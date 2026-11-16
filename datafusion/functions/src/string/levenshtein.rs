// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, OffsetSizeTrait};
use arrow::datatypes::DataType;

use crate::utils::{make_scalar_function, utf8_to_int_type};
use datafusion_common::cast::{as_generic_string_array, as_string_view_array};
use datafusion_common::types::logical_string;
use datafusion_common::utils::datafusion_strsim;
use datafusion_common::utils::take_function_args;
use datafusion_common::{Result, exec_err};
use datafusion_expr::type_coercion::binary::{
    binary_to_string_coercion, string_coercion,
};
use datafusion_expr::{
    Coercion, ColumnarValue, Documentation, ScalarFunctionArgs, ScalarUDFImpl, Signature,
    TypeSignatureClass, Volatility,
};
use datafusion_macros::user_doc;

#[user_doc(
    doc_section(label = "String Functions"),
    description = "Returns the [`Levenshtein distance`](https://en.wikipedia.org/wiki/Levenshtein_distance) between the two given strings.",
    syntax_example = "levenshtein(str1, str2)",
    sql_example = r#"```sql
> select levenshtein('kitten', 'sitting');
+---------------------------------------------+
| levenshtein(Utf8("kitten"),Utf8("sitting")) |
+---------------------------------------------+
| 3                                           |
+---------------------------------------------+
```"#,
    argument(
        name = "str1",
        description = "String expression to compute Levenshtein distance with str2."
    ),
    argument(
        name = "str2",
        description = "String expression to compute Levenshtein distance with str1."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LevenshteinFunc {
    signature: Signature,
}

impl Default for LevenshteinFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl LevenshteinFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::coercible(
                vec![
                    Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                    Coercion::new_exact(TypeSignatureClass::Native(logical_string())),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for LevenshteinFunc {
    fn name(&self) -> &str {
        "levenshtein"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if let Some(coercion_data_type) = string_coercion(&arg_types[0], &arg_types[1])
            .or_else(|| binary_to_string_coercion(&arg_types[0], &arg_types[1]))
        {
            utf8_to_int_type(&coercion_data_type, "levenshtein")
        } else {
            exec_err!(
                "Unsupported data types for levenshtein. Expected Utf8, LargeUtf8 or Utf8View"
            )
        }
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match args.args[0].data_type() {
            DataType::Utf8View | DataType::Utf8 => {
                make_scalar_function(levenshtein::<i32>, vec![])(&args.args)
            }
            DataType::LargeUtf8 => {
                make_scalar_function(levenshtein::<i64>, vec![])(&args.args)
            }
            other => {
                exec_err!("Unsupported data type {other:?} for function levenshtein")
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

/// The most character pairs, `chars(str1) * chars(str2)` summed over its rows,
/// one call may compare. The distances take that many steps with no
/// cancellation point, so a larger call would hold its thread far past any
/// query timeout.
const MAX_LEVENSHTEIN_CHAR_PAIRS: u64 = 1_000_000_000;

/// Refuses a call whose distances would take more than
/// [`MAX_LEVENSHTEIN_CHAR_PAIRS`] steps in total. A row with a null string is
/// not compared, so it does not count.
fn check_levenshtein_size<'a, I>(rows: impl Fn() -> I) -> Result<()>
where
    I: Iterator<Item = (Option<&'a str>, Option<&'a str>)>,
{
    // A character is at least one byte, so byte lengths bound the pairs.
    if total_pairs(rows(), str::len) <= MAX_LEVENSHTEIN_CHAR_PAIRS {
        return Ok(());
    }
    if total_pairs(rows(), |s| s.chars().count()) > MAX_LEVENSHTEIN_CHAR_PAIRS {
        return exec_err!(
            "levenshtein input is too long: the strings in one batch have more than {MAX_LEVENSHTEIN_CHAR_PAIRS} character pairs (chars(str1) * chars(str2) summed over the batch's rows)"
        );
    }
    Ok(())
}

/// `len(str1) * len(str2)` summed over `rows`, stopping once the sum passes
/// [`MAX_LEVENSHTEIN_CHAR_PAIRS`].
fn total_pairs<'a>(
    rows: impl Iterator<Item = (Option<&'a str>, Option<&'a str>)>,
    len: impl Fn(&str) -> usize,
) -> u64 {
    let mut total = 0u64;
    for (str1, str2) in rows {
        if let (Some(str1), Some(str2)) = (str1, str2) {
            let pairs = (len(str1) as u64).saturating_mul(len(str2) as u64);
            total = total.saturating_add(pairs);
            if total > MAX_LEVENSHTEIN_CHAR_PAIRS {
                break;
            }
        }
    }
    total
}

///Returns the Levenshtein distance between the two given strings.
/// LEVENSHTEIN('kitten', 'sitting') = 3
fn levenshtein<T: OffsetSizeTrait>(args: &[ArrayRef]) -> Result<ArrayRef> {
    let [str1, str2] = take_function_args("levenshtein", args)?;

    if let Some(coercion_data_type) =
        string_coercion(args[0].data_type(), args[1].data_type()).or_else(|| {
            binary_to_string_coercion(args[0].data_type(), args[1].data_type())
        })
    {
        let str1 = if str1.data_type() == &coercion_data_type {
            Arc::clone(str1)
        } else {
            arrow::compute::kernels::cast::cast(&str1, &coercion_data_type)?
        };
        let str2 = if str2.data_type() == &coercion_data_type {
            Arc::clone(str2)
        } else {
            arrow::compute::kernels::cast::cast(&str2, &coercion_data_type)?
        };

        match coercion_data_type {
            DataType::Utf8View => {
                let str1_array = as_string_view_array(&str1)?;
                let str2_array = as_string_view_array(&str2)?;
                check_levenshtein_size(|| str1_array.iter().zip(str2_array.iter()))?;

                // Reusable buffer to avoid allocating for each row
                let mut cache = Vec::new();

                let result = str1_array
                    .iter()
                    .zip(str2_array.iter())
                    .map(|(string1, string2)| match (string1, string2) {
                        (Some(string1), Some(string2)) => {
                            Some(datafusion_strsim::levenshtein_with_buffer(
                                string1, string2, &mut cache,
                            ) as i32)
                        }
                        _ => None,
                    })
                    .collect::<Int32Array>();
                Ok(Arc::new(result) as ArrayRef)
            }
            DataType::Utf8 => {
                let str1_array = as_generic_string_array::<T>(&str1)?;
                let str2_array = as_generic_string_array::<T>(&str2)?;
                check_levenshtein_size(|| str1_array.iter().zip(str2_array.iter()))?;

                // Reusable buffer to avoid allocating for each row
                let mut cache = Vec::new();

                let result = str1_array
                    .iter()
                    .zip(str2_array.iter())
                    .map(|(string1, string2)| match (string1, string2) {
                        (Some(string1), Some(string2)) => {
                            Some(datafusion_strsim::levenshtein_with_buffer(
                                string1, string2, &mut cache,
                            ) as i32)
                        }
                        _ => None,
                    })
                    .collect::<Int32Array>();
                Ok(Arc::new(result) as ArrayRef)
            }
            DataType::LargeUtf8 => {
                let str1_array = as_generic_string_array::<T>(&str1)?;
                let str2_array = as_generic_string_array::<T>(&str2)?;
                check_levenshtein_size(|| str1_array.iter().zip(str2_array.iter()))?;

                // Reusable buffer to avoid allocating for each row
                let mut cache = Vec::new();

                let result = str1_array
                    .iter()
                    .zip(str2_array.iter())
                    .map(|(string1, string2)| match (string1, string2) {
                        (Some(string1), Some(string2)) => {
                            Some(datafusion_strsim::levenshtein_with_buffer(
                                string1, string2, &mut cache,
                            ) as i64)
                        }
                        _ => None,
                    })
                    .collect::<Int64Array>();
                Ok(Arc::new(result) as ArrayRef)
            }
            other => {
                exec_err!(
                    "levenshtein was called with {other} datatype arguments. It requires Utf8View, Utf8 or LargeUtf8."
                )
            }
        }
    } else {
        exec_err!(
            "Unsupported data types for levenshtein. Expected Utf8, LargeUtf8 or Utf8View"
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::StringArray;
    use arrow::datatypes::Field;

    use datafusion_common::ScalarValue;
    use datafusion_common::cast::{as_int32_array, as_int64_array};
    use datafusion_common::config::ConfigOptions;

    use super::*;

    const STRING_TYPES: [DataType; 3] =
        [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View];

    const TOO_LONG: &str = "Execution error: levenshtein input is too long: the strings in one batch have more than 1000000000 character pairs (chars(str1) * chars(str2) summed over the batch's rows)";

    /// The size check over `rows` copies of one row.
    fn check_rows(str1: &str, str2: &str, rows: usize) -> Result<()> {
        check_levenshtein_size(|| std::iter::repeat_n((Some(str1), Some(str2)), rows))
    }

    fn strings(data_type: &DataType, rows: Vec<Option<&str>>) -> ArrayRef {
        arrow::compute::cast(&StringArray::from(rows), data_type).unwrap()
    }

    fn scalar(data_type: &DataType, value: &str) -> ColumnarValue {
        let array = strings(data_type, vec![Some(value)]);
        ColumnarValue::Scalar(ScalarValue::try_from_array(&array, 0).unwrap())
    }

    fn array(data_type: &DataType, rows: Vec<Option<&str>>) -> ColumnarValue {
        ColumnarValue::Array(strings(data_type, rows))
    }

    /// Calls the function the way a physical plan does.
    fn invoke(args: Vec<ColumnarValue>) -> Result<ColumnarValue> {
        let func = LevenshteinFunc::new();
        let arg_types: Vec<_> = args.iter().map(ColumnarValue::data_type).collect();
        let return_type = func.return_type(&arg_types)?;
        let number_rows = args
            .iter()
            .find_map(|arg| match arg {
                ColumnarValue::Array(array) => Some(array.len()),
                ColumnarValue::Scalar(_) => None,
            })
            .unwrap_or(1);
        func.invoke_with_args(ScalarFunctionArgs {
            arg_fields: arg_types
                .into_iter()
                .map(|data_type| Arc::new(Field::new("s", data_type, true)))
                .collect(),
            args,
            number_rows,
            return_field: Arc::new(Field::new("d", return_type, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })
    }

    fn distances(result: ColumnarValue) -> Vec<Option<i64>> {
        let result =
            arrow::compute::cast(&result.to_array(1).unwrap(), &DataType::Int64).unwrap();
        as_int64_array(&result).unwrap().iter().collect()
    }

    #[test]
    fn size_check_accepts_the_limit_and_refuses_past_it() {
        // One long row: 40,000 x 25,000 is the limit.
        let (long1, long2) = ("a".repeat(40_000), "b".repeat(25_000));
        check_rows(&long1, &long2, 1).unwrap();
        let error = check_rows(&"a".repeat(40_001), &long2, 1).unwrap_err();
        assert_eq!(error.strip_backtrace(), TOO_LONG);

        // Many shorter rows: 1,000 rows of 1,000 x 1,000 is the limit.
        let short = "a".repeat(1_000);
        check_rows(&short, &short, 1_000).unwrap();
        let error = check_rows(&short, &short, 1_001).unwrap_err();
        assert_eq!(error.strip_backtrace(), TOO_LONG);

        // Two bytes per character: past the limit in bytes, at it in characters.
        check_rows(&"\u{e9}".repeat(40_000), &"\u{e9}".repeat(25_000), 1).unwrap();

        // A row with a null or an empty string compares nothing.
        let huge = "a".repeat(1_000_000);
        check_levenshtein_size(|| {
            [
                (None, Some(huge.as_str())),
                (Some(""), Some(huge.as_str())),
                (Some(long1.as_str()), Some(long2.as_str())),
            ]
            .into_iter()
        })
        .unwrap();
    }

    #[test]
    fn every_string_type_refuses_a_call_past_the_size_limit() {
        let (long1, long2) = ("a".repeat(40_001), "b".repeat(25_000));
        let short = "a".repeat(1_000);
        let shorts = vec![Some(short.as_str()); 1_001];
        for data_type in &STRING_TYPES {
            for args in [
                // One long row.
                vec![scalar(data_type, &long1), scalar(data_type, &long2)],
                vec![
                    array(data_type, vec![Some("kitten"), Some(&long1)]),
                    array(data_type, vec![Some("sitting"), Some(&long2)]),
                ],
                // Many shorter rows; a scalar counts once for each row.
                vec![
                    array(data_type, shorts.clone()),
                    array(data_type, shorts.clone()),
                ],
                vec![array(data_type, shorts.clone()), scalar(data_type, &short)],
            ] {
                let error = invoke(args).unwrap_err();
                assert_eq!(error.strip_backtrace(), TOO_LONG, "{data_type}");
            }
        }
    }

    #[test]
    fn every_string_type_keeps_its_distances() {
        for data_type in &STRING_TYPES {
            let result = invoke(vec![
                scalar(data_type, "kitten"),
                scalar(data_type, "sitting"),
            ])
            .unwrap();
            assert!(matches!(result, ColumnarValue::Scalar(_)), "{data_type}");
            assert_eq!(distances(result), vec![Some(3)], "{data_type}");

            let result = invoke(vec![
                array(
                    data_type,
                    vec![Some("123"), Some("na\u{ef}ve"), None, Some("")],
                ),
                array(
                    data_type,
                    vec![Some("321"), Some("naive"), Some("x"), Some("abc")],
                ),
            ])
            .unwrap();
            assert_eq!(
                distances(result),
                vec![Some(2), Some(1), None, Some(3)],
                "{data_type}"
            );
        }
    }

    #[test]
    fn to_levenshtein() -> Result<()> {
        let string1_array =
            Arc::new(StringArray::from(vec!["123", "abc", "xyz", "kitten"]));
        let string2_array =
            Arc::new(StringArray::from(vec!["321", "def", "zyx", "sitting"]));
        let res = levenshtein::<i32>(&[string1_array, string2_array]).unwrap();
        let result =
            as_int32_array(&res).expect("failed to initialized function levenshtein");
        let expected = Int32Array::from(vec![2, 3, 2, 3]);
        assert_eq!(&expected, result);

        Ok(())
    }
}
