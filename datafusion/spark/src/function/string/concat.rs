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

use arrow::array::new_null_array;
use arrow::datatypes::{DataType, Field};
use datafusion_common::arrow::datatypes::FieldRef;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::ReturnFieldArgs;
use datafusion_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_functions::string::concat::ConcatFunc;
use std::sync::Arc;

use crate::function::null_utils::{
    NullMaskResolution, apply_null_mask, compute_null_mask,
};

/// Spark-compatible `concat` expression
/// <https://spark.apache.org/docs/latest/api/sql/index.html#concat>
///
/// Concatenates multiple input strings into a single string.
/// Returns NULL if any input is NULL.
///
/// Differences with DataFusion concat:
/// - Support 0 arguments
/// - Return NULL if any input is NULL
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SparkConcat {
    signature: Signature,
}

impl Default for SparkConcat {
    fn default() -> Self {
        Self::new()
    }
}

impl SparkConcat {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SparkConcat {
    fn name(&self) -> &str {
        "concat"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        spark_concat(args)
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        // Accept any string types, including zero arguments. An untyped `Null`
        // has to be given one: `spark_concat` hands its arguments to
        // `ConcatFunc`, whose array branch matches the concrete string and
        // binary variants and reaches `unreachable!("concat")` for anything
        // else. Under Spark semantics a NULL argument makes the whole call
        // NULL, so the type only has to be one that branch accepts.
        Ok(arg_types
            .iter()
            .map(|arg_type| match arg_type {
                DataType::Null => DataType::Utf8,
                other => other.clone(),
            })
            .collect())
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion_common::internal_err!(
            "return_type should not be called for Spark concat"
        )
    }
    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        use DataType::*;

        // Spark semantics: concat returns NULL if ANY input is NULL
        let nullable = args.arg_fields.iter().any(|f| f.is_nullable());

        // Determine return type: Utf8View > LargeUtf8 > Utf8
        let mut dt = &Utf8;
        for field in args.arg_fields {
            let data_type = field.data_type();
            if data_type == &Utf8View || (data_type == &LargeUtf8 && dt != &Utf8View) {
                dt = data_type;
            }
        }

        Ok(Arc::new(Field::new("concat", dt.clone(), nullable)))
    }
}

/// Concatenates strings, returning NULL if any input is NULL
/// This is a Spark-specific wrapper around DataFusion's concat that returns NULL
/// if any argument is NULL (Spark behavior), whereas DataFusion's concat ignores NULLs.
fn spark_concat(args: ScalarFunctionArgs) -> Result<ColumnarValue> {
    let ScalarFunctionArgs {
        args: arg_values,
        arg_fields,
        number_rows,
        return_field,
        config_options,
    } = args;

    // Handle zero-argument case: return empty string
    if arg_values.is_empty() {
        let return_type = return_field.data_type();
        return match return_type {
            DataType::Utf8View => Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(Some(
                String::new(),
            )))),
            DataType::LargeUtf8 => Ok(ColumnarValue::Scalar(ScalarValue::LargeUtf8(
                Some(String::new()),
            ))),
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Utf8(
                Some(String::new()),
            ))),
        };
    }

    // Step 1: Check for NULL mask in incoming args
    let null_mask = compute_null_mask(&arg_values);

    // If all scalars and any is NULL, return NULL immediately
    if matches!(null_mask, NullMaskResolution::ReturnNull) {
        let return_type = return_field.data_type();
        return match return_type {
            DataType::Utf8View => Ok(ColumnarValue::Scalar(ScalarValue::Utf8View(None))),
            DataType::LargeUtf8 => {
                Ok(ColumnarValue::Scalar(ScalarValue::LargeUtf8(None)))
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Utf8(None))),
        };
    }

    // Every row is NULL, so nothing DataFusion's concat computes survives
    // `apply_null_mask`. Answer directly rather than concatenating values that
    // are about to be masked away.
    if let NullMaskResolution::Apply(mask) = &null_mask
        && mask.null_count() == mask.len()
    {
        return Ok(ColumnarValue::Array(new_null_array(
            return_field.data_type(),
            mask.len(),
        )));
    }

    // Step 2: Delegate to DataFusion's concat
    let concat_func = ConcatFunc::new();
    let return_type = return_field.data_type().clone();
    let func_args = ScalarFunctionArgs {
        args: arg_values,
        arg_fields,
        number_rows,
        return_field,
        config_options,
    };
    let result = concat_func.invoke_with_args(func_args)?;

    // Step 3: Apply NULL mask to result
    apply_null_mask(result, null_mask, &return_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::utils::test::test_scalar_function;
    use arrow::array::{Array, ArrayRef, StringArray};
    use datafusion_common::config::ConfigOptions;

    /// `concat(<array>, NULL)` where the NULL literal carries no type.
    /// Without the coercion in `coerce_types` the untyped `Null` reached
    /// `ConcatFunc`'s array branch, which matches only the concrete string and
    /// binary variants, and panicked on its `unreachable!("concat")`.
    #[test]
    fn an_untyped_null_argument_is_coerced_to_a_string_type() -> Result<()> {
        let func = SparkConcat::new();

        assert_eq!(
            func.coerce_types(&[DataType::Utf8, DataType::Null])?,
            vec![DataType::Utf8, DataType::Utf8],
            "an untyped Null argument must be given a string type"
        );

        // Controls: the string and binary types the kernel already handles are
        // passed through untouched, and zero arguments stay zero arguments.
        assert_eq!(
            func.coerce_types(&[DataType::Utf8View, DataType::LargeUtf8])?,
            vec![DataType::Utf8View, DataType::LargeUtf8]
        );
        assert_eq!(
            func.coerce_types(&[DataType::Binary])?,
            vec![DataType::Binary]
        );
        assert_eq!(func.coerce_types(&[])?, Vec::<DataType>::new());

        Ok(())
    }

    /// The coerced call answers NULL for every row, matching what an
    /// already-typed NULL literal (`arrow_cast(NULL, 'Utf8')`) answers.
    #[test]
    fn an_array_beside_a_null_literal_answers_null_for_every_row() -> Result<()> {
        let names: ArrayRef = Arc::new(StringArray::from(vec![
            Some("alpha"),
            Some("beta"),
            None,
            Some("delta"),
        ]));
        let coerced =
            SparkConcat::new().coerce_types(&[DataType::Utf8, DataType::Null])?;

        let result = SparkConcat::new().invoke_with_args(ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::clone(&names)),
                ColumnarValue::Scalar(ScalarValue::try_from(&coerced[1])?),
            ],
            arg_fields: vec![
                Arc::new(Field::new("name", coerced[0].clone(), true)),
                Arc::new(Field::new("lit", coerced[1].clone(), true)),
            ],
            number_rows: names.len(),
            return_field: Arc::new(Field::new("concat", DataType::Utf8, true)),
            config_options: Arc::new(ConfigOptions::default()),
        })?;

        let array = result.to_array(names.len())?;
        assert_eq!(array.len(), names.len());
        assert_eq!(
            array.null_count(),
            names.len(),
            "every row must be NULL, got {array:?}"
        );

        Ok(())
    }

    #[test]
    fn test_concat_basic() -> Result<()> {
        test_scalar_function!(
            SparkConcat::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("Spark".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("SQL".to_string()))),
            ],
            Ok(Some("SparkSQL")),
            &str,
            DataType::Utf8,
            StringArray
        );
        Ok(())
    }

    #[test]
    fn test_concat_with_null() -> Result<()> {
        test_scalar_function!(
            SparkConcat::new(),
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("Spark".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("SQL".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(None)),
            ],
            Ok(None),
            &str,
            DataType::Utf8,
            StringArray
        );
        Ok(())
    }

    #[test]
    fn test_spark_concat_return_field_non_nullable() -> Result<()> {
        let func = SparkConcat::new();

        let fields = vec![
            Arc::new(Field::new("a", DataType::Utf8, false)),
            Arc::new(Field::new("b", DataType::Utf8, false)),
        ];

        let args = ReturnFieldArgs {
            arg_fields: &fields,
            scalar_arguments: &[],
        };

        let field = func.return_field_from_args(args)?;

        assert!(
            !field.is_nullable(),
            "Expected concat result to be non-nullable when all inputs are non-nullable"
        );

        Ok(())
    }
    #[test]
    fn test_spark_concat_return_field_nullable() -> Result<()> {
        let func = SparkConcat::new();

        let fields = vec![
            Arc::new(Field::new("a", DataType::Utf8, false)),
            Arc::new(Field::new("b", DataType::Utf8, true)),
        ];

        let args = ReturnFieldArgs {
            arg_fields: &fields,
            scalar_arguments: &[],
        };

        let field = func.return_field_from_args(args)?;

        assert!(
            field.is_nullable(),
            "Expected concat result to be nullable when any input is nullable"
        );

        Ok(())
    }
}
