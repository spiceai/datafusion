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

//! Support for LATERAL table-function calls whose arguments reference columns
//! of an earlier FROM item, such as
//! `SELECT * FROM t CROSS JOIN LATERAL generate_series(1, t.n)`.
//!
//! The SQL planner lowers such a call to a projection that evaluates a scalar
//! UDF (see [`ContextProvider::create_lateral_table_function_udf`]) followed by
//! `UNNEST`. The UDF calls the table function once per distinct tuple of
//! column values and returns all rows of that call as one
//! `List<Struct<..>>` value.
//!
//! [`ContextProvider::create_lateral_table_function_udf`]: crate::planner::ContextProvider::create_lateral_table_function_udf

use arrow::datatypes::{DataType, FieldRef, Fields};
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Column, Result, ScalarValue, internal_err};

use crate::Expr;
use crate::expr::{Alias, FieldMetadata};

/// Metadata key that the SQL planner puts on a named table-function argument
/// (`limit => 10`). The value is the parameter name.
pub const TABLE_FUNCTION_PARAMETER_NAME_KEY: &str = "spice.parameter_name";

/// String value that stands in for a string column while the output schema of
/// a call is derived. Some table functions reject an empty string.
const PLACEHOLDER_TEXT: &str = "placeholder";

/// A table-function call whose arguments reference columns of an earlier FROM
/// item. Input to
/// [`ContextProvider::create_lateral_table_function_udf`](crate::planner::ContextProvider::create_lateral_table_function_udf).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LateralTableFunctionCall {
    /// Name of the table function.
    pub name: String,
    /// The planned arguments. Each [`Expr::OuterReferenceColumn`] is replaced
    /// with the current row's value before the table function is called.
    pub args: Vec<Expr>,
    /// The distinct outer columns that `args` reference. The UDF takes one
    /// argument per column, in this order.
    pub outer_columns: Vec<Column>,
    /// Fields of one result row. All are nullable. The UDF returns
    /// `List<Struct<row_fields>>`.
    pub row_fields: Fields,
}

impl LateralTableFunctionCall {
    /// The type of the UDF result: one list element per result row.
    pub fn return_type(&self) -> DataType {
        DataType::new_list(DataType::Struct(self.row_fields.clone()), true)
    }
}

/// Returns the distinct outer columns that `args` reference, in first-use
/// order, with their fields.
pub fn outer_references(args: &[Expr]) -> Result<Vec<(FieldRef, Column)>> {
    let mut found: Vec<(FieldRef, Column)> = Vec::new();
    for arg in args {
        arg.apply(|expr| {
            if let Expr::OuterReferenceColumn(field, column) = expr
                && !found.iter().any(|(_, seen)| seen == column)
            {
                found.push((FieldRef::clone(field), column.clone()));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
    }
    Ok(found)
}

/// Replaces each reference to `columns[i]` in `args` with the literal
/// `values[i]`, also inside nested expressions such as
/// `rrf(text_search(t, q.text))`.
///
/// Table functions match literal values, so a value is bound as the literal
/// that SQL text would produce: strings are `Utf8` and a NULL is the untyped
/// `NULL`. A named argument whose value was a column is planned as an alias
/// that carries the parameter name; once bound it becomes a literal that
/// carries the name, the same as a named literal argument.
pub fn bind_outer_references(
    args: &[Expr],
    columns: &[Column],
    values: &[ScalarValue],
) -> Result<Vec<Expr>> {
    if columns.len() != values.len() {
        return internal_err!(
            "LATERAL table function has {} outer columns but {} values",
            columns.len(),
            values.len()
        );
    }
    args.iter()
        .map(|arg| {
            let bound = arg
                .clone()
                .transform_up(|expr| match expr {
                    Expr::OuterReferenceColumn(_, ref column) => {
                        let Some(index) = columns.iter().position(|c| c == column) else {
                            return internal_err!(
                                "LATERAL table function has no value for outer column {column}"
                            );
                        };
                        Ok(Transformed::yes(Expr::Literal(
                            normalize_value(values[index].clone()),
                            None,
                        )))
                    }
                    other => Ok(Transformed::no(other)),
                })?
                .data;
            Ok(named_literal(bound))
        })
        .collect()
}

/// A value of `data_type` that stands in for a column while the output schema
/// of a call is derived. The call is only planned, not executed.
pub fn placeholder_value(data_type: &DataType) -> Result<ScalarValue> {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            Ok(ScalarValue::Utf8(Some(PLACEHOLDER_TEXT.to_string())))
        }
        other => ScalarValue::new_default(other),
    }
}

/// Converts a column value to the literal that SQL text would produce.
fn normalize_value(value: ScalarValue) -> ScalarValue {
    if value.is_null() {
        return ScalarValue::Null;
    }
    match value {
        ScalarValue::LargeUtf8(s) | ScalarValue::Utf8View(s) => ScalarValue::Utf8(s),
        other => other,
    }
}

/// Turns `Alias(Literal)` with a parameter name into `Literal` with that name.
fn named_literal(expr: Expr) -> Expr {
    match expr {
        Expr::Alias(Alias {
            expr: inner,
            relation,
            name,
            metadata,
        }) => match (*inner, metadata) {
            (Expr::Literal(value, literal_metadata), Some(alias_metadata))
                if alias_metadata
                    .inner()
                    .contains_key(TABLE_FUNCTION_PARAMETER_NAME_KEY) =>
            {
                let mut merged = literal_metadata.unwrap_or_else(FieldMetadata::default);
                merged.extend(alias_metadata);
                Expr::Literal(value, Some(merged))
            }
            (inner, metadata) => Expr::Alias(Alias {
                expr: Box::new(inner),
                relation,
                name,
                metadata,
            }),
        },
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use arrow::datatypes::Field;

    use super::*;
    use crate::{col, lit};

    fn outer(name: &str, data_type: DataType) -> Expr {
        Expr::OuterReferenceColumn(
            Arc::new(Field::new(name, data_type, true)),
            Column::new(Some("q"), name),
        )
    }

    fn parameter_name(name: &str) -> FieldMetadata {
        FieldMetadata::new(BTreeMap::from([(
            TABLE_FUNCTION_PARAMETER_NAME_KEY.to_string(),
            name.to_string(),
        )]))
    }

    #[test]
    fn outer_references_are_distinct_and_ordered() -> Result<()> {
        let args = vec![
            col("corpus"),
            outer("text", DataType::LargeUtf8),
            outer("k", DataType::Int64),
            outer("text", DataType::LargeUtf8),
        ];
        let found = outer_references(&args)?;
        let names: Vec<_> = found.iter().map(|(_, c)| c.name.as_str()).collect();
        assert_eq!(names, vec!["text", "k"]);
        assert_eq!(found[0].0.data_type(), &DataType::LargeUtf8);
        Ok(())
    }

    #[test]
    fn binds_outer_references_and_keeps_table_argument() -> Result<()> {
        let args = vec![col("corpus"), outer("text", DataType::LargeUtf8)];
        let columns = vec![Column::new(Some("q"), "text")];
        let values = vec![ScalarValue::LargeUtf8(Some("cancer".to_string()))];

        let bound = bind_outer_references(&args, &columns, &values)?;

        assert_eq!(bound[0], col("corpus"));
        // `LargeUtf8` is passed as `Utf8`, the type table functions match.
        assert_eq!(bound[1], lit("cancer"));
        Ok(())
    }

    #[test]
    fn binds_utf8_view_as_utf8() -> Result<()> {
        let args = vec![outer("text", DataType::Utf8View)];
        let columns = vec![Column::new(Some("q"), "text")];
        let values = vec![ScalarValue::Utf8View(Some("x".to_string()))];

        let bound = bind_outer_references(&args, &columns, &values)?;

        assert_eq!(bound, vec![lit("x")]);
        Ok(())
    }

    #[test]
    fn binds_typed_null_as_untyped_null() -> Result<()> {
        let args = vec![lit(1i64), outer("n", DataType::Int64)];
        let columns = vec![Column::new(Some("q"), "n")];
        let values = vec![ScalarValue::Int64(None)];

        let bound = bind_outer_references(&args, &columns, &values)?;

        assert_eq!(bound[1], Expr::Literal(ScalarValue::Null, None));
        Ok(())
    }

    #[test]
    fn binds_inside_nested_expressions() -> Result<()> {
        // A nested call such as `rrf(text_search(t, q.text))` is an expression
        // argument; its outer references are bound too.
        let nested = outer("text", DataType::Utf8).eq(lit("a"));
        let columns = vec![Column::new(Some("q"), "text")];
        let values = vec![ScalarValue::Utf8(Some("x".to_string()))];

        let bound = bind_outer_references(&[nested], &columns, &values)?;

        assert_eq!(bound[0], lit("x").eq(lit("a")));
        Ok(())
    }

    #[test]
    fn named_column_argument_becomes_named_literal() -> Result<()> {
        let named = outer("text", DataType::Utf8)
            .alias_with_metadata("query", Some(parameter_name("query")));
        let columns = vec![Column::new(Some("q"), "text")];
        let values = vec![ScalarValue::Utf8(Some("x".to_string()))];

        let bound = bind_outer_references(&[named], &columns, &values)?;

        assert_eq!(
            bound[0],
            Expr::Literal(
                ScalarValue::Utf8(Some("x".into())),
                Some(parameter_name("query"))
            )
        );
        Ok(())
    }

    #[test]
    fn plain_alias_is_kept() -> Result<()> {
        let aliased = outer("text", DataType::Utf8).alias("not_a_parameter");
        let columns = vec![Column::new(Some("q"), "text")];
        let values = vec![ScalarValue::Utf8(Some("x".to_string()))];

        let bound = bind_outer_references(&[aliased], &columns, &values)?;

        assert_eq!(bound[0], lit("x").alias("not_a_parameter"));
        Ok(())
    }

    #[test]
    fn mismatched_value_count_is_an_error() {
        let columns = vec![Column::new(Some("q"), "text")];
        assert!(bind_outer_references(&[], &columns, &[]).is_err());
    }

    #[test]
    fn placeholder_strings_are_non_empty_utf8() -> Result<()> {
        for data_type in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            assert_eq!(
                placeholder_value(&data_type)?,
                ScalarValue::Utf8(Some(PLACEHOLDER_TEXT.to_string()))
            );
        }
        assert_eq!(
            placeholder_value(&DataType::Int64)?,
            ScalarValue::Int64(Some(0))
        );
        Ok(())
    }

    #[test]
    fn return_type_is_list_of_nullable_struct() {
        let call = LateralTableFunctionCall {
            name: "f".to_string(),
            args: vec![],
            outer_columns: vec![],
            row_fields: Fields::from(vec![Field::new("value", DataType::Int64, true)]),
        };
        let DataType::List(item) = call.return_type() else {
            panic!("expected a list");
        };
        assert!(item.is_nullable());
        assert_eq!(
            item.data_type(),
            &DataType::Struct(Fields::from(vec![Field::new(
                "value",
                DataType::Int64,
                true
            )]))
        );
    }
}
