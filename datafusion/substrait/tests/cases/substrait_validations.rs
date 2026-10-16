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

#[cfg(test)]
mod tests {

    // verify the schema compatibility validations
    mod schema_compatibility {
        use crate::utils::test::read_json;
        use datafusion::arrow::array::{
            LargeBinaryArray, LargeStringArray, RecordBatch, StringViewArray,
        };
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::arrow::util::pretty::pretty_format_batches;
        use datafusion::common::{DFSchema, Result, TableReference};
        use datafusion::datasource::MemTable;
        use datafusion::datasource::empty::EmptyTable;
        use datafusion::prelude::SessionContext;
        use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
        use insta::assert_snapshot;
        use std::collections::HashMap;
        use std::sync::Arc;

        fn generate_context_with_table(
            table_name: &str,
            fields: Vec<(&str, DataType, bool)>,
        ) -> Result<SessionContext> {
            let table_ref = TableReference::bare(table_name);
            let fields: Vec<(Option<TableReference>, Arc<Field>)> = fields
                .into_iter()
                .map(|pair| {
                    let (field_name, data_type, nullable) = pair;
                    (
                        Some(table_ref.clone()),
                        Arc::new(Field::new(field_name, data_type, nullable)),
                    )
                })
                .collect();

            let df_schema = DFSchema::new_with_metadata(fields, HashMap::default())?;

            let ctx = SessionContext::new();
            ctx.register_table(
                table_ref,
                Arc::new(EmptyTable::new(Arc::clone(df_schema.inner()))),
            )?;
            Ok(ctx)
        }

        #[tokio::test]
        async fn ensure_schema_match_exact() -> Result<()> {
            let proto_plan =
                read_json("tests/testdata/test_plans/simple_select.substrait.json");
            // this is the exact schema of the Substrait plan
            let df_schema =
                vec![("a", DataType::Int32, true), ("b", DataType::Int32, false)];

            let ctx = generate_context_with_table("DATA", df_schema)?;
            let plan = from_substrait_plan(&ctx.state(), &proto_plan).await?;

            assert_snapshot!(
            plan,
            @r"
            Projection: DATA.a, DATA.b
              TableScan: DATA
            "
                        );
            Ok(())
        }

        #[tokio::test]
        async fn ensure_schema_match_subset() -> Result<()> {
            let proto_plan =
                read_json("tests/testdata/test_plans/simple_select.substrait.json");
            // the DataFusion schema { b, a, c } contains the Substrait schema { a, b }
            let df_schema = vec![
                ("b", DataType::Int32, false),
                ("a", DataType::Int32, true),
                ("c", DataType::Int32, false),
            ];
            let ctx = generate_context_with_table("DATA", df_schema)?;
            let plan = from_substrait_plan(&ctx.state(), &proto_plan).await?;

            assert_snapshot!(
            plan,
            @r"
            Projection: DATA.a, DATA.b
              TableScan: DATA projection=[a, b]
            "
                        );
            Ok(())
        }

        #[tokio::test]
        async fn ensure_schema_match_subset_with_mask() -> Result<()> {
            let proto_plan = read_json(
                "tests/testdata/test_plans/simple_select_with_mask.substrait.json",
            );
            // the DataFusion schema { d, a, c, b } contains the Substrait schema { a, b, c }
            let df_schema = vec![
                ("d", DataType::Int32, true),
                ("a", DataType::Int32, false),
                ("c", DataType::Int32, false),
                ("b", DataType::Int32, false),
            ];
            let ctx = generate_context_with_table("DATA", df_schema)?;
            let plan = from_substrait_plan(&ctx.state(), &proto_plan).await?;

            assert_snapshot!(
            plan,
            @r"
            Projection: DATA.a, DATA.b
              TableScan: DATA projection=[a, b]
            "
                        );
            Ok(())
        }

        #[tokio::test]
        async fn ensure_schema_match_not_subset() -> Result<()> {
            let proto_plan =
                read_json("tests/testdata/test_plans/simple_select.substrait.json");
            // the substrait plans contains a field b which is not in the schema
            let df_schema =
                vec![("a", DataType::Int32, false), ("c", DataType::Int32, true)];

            let ctx = generate_context_with_table("DATA", df_schema)?;
            let res = from_substrait_plan(&ctx.state(), &proto_plan).await;
            assert!(res.is_err());
            Ok(())
        }

        #[tokio::test]
        async fn reject_plans_with_incompatible_field_types() -> Result<()> {
            let proto_plan =
                read_json("tests/testdata/test_plans/simple_select.substrait.json");

            let ctx =
                generate_context_with_table("DATA", vec![("a", DataType::Date32, true)])?;
            let res = from_substrait_plan(&ctx.state(), &proto_plan).await;
            assert!(res.is_err());
            Ok(())
        }

        /// A Substrait `string`, `varchar` or `fixedchar` field reads a string
        /// column of any width, and a `binary` field a binary column of any
        /// width. Producers other than DataFusion cannot name the width, and the
        /// scan keeps the table's own types, which the rows below come back in.
        #[tokio::test]
        async fn read_string_and_binary_fields_over_columns_of_any_width() -> Result<()> {
            let proto_plan = read_json(
                "tests/testdata/test_plans/read_string_and_binary_widths.substrait.json",
            );
            let schema = Arc::new(Schema::new(vec![
                Field::new("s", DataType::LargeUtf8, true),
                Field::new("v", DataType::Utf8View, true),
                Field::new("c", DataType::LargeUtf8, true),
                Field::new("b", DataType::LargeBinary, true),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(vec![
                        Some("a"),
                        Some("b"),
                        None,
                        Some("b"),
                    ])),
                    Arc::new(StringViewArray::from(vec![
                        Some("w"),
                        Some("x"),
                        Some("y"),
                        None,
                    ])),
                    Arc::new(LargeStringArray::from(vec![
                        Some("1"),
                        Some("2"),
                        Some("3"),
                        Some("4"),
                    ])),
                    Arc::new(LargeBinaryArray::from_opt_vec(vec![
                        Some(b"p".as_slice()),
                        Some(b"q".as_slice()),
                        None,
                        None,
                    ])),
                ],
            )?;
            let ctx = SessionContext::new();
            ctx.register_table(
                TableReference::bare("DATA"),
                Arc::new(MemTable::try_new(Arc::clone(&schema), vec![vec![batch]])?),
            )?;

            let plan = from_substrait_plan(&ctx.state(), &proto_plan).await?;
            let df = ctx.execute_logical_plan(plan).await?;
            let types: Vec<DataType> = df
                .schema()
                .fields()
                .iter()
                .map(|field| field.data_type().clone())
                .collect();
            assert_eq!(
                types,
                vec![
                    DataType::LargeUtf8,
                    DataType::Utf8View,
                    DataType::LargeUtf8,
                    DataType::LargeBinary
                ]
            );
            let results = df.collect().await?;
            assert_snapshot!(
                pretty_format_batches(&results)?,
                @r"
            +---+---+---+----+
            | s | v | c | b  |
            +---+---+---+----+
            | b | x | 2 | 71 |
            | b |   | 4 |    |
            +---+---+---+----+
            "
            );
            Ok(())
        }

        #[tokio::test]
        async fn reject_string_fields_over_binary_columns() -> Result<()> {
            let proto_plan = read_json(
                "tests/testdata/test_plans/read_string_and_binary_widths.substrait.json",
            );
            let ctx = generate_context_with_table(
                "DATA",
                vec![
                    ("s", DataType::LargeBinary, true),
                    ("v", DataType::Utf8, true),
                    ("c", DataType::Utf8, true),
                    ("b", DataType::Binary, true),
                ],
            )?;
            let res = from_substrait_plan(&ctx.state(), &proto_plan).await;
            assert_snapshot!(
                res.unwrap_err().strip_backtrace(),
                @"Substrait error: Field 's' in Substrait schema has a different type (Utf8) than the corresponding field in the table schema (LargeBinary)."
            );
            Ok(())
        }

        #[tokio::test]
        async fn reject_plans_with_incompatible_field_nullability() -> Result<()> {
            let proto_plan =
                read_json("tests/testdata/test_plans/simple_select.substrait.json");
            let df_schema =
                vec![("a", DataType::Int32, true), ("b", DataType::Int32, true)];

            let ctx = generate_context_with_table("DATA", df_schema)?;
            let res = from_substrait_plan(&ctx.state(), &proto_plan).await;

            assert_snapshot!(
                res.unwrap_err().strip_backtrace(),
                @r#"Substrait error: Field 'b' is nullable in the DataFusion schema but not nullable in the Substrait schema."#
            );
            Ok(())
        }
    }
}
