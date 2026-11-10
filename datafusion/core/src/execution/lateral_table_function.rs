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

//! The async scalar UDF that evaluates a LATERAL table-function call with
//! column arguments. See [`datafusion_expr::lateral_table_function`].

use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, ListArray, StructArray, UInt64Array, new_empty_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::{cast, concat, take};
use arrow::datatypes::{DataType, Field, Fields, SchemaRef};
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion_catalog::TableFunction;
use datafusion_common::config::ConfigOptions;
use datafusion_common::{Result, ScalarValue, TableReference, exec_err, internal_err};
use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
use datafusion_expr::lateral_table_function::{
    LateralTableFunctionCall, bind_outer_references,
};
use datafusion_expr::{
    ColumnarValue, LogicalPlanBuilder, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature, Volatility,
};
use datafusion_physical_plan::ExecutionPlanProperties;
use futures::{StreamExt, TryStreamExt, stream};

use crate::datasource::provider_as_source;
use crate::execution::session_state::{SessionState, table_function_provider};

/// Calls a table function once per distinct tuple of outer column values and
/// returns each call's rows as one `List<Struct<..>>` value.
///
/// Each call binds the outer columns to literals before the table function
/// runs, because some table functions keep their arguments and build a new
/// plan from them when scanned. The provider is then planned and executed
/// through the session, the same as `SELECT * FROM f(<literals>)`.
pub(crate) struct LateralTableFunctionUdf {
    name: String,
    signature: Signature,
    tbl_func: Arc<TableFunction>,
    call: LateralTableFunctionCall,
    /// The session that planned the query. The expression API has no session
    /// at invoke time, and every call needs one to plan its provider.
    state: Arc<SessionState>,
}

impl LateralTableFunctionUdf {
    pub(crate) fn new_udf(
        tbl_func: Arc<TableFunction>,
        call: LateralTableFunctionCall,
        state: Arc<SessionState>,
    ) -> Arc<ScalarUDF> {
        let udf = Self {
            name: format!("lateral_{}", call.name),
            signature: Signature::any(call.outer_columns.len(), Volatility::Volatile),
            tbl_func,
            call,
            state,
        };
        Arc::new(AsyncScalarUDF::new(Arc::new(udf)).into_scalar_udf())
    }

    /// Runs one call and returns its rows as a struct array.
    async fn call_once(&self, values: Vec<ScalarValue>) -> Result<StructArray> {
        let args =
            bind_outer_references(&self.call.args, &self.call.outer_columns, &values)?;
        let provider = table_function_provider(&self.state, &self.tbl_func, args)?;
        let plan = LogicalPlanBuilder::scan(
            TableReference::bare(self.call.name.as_str()),
            provider_as_source(provider),
            None,
        )?
        .build()?;
        let plan = self.state.create_physical_plan(&plan).await?;

        // Poll every partition in this task, so task-local state of the
        // caller is visible to the scan.
        let task_ctx = self.state.task_ctx();
        let partitions = (0..plan.output_partitioning().partition_count()).map(|p| {
            let stream = plan.execute(p, Arc::clone(&task_ctx));
            async move { stream?.try_collect::<Vec<_>>().await }
        });
        let batches: Vec<RecordBatch> = futures::future::try_join_all(partitions)
            .await?
            .into_iter()
            .flatten()
            .collect();
        self.rows_to_struct(&plan.schema(), &batches)
    }

    /// Packs the rows of one call into a struct array of the planned row type.
    fn rows_to_struct(
        &self,
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<StructArray> {
        let expected = &self.call.row_fields;
        let matches = schema.fields().len() == expected.len()
            && schema
                .fields()
                .iter()
                .zip(expected.iter())
                .all(|(got, want)| {
                    got.name() == want.name()
                        && got.data_type().equals_datatype(want.data_type())
                });
        if !matches {
            return exec_err!(
                "LATERAL table function '{}' returned columns ({}), but the query was planned for ({})",
                self.call.name,
                describe_fields(schema.fields()),
                describe_fields(expected)
            );
        }

        let columns = expected
            .iter()
            .enumerate()
            .map(|(i, field)| {
                let parts: Vec<&dyn Array> =
                    batches.iter().map(|b| b.column(i).as_ref()).collect();
                let column = if parts.is_empty() {
                    new_empty_array(field.data_type())
                } else {
                    concat(&parts)?
                };
                // Nested field nullability can differ from the planned type.
                if column.data_type() == field.data_type() {
                    Ok(column)
                } else {
                    Ok(cast(&column, field.data_type())?)
                }
            })
            .collect::<Result<Vec<ArrayRef>>>()?;
        Ok(StructArray::try_new(expected.clone(), columns, None)?)
    }
}

fn describe_fields(fields: &Fields) -> String {
    fields
        .iter()
        .map(|f| format!("{} {}", f.name(), f.data_type()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The maximum number of calls in flight for one batch.
fn concurrency(options: &ConfigOptions) -> usize {
    options.execution.target_partitions.max(1)
}

impl Debug for LateralTableFunctionUdf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LateralTableFunctionUdf")
            .field("name", &self.name)
            .field("call", &self.call)
            .finish_non_exhaustive()
    }
}

impl PartialEq for LateralTableFunctionUdf {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.call == other.call
            && Arc::ptr_eq(&self.tbl_func, &other.tbl_func)
            && Arc::ptr_eq(&self.state, &other.state)
    }
}

impl Eq for LateralTableFunctionUdf {}

impl Hash for LateralTableFunctionUdf {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.call.hash(state);
    }
}

impl ScalarUDFImpl for LateralTableFunctionUdf {
    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(self.call.return_type())
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        internal_err!("{} can only be invoked asynchronously", self.name)
    }
}

#[async_trait]
impl AsyncScalarUDFImpl for LateralTableFunctionUdf {
    async fn invoke_async_with_args(
        &self,
        args: ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;
        let columns = args
            .args
            .iter()
            .map(|arg| arg.to_array(num_rows))
            .collect::<Result<Vec<_>>>()?;

        // Call the table function once per distinct tuple of values.
        let mut distinct: Vec<Vec<ScalarValue>> = Vec::new();
        let mut call_index: HashMap<Vec<ScalarValue>, usize> = HashMap::new();
        let mut row_calls = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let values = columns
                .iter()
                .map(|column| ScalarValue::try_from_array(column, row))
                .collect::<Result<Vec<_>>>()?;
            let index = match call_index.get(&values) {
                Some(index) => *index,
                None => {
                    let index = distinct.len();
                    call_index.insert(values.clone(), index);
                    distinct.push(values);
                    index
                }
            };
            row_calls.push(index);
        }

        let results: Vec<StructArray> =
            stream::iter(distinct.into_iter().map(|values| self.call_once(values)))
                .buffered(concurrency(&args.config_options))
                .try_collect()
                .await?;

        // Lay out one list entry per input row; rows with equal values share
        // the rows of one call.
        let row_type = DataType::Struct(self.call.row_fields.clone());
        let mut starts = Vec::with_capacity(results.len());
        let mut total = 0usize;
        for result in &results {
            starts.push(total);
            total += result.len();
        }
        let values: ArrayRef = if results.is_empty() {
            new_empty_array(&row_type)
        } else {
            let parts: Vec<&dyn Array> =
                results.iter().map(|r| r as &dyn Array).collect();
            concat(&parts)?
        };
        let mut indices = Vec::new();
        let mut lengths = Vec::with_capacity(num_rows);
        for &call in &row_calls {
            let len = results[call].len();
            let start = starts[call];
            indices.extend((start..start + len).map(|i| i as u64));
            lengths.push(len);
        }
        let values = take(values.as_ref(), &UInt64Array::from(indices), None)?;
        let list = ListArray::try_new(
            Arc::new(Field::new_list_field(row_type, true)),
            OffsetBuffer::from_lengths(lengths),
            values,
            None,
        )?;
        Ok(ColumnarValue::Array(Arc::new(list)))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray, StringViewArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_catalog::{TableFunctionArgs, TableFunctionImpl};
    use datafusion_common::{
        Result, ScalarValue, assert_batches_eq, assert_contains, plan_err,
    };
    use datafusion_expr::Expr;
    use datafusion_physical_plan::displayable;

    use crate::catalog::TableProvider;
    use crate::datasource::MemTable;
    use crate::prelude::{SessionConfig, SessionContext};

    /// `echo(text)` returns two rows, `text` and `text!`. It records the
    /// arguments of every call.
    #[derive(Debug, Default)]
    struct Echo {
        calls: Arc<Mutex<Vec<Vec<Expr>>>>,
        /// Return an Int64 column for real calls, to test schema validation.
        change_schema: bool,
    }

    impl TableFunctionImpl for Echo {
        fn call_with_args(
            &self,
            args: TableFunctionArgs,
        ) -> Result<Arc<dyn TableProvider>> {
            self.calls.lock().unwrap().push(args.exprs().to_vec());
            let [Expr::Literal(ScalarValue::Utf8(Some(text)), _)] = args.exprs() else {
                return plan_err!("echo takes one Utf8 literal, got {:?}", args.exprs());
            };
            let batch = if self.change_schema && text != "placeholder" {
                RecordBatch::try_from_iter(vec![(
                    "word",
                    Arc::new(Int64Array::from(vec![1])) as ArrayRef,
                )])?
            } else {
                RecordBatch::try_from_iter(vec![(
                    "word",
                    Arc::new(StringArray::from(vec![text.clone(), format!("{text}!")]))
                        as ArrayRef,
                )])?
            };
            Ok(Arc::new(MemTable::try_new(
                batch.schema(),
                vec![vec![batch]],
            )?))
        }
    }

    /// A context with `echo` and a table `q(id, text)` whose `text` column is
    /// `Utf8View` with a repeated value.
    fn context(echo: Echo) -> Result<SessionContext> {
        let ctx = SessionContext::new_with_config(
            SessionConfig::new().with_target_partitions(1),
        );
        ctx.register_udtf("echo", Arc::new(echo));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8View, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringViewArray::from(vec!["a", "a", "b"])),
            ],
        )?;
        ctx.register_table("q", Arc::new(MemTable::try_new(schema, vec![vec![batch]])?))?;
        Ok(ctx)
    }

    #[tokio::test]
    async fn calls_once_per_distinct_value_with_bound_utf8_literals() -> Result<()> {
        let calls = Arc::new(Mutex::new(vec![]));
        let ctx = context(Echo {
            calls: Arc::clone(&calls),
            change_schema: false,
        })?;

        let batches = ctx
            .sql(
                "SELECT q.id, r.word FROM q CROSS JOIN LATERAL echo(q.text) AS r \
                 ORDER BY q.id, r.word",
            )
            .await?
            .collect()
            .await?;

        assert_batches_eq!(
            [
                "+----+------+",
                "| id | word |",
                "+----+------+",
                "| 1  | a    |",
                "| 1  | a!   |",
                "| 2  | a    |",
                "| 2  | a!   |",
                "| 3  | b    |",
                "| 3  | b!   |",
                "+----+------+",
            ],
            &batches
        );

        // One placeholder call while planning, then one call per distinct
        // value. `Utf8View` values arrive as `Utf8` literals.
        let mut bound: Vec<Vec<Expr>> = calls.lock().unwrap().clone();
        let placeholder = bound.remove(0);
        assert_eq!(
            placeholder,
            vec![Expr::Literal(ScalarValue::from("placeholder"), None)]
        );
        bound.sort_by_key(|args| args[0].to_string());
        assert_eq!(
            bound,
            vec![
                vec![Expr::Literal(ScalarValue::from("a"), None)],
                vec![Expr::Literal(ScalarValue::from("b"), None)],
            ]
        );
        Ok(())
    }

    #[tokio::test]
    async fn plans_to_async_func_and_unnest() -> Result<()> {
        let ctx = context(Echo::default())?;
        let plan = ctx
            .sql("SELECT q.id, r.word FROM q CROSS JOIN LATERAL echo(q.text) AS r")
            .await?
            .create_physical_plan()
            .await?;
        let plan = displayable(plan.as_ref()).indent(true).to_string();
        assert_contains!(&plan, "AsyncFuncExec");
        assert_contains!(&plan, "lateral_echo(text@1)");
        assert_contains!(&plan, "UnnestExec");
        Ok(())
    }

    #[tokio::test]
    async fn left_join_keeps_rows_whose_call_is_empty() -> Result<()> {
        let ctx = context(Echo::default())?;
        // `generate_series(1, 0)` returns no rows.
        let batches = ctx
            .sql(
                "SELECT q.id, s.value FROM q \
                 LEFT JOIN LATERAL generate_series(1, q.id - 2) AS s ON true \
                 ORDER BY q.id, s.value",
            )
            .await?
            .collect()
            .await?;
        assert_batches_eq!(
            [
                "+----+-------+",
                "| id | value |",
                "+----+-------+",
                "| 1  |       |",
                "| 2  |       |",
                "| 3  | 1     |",
                "+----+-------+",
            ],
            &batches
        );
        Ok(())
    }

    #[tokio::test]
    async fn schema_that_differs_from_the_planned_schema_is_an_error() -> Result<()> {
        let ctx = context(Echo {
            calls: Arc::default(),
            change_schema: true,
        })?;
        let err = ctx
            .sql("SELECT r.word FROM q CROSS JOIN LATERAL echo(q.text) AS r")
            .await?
            .collect()
            .await
            .unwrap_err();
        assert_contains!(
            err.to_string(),
            "LATERAL table function 'echo' returned columns (word Int64), but the query was planned for (word Utf8)"
        );
        Ok(())
    }
}
