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

//! Streaming execution of input-aware table functions.

use arrow::array::{
    Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions, UInt32Array,
    new_null_array,
};
use arrow::compute::{filter, filter_record_batch, take};
use arrow::datatypes::{SchemaRef, UInt32Type};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, exec_err, internal_err};
use datafusion_execution::{TaskContext, async_try_stream};
use datafusion_expr::{
    Expr,
    table_function::{BoundTableFunction, TableFunctionSource},
};
use datafusion_physical_expr::EquivalenceProperties;
use futures::{TryStreamExt, stream::BoxStream};
use std::{fmt, sync::Arc};

use crate::execution_plan::EmissionType;
use crate::stream::RecordBatchStreamAdapter;
use crate::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PhysicalExpr, PlanProperties, SendableRecordBatchStream, apply_expression_roots,
};

/// Rows produced for one input batch. Each output row has one non-null, zero-based
/// index into that input batch. A function may emit several chunks per input batch.
pub struct TableFunctionRows {
    /// Projected function output.
    pub batch: RecordBatch,
    /// Input row that produced each output row.
    pub input_rows: UInt32Array,
}

/// An implementation bound to the static arguments of one table-function call.
/// Each invocation receives a batch of runtime argument arrays. Results are polled
/// on demand and dropping the stream cancels the invocation.
pub trait TableFunctionEvaluator: fmt::Debug + Send + Sync {
    /// Full output schema before projection.
    fn schema(&self) -> SchemaRef;
    /// Evaluate one input batch. Each row is a separate invocation, including
    /// rows whose arguments compare equal. Output batches use the given projection.
    fn evaluate(
        &self,
        arguments: Vec<ArrayRef>,
        num_rows: usize,
        projection: &[usize],
        context: Arc<TaskContext>,
    ) -> Result<BoxStream<'static, Result<TableFunctionRows>>>;
}

/// Runtime arguments and evaluator produced by binding a table function.
#[derive(Debug)]
pub struct TableFunctionBinding {
    /// Expressions evaluated against the child plan.
    pub arguments: Vec<Expr>,
    /// Bound evaluator. Shared between executions, with per-stream state in evaluate.
    pub evaluator: Arc<dyn TableFunctionEvaluator>,
}

#[derive(Debug)]
struct BoundEvaluator(Arc<dyn TableFunctionEvaluator>);
impl TableFunctionSource for BoundEvaluator {
    fn schema(&self) -> SchemaRef {
        self.0.schema()
    }
}

impl TableFunctionBinding {
    /// Erase physical types for logical planning.
    pub fn into_logical(self) -> BoundTableFunction {
        BoundTableFunction {
            arguments: self.arguments,
            source: Arc::new(BoundEvaluator(self.evaluator)),
        }
    }
}

/// Applies a bound function to one child partition at a time.
#[derive(Debug, Clone)]
pub struct TableFunctionExec {
    input: Arc<dyn ExecutionPlan>,
    evaluator: Arc<dyn TableFunctionEvaluator>,
    arguments: Vec<Arc<dyn PhysicalExpr>>,
    predicate: Option<Arc<dyn PhysicalExpr>>,
    preserve_unmatched: bool,
    projection: Vec<usize>,
    properties: Arc<PlanProperties>,
}

impl TableFunctionExec {
    /// Create the physical operator for a bound call.
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        call: &datafusion_expr::table_function::TableFunctionApply,
        arguments: Vec<Arc<dyn PhysicalExpr>>,
        predicate: Option<Arc<dyn PhysicalExpr>>,
    ) -> Result<Self> {
        let Some(bound) = (call.function.source.as_ref() as &dyn std::any::Any)
            .downcast_ref::<BoundEvaluator>()
        else {
            return internal_err!(
                "Unsupported table function source for {}",
                call.function_name
            );
        };
        let properties = Self::properties_for(&input, Arc::clone(call.schema.inner()));
        Ok(Self {
            input,
            evaluator: Arc::clone(&bound.0),
            arguments,
            predicate,
            preserve_unmatched: call.preserve_unmatched,
            projection: call.projection.clone(),
            properties,
        })
    }
    fn properties_for(
        input: &Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> Arc<PlanProperties> {
        Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(
                input.output_partitioning().partition_count(),
            ),
            EmissionType::Incremental,
            input.boundedness(),
        ))
    }
}

impl DisplayAs for TableFunctionExec {
    fn fmt_as(&self, _: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TableFunctionExec: {:?}, projection={:?}",
            self.evaluator, self.projection
        )
    }
}
impl ExecutionPlan for TableFunctionExec {
    fn name(&self) -> &'static str {
        "TableFunctionExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(self.arguments.iter().chain(&self.predicate), f)
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = children.try_into().map_err(|_| {
            datafusion_common::internal_datafusion_err!(
                "Expected one table function input"
            )
        })?;
        let properties = Self::properties_for(&input, self.schema());
        Ok(Arc::new(Self {
            input,
            properties,
            ..self.as_ref().clone()
        }))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut input = self.input.execute(partition, Arc::clone(&context))?;
        let call = self.clone();
        let expected = call.evaluator.schema().project(&call.projection)?;
        let output = async_try_stream(move |mut out| async move {
            while let Some(batch) = input.try_next().await? {
                if batch.num_rows() > u32::MAX as usize {
                    return exec_err!(
                        "Table function input exceeds the UInt32 row index range"
                    );
                }
                let arguments = call
                    .arguments
                    .iter()
                    .map(|expr| expr.evaluate(&batch)?.into_array(batch.num_rows()))
                    .collect::<Result<_>>()?;
                let mut results = call.evaluator.evaluate(
                    arguments,
                    batch.num_rows(),
                    &call.projection,
                    Arc::clone(&context),
                )?;
                let mut matched = call
                    .preserve_unmatched
                    .then(|| vec![false; batch.num_rows()]);
                while let Some(rows) = results.try_next().await? {
                    // A ready stream of chunks rejected by ON can otherwise monopolize a task.
                    tokio::task::coop::consume_budget().await;
                    let mut ids = rows.input_rows;
                    if rows.batch.schema().as_ref() != &expected
                        || ids.len() != rows.batch.num_rows()
                        || ids.null_count() != 0
                        || ids.values().iter().any(|&i| i as usize >= batch.num_rows())
                    {
                        return exec_err!(
                            "Invalid result schema or input row mapping from table function"
                        );
                    }
                    let mut joined =
                        join_rows(&call.schema(), &batch, &ids, rows.batch.columns())?;
                    if let Some(predicate) = &call.predicate {
                        let mask =
                            predicate.evaluate(&joined)?.into_array(joined.num_rows())?;
                        let mask = mask.as_boolean_opt().ok_or_else(|| {
                            datafusion_common::exec_datafusion_err!(
                                "Table function join predicate must be Boolean"
                            )
                        })?;
                        joined = filter_record_batch(&joined, mask)?;
                        if matched.is_some() {
                            ids =
                                filter(&ids, mask)?.as_primitive::<UInt32Type>().clone();
                        }
                    }
                    if let Some(matched) = &mut matched {
                        for &i in ids.values() {
                            matched[i as usize] = true;
                        }
                    }
                    if joined.num_rows() > 0 {
                        out.emit(joined).await;
                    }
                }
                if let Some(matched) = matched {
                    let missing = UInt32Array::from_iter_values(
                        matched
                            .into_iter()
                            .enumerate()
                            .filter(|(_, hit)| !hit)
                            .map(|(i, _)| i as u32),
                    );
                    if !missing.is_empty() {
                        let nulls = expected
                            .fields()
                            .iter()
                            .map(|f| new_null_array(f.data_type(), missing.len()))
                            .collect::<Vec<_>>();
                        out.emit(join_rows(&call.schema(), &batch, &missing, &nulls)?)
                            .await;
                    }
                }
            }
            Ok(())
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            output,
        )))
    }
}

fn join_rows(
    schema: &SchemaRef,
    input: &RecordBatch,
    rows: &UInt32Array,
    output: &[ArrayRef],
) -> Result<RecordBatch> {
    let mut columns = input
        .columns()
        .iter()
        .map(|a| take(a.as_ref(), rows, None))
        .collect::<arrow::error::Result<Vec<_>>>()?;
    columns.extend_from_slice(output);
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )?)
}
