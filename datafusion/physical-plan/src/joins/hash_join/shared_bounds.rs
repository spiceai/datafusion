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

//! Utilities for shared bounds. Used in dynamic filter pushdown in Hash Joins.
// TODO: include the link to the Dynamic Filter blog post.

use std::any::Any;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::joins::PartitionMode;
use crate::ExecutionPlan;
use crate::ExecutionPlanProperties;

use arrow::array::Array;
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{ColumnarValue, Operator};
use datafusion_physical_expr::expressions::InListExpr;
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::expressions::{lit, BinaryExpr, DynamicFilterPhysicalExpr};
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};

use itertools::Itertools;
use parking_lot::Mutex;

/// A wrapper around InListExpr that displays a summarized count in explain plans
/// instead of listing all items, which could be millions of values.
///
/// This struct delegates all PhysicalExpr operations to the inner InListExpr
/// but provides a custom Display implementation for better explain plan readability.
#[derive(Debug, Clone)]
struct CompactInListExpr {
    inner: Arc<InListExpr>,
}

impl CompactInListExpr {
    fn new(inner: Arc<InListExpr>) -> Self {
        Self { inner }
    }
}

impl fmt::Display for CompactInListExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let item_count = self.inner.list().len();
        if self.inner.negated() {
            write!(f, "{} NOT IN (<{} items>)", self.inner.expr(), item_count)
        } else {
            write!(f, "{} IN (<{} items>)", self.inner.expr(), item_count)
        }
    }
}

impl PhysicalExpr for CompactInListExpr {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn data_type(&self, input_schema: &Schema) -> Result<arrow::datatypes::DataType> {
        self.inner.data_type(input_schema)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.inner.nullable(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        self.inner.evaluate(batch)
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        self.inner.children()
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        // Create a new InListExpr with updated children
        let new_inner = Arc::new(InListExpr::new(
            Arc::clone(&children[0]),
            children[1..].to_vec(),
            self.inner.negated(),
            None,
        ));
        Ok(Arc::new(CompactInListExpr::new(new_inner)))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt_sql(f)
    }
}

impl PartialEq for CompactInListExpr {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl Eq for CompactInListExpr {}

impl Hash for CompactInListExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

/// Represents the minimum and maximum values for a specific column.
/// Used in dynamic filter pushdown to establish value boundaries.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ColumnBounds {
    /// The minimum value observed for this column
    min: ScalarValue,
    /// The maximum value observed for this column  
    max: ScalarValue,

    expr_array: Option<Arc<dyn Array>>,
}

impl ColumnBounds {
    pub(crate) fn new(min: ScalarValue, max: ScalarValue) -> Self {
        Self {
            min,
            max,
            expr_array: None,
        }
    }

    pub(crate) fn with_expr_array(mut self, expr_array: Option<Arc<dyn Array>>) -> Self {
        self.expr_array = expr_array;
        self
    }
}

/// Represents the bounds for all join key columns from a single partition.
/// This contains the min/max values computed from one partition's build-side data.
#[derive(Debug, Clone)]
pub(crate) struct PartitionBounds {
    /// Partition identifier for debugging and determinism (not strictly necessary)
    partition: usize,
    /// Min/max bounds for each join key column in this partition.
    /// Index corresponds to the join key expression index.
    column_bounds: Vec<ColumnBounds>,
}

impl PartitionBounds {
    pub(crate) fn new(partition: usize, column_bounds: Vec<ColumnBounds>) -> Self {
        Self {
            partition,
            column_bounds,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.column_bounds.len()
    }

    pub(crate) fn get_column_bounds(&self, index: usize) -> Option<&ColumnBounds> {
        self.column_bounds.get(index)
    }
}

/// Coordinates dynamic filter bounds collection across multiple partitions
///
/// This structure ensures that dynamic filters are built with complete information from all
/// relevant partitions before being applied to probe-side scans. Incomplete filters would
/// incorrectly eliminate valid join results.
///
/// ## Synchronization Strategy
///
/// 1. Each partition computes bounds from its build-side data
/// 2. Bounds are stored in the shared HashMap (indexed by partition_id)  
/// 3. A counter tracks how many partitions have reported their bounds
/// 4. When the last partition reports (completed == total), bounds are merged and filter is updated
///
/// ## Partition Counting
///
/// The `total_partitions` count represents how many times `collect_build_side` will be called:
/// - **CollectLeft**: Number of output partitions (each accesses shared build data)
/// - **Partitioned**: Number of input partitions (each builds independently)
///
/// ## Thread Safety
///
/// All fields use a single mutex to ensure correct coordination between concurrent
/// partition executions.
pub(crate) struct SharedBoundsAccumulator {
    /// Shared state protected by a single mutex to avoid ordering concerns
    inner: Mutex<SharedBoundsState>,
    /// Total number of partitions.
    /// Need to know this so that we can update the dynamic filter once we are done
    /// building *all* of the hash tables.
    total_partitions: usize,
    /// Dynamic filter for pushdown to probe side
    dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
    /// Right side join expressions needed for creating filter bounds
    on_right: Vec<PhysicalExprRef>,
}

/// State protected by SharedBoundsAccumulator's mutex
struct SharedBoundsState {
    /// Bounds from completed partitions.
    /// Each element represents the column bounds computed by one partition.
    bounds: Vec<PartitionBounds>,
    /// Number of partitions that have reported completion.
    completed_partitions: usize,
}

impl SharedBoundsAccumulator {
    /// Creates a new SharedBoundsAccumulator configured for the given partition mode
    ///
    /// This method calculates how many times `collect_build_side` will be called based on the
    /// partition mode's execution pattern. This count is critical for determining when we have
    /// complete information from all partitions to build the dynamic filter.
    ///
    /// ## Partition Mode Execution Patterns
    ///
    /// - **CollectLeft**: Build side is collected ONCE from partition 0 and shared via `OnceFut`
    ///   across all output partitions. Each output partition calls `collect_build_side` to access
    ///   the shared build data. Expected calls = number of output partitions.
    ///
    /// - **Partitioned**: Each partition independently builds its own hash table by calling
    ///   `collect_build_side` once. Expected calls = number of build partitions.
    ///
    /// - **Auto**: Placeholder mode resolved during optimization. Uses 1 as safe default since
    ///   the actual mode will be determined and a new bounds_accumulator created before execution.
    ///
    /// ## Why This Matters
    ///
    /// We cannot build a partial filter from some partitions - it would incorrectly eliminate
    /// valid join results. We must wait until we have complete bounds information from ALL
    /// relevant partitions before updating the dynamic filter.
    pub(crate) fn new_from_partition_mode(
        partition_mode: PartitionMode,
        left_child: &dyn ExecutionPlan,
        right_child: &dyn ExecutionPlan,
        dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
        on_right: Vec<PhysicalExprRef>,
    ) -> Self {
        // Troubleshooting: If partition counts are incorrect, verify this logic matches
        // the actual execution pattern in collect_build_side()
        let expected_calls = match partition_mode {
            // Each output partition accesses shared build data
            PartitionMode::CollectLeft => {
                right_child.output_partitioning().partition_count()
            }
            // Each partition builds its own data
            PartitionMode::Partitioned => {
                left_child.output_partitioning().partition_count()
            }
            // Default value, will be resolved during optimization (does not exist once `execute()` is called; will be replaced by one of the other two)
            PartitionMode::Auto => unreachable!("PartitionMode::Auto should not be present at execution time. This is a bug in DataFusion, please report it!"),
        };
        Self {
            inner: Mutex::new(SharedBoundsState {
                bounds: Vec::with_capacity(expected_calls),
                completed_partitions: 0,
            }),
            total_partitions: expected_calls,
            dynamic_filter,
            on_right,
        }
    }

    /// Create a filter expression from individual partition bounds using OR logic.
    ///
    /// This creates a filter where each partition's bounds form a conjunction (AND)
    /// of column range predicates, and all partitions are combined with OR.
    ///
    /// For example, with 2 partitions and 2 columns:
    /// ((col0 >= p0_min0 AND col0 <= p0_max0 AND col1 >= p0_min1 AND col1 <= p0_max1)
    ///  OR
    ///  (col0 >= p1_min0 AND col0 <= p1_max0 AND col1 >= p1_min1 AND col1 <= p1_max1))
    pub(crate) fn create_filter_from_partition_bounds(
        &self,
        bounds: &[PartitionBounds],
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if bounds.is_empty() {
            return Ok(lit(true));
        }

        // Create a predicate for each partition
        let mut partition_predicates = Vec::with_capacity(bounds.len());

        for partition_bounds in bounds.iter().sorted_by_key(|b| b.partition) {
            // Create range predicates for each join key in this partition
            let mut column_predicates = Vec::with_capacity(partition_bounds.len());

            for (col_idx, right_expr) in self.on_right.iter().enumerate() {
                if let Some(column_bounds) = partition_bounds.get_column_bounds(col_idx) {
                    if let Some(expr_values) = &column_bounds.expr_array {
                        // convert the dyn Array into a PhysicalExpr of Vec<ScalarValue>
                        let expr_values: Vec<ScalarValue> = (0..expr_values.len())
                            .map(|i| ScalarValue::try_from_array(expr_values.as_ref(), i))
                            .collect::<Result<Vec<ScalarValue>>>()?;

                        let expr_values: Vec<Arc<dyn PhysicalExpr>> = expr_values
                            .into_iter()
                            .map(|sv| Arc::new(Literal::new(sv)) as _)
                            .collect();

                        // Create IN list predicate: col IN (expr_values)
                        // Wrap in CompactInListExpr to avoid displaying millions of items in explain plans
                        let in_list_expr = Arc::new(InListExpr::new(
                            Arc::clone(right_expr),
                            expr_values,
                            false,
                            None,
                        ));
                        let compact_in_list =
                            Arc::new(CompactInListExpr::new(in_list_expr))
                                as Arc<dyn PhysicalExpr>;
                        column_predicates.push(compact_in_list);
                        continue;
                    }

                    // Create predicate: col >= min AND col <= max
                    let min_expr = Arc::new(BinaryExpr::new(
                        Arc::clone(right_expr),
                        Operator::GtEq,
                        lit(column_bounds.min.clone()),
                    )) as Arc<dyn PhysicalExpr>;
                    let max_expr = Arc::new(BinaryExpr::new(
                        Arc::clone(right_expr),
                        Operator::LtEq,
                        lit(column_bounds.max.clone()),
                    )) as Arc<dyn PhysicalExpr>;
                    let range_expr =
                        Arc::new(BinaryExpr::new(min_expr, Operator::And, max_expr))
                            as Arc<dyn PhysicalExpr>;
                    column_predicates.push(range_expr);
                }
            }

            // Combine all column predicates for this partition with AND
            if !column_predicates.is_empty() {
                let partition_predicate = column_predicates
                    .into_iter()
                    .reduce(|acc, pred| {
                        Arc::new(BinaryExpr::new(acc, Operator::And, pred))
                            as Arc<dyn PhysicalExpr>
                    })
                    .unwrap();
                partition_predicates.push(partition_predicate);
            }
        }

        // Combine all partition predicates with OR
        let combined_predicate = partition_predicates
            .into_iter()
            .reduce(|acc, pred| {
                Arc::new(BinaryExpr::new(acc, Operator::Or, pred))
                    as Arc<dyn PhysicalExpr>
            })
            .unwrap_or_else(|| lit(true));

        Ok(combined_predicate)
    }

    /// Report bounds from a completed partition and update dynamic filter if all partitions are done
    ///
    /// This method coordinates the dynamic filter updates across all partitions. It stores the
    /// bounds from the current partition, increments the completion counter, and when all
    /// partitions have reported, creates an OR'd filter from individual partition bounds.
    ///
    /// # Arguments
    /// * `partition_bounds` - The bounds computed by this partition (if any)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err if filter update failed
    pub(crate) fn report_partition_bounds(
        &self,
        partition: usize,
        partition_bounds: Option<Vec<ColumnBounds>>,
    ) -> Result<()> {
        let mut inner = self.inner.lock();

        // Store bounds in the accumulator - this runs once per partition
        if let Some(bounds) = partition_bounds {
            // Only push actual bounds if they exist
            inner.bounds.push(PartitionBounds::new(partition, bounds));
        }

        // Increment the completion counter
        // Even empty partitions must report to ensure proper termination
        inner.completed_partitions += 1;
        let completed = inner.completed_partitions;
        let total_partitions = self.total_partitions;

        // Critical synchronization point: Only update the filter when ALL partitions are complete
        // Troubleshooting: If you see "completed > total_partitions", check partition
        // count calculation in new_from_partition_mode() - it may not match actual execution calls
        if completed == total_partitions && !inner.bounds.is_empty() {
            let filter_expr = self.create_filter_from_partition_bounds(&inner.bounds)?;
            self.dynamic_filter.update(filter_expr)?;
        }

        Ok(())
    }
}

impl fmt::Debug for SharedBoundsAccumulator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SharedBoundsAccumulator")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::cast::as_boolean_array;
    use datafusion_physical_expr::expressions::Column;

    #[test]
    fn test_compact_in_list_display() {
        // Create a column expression
        let col_expr = Arc::new(Column::new("col1", 0)) as Arc<dyn PhysicalExpr>;

        // Create a list of literal values (simulating a large list)
        let values: Vec<Arc<dyn PhysicalExpr>> = (0..1000)
            .map(|i| Arc::new(Literal::new(ScalarValue::Int32(Some(i)))) as _)
            .collect();

        // Create an InListExpr
        let in_list = Arc::new(InListExpr::new(
            Arc::clone(&col_expr),
            values.clone(),
            false,
            None,
        ));

        // Create a CompactInListExpr
        let compact_in_list = CompactInListExpr::new(Arc::clone(&in_list));

        // Test Display implementation
        let display_str = format!("{}", compact_in_list);
        assert!(
            display_str.contains("<1000 items>"),
            "Display should show item count, got: {}",
            display_str
        );
        assert!(
            display_str.contains("IN"),
            "Display should contain IN keyword, got: {}",
            display_str
        );

        // Test that the regular InListExpr shows the full list (for comparison)
        let regular_display = format!("{}", in_list);
        assert!(
            !regular_display.contains("<1000 items>"),
            "Regular InListExpr should not use compact format"
        );
    }

    #[test]
    fn test_compact_in_list_display_negated() {
        let col_expr = Arc::new(Column::new("col1", 0)) as Arc<dyn PhysicalExpr>;

        let values: Vec<Arc<dyn PhysicalExpr>> = (0..500)
            .map(|i| Arc::new(Literal::new(ScalarValue::Int32(Some(i)))) as _)
            .collect();

        // Create a negated InListExpr
        let in_list = Arc::new(InListExpr::new(
            Arc::clone(&col_expr),
            values,
            true, // negated
            None,
        ));

        let compact_in_list = CompactInListExpr::new(in_list);
        let display_str = format!("{}", compact_in_list);

        assert!(
            display_str.contains("NOT IN"),
            "Display should show NOT IN for negated, got: {}",
            display_str
        );
        assert!(
            display_str.contains("<500 items>"),
            "Display should show item count, got: {}",
            display_str
        );
    }

    #[test]
    fn test_compact_in_list_functionality() {
        use arrow::array::Int32Array;
        use arrow::record_batch::RecordBatch;

        // Create a schema and record batch
        let schema = Arc::new(Schema::new(vec![Field::new(
            "col1",
            DataType::Int32,
            false,
        )]));
        let col_expr = Arc::new(Column::new("col1", 0)) as Arc<dyn PhysicalExpr>;

        let values: Vec<Arc<dyn PhysicalExpr>> = vec![
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))) as _,
            Arc::new(Literal::new(ScalarValue::Int32(Some(2)))) as _,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))) as _,
        ];

        let in_list =
            Arc::new(InListExpr::new(Arc::clone(&col_expr), values, false, None));

        let compact_in_list =
            Arc::new(CompactInListExpr::new(in_list)) as Arc<dyn PhysicalExpr>;

        // Create test data: [1, 2, 4, 5]
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 4, 5]))],
        )
        .unwrap();

        // Evaluate the expression
        let result = compact_in_list.evaluate(&batch).unwrap();
        let result_array = result.into_array(4).unwrap();
        let bool_array = as_boolean_array(&result_array).unwrap();

        // Should return [true, true, false, false] for values [1, 2, 4, 5]
        assert_eq!(bool_array.value(0), true);
        assert_eq!(bool_array.value(1), true);
        assert_eq!(bool_array.value(2), false);
        assert_eq!(bool_array.value(3), false);
    }
}
