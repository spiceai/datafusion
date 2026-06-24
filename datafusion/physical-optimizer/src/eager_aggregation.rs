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

//! [`EagerAggregation`]: a **cost-based**, physical alternative to the logical
//! eager-aggregation rule. It pushes a partial aggregation below a join when a
//! statistics-based cost model predicts a win, then lets the existing two-phase
//! aggregate machinery (and `EnforceDistribution`) finalize the result.
//!
//! # Why physical / cost-based
//!
//! Eager aggregation is profitable only sometimes: pushing an aggregate below a
//! *selective* join (one whose output is much smaller than the pre-aggregated
//! row count) is wasted work — the join would have discarded those rows anyway.
//! That decision needs cardinality estimates, which only exist at physical
//! planning time (the logical layer is statistics-blind). This rule reads
//! `Statistics` and fires only when the push-down is estimated to pay off:
//!
//! ```text
//!     join_out > grouped                       (not behind a selective join)
//!  && push_rows >= grouped * min_reduction      (a substantial, not token, reduction)
//!  && grouped <= max_pushed_groups (if capped)  (bounded pre-aggregation memory)
//! ```
//!
//! where `push_rows` is the push side's row count, `grouped` the estimated
//! pre-aggregated group count, and `join_out` the rows reaching the aggregate.
//! The reduction factor and the absolute group cap are tunable via
//! `datafusion.optimizer.eager_aggregation_min_reduction_factor` and
//! `…_max_pushed_groups`. The factor guard matters because a push that barely
//! reduces (e.g. 1.7x) yet materializes millions of groups costs more than it
//! saves.
//!
//! # Scope (first milestone)
//!
//! Matches `Final/FinalPartitioned( Partial( HashJoinExec ) )` (the shape the
//! default physical planner emits for `GROUP BY` over a join) and pushes a
//! pre-aggregation into the join input that supplies all aggregate arguments.
//! Supports `SUM`/`MIN`/`MAX` (whose merge reuses the same UDAF, so no function
//! registry is needed), inner joins with no residual (non-equi) filter, and
//! plain-column join keys (group-by may be any expression — its push-side
//! columns are pushed into the pre-aggregation grouping). It peels an intervening chain of
//! column-only `ProjectionExec`s between the partial aggregate and the join
//! (the shape the planner emits before `ProjectionPushdown`). Anything outside
//! this leaves the plan unchanged.
//!
//! # Cost-gated cascade
//!
//! The pushed pre-aggregation has the same `Final(Partial(..))` shape this rule
//! matches, so when its input is itself a join it re-matches and the cost gate
//! runs again. The rewrite therefore **cascades down a join chain as far as each
//! level's gate allows** — compounding the reduction across multi-join (e.g.
//! TPC-H Q3/Q5) shapes — but is *self-limiting*: it stops at a selective join,
//! where `join_output <= pre_aggregated_rows`. (This is why the decision must be
//! physical/cost-based: a logical cascade — the existing logical rule — cannot
//! tell a beneficial chain from a wasteful push below a selective join.)

use std::sync::Arc;

use crate::PhysicalOptimizerRule;

use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_expr::JoinType;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::aggregate::{AggregateExprBuilder, AggregateFunctionExpr};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::joins::HashJoinExec;

/// Cost-based physical eager-aggregation rule. See the [module docs](self).
#[derive(Debug, Default)]
pub struct EagerAggregation {}

impl EagerAggregation {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PhysicalOptimizerRule for EagerAggregation {
    fn name(&self) -> &str {
        "eager_aggregation"
    }

    fn schema_check(&self) -> bool {
        // The rewrite must preserve the output schema exactly; let the framework
        // assert it.
        true
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_eager_aggregation {
            return Ok(plan);
        }
        plan.transform_down(|plan| match try_push_aggregate(&plan, config)? {
            Some(new_plan) => Ok(Transformed::yes(new_plan)),
            None => Ok(Transformed::no(plan)),
        })
        .map(|t| t.data)
    }
}

/// Side of the join the aggregation is pushed into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Left,
    Right,
}

/// Outcome of the statistics-based cost gate.
enum CostGate {
    Allow {
        push_rows: usize,
        grouped: usize,
        join_out: usize,
    },
    Decline(String),
}

/// Try to rewrite `Final(Partial(HashJoin))` by pushing a pre-aggregation into
/// one join input. Returns `Ok(None)` if the pattern/guards/cost-gate do not
/// hold.
fn try_push_aggregate(
    plan: &Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    // Match Final/FinalPartitioned <- Partial <- HashJoinExec.
    let Some(top_final) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(None);
    };
    if !matches!(
        top_final.mode(),
        AggregateMode::Final | AggregateMode::FinalPartitioned
    ) {
        return Ok(None);
    }
    let Some(top_partial) = top_final.input().downcast_ref::<AggregateExec>() else {
        log::debug!(
            target: "eager_aggregation",
            "shape: final aggregate's input is {}, not a partial aggregate",
            top_final.input().name()
        );
        return Ok(None);
    };
    if *top_partial.mode() != AggregateMode::Partial {
        return Ok(None);
    }

    // Between the partial aggregate and the join the planner emits a *chain* of
    // column-only `ProjectionExec`s (this rule runs before ProjectionPushdown
    // folds them into the join). Peel the whole chain down to the HashJoinExec,
    // composing the column-index map so aggregate-input indices map back to raw
    // (left ++ right) join indices. We rebuild the join with full output and drop
    // these projections; ProjectionPushdown re-prunes later.
    let agg_input = top_partial.input();
    let mut node: Arc<dyn ExecutionPlan> = Arc::clone(agg_input);
    let mut agg_to_raw: Vec<usize> = (0..agg_input.schema().fields().len()).collect();
    loop {
        let next = {
            if node.downcast_ref::<HashJoinExec>().is_some() {
                break;
            }
            let Some(proj) = node
                .downcast_ref::<datafusion_physical_plan::projection::ProjectionExec>()
            else {
                log::debug!(
                    target: "eager_aggregation",
                    "shape: aggregate input chain reaches {}, not a (projected) hash join",
                    node.name()
                );
                return Ok(None);
            };
            // Column-only projection: output index -> input column index.
            let mut proj_map = Vec::with_capacity(proj.expr().len());
            for pe in proj.expr() {
                let Some(c) = pe.expr.downcast_ref::<Column>() else {
                    log::debug!(
                        target: "eager_aggregation",
                        "shape: intervening projection has a computed expression (not a plain column)"
                    );
                    return Ok(None);
                };
                proj_map.push(c.index());
            }
            for m in agg_to_raw.iter_mut() {
                *m = proj_map[*m];
            }
            Arc::clone(proj.input())
        };
        node = next;
    }
    let join = node.downcast_ref::<HashJoinExec>().unwrap();
    // Apply the join's own (folded) projection: node output index -> raw index.
    if let Some(p) = join.projection.as_ref() {
        for m in agg_to_raw.iter_mut() {
            *m = p[*m];
        }
    }

    // From here the shape matched (Final <- Partial <- [Projection] <- Join), so
    // any decline is informative: log the reason at debug level. Enable with
    // `RUST_LOG=datafusion_physical_optimizer::eager_aggregation=debug`.
    macro_rules! decline {
        ($($arg:tt)*) => {{
            log::debug!(target: "eager_aggregation", "decline: {}", format!($($arg)*));
            return Ok(None);
        }};
    }

    // Guards: inner join, no extra filter.
    if *join.join_type() != JoinType::Inner {
        decline!("join_type is {:?}, only Inner supported", join.join_type());
    }
    if join.filter().is_some() {
        decline!("join carries a non-equi filter");
    }

    let left = join.left();
    let right = join.right();
    let left_len = left.schema().fields().len();

    // Map an aggregate-input column index to the raw left ++ right join index.
    let raw_index = |agg_idx: usize| -> usize { agg_to_raw[agg_idx] };

    // Equi-keys must be plain columns.
    let mut left_key_cols: Vec<Column> = Vec::with_capacity(join.on().len());
    let mut right_key_cols: Vec<Column> = Vec::with_capacity(join.on().len());
    if join.on().is_empty() {
        decline!("join has no equi-keys");
    }
    for (l, r) in join.on() {
        let (Some(lc), Some(rc)) =
            (l.downcast_ref::<Column>(), r.downcast_ref::<Column>())
        else {
            decline!("join key is not a plain column");
        };
        left_key_cols.push(lc.clone());
        right_key_cols.push(rc.clone());
    }

    // Aggregates must be decomposable with a self-merge (SUM/MIN/MAX) and have
    // no per-aggregate filter. Collect their argument columns to pick a side.
    let aggrs = top_partial.aggr_expr();
    if aggrs.is_empty() {
        decline!("no aggregate expressions");
    }
    if top_partial.filter_expr().iter().any(Option::is_some) {
        decline!("aggregate has a FILTER clause");
    }
    let mut arg_indices: Vec<usize> = Vec::new();
    for agg in aggrs {
        if !is_self_mergeable(agg) {
            decline!(
                "aggregate {:?} (distinct={}) is not self-mergeable (only SUM/MIN/MAX)",
                agg.fun().name(),
                agg.is_distinct()
            );
        }
        for e in agg.expressions() {
            collect_column_indices(&e, &mut arg_indices);
        }
    }
    // All aggregate arguments must come from a single side (compared in raw
    // left ++ right index space).
    let all_left = arg_indices.iter().all(|&i| raw_index(i) < left_len);
    let all_right = arg_indices.iter().all(|&i| raw_index(i) >= left_len);
    let side = if all_left && !arg_indices.is_empty() {
        Side::Left
    } else if all_right && !arg_indices.is_empty() {
        Side::Right
    } else {
        decline!(
            "aggregate args span both sides or are empty: raw_indices={:?} left_len={left_len}",
            arg_indices
                .iter()
                .map(|&i| raw_index(i))
                .collect::<Vec<_>>()
        );
    };

    // Group-by may include computed expressions (e.g. `extract(year from ...)`),
    // not just plain columns. Collect the columns each group expression
    // references; the push-side ones are added to the pushdown grouping below.
    // Grouping the pre-aggregation by those underlying columns refines grouping
    // by the expression (the expression is constant within each pre-agg group),
    // so re-aggregating above the join reproduces the original groups. Keep-side
    // group expressions simply pass through to the final grouping unchanged.
    let group = top_partial.group_expr();
    let mut group_cols: Vec<Column> = Vec::new();
    for (e, _) in group.expr() {
        collect_columns(e, &mut group_cols);
    }

    let (push_plan, push_keys) = match side {
        Side::Left => (left, &left_key_cols),
        Side::Right => (right, &right_key_cols),
    };

    // Build the pushed-down grouping over the push side: its join keys plus any
    // group-by columns originating from that side. Columns are expressed in the
    // push side's own schema (left columns keep their index; right columns are
    // shifted by `left_len` in the join output).
    let to_push_index = |join_idx: usize| -> usize {
        match side {
            Side::Left => join_idx,
            Side::Right => join_idx - left_len,
        }
    };
    let mut pushdown_group: Vec<(Arc<dyn PhysicalExpr>, String)> = Vec::new();
    let mut seen: Vec<usize> = Vec::new();
    for k in push_keys {
        if !seen.contains(&k.index()) {
            seen.push(k.index());
            pushdown_group.push((
                Arc::new(k.clone()) as Arc<dyn PhysicalExpr>,
                k.name().to_string(),
            ));
        }
    }
    for c in &group_cols {
        let raw = raw_index(c.index());
        let on_side = (side == Side::Left && raw < left_len)
            || (side == Side::Right && raw >= left_len);
        if on_side {
            let pidx = to_push_index(raw);
            if !seen.contains(&pidx) {
                seen.push(pidx);
                pushdown_group.push((
                    Arc::new(Column::new(c.name(), pidx)) as Arc<dyn PhysicalExpr>,
                    c.name().to_string(),
                ));
            }
        }
    }

    let push_schema = push_plan.schema();

    // Cost gate: only push when the pre-aggregation produces fewer rows than the
    // join emits (otherwise the join is selective enough that pre-aggregating is
    // wasted work). Requires statistics; absent stats => do not fire.
    match cost_gate(join, push_plan, &pushdown_group, config)? {
        CostGate::Allow {
            push_rows,
            grouped,
            join_out,
        } => {
            log::debug!(
                target: "eager_aggregation",
                "accept: side={:?} push_rows={push_rows} grouped={grouped} join_out={join_out}",
                side
            );
        }
        CostGate::Decline(reason) => decline!("cost gate: {reason}"),
    }

    // Build the pushed (pre-)aggregation expressions: reuse each original UDAF,
    // remap its argument columns into the push side's schema, alias to a stable
    // internal name. The merge above reuses the same UDAF over this column.
    let mut pushed_aggrs: Vec<Arc<AggregateFunctionExpr>> =
        Vec::with_capacity(aggrs.len());
    let mut internal_names: Vec<String> = Vec::with_capacity(aggrs.len());
    for (i, agg) in aggrs.iter().enumerate() {
        let internal = format!("__eager_phys_{i}");
        let args = agg
            .expressions()
            .into_iter()
            .map(|e| {
                // Remap join-output indices to push-side indices, then fix by name.
                reassign_expr_columns(e, push_schema.as_ref())
            })
            .collect::<Result<Vec<_>>>()?;
        let built = AggregateExprBuilder::new(Arc::new(agg.fun().clone()), args)
            .schema(Arc::clone(&push_schema))
            .alias(internal.clone())
            .build()?;
        pushed_aggrs.push(Arc::new(built));
        internal_names.push(internal);
    }

    let pushdown_group_by = PhysicalGroupBy::new_single(pushdown_group);

    // Pre-aggregation as Partial -> FinalPartitioned (EnforceDistribution will
    // insert the hash repartition between them).
    let pre_partial = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        pushdown_group_by.clone(),
        pushed_aggrs.clone(),
        vec![None; pushed_aggrs.len()],
        Arc::clone(push_plan),
        Arc::clone(&push_schema),
    )?);
    let pre_final = Arc::new(AggregateExec::try_new(
        AggregateMode::FinalPartitioned,
        pushdown_group_by.as_final(),
        pushed_aggrs,
        vec![None; internal_names.len()],
        pre_partial,
        Arc::clone(&push_schema),
    )?) as Arc<dyn ExecutionPlan>;

    // Rebuild the join with the chosen side replaced by the pre-aggregation.
    let (new_left, new_right) = match side {
        Side::Left => (Arc::clone(&pre_final), Arc::clone(right)),
        Side::Right => (Arc::clone(left), Arc::clone(&pre_final)),
    };
    // Remap the join keys: the pushed side's keys are now group columns in the
    // pre-aggregation's (reordered, narrower) schema.
    let new_on = join
        .on()
        .iter()
        .map(|(l, r)| -> Result<_> {
            let (nl, nr) = match side {
                Side::Left => (
                    reassign_expr_columns(Arc::clone(l), new_left.schema().as_ref())?,
                    Arc::clone(r),
                ),
                Side::Right => (
                    Arc::clone(l),
                    reassign_expr_columns(Arc::clone(r), new_right.schema().as_ref())?,
                ),
            };
            Ok((nl, nr))
        })
        .collect::<Result<Vec<_>>>()?;

    let new_join = Arc::new(HashJoinExec::try_new(
        new_left,
        new_right,
        new_on,
        None,
        join.join_type(),
        None,
        *join.partition_mode(),
        join.null_equality(),
        join.null_aware,
    )?) as Arc<dyn ExecutionPlan>;
    let new_join_schema = new_join.schema();

    // We remap the parent aggregate's columns into the new (full) join output by
    // name, so the names must be unambiguous. Bail out otherwise (physical
    // schemas permit duplicate names).
    {
        let mut names: Vec<&str> = new_join_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            decline!(
                "new join schema has duplicate column names (name-based remap unsafe)"
            );
        }
    }

    // Rebuild the top aggregation to merge the partials. Group columns are
    // remapped into the new join output (by name); each aggregate becomes
    // merge(internal_col) aliased back to the original output name.
    let new_group_exprs = group
        .expr()
        .iter()
        .map(|(e, name)| -> Result<_> {
            Ok((
                reassign_expr_columns(Arc::clone(e), new_join_schema.as_ref())?,
                name.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let new_group_by = PhysicalGroupBy::new_single(new_group_exprs);

    let mut merge_aggrs: Vec<Arc<AggregateFunctionExpr>> =
        Vec::with_capacity(aggrs.len());
    for (i, agg) in aggrs.iter().enumerate() {
        let idx = new_join_schema.index_of(&internal_names[i])?;
        let arg = Arc::new(Column::new(&internal_names[i], idx)) as Arc<dyn PhysicalExpr>;
        let built = AggregateExprBuilder::new(Arc::new(agg.fun().clone()), vec![arg])
            .schema(Arc::clone(&new_join_schema))
            .alias(agg.name().to_string())
            .build()?;
        merge_aggrs.push(Arc::new(built));
    }

    let new_top_partial = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        new_group_by.clone(),
        merge_aggrs.clone(),
        vec![None; merge_aggrs.len()],
        new_join,
        Arc::clone(&new_join_schema),
    )?);
    let new_top_final = Arc::new(AggregateExec::try_new(
        *top_final.mode(),
        new_group_by.as_final(),
        merge_aggrs,
        vec![None; aggrs.len()],
        new_top_partial,
        new_join_schema,
    )?) as Arc<dyn ExecutionPlan>;

    log::debug!(
        target: "eager_aggregation",
        "pushed aggregation into {side:?} side of join"
    );
    Ok(Some(new_top_final))
}

/// True if `agg` merges by re-applying its own function (SUM/MIN/MAX), is not
/// DISTINCT, and has no ORDER BY.
fn is_self_mergeable(agg: &AggregateFunctionExpr) -> bool {
    if agg.is_distinct() {
        return false;
    }
    matches!(agg.fun().name(), "sum" | "min" | "max")
}

/// Collect the indices of all `Column`s referenced by a physical expression.
fn collect_column_indices(expr: &Arc<dyn PhysicalExpr>, out: &mut Vec<usize>) {
    if let Some(c) = expr.downcast_ref::<Column>() {
        out.push(c.index());
    }
    for child in expr.children() {
        collect_column_indices(child, out);
    }
}

/// Collect all `Column`s referenced by a physical expression.
fn collect_columns(expr: &Arc<dyn PhysicalExpr>, out: &mut Vec<Column>) {
    if let Some(c) = expr.downcast_ref::<Column>() {
        out.push(c.clone());
    }
    for child in expr.children() {
        collect_columns(child, out);
    }
}

/// Cost gate: push only when the join's estimated output exceeds the estimated
/// pre-aggregated row count (and pre-aggregation actually reduces rows). Declines
/// (with a reason) when the required statistics are absent (conservative).
///
/// The pre-aggregated row estimate is the product of the **combined** pushdown
/// grouping columns' `distinct_count` (join keys ∪ push-side group columns) — the
/// Q2 lesson: a widened key like `[su_nationkey, s_i_id]` that is near-unique
/// must register as "no reduction".
fn cost_gate(
    join: &HashJoinExec,
    push_plan: &Arc<dyn ExecutionPlan>,
    pushdown_group: &[(Arc<dyn PhysicalExpr>, String)],
    config: &ConfigOptions,
) -> Result<CostGate> {
    let min_factor = config
        .optimizer
        .eager_aggregation_min_reduction_factor
        .max(1);
    let max_groups = config.optimizer.eager_aggregation_max_pushed_groups;
    let push_stats = push_plan.partition_statistics(None)?;
    let push_rows = match push_stats.num_rows {
        Precision::Exact(n) | Precision::Inexact(n) => n,
        Precision::Absent => {
            return Ok(CostGate::Decline("push side num_rows absent".into()));
        }
    };
    let join_out = match join.partition_statistics(None)?.num_rows {
        Precision::Exact(n) | Precision::Inexact(n) => n,
        Precision::Absent => {
            return Ok(CostGate::Decline("join output num_rows absent".into()));
        }
    };

    // Estimated pre-aggregated rows = product of the grouping columns' distinct
    // counts (bounded by the input row count).
    let mut grouped: usize = 1;
    for (e, name) in pushdown_group {
        let Some(col) = e.downcast_ref::<Column>() else {
            return Ok(CostGate::Decline("group column not plain".into()));
        };
        let ndv = match push_stats
            .column_statistics
            .get(col.index())
            .map(|c| &c.distinct_count)
        {
            Some(Precision::Exact(n)) | Some(Precision::Inexact(n)) => *n,
            _ => {
                return Ok(CostGate::Decline(format!(
                    "distinct_count absent for group column {name}"
                )));
            }
        };
        grouped = grouped.saturating_mul(ndv.max(1));
    }
    let grouped = grouped.min(push_rows);

    if push_is_beneficial(push_rows, grouped, join_out, min_factor, max_groups) {
        Ok(CostGate::Allow {
            push_rows,
            grouped,
            join_out,
        })
    } else {
        Ok(CostGate::Decline(format!(
            "not beneficial: push_rows={push_rows} grouped={grouped} join_out={join_out} \
             min_reduction_factor={min_factor} max_pushed_groups={max_groups} \
             (need grouped<push_rows, join_out>grouped, push_rows>=grouped*factor, grouped<=cap)"
        )))
    }
}

/// The cost decision, factored out for testing.
///
/// Push only if all hold:
/// * pre-aggregation reduces rows (`grouped < push_rows`);
/// * the join emits at least as many rows as the pre-aggregation produces
///   (`join_out > grouped`) — otherwise the join is so selective that
///   aggregating its (discarded) input first is wasted work (the original Q2
///   regression: a `MIN` pushed below the selective `region = 'EUROPE'` chain);
/// * the reduction is *substantial*: `push_rows >= grouped * min_factor`. A
///   barely-reducing push (e.g. 1.7x) that still materializes millions of groups
///   is not worth its hash-table cost — this is the second Q2 site;
/// * the pre-aggregated group count is within `max_groups` (0 = no cap) — an
///   absolute guard on the pre-aggregation's memory footprint.
fn push_is_beneficial(
    push_rows: usize,
    grouped: usize,
    join_out: usize,
    min_factor: usize,
    max_groups: usize,
) -> bool {
    grouped < push_rows
        && join_out > grouped
        && push_rows >= grouped.saturating_mul(min_factor)
        && (max_groups == 0 || grouped <= max_groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PhysicalOptimizerRule;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::NullEquality;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::stats::{ColumnStatistics, Statistics};
    use datafusion_functions_aggregate::sum::sum_udaf;
    use datafusion_physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::test::exec::StatisticsExec;

    fn stats_leaf(
        fields: Vec<Field>,
        num_rows: usize,
        distinct: &[Option<usize>],
    ) -> Arc<dyn ExecutionPlan> {
        let schema = Schema::new(fields);
        let column_statistics = distinct
            .iter()
            .map(|d| {
                let cs = ColumnStatistics::new_unknown();
                match d {
                    Some(n) => cs.with_distinct_count(Precision::Inexact(*n)),
                    None => cs,
                }
            })
            .collect();
        let stats = Statistics {
            num_rows: Precision::Inexact(num_rows),
            total_byte_size: Precision::Absent,
            column_statistics,
        };
        Arc::new(StatisticsExec::new(stats, schema))
    }

    /// Build `Final(Partial(Join(fact, dim)))` for `SELECT d_name, SUM(f_amount)
    /// ... GROUP BY d_name`, with the fact join key having `fact_key_ndv` distinct
    /// values (controls whether pre-aggregation reduces rows).
    fn agg_over_join(fact_key_ndv: usize) -> Arc<dyn ExecutionPlan> {
        let fact = stats_leaf(
            vec![
                Field::new("f_dim", DataType::Int32, false),
                Field::new("f_amount", DataType::Float64, true),
            ],
            10_000_000,
            &[Some(fact_key_ndv), None],
        );
        let dim = stats_leaf(
            vec![
                Field::new("d_id", DataType::Int32, false),
                Field::new("d_name", DataType::Utf8, true),
            ],
            100,
            &[Some(100), Some(100)],
        );
        let join = Arc::new(
            HashJoinExec::try_new(
                fact,
                dim,
                vec![(
                    Arc::new(Column::new("f_dim", 0)),
                    Arc::new(Column::new("d_id", 0)),
                )],
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let join_schema = join.schema();

        let sum_expr = Arc::new(
            AggregateExprBuilder::new(
                sum_udaf(),
                vec![Arc::new(Column::new("f_amount", 1))],
            )
            .schema(Arc::clone(&join_schema))
            .alias("sum(f_amount)")
            .build()
            .unwrap(),
        );
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("d_name", 3)) as Arc<dyn PhysicalExpr>,
            "d_name".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group.clone(),
                vec![Arc::clone(&sum_expr)],
                vec![None],
                join,
                Arc::clone(&join_schema),
            )
            .unwrap(),
        );
        let final_group = partial.group_expr().as_final();
        Arc::new(
            AggregateExec::try_new(
                AggregateMode::FinalPartitioned,
                final_group,
                vec![sum_expr],
                vec![None],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>
    }

    fn run_rule(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let mut opts = ConfigOptions::default();
        opts.optimizer.enable_eager_aggregation = true;
        EagerAggregation::new().optimize(plan, &opts).unwrap()
    }

    /// True if any `HashJoinExec` in the tree has an `AggregateExec` child
    /// (i.e. a pre-aggregation was pushed below the join).
    fn join_child_is_aggregate(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if let Some(j) = plan.downcast_ref::<HashJoinExec>() {
            return j.left().downcast_ref::<AggregateExec>().is_some()
                || j.right().downcast_ref::<AggregateExec>().is_some();
        }
        plan.children()
            .into_iter()
            .any(|c| join_child_is_aggregate(c))
    }

    // Accept path: a large fact joined N:1 to a small dim, grouping key reduces
    // 10M rows to ~100 groups, and the join emits ~10M rows -> push fires.
    #[test]
    fn fires_on_beneficial_join() {
        let optimized = run_rule(agg_over_join(100));
        assert!(
            join_child_is_aggregate(&optimized),
            "expected a pre-aggregation pushed below the join"
        );
    }

    // Fix #1: a chain of column-only projections between the partial aggregate
    // and the join (the shape the planner emits before ProjectionPushdown) must
    // be peeled, with the column-index map composed through it, so the rule still
    // reaches the gate and fires.
    #[test]
    fn fires_through_projection_chain() {
        let beneficial = agg_over_join(100);
        // Rebuild the same logical content but insert two column-only projections
        // between the partial aggregate and the join.
        let final_agg = beneficial.downcast_ref::<AggregateExec>().unwrap();
        let partial = final_agg.input().downcast_ref::<AggregateExec>().unwrap();
        let join = Arc::clone(partial.input()); // HashJoinExec, schema [f_dim,f_amount,d_id,d_name]

        // proj1: [f_amount@1, d_name@3, f_dim@0]
        let proj1 = Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                vec![
                    (
                        Arc::new(Column::new("f_amount", 1)) as Arc<dyn PhysicalExpr>,
                        "f_amount".to_string(),
                    ),
                    (Arc::new(Column::new("d_name", 3)), "d_name".to_string()),
                    (Arc::new(Column::new("f_dim", 0)), "f_dim".to_string()),
                ],
                join,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        // proj2: [d_name@1, f_amount@0]
        let proj2 = Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                vec![
                    (
                        Arc::new(Column::new("d_name", 1)) as Arc<dyn PhysicalExpr>,
                        "d_name".to_string(),
                    ),
                    (Arc::new(Column::new("f_amount", 0)), "f_amount".to_string()),
                ],
                proj1,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let proj2_schema = proj2.schema();

        let sum_expr = Arc::new(
            AggregateExprBuilder::new(
                sum_udaf(),
                vec![Arc::new(Column::new("f_amount", 1))],
            )
            .schema(Arc::clone(&proj2_schema))
            .alias("sum(f_amount)")
            .build()
            .unwrap(),
        );
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("d_name", 0)) as Arc<dyn PhysicalExpr>,
            "d_name".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group,
                vec![Arc::clone(&sum_expr)],
                vec![None],
                proj2,
                Arc::clone(&proj2_schema),
            )
            .unwrap(),
        );
        let final_group = partial.group_expr().as_final();
        let plan = Arc::new(
            AggregateExec::try_new(
                AggregateMode::FinalPartitioned,
                final_group,
                vec![sum_expr],
                vec![None],
                partial,
                proj2_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let optimized = run_rule(plan);
        assert!(
            join_child_is_aggregate(&optimized),
            "expected push-down through a column-only projection chain"
        );
    }

    // Group-by a computed expression over a keep-side column (e.g. `d_id + 1`)
    // must still fire: the expression passes through to the final grouping and
    // the pushdown groups only by the join key.
    #[test]
    fn fires_with_computed_group_expr() {
        use datafusion_common::ScalarValue;
        use datafusion_expr::Operator;
        use datafusion_physical_expr::expressions::{BinaryExpr, Literal};

        let fact = stats_leaf(
            vec![
                Field::new("f_dim", DataType::Int32, false),
                Field::new("f_amount", DataType::Float64, true),
            ],
            10_000_000,
            &[Some(100), None],
        );
        let dim = stats_leaf(
            vec![
                Field::new("d_id", DataType::Int32, false),
                Field::new("d_name", DataType::Utf8, true),
            ],
            100,
            &[Some(100), Some(100)],
        );
        let join = Arc::new(
            HashJoinExec::try_new(
                fact,
                dim,
                vec![(
                    Arc::new(Column::new("f_dim", 0)),
                    Arc::new(Column::new("d_id", 0)),
                )],
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let join_schema = join.schema();

        let sum_expr = Arc::new(
            AggregateExprBuilder::new(
                sum_udaf(),
                vec![Arc::new(Column::new("f_amount", 1))],
            )
            .schema(Arc::clone(&join_schema))
            .alias("sum(f_amount)")
            .build()
            .unwrap(),
        );
        // GROUP BY (d_id + 1) — a computed expression over a keep-side column.
        let grp_expr = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("d_id", 2)),
            Operator::Plus,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        )) as Arc<dyn PhysicalExpr>;
        let group = PhysicalGroupBy::new_single(vec![(grp_expr, "grp".to_string())]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group,
                vec![Arc::clone(&sum_expr)],
                vec![None],
                join,
                Arc::clone(&join_schema),
            )
            .unwrap(),
        );
        let final_group = partial.group_expr().as_final();
        let plan = Arc::new(
            AggregateExec::try_new(
                AggregateMode::FinalPartitioned,
                final_group,
                vec![sum_expr],
                vec![None],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let optimized = run_rule(plan);
        assert!(
            join_child_is_aggregate(&optimized),
            "expected push-down with a computed keep-side group expression"
        );
    }

    // Decline path: the fact join key is unique (NDV == row count) -> no
    // reduction -> rule must not fire.
    #[test]
    fn declines_when_no_reduction() {
        let optimized = run_rule(agg_over_join(10_000_000));
        assert!(
            !join_child_is_aggregate(&optimized),
            "expected no push-down when pre-aggregation does not reduce rows"
        );
    }

    use super::push_is_beneficial;

    // Default thresholds (mirror config defaults): require >= 4x reduction, no
    // absolute group cap.
    const FACTOR: usize = 4;
    const NO_CAP: usize = 0;

    #[test]
    fn beneficial_when_join_output_exceeds_pre_aggregated_rows() {
        // Large fact side reduced well by grouping (100x), and the join is not
        // selective (emits more rows than the pre-aggregation) -> push.
        assert!(push_is_beneficial(
            10_000_000, 100_000, 5_000_000, FACTOR, NO_CAP
        ));
        assert!(push_is_beneficial(
            10_000_000, 100_000, 100_001, FACTOR, NO_CAP
        ));
    }

    #[test]
    fn not_beneficial_below_selective_join_q2_shape() {
        // TPC-H Q2 site A: ~10M `stock` rows grouped to ~100k, but the EUROPE
        // chain only emits ~946 rows. join_out (946) <= grouped (100k) => the
        // join would discard those rows. Must NOT fire.
        assert!(!push_is_beneficial(
            10_000_000, 100_000, 946, FACTOR, NO_CAP
        ));
        assert!(!push_is_beneficial(
            10_000_000, 100_000, 100_000, FACTOR, NO_CAP
        ));
    }

    #[test]
    fn not_beneficial_barely_reducing_q2_site_b() {
        // TPC-H Q2 site B (the residual +30%): push=10M grouped=6M join_out=10.17M.
        // It "reduces" and the join isn't selective, but only ~1.7x while
        // materializing 6M groups (~506 MB) -> not worth it. Declined at the
        // default 4x factor, but would have fired under the old "any reduction"
        // gate (factor=1).
        assert!(!push_is_beneficial(
            10_006_168, 6_003_681, 10_170_203, FACTOR, NO_CAP
        ));
        assert!(push_is_beneficial(
            10_006_168, 6_003_681, 10_170_203, 1, NO_CAP
        ));
        // Q2 site that should still fire: ~20x reduction.
        assert!(push_is_beneficial(
            10_170_203, 492_105, 2_034_040, FACTOR, NO_CAP
        ));
    }

    #[test]
    fn absolute_group_cap_declines_large_pre_aggregation() {
        // Even a strong reduction is declined if the absolute pushed-group count
        // exceeds the cap (memory guard).
        assert!(push_is_beneficial(
            100_000_000,
            2_000_000,
            100_000_000,
            FACTOR,
            NO_CAP
        ));
        assert!(!push_is_beneficial(
            100_000_000,
            2_000_000,
            100_000_000,
            FACTOR,
            1_000_000
        ));
    }

    #[test]
    fn not_beneficial_when_no_reduction() {
        // Group key is (near-)unique on the push side: pre-aggregation does not
        // reduce rows, so it is pure overhead even if the join is not selective.
        assert!(!push_is_beneficial(
            1_000_000, 1_000_000, 1_000_000, FACTOR, NO_CAP
        ));
    }
}
