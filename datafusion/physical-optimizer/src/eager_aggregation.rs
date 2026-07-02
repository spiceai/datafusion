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
//!     join_out > grouped                          (not behind a selective join)
//!  && push_rows >= grouped * min_reduction         (a substantial, not token, reduction)
//!  && grouped <= max_pushed_groups (if capped)     (bounded pre-aggregation memory)
//!  && grouped <= retained_groups_bound * 2 (if known)  (not over-producing groups the join drops)
//! ```
//!
//! where `push_rows` is the push side's row count, `grouped` the estimated
//! pre-aggregated group count, and `join_out` the rows reaching the aggregate.
//! The reduction factor and the absolute group cap are tunable via
//! `datafusion.optimizer.eager_aggregation_min_reduction_factor` and
//! `…_max_pushed_groups`. The factor guard matters because a push that barely
//! reduces (e.g. 1.7x) yet materializes millions of groups costs more than it
//! saves. `retained_groups_bound` is the cap (when derivable) on how many
//! pre-aggregated groups the join can keep — the over-production guard against a
//! `join_out` over-estimate that would otherwise wave a wasteful push through;
//! see [`cost_gate`].
//!
//! # Scope (first milestone)
//!
//! Matches `Final/FinalPartitioned( Partial( HashJoinExec ) )` (the shape the
//! default physical planner emits for `GROUP BY` over a join) and pushes a
//! pre-aggregation into the join input that supplies all aggregate arguments.
//! Supports `SUM`/`MIN`/`MAX` (whose merge reuses the same UDAF) and `COUNT`
//! (whose per-group partial counts are merged with `SUM`), and plain-column
//! join keys (group-by may be any expression — its push-side columns are pushed
//! into the pre-aggregation grouping). `COUNT(*)` (which names no column, so no
//! side can be inferred) is pushed only for semi/anti joins, where the surviving
//! side is forced. A residual (non-equi) `join.filter()` is preserved by adding
//! its push-side columns to the pre-aggregation grouping (so the filter, remapped
//! onto the pre-aggregated side, still accepts/rejects each group as a unit).
//! Join types:
//! * **inner** — push into the side that supplies the aggregate arguments;
//! * **`LeftSemi`/`LeftAnti`/`RightSemi`/`RightAnti`** (from IN/EXISTS) — the
//!   output is a single side; push into it. A row of that side survives the join
//!   based only on its join key's (non-)match on the other side, so collapsing it
//!   by [join key ∪ group columns] before the join preserves which groups
//!   survive.
//!
//! It peels an intervening chain of column-only `ProjectionExec`s between the
//! partial aggregate and the join (the shape the planner emits before
//! `ProjectionPushdown`), including ones that *rename* columns (`x AS y`): the
//! aggregate arg/group expressions are normalized out of the (possibly aliased)
//! post-projection space into the join's raw-named output space before any
//! by-name remapping (see `remap_to_raw`). Anything outside this leaves the plan
//! unchanged.
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
//!
//! # Deferred / not yet supported
//!
//! The shapes below are recognized but intentionally declined for now; each is a
//! follow-up. Search for `TODO(eager-agg)` for the corresponding bail points.
//!
//! * **`FilterExec` between the aggregate and the join.** The peel loop walks
//!   only column-only `ProjectionExec`s; a `FilterExec` makes it bail. Supporting
//!   it means keeping the filter (its push-side columns folded into the
//!   pre-aggregation grouping, exactly like a residual `join.filter()`) and
//!   re-inserting it above the rebuilt join with its predicate remapped. In
//!   practice cross-side predicates already arrive as a `join.filter()` (handled);
//!   single-side ones are sunk below the join before this rule runs — so this is
//!   low priority until a decline map shows a real case.
//! * **A nested aggregate between the top aggregate and the join** (subquery
//!   aggregate; CH-benCHmark q13/q16). The peel reaches an `AggregateExec` rather
//!   than a join and bails; handling it needs a distinct matcher.
//! * **Aggregate measures spanning both join inputs** (e.g. `SUM(l.a + r.b)` or a
//!   `CASE` over both sides; CH-benCHmark q14). Requires linear decomposition
//!   (`SUM(l.a + r.b) = SUM(l.a)·fanout + SUM(r.b)·fanout`) with the
//!   Yan–Larson/Fent–Neumann fan-out factors — a larger piece of work.
//! * **`COUNT(*)` over an inner join.** With no column argument no side can be
//!   inferred; only forced (semi/anti) sides are handled. An inner `COUNT(*)`
//!   could ride a heuristic side choice but is declined for now.
//! * **Decimal `AVG`.** Only Float64 `AVG` is recombined; a decimal result would
//!   need its exact output scale reproduced in the division projection.
//! * **`schema_check` is disabled** (COUNT→SUM widens non-null→nullable). A
//!   `coalesce(_, 0)` in a top projection would let it stay enabled.

use std::sync::Arc;

use crate::PhysicalOptimizerRule;

use arrow::datatypes::DataType;
use datafusion_common::JoinSide;
use datafusion_common::Result;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_expr::{AggregateUDF, JoinType, Operator};
use datafusion_functions_aggregate::count::count_udaf;
use datafusion_functions_aggregate::sum::sum_udaf;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::aggregate::{AggregateExprBuilder, AggregateFunctionExpr};
use datafusion_physical_expr::expressions::{Column, binary, cast};
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, build_join_schema,
};
use datafusion_physical_plan::projection::ProjectionExec;

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
        // The rewrite preserves column types and order, but COUNT decomposes to
        // SUM(partial_count), and SUM is nullable while COUNT is not — a
        // non-null -> nullable widening that the framework's schema check
        // forbids (it only permits the reverse). The widening is value-safe
        // (every final group has >= 1 partial, so the merged count is never
        // actually NULL), so we opt out of the check, as the trait docs sanction
        // for rules that change nullability.
        false
    }

    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !config.optimizer.enable_eager_aggregation {
            return Ok(plan);
        }
        Ok(plan
            .transform_down(|plan| match try_push_aggregate(&plan, config)? {
                Some(new_plan) => Ok(Transformed::yes(new_plan)),
                None => Ok(Transformed::no(plan)),
            })?
            .data)
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
        // High-frequency shape miss (fires for every non-two-phase aggregate);
        // stay silent so the decline diagnostics below aren't drowned out.
        return Ok(None);
    };
    if *top_partial.mode() != AggregateMode::Partial {
        return Ok(None);
    }

    // The `Final <- Partial` shape matched, so from here any bail is informative:
    // emit the reason at debug level (target `datafusion::eager_aggregation`, so it
    // can be filtered) — e.g. an unsupported shape, an unsupported aggregate, or a
    // cost-gate decline.
    macro_rules! decline {
        ($($arg:tt)*) => {{
            log::debug!(target: "datafusion::eager_aggregation", "decline: {}", format!($($arg)*));
            return Ok(None);
        }};
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
            // TODO(eager-agg): only column-only projections are peeled. A
            // `FilterExec` here (filter above the join, pre-FilterPushdown) could
            // be kept and re-applied with its push-side columns folded into the
            // pre-aggregation grouping; an intervening `AggregateExec` (nested /
            // subquery aggregate, e.g. CH-benCHmark q13/q16) would need a separate
            // matcher. Both bail here for now. See the module "Deferred" section.
            let Some(proj) = node.downcast_ref::<ProjectionExec>() else {
                decline!(
                    "aggregate input chain reaches {}, not a (projected) hash join",
                    node.name()
                );
            };
            // Column-only projection: output index -> input column index. We
            // compose only the *indices*. A projection that *renames* a column
            // (`x AS y`) is fine: the aggregate arg/group expressions are later
            // normalized out of this (possibly-renamed) post-projection space into
            // the join's raw-named output space via `agg_to_raw` (see
            // `remap_to_raw`), so the downstream by-name remaps never see the alias.
            let mut proj_map = Vec::with_capacity(proj.expr().len());
            for pe in proj.expr() {
                let Some(c) = pe.expr.downcast_ref::<Column>() else {
                    decline!(
                        "intervening projection has a computed expression (not a plain column)"
                    );
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
    // The peel loop only `break`s once `node` downcasts to a `HashJoinExec`.
    let join = node
        .downcast_ref::<HashJoinExec>()
        .expect("peel loop exits only at a HashJoinExec");
    // Apply the join's own (folded) projection: node output index -> raw index.
    if let Some(p) = join.projection.as_ref() {
        for m in agg_to_raw.iter_mut() {
            *m = p[*m];
        }
    }

    // Supported join types: inner, and the semi/anti joins produced by
    // IN/EXISTS. For semi/anti the output is a *single* side (left for `Left*`,
    // right for `Right*`); the aggregate's columns and group-by live entirely on
    // that side, and we push into it. A row of that side survives the join based
    // only on whether its join key has (semi) / lacks (anti) a match on the other
    // side, so collapsing it by [join key ∪ group columns] before the join is
    // sound — the whole key-group survives or not together.
    let join_type = *join.join_type();
    let output_side = match join_type {
        JoinType::Inner => None,
        JoinType::LeftSemi | JoinType::LeftAnti => Some(Side::Left),
        JoinType::RightSemi | JoinType::RightAnti => Some(Side::Right),
        _ => decline!("join_type {join_type:?} not supported"),
    };
    // A residual (non-equi) `join.filter()` is supported by pushing its push-side
    // columns into the pre-aggregation grouping (below) and re-applying the filter
    // at the rebuilt join; it is not a reason to decline.

    let left = join.left();
    let right = join.right();
    let left_len = left.schema().fields().len();

    // Map an aggregate-input column index to the raw left ++ right join index.
    let raw_index = |agg_idx: usize| -> usize { agg_to_raw[agg_idx] };

    // The join's *raw* (unprojected) output schema — left ++ right for an inner
    // join, or the single surviving side for a semi/anti join. This is exactly the
    // index space `agg_to_raw` now points into, so it carries the canonical
    // (raw) column names.
    let (raw_join_schema, _) =
        build_join_schema(left.schema().as_ref(), right.schema().as_ref(), &join_type);

    // Rewrite an aggregate arg/group expression out of the (possibly projection-
    // *renamed*) aggregate-input space into the raw join-output space: each
    // `Column` is re-pointed via `agg_to_raw` to its raw index and given the raw
    // name at that index. Downstream by-name remaps (`reassign_expr_columns` into
    // the push-side / rebuilt-join schemas) then never see a projection alias, so
    // an intervening `x AS y` projection is handled rather than declined.
    let remap_to_raw = |e: Arc<dyn PhysicalExpr>| -> Result<Arc<dyn PhysicalExpr>> {
        e.transform_down(|node| {
            if let Some(c) = node.downcast_ref::<Column>() {
                let raw = agg_to_raw[c.index()];
                let name = raw_join_schema.field(raw).name();
                Ok(Transformed::yes(
                    Arc::new(Column::new(name, raw)) as Arc<dyn PhysicalExpr>
                ))
            } else {
                Ok(Transformed::no(node))
            }
        })
        .data()
    };

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
        if !is_decomposable(agg) {
            decline!(
                "aggregate {:?} (distinct={}) is not decomposable (only SUM/MIN/MAX/COUNT/AVG)",
                agg.fun().name(),
                agg.is_distinct()
            );
        }
        // AVG decomposes to SUM/COUNT and is recombined by division. We only
        // handle the Float64 case (numeric inputs); decimal AVG keeps a Decimal
        // output whose scale we'd have to reproduce exactly, so decline it.
        // TODO(eager-agg): support decimal AVG by reproducing the output scale.
        if agg.fun().name() == "avg" && agg.field().data_type() != &DataType::Float64 {
            decline!(
                "AVG output type {:?} unsupported (only Float64)",
                agg.field().data_type()
            );
        }
        for e in agg.expressions() {
            collect_column_indices(&e, &mut arg_indices);
        }
    }
    // Choose the side to push into. For semi/anti joins it is forced to the
    // output side (the only side with columns; `agg_to_raw` already maps into its
    // schema). For inner joins, the aggregate arguments must all come from one
    // side (compared in raw left ++ right index space).
    let side = if let Some(os) = output_side {
        os
    } else {
        let all_left = arg_indices.iter().all(|&i| raw_index(i) < left_len);
        let all_right = arg_indices.iter().all(|&i| raw_index(i) >= left_len);
        if all_left && !arg_indices.is_empty() {
            Side::Left
        } else if all_right && !arg_indices.is_empty() {
            Side::Right
        } else {
            // TODO(eager-agg): empty args here is `COUNT(*)` over an inner join —
            // no column pins a side. It could ride a heuristic side choice (push
            // into the grouped side); declined for now. Args genuinely spanning
            // both sides is the cross-side-measure case (see module "Deferred").
            decline!(
                "aggregate args span both sides or are empty: raw_indices={:?} left_len={left_len}",
                arg_indices
                    .iter()
                    .map(|&i| raw_index(i))
                    .collect::<Vec<_>>()
            );
        }
    };

    // Group-by may include computed expressions (e.g. `extract(year from ...)`),
    // not just plain columns. Normalize the group expressions into raw join-output
    // space (rename-safe; see `remap_to_raw`) and collect the columns each one
    // references; the push-side ones are added to the pushdown grouping below.
    // Grouping the pre-aggregation by those underlying columns refines grouping
    // by the expression (the expression is constant within each pre-agg group),
    // so re-aggregating above the join reproduces the original groups. Keep-side
    // group expressions simply pass through to the final grouping unchanged. The
    // normalized exprs are reused to rebuild the final grouping over the new join.
    let group = top_partial.group_expr();
    let raw_group_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = group
        .expr()
        .iter()
        .map(|(e, name)| Ok((remap_to_raw(Arc::clone(e))?, name.clone())))
        .collect::<Result<Vec<_>>>()?;
    let mut group_cols: Vec<Column> = Vec::new();
    for (e, _) in &raw_group_exprs {
        collect_columns(e, &mut group_cols);
    }

    let (push_plan, push_keys) = match side {
        Side::Left => (left, &left_key_cols),
        Side::Right => (right, &right_key_cols),
    };

    // Build the pushed-down grouping over the push side: its join keys plus any
    // group-by columns originating from that side. Columns are expressed in the
    // push side's own schema. For an inner join, `raw_index` is a left ++ right
    // index, so left columns keep their index and right columns are shifted by
    // `left_len`. For a semi/anti join the output is a single side, so
    // `raw_index` is already push-side-local and every group column is on it.
    let to_push_index = |raw: usize| -> usize {
        match (output_side, side) {
            (Some(_), _) => raw,
            (None, Side::Left) => raw,
            (None, Side::Right) => raw - left_len,
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
    // Group count when the pushdown grouping is *exactly* the push-side join
    // keys (no extra group/filter columns added below). In that case the
    // pre-aggregation collapses the push side to one row per join key, which
    // lets us bound how many groups the join can retain (the over-production
    // guard below).
    let n_key_groups = pushdown_group.len();
    for c in &group_cols {
        // `group_cols` are already in raw join-output space, so the index is the
        // raw index and the name is the raw (un-aliased) column name.
        let raw = c.index();
        let on_side = output_side.is_some()
            || (side == Side::Left && raw < left_len)
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
    // We remap the aggregate arguments, join keys, and residual-filter columns
    // into `push_schema` *by name* (via `reassign_expr_columns` / `index_of`),
    // which resolves to the first field of a given name. If the push side carries
    // duplicate column names a lookup could silently bind to the wrong column, so
    // bail (mirrors the output-schema duplicate-name guard below).
    {
        let mut names: Vec<&str> = push_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        names.sort_unstable();
        if names.windows(2).any(|w| w[0] == w[1]) {
            decline!("push side has duplicate column names (name-based remap unsafe)");
        }
    }

    // The join side the pre-aggregation replaces, in `JoinFilter` terms.
    let push_join_side = match side {
        Side::Left => JoinSide::Left,
        Side::Right => JoinSide::Right,
    };
    // To preserve a residual join filter, add every push-side column it
    // references to the pushdown grouping. Each pre-aggregated group then has a
    // single value for those columns, so the filter (re-applied at the rebuilt
    // join) accepts/rejects the whole group exactly as it would the original
    // rows. `JoinFilter` column indices are already side-local. (A high-NDV
    // filter column inflates the group count, which the cost gate then weighs.)
    if let Some(filter) = join.filter() {
        for ci in filter.column_indices() {
            if ci.side != push_join_side {
                continue;
            }
            if !seen.contains(&ci.index) {
                seen.push(ci.index);
                let name = push_schema.field(ci.index).name();
                pushdown_group.push((
                    Arc::new(Column::new(name, ci.index)) as Arc<dyn PhysicalExpr>,
                    name.to_string(),
                ));
            }
        }
    }

    // Upper bound on how many pre-aggregated groups the join can *retain*, for
    // the over-production guard in `cost_gate`. Derivable only when the grouping
    // is exactly the push-side join keys (`group_is_just_keys`) on an inner join:
    // the pre-aggregation then leaves one row per key, and such a group survives
    // only if its key exists on the other side, so
    //     retained groups <= distinct keys(other side) <= other side row count.
    // The other side's row count is therefore a (loose) upper bound. Materializing
    // far more groups than this is hash-aggregation the join immediately discards.
    // The pre-push `join_out` estimate misses this collapse when the other side
    // carries selective filters whose selectivity never propagated through the
    // join chain (CH-benCHmark q3: 3.1M order_line groups pre-aggregated, but the
    // selective customer/oorder side retains ~1.0M est / ~35K actual). A semi/anti
    // join's other side merely filters the single output side, so it bounds
    // nothing here — leave it unset (no guard).
    let group_is_just_keys = pushdown_group.len() == n_key_groups;
    let retained_groups_bound = if output_side.is_none() && group_is_just_keys {
        let other_plan = match side {
            Side::Left => right,
            Side::Right => left,
        };
        match other_plan.partition_statistics(None)?.num_rows {
            Precision::Exact(n) | Precision::Inexact(n) => Some(n),
            Precision::Absent => None,
        }
    } else {
        None
    };

    // Cost gate: only push when the pre-aggregation produces fewer rows than the
    // join emits (otherwise the join is selective enough that pre-aggregating is
    // wasted work). Requires statistics; absent stats => do not fire.
    match cost_gate(
        join,
        push_plan,
        &pushdown_group,
        retained_groups_bound,
        config,
    )? {
        CostGate::Allow {
            push_rows,
            grouped,
            join_out,
        } => {
            log::debug!(
                target: "datafusion::eager_aggregation",
                "accept: side={side:?} push_rows={push_rows} grouped={grouped} join_out={join_out}"
            );
        }
        CostGate::Decline(reason) => decline!("cost gate: {reason}"),
    }

    // Decompose each aggregate into the partial(s) to push and how to recombine
    // them above the join. AVG splits into SUM(x)+COUNT(x) and needs a top
    // projection (the division); SUM/MIN/MAX/COUNT each produce a single merged
    // column that *is* the output, so no projection is added then.
    let has_avg = aggrs.iter().any(|a| a.fun().name() == "avg");

    /// Per-input-aggregate plan: how to merge each pushed partial and how to
    /// recombine the merged columns into the aggregate's output value.
    struct AggPlan {
        /// `(merge udaf, pushed partial alias, merged column alias)` per partial.
        merges: Vec<(Arc<AggregateUDF>, String, String)>,
        out: OutKind,
        out_name: String,
    }

    // Pushed (pre-)aggregation expressions: each partial UDAF over the original
    // argument columns remapped into the push side's schema, aliased to a stable
    // internal name.
    let mut pushed_aggrs: Vec<Arc<AggregateFunctionExpr>> =
        Vec::with_capacity(aggrs.len());
    let mut agg_plans: Vec<AggPlan> = Vec::with_capacity(aggrs.len());
    for (i, agg) in aggrs.iter().enumerate() {
        let args = agg
            .expressions()
            .into_iter()
            .map(|e| {
                // Normalize out of the (possibly renamed) aggregate-input space
                // into raw join-output space, then rebind by name onto the push
                // side's schema.
                reassign_expr_columns(remap_to_raw(e)?, push_schema.as_ref())
            })
            .collect::<Result<Vec<_>>>()?;
        let Decomp { partials, out } = decompose(agg);
        let mut merges = Vec::with_capacity(partials.len());
        for (p, (partial_udaf, merge)) in partials.into_iter().enumerate() {
            let pushed_alias = format!("__eager_p{i}_{p}");
            let built = AggregateExprBuilder::new(partial_udaf, args.clone())
                .schema(Arc::clone(&push_schema))
                .alias(pushed_alias.clone())
                .build()?;
            pushed_aggrs.push(Arc::new(built));
            // The merged column gets an internal alias when a projection will
            // remap it; with no AVG the single passthrough merge is aliased
            // straight to the output name (no projection, unchanged shape).
            let merged_alias = if has_avg {
                format!("__eager_m{i}_{p}")
            } else {
                agg.name().to_string()
            };
            merges.push((merge, pushed_alias, merged_alias));
        }
        agg_plans.push(AggPlan {
            merges,
            out,
            out_name: agg.name().to_string(),
        });
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
    let n_pushed = pushed_aggrs.len();
    let pre_final = Arc::new(AggregateExec::try_new(
        AggregateMode::FinalPartitioned,
        pushdown_group_by.as_final(),
        pushed_aggrs,
        vec![None; n_pushed],
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

    // Remap a residual filter onto the pre-aggregated side: its push-side column
    // indices now point at the (reordered, narrower) pre-aggregation output,
    // located by name (each such column was added to the grouping above). The
    // filter expression and intermediate schema are unchanged — only the
    // side-local indices move.
    let new_filter = match join.filter() {
        None => None,
        Some(filter) => {
            let pf_schema = pre_final.schema();
            let mut new_cols = Vec::with_capacity(filter.column_indices().len());
            for ci in filter.column_indices() {
                if ci.side == push_join_side {
                    let name = push_schema.field(ci.index).name();
                    let Ok(index) = pf_schema.index_of(name) else {
                        decline!(
                            "filter push-side column {name} missing after pre-aggregation"
                        );
                    };
                    new_cols.push(ColumnIndex {
                        index,
                        side: ci.side,
                    });
                } else {
                    new_cols.push(ci.clone());
                }
            }
            Some(JoinFilter::new(
                Arc::clone(filter.expression()),
                new_cols,
                Arc::clone(filter.schema()),
            ))
        }
    };

    let new_join = Arc::new(HashJoinExec::try_new(
        new_left,
        new_right,
        new_on,
        new_filter,
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

    // Rebuild the top aggregation to merge the partials. Group columns are the
    // raw-normalized group expressions remapped into the new join output (by name,
    // now rename-safe); the output alias (`name`) is preserved so the final schema
    // is unchanged. Each aggregate becomes merge(internal_col) aliased back to the
    // original output name.
    let new_group_exprs = raw_group_exprs
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
        Vec::with_capacity(agg_plans.len());
    for plan in &agg_plans {
        for (merge, pushed_alias, merged_alias) in &plan.merges {
            let idx = new_join_schema.index_of(pushed_alias)?;
            let arg = Arc::new(Column::new(pushed_alias, idx)) as Arc<dyn PhysicalExpr>;
            let built = AggregateExprBuilder::new(Arc::clone(merge), vec![arg])
                .schema(Arc::clone(&new_join_schema))
                .alias(merged_alias.clone())
                .build()?;
            merge_aggrs.push(Arc::new(built));
        }
    }
    let n_merge = merge_aggrs.len();

    let new_top_partial = Arc::new(AggregateExec::try_new(
        AggregateMode::Partial,
        new_group_by.clone(),
        merge_aggrs.clone(),
        vec![None; n_merge],
        new_join,
        Arc::clone(&new_join_schema),
    )?);
    let new_top_final = Arc::new(AggregateExec::try_new(
        *top_final.mode(),
        new_group_by.as_final(),
        merge_aggrs,
        vec![None; n_merge],
        new_top_partial,
        new_join_schema,
    )?) as Arc<dyn ExecutionPlan>;

    // For AVG, the top aggregate produces internal SUM/COUNT columns; add a
    // projection that passes group columns through and divides SUM/COUNT (as
    // Float64) back into the AVG output, restoring the original output schema.
    // Without AVG the merged columns already carry the output names and order.
    let result: Arc<dyn ExecutionPlan> = if has_avg {
        let final_schema = new_top_final.schema();
        let col = |name: &str| -> Result<Arc<dyn PhysicalExpr>> {
            let idx = final_schema.index_of(name)?;
            Ok(Arc::new(Column::new(name, idx)) as Arc<dyn PhysicalExpr>)
        };
        let mut proj: Vec<(Arc<dyn PhysicalExpr>, String)> =
            Vec::with_capacity(group.expr().len() + agg_plans.len());
        for (_, name) in group.expr() {
            proj.push((col(name)?, name.clone()));
        }
        for plan in &agg_plans {
            let expr = match plan.out {
                OutKind::Passthrough => col(&plan.merges[0].2)?,
                OutKind::AvgDiv => {
                    let sum_f = cast(
                        col(&plan.merges[0].2)?,
                        final_schema.as_ref(),
                        DataType::Float64,
                    )?;
                    let cnt_f = cast(
                        col(&plan.merges[1].2)?,
                        final_schema.as_ref(),
                        DataType::Float64,
                    )?;
                    binary(sum_f, Operator::Divide, cnt_f, final_schema.as_ref())?
                }
            };
            proj.push((expr, plan.out_name.clone()));
        }
        Arc::new(ProjectionExec::try_new(proj, new_top_final)?)
    } else {
        new_top_final
    };

    Ok(Some(result))
}

/// True if `agg` can be split into pushed partial(s) plus a merge above the
/// join, is not DISTINCT, and has no ORDER BY. SUM/MIN/MAX merge by re-applying
/// themselves; COUNT merges its partial counts with SUM; AVG splits into
/// SUM(x)+COUNT(x) and is recombined by division (see [`decompose`]).
fn is_decomposable(agg: &AggregateFunctionExpr) -> bool {
    if agg.is_distinct() {
        return false;
    }
    // An order-sensitive aggregate (`ORDER BY` inside the call) cannot be split
    // into pushed partials and merged, since the pre-aggregation would collapse
    // the rows the ordering depends on. None of the supported functions below are
    // order-sensitive, but guard explicitly to honor the documented contract.
    if !agg.order_bys().is_empty() {
        return false;
    }
    matches!(agg.fun().name(), "sum" | "min" | "max" | "count" | "avg")
}

/// The partial aggregate(s) to push below the join for `agg`, and how its final
/// output value is recombined above the join.
struct Decomp {
    /// `(udaf, merge_udaf)` for each pushed partial: the partial computes `udaf`
    /// on the push side, and `merge_udaf` merges it above the join.
    partials: Vec<(Arc<AggregateUDF>, Arc<AggregateUDF>)>,
    /// How to turn the merged column(s) into `agg`'s output value.
    out: OutKind,
}

/// How an aggregate's merged partial column(s) recombine into its output.
enum OutKind {
    /// Output is the single merged column (SUM/MIN/MAX/COUNT).
    Passthrough,
    /// Output is `merged_sum / merged_count`, cast to Float64 (AVG).
    AvgDiv,
}

/// Decompose `agg` into the partial(s) to push and the recombination kind. All
/// inputs are pre-validated by [`is_decomposable`] (+ the AVG Float64 gate).
fn decompose(agg: &AggregateFunctionExpr) -> Decomp {
    match agg.fun().name() {
        // COUNT's per-group partial counts must be summed, not re-counted.
        "count" => Decomp {
            partials: vec![(Arc::new(agg.fun().clone()), sum_udaf())],
            out: OutKind::Passthrough,
        },
        // AVG(x) = SUM(x) / COUNT(x); both partials merge with SUM.
        "avg" => Decomp {
            partials: vec![(sum_udaf(), sum_udaf()), (count_udaf(), sum_udaf())],
            out: OutKind::AvgDiv,
        },
        // SUM/MIN/MAX are self-merging.
        _ => Decomp {
            partials: vec![(Arc::new(agg.fun().clone()), Arc::new(agg.fun().clone()))],
            out: OutKind::Passthrough,
        },
    }
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
    retained_groups_bound: Option<usize>,
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

    // Over-production guard: `retained_groups_bound` (when known) caps how many
    // pre-aggregated groups the join can retain (see caller for the derivation).
    // Allow up to 2x over-production to absorb estimate noise on a healthy
    // key-to-key join (q18: grouped ~= bound), but decline a gross overshoot
    // (q3: grouped ~3x the bound, so most pre-aggregation is discarded).
    if let Some(bound) = retained_groups_bound
        && grouped > bound.saturating_mul(2)
    {
        return Ok(CostGate::Decline(format!(
            "over-produces vs join: grouped={grouped} retained_groups_bound={bound} \
             (pre-agg groups the selective join discards)"
        )));
    }

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

    use crate::enforce_distribution::EnforceDistribution;
    use arrow::array::{Float64Array, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use arrow::record_batch::RecordBatch;
    use datafusion_common::NullEquality;
    use datafusion_common::ScalarValue;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::stats::{ColumnStatistics, Statistics};
    use datafusion_execution::TaskContext;
    use datafusion_functions_aggregate::average::avg_udaf;
    use datafusion_functions_aggregate::count::count_udaf;
    use datafusion_functions_aggregate::sum::sum_udaf;
    use datafusion_physical_expr::EquivalenceProperties;
    use datafusion_physical_expr::expressions::{BinaryExpr, Literal, lit};
    use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
    use datafusion_physical_plan::joins::PartitionMode;
    use datafusion_physical_plan::memory::MemoryStream;
    use datafusion_physical_plan::test::exec::StatisticsExec;
    use datafusion_physical_plan::{
        DisplayAs, DisplayFormatType, Partitioning, PlanProperties,
        SendableRecordBatchStream, collect, displayable,
    };
    use insta::assert_snapshot;

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
        agg_over_join_dim(fact_key_ndv, 100, 100)
    }

    /// Like [`agg_over_join`] but with a configurable dim (the join's *other*
    /// side): `dim_rows` rows and `dim_ndv` distinct join keys. The grouping is
    /// exactly the fact (push-side) join key, so the over-production guard is in
    /// play — `dim_rows` is the bound on surviving pre-aggregated groups.
    fn agg_over_join_dim(
        fact_key_ndv: usize,
        dim_rows: usize,
        dim_ndv: usize,
    ) -> Arc<dyn ExecutionPlan> {
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
            dim_rows,
            &[Some(dim_ndv), Some(dim_ndv)],
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

    /// Render a plan tree as an indented string, for the snapshot tests below.
    fn plan_str(plan: &Arc<dyn ExecutionPlan>) -> String {
        displayable(plan.as_ref()).indent(true).to_string()
    }

    /// True if any `HashJoinExec` in the tree has an `AggregateExec` child
    /// (i.e. a pre-aggregation was pushed below the join).
    fn join_child_is_aggregate(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if let Some(j) = plan.downcast_ref::<HashJoinExec>() {
            return j.left().downcast_ref::<AggregateExec>().is_some()
                || j.right().downcast_ref::<AggregateExec>().is_some();
        }
        plan.children().into_iter().any(join_child_is_aggregate)
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

    // Executable example (canonical SUM push). Input is the plan the planner emits
    // for `SELECT d_name, SUM(f_amount) FROM fact JOIN dim ON f_dim = d_id GROUP BY
    // d_name`: a two-phase aggregate directly over the join. The rule pushes a
    // pre-aggregation over the fact's join key (`f_dim`) below the join, so the
    // join probes ~100 pre-aggregated rows instead of 10M, and the original top
    // aggregate merges the partial sums. The pushed aggregate is emitted as
    // Partial -> FinalPartitioned; `EnforceDistribution` (which runs after this
    // rule) inserts the hash repartition between the two halves.
    #[test]
    fn example_sum_push_plan() {
        let input = agg_over_join(100);
        assert_snapshot!(plan_str(&input), @"
        AggregateExec: mode=FinalPartitioned, gby=[d_name@0 as d_name], aggr=[sum(f_amount)]
          AggregateExec: mode=Partial, gby=[d_name@3 as d_name], aggr=[sum(f_amount)]
            HashJoinExec: mode=CollectLeft, join_type=Inner, accumulator=MinMaxLeftAccumulator, on=[(f_dim@0, d_id@0)]
              StatisticsExec: col_count=2, row_count=Inexact(10000000)
              StatisticsExec: col_count=2, row_count=Inexact(100)
        ");

        let optimized = run_rule(input);
        assert_snapshot!(plan_str(&optimized), @"
        AggregateExec: mode=FinalPartitioned, gby=[d_name@0 as d_name], aggr=[sum(f_amount)]
          AggregateExec: mode=Partial, gby=[d_name@3 as d_name], aggr=[sum(f_amount)]
            HashJoinExec: mode=CollectLeft, join_type=Inner, accumulator=MinMaxLeftAccumulator, on=[(f_dim@0, d_id@0)]
              AggregateExec: mode=FinalPartitioned, gby=[f_dim@0 as f_dim], aggr=[__eager_p0_0]
                AggregateExec: mode=Partial, gby=[f_dim@0 as f_dim], aggr=[__eager_p0_0]
                  StatisticsExec: col_count=2, row_count=Inexact(10000000)
              StatisticsExec: col_count=2, row_count=Inexact(100)
        ");
    }

    // Decline path (CH-benCHmark q3 shape): a large fact pre-aggregates to many
    // groups (500k) but the join's other side is comparatively small (100k rows),
    // so the inner join can retain at most ~100k of those groups — the rest of
    // the pre-aggregation is wasted work the selective join discards. The naive
    // `join_out > grouped` test would still fire here (the join-output estimate
    // ignores that the push collapses the fan-out), so the over-production guard
    // is what holds it back. Contrast `fires_on_beneficial_join`, where the dim
    // is tiny (100) *and* the fact reduces to 100 groups (no over-production).
    #[test]
    fn declines_over_production_q3_shape() {
        let optimized = run_rule(agg_over_join_dim(500_000, 100_000, 100_000));
        assert!(
            !join_child_is_aggregate(&optimized),
            "expected NO push: pre-agg over-produces vs the selective join's other side"
        );
    }

    // A healthy key-to-key join (q18 shape): the fact reduces to ~grouped groups
    // and the other side is about the same size, so the join retains essentially
    // all pre-aggregated groups -> the over-production guard allows it.
    #[test]
    fn fires_on_balanced_key_to_key_join() {
        let optimized = run_rule(agg_over_join_dim(2_000_000, 2_000_000, 2_000_000));
        assert!(
            join_child_is_aggregate(&optimized),
            "expected a push: grouped ~= other side rows (no over-production)"
        );
    }

    // Non-inner (q22 shape): RightAnti join, output = right (customer) side; the
    // measure and group-by live on that side, so the aggregation is pushed into
    // it. Synthetic stats: the right join key is low-NDV so the pre-aggregation
    // reduces (a real PK anti-key would not — see the SF100 caveat — so this test
    // validates the anti-join mapping/mechanics, not a real-workload win).
    #[test]
    fn fires_on_right_anti_join() {
        // left = existence side ("oorder"); right = output/measured side ("customer").
        // oorder's join key matches few customer keys (NDV 10 vs 10k), so the
        // anti-join keeps most customer rows -> join_out stays large.
        let oorder = stats_leaf(
            vec![Field::new("o_c_id", DataType::Int32, false)],
            1_000,
            &[Some(10)],
        );
        let customer = stats_leaf(
            vec![
                Field::new("c_id", DataType::Int32, false),
                Field::new("c_state", DataType::Utf8, true),
                Field::new("c_balance", DataType::Float64, true),
            ],
            10_000_000,
            &[Some(10_000), Some(50), None],
        );
        let join = Arc::new(
            HashJoinExec::try_new(
                oorder,
                customer,
                vec![(
                    Arc::new(Column::new("o_c_id", 0)),
                    Arc::new(Column::new("c_id", 0)),
                )],
                None,
                &JoinType::RightAnti,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let join_schema = join.schema(); // RightAnti output = customer columns

        let sum_expr = Arc::new(
            AggregateExprBuilder::new(
                sum_udaf(),
                vec![Arc::new(Column::new("c_balance", 2))],
            )
            .schema(Arc::clone(&join_schema))
            .alias("sum(c_balance)")
            .build()
            .unwrap(),
        );
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("c_state", 1)) as Arc<dyn PhysicalExpr>,
            "c_state".to_string(),
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
            "expected the aggregation pushed into the RightAnti output (right) side"
        );
    }

    /// Aggregate function names of the root (top) `AggregateExec`.
    fn outer_agg_fun_names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
        let agg = plan
            .downcast_ref::<AggregateExec>()
            .expect("root is an AggregateExec");
        agg.aggr_expr()
            .iter()
            .map(|a| a.fun().name().to_string())
            .collect()
    }

    /// Aggregate function names of the first `AggregateExec` found as a direct
    /// child of a `HashJoinExec` (the pushed pre-aggregation).
    fn pushed_agg_fun_names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
        fn rec(p: &Arc<dyn ExecutionPlan>) -> Option<Vec<String>> {
            if let Some(j) = p.downcast_ref::<HashJoinExec>() {
                for child in [j.left(), j.right()] {
                    if let Some(a) = child.downcast_ref::<AggregateExec>() {
                        return Some(
                            a.aggr_expr()
                                .iter()
                                .map(|x| x.fun().name().to_string())
                                .collect(),
                        );
                    }
                }
            }
            p.children().into_iter().find_map(rec)
        }
        rec(plan).expect("a pre-aggregation pushed below a join")
    }

    // COUNT decomposition over an inner join: the pushed side computes per-group
    // partial COUNTs, and the top aggregate merges them with SUM (not COUNT).
    #[test]
    fn fires_with_count_on_inner_join() {
        // Same beneficial fact/dim shape as `agg_over_join`, but COUNT(f_amount).
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

        let count_expr = Arc::new(
            AggregateExprBuilder::new(
                count_udaf(),
                vec![Arc::new(Column::new("f_amount", 1))],
            )
            .schema(Arc::clone(&join_schema))
            .alias("count(f_amount)")
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
                vec![Arc::clone(&count_expr)],
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
                vec![count_expr],
                vec![None],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let optimized = run_rule(plan);
        assert!(
            join_child_is_aggregate(&optimized),
            "expected a pre-aggregation pushed below the join"
        );
        assert_eq!(
            pushed_agg_fun_names(&optimized),
            vec!["count"],
            "pushed side should compute partial COUNTs"
        );
        assert_eq!(
            outer_agg_fun_names(&optimized),
            vec!["sum"],
            "top aggregate should merge partial counts with SUM"
        );
    }

    // COUNT(*) (no column argument) over a LeftSemi join: no side can be inferred
    // from the (empty) arguments, but the semi join's surviving side is forced, so
    // the count is pushed there and merged with SUM above the join.
    #[test]
    fn fires_with_count_star_on_semi_join() {
        // left = output side ("customer"); right = existence side ("oorder").
        // Low-NDV join key (100) so the pushed [c_id, c_state] grouping (100*50)
        // still reduces 10M rows, and the semi join keeps ~10% -> join_out large.
        let customer = stats_leaf(
            vec![
                Field::new("c_id", DataType::Int32, false),
                Field::new("c_state", DataType::Utf8, true),
            ],
            10_000_000,
            &[Some(100), Some(50)],
        );
        let oorder = stats_leaf(
            vec![Field::new("o_c_id", DataType::Int32, false)],
            1_000,
            &[Some(10)],
        );
        let join = Arc::new(
            HashJoinExec::try_new(
                customer,
                oorder,
                vec![(
                    Arc::new(Column::new("c_id", 0)),
                    Arc::new(Column::new("o_c_id", 0)),
                )],
                None,
                &JoinType::LeftSemi,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let join_schema = join.schema(); // LeftSemi output = customer columns

        let count_expr = Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![lit(1i64)])
                .schema(Arc::clone(&join_schema))
                .alias("count(*)")
                .build()
                .unwrap(),
        );
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("c_state", 1)) as Arc<dyn PhysicalExpr>,
            "c_state".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group.clone(),
                vec![Arc::clone(&count_expr)],
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
                vec![count_expr],
                vec![None],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let optimized = run_rule(plan);
        assert!(
            join_child_is_aggregate(&optimized),
            "expected COUNT(*) pushed into the LeftSemi output (left) side"
        );
        assert_eq!(pushed_agg_fun_names(&optimized), vec!["count"]);
        assert_eq!(outer_agg_fun_names(&optimized), vec!["sum"]);
    }

    // AVG decomposition over an inner join: the push side computes partial
    // SUM(x)+COUNT(x), the top aggregate merges both with SUM, and a top
    // ProjectionExec divides them back into the AVG output (restoring the schema).
    #[test]
    fn fires_with_avg_on_inner_join() {
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

        let avg_expr = Arc::new(
            AggregateExprBuilder::new(
                avg_udaf(),
                vec![Arc::new(Column::new("f_amount", 1))],
            )
            .schema(Arc::clone(&join_schema))
            .alias("avg(f_amount)")
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
                vec![Arc::clone(&avg_expr)],
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
                vec![avg_expr],
                vec![None],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let original_schema = plan.schema();

        let optimized = run_rule(plan);
        assert!(
            join_child_is_aggregate(&optimized),
            "expected a pre-aggregation pushed below the join"
        );
        // The AVG path wraps the merged aggregate in a division projection.
        let proj = optimized
            .downcast_ref::<ProjectionExec>()
            .expect("AVG output should be recombined by a top projection");
        assert_eq!(
            pushed_agg_fun_names(&optimized),
            vec!["sum", "count"],
            "push side should compute partial SUM and COUNT"
        );
        let top = proj
            .input()
            .downcast_ref::<AggregateExec>()
            .expect("projection over the merged aggregate");
        let top_funs: Vec<_> = top.aggr_expr().iter().map(|a| a.fun().name()).collect();
        assert_eq!(
            top_funs,
            vec!["sum", "sum"],
            "both AVG partials merge with SUM"
        );
        // Output schema (names, types, order) is preserved.
        assert_eq!(optimized.schema(), original_schema);

        // Executable example of the AVG decomposition: the push side computes
        // partial SUM(x) + COUNT(x); above the join both merge with SUM; and a top
        // ProjectionExec divides the merged sum by the merged count (as Float64) to
        // rebuild `avg(f_amount)`, restoring the original output schema.
        assert_snapshot!(plan_str(&optimized), @"
        ProjectionExec: expr=[d_name@0 as d_name, __eager_m0_0@1 / CAST(__eager_m0_1@2 AS Float64) as avg(f_amount)]
          AggregateExec: mode=FinalPartitioned, gby=[d_name@0 as d_name], aggr=[__eager_m0_0, __eager_m0_1]
            AggregateExec: mode=Partial, gby=[d_name@4 as d_name], aggr=[__eager_m0_0, __eager_m0_1]
              HashJoinExec: mode=CollectLeft, join_type=Inner, accumulator=MinMaxLeftAccumulator, on=[(f_dim@0, d_id@0)]
                AggregateExec: mode=FinalPartitioned, gby=[f_dim@0 as f_dim], aggr=[__eager_p0_0, __eager_p0_1]
                  AggregateExec: mode=Partial, gby=[f_dim@0 as f_dim], aggr=[__eager_p0_0, __eager_p0_1]
                    StatisticsExec: col_count=2, row_count=Inexact(10000000)
                StatisticsExec: col_count=2, row_count=Inexact(100)
        ");
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
            ProjectionExec::try_new(
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
            ProjectionExec::try_new(
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

    // A *renaming* (and reordering / pruning) column-only projection between the
    // aggregate and the join: `d_name AS d_label`, `f_amount AS amt`. The aggregate
    // references the renamed columns. The rule normalizes them back to the join's
    // raw names (`remap_to_raw`) and still pushes — a rename no longer disables the
    // optimization (it previously declined).
    #[test]
    fn fires_through_renaming_projection() {
        let beneficial = agg_over_join(100);
        let final_agg = beneficial.downcast_ref::<AggregateExec>().unwrap();
        let partial = final_agg.input().downcast_ref::<AggregateExec>().unwrap();
        let join = Arc::clone(partial.input()); // [f_dim, f_amount, d_id, d_name]

        // Rename + reorder + prune: [d_name@3 as d_label, f_amount@1 as amt].
        let proj = Arc::new(
            ProjectionExec::try_new(
                vec![
                    (
                        Arc::new(Column::new("d_name", 3)) as Arc<dyn PhysicalExpr>,
                        "d_label".to_string(),
                    ),
                    (Arc::new(Column::new("f_amount", 1)), "amt".to_string()),
                ],
                join,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let proj_schema = proj.schema(); // [d_label, amt]

        let sum_expr = Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![Arc::new(Column::new("amt", 1))])
                .schema(Arc::clone(&proj_schema))
                .alias("sum(amt)")
                .build()
                .unwrap(),
        );
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("d_label", 0)) as Arc<dyn PhysicalExpr>,
            "d_label".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group,
                vec![Arc::clone(&sum_expr)],
                vec![None],
                proj,
                Arc::clone(&proj_schema),
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
                proj_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let optimized = run_rule(plan);
        assert!(
            join_child_is_aggregate(&optimized),
            "expected push-down through a renaming projection"
        );
    }

    // End-to-end value correctness through a renaming projection: `f_amount AS amt`
    // (aggregate arg) and `d_name AS label` (group key) are both renamed by a
    // projection between the aggregate and the join. Eager-ON must match eager-OFF,
    // proving the raw-name normalization rebinds the pushed aggregate and the final
    // grouping to the right underlying columns.
    #[tokio::test]
    async fn renaming_projection_eager_push_preserves_values() {
        let fact_schema = Arc::new(Schema::new(vec![
            Field::new("f_dim", DataType::Int32, false),
            Field::new("f_amount", DataType::Float64, true),
        ]));
        let fact_batch = RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2, 1])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            ],
        )
        .unwrap();
        let fact = Arc::new(DataStatsExec::new(
            vec![fact_batch],
            stats_with(10_000_000, &[Some(2), None]),
        )) as Arc<dyn ExecutionPlan>;

        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("d_id", DataType::Int32, false),
            Field::new("d_name", DataType::Utf8, true),
        ]));
        let dim_batch = RecordBatch::try_new(
            Arc::clone(&dim_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let dim = Arc::new(DataStatsExec::new(
            vec![dim_batch],
            stats_with(2, &[Some(2), Some(2)]),
        )) as Arc<dyn ExecutionPlan>;

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
        ) as Arc<dyn ExecutionPlan>; // [f_dim, f_amount, d_id, d_name]

        // Renaming + reordering projection: [d_name@3 as label, f_amount@1 as amt].
        let proj = Arc::new(
            ProjectionExec::try_new(
                vec![
                    (
                        Arc::new(Column::new("d_name", 3)) as Arc<dyn PhysicalExpr>,
                        "label".to_string(),
                    ),
                    (Arc::new(Column::new("f_amount", 1)), "amt".to_string()),
                ],
                join,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let proj_schema = proj.schema(); // [label, amt]

        let mk = |alias: &str, udaf: Arc<AggregateUDF>| {
            Arc::new(
                AggregateExprBuilder::new(udaf, vec![Arc::new(Column::new("amt", 1))])
                    .schema(Arc::clone(&proj_schema))
                    .alias(alias.to_string())
                    .build()
                    .unwrap(),
            )
        };
        let aggrs = vec![mk("count(amt)", count_udaf()), mk("sum(amt)", sum_udaf())];
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("label", 0)) as Arc<dyn PhysicalExpr>,
            "label".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group.clone(),
                aggrs.clone(),
                vec![None; aggrs.len()],
                proj,
                Arc::clone(&proj_schema),
            )
            .unwrap(),
        );
        let final_group = partial.group_expr().as_final();
        let plan = Arc::new(
            AggregateExec::try_new(
                AggregateMode::FinalPartitioned,
                final_group,
                aggrs.clone(),
                vec![None; aggrs.len()],
                partial,
                proj_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let eager_on = run_rule(Arc::clone(&plan));
        assert!(
            join_child_is_aggregate(&eager_on),
            "eager aggregation should fire through the renaming projection"
        );
        let on = enforce_distribution(eager_on);
        let off = enforce_distribution(plan);

        let ctx = Arc::new(TaskContext::default());
        let on_rows = collect(on, Arc::clone(&ctx)).await.unwrap();
        let off_rows = collect(off, ctx).await.unwrap();

        // Expected: a -> {count 3, sum 90}; b -> {count 3, sum 120}.
        let total: usize = off_rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "two groups expected");
        assert_eq!(
            sorted_rows(&on_rows),
            sorted_rows(&off_rows),
            "eager-on results must match eager-off across a renaming projection"
        );
    }

    // Group-by a computed expression over a keep-side column (e.g. `d_id + 1`)
    // must still fire: the expression passes through to the final grouping and
    // the pushdown groups only by the join key.
    #[test]
    fn fires_with_computed_group_expr() {
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

    /// A leaf exec that yields fixed `RecordBatch`es **and** reports injected
    /// `Statistics`. The stats (large NDV/row counts) drive the cost gate so the
    /// rewrite fires; the batches (tiny) are what actually executes, so the
    /// ON-vs-OFF value comparison runs on real data.
    #[derive(Debug)]
    struct DataStatsExec {
        batches: Vec<RecordBatch>,
        schema: SchemaRef,
        stats: Statistics,
        cache: Arc<PlanProperties>,
    }

    impl DataStatsExec {
        fn new(batches: Vec<RecordBatch>, stats: Statistics) -> Self {
            let schema = batches[0].schema();
            let cache = Arc::new(PlanProperties::new(
                EquivalenceProperties::new(Arc::clone(&schema)),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Incremental,
                Boundedness::Bounded,
            ));
            Self {
                batches,
                schema,
                stats,
                cache,
            }
        }
    }

    impl DisplayAs for DataStatsExec {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut std::fmt::Formatter,
        ) -> std::fmt::Result {
            write!(f, "DataStatsExec")
        }
    }

    impl ExecutionPlan for DataStatsExec {
        fn name(&self) -> &'static str {
            "DataStatsExec"
        }
        fn properties(&self) -> &Arc<PlanProperties> {
            &self.cache
        }
        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }
        fn with_new_children(
            self: Arc<Self>,
            _: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            Ok(self)
        }
        fn execute(
            &self,
            _partition: usize,
            _context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            Ok(Box::pin(MemoryStream::try_new(
                self.batches.clone(),
                Arc::clone(&self.schema),
                None,
            )?))
        }
        fn partition_statistics(
            &self,
            partition: Option<usize>,
        ) -> Result<Arc<Statistics>> {
            Ok(Arc::new(if partition.is_some() {
                Statistics::new_unknown(&self.schema)
            } else {
                self.stats.clone()
            }))
        }
    }

    fn stats_with(num_rows: usize, distinct: &[Option<usize>]) -> Statistics {
        Statistics {
            num_rows: Precision::Inexact(num_rows),
            total_byte_size: Precision::Absent,
            column_statistics: distinct
                .iter()
                .map(|d| {
                    let cs = ColumnStatistics::new_unknown();
                    match d {
                        Some(n) => cs.with_distinct_count(Precision::Inexact(*n)),
                        None => cs,
                    }
                })
                .collect(),
        }
    }

    /// Format result rows as a sorted multiset of strings, so the comparison is
    /// insensitive to partition/row ordering.
    fn sorted_rows(batches: &[RecordBatch]) -> Vec<String> {
        let s = arrow::util::pretty::pretty_format_batches(batches)
            .unwrap()
            .to_string();
        let mut lines: Vec<String> = s.lines().map(str::to_string).collect();
        lines.sort();
        lines
    }

    fn enforce_distribution(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        // EnforceDistribution inserts the repartitioning the pushed/top
        // FinalPartitioned aggregates require to execute correctly.
        EnforceDistribution::new()
            .optimize(plan, &ConfigOptions::default())
            .unwrap()
    }

    // End-to-end value correctness: COUNT, SUM and AVG over an inner join all
    // produce identical results with eager aggregation ON vs OFF, and the ON plan
    // actually pushes a pre-aggregation below the join. Tiny data, big fake stats.
    #[tokio::test]
    async fn count_sum_avg_eager_push_preserves_values() {
        // fact(f_dim, f_amount): real rows are tiny; stats claim 10M rows / NDV 2.
        let fact_schema = Arc::new(Schema::new(vec![
            Field::new("f_dim", DataType::Int32, false),
            Field::new("f_amount", DataType::Float64, true),
        ]));
        let fact_batch = RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2, 1])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
            ],
        )
        .unwrap();
        let fact = Arc::new(DataStatsExec::new(
            vec![fact_batch],
            stats_with(10_000_000, &[Some(2), None]),
        )) as Arc<dyn ExecutionPlan>;

        // dim(d_id, d_name).
        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("d_id", DataType::Int32, false),
            Field::new("d_name", DataType::Utf8, true),
        ]));
        let dim_batch = RecordBatch::try_new(
            Arc::clone(&dim_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let dim = Arc::new(DataStatsExec::new(
            vec![dim_batch],
            stats_with(2, &[Some(2), Some(2)]),
        )) as Arc<dyn ExecutionPlan>;

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
        let join_schema = join.schema(); // [f_dim, f_amount, d_id, d_name]

        let mk = |alias: &str, udaf: Arc<AggregateUDF>| {
            Arc::new(
                AggregateExprBuilder::new(
                    udaf,
                    vec![Arc::new(Column::new("f_amount", 1))],
                )
                .schema(Arc::clone(&join_schema))
                .alias(alias.to_string())
                .build()
                .unwrap(),
            )
        };
        let aggrs = vec![
            mk("count(f_amount)", count_udaf()),
            mk("sum(f_amount)", sum_udaf()),
            mk("avg(f_amount)", avg_udaf()),
        ];
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("d_name", 3)) as Arc<dyn PhysicalExpr>,
            "d_name".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group.clone(),
                aggrs.clone(),
                vec![None; aggrs.len()],
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
                aggrs.clone(),
                vec![None; aggrs.len()],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        // Run the rule, assert it fired (before EnforceDistribution inserts a
        // RepartitionExec between the join and the pushed aggregate), then
        // distribute both plans so they execute correctly.
        let eager_on = run_rule(Arc::clone(&plan));
        assert!(
            join_child_is_aggregate(&eager_on),
            "eager aggregation should have fired"
        );
        let on = enforce_distribution(eager_on);
        let off = enforce_distribution(plan);

        let ctx = Arc::new(TaskContext::default());
        let on_rows = collect(on, Arc::clone(&ctx)).await.unwrap();
        let off_rows = collect(off, ctx).await.unwrap();

        // Expected per-group: a -> {count 3, sum 90, avg 30}; b -> {3, 120, 40}.
        let total: usize = off_rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "two groups expected");
        assert_eq!(
            sorted_rows(&on_rows),
            sorted_rows(&off_rows),
            "eager-on results must match eager-off"
        );
    }

    // End-to-end value correctness with a *non-unique keep side* (join fan-out) —
    // the classic eager-aggregation correctness trap. The dim (keep) side repeats
    // join key 1, so each pushed pre-aggregated fact row for f_dim=1 is duplicated
    // by the join before the top aggregate re-aggregates. SUM/COUNT are fan-out
    // linear (Σ over fact⋈dim = Σ_key preSum(key)·fanout(key)), so the top
    // re-aggregation must still match eager-off. Guards against a future change
    // that (wrongly) assumes a unique keep side.
    #[tokio::test]
    async fn fan_out_keep_side_eager_push_preserves_values() {
        // fact(f_dim, f_amount): tiny real rows, big fake stats (NDV 2) so it fires.
        let fact_schema = Arc::new(Schema::new(vec![
            Field::new("f_dim", DataType::Int32, false),
            Field::new("f_amount", DataType::Float64, true),
        ]));
        let fact_batch = RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
            ],
        )
        .unwrap();
        let fact = Arc::new(DataStatsExec::new(
            vec![fact_batch],
            stats_with(10_000_000, &[Some(2), None]),
        )) as Arc<dyn ExecutionPlan>;

        // dim(d_id, d_name): key 1 repeats (both named "a") -> f_dim=1 fans out 2x.
        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("d_id", DataType::Int32, false),
            Field::new("d_name", DataType::Utf8, true),
        ]));
        let dim_batch = RecordBatch::try_new(
            Arc::clone(&dim_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2])),
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
            ],
        )
        .unwrap();
        let dim = Arc::new(DataStatsExec::new(
            vec![dim_batch],
            stats_with(3, &[Some(2), Some(2)]),
        )) as Arc<dyn ExecutionPlan>;

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
        let join_schema = join.schema(); // [f_dim, f_amount, d_id, d_name]

        let mk = |alias: &str, udaf: Arc<AggregateUDF>| {
            Arc::new(
                AggregateExprBuilder::new(
                    udaf,
                    vec![Arc::new(Column::new("f_amount", 1))],
                )
                .schema(Arc::clone(&join_schema))
                .alias(alias.to_string())
                .build()
                .unwrap(),
            )
        };
        let aggrs = vec![
            mk("count(f_amount)", count_udaf()),
            mk("sum(f_amount)", sum_udaf()),
        ];
        let group = PhysicalGroupBy::new_single(vec![(
            Arc::new(Column::new("d_name", 3)) as Arc<dyn PhysicalExpr>,
            "d_name".to_string(),
        )]);
        let partial = Arc::new(
            AggregateExec::try_new(
                AggregateMode::Partial,
                group.clone(),
                aggrs.clone(),
                vec![None; aggrs.len()],
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
                aggrs.clone(),
                vec![None; aggrs.len()],
                partial,
                join_schema,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;

        let eager_on = run_rule(Arc::clone(&plan));
        assert!(
            join_child_is_aggregate(&eager_on),
            "eager aggregation should have fired"
        );
        let on = enforce_distribution(eager_on);
        let off = enforce_distribution(plan);

        let ctx = Arc::new(TaskContext::default());
        let on_rows = collect(on, Arc::clone(&ctx)).await.unwrap();
        let off_rows = collect(off, ctx).await.unwrap();

        // Expected (fan-out): a -> {count 4, sum 60}; b -> {count 1, sum 30}.
        let total: usize = off_rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "two groups expected");
        assert_eq!(
            sorted_rows(&on_rows),
            sorted_rows(&off_rows),
            "eager-on results must match eager-off across a fan-out keep side"
        );
    }

    /// True if some `HashJoinExec` in the tree carries a residual filter.
    fn any_join_has_filter(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if let Some(j) = plan.downcast_ref::<HashJoinExec>()
            && j.filter().is_some()
        {
            return true;
        }
        plan.children().into_iter().any(any_join_has_filter)
    }

    // End-to-end value correctness with a *cross-side* residual join filter
    // (`f_date >= d_since`, the shape of CH-benCHmark q12). The push-side filter
    // column (f_date) is folded into the pre-aggregation grouping and the filter
    // is re-applied at the rebuilt join; results must match eager-off.
    #[tokio::test]
    async fn cross_side_join_filter_eager_push_preserves_values() {
        // fact(f_dim, f_amount, f_date).
        let fact_schema = Arc::new(Schema::new(vec![
            Field::new("f_dim", DataType::Int32, false),
            Field::new("f_amount", DataType::Float64, true),
            Field::new("f_date", DataType::Int32, false),
        ]));
        let fact_batch = RecordBatch::try_new(
            Arc::clone(&fact_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2, 1])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])),
                Arc::new(Int32Array::from(vec![5, 9, 3, 7, 8, 2])),
            ],
        )
        .unwrap();
        let fact = Arc::new(DataStatsExec::new(
            vec![fact_batch],
            stats_with(10_000_000, &[Some(2), None, Some(10)]),
        )) as Arc<dyn ExecutionPlan>;

        // dim(d_id, d_name, d_since).
        let dim_schema = Arc::new(Schema::new(vec![
            Field::new("d_id", DataType::Int32, false),
            Field::new("d_name", DataType::Utf8, true),
            Field::new("d_since", DataType::Int32, false),
        ]));
        let dim_batch = RecordBatch::try_new(
            Arc::clone(&dim_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Int32Array::from(vec![4, 6])),
            ],
        )
        .unwrap();
        let dim = Arc::new(DataStatsExec::new(
            vec![dim_batch],
            stats_with(2, &[Some(2), Some(2), Some(2)]),
        )) as Arc<dyn ExecutionPlan>;

        // Residual filter: f_date (Left@2) >= d_since (Right@2).
        let intermediate = Schema::new(vec![
            Field::new("f_date", DataType::Int32, false),
            Field::new("d_since", DataType::Int32, false),
        ]);
        let filter_expr = binary(
            Arc::new(Column::new("f_date", 0)),
            Operator::GtEq,
            Arc::new(Column::new("d_since", 1)),
            &intermediate,
        )
        .unwrap();
        let filter = JoinFilter::new(
            filter_expr,
            vec![
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 2,
                    side: JoinSide::Right,
                },
            ],
            Arc::new(intermediate),
        );

        let join = Arc::new(
            HashJoinExec::try_new(
                fact,
                dim,
                vec![(
                    Arc::new(Column::new("f_dim", 0)),
                    Arc::new(Column::new("d_id", 0)),
                )],
                Some(filter),
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap(),
        ) as Arc<dyn ExecutionPlan>;
        let join_schema = join.schema(); // [f_dim, f_amount, f_date, d_id, d_name, d_since]

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
            Arc::new(Column::new("d_name", 4)) as Arc<dyn PhysicalExpr>,
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

        let eager_on = run_rule(Arc::clone(&plan));
        assert!(
            join_child_is_aggregate(&eager_on),
            "eager aggregation should have fired"
        );
        assert!(
            any_join_has_filter(&eager_on),
            "rebuilt join must retain the residual filter"
        );
        let on = enforce_distribution(eager_on);
        let off = enforce_distribution(plan);

        let ctx = Arc::new(TaskContext::default());
        let on_rows = collect(on, Arc::clone(&ctx)).await.unwrap();
        let off_rows = collect(off, ctx).await.unwrap();

        // Expected: a -> 10+20 = 30 (date>=4); b -> 40+50 = 90 (date>=6).
        let total: usize = off_rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2, "two groups expected");
        assert_eq!(
            sorted_rows(&on_rows),
            sorted_rows(&off_rows),
            "eager-on results must match eager-off with a cross-side filter"
        );
    }
}
