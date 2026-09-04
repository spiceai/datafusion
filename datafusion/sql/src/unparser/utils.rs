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

use std::{cmp::Ordering, collections::HashSet, sync::Arc, vec};

use super::{
    Unparser, dialect::CharacterLengthStyle, dialect::DateFieldExtractStyle,
    rewrite::TableAliasRewriter,
};
use arrow::datatypes::DataType;
use datafusion_common::{
    Column, DFSchema, DataFusionError, Result, ScalarValue, TableReference,
    assert_eq_or_internal_err, internal_err, not_impl_err,
    tree_node::{Transformed, TransformedResult, TreeNode},
};
use datafusion_expr::{
    Aggregate, Distinct, DistinctOn, Expr, LogicalPlan, LogicalPlanBuilder, Projection,
    SortExpr, Unnest, Window, expr,
    expr::{Cast, TryCast},
    utils::grouping_set_to_exprlist,
};

use indexmap::IndexSet;
use sqlparser::ast;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::tokenizer::Span;

/// Recursively searches children of [LogicalPlan] to find an Aggregate node if exists
/// prior to encountering a Join, TableScan, or a nested subquery (derived table factor).
/// If an Aggregate or node is not found prior to this or at all before reaching the end
/// of the tree, None is returned.
pub(crate) fn find_agg_node_within_select(
    plan: &LogicalPlan,
    already_projected: bool,
) -> Option<&Aggregate> {
    // Note that none of the nodes that have a corresponding node can have more
    // than 1 input node. E.g. Projection / Filter always have 1 input node.
    let input = plan.inputs();
    let input = if input.len() > 1 {
        return None;
    } else {
        input.first()?
    };
    // Agg nodes explicitly return immediately with a single node
    if let LogicalPlan::Aggregate(agg) = input {
        Some(agg)
    } else if matches!(
        input,
        LogicalPlan::TableScan(_)
            | LogicalPlan::Subquery(_)
            | LogicalPlan::SubqueryAlias(_)
    ) {
        None
    } else if let LogicalPlan::Projection(_) = input {
        if already_projected {
            None
        } else {
            find_agg_node_within_select(input, true)
        }
    } else {
        find_agg_node_within_select(input, already_projected)
    }
}

/// Recursively searches children of [LogicalPlan] to find Unnest node if exist
pub(crate) fn find_unnest_node_within_select(plan: &LogicalPlan) -> Option<&Unnest> {
    // Note that none of the nodes that have a corresponding node can have more
    // than 1 input node. E.g. Projection / Filter always have 1 input node.
    let input = plan.inputs();
    let input = if input.len() > 1 {
        return None;
    } else {
        input.first()?
    };

    if let LogicalPlan::Unnest(unnest) = input {
        Some(unnest)
    } else if let LogicalPlan::TableScan(_) = input {
        None
    } else if let LogicalPlan::Projection(_) = input {
        None
    } else {
        find_unnest_node_within_select(input)
    }
}

/// Recursively searches children of [LogicalPlan] to find Unnest node if exist
/// until encountering a Relation node with single input
pub(crate) fn find_unnest_node_until_relation(plan: &LogicalPlan) -> Option<&Unnest> {
    // Note that none of the nodes that have a corresponding node can have more
    // than 1 input node. E.g. Projection / Filter always have 1 input node.
    let input = plan.inputs();
    let input = if input.len() > 1 {
        return None;
    } else {
        input.first()?
    };

    if let LogicalPlan::Unnest(unnest) = input {
        Some(unnest)
    } else if let LogicalPlan::TableScan(_) = input {
        None
    } else if let LogicalPlan::Subquery(_) = input {
        None
    } else if let LogicalPlan::SubqueryAlias(_) = input {
        None
    } else {
        find_unnest_node_within_select(input)
    }
}

/// Recursively searches children of [LogicalPlan] to find Window nodes if exist
/// prior to encountering a Join, TableScan, or a nested subquery (derived table factor).
/// If Window node is not found prior to this or at all before reaching the end
/// of the tree, None is returned.
pub(crate) fn find_window_nodes_within_select<'a>(
    plan: &'a LogicalPlan,
    mut prev_windows: Option<Vec<&'a Window>>,
    already_projected: bool,
) -> Option<Vec<&'a Window>> {
    // Note that none of the nodes that have a corresponding node can have more
    // than 1 input node. E.g. Projection / Filter always have 1 input node.
    let input = plan.inputs();
    let input = if input.len() > 1 {
        return prev_windows;
    } else {
        input.first()?
    };

    // Window nodes accumulate in a vec until encountering a TableScan or 2nd projection
    match input {
        LogicalPlan::Window(window) => {
            prev_windows = match &mut prev_windows {
                Some(windows) => {
                    windows.push(window);
                    prev_windows
                }
                _ => Some(vec![window]),
            };
            find_window_nodes_within_select(input, prev_windows, already_projected)
        }
        LogicalPlan::Projection(_) => {
            if already_projected {
                prev_windows
            } else {
                find_window_nodes_within_select(input, prev_windows, true)
            }
        }
        LogicalPlan::TableScan(_) => prev_windows,
        _ => find_window_nodes_within_select(input, prev_windows, already_projected),
    }
}

/// Recursively identify Column expressions and transform them into the appropriate unnest expression
///
/// For example, if expr contains the column expr "__unnest_placeholder(make_array(Int64(1),Int64(2),Int64(2),Int64(5),NULL),depth=1)"
/// it will be transformed into an actual unnest expression UNNEST([1, 2, 2, 5, NULL])
pub(crate) fn unproject_unnest_expr(expr: Expr, unnest: &Unnest) -> Result<Expr> {
    expr.transform(|sub_expr| {
            if let Expr::Column(col_ref) = &sub_expr {
                // Check if the column is among the columns to run unnest on.
                // Currently, only List/Array columns (defined in `list_type_columns`) are supported for unnesting.
                if unnest.list_type_columns.iter().any(|e| e.1.output_column.name == col_ref.name) {
                    if let Ok(idx) = unnest.schema.index_of_column(col_ref)
                        && let LogicalPlan::Projection(Projection { expr, .. }) = unnest.input.as_ref()
                            && let Some(unprojected_expr) = expr.get(idx) {
                                let unnest_expr = Expr::Unnest(expr::Unnest::new(unprojected_expr.clone()));
                                return Ok(Transformed::yes(unnest_expr));
                            }
                    return internal_err!(
                        "Tried to unproject unnest expr for column '{}' that was not found in the provided Unnest!", &col_ref.name
                    );
                }
            }

            Ok(Transformed::no(sub_expr))

        }).map(|e| e.data)
}

/// Like `unproject_unnest_expr`, but for Snowflake FLATTEN:
/// transforms `__unnest_placeholder(...)` column references into
/// `Expr::Column(Column { relation: Some(alias), name: "VALUE" })`.
pub(crate) fn unproject_unnest_expr_as_flatten_value(
    expr: Expr,
    unnest: &Unnest,
    flatten_alias: &str,
) -> Result<Expr> {
    expr.transform(|sub_expr| {
        if let Expr::Column(col_ref) = &sub_expr
            && unnest
                .list_type_columns
                .iter()
                .any(|e| e.1.output_column.name == col_ref.name)
        {
            let value_col = Expr::Column(Column::new(
                Some(TableReference::bare(flatten_alias)),
                "VALUE",
            ));
            return Ok(Transformed::yes(value_col));
        }
        Ok(Transformed::no(sub_expr))
    })
    .map(|e| e.data)
}

/// Recursively identify all Column expressions and transform them into the appropriate
/// aggregate expression contained in agg.
///
/// For example, if expr contains the column expr "COUNT(*)" it will be transformed
/// into an actual aggregate expression COUNT(*) as identified in the aggregate node.
pub(crate) fn unproject_agg_exprs(
    expr: Expr,
    agg: &Aggregate,
    windows: Option<&[&Window]>,
) -> Result<Expr> {
    expr.transform(|sub_expr| {
            if let Expr::Column(c) = sub_expr {
                if let Some(unprojected_expr) = find_agg_expr(agg, &c)? {
                    Ok(Transformed::yes(unprojected_expr.clone()))
                } else if let Some(unprojected_expr) =
                    windows.and_then(|w| find_window_expr(w, &c.name).cloned())
                {
                    // Window function can contain an aggregation columns, e.g., 'avg(sum(ss_sales_price)) over ...' that needs to be unprojected
                    Ok(Transformed::yes(unproject_agg_exprs(unprojected_expr, agg, None)?))
                } else {
                    internal_err!(
                        "Tried to unproject agg expr for column '{}' that was not found in the provided Aggregate!", &c.name
                    )
                }
            } else {
                Ok(Transformed::no(sub_expr))
            }
        })
        .map(|e| e.data)
}

/// The expression an `Alias`, however many layers of it, finally names.
fn strip_aliases(expr: &Expr) -> &Expr {
    let mut expr = expr;
    while let Expr::Alias(alias) = expr {
        expr = alias.expr.as_ref();
    }
    expr
}

/// Whether any select item *wraps* a computed grouping expression — reaches one
/// through a wrapper rather than being that reference itself.
///
/// This is the shape a dialect that resolves `GROUP BY` against whole select items
/// cannot bind (see
/// [`Dialect::group_by_matches_select_subexpressions`](super::dialect::Dialect::group_by_matches_select_subexpressions)).
/// Two shapes are deliberately not it:
///
/// * a grouping expression that is a bare column, because a column reference is
///   matched wherever it appears, nested or not;
/// * a wrapped *aggregate*, which every dialect accepts — `CAST(min(t) AS DATE)`
///   is not a grouped column reference.
///
/// A bare copy of the grouping expression elsewhere in the select list does not
/// rescue a wrapped one, so this asks whether *any* item wraps, not whether all of
/// them do.
pub(crate) fn select_list_wraps_a_grouping_expr(exprs: &[Expr], agg: &Aggregate) -> bool {
    // A `ROLLUP`/`CUBE`/`GROUPING SETS` is one `Expr::GroupingSet` covering several
    // grouping outputs, and adds an internal grouping-id output of its own, so the
    // positional pairing below does not describe it. Such a plan keeps the flat
    // rendering it has today rather than being paired up wrongly.
    if agg
        .group_expr
        .iter()
        .any(|expr| matches!(strip_aliases(expr), Expr::GroupingSet(_)))
    {
        return false;
    }
    // The scope names its outputs by the name its schema reports, which drops the
    // qualifier, so two outputs the schema tells apart only by qualifier — grouping
    // a join by `left.id` and `right.id` — would collide into one name and the
    // rewrite would fail to build. Leave that shape as it renders today.
    let mut names = HashSet::new();
    if !agg
        .schema
        .fields()
        .iter()
        .all(|field| names.insert(field.name()))
    {
        return false;
    }
    // An Aggregate's schema reports its grouping outputs first, so zipping pairs
    // each group expression with the field the projection refers to it by.
    let computed_group_outputs: HashSet<Column> = agg
        .group_expr
        .iter()
        .zip(agg.schema.iter())
        .filter(|(group_expr, _)| !matches!(strip_aliases(group_expr), Expr::Column(_)))
        .map(|(_, (qualifier, field))| Column::new(qualifier.cloned(), field.name()))
        .collect();
    if computed_group_outputs.is_empty() {
        return false;
    }

    exprs.iter().any(|expr| {
        let item = strip_aliases(expr);
        // The item *is* the reference, so nothing wraps it.
        if matches!(item, Expr::Column(_)) {
            return false;
        }
        item.exists(|sub| {
            Ok(matches!(sub, Expr::Column(column) if computed_group_outputs.contains(column)))
        })
        .unwrap_or(false)
    })
}

/// `plan` with a projection that names every one of its outputs, so an enclosing
/// scope can address them.
///
/// A derived table is referred to by the names its schema reports, and an
/// `Aggregate` leaves every computed output for the engine to name. The names go
/// into a projection *above* the plan rather than into the `Aggregate` itself,
/// because `aggr_expr` has to stay aggregate expressions.
///
/// Each alias is the name the schema already reports, so the enclosing scope's
/// references still resolve.
pub(crate) fn name_scope_outputs(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let exprs = plan
        .schema()
        .iter()
        .map(|(qualifier, field)| {
            Expr::Column(Column::new(qualifier.cloned(), field.name()))
                .alias(field.name().clone())
        })
        .collect::<Vec<_>>();
    Projection::try_new(exprs, Arc::new(plan.clone())).map(LogicalPlan::Projection)
}

/// Recursively identify all Column expressions and transform them into the appropriate
/// window expression contained in window.
///
/// For example, if expr contains the column expr "COUNT(*) PARTITION BY id" it will be transformed
/// into an actual window expression as identified in the window node.
pub(crate) fn unproject_window_exprs(expr: Expr, windows: &[&Window]) -> Result<Expr> {
    expr.transform(|sub_expr| {
        if let Expr::Column(c) = sub_expr {
            if let Some(unproj) = find_window_expr(windows, &c.name) {
                Ok(Transformed::yes(unproj.clone()))
            } else {
                Ok(Transformed::no(Expr::Column(c)))
            }
        } else {
            Ok(Transformed::no(sub_expr))
        }
    })
    .map(|e| e.data)
}

/// Recursively searches children of [LogicalPlan] for the [Projection] that will be
/// flattened into the same `SELECT` as `plan`, if there is one.
///
/// Stops at a node that opens a scope of its own: the columns of a relation, a
/// subquery, or a derived table are addressable by name from outside, so a
/// reference to one of them already binds.
pub(crate) fn find_projection_node_within_select(
    plan: &LogicalPlan,
    already_projected: bool,
) -> Option<&Projection> {
    // A projection reached once the `SELECT` list is taken becomes a derived
    // table rather than part of this statement, so it is not what a predicate
    // here would be referring to.
    if already_projected {
        return None;
    }
    let input = plan.inputs();
    if input.len() > 1 {
        return None;
    }
    let input = input.first()?;
    match input {
        LogicalPlan::Projection(projection) => Some(projection),
        // Stacked filters collapse into one `WHERE`, so keep looking through them.
        LogicalPlan::Filter(_) => {
            find_projection_node_within_select(input, already_projected)
        }
        _ => None,
    }
}

/// Names the outputs a derived table exposes, where the [Projection] that supplies
/// them leaves them unnamed.
///
/// A derived table is addressed from the scope that encloses it, and the unparser
/// spells such a reference with the output's logical name — `t.a + t.b` for
/// `Projection: t.a + t.b`. That name describes the expression; it is not an
/// identifier the derived table carries, since an engine names an unaliased
/// expression itself (`?column?` on PostgreSQL). Aliasing each unnamed output to
/// the logical name the enclosing scope uses makes the reference bind, and — unlike
/// repeating the expression at the point of use — evaluates it exactly once.
///
/// Returns `None` when nothing needs naming, so the plan is unparsed as it stands.
pub(crate) fn name_derived_scope_outputs(
    plan: &LogicalPlan,
) -> Result<Option<LogicalPlan>> {
    let Some(named) = name_scope_projection_outputs(plan)? else {
        return Ok(None);
    };
    // Naming an output must not rename it: the alias is the name the schema
    // already reports, so the enclosing scope's references still resolve. That
    // holds by construction — the alias is taken from the schema itself — and a
    // node added to the walk below that renames its outputs would break it.
    debug_assert!(
        named
            .schema()
            .logically_equivalent_names_and_types(plan.schema()),
        "naming a derived table's outputs renamed them: {} became {}",
        plan.schema(),
        named.schema()
    );
    Ok(Some(named))
}

/// Aliases the unnamed outputs of the [Projection] that becomes this scope's `SELECT`
/// list, rebuilding the nodes walked through on the way down to it.
///
/// The nodes walked through are the ones that carry the projection's output names
/// out to the derived table unchanged. [`find_projection_node_within_select`] walks
/// the same way for a different purpose and stops at a narrower set, because a
/// predicate can only reach a projection that stays in its own `SELECT`.
fn name_scope_projection_outputs(plan: &LogicalPlan) -> Result<Option<LogicalPlan>> {
    // Each node is rebuilt by replacing its input and leaving every other field as
    // it stands. `LogicalPlan::with_new_exprs` is the shorter spelling and the
    // wrong one: it reconstructs a node from the expressions a plan reports, and
    // that round trip does not carry a `DISTINCT ON`'s sort expressions — it
    // panics on a plan that has them.
    match plan {
        LogicalPlan::Projection(projection) => name_projection_outputs(projection),
        // The nodes below carry the projection's output names out to the derived
        // table unchanged, so the projection under them is still the one exposing
        // its columns.
        LogicalPlan::Filter(filter) => with_named_input(&filter.input, |input| {
            let mut filter = filter.clone();
            filter.input = input;
            LogicalPlan::Filter(filter)
        }),
        LogicalPlan::Sort(sort) => with_named_input(&sort.input, |input| {
            let mut sort = sort.clone();
            sort.input = input;
            LogicalPlan::Sort(sort)
        }),
        LogicalPlan::Limit(limit) => with_named_input(&limit.input, |input| {
            let mut limit = limit.clone();
            limit.input = input;
            LogicalPlan::Limit(limit)
        }),
        LogicalPlan::Distinct(Distinct::All(input)) => {
            with_named_input(input, |input| LogicalPlan::Distinct(Distinct::All(input)))
        }
        // A `DISTINCT ON` is the one node here that emits its own `SELECT` list:
        // `select_expr` becomes the projection, so those — not the projection
        // below — are the outputs the enclosing scope binds. Naming only the input
        // therefore misses a computed one: `DISTINCT ON (a) a + b` straight off a
        // scan emits `(t.a + t.b)` for the engine to name. Both are named, since
        // the input is emitted as its own derived table (the projection beneath
        // sees `already_projected()`) and an enclosing scope can reach either.
        LogicalPlan::Distinct(Distinct::On(distinct_on)) => {
            let named_input = name_scope_projection_outputs(&distinct_on.input)?;
            let named_select = name_distinct_on_outputs(distinct_on);
            if named_input.is_none() && named_select.is_none() {
                return Ok(None);
            }
            let mut distinct_on = distinct_on.clone();
            if let Some(input) = named_input {
                distinct_on.input = Arc::new(input);
            }
            if let Some(select_expr) = named_select {
                distinct_on.select_expr = select_expr;
            }
            Ok(Some(LogicalPlan::Distinct(Distinct::On(distinct_on))))
        }
        LogicalPlan::SubqueryAlias(subquery_alias) => {
            with_named_input(&subquery_alias.input, |input| {
                let mut subquery_alias = subquery_alias.clone();
                subquery_alias.input = input;
                LogicalPlan::SubqueryAlias(subquery_alias)
            })
        }
        _ => Ok(None),
    }
}

/// Rebuilds a node around a named input, or `None` where the input needs no naming.
fn with_named_input(
    input: &Arc<LogicalPlan>,
    rebuild: impl FnOnce(Arc<LogicalPlan>) -> LogicalPlan,
) -> Result<Option<LogicalPlan>> {
    Ok(name_scope_projection_outputs(input)?.map(|named| rebuild(Arc::new(named))))
}

/// The projection with each unnamed output aliased to the name its schema reports.
fn name_projection_outputs(projection: &Projection) -> Result<Option<LogicalPlan>> {
    let Some(named) =
        name_unnamed_outputs(&projection.expr, &projection.schema, |_| false)
    else {
        return Ok(None);
    };
    Projection::try_new(named, Arc::clone(&projection.input))
        .map(|projection| Some(LogicalPlan::Projection(projection)))
}

/// The `select_expr` of a `DISTINCT ON` with each unnamed output aliased, or `None`
/// where none needs it.
///
/// Skips any output whose name the node's own `ON` or `ORDER BY` already spells as a
/// bare column, because an output alias would silently capture that reference. In
/// PostgreSQL both clauses resolve a bare name against the output list *first*, so for
///
/// ```text
/// on_expr:     col("a + b")            // the input column of that name
/// select_expr: col("a") + col("b")     // logical name, also "a + b"
/// ```
///
/// aliasing the output rebinds `DISTINCT ON ("a + b")` from the input column to the sum
/// — the same rows grouped by a different key. Leaving that output unnamed keeps the
/// enclosing reference unbound, which is the pre-existing bug rather than a new wrong
/// answer, so it is the safe side to fail to. Emitting the key qualified (or naming the
/// derived table through a column-alias list) would fix both; see
/// spiceai/spiceai#13444.
///
/// Only a *bare* reference is at risk: a qualified `t."a + b"` resolves to the relation,
/// and a name inside a larger expression resolves to the input columns, in both clauses.
///
/// Names are compared case-folded, because quoting does not say how the engine compares
/// what was written: DuckDB matches identifiers case-insensitively even quoted. Comparing
/// byte-for-byte would leave a key spelled `a + b` unmatched against an output named
/// `A + B`, emit `AS "A + B"`, and capture the key on exactly those dialects — the
/// failure this guard exists to prevent. Folding instead over-refuses on a
/// case-sensitive dialect, which costs an unbound reference rather than rows, and is the
/// same trade and the same unconditional fold as
/// [`Unparser::identifier_comparison_key`]; asking the dialect properly is
/// spiceai/spiceai#13474.
fn name_distinct_on_outputs(distinct_on: &DistinctOn) -> Option<Vec<Expr>> {
    let key_names: Vec<String> = distinct_on
        .on_expr
        .iter()
        .chain(
            distinct_on
                .sort_expr
                .iter()
                .flatten()
                .map(|sort| &sort.expr),
        )
        .filter_map(|expr| match expr {
            // Both variants unparse through the same `col_to_sql`, so an unqualified
            // one of either emits a bare name that an output alias can capture.
            Expr::Column(column) | Expr::OuterReferenceColumn(_, column)
                if column.relation.is_none() =>
            {
                Some(column.name.to_lowercase())
            }
            _ => None,
        })
        .collect();

    name_unnamed_outputs(&distinct_on.select_expr, &distinct_on.schema, |name| {
        key_names.contains(&name.to_lowercase())
    })
}

/// `exprs` with each unnamed output aliased to the name `schema` reports for it, or
/// `None` where no output is both unnamed and nameable.
///
/// `schema` is the schema the expressions produce, so its fields are positionally
/// aligned with them — true of a [`Projection`] and of a `DISTINCT ON`, whose schema is
/// likewise built from its `select_expr` alone.
///
/// `is_reserved` names the outputs that must be left alone even when unnamed, for a
/// caller where introducing the alias would change what another clause resolves to.
fn name_unnamed_outputs(
    exprs: &[Expr],
    schema: &DFSchema,
    is_reserved: impl Fn(&str) -> bool,
) -> Option<Vec<Expr>> {
    let nameable =
        |expr: &Expr, name: &str| output_is_unnamed(expr) && !is_reserved(name);
    if !exprs
        .iter()
        .zip(schema.fields())
        .any(|(expr, field)| nameable(expr, field.name()))
    {
        return None;
    }
    Some(
        exprs
            .iter()
            .zip(schema.fields())
            .map(|(expr, field)| {
                if nameable(expr, field.name()) {
                    // The schema's name for the output, rather than a second derivation
                    // of it: this is the name the enclosing scope refers to it by.
                    expr.clone().alias(field.name().clone())
                } else {
                    expr.clone()
                }
            })
            .collect(),
    )
}

/// Whether the emitted `SELECT` leaves this output for the engine to name.
///
/// An alias names it explicitly and a bare column keeps the name it already had,
/// so in both cases the emitted name is the one the schema reports. A wildcard
/// stands for a list of columns that each keep their own name, and cannot take an
/// alias at all. Every other expression is emitted unnamed.
fn output_is_unnamed(expr: &Expr) -> bool {
    #[expect(deprecated)]
    !matches!(
        expr,
        Expr::Alias(_) | Expr::Column(_) | Expr::Wildcard { .. }
    )
}

/// Replaces a reference to a [Projection] output that the projection does not name
/// with the expression that produces it.
///
/// The unparser names such an output by its logical name — `t.a + t.b` for
/// `Projection: t.a + t.b` — which is a description of the expression, not an
/// identifier the emitted statement carries. Emitting it as one yields SQL that
/// no engine can bind, so the expression is inlined at the point of use instead,
/// which needs no name at all.
pub(crate) fn unproject_unnamed_projection_exprs(
    expr: Expr,
    projection: &Projection,
) -> Result<Expr> {
    expr.transform(|sub_expr| {
        if let Expr::Column(c) = &sub_expr
            && let Some(unprojected) = find_unnamed_projection_expr(projection, c)
        {
            return Ok(Transformed::yes(unprojected.clone()));
        }
        Ok(Transformed::no(sub_expr))
    })
    .map(|e| e.data)
}

/// The expression behind `column`, but only when the projection leaves it unnamed.
///
/// An alias names the output explicitly and a bare column carries the name it
/// already had, so in both cases the emitted `SELECT` carries a name matching the
/// reference and there is nothing to repair.
///
/// A volatile expression is left alone as well. Inlining evaluates it a second
/// time, in a clause that may see a different value than the `SELECT` list did,
/// which would answer the query with silently wrong rows. The unbindable
/// reference this repairs is at least a loud failure, so it is the safer of the
/// two to leave in place. Volatility is a fact about whether inlining is safe, not
/// about whether the output is named, so it is checked separately from
/// [`output_is_unnamed`].
fn find_unnamed_projection_expr<'a>(
    projection: &'a Projection,
    column: &Column,
) -> Option<&'a Expr> {
    let index = projection.schema.index_of_column(column).ok()?;
    let expr = projection.expr.get(index)?;
    (output_is_unnamed(expr) && !expr.is_volatile()).then_some(expr)
}

fn find_agg_expr<'a>(agg: &'a Aggregate, column: &Column) -> Result<Option<&'a Expr>> {
    if let Ok(index) = agg.schema.index_of_column(column) {
        if matches!(agg.group_expr.as_slice(), [Expr::GroupingSet(_)]) {
            // For grouping set expr, we must operate by expression list from the grouping set
            let grouping_expr = grouping_set_to_exprlist(agg.group_expr.as_slice())?;
            match index.cmp(&grouping_expr.len()) {
                Ordering::Less => Ok(grouping_expr.into_iter().nth(index)),
                Ordering::Equal => {
                    internal_err!(
                        "Tried to unproject column referring to internal grouping id"
                    )
                }
                Ordering::Greater => {
                    Ok(agg.aggr_expr.get(index - grouping_expr.len() - 1))
                }
            }
        } else {
            Ok(agg.group_expr.iter().chain(agg.aggr_expr.iter()).nth(index))
        }
    } else {
        Ok(None)
    }
}

fn find_window_expr<'a>(
    windows: &'a [&'a Window],
    column_name: &'a str,
) -> Option<&'a Expr> {
    windows
        .iter()
        .flat_map(|w| w.window_expr.iter())
        .find(|expr| expr.schema_name().to_string() == column_name)
}

/// Transforms all Column expressions in a sort expression into the actual expression from aggregation or projection if found.
/// This is required because if an ORDER BY expression is present in an Aggregate or Select, it is replaced
/// with a Column expression (e.g., "sum(catalog_returns.cr_net_loss)"). We need to transform it back to
/// the actual expression, such as sum("catalog_returns"."cr_net_loss").
pub(crate) fn unproject_sort_expr(
    mut sort_expr: SortExpr,
    agg: Option<&Aggregate>,
    windows: Option<&[&Window]>,
    input: &LogicalPlan,
) -> Result<SortExpr> {
    // When the *entire* sort key is a bare unqualified column reference that
    // maps to an explicitly aliased projection expression, the column name IS
    // the output-column alias — a valid top-level ORDER BY key in all dialects.
    // Return it as-is; inlining the full aliased expression (which may be a
    // complex window or arithmetic formula) is both unnecessary and harmful:
    // remote engines that re-unparse the SQL can produce dangling quoted
    // identifiers for inner aggregate/window references.
    //
    // Inlining IS still needed when the same alias appears *nested* inside a
    // larger sort expression (e.g. `CASE WHEN alias = 0 THEN ...`), because
    // PostgreSQL and similar dialects reject output-column aliases inside
    // expressions. That path is handled by the recursive `transform` below,
    // which is only reached when `sort_expr.expr` is not a bare column.
    if let Expr::Column(Column {
        relation: None,
        name,
        ..
    }) = &sort_expr.expr
        && let LogicalPlan::Projection(Projection { expr, schema, .. }) = input
        && let Some(idx) = schema.index_of_column_by_name(None, name)
        && let Some(Expr::Alias(_)) = expr.get(idx)
    {
        return Ok(sort_expr);
    }

    sort_expr.expr = sort_expr
        .expr
        .transform(|sub_expr| {
            match sub_expr {
                // Remove alias if present, because ORDER BY cannot use aliases
                Expr::Alias(alias) => Ok(Transformed::yes(*alias.expr)),
                // Qualified columns reference FROM relations directly; only
                // unqualified columns can name aggregate/projection outputs
                // that need unprojecting below.
                Expr::Column(Column {
                    relation: Some(_), ..
                }) => Ok(Transformed::no(sub_expr)),
                // In case of aggregation there could be columns containing aggregation functions we need to unproject
                Expr::Column(col)
                    if let Some(agg) = agg
                        && agg.schema.is_column_from_schema(&col) =>
                {
                    return Ok(Transformed::yes(unproject_agg_exprs(
                        Expr::Column(col),
                        agg,
                        None,
                    )?));
                }
                Expr::Column(col) => {
                    // When an expression in the `ORDER BY` contains an alias from the `SELECT`
                    // we need to transform it back to the actual expression so that it is
                    // valid SQL in all positions inside ORDER BY (PostgreSQL only allows bare
                    // output-column aliases as top-level sort keys, not inside larger expressions
                    // such as CASE WHEN).
                    //
                    // We do NOT re-inline when the underlying expression (after stripping aliases)
                    // is a plain Expr::Column (simple rename), an AggregateFunction, or a
                    // WindowFunction — those cases either don't need inlining or are handled
                    // separately by the aggregate/window unproject branches.
                    //
                    // When the inlined expression contains aggregate or window function output
                    // column references (e.g., `sum(ws_ext_sales_price) * 100 / window_result`),
                    // we apply best-effort unprojection so those references resolve to their
                    // actual function-call representations rather than emitting as quoted identifiers.
                    if let LogicalPlan::Projection(Projection { expr, schema, .. }) =
                        input
                        && let Ok(idx) = schema.index_of_column(&col)
                        && let Some(proj_expr) = expr.get(idx)
                    {
                        let unaliased = proj_expr.clone().unalias_nested().data;
                        if !matches!(
                            unaliased,
                            Expr::Column(_)
                                | Expr::AggregateFunction(_)
                                | Expr::WindowFunction(_)
                        ) {
                            // Resolve aggregate and window function output column
                            // references inside the inlined expression. Delegate to the
                            // same helpers the SELECT projection uses so nested
                            // references are fully unprojected — in particular a window
                            // whose argument is itself an aggregate output, e.g.
                            // `sum(sum(x)) OVER (...)`, where `unproject_agg_exprs`
                            // recurses into the substituted window expression. The
                            // previous inline transform did not, leaving the inner
                            // `sum(x)` as a dangling quoted identifier.
                            let resolved = match (agg, windows) {
                                (Some(agg), windows) => {
                                    unproject_agg_exprs(unaliased, agg, windows)?
                                }
                                (None, Some(windows)) => {
                                    unproject_window_exprs(unaliased, windows)?
                                }
                                (None, None) => unaliased,
                            };
                            return Ok(Transformed::yes(resolved));
                        }
                    }

                    Ok(Transformed::no(Expr::Column(col)))
                }
                _ => Ok(Transformed::no(sub_expr)),
            }
        })
        .map(|e| e.data)?;
    Ok(sort_expr)
}

/// What [`try_transform_to_simple_table_scan_with_filters`] peeled off a join input.
pub(crate) struct SimpleTableScan {
    /// The `TableScan` (optionally under a `SubqueryAlias`) with the filters and the
    /// `fetch` removed.
    pub plan: LogicalPlan,
    /// Every filter collected, from the `Filter` nodes and from the scan itself, in the
    /// order they were found.
    pub filters: Vec<Expr>,
    /// The subset of `filters` that came from the scan itself. The scan applies these
    /// *before* its `fetch`, while a `Filter` node above it applies afterwards — a
    /// distinction that only matters when there is a `fetch` to sit between them.
    pub scan_filters: Vec<Expr>,
    /// The scan's row limit.
    pub fetch: Option<usize>,
}

/// Iterates through the children of a [LogicalPlan] to find a TableScan node before encountering
/// a Projection or any unexpected node that indicates the presence of a Projection (SELECT) in the plan.
/// If a TableScan node is found, returns the TableScan node without filters, along with the collected
/// filters and the scan's `fetch` separately.
/// If the plan contains a Projection, returns None.
///
/// The returned plan carries neither the filters nor the `fetch`: both are the caller's to re-emit,
/// and a caller that ignores either one silently widens the scan.
///
/// Note: If a table alias is present, TableScan filters are rewritten to reference the alias.
///
/// LogicalPlan example:
///   Filter: ta.j1_id < 5
///     Alias:  ta
///       TableScan: j1, j1_id > 10, fetch=5
///
/// Will return LogicalPlan below:
///     Alias:  ta
///       TableScan: j1
/// And filters: [ta.j1_id < 5, ta.j1_id > 10], fetch: Some(5)
pub(crate) fn try_transform_to_simple_table_scan_with_filters(
    plan: &LogicalPlan,
) -> Result<Option<SimpleTableScan>> {
    let mut filters: IndexSet<Expr> = IndexSet::new();
    let mut plan_stack = vec![plan];
    let mut table_alias = None;

    while let Some(current_plan) = plan_stack.pop() {
        match current_plan {
            LogicalPlan::SubqueryAlias(alias) => {
                table_alias = Some(alias.alias.clone());
                plan_stack.push(alias.input.as_ref());
            }
            LogicalPlan::Filter(filter) => {
                if !filters.contains(&filter.predicate) {
                    filters.insert(filter.predicate.clone());
                }
                plan_stack.push(filter.input.as_ref());
            }
            LogicalPlan::TableScan(table_scan) => {
                let table_schema = table_scan.source.schema();
                // optional rewriter if table has an alias
                let mut filter_alias_rewriter =
                    table_alias.as_ref().map(|alias_name| TableAliasRewriter {
                        table_schema: &table_schema,
                        alias_name: alias_name.clone(),
                    });

                // Rewrite already-collected Filter node predicates to use the
                // table alias so they can be properly deduplicated against the
                // rewritten TableScan filters below.
                if let Some(ref mut rewriter) = filter_alias_rewriter {
                    filters = filters
                        .into_iter()
                        .map(|expr| expr.rewrite(rewriter).data())
                        .collect::<Result<IndexSet<_>, _>>()?;
                }

                // rewrite filters to use table alias if present
                let table_scan_filters = table_scan
                    .filters
                    .iter()
                    .cloned()
                    .map(|expr| {
                        if let Some(ref mut rewriter) = filter_alias_rewriter {
                            expr.rewrite(rewriter).data()
                        } else {
                            Ok(expr)
                        }
                    })
                    .collect::<Result<Vec<_>, DataFusionError>>()?;

                let scan_filters = table_scan_filters.clone();
                for table_scan_filter in table_scan_filters {
                    if !filters.contains(&table_scan_filter) {
                        filters.insert(table_scan_filter);
                    }
                }

                let mut builder = LogicalPlanBuilder::scan(
                    table_scan.table_name.clone(),
                    Arc::clone(&table_scan.source),
                    table_scan.projection.clone(),
                )?;

                if let Some(alias) = table_alias.take() {
                    builder = builder.alias(alias)?;
                }

                let plan = builder.build()?;
                let filters = filters.into_iter().collect();

                return Ok(Some(SimpleTableScan {
                    plan,
                    filters,
                    scan_filters,
                    fetch: table_scan.fetch,
                }));
            }
            _ => {
                return Ok(None);
            }
        }
    }

    Ok(None)
}

/// Returns `true` if the expression contains a subquery (scalar, IN, or EXISTS).
pub(crate) fn expr_contains_subquery(expr: &Expr) -> bool {
    expr.exists(|e| {
        Ok(matches!(
            e,
            Expr::ScalarSubquery(_) | Expr::InSubquery(_) | Expr::Exists(_)
        ))
    })
    .unwrap_or(false)
}

/// Partitions filters into `(non_subquery, subquery)` based on whether
/// each filter contains a subquery expression.
pub(crate) fn partition_subquery_filters(filters: Vec<Expr>) -> (Vec<Expr>, Vec<Expr>) {
    filters
        .into_iter()
        .partition(|f| !expr_contains_subquery(f))
}

/// Converts a date_part function to SQL, tailoring it to the supported date field extraction style.
pub(crate) fn date_part_to_sql(
    unparser: &Unparser,
    style: DateFieldExtractStyle,
    date_part_args: &[Expr],
) -> Result<Option<ast::Expr>> {
    match (style, date_part_args.len()) {
        (DateFieldExtractStyle::Extract, 2) => {
            let date_expr = unparser.expr_to_sql(&date_part_args[1])?;
            if let Expr::Literal(ScalarValue::Utf8(Some(field)), _) = &date_part_args[0] {
                let field = match field.to_lowercase().as_str() {
                    "year" => ast::DateTimeField::Year,
                    "month" => ast::DateTimeField::Month,
                    "day" => ast::DateTimeField::Day,
                    "hour" => ast::DateTimeField::Hour,
                    "minute" => ast::DateTimeField::Minute,
                    "second" => ast::DateTimeField::Second,
                    _ => return Ok(None),
                };

                return Ok(Some(ast::Expr::Extract {
                    field,
                    expr: Box::new(date_expr),
                    syntax: ast::ExtractSyntax::From,
                }));
            }
        }
        (DateFieldExtractStyle::Strftime, 2) => {
            let column = unparser.expr_to_sql(&date_part_args[1])?;

            if let Expr::Literal(ScalarValue::Utf8(Some(field)), _) = &date_part_args[0] {
                let field = match field.to_lowercase().as_str() {
                    "year" => "%Y",
                    "month" => "%m",
                    "day" => "%d",
                    "hour" => "%H",
                    "minute" => "%M",
                    "second" => "%S",
                    _ => return Ok(None),
                };

                return Ok(Some(ast::Expr::Function(ast::Function {
                    name: ast::ObjectName::from(vec![ast::Ident {
                        value: "strftime".to_string(),
                        quote_style: None,
                        span: Span::empty(),
                    }]),
                    args: ast::FunctionArguments::List(ast::FunctionArgumentList {
                        duplicate_treatment: None,
                        args: vec![
                            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                                ast::Expr::value(ast::Value::SingleQuotedString(
                                    field.to_string(),
                                )),
                            )),
                            ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(column)),
                        ],
                        clauses: vec![],
                    }),
                    filter: None,
                    null_treatment: None,
                    over: None,
                    within_group: vec![],
                    parameters: ast::FunctionArguments::None,
                    uses_odbc_syntax: false,
                })));
            }
        }
        (DateFieldExtractStyle::DatePart, _) => {
            return Ok(Some(
                unparser.scalar_function_to_sql("date_part", date_part_args)?,
            ));
        }
        _ => {}
    };

    Ok(None)
}

pub(crate) fn character_length_to_sql(
    unparser: &Unparser,
    style: CharacterLengthStyle,
    character_length_args: &[Expr],
) -> Result<Option<ast::Expr>> {
    let func_name = match style {
        CharacterLengthStyle::CharacterLength => "character_length",
        CharacterLengthStyle::Length => "length",
    };

    Ok(Some(unparser.scalar_function_to_sql(
        func_name,
        character_length_args,
    )?))
}

/// SQLite does not support timestamp/date scalars like `to_timestamp`, `from_unixtime`, `date_trunc`, etc.
/// This remaps `from_unixtime` to `datetime(expr, 'unixepoch')`, expecting the input to be in seconds.
/// It supports no other arguments, so if any are supplied it will return an error.
///
/// # Errors
///
/// - If the number of arguments is not 1 - the column or expression to convert.
/// - If the scalar function cannot be converted to SQL.
pub(crate) fn sqlite_from_unixtime_to_sql(
    unparser: &Unparser,
    from_unixtime_args: &[Expr],
) -> Result<Option<ast::Expr>> {
    assert_eq_or_internal_err!(
        from_unixtime_args.len(),
        1,
        "from_unixtime for SQLite expects 1 argument, found {}",
        from_unixtime_args.len()
    );

    Ok(Some(unparser.scalar_function_to_sql(
        "datetime",
        &[
            from_unixtime_args[0].clone(),
            Expr::Literal(ScalarValue::Utf8(Some("unixepoch".to_string())), None),
        ],
    )?))
}

/// SQLite does not support timestamp/date scalars like `to_timestamp`, `from_unixtime`, `date_trunc`, etc.
/// This uses the `strftime` function to format the timestamp as a string depending on the truncation unit.
///
/// # Errors
///
/// - If the number of arguments is not 2 - truncation unit and the column or expression to convert.
/// - If the scalar function cannot be converted to SQL.
pub(crate) fn sqlite_date_trunc_to_sql(
    unparser: &Unparser,
    date_trunc_args: &[Expr],
) -> Result<Option<ast::Expr>> {
    assert_eq_or_internal_err!(
        date_trunc_args.len(),
        2,
        "date_trunc for SQLite expects 2 arguments, found {}",
        date_trunc_args.len()
    );

    if let Expr::Literal(ScalarValue::Utf8(Some(unit)), _) = &date_trunc_args[0] {
        let format = match unit.to_lowercase().as_str() {
            "year" => "%Y",
            "month" => "%Y-%m",
            "day" => "%Y-%m-%d",
            "hour" => "%Y-%m-%d %H",
            "minute" => "%Y-%m-%d %H:%M",
            "second" => "%Y-%m-%d %H:%M:%S",
            _ => return Ok(None),
        };

        return Ok(Some(unparser.scalar_function_to_sql(
            "strftime",
            &[
                Expr::Literal(ScalarValue::Utf8(Some(format.to_string())), None),
                date_trunc_args[1].clone(),
            ],
        )?));
    }

    Ok(None)
}

/// The type an expression states about itself, without a schema.
///
/// Only a cast and a literal carry their own type; everything else answers
/// `None`, including a column whose type is known only to the schema.
pub(crate) fn provable_data_type(expr: &Expr) -> Option<DataType> {
    match expr {
        Expr::Alias(alias) => provable_data_type(&alias.expr),
        Expr::Cast(Cast { field, .. }) | Expr::TryCast(TryCast { field, .. }) => {
            Some(field.data_type().clone())
        }
        Expr::Literal(value, _) => Some(value.data_type()),
        // A function's return type follows from its arguments, so it is provable
        // whenever they are. Without this a call is opaque, and a rendering that
        // needs the type declines — which is how `ts >= date_trunc(...)` kept its
        // zone disagreement: the column resolved, the call did not, and a
        // comparison needs both sides to agree on one.
        Expr::ScalarFunction(function) => {
            let argument_types = function
                .args
                .iter()
                .map(provable_data_type)
                .collect::<Option<Vec<_>>>()?;
            function.func.return_type(&argument_types).ok()
        }
        _ => None,
    }
}

/// Wraps `arg` in BigQuery's `TIMESTAMP(...)`, which reads a civil date or
/// date-and-time as an instant in UTC — the same reading DataFusion gives a
/// tz-naive value.
fn bigquery_as_instant(arg: ast::Expr) -> ast::Expr {
    bigquery_call("TIMESTAMP", vec![arg])
}

/// Builds a BigQuery function call from already-unparsed arguments.
fn bigquery_call(name: &str, args: Vec<ast::Expr>) -> ast::Expr {
    ast::Expr::Function(ast::Function {
        name: ast::ObjectName::from(vec![ast::Ident {
            value: name.to_string(),
            quote_style: None,
            span: Span::empty(),
        }]),
        args: ast::FunctionArguments::List(ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: args
                .into_iter()
                .map(|arg| ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(arg)))
                .collect(),
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
        parameters: ast::FunctionArguments::None,
        uses_odbc_syntax: false,
    })
}

/// Converts DataFusion's `to_unixtime(expr)` — seconds since the epoch — to
/// BigQuery's `UNIX_SECONDS`.
///
/// `UNIX_SECONDS` takes only a `TIMESTAMP`; it refuses a `DATETIME` ("No matching
/// signature for function UNIX_SECONDS"), so a civil operand is read as an
/// instant in UTC first, which is how DataFusion reads a tz-naive timestamp and a
/// date.
///
/// Returns `Ok(None)` when the operand's type is not evident from the expression,
/// leaving the default rendering, because the right form depends on that type.
pub(crate) fn bigquery_to_unixtime_to_sql(
    unparser: &Unparser,
    args: &[Expr],
) -> Result<Option<ast::Expr>> {
    let [arg] = args else {
        return Ok(None);
    };

    let Some(data_type) = unparser.resolved_data_type(arg) else {
        return Ok(None);
    };

    let operand = unparser.expr_to_sql(arg)?;
    let instant = match data_type {
        DataType::Timestamp(_, Some(_)) => operand,
        DataType::Timestamp(_, None) | DataType::Date32 => bigquery_as_instant(operand),
        // A number is already seconds and a string needs parsing rules BigQuery
        // does not share; neither is a rename.
        _ => return Ok(None),
    };

    Ok(Some(bigquery_call("UNIX_SECONDS", vec![instant])))
}

/// Converts DataFusion's `to_timestamp(expr)` to BigQuery.
///
/// DataFusion reads an integer as seconds since the epoch and returns a tz-naive
/// timestamp. BigQuery has no `TIMESTAMP(int)` ("No matching signature for
/// function TIMESTAMP"); `TIMESTAMP_SECONDS` is the integer form, and its result
/// is an instant, so it is brought back to the civil type the plan declares.
///
/// Returns `Ok(None)` for any other operand type, including one that is not
/// evident from the expression.
pub(crate) fn bigquery_to_timestamp_to_sql(
    unparser: &Unparser,
    args: &[Expr],
) -> Result<Option<ast::Expr>> {
    let [arg] = args else {
        return Ok(None);
    };

    let Some(data_type) = unparser.resolved_data_type(arg) else {
        return Ok(None);
    };

    if !data_type.is_integer() {
        return Ok(None);
    }

    let seconds = unparser.expr_to_sql(arg)?;
    Ok(Some(bigquery_call(
        "DATETIME",
        vec![bigquery_call("TIMESTAMP_SECONDS", vec![seconds])],
    )))
}

/// Renders a percentile over a group for BigQuery, which has no aggregate
/// percentile of its own.
///
/// The group is ordered into an array and the two values the percentile falls
/// between are read off it:
///
/// ```text
/// idx = p * (n - 1)
/// lo  = a[FLOOR(idx)]
/// hi  = a[CEIL(idx)]
/// ```
///
/// A median is `(lo + hi) / 2`, which is exact for any group size — at an integer
/// `idx` both offsets address the same element, so the average is that element —
/// and keeps the input's own type, so a `NUMERIC` column keeps its precision
/// rather than going through a float. That matches DataFusion, which averages the
/// two middle values and declares the same decimal type it was given, coercing
/// only integers to float, which BigQuery's `/` does too.
///
/// Any other percentile has to interpolate between the two, which brings in a
/// fractional weight and so a float result.
///
/// `IGNORE NULLS` matches DataFusion, where a null takes no position.
///
/// One cost worth stating: `ARRAY_AGG` materialises and sorts each group, where
/// DataFusion's `approx_percentile_cont` keeps bounded state. A group large
/// enough to exceed BigQuery's array limits will fail rather than degrade — a
/// visible error, not a wrong number, which is the better failure of the two, but
/// it does mean this is not a bounded-memory rendering.
///
/// `percentile_arg` is the index of the argument carrying the percentile, or
/// `None` for a median. Returns `Ok(None)` when the arguments are not the
/// expected shape, so the caller falls back rather than emitting something
/// malformed.
pub(crate) fn bigquery_percentile_to_sql(
    unparser: &Unparser,
    args: &[Expr],
    percentile_arg: Option<usize>,
    distinct: bool,
) -> Result<Option<ast::Expr>> {
    let Some(value) = args.first() else {
        return Ok(None);
    };

    let percentile = match percentile_arg {
        None => None,
        Some(i) => match args.get(i) {
            // Only a constant percentile can be folded into the offsets; a
            // per-row one would need a different shape entirely.
            Some(literal @ Expr::Literal(scalar, _)) if !scalar.is_null() => {
                Some(unparser.expr_to_sql(literal)?)
            }
            _ => return Ok(None),
        },
    };

    let value_sql = unparser.expr_to_sql(value)?;
    let sorted = ast::Expr::Function(ast::Function {
        name: ast::ObjectName::from(vec![ast::Ident {
            value: "ARRAY_AGG".to_string(),
            quote_style: None,
            span: Span::empty(),
        }]),
        args: ast::FunctionArguments::List(ast::FunctionArgumentList {
            duplicate_treatment: distinct.then_some(ast::DuplicateTreatment::Distinct),
            args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                value_sql.clone(),
            ))],
            clauses: vec![
                // Inside the parentheses, and before the ordering: BigQuery
                // takes both as argument clauses of ARRAY_AGG. A null takes no
                // position, which is how DataFusion reads it too.
                ast::FunctionArgumentClause::IgnoreOrRespectNulls(
                    ast::NullTreatment::IgnoreNulls,
                ),
                ast::FunctionArgumentClause::OrderBy(vec![ast::OrderByExpr {
                    expr: value_sql,
                    options: ast::OrderByOptions {
                        asc: None,
                        nulls_first: None,
                    },
                    with_fill: None,
                }]),
            ],
        }),
        null_treatment: None,
        filter: None,
        over: None,
        within_group: vec![],
        parameters: ast::FunctionArguments::None,
        uses_odbc_syntax: false,
    });

    // idx = p * (ARRAY_LENGTH(sorted) - 1), the 0-based position the percentile
    // falls at.
    let index = binary(
        percentile.clone().unwrap_or_else(|| number("0.5")),
        ast::BinaryOperator::Multiply,
        nest(binary(
            bigquery_call("ARRAY_LENGTH", vec![sorted.clone()]),
            ast::BinaryOperator::Minus,
            number("1"),
        )),
    );

    // The two elements it falls between. At an integer index both address the
    // same element, so the interpolation below degenerates to that element.
    let at = |round: &str| ast::Expr::CompoundFieldAccess {
        root: Box::new(sorted.clone()),
        access_chain: vec![ast::AccessExpr::Subscript(ast::Subscript::Index {
            index: bigquery_call(
                "OFFSET",
                vec![cast_to(
                    bigquery_call(round, vec![index.clone()]),
                    ast::DataType::Int64,
                )],
            ),
        })],
    };
    let (low, high) = (at("FLOOR"), at("CEIL"));

    let rendered = if percentile.is_none() {
        // A median needs no weight, so the input's own type survives.
        let mean = binary(
            nest(binary(nest(low), ast::BinaryOperator::Plus, nest(high))),
            ast::BinaryOperator::Divide,
            number("2"),
        );
        // DataFusion's median over a decimal averages the two middle values as
        // scaled integers, so the result is truncated to the input's own scale:
        // the median of 1 and 2 at scale 0 is 1, not 1.5. BigQuery's `/` keeps
        // the extra digits, so they are truncated off to match.
        match unparser
            .resolved_data_type(value)
            .as_ref()
            .and_then(decimal_scale)
        {
            Some(scale) => bigquery_call("TRUNC", vec![mean, number(&scale.to_string())]),
            None => mean,
        }
    } else {
        // low + (high - low) * (idx - FLOOR(idx))
        let float = |expr| cast_to(expr, ast::DataType::Float64);
        binary(
            float(low.clone()),
            ast::BinaryOperator::Plus,
            nest(binary(
                nest(binary(float(high), ast::BinaryOperator::Minus, float(low))),
                ast::BinaryOperator::Multiply,
                nest(binary(
                    index.clone(),
                    ast::BinaryOperator::Minus,
                    bigquery_call("FLOOR", vec![index]),
                )),
            )),
        )
    };

    Ok(Some(nest(rendered)))
}

/// The scale of a decimal type, when it is one and the scale is not negative.
///
/// A negative scale stores a value coarser than one unit, which no BigQuery
/// numeric type has, so there is nothing to truncate to.
fn decimal_scale(data_type: &DataType) -> Option<i8> {
    match data_type {
        DataType::Decimal32(_, scale)
        | DataType::Decimal64(_, scale)
        | DataType::Decimal128(_, scale)
        | DataType::Decimal256(_, scale) => (*scale >= 0).then_some(*scale),
        _ => None,
    }
}

/// `lhs op rhs`.
fn binary(lhs: ast::Expr, op: ast::BinaryOperator, rhs: ast::Expr) -> ast::Expr {
    ast::Expr::BinaryOp {
        left: Box::new(lhs),
        op,
        right: Box::new(rhs),
    }
}

/// `(expr)`, so composed arithmetic keeps the grouping it was built with.
fn nest(expr: ast::Expr) -> ast::Expr {
    ast::Expr::Nested(Box::new(expr))
}

/// An unquoted numeric literal.
fn number(value: &str) -> ast::Expr {
    ast::Expr::Value(ast::Value::Number(value.to_string(), false).into())
}

/// `CAST(expr AS data_type)`.
fn cast_to(expr: ast::Expr, data_type: ast::DataType) -> ast::Expr {
    ast::Expr::Cast {
        kind: ast::CastKind::Cast,
        expr: Box::new(expr),
        data_type,
        format: None,
        array: false,
    }
}

/// Re-emits a scalar function under the name BigQuery spells it, with the
/// arguments unchanged.
///
/// Only for functions whose BigQuery counterpart takes the same arguments in the
/// same order and means the same thing, so the rename is the whole translation.
/// A function whose BigQuery form depends on its argument *types* cannot be
/// handled here, because those types are not available at unparse time.
pub(crate) fn bigquery_renamed_scalar_fn(
    unparser: &Unparser,
    bigquery_name: &str,
    args: &[Expr],
) -> Result<Option<ast::Expr>> {
    let args = args
        .iter()
        .map(|arg| {
            Ok(ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                unparser.expr_to_sql(arg)?,
            )))
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Some(ast::Expr::Function(ast::Function {
        name: ast::ObjectName::from(vec![ast::Ident {
            value: bigquery_name.to_string(),
            quote_style: None,
            span: Span::empty(),
        }]),
        args: ast::FunctionArguments::List(ast::FunctionArgumentList {
            duplicate_treatment: None,
            args,
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
        parameters: ast::FunctionArguments::None,
        uses_odbc_syntax: false,
    })))
}

/// Converts DataFusion's `date_trunc(granularity, expr)` to BigQuery's
/// `TIMESTAMP_TRUNC(expr, GRANULARITY)`.
///
/// BigQuery's truncation function differs from DataFusion's in three ways, all
/// of which this rewrite handles:
/// 1. **Argument order is reversed** — the value comes first, the granularity
///    second (`TIMESTAMP_TRUNC(ts, MONTH)` vs `date_trunc('month', ts)`).
/// 2. **The granularity is a bare keyword**, not a quoted string literal
///    (`MONTH`, not `'month'`), so it is emitted as an unquoted identifier.
/// 3. The function name is type-specific. We always emit `TIMESTAMP_TRUNC`,
///    which is correct for the common case of a `Timestamp` (with timezone)
///    input. DataFusion's `date_trunc` also accepts `Timestamp` *without* a
///    timezone, `Date32`, and `Time32`/`Time64`, which in BigQuery have
///    `DATETIME_TRUNC`, `DATE_TRUNC` and `TIME_TRUNC` of their own. All of them
///    are still rendered as `TIMESTAMP_TRUNC`, which BigQuery accepts over a
///    `DATETIME` and returns a `DATETIME` for, so the common cases hold. The
///    operand's type is now reachable through [`Unparser::resolved_data_type`]
///    whenever plan unparsing supplied a schema, so dispatching properly here is
///    possible; it is left alone because no reported statement needs it.
///
/// Returns `Ok(None)` (falling back to default unparsing) when the arguments
/// don't match the expected shape or the granularity is not one BigQuery
/// supports, so the caller can decide how to proceed.
pub(crate) fn bigquery_date_trunc_to_sql(
    unparser: &Unparser,
    date_trunc_args: &[Expr],
) -> Result<Option<ast::Expr>> {
    // Fall back to default unparsing rather than erroring if the shape is
    // unexpected; date_trunc is documented as taking exactly 2 arguments.
    if date_trunc_args.len() != 2 {
        return Ok(None);
    }

    let Expr::Literal(ScalarValue::Utf8(Some(granularity)), _) = &date_trunc_args[0]
    else {
        return Ok(None);
    };

    // Map DataFusion granularities to BigQuery `TIMESTAMP_TRUNC` date parts.
    // https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions#timestamp_trunc
    let part = match granularity.to_lowercase().as_str() {
        "year" => "YEAR",
        "quarter" => "QUARTER",
        "month" => "MONTH",
        // DataFusion truncates weeks to Monday; BigQuery's bare `WEEK` is
        // Sunday-based, so use `ISOWEEK`, which truncates to Monday.
        "week" => "ISOWEEK",
        "day" => "DAY",
        "hour" => "HOUR",
        "minute" => "MINUTE",
        "second" => "SECOND",
        "millisecond" => "MILLISECOND",
        "microsecond" => "MICROSECOND",
        _ => return Ok(None),
    };

    let value_expr = unparser.expr_to_sql(&date_trunc_args[1])?;

    Ok(Some(ast::Expr::Function(ast::Function {
        name: ast::ObjectName::from(vec![ast::Ident {
            value: "TIMESTAMP_TRUNC".to_string(),
            quote_style: None,
            span: Span::empty(),
        }]),
        args: ast::FunctionArguments::List(ast::FunctionArgumentList {
            duplicate_treatment: None,
            args: vec![
                ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(value_expr)),
                // The date part must be an unquoted keyword, not a string literal.
                ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                    ast::Expr::Identifier(ast::Ident {
                        value: part.to_string(),
                        quote_style: None,
                        span: Span::empty(),
                    }),
                )),
            ],
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
        parameters: ast::FunctionArguments::None,
        uses_odbc_syntax: false,
    })))
}

/// Renders `array_element` for BigQuery.
///
/// DataFusion's `array_element` is 1-based, counts from the end for a negative
/// index, and yields NULL for an index outside the array. BigQuery's bare
/// subscript is 0-based, and its `ORDINAL` raises on a miss rather than yielding
/// NULL, so `SAFE_ORDINAL` is the form that agrees on every non-negative index.
///
/// BigQuery has no end-relative subscript, so an index that is not a
/// non-negative integer literal is refused: a bare subscript would read the
/// neighbouring element instead, and returning wrong rows is worse than
/// declining to render.
pub(crate) fn bigquery_array_element_to_sql(
    unparser: &Unparser,
    args: &[Expr],
) -> Result<Option<ast::Expr>> {
    let [array, index] = args else {
        return Ok(None);
    };

    let non_negative = match index {
        Expr::Literal(value, _) if value.data_type().is_integer() => matches!(
            value.cast_to(&DataType::Int64),
            Ok(ScalarValue::Int64(Some(index))) if index >= 0
        ),
        _ => false,
    };
    if !non_negative {
        return not_impl_err!(
            "BigQuery has no end-relative array subscript, so array_element needs a non-negative integer index"
        );
    }

    let array = unparser.expr_to_sql(array)?;
    let index = unparser.expr_to_sql(index)?;

    Ok(Some(ast::Expr::CompoundFieldAccess {
        root: Box::new(array),
        access_chain: vec![ast::AccessExpr::Subscript(ast::Subscript::Index {
            index: bigquery_call("SAFE_ORDINAL", vec![index]),
        })],
    }))
}

/// Parses text into a BigQuery timestamp, as a cast cannot.
///
/// `CAST(text AS DATETIME)` refuses any string carrying a `Z` or a UTC offset —
/// the very strings `CAST(text AS TIMESTAMP)` accepts — so a civil target has to
/// parse as an instant first and then take the civil value. Measured on
/// BigQuery: `'2026-09-02T04:53:18.789421+00:00'` casts to `TIMESTAMP` and not to
/// `DATETIME`, and `DATETIME(TIMESTAMP(x))` yields `2026-09-02 04:53:18.789421`.
///
/// Sub-second digits past six are dropped first, because neither cast takes them
/// and a BigQuery timestamp cannot hold them: `'…04.390436170Z'` fails both casts
/// outright. DataFusion reads nanoseconds here, so this narrows the value to what
/// the engine can represent rather than failing the statement.
pub(crate) fn bigquery_string_to_timestamp_to_sql(
    value: ast::Expr,
    instant: bool,
) -> ast::Expr {
    let microseconds = bigquery_call(
        "REGEXP_REPLACE",
        vec![value, raw_string(r"(\.\d{6})\d+"), raw_string(r"\1")],
    );
    let parsed = bigquery_call("TIMESTAMP", vec![microseconds]);
    if instant {
        parsed
    } else {
        bigquery_call("DATETIME", vec![parsed])
    }
}

/// A `R'...'` literal, so a regex's backslashes reach the engine unescaped.
fn raw_string(value: &str) -> ast::Expr {
    ast::Expr::Value(ast::Value::SingleQuotedRawStringLiteral(value.to_string()).into())
}

/// Renders an aggregate's `FILTER (WHERE ...)` for BigQuery, which has no such
/// clause and rejects the statement outright.
///
/// The predicate moves inside the aggregate. `COUNT(*)` becomes `COUNTIF(p)`;
/// everything else takes `CASE WHEN p THEN arg END`, which every aggregate here
/// reads the same way because they all skip nulls — measured on BigQuery over
/// `[1,2,3]` with `x > 1`: `COUNTIF` and `COUNT(CASE …)` both give 2, and
/// `SUM(CASE …)` gives 5.
///
/// Declines for a multi-argument aggregate, where which argument the predicate
/// should guard is not obvious and guessing would compute over the wrong rows.
pub(crate) fn bigquery_filtered_aggregate_to_sql(
    unparser: &Unparser,
    func_name: &str,
    args: &[Expr],
    distinct: bool,
    predicate: &Expr,
) -> Result<Option<ast::Expr>> {
    let condition = unparser.expr_to_sql(predicate)?;

    // `COUNT(*)` counts rows, so the predicate is the whole of it.
    let counts_rows = func_name.eq_ignore_ascii_case("count")
        && (args.is_empty() || matches!(args, [Expr::Literal(..)]));
    if counts_rows && !distinct {
        return Ok(Some(bigquery_call("COUNTIF", vec![condition])));
    }

    let [value] = args else {
        return Ok(None);
    };
    let guarded = ast::Expr::Case {
        case_token: AttachedToken::empty(),
        end_token: AttachedToken::empty(),
        operand: None,
        conditions: vec![ast::CaseWhen {
            condition,
            result: unparser.expr_to_sql(value)?,
        }],
        else_result: None,
    };

    Ok(Some(ast::Expr::Function(ast::Function {
        name: ast::ObjectName::from(vec![ast::Ident {
            value: func_name.to_string(),
            quote_style: None,
            span: Span::empty(),
        }]),
        args: ast::FunctionArguments::List(ast::FunctionArgumentList {
            duplicate_treatment: distinct.then_some(ast::DuplicateTreatment::Distinct),
            args: vec![ast::FunctionArg::Unnamed(ast::FunctionArgExpr::Expr(
                guarded,
            ))],
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
        parameters: ast::FunctionArguments::None,
        uses_odbc_syntax: false,
    })))
}
