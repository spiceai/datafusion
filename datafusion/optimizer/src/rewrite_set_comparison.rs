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

//! Optimizer rule rewriting `SetComparison` subqueries (e.g. `= ANY`,
//! `> ALL`) into boolean expressions built from `EXISTS` subqueries
//! that capture SQL three-valued logic.

use crate::utils::NamePreserver;
use crate::{OptimizerConfig, OptimizerRule};
use datafusion_common::alias::AliasGenerator;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Column, DFSchema, ExprSchema, Result, ScalarValue, plan_err};
use datafusion_expr::expr::{self, Exists, SetComparison, SetQuantifier};
use datafusion_expr::logical_plan::builder::LogicalPlanBuilder;
use datafusion_expr::logical_plan::{Filter, Projection, Subquery, SubqueryAlias};
use datafusion_expr::{DmlStatement, Expr, LogicalPlan, WriteOp, lit};
use std::sync::Arc;

use datafusion_expr::utils::merge_schema;

/// Rewrite `SetComparison` expressions to scalar subqueries that return the
/// correct boolean value (including SQL NULL semantics). After this rule
/// runs, later rules such as `ScalarSubqueryToJoin` can decorrelate and
/// remove the remaining subquery.
#[derive(Debug, Default)]
pub struct RewriteSetComparison;

impl RewriteSetComparison {
    /// Create a new `RewriteSetComparison` optimizer rule.
    pub fn new() -> Self {
        Self
    }

    fn rewrite_plan(
        &self,
        plan: LogicalPlan,
        alias_generator: &Arc<AliasGenerator>,
    ) -> Result<Transformed<LogicalPlan>> {
        let mut schema = merge_schema(&plan.inputs());
        if let LogicalPlan::Dml(DmlStatement {
            op: WriteOp::MergeInto(_),
            table_name,
            target,
            ..
        }) = &plan
        {
            schema.merge(&DFSchema::try_from_qualified_schema(
                table_name.clone(),
                &target.schema(),
            )?);
        }
        // This rule replaces `x > ALL (…)` with a `CASE WHEN EXISTS …`, whose display
        // name is not the name of the field it produces. Rewriting an expression
        // without preserving its name therefore changes the output schema, and the
        // optimizer's "the schema is invariant under optimization" check fails with an
        // internal error rather than returning the answer. `simplify_exprs` and
        // `type_coercion` preserve names around `map_expressions` for the same reason;
        // doing it here also stops this rule leaving mismatched names in `Aggregate`
        // and `Window` nodes, which a projection-only fix could not reach.
        let name_preserver = NamePreserver::new(&plan);
        let correlated_scalar = match &plan {
            LogicalPlan::Projection(_) | LogicalPlan::Filter(_) => {
                CorrelatedScalarSubquery::Allowed
            }
            _ => CorrelatedScalarSubquery::NotAllowed,
        };
        plan.map_expressions(|expr| {
            let original_name = name_preserver.save(&expr);
            expr.transform_up(|expr| {
                rewrite_set_comparison(expr, &schema, alias_generator, correlated_scalar)
            })
            .map(|t| t.update_data(|expr| original_name.restore(expr)))
        })
    }
}

impl OptimizerRule for RewriteSetComparison {
    fn name(&self) -> &str {
        "rewrite_set_comparison"
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let alias_generator = config.alias_generator();
        plan.transform_up_with_subqueries(|plan| self.rewrite_plan(plan, alias_generator))
    }
}

/// Whether the plan node that holds a set comparison may hold a correlated
/// scalar subquery. Only projections and filters may: `ScalarSubqueryToJoin`
/// decorrelates only those, and the plan's subquery checks reject a correlated
/// scalar subquery in most other nodes.
#[derive(Clone, Copy)]
enum CorrelatedScalarSubquery {
    Allowed,
    NotAllowed,
}

fn rewrite_set_comparison(
    expr: Expr,
    outer_schema: &DFSchema,
    alias_generator: &Arc<AliasGenerator>,
    correlated_scalar: CorrelatedScalarSubquery,
) -> Result<Transformed<Expr>> {
    match expr {
        Expr::SetComparison(set_comparison) => {
            let rewritten = build_set_comparison_subquery(
                set_comparison,
                outer_schema,
                alias_generator,
                correlated_scalar,
            )?;
            Ok(Transformed::yes(rewritten))
        }
        _ => Ok(Transformed::no(expr)),
    }
}

fn build_set_comparison_subquery(
    set_comparison: SetComparison,
    outer_schema: &DFSchema,
    alias_generator: &Arc<AliasGenerator>,
    correlated_scalar: CorrelatedScalarSubquery,
) -> Result<Expr> {
    let SetComparison {
        expr,
        subquery,
        op,
        quantifier,
    } = set_comparison;

    let subquery_schema = subquery.subquery.schema();
    if subquery_schema.fields().is_empty() {
        return plan_err!("single expression required.");
    }
    // avoid `head_output_expr` for aggr/window plan, it will gives group-by expr if exists
    let first_column = Column::from(subquery_schema.qualified_field(0));

    // ANY and ALL over exactly one value are the comparison with that value.
    // The scalar form keeps the aggregate's row for an outer row that matches
    // no subquery row (`max` NULL, `count` 0), which the EXISTS form below
    // loses once decorrelation turns the subquery into a join.
    let scalar_allowed = subquery.outer_ref_columns.is_empty()
        || matches!(correlated_scalar, CorrelatedScalarSubquery::Allowed);
    if scalar_allowed && returns_exactly_one_row(&subquery.subquery) {
        let scalar = Subquery {
            subquery: Arc::new(
                LogicalPlanBuilder::from(subquery.subquery.as_ref().clone())
                    .project(vec![Expr::Column(first_column)])?
                    .build()?,
            ),
            ..subquery
        };
        return Ok(Expr::BinaryExpr(expr::BinaryExpr::new(
            expr,
            op,
            Box::new(Expr::ScalarSubquery(scalar)),
        )));
    }

    // A HAVING over that aggregate leaves zero rows or one, so ANY and ALL are
    // the comparison with the subquery's value when the HAVING holds, and false
    // (ANY) or true (ALL) when it does not. The HAVING is read as a scalar
    // subquery over the one-row aggregate. Decorrelation gives an outer row with
    // no inner row the aggregate's values over no input, which a constant does
    // not get and a subquery inside the HAVING cannot be evaluated for, so those
    // keep the EXISTS form.
    if scalar_allowed
        && let Some((having, one_row)) = split_having_over_one_row(&subquery.subquery)
        && !having.contains_outer()
        && !having.column_refs().is_empty()
        && !having.exists(|e| Ok(is_subquery(e)))?
        && reads_input(&subquery.subquery, 0)
    {
        let value = scalar_subquery(
            LogicalPlanBuilder::from(subquery.subquery.as_ref().clone())
                .project(vec![Expr::Column(first_column)])?
                .build()?,
            &subquery,
        );
        let having = scalar_subquery(
            LogicalPlanBuilder::from(one_row)
                .project(vec![having])?
                .build()?,
            &subquery,
        );
        let otherwise = match quantifier {
            SetQuantifier::Any => lit(false),
            SetQuantifier::All => lit(true),
        };
        return Ok(Expr::Case(expr::Case {
            expr: None,
            when_then_expr: vec![(
                Box::new(having),
                Box::new(Expr::BinaryExpr(expr::BinaryExpr::new(
                    expr,
                    op,
                    Box::new(value),
                ))),
            )],
            else_expr: Some(Box::new(otherwise)),
        }));
    }

    let left_expr = to_outer_reference(*expr, outer_schema)?;
    // The subquery can read a relation with the outer query's name, as in
    // `x = ANY (SELECT x FROM t)` over `t`. Decorrelation drops the outer
    // reference from `outer.x = inner.x`, so the subquery column gets a name
    // of its own.
    let value_name = alias_generator.next("__set_comparison_value");
    let subquery = Subquery {
        subquery: Arc::new(
            LogicalPlanBuilder::from(subquery.subquery.as_ref().clone())
                .project(vec![Expr::Column(first_column).alias(&value_name)])?
                .build()?,
        ),
        ..subquery
    };
    let right_expr = Expr::Column(Column::new_unqualified(value_name));

    let comparison = Expr::BinaryExpr(expr::BinaryExpr::new(
        Box::new(left_expr),
        op,
        Box::new(right_expr),
    ));

    let true_exists =
        exists_subquery(&subquery, Expr::IsTrue(Box::new(comparison.clone())))?;
    let null_exists =
        exists_subquery(&subquery, Expr::IsNull(Box::new(comparison.clone())))?;

    let result_expr = match quantifier {
        SetQuantifier::Any => Expr::Case(expr::Case {
            expr: None,
            when_then_expr: vec![
                (Box::new(true_exists), Box::new(lit(true))),
                (
                    Box::new(null_exists),
                    Box::new(Expr::Literal(ScalarValue::Boolean(None), None)),
                ),
            ],
            else_expr: Some(Box::new(lit(false))),
        }),
        SetQuantifier::All => {
            let false_exists =
                exists_subquery(&subquery, Expr::IsFalse(Box::new(comparison.clone())))?;
            Expr::Case(expr::Case {
                expr: None,
                when_then_expr: vec![
                    (Box::new(false_exists), Box::new(lit(false))),
                    (
                        Box::new(null_exists),
                        Box::new(Expr::Literal(ScalarValue::Boolean(None), None)),
                    ),
                ],
                else_expr: Some(Box::new(lit(true))),
            })
        }
    };

    Ok(result_expr)
}

/// An aggregate without `GROUP BY` returns one row, also over no input.
/// Projections and aliases above it keep that row; a filter (`HAVING`) or a
/// limit can drop it.
fn returns_exactly_one_row(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Projection(projection) => returns_exactly_one_row(&projection.input),
        LogicalPlan::SubqueryAlias(alias) => returns_exactly_one_row(&alias.input),
        LogicalPlan::Aggregate(aggregate) => aggregate.group_expr.is_empty(),
        _ => false,
    }
}

/// The HAVING of a subquery whose root is a filter over a plan that returns
/// exactly one row, under projections and aliases, and the plan under it.
fn split_having_over_one_row(plan: &LogicalPlan) -> Option<(Expr, LogicalPlan)> {
    match plan {
        LogicalPlan::Projection(Projection { input, .. })
        | LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. }) => {
            split_having_over_one_row(input)
        }
        LogicalPlan::Filter(filter) if returns_exactly_one_row(&filter.input) => {
            Some((filter.predicate.clone(), filter.input.as_ref().clone()))
        }
        _ => None,
    }
}

/// Whether output column `index` of `plan` is computed from an aggregate or a
/// column, rather than being a constant.
fn reads_input(plan: &LogicalPlan, index: usize) -> bool {
    match plan {
        LogicalPlan::Projection(projection) => {
            let Some(expr) = projection.expr.get(index) else {
                return false;
            };
            match expr.clone().unalias_nested().data {
                Expr::Column(column) => projection
                    .input
                    .schema()
                    .maybe_index_of_column(&column)
                    .is_some_and(|i| reads_input(&projection.input, i)),
                expr => !expr.column_refs().is_empty(),
            }
        }
        LogicalPlan::SubqueryAlias(SubqueryAlias { input, .. })
        | LogicalPlan::Filter(Filter { input, .. }) => reads_input(input, index),
        LogicalPlan::Aggregate(_) => true,
        _ => false,
    }
}

fn is_subquery(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Exists(_)
            | Expr::InSubquery(_)
            | Expr::ScalarSubquery(_)
            | Expr::SetComparison(_)
    )
}

fn scalar_subquery(plan: LogicalPlan, subquery: &Subquery) -> Expr {
    let outer_ref_columns = plan.all_out_ref_exprs();
    Expr::ScalarSubquery(Subquery {
        subquery: Arc::new(plan),
        outer_ref_columns,
        spans: subquery.spans.clone(),
    })
}

fn exists_subquery(subquery: &Subquery, filter: Expr) -> Result<Expr> {
    let plan = LogicalPlanBuilder::from(subquery.subquery.as_ref().clone())
        .filter(filter)?
        .build()?;
    let outer_ref_columns = plan.all_out_ref_exprs();
    Ok(Expr::Exists(Exists {
        subquery: Subquery {
            subquery: Arc::new(plan),
            outer_ref_columns,
            spans: subquery.spans.clone(),
        },
        negated: false,
    }))
}

fn to_outer_reference(expr: Expr, outer_schema: &DFSchema) -> Result<Expr> {
    expr.transform_up(|expr| match expr {
        Expr::Column(col) => {
            let field = outer_schema.field_from_column(&col)?;
            Ok(Transformed::yes(Expr::OuterReferenceColumn(
                Arc::clone(field),
                col,
            )))
        }
        Expr::OuterReferenceColumn(_, _) => Ok(Transformed::no(expr)),
        _ => Ok(Transformed::no(expr)),
    })
    .map(|t| t.data)
}
