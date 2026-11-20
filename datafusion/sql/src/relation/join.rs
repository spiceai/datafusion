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

use super::{LateralRelation, TableFunctionCall, is_table_function_call};
use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use arrow::datatypes::Fields;
use datafusion_common::{
    Column, DFSchema, NullHandling, Result, ScalarValue, TableReference, UnnestOptions,
    not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::lateral_table_function::{
    LateralTableFunctionCall, bind_outer_references, outer_references, placeholder_value,
};
use datafusion_expr::{Expr, JoinType, LogicalPlan, LogicalPlanBuilder};
use sqlparser::ast::{
    Expr as SQLExpr, Join, JoinConstraint, JoinOperator, ObjectName, TableFactor,
    TableWithJoins,
};
use std::collections::HashSet;
use std::sync::Arc;

impl<S: ContextProvider> SqlToRel<'_, S> {
    pub(crate) fn plan_table_with_joins(
        &self,
        t: TableWithJoins,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let left = if is_lateral(&t.relation) {
            self.create_relation_subquery(t.relation, planner_context)?
        } else {
            self.create_relation(t.relation, planner_context)?
        };
        self.plan_joins(left, t.joins, planner_context)
    }

    /// Plans `left, <input>`: an item of a comma-separated FROM list after
    /// the first one.
    pub(crate) fn plan_comma_join(
        &self,
        left: LogicalPlan,
        input: TableWithJoins,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        if !(is_lateral(&input.relation) && is_table_function_call(&input.relation)) {
            let right = self.plan_table_with_joins(input, planner_context)?;
            return LogicalPlanBuilder::from(left).cross_join(right)?.build();
        }

        let TableWithJoins { relation, joins } = input;
        match self.create_lateral_relation(relation, planner_context)? {
            LateralRelation::Plan(right) => {
                let right = self.plan_joins(right, joins, planner_context)?;
                LogicalPlanBuilder::from(left).cross_join(right)?.build()
            }
            LateralRelation::CorrelatedTableFunction(call) => {
                // `left, LATERAL f(..) AS r JOIN x ON c` is planned as
                // `(left CROSS JOIN LATERAL f(..) AS r) JOIN x ON c`. This is
                // equal when each later join keeps or filters the rows of its
                // left side, because `c` cannot reference `left`.
                if let Some(join) = joins
                    .iter()
                    .find(|join| !keeps_or_filters_left_rows(&join.join_operator))
                {
                    return not_impl_err!(
                        "{:?} after a LATERAL table function call with column arguments in a comma-separated FROM list",
                        join.join_operator
                    );
                }
                let plan = self.plan_lateral_table_function(
                    left,
                    call,
                    LateralJoinKind::Inner,
                    None,
                    planner_context,
                )?;
                self.plan_joins(plan, joins, planner_context)
            }
        }
    }

    /// Joins each of `joins` onto `left`, in order.
    fn plan_joins(
        &self,
        mut left: LogicalPlan,
        joins: Vec<Join>,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let old_outer_from_schema = planner_context.outer_from_schema();
        for join in joins {
            planner_context.extend_outer_from_schema(left.schema())?;
            left = self.parse_relation_join(left, join, planner_context)?;
        }
        planner_context.set_outer_from_schema(old_outer_from_schema);
        Ok(left)
    }

    pub(crate) fn parse_relation_join(
        &self,
        left: LogicalPlan,
        join: Join,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let right = if is_lateral_join(&join)? {
            match self.create_lateral_relation(join.relation, planner_context)? {
                LateralRelation::Plan(right) => right,
                LateralRelation::CorrelatedTableFunction(call) => {
                    return self.plan_lateral_table_function_join(
                        left,
                        call,
                        join.join_operator,
                        planner_context,
                    );
                }
            }
        } else {
            self.create_relation(join.relation, planner_context)?
        };
        match join.join_operator {
            JoinOperator::LeftOuter(constraint) | JoinOperator::Left(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Left, planner_context)
            }
            JoinOperator::RightOuter(constraint) | JoinOperator::Right(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Right, planner_context)
            }
            JoinOperator::Inner(constraint) | JoinOperator::Join(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Inner, planner_context)
            }
            JoinOperator::LeftSemi(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::LeftSemi,
                planner_context,
            ),
            JoinOperator::RightSemi(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::RightSemi,
                planner_context,
            ),
            JoinOperator::LeftAnti(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::LeftAnti,
                planner_context,
            ),
            JoinOperator::RightAnti(constraint) => self.parse_join(
                left,
                right,
                constraint,
                JoinType::RightAnti,
                planner_context,
            ),
            JoinOperator::FullOuter(constraint) => {
                self.parse_join(left, right, constraint, JoinType::Full, planner_context)
            }
            JoinOperator::CrossJoin(JoinConstraint::None) => {
                self.parse_cross_join(left, right)
            }
            other => not_impl_err!("Unsupported JOIN operator {other:?}"),
        }
    }

    /// Plans `left <join_operator> LATERAL f(args)` where `args` reference
    /// columns of `left`.
    fn plan_lateral_table_function_join(
        &self,
        left: LogicalPlan,
        call: TableFunctionCall,
        join_operator: JoinOperator,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let (kind, constraint) = match join_operator {
            JoinOperator::CrossJoin(JoinConstraint::None) | JoinOperator::CrossApply => {
                (LateralJoinKind::Inner, JoinConstraint::None)
            }
            JoinOperator::Inner(constraint) | JoinOperator::Join(constraint) => {
                (LateralJoinKind::Inner, constraint)
            }
            JoinOperator::Left(constraint) | JoinOperator::LeftOuter(constraint) => {
                (LateralJoinKind::Left, constraint)
            }
            JoinOperator::OuterApply => (LateralJoinKind::Left, JoinConstraint::None),
            other => {
                return not_impl_err!(
                    "Unsupported JOIN operator {other:?} for LATERAL table function '{}' with column arguments",
                    call.name
                );
            }
        };
        let on = match constraint {
            JoinConstraint::On(expr) => Some(expr),
            JoinConstraint::None => None,
            JoinConstraint::Using(_) | JoinConstraint::Natural => {
                return not_impl_err!(
                    "USING and NATURAL are not supported for LATERAL table function '{}' with column arguments",
                    call.name
                );
            }
        };
        self.plan_lateral_table_function(left, call, kind, on, planner_context)
    }

    /// Lowers a LATERAL table-function call whose arguments reference
    /// columns of `left` to existing operators:
    ///
    /// ```text
    /// Projection: <left columns>, <rows>.<col> AS r.<col>, ...
    ///   Unnest: <rows>                        (struct: one column per field)
    ///     Unnest: <rows>                      (list: one row per call row)
    ///       Projection: <left columns>, lateral_f(<outer columns>) AS <rows>
    ///         <left>
    /// ```
    ///
    /// `lateral_f` calls `f` once per distinct tuple of outer column values
    /// and returns the rows as `List<Struct<..>>`. An INNER join's ON
    /// condition becomes a filter on top. A LEFT join keeps a row whose call
    /// returned no rows by unnesting the empty list to one NULL row.
    fn plan_lateral_table_function(
        &self,
        left: LogicalPlan,
        call: TableFunctionCall,
        kind: LateralJoinKind,
        on: Option<SQLExpr>,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let TableFunctionCall {
            name, args, alias, ..
        } = call;

        let outer = outer_references(&args)?;
        if let Some((_, column)) = outer
            .iter()
            .find(|(_, column)| !left.schema().has_column(column))
        {
            return not_impl_err!(
                "LATERAL table function '{name}' references {column}, which is not a column of an earlier FROM item"
            );
        }
        let outer_columns: Vec<Column> =
            outer.iter().map(|(_, column)| column.clone()).collect();

        // Plan (but do not run) one call to learn the output schema.
        let placeholders = outer
            .iter()
            .map(|(field, _)| placeholder_value(field.data_type()))
            .collect::<Result<Vec<_>>>()?;
        let placeholder_args =
            bind_outer_references(&args, &outer_columns, &placeholders)?;
        let source = self
            .context_provider
            .get_table_function_source(&name, placeholder_args)?;
        // Any call can return NULLs after a LEFT join.
        let row_fields: Fields = source
            .schema()
            .fields()
            .iter()
            .map(|field| Arc::new(field.as_ref().clone().with_nullable(true)))
            .collect();
        if row_fields.is_empty() {
            return plan_err!("LATERAL table function '{name}' returns no columns");
        }

        let alias_name = match &alias {
            Some(alias) => self.ident_normalizer.normalize(alias.name.clone()),
            None => name.clone(),
        };
        let output_names: Vec<String> = match alias {
            Some(alias) if !alias.columns.is_empty() => {
                if alias.columns.len() != row_fields.len() {
                    return plan_err!(
                        "Source table contains {} columns but only {} \
                        names given as column alias",
                        row_fields.len(),
                        alias.columns.len()
                    );
                }
                alias
                    .columns
                    .into_iter()
                    .map(|column| self.ident_normalizer.normalize(column.name))
                    .collect()
            }
            _ => row_fields
                .iter()
                .map(|field| field.name().clone())
                .collect(),
        };

        let udf = self.context_provider.create_lateral_table_function_udf(
            LateralTableFunctionCall {
                name,
                args,
                outer_columns: outer_columns.clone(),
                row_fields: row_fields.clone(),
            },
        )?;

        let rows_name = unused_column_name(left.schema(), "__lateral_rows");
        let left_columns: Vec<Expr> = left
            .schema()
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect();
        let call_expr = Expr::ScalarFunction(ScalarFunction::new_udf(
            udf,
            outer_columns.into_iter().map(Expr::Column).collect(),
        ))
        .alias(&rows_name);
        let null_handling = match kind {
            LateralJoinKind::Inner => NullHandling::Drop,
            LateralJoinKind::Left => NullHandling::PreserveAndExpandEmpty,
        };
        let qualifier = TableReference::bare(alias_name);
        let output_columns = row_fields.iter().zip(output_names).map(|(field, out)| {
            Expr::Column(Column::from_name(format!("{rows_name}.{}", field.name())))
                .alias_qualified(Some(qualifier.clone()), out)
        });

        let mut plan = LogicalPlanBuilder::from(left)
            .project(left_columns.iter().cloned().chain([call_expr]))?
            .unnest_column_with_options(
                Column::from_name(&rows_name),
                UnnestOptions::new().with_null_handling(null_handling),
            )?
            .unnest_column(Column::from_name(&rows_name))?
            .project(left_columns.into_iter().chain(output_columns))?;

        if let Some(on) = on {
            self.warn_on_null_equality_predicate(&on);
            let predicate = self.sql_to_expr(on, plan.schema(), planner_context)?;
            match kind {
                LateralJoinKind::Inner => plan = plan.filter(predicate)?,
                LateralJoinKind::Left => {
                    if !matches!(
                        predicate,
                        Expr::Literal(ScalarValue::Boolean(Some(true)), _)
                    ) {
                        return not_impl_err!(
                            "LEFT JOIN LATERAL table function with column arguments supports only ON true, got ON {predicate}"
                        );
                    }
                }
            }
        }
        plan.build()
    }

    fn parse_cross_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
    ) -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(left).cross_join(right)?.build()
    }

    fn parse_join(
        &self,
        left: LogicalPlan,
        right: LogicalPlan,
        constraint: JoinConstraint,
        join_type: JoinType,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        match constraint {
            JoinConstraint::On(sql_expr) => {
                let join_schema = left.schema().join(right.schema())?;
                // parse ON expression
                self.warn_on_null_equality_predicate(&sql_expr);
                let expr = self.sql_to_expr(sql_expr, &join_schema, planner_context)?;
                LogicalPlanBuilder::from(left)
                    .join_on(right, join_type, Some(expr))?
                    .build()
            }
            JoinConstraint::Using(object_names) => {
                let keys = object_names
                    .into_iter()
                    .map(|object_name| {
                        let ObjectName(mut object_names) = object_name;
                        if object_names.len() != 1 {
                            not_impl_err!(
                                "Invalid identifier in USING clause. Expected single identifier, got {}", ObjectName(object_names)
                            )
                        } else {
                            let id = object_names.swap_remove(0);
                            id.as_ident()
                                .ok_or_else(|| {
                                    plan_datafusion_err!(
                                        "Expected identifier in USING clause"
                                    )
                                })
                                .map(|ident| Column::from_name(self.ident_normalizer.normalize(ident.clone())))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                LogicalPlanBuilder::from(left)
                    .join_using(right, join_type, keys)?
                    .build()
            }
            JoinConstraint::Natural => {
                let left_cols: HashSet<&String> =
                    left.schema().fields().iter().map(|f| f.name()).collect();
                let keys: Vec<Column> = right
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .filter(|f| left_cols.contains(f))
                    .map(Column::from_name)
                    .collect();
                if keys.is_empty() {
                    self.parse_cross_join(left, right)
                } else {
                    LogicalPlanBuilder::from(left)
                        .join_using(right, join_type, keys)?
                        .build()
                }
            }
            JoinConstraint::None => LogicalPlanBuilder::from(left)
                .join_on(right, join_type, [])?
                .build(),
        }
    }
}

/// How a lowered LATERAL table-function call keeps the rows of its left side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LateralJoinKind {
    /// A row is repeated once per row of its call; a call with no rows drops it.
    Inner,
    /// As `Inner`, but a call with no rows keeps the row once, with NULLs.
    Left,
}

/// Returns `true` if a join keeps (or only filters) the rows of its left side,
/// so it can be applied after a lowered LATERAL call instead of before it.
fn keeps_or_filters_left_rows(join_operator: &JoinOperator) -> bool {
    matches!(
        join_operator,
        JoinOperator::Join(_)
            | JoinOperator::Inner(_)
            | JoinOperator::Left(_)
            | JoinOperator::LeftOuter(_)
            | JoinOperator::CrossJoin(_)
            | JoinOperator::LeftSemi(_)
            | JoinOperator::LeftAnti(_)
            | JoinOperator::CrossApply
            | JoinOperator::OuterApply
    )
}

/// Returns `base`, or `base` with a numeric suffix, so that the name is not
/// used by any field of `schema`.
fn unused_column_name(schema: &DFSchema, base: &str) -> String {
    let is_used = |name: &str| schema.fields().iter().any(|f| f.name() == name);
    let mut name = base.to_string();
    let mut suffix = 0;
    while is_used(&name) {
        suffix += 1;
        name = format!("{base}_{suffix}");
    }
    name
}

/// Returns `true` if the given [`TableFactor`] is lateral.
pub(crate) fn is_lateral(factor: &TableFactor) -> bool {
    match factor {
        TableFactor::Derived { lateral, .. } => *lateral,
        TableFactor::Function { lateral, .. } => *lateral,
        TableFactor::UNNEST { .. } => true,
        _ => false,
    }
}

/// Returns `true` if the given [`Join`] is lateral.
pub(crate) fn is_lateral_join(join: &Join) -> Result<bool> {
    let is_lateral_syntax = is_lateral(&join.relation);
    let is_apply_syntax = match join.join_operator {
        JoinOperator::FullOuter(..)
        | JoinOperator::Right(..)
        | JoinOperator::RightOuter(..)
        | JoinOperator::RightAnti(..)
        | JoinOperator::RightSemi(..)
            if is_lateral_syntax =>
        {
            return not_impl_err!(
                "LATERAL syntax is not supported for \
                 FULL OUTER and RIGHT [OUTER | ANTI | SEMI] joins"
            );
        }
        JoinOperator::CrossApply | JoinOperator::OuterApply => true,
        _ => false,
    };
    Ok(is_lateral_syntax || is_apply_syntax)
}
