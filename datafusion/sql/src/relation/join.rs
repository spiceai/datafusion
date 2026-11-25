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
use arrow::datatypes::{Fields, Schema};
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{
    Column, DFSchema, Result, TableReference, not_impl_err, plan_datafusion_err, plan_err,
};
use datafusion_expr::table_function::TableFunctionApply;
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

        let args = args.into_iter().map(|arg| arg.transform_up(|expr| {
            if let Expr::OuterReferenceColumn(_, column) = expr {
                if !left.schema().has_column(&column) {
                    return not_impl_err!("Table function '{name}' references {column} outside its input");
                }
                Ok(Transformed::yes(Expr::Column(column)))
            } else {
                Ok(Transformed::no(expr))
            }
        }).map(|expr| expr.data)).collect::<Result<Vec<_>>>()?;
        let Some(function) =
            self.context_provider
                .bind_table_function(&name, &args, left.schema())?
        else {
            return not_impl_err!(
                "Table function '{name}' does not support arguments from an input relation"
            );
        };
        let source_schema = function.source.schema();
        let row_fields = source_schema.fields();
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

        let preserve_unmatched = matches!(kind, LateralJoinKind::Left);
        let fields: Fields = row_fields
            .iter()
            .zip(output_names)
            .map(|(field, name)| {
                field
                    .as_ref()
                    .clone()
                    .with_name(name)
                    .with_nullable(preserve_unmatched || field.is_nullable())
            })
            .collect();
        let right = DFSchema::try_from_qualified_schema(
            TableReference::bare(alias_name),
            &Schema::new(fields),
        )?;
        let schema = Arc::new(left.schema().join(&right)?);
        let predicate = on
            .map(|expr| self.sql_to_expr(expr, &schema, planner_context))
            .transpose()?;
        Ok(TableFunctionApply {
            function_name: name,
            input: Arc::new(left),
            function,
            preserve_unmatched,
            predicate,
            projection: (0..row_fields.len()).collect(),
            schema,
        }
        .into_plan())
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
