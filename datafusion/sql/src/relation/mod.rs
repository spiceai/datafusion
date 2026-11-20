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

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};

use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{
    DFSchema, DFSchemaRef, Diagnostic, Result, Span, Spans, TableReference, internal_err,
    not_impl_err, plan_err,
};
use datafusion_expr::builder::subquery_alias;
use datafusion_expr::expr::FieldMetadata;
use datafusion_expr::lateral_table_function::TABLE_FUNCTION_PARAMETER_NAME_KEY;
use datafusion_expr::planner::{
    PlannedRelation, RelationPlannerContext, RelationPlanning,
};
use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder, expr::Unnest};
use datafusion_expr::{Subquery, SubqueryAlias};
use sqlparser::ast::{
    Expr as SQLExpr, FunctionArg, FunctionArgExpr, Spanned, TableAlias, TableFactor,
};

mod join;

struct SqlToRelRelationContext<'a, 'b, S: ContextProvider> {
    planner: &'a SqlToRel<'b, S>,
    planner_context: &'a mut PlannerContext,
}

// Implement RelationPlannerContext
impl<'a, 'b, S: ContextProvider> RelationPlannerContext
    for SqlToRelRelationContext<'a, 'b, S>
{
    fn context_provider(&self) -> &dyn ContextProvider {
        self.planner.context_provider
    }

    fn plan(&mut self, relation: TableFactor) -> Result<LogicalPlan> {
        self.planner.create_relation(relation, self.planner_context)
    }

    fn sql_to_expr(
        &mut self,
        expr: sqlparser::ast::Expr,
        schema: &DFSchema,
    ) -> Result<Expr> {
        self.planner.sql_to_expr(expr, schema, self.planner_context)
    }

    fn sql_expr_to_logical_expr(
        &mut self,
        expr: sqlparser::ast::Expr,
        schema: &DFSchema,
    ) -> Result<Expr> {
        self.planner
            .sql_expr_to_logical_expr(expr, schema, self.planner_context)
    }

    fn normalize_ident(&self, ident: sqlparser::ast::Ident) -> String {
        self.planner.ident_normalizer.normalize(ident)
    }

    fn object_name_to_table_reference(
        &self,
        name: sqlparser::ast::ObjectName,
    ) -> Result<TableReference> {
        self.planner.object_name_to_table_reference(name)
    }
}

impl<S: ContextProvider> SqlToRel<'_, S> {
    /// Create a `LogicalPlan` that scans the named relation.
    ///
    /// First tries any registered extension planners. If no extension handles
    /// the relation, falls back to the default planner.
    fn create_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let planned_relation =
            match self.create_extension_relation(relation, planner_context)? {
                RelationPlanning::Planned(planned) => *planned,
                RelationPlanning::Original(original) => {
                    self.create_default_relation(*original, planner_context)?
                }
            };
        self.finish_relation(planned_relation)
    }

    /// Applies the alias of a planned relation and removes sorts that cannot
    /// affect its result.
    fn finish_relation(&self, planned_relation: PlannedRelation) -> Result<LogicalPlan> {
        let optimized_plan = optimize_subquery_sort(
            planned_relation.plan,
            self.context_provider
                .options()
                .sql_parser
                .enable_subquery_sort_elimination,
        )?
        .data;
        if let Some(alias) = planned_relation.alias {
            self.apply_table_alias(optimized_plan, alias)
        } else {
            Ok(optimized_plan)
        }
    }

    fn create_extension_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<RelationPlanning> {
        let planners = self.context_provider.get_relation_planners();
        if planners.is_empty() {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        }

        let mut current_relation = relation;
        for planner in planners.iter() {
            let mut context = SqlToRelRelationContext {
                planner: self,
                planner_context,
            };

            match planner.plan_relation(current_relation, &mut context)? {
                RelationPlanning::Planned(planned) => {
                    return Ok(RelationPlanning::Planned(planned));
                }
                RelationPlanning::Original(original) => {
                    current_relation = *original;
                }
            }
        }

        Ok(RelationPlanning::Original(Box::new(current_relation)))
    }

    fn create_default_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<PlannedRelation> {
        let relation_span = relation.span();
        if is_table_function_call(&relation) {
            let call = self.plan_table_function_call(relation, planner_context)?;
            let alias = call.alias.clone();
            let plan = self.table_function_call_to_plan(call)?;
            return Ok(PlannedRelation::new(plan, alias));
        }
        let (plan, alias) = match relation {
            TableFactor::Table { name, alias, .. } => {
                // Normalize name and alias
                let table_ref = self.object_name_to_table_reference(name)?;
                let table_name = table_ref.to_string();
                let cte = planner_context.get_cte(&table_name);
                (
                    match (
                        cte,
                        self.context_provider.get_table_source(table_ref.clone()),
                    ) {
                        (Some(cte_plan), _) => Ok(cte_plan.clone()),
                        (_, Ok(provider)) => {
                            LogicalPlanBuilder::scan(table_ref.clone(), provider, None)?
                                .build()
                        }
                        (None, Err(e)) => {
                            let e = e.with_diagnostic(Diagnostic::new_error(
                                format!("table '{table_ref}' not found"),
                                Span::try_from_sqlparser_span(relation_span),
                            ));
                            Err(e)
                        }
                    }?,
                    alias,
                )
            }
            TableFactor::Derived {
                subquery, alias, ..
            } => {
                let logical_plan = self.query_to_plan(*subquery, planner_context)?;
                (logical_plan, alias)
            }
            TableFactor::NestedJoin {
                table_with_joins,
                alias,
            } => (
                self.plan_table_with_joins(*table_with_joins, planner_context)?,
                alias,
            ),
            TableFactor::UNNEST {
                alias,
                array_exprs,
                with_offset: false,
                with_offset_alias: None,
                with_ordinality,
            } => {
                if with_ordinality {
                    return not_impl_err!("UNNEST with ordinality is not supported yet");
                }

                // Unnest table factor has empty input
                let schema = DFSchema::empty();
                let input = LogicalPlanBuilder::empty(true).build()?;
                // Unnest table factor can have multiple arguments.
                // We treat each argument as a separate unnest expression.
                let unnest_exprs = array_exprs
                    .into_iter()
                    .map(|sql_expr| {
                        let expr = self.sql_expr_to_logical_expr(
                            sql_expr,
                            &schema,
                            planner_context,
                        )?;
                        Self::check_unnest_arg(&expr, &schema)?;
                        Ok(Expr::Unnest(Unnest::new(expr)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if unnest_exprs.is_empty() {
                    return plan_err!("UNNEST must have at least one argument");
                }
                let logical_plan = self.try_process_unnest(input, unnest_exprs)?;
                (logical_plan, alias)
            }
            TableFactor::UNNEST { .. } => {
                return not_impl_err!(
                    "UNNEST table factor with offset is not supported yet"
                );
            }
            // @todo Support TableFactory::TableFunction?
            _ => {
                return not_impl_err!(
                    "Unsupported ast node {relation:?} in create_relation"
                );
            }
        };
        Ok(PlannedRelation::new(plan, alias))
    }

    pub(crate) fn create_relation_subquery(
        &self,
        subquery: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LogicalPlan> {
        let scope = enter_lateral_scope(planner_context);
        let plan = self.create_relation(subquery, planner_context)?;
        exit_lateral_scope(planner_context, scope);
        wrap_correlated_relation(plan)
    }

    /// Plans a LATERAL relation.
    ///
    /// A table-function call whose arguments reference columns of earlier
    /// FROM items is returned unplanned as
    /// [`LateralRelation::CorrelatedTableFunction`], for the join planner to
    /// lower. Everything else is planned as by [`Self::create_relation_subquery`].
    pub(crate) fn create_lateral_relation(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<LateralRelation> {
        if !is_table_function_call(&relation) {
            return self
                .create_relation_subquery(relation, planner_context)
                .map(LateralRelation::Plan);
        }

        let scope = enter_lateral_scope(planner_context);
        let planned_relation = match self
            .create_extension_relation(relation, planner_context)?
        {
            RelationPlanning::Planned(planned) => *planned,
            RelationPlanning::Original(original) => {
                let call = self.plan_table_function_call(*original, planner_context)?;
                if call.args.iter().any(Expr::contains_outer) {
                    exit_lateral_scope(planner_context, scope);
                    return Ok(LateralRelation::CorrelatedTableFunction(call));
                }
                let alias = call.alias.clone();
                PlannedRelation::new(self.table_function_call_to_plan(call)?, alias)
            }
        };
        let plan = self.finish_relation(planned_relation)?;
        exit_lateral_scope(planner_context, scope);
        wrap_correlated_relation(plan).map(LateralRelation::Plan)
    }

    /// Plans the arguments of a table-function call relation.
    ///
    /// Arguments are planned against an empty schema, so inside a LATERAL
    /// scope a column of an earlier FROM item becomes an
    /// [`Expr::OuterReferenceColumn`], and a bare name that matches no column
    /// (such as a table name argument) stays an [`Expr::Column`].
    fn plan_table_function_call(
        &self,
        relation: TableFactor,
        planner_context: &mut PlannerContext,
    ) -> Result<TableFunctionCall> {
        match relation {
            TableFactor::Table {
                name,
                alias,
                args: Some(func_args),
                ..
            } => {
                let tbl_func_name =
                    name.0.first().unwrap().as_ident().unwrap().to_string();
                let args =
                    self.plan_table_function_args(func_args.args, planner_context)?;
                let table_name_arg_str = args
                    .iter()
                    .map(|e| e.to_string())
                    .reduce(|a, b| format!("{a}, {b}"))
                    .unwrap_or_default();
                Ok(TableFunctionCall {
                    scan_name: TableReference::Bare {
                        table: format!("{tbl_func_name}({table_name_arg_str})").into(),
                    },
                    name: tbl_func_name,
                    args,
                    alias,
                })
            }
            TableFactor::Function {
                name, args, alias, ..
            } => {
                let tbl_func_ref = self.object_name_to_table_reference(name)?;
                let args = self.plan_table_function_args(args, planner_context)?;
                Ok(TableFunctionCall {
                    name: tbl_func_ref.table().to_string(),
                    scan_name: TableReference::from(tbl_func_ref.table()),
                    args,
                    alias,
                })
            }
            other => internal_err!("Expected a table function call, got {other}"),
        }
    }

    fn plan_table_function_args(
        &self,
        args: Vec<FunctionArg>,
        planner_context: &mut PlannerContext,
    ) -> Result<Vec<Expr>> {
        args.into_iter()
            .map(|arg| match arg {
                FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => self
                    .sql_expr_to_logical_expr(expr, &DFSchema::empty(), planner_context),
                FunctionArg::ExprNamed {
                    name: SQLExpr::Identifier(name),
                    arg: FunctionArgExpr::Expr(arg),
                    operator: _,
                }
                | FunctionArg::Named {
                    name,
                    arg: FunctionArgExpr::Expr(arg),
                    operator: _,
                } => self.plan_named_table_function_arg(name.value, arg, planner_context),
                _ => plan_err!("Unsupported function argument type: {}", arg),
            })
            .collect()
    }

    /// Plans a named argument (`name => value`). The name is kept in the
    /// field metadata under [`TABLE_FUNCTION_PARAMETER_NAME_KEY`].
    fn plan_named_table_function_arg(
        &self,
        param_name: String,
        arg: SQLExpr,
        planner_context: &mut PlannerContext,
    ) -> Result<Expr> {
        let parameter_metadata = FieldMetadata::new(BTreeMap::from([(
            TABLE_FUNCTION_PARAMETER_NAME_KEY.to_string(),
            param_name.clone(),
        )]));
        match self.sql_expr_to_logical_expr(arg, &DFSchema::empty(), planner_context)? {
            Expr::Literal(scalar, meta) => {
                let mut meta = meta.unwrap_or_else(FieldMetadata::default);
                meta.extend(parameter_metadata);
                Ok(Expr::Literal(scalar, Some(meta)))
            }
            // For non-literal named arguments (like function calls),
            // we wrap in an alias to preserve the parameter name.
            other => Ok(other.alias_with_metadata(param_name, Some(parameter_metadata))),
        }
    }

    /// Plans an uncorrelated table-function call as a scan of its provider.
    fn table_function_call_to_plan(
        &self,
        call: TableFunctionCall,
    ) -> Result<LogicalPlan> {
        let TableFunctionCall {
            name,
            scan_name,
            args,
            alias: _,
        } = call;
        let provider = self
            .context_provider
            .get_table_function_source(&name, args)?;
        LogicalPlanBuilder::scan(scan_name, provider, None)?.build()
    }
}

/// A table-function call in a FROM clause, with its arguments planned.
pub(crate) struct TableFunctionCall {
    /// The table function name.
    pub(crate) name: String,
    /// Table name of the scan that an uncorrelated call plans.
    scan_name: TableReference,
    pub(crate) args: Vec<Expr>,
    pub(crate) alias: Option<TableAlias>,
}

/// A planned LATERAL relation.
pub(crate) enum LateralRelation {
    /// A plan, as for any other relation.
    Plan(LogicalPlan),
    /// A table-function call whose arguments reference columns of earlier
    /// FROM items. The join planner lowers it to a projection and `UNNEST`.
    CorrelatedTableFunction(TableFunctionCall),
}

/// Returns `true` if `factor` calls a table function (`f(...)`,
/// `LATERAL f(...)`, `CROSS APPLY f(...)`).
pub(crate) fn is_table_function_call(factor: &TableFactor) -> bool {
    matches!(
        factor,
        TableFactor::Table { args: Some(_), .. } | TableFactor::Function { .. }
    )
}

/// Planner state replaced while a LATERAL relation is planned.
struct LateralScope {
    from_schema: DFSchemaRef,
    outer_query_schema: Option<DFSchemaRef>,
}

/// Makes the earlier FROM items visible to a LATERAL relation as the
/// innermost outer query schema, so that references to them plan as
/// [`Expr::OuterReferenceColumn`].
fn enter_lateral_scope(planner_context: &mut PlannerContext) -> LateralScope {
    // For a syntactically valid query the outer FROM schema is set: a lateral
    // relation is never the first factor in a FROM list, so it is reached
    // through the `for` loop in `plan_from_tables` or in `plan_joins`.
    let from_schema = planner_context
        .set_outer_from_schema(None)
        .unwrap_or_else(|| Arc::new(DFSchema::empty()));
    let outer_query_schema = planner_context.pop_outer_query_schema();
    let new_query_schema = match outer_query_schema {
        Some(ref old_query_schema) => {
            let mut new_query_schema = from_schema.as_ref().clone();
            new_query_schema.merge(old_query_schema.as_ref());
            Arc::new(new_query_schema)
        }
        None => Arc::clone(&from_schema),
    };
    planner_context.append_outer_query_schema(new_query_schema);
    LateralScope {
        from_schema,
        outer_query_schema,
    }
}

fn exit_lateral_scope(planner_context: &mut PlannerContext, scope: LateralScope) {
    planner_context.pop_outer_query_schema();
    if let Some(schema) = scope.outer_query_schema {
        planner_context.append_outer_query_schema(schema);
    }
    planner_context.set_outer_from_schema(Some(scope.from_schema));
}

/// Wraps a LATERAL relation that references columns of the outer scope in a
/// [`Subquery`], for the decorrelation rules.
fn wrap_correlated_relation(plan: LogicalPlan) -> Result<LogicalPlan> {
    let outer_ref_columns = plan.all_out_ref_exprs();

    // We can omit the subquery wrapper if there are no columns
    // referencing the outer scope.
    if outer_ref_columns.is_empty() {
        return Ok(plan);
    }

    match plan {
        LogicalPlan::SubqueryAlias(SubqueryAlias { input, alias, .. }) => subquery_alias(
            LogicalPlan::Subquery(Subquery {
                subquery: input,
                outer_ref_columns,
                spans: Spans::new(),
            }),
            alias,
        ),
        plan => Ok(LogicalPlan::Subquery(Subquery {
            subquery: Arc::new(plan),
            outer_ref_columns,
            spans: Spans::new(),
        })),
    }
}

fn optimize_subquery_sort(
    plan: LogicalPlan,
    enable_subquery_sort_elimination: bool,
) -> Result<Transformed<LogicalPlan>> {
    if !enable_subquery_sort_elimination {
        return Ok(Transformed::no(plan));
    }

    // When initializing subqueries, we examine sort options since they might be unnecessary.
    // They are only important if the subquery result is affected by the ORDER BY statement,
    // which can happen when we have:
    // 1. DISTINCT ON / ARRAY_AGG ... => Handled by an `Aggregate` and its requirements.
    // 2. RANK / ROW_NUMBER ... => Handled by a `WindowAggr` and its requirements.
    // 3. LIMIT => Handled by a `Sort`, so we need to search for it.
    let mut has_limit = false;

    plan.transform_down(|c| {
        if let LogicalPlan::Limit(_) = c {
            has_limit = true;
            return Ok(Transformed::no(c));
        }
        match c {
            LogicalPlan::Sort(s) => {
                if !has_limit {
                    has_limit = false;
                    return Ok(Transformed::yes(Arc::unwrap_or_clone(s.input)));
                }
                Ok(Transformed::no(LogicalPlan::Sort(s)))
            }
            _ => Ok(Transformed::no(c)),
        }
    })
}
