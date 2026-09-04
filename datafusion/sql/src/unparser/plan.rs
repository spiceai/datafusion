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

use super::{
    Unparser,
    ast::{
        BuilderError, DERIVED_DISTINCT_ALIAS, DERIVED_LIMIT_ALIAS,
        DERIVED_PROJECTION_ALIAS, DERIVED_SORT_ALIAS, DERIVED_TABLE_ALIASES,
        DERIVED_UNION_ALIAS, DERIVED_UNNEST_ALIAS, DERIVED_WINDOW_INPUT_ALIAS,
        DerivedRelationBuilder, QueryBuilder, RelationBuilder, SelectBuilder,
        TableRelationBuilder, TableWithJoinsBuilder, is_numbered_alias,
    },
    rewrite::{
        TableAliasRewriter, inject_column_aliases_into_subquery, normalize_union_schema,
        remove_dangling_identifiers, requalify_column_onto_derived_table,
        rewrite_plan_for_sort_on_non_projected_fields,
        subquery_alias_inner_query_and_columns,
    },
    utils::{
        expr_contains_subquery, find_agg_node_within_select,
        find_projection_node_within_select, find_unnest_node_within_select,
        find_window_nodes_within_select, name_derived_scope_outputs, name_scope_outputs,
        partition_subquery_filters, select_list_wraps_a_grouping_expr,
        try_transform_to_simple_table_scan_with_filters, unproject_sort_expr,
        unproject_unnamed_projection_exprs, unproject_unnest_expr,
        unproject_unnest_expr_as_flatten_value, unproject_window_exprs,
    },
};
use crate::unparser::extension_unparser::{
    UnparseToStatementResult, UnparseWithinStatementResult,
};
use crate::unparser::utils::{find_unnest_node_until_relation, unproject_agg_exprs};
use crate::unparser::{
    ast::FlattenRelationBuilder, ast::UnnestRelationBuilder, rewrite::rewrite_qualify,
};
use crate::utils::UNNEST_PLACEHOLDER;
use arrow::datatypes::SchemaRef;
use datafusion_common::{
    Column, DFSchema, DataFusionError, Result, ScalarValue, TableReference,
    assert_or_internal_err, internal_datafusion_err, internal_err, not_impl_err,
    tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion},
};
use datafusion_expr::expr::{Cast, OUTER_REFERENCE_COLUMN_PREFIX, UNNEST_COLUMN_PREFIX};
use datafusion_expr::{
    Aggregate, BinaryExpr, Distinct, Expr, Join, JoinConstraint, JoinType, LogicalPlan,
    LogicalPlanBuilder, Operator, Projection, SortExpr, Subquery, TableScan, Unnest,
    UserDefinedLogicalNode, Window, expr::Alias, utils::split_conjunction,
};
use sqlparser::ast::{self, Ident, OrderByKind, SetExpr, TableAliasColumnDef};
use sqlparser::ast::helpers::attached_token::AttachedToken;
use std::{collections::HashSet, sync::Arc, vec};

/// Convert a DataFusion [`LogicalPlan`] to [`ast::Statement`]
///
/// This function is the opposite of [`SqlToRel::sql_statement_to_plan`] and can
/// be used to, among other things, to convert `LogicalPlan`s to SQL strings.
///
/// # Errors
///
/// This function returns an error if the plan cannot be converted to SQL.
///
/// # See Also
///
/// * [`expr_to_sql`] for converting [`Expr`], a single expression to SQL
///
/// # Example
/// ```
/// use arrow::datatypes::{DataType, Field, Schema};
/// use datafusion_expr::{col, logical_plan::table_scan};
/// use datafusion_sql::unparser::plan_to_sql;
/// let schema = Schema::new(vec![
///     Field::new("id", DataType::Utf8, false),
///     Field::new("value", DataType::Utf8, false),
/// ]);
/// // Scan 'table' and select columns 'id' and 'value'
/// let plan = table_scan(Some("table"), &schema, None)
///     .unwrap()
///     .project(vec![col("id"), col("value")])
///     .unwrap()
///     .build()
///     .unwrap();
/// // convert to AST
/// let sql = plan_to_sql(&plan).unwrap();
/// // use the Display impl to convert to SQL text
/// assert_eq!(
///     sql.to_string(),
///     "SELECT \"table\".id, \"table\".\"value\" FROM \"table\""
/// )
/// ```
///
/// [`SqlToRel::sql_statement_to_plan`]: crate::planner::SqlToRel::sql_statement_to_plan
/// [`expr_to_sql`]: crate::unparser::expr_to_sql
pub fn plan_to_sql(plan: &LogicalPlan) -> Result<ast::Statement> {
    let unparser = Unparser::default();
    unparser.plan_to_sql(plan)
}

/// `have` extended with the fields of `add` it does not already carry.
///
/// `None` when the two disagree about a field's type under one name, which makes
/// a reference to it ambiguous.
fn merged_without_conflict(have: &DFSchema, add: &DFSchema) -> Option<DFSchema> {
    let mut merged = have.clone();
    for (qualifier, field) in add.iter() {
        match have.field_with_name(qualifier, field.name()) {
            Ok(existing) if existing.data_type() == field.data_type() => {}
            Ok(_) => return None,
            Err(_) => {
                let addition = DFSchema::new_with_metadata(
                    vec![(qualifier.cloned(), Arc::clone(field))],
                    std::collections::HashMap::new(),
                )
                .ok()?;
                merged = merged.join(&addition).ok()?;
            }
        }
    }
    Some(merged)
}

/// The fields a node's own expressions resolve against: everything its inputs
/// expose, which for a join is both sides.
///
/// `None` when the inputs' fields will not combine, such as two branches that both
/// expose the same unqualified name. A leaf carries its expressions itself — a
/// scan's pushed-down filters — so its own schema is the right one.
fn expression_schema(plan: &LogicalPlan) -> Option<DFSchema> {
    let inputs = plan.inputs();
    if inputs.is_empty() {
        // A scan's pushed-down filters may name columns its projection leaves
        // out, so the source's own schema is the scope, not the projected one.
        if let LogicalPlan::TableScan(scan) = plan {
            return DFSchema::try_from_qualified_schema(
                scan.table_name.clone(),
                &scan.source.schema(),
            )
            .ok();
        }
        return Some(plan.schema().as_ref().clone());
    }

    let mut schema = DFSchema::empty();
    for input in &inputs {
        schema = schema.join(input.schema()).ok()?;
    }

    // An expression unparsed here may have been unprojected from a node below —
    // a select item is substituted back into the aggregate or window expression
    // that produced it — and then it references that node's input rather than
    // this one's. Those nodes flatten into the same `SELECT`, so their inputs are
    // in scope for the expressions that end up in it.
    // Start one level down: this node's inputs are already joined above, and
    // re-joining them would collide with themselves.
    let mut below: &LogicalPlan = *inputs.first()?;
    loop {
        if !matches!(
            below,
            LogicalPlan::Projection(_)
                | LogicalPlan::Aggregate(_)
                | LogicalPlan::Window(_)
                | LogicalPlan::Filter(_)
                | LogicalPlan::Sort(_)
                | LogicalPlan::Limit(_)
                | LogicalPlan::Distinct(_)
        ) {
            break;
        }
        let below_inputs = below.inputs();
        let Some(input) = below_inputs.first() else {
            break;
        };
        // A name repeats across levels for the ordinary reason — a grouping key
        // is in both an aggregate's output and its input — and that is the same
        // column, so it can be skipped. A repeat carrying a *different* type is
        // a real ambiguity, and guessing which one a reference means is worse
        // than declining, so the walk stops there.
        match merged_without_conflict(&schema, input.schema()) {
            Some(merged) => schema = merged,
            None => break,
        }
        below = input;
    }

    Some(schema)
}

impl Unparser<'_> {
    pub fn plan_to_sql(&self, plan: &LogicalPlan) -> Result<ast::Statement> {
        let mut plan = normalize_union_schema(plan)?;
        if !self.dialect.supports_qualify() {
            plan = rewrite_qualify(plan)?;
        }

        match plan {
            LogicalPlan::Projection(_)
            | LogicalPlan::Filter(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Aggregate(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Join(_)
            | LogicalPlan::Repartition(_)
            | LogicalPlan::Union(_)
            | LogicalPlan::TableScan(_)
            | LogicalPlan::EmptyRelation(_)
            | LogicalPlan::Subquery(_)
            | LogicalPlan::SubqueryAlias(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Statement(_)
            | LogicalPlan::Values(_)
            | LogicalPlan::Distinct(_) => self.select_to_sql_statement(&plan),
            LogicalPlan::Dml(_) => self.dml_to_sql(&plan),
            LogicalPlan::Extension(extension) => {
                self.extension_to_statement(extension.node.as_ref())
            }
            LogicalPlan::RecursiveQuery(recursive)
                if self.dialect.supports_recursive_cte() =>
            {
                self.recursive_query_to_sql_statement(&recursive)
            }
            LogicalPlan::Explain(_)
            | LogicalPlan::Analyze(_)
            | LogicalPlan::Ddl(_)
            | LogicalPlan::Copy(_)
            | LogicalPlan::DescribeTable(_)
            | LogicalPlan::RecursiveQuery(_)
            | LogicalPlan::Unnest(_) => not_impl_err!("Unsupported plan: {plan:?}"),
        }
    }

    /// Try to unparse a [UserDefinedLogicalNode] to a SQL statement.
    /// If multiple unparsers are registered for the same [UserDefinedLogicalNode],
    /// the first unparsing result will be returned.
    fn extension_to_statement(
        &self,
        node: &dyn UserDefinedLogicalNode,
    ) -> Result<ast::Statement> {
        let mut statement = None;
        for unparser in &self.extension_unparsers {
            match unparser.unparse_to_statement(node, self)? {
                UnparseToStatementResult::Modified(stmt) => {
                    statement = Some(stmt);
                    break;
                }
                UnparseToStatementResult::Unmodified => {}
            }
        }
        if let Some(statement) = statement {
            Ok(statement)
        } else {
            not_impl_err!("Unsupported extension node: {node:?}")
        }
    }

    /// Try to unparse a [UserDefinedLogicalNode] to a SQL statement.
    /// If multiple unparsers are registered for the same [UserDefinedLogicalNode],
    /// the first unparser supporting the node will be used.
    fn extension_to_sql(
        &self,
        node: &dyn UserDefinedLogicalNode,
        query: &mut Option<&mut QueryBuilder>,
        select: &mut Option<&mut SelectBuilder>,
        relation: &mut Option<&mut RelationBuilder>,
    ) -> Result<()> {
        for unparser in &self.extension_unparsers {
            match unparser.unparse(node, self, query, select, relation)? {
                UnparseWithinStatementResult::Modified => return Ok(()),
                UnparseWithinStatementResult::Unmodified => {}
            }
        }
        not_impl_err!("Unsupported extension node: {node:?}")
    }

    fn select_to_sql_statement(&self, plan: &LogicalPlan) -> Result<ast::Statement> {
        let mut query_builder = Some(QueryBuilder::default());

        let body = self.select_to_sql_expr(plan, &mut query_builder)?;

        let query = query_builder.unwrap().body(Box::new(body)).build()?;

        Ok(ast::Statement::Query(Box::new(query)))
    }

    /// Builds the `<name> AS (<static> UNION [ALL] <recursive>)` a recursive CTE
    /// contributes to an enclosing query's `WITH`.
    ///
    /// The recursive term already refers to the working table as a scan of
    /// `name`, so it renders as that name without help — the self-reference the
    /// CTE needs is the ordinary table reference the plan carries.
    ///
    /// `is_distinct` picks the set quantifier, and it is the whole difference
    /// between `UNION` and `UNION ALL` here: DataFusion dedupes the working
    /// table for the distinct form, which is what `UNION` asks the remote to do.
    fn recursive_cte(
        &self,
        recursive: &datafusion_expr::RecursiveQuery,
    ) -> Result<(ast::Cte, Ident)> {
        let mut static_query = Some(QueryBuilder::default());
        let static_term =
            self.select_to_sql_expr(&recursive.static_term, &mut static_query)?;
        let mut recursive_query = Some(QueryBuilder::default());
        let recursive_term =
            self.select_to_sql_expr(&recursive.recursive_term, &mut recursive_query)?;

        let set_quantifier = if recursive.is_distinct {
            self.dialect.union_distinct_set_quantifier()
        } else {
            ast::SetQuantifier::All
        };
        let body = SetExpr::SetOperation {
            op: ast::SetOperator::Union,
            set_quantifier,
            left: Box::new(static_term),
            right: Box::new(recursive_term),
        };

        let name = self.new_ident_quoted_if_needs(recursive.name.clone());
        let cte = ast::Cte {
            alias: ast::TableAlias {
                name: name.clone(),
                columns: vec![],
                at: None,
                explicit: false,
            },
            query: Box::new(
                QueryBuilder::default()
                    .body(Box::new(body))
                    .build()
                    .map_err(|e| internal_datafusion_err!("{e}"))?,
            ),
            from: None,
            materialized: None,
            closing_paren_token: AttachedToken::empty(),
        };
        Ok((cte, name))
    }

    /// A recursive CTE as the whole statement: `WITH RECURSIVE … SELECT * FROM
    /// <name>`.
    fn recursive_query_to_sql_statement(
        &self,
        recursive: &datafusion_expr::RecursiveQuery,
    ) -> Result<ast::Statement> {
        let (cte, name) = self.recursive_cte(recursive)?;
        let select_all = ast::Select {
            select_token: AttachedToken::empty(),
            distinct: None,
            top: None,
            top_before_distinct: false,
            projection: vec![ast::SelectItem::Wildcard(
                ast::WildcardAdditionalOptions::default(),
            )],
            exclude: None,
            into: None,
            from: vec![ast::TableWithJoins {
                relation: ast::TableFactor::Table {
                    name: ast::ObjectName::from(vec![name]),
                    alias: None,
                    args: None,
                    with_hints: vec![],
                    version: None,
                    with_ordinality: false,
                    partitions: vec![],
                    json_path: None,
                    sample: None,
                    index_hints: vec![],
                },
                joins: vec![],
            }],
            lateral_views: vec![],
            prewhere: None,
            selection: None,
            group_by: ast::GroupByExpr::Expressions(vec![], vec![]),
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            having: None,
            named_window: vec![],
            qualify: None,
            window_before_qualify: false,
            value_table_mode: None,
            connect_by: vec![],
            flavor: ast::SelectFlavor::Standard,
            optimizer_hints: vec![],
            select_modifiers: None,
        };

        let mut query = QueryBuilder::default();
        query
            .push_cte(cte, true)
            .body(Box::new(SetExpr::Select(Box::new(select_all))));

        Ok(ast::Statement::Query(Box::new(
            query.build().map_err(|e| internal_datafusion_err!("{e}"))?,
        )))
    }

    fn select_to_sql_expr(
        &self,
        plan: &LogicalPlan,
        query: &mut Option<QueryBuilder>,
    ) -> Result<SetExpr> {
        let mut select_builder = SelectBuilder::default();
        select_builder.push_from(TableWithJoinsBuilder::default());
        let mut relation_builder = RelationBuilder::default();
        self.select_to_sql_recursively(
            plan,
            query,
            &mut select_builder,
            &mut relation_builder,
        )?;

        // If we were able to construct a full body (i.e. UNION ALL), return it
        if let Some(body) = query.as_mut().and_then(|q| q.take_body()) {
            return Ok(*body);
        }

        // If no projection is set, add a wildcard projection to the select
        // which will be translated to `SELECT *` in the SQL statement
        if !select_builder.already_projected() {
            select_builder.projection(vec![ast::SelectItem::Wildcard(
                ast::WildcardAdditionalOptions::default(),
            )]);
        }

        // Construct a list of all the identifiers present in query sources
        let mut all_idents = Vec::new();
        if let Some(source_alias) = relation_builder.get_alias() {
            all_idents.push(source_alias);
        } else if let Some(source_name) = relation_builder.get_name() {
            all_idents.push(source_name);
        }

        let mut twj = select_builder.pop_from().unwrap();
        twj.get_joins()
            .iter()
            .for_each(|join| match &join.relation {
                ast::TableFactor::Table { alias, name, .. } => {
                    if let Some(alias) = alias {
                        all_idents.push(alias.name.to_string());
                    } else {
                        all_idents.push(name.to_string());
                    }
                }
                ast::TableFactor::Derived { alias, .. } => {
                    if let Some(alias) = alias {
                        all_idents.push(alias.name.to_string());
                    }
                }
                _ => {}
            });

        twj.relation(relation_builder);
        select_builder.push_from(twj);

        // Ensure that the projection contains references to sources that actually exist
        let mut projection = select_builder.get_projection();
        projection.iter_mut().for_each(|select_item| {
            if let ast::SelectItem::UnnamedExpr(ast::Expr::CompoundIdentifier(idents)) =
                select_item
            {
                remove_dangling_identifiers(idents, &all_idents);
            }
        });

        // Check the order by as well
        if let Some(query) = query.as_mut()
            && let Some(OrderByKind::Expressions(mut order_by)) = query.get_order_by()
        {
            order_by.iter_mut().for_each(|sort_item| {
                if let ast::Expr::CompoundIdentifier(idents) = &mut sort_item.expr {
                    remove_dangling_identifiers(idents, &all_idents);
                }
            });

            query.order_by(OrderByKind::Expressions(order_by));
        }

        // Order by could be a sort in the select builder
        let mut sort = select_builder.get_sort_by();
        sort.iter_mut().for_each(|sort_item| {
            if let ast::Expr::CompoundIdentifier(idents) = &mut sort_item.expr {
                remove_dangling_identifiers(idents, &all_idents);
            }
        });

        select_builder.projection(projection);

        Ok(SetExpr::Select(Box::new(select_builder.build()?)))
    }

    /// Reconstructs a SELECT SQL statement from a logical plan by unprojecting column expressions
    /// found in a [Projection] node. This requires scanning the plan tree for relevant Aggregate
    /// and Window nodes and matching column expressions to the appropriate agg or window expressions.
    fn reconstruct_select_statement(
        &self,
        plan: &LogicalPlan,
        p: &Projection,
        select: &mut SelectBuilder,
    ) -> Result<()> {
        let mut exprs = p.expr.clone();

        // A projection with no output expressions (e.g. `count(*)` over a view
        // that prunes every column, producing `Projection: <empty> -> TableScan`)
        // must not be unparsed as an empty `SELECT` list: dialects such as DuckDB
        // reject `SELECT FROM t` with a parser error. Mirror the bare `TableScan`
        // handling and fall back to a dummy `SELECT 1` for dialects that do not
        // support an empty select list.
        if exprs.is_empty() {
            let items = self
                .empty_projection_fallback()
                .iter()
                .map(|e| self.select_item_to_sql(e))
                .collect::<Result<Vec<_>>>()?;
            select.projection(items);
            return Ok(());
        }

        // If an Unnest node is found within the select, find and unproject the unnest column
        let flatten_alias = select.current_flatten_alias();
        if let Some(unnest) = find_unnest_node_within_select(plan) {
            if let Some(ref alias) = flatten_alias {
                exprs = exprs
                    .into_iter()
                    .map(|e| unproject_unnest_expr_as_flatten_value(e, unnest, alias))
                    .collect::<Result<Vec<_>>>()?;
            } else {
                exprs = exprs
                    .into_iter()
                    .map(|e| unproject_unnest_expr(e, unnest))
                    .collect::<Result<Vec<_>>>()?;
            }
        };

        // Rewrite column references that point to FLATTEN table aliases:
        // in Snowflake, FLATTEN output is accessed via .VALUE, not the
        // original column name.
        if !select.flatten_table_aliases_empty() {
            exprs = exprs
                .into_iter()
                .map(|e| {
                    e.transform(|expr| {
                        if let Expr::Column(ref col) = expr
                            && let Some(ref relation) = col.relation
                            && select.is_flatten_table_alias(relation.table())
                        {
                            return Ok(Transformed::yes(Expr::Column(Column::new(
                                Some(relation.clone()),
                                "VALUE",
                            ))));
                        }
                        Ok(Transformed::no(expr))
                    })
                    .map(|t| t.data)
                })
                .collect::<Result<Vec<_>>>()?;
        }

        match (
            find_agg_node_within_select(plan, true),
            find_window_nodes_within_select(plan, None, true),
        ) {
            (Some(agg), window) => {
                let window_option = window.as_deref();
                let items = exprs
                    .into_iter()
                    .map(|proj_expr| {
                        let unproj = unproject_agg_exprs(proj_expr, agg, window_option)?;
                        self.select_item_to_sql(&unproj)
                    })
                    .collect::<Result<Vec<_>>>()?;

                select.projection(items);
                select.group_by(ast::GroupByExpr::Expressions(
                    self.group_by_keys(agg)?,
                    vec![],
                ));
            }
            (None, Some(window)) => {
                let items = exprs
                    .into_iter()
                    .map(|proj_expr| {
                        let unproj = unproject_window_exprs(proj_expr, &window)?;
                        self.select_item_to_sql(&unproj)
                    })
                    .collect::<Result<Vec<_>>>()?;

                select.projection(items);
            }
            _ => {
                let items = exprs
                    .iter()
                    .map(|e| {
                        // After unproject_unnest_expr_as_flatten_value, an
                        // internal UNNEST display-name alias may still wrap
                        // the rewritten _unnest.VALUE column. Replace it
                        // with the bare FLATTEN VALUE select item.
                        if let Some(ref alias) = flatten_alias
                            && Self::has_internal_unnest_alias(e)
                        {
                            return Ok(self.build_flatten_value_select_item(alias, None));
                        }
                        self.select_item_to_sql(e)
                    })
                    .collect::<Result<Vec<_>>>()?;
                select.projection(items);
            }
        }
        Ok(())
    }

    /// Unparses a `Projection` over an `Aggregate` as two scopes: this `SELECT`'s
    /// list reads the aggregate's outputs, and the aggregate becomes a derived
    /// table.
    ///
    /// The rendering a dialect needs when it will not resolve `GROUP BY` against a
    /// select item's sub-expressions. It is also the only rendering that keeps the
    /// plan's grouping for every projection: the cheaper `GROUP BY <output alias>`
    /// and `GROUP BY <ordinal>` group by the value the projection computes, so a
    /// projection that is not injective over the grouping expression returns one
    /// row where the plan has several, with their aggregates summed.
    fn projection_over_scoped_aggregate(
        &self,
        p: &Projection,
        select: &mut SelectBuilder,
        relation: &mut RelationBuilder,
    ) -> Result<()> {
        // The projection already reads the aggregate's output columns, so its
        // expressions go out as they stand: no unprojection, which is what would
        // substitute the grouping expression back into the select item, and no
        // GROUP BY on this SELECT, which now groups nothing.
        let items = p
            .expr
            .iter()
            .map(|expr| self.select_item_to_sql(expr))
            .collect::<Result<Vec<_>>>()?;
        select.projection(items);

        // The derived table always carries an alias, even for a dialect that would
        // not require one, so this SELECT has a name to address its columns
        // through, and it carries a projection that names its outputs so there is
        // something to address.
        //
        // The alias is numbered per SELECT for the reason the stacked-aggregate
        // scope numbers its own: a join walks both of its sides with one builder,
        // and each side requalifies its own references onto its own alias.
        let named_input = name_scope_outputs(p.input.as_ref())?;
        let alias_name = select.next_derived_aggregate_alias();
        let alias = self.new_ident_quoted_if_needs(alias_name.clone());
        self.derive(
            &named_input,
            relation,
            Some(self.new_table_alias(alias_name, vec![])),
            false,
        )?;

        // This SELECT now reads those columns from the derived table, so a
        // reference still qualified by a relation the derived table encloses binds
        // to nothing. DataFusion re-plans such SQL, but a stricter remote binder
        // rejects it, which is what breaks a federated pushdown.
        let derived_qualifiers: HashSet<String> = p
            .input
            .schema()
            .iter()
            .filter_map(|(qualifier, _)| qualifier)
            .flat_map(|qualifier| [qualifier.to_string(), qualifier.table().to_string()])
            .collect();
        select.visit_expressions_in_clauses_mut(|expr| {
            if let ast::Expr::CompoundIdentifier(idents) = expr {
                requalify_column_onto_derived_table(idents, &derived_qualifiers, &alias);
            }
        });

        Ok(())
    }

    /// Whether `plan`, a `Projection`, is going to move the aggregate below it
    /// into a scope of its own rather than folding it into this SELECT.
    ///
    /// Asked in two places, which have to agree: the projection acts on it, and a
    /// sort above the projection reads it to decide what its key may name. A sort
    /// key unprojected into a grouping expression names the relation that
    /// expression reads, and once the aggregate is a scope of its own that
    /// relation is enclosed by it and binds to nothing outside.
    fn projection_scopes_its_aggregate(
        &self,
        plan: &LogicalPlan,
        select: &SelectBuilder,
    ) -> bool {
        if self.dialect.group_by_matches_select_subexpressions()
            || select.has_grouped_predicate()
        {
            return false;
        }
        let LogicalPlan::Projection(projection) = plan else {
            return false;
        };
        find_agg_node_within_select(plan, true)
            .is_some_and(|agg| select_list_wraps_a_grouping_expr(&projection.expr, agg))
    }

    fn derive(
        &self,
        plan: &LogicalPlan,
        relation: &mut RelationBuilder,
        alias: Option<ast::TableAlias>,
        lateral: bool,
    ) -> Result<()> {
        // A derived table is referred to by the names its schema reports, so any
        // output the projection below leaves for the engine to name has to be
        // aliased before the enclosing scope can bind it. A table alias that
        // carries a column list already names every output, so it needs nothing.
        let columns_named_by_alias = relation.has_columns_named_by_alias()
            || alias
                .as_ref()
                .is_some_and(|alias| !alias.columns.is_empty());
        let named = if columns_named_by_alias {
            None
        } else {
            name_derived_scope_outputs(plan)?
        };
        let plan = named.as_ref().unwrap_or(plan);

        let mut derived_builder = DerivedRelationBuilder::default();
        derived_builder.lateral(lateral).alias(alias).subquery({
            let inner_statement = self.plan_to_sql(plan)?;
            if let ast::Statement::Query(inner_query) = inner_statement {
                inner_query
            } else {
                return internal_err!(
                    "Subquery must be a Query, but found {inner_statement:?}"
                );
            }
        });
        relation.derived(derived_builder);

        Ok(())
    }

    fn derive_with_dialect_alias(
        &self,
        alias: &str,
        plan: &LogicalPlan,
        relation: &mut RelationBuilder,
        lateral: bool,
        columns: Vec<Ident>,
    ) -> Result<()> {
        if self.dialect.requires_derived_table_alias() || !columns.is_empty() {
            self.derive(
                plan,
                relation,
                Some(self.new_table_alias(alias.to_string(), columns)),
                lateral,
            )
        } else {
            self.derive(plan, relation, None, lateral)
        }
    }

    /// Whether the subtree carrying a row limit needs a `SELECT` of its own.
    ///
    /// The walk is top-down, so any predicate already on `select` came from a
    /// node visited earlier. `WHERE`, `HAVING` and `QUALIFY` are all evaluated
    /// before `LIMIT`/`OFFSET`, while a plan that puts a filter above a limit
    /// says the opposite: the limit runs first and the predicate filters what
    /// it produced. Keeping both in one `SELECT` therefore states the reverse
    /// of the plan, and can return rows the plan excludes.
    ///
    /// A `WHERE` can stay in the enclosing query while the limited subtree
    /// moves into a derived table. `HAVING` and `QUALIFY` cannot: they name an
    /// aggregate or window expression that only the `SELECT` computing it can
    /// name. Refuse those rather than emit the reversed form — but only when
    /// the grouping they filter is in fact below this limit. Join inputs are
    /// walked with one shared `SelectBuilder`, so a predicate on it may have
    /// come from a sibling input rather than from an ancestor of this node.
    fn row_limit_needs_own_scope(
        plan: &LogicalPlan,
        select: &SelectBuilder,
    ) -> Result<bool> {
        if select.has_grouped_predicate()
            && (find_agg_node_within_select(plan, select.already_projected()).is_some()
                || find_window_nodes_within_select(
                    plan,
                    None,
                    select.already_projected(),
                )
                .is_some())
        {
            return not_impl_err!(
                "Unparsing a HAVING or QUALIFY predicate that is applied after a row limit is not supported"
            );
        }
        Ok(select.has_selection())
    }

    /// Unparses a `Limit` as a derived table, so the `WHERE` already on the
    /// enclosing `SELECT` applies to the limited rows rather than to the rows
    /// feeding the limit.
    ///
    /// The derived table takes the name of the relation it reads, because the
    /// predicate staying outside is still qualified by that name, and its
    /// columns are listed explicitly: a wildcard would expand to every
    /// relation in the enclosing `FROM`, not to this one's contribution.
    ///
    /// Both of those need the derived table's output columns to be exactly the
    /// relation's own columns, under their own names — which is why only a
    /// scan, and the clauses that can wrap one without renaming anything, are
    /// accepted here. A projection may emit a column the derived query never
    /// names (an unaliased expression) or two columns that differ only by a
    /// qualifier SQL cannot carry across the boundary; a join, union or
    /// aggregate has no single name for the alias to take. Those are refused,
    /// which costs the pushdown but never the rows.
    fn derive_row_limited_scope(
        &self,
        plan: &LogicalPlan,
        select: &mut SelectBuilder,
        relation: &mut RelationBuilder,
    ) -> Result<()> {
        let Some(table_ref) = Self::scanned_relation_of(plan) else {
            return not_impl_err!(
                "Unparsing a filter applied after a row limit is only supported when the limited input is a single table scan"
            );
        };

        // Only the last component survives as an alias, so a predicate spelled
        // with the full path would be left pointing at a name that is gone.
        if self.dialect.full_qualified_col() && table_ref.to_vec().len() > 1 {
            return not_impl_err!(
                "Unparsing a filter applied after a row limit is not supported for a qualified table name on a dialect that spells columns in full"
            );
        }

        // A scan can project no columns at all, which every other empty
        // projection in this unparser renders as `SELECT 1`. There is no
        // column list to name a derived table's output with here, so refuse.
        // (Two columns of one name cannot arrive: `DFSchema` rejects a scan
        // with a duplicate qualified field.)
        let fields = plan.schema().fields();
        if fields.is_empty() {
            return not_impl_err!(
                "Unparsing a filter applied after a row limit is not supported for an input projecting no columns"
            );
        }

        // The subtree moves into a statement of its own, so this `SELECT` no
        // longer receives a projection from the nodes below it.
        if !select.already_projected() {
            let items = fields
                .iter()
                .map(|field| {
                    self.select_item_to_sql(&Expr::Column(Column::new(
                        Some(table_ref.clone()),
                        field.name(),
                    )))
                })
                .collect::<Result<Vec<_>>>()?;
            select.projection(items);
        }

        self.derive(
            plan,
            relation,
            Some(self.new_table_alias(table_ref.table().to_string(), vec![])),
            false,
        )
    }

    /// The relation a subtree scans, when the subtree is one scan under
    /// clauses that neither rename nor add columns and neither reorder nor
    /// combine rows from elsewhere.
    ///
    /// A `Sort` is deliberately not walked through. It is the one such clause
    /// whose effect does not survive being wrapped: SQL does not carry a
    /// derived table's row order into the query selecting from it.
    fn scanned_relation_of(plan: &LogicalPlan) -> Option<TableReference> {
        match plan {
            LogicalPlan::TableScan(scan) => Some(scan.table_name.clone()),
            LogicalPlan::SubqueryAlias(alias) => {
                Self::scanned_relation_of(alias.input.as_ref())
                    .map(|_| alias.alias.clone())
            }
            LogicalPlan::Limit(limit) => Self::scanned_relation_of(limit.input.as_ref()),
            LogicalPlan::Filter(filter) => {
                Self::scanned_relation_of(filter.input.as_ref())
            }
            _ => None,
        }
    }

    /// Isolates what a join input's `TableScan` did in a derived table:
    /// `(SELECT ... FROM t WHERE ... LIMIT ...) AS t`.
    ///
    /// Two things a scan can carry have no faithful home in the enclosing
    /// query:
    ///
    /// * A `FULL JOIN` input's filters. A `FULL JOIN` preserves both sides, so
    ///   a filter on just one input cannot be expressed by folding it into `ON`
    ///   (filtered-out rows would reappear as unmatched) or by moving it to
    ///   `WHERE` (the other side's unmatched rows would be discarded).
    /// * A `fetch`, for any join type. The join's own `LIMIT` bounds its
    ///   output, not one input's contribution to it.
    ///
    /// Pre-filtering that side in its own subquery is the only clause that
    /// preserves the original meaning.
    ///
    /// The filters go on the side of the limit they came from. The scan's own
    /// filters run before its `fetch`; a `Filter` node above the scan runs
    /// after it, and needs a second scope of its own — the same `SELECT`
    /// cannot express it, because `WHERE` is evaluated before `LIMIT`.
    ///
    /// `clean_plan` is the `TableScan`/`SubqueryAlias` produced by
    /// `try_transform_to_simple_table_scan_with_filters`; `filters` is
    /// everything it extracted and `scan_filters` the subset belonging to the
    /// scan. `relation` already holds the plain table reference built for this
    /// side; it is overwritten with the derived subquery.
    /// Splits an outer join's `ON` filter into what stays and what belongs in the
    /// non-preserved input's own scope.
    ///
    /// A subquery inside a join predicate is refused outright by some dialects. On
    /// an inner join the conjunct carrying one moves to `WHERE`, which selects the
    /// same rows; on an outer join it cannot, because `WHERE` discards the rows the
    /// join preserves. Applied to the non-preserved input before the join it does
    /// select the same rows, so that input's own scope is where it goes.
    ///
    /// Only a conjunct whose every reference — its own columns and any outer
    /// reference its subquery emits past itself, at any depth — comes from that
    /// input is moved. Anything else would leave the scope that answers it
    /// behind: a non-lateral derived table has no view of the join's other side.
    ///
    /// Returns `(the filter that stays, the conjuncts the input's scope takes)`.
    fn scope_subquery_onto_non_preserved_input(join: &Join) -> (Option<Expr>, Vec<Expr>) {
        let Some(filter) = &join.filter else {
            return (None, vec![]);
        };
        let non_preserved = match join.join_type {
            JoinType::Left => &join.right,
            JoinType::Right => &join.left,
            _ => return (join.filter.clone(), vec![]),
        };
        let reachable: HashSet<Column> =
            non_preserved.schema().columns().into_iter().collect();
        let reads_only_that_input = |expr: &Expr| {
            expr.column_refs()
                .into_iter()
                .all(|column| reachable.contains(column))
                && expr
                    .apply(|sub| {
                        let Some(subquery) = Self::subquery_of(sub) else {
                            return Ok(TreeNodeRecursion::Continue);
                        };
                        let mut reaching = vec![];
                        Self::outward_references(subquery, &mut reaching)?;
                        Ok(
                            if reaching.iter().all(|column| reachable.contains(column)) {
                                TreeNodeRecursion::Continue
                            } else {
                                TreeNodeRecursion::Stop
                            },
                        )
                    })
                    // An error here is a list this cannot read, and keeping the
                    // conjunct in `ON` is the safe answer to that.
                    .is_ok_and(|recursion| recursion != TreeNodeRecursion::Stop)
        };
        let (scoped, kept): (Vec<Expr>, Vec<Expr>) = split_conjunction(filter)
            .into_iter()
            .cloned()
            .partition(|conjunct| {
                expr_contains_subquery(conjunct) && reads_only_that_input(conjunct)
            });
        (kept.into_iter().reduce(Expr::and), scoped)
    }

    /// The columns the outer references `subquery` emits past itself name, at
    /// any depth.
    ///
    /// Its own [`Subquery::outer_ref_columns`] are the references written into
    /// its body. A nested subquery's are relative to the body enclosing *it*:
    /// one naming a column the node holding that subquery can see binds there,
    /// as the planner bound it, and goes no further; the rest pass out through
    /// this level as well. `outer_ref_columns` leaves those out, for the reason
    /// [`Self::nested_subqueries_reach_captured_scope`] gives, so reaching them
    /// needs this descent.
    ///
    /// `outer_ref_columns` holds `Expr::OuterReferenceColumn`s, which
    /// [`Expr::column_refs`] passes over, so the column is read out of each one
    /// directly. Anything else on the list is an error rather than a reference
    /// silently taken as reaching nowhere.
    ///
    /// Recursion depth is the plan's subquery nesting depth, which nothing
    /// bounds, so it grows the stack the way
    /// [`Self::subquery_reaches_captured_scope`] does.
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn outward_references(subquery: &Subquery, reaching: &mut Vec<Column>) -> Result<()> {
        for outer in &subquery.outer_ref_columns {
            let Expr::OuterReferenceColumn(_, column) = outer else {
                return internal_err!(
                    "outer_ref_columns holds a {} rather than an outer reference",
                    outer.variant_name()
                );
            };
            reaching.push(column.clone());
        }
        subquery.subquery.apply(|node| {
            // What this node's expressions can see: the columns its inputs
            // present, or its own where it has none.
            let inputs = node.inputs();
            let visible: HashSet<Column> = if inputs.is_empty() {
                node.schema().columns().into_iter().collect()
            } else {
                inputs
                    .into_iter()
                    .flat_map(|input| input.schema().columns())
                    .collect()
            };
            node.apply_expressions(|expr| {
                expr.apply(|sub| {
                    let Some(nested) = Self::subquery_of(sub) else {
                        return Ok(TreeNodeRecursion::Continue);
                    };
                    let mut nested_reaching = vec![];
                    Self::outward_references(nested, &mut nested_reaching)?;
                    reaching.extend(
                        nested_reaching
                            .into_iter()
                            .filter(|column| !visible.contains(column)),
                    );
                    Ok(TreeNodeRecursion::Continue)
                })
            })
        })?;
        Ok(())
    }

    fn derive_join_side(
        &self,
        clean_plan: &LogicalPlan,
        filters: Vec<Expr>,
        scan_filters: &[Expr],
        fetch: Option<usize>,
        relation: &mut RelationBuilder,
    ) -> Result<()> {
        if filters.is_empty() && fetch.is_none() {
            return Ok(());
        }
        let table_ref = match clean_plan {
            LogicalPlan::TableScan(scan) => Some(scan.table_name.clone()),
            LogicalPlan::SubqueryAlias(alias) => Some(alias.alias.clone()),
            _ => None,
        };

        // A derived table's alias is a single identifier, so only the last
        // component of a qualified name survives it. Where the dialect spells
        // columns in full, the enclosing query's `ON` and `WHERE` still name
        // every component, leaving them qualified by a relation that is no
        // longer in scope. Refuse rather than emit that, which costs the
        // pushdown but never the rows — the same trade
        // `derive_row_limited_scope` makes for the limit it scopes.
        if self.dialect.full_qualified_col()
            && table_ref
                .as_ref()
                .is_some_and(|table_ref| table_ref.to_vec().len() > 1)
        {
            return not_impl_err!(
                "Unparsing a join input's fetch or FULL JOIN filters is not supported for a qualified table name on a dialect that spells columns in full"
            );
        }

        let (below_fetch, above_fetch): (Vec<Expr>, Vec<Expr>) = if fetch.is_some() {
            filters
                .into_iter()
                .partition(|filter| scan_filters.contains(filter))
        } else {
            // Without a limit between them the two sets commute, so keep them
            // in one `WHERE` in the order they were collected.
            (filters, vec![])
        };

        let mut builder = LogicalPlanBuilder::from(clean_plan.clone());
        if let Some(combined) = below_fetch.into_iter().reduce(Expr::and) {
            builder = builder.filter(combined)?;
        }
        if let Some(fetch) = fetch {
            builder = builder.limit(0, Some(fetch))?;
        }
        if let Some(combined) = above_fetch.into_iter().reduce(Expr::and) {
            // The alias both scopes the limit in its own `SELECT` and keeps the
            // predicate's column references resolvable against it.
            if let Some(table_ref) = table_ref.clone() {
                builder = builder.alias(table_ref)?;
            }
            builder = builder.filter(combined)?;
        }
        let derived_plan = builder.build()?;
        let alias = table_ref
            .map(|table_ref| self.new_table_alias(table_ref.table().to_string(), vec![]));
        self.derive(&derived_plan, relation, alias, false)
    }

    /// Projection unparsing when [`super::dialect::Dialect::unnest_as_lateral_flatten`] is enabled:
    /// Snowflake-style `LATERAL FLATTEN` for unnest (not other dialect spellings).
    ///
    /// [`Self::peel_to_unnest_with_modifiers`] walks through any intermediate
    /// Limit/Sort nodes (the optimizer can insert these between the Projection
    /// and the Unnest), applies their modifiers to the query, and returns the
    /// Unnest plus the [`LogicalPlan`] ref to recurse into. This bypasses the
    /// normal Limit/Sort handlers which would wrap the subtree in a derived
    /// subquery.
    ///
    /// SELECT rendering is delegated to [`Self::reconstruct_select_statement`],
    /// which rewrites placeholder columns to `alias."VALUE"` via
    /// [`unproject_unnest_expr_as_flatten_value`].
    ///
    /// Returns `Ok(true)` when this path fully handled the projection.
    fn try_projection_unnest_as_lateral_flatten(
        &self,
        plan: &LogicalPlan,
        p: &Projection,
        query: &mut Option<QueryBuilder>,
        select: &mut SelectBuilder,
        relation: &mut RelationBuilder,
        unnest_input_type: Option<&UnnestInputType>,
    ) -> Result<bool> {
        // unnest_as_lateral_flatten: Snowflake LATERAL FLATTEN
        //
        // Generate the alias up front so that peel_to_unnest_with_modifiers
        // can rewrite ORDER BY placeholder columns to alias.VALUE.
        if self.dialect.unnest_as_lateral_flatten() && unnest_input_type.is_some() {
            let flatten_alias_name = if !select.already_projected() {
                select.next_flatten_alias()
            } else {
                select
                    .current_flatten_alias()
                    .unwrap_or_else(|| select.next_flatten_alias())
            };

            if let Some((unnest, unnest_plan)) = self.peel_to_unnest_with_modifiers(
                p.input.as_ref(),
                query,
                Some(&flatten_alias_name),
            )? && let Some(mut flatten) =
                self.try_unnest_to_lateral_flatten_sql(unnest)?
            {
                let inner_projection = Self::peel_to_inner_projection(
                    unnest.input.as_ref(),
                )
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "Unnest input is not a Projection: {:?}",
                        unnest.input
                    )
                })?;

                flatten.alias(Some(ast::TableAlias {
                    name: Ident::with_quote('"', &flatten_alias_name),
                    columns: vec![],
                    explicit: true,
                    at: None,
                }));

                if !select.already_projected() {
                    self.reconstruct_select_statement(plan, p, select)?;
                }

                if matches!(
                    inner_projection.input.as_ref(),
                    LogicalPlan::EmptyRelation(_)
                ) {
                    relation.flatten(flatten);
                    self.select_to_sql_recursively(unnest_plan, query, select, relation)?;
                    return Ok(true);
                }

                self.select_to_sql_recursively(unnest_plan, query, select, relation)?;

                let flatten_factor = flatten.build().map_err(|e| {
                    internal_datafusion_err!("Failed to build FLATTEN: {e}")
                })?;
                let cross_join = ast::Join {
                    relation: flatten_factor,
                    global: false,
                    join_operator: ast::JoinOperator::CrossJoin(
                        ast::JoinConstraint::None,
                    ),
                };
                if let Some(mut from) = select.pop_from() {
                    from.push_join(cross_join);
                    select.push_from(from);
                } else {
                    let mut twj = TableWithJoinsBuilder::default();
                    twj.push_join(cross_join);
                    select.push_from(twj);
                }

                return Ok(true);
            }
        }

        Ok(false)
    }

    fn project_window_output(
        &self,
        window_expr: &[Expr],
        select: &mut SelectBuilder,
        agg: Option<&Aggregate>,
    ) -> Result<()> {
        let mut items = if select.already_projected() {
            select.pop_projections()
        } else {
            vec![ast::SelectItem::Wildcard(
                ast::WildcardAdditionalOptions::default(),
            )]
        };

        items.extend(
            window_expr
                .iter()
                .map(|expr| {
                    let expr = if let Some(agg) = agg {
                        unproject_agg_exprs(expr.clone(), agg, None)?
                    } else {
                        expr.clone()
                    };
                    self.select_item_to_sql(&expr)
                })
                .collect::<Result<Vec<_>>>()?,
        );
        select.projection(items);

        Ok(())
    }

    fn window_input_requires_derived_subquery(plan: &LogicalPlan) -> bool {
        // These operators either produce a SELECT list or apply SQL clauses
        // that are evaluated after window functions in a single SELECT block.
        // Keep them below the Window node by emitting a derived table.
        matches!(
            plan,
            LogicalPlan::Projection(_)
                | LogicalPlan::Distinct(_)
                | LogicalPlan::Limit(_)
                | LogicalPlan::Sort(_)
                | LogicalPlan::Union(_)
        )
    }

    fn window_to_sql_with_derived_input(
        &self,
        window: &Window,
        select: &mut SelectBuilder,
        relation: &mut RelationBuilder,
    ) -> Result<()> {
        let input_alias = DERIVED_WINDOW_INPUT_ALIAS;
        self.derive(
            window.input.as_ref(),
            relation,
            Some(self.new_table_alias(input_alias.to_string(), vec![])),
            false,
        )?;

        let input_schema = window.input.schema();
        let mut alias_rewriter = TableAliasRewriter {
            table_schema: input_schema.as_arrow(),
            alias_name: TableReference::bare(input_alias),
        };
        let window_expr = window
            .window_expr
            .iter()
            .map(|expr| expr.clone().rewrite(&mut alias_rewriter).data())
            .collect::<Result<Vec<_>>>()?;

        self.project_window_output(&window_expr, select, None)
    }

    /// The `GROUP BY` keys to emit, with a constant key cast to its own type.
    ///
    /// A bare literal is not a portable grouping key. BigQuery refuses one
    /// outright ("Cannot GROUP BY literal values"), and an engine that reads a
    /// bare integer there as a select-list ordinal groups by something else
    /// entirely. Casting the literal to the type it already has is accepted
    /// everywhere and cannot be read as an ordinal.
    ///
    /// The key is kept rather than dropped, because dropping the last one turns
    /// a grouped aggregate into a global one: over an empty input the first
    /// yields no rows and the second yields a row of zeros.
    fn group_by_keys(&self, aggregate: &Aggregate) -> Result<Vec<ast::Expr>> {
        aggregate
            .group_expr
            .iter()
            .map(|expr| match expr {
                Expr::Literal(value, _) => self.expr_to_sql(&Expr::Cast(Cast::new(
                    Box::new(expr.clone()),
                    value.data_type(),
                ))),
                _ => self.expr_to_sql(expr),
            })
            .collect()
    }

    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn select_to_sql_recursively(
        &self,
        plan: &LogicalPlan,
        query: &mut Option<QueryBuilder>,
        select: &mut SelectBuilder,
        relation: &mut RelationBuilder,
    ) -> Result<()> {
        // Bind this node's schema once, here, so everything the node's handling
        // reaches — helpers included — resolves expression types against it. The
        // recursion re-enters through this wrapper, so each node rebinds to its
        // own inputs rather than inheriting an ancestor's.
        //
        // A schema whose fields will not combine is not bound, and a rendering
        // that needs a type then applies only where an expression states its own.
        match expression_schema(plan) {
            Some(schema) => self
                .with_schema(Arc::new(schema))
                .select_to_sql_recursively_inner(plan, query, select, relation),
            None => self.select_to_sql_recursively_inner(plan, query, select, relation),
        }
    }

    fn select_to_sql_recursively_inner(
        &self,
        plan: &LogicalPlan,
        query: &mut Option<QueryBuilder>,
        select: &mut SelectBuilder,
        relation: &mut RelationBuilder,
    ) -> Result<()> {
        match plan {
            LogicalPlan::TableScan(scan) => {
                if let Some(unparsed_table_scan) = self.unparse_table_scan_pushdown(
                    plan,
                    None,
                    select.already_projected(),
                )? {
                    return self.select_to_sql_recursively(
                        &unparsed_table_scan,
                        query,
                        select,
                        relation,
                    );
                }
                let mut builder = TableRelationBuilder::default();
                let mut table_parts = vec![];
                if let Some(catalog_name) = scan.table_name.catalog() {
                    table_parts
                        .push(self.new_ident_quoted_if_needs(catalog_name.to_string()));
                }
                if let Some(schema_name) = scan.table_name.schema() {
                    table_parts
                        .push(self.new_ident_quoted_if_needs(schema_name.to_string()));
                }
                table_parts.push(
                    self.new_ident_quoted_if_needs(scan.table_name.table().to_string()),
                );
                builder.name(ast::ObjectName::from(table_parts));
                relation.table(builder);

                Ok(())
            }
            LogicalPlan::Projection(p) => {
                if let Some(new_plan) = rewrite_plan_for_sort_on_non_projected_fields(p) {
                    return self
                        .select_to_sql_recursively(&new_plan, query, select, relation);
                }

                // Projection can be top-level plan for unnest relation.
                // The projection generated by the `RecursiveUnnestRewriter`
                // will have at least one expression referencing an unnest
                // placeholder column.
                let unnest_input_type: Option<UnnestInputType> =
                    p.expr.iter().find_map(Self::find_unnest_placeholder);

                // --- UNNEST table factor path (BigQuery, etc.) ---
                // Only fires for a single bare-placeholder projection.
                // Uses peel_to_unnest_with_modifiers (rather than matching
                // p.input directly) to handle Limit/Sort between Projection
                // and Unnest.
                if self.dialect.unnest_as_table_factor()
                    && p.expr.len() == 1
                    && Self::is_bare_unnest_placeholder(&p.expr[0])
                    && let Some((unnest, unnest_plan)) =
                        self.peel_to_unnest_with_modifiers(p.input.as_ref(), query, None)?
                    && let Some(unnest_relation) =
                        self.try_unnest_to_table_factor_sql(unnest)?
                {
                    relation.unnest(unnest_relation);
                    return self.select_to_sql_recursively(
                        unnest_plan,
                        query,
                        select,
                        relation,
                    );
                }

                if self.try_projection_unnest_as_lateral_flatten(
                    plan,
                    p,
                    query,
                    select,
                    relation,
                    unnest_input_type.as_ref(),
                )? {
                    return Ok(());
                }

                // If it's a unnest projection, we should provide the table column alias
                // to provide a column name for the unnest relation.
                let columns = if unnest_input_type.is_some() {
                    p.expr
                        .iter()
                        .map(|e| {
                            self.new_ident_quoted_if_needs(e.schema_name().to_string())
                        })
                        .collect()
                } else {
                    vec![]
                };
                // Projection can be top-level plan for derived table
                if select.already_projected() {
                    return self.derive_with_dialect_alias(
                        DERIVED_PROJECTION_ALIAS,
                        plan,
                        relation,
                        unnest_input_type
                            .filter(|t| matches!(t, UnnestInputType::OuterReference))
                            .is_some(),
                        columns,
                    );
                }
                // For Snowflake FLATTEN: when the outer Projection has
                // UNNEST(...) display-name columns (from SELECT * / SELECT
                // UNNEST(...)), generate a flatten alias now so that
                // reconstruct_select_statement and the downstream Unnest
                // handler both use the same alias.
                if self.dialect.unnest_as_lateral_flatten()
                    && p.expr.iter().any(Self::has_internal_unnest_alias)
                {
                    select.next_flatten_alias();
                }
                // Pre-register FLATTEN table aliases from SubqueryAlias
                // nodes in the plan tree so that
                // reconstruct_select_statement can rewrite column
                // references (e.g. a.col → a.VALUE) before the
                // SubqueryAlias handler runs.
                if self.dialect.unnest_as_lateral_flatten() {
                    Self::collect_flatten_aliases(p.input.as_ref(), select);
                }
                // A dialect that resolves GROUP BY against whole select items and
                // column references only cannot be handed a select list that wraps a
                // computed grouping expression: it reports the columns inside the
                // wrapper as neither grouped nor aggregated and refuses the
                // statement. The aggregate needs a scope of its own — and only a
                // scope, since grouping by the output alias or the ordinal groups by
                // the *wrapped* value and merges groups a non-injective wrapper
                // collides.
                //
                // `find_agg_node_within_select` is asked the same question
                // `reconstruct_select_statement` asks below, so the aggregate scoped
                // here is exactly the one that would otherwise fold into this SELECT.
                // A `HAVING`/`QUALIFY` already on this SELECT was classified against
                // the aggregate below, and names an expression only the SELECT that
                // computes it can name. Moving the aggregate into a scope would
                // strand it on a SELECT that no longer aggregates.
                if self.projection_scopes_its_aggregate(plan, select) {
                    return self.projection_over_scoped_aggregate(p, select, relation);
                }
                self.reconstruct_select_statement(plan, p, select)?;
                self.select_to_sql_recursively(p.input.as_ref(), query, select, relation)
            }
            LogicalPlan::Filter(filter) => {
                let window = find_window_nodes_within_select(
                    plan,
                    None,
                    select.already_projected(),
                );
                let agg = find_agg_node_within_select(plan, select.already_projected());

                if let (Some(window), true) =
                    (window.as_deref(), self.dialect.supports_qualify())
                {
                    let mut unprojected =
                        unproject_window_exprs(filter.predicate.clone(), window)?;
                    if let Some(agg) = agg {
                        unprojected = unproject_agg_exprs(unprojected, agg, None)?;
                    }
                    let filter_expr = self.expr_to_sql(&unprojected)?;
                    select.qualify(Some(filter_expr));
                } else if let Some(agg) = agg {
                    let unprojected =
                        unproject_agg_exprs(filter.predicate.clone(), agg, None)?;
                    let filter_expr = self.expr_to_sql(&unprojected)?;
                    select.having(Some(filter_expr));
                } else {
                    // A predicate can reference a projection output that the
                    // projection does not name, whose logical name is not an
                    // identifier the emitted statement carries.
                    let predicate = match find_projection_node_within_select(
                        plan,
                        select.already_projected(),
                    ) {
                        Some(projection) => unproject_unnamed_projection_exprs(
                            filter.predicate.clone(),
                            projection,
                        )?,
                        None => filter.predicate.clone(),
                    };
                    let filter_expr = self.expr_to_sql(&predicate)?;
                    select.selection(Some(filter_expr));
                }

                self.select_to_sql_recursively(
                    filter.input.as_ref(),
                    query,
                    select,
                    relation,
                )
            }
            LogicalPlan::Limit(limit) => {
                // Limit can be top-level plan for derived table
                if select.already_projected() {
                    return self.derive_with_dialect_alias(
                        DERIVED_LIMIT_ALIAS,
                        plan,
                        relation,
                        false,
                        vec![],
                    );
                }
                if (limit.fetch.is_some() || limit.skip.is_some())
                    && Self::row_limit_needs_own_scope(plan, select)?
                {
                    return self.derive_row_limited_scope(plan, select, relation);
                }
                if let Some(fetch) = &limit.fetch {
                    let Some(query) = query.as_mut() else {
                        return internal_err!(
                            "Limit operator only valid in a statement context."
                        );
                    };
                    query.limit(Some(self.expr_to_sql(fetch)?));
                }

                if let Some(skip) = &limit.skip {
                    let Some(query) = query.as_mut() else {
                        return internal_err!(
                            "Offset operator only valid in a statement context."
                        );
                    };

                    query.offset(Some(ast::Offset {
                        rows: ast::OffsetRows::None,
                        value: self.expr_to_sql(skip)?,
                    }));
                }

                self.select_to_sql_recursively(
                    limit.input.as_ref(),
                    query,
                    select,
                    relation,
                )
            }
            LogicalPlan::Sort(sort) => {
                // Sort can be top-level plan for derived table
                if select.already_projected() {
                    return self.derive_with_dialect_alias(
                        DERIVED_SORT_ALIAS,
                        plan,
                        relation,
                        false,
                        vec![],
                    );
                }
                // A `Sort` carrying a `fetch` renders that fetch as this
                // query's `LIMIT`, so it reorders against a predicate above it
                // exactly as a `Limit` node does — but it cannot be moved into
                // a derived table the way a `Limit` can, because SQL does not
                // carry a derived table's row order out to the query selecting
                // from it. (A sort without a fetch reorders nothing: `ORDER BY`
                // is evaluated after `WHERE` either way.)
                if sort.fetch.is_some() && Self::row_limit_needs_own_scope(plan, select)?
                {
                    return not_impl_err!(
                        "Unparsing a filter applied after a sort's fetch is not supported"
                    );
                }

                let Some(query_ref) = query else {
                    return internal_err!(
                        "Sort operator only valid in a statement context."
                    );
                };

                if let Some(fetch) = sort.fetch {
                    query_ref.limit(Some(ast::Expr::value(ast::Value::Number(
                        fetch.to_string(),
                        false,
                    ))));
                };

                // The projection below may be about to move the aggregate into a
                // scope of its own. The aggregate is then not this SELECT's, and the
                // sort key has to stay the reference the plan holds so it names that
                // scope's output rather than the grouping expression inside it.
                let agg = if self
                    .projection_scopes_its_aggregate(sort.input.as_ref(), select)
                {
                    None
                } else {
                    find_agg_node_within_select(plan, select.already_projected())
                };
                let window_nodes = find_window_nodes_within_select(
                    plan,
                    None,
                    select.already_projected(),
                );
                let windows: Option<Vec<&Window>> = window_nodes
                    .as_deref()
                    .map(|ws| ws.iter().copied().collect());
                // unproject sort expressions
                let sort_exprs: Vec<SortExpr> = sort
                    .expr
                    .iter()
                    .map(|sort_expr| {
                        unproject_sort_expr(
                            sort_expr.clone(),
                            agg,
                            windows.as_deref(),
                            sort.input.as_ref(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;

                query_ref.order_by(self.sorts_to_sql(&sort_exprs)?);

                self.select_to_sql_recursively(
                    sort.input.as_ref(),
                    query,
                    select,
                    relation,
                )
            }
            LogicalPlan::Aggregate(agg) => {
                // A SELECT expresses a single grouping, so an aggregate stacked below the
                // one this SELECT already carries has to become a derived table. Stacked
                // aggregates are what `single_distinct_to_groupby` produces for
                // `count(DISTINCT c)`: an outer `count(alias1)` over an inner
                // `GROUP BY c AS alias1`. Folding both into one SELECT would emit
                // `count(alias1)` against the base table — `alias1` does not exist there,
                // and where it happens to, the DISTINCT is silently gone.
                if select.already_aggregated() {
                    // The derived table always carries an alias, even for a dialect that
                    // would not require one, so this SELECT has a name to address its
                    // columns through. A bare column name would do only where the derived
                    // table is the SELECT's sole relation, which is not true under a join.
                    //
                    // The alias is numbered per SELECT because a join walks both of
                    // its sides with this one builder, so both sides can derive an
                    // aggregate into the same FROM clause. A fixed name would repeat
                    // there, and since each side requalifies its own references onto
                    // its own alias, the two sides' distinct columns would collapse
                    // onto a single qualifier.
                    let alias_name = select.next_derived_aggregate_alias();
                    let alias = self.new_ident_quoted_if_needs(alias_name.clone());
                    self.derive(
                        plan,
                        relation,
                        Some(self.new_table_alias(alias_name, vec![])),
                        false,
                    )?;

                    // This SELECT now reads those columns from the derived table, so a
                    // reference still qualified by a relation the derived table encloses
                    // binds to nothing. DataFusion re-plans such SQL, but a stricter remote
                    // binder rejects it, which is what breaks a federated pushdown.
                    let derived_qualifiers: HashSet<String> = plan
                        .schema()
                        .iter()
                        .filter_map(|(qualifier, _)| qualifier)
                        .flat_map(|qualifier| {
                            [qualifier.to_string(), qualifier.table().to_string()]
                        })
                        .collect();
                    select.visit_expressions_in_clauses_mut(|expr| {
                        if let ast::Expr::CompoundIdentifier(idents) = expr {
                            requalify_column_onto_derived_table(
                                idents,
                                &derived_qualifiers,
                                &alias,
                            );
                        }
                    });

                    return Ok(());
                }
                select.mark_aggregated();

                // Aggregation can be already handled in the projection case
                if !select.already_projected() {
                    // The query returns aggregate and group expressions. If that weren't the case,
                    // the aggregate would have been placed inside a projection, making the check above^ false
                    let exprs: Vec<_> = agg
                        .aggr_expr
                        .iter()
                        .chain(agg.group_expr.iter())
                        .map(|expr| self.select_item_to_sql(expr))
                        .collect::<Result<Vec<_>>>()?;
                    select.projection(exprs);

                    select.group_by(ast::GroupByExpr::Expressions(
                        self.group_by_keys(agg)?,
                        vec![],
                    ));
                }

                self.select_to_sql_recursively(
                    agg.input.as_ref(),
                    query,
                    select,
                    relation,
                )
            }
            LogicalPlan::Distinct(distinct) => {
                // Distinct can be top-level plan for derived table
                if select.already_projected() {
                    return self.derive_with_dialect_alias(
                        DERIVED_DISTINCT_ALIAS,
                        plan,
                        relation,
                        false,
                        vec![],
                    );
                }

                // If this distinct is the parent of a Union and we're in a query context,
                // then we need to unparse as a `UNION` rather than a `UNION ALL`.
                if let Distinct::All(input) = distinct
                    && matches!(input.as_ref(), LogicalPlan::Union(_))
                    && let Some(query_mut) = query.as_mut()
                {
                    query_mut.distinct_union();
                    return self.select_to_sql_recursively(
                        input.as_ref(),
                        query,
                        select,
                        relation,
                    );
                }

                let (select_distinct, input) = match distinct {
                    Distinct::All(input) => (ast::Distinct::Distinct, input.as_ref()),
                    Distinct::On(on) => {
                        let exprs = on
                            .on_expr
                            .iter()
                            .map(|e| self.expr_to_sql(e))
                            .collect::<Result<Vec<_>>>()?;
                        let items = on
                            .select_expr
                            .iter()
                            .map(|e| self.select_item_to_sql(e))
                            .collect::<Result<Vec<_>>>()?;
                        if let Some(sort_expr) = &on.sort_expr {
                            if let Some(query_ref) = query {
                                query_ref.order_by(self.sorts_to_sql(sort_expr)?);
                            } else {
                                return internal_err!(
                                    "Sort operator only valid in a statement context."
                                );
                            }
                        }
                        select.projection(items);
                        (ast::Distinct::On(exprs), on.input.as_ref())
                    }
                };
                select.distinct(Some(select_distinct));
                self.select_to_sql_recursively(input, query, select, relation)
            }
            LogicalPlan::Join(join) => {
                // Kept apart by input: where a filter may be re-emitted depends
                // on which side of the join it came from (see
                // `split_join_on_and_where_filters`).
                let mut left_scan_filters = vec![];
                let mut right_scan_filters = vec![];
                let (left_plan, right_plan) = if Self::swaps_join_inputs(join.join_type) {
                    (&join.right, &join.left)
                } else {
                    (&join.left, &join.right)
                };
                // A subquery in `ON` is refused by some dialects, and an outer join
                // cannot move it to `WHERE`. The non-preserved input's own scope is
                // the remaining clause that selects the same rows.
                let (scoped_join_filter, scoped_for_input) =
                    Self::scope_subquery_onto_non_preserved_input(join);
                let (mut left_scoped, mut right_scoped) = match join.join_type {
                    JoinType::Right => (scoped_for_input, vec![]),
                    _ => (vec![], scoped_for_input),
                };
                // If there's an outer projection plan, it will already set up the projection.
                // In that case, we don't need to worry about setting up the projection here.
                // The outer projection plan will handle projecting the correct columns.
                let already_projected = select.already_projected();

                let mut left_scan_fetch = None;
                let mut left_scan_only_filters = vec![];
                let left_plan =
                    match try_transform_to_simple_table_scan_with_filters(left_plan)? {
                        Some(scan) => {
                            left_scan_filters.extend(scan.filters);
                            left_scan_only_filters = scan.scan_filters;
                            left_scan_fetch = scan.fetch;
                            Arc::new(scan.plan)
                        }
                        None => Arc::clone(left_plan),
                    };

                if join.join_type == JoinType::Right {
                    let (kept, scoped) = partition_subquery_filters(std::mem::take(
                        &mut left_scan_filters,
                    ));
                    left_scan_filters = kept;
                    left_scoped.extend(scoped);
                }

                // A join that null-extends its left input must not let a
                // predicate from that subtree reach the SELECT-global `WHERE`:
                // `WHERE` is evaluated after this join and would discard the
                // very rows this join preserves. Set the accumulated predicate
                // aside, see what the left subtree adds, and fold that into
                // this join's `ON` instead (where, for the non-preserved side,
                // it means the same thing).
                // This relocation is only valid when the left input is not
                // preserved. A FULL JOIN also null-extends its left input, but
                // preserves left rows, so moving the predicate into ON would
                // make filtered-out left rows reappear as unmatched rows.
                let left_is_null_extended = matches!(join.join_type, JoinType::Right);
                let outer_selection = if left_is_null_extended {
                    select.take_selection()
                } else {
                    None
                };

                self.select_to_sql_recursively(
                    left_plan.as_ref(),
                    query,
                    select,
                    relation,
                )?;

                // A FULL JOIN preserves both sides, so neither `ON` nor
                // `WHERE` can express a filter that came from just one
                // side's `TableScan` (see `split_join_on_and_where_filters`),
                // and no clause of the enclosing query can express one input's
                // `fetch`. Isolate that side in a derived table instead, and
                // drop the filters from `left_scan_filters` so they are not
                // also routed to `ON`/`WHERE` below.
                if left_scan_fetch.is_some()
                    || (join.join_type == JoinType::Full && !left_scan_filters.is_empty())
                    || !left_scoped.is_empty()
                {
                    let mut side_filters = std::mem::take(&mut left_scoped);
                    if left_scan_fetch.is_some() || join.join_type == JoinType::Full {
                        side_filters.append(&mut left_scan_filters);
                    }
                    self.derive_join_side(
                        left_plan.as_ref(),
                        side_filters,
                        &left_scan_only_filters,
                        left_scan_fetch,
                        relation,
                    )?;
                }

                let hoisted_from_left = if left_is_null_extended {
                    let contributed = select.take_selection();
                    select.selection(outer_selection);
                    contributed
                } else {
                    None
                };

                let left_projection: Option<Vec<ast::SelectItem>> = if !already_projected
                {
                    Some(select.pop_projections())
                } else {
                    None
                };

                let is_exists_join = matches!(
                    join.join_type,
                    JoinType::LeftSemi
                        | JoinType::LeftAnti
                        | JoinType::LeftMark
                        | JoinType::RightSemi
                        | JoinType::RightAnti
                        | JoinType::RightMark
                );

                // The build (right) side of an EXISTS-style join is emitted as
                // a self-contained subquery (see `build_exists_subquery`), so
                // it must not be unparsed into the shared `select` here. Doing
                // so would leak its projection, joins, DISTINCT and WHERE
                // clauses into the outer query. Regular joins unparse it into
                // the shared `select` as usual.
                let mut right_relation = RelationBuilder::default();
                let right_plan: Arc<LogicalPlan> = if is_exists_join {
                    Arc::clone(right_plan)
                } else {
                    let mut right_scan_fetch = None;
                    let mut right_scan_only_filters = vec![];
                    let right_plan =
                        match try_transform_to_simple_table_scan_with_filters(right_plan)?
                        {
                            Some(scan) => {
                                right_scan_filters.extend(scan.filters);
                                right_scan_only_filters = scan.scan_filters;
                                right_scan_fetch = scan.fetch;
                                Arc::new(scan.plan)
                            }
                            None => Arc::clone(right_plan),
                        };
                    // A predicate lifted out of the non-preserved input carrying a
                    // subquery cannot go to `ON`, which some dialects refuse it in,
                    // nor to `WHERE`, which would discard the rows the join
                    // preserves. It selects the same rows applied to that input, so
                    // it stays in the input's own scope.
                    if join.join_type == JoinType::Left {
                        let (kept, scoped) = partition_subquery_filters(std::mem::take(
                            &mut right_scan_filters,
                        ));
                        right_scan_filters = kept;
                        right_scoped.extend(scoped);
                    }

                    self.select_to_sql_recursively(
                        right_plan.as_ref(),
                        query,
                        select,
                        &mut right_relation,
                    )?;
                    if right_scan_fetch.is_some()
                        || (join.join_type == JoinType::Full
                            && !right_scan_filters.is_empty())
                        || !right_scoped.is_empty()
                    {
                        let mut side_filters = std::mem::take(&mut right_scoped);
                        if right_scan_fetch.is_some() || join.join_type == JoinType::Full
                        {
                            side_filters.append(&mut right_scan_filters);
                        }
                        self.derive_join_side(
                            right_plan.as_ref(),
                            side_filters,
                            &right_scan_only_filters,
                            right_scan_fetch,
                            &mut right_relation,
                        )?;
                    }
                    right_plan
                };

                // Table-scan filters extracted from the inputs must land in the
                // outer query. EXISTS-style joins have no outer `ON` clause, so
                // all such filters go to the outer `WHERE` (only the preserved
                // side is transformed above, so `right_scan_filters` is empty);
                // regular joins split them between `ON` and `WHERE`.
                let (join_filters, where_filters) = if is_exists_join {
                    (join.filter.clone(), left_scan_filters)
                } else {
                    Self::split_join_on_and_where_filters(
                        join.join_type,
                        &scoped_join_filter,
                        left_scan_filters,
                        right_scan_filters,
                    )
                };
                for filter in where_filters {
                    let filter_expr = self.expr_to_sql(&filter)?;
                    select.selection(Some(filter_expr));
                }

                let mut join_constraint = self.join_constraint_to_sql(
                    join.join_constraint,
                    &join.on,
                    join_filters.as_ref(),
                )?;
                // `USING`/`NATURAL` cannot carry an extra predicate. Normally
                // the predicate goes back where it was, since nothing is
                // gained by mangling the join — but a predicate hoisted from
                // the non-preserved side of a null-extending join has nowhere
                // else to go: returning it to the SELECT-global `WHERE` would
                // discard unmatched rows from the preserved side. Downgrade
                // to an equivalent `ON` constraint so it can be appended.
                if hoisted_from_left.is_some()
                    && matches!(
                        join_constraint,
                        ast::JoinConstraint::Using(_) | ast::JoinConstraint::Natural
                    )
                {
                    join_constraint =
                        self.join_conditions_to_sql_on(&join.on, join_filters.as_ref())?;
                }
                let (join_constraint, unhoistable) =
                    Self::and_into_join_constraint(join_constraint, hoisted_from_left);
                select.selection(unhoistable);

                let right_projection: Option<Vec<ast::SelectItem>> =
                    if !already_projected && !is_exists_join {
                        Some(select.pop_projections())
                    } else {
                        None
                    };

                match join.join_type {
                    JoinType::LeftSemi
                    | JoinType::LeftAnti
                    | JoinType::LeftMark
                    | JoinType::RightSemi
                    | JoinType::RightAnti
                    | JoinType::RightMark => {
                        let subquery =
                            self.build_exists_subquery(right_plan.as_ref(), join)?;

                        let negated = match join.join_type {
                            JoinType::LeftSemi
                            | JoinType::RightSemi
                            | JoinType::LeftMark
                            | JoinType::RightMark => false,
                            JoinType::LeftAnti | JoinType::RightAnti => true,
                            _ => unreachable!(),
                        };
                        let exists_expr = ast::Expr::Exists {
                            subquery: Box::new(subquery),
                            negated,
                        };

                        match join.join_type {
                            JoinType::LeftMark | JoinType::RightMark => {
                                let source_schema =
                                    if join.join_type == JoinType::LeftMark {
                                        right_plan.schema()
                                    } else {
                                        left_plan.schema()
                                    };
                                let (table_ref, _) = source_schema.qualified_field(0);
                                let column = self.col_to_sql(&Column::new(
                                    table_ref.cloned(),
                                    "mark",
                                ))?;
                                select.replace_mark(&column, &exists_expr);
                            }
                            _ => {
                                select.selection(Some(exists_expr));
                            }
                        }
                        if let Some(projection) = left_projection {
                            select.projection(projection);
                        }
                    }
                    JoinType::Inner
                    | JoinType::Left
                    | JoinType::Right
                    | JoinType::Full => {
                        let Ok(Some(relation)) = right_relation.build() else {
                            return internal_err!("Failed to build right relation");
                        };
                        let ast_join = ast::Join {
                            relation,
                            global: false,
                            join_operator: self
                                .join_operator_to_sql(join.join_type, join_constraint)?,
                        };
                        let mut from = select.pop_from().unwrap();
                        from.push_join(ast_join);
                        select.push_from(from);
                        if !already_projected {
                            let Some(left_projection) = left_projection else {
                                return internal_err!("Left projection is missing");
                            };

                            let Some(right_projection) = right_projection else {
                                return internal_err!("Right projection is missing");
                            };

                            let projection = left_projection
                                .into_iter()
                                .chain(right_projection)
                                .collect();
                            select.projection(projection);
                        }
                    }
                };

                Ok(())
            }
            LogicalPlan::SubqueryAlias(plan_alias) => {
                let (plan, mut columns) =
                    subquery_alias_inner_query_and_columns(plan_alias);
                let unparsed_table_scan = self.unparse_table_scan_pushdown(
                    plan,
                    Some(plan_alias.alias.clone()),
                    select.already_projected(),
                )?;

                // If the (possibly rewritten) inner plan builds its own
                // SELECT clauses (e.g. Aggregate adds GROUP BY, Window adds
                // OVER, etc.) and unparse_table_scan_pushdown couldn't reduce it,
                // we must emit a derived subquery: (SELECT ...) AS alias.
                // Without this, the recursive handler would merge those clauses
                // into the outer SELECT, losing the subquery structure entirely.
                if unparsed_table_scan.is_none() && Self::requires_derived_subquery(plan)
                {
                    // When the dialect does not support column aliases in
                    // table aliases (e.g. SQLite), inject the aliases into
                    // the inner projection before wrapping as a derived
                    // subquery.
                    if !columns.is_empty()
                        && !self.dialect.supports_column_alias_in_table_alias()
                    {
                        let Ok(rewritten_plan) =
                            inject_column_aliases_into_subquery(plan.clone(), columns)
                        else {
                            return internal_err!(
                                "Failed to transform SubqueryAlias plan"
                            );
                        };
                        return self.derive(
                            &rewritten_plan,
                            relation,
                            Some(self.new_table_alias(
                                plan_alias.alias.table().to_string(),
                                vec![],
                            )),
                            false,
                        );
                    }
                    return self.derive(
                        plan,
                        relation,
                        Some(self.new_table_alias(
                            plan_alias.alias.table().to_string(),
                            columns,
                        )),
                        false,
                    );
                }

                // if the child plan is a TableScan with pushdown operations, we don't need to
                // create an additional subquery for it
                if !select.already_projected() && unparsed_table_scan.is_none() {
                    select.projection(vec![ast::SelectItem::Wildcard(
                        ast::WildcardAdditionalOptions::default(),
                    )]);
                }
                let plan = unparsed_table_scan.unwrap_or_else(|| plan.clone());
                if !columns.is_empty()
                    && !self.dialect.supports_column_alias_in_table_alias()
                {
                    // Instead of specifying column aliases as part of the outer table, inject them directly into the inner projection
                    let rewritten_plan =
                        match inject_column_aliases_into_subquery(plan, columns) {
                            Ok(p) => p,
                            Err(e) => {
                                return internal_err!(
                                    "Failed to transform SubqueryAlias plan: {e}"
                                );
                            }
                        };

                    columns = vec![];

                    self.select_to_sql_recursively(
                        &rewritten_plan,
                        query,
                        select,
                        relation,
                    )?;
                } else {
                    // The alias attached below carries these names, so a derived
                    // table built during the walk does not have to name its own
                    // outputs to be addressable. Restored afterwards because this
                    // builder outlives the walk — a join derives its other side
                    // through the same one.
                    let outer = relation.has_columns_named_by_alias();
                    relation.columns_named_by_alias(outer || !columns.is_empty());
                    self.select_to_sql_recursively(&plan, query, select, relation)?;
                    relation.columns_named_by_alias(outer);
                }

                relation.alias(Some(
                    self.new_table_alias(plan_alias.alias.table().to_string(), columns),
                ));

                // If this SubqueryAlias wraps a FLATTEN (Snowflake unnest),
                // register the alias so the outer Projection can rewrite
                // column references to use VALUE.
                if self.dialect.unnest_as_lateral_flatten()
                    && find_unnest_node_until_relation(plan_alias.input.as_ref())
                        .is_some()
                {
                    select.add_flatten_table_alias(plan_alias.alias.table().to_string());
                }

                Ok(())
            }
            LogicalPlan::Union(union) => {
                // Covers cases where the UNION is a subquery and the projection is at the top level
                if select.already_projected() {
                    return self.derive_with_dialect_alias(
                        DERIVED_UNION_ALIAS,
                        plan,
                        relation,
                        false,
                        vec![],
                    );
                }

                let input_exprs: Vec<SetExpr> = union
                    .inputs
                    .iter()
                    .map(|input| self.select_to_sql_expr(input, query))
                    .collect::<Result<Vec<_>>>()?;

                assert_or_internal_err!(
                    input_exprs.len() >= 2,
                    "UNION operator requires at least 2 inputs"
                );

                let set_quantifier =
                    if query.as_ref().is_some_and(|q| q.is_distinct_union()) {
                        self.dialect.union_distinct_set_quantifier()
                    } else {
                        ast::SetQuantifier::All
                    };

                // Build the union expression tree bottom-up by reversing the order
                // note that we are also swapping left and right inputs because of the rev
                let union_expr = input_exprs
                    .into_iter()
                    .rev()
                    .reduce(|a, b| SetExpr::SetOperation {
                        op: ast::SetOperator::Union,
                        set_quantifier,
                        left: Box::new(b),
                        right: Box::new(a),
                    })
                    .unwrap();

                let Some(query) = query.as_mut() else {
                    return internal_err!(
                        "UNION ALL operator only valid in a statement context"
                    );
                };
                query.body(Box::new(union_expr));

                Ok(())
            }
            LogicalPlan::Window(window) => {
                // Window nodes are usually handled simultaneously with Projection
                // nodes, where projected columns are unprojected back into their
                // corresponding window expressions. Manually built plans can have
                // Window nodes without an enclosing Projection, so in that case
                // the Window node itself must contribute its output expressions.
                let project_window_output = !select.already_projected();
                if project_window_output
                    && Self::window_input_requires_derived_subquery(window.input.as_ref())
                {
                    return self
                        .window_to_sql_with_derived_input(window, select, relation);
                }

                let agg = if project_window_output {
                    find_agg_node_within_select(plan, false)
                } else {
                    None
                };

                self.select_to_sql_recursively(
                    window.input.as_ref(),
                    query,
                    select,
                    relation,
                )?;

                if project_window_output {
                    self.project_window_output(&window.window_expr, select, agg)?;
                }

                Ok(())
            }
            LogicalPlan::EmptyRelation(_) => {
                // An EmptyRelation could be behind an UNNEST node. If the dialect supports UNNEST as a table factor,
                // a TableRelationBuilder will be created for the UNNEST node first.
                if !relation.has_relation() {
                    relation.empty();
                }
                Ok(())
            }
            LogicalPlan::Extension(extension) => {
                if let Some(query) = query.as_mut() {
                    self.extension_to_sql(
                        extension.node.as_ref(),
                        &mut Some(query),
                        &mut Some(select),
                        &mut Some(relation),
                    )
                } else {
                    self.extension_to_sql(
                        extension.node.as_ref(),
                        &mut None,
                        &mut Some(select),
                        &mut Some(relation),
                    )
                }
            }
            LogicalPlan::Unnest(unnest) => {
                if !unnest.struct_type_columns.is_empty() {
                    if self.dialect.unnest_as_lateral_flatten() {
                        return not_impl_err!(
                            "Snowflake FLATTEN cannot unparse struct unnest: \
                             DataFusion expands struct fields into columns (horizontal), \
                             but Snowflake FLATTEN expands them into rows (vertical). \
                             Columns: {:?}",
                            unnest.struct_type_columns
                        );
                    }
                    return internal_err!(
                        "Struct type columns are not currently supported in UNNEST: {:?}",
                        unnest.struct_type_columns
                    );
                }

                // For Snowflake FLATTEN: if the relation hasn't been set yet
                // (UNNEST was in SELECT clause, not FROM clause), set the FLATTEN
                // relation here so the FROM clause is emitted.
                if self.dialect.unnest_as_lateral_flatten()
                    && !relation.has_relation()
                    && let Some(mut flatten_relation) =
                        self.try_unnest_to_lateral_flatten_sql(unnest)?
                {
                    // Use the alias already generated by the Projection
                    // handler so SELECT items and the FLATTEN relation
                    // reference the same name.
                    if let Some(alias) = select.current_flatten_alias() {
                        flatten_relation.alias(Some(ast::TableAlias {
                            name: Ident::with_quote('"', &alias),
                            columns: vec![],
                            explicit: true,
                            at: None,
                        }));
                    }
                    relation.flatten(flatten_relation);
                }

                // In the case of UNNEST, the Unnest node is followed by a duplicate Projection node that we should skip.
                // Otherwise, there will be a duplicate SELECT clause.
                // | Projection: table.col1, UNNEST(table.col2)
                // |   Unnest: UNNEST(table.col2)
                // |     Projection: table.col1, table.col2 AS UNNEST(table.col2)
                // |       Filter: table.col3 = Int64(3)
                // |         TableScan: table projection=None
                if let Some(p) = Self::peel_to_inner_projection(unnest.input.as_ref()) {
                    // Skip the inner Projection (synthetic rewriter node)
                    // and continue with its input.
                    self.select_to_sql_recursively(&p.input, query, select, relation)
                } else {
                    internal_err!("Unnest input is not a Projection: {unnest:?}")
                }
            }
            LogicalPlan::Subquery(subquery)
                if find_unnest_node_until_relation(subquery.subquery.as_ref())
                    .is_some() =>
            {
                if self.dialect.unnest_as_table_factor()
                    || self.dialect.unnest_as_lateral_flatten()
                {
                    self.select_to_sql_recursively(
                        subquery.subquery.as_ref(),
                        query,
                        select,
                        relation,
                    )
                } else {
                    self.derive_with_dialect_alias(
                        DERIVED_UNNEST_ALIAS,
                        subquery.subquery.as_ref(),
                        relation,
                        true,
                        vec![],
                    )
                }
            }
            LogicalPlan::RecursiveQuery(recursive)
                if self.dialect.supports_recursive_cte() =>
            {
                // A recursive CTE nested inside a larger statement becomes a CTE
                // on the enclosing query and a plain table reference where it
                // stood, which is the shape the plan already describes: the
                // recursive term refers to the working table by this same name.
                let (cte, name) = self.recursive_cte(recursive)?;
                let Some(query) = query.as_mut() else {
                    return internal_err!(
                        "a recursive CTE is only valid in a statement context"
                    );
                };
                query.push_cte(cte, true);

                let mut builder = TableRelationBuilder::default();
                builder.name(ast::ObjectName::from(vec![name]));
                relation.table(builder);

                Ok(())
            }
            _ => {
                not_impl_err!("Unsupported operator: {plan:?}")
            }
        }
    }

    /// Walk through transparent nodes (SubqueryAlias) to find the inner
    /// Projection that feeds an Unnest node.
    ///
    /// The inner Projection is created atomically by the
    /// `RecursiveUnnestRewriter` and contains the array expression that the
    /// Unnest operates on. A `SubqueryAlias` (e.g. from a virtual/passthrough
    /// table) may wrap the Projection.
    fn peel_to_inner_projection(plan: &LogicalPlan) -> Option<&Projection> {
        match plan {
            LogicalPlan::Projection(p) => Some(p),
            LogicalPlan::SubqueryAlias(alias) => {
                Self::peel_to_inner_projection(alias.input.as_ref())
            }
            _ => None,
        }
    }

    /// Walk through transparent nodes (Limit, Sort) between the outer
    /// Projection and the Unnest, applying their SQL modifiers (LIMIT,
    /// OFFSET, ORDER BY) to the query builder. Returns the `Unnest` node
    /// and a reference to the enclosing `LogicalPlan` for recursion, or
    /// `Ok(None)` if no Unnest is found.
    ///
    /// By processing Limit/Sort inline and then recursing into the Unnest
    /// plan directly, we bypass the normal Limit/Sort handlers which would
    /// create unwanted derived subqueries (since `already_projected` is
    /// set at the point this is called).
    fn peel_to_unnest_with_modifiers<'a>(
        &self,
        plan: &'a LogicalPlan,
        query: &mut Option<QueryBuilder>,
        flatten_alias: Option<&str>,
    ) -> Result<Option<(&'a Unnest, &'a LogicalPlan)>> {
        match plan {
            LogicalPlan::Unnest(unnest) => Ok(Some((unnest, plan))),
            LogicalPlan::Limit(limit) => {
                if let Some(fetch) = &limit.fetch
                    && let Some(q) = query.as_mut()
                {
                    q.limit(Some(self.expr_to_sql(fetch)?));
                }
                if let Some(skip) = &limit.skip
                    && let Some(q) = query.as_mut()
                {
                    q.offset(Some(ast::Offset {
                        rows: ast::OffsetRows::None,
                        value: self.expr_to_sql(skip)?,
                    }));
                }
                self.peel_to_unnest_with_modifiers(
                    limit.input.as_ref(),
                    query,
                    flatten_alias,
                )
            }
            LogicalPlan::Sort(sort) => {
                let Some(query_ref) = query.as_mut() else {
                    return internal_err!(
                        "Sort between Projection and Unnest requires a statement context."
                    );
                };
                if let Some(fetch) = sort.fetch {
                    query_ref.limit(Some(ast::Expr::value(ast::Value::Number(
                        fetch.to_string(),
                        false,
                    ))));
                }
                // When a flatten_alias is provided, rewrite
                // __unnest_placeholder(...) columns in sort expressions to
                // alias.VALUE so ORDER BY references the FLATTEN output.
                let unnest_node = match sort.input.as_ref() {
                    LogicalPlan::Unnest(u) => Some(u),
                    _ => find_unnest_node_within_select(sort.input.as_ref()),
                };
                let sort_exprs = if let Some(alias) = flatten_alias
                    && let Some(unnest) = unnest_node
                {
                    sort.expr
                        .iter()
                        .map(|s| {
                            let rewritten = unproject_unnest_expr_as_flatten_value(
                                s.expr.clone(),
                                unnest,
                                alias,
                            )?;
                            Ok(SortExpr {
                                expr: rewritten,
                                ..s.clone()
                            })
                        })
                        .collect::<Result<Vec<_>>>()?
                } else {
                    sort.expr.clone()
                };
                query_ref.order_by(self.sorts_to_sql(&sort_exprs)?);
                self.peel_to_unnest_with_modifiers(
                    sort.input.as_ref(),
                    query,
                    flatten_alias,
                )
            }
            _ => Ok(None),
        }
    }

    /// Search an expression tree for an unnest placeholder column reference.
    ///
    /// Returns the [`UnnestInputType`] if any sub-expression is a column
    /// whose name starts with `__unnest_placeholder`. The placeholder may
    /// be at the top level (bare), inside a function call, or one of several
    /// expressions — this function finds it regardless.
    fn find_unnest_placeholder(expr: &Expr) -> Option<UnnestInputType> {
        let mut result = None;
        let _ = expr.apply(|e| {
            if let Some(t) = Self::classify_placeholder_column(e) {
                result = Some(t);
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        result
    }

    /// Returns true if `expr` is a placeholder column, optionally wrapped
    /// in a single alias (the rewriter's internal `UNNEST(...)` name).
    /// Does NOT match when a user alias wraps the internal alias
    /// (e.g. `Alias("c1", Alias("UNNEST(...)", Column(placeholder)))`),
    /// so the table-factor path correctly falls through to
    /// `reconstruct_select_statement` which preserves user aliases.
    fn is_bare_unnest_placeholder(expr: &Expr) -> bool {
        // Peel at most one alias layer (the rewriter's internal name).
        let inner = match expr {
            Expr::Alias(Alias { expr, .. }) => expr.as_ref(),
            other => other,
        };
        Self::classify_placeholder_column(inner).is_some()
    }

    /// If `expr` is a `Column` whose name starts with `__unnest_placeholder`,
    /// classify it as [`UnnestInputType::OuterReference`] or
    /// [`UnnestInputType::Scalar`].
    fn classify_placeholder_column(expr: &Expr) -> Option<UnnestInputType> {
        if let Expr::Column(Column { name, .. }) = expr
            && let Some(prefix) = name.strip_prefix(UNNEST_PLACEHOLDER)
        {
            if prefix.starts_with(&format!("({OUTER_REFERENCE_COLUMN_PREFIX}(")) {
                return Some(UnnestInputType::OuterReference);
            }
            return Some(UnnestInputType::Scalar);
        }
        None
    }

    /// Check whether an expression carries an internal `UNNEST(...)` display
    /// name as its column name or outermost alias. After
    /// [`unproject_unnest_expr_as_flatten_value`] rewrites the placeholder
    /// column to `_unnest.VALUE`, the internal alias may still linger
    /// (e.g. `Alias("UNNEST(make_array(...))", Column("_unnest.VALUE"))`).
    /// Callers use this to replace the expression with a clean
    /// `_unnest."VALUE"` select item.
    fn has_internal_unnest_alias(expr: &Expr) -> bool {
        match expr {
            Expr::Column(col) => {
                col.name.starts_with(&format!("{UNNEST_COLUMN_PREFIX}("))
            }
            Expr::Alias(Alias { name, .. }) => {
                name.starts_with(&format!("{UNNEST_COLUMN_PREFIX}("))
            }
            _ => false,
        }
    }

    /// Walk the plan tree and register any SubqueryAlias that wraps an
    /// unnest as a FLATTEN table alias on the SelectBuilder. This allows
    /// `reconstruct_select_statement` to rewrite column references (e.g.
    /// `a.col` → `a.VALUE`) before the SubqueryAlias handler runs.
    /// Returns true if a plan tree contains an Unnest node, searching
    /// through Projection, Subquery, and SubqueryAlias wrappers.
    fn contains_unnest(plan: &LogicalPlan) -> bool {
        match plan {
            LogicalPlan::Unnest(_) => true,
            LogicalPlan::Projection(p) => Self::contains_unnest(&p.input),
            LogicalPlan::Subquery(s) => Self::contains_unnest(&s.subquery),
            LogicalPlan::SubqueryAlias(a) => Self::contains_unnest(&a.input),
            _ => false,
        }
    }

    fn collect_flatten_aliases(plan: &LogicalPlan, select: &mut SelectBuilder) {
        match plan {
            LogicalPlan::SubqueryAlias(alias)
                if Self::contains_unnest(alias.input.as_ref()) =>
            {
                select.add_flatten_table_alias(alias.alias.table().to_string());
            }
            LogicalPlan::Join(join) => {
                Self::collect_flatten_aliases(&join.left, select);
                Self::collect_flatten_aliases(&join.right, select);
            }
            _ => {}
        }
    }

    fn try_unnest_to_table_factor_sql(
        &self,
        unnest: &Unnest,
    ) -> Result<Option<UnnestRelationBuilder>> {
        let mut unnest_relation = UnnestRelationBuilder::default();
        let LogicalPlan::Projection(projection) = unnest.input.as_ref() else {
            return Ok(None);
        };

        if !matches!(projection.input.as_ref(), LogicalPlan::EmptyRelation(_)) {
            // It may be possible that UNNEST is used as a source for the query.
            // However, at this point, we don't yet know if it is just a single expression
            // from another source or if it's from UNNEST.
            //
            // Unnest(Projection(EmptyRelation)) denotes a case with `UNNEST([...])`,
            // which is normally safe to unnest as a table factor.
            // However, in the future, more comprehensive checks can be added here.
            return Ok(None);
        };

        let exprs = projection
            .expr
            .iter()
            .map(|e| self.expr_to_sql(e))
            .collect::<Result<Vec<_>>>()?;
        unnest_relation.array_exprs(exprs);

        Ok(Some(unnest_relation))
    }

    /// Build a `SELECT alias."VALUE"` item for Snowflake FLATTEN output.
    fn build_flatten_value_select_item(
        &self,
        flatten_alias: &str,
        user_alias: Option<&str>,
    ) -> ast::SelectItem {
        let compound = ast::Expr::CompoundIdentifier(vec![
            self.new_ident_quoted_if_needs(flatten_alias.to_string()),
            Ident::with_quote('"', "VALUE"),
        ]);
        match user_alias {
            Some(alias) => ast::SelectItem::ExprWithAlias {
                expr: compound,
                alias: self.new_ident_quoted_if_needs(alias.to_string()),
            },
            None => ast::SelectItem::UnnamedExpr(compound),
        }
    }

    /// Convert an `Unnest` logical plan node to a `LATERAL FLATTEN(INPUT => expr, ...)`
    /// table factor for Snowflake-style SQL output.
    fn try_unnest_to_lateral_flatten_sql(
        &self,
        unnest: &Unnest,
    ) -> Result<Option<FlattenRelationBuilder>> {
        let Some(projection) = Self::peel_to_inner_projection(unnest.input.as_ref())
        else {
            return Ok(None);
        };

        // For now, handle the simple case of a single expression to flatten.
        // Multi-expression would require multiple LATERAL FLATTEN calls chained together.
        let Some(first_expr) = projection.expr.first() else {
            return Ok(None);
        };

        let input_expr = self.expr_to_sql(first_expr)?;

        let mut flatten = FlattenRelationBuilder::default();
        flatten.input_expr(input_expr);
        flatten.outer(unnest.options.preserve_nulls);

        Ok(Some(flatten))
    }

    fn is_scan_with_pushdown(scan: &TableScan) -> bool {
        scan.projection.is_some() || !scan.filters.is_empty() || scan.fetch.is_some()
    }

    /// Returns true if a plan, when used as the direct child of a SubqueryAlias,
    /// must be emitted as a derived subquery `(SELECT ...) AS alias`.
    ///
    /// Plans like Aggregate or Window build their own SELECT clauses (GROUP BY,
    /// window functions).
    fn requires_derived_subquery(plan: &LogicalPlan) -> bool {
        match plan {
            LogicalPlan::Aggregate(_)
            | LogicalPlan::Window(_)
            | LogicalPlan::Sort(_)
            | LogicalPlan::Limit(_)
            | LogicalPlan::Union(_) => true,
            // Semi/anti joins generate correlated EXISTS subqueries whose WHERE conditions
            // reference the join's left-branch alias via `select.selection()`. When a
            // SubqueryAlias later calls `relation.alias(outer_name)` it overwrites that
            // alias, causing the EXISTS correlation to reference a table name that no
            // longer appears in FROM. Wrapping as a derived subquery isolates the EXISTS
            // conditions inside their own SELECT scope, so the alias overwrite only
            // affects the outer wrapper and the correlation remains valid.
            LogicalPlan::Join(join) => matches!(
                join.join_type,
                JoinType::LeftSemi
                    | JoinType::LeftAnti
                    | JoinType::RightSemi
                    | JoinType::RightAnti
                    | JoinType::LeftMark
                    | JoinType::RightMark
            ),
            // Peek through Distinct: INTERSECT produces Distinct(LeftSemiJoin(...))
            LogicalPlan::Distinct(distinct) => {
                Self::requires_derived_subquery(distinct.input())
            }
            _ => false,
        }
    }

    /// Try to unparse a table scan with pushdown operations into a new subquery plan.
    /// If the table scan is without any pushdown operations, return None.
    fn unparse_table_scan_pushdown(
        &self,
        plan: &LogicalPlan,
        alias: Option<TableReference>,
        already_projected: bool,
    ) -> Result<Option<LogicalPlan>> {
        match plan {
            LogicalPlan::TableScan(table_scan) => {
                if !Self::is_scan_with_pushdown(table_scan) {
                    return Ok(None);
                }
                let table_schema = table_scan.source.schema();
                let mut filter_alias_rewriter =
                    alias.as_ref().map(|alias_name| TableAliasRewriter {
                        table_schema: &table_schema,
                        alias_name: alias_name.clone(),
                    });

                let mut builder = LogicalPlanBuilder::scan(
                    table_scan.table_name.clone(),
                    Arc::clone(&table_scan.source),
                    None,
                )?;
                // We will rebase the column references to the new alias if it exists.
                // If the projection or filters are empty, we will append alias to the table scan.
                //
                // Example:
                //   select t1.c1 from t1 where t1.c1 > 1 -> select a.c1 from t1 as a where a.c1 > 1
                if let Some(ref alias) = alias
                    && (table_scan.projection.is_some() || !table_scan.filters.is_empty())
                {
                    builder = builder.alias(alias.clone())?;
                }

                // Avoid creating a duplicate Projection node, which would result in an additional subquery if a projection already exists.
                // For example, if the `optimize_projection` rule is applied, there will be a Projection node, and duplicate projection
                // information included in the TableScan node.
                if !already_projected && let Some(project_vec) = &table_scan.projection {
                    if project_vec.is_empty() {
                        builder = builder.project(self.empty_projection_fallback())?;
                    } else {
                        let project_columns = project_vec
                            .iter()
                            .cloned()
                            .map(|i| {
                                let schema = table_scan.source.schema();
                                let field = schema.field(i);
                                if alias.is_some() {
                                    Column::new(alias.clone(), field.name().clone())
                                } else {
                                    Column::new(
                                        Some(table_scan.table_name.clone()),
                                        field.name().clone(),
                                    )
                                }
                            })
                            .collect::<Vec<_>>();
                        builder = builder.project(project_columns)?;
                    };
                }

                let filter_expr: Result<Option<Expr>> = table_scan
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
                    .reduce(|acc, expr_result| {
                        acc.and_then(|acc_expr| {
                            expr_result.map(|expr| acc_expr.and(expr))
                        })
                    })
                    .transpose();

                if let Some(filter) = filter_expr? {
                    builder = builder.filter(filter)?;
                }

                if let Some(fetch) = table_scan.fetch {
                    builder = builder.limit(0, Some(fetch))?;
                }

                // If the table scan has an alias but no projection or filters, it means no column references are rebased.
                // So we will append the alias to this subquery.
                // Example:
                //   select * from t1 limit 10 -> (select * from t1 limit 10) as a
                if let Some(alias) = alias
                    && table_scan.projection.is_none()
                    && table_scan.filters.is_empty()
                {
                    builder = builder.alias(alias)?;
                }

                Ok(Some(builder.build()?))
            }
            LogicalPlan::SubqueryAlias(subquery_alias) => {
                let ret = self.unparse_table_scan_pushdown(
                    &subquery_alias.input,
                    Some(subquery_alias.alias.clone()),
                    already_projected,
                )?;
                if let Some(alias) = alias
                    && let Some(plan) = ret
                {
                    let plan = LogicalPlanBuilder::new(plan).alias(alias)?.build()?;
                    return Ok(Some(plan));
                }
                Ok(ret)
            }
            // Handle Filter between SubqueryAlias and TableScan (e.g. Inexact/Unsupported
            // filter pushdown). Rewrite predicate column references to use the alias.
            // Skip predicates with subquery expressions — TableAliasRewriter
            // cannot rewrite OuterReferenceColumn inside subquery LogicalPlans.
            // Returning None lets the caller wrap the plan as a derived table,
            // preserving the original table name for outer references and generate correct SQL.
            LogicalPlan::Filter(filter) => {
                if filter.predicate.exists(|e| {
                    Ok(matches!(
                        e,
                        Expr::Exists(_) | Expr::InSubquery(_) | Expr::ScalarSubquery(_)
                    ))
                })? {
                    return Ok(None);
                }

                if let Some(plan) = self.unparse_table_scan_pushdown(
                    &filter.input,
                    alias.clone(),
                    already_projected,
                )? {
                    let predicate = if let Some(ref alias_name) = alias {
                        let mut rewriter = TableAliasRewriter {
                            table_schema: plan.schema().as_arrow(),
                            alias_name: alias_name.clone(),
                        };
                        filter.predicate.clone().rewrite(&mut rewriter).data()?
                    } else {
                        filter.predicate.clone()
                    };
                    Ok(Some(
                        LogicalPlanBuilder::from(plan).filter(predicate)?.build()?,
                    ))
                } else {
                    Ok(None)
                }
            }
            // SubqueryAlias could be rewritten to a plan with a projection as the top node by [rewrite::subquery_alias_inner_query_and_columns].
            // The inner table scan could be a scan with pushdown operations.
            LogicalPlan::Projection(projection) => {
                if let Some(plan) = self.unparse_table_scan_pushdown(
                    &projection.input,
                    alias.clone(),
                    already_projected,
                )? {
                    let exprs = if alias.is_some() {
                        let mut alias_rewriter =
                            alias.as_ref().map(|alias_name| TableAliasRewriter {
                                table_schema: plan.schema().as_arrow(),
                                alias_name: alias_name.clone(),
                            });
                        projection
                            .expr
                            .iter()
                            .cloned()
                            .map(|expr| {
                                if let Some(ref mut rewriter) = alias_rewriter {
                                    expr.rewrite(rewriter).data()
                                } else {
                                    Ok(expr)
                                }
                            })
                            .collect::<Result<Vec<_>>>()?
                    } else {
                        projection.expr.clone()
                    };
                    Ok(Some(
                        LogicalPlanBuilder::from(plan).project(exprs)?.build()?,
                    ))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    fn select_item_to_sql(&self, expr: &Expr) -> Result<ast::SelectItem> {
        match expr {
            Expr::Alias(Alias { expr, name, .. }) => {
                let inner = self.expr_to_sql(expr)?;

                let col_name = self.emitted_column_name(name)?;

                Ok(ast::SelectItem::ExprWithAlias {
                    expr: inner,
                    alias: self.new_ident_quoted_if_needs(col_name),
                })
            }
            _ => {
                let inner = self.expr_to_sql(expr)?;

                Ok(ast::SelectItem::UnnamedExpr(inner))
            }
        }
    }

    fn sorts_to_sql(&self, sort_exprs: &[SortExpr]) -> Result<OrderByKind> {
        Ok(OrderByKind::Expressions(
            sort_exprs
                .iter()
                .map(|sort_expr| self.sort_to_sql(sort_expr))
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    fn join_operator_to_sql(
        &self,
        join_type: JoinType,
        constraint: ast::JoinConstraint,
    ) -> Result<ast::JoinOperator> {
        Ok(match join_type {
            JoinType::Inner => match &constraint {
                ast::JoinConstraint::On(_)
                | ast::JoinConstraint::Using(_)
                | ast::JoinConstraint::Natural => ast::JoinOperator::Inner(constraint),
                ast::JoinConstraint::None => {
                    // Inner joins with no conditions or filters are not valid SQL in most systems,
                    // return a CROSS JOIN instead
                    ast::JoinOperator::CrossJoin(constraint)
                }
            },
            JoinType::Left => ast::JoinOperator::LeftOuter(constraint),
            JoinType::Right => ast::JoinOperator::RightOuter(constraint),
            JoinType::Full => ast::JoinOperator::FullOuter(constraint),
            JoinType::LeftAnti => ast::JoinOperator::LeftAnti(constraint),
            JoinType::LeftSemi => ast::JoinOperator::LeftSemi(constraint),
            JoinType::RightAnti => ast::JoinOperator::RightAnti(constraint),
            JoinType::RightSemi => ast::JoinOperator::RightSemi(constraint),
            JoinType::LeftMark | JoinType::RightMark => {
                unimplemented!("Unparsing of Mark join type")
            }
        })
    }

    /// Convert the components of a USING clause to the USING AST. Returns
    /// 'None' if the conditions are not compatible with a USING expression,
    /// e.g. non-column expressions or non-matching names.
    fn join_using_to_sql(
        &self,
        join_conditions: &[(Expr, Expr)],
    ) -> Option<ast::JoinConstraint> {
        let mut object_names = Vec::with_capacity(join_conditions.len());
        for (left, right) in join_conditions {
            match (left, right) {
                (
                    Expr::Column(Column {
                        relation: _,
                        name: left_name,
                        spans: _,
                    }),
                    Expr::Column(Column {
                        relation: _,
                        name: right_name,
                        spans: _,
                    }),
                ) if left_name == right_name => {
                    // For example, if the join condition `t1.id = t2.id`
                    // this is represented as two columns like `[t1.id, t2.id]`
                    // This code forms `id` (without relation name)
                    let ident = self.new_ident_quoted_if_needs(left_name.to_string());
                    object_names.push(ast::ObjectName::from(vec![ident]));
                }
                // USING is only valid with matching column names; arbitrary expressions
                // are not allowed
                _ => return None,
            }
        }
        Some(ast::JoinConstraint::Using(object_names))
    }

    /// Convert a join constraint and associated conditions and filter to a SQL AST node
    fn join_constraint_to_sql(
        &self,
        constraint: JoinConstraint,
        conditions: &[(Expr, Expr)],
        filter: Option<&Expr>,
    ) -> Result<ast::JoinConstraint> {
        match (constraint, conditions, filter) {
            // No constraints
            (JoinConstraint::On | JoinConstraint::Using, [], None) => {
                Ok(ast::JoinConstraint::None)
            }

            (JoinConstraint::Using, conditions, None) => {
                match self.join_using_to_sql(conditions) {
                    Some(using) => Ok(using),
                    // As above, this should not be reachable from parsed SQL,
                    // but a user could create this; we "downgrade" to ON.
                    None => self.join_conditions_to_sql_on(conditions, None),
                }
            }

            // Two cases here:
            // 1. Straightforward ON case, with possible equi-join conditions
            //    and additional filters
            // 2. USING with additional filters; we "downgrade" to ON, because
            //    you can't use USING with arbitrary filters. (This should not
            //    be accessible from parsed SQL, but may have been a
            //    custom-built JOIN by a user.)
            (JoinConstraint::On | JoinConstraint::Using, conditions, filter) => {
                self.join_conditions_to_sql_on(conditions, filter)
            }
        }
    }

    // Convert a list of equi0join conditions and an optional filter to a SQL ON
    // AST node, with the equi-join conditions and the filter merged into a
    // single conditional expression
    fn join_conditions_to_sql_on(
        &self,
        join_conditions: &[(Expr, Expr)],
        filter: Option<&Expr>,
    ) -> Result<ast::JoinConstraint> {
        let mut condition = None;
        // AND the join conditions together to create the overall condition
        for (left, right) in join_conditions {
            // Parse left and right
            let l = self.expr_to_sql(left)?;
            let r = self.expr_to_sql(right)?;
            let e = self.binary_op_to_sql(l, r, ast::BinaryOperator::Eq);
            condition = match condition {
                Some(expr) => Some(self.and_op_to_sql(expr, e)),
                None => Some(e),
            };
        }

        // Then AND the non-equijoin filter condition as well
        condition = match (condition, filter) {
            (Some(expr), Some(filter)) => {
                Some(self.and_op_to_sql(expr, self.expr_to_sql(filter)?))
            }
            (Some(expr), None) => Some(expr),
            (None, Some(filter)) => Some(self.expr_to_sql(filter)?),
            (None, None) => None,
        };

        let constraint = match condition {
            Some(filter) => ast::JoinConstraint::On(filter),
            None => ast::JoinConstraint::None,
        };

        Ok(constraint)
    }

    fn and_op_to_sql(&self, lhs: ast::Expr, rhs: ast::Expr) -> ast::Expr {
        self.binary_op_to_sql(lhs, rhs, ast::BinaryOperator::And)
    }

    fn new_table_alias(&self, alias: String, columns: Vec<Ident>) -> ast::TableAlias {
        let columns = columns
            .into_iter()
            .map(|ident| TableAliasColumnDef {
                name: ident,
                data_type: None,
            })
            .collect();
        ast::TableAlias {
            name: self.new_ident_quoted_if_needs(alias),
            columns,
            explicit: true,
            at: None,
        }
    }

    fn dml_to_sql(&self, plan: &LogicalPlan) -> Result<ast::Statement> {
        not_impl_err!("Unsupported plan: {plan:?}")
    }

    /// Generates appropriate projection expression for empty projection lists.
    /// Returns an empty vec for dialects supporting empty select lists,
    /// or a dummy literal `1` for other dialects.
    fn empty_projection_fallback(&self) -> Vec<Expr> {
        if self.dialect.supports_empty_select_list() {
            Vec::new()
        } else {
            vec![Expr::Literal(ScalarValue::Int64(Some(1)), None)]
        }
    }

    /// Builds the body of an `EXISTS` subquery for a semi/anti/mark join.
    ///
    /// The build (right) side is unparsed as a complete, self-contained
    /// subquery so that any SELECT-level clauses it needs (projection, joins,
    /// DISTINCT, WHERE) stay inside the subquery instead of leaking into the
    /// outer SELECT. Its projection is replaced with `SELECT 1` and the join
    /// predicates (`ON` equalities plus any residual join filter) are
    /// AND-combined into its WHERE clause, becoming correlated references to
    /// the outer query.
    fn build_exists_subquery(
        &self,
        right_plan: &LogicalPlan,
        join: &Join,
    ) -> Result<ast::Query> {
        // Checked before any of the body is built: this refusal is about the
        // correlation's qualifiers alone, and holds whether or not a bound is
        // found below.
        self.ensure_exists_correlation_not_shadowed(join)?;

        let mut query_builder = Some(QueryBuilder::default());
        let body = self.select_to_sql_expr(right_plan, &mut query_builder)?;
        let mut query_builder = query_builder.unwrap();

        // Reduce the build side to a single SELECT to use as the EXISTS body.
        // A non-SELECT body (e.g. a set operation) is wrapped as a derived table.
        let mut select = match body {
            SetExpr::Select(select) => select,
            other => Box::new(Self::wrap_setexpr_as_derived_select(other)?),
        };

        // A row bound on the build side (`LIMIT`/`OFFSET`/`FETCH`/`LIMIT BY`)
        // is applied after that side's own `WHERE`, so a correlated predicate
        // added there would decide which rows the bound keeps: the subquery
        // would search the whole relation and report a match among rows the
        // plan never read. A semi or mark join then reports a match it does
        // not have and an anti join drops a row it should return — wrong rows
        // rather than too many. Give the bounded body a scope of its own and
        // correlate outside it.
        //
        // The rewrites below then belong to the outer select: inside the bound,
        // both the projection and `DISTINCT` still decide which rows survive
        // it, so neither is redundant there.
        //
        // `bounds_rows()` is tested first so the scope name is only demanded
        // when a bound actually has to be moved: naming it can refuse the plan
        // outright, and an unbounded build side has nothing to get wrong.
        if query_builder.bounds_rows() {
            let scope_name = self.exists_scope_name(join)?;
            query_builder.body(Box::new(SetExpr::Select(select)));
            let bounded = query_builder.build()?;
            let alias = self.new_table_alias(scope_name, vec![]);
            select = Box::new(Self::wrap_query_as_derived_select(bounded, Some(alias))?);
            // The bound now lives on the derived table; a second copy out here
            // would re-apply it to the correlated result.
            query_builder = QueryBuilder::default();
        }

        // `EXISTS` only needs `SELECT 1`; DISTINCT would be redundant.
        select.projection = vec![ast::SelectItem::UnnamedExpr(ast::Expr::value(
            ast::Value::Number("1".to_string(), false),
        ))];
        select.distinct = None;

        // AND the correlated join predicates — the `ON` equalities plus any
        // residual (non-equi) join filter — into the WHERE clause.
        let predicates = join
            .filter
            .iter()
            .cloned()
            .chain(join.on.iter().map(|(l, r)| l.clone().eq(r.clone())));
        for predicate in predicates {
            let predicate = self.expr_to_sql(&predicate)?;
            select.selection = Some(match select.selection.take() {
                Some(existing) => ast::Expr::BinaryOp {
                    left: Box::new(existing),
                    op: ast::BinaryOperator::And,
                    right: Box::new(predicate),
                },
                None => predicate,
            });
        }

        query_builder.body(Box::new(SetExpr::Select(select)));
        Ok(query_builder.build()?)
    }

    /// Wraps a non-SELECT set expression (e.g. a `UNION`) as an unaliased
    /// derived table inside a bare `SELECT`, so it can serve as an `EXISTS`
    /// body that [`build_exists_subquery`] then adds `SELECT 1` and correlated
    /// predicates to.
    ///
    /// [`build_exists_subquery`]: Self::build_exists_subquery
    fn wrap_setexpr_as_derived_select(body: SetExpr) -> Result<ast::Select> {
        let mut subquery = QueryBuilder::default();
        subquery.body(Box::new(body));
        let mut select = Self::wrap_query_as_derived_select(subquery.build()?, None)?;
        // An unset projection renders as `SELECT FROM`. A caller that goes on
        // to ask for `SELECT 1` overwrites this; one that wraps this select in
        // a further scope needs it to expose the body's columns to that scope.
        select.projection = vec![ast::SelectItem::Wildcard(
            ast::WildcardAdditionalOptions::default(),
        )];
        Ok(select)
    }

    /// Wraps a complete query as a derived table inside a bare `SELECT`, so
    /// clauses added to that `SELECT` are evaluated on the query's result
    /// rather than alongside its own.
    fn wrap_query_as_derived_select(
        query: ast::Query,
        alias: Option<ast::TableAlias>,
    ) -> Result<ast::Select> {
        let mut derived = DerivedRelationBuilder::default();
        derived
            .lateral(false)
            .alias(alias)
            .subquery(Box::new(query));

        let mut relation = RelationBuilder::default();
        relation.derived(derived);

        let mut from = TableWithJoinsBuilder::default();
        from.relation(relation);

        let mut select = SelectBuilder::default();
        select.push_from(from);
        Ok(select.build()?)
    }

    /// Whether a join presents its inputs to the unparser the other way round
    /// from the way the plan holds them: `RightSemi` and `RightAnti` correlate
    /// `join.right` and build the `EXISTS` body from `join.left`.
    ///
    /// Read from here rather than restated, so that everything deciding which
    /// side is which agrees — the join arm swaps the plans, and anything
    /// reading `join.on` has to take the key from the matching side of each
    /// pair or it will name the relation the correlation does not.
    const fn swaps_join_inputs(join_type: JoinType) -> bool {
        matches!(join_type, JoinType::RightSemi | JoinType::RightAnti)
    }

    /// Whether `qualifier`, as the emitted SQL spells it, is a name the unparser
    /// can invent for a derived table of its own.
    ///
    /// A derived table's alias is a single identifier, so a qualifier a dialect
    /// spells with a schema or catalog is not one of these however its last
    /// component reads — that reference names a relation the alias cannot answer
    /// to.
    fn is_unparser_derived_alias(qualifier: &[String]) -> bool {
        let [name] = qualifier else {
            return false;
        };
        DERIVED_TABLE_ALIASES.contains(&name.as_str()) || is_numbered_alias(name)
    }

    /// What the `EXISTS` body's `FROM` will answer to: its relation names, and the
    /// column names a reference inside it can collide with — including one the
    /// body renames, which no relation exposes.
    ///
    /// The qualifiers are read from the relations the `FROM` introduces rather
    /// than from `plan`'s output schema, which is a proxy that is wrong in both
    /// directions:
    ///
    /// * A projection pruning every column of a relation leaves it with no
    ///   qualified output field, while it is still named in the `FROM` and still
    ///   captures a reference using its name.
    /// * A `SubqueryAlias` puts its alias on the schema as the whole
    ///   [`TableReference`] it was built from, but a derived table's alias is a
    ///   single identifier, so only `table()` is emitted — and dialect-independently
    ///   so, unlike a scanned relation. Keying an alias `s.a` off the schema would
    ///   miss an outer `a.c` that `AS a` really does capture, and refuse an outer
    ///   `s.a.c` that it does not.
    ///
    /// The column names take both: the relations the walk finds expose every
    /// column they have rather than the projected ones, and the output schema
    /// carries names no relation in the plan has, because a rename names a
    /// column something. Neither alone is the set an unqualified reference can
    /// collide with. The seed below says where each of those lands in the
    /// emitted SQL.
    ///
    /// The walk stops at a `SubqueryAlias`: an alias replaces the name it is given
    /// to, so the relations it encloses are not addressable through it and
    /// collecting them would refuse references that bind correctly.
    ///
    /// Aliases the unparser invents for derived tables of its own are not here —
    /// they appear nowhere in the plan, so no walk can find them. They are
    /// recognised by [`Self::is_unparser_derived_alias`] instead.
    ///
    /// A node whose emitted relation the plan does not describe costs the walk
    /// either the whole scope or just its column names — see
    /// [`Self::unreadable_part`] for which those are and why, and
    /// [`Self::introduces_unreadable`] for why an enclosing alias shields
    /// neither.
    ///
    /// Only the plan's own inputs are walked, never the plans inside an
    /// `Expr::ScalarSubquery` or `Expr::Exists`. That is what makes the model
    /// hold: a relation in a nested subquery's `FROM` is not in scope for a
    /// predicate emitted at this body's level, so it cannot capture one.
    fn emitted_scope(&self, plan: &LogicalPlan) -> Result<EmittedScope> {
        // Collected during the walk and keyed after it: the walk's closure
        // reports a `DataFusionError`, while keying a name asks the dialect and
        // can fail with one of its own.
        let mut qualifiers: Vec<Vec<String>> = Vec::new();
        // Every schema whose column names the emitted `FROM` can answer to,
        // keyed after the walk rather than during it.
        let mut schemas: Vec<SchemaRef> = Vec::new();
        let mut unreadable = false;
        let mut columns_unreadable = false;

        plan.apply(|node| {
            // Asked of every node before it is classified, so there is one
            // definition of what this walk cannot read rather than one per arm.
            match self.unreadable_part(node) {
                // Nothing below can be worth collecting.
                Some(UnreadablePart::Relation) => {
                    unreadable = true;
                    return Ok(TreeNodeRecursion::Stop);
                }
                // The relation is still named where the walk can see it, so the
                // qualifiers below are still worth having — only the column
                // names are lost. Keep going.
                Some(UnreadablePart::ColumnNames) => columns_unreadable = true,
                None => {}
            }
            match node {
                LogicalPlan::TableScan(scan) => {
                    let emitted = self.emitted_qualifier_key(&scan.table_name);
                    // A qualified `FROM` introduces the bare table name as a
                    // correlation name too: `FROM s.t` is addressed as `t` as
                    // readily as `s.t`. Keeping only the full key misses a
                    // correlation qualified by the bare name, which is then read
                    // as reaching outward while the emitted SQL binds it here:
                    // `... FROM s.t WHERE t.c = s.t.c` compares the inner row
                    // with itself. The full key is kept alongside it, so `s1.t`
                    // and `s2.t` are still told apart.
                    //
                    // Matching two components or more only skips a duplicate —
                    // an unqualified name is already its own last component — so
                    // it decides nothing this list is read for.
                    if let [_, .., bare] = emitted.as_slice() {
                        qualifiers.push(vec![bare.clone()]);
                    }
                    qualifiers.push(emitted);
                    // A relation emitted bare answers to every column it has,
                    // not just the projected ones — the same reason the
                    // qualifier is read from the `FROM` rather than from the
                    // output schema.
                    //
                    // Not asked once the column names are already unknowable:
                    // `TableSource::schema` is a trait call some
                    // implementations build a `Schema` in, and the result would
                    // be dropped unread.
                    if !columns_unreadable {
                        schemas.push(scan.source.schema());
                    }
                    Ok(TreeNodeRecursion::Continue)
                }
                LogicalPlan::SubqueryAlias(alias) => {
                    match self.introduces_unreadable(&alias.input)? {
                        Some(UnreadablePart::Relation) => {
                            unreadable = true;
                            return Ok(TreeNodeRecursion::Stop);
                        }
                        Some(UnreadablePart::ColumnNames) => columns_unreadable = true,
                        None => {}
                    }
                    qualifiers
                        .push(vec![self.identifier_comparison_key(alias.alias.table())]);

                    // An alias replaces the *names* below it, which is why the
                    // walk stops here — but it does not replace the *columns*.
                    // The relation it introduces answers to whatever it
                    // exposes, and the emitter writes it either as an aliased
                    // scan (`FROM "t" AS "a"`, exposing every column `t` has)
                    // or as a derived table (exposing the ones it selects).
                    // Which of the two is not decided until the body is built,
                    // so both are taken, in the direction that refuses.
                    //
                    // Jumping without this lost the whole set: `exposed` is
                    // otherwise collected at the plan's root and at its leaf
                    // scans, and an alias is a relation boundary in between.
                    schemas.push(Arc::clone(alias.input.schema().inner()));

                    alias.input.apply(|inner| {
                        if let LogicalPlan::TableScan(scan) = inner {
                            schemas.push(scan.source.schema());
                        }
                        Ok(TreeNodeRecursion::Continue)
                    })?;
                    // The names below are hidden by the alias only while there
                    // is a single relation for it to replace. `FROM "t" AS "derived"`
                    // really does put `"t"` out of reach — the pushdown that
                    // depends on it is pinned by
                    // `test_unparse_left_semi_join_keeps_relation_enclosed_by_build_side_alias`.
                    //
                    // An alias over a *join* is the other shape. SQL has no way
                    // to name a join, so unless the emitter wraps it in a
                    // derived table — which `requires_derived_subquery` declines
                    // to do for inner, outer and cross joins — `relation.alias`
                    // renames the primary relation alone and every relation
                    // joined to it keeps the name it was scanned under, still
                    // addressable beside the alias: `FROM "safe" AS "a" CROSS
                    // JOIN "t"`. Recording only `"a"` there leaves a correlated
                    // `"t"."c"` matching no name in this scope, so the guard
                    // reads a capture as a reference that reaches outward and
                    // emits SQL binding it to the inner `"t"` — valid, silent,
                    // and answering from the wrong rows.
                    //
                    // Which relation the emitter renames is not decided here, so
                    // all of them are taken, including the one that will be
                    // replaced — and so is the join wrapped in a derived table
                    // after all, whose names really are hidden. These are read
                    // only to refuse, so a name too many costs a pushdown and
                    // never a row.
                    if Self::alias_input_holds_a_join(&alias.input) {
                        alias.input.apply(|inner| {
                            match inner {
                                LogicalPlan::TableScan(scan) => {
                                    qualifiers.push(
                                        self.emitted_qualifier_key(&scan.table_name),
                                    );
                                    Ok(TreeNodeRecursion::Continue)
                                }
                                LogicalPlan::SubqueryAlias(inner_alias) => {
                                    qualifiers.push(vec![
                                        self.identifier_comparison_key(
                                            inner_alias.alias.table(),
                                        ),
                                    ]);
                                    // Its own input is hidden behind it, on the
                                    // same terms this arm is deciding.
                                    Ok(TreeNodeRecursion::Jump)
                                }
                                _ => Ok(TreeNodeRecursion::Continue),
                            }
                        })?;
                    }
                    Ok(TreeNodeRecursion::Jump)
                }
                _ => Ok(TreeNodeRecursion::Continue),
            }
        })?;

        // Nothing collected above can be read once the scope is unreadable, so
        // it is not built: the naming below is the expensive half of this walk,
        // and a list that cannot be consulted is a list a later reader can
        // consult by mistake.
        if unreadable {
            return Ok(EmittedScope::Unreadable);
        }

        // The build side's own output names, which no walk above can find: a
        // rename names a column something the plan holds no relation for. Both
        // ways the body is emitted put these names where an unqualified
        // correlation collides with them — as the columns of a derived table
        // when the emitter wraps the body, and as bare names that escape
        // outward, to be compared with the outer query's own column, when it
        // folds the projection into the `SELECT 1` instead.
        //
        // Taken unconditionally, which over-approximates the case where the
        // body folds and the correlation's build half names a column of the
        // scan rather than the renamed one: there the outer reference does bind
        // outward and the refusal costs a pushdown it did not need to. That is
        // the same approximation, and the same fix, as spiceai/spiceai#13469.
        // Skipped entirely when the emitted `FROM` presents column names the
        // plan does not hold — see the field's own doc for why that is `None`
        // and not a partial list.
        let exposed = if columns_unreadable {
            None
        } else {
            let mut exposed = HashSet::new();
            self.expose_columns(&mut exposed, plan.schema().inner())?;
            for schema in &schemas {
                self.expose_columns(&mut exposed, schema)?;
            }
            Some(exposed)
        };
        Ok(EmittedScope::Readable {
            addressable: self.addressable_relations(plan),
            qualifiers,
            exposed,
        })
    }

    /// Whether an alias's input puts more than one relation in the emitted
    /// `FROM`, so the alias cannot replace all of them.
    ///
    /// Asked as "is there a join here" rather than by counting relations,
    /// because a join is the only thing that puts a second one there and the
    /// count is the fragile half: it would have to enumerate every node the
    /// emitter can write as a table factor — a scan, a nested alias, an `UNNEST`
    /// — and a factor left off that list reads as one relation, which is the
    /// answer that declines to collect. A join present is enough to know the
    /// alias renames one side and leaves the other named as it was.
    ///
    /// A nested `SubqueryAlias` ends the descent: whatever it holds is its own
    /// boundary, decided on these same terms when the walk reaches it.
    fn alias_input_holds_a_join(plan: &LogicalPlan) -> bool {
        let mut pending = vec![plan];
        while let Some(node) = pending.pop() {
            match node {
                LogicalPlan::Join(_) => return true,
                LogicalPlan::SubqueryAlias(_) => {}
                _ => pending.extend(node.inputs()),
            }
        }
        false
    }

    fn addressable_relations(&self, plan: &LogicalPlan) -> Vec<Vec<String>> {
        let mut addressable = Vec::new();
        let mut pending = vec![plan];
        // A worklist rather than recursion: this is reached for every scope
        // built, and plan depth is not bounded by anything the unparser owns.
        while let Some(node) = pending.pop() {
            match node {
                LogicalPlan::TableScan(scan) => {
                    addressable.push(scan.table_name.to_vec());
                }
                LogicalPlan::SubqueryAlias(alias) => {
                    addressable.push(vec![alias.alias.table().to_string()]);
                }
                LogicalPlan::Filter(_) | LogicalPlan::Join(_) => {
                    pending.extend(node.inputs());
                }
                _ => {}
            }
        }
        addressable
    }

    /// Whether this node's emitted relation is decided outside the plan, so no
    /// walk over the plan can say what the `FROM` will answer to.
    ///
    /// An extension node is one such shape. Its
    /// [`UserDefinedLogicalNodeUnparser`] is handed the `RelationBuilder` and
    /// decides the emitted `FROM` outright, so a walk is wrong in both
    /// directions at once: it collects the extension's inputs, which the
    /// emitted SQL does not name — `select_to_sql_recursively` returns without
    /// descending once the unparser writes the relation — and misses whatever
    /// the unparser wrote, which it does name.
    ///
    /// An `Unnest` on a dialect that emits Snowflake `LATERAL FLATTEN` keeps
    /// less from the walk, and that difference is the whole of
    /// [`UnreadablePart::ColumnNames`]. The FLATTEN relation presents the columns
    /// Snowflake defines for it — `VALUE` among them — which the plan holds
    /// none of, so no reading of the walk can put them in `exposed`, and an
    /// unqualified correlation on such a name binds to the FLATTEN. Its *name*
    /// is not a mystery at all: the emitter aliases it `_unnest_N`, which
    /// [`Self::is_unparser_derived_alias`] already answers for, and no other
    /// qualifier can reach it. So a qualified reference stays decidable and
    /// keeps its pushdown; only the unqualified ones are refused.
    ///
    /// Enumerating the FLATTEN columns instead would mean writing a vendor's
    /// output schema into this walk, which under-refuses — the wrong-rows
    /// direction — the moment that schema grows.
    ///
    /// BigQuery's `UNNEST(...)` table factor is the same shape and is
    /// deliberately not here: the unparser emits it unaliased and never calls
    /// `with_offset`, and BigQuery gives no way to reference an unaliased
    /// `UNNEST` column, so nothing it presents can be collided with. That is a
    /// judgement about a second dialect rather than a difference in kind, which
    /// is why it is written down instead of left as an asymmetry between two
    /// arms.
    ///
    /// This is the one place either judgement lives, so the refinement they are
    /// waiting for — reading the emitted `FROM` instead of predicting it,
    /// spiceai/spiceai#13469, or a trait method by which an unparser declares
    /// the scope it will emit — replaces one function rather than several match
    /// arms. That both shapes here are relations the emitter knows how to write
    /// and this walk has to be told about separately is the argument in
    /// spiceai/spiceai#13480.
    ///
    /// [`UserDefinedLogicalNodeUnparser`]: crate::unparser::extension_unparser::UserDefinedLogicalNodeUnparser
    fn unreadable_part(&self, node: &LogicalPlan) -> Option<UnreadablePart> {
        match node {
            LogicalPlan::Extension(_) => Some(UnreadablePart::Relation),
            // Wider than the emitter's own gates, which also require an
            // unnest input type and an unset relation: this asks only the
            // dialect, so it answers for every `Unnest` on one. Over-refusing
            // is the safe direction, and predicting those gates from here is
            // the mistake the rest of this walk keeps paying for.
            LogicalPlan::Unnest(_) if self.dialect.unnest_as_lateral_flatten() => {
                Some(UnreadablePart::ColumnNames)
            }
            _ => None,
        }
    }

    /// The strongest [`UnreadablePart`] any node below `plan` reports.
    ///
    /// Asked of a `SubqueryAlias`'s input, because an alias contains neither
    /// kind. The alias reaches the single relation the `RelationBuilder` holds,
    /// while the same unparser is handed the `SelectBuilder` and can join a
    /// second relation onto the same `FROM` — and that one keeps its own name
    /// and its own columns. So the walk looks past an alias for this and for
    /// nothing else: the relations an alias really does shield are why it stops
    /// there at all, and collecting them would refuse references that bind
    /// correctly.
    fn introduces_unreadable(
        &self,
        plan: &LogicalPlan,
    ) -> Result<Option<UnreadablePart>> {
        let mut found = None;
        plan.apply(|node| {
            match self.unreadable_part(node) {
                // Nothing outranks it, so there is no reason to keep looking.
                Some(UnreadablePart::Relation) => {
                    found = Some(UnreadablePart::Relation);
                    return Ok(TreeNodeRecursion::Stop);
                }
                Some(UnreadablePart::ColumnNames) => {
                    found = Some(UnreadablePart::ColumnNames)
                }
                None => {}
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(found)
    }

    /// Adds every name `schema`'s columns let the body answer to.
    ///
    /// Each column is exposed under two names: its own, which a relation emitted
    /// bare answers to, and the one this dialect would rewrite it to, which is
    /// what a derived table carries it under. Which of the two the body will be
    /// is not decided here — that is the approximation tracked by
    /// spiceai/spiceai#13469 — so both are taken, in the direction that refuses
    /// rather than emits.
    fn expose_columns(
        &self,
        exposed: &mut HashSet<String>,
        schema: &SchemaRef,
    ) -> Result<()> {
        for field in schema.fields() {
            exposed.insert(self.identifier_comparison_key(field.name()));
            exposed.insert(self.emitted_column_key(field.name())?);
        }
        Ok(())
    }

    /// Whether any column in `expr` emits a reference the `EXISTS` body captures.
    ///
    /// Both column variants are walked. [`Expr::column_refs`] collects only
    /// [`Expr::Column`], but [`Self::col_to_sql`] renders
    /// [`Expr::OuterReferenceColumn`] through that same path, so the two are
    /// emitted as the same qualified identifier and bind by the same rules. A
    /// guard that reasons about where an emitted reference resolves has to see
    /// both, or the variant it does not walk goes on being captured.
    ///
    /// What separates them is *attribution*, which is why `probe_scope` is
    /// asked for one and not the other:
    ///
    /// * An [`Expr::Column`] in a `join.filter` carries no split by side, but it
    ///   can only mean one of the join's own two inputs — so a name or qualifier
    ///   a single input owns is attributable after all and binds where it was
    ///   meant to. Only what *both* answer to is ambiguous, which is what asking
    ///   the probe scope as well says. For the `on` pairs the split is already
    ///   made by the caller, so there is no probe scope to ask.
    /// * An [`Expr::OuterReferenceColumn`] names neither input by construction:
    ///   it reaches past this join to an enclosing query. Nothing about the
    ///   probe can make it a local reference, so the two scopes stop excusing
    ///   each other and start adding up — the reference passes *through* the
    ///   probe's scope on its way out, so either scope answering to it is a
    ///   capture. Requiring both hides one; requiring only the build hides the
    ///   other.
    fn references_captured_scope(
        &self,
        expr: &Expr,
        kind: ReferenceKind,
        build_scope: &EmittedScope,
        probe_scope: Option<&EmittedScope>,
    ) -> Result<bool> {
        expr.exists(|node| {
            // A subquery is a leaf here, so the outer references it carries have
            // to be reached for explicitly — and at every depth, since the list
            // one subquery holds leaves out the references a subquery nested
            // inside it holds.
            if let Some(subquery) = Self::subquery_of(node) {
                return self.subquery_reaches_captured_scope(
                    subquery,
                    &[],
                    build_scope,
                    probe_scope,
                );
            }

            let (column, reaches_past_the_join) = match node {
                Expr::Column(column) => (column, false),
                Expr::OuterReferenceColumn(_, column) => (column, true),
                _ => return Ok(false),
            };

            // Asked before anything else, including the invented-alias answer
            // below: a caller that narrowed to the references reaching past the
            // join has already asked about the rest under the other rule, and
            // answering for them here would apply this one twice.
            if matches!(kind, ReferenceKind::ReachingPastTheJoin)
                && !reaches_past_the_join
            {
                return Ok(false);
            }

            // An alias the unparser invents is in the emitted `FROM` whatever
            // the plan holds, so no scope can claim it. Asked of the keyed
            // spelling, since a relation a user named `DERIVED_PROJECTION` is
            // the alias `derived_projection` once a dialect that folds case has
            // emitted them both.
            let names_invented_alias = match column.relation.as_ref() {
                Some(relation) => {
                    let emitted = self.emitted_qualifier(relation);
                    Self::is_unparser_derived_alias(&self.qualifier_key(&emitted))
                }
                None => false,
            };

            // Counted as the build side answering rather than answered on its
            // own, so the attribution below still applies. Returning early here
            // would refuse a *build-local* reference to a relation the user
            // happened to name `derived_limit`, which binds inside the body on
            // purpose — the same reference the side split exists to allow.
            let captured_by_build =
                names_invented_alias || self.scope_answers(build_scope, column)?;
            Ok(match probe_scope {
                // An outer reference passes *through* the probe's scope on its
                // way out, so the probe shadowing it is as much a capture as the
                // build shadowing it. Neither can be excused by the other.
                Some(probe) if reaches_past_the_join => {
                    captured_by_build || self.scope_answers(probe, column)?
                }
                // A plain column can only mean one of the join's own two inputs,
                // so a name or qualifier a single input owns is attributable
                // after all and binds where it was meant to. Only what both
                // answer to is ambiguous.
                Some(probe) => captured_by_build && self.scope_answers(probe, column)?,
                // The caller already split this reference by side, so there is
                // no other scope to attribute it to.
                None => captured_by_build,
            })
        })
    }

    /// The [`Subquery`] a subquery-bearing expression holds.
    ///
    /// A subquery's plan is out of reach of `Expr` traversal, in one of two
    /// ways: `Expr::apply_children` reports `Exists` and `ScalarSubquery` as
    /// leaves outright, and for `InSubquery` and `SetComparison` it descends
    /// into the compared expression only. Either way a walk over an expression
    /// never reaches the plan inside one, and so never sees the outer
    /// references that plan carries. Those are held separately, on
    /// [`Subquery::outer_ref_columns`], and have to be asked for by name.
    ///
    /// Every subquery-bearing variant is answered for. Leaving one out is the
    /// wrong-rows direction — the walk reports no capture for a reference it
    /// never looked at — and the compiler cannot point at the omission, since
    /// the catch-all arm this needs for the non-subquery variants absorbs it.
    const fn subquery_of(expr: &Expr) -> Option<&Subquery> {
        match expr {
            Expr::Exists(exists) => Some(&exists.subquery),
            Expr::ScalarSubquery(subquery) => Some(subquery),
            Expr::InSubquery(in_subquery) => Some(&in_subquery.subquery),
            Expr::SetComparison(set_comparison) => Some(&set_comparison.subquery),
            _ => None,
        }
    }

    /// Whether `expr` carries a reference that reaches past this join, counting
    /// the ones held by a nested subquery.
    ///
    /// [`Expr::contains_outer`] cannot see those, for the reason
    /// [`Self::subquery_of`] gives, so asking it alone would leave the probe
    /// scope unbuilt and the reference unexamined.
    fn carries_outer_reference(expr: &Expr) -> Result<bool> {
        Ok(expr.contains_outer()
            || expr.exists(|node| Ok(Self::subquery_of(node).is_some()))?)
    }

    /// Whether an outer reference `subquery` carries at any depth is captured by
    /// one of the scopes this join emits.
    ///
    /// `enclosing` holds the bodies between `subquery` and this join, innermost
    /// first; it is empty for a subquery written directly into the predicate,
    /// which is what makes that case identical to asking about its own list
    /// alone. A reference one of those bodies resolves — bound there or captured
    /// there — never reaches this join, and [`Self::outward_reference`] says
    /// which. Only what survives the whole chain reaches the join, and that
    /// follows the same rule as an `OuterReferenceColumn` written directly into
    /// the predicate: whichever scope the join emits shadows one captures it.
    /// Recursion depth here is the plan's subquery nesting depth, which no part of
    /// the unparser bounds, so it grows the stack the same way
    /// `select_to_sql_recursively` does.
    #[cfg_attr(feature = "recursive_protection", recursive::recursive)]
    fn subquery_reaches_captured_scope(
        &self,
        subquery: &Subquery,
        enclosing: &[&EmittedScope],
        build_scope: &EmittedScope,
        probe_scope: Option<&EmittedScope>,
    ) -> Result<bool> {
        for outer in &subquery.outer_ref_columns {
            match self.outward_reference(outer, enclosing)? {
                OutwardReference::BindsToAnEnclosingBody => continue,
                OutwardReference::CapturedByAnEnclosingBody => return Ok(true),
                OutwardReference::ReachesTheJoin => {}
            }
            if self.references_captured_scope(
                outer,
                ReferenceKind::ReachingPastTheJoin,
                build_scope,
                probe_scope,
            )? {
                return Ok(true);
            }
        }

        self.nested_subqueries_reach_captured_scope(
            &subquery.subquery,
            enclosing,
            build_scope,
            probe_scope,
        )
    }

    /// [`Self::subquery_reaches_captured_scope`] for the subqueries `plan`'s own
    /// expressions hold — the ones its [`Subquery::outer_ref_columns`] leaves out.
    ///
    /// Every site that builds that list uses [`LogicalPlan::all_out_ref_exprs`],
    /// which collects over `apply_expressions` and `inputs()`. `inputs()`
    /// documents itself as not including subqueries, and `Expr` traversal reports
    /// a subquery-bearing expression as a leaf (see [`Self::subquery_of`]), so a
    /// reference held two levels down appears on no list at all:
    /// `EXISTS (S1 WHERE EXISTS (S2 WHERE outer_ref(p.c)))` leaves S1's list
    /// empty and `p.c` unexamined, and an emitted `FROM` that shadows `p` then
    /// captures it. Reaching it needs this descent.
    ///
    /// Each level contributes one scope to the chain — the body's own, built once
    /// here rather than per node holding a subquery. A body whose emitted form is
    /// a single `SELECT` has one `FROM`, so every correlation inside it is emitted
    /// against the same relations; a set-operation body is emitted as one `SELECT`
    /// per branch, and [`Self::set_operation_branches`] scopes those separately.
    fn nested_subqueries_reach_captured_scope(
        &self,
        plan: &LogicalPlan,
        enclosing: &[&EmittedScope],
        build_scope: &EmittedScope,
        probe_scope: Option<&EmittedScope>,
    ) -> Result<bool> {
        // Asked before the scope below is built, which walks the plan again and
        // collects two sets of names: most subqueries hold no further subquery,
        // and for those there is nothing the scope would be asked about.
        if !Self::holds_subquery(plan)? {
            return Ok(false);
        }
        // Each branch is emitted as its own `SELECT` with its own `FROM`, so a
        // relation named in one branch never answers a correlation held in
        // another. Scoping the branches together would read such a pairing as a
        // capture and refuse a query the emitter renders correctly.
        if let Some(branches) = Self::set_operation_branches(plan) {
            for branch in branches {
                if self.nested_subqueries_reach_captured_scope(
                    branch,
                    enclosing,
                    build_scope,
                    probe_scope,
                )? {
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        let scope = self.emitted_scope(plan)?;
        let mut chain = Vec::with_capacity(enclosing.len() + 1);
        chain.push(&scope);
        chain.extend_from_slice(enclosing);

        let mut captured = false;
        plan.apply(|node| {
            node.apply_expressions(|expr| {
                expr.apply(|node| {
                    let Some(nested) = Self::subquery_of(node) else {
                        return Ok(TreeNodeRecursion::Continue);
                    };
                    if self.subquery_reaches_captured_scope(
                        nested,
                        &chain,
                        build_scope,
                        probe_scope,
                    )? {
                        captured = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
            })
        })?;
        Ok(captured)
    }

    /// The bodies a set-operation `plan` is emitted as, or `None` when `plan` is
    /// emitted as a single `SELECT`.
    ///
    /// Mirrors what `select_to_sql_recursively` writes: its [`LogicalPlan::Union`]
    /// arm emits every input as its own `SetExpr`, and its [`Distinct::All`] arm
    /// delegates a distinct union straight to that same arm, so both shapes reach
    /// the reader as one `SELECT` per branch.
    ///
    /// Only a set operation standing as the body itself is reported. Below a
    /// [`LogicalPlan::SubqueryAlias`] there is nothing to report:
    /// `requires_derived_subquery` sends a union there to a derived table, and the
    /// name a correlation can reach is that alias, which [`Self::emitted_scope`]
    /// records as it stops. A union buried under any other node is left to the
    /// single-scope path, which reads its branches together — an
    /// over-approximation that costs a refusal rather than a wrong emission, and
    /// which splitting here would not fix, since the derived table's own alias is
    /// the name that path is missing.
    fn set_operation_branches(plan: &LogicalPlan) -> Option<&[Arc<LogicalPlan>]> {
        match plan {
            LogicalPlan::Union(union) => Some(&union.inputs),
            LogicalPlan::Distinct(Distinct::All(input)) => match input.as_ref() {
                LogicalPlan::Union(union) => Some(&union.inputs),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether any expression anywhere in `plan` carries a subquery.
    fn holds_subquery(plan: &LogicalPlan) -> Result<bool> {
        let mut found = false;
        plan.apply(|node| {
            node.apply_expressions(|expr| {
                found =
                    found || expr.exists(|node| Ok(Self::subquery_of(node).is_some()))?;
                Ok(if found {
                    TreeNodeRecursion::Stop
                } else {
                    TreeNodeRecursion::Continue
                })
            })
        })?;
        Ok(found)
    }

    /// What becomes of the reference `outer` emits on its way out to this join,
    /// given the bodies `enclosing` between the two, innermost first.
    ///
    /// A nested subquery's outer references are relative to the scope enclosing
    /// *it*, which is usually the body one level out rather than anything past
    /// this join, so the scopes in between have to be asked before the join's own
    /// are. In
    /// `EXISTS (SELECT 1 FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.a = t1.b))`
    /// the inner list holds `t1.b`, which binds against `FROM t1` and reaches
    /// nothing this join emits: taking every nested reference straight to the
    /// join's scopes would refuse it, and a refusal is a hard error rather than a
    /// different emission, so that trades wrong SQL for a wrong error.
    ///
    /// SQL resolves the reference at the first scope its emitted qualifier
    /// collides with, so a colliding body ends the journey either way; what the
    /// two answers separate is whether that body is the relation the plan named.
    /// Collapsing them into "a colliding body consumes it" is the wrong-rows
    /// direction, and not hypothetically: on a dialect that does not spell
    /// columns in full, a body selecting from `mid.t` collides with a reference
    /// to `public.t`, and reading that as arrival drops the reference unexamined
    /// and emits the capture.
    ///
    /// The two questions consult different halves of a scope, because the two
    /// errors they can make are not equally bad: arrival is asked of the
    /// relations the emitted `FROM` can address, collision of every relation the
    /// plan mentions. [`Self::addressable_relations`] is where that asymmetry is
    /// stated.
    fn outward_reference(
        &self,
        outer: &Expr,
        enclosing: &[&EmittedScope],
    ) -> Result<OutwardReference> {
        // Only a reference this can place is placed. Anything else is left to
        // `references_captured_scope`, which walks the expression itself.
        let Expr::OuterReferenceColumn(_, column) = outer else {
            return Ok(OutwardReference::ReachesTheJoin);
        };
        for scope in enclosing {
            if self.scope_names_relation(scope, column) {
                return Ok(OutwardReference::BindsToAnEnclosingBody);
            }
            if self.scope_answers(scope, column)? {
                return Ok(OutwardReference::CapturedByAnEnclosingBody);
            }
        }
        Ok(OutwardReference::ReachesTheJoin)
    }

    /// Whether `scope` names the very relation `column` names, *and* the emitted
    /// `FROM` can address it — so the reference resolves there, as the plan
    /// intends, rather than merely colliding with something spelled alike.
    ///
    /// Asked of [`Self::addressable_relations`] and not of the scope's full list,
    /// for the reason that function gives: this is the answer that ends the
    /// enquiry into a reference, so a relation it wrongly credits is a reference
    /// dropped unexamined.
    ///
    /// Compared exactly, and it is the one comparison in this guard that is —
    /// everywhere else identifiers are folded through
    /// [`Self::identifier_comparison_key`], which errs toward calling two of
    /// them the same. That direction refuses, and refusing is what the rest of
    /// the guard wants; here it *permits*, so the same fold would end the
    /// enquiry into a reference the emitted SQL never binds where this claims.
    /// On PostgreSQL a body scanning `"t"` does not resolve a reference to
    /// `"T"`, yet folded they are one relation, and the scopes past this one —
    /// which may genuinely capture it — would never be asked. An exact
    /// comparison declines to credit the arrival instead, and the reference goes
    /// on to [`Self::scope_answers`], whose folded reading refuses it. The cost
    /// runs the other way, on a dialect that really does bind them alike: a
    /// pushdown, not a row. Asking the dialect rather than guessing is
    /// spiceai/spiceai#13474.
    ///
    /// An unqualified reference names no relation, so nothing can be said to be
    /// it: such a reference binds to whichever relation in the colliding scope
    /// exposes the column, and which one the plan meant is not recoverable —
    /// least of all from `exposed`, which is deliberately over-collected in the
    /// refusing direction and so cannot be read as an arrival. That is answered
    /// `false`, the same stance [`Self::scope_answers`] takes on an unqualified
    /// reference against the join's own scopes.
    fn scope_names_relation(&self, scope: &EmittedScope, column: &Column) -> bool {
        let EmittedScope::Readable { addressable, .. } = scope else {
            return false;
        };
        let Some(relation) = column.relation.as_ref() else {
            return false;
        };
        addressable.contains(&relation.to_vec())
    }

    /// Whether `scope` answers to the reference `column` emits.
    ///
    /// Both halves are asked in the emitted, keyed form — the qualifier through
    /// [`Self::emitted_qualifier_key`] and the name through
    /// [`Self::emitted_column_key`] — against a scope that already holds only
    /// that form.
    ///
    /// A scope whose names are not knowable — an [`EmittedScope::Unreadable`]
    /// one, or a `Readable` one whose column list is `None` — answers to
    /// everything, so neither half is asked, and neither exists to ask: a
    /// reference cannot be cleared against a name nobody knows. Both callers need
    /// that reading, for opposite-looking reasons that come to the same thing —
    /// see [`OutwardReference::CapturedByAnEnclosingBody`].
    fn scope_answers(&self, scope: &EmittedScope, column: &Column) -> Result<bool> {
        let EmittedScope::Readable {
            qualifiers,
            exposed,
            ..
        } = scope
        else {
            return Ok(true);
        };
        Ok(match column.relation.as_ref() {
            Some(relation) => qualifiers.contains(&self.emitted_qualifier_key(relation)),
            // An unqualified reference names no relation to disagree with, so
            // it binds to whichever relation in the innermost scope exposes the
            // column — the `EXISTS` body's own, whenever that body exposes the
            // name at all. The names missing from a list that is not knowable
            // are exactly the ones the plan never held, so nothing about such a
            // reference can be settled by looking.
            None => match exposed {
                Some(exposed) => {
                    exposed.contains(&self.emitted_column_key(&column.name)?)
                }
                None => true,
            },
        })
    }

    /// Refuses a correlation that the `EXISTS` body's own `FROM` would capture.
    ///
    /// The correlated predicates are emitted inside the subquery, so every
    /// reference in them binds to the innermost scope answering to its qualifier.
    /// A reference meant for the outer query whose qualifier the build side also
    /// answers to therefore binds to the subquery's own relation instead: the
    /// correlation becomes a comparison of an inner row with itself, `EXISTS`
    /// degenerates to "this relation has a row", and a semi or mark join then
    /// keeps every probe row while an anti join drops every one. That is valid
    /// SQL, so a database runs it and returns those wrong rows.
    ///
    /// SQL's scoping rules decide it on their own, so the capture does not need a
    /// row bound to happen. Refusing costs the pushdown, which is the trade
    /// [`Self::exists_scope_name`] already makes for output that binds and runs
    /// rather than failing.
    ///
    /// Every part of this compares *what the statement will say*, not what the
    /// plan holds, and each one costs correctness one way and pushdown the other:
    ///
    /// * **Ask how the qualifier will be spelled**, via
    ///   [`Self::emitted_qualifier`], rather than comparing the
    ///   [`TableReference`]. A dialect that elides the prefix emits `s1.t` and
    ///   `s2.t` alike as `t`, so distinct references collide; a dialect that
    ///   spells columns in full keeps them apart, and refusing there would cost
    ///   the pushdown on SQL that binds correctly. Neither is inferable from the
    ///   reference alone.
    /// * **Ask how the column name will be spelled**, via
    ///   [`Self::emitted_column_name`]. A dialect may rewrite a name it cannot
    ///   spell — BigQuery writes `min(a)` as `min_40a_41` — and a build relation
    ///   with a column of the rewritten name then captures the reference while
    ///   the two plan names look nothing alike.
    /// * **Compare identifiers the way the engine will resolve them**, via
    ///   [`Self::identifier_comparison_key`], which folds case
    ///   *unconditionally*. An identifier emitted unquoted is case-folded
    ///   before it binds, so `T` and `t` written bare are one name. Keying that
    ///   on the quote style instead would be false in both directions — DuckDB
    ///   folds even a quoted identifier, and BigQuery folds a column name
    ///   despite the backticks — so the fold does not ask. The price is a
    ///   PostgreSQL build side whose `"T"` and `"t"` really are two relations:
    ///   that correlation binds correctly and is refused anyway, which
    ///   `refuses_case_distinct_quoted_relations` pins and spiceai/spiceai#13474
    ///   tracks. Do not reintroduce the quote-style heuristic to buy it back.
    /// * **Attribute an `Expr::Column` by side, and an
    ///   `Expr::OuterReferenceColumn` not at all.** The two follow opposite
    ///   rules, and collapsing either into the other is a defect in one
    ///   direction or the other:
    ///   * An `Expr::Column` belongs to one of the join's own inputs. In an
    ///     `on` pair the split says which: only the probe half is the
    ///     correlated reference, and a build half naming a build relation is an
    ///     ordinary local reference that binds inside on purpose — testing both
    ///     halves refuses every join whose build side shares a qualifier with
    ///     its probe, including the correct ones, which
    ///     `keeps_build_side_key_on_shared_relation` pins. `join.filter` has no
    ///     such split, so there a qualifier *both* sides answer to is what is
    ///     refused, which no reference to a distinct relation can be.
    ///   * An `Expr::OuterReferenceColumn` belongs to neither input — it is on
    ///     its way out of both. So the half it was written on says nothing, both
    ///     halves of every pair are asked, and *either* scope shadowing it is a
    ///     capture. Asking with the `Expr::Column` rule instead would let a
    ///     reference through whenever one side alone shadows it.
    ///
    ///   [`Self::references_captured_scope`] holds both rules; the caller
    ///   selects which references to ask about with [`ReferenceKind`].
    /// * **Ask what the body exposes, not only what it is called.** An
    ///   unqualified reference names no relation to collide with, so it is a
    ///   column name that decides where it binds. Comparing qualifiers alone
    ///   lets two sides that share no relation at all emit a correlation the
    ///   body captures.
    ///
    /// One gap is left to the qualifier rewrite tracked by spiceai/spiceai#12840:
    /// a relation reached through a `SubqueryAlias`, whose own name the emitted
    /// SQL replaces with the alias. The alias is what this compares, which is
    /// right for a reference written against it, but a correlated reference
    /// still qualified by the enclosed relation is a shape the rewrite has to
    /// requalify rather than refuse.
    ///
    /// It over-refuses in the other direction as well, tracked by
    /// spiceai/spiceai#13469: the walk collects relations the emitter will bury
    /// inside a derived table of its own, which the emitted SQL cannot address.
    /// That costs the pushdown on correct SQL, which is the safe way round for
    /// a guard whose other failure is wrong rows.
    ///
    /// A build or probe side whose emitted `FROM` the plan does not describe —
    /// [`Self::unreadable_part`] — is refused wholesale on the same
    /// trade. This runs before any of the body is built, so there is no emitted
    /// `FROM` to read instead; narrowing it is the trait change noted there.
    fn ensure_exists_correlation_not_shadowed(&self, join: &Join) -> Result<()> {
        let swapped = Self::swaps_join_inputs(join.join_type);
        let (probe_plan, build_plan) = if swapped {
            (&join.right, &join.left)
        } else {
            (&join.left, &join.right)
        };

        let build_scope = self.emitted_scope(build_plan)?;
        let mut captured = false;

        // The probe half of each pair, and every reference in it: that half is
        // the correlated one, so the build side answering to it is enough.
        for correlated in join
            .on
            .iter()
            .map(|(left, right)| if swapped { right } else { left })
        {
            if self.references_captured_scope(
                correlated,
                ReferenceKind::Every,
                &build_scope,
                None,
            )? {
                captured = true;
                break;
            }
        }

        // What the probe's own scope has to be asked about as well. Two sources,
        // for the same reason: a reference that cannot be attributed to one side
        // by where it was written.
        //
        // * `join.filter` is not split into halves at all, so an
        //   `Expr::Column` in it could have come from either input.
        // * Both halves of every `on` pair, for the references that reach past
        //   the join — the rule for those is on `ReferenceKind`. One gets into a
        //   pair at all whenever the expression also carries a local column,
        //   because `find_valid_equijoin_key_pair` decides ownership from
        //   `Expr::column_refs`, which collects only `Expr::Column`.
        //
        // Collected first so the probe scope is named only when something will
        // ask it. Most joins carry no filter and no outer reference in a key,
        // and naming a scope walks every column the side exposes.
        if !captured {
            let mut needs_probe_scope: Vec<(&Expr, ReferenceKind)> = Vec::new();
            if let Some(filter) = &join.filter {
                needs_probe_scope.push((filter, ReferenceKind::Every));
            }
            for half in join.on.iter().flat_map(|(left, right)| [left, right]) {
                if Self::carries_outer_reference(half)? {
                    needs_probe_scope.push((half, ReferenceKind::ReachingPastTheJoin));
                }
            }
            if !needs_probe_scope.is_empty() {
                let probe_scope = self.emitted_scope(probe_plan)?;
                for (expr, kind) in needs_probe_scope {
                    if self.references_captured_scope(
                        expr,
                        kind,
                        &build_scope,
                        Some(&probe_scope),
                    )? {
                        captured = true;
                        break;
                    }
                }
            }
        }

        if captured {
            return not_impl_err!(
                "Unparsing an EXISTS-style join is not supported when a FROM the emitted SQL introduces would capture the correlation: it answers to the correlated reference's relation qualifier, or exposes its column name when the reference carries none, or is a relation this unparser cannot read at all, so the reference binds there instead of in the query it was written against"
            );
        }
        Ok(())
    }

    /// The name a scope around the `EXISTS` build side has to answer to, so the
    /// correlated predicates still resolve against it.
    ///
    /// The name has to come from the predicates rather than from the build
    /// side's schema: a set operation's output carries no qualifier while the
    /// join keys naming it still do, so the schema would have this scope
    /// answer to a name nothing references.
    ///
    /// * One relation — the scope takes its name and every reference resolves.
    /// * No qualifier at all — the correlation is by unqualified columns, which
    ///   resolve against the only relation in scope whatever it is called. It
    ///   still needs *a* name, since most dialects reject an unaliased derived
    ///   table.
    ///
    /// Every other shape is refused, because a scope this function cannot name
    /// is a bound this unparser cannot place — and leaving the bound beside the
    /// correlation emits SQL that *binds and runs* while answering from rows
    /// outside the bound. A wrong answer that executes is worse than one that
    /// does not, so these cost the pushdown instead — the trade
    /// `derive_row_limited_scope` makes for the limit it scopes.
    ///
    /// Refusing is not the same as repairing. Each shape below is emitted
    /// correctly only once the correlation's own qualifiers are rewritten to the
    /// scope the derived table introduces, which is a pass over the correlation
    /// expressions rather than a choice of name — tracked by spiceai/spiceai#12840.
    /// Whoever implements it should expect these refusals to become scopes.
    fn exists_scope_name(&self, join: &Join) -> Result<String> {
        // Which input is correlated against, and therefore which half of each
        // `on` pair names the side being scoped, follows the same swap the join
        // arm applies before it builds this subquery.
        let swapped = Self::swaps_join_inputs(join.join_type);
        let probe_plan = if swapped { &join.right } else { &join.left };

        // Match on the probe side's *qualifiers* rather than its columns: a
        // correlated reference can name a probe-side column the probe's own
        // projection dropped, and testing for the column would then read that
        // reference as a build-side name.
        let probe_relations: Vec<&TableReference> = probe_plan
            .schema()
            .iter()
            .filter_map(|(relation, _)| relation)
            .collect();
        let mut relations: Vec<TableReference> = Vec::new();
        let mut names_a_probe_relation = false;
        let mut collect = |expr: &Expr| {
            for column in expr.column_refs() {
                let Some(relation) = &column.relation else {
                    continue;
                };
                if probe_relations.contains(&relation) {
                    names_a_probe_relation = true;
                } else if !relations.contains(relation) {
                    relations.push(relation.clone());
                }
            }
        };
        for (left, right) in &join.on {
            collect(if swapped { left } else { right });
        }
        if let Some(filter) = &join.filter {
            collect(filter);
        }

        match relations.as_slice() {
            // Nothing to preserve, so any name will do — but only once the
            // correlation has been shown to name nothing.
            [] if !names_a_probe_relation => Ok(DERIVED_LIMIT_ALIAS.to_string()),
            // A qualifier the probe side also answers to can be a build-side
            // reference on a self-join, where naming the scope anything else
            // rebinds it to the probe.
            //
            // Where the build side answers to that qualifier too — the self-join
            // proper — `ensure_exists_correlation_not_shadowed` has already
            // refused the plan, bound or not. What reaches here is the residue:
            // a correlation qualified only by a probe relation the build side
            // does not share, so nothing is captured, but there is also no
            // build-side name for the scope to keep bound. Renaming it would
            // rebind the reference to the probe, so this refuses instead.
            [] => {
                not_impl_err!(
                    "Unparsing a row bound on an EXISTS-style join's build side is not supported when the correlation's only qualifier is one the probe side also answers to"
                )
            }
            // A derived table's alias is a single identifier, so only the last
            // component of a qualified name survives it, while a dialect that
            // spells columns in full still writes every component in the
            // correlated predicate — leaving it qualified by a relation that is
            // no longer in scope.
            [relation]
                if self.dialect.full_qualified_col() && relation.to_vec().len() > 1 =>
            {
                not_impl_err!(
                    "Unparsing a row bound on an EXISTS-style join's build side is not supported for a qualified table name on a dialect that spells columns in full"
                )
            }
            [relation] => Ok(relation.table().to_string()),
            // The build side is itself a join and the correlation names more
            // than one of its inputs. A single scope can expose only one of
            // those names, so there is no name that keeps every reference
            // bound to the relation it came from.
            _ => {
                not_impl_err!(
                    "Unparsing a row bound on an EXISTS-style join's build side is not supported when the correlation names more than one of the build side's inputs"
                )
            }
        }
    }

    /// AND-folds `predicate` into a join's `ON` clause.
    ///
    /// Returns the constraint together with the predicate it could not take:
    /// `USING` and `NATURAL` joins have nowhere to put one.
    fn and_into_join_constraint(
        constraint: ast::JoinConstraint,
        predicate: Option<ast::Expr>,
    ) -> (ast::JoinConstraint, Option<ast::Expr>) {
        let Some(predicate) = predicate else {
            return (constraint, None);
        };

        match constraint {
            ast::JoinConstraint::On(existing) => (
                ast::JoinConstraint::On(ast::Expr::BinaryOp {
                    left: Box::new(existing),
                    op: ast::BinaryOperator::And,
                    right: Box::new(predicate),
                }),
                None,
            ),
            ast::JoinConstraint::None => (ast::JoinConstraint::On(predicate), None),
            constraint @ (ast::JoinConstraint::Using(_)
            | ast::JoinConstraint::Natural) => (constraint, Some(predicate)),
        }
    }

    /// Decides where the table-scan filters extracted from a join's two inputs
    /// belong in the unparsed SQL: in the `JOIN ON` clause or in `WHERE`.
    ///
    /// The answer depends on the join type *and* on which input the filter came
    /// from — a filter on a preserved side means something different in `ON`
    /// than in `WHERE`. Filters routed to `ON` are AND-folded onto the join's
    /// own filter.
    ///
    /// Returns `(on_filter, where_filters)`.
    fn split_join_on_and_where_filters(
        join_type: JoinType,
        join_filter: &Option<Expr>,
        left_scan_filters: Vec<Expr>,
        right_scan_filters: Vec<Expr>,
    ) -> (Option<Expr>, Vec<Expr>) {
        // Which clause preserves a filter's meaning depends on the side it came
        // from:
        //
        // * Inner — ON and WHERE are equivalent, so both sides go to WHERE
        //   (some dialects reject subqueries inside JOIN ON).
        // * Left/Right — the preserved side keeps every row whatever ON says,
        //   so folding its filter into ON would filter nothing at all; it has
        //   to go to WHERE. The non-preserved side is the mirror image: WHERE
        //   would discard the null-extended rows and silently turn the outer
        //   join into an inner join, so its filter has to stay in ON.
        // * Full — both sides are preserved, so neither clause preserves either
        //   side's filter; only a derived table does. The caller isolates a
        //   `FULL JOIN`'s filtered side in one before reaching this function,
        //   so `left_scan_filters`/`right_scan_filters` are always empty here.
        // A subquery inside `ON` is refused outright by some dialects. Where `ON`
        // and `WHERE` are equivalent the conjunct carrying it can move; where they
        // are not — an outer join preserves rows that `WHERE` would then discard —
        // it has to stay, and the dialect's own limit applies.
        let (join_filter, subquery_filters) = match join_type {
            JoinType::Inner => {
                let (kept, moved) = partition_subquery_filters(
                    join_filter
                        .iter()
                        .flat_map(split_conjunction)
                        .cloned()
                        .collect(),
                );
                let kept = kept.into_iter().reduce(|acc, filter| {
                    Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(acc),
                        op: Operator::And,
                        right: Box::new(filter),
                    })
                });
                (kept, moved)
            }
            _ => (join_filter.clone(), vec![]),
        };
        let join_filter = &join_filter;

        let (on_scan_filters, where_scan_filters) = match join_type {
            JoinType::Inner => (
                vec![],
                left_scan_filters
                    .into_iter()
                    .chain(right_scan_filters)
                    .collect(),
            ),
            JoinType::Left => (right_scan_filters, left_scan_filters),
            JoinType::Right => (left_scan_filters, right_scan_filters),
            // Semi/anti/mark joins do not reach this function: their filters
            // are routed by the EXISTS branch of `select_to_sql_recursively`.
            JoinType::Full
            | JoinType::LeftSemi
            | JoinType::RightSemi
            | JoinType::LeftAnti
            | JoinType::RightAnti
            | JoinType::LeftMark
            | JoinType::RightMark => (
                left_scan_filters
                    .into_iter()
                    .chain(right_scan_filters)
                    .collect(),
                vec![],
            ),
        };

        let where_scan_filters: Vec<Expr> = where_scan_filters
            .into_iter()
            .chain(subquery_filters)
            .collect();

        if on_scan_filters.is_empty() {
            return (join_filter.clone(), where_scan_filters);
        }

        let combined = on_scan_filters.into_iter().reduce(|acc, filter| {
            Expr::BinaryExpr(BinaryExpr {
                left: Box::new(acc),
                op: Operator::And,
                right: Box::new(filter),
            })
        });

        let on_filter = match (join_filter, combined) {
            (Some(jf), Some(c)) => Some(Expr::BinaryExpr(BinaryExpr {
                left: Box::new(jf.clone()),
                op: Operator::And,
                right: Box::new(c),
            })),
            (Some(jf), None) => Some(jf.clone()),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };

        (on_filter, where_scan_filters)
    }
}

impl From<BuilderError> for DataFusionError {
    fn from(e: BuilderError) -> Self {
        DataFusionError::External(Box::new(e))
    }
}

/// What an `EXISTS` body's emitted `FROM` will answer to.
///
/// A reference inside the body binds to the innermost scope that answers to it,
/// and it can be addressed two ways, so both are collected:
///
/// * `qualifiers` — the relation names the `FROM` introduces, spelled the way
///   this dialect will spell a column's qualifier.
/// * `exposed` — the column names the body can answer to, which is more than
///   the emitted relations expose: a rename names a column something no relation
///   has. An unqualified reference carries no relation to compare, so a name
///   being found here is the only thing that decides whether the body captures
///   it.
///
/// Both of those answer *whether the body could capture a reference*, and are
/// over-collected on purpose, because there the doubtful case has to refuse. A
/// scope is also asked the opposite question — whether a reference has *arrived*
/// — which cannot be answered from an over-collected list at all, so
/// `addressable` is collected separately and under the opposite rule.
///
/// All three are held in the form [`Unparser::identifier_comparison_key`] gives,
/// put there once by [`Unparser::emitted_scope`] rather than derived again at
/// each comparison. That is what keeps the normalization from being forgotten:
/// the plan's own *spelling* is not in the variant, so a comparison site has
/// nothing to compare wrongly.
#[derive(Debug)]
enum EmittedScope {
    Readable {
        qualifiers: Vec<Vec<String>>,
        /// The relations the emitted `FROM` at this level can actually name, each
        /// as its own path rather than as the qualifier this dialect will spell
        /// for it — see [`Unparser::addressable_relations`] for why this is not
        /// the same list as `qualifiers`, and why only one of the two questions
        /// asked of a scope may consult it.
        addressable: Vec<Vec<String>>,
        /// The column names an unqualified reference can collide with, or
        /// `None` when the emitted `FROM` presents names the plan does not hold
        /// — see [`UnreadablePart::ColumnNames`]. `None` rather than an empty set,
        /// so that every read has to say what it does when the list is not
        /// knowable instead of treating it as knowably empty.
        exposed: Option<HashSet<String>>,
    },
    /// The emitted `FROM` holds a relation the plan does not describe, so
    /// nothing was learned by looking — see
    /// [`UnreadablePart::Relation`].
    ///
    /// This carries no lists rather than empty or unread ones. A scope that
    /// answers to every reference is only safe while nothing reads *past* that
    /// answer, and the one enforcement that cannot be forgotten is for the
    /// names not to be there: the next reader of a scope — the qualifier
    /// rewrite spiceai/spiceai#12840 is waiting for, or any narrowing of this
    /// guard — then cannot consult a name list without first saying what it
    /// does when there is none.
    Unreadable,
}

/// What becomes of a nested subquery's outer reference on its way out to a join
/// whose `EXISTS`-style body is being emitted.
///
/// The three are distinct because "which scope resolves this reference" and
/// "which scope was it sent to" are different questions, and a two-valued answer
/// has to fold one into the other — either refusing references that bind
/// correctly, or emitting captures unexamined.
#[derive(Clone, Copy, Debug)]
enum OutwardReference {
    /// A body between the reference and the join emits a `FROM` that can address
    /// the very relation the reference names, so the reference resolves there, as
    /// the plan intends, and this join never sees it.
    BindsToAnEnclosingBody,
    /// A body between the two answers to the reference without that being where
    /// the plan sent it, so the reference resolves there instead. Either the body
    /// names a different relation this dialect spells the same way, or it holds
    /// the right one behind a derived table of the emitter's own and so cannot
    /// address it. The capture is decided whatever the join goes on to emit.
    CapturedByAnEnclosingBody,
    /// Nothing between the two answers to the reference, so it reaches the join
    /// and the scopes the join emits decide it.
    ReachesTheJoin,
}

/// What a node's emitted relation keeps [`Unparser::emitted_scope`] from
/// knowing.
///
/// Ordered by how much is lost: `Relation` subsumes `ColumnNames`, which is why
/// [`Unparser::introduces_unreadable`] stops at the first `Relation` it finds.
///
/// Named for the *part* rather than the state, because the two do not line up:
/// `Relation` yields [`EmittedScope::Unreadable`], while `ColumnNames` yields a
/// `Readable` scope whose column list is simply not knowable.
#[derive(Clone, Copy, Debug)]
enum UnreadablePart {
    /// Its name and its columns both, so nothing collected about the side is
    /// worth having.
    Relation,
    /// Only its column names: the relation is named where the walk can see it,
    /// so a qualified reference stays decidable, but it presents columns the
    /// plan does not hold.
    ColumnNames,
}

/// Which of an expression's column references a capture check is being asked
/// about.
///
/// The two kinds follow opposite rules — see
/// [`Unparser::ensure_exists_correlation_not_shadowed`] — so an expression that
/// carries both is asked twice, once under each, rather than once under
/// whichever rule happens to be looser.
#[derive(Clone, Copy, Debug)]
enum ReferenceKind {
    /// Every reference the expression carries.
    Every,
    /// Only [`Expr::OuterReferenceColumn`], which belongs to neither of the
    /// join's inputs and so cannot be attributed by the side it was written on.
    ReachingPastTheJoin,
}

/// The type of the input to the UNNEST table factor.
#[derive(Debug)]
enum UnnestInputType {
    /// The input is a column reference. It will be presented like `outer_ref(column_name)`.
    OuterReference,
    /// The input is a scalar value. It will be presented like a scalar array or struct.
    Scalar,
}
