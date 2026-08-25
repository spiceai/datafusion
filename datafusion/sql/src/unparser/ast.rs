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

use core::fmt;
use std::ops::ControlFlow;

use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    self, LimitClause, OrderByKind, SelectFlavor, visit_expressions,
    visit_expressions_mut,
};

#[derive(Clone)]
pub struct QueryBuilder {
    with: Option<ast::With>,
    body: Option<Box<ast::SetExpr>>,
    order_by_kind: Option<OrderByKind>,
    limit: Option<ast::Expr>,
    limit_by: Vec<ast::Expr>,
    offset: Option<ast::Offset>,
    fetch: Option<ast::Fetch>,
    locks: Vec<ast::LockClause>,
    for_clause: Option<ast::ForClause>,
    // If true, we need to unparse LogicalPlan::Union as a SQL `UNION` rather than a `UNION ALL`.
    distinct_union: bool,
}

impl QueryBuilder {
    pub fn with(&mut self, value: Option<ast::With>) -> &mut Self {
        self.with = value;
        self
    }
    pub fn body(&mut self, value: Box<ast::SetExpr>) -> &mut Self {
        self.body = Some(value);
        self
    }
    pub fn take_body(&mut self) -> Option<Box<ast::SetExpr>> {
        self.body.take()
    }
    pub fn order_by(&mut self, value: OrderByKind) -> &mut Self {
        self.order_by_kind = Some(value);
        self
    }
    pub fn get_order_by(&self) -> Option<OrderByKind> {
        self.order_by_kind.clone()
    }
    pub fn limit(&mut self, value: Option<ast::Expr>) -> &mut Self {
        self.limit = value;
        self
    }
    pub fn limit_by(&mut self, value: Vec<ast::Expr>) -> &mut Self {
        self.limit_by = value;
        self
    }
    pub fn offset(&mut self, value: Option<ast::Offset>) -> &mut Self {
        self.offset = value;
        self
    }
    pub fn fetch(&mut self, value: Option<ast::Fetch>) -> &mut Self {
        self.fetch = value;
        self
    }
    pub fn locks(&mut self, value: Vec<ast::LockClause>) -> &mut Self {
        self.locks = value;
        self
    }
    pub fn for_clause(&mut self, value: Option<ast::ForClause>) -> &mut Self {
        self.for_clause = value;
        self
    }
    pub fn distinct_union(&mut self) -> &mut Self {
        self.distinct_union = true;
        self
    }
    pub fn is_distinct_union(&self) -> bool {
        self.distinct_union
    }
    /// Whether the query bounds *which* rows it returns rather than only
    /// filtering them: `LIMIT`, `OFFSET`, `FETCH` or `LIMIT BY`.
    ///
    /// SQL evaluates all of these after the body's `WHERE`, so a predicate
    /// added to that `WHERE` decides which rows the bound then keeps. A caller
    /// that needs its predicate applied to the bounded result has to put the
    /// bounded query in a scope of its own instead.
    pub fn bounds_rows(&self) -> bool {
        self.limit.is_some()
            || self.offset.is_some()
            || self.fetch.is_some()
            || !self.limit_by.is_empty()
    }
    pub fn build(&self) -> Result<ast::Query, BuilderError> {
        let order_by = self
            .order_by_kind
            .as_ref()
            .map(|order_by_kind| ast::OrderBy {
                kind: order_by_kind.clone(),
                interpolate: None,
            });

        Ok(ast::Query {
            with: self.with.clone(),
            body: match self.body {
                Some(ref value) => value.clone(),
                None => return Err(Into::into(UninitializedFieldError::from("body"))),
            },
            order_by,
            limit_clause: Some(LimitClause::LimitOffset {
                limit: self.limit.clone(),
                offset: self.offset.clone(),
                limit_by: self.limit_by.clone(),
            }),
            fetch: self.fetch.clone(),
            locks: self.locks.clone(),
            for_clause: self.for_clause.clone(),
            settings: None,
            format_clause: None,
            pipe_operators: vec![],
        })
    }
    fn create_empty() -> Self {
        Self {
            with: Default::default(),
            body: Default::default(),
            order_by_kind: Default::default(),
            limit: Default::default(),
            limit_by: Default::default(),
            offset: Default::default(),
            fetch: Default::default(),
            locks: Default::default(),
            for_clause: Default::default(),
            distinct_union: false,
        }
    }
}
impl Default for QueryBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

/// Returns true if `expr` holds a subquery anywhere within it.
fn contains_subquery(expr: &ast::Expr) -> bool {
    visit_expressions(expr, |expr| {
        if matches!(
            expr,
            ast::Expr::Subquery(_)
                | ast::Expr::InSubquery { .. }
                | ast::Expr::Exists { .. }
        ) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}

#[derive(Clone)]
pub struct SelectBuilder {
    distinct: Option<ast::Distinct>,
    top: Option<ast::Top>,
    /// Projection items for the SELECT clause.
    ///
    /// This field uses `Option` to distinguish between three distinct states:
    /// - `None`: No projection has been set (not yet initialized)
    /// - `Some(vec![])`: Empty projection explicitly set (generates `SELECT FROM ...` or `SELECT 1 FROM ...`)
    /// - `Some(vec![SelectItem::Wildcard(...)])`: Wildcard projection (generates `SELECT * FROM ...`)
    /// - `Some(vec![...])`: Non-empty projection with specific columns/expressions
    ///
    /// Use `projection()` to set this field and `already_projected()` to check if it has been set.
    projection: Option<Vec<ast::SelectItem>>,
    into: Option<ast::SelectInto>,
    from: Vec<TableWithJoinsBuilder>,
    lateral_views: Vec<ast::LateralView>,
    selection: Option<ast::Expr>,
    group_by: Option<ast::GroupByExpr>,
    cluster_by: Vec<ast::Expr>,
    distribute_by: Vec<ast::Expr>,
    sort_by: Vec<ast::OrderByExpr>,
    having: Option<ast::Expr>,
    named_window: Vec<ast::NamedWindowDefinition>,
    qualify: Option<ast::Expr>,
    value_table_mode: Option<ast::ValueTableMode>,
    flavor: Option<SelectFlavor>,
    /// Counter for generating unique LATERAL FLATTEN aliases within this SELECT.
    flatten_alias_counter: usize,
    /// Counter for generating unique derived-aggregate aliases within this SELECT.
    derived_aggregate_alias_counter: usize,
    /// Table aliases that correspond to LATERAL FLATTEN relations.
    /// Column references into these aliases must use `VALUE` as the column name.
    flatten_table_aliases: Vec<String>,
    /// Whether a `LogicalPlan::Aggregate` has already been folded into this SELECT,
    /// as its select list and `GROUP BY`. A SELECT expresses at most one grouping, so
    /// a second aggregate below it belongs in a derived table.
    ///
    /// Set with `mark_aggregated()` and read with `already_aggregated()`.
    aggregated: bool,
}

/// Prefix used for auto-generated LATERAL FLATTEN table aliases.
const FLATTEN_ALIAS_PREFIX: &str = "_unnest";

/// Prefix used for the auto-generated alias of a derived table that carries an
/// aggregate stacked below the one its enclosing SELECT already expresses.
const DERIVED_AGGREGATE_ALIAS_PREFIX: &str = "derived_aggregate";

/// The prefixes [`numbered_alias`] builds a name from.
const NUMBERED_ALIAS_PREFIXES: [&str; 2] =
    [FLATTEN_ALIAS_PREFIX, DERIVED_AGGREGATE_ALIAS_PREFIX];

/// Aliases the unparser gives verbatim to a derived table it introduces, one
/// per kind of node that has to be wrapped.
///
/// Public because a name the unparser can invent is in the emitted scope
/// without the plan holding it anywhere, so code reasoning about what that
/// scope answers to has to be able to ask.
pub(crate) const DERIVED_DISTINCT_ALIAS: &str = "derived_distinct";
pub(crate) const DERIVED_LIMIT_ALIAS: &str = "derived_limit";
pub(crate) const DERIVED_PROJECTION_ALIAS: &str = "derived_projection";
pub(crate) const DERIVED_SORT_ALIAS: &str = "derived_sort";
pub(crate) const DERIVED_UNION_ALIAS: &str = "derived_union";
pub(crate) const DERIVED_UNNEST_ALIAS: &str = "derived_unnest";
pub(crate) const DERIVED_WINDOW_INPUT_ALIAS: &str = "derived_window_input";

/// The name the counter-numbered generators below build: a prefix, an
/// underscore, and the count.
///
/// Written once so that [`is_numbered_alias`] is its exact inverse. A guard
/// that has to recognise these names without generating them lives in another
/// module, and the two agreeing is what keeps such a guard from quietly
/// matching nothing.
fn numbered_alias(prefix: &str, counter: usize) -> String {
    format!("{prefix}_{counter}")
}

/// Whether `name` is one this module's counter-numbered generators can produce.
///
/// The inverse of [`numbered_alias`], and exactly it: the counters start at 1 and
/// `usize` renders canonically, so `_unnest_0`, `_unnest_01` and a run of digits
/// too long to be a `usize` are names no generator here can build. Recognising
/// one of those would treat a relation a user happened to give that name as an
/// alias the unparser invented, and refuse a correlation against it that binds
/// correctly.
///
/// Decided by rebuilding the name rather than by inspecting the digits, so the
/// two cannot disagree about a form neither anticipated.
pub(crate) fn is_numbered_alias(name: &str) -> bool {
    let Some((prefix, counter)) = name.rsplit_once('_') else {
        return false;
    };
    NUMBERED_ALIAS_PREFIXES.contains(&prefix)
        && counter
            .parse::<usize>()
            .is_ok_and(|counter| counter > 0 && numbered_alias(prefix, counter) == name)
}

impl SelectBuilder {
    /// Generate a unique alias for a LATERAL FLATTEN relation
    /// (`_unnest_1`, `_unnest_2`, …). Each call returns a fresh name.
    pub fn next_flatten_alias(&mut self) -> String {
        self.flatten_alias_counter += 1;
        numbered_alias(FLATTEN_ALIAS_PREFIX, self.flatten_alias_counter)
    }

    /// Generate a unique alias for a derived table holding a stacked aggregate
    /// (`derived_aggregate_1`, `derived_aggregate_2`, …). Each call returns a fresh
    /// name. A join walks both of its sides with one builder, so both sides can
    /// derive an aggregate into the same FROM clause; the number is what keeps the
    /// two apart, both as table names and as the qualifier each side's own column
    /// references are rewritten onto.
    pub fn next_derived_aggregate_alias(&mut self) -> String {
        self.derived_aggregate_alias_counter += 1;
        numbered_alias(
            DERIVED_AGGREGATE_ALIAS_PREFIX,
            self.derived_aggregate_alias_counter,
        )
    }

    /// Register a table alias as pointing to a LATERAL FLATTEN relation.
    pub fn add_flatten_table_alias(&mut self, alias: String) {
        self.flatten_table_aliases.push(alias);
    }

    /// Returns true if no FLATTEN table aliases have been registered.
    pub fn flatten_table_aliases_empty(&self) -> bool {
        self.flatten_table_aliases.is_empty()
    }

    /// Returns true if the given table alias refers to a FLATTEN relation.
    pub fn is_flatten_table_alias(&self, alias: &str) -> bool {
        self.flatten_table_aliases.iter().any(|a| a == alias)
    }

    /// Record that an aggregate node is now expressed by this SELECT.
    pub fn mark_aggregated(&mut self) {
        self.aggregated = true;
    }

    /// Returns true if an aggregate node has already been folded into this SELECT.
    pub fn already_aggregated(&self) -> bool {
        self.aggregated
    }

    /// Returns the most recently generated flatten alias, or `None` if
    /// `next_flatten_alias` has not been called yet.
    pub fn current_flatten_alias(&self) -> Option<String> {
        if self.flatten_alias_counter > 0 {
            Some(numbered_alias(
                FLATTEN_ALIAS_PREFIX,
                self.flatten_alias_counter,
            ))
        } else {
            None
        }
    }

    pub fn distinct(&mut self, value: Option<ast::Distinct>) -> &mut Self {
        self.distinct = value;
        self
    }
    pub fn top(&mut self, value: Option<ast::Top>) -> &mut Self {
        self.top = value;
        self
    }
    pub fn get_projection(&self) -> Vec<ast::SelectItem> {
        self.projection.clone().unwrap_or_default()
    }
    pub fn projection(&mut self, value: Vec<ast::SelectItem>) -> &mut Self {
        self.projection = Some(value);
        self
    }
    pub fn pop_projections(&mut self) -> Vec<ast::SelectItem> {
        self.projection.take().unwrap_or_default()
    }
    /// Returns true if a projection has been explicitly set via `projection()`.
    ///
    /// This method is used to determine whether the SELECT clause has already been
    /// defined, which helps avoid creating duplicate projection nodes during query
    /// unparsing. It returns `true` for both empty and non-empty projections.
    ///
    /// # Returns
    ///
    /// - `true` if `projection()` has been called (regardless of whether it was empty or not)
    /// - `false` if no projection has been set yet
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut builder = SelectBuilder::default();
    /// assert!(!builder.already_projected());
    ///
    /// builder.projection(vec![]);
    /// assert!(builder.already_projected()); // true even for empty projection
    ///
    /// builder.projection(vec![SelectItem::Wildcard(...)]);
    /// assert!(builder.already_projected()); // true for non-empty projection
    /// ```
    pub fn already_projected(&self) -> bool {
        self.projection.is_some()
    }
    pub fn into(&mut self, value: Option<ast::SelectInto>) -> &mut Self {
        self.into = value;
        self
    }
    pub fn from(&mut self, value: Vec<TableWithJoinsBuilder>) -> &mut Self {
        self.from = value;
        self
    }
    pub fn push_from(&mut self, value: TableWithJoinsBuilder) -> &mut Self {
        self.from.push(value);
        self
    }
    pub fn pop_from(&mut self) -> Option<TableWithJoinsBuilder> {
        self.from.pop()
    }
    pub fn lateral_views(&mut self, value: Vec<ast::LateralView>) -> &mut Self {
        self.lateral_views = value;
        self
    }

    /// Replaces the selection with a new value.
    ///
    /// This function is used to replace a specific expression within the selection.
    /// Unlike the `selection` method which combines existing and new selections with AND,
    /// this method searches for and replaces occurrences of a specific expression.
    ///
    /// This method is primarily used to modify LEFT MARK JOIN expressions.
    /// When processing a LEFT MARK JOIN, we need to replace the placeholder expression
    /// with the actual join condition in the selection clause.
    ///
    /// # Arguments
    ///
    /// * `existing_expr` - The expression to replace
    /// * `value` - The new expression to set as the selection
    pub fn replace_mark(
        &mut self,
        existing_expr: &ast::Expr,
        value: &ast::Expr,
    ) -> &mut Self {
        if let Some(selection) = &mut self.selection {
            let _ = visit_expressions_mut(selection, |expr| {
                if expr == existing_expr {
                    *expr = value.clone();
                }
                ControlFlow::<()>::Continue(())
            });
        }
        self
    }

    pub fn selection(&mut self, value: Option<ast::Expr>) -> &mut Self {
        // With filter pushdown optimization, the LogicalPlan can have filters defined as part of `TableScan` and `Filter` nodes.
        // To avoid overwriting one of the filters, we combine the existing filter with the additional filter.
        // Example:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
        // |  Projection: customer.c_phone AS cntrycode, customer.c_acctbal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
        // |   Filter: CAST(customer.c_acctbal AS Decimal128(38, 6)) > (<subquery>)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
        // |     Subquery:
        // |     ..                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
        // |     TableScan: customer, full_filters=[customer.c_mktsegment = Utf8("BUILDING")]
        match (&self.selection, value) {
            (Some(existing_selection), Some(new_selection)) => {
                self.selection = Some(ast::Expr::BinaryOp {
                    left: Box::new(existing_selection.clone()),
                    op: ast::BinaryOperator::And,
                    right: Box::new(new_selection),
                });
            }
            (None, Some(new_selection)) => {
                self.selection = Some(new_selection);
            }
            (_, None) => (),
        }

        self
    }

    /// Whether this `SELECT` already carries a `WHERE` predicate.
    pub fn has_selection(&self) -> bool {
        self.selection.is_some()
    }

    /// Whether this `SELECT` already carries a predicate that can only be
    /// stated alongside the grouping or windowing it filters — `HAVING` or
    /// `QUALIFY`.
    ///
    /// Both are evaluated before `LIMIT`/`OFFSET`, like `WHERE`, but unlike
    /// `WHERE` they cannot simply be lifted into an enclosing query: the
    /// aggregate or window expression they reference is only nameable in the
    /// `SELECT` that computes it.
    pub fn has_grouped_predicate(&self) -> bool {
        self.having.is_some() || self.qualify.is_some()
    }

    /// Removes the `WHERE` predicate accumulated so far and returns it, so a
    /// caller can tell what a sub-plan contributed and re-place it elsewhere.
    pub fn take_selection(&mut self) -> Option<ast::Expr> {
        self.selection.take()
    }

    /// Applies `f` to every expression this SELECT carries: the projection, `WHERE`,
    /// `GROUP BY`, `HAVING`, `QUALIFY` and the builder's own sort. `f` sees nested
    /// expressions too, so a rewrite reaches a column reference wherever it sits.
    ///
    /// This exists so a caller that replaces the SELECT's relation — unparsing a sub-plan
    /// as a derived table — can re-point the column references that addressed the old one,
    /// without the builder having to expose each clause for reading.
    ///
    /// An expression containing a subquery is skipped whole. A correlated subquery
    /// references an enclosing query's relation, which stays in scope and is
    /// indistinguishable by name from a reference to this SELECT's own relation, so
    /// rewriting inside one would silently change which column the subquery reads. That
    /// leaves such an expression untouched rather than risk rewriting it wrongly.
    pub fn visit_expressions_in_clauses_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut ast::Expr),
    {
        let mut visit = |expr: &mut ast::Expr| {
            if contains_subquery(expr) {
                return;
            }
            let _ = visit_expressions_mut(expr, |expr| {
                f(expr);
                ControlFlow::<()>::Continue(())
            });
        };

        for item in self.projection.iter_mut().flatten() {
            match item {
                ast::SelectItem::UnnamedExpr(expr)
                | ast::SelectItem::ExprWithAlias { expr, .. }
                | ast::SelectItem::ExprWithAliases { expr, .. } => visit(expr),
                ast::SelectItem::QualifiedWildcard(..) | ast::SelectItem::Wildcard(_) => {
                }
            }
        }
        for expr in self
            .selection
            .iter_mut()
            .chain(self.having.iter_mut())
            .chain(self.qualify.iter_mut())
        {
            visit(expr);
        }
        if let Some(ast::GroupByExpr::Expressions(exprs, _)) = self.group_by.as_mut() {
            for expr in exprs {
                visit(expr);
            }
        }
        for sort in &mut self.sort_by {
            visit(&mut sort.expr);
        }
    }

    pub fn group_by(&mut self, value: ast::GroupByExpr) -> &mut Self {
        self.group_by = Some(value);
        self
    }
    pub fn cluster_by(&mut self, value: Vec<ast::Expr>) -> &mut Self {
        self.cluster_by = value;
        self
    }
    pub fn distribute_by(&mut self, value: Vec<ast::Expr>) -> &mut Self {
        self.distribute_by = value;
        self
    }
    pub fn sort_by(&mut self, value: Vec<ast::OrderByExpr>) -> &mut Self {
        self.sort_by = value;
        self
    }
    pub fn get_sort_by(&self) -> Vec<ast::OrderByExpr> {
        self.sort_by.clone()
    }
    pub fn having(&mut self, value: Option<ast::Expr>) -> &mut Self {
        self.having = value;
        self
    }
    pub fn named_window(&mut self, value: Vec<ast::NamedWindowDefinition>) -> &mut Self {
        self.named_window = value;
        self
    }
    pub fn qualify(&mut self, value: Option<ast::Expr>) -> &mut Self {
        self.qualify = value;
        self
    }
    pub fn value_table_mode(&mut self, value: Option<ast::ValueTableMode>) -> &mut Self {
        self.value_table_mode = value;
        self
    }
    pub fn build(&self) -> Result<ast::Select, BuilderError> {
        Ok(ast::Select {
            optimizer_hints: vec![],
            distinct: self.distinct.clone(),
            select_modifiers: None,
            top_before_distinct: false,
            top: self.top.clone(),
            projection: self.projection.clone().unwrap_or_default(),
            into: self.into.clone(),
            from: self
                .from
                .iter()
                .filter_map(|b| b.build().transpose())
                .collect::<Result<Vec<_>, BuilderError>>()?,
            lateral_views: self.lateral_views.clone(),
            selection: self.selection.clone(),
            group_by: match self.group_by {
                Some(ref value) => value.clone(),
                None => {
                    return Err(Into::into(UninitializedFieldError::from("group_by")));
                }
            },
            cluster_by: self.cluster_by.clone(),
            distribute_by: self.distribute_by.clone(),
            sort_by: self.sort_by.clone(),
            having: self.having.clone(),
            named_window: self.named_window.clone(),
            qualify: self.qualify.clone(),
            value_table_mode: self.value_table_mode,
            connect_by: Vec::new(),
            window_before_qualify: false,
            prewhere: None,
            select_token: AttachedToken::empty(),
            flavor: match self.flavor {
                Some(ref value) => *value,
                None => return Err(Into::into(UninitializedFieldError::from("flavor"))),
            },
            exclude: None,
        })
    }
    fn create_empty() -> Self {
        Self {
            distinct: Default::default(),
            top: Default::default(),
            projection: None,
            into: Default::default(),
            from: Default::default(),
            lateral_views: Default::default(),
            selection: Default::default(),
            group_by: Some(ast::GroupByExpr::Expressions(Vec::new(), Vec::new())),
            cluster_by: Default::default(),
            distribute_by: Default::default(),
            sort_by: Default::default(),
            having: Default::default(),
            named_window: Default::default(),
            qualify: Default::default(),
            value_table_mode: Default::default(),
            flavor: Some(SelectFlavor::Standard),
            flatten_alias_counter: 0,
            derived_aggregate_alias_counter: 0,
            flatten_table_aliases: Vec::new(),
            aggregated: false,
        }
    }
}
impl Default for SelectBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

#[derive(Clone)]
pub struct TableWithJoinsBuilder {
    relation: Option<RelationBuilder>,
    joins: Vec<ast::Join>,
}

impl TableWithJoinsBuilder {
    pub fn relation(&mut self, value: RelationBuilder) -> &mut Self {
        self.relation = Some(value);
        self
    }
    pub fn get_joins(&self) -> Vec<ast::Join> {
        self.joins.clone()
    }
    pub fn joins(&mut self, value: Vec<ast::Join>) -> &mut Self {
        self.joins = value;
        self
    }
    pub fn push_join(&mut self, value: ast::Join) -> &mut Self {
        self.joins.push(value);
        self
    }

    pub fn build(&self) -> Result<Option<ast::TableWithJoins>, BuilderError> {
        match self.relation {
            Some(ref value) => match value.build()? {
                Some(relation) => Ok(Some(ast::TableWithJoins {
                    relation,
                    joins: self.joins.clone(),
                })),
                None => Ok(None),
            },
            None => Err(Into::into(UninitializedFieldError::from("relation"))),
        }
    }
    fn create_empty() -> Self {
        Self {
            relation: Default::default(),
            joins: Default::default(),
        }
    }
}
impl Default for TableWithJoinsBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

#[derive(Clone)]
pub struct RelationBuilder {
    relation: Option<TableFactorBuilder>,
}

#[derive(Clone)]
// Boxing variants would penalize the common builder path; this enum is
// constructed-then-consumed locally rather than stored at scale.
#[expect(clippy::large_enum_variant)]
enum TableFactorBuilder {
    Table(TableRelationBuilder),
    Derived(DerivedRelationBuilder),
    Unnest(UnnestRelationBuilder),
    Flatten(FlattenRelationBuilder),
    Empty,
}

impl RelationBuilder {
    pub fn has_relation(&self) -> bool {
        self.relation.is_some()
    }
    pub fn get_name(&self) -> Option<String> {
        match self.relation {
            Some(TableFactorBuilder::Table(ref value)) => {
                value.name.as_ref().map(|a| a.to_string())
            }
            _ => None,
        }
    }
    pub fn get_alias(&self) -> Option<String> {
        match self.relation {
            Some(TableFactorBuilder::Table(ref value)) => {
                value.alias.as_ref().map(|a| a.name.to_string())
            }
            Some(TableFactorBuilder::Derived(ref value)) => {
                value.alias.as_ref().map(|a| a.name.to_string())
            }
            _ => None,
        }
    }
    pub fn table(&mut self, value: TableRelationBuilder) -> &mut Self {
        self.relation = Some(TableFactorBuilder::Table(value));
        self
    }
    pub fn derived(&mut self, value: DerivedRelationBuilder) -> &mut Self {
        self.relation = Some(TableFactorBuilder::Derived(value));
        self
    }

    pub fn unnest(&mut self, value: UnnestRelationBuilder) -> &mut Self {
        self.relation = Some(TableFactorBuilder::Unnest(value));
        self
    }

    pub fn flatten(&mut self, value: FlattenRelationBuilder) -> &mut Self {
        self.relation = Some(TableFactorBuilder::Flatten(value));
        self
    }

    pub fn empty(&mut self) -> &mut Self {
        self.relation = Some(TableFactorBuilder::Empty);
        self
    }
    pub fn alias(&mut self, value: Option<ast::TableAlias>) -> &mut Self {
        let new = self;
        match new.relation {
            Some(TableFactorBuilder::Table(ref mut rel_builder)) => {
                rel_builder.alias = value;
            }
            Some(TableFactorBuilder::Derived(ref mut rel_builder)) => {
                rel_builder.alias = value;
            }
            Some(TableFactorBuilder::Unnest(ref mut rel_builder)) => {
                rel_builder.alias = value;
            }
            Some(TableFactorBuilder::Flatten(ref mut rel_builder)) => {
                rel_builder.alias = value;
            }
            Some(TableFactorBuilder::Empty) => (),
            None => (),
        }
        new
    }
    pub fn build(&self) -> Result<Option<ast::TableFactor>, BuilderError> {
        Ok(match self.relation {
            Some(TableFactorBuilder::Table(ref value)) => Some(value.build()?),
            Some(TableFactorBuilder::Derived(ref value)) => Some(value.build()?),
            Some(TableFactorBuilder::Unnest(ref value)) => Some(value.build()?),
            Some(TableFactorBuilder::Flatten(ref value)) => Some(value.build()?),
            Some(TableFactorBuilder::Empty) => None,
            None => return Err(Into::into(UninitializedFieldError::from("relation"))),
        })
    }
    fn create_empty() -> Self {
        Self {
            relation: Default::default(),
        }
    }
}
impl Default for RelationBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

#[derive(Clone)]
pub struct TableRelationBuilder {
    name: Option<ast::ObjectName>,
    alias: Option<ast::TableAlias>,
    args: Option<Vec<ast::FunctionArg>>,
    with_hints: Vec<ast::Expr>,
    version: Option<ast::TableVersion>,
    partitions: Vec<ast::Ident>,
    index_hints: Vec<ast::TableIndexHints>,
}

impl TableRelationBuilder {
    pub fn name(&mut self, value: ast::ObjectName) -> &mut Self {
        self.name = Some(value);
        self
    }
    pub fn alias(&mut self, value: Option<ast::TableAlias>) -> &mut Self {
        self.alias = value;
        self
    }
    pub fn args(&mut self, value: Option<Vec<ast::FunctionArg>>) -> &mut Self {
        self.args = value;
        self
    }
    pub fn with_hints(&mut self, value: Vec<ast::Expr>) -> &mut Self {
        self.with_hints = value;
        self
    }
    pub fn version(&mut self, value: Option<ast::TableVersion>) -> &mut Self {
        self.version = value;
        self
    }
    pub fn partitions(&mut self, value: Vec<ast::Ident>) -> &mut Self {
        self.partitions = value;
        self
    }
    pub fn index_hints(&mut self, value: Vec<ast::TableIndexHints>) -> &mut Self {
        self.index_hints = value;
        self
    }
    pub fn build(&self) -> Result<ast::TableFactor, BuilderError> {
        Ok(ast::TableFactor::Table {
            name: match self.name {
                Some(ref value) => value.clone(),
                None => return Err(Into::into(UninitializedFieldError::from("name"))),
            },
            alias: self.alias.clone(),
            args: self.args.clone().map(|args| ast::TableFunctionArgs {
                args,
                settings: None,
            }),
            with_hints: self.with_hints.clone(),
            version: self.version.clone(),
            partitions: self.partitions.clone(),
            with_ordinality: false,
            json_path: None,
            sample: None,
            index_hints: self.index_hints.clone(),
        })
    }
    fn create_empty() -> Self {
        Self {
            name: Default::default(),
            alias: Default::default(),
            args: Default::default(),
            with_hints: Default::default(),
            version: Default::default(),
            partitions: Default::default(),
            index_hints: Default::default(),
        }
    }
}
impl Default for TableRelationBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}
#[derive(Clone)]
pub struct DerivedRelationBuilder {
    lateral: Option<bool>,
    subquery: Option<Box<ast::Query>>,
    alias: Option<ast::TableAlias>,
}

impl DerivedRelationBuilder {
    pub fn lateral(&mut self, value: bool) -> &mut Self {
        self.lateral = Some(value);
        self
    }
    pub fn subquery(&mut self, value: Box<ast::Query>) -> &mut Self {
        self.subquery = Some(value);
        self
    }
    pub fn alias(&mut self, value: Option<ast::TableAlias>) -> &mut Self {
        self.alias = value;
        self
    }
    fn build(&self) -> Result<ast::TableFactor, BuilderError> {
        Ok(ast::TableFactor::Derived {
            lateral: match self.lateral {
                Some(ref value) => *value,
                None => return Err(Into::into(UninitializedFieldError::from("lateral"))),
            },
            subquery: match self.subquery {
                Some(ref value) => value.clone(),
                None => {
                    return Err(Into::into(UninitializedFieldError::from("subquery")));
                }
            },
            alias: self.alias.clone(),
            sample: None,
        })
    }
    fn create_empty() -> Self {
        Self {
            lateral: Default::default(),
            subquery: Default::default(),
            alias: Default::default(),
        }
    }
}
impl Default for DerivedRelationBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

#[derive(Clone)]
pub struct UnnestRelationBuilder {
    pub alias: Option<ast::TableAlias>,
    pub array_exprs: Vec<ast::Expr>,
    with_offset: bool,
    with_offset_alias: Option<ast::Ident>,
    with_ordinality: bool,
}

impl UnnestRelationBuilder {
    pub fn alias(&mut self, value: Option<ast::TableAlias>) -> &mut Self {
        self.alias = value;
        self
    }
    pub fn array_exprs(&mut self, value: Vec<ast::Expr>) -> &mut Self {
        self.array_exprs = value;
        self
    }

    pub fn with_offset(&mut self, value: bool) -> &mut Self {
        self.with_offset = value;
        self
    }

    pub fn with_offset_alias(&mut self, value: Option<ast::Ident>) -> &mut Self {
        self.with_offset_alias = value;
        self
    }

    pub fn with_ordinality(&mut self, value: bool) -> &mut Self {
        self.with_ordinality = value;
        self
    }

    pub fn build(&self) -> Result<ast::TableFactor, BuilderError> {
        Ok(ast::TableFactor::UNNEST {
            alias: self.alias.clone(),
            array_exprs: self.array_exprs.clone(),
            with_offset: self.with_offset,
            with_offset_alias: self.with_offset_alias.clone(),
            with_ordinality: self.with_ordinality,
        })
    }

    fn create_empty() -> Self {
        Self {
            alias: Default::default(),
            array_exprs: Default::default(),
            with_offset: Default::default(),
            with_offset_alias: Default::default(),
            with_ordinality: Default::default(),
        }
    }
}

impl Default for UnnestRelationBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

/// Builds a `LATERAL FLATTEN(INPUT => expr, OUTER => bool)` table factor
/// for Snowflake-style unnesting.
#[derive(Clone)]
pub struct FlattenRelationBuilder {
    pub alias: Option<ast::TableAlias>,
    /// The input expression to flatten (e.g. a column reference).
    pub input_expr: Option<ast::Expr>,
    /// Whether to preserve rows for NULL/empty inputs (Snowflake `OUTER` param).
    pub outer: bool,
}

impl FlattenRelationBuilder {
    pub fn alias(&mut self, value: Option<ast::TableAlias>) -> &mut Self {
        self.alias = value;
        self
    }

    pub fn input_expr(&mut self, value: ast::Expr) -> &mut Self {
        self.input_expr = Some(value);
        self
    }

    pub fn outer(&mut self, value: bool) -> &mut Self {
        self.outer = value;
        self
    }

    pub fn build(&self) -> Result<ast::TableFactor, BuilderError> {
        let input = self.input_expr.clone().ok_or_else(|| {
            BuilderError::from(UninitializedFieldError::from("input_expr"))
        })?;

        let mut args = vec![ast::FunctionArg::Named {
            name: ast::Ident::new("INPUT"),
            arg: ast::FunctionArgExpr::Expr(input),
            operator: ast::FunctionArgOperator::RightArrow,
        }];

        if self.outer {
            args.push(ast::FunctionArg::Named {
                name: ast::Ident::new("OUTER"),
                arg: ast::FunctionArgExpr::Expr(ast::Expr::Value(
                    ast::Value::Boolean(true).into(),
                )),
                operator: ast::FunctionArgOperator::RightArrow,
            });
        }

        Ok(ast::TableFactor::Function {
            lateral: true,
            name: ast::ObjectName::from(vec![ast::Ident::new("FLATTEN")]),
            args,
            with_ordinality: false,
            alias: self.alias.clone(),
        })
    }

    fn create_empty() -> Self {
        Self {
            alias: None,
            input_expr: None,
            outer: false,
        }
    }
}

impl Default for FlattenRelationBuilder {
    fn default() -> Self {
        Self::create_empty()
    }
}

/// Runtime error when a `build()` method is called and one or more required fields
/// do not have a value.
#[derive(Debug, Clone)]
pub struct UninitializedFieldError(&'static str);

impl UninitializedFieldError {
    /// Create a new `UninitializedFieldError` for the specified field name.
    pub fn new(field_name: &'static str) -> Self {
        UninitializedFieldError(field_name)
    }

    /// Get the name of the first-declared field that wasn't initialized
    pub fn field_name(&self) -> &'static str {
        self.0
    }
}

impl fmt::Display for UninitializedFieldError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Field not initialized: {}", self.0)
    }
}

impl From<&'static str> for UninitializedFieldError {
    fn from(field_name: &'static str) -> Self {
        Self::new(field_name)
    }
}
impl std::error::Error for UninitializedFieldError {}

#[derive(Debug)]
pub enum BuilderError {
    UninitializedField(&'static str),
    ValidationError(String),
}
impl From<UninitializedFieldError> for BuilderError {
    fn from(s: UninitializedFieldError) -> Self {
        Self::UninitializedField(s.field_name())
    }
}
impl From<String> for BuilderError {
    fn from(s: String) -> Self {
        Self::ValidationError(s)
    }
}
impl fmt::Display for BuilderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::UninitializedField(field) => {
                write!(f, "`{field}` must be initialized")
            }
            Self::ValidationError(error) => write!(f, "{error}"),
        }
    }
}
impl std::error::Error for BuilderError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every name the counter-numbered generators actually produce is recognised.
    ///
    /// Driven from the generators rather than from written-out strings, so the
    /// pairing is pinned by the code that builds the names instead of by a copy
    /// of it.
    #[test]
    fn generated_numbered_aliases_are_recognised() {
        let mut select = SelectBuilder::default();
        for _ in 0..3 {
            let flatten = select.next_flatten_alias();
            assert!(
                is_numbered_alias(&flatten),
                "the flatten generator produced {flatten}, which is not recognised"
            );
            let aggregate = select.next_derived_aggregate_alias();
            assert!(
                is_numbered_alias(&aggregate),
                "the aggregate generator produced {aggregate}, which is not recognised"
            );
        }
    }

    /// A name no generator can build is not one, however much it looks like one.
    ///
    /// The counters start at 1 and render canonically, so a zero, a leading zero
    /// and a run of digits wider than a `usize` are all names the unparser never
    /// invents. Recognising one would treat a user's own relation as an invented
    /// alias and refuse a correlation against it.
    #[test]
    fn numbered_alias_lookalikes_are_not_recognised() {
        for name in [
            "_unnest_0",
            "_unnest_01",
            "_unnest_+1",
            "_unnest_",
            "_unnest",
            "_unnest_1x",
            "_unnest_99999999999999999999999999999999999999",
            "derived_aggregate_0",
            "derived_aggregate_007",
            "unnest_1",
            "_flatten_1",
        ] {
            assert!(
                !is_numbered_alias(name),
                "{name} is not a name any generator here builds, so it must not be recognised as one"
            );
        }
    }
}
