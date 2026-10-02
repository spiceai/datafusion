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

use std::{collections::HashSet, sync::Arc};

use arrow::datatypes::Schema;
use datafusion_common::tree_node::TreeNodeContainer;
use datafusion_common::{
    Column, HashMap, Result, TableReference,
    tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRewriter},
};
use datafusion_expr::expr::{Alias, UNNEST_COLUMN_PREFIX};
use datafusion_expr::{Expr, LogicalPlan, Projection, Sort, SortExpr};
use sqlparser::ast::{self, Ident};

/// Normalize the schema of a union plan to remove qualifiers from the schema fields and sort expressions.
///
/// DataFusion will return an error if two columns in the schema have the same name with no table qualifiers.
/// There are certain types of UNION queries that can result in having two columns with the same name, and the
/// solution was to add table qualifiers to the schema fields.
/// See <https://github.com/apache/datafusion/issues/5410> for more context on this decision.
///
/// However, this causes a problem when unparsing these queries back to SQL - as the table qualifier has
/// logically been erased and is no longer a valid reference.
///
/// The following input SQL:
/// ```sql
/// SELECT table1.foo FROM table1
/// UNION ALL
/// SELECT table2.foo FROM table2
/// ORDER BY foo
/// ```
///
/// Would be unparsed into the following invalid SQL without this transformation:
/// ```sql
/// SELECT table1.foo FROM table1
/// UNION ALL
/// SELECT table2.foo FROM table2
/// ORDER BY table1.foo
/// ```
///
/// Which would result in a SQL error, as `table1.foo` is not a valid reference in the context of the UNION.
pub(super) fn normalize_union_schema(plan: &LogicalPlan) -> Result<LogicalPlan> {
    let plan = plan.clone();

    let transformed_plan = plan.transform_up(|plan| match plan {
        LogicalPlan::Union(mut union) => {
            let schema = Arc::unwrap_or_clone(union.schema);
            let schema = schema.strip_qualifiers();

            union.schema = Arc::new(schema);
            Ok(Transformed::yes(LogicalPlan::Union(union)))
        }
        LogicalPlan::Sort(sort) => {
            // Only rewrite Sort expressions that have a UNION as their input
            if !matches!(&*sort.input, LogicalPlan::Union(_)) {
                return Ok(Transformed::no(LogicalPlan::Sort(sort)));
            }

            Ok(Transformed::yes(LogicalPlan::Sort(Sort {
                expr: rewrite_sort_expr_for_union(sort.expr)?,
                input: sort.input,
                fetch: sort.fetch,
            })))
        }
        _ => Ok(Transformed::no(plan)),
    });
    transformed_plan.data()
}

/// Rewrite sort expressions that have a UNION plan as their input to remove the table reference.
fn rewrite_sort_expr_for_union(exprs: Vec<SortExpr>) -> Result<Vec<SortExpr>> {
    let sort_exprs = exprs
        .map_elements(&mut |expr: Expr| expr.transform_up(unqualify_column))
        .data()?;

    Ok(sort_exprs)
}

/// Drops the relation qualifier from every column reference in `expr`, so each binds
/// by name to whatever the `SELECT` reads — a set operation's output, or a derived
/// table that is the `SELECT`'s only relation.
pub(super) fn unqualify_columns(expr: Expr) -> Result<Expr> {
    expr.transform_up(unqualify_column).data()
}

fn unqualify_column(expr: Expr) -> Result<Transformed<Expr>> {
    if let Expr::Column(mut col) = expr {
        col.relation = None;
        Ok(Transformed::yes(Expr::Column(col)))
    } else {
        Ok(Transformed::no(expr))
    }
}

/// Rewrite Filter plans that have a Window as their input by inserting a SubqueryAlias.
///
/// When a Filter directly operates on a Window plan, it can cause issues during SQL unparsing
/// because window functions in a WHERE clause are not valid SQL. The solution is to wrap
/// the Window plan in a SubqueryAlias, effectively creating a derived table.
///
/// Example transformation:
///
/// Filter: condition
///   Window: window_function
///     TableScan: table
///
/// becomes:
///
/// Filter: condition
///   SubqueryAlias: __qualify_subquery
///     Projection: table.column1, table.column2
///       Window: window_function
///         TableScan: table
pub(super) fn rewrite_qualify(plan: LogicalPlan) -> Result<LogicalPlan> {
    let transformed_plan = plan.transform_up(|plan| match plan {
        // Check if the filter's input is a Window plan
        LogicalPlan::Filter(mut filter) => {
            if matches!(&*filter.input, LogicalPlan::Window(_)) {
                // Create a SubqueryAlias around the Window plan
                let qualifier = filter
                    .input
                    .schema()
                    .iter()
                    .find_map(|(q, _)| q)
                    .map(|q| q.to_string())
                    .unwrap_or_else(|| "__qualify_subquery".to_string());

                // for Postgres, name of column for 'rank() over (...)' is 'rank'
                // but in Datafusion, it is 'rank() over (...)'
                // without projection, it's still an invalid sql in Postgres

                let project_exprs = filter
                    .input
                    .schema()
                    .iter()
                    .map(|(_, f)| datafusion_expr::col(f.name()).alias(f.name()))
                    .collect::<Vec<_>>();

                let input =
                    datafusion_expr::LogicalPlanBuilder::from(Arc::clone(&filter.input))
                        .project(project_exprs)?
                        .build()?;

                let subquery_alias =
                    datafusion_expr::SubqueryAlias::try_new(Arc::new(input), qualifier)?;

                filter.input = Arc::new(LogicalPlan::SubqueryAlias(subquery_alias));
                Ok(Transformed::yes(LogicalPlan::Filter(filter)))
            } else {
                Ok(Transformed::no(LogicalPlan::Filter(filter)))
            }
        }

        _ => Ok(Transformed::no(plan)),
    });

    transformed_plan.data()
}

/// Rewrite logic plan for query that order by columns are not in projections
/// Plan before rewrite:
///
/// Projection: j1.j1_string, j2.j2_string
///   Sort: j1.j1_id DESC NULLS FIRST, j2.j2_id DESC NULLS FIRST
///     Projection: j1.j1_string, j2.j2_string, j1.j1_id, j2.j2_id
///       Inner Join:  Filter: j1.j1_id = j2.j2_id
///         TableScan: j1
///         TableScan: j2
///
/// Plan after rewrite
///
/// Sort: j1.j1_id DESC NULLS FIRST, j2.j2_id DESC NULLS FIRST
///   Projection: j1.j1_string, j2.j2_string
///     Inner Join:  Filter: j1.j1_id = j2.j2_id
///       TableScan: j1
///       TableScan: j2
///
/// This prevents the original plan generate query with derived table but missing alias.
///
/// It also keeps the ORDER BY at the top level of the emitted statement. Left as
/// `Projection -> Sort`, the `Sort` is unparsed as a derived table:
///
/// ```sql
/// SELECT id FROM (SELECT person.id, person.age FROM person ORDER BY person.age)
/// ```
///
/// SQL does not require the enclosing query to preserve the ordering of a derived
/// table, so the rows come back in an arbitrary order.
///
/// A sort key does not have to *be* one of the inner Projection's outputs for the
/// hoist to be valid -- it only has to be computable from them, as `age + 1` is from
/// `age`.
pub(super) fn rewrite_plan_for_sort_on_non_projected_fields(
    p: &Projection,
) -> Option<LogicalPlan> {
    let LogicalPlan::Sort(sort) = p.input.as_ref() else {
        return None;
    };

    let LogicalPlan::Projection(inner_p) = sort.input.as_ref() else {
        return None;
    };

    let mut map = HashMap::new();
    let inner_exprs = inner_p
        .expr
        .iter()
        .enumerate()
        .map(|(i, f)| match f {
            Expr::Alias(alias) => {
                let a = Expr::Column(alias.name.clone().into());
                map.insert(a.clone(), f.clone());
                a
            }
            Expr::Column(_) => {
                map.insert(
                    Expr::Column(inner_p.schema.field(i).name().into()),
                    f.clone(),
                );
                f.clone()
            }
            _ => {
                let a = Expr::Column(inner_p.schema.field(i).name().into());
                map.insert(a.clone(), f.clone());
                a
            }
        })
        .collect::<Vec<_>>();

    // Compare outer collects Expr::to_string with inner collected transformed values
    // alias -> alias column
    // column -> remain
    // others, extract schema field name
    let inner_collects = inner_exprs
        .iter()
        .map(Expr::to_string)
        .collect::<HashSet<_>>();

    let mut collects = p.expr.clone();
    for sort in &sort.expr {
        // Strip aliases from sort expressions so the comparison matches
        // the inner Projection's raw expressions. The optimizer may add
        // sort expressions to the inner Projection without aliases, while
        // the Sort node's expressions carry aliases from the original plan.
        let mut expr = sort.expr.clone();
        while let Expr::Alias(alias) = expr {
            expr = *alias.expr;
        }
        if inner_collects.contains(&expr.to_string()) {
            collects.push(expr);
            continue;
        }
        // The sort key is not itself one of the inner Projection's outputs, but
        // it may be an expression *over* them (`ORDER BY age + 1` while the
        // inner Projection exposes `age`). Account for the columns it reads so
        // an inner Projection that exists only to expose them still matches.
        // Without this the rewrite bails out and the Sort is emitted as a
        // derived table, where SQL does not guarantee the ORDER BY is honoured
        // by the enclosing query -- the rows come back in an arbitrary order.
        collects.extend(expr.column_refs().into_iter().cloned().map(Expr::Column));
    }

    let outer_collects = collects.iter().map(Expr::to_string).collect::<HashSet<_>>();

    if outer_collects == inner_collects {
        let mut sort = sort.clone();
        let mut inner_p = inner_p.clone();

        let new_exprs = p
            .expr
            .iter()
            .map(|e| map.get(e).unwrap_or(e).clone())
            .collect::<Vec<_>>();

        // The inner Projection may define aliases that the Sort references
        // but the outer Projection does not include.  Since we are about to
        // replace the inner Projection's expressions with `new_exprs` (which
        // only contains the outer Projection's columns), those alias
        // definitions will be lost.  To keep the Sort valid, rewrite any
        // sort expression that references a dropped alias so that it uses
        // the alias's underlying expression instead.
        let projected_aliases: HashSet<&str> = new_exprs
            .iter()
            .filter_map(|e| match e {
                Expr::Alias(alias) => Some(alias.name.as_str()),
                _ => None,
            })
            .collect();

        let dropped_aliases: HashMap<String, Expr> = inner_p
            .expr
            .iter()
            .filter_map(|e| match e {
                Expr::Alias(alias)
                    if !projected_aliases.contains(alias.name.as_str()) =>
                {
                    Some((alias.name.clone(), (*alias.expr).clone()))
                }
                _ => None,
            })
            .collect();

        if !dropped_aliases.is_empty() {
            for sort_expr in &mut sort.expr {
                let mut expr = sort_expr.expr.clone();
                while let Expr::Alias(alias) = expr {
                    expr = *alias.expr;
                }
                // Substitute nested references too, not just a whole-expression
                // one: `ORDER BY x + 1` over a dropped `expr AS x` has to become
                // `ORDER BY expr + 1`.
                sort_expr.expr = expr
                    .clone()
                    .transform_down(|e| {
                        Ok(match &e {
                            // Only an unqualified column can name a projection
                            // alias: `t.x` refers to `t`'s own column even when a
                            // dropped alias happens to share the name.
                            Expr::Column(col) if col.relation.is_none() => {
                                match dropped_aliases.get(col.name()) {
                                    Some(underlying) => {
                                        Transformed::yes(underlying.clone())
                                    }
                                    None => Transformed::no(e),
                                }
                            }
                            _ => Transformed::no(e),
                        })
                    })
                    .map_or(expr, |transformed| transformed.data);
            }
        }

        inner_p.expr.clone_from(&new_exprs);
        sort.input = Arc::new(LogicalPlan::Projection(inner_p));

        Some(LogicalPlan::Sort(sort))
    } else {
        None
    }
}

/// This logic is to work out the columns and inner query for SubqueryAlias plan for some types of
/// subquery or unnest
/// - `(SELECT column_a as a from table) AS A`
/// - `(SELECT column_a from table) AS A (a)`
/// - `SELECT * FROM t1 CROSS JOIN UNNEST(t1.c1) AS u(c1)` (see [find_unnest_column_alias])
///
/// A roundtrip example for table alias with columns
///
/// query: SELECT id FROM (SELECT j1_id from j1) AS c (id)
///
/// LogicPlan:
/// Projection: c.id
///   SubqueryAlias: c
///     Projection: j1.j1_id AS id
///       Projection: j1.j1_id
///         TableScan: j1
///
/// Before introducing this logic, the unparsed query would be `SELECT c.id FROM (SELECT j1.j1_id AS
/// id FROM (SELECT j1.j1_id FROM j1)) AS c`.
/// The query is invalid as `j1.j1_id` is not a valid identifier in the derived table
/// `(SELECT j1.j1_id FROM j1)`
///
/// With this logic, the unparsed query will be:
/// `SELECT c.id FROM (SELECT j1.j1_id FROM j1) AS c (id)`
///
/// Caveat: this won't handle the case like `select * from (select 1, 2) AS a (b, c)`
/// as the parser gives a wrong plan which has mismatch `Int(1)` types: Literal and
/// Column in the Projections. Once the parser side is fixed, this logic should work
pub(super) fn subquery_alias_inner_query_and_columns(
    subquery_alias: &datafusion_expr::SubqueryAlias,
) -> (&LogicalPlan, Vec<Ident>) {
    let plan: &LogicalPlan = subquery_alias.input.as_ref();

    if let LogicalPlan::Subquery(subquery) = plan {
        let (inner_projection, Some(column)) =
            find_unnest_column_alias(subquery.subquery.as_ref())
        else {
            return (plan, vec![]);
        };
        return (inner_projection, vec![Ident::new(column)]);
    }

    let LogicalPlan::Projection(outer_projections) = plan else {
        return (plan, vec![]);
    };

    // Check if it's projection inside projection
    let Some(inner_projection) = find_projection(outer_projections.input.as_ref()) else {
        return (plan, vec![]);
    };

    let mut columns: Vec<Ident> = vec![];
    // Check if the inner projection and outer projection have a matching pattern like
    //     Projection: j1.j1_id AS id
    //       Projection: j1.j1_id
    if outer_projections.expr.len() != inner_projection.expr.len() {
        return (plan, vec![]);
    }

    for (i, inner_expr) in inner_projection.expr.iter().enumerate() {
        let Expr::Alias(outer_alias) = &outer_projections.expr[i] else {
            return (plan, vec![]);
        };

        // Inner projection schema fields store the projection name which is used in outer
        // projection expr
        let inner_expr_string = match inner_expr {
            Expr::Column(_) => inner_expr.to_string(),
            _ => inner_projection.schema.field(i).name().clone(),
        };

        if outer_alias.expr.to_string() != inner_expr_string {
            return (plan, vec![]);
        };

        columns.push(outer_alias.name.as_str().into());
    }

    (outer_projections.input.as_ref(), columns)
}

/// Try to find the column alias for UNNEST in the inner projection.
/// For example:
/// ```sql
///     SELECT * FROM t1 CROSS JOIN UNNEST(t1.c1) AS u(c1)
/// ```
/// The above query will be parsed into the following plan:
/// ```text
/// Projection: *
///   Cross Join:
///     SubqueryAlias: t1
///       TableScan: t
///     SubqueryAlias: u
///       Subquery:
///         Projection: UNNEST(outer_ref(t1.c1)) AS c1
///           Projection: __unnest_placeholder(outer_ref(t1.c1),depth=1) AS UNNEST(outer_ref(t1.c1))
///             Unnest: lists[__unnest_placeholder(outer_ref(t1.c1))|depth=1] structs[]
///               Projection: outer_ref(t1.c1) AS __unnest_placeholder(outer_ref(t1.c1))
///                 EmptyRelation
/// ```
/// The function will return the inner projection and the column alias `c1` if the column name
/// starts with `UNNEST(` (the `Display` result of [Expr::Unnest]) in the inner projection.
pub(super) fn find_unnest_column_alias(
    plan: &LogicalPlan,
) -> (&LogicalPlan, Option<String>) {
    if let LogicalPlan::Projection(projection) = plan {
        if projection.expr.len() != 1 {
            return (plan, None);
        }
        if let Some(Expr::Alias(alias)) = projection.expr.first()
            && alias
                .expr
                .schema_name()
                .to_string()
                .starts_with(&format!("{UNNEST_COLUMN_PREFIX}("))
        {
            return (projection.input.as_ref(), Some(alias.name.clone()));
        }
    }
    (plan, None)
}

/// Injects column aliases into a subquery's logical plan. The function searches for a `Projection`
/// within the given plan, which may be wrapped by other operators (e.g., LIMIT, SORT).
/// If the top-level plan is a `Projection`, it directly injects the column aliases.
/// Otherwise, it iterates through the plan's children to locate and transform the `Projection`.
///
/// Example:
/// - `SELECT col1, col2 FROM table LIMIT 10` plan with aliases `["alias_1", "some_alias_2"]` will be transformed to
/// - `SELECT col1 AS alias_1, col2 AS some_alias_2 FROM table LIMIT 10`
pub(super) fn inject_column_aliases_into_subquery(
    plan: LogicalPlan,
    aliases: Vec<Ident>,
) -> Result<LogicalPlan> {
    match &plan {
        LogicalPlan::Projection(inner_p) => Ok(inject_column_aliases(inner_p, aliases)),
        _ => {
            // projection is wrapped by other operator (LIMIT, SORT, etc), iterate through the plan to find it
            plan.map_children(|child| {
                if let LogicalPlan::Projection(p) = &child {
                    Ok(Transformed::yes(inject_column_aliases(p, aliases.clone())))
                } else {
                    Ok(Transformed::no(child))
                }
            })
            .map(|plan| plan.data)
        }
    }
}

/// Injects column aliases into the projection of a logical plan by wrapping expressions
/// with `Expr::Alias` using the provided list of aliases.
///
/// Example:
/// - `SELECT col1, col2 FROM table` with aliases `["alias_1", "some_alias_2"]` will be transformed to
/// - `SELECT col1 AS alias_1, col2 AS some_alias_2 FROM table`
pub(super) fn inject_column_aliases(
    projection: &Projection,
    aliases: impl IntoIterator<Item = Ident>,
) -> LogicalPlan {
    let mut updated_projection = projection.clone();

    let new_exprs = updated_projection
        .expr
        .into_iter()
        .zip(aliases)
        .map(|(expr, col_alias)| {
            let relation = match &expr {
                Expr::Column(col) => col.relation.clone(),
                _ => None,
            };

            Expr::Alias(Alias {
                expr: Box::new(expr.clone()),
                relation,
                name: col_alias.value,
                metadata: None,
            })
        })
        .collect::<Vec<_>>();

    updated_projection.expr = new_exprs;

    LogicalPlan::Projection(updated_projection)
}

fn find_projection(logical_plan: &LogicalPlan) -> Option<&Projection> {
    match logical_plan {
        LogicalPlan::Projection(p) => Some(p),
        LogicalPlan::Limit(p) => find_projection(p.input.as_ref()),
        LogicalPlan::Distinct(p) => find_projection(p.input().as_ref()),
        LogicalPlan::Sort(p) => find_projection(p.input.as_ref()),
        _ => None,
    }
}

/// A `TreeNodeRewriter` implementation that rewrites `Expr::Column` expressions by
/// replacing the column's name with an alias if the column exists in the provided schema.
///
/// This is typically used to apply table aliases in query plans, ensuring that
/// the column references in the expressions use the correct table alias.
///
/// # Fields
///
/// * `table_schema`: The schema (`SchemaRef`) representing the table structure
///   from which the columns are referenced. This is used to look up columns by their names.
/// * `alias_name`: The alias (`TableReference`) that will replace the table name
///   in the column references when applicable.
pub struct TableAliasRewriter<'a> {
    pub table_schema: &'a Schema,
    pub alias_name: TableReference,
}

impl TreeNodeRewriter for TableAliasRewriter<'_> {
    type Node = Expr;

    fn f_down(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        match expr {
            Expr::Column(column) => {
                if let Ok(field) = self.table_schema.field_with_name(&column.name) {
                    let new_column =
                        Column::new(Some(self.alias_name.clone()), field.name().clone());
                    Ok(Transformed::yes(Expr::Column(new_column)))
                } else {
                    Ok(Transformed::no(Expr::Column(column)))
                }
            }
            _ => Ok(Transformed::no(expr)),
        }
    }
}

/// Re-points a column reference at the derived table that replaced the relation it names.
///
/// When a sub-plan is unparsed as a derived table, the SELECT above it stops reading from
/// the relation its expressions are qualified by and reads from that derived table instead.
/// A qualifier naming the original relation then binds to nothing, so a strict remote binder
/// rejects the query even though DataFusion re-plans it.
///
/// A reference is rewritten only when its qualifier names one of the relations the derived
/// table encloses (`derived_qualifiers`), so a reference to a relation the SELECT still
/// reads directly — the other side of a join, say — keeps the qualifier it needs. With
/// `Some(alias)` the column is then addressed through `alias`, which the derived table
/// carries, rather than reduced to a bare name: bare would be ambiguous wherever the derived
/// table is not the SELECT's only relation. `None` is for a derived table that *is* the
/// SELECT's only relation and carries no alias: its outputs are addressed by name alone.
///
/// Both tests are on names alone, so neither distinguishes a correlated reference to an
/// enclosing query — that qualifier can name the very same relation. A caller must not
/// offer one, which is why [`SelectBuilder::visit_expressions_in_clauses_mut`] skips any
/// expression holding a subquery.
///
/// [`SelectBuilder::visit_expressions_in_clauses_mut`]: super::ast::SelectBuilder::visit_expressions_in_clauses_mut
pub fn requalify_column_onto_derived_table(
    idents: &mut Vec<Ident>,
    derived_qualifiers: &HashSet<String>,
    alias: Option<&Ident>,
) {
    let Some((last, qualifier)) = idents.split_last() else {
        return;
    };
    if qualifier.is_empty() {
        return;
    }
    let qualifier = qualifier
        .iter()
        .map(|ident| ident.value.clone())
        .collect::<Vec<String>>()
        .join(".");
    if !derived_qualifiers.contains(&qualifier) {
        return;
    }
    let last = last.clone();
    *idents = match alias {
        Some(alias) => vec![alias.clone(), last],
        None => vec![last],
    };
}

/// The relations an emitted `SELECT` can name.
///
/// Every qualifier its `FROM` makes visible, spelled as the identifier parts a column
/// reference uses for it: an aliased relation is visible only as its alias; a bare
/// table as its last name part, the form [`super::Unparser::emitted_qualifier`]
/// writes by default, and as its full name for a dialect that writes every part.
/// A correlated reference binds in an enclosing query rather than in this `FROM`, so
/// the qualifiers the plan's outer references carry count as visible too.
#[derive(Debug, Default)]
pub(super) struct VisibleScope {
    relations: Vec<Vec<String>>,
}

impl VisibleScope {
    /// Everything `factor`, a relation of the emitted `FROM`, makes visible.
    pub(super) fn add_relation(&mut self, factor: &ast::TableFactor) {
        match factor {
            ast::TableFactor::Table { name, alias, .. } => match alias {
                Some(alias) => self.add_alias(alias),
                None => self.add_table_name(name),
            },
            ast::TableFactor::Derived { alias, .. }
            | ast::TableFactor::TableFunction { alias, .. }
            | ast::TableFactor::Function { alias, .. }
            | ast::TableFactor::UNNEST { alias, .. }
            | ast::TableFactor::JsonTable { alias, .. }
            | ast::TableFactor::OpenJsonTable { alias, .. }
            | ast::TableFactor::NestedJoin { alias, .. }
            | ast::TableFactor::Pivot { alias, .. }
            | ast::TableFactor::Unpivot { alias, .. }
            | ast::TableFactor::MatchRecognize { alias, .. }
            | ast::TableFactor::XmlTable { alias, .. }
            | ast::TableFactor::SemanticView { alias, .. } => {
                if let Some(alias) = alias {
                    self.add_alias(alias);
                }
            }
        }
    }

    /// An aliased relation is visible as its alias alone.
    pub(super) fn add_alias(&mut self, alias: &ast::TableAlias) {
        self.relations.push(vec![alias.name.value.clone()]);
    }

    /// A qualifier that binds outside this `SELECT`: the relation an outer reference
    /// reads from, spelled as the unparser emits it.
    pub(super) fn add_outer_qualifier(&mut self, parts: Vec<String>) {
        if !parts.is_empty() {
            self.relations.push(parts);
        }
    }

    /// A bare table is visible as its last name part and, for a dialect that writes
    /// every part of a qualifier, as its full name.
    pub(super) fn add_table_name(&mut self, name: &ast::ObjectName) {
        let parts: Vec<String> = name
            .0
            .iter()
            .filter_map(|part| part.as_ident().map(|ident| ident.value.clone()))
            .collect();
        if let Some(last) = parts.last() {
            self.relations.push(vec![last.clone()]);
        }
        if parts.len() > 1 {
            self.relations.push(parts);
        }
    }

    /// Whether `idents` is a qualifier this scope makes visible followed by exactly
    /// one column name. A shorter visible name that merely prefixes the qualifier does
    /// not count: an alias `d` does not make `d.t.id` bind when `d.t` is enclosed.
    fn names_visible_relation(&self, idents: &[Ident]) -> bool {
        self.relations.iter().any(|relation| {
            relation.len() + 1 == idents.len()
                && relation
                    .iter()
                    .zip(idents)
                    .all(|(part, ident)| *part == ident.value)
        })
    }
}

/// Strips a relation qualifier the enclosing `FROM` does not make visible from a
/// column reference, so the statement binds in an engine that resolves qualifiers
/// against the `FROM` alone.
///
/// DataFusion keeps a derived table's qualifiers on its output columns, so a
/// projection over an un-aliased derived table — or one aliased under another name —
/// carries `ta.j1_id` with no `ta` in the `FROM`, and PostgreSQL, MySQL and SQLite
/// refuse the statement.
///
/// A [`ast::Expr::CompoundIdentifier`] here is a qualifier followed by one column
/// name, whatever the dialect's qualifier width: the unparser emits a struct field
/// path as an [`ast::Expr::CompoundFieldAccess`] whose root is the column reference,
/// so the path is never mistaken for part of the qualifier and stays as it is while
/// the root is judged on its own. A reference left with one part becomes a plain
/// identifier, which is what the parser produces for it and what a planner handed the
/// AST directly expects.
pub(super) fn strip_dangling_qualifier(expr: &mut ast::Expr, scope: &VisibleScope) {
    match expr {
        ast::Expr::CompoundFieldAccess { root, .. } => {
            strip_dangling_qualifier(root, scope);
        }
        ast::Expr::CompoundIdentifier(idents) => {
            // `@root.name` is a scalar variable, not a qualified column (the planner
            // draws the same line on an unquoted leading `@`).
            let is_variable = idents.first().is_some_and(|root| {
                root.value.starts_with('@') && root.quote_style.is_none()
            });
            if idents.len() < 2 || is_variable || scope.names_visible_relation(idents) {
                return;
            }
            if let Some(column) = idents.pop() {
                *expr = ast::Expr::Identifier(column);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use datafusion_expr::{LogicalPlanBuilder, col, table_scan};
    use sqlparser::dialect::GenericDialect;
    use sqlparser::parser::Parser;

    fn idents(parts: &[&str]) -> Vec<Ident> {
        parts.iter().map(|part| Ident::new(*part)).collect()
    }

    fn compound(parts: &[&str]) -> ast::Expr {
        ast::Expr::CompoundIdentifier(idents(parts))
    }

    fn field_path(root: ast::Expr, fields: &[&str]) -> ast::Expr {
        ast::Expr::CompoundFieldAccess {
            root: Box::new(root),
            access_chain: fields
                .iter()
                .map(|field| {
                    ast::AccessExpr::Dot(ast::Expr::Identifier(Ident::new(*field)))
                })
                .collect(),
        }
    }

    fn stripped(mut expr: ast::Expr, scope: &VisibleScope) -> ast::Expr {
        strip_dangling_qualifier(&mut expr, scope);
        expr
    }

    fn alias(name: &str) -> ast::TableAlias {
        ast::TableAlias {
            explicit: true,
            name: Ident::new(name),
            columns: vec![],
            at: None,
        }
    }

    #[test]
    fn a_qualifier_the_from_makes_visible_is_kept() {
        let mut scope = VisibleScope::default();
        scope.add_alias(&alias("ta"));
        assert_eq!(
            stripped(compound(&["ta", "j1_id"]), &scope),
            compound(&["ta", "j1_id"])
        );
    }

    #[test]
    fn a_dangling_qualifier_is_stripped_to_a_plain_identifier() {
        let scope = VisibleScope::default();
        assert_eq!(
            stripped(compound(&["ta", "j1_id"]), &scope),
            ast::Expr::Identifier(Ident::new("j1_id"))
        );
    }

    #[test]
    fn a_field_path_is_judged_by_its_root_and_kept_whole() {
        let mut scope = VisibleScope::default();
        scope.add_alias(&alias("ta"));
        let visible = field_path(compound(&["ta", "metadata"]), &["product", "name"]);
        assert_eq!(stripped(visible.clone(), &scope), visible);

        let dangling = field_path(compound(&["tb", "metadata"]), &["product", "name"]);
        assert_eq!(
            stripped(dangling, &scope),
            field_path(
                ast::Expr::Identifier(Ident::new("metadata")),
                &["product", "name"]
            )
        );
    }

    #[test]
    fn a_field_path_on_an_unqualified_column_needs_no_scope() {
        let scope = VisibleScope::default();
        let path = field_path(ast::Expr::Identifier(Ident::new("s")), &["a"]);
        assert_eq!(stripped(path.clone(), &scope), path);
    }

    #[test]
    fn a_bare_table_is_visible_by_its_last_name_part_and_by_its_full_name() {
        let mut scope = VisibleScope::default();
        scope.add_table_name(&ast::ObjectName::from(idents(&[
            "catalog", "schema", "table",
        ])));
        assert_eq!(
            stripped(compound(&["table", "id"]), &scope),
            compound(&["table", "id"])
        );
        assert_eq!(
            stripped(compound(&["catalog", "schema", "table", "id"]), &scope),
            compound(&["catalog", "schema", "table", "id"])
        );
    }

    #[test]
    fn a_dangling_qualifier_of_any_width_leaves_only_the_column() {
        let scope = VisibleScope::default();
        assert_eq!(
            stripped(compound(&["catalog", "schema", "table", "id"]), &scope),
            ast::Expr::Identifier(Ident::new("id"))
        );
    }

    #[test]
    fn a_derived_table_is_visible_only_through_its_alias() {
        let mut scope = VisibleScope::default();
        let query = Parser::new(&GenericDialect {})
            .try_with_sql("SELECT t.t, t.id FROM t")
            .expect("parser accepts the statement")
            .parse_query()
            .expect("statement is a query");
        scope.add_relation(&ast::TableFactor::Derived {
            lateral: false,
            subquery: query,
            alias: None,
            sample: None,
        });
        // The enclosed table is not visible, whatever its columns are named.
        assert_eq!(
            stripped(compound(&["t", "id"]), &scope),
            ast::Expr::Identifier(Ident::new("id"))
        );
    }

    #[test]
    fn a_visible_name_that_only_prefixes_the_qualifier_does_not_bind_it() {
        let mut scope = VisibleScope::default();
        scope.add_alias(&alias("derived_projection"));
        // `derived_projection.t` is the enclosed table; the alias alone is visible.
        assert_eq!(
            stripped(compound(&["derived_projection", "t", "id"]), &scope),
            ast::Expr::Identifier(Ident::new("id"))
        );
    }

    #[test]
    fn a_scalar_variable_is_not_a_qualified_column() {
        let scope = VisibleScope::default();
        let variable = compound(&["@root", "foo"]);
        assert_eq!(stripped(variable.clone(), &scope), variable);
        // Quoted, it is an ordinary (dangling) qualifier.
        let mut quoted = idents(&["@root", "foo"]);
        quoted[0].quote_style = Some('"');
        assert_eq!(
            stripped(ast::Expr::CompoundIdentifier(quoted), &scope),
            ast::Expr::Identifier(Ident::new("foo"))
        );
    }

    #[test]
    fn an_outer_references_qualifier_is_visible() {
        let mut scope = VisibleScope::default();
        scope.add_outer_qualifier(vec!["o".to_string()]);
        assert_eq!(
            stripped(compound(&["o", "id"]), &scope),
            compound(&["o", "id"])
        );
    }

    #[test]
    fn anything_but_a_column_reference_is_left_alone() {
        let scope = VisibleScope::default();
        let expr = ast::Expr::Identifier(Ident::new("j1_id"));
        assert_eq!(stripped(expr.clone(), &scope), expr);
    }

    // this is a regression test: when the outer projection has fewer expressions than
    // the inner projection, `subquery_alias_inner_query_and_columns` must not panic
    // with an index oob error
    // note: this happens when optimizer passes (e.g. CommonSubexprEliminate)
    // insert an inner projection with extra columns that a subsequent projection narrows
    // back down
    #[test]
    fn test_stacked_projections_mismatched_lengths_no_panic() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]);

        // Inner projection has 2 expressions, outer has 0 (empty).
        let inner_plan = LogicalPlanBuilder::from(
            table_scan(Some("t"), &schema, Some(vec![0, 1]))
                .unwrap()
                .build()
                .unwrap(),
        )
        .project(vec![col("t.id"), col("t.name")])
        .unwrap()
        .build()
        .unwrap();

        // Build an empty outer projection over the inner.
        let outer_plan = LogicalPlanBuilder::from(inner_plan)
            .project(Vec::<Expr>::new())
            .unwrap()
            .alias("sub")
            .unwrap()
            .build()
            .unwrap();

        let LogicalPlan::SubqueryAlias(subquery_alias) = &outer_plan else {
            panic!("expected SubqueryAlias");
        };

        // should return early without panicking
        let (_plan, columns) = subquery_alias_inner_query_and_columns(subquery_alias);
        assert!(columns.is_empty());
    }
}
