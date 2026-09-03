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

use arrow::datatypes::{DataType, Schema};
use datafusion_common::tree_node::TreeNodeContainer;
use datafusion_common::{
    Column, DFSchema, HashMap, Result, TableReference,
    tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRewriter},
};
use datafusion_expr::expr::{Alias, BinaryExpr, UNNEST_COLUMN_PREFIX};
use datafusion_expr::type_coercion::binary::comparison_coercion;
use datafusion_expr::{
    Expr, ExprSchemable, LogicalPlan, Operator, Projection, Sort, SortExpr,
};
use sqlparser::ast::Ident;

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
        .map_elements(&mut |expr: Expr| {
            expr.transform_up(|expr| {
                if let Expr::Column(mut col) = expr {
                    col.relation = None;
                    Ok(Transformed::yes(Expr::Column(col)))
                } else {
                    Ok(Transformed::no(expr))
                }
            })
        })
        .data()?;

    Ok(sort_exprs)
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
/// reads directly — the other side of a join, say — keeps the qualifier it needs. The
/// column is then addressed through `alias`, which the derived table always carries, rather
/// than reduced to a bare name: bare would be ambiguous wherever the derived table is not
/// the SELECT's only relation.
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
    alias: &Ident,
) {
    if idents.len() < 2 {
        return;
    }
    let qualifier = idents
        .iter()
        .take(idents.len() - 1)
        .map(|ident| ident.value.clone())
        .collect::<Vec<String>>()
        .join(".");
    if !derived_qualifiers.contains(&qualifier) {
        return;
    }
    let Some(last) = idents.last() else {
        unreachable!("CompoundIdentifier must have a last element");
    };
    *idents = vec![alias.clone(), last.clone()];
}

/// Takes an input list of identifiers and a list of identifiers that are available from relations or joins.
/// Removes any table identifiers that are not present in the list of available identifiers, retains original column names.
pub fn remove_dangling_identifiers(
    idents: &mut Vec<Ident>,
    available_idents: &Vec<String>,
) {
    if idents.len() > 1 {
        // sqlparser 0.61 made `display_separated` pub(crate); join via Display instead.
        let ident_source = idents
            .iter()
            .take(idents.len() - 1)
            .map(ToString::to_string)
            .collect::<Vec<String>>()
            .join(".");
        // If the identifier is not present in the list of all identifiers, it refers to a table that does not exist
        if !available_idents.contains(&ident_source) {
            let Some(last) = idents.last() else {
                unreachable!("CompoundIdentifier must have a last element");
            };
            // Reset the identifiers to only the last element, which is the column name
            *idents = vec![last.clone()];
        }
    }
}

/// Makes the date type of an operand visible in the expression itself, so the
/// dialect can render date arithmetic without a schema of its own.
///
/// Expression unparsing has no schema, so it can only act on types an expression
/// carries — an explicit cast, or a literal. That is enough for a subtraction of
/// two explicit date casts, and not enough for `CAST(a AS DATE) - b` or for an
/// integer cast of a bare date column, where the type lives only in the schema.
///
/// A plan node does have the schema its expressions resolve against, so this pass
/// wraps those operands in a cast to the date type they already have. The cast is
/// a no-op — the operand is that type — but it puts the type where the unparser
/// can read it.
///
/// Only date operands are wrapped, and only under a subtraction or an integer
/// cast, so nothing else changes shape. Any expression whose type cannot be
/// resolved is left exactly as it was.
pub(super) fn expose_date_operand_types(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_up(|plan| {
        // Nothing here is required for correct SQL — it only lets the unparser see
        // a type it would otherwise miss — so a node whose schema will not
        // combine is skipped rather than failed. Such a schema is ambiguous
        // anyway, and its own unparsing reports that far better than this pass
        // could.
        let Some(schema) = combined_input_schema(&plan) else {
            return Ok(Transformed::no(plan));
        };

        plan.map_expressions(|expr| {
            expr.transform_up(|expr| Ok(expose_in_expr(expr, &schema)))
        })
    })
    .data()
}

/// The schema this node's expressions resolve against: its inputs' fields, which
/// for a join is both sides.
///
/// `None` when the inputs' fields will not combine, such as two branches that
/// both expose the same unqualified name.
fn combined_input_schema(plan: &LogicalPlan) -> Option<DFSchema> {
    let inputs = plan.inputs();
    // A leaf carries its expressions itself — a scan's pushed-down filters — and
    // they resolve against its own schema. A column the scan does not expose
    // stays unresolved, which leaves the expression as it was.
    if inputs.is_empty() {
        return Some(plan.schema().as_ref().clone());
    }

    let mut schema = DFSchema::empty();
    for input in inputs {
        schema = schema.join(input.schema()).ok()?;
    }
    Some(schema)
}

/// Wraps `operand` in a cast to its own type when that type is one the dialect
/// dispatches on, so the type travels with the expression.
///
/// A cast or a literal already states its type, so re-wrapping it would only add
/// noise. Any expression whose type does not resolve is returned untouched.
fn expose_type(operand: Expr, schema: &DFSchema) -> (Expr, bool) {
    if matches!(
        operand,
        Expr::Cast(_) | Expr::TryCast(_) | Expr::Literal(..)
    ) {
        return (operand, false);
    }
    match operand.get_type(schema) {
        Ok(
            data_type @ (DataType::Date32
            | DataType::Timestamp(_, _)
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64),
        ) => (
            Expr::Cast(datafusion_expr::expr::Cast::new(
                Box::new(operand),
                data_type,
            )),
            true,
        ),
        _ => (operand, false),
    }
}

/// As [`expose_type`], restricted to the date types, so date arithmetic never
/// gains a cast around an operand that is not a date.
fn expose_date_type(operand: Expr, schema: &DFSchema) -> (Expr, bool) {
    match operand.get_type(schema) {
        Ok(DataType::Date32) => expose_type(operand, schema),
        _ => (operand, false),
    }
}

/// Whether `expr` resolves to one of Arrow's date types against `schema`.
fn is_date(expr: &Expr, schema: &DFSchema) -> bool {
    matches!(expr.get_type(schema), Ok(DataType::Date32))
}

/// Wraps a bare date operand in a cast to its own type. Returns the expression
/// untouched when it is not one of the two shapes that need it.
fn expose_in_expr(expr: Expr, schema: &DFSchema) -> Transformed<Expr> {
    let expose_date =
        |operand: Expr| -> (Expr, bool) { expose_date_type(operand, schema) };

    match expr {
        // `date - date` is a day count, but only if both sides say they are dates.
        Expr::BinaryExpr(BinaryExpr { left, op, right })
            if matches!(op, Operator::Minus)
                && is_date(&left, schema)
                && is_date(&right, schema) =>
        {
            let (left, left_changed) = expose_date(*left);
            let (right, right_changed) = expose_date(*right);
            let rewritten =
                Expr::BinaryExpr(BinaryExpr::new(Box::new(left), op, Box::new(right)));
            if left_changed || right_changed {
                Transformed::yes(rewritten)
            } else {
                Transformed::no(rewritten)
            }
        }
        // `to_unixtime` and `to_timestamp` render differently per operand type,
        // so the operand has to say what it is.
        Expr::ScalarFunction(mut function)
            if matches!(function.name(), "to_unixtime" | "to_timestamp") =>
        {
            let mut changed = false;
            function.args = function
                .args
                .into_iter()
                .map(|arg| {
                    let (arg, arg_changed) = expose_type(arg, schema);
                    changed |= arg_changed;
                    arg
                })
                .collect();
            let rewritten = Expr::ScalarFunction(function);
            if changed {
                Transformed::yes(rewritten)
            } else {
                Transformed::no(rewritten)
            }
        }
        // `CAST(date AS INT64)` is the same day count, spelled as a cast.
        Expr::Cast(cast) if cast.field.data_type().is_integer() => {
            let (inner, changed) = expose_date(*cast.expr);
            let rewritten = Expr::Cast(datafusion_expr::expr::Cast::new(
                Box::new(inner),
                cast.field.data_type().clone(),
            ));
            if changed {
                Transformed::yes(rewritten)
            } else {
                Transformed::no(rewritten)
            }
        }
        other => Transformed::no(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use datafusion_expr::{LogicalPlanBuilder, col, table_scan};

    #[test]
    fn test_remove_dangling_identifiers() {
        let tests = vec![
            (vec![], vec![Ident::new("column1".to_string())]),
            (
                vec!["table1.table2".to_string()],
                vec![
                    Ident::new("table1".to_string()),
                    Ident::new("table2".to_string()),
                    Ident::new("column1".to_string()),
                ],
            ),
            (
                vec!["table1".to_string()],
                vec![Ident::new("column1".to_string())],
            ),
        ];

        for test in tests {
            let test_in = test.0;
            let test_out = test.1;

            let mut idents = vec![
                Ident::new("table1".to_string()),
                Ident::new("table2".to_string()),
                Ident::new("column1".to_string()),
            ];

            remove_dangling_identifiers(&mut idents, &test_in);
            assert_eq!(idents, test_out);
        }
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

/// Makes a comparison's two operands agree when the dialect will not do it
/// itself.
///
/// DataFusion leaves a mismatched comparison in the plan and coerces at
/// execution: it reads a tz-naive timestamp against a tz-aware one as UTC, and a
/// string against a number by reading the string as that number. An engine that
/// has no common supertype for such a pair refuses the whole statement instead —
/// BigQuery says "No matching signature for operator >= for argument types:
/// DATETIME, TIMESTAMP", or "... INT64, STRING".
///
/// The target type is DataFusion's own [`comparison_coercion`], so the rewrite
/// says what the plan already meant rather than choosing for it. Only the two
/// pairs an engine is known to refuse are touched; a mismatch it coerces happily,
/// such as `INT64` against `FLOAT64`, is left alone rather than dressed up.
pub(super) fn unify_comparison_operands(plan: LogicalPlan) -> Result<LogicalPlan> {
    plan.transform_up(|plan| {
        let Some(schema) = combined_input_schema(&plan) else {
            return Ok(Transformed::no(plan));
        };

        plan.map_expressions(|expr| {
            expr.transform_up(|expr| Ok(unify_in_expr(expr, &schema)))
        })
    })
    .data()
}

/// Whether the engine is known to refuse this pair outright rather than coerce it.
fn needs_saying_out_loud(left: &DataType, right: &DataType) -> bool {
    let timestamp_zone_disagreement = matches!(
        (left, right),
        (
            DataType::Timestamp(_, None),
            DataType::Timestamp(_, Some(_))
        ) | (
            DataType::Timestamp(_, Some(_)),
            DataType::Timestamp(_, None)
        )
    );
    let is_text = |t: &DataType| {
        matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
    };
    let number_against_text =
        (left.is_numeric() && is_text(right)) || (is_text(left) && right.is_numeric());

    timestamp_zone_disagreement || number_against_text
}

fn unify_in_expr(expr: Expr, schema: &DFSchema) -> Transformed<Expr> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = expr else {
        return Transformed::no(expr);
    };

    let rebuilt = || Expr::BinaryExpr(BinaryExpr::new(left.clone(), op, right.clone()));

    if !matches!(
        op,
        Operator::Eq
            | Operator::NotEq
            | Operator::Lt
            | Operator::LtEq
            | Operator::Gt
            | Operator::GtEq
    ) {
        return Transformed::no(rebuilt());
    }

    let (Ok(left_type), Ok(right_type)) = (left.get_type(schema), right.get_type(schema))
    else {
        return Transformed::no(rebuilt());
    };

    if !needs_saying_out_loud(&left_type, &right_type) {
        return Transformed::no(rebuilt());
    }

    let Some(common) = comparison_coercion(&left_type, &right_type) else {
        return Transformed::no(rebuilt());
    };

    let converge = |operand: &Expr, operand_type: DataType| -> Box<Expr> {
        if operand_type == common {
            return Box::new(operand.clone());
        }
        Box::new(Expr::Cast(datafusion_expr::expr::Cast::new(
            Box::new(operand.clone()),
            common.clone(),
        )))
    };

    Transformed::yes(Expr::BinaryExpr(BinaryExpr::new(
        converge(&left, left_type),
        op,
        converge(&right, right_type),
    )))
}
