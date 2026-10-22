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

use arrow::datatypes::DataType;
use datafusion_common::tree_node::TreeNodeContainer;
use datafusion_common::{
    Column, DFSchema, HashMap, Result, TableReference,
    tree_node::{
        Transformed, TransformedResult, TreeNode, TreeNodeRecursion, TreeNodeRewriter,
    },
};
use datafusion_expr::expr::{Alias, UNNEST_COLUMN_PREFIX};
use datafusion_expr::{Distinct, Expr, LogicalPlan, Projection, Sort, SortExpr};
use sqlparser::ast::{self, Ident};

use super::Unparser;
use super::utils::output_is_repeatable;

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
            // Only rewrite Sort expressions that read a UNION's output: a
            // `UNION ALL`, or a `UNION` (`Distinct::All` over the union)
            if !is_union_output(&sort.input) {
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

/// Whether `plan` is a `UNION ALL` or a `UNION`, whose output columns carry no
/// relation an `ORDER BY` above them could name. A `UNION` is any chain of
/// `Distinct::All` over the union: the unparser folds them all into the one
/// set operation.
fn is_union_output(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Union(_) => true,
        LogicalPlan::Distinct(Distinct::All(input)) => is_union_output(input),
        _ => false,
    }
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
    unparser: &Unparser,
    p: &Projection,
) -> Option<LogicalPlan> {
    let LogicalPlan::Sort(sort) = p.input.as_ref() else {
        return None;
    };

    let LogicalPlan::Projection(inner_p) = sort.input.as_ref() else {
        return None;
    };

    // The fold reads an output by the name its expression gives it, so it is
    // taken only where that is the name the schema declares for it: a
    // projection built with a declared schema that renames an alias or a column
    // (`(t.a + 1) AS x` declared `y`) would have it substitute one output for
    // another.
    if !outputs_named_as_declared(inner_p) {
        return None;
    }

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

    // The inner output each outer expression reads, matched by the spelling the
    // comparison below uses, so every expression the fold accepts finds its own.
    let position = inner_exprs
        .iter()
        .enumerate()
        .map(|(i, e)| (e.to_string(), i))
        .collect::<HashMap<_, _>>();
    // That spelling is not a unique identity: `t."a.b"` and an alias `"t.a.b"`
    // both display as `t.a.b`. Two outputs that share one leave the plan
    // unfolded, since the fold could not tell which of them an expression reads.
    if position.len() != inner_exprs.len() {
        return None;
    }

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

    // The inner projection may expose outputs that neither the outer projection
    // nor the sort reads. Dropping them changes nothing above, and leaving the
    // sort unhoisted over them puts its `ORDER BY` in a derived table, whose
    // order SQL does not carry out (spiceai/spiceai#14733). That widening of the
    // fold (a strict subset, where the fold has always required equal sets) is
    // taken only where it is safe by construction: see `widened_fold_is_safe`.
    // Anything else keeps the sort in its own scope, as it did before.
    if outer_collects.is_subset(&inner_collects)
        && (outer_collects == inner_collects
            || widened_fold_is_safe(p, inner_p, &position))
    {
        let mut sort = sort.clone();
        let inner_p = inner_p.clone();

        let new_exprs = p
            .expr
            .iter()
            .map(|e| map.get(e).unwrap_or(e).clone())
            .collect::<Vec<_>>();

        // The inner Projection may define outputs that the Sort references but
        // the outer Projection does not include — an alias, or an unnamed
        // expression the sort reads by its schema name. Since the inner
        // Projection's expressions are about to be replaced with `new_exprs`
        // (which only contains the outer Projection's columns), those outputs
        // will be lost. To keep the Sort valid, rewrite any sort expression that
        // references a dropped output so that it uses the underlying expression
        // instead.
        //
        // A projection built with a declared schema (`Projection::try_new_with_schema`)
        // names an output by its field rather than by its expression (`a + b`
        // declared `c`), and the kept output keeps that field, so the name it
        // is read by above is the one the folded projection's schema gives it.
        //
        // The inner Projection takes the outer one's expressions, and each is the
        // inner output the outer list reads, so it keeps that output's field: the
        // name the sort's keys and the inliner (`unproject_sort_expr` resolves a
        // key's column by position) look it up by, at the position the expression
        // takes in the outer list. Carrying the inner schema over unchanged would
        // name each output after the inner expression at the same index, `ORDER BY
        // x + 1` emitted over the expression at `x`'s inner index; taking the outer
        // schema would rename an output the outer list renames, so the key that
        // reads it by its inner name resolves to nothing. An output the outer list
        // repeats takes the outer field the second time, since one schema cannot
        // name two outputs alike. Fields that still collide leave the plan
        // unfolded: the outer schema names outputs by the outer list's
        // positions, so the inliner would resolve a key against the wrong one.
        //
        // A repeat named by an outer field that is the name of another inner
        // output (`x` repeated and declared `y` over an inner `y`) keeps that
        // output from being substituted, so a key reading it by that name would
        // read the repeat instead. Such a key leaves the plan unfolded too.
        //
        // A repeat of an output that cannot be repeated (`random() AS r` read
        // twice) would compute it once per copy, two draws where the plan copies
        // one value. That leaves the plan unfolded as well.
        let mut read = HashSet::new();
        let mut borrowed_names = HashSet::new();
        let mut fields = Vec::with_capacity(p.expr.len());
        for (i, e) in p.expr.iter().enumerate() {
            let reads = position.get(&e.to_string()).copied();
            let (schema, index) = match reads {
                Some(k) if read.insert(k) => (&inner_p.schema, k),
                Some(k) if !output_is_repeatable(&inner_p.expr[k]) => return None,
                _ => (&p.schema, i),
            };
            let (qualifier, field) = schema.qualified_field(index);
            if std::ptr::eq(schema, &p.schema)
                && inner_p
                    .schema
                    .fields()
                    .iter()
                    .enumerate()
                    .any(|(k, inner)| inner.name() == field.name() && Some(k) != reads)
            {
                borrowed_names.insert(field.name().clone());
            }
            fields.push((qualifier.cloned(), Arc::clone(&schema.fields()[index])));
        }
        let key_reads_borrowed_name =
            sort.expr.iter().any(|sort_expr| {
                sort_expr.expr.column_refs().iter().any(|col| {
                    col.relation.is_none() && borrowed_names.contains(col.name())
                })
            });
        if key_reads_borrowed_name {
            return None;
        }
        let schema = Arc::new(
            DFSchema::new_with_metadata(fields, inner_p.schema.metadata().clone())
                .ok()?,
        );
        let declared_names = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();
        // An unnamed kept expression is read by its declared field, not by the
        // name its expression spells: a dropped output declared under that
        // spelling (`t.c + 1` declared `t.a + t.b`) is still dropped.
        let kept_names: HashSet<String> = new_exprs
            .iter()
            .filter_map(|e| match e {
                Expr::Alias(alias) => Some(alias.name.clone()),
                _ => None,
            })
            .chain(declared_names.iter().cloned())
            .collect();

        let dropped_outputs: HashMap<String, Expr> = inner_p
            .expr
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                let (name, underlying) = match e {
                    Expr::Alias(alias) => (alias.name.clone(), (*alias.expr).clone()),
                    // A plain column keeps its own name wherever it is read.
                    Expr::Column(_) => return None,
                    other => (inner_p.schema.field(i).name().clone(), other.clone()),
                };
                (!kept_names.contains(&name)).then_some((name, underlying))
            })
            .collect();

        // A dropped output that cannot be repeated — a volatile expression, or
        // one holding a subquery — must not be substituted at more than one
        // reference the keys make to it: `(r - r)` over `random() AS r` is 0
        // drawn once and a difference of two draws when repeated. The fold is
        // skipped, and the sort keeps the scope it has, where the output is
        // computed once. A single reference evaluates it once either way.
        let mut unrepeatable_reads = HashMap::<String, usize>::new();
        for sort_expr in &sort.expr {
            let _ = sort_expr.expr.apply(|e| {
                if let Expr::Column(col) = e
                    && col.relation.is_none()
                    && dropped_outputs
                        .get(col.name())
                        .is_some_and(|underlying| !output_is_repeatable(underlying))
                {
                    *unrepeatable_reads
                        .entry(col.name().to_string())
                        .or_default() += 1;
                }
                Ok(TreeNodeRecursion::Continue)
            });
        }
        if unrepeatable_reads.values().any(|&reads| reads > 1) {
            return None;
        }
        // A substituted expression reads the scope below by bare column names.
        // Where such a name is also the name of a surviving alias (`0 - s AS s`
        // kept, `s AS x` dropped, the key reading `x`), the substituted `s`
        // would be bound to the alias by the unprojection that follows, and the
        // key would order by `0 - s` instead of `s`. The fold is skipped. A
        // computed output named by a declared schema (`a + b` declared `c`)
        // captures a substituted `c` the same way; a plain column kept under
        // its own name is the column the substitution reads. An unqualified
        // outer reference renders by its bare name too, so any output the fold
        // keeps under that name captures it the same way. A qualified read is
        // safe while its relation stays in scope, but an aggregate over a
        // derived projection, or a projection over one, renders `t.k` as the
        // bare `k`. A key that is that bare name alone binds to the kept
        // `(0 - t.k) AS k`; inside a compound key (`k + 1`) the name reads the
        // input column, so that key folds. Names are compared the way the
        // engine binds them: `t."K"` and `outer_ref(S)` render as `"K"` and
        // `"S"`, which `k` and `s` capture wherever identifiers fold case.
        let qualifiers_stripped = scope_strips_qualifiers(&inner_p.input);
        let mut kept_aliases = HashSet::new();
        for (e, declared) in new_exprs.iter().zip(&declared_names) {
            if let Expr::Alias(alias) = e {
                kept_aliases.insert(unparser.emitted_column_key(&alias.name).ok()?);
            }
            if !matches!(e, Expr::Column(column) if column.name() == declared.as_str()) {
                kept_aliases.insert(unparser.emitted_column_key(declared).ok()?);
            }
        }
        let kept_keys = declared_names
            .iter()
            .map(|name| unparser.emitted_column_key(name).ok())
            .collect::<Option<HashSet<_>>>()?;
        let substitution_shadowed = sort.expr.iter().any(|sort_expr| {
            let mut key = &sort_expr.expr;
            while let Expr::Alias(alias) = key {
                key = &alias.expr;
            }
            let bare_key = matches!(key, Expr::Column(_));
            sort_expr.expr.column_refs().iter().any(|col| {
                col.relation.is_none()
                    && dropped_outputs.get(col.name()).is_some_and(|underlying| {
                        underlying.column_refs().iter().any(|read| {
                            (read.relation.is_none() || (qualifiers_stripped && bare_key))
                                && unparser
                                    .emitted_column_key(read.name())
                                    .ok()
                                    .is_none_or(|key| kept_aliases.contains(&key))
                        }) || reads_outer_reference_named_in(
                            unparser, underlying, &kept_keys,
                        )
                    })
            })
        });
        if substitution_shadowed {
            return None;
        }

        if !dropped_outputs.is_empty() {
            for sort_expr in &mut sort.expr {
                let mut expr = sort_expr.expr.clone();
                while let Expr::Alias(alias) = expr {
                    expr = *alias.expr;
                }
                // Substitute nested references too, not just a whole-expression
                // one: `ORDER BY x + 1` over a dropped `expr AS x` has to become
                // `ORDER BY expr + 1`. Bottom-up, so the substituted expression
                // is not walked again: `(s + 1) AS s` reads an `s` of the scope
                // below, which a top-down walk would substitute without end.
                sort_expr.expr = expr
                    .clone()
                    .transform_up(|e| {
                        Ok(match &e {
                            // Only an unqualified column can name a projection
                            // output: `t.x` refers to `t`'s own column even when a
                            // dropped output happens to share the name.
                            Expr::Column(col) if col.relation.is_none() => {
                                match dropped_outputs.get(col.name()) {
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
            // A key that is now a literal orders nothing, and an integer literal
            // in `ORDER BY` would be read as a column position. Such a key is
            // dropped; a sort left with no key is not folded.
            sort.expr
                .retain(|sort_expr| !matches!(sort_expr.expr, Expr::Literal(..)));
            if sort.expr.is_empty() {
                return None;
            }
        }

        let inner_p = Projection::try_new_with_schema(
            new_exprs,
            Arc::clone(&inner_p.input),
            schema,
        )
        .ok()?;
        sort.input = Arc::new(LogicalPlan::Projection(inner_p));

        Some(LogicalPlan::Sort(sort))
    } else {
        None
    }
}

/// Whether the fold may drop inner outputs nobody above reads, which the fold
/// did not do before spiceai/spiceai#14733. It is taken only in a subset that
/// is safe by construction, rather than guarded shape by shape:
///
/// - every output the outer list keeps is a plain column, named by that column
///   in the outer schema as well (the inner one already is, see
///   [`outputs_named_as_declared`]), so the folded `SELECT` list introduces no
///   alias that a bare `ORDER BY` name, a substituted expression or an outer
///   reference could bind to, and the result keeps the plan's names;
/// - the outer list reads each kept output once;
/// - no inner output is volatile or holds a subquery, so substituting a dropped
///   output into a key evaluates nothing the plan evaluates once.
///
/// Outside it the sort keeps its own scope, as it did before the widening.
fn widened_fold_is_safe(
    p: &Projection,
    inner_p: &Projection,
    position: &HashMap<String, usize>,
) -> bool {
    if !inner_p.expr.iter().all(output_is_repeatable) {
        return false;
    }
    let mut kept = HashSet::new();
    p.expr.iter().enumerate().all(|(j, e)| {
        position.get(&e.to_string()).is_some_and(|&i| {
            matches!(&inner_p.expr[i], Expr::Column(column) if {
                let (qualifier, field) = p.schema.qualified_field(j);
                field.name() == &column.name && qualifier == column.relation.as_ref()
            }) && kept.insert(i)
        })
    })
}

/// Whether every aliased or plain-column output of `projection` carries the
/// name its schema declares for it, and any relation the expression names is
/// the declared qualifier. An unnamed computed output is read by its declared
/// name throughout the fold, so it always agrees.
fn outputs_named_as_declared(projection: &Projection) -> bool {
    projection.expr.len() == projection.schema.fields().len()
        && projection.expr.iter().enumerate().all(|(i, e)| {
            let (relation, name) = match e {
                Expr::Alias(alias) => (alias.relation.as_ref(), &alias.name),
                Expr::Column(column) => (column.relation.as_ref(), &column.name),
                _ => return true,
            };
            let (qualifier, field) = projection.schema.qualified_field(i);
            field.name() == name && relation.is_none_or(|r| Some(r) == qualifier)
        })
}

/// Whether `plan` renders in a scope that hides the relations its input
/// columns are qualified by, where the unparser strips their qualifiers: a
/// derived projection, or an aggregate whose input is one (see
/// `UnparserAggScope`).
fn scope_strips_qualifiers(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Aggregate(agg) => {
            Unparser::contains_projection_before_relation(&agg.input)
        }
        LogicalPlan::Projection(_) => true,
        LogicalPlan::TableScan(_)
        | LogicalPlan::Subquery(_)
        | LogicalPlan::SubqueryAlias(_)
        | LogicalPlan::Join(_)
        | LogicalPlan::EmptyRelation(_)
        | LogicalPlan::Values(_) => false,
        _ => matches!(
            plan.inputs().as_slice(),
            [input] if scope_strips_qualifiers(input)
        ),
    }
}

/// Whether `expr` holds an unqualified outer reference whose bare name folds to
/// one of `names` (keys from [`Unparser::emitted_column_key`]), so that a scope
/// projecting those names would bind it to its own output rather than to the
/// enclosing query.
fn reads_outer_reference_named_in(
    unparser: &Unparser,
    expr: &Expr,
    names: &HashSet<String>,
) -> bool {
    expr.exists(|e| {
        Ok(match e {
            Expr::OuterReferenceColumn(_, column) if column.relation.is_none() => {
                names.contains(&unparser.emitted_column_key(column.name())?)
            }
            _ => false,
        })
    })
    .unwrap_or(true)
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
/// replacing the column's qualifier with an alias if the column resolves to a
/// qualified field in the provided schema.
///
/// This is typically used to apply table aliases in query plans, ensuring that
/// the column references in the expressions use the correct table alias.
///
/// # Fields
///
/// * `table_schema`: The schema representing the table structure from which the
///   columns are referenced. This is used to look up columns by their names and qualifiers.
/// * `alias_name`: The alias (`TableReference`) that will replace the table name
///   in the column references when applicable.
/// * `rewrite_unqualified`: Whether columns that resolve to unqualified fields
///   in `table_schema` should also be rewritten to `alias_name`.
pub struct TableAliasRewriter<'a> {
    pub table_schema: &'a DFSchema,
    pub alias_name: TableReference,
    pub rewrite_unqualified: bool,
}

impl TreeNodeRewriter for TableAliasRewriter<'_> {
    type Node = Expr;

    fn f_down(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        match expr {
            Expr::Column(column) => {
                match self
                    .table_schema
                    .qualified_field_from_column(&column)
                    .or_else(|_| {
                        self.table_schema
                            .qualified_field_with_unqualified_name(&column.name)
                    }) {
                    Ok((qualifier, field))
                        if qualifier.is_some() || self.rewrite_unqualified =>
                    {
                        let new_column = Column::new(
                            Some(self.alias_name.clone()),
                            field.name().clone(),
                        );
                        Ok(Transformed::yes(Expr::Column(new_column)))
                    }
                    Ok(_) | Err(_) => Ok(Transformed::no(Expr::Column(column))),
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
/// table encloses (`derived_relations`), so a reference to a relation the SELECT still
/// reads directly — the other side of a join, say — keeps the qualifier it needs. With
/// `Some(alias)` the column is then addressed through `alias`, which the derived table
/// carries, rather than reduced to a bare name: bare would be ambiguous wherever the derived
/// table is not the SELECT's only relation. `None` is for a derived table that *is* the
/// SELECT's only relation and carries no alias: its outputs are addressed by name alone.
///
/// The qualifier is everything before the last identifier when that names an enclosed
/// relation, so a multi-part one (`db.schema.t`) is recognised whole. A reference that
/// goes on past the column — a field path, `t.payload.x` — has a shorter qualifier, and
/// is re-pointed only where the derived table is the SELECT's only relation (`None`):
/// there the qualifier is the longest leading run naming an enclosed relation through
/// whose columns the rest is a field path — a column, then a field of each struct in
/// turn — and the column and its path are kept. Beside another relation, a join's other
/// input, a shorter run cannot be told apart on names alone from that relation's own
/// name — `t.s.a` is as much column `a` of a relation `t.s` — so with `Some(alias)` only
/// the full qualifier is matched. Requiring the path to exist in the enclosed types keeps
/// a correlated reference to an enclosing `t.s` as it is unless `t` also has a struct
/// column `s` with a field `a`, where the SQL text itself reads either way.
///
/// Both tests are on names alone, so neither distinguishes a correlated reference to an
/// enclosing query — that qualifier can name the very same relation. A caller must not
/// offer one, which is why [`SelectBuilder::visit_expressions_in_clauses_mut`] skips any
/// expression holding a subquery.
///
/// [`SelectBuilder::visit_expressions_in_clauses_mut`]: super::ast::SelectBuilder::visit_expressions_in_clauses_mut
pub fn requalify_column_onto_derived_table(
    idents: &mut Vec<Ident>,
    derived_relations: &HashMap<String, HashMap<String, DataType>>,
    alias: Option<&Ident>,
) {
    let shortest = if alias.is_some() {
        idents.len().saturating_sub(1).max(1)
    } else {
        1
    };
    let qualifier_len = (shortest..idents.len()).rev().find(|&len| {
        let qualifier = idents[..len]
            .iter()
            .map(|ident| ident.value.clone())
            .collect::<Vec<String>>()
            .join(".");
        derived_relations.get(&qualifier).is_some_and(|columns| {
            len + 1 == idents.len() || is_field_path(columns, &idents[len..])
        })
    });
    let Some(qualifier_len) = qualifier_len else {
        return;
    };
    let rest = idents.split_off(qualifier_len);
    *idents = match alias {
        Some(alias) => std::iter::once(alias.clone()).chain(rest).collect(),
        None => rest,
    };
}

/// Whether `path` — a column, then a field of each struct in turn — names a field
/// path through `columns`.
fn is_field_path(columns: &HashMap<String, DataType>, path: &[Ident]) -> bool {
    let Some((column, fields)) = path.split_first() else {
        return false;
    };
    let Some(mut data_type) = columns.get(&column.value) else {
        return false;
    };
    for field in fields {
        let DataType::Struct(struct_fields) = data_type else {
            return false;
        };
        let Some(next) = struct_fields.iter().find(|f| *f.name() == field.value) else {
            return false;
        };
        data_type = next.data_type();
    }
    true
}

/// Takes an input list of identifiers and a list of identifiers that are available from relations or joins.
/// Removes any table identifiers that are not present in the list of available identifiers, retains original column names.
pub fn remove_dangling_identifiers(idents: &mut Vec<Ident>, available_idents: &[String]) {
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

/// [`remove_dangling_identifiers`] over a column reference, or over the column a
/// struct field path (`t.s.f`) is rooted at, keeping the path.
///
/// Only a path of `.field` steps is judged. That is the shape `get_field` is emitted
/// in, and its root is always a column of this query's own relations: a correlated
/// column is refused there. A subscript (`t.a[1]`) is left alone, as its root can be
/// a correlated reference, which binds in the enclosing query whatever this `FROM`
/// reads.
pub fn remove_dangling_qualifier(expr: &mut ast::Expr, available_idents: &[String]) {
    match expr {
        ast::Expr::CompoundIdentifier(idents) => {
            remove_dangling_identifiers(idents, available_idents);
        }
        ast::Expr::CompoundFieldAccess { root, access_chain }
            if access_chain
                .iter()
                .all(|access| matches!(access, ast::AccessExpr::Dot(_))) =>
        {
            remove_dangling_qualifier(root, available_idents);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
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

    #[test]
    fn test_remove_dangling_qualifier() {
        let column = |parts: &[&str]| {
            ast::Expr::CompoundIdentifier(
                parts.iter().map(|part| Ident::new(*part)).collect(),
            )
        };
        let field_path = |root: ast::Expr| ast::Expr::CompoundFieldAccess {
            root: Box::new(root),
            access_chain: vec![ast::AccessExpr::Dot(ast::Expr::Identifier(Ident::new(
                "product",
            )))],
        };
        let subscript = |root: ast::Expr| ast::Expr::CompoundFieldAccess {
            root: Box::new(root),
            access_chain: vec![ast::AccessExpr::Subscript(ast::Subscript::Index {
                index: ast::Expr::value(ast::Value::Number("1".to_string(), false)),
            })],
        };

        // A field path is judged by the column it is rooted at, and keeps its path.
        let mut expr = field_path(column(&["t", "metadata"]));
        remove_dangling_qualifier(&mut expr, &[]);
        assert_eq!(expr, field_path(column(&["metadata"])));

        let mut expr = field_path(column(&["t", "metadata"]));
        remove_dangling_qualifier(&mut expr, &["t".to_string()]);
        assert_eq!(expr, field_path(column(&["t", "metadata"])));

        // A subscript's root may be a correlated reference, so it is left alone.
        let mut expr = subscript(column(&["o", "array_col"]));
        remove_dangling_qualifier(&mut expr, &[]);
        assert_eq!(expr, subscript(column(&["o", "array_col"])));
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
