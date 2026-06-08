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

//! Benchmarks for the file-opening pruner construction path.
//!
//! `PruningPredicate::try_new` is invoked once per opened file (and again per
//! single-column page predicate and per candidate row group). Profiling shows
//! that nearly all of its cost lives in [`PhysicalExprSimplifier::simplify`],
//! which `try_new` runs on the rewritten predicate expression.
//!
//! Each predicate is benchmarked two ways so the simplifier's share is visible
//! live as the bench runs:
//!
//! * `try_new`  — full [`PruningPredicate::try_new`], i.e. the real per-file cost.
//! * `simplify` — [`PhysicalExprSimplifier::simplify`] alone, on the same input
//!   expression, so it can be compared directly against `try_new`.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p datafusion-pruning --bench pruning_predicate
//! ```

use std::hint::black_box;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use datafusion_common::ScalarValue;
use datafusion_expr::Operator;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, IsNullExpr, Literal,
};
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_pruning::PruningPredicate;

fn col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name, index))
}

fn lit_i32(value: i32) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int32(Some(value))))
}

fn lit_i64(value: i64) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int64(Some(value))))
}

fn lit_bool(value: bool) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Boolean(Some(value))))
}

fn bin(
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(left, op, right))
}

fn and(
    left: Arc<dyn PhysicalExpr>,
    right: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    bin(left, Operator::And, right)
}

/// A conjunction of `n_terms` ranged comparisons, representative of a typical
/// analytical filter pushed down to a scan.
///
/// Each term is `cast(c_i AS Int64) >= lo AND cast(c_i AS Int64) <= hi`, which
/// forces the simplifier's `unwrap_cast` pass to run, and the chain is prefixed
/// with a folded partition equality (`7 = 7`) to mimic the constant subterms
/// produced when `replace_columns_with_literals` substitutes partition values
/// during file opening.
fn range_conjunction(n_terms: usize) -> (SchemaRef, Arc<dyn PhysicalExpr>) {
    let fields: Vec<Field> = (0..n_terms)
        .map(|i| Field::new(format!("c{i}"), DataType::Int32, true))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let part_eq = bin(lit_i32(7), Operator::Eq, lit_i32(7));

    let expr = (0..n_terms)
        .map(|i| {
            let column = col(&format!("c{i}"), i);
            let cast: Arc<dyn PhysicalExpr> =
                Arc::new(CastExpr::new(column, DataType::Int64, None));
            and(
                bin(
                    Arc::clone(&cast),
                    Operator::GtEq,
                    lit_i64(i64::try_from(i).expect("term index fits in i64")),
                ),
                bin(cast, Operator::LtEq, lit_i64(1_000_000)),
            )
        })
        .fold(part_eq, and);

    (schema, expr)
}

/// A TPC-DS q76-style predicate: `ship_addr IS NULL AND CASE(...) AND CASE(...)`
/// where each `CASE` has `n_branches` arms keyed on a hash-partition modulo.
///
/// This is the worst case for simplification — large `CASE` trees with many
/// literal-heavy branches that the constant evaluator must walk repeatedly.
fn case_predicate(n_branches: usize) -> (SchemaRef, Arc<dyn PhysicalExpr>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("item_sk", DataType::Int64, true),
        Field::new("sold_date_sk", DataType::Int64, true),
        Field::new("ship_addr_sk", DataType::Int64, true),
    ]));

    let item = col("item_sk", 0);
    let date = col("sold_date_sk", 1);
    let addr = col("ship_addr_sk", 2);

    let n = i64::try_from(n_branches).expect("branch count fits in i64");

    let case = |key_mod: Arc<dyn PhysicalExpr>,
                key_col: &Arc<dyn PhysicalExpr>,
                hi: i64| {
        let branches: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = (0
            ..n_branches)
            .map(|p| {
                let partition = i32::try_from(p).expect("partition fits in i32");
                let when = bin(Arc::clone(&key_mod), Operator::Eq, lit_i32(partition));
                let then = and(
                    bin(
                        Arc::clone(key_col),
                        Operator::GtEq,
                        lit_i64(i64::from(partition)),
                    ),
                    bin(Arc::clone(key_col), Operator::LtEq, lit_i64(hi)),
                );
                (when, then)
            })
            .collect();
        let case: Arc<dyn PhysicalExpr> = Arc::new(
            CaseExpr::try_new(None, branches, Some(lit_bool(false))).expect("case expr"),
        );
        case
    };

    let item_case = case(
        bin(Arc::clone(&item), Operator::Modulo, lit_i64(n)),
        &item,
        18_000,
    );
    let date_case = case(
        bin(Arc::clone(&date), Operator::Modulo, lit_i64(n)),
        &date,
        2_488_070,
    );

    let is_null: Arc<dyn PhysicalExpr> = Arc::new(IsNullExpr::new(addr));
    let expr = and(and(is_null, item_case), date_case);

    (schema, expr)
}

/// A complex dynamic-filter predicate: `n_ranges` disjoint range bands on a
/// single column, OR'd together — e.g.
/// `(a > 1 AND a < 2) OR (a > 2 AND a < 3) OR ...`.
///
/// This is the shape produced when a dynamic join filter encodes many distinct
/// build-side key bands. It is a single dynamic filter, but it forces the
/// simplifier to walk a wide OR tree of range comparisons (none of which can be
/// folded away), so the simplification cost grows with the number of bands.
fn or_of_ranges(n_ranges: usize) -> (SchemaRef, Arc<dyn PhysicalExpr>) {
    let schema: SchemaRef =
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));

    let expr = (0..n_ranges)
        .map(|i| {
            let lo = i32::try_from(i + 1).expect("range bound fits in i32");
            and(
                bin(col("a", 0), Operator::Gt, lit_i32(lo)),
                bin(col("a", 0), Operator::Lt, lit_i32(lo + 1)),
            )
        })
        .reduce(|acc, band| bin(acc, Operator::Or, band))
        .expect("n_ranges must be non-zero");

    (schema, expr)
}

fn bench_predicate(
    c: &mut Criterion,
    group_name: &str,
    sizes: &[usize],
    build: impl Fn(usize) -> (SchemaRef, Arc<dyn PhysicalExpr>),
) {
    let mut group = c.benchmark_group(group_name);
    for &n in sizes {
        let (schema, expr) = build(n);

        // Full pruner construction — this is what runs once per opened file.
        group.bench_with_input(BenchmarkId::new("try_new", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    PruningPredicate::try_new(
                        black_box(Arc::clone(&expr)),
                        Arc::clone(&schema),
                    )
                    .expect("try_new succeeds"),
                )
            });
        });

        // The simplifier alone, on the same input expression.
        let simplifier = PhysicalExprSimplifier::new(schema.as_ref());
        group.bench_with_input(BenchmarkId::new("simplify", n), &n, |b, _| {
            b.iter(|| {
                black_box(
                    simplifier
                        .simplify(black_box(Arc::clone(&expr)))
                        .expect("simplify succeeds"),
                )
            });
        });
    }
    group.finish();
}

/// Small predicates typical of dynamic filter pushdown from a join build side:
/// a single bound (`a > 15`) and a two-sided range (`a > 10 AND a < 50`).
///
/// These are cheap to simplify individually, but the pruner builds and
/// simplifies them once per opened file — often hundreds of thousands of times
/// in a single query — so the per-call fixed overhead dominates.
fn bench_dynamic_filter(c: &mut Criterion) {
    let schema: SchemaRef =
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));

    let gt = bin(col("a", 0), Operator::Gt, lit_i32(15));
    let range = and(
        bin(col("a", 0), Operator::Gt, lit_i32(10)),
        bin(col("a", 0), Operator::Lt, lit_i32(50)),
    );
    let cases: [(&str, &Arc<dyn PhysicalExpr>); 2] = [("gt", &gt), ("range", &range)];

    let mut group = c.benchmark_group("pruning_predicate/dynamic_filter");
    for (label, expr) in cases {
        // Full pruner construction — this is what runs once per opened file.
        group.bench_with_input(BenchmarkId::new("try_new", label), label, |b, _| {
            b.iter(|| {
                black_box(
                    PruningPredicate::try_new(
                        black_box(Arc::clone(expr)),
                        Arc::clone(&schema),
                    )
                    .expect("try_new succeeds"),
                )
            });
        });

        // The simplifier alone, on the same input expression.
        let simplifier = PhysicalExprSimplifier::new(schema.as_ref());
        group.bench_with_input(BenchmarkId::new("simplify", label), label, |b, _| {
            b.iter(|| {
                black_box(
                    simplifier
                        .simplify(black_box(Arc::clone(expr)))
                        .expect("simplify succeeds"),
                )
            });
        });
    }
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_dynamic_filter(c);
    bench_predicate(
        c,
        "pruning_predicate/or_ranges",
        &[16, 50, 128],
        or_of_ranges,
    );
    bench_predicate(
        c,
        "pruning_predicate/range",
        &[16, 64, 256],
        range_conjunction,
    );
    bench_predicate(c, "pruning_predicate/case", &[16, 64, 128], case_predicate);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
