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

//! Executes consumed plans whose correlated subqueries read the same table as
//! the enclosing scope, where the plan string alone cannot show whether the
//! correlation survived decorrelation.

#[cfg(test)]
mod tests {
    use crate::utils::test::read_json;
    use datafusion::arrow::array::{Int64Array, RecordBatch};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::common::{Result, assert_batches_sorted_eq};
    use datafusion::prelude::SessionContext;
    use datafusion_substrait::logical_plan::consumer::from_substrait_plan;
    use insta::assert_snapshot;
    use std::sync::Arc;

    /// `SELECT * FROM t WHERE EXISTS (SELECT * FROM t t2 WHERE t2.a = t.a AND t2.b <> t.b)`
    /// as Substrait: both scans read `t`, and the correlated predicate refers to
    /// the outer `t` one step out. Without a qualifier of its own on the inner
    /// scan, decorrelation matched both sides of the predicate to the inner scan,
    /// dropped the join condition, and kept `t.b != t.b`, returning no rows.
    #[tokio::test]
    async fn correlated_exists_over_the_same_table_keeps_its_correlation() -> Result<()> {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )?;
        ctx.register_batch("t", batch)?;

        let proto =
            read_json("tests/testdata/test_plans/self_correlated_exists.substrait.json");
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Filter: EXISTS (<subquery>)
          Subquery:
            Filter: t_1.a = outer_ref(t.a) AND t_1.b != outer_ref(t.b)
              SubqueryAlias: t_1
                TableScan: t
          TableScan: t
        ");

        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        // Rows 1|10 and 1|20 each have another row with the same `a` and a
        // different `b`; 2|30 has none.
        assert_batches_sorted_eq!(
            [
                "+---+----+",
                "| a | b  |",
                "+---+----+",
                "| 1 | 10 |",
                "| 1 | 20 |",
                "+---+----+",
            ],
            &batches
        );
        Ok(())
    }

    fn three_rows() -> Result<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 2])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )?)
    }

    /// The same correlated predicate carried as the inner `ReadRel.filter`
    /// rather than a `FilterRel`: it must be applied above the alias, against
    /// the aliased schema, not pushed into the scan where the decorrelation
    /// rules cannot reach it.
    #[tokio::test]
    async fn correlated_read_filter_over_the_same_table_keeps_its_correlation()
    -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_batch("t", three_rows()?)?;

        let proto = read_json(
            "tests/testdata/test_plans/self_correlated_exists_read_filter.substrait.json",
        );
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Filter: EXISTS (<subquery>)
          Subquery:
            Filter: t_1.a = outer_ref(t.a) AND t_1.b != outer_ref(t.b)
              SubqueryAlias: t_1
                TableScan: t
          TableScan: t
        ");

        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        assert_batches_sorted_eq!(
            [
                "+---+----+",
                "| a | b  |",
                "+---+----+",
                "| 1 | 10 |",
                "| 1 | 20 |",
                "+---+----+",
            ],
            &batches
        );
        Ok(())
    }

    /// A `ReadRel.filter`'s field indices are defined against the Substrait
    /// base schema; the provider may carry more, or differently ordered,
    /// fields (`ensure_schema_compatibility` allows both). The filter must
    /// bind to the Substrait fields, and the aliased scan must be projected
    /// to them by name.
    #[tokio::test]
    async fn correlated_read_filter_binds_fields_to_the_substrait_schema() -> Result<()> {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("extra", DataType::Utf8, false),
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(datafusion::arrow::array::StringArray::from(vec![
                    "x", "y", "z",
                ])),
                Arc::new(Int64Array::from(vec![1, 1, 2])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )?;
        ctx.register_batch("t", batch)?;

        let proto = read_json(
            "tests/testdata/test_plans/self_correlated_exists_read_filter.substrait.json",
        );
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Filter: EXISTS (<subquery>)
          Subquery:
            Projection: t_1.a, t_1.b
              Filter: t_1.a = outer_ref(t.a) AND t_1.b != outer_ref(t.b)
                SubqueryAlias: t_1
                  TableScan: t
          TableScan: t projection=[a, b]
        ");

        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        assert_batches_sorted_eq!(
            [
                "+---+----+",
                "| a | b  |",
                "+---+----+",
                "| 1 | 10 |",
                "| 1 | 20 |",
                "+---+----+",
            ],
            &batches
        );
        Ok(())
    }

    /// Both scopes self-join `t`, so both joins requalify their sides. The
    /// inner sides must not take the enclosing join's `left`/`right`, or the
    /// predicate correlating to the outer `left` collapses; they become
    /// `left_1`/`right_1`.
    #[tokio::test]
    async fn requalified_join_inside_a_subquery_avoids_the_enclosing_left_and_right()
    -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_batch("t", three_rows()?)?;

        let proto = read_json(
            "tests/testdata/test_plans/self_correlated_exists_requalified_joins.substrait.json",
        );
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Projection: left.a AS a1, left.b AS b1, right.a AS a2, right.b AS b2
          Filter: EXISTS (<subquery>)
            Subquery:
              Filter: left_1.a = outer_ref(left.a) AND left_1.b != outer_ref(left.b)
                Cross Join:
                  SubqueryAlias: left_1
                    TableScan: t
                  SubqueryAlias: right_1
                    TableScan: t
            Cross Join:
              SubqueryAlias: left
                TableScan: t
              SubqueryAlias: right
                TableScan: t
        ");

        // The outer `left` rows 1|10 and 1|20 have a partner in `t` with the
        // same `a` and a different `b`; 2|30 has none. Each keeps its three
        // outer `right` partners.
        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        assert_batches_sorted_eq!(
            [
                "+----+----+----+----+",
                "| a1 | b1 | a2 | b2 |",
                "+----+----+----+----+",
                "| 1  | 10 | 1  | 10 |",
                "| 1  | 10 | 1  | 20 |",
                "| 1  | 10 | 2  | 30 |",
                "| 1  | 20 | 1  | 10 |",
                "| 1  | 20 | 1  | 20 |",
                "| 1  | 20 | 2  | 30 |",
                "+----+----+----+----+",
            ],
            &batches
        );
        Ok(())
    }

    /// A self-`INTERSECT` inside the subquery: `intersect`/`except` also
    /// requalify their sides, so the sides must keep clear of the enclosing
    /// join's `left`/`right` as joins do.
    #[tokio::test]
    async fn intersect_inside_a_subquery_avoids_the_enclosing_left_and_right()
    -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_batch("t", three_rows()?)?;

        let proto = read_json(
            "tests/testdata/test_plans/self_correlated_exists_intersect.substrait.json",
        );
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Projection: left.a AS a1, left.b AS b1, right.a AS a2, right.b AS b2
          Filter: EXISTS (<subquery>)
            Subquery:
              Filter: left_1.a = outer_ref(left.a) AND left_1.b != outer_ref(left.b)
                LeftSemi Join: left_1.a = right_1.a, left_1.b = right_1.b
                  Distinct:
                    SubqueryAlias: left_1
                      TableScan: t
                  SubqueryAlias: right_1
                    TableScan: t
            Cross Join:
              SubqueryAlias: left
                TableScan: t
              SubqueryAlias: right
                TableScan: t
        ");

        // `t INTERSECT t` is `t`; the outer `left` rows 1|10 and 1|20 have a
        // partner with the same `a` and a different `b`, 2|30 has none.
        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        assert_batches_sorted_eq!(
            [
                "+----+----+----+----+",
                "| a1 | b1 | a2 | b2 |",
                "+----+----+----+----+",
                "| 1  | 10 | 1  | 10 |",
                "| 1  | 10 | 1  | 20 |",
                "| 1  | 10 | 2  | 30 |",
                "| 1  | 20 | 1  | 10 |",
                "| 1  | 20 | 1  | 20 |",
                "| 1  | 20 | 2  | 30 |",
                "+----+----+----+----+",
            ],
            &batches
        );
        Ok(())
    }

    /// A correlated `ReadRel.filter` on a table no enclosing scope reads: no
    /// alias is needed, but the filter still belongs above the scan, because
    /// a `TableScan`'s filters cannot evaluate an outer reference and the
    /// decorrelation rules cannot lift one from there.
    #[tokio::test]
    async fn correlated_read_filter_on_another_table_stays_above_the_scan() -> Result<()>
    {
        let ctx = SessionContext::new();
        ctx.register_batch("t", three_rows()?)?;
        ctx.register_batch("u", three_rows()?)?;

        let proto = read_json(
            "tests/testdata/test_plans/correlated_read_filter_other_table.substrait.json",
        );
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Filter: EXISTS (<subquery>)
          Subquery:
            Filter: u.a = outer_ref(t.a) AND u.b != outer_ref(t.b)
              TableScan: u
          TableScan: t
        ");

        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        assert_batches_sorted_eq!(
            [
                "+---+----+",
                "| a | b  |",
                "+---+----+",
                "| 1 | 10 |",
                "| 1 | 20 |",
                "+---+----+",
            ],
            &batches
        );
        Ok(())
    }

    /// The enclosing scope reads `t` and a table that is already named `t_1`;
    /// the inner scan of `t`, correlated to that `t_1`, must not take the name
    /// `t_1` or the collision comes straight back. It becomes `t_2`.
    #[tokio::test]
    async fn subquery_scan_alias_skips_a_name_an_enclosing_scope_uses() -> Result<()> {
        let ctx = SessionContext::new();
        ctx.register_batch("t", three_rows()?)?;
        ctx.register_batch("t_1", three_rows()?)?;

        let proto = read_json(
            "tests/testdata/test_plans/self_correlated_exists_alias_taken.substrait.json",
        );
        let plan = from_substrait_plan(&ctx.state(), &proto).await?;
        assert_snapshot!(plan.display_indent(), @"
        Projection: t.a, t.b, t_1.a AS a1, t_1.b AS b1
          Filter: EXISTS (<subquery>)
            Subquery:
              Filter: t_2.a = outer_ref(t_1.a) AND t_2.b != outer_ref(t_1.b)
                SubqueryAlias: t_2
                  TableScan: t
            Cross Join:
              TableScan: t
              TableScan: t_1
        ");

        // Every `t` row pairs with the two `t_1` rows that have a partner in
        // `t` with the same `a` and a different `b`; `t_1`'s 2|30 has none.
        let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
        assert_batches_sorted_eq!(
            [
                "+---+----+----+----+",
                "| a | b  | a1 | b1 |",
                "+---+----+----+----+",
                "| 1 | 10 | 1  | 10 |",
                "| 1 | 10 | 1  | 20 |",
                "| 1 | 20 | 1  | 10 |",
                "| 1 | 20 | 1  | 20 |",
                "| 2 | 30 | 1  | 10 |",
                "| 2 | 30 | 1  | 20 |",
                "+---+----+----+----+",
            ],
            &batches
        );
        Ok(())
    }
}
