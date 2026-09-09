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
}
