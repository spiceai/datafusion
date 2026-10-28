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

//! A join that fans out emits many output batches for every input batch it
//! reads. Its output stream has to hand control back to Tokio while it does,
//! or a task polling it can't be aborted until the join finishes
//! (apache/datafusion#19358).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;

use arrow::array::{Int32Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema};
use datafusion_common::{JoinType, NullEquality};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_physical_expr::expressions::col;
use futures::StreamExt;
use futures::future::poll_fn;

use crate::ExecutionPlan;
use crate::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode, SortMergeJoinExec,
};
use crate::test::TestMemoryExec;

/// Tokio's per-task budget: a cooperative stream returns `Pending` after this
/// many units.
const TASK_BUDGET: usize = 128;

/// Output batch size. With `ROWS` rows per side that all share one key, an
/// equi-join or cross join emits `ROWS * ROWS / BATCH_SIZE` batches (4,096)
/// from a single input batch per side.
const BATCH_SIZE: usize = 64;
const ROWS: usize = 512;

/// One in-memory batch of `ROWS` rows whose only column is the constant 0.
/// The scan never returns `Pending` and consumes no Tokio budget.
fn one_key_table(name: &str) -> Arc<dyn ExecutionPlan> {
    let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(vec![0; ROWS]))],
    )
    .unwrap();
    TestMemoryExec::try_new_exec(&[vec![batch]], schema, None).unwrap()
}

fn task_ctx() -> Arc<TaskContext> {
    Arc::new(
        TaskContext::default()
            .with_session_config(SessionConfig::new().with_batch_size(BATCH_SIZE)),
    )
}

/// Checks that `join`'s output stream yields to Tokio within one task budget,
/// that a task draining it can be aborted mid-join, and that the wrapper
/// doesn't change the result.
async fn assert_join_yields(join: Arc<dyn ExecutionPlan>) {
    let name = join.name().to_string();
    let expected_rows = ROWS * ROWS;

    // 1. Poll the stream in a loop the way a parent operator does. A
    //    non-cooperative join only stops when it is exhausted.
    let mut stream = join.execute(0, task_ctx()).unwrap();
    let mut ready = 0usize;
    let yielded = poll_fn(|cx| {
        loop {
            match stream.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Ready(true),
                Poll::Ready(None) => return Poll::Ready(false),
                Poll::Ready(Some(batch)) => {
                    batch.unwrap();
                    ready += 1;
                }
            }
        }
    })
    .await;
    assert!(
        yielded,
        "{name} emitted all {ready} batches without returning Pending, so the \
         task polling it can't be aborted until the join finishes"
    );
    assert!(
        ready <= TASK_BUDGET,
        "{name} emitted {ready} batches before returning Pending (budget {TASK_BUDGET})"
    );
    drop(stream);

    // 2. Abort a task that drains the join. On the current-thread runtime the
    //    abort only lands once the draining task returns Pending.
    let produced = Arc::new(AtomicUsize::new(0));
    let mut stream = join.execute(0, task_ctx()).unwrap();
    let counter = Arc::clone(&produced);
    let drain = tokio::spawn(async move {
        while let Some(batch) = stream.next().await {
            batch.unwrap();
            counter.fetch_add(1, Ordering::Relaxed);
        }
    });
    tokio::task::yield_now().await;
    drain.abort();
    let outcome = drain.await;
    let produced = produced.load(Ordering::Relaxed);
    assert!(
        outcome.as_ref().is_err_and(|e| e.is_cancelled()),
        "{name}: the drain ran to completion ({produced} batches) before the \
         abort could land"
    );
    assert!(
        produced < expected_rows / BATCH_SIZE,
        "{name}: produced {produced} batches before the abort landed"
    );

    // 3. Same rows as before.
    let mut stream = join.execute(0, task_ctx()).unwrap();
    let mut rows = 0;
    while let Some(batch) = stream.next().await {
        rows += batch.unwrap().num_rows();
    }
    assert_eq!(rows, expected_rows, "{name} row count");
}

#[tokio::test]
async fn hash_join_output_yields_and_is_cancellable() {
    let left = one_key_table("l");
    let right = one_key_table("r");
    let on = vec![(
        col("l", &left.schema()).unwrap(),
        col("r", &right.schema()).unwrap(),
    )];
    let join = HashJoinExec::try_new(
        left,
        right,
        on,
        None,
        &JoinType::Inner,
        None,
        PartitionMode::CollectLeft,
        NullEquality::NullEqualsNothing,
        false,
    )
    .unwrap();
    assert_join_yields(Arc::new(join)).await;
}

#[tokio::test]
async fn sort_merge_join_output_yields_and_is_cancellable() {
    let left = one_key_table("l");
    let right = one_key_table("r");
    let on = vec![(
        col("l", &left.schema()).unwrap(),
        col("r", &right.schema()).unwrap(),
    )];
    let join = SortMergeJoinExec::try_new(
        left,
        right,
        on,
        None,
        JoinType::Inner,
        vec![SortOptions::default()],
        NullEquality::NullEqualsNothing,
    )
    .unwrap();
    assert_join_yields(Arc::new(join)).await;
}

#[tokio::test]
async fn nested_loop_join_output_yields_and_is_cancellable() {
    let join = NestedLoopJoinExec::try_new(
        one_key_table("l"),
        one_key_table("r"),
        None,
        &JoinType::Inner,
        None,
    )
    .unwrap();
    assert_join_yields(Arc::new(join)).await;
}

#[tokio::test]
async fn cross_join_output_yields_and_is_cancellable() {
    let join = CrossJoinExec::new(one_key_table("l"), one_key_table("r"));
    assert_join_yields(Arc::new(join)).await;
}
