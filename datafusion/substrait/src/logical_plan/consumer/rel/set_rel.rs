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

use crate::logical_plan::consumer::SubstraitConsumer;
use crate::logical_plan::consumer::utils::requalify_sides_for_scope;
use datafusion::common::{not_impl_err, substrait_err};
use datafusion::logical_expr::{LogicalPlan, LogicalPlanBuilder};
use substrait::proto::set_rel::SetOp;
use substrait::proto::{Rel, SetRel};

pub async fn from_set_rel(
    consumer: &impl SubstraitConsumer,
    set: &SetRel,
) -> datafusion::common::Result<LogicalPlan> {
    if set.inputs.len() < 2 {
        substrait_err!("Set operation requires at least two inputs")
    } else {
        match set.op() {
            SetOp::UnionAll => union_rels(consumer, &set.inputs, true).await,
            SetOp::UnionDistinct => union_rels(consumer, &set.inputs, false).await,
            SetOp::IntersectionPrimary => {
                let (left, right) = scoped_sides(
                    consumer,
                    consumer.consume_rel(&set.inputs[0]).await?,
                    union_rels(consumer, &set.inputs[1..], true).await?,
                )?;
                LogicalPlanBuilder::intersect(left, right, false)
            }
            SetOp::IntersectionMultiset => {
                intersect_rels(consumer, &set.inputs, false).await
            }
            SetOp::IntersectionMultisetAll => {
                intersect_rels(consumer, &set.inputs, true).await
            }
            SetOp::MinusPrimary => except_rels(consumer, &set.inputs, false).await,
            SetOp::MinusPrimaryAll => except_rels(consumer, &set.inputs, true).await,
            set_op => not_impl_err!("Unsupported set operator: {set_op:?}"),
        }
    }
}

async fn union_rels(
    consumer: &impl SubstraitConsumer,
    rels: &[Rel],
    is_all: bool,
) -> datafusion::common::Result<LogicalPlan> {
    let mut union_builder = Ok(LogicalPlanBuilder::from(
        consumer.consume_rel(&rels[0]).await?,
    ));
    for input in &rels[1..] {
        let rel_plan = consumer.consume_rel(input).await?;

        union_builder = if is_all {
            union_builder?.union(rel_plan)
        } else {
            union_builder?.union_distinct(rel_plan)
        };
    }
    union_builder?.build()
}

async fn intersect_rels(
    consumer: &impl SubstraitConsumer,
    rels: &[Rel],
    is_all: bool,
) -> datafusion::common::Result<LogicalPlan> {
    let mut rel = consumer.consume_rel(&rels[0]).await?;

    for input in &rels[1..] {
        let (left, right) =
            scoped_sides(consumer, rel, consumer.consume_rel(input).await?)?;
        rel = LogicalPlanBuilder::intersect(left, right, is_all)?;
    }

    Ok(rel)
}

async fn except_rels(
    consumer: &impl SubstraitConsumer,
    rels: &[Rel],
    is_all: bool,
) -> datafusion::common::Result<LogicalPlan> {
    let mut rel = consumer.consume_rel(&rels[0]).await?;

    for input in &rels[1..] {
        let (left, right) =
            scoped_sides(consumer, rel, consumer.consume_rel(input).await?)?;
        rel = LogicalPlanBuilder::except(left, right, is_all)?;
    }

    Ok(rel)
}

/// `LogicalPlanBuilder::intersect` / `except` requalify conflicting sides to
/// the fixed `left`/`right`; inside a subquery those can collide with the
/// enclosing scope's, so the sides are requalified for the scope first and
/// the builder then finds nothing left to rename.
fn scoped_sides(
    consumer: &impl SubstraitConsumer,
    left: LogicalPlan,
    right: LogicalPlan,
) -> datafusion::common::Result<(LogicalPlan, LogicalPlan)> {
    let (left, right) = requalify_sides_for_scope(
        consumer,
        LogicalPlanBuilder::from(left),
        LogicalPlanBuilder::from(right),
    )?;
    Ok((left.build()?, right.build()?))
}
