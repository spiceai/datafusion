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

//! `ListingTable` plan-level tests for metadata-column pushdown.
//!
//! These exercise the full path — `supports_filters_pushdown`, `scan_with_args`, and the
//! physical plan the optimizer builds — rather than calling
//! `pruned_partition_list_with_metadata` directly, so a regression that removes the
//! `Exact` marking or the file-pruning wiring (while leaving the listing helper itself
//! correct) would be caught here.

use std::sync::Arc;

use datafusion::datasource::TableProvider;
use datafusion::datasource::listing::{ListingOptions, ListingTable, ListingTableConfig};
use datafusion::prelude::SessionContext;
use datafusion_datasource::ListingTableUrl;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::metadata::MetadataColumn;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_json::JsonFormat;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::{Expr, TableProviderFilterPushDown, col, lit};
use datafusion_physical_plan::{ExecutionPlan, displayable};
use object_store::{ObjectStoreExt, memory::InMemory, path::Path};

/// Registers an in-memory JSON-backed `ListingTable`, one line per file, pruned by
/// `metadata_cols`. Files are named `a.json`, `b.json`, ... in the order given.
async fn build_table(
    ctx: &SessionContext,
    file_row_counts: &[usize],
    metadata_cols: Vec<MetadataColumn>,
) -> Arc<ListingTable> {
    let store = Arc::new(InMemory::new());
    for (i, rows) in file_row_counts.iter().enumerate() {
        let name = format!("tablepath/{}.json", (b'a' + i as u8) as char);
        let content: String = (0..*rows).map(|n| format!("{{\"id\": {n}}}\n")).collect();
        store
            .put(&Path::from(name), content.into_bytes().into())
            .await
            .unwrap();
    }

    let store_url = ObjectStoreUrl::parse("memory://").unwrap();
    ctx.register_object_store(store_url.as_ref(), store);

    let options = ListingOptions::new(Arc::new(JsonFormat::default()))
        .with_file_extension(".json")
        .with_metadata_cols(metadata_cols);

    let config =
        ListingTableConfig::new(ListingTableUrl::parse("memory:///tablepath/").unwrap())
            .with_listing_options(options)
            .infer_schema(&ctx.state())
            .await
            .unwrap();

    Arc::new(ListingTable::try_new(config).unwrap())
}

/// Total number of files across every file group in the physical plan, found by
/// searching for `DataSourceExec` nodes.
fn total_files(plan: &Arc<dyn ExecutionPlan>) -> usize {
    if let Some(ds) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = ds.data_source().downcast_ref::<FileScanConfig>()
    {
        return config.file_groups.iter().map(|g| g.len()).sum();
    }
    plan.children().iter().map(|c| total_files(c)).sum()
}

async fn physical_plan_for(
    ctx: &SessionContext,
    table: Arc<ListingTable>,
    filter: Expr,
) -> Arc<dyn ExecutionPlan> {
    ctx.register_table("t", table).unwrap();
    let df = ctx
        .table("t")
        .await
        .unwrap()
        .filter(filter)
        .unwrap()
        .select_columns(&["id"])
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    ctx.deregister_table("t").unwrap();
    plan
}

#[tokio::test]
async fn metadata_only_filter_is_exact_and_prunes_file_groups() {
    let ctx = SessionContext::new();
    // 3 one-row files and one 50-row file: `_size` separates them cleanly.
    let table = build_table(&ctx, &[1, 1, 1, 50], vec![MetadataColumn::Size]).await;

    let filter = col("_size").lt(lit(50u64));
    assert_eq!(
        table
            .supports_filters_pushdown(&[&filter])
            .unwrap()
            .as_slice(),
        [TableProviderFilterPushDown::Exact],
        "a filter fully evaluable from metadata columns must be reported Exact"
    );

    let plan = physical_plan_for(&ctx, table, filter).await;
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        !plan_str.contains("FilterExec"),
        "an Exact metadata predicate must not need a residual FilterExec:\n{plan_str}"
    );
    assert_eq!(
        total_files(&plan),
        3,
        "the 50-row file must be pruned from the file groups before any file is opened:\n{plan_str}"
    );
}

#[tokio::test]
async fn metadata_filter_matching_nothing_yields_empty_exec() {
    let ctx = SessionContext::new();
    let table = build_table(&ctx, &[1, 1, 50], vec![MetadataColumn::Size]).await;

    // No file is anywhere near this large.
    let filter = col("_size").gt(lit(1_000_000u64));
    let plan = physical_plan_for(&ctx, table, filter).await;
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        plan_str.contains("EmptyExec"),
        "a metadata filter excluding every file must yield EmptyExec:\n{plan_str}"
    );
}

#[tokio::test]
async fn mixed_predicate_stays_inexact_with_residual_filter() {
    let ctx = SessionContext::new();
    let table = build_table(&ctx, &[1, 1, 50], vec![MetadataColumn::Size]).await;

    // `id` is a data column, not a metadata column, so this predicate cannot be
    // evaluated purely from `ObjectMeta` and must remain Inexact.
    let filter = col("_size").lt(lit(50u64)).or(col("id").eq(lit(0i64)));
    assert_eq!(
        table
            .supports_filters_pushdown(&[&filter])
            .unwrap()
            .as_slice(),
        [TableProviderFilterPushDown::Inexact],
        "a predicate mixing metadata and data columns must stay Inexact"
    );

    let plan = physical_plan_for(&ctx, table, filter).await;
    let plan_str = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        plan_str.contains("FilterExec"),
        "an Inexact predicate must keep a residual FilterExec:\n{plan_str}"
    );
}
