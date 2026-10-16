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

//! `ListingFileFilter` (Spice extension) in plan serialization:
//! `ListingTableScanNode` cannot carry the filter, so a filtered `ListingTable`
//! is encoded by the `LogicalExtensionCodec` or not at all.

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::TableProvider;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingFileFilter, ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::datasource::source_as_provider;
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::prelude::SessionContext;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, TableReference, internal_err, not_impl_err};
use datafusion_execution::TaskContext;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::{Extension, LogicalPlan};
use datafusion_proto::bytes::{
    logical_plan_from_bytes_with_extension_codec, logical_plan_to_bytes,
    logical_plan_to_bytes_with_extension_codec,
};
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use object_store::{ObjectMeta, ObjectStoreExt, memory::InMemory, path::Path};

/// Hive's convention: a name starting with `_` or `.` is a marker, and a data
/// object carries no extension.
#[derive(Debug)]
struct HiveDataFiles;

impl ListingFileFilter for HiveDataFiles {
    fn is_data_file(&self, object: &ObjectMeta) -> Result<bool> {
        let name = object.location.filename().unwrap_or_default();
        Ok(!name.starts_with(['_', '.']) && !name.contains('.'))
    }

    fn accepts_inserted_files(&self, file_extension: &str) -> bool {
        file_extension.is_empty()
    }
}

/// Encodes the one table it holds by name, and decodes that name back to the
/// same provider, as a codec that resolves tables from a catalog does.
#[derive(Debug)]
struct RegisteredTableCodec {
    name: String,
    table: Arc<dyn TableProvider>,
}

impl LogicalExtensionCodec for RegisteredTableCodec {
    fn try_decode(
        &self,
        _buf: &[u8],
        _inputs: &[LogicalPlan],
        _ctx: &TaskContext,
    ) -> Result<Extension> {
        not_impl_err!("RegisteredTableCodec decodes no extension nodes")
    }

    fn try_encode(&self, _node: &Extension, _buf: &mut Vec<u8>) -> Result<()> {
        not_impl_err!("RegisteredTableCodec encodes no extension nodes")
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        _table_ref: &TableReference,
        _schema: SchemaRef,
        _ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        if buf != self.name.as_bytes() {
            return internal_err!("RegisteredTableCodec holds no table {buf:?}");
        }
        Ok(Arc::clone(&self.table))
    }

    fn try_encode_table_provider(
        &self,
        _table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        if !Arc::ptr_eq(&node, &self.table) {
            return internal_err!("RegisteredTableCodec holds a different table");
        }
        buf.extend_from_slice(self.name.as_bytes());
        Ok(())
    }
}

fn parquet_bytes(ids: &[i32]) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .expect("valid batch");
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, schema, None).expect("writer");
    writer.write(&batch).expect("write parquet");
    writer.close().expect("close parquet");
    bytes
}

/// A context over `memory:///tablepath/`, which holds the data object
/// `000000_0` (ids 1, 2, 3) and `stale.parquet` (id 100), a Parquet file the
/// filter excludes. Returns the context and table `t` over that path, which
/// uses `file_filter` and is not yet registered.
async fn context_and_table(
    file_filter: Option<Arc<dyn ListingFileFilter>>,
) -> Result<(SessionContext, Arc<dyn TableProvider>)> {
    let store = Arc::new(InMemory::new());
    store
        .put(
            &Path::from("tablepath/000000_0"),
            parquet_bytes(&[1, 2, 3]).into(),
        )
        .await?;
    store
        .put(
            &Path::from("tablepath/stale.parquet"),
            parquet_bytes(&[100]).into(),
        )
        .await?;
    let ctx = SessionContext::new();
    ctx.register_object_store(
        ObjectStoreUrl::parse("memory://")?.as_ref(),
        Arc::clone(&store) as _,
    );
    let options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_file_extension("")
        .with_file_filter(file_filter);
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let table = ListingTable::try_new(
        ListingTableConfig::new(ListingTableUrl::parse("memory:///tablepath/")?)
            .with_listing_options(options)
            .with_schema(schema),
    )?;
    Ok((ctx, Arc::new(table)))
}

async fn sum_of_ids(ctx: &SessionContext, plan: LogicalPlan) -> Result<String> {
    let batches = ctx.execute_logical_plan(plan).await?.collect().await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// The provider of the plan's only table scan.
fn scanned_table(plan: &LogicalPlan) -> Result<Arc<dyn TableProvider>> {
    let mut scanned = Vec::new();
    plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            scanned.push(source_as_provider(&scan.source)?);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    assert_eq!(scanned.len(), 1, "expected one table scan in {plan}");
    Ok(scanned.remove(0))
}

const SUM_6: &str = "+---+\n| s |\n+---+\n| 6 |\n+---+";

/// Without a codec for it, a filtered table fails to serialize instead of
/// decoding as a table that also reads `stale.parquet` and sums to 106.
#[tokio::test]
async fn the_default_codec_refuses_a_listing_table_with_a_file_filter() -> Result<()> {
    let (ctx, table) = context_and_table(Some(Arc::new(HiveDataFiles))).await?;
    ctx.register_table("t", table)?;
    let plan = ctx
        .sql("SELECT sum(id) AS s FROM t")
        .await?
        .logical_plan()
        .clone();
    assert_eq!(sum_of_ids(&ctx, plan.clone()).await?, SUM_6);

    let err = logical_plan_to_bytes(&plan).expect_err("a filtered table must not encode");
    assert_eq!(
        err.strip_backtrace(),
        "Error serializing ListingTable t: ListingTableScanNode cannot encode its \
         ListingFileFilter, so the LogicalExtensionCodec must encode the table\n\
         caused by\n\
         This feature is not implemented: LogicalExtensionCodec is not provided"
    );

    // The same table without a filter still encodes as a `ListingTableScanNode`.
    let (ctx, table) = context_and_table(None).await?;
    ctx.register_table("t", table)?;
    let plan = ctx
        .sql("SELECT sum(id) AS s FROM t")
        .await?
        .logical_plan()
        .clone();
    logical_plan_to_bytes(&plan)?;
    Ok(())
}

/// A codec that encodes the table round-trips it with its filter, so the
/// decoded plan still skips `stale.parquet`.
#[tokio::test]
async fn an_extension_codec_round_trips_a_listing_table_with_its_file_filter()
-> Result<()> {
    let (ctx, table) = context_and_table(Some(Arc::new(HiveDataFiles))).await?;
    ctx.register_table("t", Arc::clone(&table))?;
    let codec = RegisteredTableCodec {
        name: "t".to_string(),
        table: Arc::clone(&table),
    };
    let plan = ctx
        .sql("SELECT sum(id) AS s FROM t")
        .await?
        .logical_plan()
        .clone();

    let bytes = logical_plan_to_bytes_with_extension_codec(&plan, &codec)?;
    let decoded =
        logical_plan_from_bytes_with_extension_codec(&bytes, &ctx.task_ctx(), &codec)?;

    let decoded_table = scanned_table(&decoded)?;
    assert!(Arc::ptr_eq(&decoded_table, &table));
    let listing = decoded_table
        .downcast_ref::<ListingTable>()
        .expect("decoded table is a ListingTable");
    assert!(listing.options().file_filter.is_some());
    assert_eq!(sum_of_ids(&ctx, decoded).await?, SUM_6);
    Ok(())
}
