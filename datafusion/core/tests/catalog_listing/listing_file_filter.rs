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

//! `ListingFileFilter` (Spice extension): a table whose data files no suffix
//! describes, such as extensionless Hive objects next to job markers, still
//! lists through `ListingTable`, so its scans carry per-file statistics.

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::TableProvider;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingFileFilter, ListingOptions, ListingTable, ListingTableConfig,
};
use datafusion::parquet::arrow::ArrowWriter;
use datafusion::prelude::SessionContext;
use datafusion_catalog_listing::helpers::pruned_partition_list_with_metadata;
use datafusion_common::stats::Precision;
use datafusion_common::{DataFusionError, Result};
use datafusion_datasource::ListingTableUrl;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_physical_plan::{StatisticsArgs, StatisticsContext, displayable};
use futures::TryStreamExt;
use object_store::{ObjectMeta, ObjectStoreExt, memory::InMemory, path::Path};

use super::pruned_partition_list::make_test_store_and_state;

/// Hive's convention: a name starting with `_` or `.` is a marker, and a data
/// object carries no extension.
#[derive(Debug)]
struct HiveDataFiles;

impl ListingFileFilter for HiveDataFiles {
    fn is_data_file(&self, object: &ObjectMeta) -> Result<bool> {
        let name = object.location.filename().unwrap_or_default();
        Ok(!name.starts_with(['_', '.']) && !name.contains('.'))
    }
}

/// Refuses to list an object named `stray`.
#[derive(Debug)]
struct RejectStray;

impl ListingFileFilter for RejectStray {
    fn is_data_file(&self, object: &ObjectMeta) -> Result<bool> {
        if object.location.filename() == Some("stray") {
            return Err(DataFusionError::Configuration(format!(
                "'{}' is not under a partition directory",
                object.location
            )));
        }
        Ok(true)
    }
}

async fn list(
    files: &[(&str, u64)],
    partition_cols: &[(String, DataType)],
    file_filter: Option<&dyn ListingFileFilter>,
) -> Result<Vec<String>> {
    let (store, state) = make_test_store_and_state(files);
    let mut listed: Vec<String> = pruned_partition_list_with_metadata(
        state.as_ref(),
        store.as_ref(),
        &ListingTableUrl::parse("file:///tablepath/").unwrap(),
        &[],
        "",
        partition_cols,
        &[],
        &[],
        file_filter,
    )
    .await?
    .map_ok(|file| file.object_meta.location.to_string())
    .try_collect()
    .await?;
    listed.sort();
    Ok(listed)
}

#[tokio::test]
async fn the_file_filter_drops_markers_an_empty_extension_lists() {
    let files = [
        ("tablepath/p=1/000000_0", 100),
        ("tablepath/p=1/_SUCCESS", 10),
        ("tablepath/p=1/.000000_0.crc", 10),
        ("tablepath/p=2/000001_0", 100),
    ];
    let partition_cols = [("p".to_string(), DataType::Utf8)];

    assert_eq!(
        list(&files, &partition_cols, None).await.unwrap().len(),
        4,
        "an empty extension lists every object, markers included"
    );
    assert_eq!(
        list(&files, &partition_cols, Some(&HiveDataFiles))
            .await
            .unwrap(),
        ["tablepath/p=1/000000_0", "tablepath/p=2/000001_0"]
    );
}

#[tokio::test]
async fn a_file_filter_error_is_not_hidden_by_partition_parsing() {
    let files = [("tablepath/p=1/000000_0", 100), ("tablepath/stray", 100)];
    let partition_cols = [("p".to_string(), DataType::Utf8)];

    assert_eq!(
        list(&files, &partition_cols, None).await.unwrap(),
        ["tablepath/p=1/000000_0"],
        "partition parsing skips an object outside the partition directories"
    );
    let err = list(&files, &partition_cols, Some(&RejectStray))
        .await
        .expect_err("the filter sees the object before partition parsing skips it");
    assert_eq!(
        err.strip_backtrace(),
        "Invalid or Unsupported Configuration: 'tablepath/stray' is not under a partition directory"
    );
}

fn parquet_bytes(ids: &[i32]) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int32Array::from(ids.to_vec()))],
    )
    .unwrap();
    let mut out = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut out, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    out
}

#[tokio::test]
async fn a_filtered_scan_keeps_per_file_statistics_and_skips_markers() -> Result<()> {
    let store = Arc::new(InMemory::new());
    for (name, bytes) in [
        ("tablepath/000000_0", parquet_bytes(&[1, 2, 3])),
        ("tablepath/000001_0", parquet_bytes(&[4, 5])),
        ("tablepath/_SUCCESS", b"not parquet".to_vec()),
    ] {
        store.put(&Path::from(name), bytes.into()).await?;
    }
    let ctx = SessionContext::new();
    ctx.register_object_store(ObjectStoreUrl::parse("memory://")?.as_ref(), store);

    let options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_file_extension("")
        .with_file_filter(Some(Arc::new(HiveDataFiles)));
    let table_path = ListingTableUrl::parse("memory:///tablepath/")?;
    let schema = options.infer_schema(&ctx.state(), &table_path).await?;
    let table = ListingTable::try_new(
        ListingTableConfig::new(table_path)
            .with_listing_options(options)
            .with_schema(schema),
    )?;

    let plan = table.scan(&ctx.state(), None, &[], None).await?;
    let rendered = displayable(plan.as_ref()).indent(true).to_string();
    assert!(
        rendered.contains("000000_0") && rendered.contains("000001_0"),
        "both data objects are scanned: {rendered}"
    );
    assert!(
        !rendered.contains("_SUCCESS"),
        "the marker is not scanned: {rendered}"
    );

    let statistics =
        StatisticsContext::new().compute(plan.as_ref(), &StatisticsArgs::new())?;
    assert_eq!(statistics.num_rows, Precision::Exact(5));

    ctx.register_table("hive", Arc::new(table))?;
    let sum = ctx
        .sql("SELECT sum(id) AS s FROM hive")
        .await?
        .collect()
        .await?;
    assert_eq!(
        sum[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .expect("sum(Int32) is Int64")
            .value(0),
        15
    );
    Ok(())
}

/// A zero-byte object reaches the filter in every listing, so its answer for
/// the object does not depend on the operation that listed it, and an object
/// the filter accepts is still never read while it is empty.
#[tokio::test]
async fn every_listing_shows_the_file_filter_zero_byte_objects() -> Result<()> {
    let store = Arc::new(InMemory::new());
    for (name, bytes) in [
        ("tablepath/p=1/000000_0", parquet_bytes(&[1, 2, 3])),
        ("tablepath/stray", Vec::new()),
    ] {
        store.put(&Path::from(name), bytes.into()).await?;
    }
    let ctx = SessionContext::new();
    ctx.register_object_store(ObjectStoreUrl::parse("memory://")?.as_ref(), store);
    let table_path = ListingTableUrl::parse("memory:///tablepath/")?;
    let options = ListingOptions::new(Arc::new(ParquetFormat::default()))
        .with_file_extension("")
        .with_table_partition_cols(vec![("p".to_string(), DataType::Utf8)])
        .with_file_filter(Some(Arc::new(RejectStray)));
    let expected = "Invalid or Unsupported Configuration: 'tablepath/stray' is not under a partition directory";

    let err = options
        .infer_schema(&ctx.state(), &table_path)
        .await
        .expect_err("schema inference lists the zero-byte stray");
    assert_eq!(err.strip_backtrace(), expected);
    let err = options
        .infer_partitions(&ctx.state(), &table_path)
        .await
        .expect_err("partition inference lists the zero-byte stray");
    assert_eq!(err.strip_backtrace(), expected);

    let file_schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let table = Arc::new(ListingTable::try_new(
        ListingTableConfig::new(table_path)
            .with_listing_options(options)
            .with_schema(file_schema),
    )?);
    let err = table
        .scan(&ctx.state(), None, &[], None)
        .await
        .expect_err("a scan lists the zero-byte stray");
    assert_eq!(err.strip_backtrace(), expected);
    ctx.register_table("t", table)?;
    let err = ctx
        .sql("INSERT INTO t VALUES (4, '1')")
        .await?
        .collect()
        .await
        .expect_err("an insert lists the zero-byte stray");
    assert_eq!(err.strip_backtrace(), expected);

    let files = [
        ("tablepath/p=1/000000_0", 100),
        ("tablepath/p=1/000001_0", 0),
    ];
    assert_eq!(
        list(
            &files,
            &[("p".to_string(), DataType::Utf8)],
            Some(&HiveDataFiles)
        )
        .await?,
        ["tablepath/p=1/000000_0"],
        "an accepted zero-byte object is not scanned"
    );
    Ok(())
}
