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

//! [`ParquetFileReaderFactory`] and [`DefaultParquetFileReaderFactory`] for
//! low level control of parquet file readers

use crate::ParquetFileMetrics;
use crate::metadata::{DFParquetMetadata, version_from_head_if_same_generation};
use bytes::Bytes;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::cache::cache_manager::FileMetadata;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::{ObjectMeta, ObjectStore, ObjectStoreExt};
use parking_lot::Mutex;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::{
    AsyncFileReader, ObjectVersionType, ParquetObjectReader,
};
use parquet::file::metadata::ParquetMetaData;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// `(location, listed ETag)` → version id discovered by `HEAD` during
/// metadata load.
///
/// ListObjectsV2 omits version ids. `CachedParquetFileReader::get_metadata`
/// promotes one from HEAD when the ETag still matches. Bloom-filter scans
/// then build a second reader from the original [`PartitionedFile`] (still
/// `version: None`); looking this map up keeps that reader on the same
/// generation pin instead of falling back to a stale `If-Match`.
type DiscoveredVersions = Arc<Mutex<HashMap<(String, String), String>>>;

fn discovered_version_key(meta: &ObjectMeta) -> Option<(String, String)> {
    Some((meta.location.to_string(), meta.e_tag.clone()?))
}

fn apply_discovered_version(pins: &DiscoveredVersions, meta: &mut ObjectMeta) {
    if meta.version.is_some() {
        return;
    }
    let Some(key) = discovered_version_key(meta) else {
        return;
    };
    if let Some(version) = pins.lock().get(&key).cloned() {
        meta.version = Some(version);
    }
}

fn record_discovered_version(
    pins: &DiscoveredVersions,
    meta: &ObjectMeta,
    version: &str,
) {
    let Some(key) = discovered_version_key(meta) else {
        return;
    };
    pins.lock().insert(key, version.to_string());
}

/// Interface for reading Apache Parquet files.
///
/// The combined implementations of [`ParquetFileReaderFactory`] and
/// [`AsyncFileReader`] can be used to provide custom data access operations
/// such as pre-cached metadata, I/O coalescing, etc.
///
/// See [`DefaultParquetFileReaderFactory`] for a simple implementation.
pub trait ParquetFileReaderFactory: Debug + Send + Sync + 'static {
    /// Provides an `AsyncFileReader` for reading data from a parquet file specified
    ///
    /// # Notes
    ///
    /// If the resulting [`AsyncFileReader`]  returns `ParquetMetaData` without
    /// page index information, the reader will load it on demand. Thus it is important
    /// to ensure that the returned `ParquetMetaData` has the necessary information
    /// if you wish to avoid a subsequent I/O
    ///
    /// # Arguments
    /// * partition_index - Index of the partition (for reporting metrics)
    /// * file - The file to be read
    /// * metadata_size_hint - If specified, the first IO reads this many bytes from the footer
    /// * metrics - Execution metrics
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>>;
}

/// Default implementation of [`ParquetFileReaderFactory`]
///
/// This implementation:
/// 1. Reads parquet directly from an underlying [`ObjectStore`] instance.
/// 2. Reads the footer and page metadata on demand.
/// 3. Does not cache metadata or coalesce I/O operations.
#[derive(Debug)]
pub struct DefaultParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    object_versioning_type: Option<ObjectVersionType>,
}

impl DefaultParquetFileReaderFactory {
    /// Create a new `DefaultParquetFileReaderFactory`.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            object_versioning_type: None,
        }
    }

    /// Set the object versioning type for reading files.
    ///
    /// This is used to handle different versions of objects in object stores,
    /// ensuring that objects listed during planning are consistent with objects
    /// read during execution.
    pub fn with_object_versioning_type(
        mut self,
        object_versioning_type: Option<ObjectVersionType>,
    ) -> Self {
        self.object_versioning_type = object_versioning_type;
        self
    }
}

/// Implements [`AsyncFileReader`] for a parquet file in object storage.
///
/// This implementation uses the [`ParquetObjectReader`] to read data from the
/// object store on demand, as required, tracking the number of bytes read.
///
/// This implementation does not coalesce I/O operations or cache bytes. Such
/// optimizations can be done either at the object store level or by providing a
/// custom implementation of [`ParquetFileReaderFactory`].
pub struct ParquetFileReader {
    pub file_metrics: ParquetFileMetrics,
    pub inner: ParquetObjectReader,
    pub partitioned_file: PartitionedFile,
}

impl AsyncFileReader for ParquetFileReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let bytes_scanned = range.end - range.start;
        self.file_metrics.bytes_scanned.add(bytes_scanned as usize);
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        self.inner.get_metadata(options)
    }
}

impl Drop for ParquetFileReader {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        // Multiple ParquetFileReaders may run, so we set_total to avoid adding the total multiple times
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.partitioned_file.object_meta.size as usize);
    }
}

impl ParquetFileReaderFactory for DefaultParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_metrics = ParquetFileMetrics::new(
            partition_index,
            partitioned_file.object_meta.location.as_ref(),
            metrics,
        );
        let store = Arc::clone(&self.store);
        let mut inner = ParquetObjectReader::new_with_meta(
            store,
            partitioned_file.object_meta.clone(),
        )
        .with_object_versioning_type(self.object_versioning_type.clone());

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        Ok(Box::new(ParquetFileReader {
            inner,
            file_metrics,
            partitioned_file,
        }))
    }
}

/// Implementation of [`ParquetFileReaderFactory`] supporting the caching of footer and page
/// metadata. Reads and updates the [`FileMetadataCache`] with the [`ParquetMetaData`] data.
///
/// [`CachedParquetFileReader::get_metadata`] forwards the [`parquet::file::metadata::PageIndexPolicy`] from
/// [`ArrowReaderOptions`] to [`DFParquetMetadata::fetch_metadata`], so callers such as the
/// parquet opener can skip page-index I/O during the initial metadata load.
#[derive(Debug)]
pub struct CachedParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    metadata_cache: Arc<dyn FileMetadataCache>,
    object_versioning_type: Option<ObjectVersionType>,
    discovered_versions: DiscoveredVersions,
}

impl CachedParquetFileReaderFactory {
    pub fn new(
        store: Arc<dyn ObjectStore>,
        metadata_cache: Arc<dyn FileMetadataCache>,
    ) -> Self {
        Self {
            store,
            metadata_cache,
            object_versioning_type: None,
            discovered_versions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set the object versioning type for reading files.
    ///
    /// This is used to handle different versions of objects in object stores,
    /// ensuring that objects listed during planning are consistent with objects
    /// read during execution.
    pub fn with_object_versioning_type(
        mut self,
        object_versioning_type: Option<ObjectVersionType>,
    ) -> Self {
        self.object_versioning_type = object_versioning_type;
        self
    }
}

impl ParquetFileReaderFactory for CachedParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_metrics = ParquetFileMetrics::new(
            partition_index,
            partitioned_file.object_meta.location.as_ref(),
            metrics,
        );
        let store = Arc::clone(&self.store);
        let mut partitioned_file = partitioned_file;
        apply_discovered_version(
            &self.discovered_versions,
            &mut partitioned_file.object_meta,
        );

        let mut inner = ParquetObjectReader::new_with_meta(
            store,
            partitioned_file.object_meta.clone(),
        )
        .with_object_versioning_type(self.object_versioning_type.clone());

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        Ok(Box::new(
            CachedParquetFileReader::new(
                file_metrics,
                Arc::clone(&self.store),
                inner,
                partitioned_file,
                Arc::clone(&self.metadata_cache),
                metadata_size_hint,
            )
            .with_object_versioning_type(self.object_versioning_type.clone())
            .with_discovered_versions(Arc::clone(&self.discovered_versions)),
        ))
    }
}

/// Implements [`AsyncFileReader`] for a Parquet file in object storage. Reads the file metadata
/// from the [`FileMetadataCache`], if available, otherwise reads it directly from the file and then
/// updates the cache.
pub struct CachedParquetFileReader {
    pub file_metrics: ParquetFileMetrics,
    store: Arc<dyn ObjectStore>,
    pub inner: ParquetObjectReader,
    partitioned_file: PartitionedFile,
    metadata_cache: Arc<dyn FileMetadataCache>,
    metadata_size_hint: Option<usize>,
    object_versioning_type: Option<ObjectVersionType>,
    discovered_versions: DiscoveredVersions,
}

impl CachedParquetFileReader {
    pub fn new(
        file_metrics: ParquetFileMetrics,
        store: Arc<dyn ObjectStore>,
        inner: ParquetObjectReader,
        partitioned_file: PartitionedFile,
        metadata_cache: Arc<dyn FileMetadataCache>,
        metadata_size_hint: Option<usize>,
    ) -> Self {
        Self {
            file_metrics,
            store,
            inner,
            partitioned_file,
            metadata_cache,
            metadata_size_hint,
            object_versioning_type: None,
            discovered_versions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Pin footer and page reads to the listed object generation.
    ///
    /// Must match the `object_versioning_type` already applied to `inner`.
    /// Defaults to `None` so existing `new` callers keep compiling.
    pub fn with_object_versioning_type(
        mut self,
        object_versioning_type: Option<ObjectVersionType>,
    ) -> Self {
        self.object_versioning_type = object_versioning_type;
        self
    }

    fn with_discovered_versions(mut self, pins: DiscoveredVersions) -> Self {
        self.discovered_versions = pins;
        self
    }
}

impl AsyncFileReader for CachedParquetFileReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let bytes_scanned = range.end - range.start;
        self.file_metrics.bytes_scanned.add(bytes_scanned as usize);
        self.inner.get_bytes(range)
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        self.file_metrics.bytes_scanned.add(total as usize);
        self.inner.get_byte_ranges(ranges)
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        let object_meta = self.partitioned_file.object_meta.clone();
        let metadata_cache = Arc::clone(&self.metadata_cache);

        async move {
            #[cfg(feature = "parquet_encryption")]
            let file_decryption_properties = options
                .and_then(|o| o.file_decryption_properties())
                .map(Arc::clone);

            #[cfg(not(feature = "parquet_encryption"))]
            let file_decryption_properties = None;

            let page_index_policy = options.map(|o| o.column_index_policy());

            let mut object_meta = object_meta;
            // ListObjectsV2 omits version ids. HEAD can supply one, but only
            // if it is still the listed generation; otherwise later page
            // reads would pin a replacement while footer ranges still use
            // the listed size.
            if matches!(
                self.object_versioning_type,
                Some(ObjectVersionType::Version)
            ) && object_meta.version.is_none()
                && let Ok(head) = self.store.head(&object_meta.location).await
                && let Some(version) =
                    version_from_head_if_same_generation(&object_meta, &head)
            {
                object_meta.version = Some(version.clone());
                self.partitioned_file.object_meta.version = Some(version.clone());
                record_discovered_version(
                    &self.discovered_versions,
                    &object_meta,
                    &version,
                );
                self.inner.set_object_version(version);
            }

            DFParquetMetadata::new(&self.store, &object_meta)
                .with_decryption_properties(file_decryption_properties)
                .with_file_metadata_cache(Some(Arc::clone(&metadata_cache)))
                .with_metadata_size_hint(self.metadata_size_hint)
                .with_page_index_policy(page_index_policy)
                .with_object_versioning_type(self.object_versioning_type.clone())
                .fetch_metadata()
                .await
                .map_err(|e| {
                    parquet::errors::ParquetError::General(format!(
                        "Failed to fetch metadata for file {}: {e}",
                        object_meta.location,
                    ))
                })
        }
        .boxed()
    }
}

impl Drop for CachedParquetFileReader {
    fn drop(&mut self) {
        self.file_metrics
            .scan_efficiency_ratio
            .add_part(self.file_metrics.bytes_scanned.value());
        // Multiple ParquetFileReaders may run, so we set_total to avoid adding the total multiple times
        self.file_metrics
            .scan_efficiency_ratio
            .set_total(self.partitioned_file.object_meta.size as usize);
    }
}

/// Wrapper to implement [`FileMetadata`] for [`ParquetMetaData`].
pub struct CachedParquetMetaData(Arc<ParquetMetaData>);

impl CachedParquetMetaData {
    pub fn new(metadata: Arc<ParquetMetaData>) -> Self {
        Self(metadata)
    }

    pub fn parquet_metadata(&self) -> &Arc<ParquetMetaData> {
        &self.0
    }
}

impl FileMetadata for CachedParquetMetaData {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn memory_size(&self) -> usize {
        self.0.memory_size()
    }

    fn extra_info(&self) -> HashMap<String, String> {
        let page_index =
            self.0.column_index().is_some() && self.0.offset_index().is_some();
        HashMap::from([("page_index".to_owned(), page_index.to_string())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use async_trait::async_trait;
    use datafusion_execution::cache::DefaultFilesMetadataCache;
    use futures::stream::BoxStream;
    use object_store::memory::InMemory;
    use object_store::path::Path;
    use object_store::{
        CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectStore,
        ObjectStoreExt, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    };
    use parquet::arrow::ArrowWriter;
    use std::fmt;
    use std::sync::Mutex as StdMutex;

    fn object_meta(e_tag: Option<&str>, version: Option<&str>) -> ObjectMeta {
        ObjectMeta {
            location: Path::from("listing/data.parquet"),
            last_modified: chrono::DateTime::from(std::time::SystemTime::now()),
            size: 100,
            e_tag: e_tag.map(str::to_string),
            version: version.map(str::to_string),
        }
    }

    #[test]
    fn replacement_reader_reuses_the_head_promoted_version() {
        let pins: DiscoveredVersions = Arc::new(Mutex::new(HashMap::new()));
        let listed = object_meta(Some("etag-a"), None);
        record_discovered_version(&pins, &listed, "v-listed");

        let mut replacement = listed.clone();
        apply_discovered_version(&pins, &mut replacement);
        assert_eq!(
            replacement.version.as_deref(),
            Some("v-listed"),
            "a second reader for the same listing must keep the HEAD version"
        );

        let mut other_generation = object_meta(Some("etag-b"), None);
        apply_discovered_version(&pins, &mut other_generation);
        assert!(
            other_generation.version.is_none(),
            "a different listed ETag must not inherit another generation's version"
        );
    }

    /// `HEAD` reports a version id; range reads are recorded.
    #[derive(Debug)]
    struct VersionHeadStore {
        inner: InMemory,
        version: String,
        reads: StdMutex<Vec<GetOptions>>,
    }

    impl fmt::Display for VersionHeadStore {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "VersionHeadStore")
        }
    }

    #[async_trait]
    impl ObjectStore for VersionHeadStore {
        async fn put_opts(
            &self,
            location: &Path,
            payload: PutPayload,
            opts: PutOptions,
        ) -> object_store::Result<PutResult> {
            self.inner.put_opts(location, payload, opts).await
        }

        async fn put_multipart_opts(
            &self,
            location: &Path,
            opts: PutMultipartOptions,
        ) -> object_store::Result<Box<dyn MultipartUpload>> {
            self.inner.put_multipart_opts(location, opts).await
        }

        async fn get_opts(
            &self,
            location: &Path,
            options: GetOptions,
        ) -> object_store::Result<GetResult> {
            if !options.head {
                self.reads.lock().expect("reads lock").push(options.clone());
            }
            let mut result = self.inner.get_opts(location, options.clone()).await?;
            if options.head {
                result.meta.version = Some(self.version.clone());
            }
            Ok(result)
        }

        fn delete_stream(
            &self,
            locations: BoxStream<'static, object_store::Result<Path>>,
        ) -> BoxStream<'static, object_store::Result<Path>> {
            self.inner.delete_stream(locations)
        }

        fn list(
            &self,
            prefix: Option<&Path>,
        ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
            self.inner.list(prefix)
        }

        async fn list_with_delimiter(
            &self,
            prefix: Option<&Path>,
        ) -> object_store::Result<ListResult> {
            self.inner.list_with_delimiter(prefix).await
        }

        async fn copy_opts(
            &self,
            from: &Path,
            to: &Path,
            options: CopyOptions,
        ) -> object_store::Result<()> {
            self.inner.copy_opts(from, to, options).await
        }
    }

    fn write_tiny_parquet() -> Vec<u8> {
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as _],
        )
        .expect("batch");
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, None).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        buf
    }

    #[tokio::test]
    async fn bloom_filter_replacement_reader_pins_the_discovered_version() {
        let buf = write_tiny_parquet();
        let inner = InMemory::new();
        let location = Path::from("listing/data.parquet");
        inner.put(&location, buf.into()).await.expect("put parquet");
        let listed = inner.head(&location).await.expect("list");
        assert!(
            listed.version.is_none(),
            "this test needs a listing with no version id"
        );
        let store = Arc::new(VersionHeadStore {
            inner,
            version: "v-listed".to_string(),
            reads: StdMutex::new(Vec::new()),
        });
        let factory = CachedParquetFileReaderFactory::new(
            Arc::clone(&store) as Arc<dyn ObjectStore>,
            Arc::new(DefaultFilesMetadataCache::new(64 * 1024 * 1024)),
        )
        .with_object_versioning_type(Some(ObjectVersionType::Version));
        let file = PartitionedFile::new_from_meta(listed);
        let metrics = ExecutionPlanMetricsSet::new();

        let mut metadata_reader = factory
            .create_reader(0, file.clone(), None, &metrics)
            .expect("metadata reader");
        metadata_reader
            .get_metadata(None)
            .await
            .expect("metadata load must HEAD and pin the listed generation");
        drop(metadata_reader);
        store.reads.lock().expect("reads lock").clear();

        let mut replacement = factory
            .create_reader(0, file, None, &metrics)
            .expect("replacement reader");
        replacement
            .get_bytes(0..8)
            .await
            .expect("page read of the replacement reader");

        let reads = store.reads.lock().expect("reads lock").clone();
        assert!(
            !reads.is_empty(),
            "the replacement reader issued no range request"
        );
        for options in &reads {
            assert_eq!(
                options.version.as_deref(),
                Some("v-listed"),
                "bloom-filter replacement must keep the HEAD version, not fall back to If-Match: {options:?}"
            );
            assert!(
                options.if_match.is_none(),
                "a version pin must not also send If-Match: {options:?}"
            );
        }
    }
}
