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
use crate::metadata::DFParquetMetadata;
use bytes::Bytes;
use datafusion_datasource::PartitionedFile;
use datafusion_execution::cache::cache_manager::FileMetadata;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use futures::FutureExt;
use futures::future::BoxFuture;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::{AsyncFileReader, ObjectVersionType, ParquetObjectReader};
use parquet::file::metadata::ParquetMetaData;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// Interface for reading parquet files.
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
    /// This is used to handle different versions of objects in object stores.
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
        let mut inner = ParquetObjectReader::new_with_meta(store, partitioned_file.object_meta.clone())
            .with_object_versioning_type(self.object_versioning_type.clone());

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        let filename = partitioned_file.object_meta.location.clone();
        let file_size = partitioned_file.object_meta.size as u64;
        let reader = Box::new(ParquetFileReader {
            inner,
            file_metrics,
            partitioned_file,
        });

        // For small files, wrap with a prefetching reader that fetches the
        // entire file in a single GET, avoiding per-row-group round-trips.
        if file_size <= DEFAULT_PREFETCH_SIZE_THRESHOLD {
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/datafusion_debug.log") {
                    let _ = writeln!(f, "[parquet] Wrapping reader with PrefetchingAsyncReader: {filename} ({file_size} bytes)");
                }
            }
            Ok(Box::new(PrefetchingAsyncReader::new(reader, file_size)))
        } else {
            {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/datafusion_debug.log") {
                    let _ = writeln!(f, "[parquet] Regular reader: {filename} ({file_size} bytes)");
                }
            }
            Ok(reader)
        }
    }
}

/// Implementation of [`ParquetFileReaderFactory`] supporting the caching of footer and page
/// metadata. Reads and updates the [`FileMetadataCache`] with the [`ParquetMetaData`] data.
/// This reader always loads the entire metadata (including page index, unless the file is
/// encrypted), even if not required by the current query, to ensure it is always available for
/// those that need it.
#[derive(Debug)]
pub struct CachedParquetFileReaderFactory {
    store: Arc<dyn ObjectStore>,
    metadata_cache: Arc<dyn FileMetadataCache>,
    object_versioning_type: Option<ObjectVersionType>,
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
        }
    }

    /// Set the object versioning type for reading files.
    /// This is used to handle different versions of objects in object stores.
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

        let mut inner =
            ParquetObjectReader::new_with_meta(store, partitioned_file.object_meta.clone())
                .with_object_versioning_type(self.object_versioning_type.clone());

        if let Some(hint) = metadata_size_hint {
            inner = inner.with_footer_size_hint(hint)
        };

        let file_size = partitioned_file.object_meta.size as u64;
        let reader = Box::new(CachedParquetFileReader::new(
            file_metrics,
            Arc::clone(&self.store),
            inner,
            partitioned_file,
            Arc::clone(&self.metadata_cache),
            metadata_size_hint,
        ));

        if file_size <= DEFAULT_PREFETCH_SIZE_THRESHOLD {
            debug_log(&format!(
                "[CachedFactory] Wrapping with PrefetchingAsyncReader: {file_size} bytes"
            ));
            Ok(Box::new(PrefetchingAsyncReader::new(reader, file_size)))
        } else {
            Ok(reader)
        }
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
        }
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
        #[cfg_attr(not(feature = "parquet_encryption"), expect(unused_variables))]
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

            DFParquetMetadata::new(&self.store, &object_meta)
                .with_decryption_properties(file_decryption_properties)
                .with_file_metadata_cache(Some(Arc::clone(&metadata_cache)))
                .with_metadata_size_hint(self.metadata_size_hint)
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

/// Default threshold below which files are fully buffered in memory on first access.
const DEFAULT_PREFETCH_SIZE_THRESHOLD: u64 = 10 * 1024 * 1024; // 10 MB

fn debug_log(msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/datafusion_debug.log")
    {
        let _ = writeln!(f, "{msg}");
    }
}

/// An [`AsyncFileReader`] wrapper that fetches the entire file into memory on
/// the first IO call, then serves all subsequent range requests from the buffer.
///
/// This eliminates per-row-group S3 round-trips for small files where the cost
/// of N sequential HTTP range requests far exceeds a single GET for the whole file.
pub struct PrefetchingAsyncReader {
    inner: Box<dyn AsyncFileReader + Send>,
    file_size: u64,
    buffer: Option<Bytes>,
}

impl PrefetchingAsyncReader {
    /// Wrap an existing reader. `file_size` determines whether the file will be
    /// fully buffered (if <= `DEFAULT_PREFETCH_SIZE_THRESHOLD`).
    pub fn new(inner: Box<dyn AsyncFileReader + Send>, file_size: u64) -> Self {
        Self {
            inner,
            file_size,
            buffer: None,
        }
    }

    /// Returns true if this file should be fully prefetched.
    fn should_prefetch(&self) -> bool {
        self.file_size <= DEFAULT_PREFETCH_SIZE_THRESHOLD
    }

    /// Ensure the file is fully buffered. If already buffered, this is a no-op.
    async fn ensure_buffered(&mut self) -> parquet::errors::Result<()> {
        if self.buffer.is_some() {
            return Ok(());
        }
        debug_log(&format!(
            "[PrefetchingAsyncReader] fetching entire file ({} bytes) from S3",
            self.file_size
        ));
        let data = self.inner.get_bytes(0..self.file_size).await?;
        debug_log(&format!(
            "[PrefetchingAsyncReader] fetched {} bytes, buffered",
            data.len()
        ));
        self.buffer = Some(data);
        Ok(())
    }
}

impl AsyncFileReader for PrefetchingAsyncReader {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        if !self.should_prefetch() {
            debug_log(&format!(
                "[PrefetchingAsyncReader] get_bytes passthrough (file too large): {:?}",
                range
            ));
            return self.inner.get_bytes(range);
        }
        let range_clone = range.clone();
        async move {
            self.ensure_buffered().await?;
            let len = range_clone.end - range_clone.start;
            debug_log(&format!(
                "[PrefetchingAsyncReader] get_bytes from buffer: {:?} ({len} bytes)",
                range_clone
            ));
            let buf = self.buffer.as_ref().expect("buffer should be set");
            Ok(buf.slice(range_clone.start as usize..range_clone.end as usize))
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>>
    where
        Self: Send,
    {
        if !self.should_prefetch() {
            debug_log(&format!(
                "[PrefetchingAsyncReader] get_byte_ranges passthrough (file too large): {} ranges",
                ranges.len()
            ));
            return self.inner.get_byte_ranges(ranges);
        }
        async move {
            self.ensure_buffered().await?;
            let total: u64 = ranges.iter().map(|r| r.end - r.start).sum();
            debug_log(&format!(
                "[PrefetchingAsyncReader] get_byte_ranges from buffer: {} ranges, {total} bytes total",
                ranges.len()
            ));
            let buf = self.buffer.as_ref().expect("buffer should be set");
            Ok(ranges
                .into_iter()
                .map(|r| buf.slice(r.start as usize..r.end as usize))
                .collect())
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        if !self.should_prefetch() {
            debug_log("[PrefetchingAsyncReader] get_metadata passthrough (file too large)");
            return self.inner.get_metadata(options);
        }
        async move {
            debug_log("[PrefetchingAsyncReader] get_metadata: prefetching entire file first");
            self.ensure_buffered().await?;
            debug_log("[PrefetchingAsyncReader] get_metadata: delegating to inner reader (from buffer)");
            self.inner.get_metadata(options).await
        }
        .boxed()
    }
}

/// A [`ParquetFileReaderFactory`] that wraps another factory and applies
/// prefetching for files below a size threshold.
///
/// For small files, this fetches the entire file in a single S3 GET on first
/// access, then serves all subsequent range requests (metadata, row groups)
/// from an in-memory buffer. This turns N sequential S3 round-trips into 1.
#[derive(Debug)]
pub struct PrefetchingParquetFileReaderFactory {
    inner: Arc<dyn ParquetFileReaderFactory>,
}

impl PrefetchingParquetFileReaderFactory {
    /// Create a new prefetching factory wrapping the given inner factory.
    pub fn new(inner: Arc<dyn ParquetFileReaderFactory>) -> Self {
        Self { inner }
    }
}

impl ParquetFileReaderFactory for PrefetchingParquetFileReaderFactory {
    fn create_reader(
        &self,
        partition_index: usize,
        partitioned_file: PartitionedFile,
        metadata_size_hint: Option<usize>,
        metrics: &ExecutionPlanMetricsSet,
    ) -> datafusion_common::Result<Box<dyn AsyncFileReader + Send>> {
        let file_size = partitioned_file.object_meta.size as u64;
        let inner = self
            .inner
            .create_reader(partition_index, partitioned_file, metadata_size_hint, metrics)?;

        if file_size <= DEFAULT_PREFETCH_SIZE_THRESHOLD {
            Ok(Box::new(PrefetchingAsyncReader::new(inner, file_size)))
        } else {
            Ok(inner)
        }
    }
}
