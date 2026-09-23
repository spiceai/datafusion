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

use crate::cache::default_cache::DefaultCache;
pub use crate::cache::{Cache, CacheValue, SchemaFingerprint, TableScopedPath};
use datafusion_common::HashMap;
use datafusion_common::heap_size::{DFHeapSize, DFHeapSizeCtx};
use datafusion_common::{Result, Statistics};
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use object_store::ObjectMeta;
use object_store::path::Path;
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

pub const DEFAULT_LIST_FILES_CACHE_MEMORY_LIMIT: usize = 1024 * 1024; // 1MiB

/// Helper function to normalize an optional string (treats empty strings as `None`).
fn normalize_optional_string(opt: &Option<String>) -> Option<&str> {
    match opt {
        Some(s) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    }
}

/// Check if two [`ObjectMeta`] represent the same file version.
///
/// Returns `true` if the files are considered the same version, `false` otherwise.
///
/// Unlike a plain `size` + `last_modified` comparison, this also takes the
/// object `version` and `e_tag` into account, which is required to correctly
/// invalidate cache entries for versioned object stores (e.g. S3 with object
/// versioning enabled) where a new version of an object can share the same size
/// and last-modified timestamp.
///
/// Logic (in priority order):
/// - If `version` is present (non-empty) in BOTH, it is authoritative: same file
///   iff the versions are equal. A difference in `e_tag` presence/value is
///   ignored in this case, because the object store has already told us the
///   definitive version identity.
/// - Otherwise, if `version` presence differs (one side has it, the other does
///   not) -> different file (one read saw a versioned object, the other didn't).
/// - Otherwise (neither side has a usable `version`), fall back to `e_tag`:
///     - both present and equal -> same file
///     - both present and different -> different file
///     - presence differs -> different file
/// - If neither `version` nor `e_tag` is available on either side -> same file
///   (no versioning information to distinguish them).
///
/// Empty strings are normalized to "absent" (see [`normalize_optional_string`]).
pub(crate) fn is_same_file_version(cached: &ObjectMeta, current: &ObjectMeta) -> bool {
    let cached_version = normalize_optional_string(&cached.version);
    let current_version = normalize_optional_string(&current.version);
    let cached_etag = normalize_optional_string(&cached.e_tag);
    let current_etag = normalize_optional_string(&current.e_tag);

    // `version`, when present on BOTH sides, is authoritative and decides on its
    // own. e_tag presence/value differences must not force invalidation here.
    if let (Some(cv), Some(curv)) = (cached_version, current_version) {
        return cv == curv;
    }

    // `version` could not decide. If its presence differs, the two reads
    // disagree on whether the object is versioned -> treat as different files.
    if cached_version.is_some() != current_version.is_some() {
        return false;
    }

    // Neither side has a usable `version`; fall back to `e_tag` semantics.
    match (cached_etag, current_etag) {
        // Both present: same file iff the e_tags match.
        (Some(ce), Some(cure)) => ce == cure,
        // e_tag presence differs -> different files.
        (Some(_), None) | (None, Some(_)) => false,
        // No versioning information available at all -> consider same file.
        (None, None) => true,
    }
}

pub const DEFAULT_LIST_FILES_CACHE_TTL: Option<Duration> = None; // Infinite

pub const DEFAULT_FILE_STATISTICS_MEMORY_LIMIT: usize = 20 * 1024 * 1024; // 20MiB

pub const DEFAULT_METADATA_CACHE_LIMIT: usize = 50 * 1024 * 1024; // 50M

/// A cache for file statistics and orderings.
///
/// This cache stores [`CachedFileMetadata`] which includes:
/// - File metadata for validation (size, last_modified)
/// - Statistics for the file
/// - Ordering information for the file
///
/// If enabled via [`CacheManagerConfig::with_file_statistics_cache`] this
/// cache avoids inferring the same file statistics repeatedly during the
/// session lifetime.
///
/// The typical usage pattern is:
/// 1. Call `get(path)` to check for cached value
/// 2. If `Some(cached)`, validate with
///    `cached.is_valid_for(&current_meta, &current_schema_fingerprint)`
/// 3. If invalid or missing, compute new value and call `put(path, new_value)`
///
/// See [`crate::runtime_env::RuntimeEnv`] for more details
pub type FileStatisticsCache = dyn Cache<TableScopedPath, CachedFileMetadata>;

/// A cache for storing the [`ObjectMeta`]s that result from listing a path.
///
/// Listing a path means doing an object store "list" operation or `ls`
/// command on the local filesystem. This operation can be expensive,
/// especially when done over remote object stores.
///
/// The cache key is always the table's base path, ensuring a stable cache key.
/// The cached value is a [`CachedFileList`] containing the files and a timestamp.
///
/// Partition filtering is done after retrieval using [`CachedFileList::files_matching_prefix`].
///
/// See [`crate::runtime_env::RuntimeEnv`] for more details.
pub type ListFilesCache = dyn Cache<TableScopedPath, CachedFileList>;

/// A cache for storing file-embedded metadata.
///
/// This cache stores per-file metadata in the form of [`CachedFileMetadataEntry`],
/// which includes the [`ObjectMeta`] for validation.
///
/// For example, the built in [`ListingTable`] uses this cache to avoid parsing
/// Parquet footers multiple times for the same file.
///
/// The typical usage pattern is:
/// 1. Call `get(path)` to check for cached value
/// 2. If `Some(cached)`, validate with `cached.is_valid_for(&current_meta)`
/// 3. If invalid or missing, compute new value and call `put(path, new_value)`
///
/// See [`crate::runtime_env::RuntimeEnv`] for more details.
///
/// [`ListingTable`]: https://docs.rs/datafusion/latest/datafusion/datasource/listing/struct.ListingTable.html
pub type FileMetadataCache = dyn Cache<Path, CachedFileMetadataEntry>;

/// Cached metadata for a file, including statistics and ordering.
///
/// This struct embeds the [`ObjectMeta`] used for cache validation,
/// the `file_schema` fingerprint, cached statistics, and ordering information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedFileMetadata {
    /// File metadata used for cache validation (size, last_modified).
    pub meta: ObjectMeta,
    /// Fingerprint of the `file_schema` used to compute `statistics`.
    pub schema_fingerprint: Arc<SchemaFingerprint>,
    /// Cached statistics for the file, if available.
    pub statistics: Arc<Statistics>,
    /// Cached ordering for the file.
    pub ordering: Option<LexOrdering>,
}

impl CachedFileMetadata {
    /// Create a new cached file metadata entry.
    pub fn new(
        meta: ObjectMeta,
        schema_fingerprint: Arc<SchemaFingerprint>,
        statistics: Arc<Statistics>,
        ordering: Option<LexOrdering>,
    ) -> Self {
        Self {
            meta,
            schema_fingerprint,
            statistics,
            ordering,
        }
    }

    /// Check if this cached entry is still valid for the given metadata.
    ///
    /// Returns true if the file size, last modified time, object version
    /// (`version` / `e_tag`), and schema all match.
    pub fn is_valid_for(
        &self,
        current_meta: &ObjectMeta,
        current_schema_fingerprint: &Arc<SchemaFingerprint>,
    ) -> bool {
        self.meta.size == current_meta.size
            && self.meta.last_modified == current_meta.last_modified
            && is_same_file_version(&self.meta, current_meta)
            && (Arc::ptr_eq(&self.schema_fingerprint, current_schema_fingerprint)
                || self.schema_fingerprint.as_ref()
                    == current_schema_fingerprint.as_ref())
    }
}

impl CacheValue for CachedFileMetadata {
    fn size(&self) -> usize {
        DFHeapSize::heap_size(self, &mut DFHeapSizeCtx::default())
    }
}

impl DFHeapSize for CachedFileMetadata {
    fn heap_size(&self, ctx: &mut DFHeapSizeCtx) -> usize {
        self.meta.size.heap_size(ctx)
            + self.meta.last_modified.heap_size(ctx)
            + self.meta.version.heap_size(ctx)
            + self.meta.e_tag.heap_size(ctx)
            + self.meta.location.as_ref().heap_size(ctx)
            + self.statistics.heap_size(ctx)
        // Do not deep-count `schema_fingerprint`: each ListingTable shares one
        // fingerprint across all cached files.
        //TODO add ordering once LexOrdering/PhysicalExpr implements DFHeapSize
    }
}

/// Cached file listing.
///
/// TTL expiration is handled internally by the cache implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct CachedFileList {
    /// The cached file list.
    pub files: Arc<Vec<ObjectMeta>>,
}

impl CachedFileList {
    /// Create a new cached file list.
    pub fn new(files: Vec<ObjectMeta>) -> Self {
        Self {
            files: Arc::new(files),
        }
    }

    /// Filter the files by prefix.
    fn filter_by_prefix(&self, prefix: &Option<Path>) -> Vec<ObjectMeta> {
        match prefix {
            Some(prefix) => self
                .files
                .iter()
                .filter(|meta| meta.location.as_ref().starts_with(prefix.as_ref()))
                .cloned()
                .collect(),
            None => self.files.as_ref().clone(),
        }
    }

    /// Returns files matching the given prefix.
    ///
    /// When prefix is `None`, returns a clone of the `Arc` (no data copy).
    /// When filtering is needed, returns a new `Arc` with filtered results (clones each matching [`ObjectMeta`]).
    pub fn files_matching_prefix(&self, prefix: &Option<Path>) -> Arc<Vec<ObjectMeta>> {
        match prefix {
            None => Arc::clone(&self.files),
            Some(p) => Arc::new(self.filter_by_prefix(&Some(p.clone()))),
        }
    }
}

impl CacheValue for CachedFileList {
    fn size(&self) -> usize {
        self.files.capacity() * size_of::<ObjectMeta>()
            + self
                .files
                .iter()
                .map(meta_heap_bytes)
                .reduce(|acc, b| acc + b)
                .unwrap_or(0)
    }
}

/// Calculates the number of bytes an [`ObjectMeta`] occupies in the heap.
pub fn meta_heap_bytes(object_meta: &ObjectMeta) -> usize {
    let mut size = object_meta.location.as_ref().len();

    if let Some(e) = &object_meta.e_tag {
        size += e.len();
    }
    if let Some(v) = &object_meta.version {
        size += v.len();
    }

    size
}

impl Deref for CachedFileList {
    type Target = Arc<Vec<ObjectMeta>>;
    fn deref(&self) -> &Self::Target {
        &self.files
    }
}

impl From<Vec<ObjectMeta>> for CachedFileList {
    fn from(files: Vec<ObjectMeta>) -> Self {
        Self::new(files)
    }
}

/// Generic file-embedded metadata used with [`FileMetadataCache`].
///
/// For example, Parquet footers and page metadata can be represented
/// using this trait.
///
/// See [`crate::runtime_env::RuntimeEnv`] for more details
pub trait FileMetadata: Any + Send + Sync {
    /// Returns the file metadata as [`Any`] so that it can be downcast to a specific
    /// implementation.
    fn as_any(&self) -> &dyn Any;

    /// Returns the size of the metadata in bytes.
    fn memory_size(&self) -> usize;

    /// Returns extra information about this entry
    fn extra_info(&self) -> HashMap<String, String>;
}

/// Cached file metadata entry with validation information.
#[derive(Clone)]
pub struct CachedFileMetadataEntry {
    /// File metadata used for cache validation (size, last_modified).
    pub meta: ObjectMeta,
    /// The cached file metadata.
    pub file_metadata: Arc<dyn FileMetadata>,
}

impl CacheValue for CachedFileMetadataEntry {
    fn size(&self) -> usize {
        self.file_metadata.memory_size()
    }
}

impl CachedFileMetadataEntry {
    /// Create a new cached file metadata entry.
    pub fn new(meta: ObjectMeta, file_metadata: Arc<dyn FileMetadata>) -> Self {
        Self {
            meta,
            file_metadata,
        }
    }

    /// Check if this cached entry is still valid for the given metadata.
    ///
    /// Returns true if the file size, last modified time, and object version
    /// (`version` / `e_tag`) all match.
    pub fn is_valid_for(&self, current_meta: &ObjectMeta) -> bool {
        self.meta.size == current_meta.size
            && self.meta.last_modified == current_meta.last_modified
            && is_same_file_version(&self.meta, current_meta)
    }
}

impl Debug for CachedFileMetadataEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedFileMetadataEntry")
            .field("meta", &self.meta)
            .field("memory_size", &self.file_metadata.memory_size())
            .finish()
    }
}

/// Manages various caches used in DataFusion.
///
/// Following DataFusion design principles, DataFusion provides default cache
/// implementations, while also allowing users to provide their own custom cache
/// implementations by implementing the relevant traits.
///
/// See [`CacheManagerConfig`] for configuration options.
#[derive(Debug)]
pub struct CacheManager {
    file_statistic_cache: Option<Arc<FileStatisticsCache>>,
    list_files_cache: Option<Arc<ListFilesCache>>,
    file_metadata_cache: Arc<FileMetadataCache>,
}

impl CacheManager {
    pub fn try_new(config: &CacheManagerConfig) -> Result<Arc<Self>> {
        let file_statistic_cache: Option<Arc<FileStatisticsCache>> =
            match &config.file_statistics_cache {
                Some(fsc) if config.file_statistics_cache_limit > 0 => {
                    fsc.update_cache_limit(config.file_statistics_cache_limit);
                    Some(Arc::clone(fsc))
                }
                None if config.file_statistics_cache_limit > 0 => Some(Arc::new(
                    DefaultCache::<TableScopedPath, CachedFileMetadata>::new(
                        config.file_statistics_cache_limit,
                    )
                    .with_name("DefaultFileStatisticsCache"),
                )),
                _ => None,
            };

        let list_files_cache: Option<Arc<ListFilesCache>> = match &config.list_files_cache
        {
            Some(lfc) if config.list_files_cache_limit > 0 => {
                // the cache memory limit or ttl might have changed, ensure they are updated
                lfc.update_cache_limit(config.list_files_cache_limit);
                // Only update TTL if explicitly set in config, otherwise preserve the cache's existing TTL
                if let Some(ttl) = config.list_files_cache_ttl {
                    lfc.update_cache_ttl(Some(ttl));
                }
                Some(Arc::clone(lfc))
            }
            None if config.list_files_cache_limit > 0 => Some(Arc::new(
                DefaultCache::<TableScopedPath, CachedFileList>::new_with_ttl(
                    config.list_files_cache_limit,
                    config.list_files_cache_ttl,
                )
                .with_name("DefaultListFilesCache"),
            )),
            _ => None,
        };

        let file_metadata_cache = config
            .file_metadata_cache
            .as_ref()
            .map(Arc::clone)
            .unwrap_or_else(|| {
                Arc::new(
                    DefaultCache::new(config.metadata_cache_limit)
                        .with_name("DefaultFileMetadataCache"),
                )
            });

        // the cache memory limit might have changed, ensure the limit is updated
        file_metadata_cache.update_cache_limit(config.metadata_cache_limit);

        Ok(Arc::new(CacheManager {
            file_statistic_cache,
            list_files_cache,
            file_metadata_cache,
        }))
    }

    /// Get the file statistics cache.
    pub fn get_file_statistic_cache(&self) -> Option<Arc<FileStatisticsCache>> {
        self.file_statistic_cache.clone()
    }

    /// Get the memory limit of the file statistics cache.
    pub fn get_file_statistic_cache_limit(&self) -> usize {
        self.file_statistic_cache
            .as_ref()
            .map_or(0, |c| c.cache_limit())
    }

    /// Get the cache for storing the result of listing [`ObjectMeta`]s under the same path.
    pub fn get_list_files_cache(&self) -> Option<Arc<ListFilesCache>> {
        self.list_files_cache.clone()
    }

    /// Get the memory limit of the list files cache.
    pub fn get_list_files_cache_limit(&self) -> usize {
        self.list_files_cache
            .as_ref()
            .map_or(0, |c| c.cache_limit())
    }

    /// Get the TTL (time-to-live) of the list files cache.
    pub fn get_list_files_cache_ttl(&self) -> Option<Duration> {
        self.list_files_cache.as_ref().and_then(|c| c.cache_ttl())
    }

    /// Get the file embedded metadata cache.
    pub fn get_file_metadata_cache(&self) -> Arc<FileMetadataCache> {
        Arc::clone(&self.file_metadata_cache)
    }

    /// Get the limit of the file embedded metadata cache.
    pub fn get_metadata_cache_limit(&self) -> usize {
        self.file_metadata_cache.cache_limit()
    }
}

#[derive(Clone)]
pub struct CacheManagerConfig {
    /// Enable caching of file statistics when listing files.
    /// Enabling the cache avoids repeatedly reading file statistics in a DataFusion session.
    /// Default is enabled. Currently only Parquet files are supported.
    pub file_statistics_cache: Option<Arc<FileStatisticsCache>>,
    /// Limit of the file statistics cache, in bytes. Default: 20MiB.
    pub file_statistics_cache_limit: usize,
    /// Enable caching of file metadata when listing files.
    /// Enabling the cache avoids repeat list and object metadata fetch operations, which may be
    /// expensive in certain situations (e.g. remote object storage), for objects under paths that
    /// are cached.
    /// Note that if this option is enabled, DataFusion will not see any updates to the underlying
    /// storage for at least `list_files_cache_ttl` duration.
    /// Default is enabled.
    pub list_files_cache: Option<Arc<ListFilesCache>>,
    /// Limit of the `list_files_cache`, in bytes. Default: 1MiB.
    pub list_files_cache_limit: usize,
    /// The duration the list files cache will consider an entry valid after insertion. Note that
    /// changes to the underlying storage system, such as adding or removing data, will not be
    /// visible until an entry expires. Default: None (infinite).
    pub list_files_cache_ttl: Option<Duration>,
    /// Cache of file-embedded metadata, used to avoid reading it multiple times when processing a
    /// data file (e.g., Parquet footer and page metadata).
    /// If not provided, the [`CacheManager`] will create it.
    pub file_metadata_cache: Option<Arc<FileMetadataCache>>,
    /// Limit of the file-embedded metadata cache, in bytes.
    pub metadata_cache_limit: usize,
}

impl Default for CacheManagerConfig {
    fn default() -> Self {
        Self {
            file_statistics_cache: Default::default(),
            file_statistics_cache_limit: DEFAULT_FILE_STATISTICS_MEMORY_LIMIT,
            list_files_cache: Default::default(),
            list_files_cache_limit: DEFAULT_LIST_FILES_CACHE_MEMORY_LIMIT,
            list_files_cache_ttl: DEFAULT_LIST_FILES_CACHE_TTL,
            file_metadata_cache: Default::default(),
            metadata_cache_limit: DEFAULT_METADATA_CACHE_LIMIT,
        }
    }
}

impl CacheManagerConfig {
    /// Set the cache for file statistics.
    pub fn with_file_statistics_cache(
        mut self,
        cache: Option<Arc<FileStatisticsCache>>,
    ) -> Self {
        self.file_statistics_cache = cache;
        self
    }

    /// Specifies the memory limit for the file statistics cache, in bytes.
    pub fn with_file_statistics_cache_limit(mut self, limit: usize) -> Self {
        self.file_statistics_cache_limit = limit;
        self
    }

    /// Set the cache for listing files.
    ///
    /// Default is `None` (disabled).
    pub fn with_list_files_cache(mut self, cache: Option<Arc<ListFilesCache>>) -> Self {
        self.list_files_cache = cache;
        self
    }

    /// Sets the limit of the list files cache, in bytes.
    ///
    /// Default: 1MiB (1,048,576 bytes).
    pub fn with_list_files_cache_limit(mut self, limit: usize) -> Self {
        self.list_files_cache_limit = limit;
        self
    }

    /// Sets the TTL (time-to-live) for entries in the list files cache.
    ///
    /// Default: None (infinite).
    pub fn with_list_files_cache_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.list_files_cache_ttl = ttl;
        self
    }

    /// Sets the cache for file-embedded metadata.
    pub fn with_file_metadata_cache(
        mut self,
        cache: Option<Arc<FileMetadataCache>>,
    ) -> Self {
        self.file_metadata_cache = cache;
        self
    }

    /// Sets the limit of the file-embedded metadata cache, in bytes.
    pub fn with_metadata_cache_limit(mut self, limit: usize) -> Self {
        self.metadata_cache_limit = limit;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::DateTime;

    /// Build an [`ObjectMeta`] with the given optional `version` and `e_tag`.
    /// Empty strings are passed through verbatim to exercise the empty-string
    /// normalization in [`is_same_file_version`].
    fn meta_with_version_etag(version: Option<&str>, e_tag: Option<&str>) -> ObjectMeta {
        ObjectMeta {
            location: Path::from("test"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: e_tag.map(str::to_string),
            version: version.map(str::to_string),
        }
    }

    #[test]
    fn test_is_same_file_version_both_none() {
        // No versioning info on either side -> same file.
        let cached = meta_with_version_etag(None, None);
        let current = meta_with_version_etag(None, None);
        assert!(is_same_file_version(&cached, &current));
    }

    #[test]
    fn test_is_same_file_version_empty_string_normalization() {
        // Empty strings are treated as absent, so empty/None on both sides ->
        // same file (no versioning info), and empty == None is indistinguishable.
        let cached = meta_with_version_etag(Some(""), Some(""));
        let current = meta_with_version_etag(None, None);
        assert!(is_same_file_version(&cached, &current));

        // An empty `version` must not be treated as an authoritative match
        // against a real version; it normalizes to "absent" and the present
        // version on the other side forces a difference.
        let cached_empty = meta_with_version_etag(Some(""), None);
        let current_v = meta_with_version_etag(Some("v2"), None);
        assert!(!is_same_file_version(&cached_empty, &current_v));
    }

    #[test]
    fn test_is_same_file_version_version_match_etag_one_side() {
        // Versions present and equal on both sides: `version` is authoritative,
        // so e_tag presence differing on one side must NOT force invalidation.
        let cached = meta_with_version_etag(Some("v1"), None);
        let current = meta_with_version_etag(Some("v1"), Some("etag-current"));
        assert!(is_same_file_version(&cached, &current));

        let cached = meta_with_version_etag(Some("v1"), Some("etag-cached"));
        let current = meta_with_version_etag(Some("v1"), None);
        assert!(is_same_file_version(&cached, &current));

        // Versions present and equal but e_tags differ on both sides ->
        // still the same file (version wins over a stale/changed e_tag).
        let cached = meta_with_version_etag(Some("v1"), Some("etag-a"));
        let current = meta_with_version_etag(Some("v1"), Some("etag-b"));
        assert!(is_same_file_version(&cached, &current));
    }

    #[test]
    fn test_is_same_file_version_version_mismatch() {
        // Versions present on both sides but different -> different file,
        // regardless of e_tag.
        let cached = meta_with_version_etag(Some("v1"), Some("same-etag"));
        let current = meta_with_version_etag(Some("v2"), Some("same-etag"));
        assert!(!is_same_file_version(&cached, &current));
    }

    #[test]
    fn test_is_same_file_version_version_presence_differs() {
        // `version` present on one side only -> different file (the reads
        // disagree on whether the object is versioned).
        let cached = meta_with_version_etag(Some("v1"), None);
        let current = meta_with_version_etag(None, None);
        assert!(!is_same_file_version(&cached, &current));

        // Even with a matching e_tag, a one-sided version still invalidates.
        let cached = meta_with_version_etag(Some("v1"), Some("etag"));
        let current = meta_with_version_etag(None, Some("etag"));
        assert!(!is_same_file_version(&cached, &current));
    }

    #[test]
    fn test_is_same_file_version_etag_only() {
        // No versions on either side: fall back to e_tag.
        // Matching e_tag -> same file.
        let cached = meta_with_version_etag(None, Some("etag-1"));
        let current = meta_with_version_etag(None, Some("etag-1"));
        assert!(is_same_file_version(&cached, &current));

        // Mismatched e_tag with no versions -> different file.
        let cached = meta_with_version_etag(None, Some("etag-1"));
        let current = meta_with_version_etag(None, Some("etag-2"));
        assert!(!is_same_file_version(&cached, &current));

        // e_tag presence differs (no versions) -> different file.
        let cached = meta_with_version_etag(None, Some("etag-1"));
        let current = meta_with_version_etag(None, None);
        assert!(!is_same_file_version(&cached, &current));
    }

    /// Test to verify that TTL is preserved when not explicitly set in config.
    /// This fixes issue #19396 where TTL was being unset from DefaultListFilesCache
    /// when CacheManagerConfig::list_files_cache_ttl was not set explicitly.
    #[test]
    fn test_ttl_preserved_when_not_set_in_config() {
        // Create a cache with TTL = 1 second
        let list_file_cache =
            DefaultCache::new_with_ttl(1024, Some(Duration::from_secs(1)));

        // Verify the cache has TTL set initially
        assert_eq!(
            list_file_cache.cache_ttl(),
            Some(Duration::from_secs(1)),
            "Cache should have TTL = 1 second initially"
        );

        // Put cache in config WITHOUT setting list_files_cache_ttl
        let config = CacheManagerConfig::default()
            .with_list_files_cache(Some(Arc::new(list_file_cache)));

        // Create CacheManager from config
        let cache_manager = CacheManager::try_new(&config).unwrap();

        // Verify TTL is preserved (not unset)
        let cache_ttl = cache_manager.get_list_files_cache().unwrap().cache_ttl();

        assert!(
            cache_ttl.is_some(),
            "TTL should be preserved when not set in config. Expected Some(Duration::from_secs(1)), got {cache_ttl:?}"
        );

        // Verify it's the correct TTL value
        assert_eq!(
            cache_ttl,
            Some(Duration::from_secs(1)),
            "TTL should be exactly 1 second"
        );
    }

    /// Test to verify that TTL can still be overridden when explicitly set in config.
    #[test]
    fn test_ttl_overridden_when_set_in_config() {
        // Create a cache with TTL = 1 second
        let list_file_cache =
            DefaultCache::new_with_ttl(1024, Some(Duration::from_secs(1)));

        // Put cache in config WITH a different TTL set
        let config = CacheManagerConfig::default()
            .with_list_files_cache(Some(Arc::new(list_file_cache)))
            .with_list_files_cache_ttl(Some(Duration::from_secs(60)));

        // Create CacheManager from config
        let cache_manager = CacheManager::try_new(&config).unwrap();

        // Verify TTL is overridden to the config value
        let cache_ttl = cache_manager.get_list_files_cache().unwrap().cache_ttl();

        assert_eq!(
            cache_ttl,
            Some(Duration::from_secs(60)),
            "TTL should be overridden to 60 seconds when set in config"
        );
    }
}
