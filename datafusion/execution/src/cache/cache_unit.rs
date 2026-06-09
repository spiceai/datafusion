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

use std::collections::HashMap;

use crate::cache::CacheAccessor;
use crate::cache::cache_manager::{
    CachedFileMetadata, FileStatisticsCache, FileStatisticsCacheEntry,
};

use dashmap::DashMap;
use object_store::ObjectMeta;
use object_store::path::Path;

pub use crate::cache::DefaultFilesMetadataCache;

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

/// Default implementation of [`FileStatisticsCache`]
///
/// Stores cached file metadata (statistics and orderings) for files.
///
/// The typical usage pattern is:
/// 1. Call `get(path)` to check for cached value
/// 2. If `Some(cached)`, validate with `cached.is_valid_for(&current_meta)`
/// 3. If invalid or missing, compute new value and call `put(path, new_value)`
///
/// Uses DashMap for lock-free concurrent access.
///
/// [`FileStatisticsCache`]: crate::cache::cache_manager::FileStatisticsCache
#[derive(Default)]
pub struct DefaultFileStatisticsCache {
    cache: DashMap<Path, CachedFileMetadata>,
}

impl CacheAccessor<Path, CachedFileMetadata> for DefaultFileStatisticsCache {
    fn get(&self, key: &Path) -> Option<CachedFileMetadata> {
        self.cache.get(key).map(|entry| entry.value().clone())
    }

    fn put(&self, key: &Path, value: CachedFileMetadata) -> Option<CachedFileMetadata> {
        self.cache.insert(key.clone(), value)
    }

    fn remove(&self, k: &Path) -> Option<CachedFileMetadata> {
        self.cache.remove(k).map(|(_, entry)| entry)
    }

    fn contains_key(&self, k: &Path) -> bool {
        self.cache.contains_key(k)
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn clear(&self) {
        self.cache.clear();
    }

    fn name(&self) -> String {
        "DefaultFileStatisticsCache".to_string()
    }
}

impl FileStatisticsCache for DefaultFileStatisticsCache {
    fn list_entries(&self) -> HashMap<Path, FileStatisticsCacheEntry> {
        let mut entries = HashMap::<Path, FileStatisticsCacheEntry>::new();

        for entry in self.cache.iter() {
            let path = entry.key();
            let cached = entry.value();
            entries.insert(
                path.clone(),
                FileStatisticsCacheEntry {
                    object_meta: cached.meta.clone(),
                    num_rows: cached.statistics.num_rows,
                    num_columns: cached.statistics.column_statistics.len(),
                    table_size_bytes: cached.statistics.total_byte_size,
                    statistics_size_bytes: 0, // TODO: set to the real size in the future
                    has_ordering: cached.ordering.is_some(),
                },
            );
        }

        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheAccessor;
    use crate::cache::cache_manager::{
        CachedFileMetadata, FileStatisticsCache, FileStatisticsCacheEntry,
    };
    use arrow::array::RecordBatch;
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use chrono::DateTime;
    use datafusion_common::Statistics;
    use datafusion_common::stats::Precision;
    use datafusion_expr::ColumnarValue;
    use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
    use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
    use object_store::ObjectMeta;
    use object_store::path::Path;
    use std::sync::Arc;

    fn create_test_meta(path: &str, size: u64) -> ObjectMeta {
        ObjectMeta {
            location: Path::from(path),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size,
            e_tag: None,
            version: None,
        }
    }

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

    #[test]
    fn test_statistics_cache() {
        let meta = create_test_meta("test", 1024);
        let cache = DefaultFileStatisticsCache::default();

        let schema = Schema::new(vec![Field::new(
            "test_column",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        )]);

        // Cache miss
        assert!(cache.get(&meta.location).is_none());

        // Put a value
        let cached_value = CachedFileMetadata::new(
            meta.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            None,
        );
        cache.put(&meta.location, cached_value);

        // Cache hit
        let result = cache.get(&meta.location);
        assert!(result.is_some());
        let cached = result.unwrap();
        assert!(cached.is_valid_for(&meta));

        // File size changed - validation should fail
        let meta2 = create_test_meta("test", 2048);
        let cached = cache.get(&meta2.location).unwrap();
        assert!(!cached.is_valid_for(&meta2));

        // Update with new value
        let cached_value2 = CachedFileMetadata::new(
            meta2.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            None,
        );
        cache.put(&meta2.location, cached_value2);

        // Test list_entries
        let entries = cache.list_entries();
        assert_eq!(entries.len(), 1);
        let entry = entries.get(&Path::from("test")).unwrap();
        assert_eq!(entry.object_meta.size, 2048); // Should be updated value
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct MockExpr {}

    impl std::fmt::Display for MockExpr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockExpr")
        }
    }

    impl PhysicalExpr for MockExpr {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn data_type(
            &self,
            _input_schema: &Schema,
        ) -> datafusion_common::Result<DataType> {
            Ok(DataType::Int32)
        }

        fn nullable(&self, _input_schema: &Schema) -> datafusion_common::Result<bool> {
            Ok(false)
        }

        fn evaluate(
            &self,
            _batch: &RecordBatch,
        ) -> datafusion_common::Result<ColumnarValue> {
            unimplemented!()
        }

        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> datafusion_common::Result<Arc<dyn PhysicalExpr>> {
            assert!(children.is_empty());
            Ok(self)
        }

        fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "MockExpr")
        }
    }

    fn ordering() -> LexOrdering {
        let expr = Arc::new(MockExpr {}) as Arc<dyn PhysicalExpr>;
        LexOrdering::new(vec![PhysicalSortExpr::new_default(expr)]).unwrap()
    }

    #[test]
    fn test_ordering_cache() {
        let meta = create_test_meta("test.parquet", 100);
        let cache = DefaultFileStatisticsCache::default();

        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        // Cache statistics with no ordering
        let cached_value = CachedFileMetadata::new(
            meta.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            None, // No ordering yet
        );
        cache.put(&meta.location, cached_value);

        let result = cache.get(&meta.location).unwrap();
        assert!(result.ordering.is_none());

        // Update to add ordering
        let mut cached = cache.get(&meta.location).unwrap();
        if cached.is_valid_for(&meta) && cached.ordering.is_none() {
            cached.ordering = Some(ordering());
        }
        cache.put(&meta.location, cached);

        let result2 = cache.get(&meta.location).unwrap();
        assert!(result2.ordering.is_some());

        // Verify list_entries shows has_ordering = true
        let entries = cache.list_entries();
        assert_eq!(entries.len(), 1);
        assert!(entries.get(&meta.location).unwrap().has_ordering);
    }

    #[test]
    fn test_cache_invalidation_on_file_modification() {
        let cache = DefaultFileStatisticsCache::default();
        let path = Path::from("test.parquet");
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let meta_v1 = create_test_meta("test.parquet", 100);

        // Cache initial value
        let cached_value = CachedFileMetadata::new(
            meta_v1.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            None,
        );
        cache.put(&path, cached_value);

        // File modified (size changed)
        let meta_v2 = create_test_meta("test.parquet", 200);

        let cached = cache.get(&path).unwrap();
        // Should not be valid for new meta
        assert!(!cached.is_valid_for(&meta_v2));

        // Compute new value and update
        let new_cached = CachedFileMetadata::new(
            meta_v2.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            None,
        );
        cache.put(&path, new_cached);

        // Should have new metadata
        let result = cache.get(&path).unwrap();
        assert_eq!(result.meta.size, 200);
    }

    #[test]
    fn test_ordering_cache_invalidation_on_file_modification() {
        let cache = DefaultFileStatisticsCache::default();
        let path = Path::from("test.parquet");
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        // Cache with original metadata and ordering
        let meta_v1 = ObjectMeta {
            location: path.clone(),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 100,
            e_tag: None,
            version: None,
        };
        let ordering_v1 = ordering();
        let cached_v1 = CachedFileMetadata::new(
            meta_v1.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            Some(ordering_v1),
        );
        cache.put(&path, cached_v1);

        // Verify cached ordering is valid
        let cached = cache.get(&path).unwrap();
        assert!(cached.is_valid_for(&meta_v1));
        assert!(cached.ordering.is_some());

        // File modified (size changed)
        let meta_v2 = ObjectMeta {
            location: path.clone(),
            last_modified: DateTime::parse_from_rfc3339("2022-09-28T10:00:00+02:00")
                .unwrap()
                .into(),
            size: 200, // Changed
            e_tag: None,
            version: None,
        };

        // Cache entry exists but should be invalid for new metadata
        let cached = cache.get(&path).unwrap();
        assert!(!cached.is_valid_for(&meta_v2));

        // Cache new version with different ordering
        let ordering_v2 = ordering(); // New ordering instance
        let cached_v2 = CachedFileMetadata::new(
            meta_v2.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            Some(ordering_v2),
        );
        cache.put(&path, cached_v2);

        // Old metadata should be invalid
        let cached = cache.get(&path).unwrap();
        assert!(!cached.is_valid_for(&meta_v1));

        // New metadata should be valid
        assert!(cached.is_valid_for(&meta_v2));
        assert!(cached.ordering.is_some());
    }

    #[test]
    fn test_list_entries() {
        let cache = DefaultFileStatisticsCache::default();
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let meta1 = create_test_meta("test1.parquet", 100);

        let cached_value = CachedFileMetadata::new(
            meta1.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            None,
        );
        cache.put(&meta1.location, cached_value);
        let meta2 = create_test_meta("test2.parquet", 200);
        let cached_value = CachedFileMetadata::new(
            meta2.clone(),
            Arc::new(Statistics::new_unknown(&schema)),
            Some(ordering()),
        );
        cache.put(&meta2.location, cached_value);

        let entries = cache.list_entries();
        assert_eq!(
            entries,
            HashMap::from([
                (
                    Path::from("test1.parquet"),
                    FileStatisticsCacheEntry {
                        object_meta: meta1,
                        num_rows: Precision::Absent,
                        num_columns: 1,
                        table_size_bytes: Precision::Absent,
                        statistics_size_bytes: 0,
                        has_ordering: false,
                    }
                ),
                (
                    Path::from("test2.parquet"),
                    FileStatisticsCacheEntry {
                        object_meta: meta2,
                        num_rows: Precision::Absent,
                        num_columns: 1,
                        table_size_bytes: Precision::Absent,
                        statistics_size_bytes: 0,
                        has_ordering: true,
                    }
                ),
            ])
        );
    }
}
