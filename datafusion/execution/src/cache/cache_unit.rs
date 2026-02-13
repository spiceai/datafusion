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
use std::sync::Arc;

use crate::cache::CacheAccessor;
use crate::cache::cache_manager::{FileStatisticsCache, FileStatisticsCacheEntry};

use datafusion_common::Statistics;

use dashmap::DashMap;
use object_store::ObjectMeta;
use object_store::path::Path;

pub use crate::cache::DefaultFilesMetadataCache;

/// Helper function to normalize an optional string (treats empty strings as None)
pub(super) fn normalize_optional_string(opt: &Option<String>) -> Option<&str> {
    match opt {
        Some(s) if !s.is_empty() => Some(s.as_str()),
        _ => None,
    }
}

/// Helper function to check if two ObjectMeta represent the same file version.
/// Returns true if the files are considered the same version, false otherwise.
///
/// Logic:
/// - If BOTH version and e_tag are absent (None or empty) -> same file (no versioning available)
/// - If version is present in BOTH and matches -> same file
/// - If e_tag is present in BOTH and matches -> same file
/// - If version is present in one but not the other -> different file
/// - If e_tag is present in one but not the other -> different file
/// - Otherwise -> different file
pub(super) fn is_same_file_version(cached: &ObjectMeta, current: &ObjectMeta) -> bool {
    let cached_version = normalize_optional_string(&cached.version);
    let current_version = normalize_optional_string(&current.version);
    let cached_etag = normalize_optional_string(&cached.e_tag);
    let current_etag = normalize_optional_string(&current.e_tag);

    // Both version and etag are absent in both - no versioning info available, consider same
    if cached_version.is_none() && current_version.is_none()
        && cached_etag.is_none() && current_etag.is_none() {
        return true;
    }

    // Check if version or etag presence differs (one has it, other doesn't) - different files
    if (cached_version.is_some() != current_version.is_some()) || (cached_etag.is_some() != current_etag.is_some()) {
        return false;
    }

    // If version is present in BOTH and matches, it's the authoritative check
    if let (Some(cv), Some(curv)) = (cached_version, current_version) {
        return cv == curv
    }

    // If etag is present in BOTH and matches, files are the same
    if let (Some(ce), Some(cure)) = (cached_etag, current_etag) {
        if ce == cure {
            return true;
        }
    }

    // Otherwise, files are different
    false
}

/// Default implementation of [`FileStatisticsCache`]
///
/// Stores collected statistics for files
///
/// Cache is invalided when file size or last modification has changed
///
/// [`FileStatisticsCache`]: crate::cache::cache_manager::FileStatisticsCache
#[derive(Default)]
pub struct DefaultFileStatisticsCache {
    statistics: DashMap<Path, (ObjectMeta, Arc<Statistics>)>,
}

impl FileStatisticsCache for DefaultFileStatisticsCache {
    fn list_entries(&self) -> HashMap<Path, FileStatisticsCacheEntry> {
        let mut entries = HashMap::<Path, FileStatisticsCacheEntry>::new();

        for entry in &self.statistics {
            let path = entry.key();
            let (object_meta, stats) = entry.value();
            entries.insert(
                path.clone(),
                FileStatisticsCacheEntry {
                    object_meta: object_meta.clone(),
                    num_rows: stats.num_rows,
                    num_columns: stats.column_statistics.len(),
                    table_size_bytes: stats.total_byte_size,
                    statistics_size_bytes: 0, // TODO: set to the real size in the future
                },
            );
        }

        entries
    }
}

impl CacheAccessor<Path, Arc<Statistics>> for DefaultFileStatisticsCache {
    type Extra = ObjectMeta;

    /// Get `Statistics` for file location.
    fn get(&self, k: &Path) -> Option<Arc<Statistics>> {
        self.statistics
            .get(k)
            .map(|s| Some(Arc::clone(&s.value().1)))
            .unwrap_or(None)
    }

    /// Get `Statistics` for file location. Returns None if file has changed or not found.
    fn get_with_extra(&self, k: &Path, e: &Self::Extra) -> Option<Arc<Statistics>> {
        self.statistics
            .get(k)
            .map(|s| {
                let (saved_meta, statistics) = s.value();
                if saved_meta.size != e.size
                    || saved_meta.last_modified != e.last_modified
                    || !is_same_file_version(saved_meta, e)
                {
                    // file has changed
                    None
                } else {
                    Some(Arc::clone(statistics))
                }
            })
            .unwrap_or(None)
    }

    /// Save collected file statistics
    fn put(&self, _key: &Path, _value: Arc<Statistics>) -> Option<Arc<Statistics>> {
        panic!("Put cache in DefaultFileStatisticsCache without Extra not supported.")
    }

    fn put_with_extra(
        &self,
        key: &Path,
        value: Arc<Statistics>,
        e: &Self::Extra,
    ) -> Option<Arc<Statistics>> {
        self.statistics
            .insert(key.clone(), (e.clone(), value))
            .map(|x| x.1)
    }

    fn remove(&self, k: &Path) -> Option<Arc<Statistics>> {
        self.statistics.remove(k).map(|x| x.1.1)
    }

    fn contains_key(&self, k: &Path) -> bool {
        self.statistics.contains_key(k)
    }

    fn len(&self) -> usize {
        self.statistics.len()
    }

    fn clear(&self) {
        self.statistics.clear()
    }
    fn name(&self) -> String {
        "DefaultFileStatisticsCache".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CacheAccessor;
    use crate::cache::cache_manager::{FileStatisticsCache, FileStatisticsCacheEntry};
    use crate::cache::cache_unit::DefaultFileStatisticsCache;
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use chrono::DateTime;
    use datafusion_common::Statistics;
    use datafusion_common::stats::Precision;
    use object_store::ObjectMeta;
    use object_store::path::Path;

    #[test]
    fn test_statistics_cache() {
        let meta = ObjectMeta {
            location: Path::from("test"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: None,
            version: None,
        };
        let cache = DefaultFileStatisticsCache::default();
        assert!(cache.get_with_extra(&meta.location, &meta).is_none());

        cache.put_with_extra(
            &meta.location,
            Statistics::new_unknown(&Schema::new(vec![Field::new(
                "test_column",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            )]))
            .into(),
            &meta,
        );
        assert!(cache.get_with_extra(&meta.location, &meta).is_some());

        // file size changed
        let mut meta2 = meta.clone();
        meta2.size = 2048;
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());

        // file last_modified changed
        let mut meta2 = meta.clone();
        meta2.last_modified = DateTime::parse_from_rfc3339("2022-09-27T22:40:00+02:00")
            .unwrap()
            .into();
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());

        // different file
        let mut meta2 = meta.clone();
        meta2.location = Path::from("test2");
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());

        // test the list_entries method
        let entries = cache.list_entries();
        assert_eq!(
            entries,
            HashMap::from([(
                Path::from("test"),
                FileStatisticsCacheEntry {
                    object_meta: meta.clone(),
                    num_rows: Precision::Absent,
                    num_columns: 1,
                    table_size_bytes: Precision::Absent,
                    statistics_size_bytes: 0,
                }
            )])
        );
    }

    #[test]
    fn test_statistics_cache_version_etag() {
        let meta = ObjectMeta {
            location: Path::from("test"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: Some("etag1".to_string()),
            version: Some("v1".to_string()),
        };
        let cache = DefaultFileStatisticsCache::default();
        assert!(cache.get_with_extra(&meta.location, &meta).is_none());

        cache.put_with_extra(
            &meta.location,
            Statistics::new_unknown(&Schema::new(vec![Field::new(
                "test_column",
                DataType::Timestamp(TimeUnit::Second, None),
                false,
            )]))
            .into(),
            &meta,
        );
        assert!(cache.get_with_extra(&meta.location, &meta).is_some());

        // e_tag changed but version still matches - cache should HIT
        let mut meta2 = meta.clone();
        meta2.e_tag = Some("etag2".to_string());
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_some());

        // version changed - cache should MISS (different file, even if etag matches)
        let mut meta2 = meta.clone();
        meta2.version = Some("v2".to_string());
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());

        // both version and e_tag changed - cache should miss
        let mut meta2 = meta.clone();
        meta2.version = Some("v2".to_string());
        meta2.e_tag = Some("etag2".to_string());
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());

        // e_tag changed from Some("etag1") to None - cache should miss
        let mut meta2 = meta.clone();
        meta2.e_tag = None;
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());

        // version changed from Some("v1") to None - cache should miss
        let mut meta2 = meta.clone();
        meta2.version = None;
        assert!(cache.get_with_extra(&meta2.location, &meta2).is_none());
    }

    #[test]
    fn test_statistics_cache_version_matching_logic() {
        // Test the version/etag matching logic:
        // - Both None/empty -> same file
        // - At least one matches -> same file
        // - Neither matches -> different file
        let cache = DefaultFileStatisticsCache::default();

        let stats = Statistics::new_unknown(&Schema::new(vec![Field::new(
            "test_column",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        )]))
        .into();

        // Case 1: Cache with None, query with None -> HIT (both absent)
        let meta_none = ObjectMeta {
            location: Path::from("test1"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: None,
            version: None,
        };
        cache.put_with_extra(&meta_none.location, Arc::clone(&stats), &meta_none);
        assert!(cache.get_with_extra(&meta_none.location, &meta_none).is_some());

        // Case 2: Cache with None, query with empty string -> HIT (both absent)
        let meta_empty = ObjectMeta {
            location: Path::from("test1"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: Some("".to_string()),
            version: Some("".to_string()),
        };
        assert!(cache.get_with_extra(&meta_empty.location, &meta_empty).is_some());

        // Case 3: Cache with version="v1", query with version="v1" -> HIT (version matches)
        let meta_v1 = ObjectMeta {
            location: Path::from("test2"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: None,
            version: Some("v1".to_string()),
        };
        cache.put_with_extra(&meta_v1.location, Arc::clone(&stats), &meta_v1);
        assert!(cache.get_with_extra(&meta_v1.location, &meta_v1).is_some());

        // Case 4: Cache with etag="etag1", query with etag="etag1" -> HIT (etag matches)
        let meta_etag1 = ObjectMeta {
            location: Path::from("test3"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: Some("etag1".to_string()),
            version: None,
        };
        cache.put_with_extra(&meta_etag1.location, Arc::clone(&stats), &meta_etag1);
        assert!(cache.get_with_extra(&meta_etag1.location, &meta_etag1).is_some());

        // Case 5: Cache with version="v1", query with version="v2" and no etag -> MISS
        let meta_v2 = ObjectMeta {
            location: Path::from("test2"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: None,
            version: Some("v2".to_string()),
        };
        assert!(cache.get_with_extra(&meta_v2.location, &meta_v2).is_none());

        // Case 6: Cache with version="v1", query with None -> MISS (versioning info mismatch)
        let meta_v1_to_none = ObjectMeta {
            location: Path::from("test2"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: None,
            version: None,
        };
        assert!(cache.get_with_extra(&meta_v1_to_none.location, &meta_v1_to_none).is_none());

        // Case 7: Cache with version="v1" + etag="e1", query with version="v1" + etag="e2" -> HIT (version matches)
        let meta_both = ObjectMeta {
            location: Path::from("test4"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: Some("e1".to_string()),
            version: Some("v1".to_string()),
        };
        cache.put_with_extra(&meta_both.location, Arc::clone(&stats), &meta_both);

        let meta_both_diff_etag = ObjectMeta {
            location: Path::from("test4"),
            last_modified: DateTime::parse_from_rfc3339("2022-09-27T22:36:00+02:00")
                .unwrap()
                .into(),
            size: 1024,
            e_tag: Some("e2".to_string()),
            version: Some("v1".to_string()),
        };
        assert!(cache.get_with_extra(&meta_both_diff_etag.location, &meta_both_diff_etag).is_some());
    }
}
