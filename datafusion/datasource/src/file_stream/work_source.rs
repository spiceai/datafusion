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

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use crate::PartitionedFile;
use crate::file_scan_config::FileScanConfig;
use parking_lot::Mutex;

/// Source of work for `ScanState`.
///
/// Streams that may share work across siblings use [`WorkSource::Shared`],
/// while streams that can not share work (e.g. because they must preserve file
/// order) use  [`WorkSource::Local`].
#[derive(Debug, Clone)]
pub(super) enum WorkSource {
    /// Files this stream will plan locally without sharing them.
    Local(VecDeque<PartitionedFile>),
    /// Files shared with sibling streams.
    Shared(SharedWorkSource),
}

impl WorkSource {
    /// Pop the next file to plan from this work source.
    pub(super) fn pop_front(&mut self) -> Option<PartitionedFile> {
        match self {
            Self::Local(files) => files.pop_front(),
            Self::Shared(shared) => shared.pop_front(),
        }
    }

    /// Return how many queued files should be counted as already processed
    /// when this stream stops early after hitting a global limit.
    pub(super) fn skipped_on_limit(&self) -> usize {
        match self {
            Self::Local(files) => files.len(),
            Self::Shared(_) => 0,
        }
    }
}

/// Shared source of work for sibling `FileStream`s.
///
/// The source is created once per execution and shared by all reorderable
/// sibling streams for that execution. It starts empty: each stream registers
/// its own partition's files when it is opened. Whichever stream becomes idle
/// first may then take the next unopened file from any partition that has been
/// registered in this process.
///
/// Registering files lazily keeps distributed execution correct. If a process
/// opens only partition `p`, only that partition's files are available to the
/// shared queue; if it opens all partitions, all files are available for local
/// work stealing.
#[derive(Debug, Clone)]
pub(crate) struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Debug, Default)]
pub(super) struct SharedWorkSourceInner {
    state: Mutex<SharedWorkSourceState>,
}

#[derive(Debug, Default)]
struct SharedWorkSourceState {
    files: VecDeque<PartitionedFile>,
    registered_partitions: HashSet<usize>,
}

impl SharedWorkSource {
    /// Create an empty shared work source.
    pub(crate) fn empty() -> Self {
        Self {
            inner: Arc::new(SharedWorkSourceInner::default()),
        }
    }

    /// Register the files for `partition` in this process-local work source.
    ///
    /// A partition is registered at most once. Files are reordered by the file
    /// source (e.g. by statistics for TopK) together with any already queued
    /// files before becoming available for sibling streams to steal.
    pub(crate) fn register_partition(&self, partition: usize, config: &FileScanConfig) {
        let Some(file_group) = config.file_groups.get(partition) else {
            return;
        };

        let mut state = self.inner.state.lock();
        if !state.registered_partitions.insert(partition) || file_group.is_empty() {
            return;
        }

        state.files.extend(file_group.iter().cloned());
        let files = state.files.drain(..).collect();
        state.files = config.file_source.reorder_files(files).into();
    }

    /// Pop the next file from the shared work queue.
    ///
    /// Returns `None` if the queue is empty.
    fn pop_front(&self) -> Option<PartitionedFile> {
        self.inner.state.lock().files.pop_front()
    }
}
