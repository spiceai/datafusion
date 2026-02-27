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

//! [`FileScanConfig`] to configure scanning of possibly partitioned
//! file sources.

use crate::file_groups::FileGroup;
use crate::metadata::MetadataColumn;
#[allow(unused_imports)]
use crate::schema_adapter::SchemaAdapterFactory;
use crate::{
    PartitionedFile, display::FileGroupsDisplay, file::FileSource,
    file_compression_type::FileCompressionType, file_stream::FileStream,
    source::DataSource, statistics::MinMaxStatistics,
};
use arrow::array::{ArrayData, ArrayRef, BufferBuilder, DictionaryArray};
use arrow::buffer::Buffer;
use arrow::datatypes::{
    ArrowNativeType, DataType, FieldRef, Schema, SchemaRef, UInt16Type,
};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use datafusion_common::config::ConfigOptions;
use datafusion_common::{
    Constraints, Result, ScalarValue, Statistics, exec_datafusion_err, exec_err,
    internal_datafusion_err, internal_err,
};
use datafusion_execution::{
    SendableRecordBatchStream, TaskContext, object_store::ObjectStoreUrl,
};
use datafusion_expr::Operator;
use datafusion_physical_expr::equivalence::project_orderings;
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::{collect_columns, reassign_expr_columns};
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, split_conjunction};
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::SortOrderPushdownResult;
use datafusion_physical_plan::coop::cooperative;
use datafusion_physical_plan::execution_plan::SchedulingType;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType,
    display::{ProjectSchemaDisplay, display_orderings},
    filter_pushdown::{FilterPushdownPropagation, PushedDown},
    metrics::ExecutionPlanMetricsSet,
};
use log::{debug, warn};
use object_store::ObjectMeta;
#[cfg(feature = "parquet")]
use parquet::arrow::async_reader::ObjectVersionType;
use std::{
    any::Any, borrow::Cow, collections::HashMap, fmt::Debug, fmt::Formatter,
    fmt::Result as FmtResult, marker::PhantomData, sync::Arc,
};

/// The base configurations for a [`DataSourceExec`], the a physical plan for
/// any given file format.
///
/// Use [`DataSourceExec::from_data_source`] to create a [`DataSourceExec`] from a ``FileScanConfig`.
///
/// # Example
/// ```
/// # use std::any::Any;
/// # use std::sync::Arc;
/// # use arrow::datatypes::{Field, Fields, DataType, Schema, SchemaRef};
/// # use object_store::ObjectStore;
/// # use datafusion_common::Result;
/// # use datafusion_datasource::file::FileSource;
/// # use datafusion_datasource::file_groups::FileGroup;
/// # use datafusion_datasource::PartitionedFile;
/// # use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
/// # use datafusion_datasource::file_stream::FileOpener;
/// # use datafusion_datasource::source::DataSourceExec;
/// # use datafusion_datasource::table_schema::TableSchema;
/// # use datafusion_execution::object_store::ObjectStoreUrl;
/// # use datafusion_physical_expr::projection::ProjectionExprs;
/// # use datafusion_physical_plan::ExecutionPlan;
/// # use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
/// # let file_schema = Arc::new(Schema::new(vec![
/// #  Field::new("c1", DataType::Int32, false),
/// #  Field::new("c2", DataType::Int32, false),
/// #  Field::new("c3", DataType::Int32, false),
/// #  Field::new("c4", DataType::Int32, false),
/// # ]));
/// # // Note: crate mock ParquetSource, as ParquetSource is not in the datasource crate
/// #[derive(Clone)]
/// # struct ParquetSource {
/// #    table_schema: TableSchema,
/// # };
/// # impl FileSource for ParquetSource {
/// #  fn create_file_opener(&self, _: Arc<dyn ObjectStore>, _: &FileScanConfig, _: usize) -> Result<Arc<dyn FileOpener>> { unimplemented!() }
/// #  fn as_any(&self) -> &dyn Any { self  }
/// #  fn table_schema(&self) -> &TableSchema { &self.table_schema }
/// #  fn with_batch_size(&self, _: usize) -> Arc<dyn FileSource> { unimplemented!() }
/// #  fn metrics(&self) -> &ExecutionPlanMetricsSet { unimplemented!() }
/// #  fn file_type(&self) -> &str { "parquet" }
/// #  // Note that this implementation drops the projection on the floor, it is not complete!
/// #  fn try_pushdown_projection(&self, projection: &ProjectionExprs) -> Result<Option<Arc<dyn FileSource>>> { Ok(Some(Arc::new(self.clone()) as Arc<dyn FileSource>)) }
/// #  }
/// # impl ParquetSource {
/// #  fn new(table_schema: impl Into<TableSchema>) -> Self { Self {table_schema: table_schema.into()} }
/// # }
/// // create FileScan config for reading parquet files from file://
/// let object_store_url = ObjectStoreUrl::local_filesystem();
/// let file_source = Arc::new(ParquetSource::new(file_schema.clone()));
/// let config = FileScanConfigBuilder::new(object_store_url, file_source)
///   .with_limit(Some(1000))            // read only the first 1000 records
///   .with_projection_indices(Some(vec![2, 3])) // project columns 2 and 3
///   .expect("Failed to push down projection")
///    // Read /tmp/file1.parquet with known size of 1234 bytes in a single group
///   .with_file(PartitionedFile::new("file1.parquet", 1234))
///   // Read /tmp/file2.parquet 56 bytes and /tmp/file3.parquet 78 bytes
///   // in a  single row group
///   .with_file_group(FileGroup::new(vec![
///    PartitionedFile::new("file2.parquet", 56),
///    PartitionedFile::new("file3.parquet", 78),
///   ])).build();
/// // create an execution plan from the config
/// let plan: Arc<dyn ExecutionPlan> = DataSourceExec::from_data_source(config);
/// ```
///
/// [`DataSourceExec`]: crate::source::DataSourceExec
/// [`DataSourceExec::from_data_source`]: crate::source::DataSourceExec::from_data_source
#[derive(Clone)]
pub struct FileScanConfig {
    /// Object store URL, used to get an [`ObjectStore`] instance from
    /// [`RuntimeEnv::object_store`]
    ///
    /// This `ObjectStoreUrl` should be the prefix of the absolute url for files
    /// as `file://` or `s3://my_bucket`. It should not include the path to the
    /// file itself. The relevant URL prefix must be registered via
    /// [`RuntimeEnv::register_object_store`]
    ///
    /// [`ObjectStore`]: object_store::ObjectStore
    /// [`RuntimeEnv::register_object_store`]: datafusion_execution::runtime_env::RuntimeEnv::register_object_store
    /// [`RuntimeEnv::object_store`]: datafusion_execution::runtime_env::RuntimeEnv::object_store
    pub object_store_url: ObjectStoreUrl,
    /// List of files to be processed, grouped into partitions
    ///
    /// Each file must have a schema of `file_schema` or a subset. If
    /// a particular file has a subset, the missing columns are
    /// padded with NULLs.
    ///
    /// DataFusion may attempt to read each partition of files
    /// concurrently, however files *within* a partition will be read
    /// sequentially, one after the next.
    pub file_groups: Vec<FileGroup>,
    /// Table constraints
    pub constraints: Constraints,
    /// The maximum number of records to read from this plan. If `None`,
    /// all records after filtering are returned.
    pub limit: Option<usize>,
    /// All equivalent lexicographical orderings that describe the schema.
    pub output_ordering: Vec<LexOrdering>,
    /// File compression type
    pub file_compression_type: FileCompressionType,
    /// File source such as `ParquetSource`, `CsvSource`, `JsonSource`, etc.
    pub file_source: Arc<dyn FileSource>,
    /// Batch size while creating new batches
    /// Defaults to [`datafusion_common::config::ExecutionOptions`] batch_size.
    pub batch_size: Option<usize>,
    /// Expression adapter used to adapt filters and projections that are pushed down into the scan
    /// from the logical schema to the physical schema of the file.
    pub expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    /// Unprojected statistics for the table (file schema + partition columns).
    /// These are projected on-demand via `projected_stats()`.
    ///
    /// Note that this field is pub(crate) because accessing it directly from outside
    /// would be incorrect if there are filters being applied, thus this should be accessed
    /// via [`FileScanConfig::statistics`].
    pub(crate) statistics: Statistics,
    /// When true, file_groups are organized by partition column values
    /// and output_partitioning will return Hash partitioning on partition columns.
    /// This allows the optimizer to skip hash repartitioning for aggregates and joins
    /// on partition columns.
    ///
    /// If the number of file partitions > target_partitions, the file partitions will be grouped
    /// in a round-robin fashion such that number of file partitions = target_partitions.
    pub partitioned_by_file_group: bool,
    /// Metadata columns to include in the output schema.
    /// These columns provide file metadata like location, size, and last_modified.
    pub metadata_cols: Vec<MetadataColumn>,
    /// Tracks (output_position, metadata_col_idx) for metadata columns in projection
    pub projected_metadata_positions: Vec<(usize, usize)>,
    /// Object versioning type for reading files.
    /// This is used to handle different versions of objects in object stores.
    #[cfg(feature = "parquet")]
    pub object_versioning_type: Option<ObjectVersionType>,
}

/// A builder for [`FileScanConfig`]'s.
///
/// Example:
///
/// ```rust
/// # use std::sync::Arc;
/// # use arrow::datatypes::{DataType, Field, Schema};
/// # use datafusion_datasource::file_scan_config::{FileScanConfigBuilder, FileScanConfig};
/// # use datafusion_datasource::file_compression_type::FileCompressionType;
/// # use datafusion_datasource::file_groups::FileGroup;
/// # use datafusion_datasource::PartitionedFile;
/// # use datafusion_datasource::table_schema::TableSchema;
/// # use datafusion_execution::object_store::ObjectStoreUrl;
/// # use datafusion_common::Statistics;
/// # use datafusion_datasource::file::FileSource;
///
/// # fn main() {
/// # fn with_source(file_source: Arc<dyn FileSource>) {
///     // Create a schema for our Parquet files
///     let file_schema = Arc::new(Schema::new(vec![
///         Field::new("id", DataType::Int32, false),
///         Field::new("value", DataType::Utf8, false),
///     ]));
///
///     // Create partition columns
///     let partition_cols = vec![
///         Arc::new(Field::new("date", DataType::Utf8, false)),
///     ];
///
///     // Create table schema with file schema and partition columns
///     let table_schema = TableSchema::new(file_schema, partition_cols);
///
///     // Create a builder for scanning Parquet files from a local filesystem
///     let config = FileScanConfigBuilder::new(
///         ObjectStoreUrl::local_filesystem(),
///         file_source,
///     )
///     // Set a limit of 1000 rows
///     .with_limit(Some(1000))
///     // Project only the first column
///     .with_projection_indices(Some(vec![0]))
///     .expect("Failed to push down projection")
///     // Add a file group with two files
///     .with_file_group(FileGroup::new(vec![
///         PartitionedFile::new("data/date=2024-01-01/file1.parquet", 1024),
///         PartitionedFile::new("data/date=2024-01-01/file2.parquet", 2048),
///     ]))
///     // Set compression type
///     .with_file_compression_type(FileCompressionType::UNCOMPRESSED)
///     // Build the final config
///     .build();
/// # }
/// # }
/// ```
#[derive(Clone)]
pub struct FileScanConfigBuilder {
    object_store_url: ObjectStoreUrl,
    file_source: Arc<dyn FileSource>,
    limit: Option<usize>,
    constraints: Option<Constraints>,
    file_groups: Vec<FileGroup>,
    statistics: Option<Statistics>,
    output_ordering: Vec<LexOrdering>,
    file_compression_type: Option<FileCompressionType>,
    batch_size: Option<usize>,
    expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    partitioned_by_file_group: bool,
    metadata_cols: Vec<MetadataColumn>,
    projected_metadata_positions: Vec<(usize, usize)>,
    #[cfg(feature = "parquet")]
    object_versioning_type: Option<ObjectVersionType>,
}

impl FileScanConfigBuilder {
    /// Create a new [`FileScanConfigBuilder`] with default settings for scanning files.
    ///
    /// # Parameters:
    /// * `object_store_url`: See [`FileScanConfig::object_store_url`]
    /// * `file_source`: See [`FileScanConfig::file_source`]. The file source must have
    ///   a schema set via its constructor.
    pub fn new(
        object_store_url: ObjectStoreUrl,
        file_source: Arc<dyn FileSource>,
    ) -> Self {
        Self {
            object_store_url,
            file_source,
            file_groups: vec![],
            statistics: None,
            output_ordering: vec![],
            file_compression_type: None,
            limit: None,
            constraints: None,
            batch_size: None,
            expr_adapter_factory: None,
            partitioned_by_file_group: false,
            metadata_cols: vec![],
            projected_metadata_positions: vec![],
            #[cfg(feature = "parquet")]
            object_versioning_type: None,
        }
    }

    /// Set the maximum number of records to read from this plan. If `None`,
    /// all records after filtering are returned.
    pub fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Set the file source for scanning files.
    ///
    /// This method allows you to change the file source implementation (e.g. ParquetSource, CsvSource, etc.)
    /// after the builder has been created.
    pub fn with_source(mut self, file_source: Arc<dyn FileSource>) -> Self {
        self.file_source = file_source;
        self
    }

    pub fn table_schema(&self) -> &SchemaRef {
        self.file_source.table_schema().table_schema()
    }

    /// Set the columns on which to project the data. Indexes that are higher than the
    /// number of columns of `file_schema` refer to `table_partition_cols`.
    ///
    /// # Deprecated
    /// Use [`Self::with_projection_indices`] instead. This method will be removed in a future release.
    #[deprecated(since = "51.0.0", note = "Use with_projection_indices instead")]
    pub fn with_projection(self, indices: Option<Vec<usize>>) -> Self {
        match self.clone().with_projection_indices(indices) {
            Ok(builder) => builder,
            Err(e) => {
                warn!(
                    "Failed to push down projection in FileScanConfigBuilder::with_projection: {e}"
                );
                self
            }
        }
    }

    /// Set the columns on which to project the data using column indices.
    ///
    /// Indexes that are higher than the number of columns of `file_schema` refer to `table_partition_cols`.
    /// Indexes beyond file + partition columns refer to metadata columns and are tracked
    /// separately in `projected_metadata_positions`.
    pub fn with_projection_indices(
        mut self,
        indices: Option<Vec<usize>>,
    ) -> Result<Self> {
        let Some(indices) = indices else {
            return Ok(self);
        };

        let table_schema = self.file_source.table_schema().table_schema();
        let table_col_count = table_schema.fields().len();

        // Separate metadata column indices from file+partition indices.
        // Metadata column indices are >= table_col_count (file + partition).
        let mut file_partition_indices = Vec::new();
        let mut metadata_positions = Vec::new();
        for (output_pos, idx) in indices.iter().enumerate() {
            if *idx >= table_col_count {
                metadata_positions.push((output_pos, idx - table_col_count));
            } else {
                file_partition_indices.push(*idx);
            }
        }

        self.projected_metadata_positions = metadata_positions;

        // Only skip projection pushdown when all projected columns are metadata.
        // When both are empty (e.g. INSERT INTO passes Some(vec![])), we must
        // still push down the empty projection so the schema reflects zero fields.
        if file_partition_indices.is_empty()
            && !self.projected_metadata_positions.is_empty()
        {
            return Ok(self);
        }

        let projection_exprs =
            ProjectionExprs::from_indices(&file_partition_indices, table_schema);
        let new_source = self
            .file_source
            .try_pushdown_projection(&projection_exprs)
            .map_err(|e| {
                internal_datafusion_err!(
                    "Failed to push down projection in FileScanConfigBuilder::build: {e}"
                )
            })?;
        if let Some(new_source) = new_source {
            self.file_source = new_source;
        } else {
            internal_err!(
                "FileSource {} does not support projection pushdown",
                self.file_source.file_type()
            )?;
        }
        Ok(self)
    }

    /// Set the table constraints
    pub fn with_constraints(mut self, constraints: Constraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Set the estimated overall statistics of the files, taking `filters` into account.
    /// Defaults to [`Statistics::new_unknown`].
    pub fn with_statistics(mut self, statistics: Statistics) -> Self {
        self.statistics = Some(statistics);
        self
    }

    /// Set the list of files to be processed, grouped into partitions.
    ///
    /// Each file must have a schema of `file_schema` or a subset. If
    /// a particular file has a subset, the missing columns are
    /// padded with NULLs.
    ///
    /// DataFusion may attempt to read each partition of files
    /// concurrently, however files *within* a partition will be read
    /// sequentially, one after the next.
    pub fn with_file_groups(mut self, file_groups: Vec<FileGroup>) -> Self {
        self.file_groups = file_groups;
        self
    }

    /// Add a new file group
    ///
    /// See [`Self::with_file_groups`] for more information
    pub fn with_file_group(mut self, file_group: FileGroup) -> Self {
        self.file_groups.push(file_group);
        self
    }

    /// Add a file as a single group
    ///
    /// See [`Self::with_file_groups`] for more information.
    pub fn with_file(self, partitioned_file: PartitionedFile) -> Self {
        self.with_file_group(FileGroup::new(vec![partitioned_file]))
    }

    /// Set the output ordering of the files
    pub fn with_output_ordering(mut self, output_ordering: Vec<LexOrdering>) -> Self {
        self.output_ordering = output_ordering;
        self
    }

    /// Set the file compression type
    pub fn with_file_compression_type(
        mut self,
        file_compression_type: FileCompressionType,
    ) -> Self {
        self.file_compression_type = Some(file_compression_type);
        self
    }

    /// Set the batch_size property
    pub fn with_batch_size(mut self, batch_size: Option<usize>) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Register an expression adapter used to adapt filters and projections that are pushed down into the scan
    /// from the logical schema to the physical schema of the file.
    /// This can include things like:
    /// - Column ordering changes
    /// - Handling of missing columns
    /// - Rewriting expression to use pre-computed values or file format specific optimizations
    pub fn with_expr_adapter(
        mut self,
        expr_adapter: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    ) -> Self {
        self.expr_adapter_factory = expr_adapter;
        self
    }

    /// Set whether file groups are organized by partition column values.
    ///
    /// When set to true, the output partitioning will be declared as Hash partitioning
    /// on the partition columns.
    pub fn with_partitioned_by_file_group(
        mut self,
        partitioned_by_file_group: bool,
    ) -> Self {
        self.partitioned_by_file_group = partitioned_by_file_group;
        self
    }

    /// Set the metadata columns to include in the output schema.
    /// These columns provide file metadata like location, size, and last_modified.
    pub fn with_metadata_cols(mut self, metadata_cols: Vec<MetadataColumn>) -> Self {
        self.metadata_cols = metadata_cols;
        self
    }

    /// Set the object versioning type for reading files.
    /// This is used to handle different versions of objects in object stores.
    #[cfg(feature = "parquet")]
    pub fn with_object_versioning_type(
        mut self,
        object_versioning_type: Option<ObjectVersionType>,
    ) -> Self {
        self.object_versioning_type = object_versioning_type;
        self
    }

    /// Build the final [`FileScanConfig`] with all the configured settings.
    ///
    /// This method takes ownership of the builder and returns the constructed `FileScanConfig`.
    /// Any unset optional fields will use their default values.
    ///
    /// # Errors
    /// Returns an error if projection pushdown fails or if schema operations fail.
    pub fn build(self) -> FileScanConfig {
        let Self {
            object_store_url,
            file_source,
            limit,
            constraints,
            file_groups,
            statistics,
            output_ordering,
            file_compression_type,
            batch_size,
            expr_adapter_factory,
            partitioned_by_file_group,
            metadata_cols,
            mut projected_metadata_positions,
            #[cfg(feature = "parquet")]
            object_versioning_type,
        } = self;

        // If no explicit metadata positions were set (e.g. projection_indices was None
        // or didn't include metadata columns), default to appending all metadata
        // columns after the table (file + partition) columns. This mirrors DF51's
        // build() behavior for the None-projection case.
        if projected_metadata_positions.is_empty() && !metadata_cols.is_empty() {
            let table_col_count =
                file_source.table_schema().table_schema().fields().len();
            projected_metadata_positions = (0..metadata_cols.len())
                .map(|i| (table_col_count + i, i))
                .collect();
        }

        let constraints = constraints.unwrap_or_default();
        let statistics = statistics.unwrap_or_else(|| {
            Statistics::new_unknown(file_source.table_schema().table_schema())
        });
        let file_compression_type =
            file_compression_type.unwrap_or(FileCompressionType::UNCOMPRESSED);

        FileScanConfig {
            object_store_url,
            file_source,
            limit,
            constraints,
            file_groups,
            output_ordering,
            file_compression_type,
            batch_size,
            expr_adapter_factory,
            statistics,
            partitioned_by_file_group,
            metadata_cols,
            projected_metadata_positions,
            #[cfg(feature = "parquet")]
            object_versioning_type,
        }
    }
}

impl From<FileScanConfig> for FileScanConfigBuilder {
    fn from(config: FileScanConfig) -> Self {
        Self {
            object_store_url: config.object_store_url,
            file_source: Arc::<dyn FileSource>::clone(&config.file_source),
            file_groups: config.file_groups,
            statistics: Some(config.statistics),
            output_ordering: config.output_ordering,
            file_compression_type: Some(config.file_compression_type),
            limit: config.limit,
            constraints: Some(config.constraints),
            batch_size: config.batch_size,
            expr_adapter_factory: config.expr_adapter_factory,
            partitioned_by_file_group: config.partitioned_by_file_group,
            metadata_cols: config.metadata_cols,
            projected_metadata_positions: config.projected_metadata_positions,
            #[cfg(feature = "parquet")]
            object_versioning_type: config.object_versioning_type,
        }
    }
}

impl DataSource for FileScanConfig {
    fn open(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let object_store = context.runtime_env().object_store(&self.object_store_url)?;
        let batch_size = self
            .batch_size
            .unwrap_or_else(|| context.session_config().batch_size());

        let source = self.file_source.with_batch_size(batch_size);

        let opener = source.create_file_opener(object_store, self, partition)?;

        let stream = FileStream::new(self, partition, opener, source.metrics())?;
        Ok(Box::pin(cooperative(stream)))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let schema = self.projected_schema().map_err(|_| std::fmt::Error {})?;
                let orderings = get_projected_output_ordering(self, &schema);

                write!(f, "file_groups=")?;
                FileGroupsDisplay(&self.file_groups).fmt_as(t, f)?;

                if !schema.fields().is_empty() {
                    if let Some(projection) = self.file_source.projection() {
                        // This matches what ProjectionExec does.
                        // TODO: can we put this into ProjectionExprs so that it's shared code?
                        let mut expr: Vec<String> = projection
                            .as_ref()
                            .iter()
                            .map(|proj_expr| {
                                if let Some(column) =
                                    proj_expr.expr.as_any().downcast_ref::<Column>()
                                {
                                    if column.name() == proj_expr.alias {
                                        column.name().to_string()
                                    } else {
                                        format!(
                                            "{} as {}",
                                            proj_expr.expr, proj_expr.alias
                                        )
                                    }
                                } else {
                                    format!("{} as {}", proj_expr.expr, proj_expr.alias)
                                }
                            })
                            .collect();

                        // Insert metadata column names at their designated output positions
                        // so the EXPLAIN display reflects the full projected schema.
                        for (output_pos, metadata_idx) in
                            &self.projected_metadata_positions
                        {
                            if *metadata_idx < self.metadata_cols.len() {
                                let name = self.metadata_cols[*metadata_idx]
                                    .field()
                                    .name()
                                    .clone();
                                if *output_pos <= expr.len() {
                                    expr.insert(*output_pos, name);
                                }
                            }
                        }

                        write!(f, ", projection=[{}]", expr.join(", "))?;
                    } else {
                        write!(f, ", projection={}", ProjectSchemaDisplay(&schema))?;
                    }
                }

                if let Some(limit) = self.limit {
                    write!(f, ", limit={limit}")?;
                }

                display_orderings(f, &orderings)?;

                if !self.constraints.is_empty() {
                    write!(f, ", {}", self.constraints)?;
                }

                self.fmt_file_source(t, f)
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "format={}", self.file_source.file_type())?;
                self.file_source.fmt_extra(t, f)?;
                let num_files = self.file_groups.iter().map(|fg| fg.len()).sum::<usize>();
                writeln!(f, "files={num_files}")?;
                Ok(())
            }
        }
    }

    /// If supported by the underlying [`FileSource`], redistribute files across partitions according to their size.
    fn repartitioned(
        &self,
        target_partitions: usize,
        repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        // When files are grouped by partition values, we cannot allow byte-range
        // splitting. It would mix rows from different partition values across
        // file groups, breaking the Hash partitioning.
        if self.partitioned_by_file_group {
            return Ok(None);
        }

        let source = self.file_source.repartitioned(
            target_partitions,
            repartition_file_min_size,
            output_ordering,
            self,
        )?;

        Ok(source.map(|s| Arc::new(s) as _))
    }

    /// Returns the output partitioning for this file scan.
    ///
    /// When `partitioned_by_file_group` is true, this returns `Partitioning::Hash` on
    /// the Hive partition columns, allowing the optimizer to skip hash repartitioning
    /// for aggregates and joins on those columns.
    ///
    /// Tradeoffs
    /// - Benefit: Eliminates `RepartitionExec` and `SortExec` for queries with
    ///   `GROUP BY` or `ORDER BY` on partition columns.
    /// - Cost: Files are grouped by partition values rather than split by byte
    ///   ranges, which may reduce I/O parallelism when partition sizes are uneven.
    ///   For simple aggregations without `ORDER BY`, this cost may outweigh the benefit.
    ///
    /// Follow-up Work
    /// - Idea: Could allow byte-range splitting within partition-aware groups,
    ///   preserving I/O parallelism while maintaining partition semantics.
    fn output_partitioning(&self) -> Partitioning {
        if self.partitioned_by_file_group {
            let partition_cols = self.table_partition_cols();
            if !partition_cols.is_empty() {
                let projected_schema = match self.projected_schema() {
                    Ok(schema) => schema,
                    Err(_) => {
                        debug!(
                            "Could not get projected schema, falling back to UnknownPartitioning."
                        );
                        return Partitioning::UnknownPartitioning(self.file_groups.len());
                    }
                };

                // Build Column expressions for partition columns based on their
                // position in the projected schema
                let mut exprs: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
                for partition_col in partition_cols {
                    if let Some((idx, _)) = projected_schema
                        .fields()
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.name() == partition_col.name())
                    {
                        exprs.push(Arc::new(Column::new(partition_col.name(), idx)));
                    }
                }

                if exprs.len() == partition_cols.len() {
                    return Partitioning::Hash(exprs, self.file_groups.len());
                }
            }
        }
        Partitioning::UnknownPartitioning(self.file_groups.len())
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        let schema = self.file_source.table_schema().table_schema();
        let mut eq_properties = EquivalenceProperties::new_with_orderings(
            Arc::clone(schema),
            self.output_ordering.clone(),
        )
        .with_constraints(self.constraints.clone());

        if let Some(filter) = self.file_source.filter() {
            // We need to remap column indexes to match the projected schema since that's what the equivalence properties deal with.
            // Note that this will *ignore* any non-projected columns: these don't factor into ordering / equivalence.
            match Self::add_filter_equivalence_info(&filter, &mut eq_properties, schema) {
                Ok(()) => {}
                Err(e) => {
                    warn!("Failed to add filter equivalence info: {e}");
                    #[cfg(debug_assertions)]
                    panic!("Failed to add filter equivalence info: {e}");
                }
            }
        }

        if let Some(projection) = self.file_source.projection() {
            match (
                projection.project_schema(schema),
                projection.projection_mapping(schema),
            ) {
                (Ok(output_schema), Ok(mapping)) => {
                    eq_properties =
                        eq_properties.project(&mapping, Arc::new(output_schema));
                }
                (Err(e), _) | (_, Err(e)) => {
                    warn!("Failed to project equivalence properties: {e}");
                    #[cfg(debug_assertions)]
                    panic!("Failed to project equivalence properties: {e}");
                }
            }
        }

        // Append metadata column fields to the schema. These columns are
        // injected by ExtendedColumnProjector in FileStream but are not part
        // of the table schema. The schema advertised by eq_properties must
        // include them so that DataSourceExec::schema() matches the actual
        // batches produced.
        if !self.metadata_cols.is_empty() {
            eq_properties = eq_properties.with_extra_fields(
                self.metadata_cols.iter().map(|c| Arc::new(c.field())),
            );
        }

        eq_properties
    }

    fn scheduling_type(&self) -> SchedulingType {
        SchedulingType::Cooperative
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Statistics> {
        if let Some(partition) = partition {
            // Get statistics for a specific partition
            // Note: FileGroup statistics include partition columns (computed from partition_values)
            if let Some(file_group) = self.file_groups.get(partition)
                && let Some(stat) = file_group.file_statistics(None)
            {
                // Project the statistics based on the projection
                let output_schema = self.projected_schema()?;
                return if let Some(projection) = self.file_source.projection() {
                    projection.project_statistics(stat.clone(), &output_schema)
                } else {
                    Ok(stat.clone())
                };
            }
            // If no statistics available for this partition, return unknown
            Ok(Statistics::new_unknown(self.projected_schema()?.as_ref()))
        } else {
            // Return aggregate statistics across all partitions
            let statistics = self.statistics();
            let projection = self.file_source.projection();
            let output_schema = self.projected_schema()?;
            if let Some(projection) = &projection {
                projection.project_statistics(statistics.clone(), &output_schema)
            } else {
                Ok(statistics)
            }
        }
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let source = FileScanConfigBuilder::from(self.clone())
            .with_limit(limit)
            .build();
        Some(Arc::new(source))
    }

    fn fetch(&self) -> Option<usize> {
        self.limit
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.file_source.metrics().clone()
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> Result<Option<Arc<dyn DataSource>>> {
        match self.file_source.try_pushdown_projection(projection)? {
            Some(new_source) => {
                let mut new_file_scan_config = self.clone();
                new_file_scan_config.file_source = new_source;
                Ok(Some(Arc::new(new_file_scan_config) as Arc<dyn DataSource>))
            }
            None => Ok(None),
        }
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        config: &ConfigOptions,
    ) -> Result<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        // Remap filter Column indices to match the table schema (file + partition columns).
        // This is necessary because filters may have been created against a different schema
        // (e.g., after projection pushdown) and need to be remapped to the table schema
        // before being passed to the file source and ultimately serialized.
        // For example, the filter being pushed down is `c1_c2 > 5` and it was created
        // against the output schema of the this `DataSource` which has projection `c1 + c2 as c1_c2`.
        // Thus we need to rewrite the filter back to `c1 + c2 > 5` before passing it to the file source.
        let table_schema = self.file_source.table_schema().table_schema();
        // If there's a projection with aliases, first map the filters back through
        // the projection expressions before remapping to the table schema.
        //
        // Metadata columns (e.g. `location`) are appended after the projection by
        // ExtendedColumnProjector and are NOT part of the projection expressions.
        // Filters that reference metadata columns cannot be remapped through the
        // projection and cannot be pushed down to the file source. We separate them
        // out and return PushedDown::No for those.
        let (filters_to_remap, metadata_filter_indices) =
            if let Some(projection) = self.file_source.projection() {
                use datafusion_physical_plan::projection::update_expr;
                let proj_len = projection.as_ref().len();
                let mut pushable = Vec::new();
                let mut metadata_indices = Vec::new();

                for (i, filter) in filters.into_iter().enumerate() {
                    let cols = collect_columns(&filter);
                    if cols.iter().any(|c| c.index() >= proj_len) {
                        // Filter references a metadata column — can't push down
                        metadata_indices.push(i);
                    } else {
                        let remapped = update_expr(&filter, projection.as_ref(), true)?
                            .ok_or_else(|| {
                            internal_datafusion_err!(
                                "Failed to map filter expression through projection: {}",
                                filter
                            )
                        })?;
                        pushable.push(remapped);
                    }
                }
                (pushable, metadata_indices)
            } else {
                (filters, vec![])
            };
        // Now remap column indices to match the table schema.
        let remapped_filters: Result<Vec<_>> = filters_to_remap
            .into_iter()
            .map(|filter| reassign_expr_columns(filter, table_schema.as_ref()))
            .collect();
        let remapped_filters = remapped_filters?;

        let result = self
            .file_source
            .try_pushdown_filters(remapped_filters, config)?;

        // Merge results: insert PushedDown::No at the positions of metadata filters.
        let mut all_filters = result.filters;
        for &idx in &metadata_filter_indices {
            all_filters.insert(idx, PushedDown::No);
        }

        match result.updated_node {
            Some(new_file_source) => {
                let mut new_file_scan_config = self.clone();
                new_file_scan_config.file_source = new_file_source;
                Ok(FilterPushdownPropagation {
                    filters: all_filters,
                    updated_node: Some(Arc::new(new_file_scan_config) as _),
                })
            }
            None => {
                // If the file source does not support filter pushdown, return the original config
                Ok(FilterPushdownPropagation {
                    filters: all_filters,
                    updated_node: None,
                })
            }
        }
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> Result<SortOrderPushdownResult<Arc<dyn DataSource>>> {
        // Delegate to FileSource to check if reverse scanning can satisfy the request.
        let pushdown_result = self
            .file_source
            .try_reverse_output(order, &self.eq_properties())?;

        match pushdown_result {
            SortOrderPushdownResult::Exact { inner } => {
                Ok(SortOrderPushdownResult::Exact {
                    inner: self.rebuild_with_source(inner, true)?,
                })
            }
            SortOrderPushdownResult::Inexact { inner } => {
                Ok(SortOrderPushdownResult::Inexact {
                    inner: self.rebuild_with_source(inner, false)?,
                })
            }
            SortOrderPushdownResult::Unsupported => {
                Ok(SortOrderPushdownResult::Unsupported)
            }
        }
    }
}

impl FileScanConfig {
    /// Get the file schema (schema of the files without partition columns)
    pub fn file_schema(&self) -> &SchemaRef {
        self.file_source.table_schema().file_schema()
    }

    /// Get the table partition columns
    pub fn table_partition_cols(&self) -> &Vec<FieldRef> {
        self.file_source.table_schema().table_partition_cols()
    }

    /// Returns the unprojected table statistics, marking them as inexact if filters are present.
    ///
    /// When filters are pushed down (including pruning predicates and bloom filters),
    /// we can't guarantee the statistics are exact because we don't know how many
    /// rows will be filtered out.
    pub fn statistics(&self) -> Statistics {
        if self.file_source.filter().is_some() {
            self.statistics.clone().to_inexact()
        } else {
            self.statistics.clone()
        }
    }

    pub fn projected_schema(&self) -> Result<Arc<Schema>> {
        let schema = self.file_source.table_schema().table_schema();
        let base_schema = match self.file_source.projection() {
            Some(proj) => Arc::new(proj.project_schema(schema)?),
            None => Arc::clone(schema),
        };

        if self.metadata_cols.is_empty() {
            return Ok(base_schema);
        }

        let mut fields: Vec<FieldRef> = base_schema.fields().iter().cloned().collect();

        // Insert metadata columns at their designated output positions
        // (set by with_projection_indices), preserving the user's requested column order.
        for (output_pos, metadata_idx) in &self.projected_metadata_positions {
            if *metadata_idx < self.metadata_cols.len() {
                let field = Arc::new(self.metadata_cols[*metadata_idx].field());
                if *output_pos <= fields.len() {
                    fields.insert(*output_pos, field);
                }
            }
        }

        Ok(Arc::new(Schema::new_with_metadata(
            fields,
            base_schema.metadata().clone(),
        )))
    }

    fn add_filter_equivalence_info(
        filter: &Arc<dyn PhysicalExpr>,
        eq_properties: &mut EquivalenceProperties,
        schema: &Schema,
    ) -> Result<()> {
        // Gather valid equality pairs from the filter expression
        let equal_pairs = split_conjunction(filter).into_iter().filter_map(|expr| {
            // Ignore any binary expressions that reference non-existent columns in the current schema
            // (e.g. due to unnecessary projections being removed)
            reassign_expr_columns(Arc::clone(expr), schema)
                .ok()
                .and_then(|expr| match expr.as_any().downcast_ref::<BinaryExpr>() {
                    Some(expr) if expr.op() == &Operator::Eq => {
                        Some((Arc::clone(expr.left()), Arc::clone(expr.right())))
                    }
                    _ => None,
                })
        });

        for (lhs, rhs) in equal_pairs {
            eq_properties.add_equal_conditions(lhs, rhs)?
        }

        Ok(())
    }

    /// Returns whether newlines in values are supported.
    ///
    /// This method always returns `false`. The actual newlines_in_values setting
    /// has been moved to [`CsvSource`] and should be accessed via
    /// [`CsvSource::csv_options()`] instead.
    ///
    /// [`CsvSource`]: https://docs.rs/datafusion/latest/datafusion/datasource/physical_plan/struct.CsvSource.html
    /// [`CsvSource::csv_options()`]: https://docs.rs/datafusion/latest/datafusion/datasource/physical_plan/struct.CsvSource.html#method.csv_options
    #[deprecated(
        since = "52.0.0",
        note = "newlines_in_values has moved to CsvSource. Access it via CsvSource::csv_options().newlines_in_values instead. It will be removed in 58.0.0 or 6 months after 52.0.0 is released, whichever comes first."
    )]
    pub fn newlines_in_values(&self) -> bool {
        false
    }

    #[deprecated(
        since = "52.0.0",
        note = "This method is no longer used, use eq_properties instead. It will be removed in 58.0.0 or 6 months after 52.0.0 is released, whichever comes first."
    )]
    pub fn projected_constraints(&self) -> Constraints {
        let props = self.eq_properties();
        props.constraints().clone()
    }

    #[deprecated(
        since = "52.0.0",
        note = "This method is no longer used, use eq_properties instead. It will be removed in 58.0.0 or 6 months after 52.0.0 is released, whichever comes first."
    )]
    pub fn file_column_projection_indices(&self) -> Option<Vec<usize>> {
        #[expect(deprecated)]
        self.file_source.projection().as_ref().map(|p| {
            p.ordered_column_indices()
                .into_iter()
                .filter(|&i| i < self.file_schema().fields().len())
                .collect::<Vec<_>>()
        })
    }

    /// Splits file groups into new groups based on statistics to enable efficient parallel processing.
    ///
    /// The method distributes files across a target number of partitions while ensuring
    /// files within each partition maintain sort order based on their min/max statistics.
    ///
    /// The algorithm works by:
    /// 1. Takes files sorted by minimum values
    /// 2. For each file:
    ///   - Finds eligible groups (empty or where file's min > group's last max)
    ///   - Selects the smallest eligible group
    ///   - Creates a new group if needed
    ///
    /// # Parameters
    /// * `table_schema`: Schema containing information about the columns
    /// * `file_groups`: The original file groups to split
    /// * `sort_order`: The lexicographical ordering to maintain within each group
    /// * `target_partitions`: The desired number of output partitions
    ///
    /// # Returns
    /// A new set of file groups, where files within each group are non-overlapping with respect to
    /// their min/max statistics and maintain the specified sort order.
    pub fn split_groups_by_statistics_with_target_partitions(
        table_schema: &SchemaRef,
        file_groups: &[FileGroup],
        sort_order: &LexOrdering,
        target_partitions: usize,
    ) -> Result<Vec<FileGroup>> {
        if target_partitions == 0 {
            return Err(internal_datafusion_err!(
                "target_partitions must be greater than 0"
            ));
        }

        let flattened_files = file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .collect::<Vec<_>>();

        if flattened_files.is_empty() {
            return Ok(vec![]);
        }

        let statistics = MinMaxStatistics::new_from_files(
            sort_order,
            table_schema,
            None,
            flattened_files.iter().copied(),
        )?;

        let indices_sorted_by_min = statistics.min_values_sorted();

        // Initialize with target_partitions empty groups
        let mut file_groups_indices: Vec<Vec<usize>> = vec![vec![]; target_partitions];

        for (idx, min) in indices_sorted_by_min {
            if let Some((_, group)) = file_groups_indices
                .iter_mut()
                .enumerate()
                .filter(|(_, group)| {
                    group.is_empty()
                        || min
                            > statistics
                                .max(*group.last().expect("groups should not be empty"))
                })
                .min_by_key(|(_, group)| group.len())
            {
                group.push(idx);
            } else {
                // Create a new group if no existing group fits
                file_groups_indices.push(vec![idx]);
            }
        }

        // Remove any empty groups
        file_groups_indices.retain(|group| !group.is_empty());

        // Assemble indices back into groups of PartitionedFiles
        Ok(file_groups_indices
            .into_iter()
            .map(|file_group_indices| {
                FileGroup::new(
                    file_group_indices
                        .into_iter()
                        .map(|idx| flattened_files[idx].clone())
                        .collect(),
                )
            })
            .collect())
    }

    /// Attempts to do a bin-packing on files into file groups, such that any two files
    /// in a file group are ordered and non-overlapping with respect to their statistics.
    /// It will produce the smallest number of file groups possible.
    pub fn split_groups_by_statistics(
        table_schema: &SchemaRef,
        file_groups: &[FileGroup],
        sort_order: &LexOrdering,
    ) -> Result<Vec<FileGroup>> {
        let flattened_files = file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .collect::<Vec<_>>();
        // First Fit:
        // * Choose the first file group that a file can be placed into.
        // * If it fits into no existing file groups, create a new one.
        //
        // By sorting files by min values and then applying first-fit bin packing,
        // we can produce the smallest number of file groups such that
        // files within a group are in order and non-overlapping.
        //
        // Source: Applied Combinatorics (Keller and Trotter), Chapter 6.8
        // https://www.appliedcombinatorics.org/book/s_posets_dilworth-intord.html

        if flattened_files.is_empty() {
            return Ok(vec![]);
        }

        let statistics = MinMaxStatistics::new_from_files(
            sort_order,
            table_schema,
            None,
            flattened_files.iter().copied(),
        )
        .map_err(|e| {
            e.context("construct min/max statistics for split_groups_by_statistics")
        })?;

        let indices_sorted_by_min = statistics.min_values_sorted();
        let mut file_groups_indices: Vec<Vec<usize>> = vec![];

        for (idx, min) in indices_sorted_by_min {
            let file_group_to_insert = file_groups_indices.iter_mut().find(|group| {
                // If our file is non-overlapping and comes _after_ the last file,
                // it fits in this file group.
                min > statistics.max(
                    *group
                        .last()
                        .expect("groups should be nonempty at construction"),
                )
            });
            match file_group_to_insert {
                Some(group) => group.push(idx),
                None => file_groups_indices.push(vec![idx]),
            }
        }

        // Assemble indices back into groups of PartitionedFiles
        Ok(file_groups_indices
            .into_iter()
            .map(|file_group_indices| {
                file_group_indices
                    .into_iter()
                    .map(|idx| flattened_files[idx].clone())
                    .collect()
            })
            .collect())
    }

    /// Write the data_type based on file_source
    fn fmt_file_source(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        write!(f, ", file_type={}", self.file_source.file_type())?;
        self.file_source.fmt_extra(t, f)
    }

    /// Returns the file_source
    pub fn file_source(&self) -> &Arc<dyn FileSource> {
        &self.file_source
    }

    /// Helper: Rebuild FileScanConfig with new file source
    fn rebuild_with_source(
        &self,
        new_file_source: Arc<dyn FileSource>,
        is_exact: bool,
    ) -> Result<Arc<dyn DataSource>> {
        let mut new_config = self.clone();

        // Reverse file groups (FileScanConfig's responsibility)
        new_config.file_groups = new_config
            .file_groups
            .into_iter()
            .map(|group| {
                let mut files = group.into_inner();
                files.reverse();
                files.into()
            })
            .collect();

        new_config.file_source = new_file_source;

        // Phase 1: Clear output_ordering for Inexact
        // (we're only reversing row groups, not guaranteeing perfect ordering)
        if !is_exact {
            new_config.output_ordering = vec![];
        }

        Ok(Arc::new(new_config))
    }
}

impl Debug for FileScanConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "FileScanConfig {{")?;
        write!(f, "object_store_url={:?}, ", self.object_store_url)?;

        write!(f, "statistics={:?}, ", self.statistics())?;

        DisplayAs::fmt_as(self, DisplayFormatType::Verbose, f)?;
        write!(f, "}}")
    }
}

impl DisplayAs for FileScanConfig {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> FmtResult {
        let schema = self.projected_schema().map_err(|_| std::fmt::Error {})?;
        let orderings = get_projected_output_ordering(self, &schema);

        write!(f, "file_groups=")?;
        FileGroupsDisplay(&self.file_groups).fmt_as(t, f)?;

        if !schema.fields().is_empty() {
            write!(f, ", projection={}", ProjectSchemaDisplay(&schema))?;
        }

        if let Some(limit) = self.limit {
            write!(f, ", limit={limit}")?;
        }

        display_orderings(f, &orderings)?;

        if !self.constraints.is_empty() {
            write!(f, ", {}", self.constraints)?;
        }

        Ok(())
    }
}

/// Get the indices of columns in a projection if the projection is a simple
/// list of columns.
/// If there are any expressions other than columns, returns None.
fn ordered_column_indices_from_projection(
    projection: &ProjectionExprs,
) -> Option<Vec<usize>> {
    projection
        .expr_iter()
        .map(|e| {
            let index = e.as_any().downcast_ref::<Column>()?.index();
            Some(index)
        })
        .collect::<Option<Vec<usize>>>()
}

/// A helper that projects partition columns into the file record batches.
///
/// One interesting trick is the usage of a cache for the key buffers of the partition column
/// dictionaries. Indeed, the partition columns are constant, so the dictionaries that represent them
/// have all their keys equal to 0. This enables us to re-use the same "all-zero" buffer across batches,
/// which makes the space consumption of the partition columns O(batch_size) instead of O(record_count).
pub struct PartitionColumnProjector {
    /// An Arrow buffer initialized to zeros that represents the key array of all partition
    /// columns (partition columns are materialized by dictionary arrays with only one
    /// value in the dictionary, thus all the keys are equal to zero).
    key_buffer_cache: ZeroBufferGenerators,
    /// Mapping between the indexes in the list of partition columns and the target
    /// schema. Sorted by index in the target schema so that we can iterate on it to
    /// insert the partition columns in the target record batch.
    projected_partition_indexes: Vec<(usize, usize)>,
    /// The schema of the table once the projection was applied.
    projected_schema: SchemaRef,
}

impl PartitionColumnProjector {
    // Create a projector to insert the partitioning columns into batches read from files
    // - `projected_schema`: the target schema with both file and partitioning columns
    // - `table_partition_cols`: all the partitioning column names
    pub fn new(projected_schema: SchemaRef, table_partition_cols: &[String]) -> Self {
        let mut idx_map = HashMap::new();
        for (partition_idx, partition_name) in table_partition_cols.iter().enumerate() {
            if let Ok(schema_idx) = projected_schema.index_of(partition_name) {
                idx_map.insert(partition_idx, schema_idx);
            }
        }

        let mut projected_partition_indexes: Vec<_> = idx_map.into_iter().collect();
        projected_partition_indexes.sort_by(|(_, a), (_, b)| a.cmp(b));

        Self {
            projected_partition_indexes,
            key_buffer_cache: Default::default(),
            projected_schema,
        }
    }

    // Transform the batch read from the file by inserting the partitioning columns
    // to the right positions as deduced from `projected_schema`
    // - `file_batch`: batch read from the file, with internal projection applied
    // - `partition_values`: the list of partition values, one for each partition column
    pub fn project(
        &mut self,
        file_batch: RecordBatch,
        partition_values: &[ScalarValue],
    ) -> Result<RecordBatch> {
        let expected_cols =
            self.projected_schema.fields().len() - self.projected_partition_indexes.len();

        if file_batch.columns().len() != expected_cols {
            return exec_err!(
                "Unexpected batch schema from file, expected {} cols but got {}",
                expected_cols,
                file_batch.columns().len()
            );
        }

        let mut cols = file_batch.columns().to_vec();
        for &(pidx, sidx) in &self.projected_partition_indexes {
            let p_value = partition_values.get(pidx).ok_or_else(|| {
                exec_datafusion_err!("Invalid partitioning found on disk")
            })?;

            let mut partition_value = Cow::Borrowed(p_value);

            // check if user forgot to dict-encode the partition value
            let field = self.projected_schema.field(sidx);
            let expected_data_type = field.data_type();
            let actual_data_type = partition_value.data_type();
            if let DataType::Dictionary(key_type, _) = expected_data_type {
                if !matches!(actual_data_type, DataType::Dictionary(_, _)) {
                    warn!(
                        "Partition value for column {} was not dictionary-encoded, applied auto-fix.",
                        field.name()
                    );
                    partition_value = Cow::Owned(ScalarValue::Dictionary(
                        key_type.clone(),
                        Box::new(partition_value.as_ref().clone()),
                    ));
                }
            }

            cols.insert(
                sidx,
                create_output_array(
                    &mut self.key_buffer_cache,
                    partition_value.as_ref(),
                    file_batch.num_rows(),
                )?,
            )
        }

        RecordBatch::try_new_with_options(
            Arc::clone(&self.projected_schema),
            cols,
            &RecordBatchOptions::new().with_row_count(Some(file_batch.num_rows())),
        )
        .map_err(Into::into)
    }
}

/// A helper that projects extended (i.e. partition/metadata) columns into the file record batches.
///
/// One interesting trick is the usage of a cache for the key buffers of the partition column
/// dictionaries. Indeed, the partition columns are constant, so the dictionaries that represent them
/// have all their keys equal to 0. This enables us to re-use the same "all-zero" buffer across batches,
/// which makes the space consumption of the partition columns O(batch_size) instead of O(record_count).
pub struct ExtendedColumnProjector {
    /// An Arrow buffer initialized to zeros that represents the key array of all partition
    /// columns (partition columns are materialized by dictionary arrays with only one
    /// value in the dictionary, thus all the keys are equal to zero).
    key_buffer_cache: ZeroBufferGenerators,
    /// Mapping between the indexes in the list of partition columns and the target
    /// schema. Sorted by index in the target schema so that we can iterate on it to
    /// insert the partition columns in the target record batch.
    projected_partition_indexes: Vec<(usize, usize)>,
    /// Similar to `projected_partition_indexes` but only stores the indexes in the target schema
    projected_metadata_indexes: Vec<usize>,
    /// The schema of the table once the projection was applied.
    projected_schema: SchemaRef,
    /// Mapping between the column name and the metadata column.
    metadata_map: HashMap<String, MetadataColumn>,
}

impl ExtendedColumnProjector {
    /// Create a projector to insert the partitioning/metadata columns into batches read from files
    /// - `projected_schema`: the target schema with file, partitioning and metadata columns
    /// - `table_partition_cols`: all the partitioning column names
    /// - `metadata_cols`: all the metadata columns
    pub fn new(
        projected_schema: SchemaRef,
        table_partition_cols: &[String],
        metadata_cols: &[MetadataColumn],
    ) -> Self {
        let mut idx_map = HashMap::new();
        for (partition_idx, partition_name) in table_partition_cols.iter().enumerate() {
            if let Ok(schema_idx) = projected_schema.index_of(partition_name) {
                idx_map.insert(partition_idx, schema_idx);
            }
        }

        let mut projected_partition_indexes: Vec<_> = idx_map.into_iter().collect();
        projected_partition_indexes.sort_by(|(_, a), (_, b)| a.cmp(b));

        let mut projected_metadata_indexes = vec![];
        for metadata_col in metadata_cols.iter() {
            if let Ok(schema_idx) = projected_schema.index_of(metadata_col.name()) {
                projected_metadata_indexes.push(schema_idx);
            }
        }
        // Sort to ensure that the final metadata column vector is expanded properly
        if !projected_metadata_indexes.is_empty() {
            projected_metadata_indexes.sort();
        }

        let mut metadata_map = HashMap::new();
        for metadata_col in metadata_cols.iter() {
            metadata_map.insert(metadata_col.name().to_string(), metadata_col.clone());
        }

        Self {
            key_buffer_cache: Default::default(),
            projected_partition_indexes,
            projected_metadata_indexes,
            projected_schema,
            metadata_map,
        }
    }

    /// Transform the batch read from the file by inserting both partitioning and metadata columns
    /// to the right positions as deduced from `projected_schema`
    /// - `file_batch`: batch read from the file, with internal projection applied
    /// - `partition_values`: the list of partition values, one for each partition column
    /// - `metadata`: the metadata of the file containing information like location, size, etc.
    pub fn project(
        &mut self,
        file_batch: RecordBatch,
        partition_values: &[ScalarValue],
        metadata: &ObjectMeta,
    ) -> Result<RecordBatch> {
        // Calculate expected number of columns from the file (excluding partition and metadata columns)
        let expected_cols = self.projected_schema.fields().len()
            - self.projected_partition_indexes.len()
            - self.projected_metadata_indexes.len();

        // Verify the file batch has the expected number of columns
        if file_batch.columns().len() != expected_cols {
            return exec_err!(
                "Unexpected batch schema from file, expected {} cols but got {}",
                expected_cols,
                file_batch.columns().len()
            );
        }

        // Start with the columns from the file batch
        let mut cols = file_batch.columns().to_vec();

        // Collect all columns to insert (both partition and metadata) with their schema indices
        // We need to insert them in ascending order by schema index to avoid index shifting issues
        enum InsertColumn<'a> {
            Partition { pidx: usize, sidx: usize },
            Metadata { sidx: usize, field_name: &'a str },
        }

        let mut inserts: Vec<InsertColumn> = Vec::with_capacity(
            self.projected_partition_indexes.len()
                + self.projected_metadata_indexes.len(),
        );

        for &(pidx, sidx) in &self.projected_partition_indexes {
            inserts.push(InsertColumn::Partition { pidx, sidx });
        }

        for &sidx in &self.projected_metadata_indexes {
            let field_name = self.projected_schema.field(sidx).name();
            inserts.push(InsertColumn::Metadata { sidx, field_name });
        }

        // Sort by schema index to ensure correct insertion order
        inserts.sort_by_key(|insert| match insert {
            InsertColumn::Partition { sidx, .. } => *sidx,
            InsertColumn::Metadata { sidx, .. } => *sidx,
        });

        // Insert columns in sorted order
        for insert in inserts {
            match insert {
                InsertColumn::Partition { pidx, sidx } => {
                    // Get the partition value from the provided values
                    let p_value = partition_values.get(pidx).ok_or_else(|| {
                        exec_datafusion_err!("Invalid partitioning found on disk")
                    })?;

                    let mut partition_value = Cow::Borrowed(p_value);

                    // Check if user forgot to dict-encode the partition value and apply auto-fix if needed
                    let field = self.projected_schema.field(sidx);
                    let expected_data_type = field.data_type();
                    let actual_data_type = partition_value.data_type();
                    if let DataType::Dictionary(key_type, _) = expected_data_type {
                        if !matches!(actual_data_type, DataType::Dictionary(_, _)) {
                            warn!(
                                "Partition value for column {} was not dictionary-encoded, applied auto-fix.",
                                field.name()
                            );
                            partition_value = Cow::Owned(ScalarValue::Dictionary(
                                key_type.clone(),
                                Box::new(partition_value.as_ref().clone()),
                            ));
                        }
                    }

                    // Create array and insert at the correct schema position
                    cols.insert(
                        sidx,
                        create_output_array(
                            &mut self.key_buffer_cache,
                            partition_value.as_ref(),
                            file_batch.num_rows(),
                        )?,
                    );
                }
                InsertColumn::Metadata { sidx, field_name } => {
                    // Get the metadata column type from the field name
                    let metadata_col =
                        self.metadata_map.get(field_name).ok_or_else(|| {
                            exec_datafusion_err!(
                                "Invalid metadata column: {}",
                                field_name
                            )
                        })?;

                    // Convert metadata to scalar value based on the column type
                    let scalar_value = metadata_col.to_scalar_value(metadata);

                    // Create array and insert at the correct schema position
                    cols.insert(
                        sidx,
                        scalar_value.to_array_of_size(file_batch.num_rows())?,
                    );
                }
            }
        }

        // Create a new record batch with all columns in the correct order
        RecordBatch::try_new_with_options(
            Arc::clone(&self.projected_schema),
            cols,
            &RecordBatchOptions::new().with_row_count(Some(file_batch.num_rows())),
        )
        .map_err(Into::into)
    }
}
#[derive(Debug, Default)]
struct ZeroBufferGenerators {
    gen_i8: ZeroBufferGenerator<i8>,
    gen_i16: ZeroBufferGenerator<i16>,
    gen_i32: ZeroBufferGenerator<i32>,
    gen_i64: ZeroBufferGenerator<i64>,
    gen_u8: ZeroBufferGenerator<u8>,
    gen_u16: ZeroBufferGenerator<u16>,
    gen_u32: ZeroBufferGenerator<u32>,
    gen_u64: ZeroBufferGenerator<u64>,
}

/// Generate a arrow [`Buffer`] that contains zero values.
#[derive(Debug, Default)]
struct ZeroBufferGenerator<T>
where
    T: ArrowNativeType,
{
    cache: Option<Buffer>,
    _t: PhantomData<T>,
}

impl<T> ZeroBufferGenerator<T>
where
    T: ArrowNativeType,
{
    const SIZE: usize = size_of::<T>();

    fn get_buffer(&mut self, n_vals: usize) -> Buffer {
        match &mut self.cache {
            Some(buf) if buf.len() >= n_vals * Self::SIZE => {
                buf.slice_with_length(0, n_vals * Self::SIZE)
            }
            _ => {
                let mut key_buffer_builder = BufferBuilder::<T>::new(n_vals);
                key_buffer_builder.advance(n_vals); // keys are all 0
                self.cache.insert(key_buffer_builder.finish()).clone()
            }
        }
    }
}

fn create_dict_array<T>(
    buffer_gen: &mut ZeroBufferGenerator<T>,
    dict_val: &ScalarValue,
    len: usize,
    data_type: DataType,
) -> Result<ArrayRef>
where
    T: ArrowNativeType,
{
    let dict_vals = dict_val.to_array()?;

    let sliced_key_buffer = buffer_gen.get_buffer(len);

    // assemble pieces together
    let mut builder = ArrayData::builder(data_type)
        .len(len)
        .add_buffer(sliced_key_buffer);
    builder = builder.add_child_data(dict_vals.to_data());
    Ok(Arc::new(DictionaryArray::<UInt16Type>::from(
        builder.build().unwrap(),
    )))
}

fn create_output_array(
    key_buffer_cache: &mut ZeroBufferGenerators,
    val: &ScalarValue,
    len: usize,
) -> Result<ArrayRef> {
    if let ScalarValue::Dictionary(key_type, dict_val) = &val {
        match key_type.as_ref() {
            DataType::Int8 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_i8,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::Int16 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_i16,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::Int32 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_i32,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::Int64 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_i64,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::UInt8 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_u8,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::UInt16 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_u16,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::UInt32 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_u32,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            DataType::UInt64 => {
                return create_dict_array(
                    &mut key_buffer_cache.gen_u64,
                    dict_val,
                    len,
                    val.data_type(),
                );
            }
            _ => {}
        }
    }

    val.to_array_of_size(len)
}

/// The various listing tables does not attempt to read all files
/// concurrently, instead they will read files in sequence within a
/// partition.  This is an important property as it allows plans to
/// run against 1000s of files and not try to open them all
/// concurrently.
///
/// However, it means if we assign more than one file to a partition
/// the output sort order will not be preserved as illustrated in the
/// following diagrams:
///
/// When only 1 file is assigned to each partition, each partition is
/// correctly sorted on `(A, B, C)`
///
/// ```text
/// ┏ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ┓
///   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐ ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ┐
/// ┃   ┌───────────────┐     ┌──────────────┐ │   ┌──────────────┐ │   ┌─────────────┐   ┃
///   │ │   1.parquet   │ │ │ │  2.parquet   │   │ │  3.parquet   │   │ │  4.parquet  │ │
/// ┃   │ Sort: A, B, C │     │Sort: A, B, C │ │   │Sort: A, B, C │ │   │Sort: A, B, C│   ┃
///   │ └───────────────┘ │ │ └──────────────┘   │ └──────────────┘   │ └─────────────┘ │
/// ┃                                          │                    │                     ┃
///   │                   │ │                    │                    │                 │
/// ┃                                          │                    │                     ┃
///   │                   │ │                    │                    │                 │
/// ┃                                          │                    │                     ┃
///   │                   │ │                    │                    │                 │
/// ┃  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  ─ ─ ─ ─ ─ ─ ─ ─ ─  ┃
///      DataFusion           DataFusion           DataFusion           DataFusion
/// ┃    Partition 1          Partition 2          Partition 3          Partition 4       ┃
///  ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━
///
///                                      DataSourceExec
/// ```
///
/// However, when more than 1 file is assigned to each partition, each
/// partition is NOT correctly sorted on `(A, B, C)`. Once the second
/// file is scanned, the same values for A, B and C can be repeated in
/// the same sorted stream
///
///```text
/// ┏ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━
///   ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐ ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─  ┃
/// ┃   ┌───────────────┐     ┌──────────────┐ │
///   │ │   1.parquet   │ │ │ │  2.parquet   │   ┃
/// ┃   │ Sort: A, B, C │     │Sort: A, B, C │ │
///   │ └───────────────┘ │ │ └──────────────┘   ┃
/// ┃   ┌───────────────┐     ┌──────────────┐ │
///   │ │   3.parquet   │ │ │ │  4.parquet   │   ┃
/// ┃   │ Sort: A, B, C │     │Sort: A, B, C │ │
///   │ └───────────────┘ │ │ └──────────────┘   ┃
/// ┃                                          │
///   │                   │ │                    ┃
/// ┃  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
///      DataFusion           DataFusion         ┃
/// ┃    Partition 1          Partition 2
///  ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ┛
///
///              DataSourceExec
/// ```
fn get_projected_output_ordering(
    base_config: &FileScanConfig,
    projected_schema: &SchemaRef,
) -> Vec<LexOrdering> {
    let projected_orderings =
        project_orderings(&base_config.output_ordering, projected_schema);

    let mut all_orderings = vec![];
    for new_ordering in projected_orderings {
        // Check if any file groups are not sorted
        if base_config.file_groups.iter().any(|group| {
            if group.len() <= 1 {
                // File groups with <= 1 files are always sorted
                return false;
            }

            let Some(indices) = base_config
                .file_source
                .projection()
                .as_ref()
                .map(|p| ordered_column_indices_from_projection(p))
            else {
                // Can't determine if ordered without a simple projection
                return true;
            };

            let statistics = match MinMaxStatistics::new_from_files(
                &new_ordering,
                projected_schema,
                indices.as_deref(),
                group.iter(),
            ) {
                Ok(statistics) => statistics,
                Err(e) => {
                    log::trace!("Error fetching statistics for file group: {e}");
                    // we can't prove that it's ordered, so we have to reject it
                    return true;
                }
            };

            !statistics.is_sorted()
        }) {
            debug!(
                "Skipping specified output ordering {:?}. \
                Some file groups couldn't be determined to be sorted: {:?}",
                base_config.output_ordering[0], base_config.file_groups
            );
            continue;
        }

        all_orderings.push(new_ordering);
    }
    all_orderings
}

/// Convert type to a type suitable for use as a `ListingTable`
/// partition column. Returns `Dictionary(UInt16, val_type)`, which is
/// a reasonable trade off between a reasonable number of partition
/// values and space efficiency.
///
/// This use this to specify types for partition columns. However
/// you MAY also choose not to dictionary-encode the data or to use a
/// different dictionary type.
///
/// Use [`wrap_partition_value_in_dict`] to wrap a [`ScalarValue`] in the same say.
pub fn wrap_partition_type_in_dict(val_type: DataType) -> DataType {
    DataType::Dictionary(Box::new(DataType::UInt16), Box::new(val_type))
}

/// Convert a [`ScalarValue`] of partition columns to a type, as
/// described in the documentation of [`wrap_partition_type_in_dict`],
/// which can wrap the types.
pub fn wrap_partition_value_in_dict(val: ScalarValue) -> ScalarValue {
    ScalarValue::Dictionary(Box::new(DataType::UInt16), Box::new(val))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::TableSchema;
    use crate::test_util::col;
    use crate::{
        generate_test_files, test_util::MockSource, tests::aggr_test_schema,
        verify_sort_integrity,
    };

    use arrow::array::{Int32Array, RecordBatch, StringArray, UInt64Array};
    use arrow::datatypes::Field;
    use chrono::{TimeZone, Utc};
    use datafusion_common::stats::Precision;
    use datafusion_common::{ColumnStatistics, internal_err};
    use datafusion_expr::{Operator, SortExpr};
    use datafusion_physical_expr::create_physical_sort_expr;
    use datafusion_physical_expr::expressions::{BinaryExpr, Column, Literal};
    use datafusion_physical_expr::projection::ProjectionExpr;
    use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
    use object_store::path::Path;

    /// Helper function to create a test ObjectMeta with a default timestamp
    fn create_test_object_meta(path: &str, size: usize) -> ObjectMeta {
        ObjectMeta {
            location: Path::from(path),
            last_modified: Utc.timestamp_opt(1234567890, 0).unwrap(),
            size: size.try_into().unwrap(),
            e_tag: None,
            version: None,
        }
    }

    #[test]
    fn physical_plan_config_no_projection_tab_cols_as_field() {
        let file_schema = aggr_test_schema();

        // make a table_partition_col as a field
        let table_partition_col =
            Field::new("date", wrap_partition_type_in_dict(DataType::Utf8), true)
                .with_metadata(HashMap::from_iter(vec![(
                    "key_whatever".to_owned(),
                    "value_whatever".to_owned(),
                )]));

        let conf = config_for_projection(
            Arc::clone(&file_schema),
            None,
            Statistics::new_unknown(&file_schema),
            vec![table_partition_col.clone()],
        );

        // verify the proj_schema includes the last column and exactly the same the field it is defined
        let proj_schema = conf.projected_schema().unwrap();
        assert_eq!(proj_schema.fields().len(), file_schema.fields().len() + 1);
        assert_eq!(
            *proj_schema.field(file_schema.fields().len()),
            table_partition_col,
            "partition columns are the last columns and ust have all values defined in created field"
        );
    }

    #[test]
    fn test_split_groups_by_statistics() -> Result<()> {
        use chrono::TimeZone;
        use datafusion_common::DFSchema;
        use datafusion_expr::execution_props::ExecutionProps;
        use object_store::{ObjectMeta, path::Path};

        struct File {
            name: &'static str,
            date: &'static str,
            statistics: Vec<Option<(Option<f64>, Option<f64>)>>,
        }
        impl File {
            fn new(
                name: &'static str,
                date: &'static str,
                statistics: Vec<Option<(f64, f64)>>,
            ) -> Self {
                Self::new_nullable(
                    name,
                    date,
                    statistics
                        .into_iter()
                        .map(|opt| opt.map(|(min, max)| (Some(min), Some(max))))
                        .collect(),
                )
            }

            fn new_nullable(
                name: &'static str,
                date: &'static str,
                statistics: Vec<Option<(Option<f64>, Option<f64>)>>,
            ) -> Self {
                Self {
                    name,
                    date,
                    statistics,
                }
            }
        }

        struct TestCase {
            name: &'static str,
            file_schema: Schema,
            files: Vec<File>,
            sort: Vec<SortExpr>,
            expected_result: Result<Vec<Vec<&'static str>>, &'static str>,
        }

        use datafusion_expr::col;
        let cases = vec![
            TestCase {
                name: "test sort",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![
                    File::new("0", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("1", "2023-01-01", vec![Some((0.50, 1.00))]),
                    File::new("2", "2023-01-02", vec![Some((0.00, 1.00))]),
                ],
                sort: vec![col("value").sort(true, false)],
                expected_result: Ok(vec![vec!["0", "1"], vec!["2"]]),
            },
            // same input but file '2' is in the middle
            // test that we still order correctly
            TestCase {
                name: "test sort with files ordered differently",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![
                    File::new("0", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("2", "2023-01-02", vec![Some((0.00, 1.00))]),
                    File::new("1", "2023-01-01", vec![Some((0.50, 1.00))]),
                ],
                sort: vec![col("value").sort(true, false)],
                expected_result: Ok(vec![vec!["0", "1"], vec!["2"]]),
            },
            TestCase {
                name: "reverse sort",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![
                    File::new("0", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("1", "2023-01-01", vec![Some((0.50, 1.00))]),
                    File::new("2", "2023-01-02", vec![Some((0.00, 1.00))]),
                ],
                sort: vec![col("value").sort(false, true)],
                expected_result: Ok(vec![vec!["1", "0"], vec!["2"]]),
            },
            TestCase {
                name: "nullable sort columns, nulls last",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    true,
                )]),
                files: vec![
                    File::new_nullable(
                        "0",
                        "2023-01-01",
                        vec![Some((Some(0.00), Some(0.49)))],
                    ),
                    File::new_nullable("1", "2023-01-01", vec![Some((Some(0.50), None))]),
                    File::new_nullable("2", "2023-01-02", vec![Some((Some(0.00), None))]),
                ],
                sort: vec![col("value").sort(true, false)],
                expected_result: Ok(vec![vec!["0", "1"], vec!["2"]]),
            },
            TestCase {
                name: "nullable sort columns, nulls first",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    true,
                )]),
                files: vec![
                    File::new_nullable("0", "2023-01-01", vec![Some((None, Some(0.49)))]),
                    File::new_nullable(
                        "1",
                        "2023-01-01",
                        vec![Some((Some(0.50), Some(1.00)))],
                    ),
                    File::new_nullable("2", "2023-01-02", vec![Some((None, Some(1.00)))]),
                ],
                sort: vec![col("value").sort(true, true)],
                expected_result: Ok(vec![vec!["0", "1"], vec!["2"]]),
            },
            TestCase {
                name: "all three non-overlapping",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![
                    File::new("0", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("1", "2023-01-01", vec![Some((0.50, 0.99))]),
                    File::new("2", "2023-01-02", vec![Some((1.00, 1.49))]),
                ],
                sort: vec![col("value").sort(true, false)],
                expected_result: Ok(vec![vec!["0", "1", "2"]]),
            },
            TestCase {
                name: "all three overlapping",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![
                    File::new("0", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("1", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("2", "2023-01-02", vec![Some((0.00, 0.49))]),
                ],
                sort: vec![col("value").sort(true, false)],
                expected_result: Ok(vec![vec!["0"], vec!["1"], vec!["2"]]),
            },
            TestCase {
                name: "empty input",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![],
                sort: vec![col("value").sort(true, false)],
                expected_result: Ok(vec![]),
            },
            TestCase {
                name: "one file missing statistics",
                file_schema: Schema::new(vec![Field::new(
                    "value".to_string(),
                    DataType::Float64,
                    false,
                )]),
                files: vec![
                    File::new("0", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("1", "2023-01-01", vec![Some((0.00, 0.49))]),
                    File::new("2", "2023-01-02", vec![None]),
                ],
                sort: vec![col("value").sort(true, false)],
                expected_result: Err(
                    "construct min/max statistics for split_groups_by_statistics\ncaused by\ncollect min/max values\ncaused by\nget min/max for column: 'value'\ncaused by\nError during planning: statistics not found",
                ),
            },
        ];

        for case in cases {
            let table_schema = Arc::new(Schema::new(
                case.file_schema
                    .fields()
                    .clone()
                    .into_iter()
                    .cloned()
                    .chain(Some(Arc::new(Field::new(
                        "date".to_string(),
                        DataType::Utf8,
                        false,
                    ))))
                    .collect::<Vec<_>>(),
            ));
            let Some(sort_order) = LexOrdering::new(
                case.sort
                    .into_iter()
                    .map(|expr| {
                        create_physical_sort_expr(
                            &expr,
                            &DFSchema::try_from(Arc::clone(&table_schema))?,
                            &ExecutionProps::default(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
            ) else {
                return internal_err!("This test should always use an ordering");
            };

            let partitioned_files = FileGroup::new(
                case.files.into_iter().map(From::from).collect::<Vec<_>>(),
            );
            let result = FileScanConfig::split_groups_by_statistics(
                &table_schema,
                std::slice::from_ref(&partitioned_files),
                &sort_order,
            );
            let results_by_name = result
                .as_ref()
                .map(|file_groups| {
                    file_groups
                        .iter()
                        .map(|file_group| {
                            file_group
                                .iter()
                                .map(|file| {
                                    partitioned_files
                                        .iter()
                                        .find_map(|f| {
                                            if f.object_meta == file.object_meta {
                                                Some(
                                                    f.object_meta
                                                        .location
                                                        .as_ref()
                                                        .rsplit('/')
                                                        .next()
                                                        .unwrap()
                                                        .trim_end_matches(".parquet"),
                                                )
                                            } else {
                                                None
                                            }
                                        })
                                        .unwrap()
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .map_err(|e| e.strip_backtrace().leak() as &'static str);

            assert_eq!(results_by_name, case.expected_result, "{}", case.name);
        }

        return Ok(());

        impl From<File> for PartitionedFile {
            fn from(file: File) -> Self {
                PartitionedFile {
                    object_meta: ObjectMeta {
                        location: Path::from(format!(
                            "data/date={}/{}.parquet",
                            file.date, file.name
                        )),
                        last_modified: Utc.timestamp_nanos(0),
                        size: 0,
                        e_tag: None,
                        version: None,
                    },
                    partition_values: vec![ScalarValue::from(file.date)],
                    range: None,
                    statistics: Some(Arc::new(Statistics {
                        num_rows: Precision::Absent,
                        total_byte_size: Precision::Absent,
                        column_statistics: file
                            .statistics
                            .into_iter()
                            .map(|stats| {
                                stats
                                    .map(|(min, max)| ColumnStatistics {
                                        min_value: Precision::Exact(
                                            ScalarValue::Float64(min),
                                        ),
                                        max_value: Precision::Exact(
                                            ScalarValue::Float64(max),
                                        ),
                                        ..Default::default()
                                    })
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>(),
                    })),
                    extensions: None,
                    metadata_size_hint: None,
                }
            }
        }
    }

    // sets default for configs that play no role in projections
    fn config_for_projection(
        file_schema: SchemaRef,
        projection: Option<Vec<usize>>,
        statistics: Statistics,
        table_partition_cols: Vec<Field>,
    ) -> FileScanConfig {
        let table_schema = TableSchema::new(
            file_schema,
            table_partition_cols.into_iter().map(Arc::new).collect(),
        );
        FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("test:///").unwrap(),
            Arc::new(MockSource::new(table_schema.clone())),
        )
        .with_projection_indices(projection)
        .unwrap()
        .with_statistics(statistics)
        .build()
    }

    #[test]
    fn test_file_scan_config_builder() {
        let file_schema = aggr_test_schema();
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let table_schema = TableSchema::new(
            Arc::clone(&file_schema),
            vec![Arc::new(Field::new(
                "date",
                wrap_partition_type_in_dict(DataType::Utf8),
                false,
            ))],
        );

        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // Create a builder with required parameters
        let builder = FileScanConfigBuilder::new(
            object_store_url.clone(),
            Arc::clone(&file_source),
        );

        // Build with various configurations
        let config = builder
            .with_limit(Some(1000))
            .with_projection_indices(Some(vec![0, 1]))
            .unwrap()
            .with_statistics(Statistics::new_unknown(&file_schema))
            .with_file_groups(vec![FileGroup::new(vec![PartitionedFile::new(
                "test.parquet".to_string(),
                1024,
            )])])
            .with_output_ordering(vec![
                [PhysicalSortExpr::new_default(Arc::new(Column::new(
                    "date", 0,
                )))]
                .into(),
            ])
            .with_file_compression_type(FileCompressionType::UNCOMPRESSED)
            .build();

        // Verify the built config has all the expected values
        assert_eq!(config.object_store_url, object_store_url);
        assert_eq!(*config.file_schema(), file_schema);
        assert_eq!(config.limit, Some(1000));
        assert_eq!(
            config
                .file_source
                .projection()
                .as_ref()
                .map(|p| p.column_indices()),
            Some(vec![0, 1])
        );
        assert_eq!(config.table_partition_cols().len(), 1);
        assert_eq!(config.table_partition_cols()[0].name(), "date");
        assert_eq!(config.file_groups.len(), 1);
        assert_eq!(config.file_groups[0].len(), 1);
        assert_eq!(
            config.file_groups[0][0].object_meta.location.as_ref(),
            "test.parquet"
        );
        assert_eq!(
            config.file_compression_type,
            FileCompressionType::UNCOMPRESSED
        );
        assert_eq!(config.output_ordering.len(), 1);
    }

    #[test]
    fn equivalence_properties_after_schema_change() {
        let file_schema = aggr_test_schema();
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);

        // Create a file source with a filter
        let file_source: Arc<dyn FileSource> = Arc::new(
            MockSource::new(table_schema.clone()).with_filter(Arc::new(BinaryExpr::new(
                col("c2", &file_schema).unwrap(),
                Operator::Eq,
                Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
            ))),
        );

        let config = FileScanConfigBuilder::new(
            object_store_url.clone(),
            Arc::clone(&file_source),
        )
        .with_projection_indices(Some(vec![0, 1, 2]))
        .unwrap()
        .build();

        // Simulate projection being updated. Since the filter has already been pushed down,
        // the new projection won't include the filtered column.
        let exprs = ProjectionExprs::new(vec![ProjectionExpr::new(
            col("c1", &file_schema).unwrap(),
            "c1",
        )]);
        let data_source = config
            .try_swapping_with_projection(&exprs)
            .unwrap()
            .unwrap();

        // Gather the equivalence properties from the new data source. There should
        // be no equivalence class for column c2 since it was removed by the projection.
        let eq_properties = data_source.eq_properties();
        let eq_group = eq_properties.eq_group();

        for class in eq_group.iter() {
            for expr in class.iter() {
                if let Some(col) = expr.as_any().downcast_ref::<Column>() {
                    assert_ne!(
                        col.name(),
                        "c2",
                        "c2 should not be present in any equivalence class"
                    );
                }
            }
        }
    }

    #[test]
    fn test_file_scan_config_builder_defaults() {
        let file_schema = aggr_test_schema();
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);

        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // Create a builder with only required parameters and build without any additional configurations
        let config = FileScanConfigBuilder::new(
            object_store_url.clone(),
            Arc::clone(&file_source),
        )
        .build();

        // Verify default values
        assert_eq!(config.object_store_url, object_store_url);
        assert_eq!(*config.file_schema(), file_schema);
        assert_eq!(config.limit, None);
        // When no projection is specified, the file source should have an unprojected projection
        // (i.e., all columns)
        let expected_projection: Vec<usize> = (0..file_schema.fields().len()).collect();
        assert_eq!(
            config
                .file_source
                .projection()
                .as_ref()
                .map(|p| p.column_indices()),
            Some(expected_projection)
        );
        assert!(config.table_partition_cols().is_empty());
        assert!(config.file_groups.is_empty());
        assert_eq!(
            config.file_compression_type,
            FileCompressionType::UNCOMPRESSED
        );
        assert!(config.output_ordering.is_empty());
        assert!(config.constraints.is_empty());

        // Verify statistics are set to unknown
        assert_eq!(config.statistics().num_rows, Precision::Absent);
        assert_eq!(config.statistics().total_byte_size, Precision::Absent);
        assert_eq!(
            config.statistics().column_statistics.len(),
            file_schema.fields().len()
        );
        for stat in config.statistics().column_statistics {
            assert_eq!(stat.distinct_count, Precision::Absent);
            assert_eq!(stat.min_value, Precision::Absent);
            assert_eq!(stat.max_value, Precision::Absent);
            assert_eq!(stat.null_count, Precision::Absent);
        }
    }

    #[test]
    fn test_file_scan_config_builder_new_from() {
        let schema = aggr_test_schema();
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();
        let partition_cols = vec![Field::new(
            "date",
            wrap_partition_type_in_dict(DataType::Utf8),
            false,
        )];
        let file = PartitionedFile::new("test_file.parquet", 100);

        let table_schema = TableSchema::new(
            Arc::clone(&schema),
            partition_cols.iter().map(|f| Arc::new(f.clone())).collect(),
        );

        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // Create a config with non-default values
        let original_config = FileScanConfigBuilder::new(
            object_store_url.clone(),
            Arc::clone(&file_source),
        )
        .with_projection_indices(Some(vec![0, 2]))
        .unwrap()
        .with_limit(Some(10))
        .with_file(file.clone())
        .with_constraints(Constraints::default())
        .build();

        // Create a new builder from the config
        let new_builder = FileScanConfigBuilder::from(original_config);

        // Build a new config from this builder
        let new_config = new_builder.build();

        // Verify properties match
        let partition_cols = partition_cols.into_iter().map(Arc::new).collect::<Vec<_>>();
        assert_eq!(new_config.object_store_url, object_store_url);
        assert_eq!(*new_config.file_schema(), schema);
        assert_eq!(
            new_config
                .file_source
                .projection()
                .as_ref()
                .map(|p| p.column_indices()),
            Some(vec![0, 2])
        );
        assert_eq!(new_config.limit, Some(10));
        assert_eq!(*new_config.table_partition_cols(), partition_cols);
        assert_eq!(new_config.file_groups.len(), 1);
        assert_eq!(new_config.file_groups[0].len(), 1);
        assert_eq!(
            new_config.file_groups[0][0].object_meta.location.as_ref(),
            "test_file.parquet"
        );
        assert_eq!(new_config.constraints, Constraints::default());
    }

    /// Regression test for the bug where metadata columns were excluded from the
    /// projected schema when projection_indices was None.
    ///
    /// When projection_indices is None, all columns should be included, including
    /// configured metadata columns. Before the fix, projected_metadata_positions was
    /// set to an empty vector when projection_indices was None, causing metadata
    /// columns to be excluded from projected_schema().
    #[test]
    fn test_metadata_cols_included_when_projection_indices_none() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();
        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);
        let file_source: Arc<dyn FileSource> = Arc::new(MockSource::new(table_schema));

        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        // Create a config with metadata columns but WITHOUT projection indices (None)
        // This should include all columns: file columns + metadata columns
        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols.clone())
            .build();

        // Verify that the projected schema includes both file and metadata columns
        let projected = config.projected_schema().expect("projected schema");

        // Should have 4 columns: col1, col2, location, size
        assert_eq!(
            projected.fields().len(),
            4,
            "Expected 4 columns (2 file + 2 metadata), got {}",
            projected.fields().len()
        );

        // Verify file columns
        assert_eq!(projected.field(0).name(), "col1");
        assert_eq!(projected.field(1).name(), "col2");

        // Verify metadata columns are present at the end
        assert_eq!(projected.field(2).name(), metadata_cols[0].name());
        assert_eq!(projected.field(3).name(), metadata_cols[1].name());
    }

    /// Test that metadata columns work correctly with partition columns when
    /// projection_indices is None.
    #[test]
    fn test_metadata_and_partition_cols_when_projection_indices_none() {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Int32,
            false,
        )]));
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let partition_cols = vec![Arc::new(Field::new(
            "year",
            wrap_partition_type_in_dict(DataType::Int32),
            false,
        ))];
        let metadata_cols = vec![MetadataColumn::Location(None)];

        let table_schema = TableSchema::new(Arc::clone(&file_schema), partition_cols);
        let file_source: Arc<dyn FileSource> = Arc::new(MockSource::new(table_schema));

        // Create a config with partition and metadata columns but NO projection indices
        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols.clone())
            .build();

        // Verify that the projected schema includes all columns
        let projected = config.projected_schema().expect("projected schema");

        // Should have 3 columns: data (file), year (partition), location (metadata)
        assert_eq!(
            projected.fields().len(),
            3,
            "Expected 3 columns (1 file + 1 partition + 1 metadata), got {}",
            projected.fields().len()
        );

        // Verify column order: file columns, then partition columns, then metadata columns
        assert_eq!(projected.field(0).name(), "data");
        assert_eq!(projected.field(1).name(), "year");
        assert_eq!(projected.field(2).name(), metadata_cols[0].name());
    }

    /// Regression test for the bug where metadata and partition columns were
    /// inserted incorrectly when their schema indices were interleaved.
    ///
    /// The bug was that partition columns were inserted first (using their schema indices),
    /// then metadata columns were inserted (also using their schema indices). But after
    /// inserting partition columns, the indices shift, causing metadata columns to be
    /// inserted at wrong positions.
    ///
    /// This test creates a schema where:
    /// - file column at index 0
    /// - metadata column at index 1 (location)
    /// - partition column at index 2 (year)
    /// - metadata column at index 3 (size)
    /// - partition column at index 4 (month)
    ///
    /// Without the fix, the columns would be inserted in wrong positions.
    #[test]
    fn test_partition_metadata_interleaved_schema_indices() {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Int32,
            false,
        )]));

        // Create partition and metadata columns
        let partition_cols = vec!["year".to_string(), "month".to_string()];
        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        // Create a projected schema where partition and metadata columns are interleaved:
        // [data, location, year, size, month]
        let projected_fields = vec![
            Field::new("data", DataType::Int32, false), // index 0 - file
            metadata_cols[0].field(),                   // index 1 - location (metadata)
            Field::new(
                "year",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Int32),
                ),
                true,
            ), // index 2 - year (partition)
            metadata_cols[1].field(),                   // index 3 - size (metadata)
            Field::new(
                "month",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Int32),
                ),
                true,
            ), // index 4 - month (partition)
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        // Create projector
        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &partition_cols,
            &metadata_cols,
        );

        // Create test data
        let file_batch = RecordBatch::try_new(
            Arc::clone(&file_schema),
            vec![Arc::new(Int32Array::from(vec![100, 200, 300]))],
        )
        .unwrap();

        let partition_values = vec![
            ScalarValue::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(ScalarValue::Int32(Some(2024))),
            ),
            ScalarValue::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(ScalarValue::Int32(Some(6))),
            ),
        ];

        let object_meta = create_test_object_meta("test/file.parquet", 2048);

        // Apply projection
        let result = projector
            .project(file_batch, &partition_values, &object_meta)
            .unwrap();

        // Verify the schema matches expected order
        assert_eq!(result.num_columns(), 5);
        assert_eq!(result.schema().field(0).name(), "data");
        assert_eq!(result.schema().field(1).name(), "location");
        assert_eq!(result.schema().field(2).name(), "year");
        assert_eq!(result.schema().field(3).name(), "size");
        assert_eq!(result.schema().field(4).name(), "month");

        // Verify the values are correct in each column
        let data_col = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(data_col.value(0), 100);
        assert_eq!(data_col.value(1), 200);
        assert_eq!(data_col.value(2), 300);

        let location_col = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(location_col.value(0), "test/file.parquet");

        let size_col = result
            .column(3)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(size_col.value(0), 2048);
    }

    /// Test where metadata column comes before partition column in schema
    #[test]
    fn test_metadata_before_partition_in_schema() {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));

        let partition_cols = vec!["part".to_string()];
        let metadata_cols = vec![MetadataColumn::Size];

        // Schema: [value, size, part]
        let projected_fields = vec![
            Field::new("value", DataType::Int32, false),
            metadata_cols[0].field(), // index 1 - metadata
            Field::new(
                "part",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ), // index 2 - partition
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &partition_cols,
            &metadata_cols,
        );

        let file_batch = RecordBatch::try_new(
            Arc::clone(&file_schema),
            vec![Arc::new(Int32Array::from(vec![42]))],
        )
        .unwrap();

        let partition_values = vec![ScalarValue::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(ScalarValue::from("A")),
        )];

        let object_meta = create_test_object_meta("path/to/file", 512);

        let result = projector
            .project(file_batch, &partition_values, &object_meta)
            .unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.schema().field(0).name(), "value");
        assert_eq!(result.schema().field(1).name(), "size");
        assert_eq!(result.schema().field(2).name(), "part");

        // Verify values
        let size_col = result
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(size_col.value(0), 512);
    }

    /// Test where partition column comes before metadata column in schema
    #[test]
    fn test_partition_before_metadata_in_schema() {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));

        let partition_cols = vec!["part".to_string()];
        let metadata_cols = vec![MetadataColumn::Location(None)];

        // Schema: [value, part, location]
        let projected_fields = vec![
            Field::new("value", DataType::Int32, false),
            Field::new(
                "part",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ), // index 1 - partition
            metadata_cols[0].field(), // index 2 - metadata
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &partition_cols,
            &metadata_cols,
        );

        let file_batch = RecordBatch::try_new(
            Arc::clone(&file_schema),
            vec![Arc::new(Int32Array::from(vec![99]))],
        )
        .unwrap();

        let partition_values = vec![ScalarValue::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(ScalarValue::from("B")),
        )];

        let object_meta = create_test_object_meta("another/path", 256);

        let result = projector
            .project(file_batch, &partition_values, &object_meta)
            .unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.schema().field(0).name(), "value");
        assert_eq!(result.schema().field(1).name(), "part");
        assert_eq!(result.schema().field(2).name(), "location");

        let location_col = result
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(location_col.value(0), "another/path");
    }

    /// Test with multiple file columns and interleaved partition/metadata columns
    #[test]
    fn test_multiple_file_cols_with_interleaved_partition_metadata() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("col_a", DataType::Int32, false),
            Field::new("col_b", DataType::Int32, false),
            Field::new("col_c", DataType::Int32, false),
        ]));

        let partition_cols = vec!["p1".to_string(), "p2".to_string()];
        let metadata_cols = vec![MetadataColumn::Size, MetadataColumn::LastModified];

        // Schema: [col_a, size, col_b, p1, last_modified, col_c, p2]
        let projected_fields = vec![
            Field::new("col_a", DataType::Int32, false), // 0 - file
            metadata_cols[0].field(),                    // 1 - size (metadata)
            Field::new("col_b", DataType::Int32, false), // 2 - file
            Field::new(
                "p1",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ), // 3 - partition
            metadata_cols[1].field(),                    // 4 - last_modified (metadata)
            Field::new("col_c", DataType::Int32, false), // 5 - file
            Field::new(
                "p2",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ), // 6 - partition
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &partition_cols,
            &metadata_cols,
        );

        let file_batch = RecordBatch::try_new(
            Arc::clone(&file_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![10, 20])),
                Arc::new(Int32Array::from(vec![100, 200])),
            ],
        )
        .unwrap();

        let partition_values = vec![
            ScalarValue::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(ScalarValue::from("X")),
            ),
            ScalarValue::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(ScalarValue::from("Y")),
            ),
        ];

        let object_meta = create_test_object_meta("complex/path", 4096);

        let result = projector
            .project(file_batch, &partition_values, &object_meta)
            .unwrap();

        // Verify schema order
        assert_eq!(result.num_columns(), 7);
        assert_eq!(result.schema().field(0).name(), "col_a");
        assert_eq!(result.schema().field(1).name(), "size");
        assert_eq!(result.schema().field(2).name(), "col_b");
        assert_eq!(result.schema().field(3).name(), "p1");
        assert_eq!(result.schema().field(4).name(), "last_modified");
        assert_eq!(result.schema().field(5).name(), "col_c");
        assert_eq!(result.schema().field(6).name(), "p2");

        // Verify file column values
        let col_a = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col_a.value(0), 1);
        assert_eq!(col_a.value(1), 2);

        let col_b = result
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col_b.value(0), 10);
        assert_eq!(col_b.value(1), 20);

        let col_c = result
            .column(5)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(col_c.value(0), 100);
        assert_eq!(col_c.value(1), 200);

        // Verify metadata column values
        let size_col = result
            .column(1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(size_col.value(0), 4096);
        assert_eq!(size_col.value(1), 4096);
    }

    /// Test with only partition columns (no metadata)
    #[test]
    fn test_only_partition_columns() {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Int32,
            false,
        )]));

        let partition_cols = vec!["p1".to_string(), "p2".to_string()];

        // Schema: [data, p1, p2]
        let projected_fields = vec![
            Field::new("data", DataType::Int32, false),
            Field::new(
                "p1",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ),
            Field::new(
                "p2",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ),
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &partition_cols,
            &[], // No metadata columns
        );

        let file_batch = RecordBatch::try_new(
            Arc::clone(&file_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let partition_values = vec![
            ScalarValue::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(ScalarValue::from("val1")),
            ),
            ScalarValue::Dictionary(
                Box::new(DataType::UInt16),
                Box::new(ScalarValue::from("val2")),
            ),
        ];

        let object_meta = create_test_object_meta("file", 100);

        let result = projector
            .project(file_batch, &partition_values, &object_meta)
            .unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.schema().field(0).name(), "data");
        assert_eq!(result.schema().field(1).name(), "p1");
        assert_eq!(result.schema().field(2).name(), "p2");
    }

    /// Test with only metadata columns (no partitions)
    #[test]
    fn test_only_metadata_columns() {
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "data",
            DataType::Int32,
            false,
        )]));

        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        // Schema: [data, location, size]
        let projected_fields = vec![
            Field::new("data", DataType::Int32, false),
            metadata_cols[0].field(),
            metadata_cols[1].field(),
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &[], // No partition columns
            &metadata_cols,
        );

        let file_batch = RecordBatch::try_new(
            Arc::clone(&file_schema),
            vec![Arc::new(Int32Array::from(vec![7, 8, 9]))],
        )
        .unwrap();

        let object_meta = create_test_object_meta("meta/file.dat", 999);

        let result = projector.project(file_batch, &[], &object_meta).unwrap();

        assert_eq!(result.num_columns(), 3);
        assert_eq!(result.schema().field(0).name(), "data");
        assert_eq!(result.schema().field(1).name(), "location");
        assert_eq!(result.schema().field(2).name(), "size");

        let location_col = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(location_col.value(0), "meta/file.dat");

        let size_col = result
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(size_col.value(0), 999);
    }

    /// Test with no file columns, only partition and metadata
    #[test]
    fn test_no_file_columns() {
        let file_schema = Arc::new(Schema::empty());

        let partition_cols = vec!["part".to_string()];
        let metadata_cols = vec![MetadataColumn::Size];

        // Schema: [part, size]
        let projected_fields = vec![
            Field::new(
                "part",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                ),
                true,
            ),
            metadata_cols[0].field(),
        ];
        let projected_schema = Arc::new(Schema::new(projected_fields));

        let mut projector = ExtendedColumnProjector::new(
            Arc::clone(&projected_schema),
            &partition_cols,
            &metadata_cols,
        );

        // Create an empty batch with 5 rows using RecordBatchOptions
        let file_batch = RecordBatch::try_new_with_options(
            Arc::clone(&file_schema),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(5)),
        )
        .unwrap();

        let partition_values = vec![ScalarValue::Dictionary(
            Box::new(DataType::UInt16),
            Box::new(ScalarValue::from("partition_value")),
        )];

        let object_meta = create_test_object_meta("empty_file", 0);

        let result = projector
            .project(file_batch, &partition_values, &object_meta)
            .unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.num_rows(), 5);
        assert_eq!(result.schema().field(0).name(), "part");
        assert_eq!(result.schema().field(1).name(), "size");
    }

    #[test]
    fn test_split_groups_by_statistics_with_target_partitions() -> Result<()> {
        use datafusion_common::DFSchema;
        use datafusion_expr::{col, execution_props::ExecutionProps};

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));

        // Setup sort expression
        let exec_props = ExecutionProps::new();
        let df_schema = DFSchema::try_from_qualified_schema("test", schema.as_ref())?;
        let sort_expr = [col("value").sort(true, false)];
        let sort_ordering = sort_expr
            .map(|expr| {
                create_physical_sort_expr(&expr, &df_schema, &exec_props).unwrap()
            })
            .into();

        // Test case parameters
        struct TestCase {
            name: String,
            file_count: usize,
            overlap_factor: f64,
            target_partitions: usize,
            expected_partition_count: usize,
        }

        let test_cases = vec![
            // Basic cases
            TestCase {
                name: "no_overlap_10_files_4_partitions".to_string(),
                file_count: 10,
                overlap_factor: 0.0,
                target_partitions: 4,
                expected_partition_count: 4,
            },
            TestCase {
                name: "medium_overlap_20_files_5_partitions".to_string(),
                file_count: 20,
                overlap_factor: 0.5,
                target_partitions: 5,
                expected_partition_count: 5,
            },
            TestCase {
                name: "high_overlap_30_files_3_partitions".to_string(),
                file_count: 30,
                overlap_factor: 0.8,
                target_partitions: 3,
                expected_partition_count: 7,
            },
            // Edge cases
            TestCase {
                name: "fewer_files_than_partitions".to_string(),
                file_count: 3,
                overlap_factor: 0.0,
                target_partitions: 10,
                expected_partition_count: 3, // Should only create as many partitions as files
            },
            TestCase {
                name: "single_file".to_string(),
                file_count: 1,
                overlap_factor: 0.0,
                target_partitions: 5,
                expected_partition_count: 1, // Should create only one partition
            },
            TestCase {
                name: "empty_files".to_string(),
                file_count: 0,
                overlap_factor: 0.0,
                target_partitions: 3,
                expected_partition_count: 0, // Empty result for empty input
            },
        ];

        for case in test_cases {
            println!("Running test case: {}", case.name);

            // Generate files using bench utility function
            let file_groups = generate_test_files(case.file_count, case.overlap_factor);

            // Call the function under test
            let result =
                FileScanConfig::split_groups_by_statistics_with_target_partitions(
                    &schema,
                    &file_groups,
                    &sort_ordering,
                    case.target_partitions,
                )?;

            // Verify results
            println!(
                "Created {} partitions (target was {})",
                result.len(),
                case.target_partitions
            );

            // Check partition count
            assert_eq!(
                result.len(),
                case.expected_partition_count,
                "Case '{}': Unexpected partition count",
                case.name
            );

            // Verify sort integrity
            assert!(
                verify_sort_integrity(&result),
                "Case '{}': Files within partitions are not properly ordered",
                case.name
            );

            // Distribution check for partitions
            if case.file_count > 1 && case.expected_partition_count > 1 {
                let group_sizes: Vec<usize> = result.iter().map(FileGroup::len).collect();
                let max_size = *group_sizes.iter().max().unwrap();
                let min_size = *group_sizes.iter().min().unwrap();

                // Check partition balancing - difference shouldn't be extreme
                let avg_files_per_partition =
                    case.file_count as f64 / case.expected_partition_count as f64;
                assert!(
                    (max_size as f64) < 2.0 * avg_files_per_partition,
                    "Case '{}': Unbalanced distribution. Max partition size {} exceeds twice the average {}",
                    case.name,
                    max_size,
                    avg_files_per_partition
                );

                println!("Distribution - min files: {min_size}, max files: {max_size}");
            }
        }

        // Test error case: zero target partitions
        let empty_groups: Vec<FileGroup> = vec![];
        let err = FileScanConfig::split_groups_by_statistics_with_target_partitions(
            &schema,
            &empty_groups,
            &sort_ordering,
            0,
        )
        .unwrap_err();

        assert!(
            err.to_string()
                .contains("target_partitions must be greater than 0"),
            "Expected error for zero target partitions"
        );

        Ok(())
    }

    #[test]
    fn test_partition_statistics_projection() {
        // This test verifies that partition_statistics applies projection correctly.
        // The old implementation had a bug where it returned file group statistics
        // without applying the projection, returning all column statistics instead
        // of just the projected ones.

        use crate::source::DataSourceExec;
        use datafusion_physical_plan::ExecutionPlan;

        // Create a schema with 4 columns
        let schema = Arc::new(Schema::new(vec![
            Field::new("col0", DataType::Int32, false),
            Field::new("col1", DataType::Int32, false),
            Field::new("col2", DataType::Int32, false),
            Field::new("col3", DataType::Int32, false),
        ]));

        // Create statistics for all 4 columns
        let file_group_stats = Statistics {
            num_rows: Precision::Exact(100),
            total_byte_size: Precision::Exact(1024),
            column_statistics: vec![
                ColumnStatistics {
                    null_count: Precision::Exact(0),
                    ..ColumnStatistics::new_unknown()
                },
                ColumnStatistics {
                    null_count: Precision::Exact(5),
                    ..ColumnStatistics::new_unknown()
                },
                ColumnStatistics {
                    null_count: Precision::Exact(10),
                    ..ColumnStatistics::new_unknown()
                },
                ColumnStatistics {
                    null_count: Precision::Exact(15),
                    ..ColumnStatistics::new_unknown()
                },
            ],
        };

        // Create a file group with statistics
        let file_group = FileGroup::new(vec![PartitionedFile::new("test.parquet", 1024)])
            .with_statistics(Arc::new(file_group_stats));

        let table_schema = TableSchema::new(Arc::clone(&schema), vec![]);

        // Create a FileScanConfig with projection: only keep columns 0 and 2
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("test:///").unwrap(),
            Arc::new(MockSource::new(table_schema.clone())),
        )
        .with_projection_indices(Some(vec![0, 2]))
        .unwrap() // Only project columns 0 and 2
        .with_file_groups(vec![file_group])
        .build();

        // Create a DataSourceExec from the config
        let exec = DataSourceExec::from_data_source(config);

        // Get statistics for partition 0
        let partition_stats = exec.partition_statistics(Some(0)).unwrap();

        // Verify that only 2 columns are in the statistics (the projected ones)
        assert_eq!(
            partition_stats.column_statistics.len(),
            2,
            "Expected 2 column statistics (projected), but got {}",
            partition_stats.column_statistics.len()
        );

        // Verify the column statistics are for columns 0 and 2
        assert_eq!(
            partition_stats.column_statistics[0].null_count,
            Precision::Exact(0),
            "First projected column should be col0 with 0 nulls"
        );
        assert_eq!(
            partition_stats.column_statistics[1].null_count,
            Precision::Exact(10),
            "Second projected column should be col2 with 10 nulls"
        );

        // Verify row count and byte size
        assert_eq!(partition_stats.num_rows, Precision::Exact(100));
        assert_eq!(partition_stats.total_byte_size, Precision::Exact(800));
    }

    #[test]
    fn test_output_partitioning_not_partitioned_by_file_group() {
        let file_schema = aggr_test_schema();
        let partition_col =
            Field::new("date", wrap_partition_type_in_dict(DataType::Utf8), false);

        let config = config_for_projection(
            Arc::clone(&file_schema),
            None,
            Statistics::new_unknown(&file_schema),
            vec![partition_col],
        );

        // partitioned_by_file_group defaults to false
        let partitioning = config.output_partitioning();
        assert!(matches!(partitioning, Partitioning::UnknownPartitioning(_)));
    }

    #[test]
    fn test_output_partitioning_no_partition_columns() {
        let file_schema = aggr_test_schema();
        let mut config = config_for_projection(
            Arc::clone(&file_schema),
            None,
            Statistics::new_unknown(&file_schema),
            vec![], // No partition columns
        );
        config.partitioned_by_file_group = true;

        let partitioning = config.output_partitioning();
        assert!(matches!(partitioning, Partitioning::UnknownPartitioning(_)));
    }

    #[test]
    fn test_output_partitioning_with_partition_columns() {
        let file_schema = aggr_test_schema();

        // Test single partition column
        let single_partition_col = vec![Field::new(
            "date",
            wrap_partition_type_in_dict(DataType::Utf8),
            false,
        )];

        let mut config = config_for_projection(
            Arc::clone(&file_schema),
            None,
            Statistics::new_unknown(&file_schema),
            single_partition_col,
        );
        config.partitioned_by_file_group = true;
        config.file_groups = vec![
            FileGroup::new(vec![PartitionedFile::new("f1.parquet".to_string(), 1024)]),
            FileGroup::new(vec![PartitionedFile::new("f2.parquet".to_string(), 1024)]),
            FileGroup::new(vec![PartitionedFile::new("f3.parquet".to_string(), 1024)]),
        ];

        let partitioning = config.output_partitioning();
        match partitioning {
            Partitioning::Hash(exprs, num_partitions) => {
                assert_eq!(num_partitions, 3);
                assert_eq!(exprs.len(), 1);
                assert_eq!(
                    exprs[0].as_any().downcast_ref::<Column>().unwrap().name(),
                    "date"
                );
            }
            _ => panic!("Expected Hash partitioning"),
        }

        // Test multiple partition columns
        let multiple_partition_cols = vec![
            Field::new("year", wrap_partition_type_in_dict(DataType::Utf8), false),
            Field::new("month", wrap_partition_type_in_dict(DataType::Utf8), false),
        ];

        config = config_for_projection(
            Arc::clone(&file_schema),
            None,
            Statistics::new_unknown(&file_schema),
            multiple_partition_cols,
        );
        config.partitioned_by_file_group = true;
        config.file_groups = vec![
            FileGroup::new(vec![PartitionedFile::new("f1.parquet".to_string(), 1024)]),
            FileGroup::new(vec![PartitionedFile::new("f2.parquet".to_string(), 1024)]),
        ];

        let partitioning = config.output_partitioning();
        match partitioning {
            Partitioning::Hash(exprs, num_partitions) => {
                assert_eq!(num_partitions, 2);
                assert_eq!(exprs.len(), 2);
                let col_names: Vec<_> = exprs
                    .iter()
                    .map(|e| e.as_any().downcast_ref::<Column>().unwrap().name())
                    .collect();
                assert_eq!(col_names, vec!["year", "month"]);
            }
            _ => panic!("Expected Hash partitioning"),
        }
    }

    /// Test that `try_pushdown_filters` works when a filter references
    /// a metadata column (e.g. `location`) that is appended after the projection.
    /// Metadata column indices are beyond the projection length, so they must be
    /// separated out rather than remapped through `update_expr`.
    #[test]
    fn test_try_pushdown_filters_with_metadata_column_filter() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").expect("valid url");
        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        let metadata_cols = vec![MetadataColumn::Location(None)];

        // Build config with metadata columns and a projection.
        let config =
            FileScanConfigBuilder::new(object_store_url, Arc::clone(&file_source))
                .with_metadata_cols(metadata_cols)
                .with_projection_indices(Some(vec![0, 1]))
                .expect("valid projection")
                .build();

        // Push a projection so that `file_source.projection()` is Some.
        let proj_exprs = ProjectionExprs::new(vec![
            ProjectionExpr::new(col("id", &file_schema).expect("col"), "id"),
            ProjectionExpr::new(col("value", &file_schema).expect("col"), "value"),
        ]);
        let data_source = config
            .try_swapping_with_projection(&proj_exprs)
            .expect("swap ok")
            .expect("should produce new source");

        // projected schema: [id, value, location]
        let projected = data_source
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("is FileScanConfig")
            .projected_schema()
            .expect("projected schema");
        assert_eq!(projected.fields().len(), 3);
        assert_eq!(projected.field(2).name(), "location");

        // Create a filter on the metadata column: location@2 = 's3://bucket'
        // (index 2 because projected output is [id@0, value@1, location@2])
        let location_filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("location", 2)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "s3://bucket".to_string(),
            )))),
        ));

        // This should work without crashing, even though the filter references a metadata column
        let config_options = ConfigOptions::default();
        let result =
            data_source.try_pushdown_filters(vec![location_filter], &config_options);
        let propagation = result.expect("to pushdown filters");

        // The metadata filter cannot be pushed down to the file source.
        assert_eq!(propagation.filters.len(), 1);
        assert!(
            matches!(propagation.filters[0], PushedDown::No),
            "metadata column filter should not be pushed down"
        );
    }

    /// Test that `try_pushdown_filters` correctly handles a mix of regular
    /// filters and metadata column filters — regular filters are remapped
    /// through the projection while metadata filters are returned as not pushed.
    #[test]
    fn test_try_pushdown_filters_mixed_regular_and_metadata_filters() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").expect("valid url");
        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        let metadata_cols = vec![MetadataColumn::Location(None)];

        let config =
            FileScanConfigBuilder::new(object_store_url, Arc::clone(&file_source))
                .with_metadata_cols(metadata_cols)
                .with_projection_indices(Some(vec![0, 1]))
                .expect("valid projection")
                .build();

        let proj_exprs = ProjectionExprs::new(vec![
            ProjectionExpr::new(col("id", &file_schema).expect("col"), "id"),
            ProjectionExpr::new(col("value", &file_schema).expect("col"), "value"),
        ]);
        let data_source = config
            .try_swapping_with_projection(&proj_exprs)
            .expect("swap ok")
            .expect("should produce new source");

        // Filter 0: regular filter on id@0 (within projection)
        let id_filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(5)))),
        ));
        // Filter 1: metadata column filter on location@2 (beyond projection)
        let location_filter: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("location", 2)),
            Operator::Eq,
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "s3://bucket".to_string(),
            )))),
        ));

        let config_options = ConfigOptions::default();
        let result = data_source
            .try_pushdown_filters(vec![id_filter, location_filter], &config_options);
        let propagation = result.expect("to pushdown filters");

        // Both filters should be present in results, in the original order.
        assert_eq!(propagation.filters.len(), 2);
        // Regular filter: MockSource returns PushedDown::No (default impl).
        assert!(
            matches!(propagation.filters[0], PushedDown::No),
            "regular filter: MockSource default returns No"
        );
        // Metadata filter: separated out, returned as PushedDown::No.
        assert!(
            matches!(propagation.filters[1], PushedDown::No),
            "metadata column filter should not be pushed down"
        );
    }

    /// Test that `eq_properties` includes metadata columns in its schema.
    /// `DataSourceExec::schema()` is derived from `eq_properties().schema()`,
    /// so metadata columns must be appended to match the actual batches
    /// produced by `FileStream` via `ExtendedColumnProjector`.
    #[test]
    fn test_eq_properties_schema_includes_metadata_columns() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").expect("valid url");
        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols)
            .build();

        let eq_props = config.eq_properties();
        let schema = eq_props.schema();

        // Schema should include file columns + metadata columns
        assert_eq!(
            schema.fields().len(),
            4,
            "Expected 4 fields (2 file + 2 metadata), got {}",
            schema.fields().len()
        );
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "value");
        assert_eq!(schema.field(2).name(), "location");
        assert_eq!(schema.field(3).name(), "size");
    }

    /// Same as above but with a projection applied — metadata columns must
    /// still appear in the schema after projection.
    #[test]
    fn test_eq_properties_schema_includes_metadata_columns_with_projection() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").expect("valid url");
        let table_schema = TableSchema::new(Arc::clone(&file_schema), vec![]);
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        let metadata_cols = vec![MetadataColumn::Location(None)];

        let config =
            FileScanConfigBuilder::new(object_store_url, Arc::clone(&file_source))
                .with_metadata_cols(metadata_cols)
                .with_projection_indices(Some(vec![0, 1]))
                .expect("valid projection")
                .build();

        // Push a projection so that file_source.projection() is Some
        let proj_exprs = ProjectionExprs::new(vec![
            ProjectionExpr::new(col("id", &file_schema).expect("col"), "id"),
            ProjectionExpr::new(col("value", &file_schema).expect("col"), "value"),
        ]);
        let data_source = config
            .try_swapping_with_projection(&proj_exprs)
            .expect("swap ok")
            .expect("should produce new source");

        let eq_props = data_source.eq_properties();
        let schema = eq_props.schema();

        // Projected schema: [id, value] + metadata: [location]
        assert_eq!(
            schema.fields().len(),
            3,
            "Expected 3 fields (2 projected + 1 metadata), got {}",
            schema.fields().len()
        );
        assert_eq!(schema.field(0).name(), "id");
        assert_eq!(schema.field(1).name(), "value");
        assert_eq!(schema.field(2).name(), "location");

        // Verify it matches projected_schema()
        let projected = data_source
            .as_any()
            .downcast_ref::<FileScanConfig>()
            .expect("is FileScanConfig")
            .projected_schema()
            .expect("projected schema");
        assert_eq!(schema.fields().len(), projected.fields().len());
        for (eq_field, proj_field) in
            schema.fields().iter().zip(projected.fields().iter())
        {
            assert_eq!(eq_field.name(), proj_field.name());
        }
    }

    /// Test that projection with both metadata and partition columns works correctly.
    ///
    /// Scenario: A hive-partitioned parquet table with file columns, partition columns,
    /// and metadata columns. When `SELECT *` is issued, DataFusion creates a projection
    /// that includes ALL column indices (file + partition + metadata).
    #[test]
    fn test_projection_indices_with_partition_and_metadata_columns() {
        // Simulate hive-partitioned parquet data with metadata columns
        // File schema: [id]
        let file_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        // Partition columns: [year, month, day] (hive partitioning)
        let partition_cols = vec![
            Arc::new(Field::new(
                "year",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
            Arc::new(Field::new(
                "month",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
            Arc::new(Field::new(
                "day",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
        ];

        // Metadata columns: [location, size, last_modified]
        let metadata_cols = vec![
            MetadataColumn::Location(None),
            MetadataColumn::Size,
            MetadataColumn::LastModified,
        ];

        let table_schema =
            TableSchema::new(Arc::clone(&file_schema), partition_cols.clone());
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // table_schema() returns [id, year, month, day] = 4 columns
        assert_eq!(
            table_schema.table_schema().fields().len(),
            4,
            "table_schema should have 4 fields (1 file + 3 partition)"
        );

        // SELECT * projection: all file + partition + metadata column indices
        // [0=id, 1=year, 2=month, 3=day, 4=location, 5=size, 6=last_modified]
        let all_indices = Some(vec![0, 1, 2, 3, 4, 5, 6]);

        // This should NOT panic - metadata column indices (4,5,6) exceed
        // table_schema().fields().len() (4) but should be handled gracefully
        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols.clone())
            .with_projection_indices(all_indices)
            .expect("with_projection_indices should handle metadata column indices")
            .build();

        // Verify projected schema includes all 7 columns in the correct order
        let projected = config.projected_schema().expect("projected schema");
        assert_eq!(
            projected.fields().len(),
            7,
            "Expected 7 columns (1 file + 3 partition + 3 metadata), got {}",
            projected.fields().len()
        );

        assert_eq!(projected.field(0).name(), "id");
        assert_eq!(projected.field(1).name(), "year");
        assert_eq!(projected.field(2).name(), "month");
        assert_eq!(projected.field(3).name(), "day");
        assert_eq!(projected.field(4).name(), "location");
        assert_eq!(projected.field(5).name(), "size");
        assert_eq!(projected.field(6).name(), "last_modified");
    }

    /// Test that projection with only some metadata columns works correctly.
    ///
    /// E.g. SELECT id, location FROM table_with_partitions_and_metadata
    /// projection = [0, 4] where 0=id (file), 4=location (metadata)
    #[test]
    fn test_projection_indices_partial_metadata_with_partitions() {
        let file_schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let partition_cols = vec![
            Arc::new(Field::new(
                "year",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
            Arc::new(Field::new(
                "month",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
        ];

        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        let table_schema =
            TableSchema::new(Arc::clone(&file_schema), partition_cols.clone());
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // table_schema = [id(0), year(1), month(2)] = 3 columns
        // metadata = [location(3), size(4)]
        // SELECT id, location -> projection = [0, 3]
        let partial_indices = Some(vec![0, 3]);

        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols.clone())
            .with_projection_indices(partial_indices)
            .expect("with_projection_indices should handle partial metadata indices")
            .build();

        let projected = config.projected_schema().expect("projected schema");
        assert_eq!(
            projected.fields().len(),
            2,
            "Expected 2 columns (id + location), got {}",
            projected.fields().len()
        );

        assert_eq!(projected.field(0).name(), "id");
        assert_eq!(projected.field(1).name(), "location");
    }

    /// Test that projection with metadata columns in non-sequential order works.
    ///
    /// E.g. SELECT location, id, size FROM table
    /// projection = [3, 0, 4] where reordered
    #[test]
    fn test_projection_indices_metadata_reordered() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let partition_cols = vec![Arc::new(Field::new(
            "day",
            wrap_partition_type_in_dict(DataType::Int32),
            false,
        ))];

        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        let table_schema =
            TableSchema::new(Arc::clone(&file_schema), partition_cols.clone());
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // table_schema = [id(0), value(1), day(2)] = 3 columns
        // metadata = [location(3), size(4)]
        // SELECT location, id, size -> projection = [3, 0, 4]
        let reordered_indices = Some(vec![3, 0, 4]);

        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols.clone())
            .with_projection_indices(reordered_indices)
            .expect("with_projection_indices should handle reordered metadata")
            .build();

        let projected = config.projected_schema().expect("projected schema");
        assert_eq!(
            projected.fields().len(),
            3,
            "Expected 3 columns (location + id + size), got {}",
            projected.fields().len()
        );

        assert_eq!(projected.field(0).name(), "location");
        assert_eq!(projected.field(1).name(), "id");
        assert_eq!(projected.field(2).name(), "size");
    }

    /// Test partial projection selecting file, partition, AND metadata columns.
    ///
    /// E.g. SELECT id, year, location FROM hive_partitioned_table
    /// projection = [0, 1, 4] where 0=id (file), 1=year (partition), 4=location (metadata)
    #[test]
    fn test_projection_indices_file_partition_and_metadata() {
        let file_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        let object_store_url = ObjectStoreUrl::parse("test:///").unwrap();

        let partition_cols = vec![
            Arc::new(Field::new(
                "year",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
            Arc::new(Field::new(
                "month",
                wrap_partition_type_in_dict(DataType::Int32),
                false,
            )),
        ];

        let metadata_cols = vec![MetadataColumn::Location(None), MetadataColumn::Size];

        let table_schema =
            TableSchema::new(Arc::clone(&file_schema), partition_cols.clone());
        let file_source: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));

        // table_schema = [id(0), value(1), year(2), month(3)] = 4 columns
        // metadata = [location(4), size(5)]
        // SELECT id, year, location -> projection = [0, 2, 4]
        let indices = Some(vec![0, 2, 4]);

        let config = FileScanConfigBuilder::new(object_store_url, file_source)
            .with_metadata_cols(metadata_cols.clone())
            .with_projection_indices(indices)
            .expect("with_projection_indices should handle mixed file+partition+metadata")
            .build();

        let projected = config.projected_schema().expect("projected schema");
        assert_eq!(
            projected.fields().len(),
            3,
            "Expected 3 columns (id + year + location), got {}",
            projected.fields().len()
        );

        assert_eq!(projected.field(0).name(), "id");
        assert_eq!(projected.field(1).name(), "year");
        assert_eq!(projected.field(2).name(), "location");

        // Now test reordered: SELECT location, year, id -> projection = [4, 2, 0]
        let file_source2: Arc<dyn FileSource> =
            Arc::new(MockSource::new(table_schema.clone()));
        let object_store_url2 = ObjectStoreUrl::parse("test:///").unwrap();
        let reordered_indices = Some(vec![4, 2, 0]);

        let config2 = FileScanConfigBuilder::new(object_store_url2, file_source2)
            .with_metadata_cols(metadata_cols)
            .with_projection_indices(reordered_indices)
            .expect(
                "with_projection_indices should handle reordered file+partition+metadata",
            )
            .build();

        let projected2 = config2.projected_schema().expect("projected schema");
        assert_eq!(
            projected2.fields().len(),
            3,
            "Expected 3 columns (location + year + id), got {}",
            projected2.fields().len()
        );

        assert_eq!(projected2.field(0).name(), "location");
        assert_eq!(projected2.field(1).name(), "year");
        assert_eq!(projected2.field(2).name(), "id");
    }
}
