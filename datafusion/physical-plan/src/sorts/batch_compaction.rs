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

//! Preflight allocation bounds for replacing retained scalar buffers with copies.

use std::sync::Arc;

use arrow::array::builder::{BinaryViewBuilder, StringViewBuilder};
use arrow::array::{
    Array, ArrayRef, AsArray, Capacities, MutableArrayData, RecordBatch, make_array,
    new_null_array,
};
use arrow::datatypes::DataType;
use datafusion_common::Result;

pub(super) struct BatchCompaction {
    columns: Vec<ColumnCompaction>,
    /// Upper bound on the new Arrow buffer capacities, before any copy is made.
    pub(super) capacity: usize,
}

struct ColumnCompaction {
    kind: CopyKind,
    rows_capacity: usize,
    capacity: usize,
}

enum CopyKind {
    Null,
    Plain,
    Bytes { value_bytes: usize },
    StringView { block_size: u32 },
    BinaryView { block_size: u32 },
}

impl BatchCompaction {
    /// Unsupported representations and overflowing bounds retain the input batch.
    pub(super) fn try_new(batch: &RecordBatch) -> Result<Option<Self>> {
        if batch
            .columns()
            .iter()
            .any(|column| !supported(column.data_type()))
        {
            return Ok(None);
        }
        let mut columns = Vec::with_capacity(batch.num_columns());
        let mut capacity = 0usize;
        for column in batch.columns() {
            let data = column.to_data();
            let logical_bytes = data.get_slice_memory_size()?;
            let source_null_bytes = if data.nulls().is_some() {
                column.len().div_ceil(8)
            } else {
                0
            };
            let Some(plan) =
                ColumnCompaction::new(column, logical_bytes, source_null_bytes)
            else {
                return Ok(None);
            };
            let Some(total) = capacity.checked_add(plan.capacity) else {
                return Ok(None);
            };
            capacity = total;
            columns.push(plan);
        }
        Ok(Some(Self { columns, capacity }))
    }

    /// The caller must reserve `capacity` additional live buffer bytes first.
    pub(super) fn copy(&self, input: &RecordBatch) -> Result<RecordBatch> {
        let columns = self
            .columns
            .iter()
            .zip(input.columns())
            .map(|(plan, column)| plan.copy(column))
            .collect::<Result<Vec<_>>>()?;
        Ok(RecordBatch::try_new(input.schema(), columns)?)
    }
}

fn supported(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Null
                | DataType::Boolean
                | DataType::FixedSizeBinary(_)
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::Utf8View
                | DataType::BinaryView
        )
}

fn rounded_capacity(bytes: usize) -> Option<usize> {
    bytes.checked_add(63).map(|value| value & !63)
}

fn view_block_size(bytes: usize) -> Option<u32> {
    let bytes = u32::try_from(bytes.max(1)).ok()?;
    // A completed view block must be strictly smaller than the offset limit.
    (bytes < u32::MAX).then_some(bytes)
}

impl ColumnCompaction {
    fn new(
        column: &ArrayRef,
        logical_bytes: usize,
        source_null_bytes: usize,
    ) -> Option<Self> {
        let rows = column.len();
        let mut rows_capacity = rows;
        let (kind, values_capacity) = match column.data_type() {
            DataType::Null => (CopyKind::Null, 0),
            DataType::Utf8
            | DataType::Binary
            | DataType::LargeUtf8
            | DataType::LargeBinary => {
                let offset_width =
                    if matches!(column.data_type(), DataType::Utf8 | DataType::Binary) {
                        4
                    } else {
                        8
                    };
                // MutableArrayData reserves one extra offset while extending. An
                // extra row of initial capacity prevents its doubling growth path.
                rows_capacity = rows.checked_add(1)?;
                let logical_offsets = rows_capacity.checked_mul(offset_width)?;
                let value_bytes = logical_bytes
                    .checked_sub(source_null_bytes)?
                    .checked_sub(logical_offsets)?;
                let offsets_capacity = rounded_capacity(
                    rows_capacity.checked_add(1)?.checked_mul(offset_width)?,
                )?;
                (
                    CopyKind::Bytes { value_bytes },
                    offsets_capacity.checked_add(rounded_capacity(value_bytes)?)?,
                )
            }
            DataType::Utf8View => {
                let bytes = column.as_string_view().iter().flatten().try_fold(
                    0usize,
                    |sum, value| {
                        sum.checked_add(if value.len() > 12 { value.len() } else { 0 })
                    },
                )?;
                let block_size = view_block_size(bytes)?;
                (
                    CopyKind::StringView { block_size },
                    rows.checked_mul(16)?.checked_add(bytes)?,
                )
            }
            DataType::BinaryView => {
                let bytes = column.as_binary_view().iter().flatten().try_fold(
                    0usize,
                    |sum, value| {
                        sum.checked_add(if value.len() > 12 { value.len() } else { 0 })
                    },
                )?;
                let block_size = view_block_size(bytes)?;
                (
                    CopyKind::BinaryView { block_size },
                    rows.checked_mul(16)?.checked_add(bytes)?,
                )
            }
            _ => (
                CopyKind::Plain,
                rounded_capacity(logical_bytes.checked_sub(source_null_bytes)?)?,
            ),
        };
        let null_capacity = if (column.null_count() > 0 || source_null_bytes > 0)
            && !matches!(kind, CopyKind::Null)
        {
            rounded_capacity(rows_capacity.div_ceil(8))?
        } else {
            0
        };
        Some(Self {
            kind,
            rows_capacity,
            capacity: values_capacity.checked_add(null_capacity)?,
        })
    }

    fn copy(&self, column: &ArrayRef) -> Result<ArrayRef> {
        match self.kind {
            CopyKind::Null => Ok(new_null_array(column.data_type(), column.len())),
            CopyKind::StringView { block_size } => {
                let mut builder = StringViewBuilder::with_capacity(column.len())
                    .with_fixed_block_size(block_size);
                for value in column.as_string_view().iter() {
                    match value {
                        Some(value) => builder.try_append_value(value)?,
                        None => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            CopyKind::BinaryView { block_size } => {
                let mut builder = BinaryViewBuilder::with_capacity(column.len())
                    .with_fixed_block_size(block_size);
                for value in column.as_binary_view().iter() {
                    match value {
                        Some(value) => builder.try_append_value(value)?,
                        None => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            CopyKind::Plain | CopyKind::Bytes { .. } => {
                let capacity = match self.kind {
                    CopyKind::Bytes { value_bytes } => {
                        Capacities::Binary(self.rows_capacity, Some(value_bytes))
                    }
                    _ => Capacities::Array(self.rows_capacity),
                };
                let data = column.to_data();
                let mut output = MutableArrayData::with_capacities(
                    vec![&data],
                    data.nulls().is_some(),
                    capacity,
                );
                output.try_extend(0, 0, column.len())?;
                Ok(make_array(output.freeze()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::spill::get_record_batch_memory_size;
    use arrow::array::{
        BinaryArray, BinaryViewArray, BooleanArray, Decimal128Array, DictionaryArray,
        FixedSizeBinaryArray, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        StringArray, StringViewArray,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{Field, Int32Type, Schema};

    #[test]
    fn view_block_size_respects_completed_block_limit() {
        assert_eq!(view_block_size(0), Some(1));
        assert_eq!(view_block_size(13), Some(13));
        assert_eq!(view_block_size((u32::MAX - 1) as usize), Some(u32::MAX - 1));
        assert_eq!(view_block_size(u32::MAX as usize), None);
        assert_eq!(view_block_size(usize::MAX), None);
    }

    fn check_copy(column: ArrayRef) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "a",
            column.data_type().clone(),
            true,
        )]));
        let input = RecordBatch::try_new(schema, vec![column])?;
        let plan = BatchCompaction::try_new(&input)?.expect("supported scalar array");
        let output = plan.copy(&input)?;
        assert_eq!(input, output);
        assert!(get_record_batch_memory_size(&output) <= plan.capacity);
        let addresses = |batch: &RecordBatch| {
            let data = batch.column(0).to_data();
            data.buffers()
                .iter()
                .chain(data.nulls().map(|nulls| nulls.buffer()))
                .filter(|buffer| buffer.capacity() > 0)
                .map(|buffer| buffer.data_ptr().addr())
                .collect::<HashSet<_>>()
        };
        assert!(addresses(&input).is_disjoint(&addresses(&output)));
        Ok(())
    }

    #[test]
    fn scalar_copy_capacity_and_values_at_buffer_boundaries() -> Result<()> {
        for rows in [0usize, 1, 7, 8, 15, 16, 31, 63, 64, 65, 127, 8192] {
            let long = "a".repeat(257);
            let strings = (0..rows + 2)
                .map(|i| match i % 4 {
                    0 => None,
                    1 => Some(""),
                    2 => Some("short"),
                    _ => Some(long.as_str()),
                })
                .collect::<Vec<_>>();
            let binaries = strings
                .iter()
                .map(|value| value.map(str::as_bytes))
                .collect::<Vec<_>>();
            let arrays: Vec<ArrayRef> = vec![
                Arc::new(StringArray::from(strings.clone())),
                Arc::new(LargeStringArray::from(strings.clone())),
                Arc::new(StringViewArray::from(strings)),
                Arc::new(BinaryArray::from(binaries.clone())),
                Arc::new(LargeBinaryArray::from(binaries.clone())),
                Arc::new(BinaryViewArray::from(binaries)),
                Arc::new(Int64Array::from_iter((0..rows + 2).map(|i| match i % 4 {
                    0 => None,
                    1 => Some(i64::MIN),
                    2 => Some(0),
                    _ => Some(i64::MAX),
                }))),
                Arc::new(BooleanArray::from_iter((0..rows + 2).map(|i| {
                    match i % 3 {
                        0 => None,
                        1 => Some(false),
                        _ => Some(true),
                    }
                }))),
                Arc::new(
                    Decimal128Array::from_iter_values(
                        (0..rows + 2).map(|i| i128::try_from(i).expect("small fixture")),
                    )
                    .with_precision_and_scale(38, 4)?,
                ),
                Arc::new(FixedSizeBinaryArray::try_from_iter(
                    (0..rows + 2).map(|_| [0u8, 255, 7]),
                )?),
            ];
            for array in arrays {
                check_copy(array.slice(1, rows))?;
            }
            check_copy(Arc::new(Int32Array::new(
                vec![1; rows].into(),
                Some(NullBuffer::new_valid(rows)),
            )))?;
            check_copy(Arc::new(Int32Array::new_null(rows)))?;
            check_copy(Arc::new(StringArray::from(vec![Some(""); rows])))?;
            check_copy(new_null_array(&DataType::Null, rows))?;
        }
        Ok(())
    }

    #[test]
    fn repeated_views_charge_every_copied_value() -> Result<()> {
        let mut builder = StringViewBuilder::new().with_deduplicate_strings();
        let value = "x".repeat(4096);
        for _ in 0..1024 {
            builder.append_value(&value);
        }
        let array = Arc::new(builder.finish()) as ArrayRef;
        check_copy(Arc::clone(&array))?;
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "a",
                DataType::Utf8View,
                false,
            )])),
            vec![array],
        )?;
        let plan = BatchCompaction::try_new(&input)?.expect("supported view");
        assert!(plan.capacity >= 1024 * 4096);
        assert!(plan.capacity > get_record_batch_memory_size(&input));
        Ok(())
    }

    #[test]
    fn dictionary_batch_retains_its_representation() -> Result<()> {
        let dictionary: DictionaryArray<Int32Type> =
            [Some("one"), None, Some("two")].into_iter().collect();
        let array = Arc::new(dictionary) as ArrayRef;
        let input = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "a",
                array.data_type().clone(),
                true,
            )])),
            vec![array],
        )?;
        assert!(BatchCompaction::try_new(&input)?.is_none());
        Ok(())
    }
}
