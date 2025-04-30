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

use std::any::Any;
use std::sync::Arc;

use arrow::array::types::IntervalMonthDayNanoType;
use arrow::array::{
    Date32Array, Int32Array, IntervalMonthDayNanoArray, StringArray,
    Time64NanosecondArray, TimestampMillisecondArray,
};
use arrow::compute::kernels::cast_utils::string_to_timestamp_nanos;
use arrow::datatypes::{DataType, IntervalUnit, TimeUnit};
use arrow::temporal_conversions::{MILLISECONDS_IN_DAY, NANOSECONDS_IN_DAY};
use chrono::{DateTime, Months, Utc};
use datafusion_common::{exec_err, plan_err, DataFusionError, Result, ScalarValue};
use datafusion_expr::sort_properties::{ExprProperties, SortProperties};
use datafusion_expr::TypeSignature::Exact;
use datafusion_expr::{
    ColumnarValue, Documentation, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

const NANOS_PER_MILLI: i64 = 1_000_000;

// Helper function to add months to a timestamp (in nanoseconds since epoch)
fn add_months_to_timestamp_nanos(timestamp_nanos: i64, months: i32) -> Option<i64> {
    let datetime = DateTime::<Utc>::from_timestamp_nanos(timestamp_nanos);
    let new_datetime = if months >= 0 {
        datetime.checked_add_months(Months::new(months as u32))
    } else {
        datetime.checked_sub_months(Months::new((-months) as u32))
    }?;
    Some(new_datetime.timestamp_nanos_opt()?)
}

// Helper function to add months to a Date32 (days since epoch)
fn add_months_to_date32(date_days: i32, months: i32) -> Option<i64> {
    let timestamp_nanos = (date_days as i64) * NANOSECONDS_IN_DAY;
    add_months_to_timestamp_nanos(timestamp_nanos, months)
}

#[user_doc(
    doc_section(label = "Time and Date Functions"),
    description = r#"
Adds a specified interval to a date, timestamp, or time value. The function supports adding integer days to dates or intervals (months, days, or nanoseconds) to dates, timestamps, or times. The result type depends on the input: dates with integer days return dates, while intervals typically return timestamps, and times return times.

For string timestamp inputs, the time component is reset to 00:00:00 in the result, while timestamp types preserve the time component. Month intervals are supported for date and timestamp inputs, adjusting the month component accordingly.
"#,
    syntax_example = "date_add(expression, interval)",
    sql_example = r#"```sql
-- Add 2 days to a date string
SELECT date_add('2022-01-01', 2);
-- 2022-01-03

-- Add 30 days to a date
SELECT date_add(DATE '2022-01-01', 30);
-- 2022-01-31

-- Add 2 day interval to a date string
SELECT date_add('2022-01-01', CAST(2 AS INTERVAL DAY));
-- 2022-01-03 00:00:00.000

-- Add 1 month interval to a date
SELECT date_add(DATE '2022-01-31', CAST(1 AS INTERVAL MONTH));
-- 2022-02-28 00:00:00.000

-- Add 30 day interval to a timestamp
SELECT date_add(TIMESTAMP '2022-01-01 12:00:00', CAST(30 AS INTERVAL DAY));
-- 2022-01-31 12:00:00.000

-- Add 1 month interval to a timestamp
SELECT date_add(TIMESTAMP '2022-01-31 12:00:00', CAST(1 AS INTERVAL MONTH));
-- 2022-02-28 12:00:00.000

-- Add 30 minute interval to a time
SELECT date_add(TIME '00:00:00', CAST(30 AS INTERVAL MINUTE));
-- 00:30:00.000
```"#,
    argument(
        name = "expression",
        description = "Date, timestamp, or time expression to operate on. Can be a string, constant, column, or function."
    ),
    argument(
        name = "interval",
        description = "Number of days (integer) or interval (MONTH, DAY, or nanoseconds) to add. Negative values subtract the interval."
    )
)]
#[derive(Debug)]
pub struct DateAddFunc {
    signature: Signature,
}

impl Default for DateAddFunc {
    fn default() -> Self {
        Self::new()
    }
}

impl DateAddFunc {
    pub fn new() -> Self {
        let signature = Signature::one_of(
            vec![
                // DATE_ADD(string, int32) -> Date32
                Exact(vec![DataType::Utf8, DataType::Int32]),
                // DATE_ADD(Date32, int32) -> Date32
                Exact(vec![DataType::Date32, DataType::Int32]),
                // DATE_ADD(string, IntervalMonthDayNano) -> TimestampMillisecond
                Exact(vec![
                    DataType::Utf8,
                    DataType::Interval(IntervalUnit::MonthDayNano),
                ]),
                // DATE_ADD(Date32, IntervalMonthDayNano) -> TimestampMillisecond
                Exact(vec![
                    DataType::Date32,
                    DataType::Interval(IntervalUnit::MonthDayNano),
                ]),
                // DATE_ADD(TimestampMillisecond, IntervalMonthDayNano) -> TimestampMillisecond
                Exact(vec![
                    DataType::Timestamp(TimeUnit::Millisecond, None),
                    DataType::Interval(IntervalUnit::MonthDayNano),
                ]),
                // DATE_ADD(Time64(Nanosecond), IntervalMonthDayNano) -> Time64(Nanosecond)
                Exact(vec![
                    DataType::Time64(TimeUnit::Nanosecond),
                    DataType::Interval(IntervalUnit::MonthDayNano),
                ]),
            ],
            Volatility::Immutable,
        );

        Self { signature }
    }
}

impl ScalarUDFImpl for DateAddFunc {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "date_add"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        match (&arg_types[0], &arg_types[1]) {
            (DataType::Utf8, DataType::Int32) => Ok(DataType::Date32),
            (DataType::Date32, DataType::Int32) => Ok(DataType::Date32),
            (DataType::Utf8, DataType::Interval(IntervalUnit::MonthDayNano)) => {
                Ok(DataType::Timestamp(TimeUnit::Millisecond, None))
            }
            (DataType::Date32, DataType::Interval(IntervalUnit::MonthDayNano)) => {
                Ok(DataType::Timestamp(TimeUnit::Millisecond, None))
            }
            (
                DataType::Timestamp(TimeUnit::Millisecond, _),
                DataType::Interval(IntervalUnit::MonthDayNano),
            ) => Ok(DataType::Timestamp(TimeUnit::Millisecond, None)),
            (
                DataType::Time64(TimeUnit::Nanosecond),
                DataType::Interval(IntervalUnit::MonthDayNano),
            ) => Ok(DataType::Time64(TimeUnit::Nanosecond)),
            _ => plan_err!("Unsupported argument types for date_add: {:?}", arg_types),
        }
    }

    fn invoke_batch(
        &self,
        args: &[ColumnarValue],
        _number_rows: usize,
    ) -> Result<ColumnarValue> {
        if args.len() != 2 {
            return exec_err!("date_add expects exactly two arguments");
        }

        date_add_impl(&args[0], &args[1])
    }

    fn output_ordering(&self, input: &[ExprProperties]) -> Result<SortProperties> {
        Ok(input[0].sort_properties)
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

fn date_add_impl(
    value: &ColumnarValue,
    interval: &ColumnarValue,
) -> Result<ColumnarValue> {
    match (value, interval) {
        // DATE_ADD(string, int32) -> Date32
        (ColumnarValue::Array(arr1), ColumnarValue::Array(arr2))
            if arr1.data_type() == &DataType::Utf8
                && arr2.data_type() == &DataType::Int32 =>
        {
            let strings = arr1.as_any().downcast_ref::<StringArray>().unwrap();
            let days = arr2.as_any().downcast_ref::<Int32Array>().unwrap();
            let result: Date32Array = strings
                .iter()
                .zip(days.iter())
                .map(|(s, days)| {
                    s.and_then(|s| {
                        days.and_then(|n| {
                            let nanos = string_to_timestamp_nanos(s).ok()?;
                            let date_days = (nanos / NANOSECONDS_IN_DAY) as i32;
                            Some(date_days + n)
                        })
                    })
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
        // DATE_ADD(Date32, int32) -> Date32
        (ColumnarValue::Array(arr1), ColumnarValue::Array(arr2))
            if arr1.data_type() == &DataType::Date32
                && arr2.data_type() == &DataType::Int32 =>
        {
            let dates = arr1.as_any().downcast_ref::<Date32Array>().unwrap();
            let days = arr2.as_any().downcast_ref::<Int32Array>().unwrap();
            let result: Date32Array = dates
                .iter()
                .zip(days.iter())
                .map(|(date, days)| date.and_then(|d| days.map(|n| d + n)))
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
        // DATE_ADD(string, IntervalMonthDayNano) -> TimestampMillisecond
        (ColumnarValue::Array(arr1), ColumnarValue::Array(arr2))
            if arr1.data_type() == &DataType::Utf8
                && arr2.data_type()
                    == &DataType::Interval(IntervalUnit::MonthDayNano) =>
        {
            let strings = arr1.as_any().downcast_ref::<StringArray>().unwrap();
            let intervals = arr2
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap();
            let result: TimestampMillisecondArray = strings
                .iter()
                .zip(intervals.iter())
                .map(|(s, interval)| {
                    s.and_then(|s| {
                        interval.and_then(|i| {
                            let timestamp_nanos = string_to_timestamp_nanos(s).ok()?;
                            let (interval_months, interval_days, interval_nanos) =
                                IntervalMonthDayNanoType::to_parts(i);
                            // Handle months
                            let base_timestamp_nanos = if interval_months != 0 {
                                add_months_to_timestamp_nanos(
                                    timestamp_nanos,
                                    interval_months,
                                )?
                            } else {
                                timestamp_nanos
                            };
                            // Truncate to day boundary (reset time to 00:00:00)
                            let timestamp_ms = (base_timestamp_nanos
                                / NANOSECONDS_IN_DAY)
                                * NANOSECONDS_IN_DAY
                                / NANOS_PER_MILLI;
                            let interval_ms = (interval_days as i64
                                * MILLISECONDS_IN_DAY)
                                + (interval_nanos / NANOS_PER_MILLI);
                            Some(timestamp_ms + interval_ms)
                        })
                    })
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
        // DATE_ADD(Date32, IntervalMonthDayNano) -> TimestampMillisecond
        (ColumnarValue::Array(arr1), ColumnarValue::Array(arr2))
            if arr1.data_type() == &DataType::Date32
                && arr2.data_type()
                    == &DataType::Interval(IntervalUnit::MonthDayNano) =>
        {
            let dates = arr1.as_any().downcast_ref::<Date32Array>().unwrap();
            let intervals = arr2
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap();
            let result: TimestampMillisecondArray = dates
                .iter()
                .zip(intervals.iter())
                .map(|(date, interval)| {
                    date.and_then(|d| {
                        interval.and_then(|i| {
                            let (months, days, nanos) =
                                IntervalMonthDayNanoType::to_parts(i);
                            let base_timestamp_ms = if months != 0 {
                                add_months_to_date32(d, months)? / NANOS_PER_MILLI
                            } else {
                                (d as i64) * MILLISECONDS_IN_DAY
                            };
                            let interval_ms = (days as i64 * MILLISECONDS_IN_DAY)
                                + (nanos / NANOS_PER_MILLI);
                            Some(base_timestamp_ms + interval_ms)
                        })
                    })
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
        // DATE_ADD(TimestampMillisecond, IntervalMonthDayNano) -> TimestampMillisecond
        (ColumnarValue::Array(arr1), ColumnarValue::Array(arr2))
            if arr1.data_type() == &DataType::Timestamp(TimeUnit::Millisecond, None)
                && arr2.data_type()
                    == &DataType::Interval(IntervalUnit::MonthDayNano) =>
        {
            let timestamps = arr1
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap();
            let intervals = arr2
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap();
            let result: TimestampMillisecondArray = timestamps
                .iter()
                .zip(intervals.iter())
                .map(|(ts, interval)| {
                    ts.and_then(|t| {
                        interval.and_then(|i| {
                            let (months, days, nanos) =
                                IntervalMonthDayNanoType::to_parts(i);
                            let base_timestamp_ms = if months != 0 {
                                let timestamp_nanos = t * NANOS_PER_MILLI;
                                add_months_to_timestamp_nanos(timestamp_nanos, months)?
                                    / NANOS_PER_MILLI
                            } else {
                                t
                            };
                            let interval_ms = (days as i64 * MILLISECONDS_IN_DAY)
                                + (nanos / NANOS_PER_MILLI);
                            Some(base_timestamp_ms + interval_ms)
                        })
                    })
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
        // DATE_ADD(Time64(Nanosecond), IntervalMonthDayNano) -> Time64(Nanosecond)
        (ColumnarValue::Array(arr1), ColumnarValue::Array(arr2))
            if arr1.data_type() == &DataType::Time64(TimeUnit::Nanosecond)
                && arr2.data_type()
                    == &DataType::Interval(IntervalUnit::MonthDayNano) =>
        {
            let times = arr1
                .as_any()
                .downcast_ref::<Time64NanosecondArray>()
                .unwrap();
            let intervals = arr2
                .as_any()
                .downcast_ref::<IntervalMonthDayNanoArray>()
                .unwrap();
            let result: Time64NanosecondArray = times
                .iter()
                .zip(intervals.iter())
                .map(|(time, interval)| {
                    time.and_then(|t| {
                        interval.and_then(|i| {
                            let (months, days, nanos) =
                                IntervalMonthDayNanoType::to_parts(i);
                            if months != 0 {
                                return None; // Month intervals not supported for time
                            }
                            let total_nanos =
                                t + (days as i64 * NANOSECONDS_IN_DAY) + nanos;
                            let wrapped = total_nanos % NANOSECONDS_IN_DAY;
                            Some(if wrapped < 0 {
                                wrapped + NANOSECONDS_IN_DAY
                            } else {
                                wrapped
                            })
                        })
                    })
                })
                .collect();
            Ok(ColumnarValue::Array(Arc::new(result)))
        }
        // Scalar cases
        (ColumnarValue::Scalar(scalar1), ColumnarValue::Scalar(scalar2)) => {
            match (scalar1, scalar2) {
                (ScalarValue::Utf8(Some(s)), ScalarValue::Int32(Some(days))) => {
                    let nanos = string_to_timestamp_nanos(s).map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to parse timestamp: {}",
                            e
                        ))
                    })?;
                    let date_days = (nanos / NANOSECONDS_IN_DAY) as i32;
                    Ok(ColumnarValue::Scalar(ScalarValue::Date32(Some(
                        date_days + days,
                    ))))
                }
                (ScalarValue::Date32(Some(date)), ScalarValue::Int32(Some(days))) => Ok(
                    ColumnarValue::Scalar(ScalarValue::Date32(Some(date + days))),
                ),
                (
                    ScalarValue::Utf8(Some(s)),
                    ScalarValue::IntervalMonthDayNano(Some(interval)),
                ) => {
                    let nanos = string_to_timestamp_nanos(s).map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to parse timestamp: {}",
                            e
                        ))
                    })?;
                    let (months, days, interval_nanos) =
                        IntervalMonthDayNanoType::to_parts(*interval);
                    let base_timestamp_nanos = if months != 0 {
                        add_months_to_timestamp_nanos(nanos, months).ok_or_else(|| {
                            DataFusionError::Execution(
                                "Failed to add months to timestamp".to_string(),
                            )
                        })?
                    } else {
                        nanos
                    };
                    let timestamp_ms = (base_timestamp_nanos / NANOSECONDS_IN_DAY)
                        * NANOSECONDS_IN_DAY
                        / NANOS_PER_MILLI;
                    let interval_ms = (days as i64 * MILLISECONDS_IN_DAY)
                        + (interval_nanos / NANOS_PER_MILLI);
                    Ok(ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(
                        Some(timestamp_ms + interval_ms),
                        None,
                    )))
                }
                (
                    ScalarValue::Date32(Some(date)),
                    ScalarValue::IntervalMonthDayNano(Some(interval)),
                ) => {
                    let (months, days, nanos) =
                        IntervalMonthDayNanoType::to_parts(*interval);
                    let base_timestamp_ms = if months != 0 {
                        add_months_to_date32(*date, months).ok_or_else(|| {
                            DataFusionError::Execution(
                                "Failed to add months to date".to_string(),
                            )
                        })? / NANOS_PER_MILLI
                    } else {
                        (*date as i64) * MILLISECONDS_IN_DAY
                    };
                    let interval_ms =
                        (days as i64 * MILLISECONDS_IN_DAY) + (nanos / NANOS_PER_MILLI);
                    Ok(ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(
                        Some(base_timestamp_ms + interval_ms),
                        None,
                    )))
                }
                (
                    ScalarValue::TimestampMillisecond(Some(ts), _),
                    ScalarValue::IntervalMonthDayNano(Some(interval)),
                ) => {
                    let (months, days, nanos) =
                        IntervalMonthDayNanoType::to_parts(*interval);
                    let base_timestamp_ms = if months != 0 {
                        let timestamp_nanos = *ts * NANOS_PER_MILLI;
                        add_months_to_timestamp_nanos(timestamp_nanos, months)
                            .ok_or_else(|| {
                                DataFusionError::Execution(
                                    "Failed to add months to timestamp".to_string(),
                                )
                            })?
                            / NANOS_PER_MILLI
                    } else {
                        *ts
                    };
                    let interval_ms =
                        (days as i64 * MILLISECONDS_IN_DAY) + (nanos / NANOS_PER_MILLI);
                    Ok(ColumnarValue::Scalar(ScalarValue::TimestampMillisecond(
                        Some(base_timestamp_ms + interval_ms),
                        None,
                    )))
                }
                (
                    ScalarValue::Time64Nanosecond(Some(time)),
                    ScalarValue::IntervalMonthDayNano(Some(interval)),
                ) => {
                    let (months, days, nanos) =
                        IntervalMonthDayNanoType::to_parts(*interval);
                    if months != 0 {
                        return exec_err!(
                            "Month intervals not supported for time in date_add"
                        );
                    }
                    let total_nanos = *time + (days as i64 * NANOSECONDS_IN_DAY) + nanos;
                    let wrapped = total_nanos % NANOSECONDS_IN_DAY;
                    let final_time = if wrapped < 0 {
                        wrapped + NANOSECONDS_IN_DAY
                    } else {
                        wrapped
                    };
                    Ok(ColumnarValue::Scalar(ScalarValue::Time64Nanosecond(Some(
                        final_time,
                    ))))
                }
                _ => exec_err!(
                    "Unsupported scalar arguments for date_add: {:?}",
                    (scalar1, scalar2)
                ),
            }
        }
        _ => exec_err!(
            "Unsupported argument types for date_add: {:?}",
            (value.data_type(), interval.data_type())
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        ArrayRef, Date32Array, Int32Array, IntervalMonthDayNanoArray, StringArray,
        Time64NanosecondArray, TimestampMillisecondArray,
    };
    use arrow::datatypes::{DataType, TimeUnit};
    use arrow::temporal_conversions::NANOSECONDS;
    use arrow_buffer::IntervalMonthDayNano;

    fn make_interval_month_day_nano(
        months: i32,
        days: i32,
        nanos: i64,
    ) -> IntervalMonthDayNano {
        IntervalMonthDayNanoType::make_value(months, days, nanos)
    }

    fn run_test_case(
        value: ColumnarValue,
        interval: ColumnarValue,
        expected_type: DataType,
        expected_array: ArrayRef,
    ) {
        let result = date_add_impl(&value, &interval).unwrap();
        match result {
            ColumnarValue::Array(result_array) => {
                assert_eq!(result_array.data_type(), &expected_type);
                assert_eq!(result_array.as_ref(), expected_array.as_ref());
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_date_add_string_2_days() {
        // DATE_ADD('2022-01-01', 2) -> 2022-01-03
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-01-01"]))),
            ColumnarValue::Array(Arc::new(Int32Array::from(vec![2]))),
            DataType::Date32,
            Arc::new(Date32Array::from(vec![Some(18995)])) as ArrayRef, // 2022-01-03
        );
    }

    #[test]
    fn test_date_add_string_minus_2_days() {
        // DATE_ADD('2022-01-01', -2) -> 2021-12-30
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-01-01"]))),
            ColumnarValue::Array(Arc::new(Int32Array::from(vec![-2]))),
            DataType::Date32,
            Arc::new(Date32Array::from(vec![Some(18991)])) as ArrayRef, // 2021-12-30
        );
    }

    #[test]
    fn test_date_add_date_30_days() {
        // DATE_ADD(DATE '2022-01-01', 30) -> 2022-01-31
        run_test_case(
            ColumnarValue::Array(Arc::new(Date32Array::from(vec![Some(18993)]))), // 2022-01-01
            ColumnarValue::Array(Arc::new(Int32Array::from(vec![30]))),
            DataType::Date32,
            Arc::new(Date32Array::from(vec![Some(19023)])) as ArrayRef, // 2022-01-31
        );
    }

    #[test]
    fn test_date_add_date_minus_30_days() {
        // DATE_ADD(DATE '2022-01-01', -30) -> 2021-12-02
        run_test_case(
            ColumnarValue::Array(Arc::new(Date32Array::from(vec![Some(18993)]))), // 2022-01-01
            ColumnarValue::Array(Arc::new(Int32Array::from(vec![-30]))),
            DataType::Date32,
            Arc::new(Date32Array::from(vec![Some(18963)])) as ArrayRef, // 2021-12-02
        );
    }

    #[test]
    fn test_date_add_string_2_day_interval() {
        // DATE_ADD('2022-01-01', CAST(2 AS INTERVAL DAY)) -> 2022-01-03 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-01-01"]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, 2, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1641168000000)]))
                as ArrayRef, // 2022-01-03 00:00:00
        );
    }

    #[test]
    fn test_date_add_string_minus_2_day_interval() {
        // DATE_ADD('2022-01-01', CAST(-2 AS INTERVAL DAY)) -> 2021-12-30 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-01-01"]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, -2, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1640822400000)]))
                as ArrayRef, // 2021-12-30 00:00:00
        );
    }

    #[test]
    fn test_date_add_date_30_day_interval() {
        // DATE_ADD(DATE '2022-01-01', CAST(30 AS INTERVAL DAY)) -> 2022-01-31 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(Date32Array::from(vec![Some(18993)]))), // 2022-01-01
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, 30, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1643587200000)]))
                as ArrayRef, // 2022-01-31 00:00:00
        );
    }

    #[test]
    fn test_date_add_timestamp_string_30_day_interval() {
        // DATE_ADD('2022-01-01 12:00:00', CAST(30 AS INTERVAL DAY)) -> 2022-01-31 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec![
                "2022-01-01 12:00:00",
            ]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, 30, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1643587200000)]))
                as ArrayRef, // 2022-01-31 00:00:00
        );
    }

    #[test]
    fn test_date_add_timestamp_string_minus_30_day_interval() {
        // DATE_ADD('2022-01-01 12:00:00', CAST(-30 AS INTERVAL DAY)) -> 2021-12-02 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec![
                "2022-01-01 12:00:00",
            ]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, -30, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1638403200000)]))
                as ArrayRef, // 2021-12-02 00:00:00
        );
    }

    #[test]
    fn test_date_add_timestamp_30_day_interval() {
        // DATE_ADD(TIMESTAMP '2022-01-01 12:00:00', CAST(30 AS INTERVAL DAY)) -> 2022-01-31 12:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(TimestampMillisecondArray::from(vec![Some(
                1641038400000,
            )]))), // 2022-01-01 12:00:00
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, 30, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1643630400000)]))
                as ArrayRef, // 2022-01-31 12:00:00
        );
    }

    #[test]
    fn test_date_add_timestamp_minus_30_day_interval() {
        // DATE_ADD(TIMESTAMP '2022-01-01 12:00:00', CAST(-30 AS INTERVAL DAY)) -> 2021-12-02 12:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(TimestampMillisecondArray::from(vec![Some(
                1641038400000,
            )]))), // 2022-01-01 12:00:00
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, -30, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1638446400000)]))
                as ArrayRef, // 2021-12-02 12:00:00
        );
    }

    #[test]
    fn test_date_add_time_30_minute_interval() {
        // DATE_ADD(TIME '00:00:00', CAST(30 AS INTERVAL MINUTE)) -> 00:30:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(Time64NanosecondArray::from(vec![Some(0)]))), // 00:00:00
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, 0, 30 * 60 * NANOSECONDS),
            )]))), // 30 minutes
            DataType::Time64(TimeUnit::Nanosecond),
            Arc::new(Time64NanosecondArray::from(vec![Some(
                30 * 60 * NANOSECONDS,
            )])) as ArrayRef, // 00:30:00
        );
    }

    #[test]
    fn test_date_add_time_minus_30_minute_interval() {
        // DATE_ADD(TIME '00:00:00', CAST(-30 AS INTERVAL MINUTE)) -> 23:30:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(Time64NanosecondArray::from(vec![Some(0)]))), // 00:00:00
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(0, 0, -30 * 60 * NANOSECONDS),
            )]))), // -30 minutes
            DataType::Time64(TimeUnit::Nanosecond),
            Arc::new(Time64NanosecondArray::from(vec![Some(
                (23 * 3600 + 30 * 60) * NANOSECONDS,
            )])) as ArrayRef, // 23:30:00
        );
    }

    // New tests for month intervals
    #[test]
    fn test_date_add_string_1_month_interval() {
        // DATE_ADD('2022-01-01', CAST(1 AS INTERVAL MONTH)) -> 2022-02-01 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-01-01"]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(1, 0, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1643673600000)]))
                as ArrayRef, // 2022-02-01 00:00:00
        );
    }

    #[test]
    fn test_date_add_string_minus_1_month_interval() {
        // DATE_ADD('2022-02-01', CAST(-1 AS INTERVAL MONTH)) -> 2022-01-01 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-02-01"]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(-1, 0, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1640995200000)]))
                as ArrayRef, // 2022-01-01 00:00:00
        );
    }

    #[test]
    fn test_date_add_date_1_month_interval() {
        // DATE_ADD(DATE '2022-01-31', CAST(1 AS INTERVAL MONTH)) -> 2022-02-28 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(Date32Array::from(vec![Some(19023)]))), // 2022-01-31
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(1, 0, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1646006400000)]))
                as ArrayRef, // 2022-02-28 00:00:00
        );
    }

    #[test]
    fn test_date_add_date_minus_1_month_interval() {
        // DATE_ADD(DATE '2022-02-28', CAST(-1 AS INTERVAL MONTH)) -> 2022-01-28 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(Date32Array::from(vec![Some(19051)]))), // 2022-02-28
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(-1, 0, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1643328000000)]))
                as ArrayRef, // 2022-01-28 00:00:00
        );
    }

    #[test]
    fn test_date_add_timestamp_1_month_interval() {
        // DATE_ADD(TIMESTAMP '2022-01-31 12:00:00', CAST(1 AS INTERVAL MONTH)) -> 2022-02-28 12:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(TimestampMillisecondArray::from(vec![Some(
                1643630400000,
            )]))), // 2022-01-31 12:00:00
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(1, 0, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1646049600000)]))
                as ArrayRef, // 2022-02-28 12:00:00
        );
    }

    #[test]
    fn test_date_add_timestamp_minus_1_month_interval() {
        // DATE_ADD(TIMESTAMP '2022-02-28 12:00:00', CAST(-1 AS INTERVAL MONTH)) -> 2022-01-28 12:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(TimestampMillisecondArray::from(vec![Some(
                1646049600000,
            )]))), // 2022-02-28 12:00:00
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(-1, 0, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1643371200000)]))
                as ArrayRef, // 2022-01-28 12:00:00
        );
    }

    #[test]
    fn test_date_add_string_1_month_1_day_interval() {
        // DATE_ADD('2022-01-31', CAST(1 AS INTERVAL MONTH) + CAST(1 AS INTERVAL DAY)) -> 2022-03-01 00:00:00.000
        run_test_case(
            ColumnarValue::Array(Arc::new(StringArray::from(vec!["2022-01-31"]))),
            ColumnarValue::Array(Arc::new(IntervalMonthDayNanoArray::from(vec![Some(
                make_interval_month_day_nano(1, 1, 0),
            )]))),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            Arc::new(TimestampMillisecondArray::from(vec![Some(1646092800000)]))
                as ArrayRef, // 2022-03-01 00:00:00
        );
    }

    #[test]
    fn test_invalid_arguments() {
        let cases = vec![
            // Wrong number of arguments
            vec![ColumnarValue::Scalar(ScalarValue::Utf8(Some(
                "2022-01-01".to_string(),
            )))],
            // Unsupported first argument type
            vec![
                ColumnarValue::Scalar(ScalarValue::Int32(Some(1))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(2))),
            ],
            // Unsupported second argument type
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("2022-01-01".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("2".to_string()))),
            ],
        ];

        for args in cases {
            let result = DateAddFunc::new().invoke_batch(&args, 1);
            assert!(result.is_err());
        }
    }
}
