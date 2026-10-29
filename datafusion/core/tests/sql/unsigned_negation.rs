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

use arrow::array::AsArray;
use arrow::datatypes::{DataType, Int64Type};
use datafusion::prelude::SessionContext;
use datafusion_common::{Result, assert_batches_eq};

#[tokio::test]
async fn checked_unsigned_negation_values_and_types() -> Result<()> {
    let ctx = SessionContext::new();
    let batches = ctx.sql(
        "SELECT -arrow_cast(v, 'UInt64') AS n FROM (VALUES          ('0'), ('1'), ('9223372036854775807'), ('9223372036854775808'), (NULL)) t(v)",
    ).await?.collect().await?;
    let mut actual = Vec::new();
    for batch in &batches {
        assert_eq!(batch.schema().field(0).data_type(), &DataType::Int64);
        actual.extend(batch.column(0).as_primitive::<Int64Type>().iter());
    }
    assert_eq!(
        actual,
        vec![Some(0), Some(-1), Some(-i64::MAX), Some(i64::MIN), None]
    );
    let batches = ctx.sql(
        "SELECT arrow_typeof(rank() OVER ()) AS r,          arrow_typeof(-rank() OVER ()) AS nr,          arrow_typeof(-arrow_cast('255', 'UInt8')) AS u8,          arrow_typeof(-arrow_cast('65535', 'UInt16')) AS u16,          arrow_typeof(-arrow_cast('4294967295', 'UInt32')) AS u32",
    ).await?.collect().await?;
    assert_batches_eq!(
        [
            "+--------+-------+-------+-------+-------+",
            "| r      | nr    | u8    | u16   | u32   |",
            "+--------+-------+-------+-------+-------+",
            "| UInt64 | Int64 | Int16 | Int32 | Int64 |",
            "+--------+-------+-------+-------+-------+",
        ],
        &batches
    );
    Ok(())
}

#[tokio::test]
async fn checked_unsigned_negation_rejects_overflow() -> Result<()> {
    let ctx = SessionContext::new();
    for sql in [
        "SELECT -arrow_cast(v, 'UInt64') < 0 FROM (VALUES ('1'), ('9223372036854775809')) t(v)",
        "SELECT -arrow_cast(v, 'UInt64') IN (-1, -2) FROM (VALUES ('1'), ('9223372036854775809')) t(v)",
        "SELECT -(-arrow_cast(v, 'Int64')) FROM (VALUES ('1'), ('-9223372036854775808')) t(v)",
        "SELECT -arrow_cast('9223372036854775809', 'UInt64')",
        "SELECT -arrow_cast('18446744073709551615', 'UInt64')",
        "SELECT -arrow_cast(v, 'UInt64') FROM (VALUES ('1'), ('9223372036854775809')) t(v)",
        "SELECT -arrow_cast(v, 'UInt64') FROM (VALUES ('1'), ('18446744073709551615')) t(v)",
        "SELECT -(-arrow_cast(v, 'UInt64')) FROM (VALUES ('1'), ('9223372036854775808')) t(v)",
        "SELECT -arrow_cast(v, 'Int64') FROM (VALUES ('1'), ('-9223372036854775808')) t(v)",
    ] {
        let result = async { ctx.sql(sql).await?.collect().await }.await;
        assert!(result.is_err(), "expected overflow from {sql}: {result:?}");
    }
    Ok(())
}
