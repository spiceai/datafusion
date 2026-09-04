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

use crate::logical_plan::consumer::SubstraitConsumer;
use datafusion::common::{DFSchema, ScalarValue, not_impl_err};
use datafusion::logical_expr::Expr;
use substrait::proto::FunctionArgument;
use substrait::proto::function_argument::ArgType;

/// Convert Substrait FunctionArguments to DataFusion Exprs
///
/// `ArgType::Enum` is lowered to a Utf8 literal. Isthmus emits this form for
/// option-typed parameters such as `extract`'s component (`YEAR`, `MONTH`, …).
/// DataFusion's `date_part` (and similar UDFs) take that option as a string.
pub async fn from_substrait_func_args(
    consumer: &impl SubstraitConsumer,
    arguments: &Vec<FunctionArgument>,
    input_schema: &DFSchema,
) -> datafusion::common::Result<Vec<Expr>> {
    let mut args: Vec<Expr> = vec![];
    for arg in arguments {
        let arg_expr = match &arg.arg_type {
            Some(ArgType::Value(e)) => consumer.consume_expression(e, input_schema).await,
            Some(ArgType::Enum(name)) => {
                Ok(Expr::Literal(ScalarValue::Utf8(Some(name.clone())), None))
            }
            Some(ArgType::Type(_)) => {
                not_impl_err!("Function argument Type not supported")
            }
            None => not_impl_err!("Function argument missing arg_type"),
        };
        args.push(arg_expr?);
    }
    Ok(args)
}
