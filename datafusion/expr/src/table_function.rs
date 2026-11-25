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

//! Logical application of a table function to a relation.
use crate::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use arrow::datatypes::SchemaRef;
use datafusion_common::{DFSchemaRef, Result, internal_err};
use std::{
    any::Any,
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

/// A table function bound to the static arguments of one SQL call.
pub trait TableFunctionSource: Any + fmt::Debug + Send + Sync {
    /// Output fields before aliasing, projection, and outer-join null extension.
    fn schema(&self) -> SchemaRef;
}

/// The bound implementation and expressions to evaluate on every input batch.
#[derive(Debug, Clone)]
pub struct BoundTableFunction {
    /// Function-specific binding, shared across rewrites of this call.
    pub source: Arc<dyn TableFunctionSource>,
    /// Runtime arguments in the order requested by the function.
    pub arguments: Vec<Expr>,
}

/// Applies one bound function to the rows of its child.
#[derive(Debug, Clone)]
pub struct TableFunctionApply {
    /// Registered name.
    pub function_name: String,
    /// Input relation.
    pub input: Arc<LogicalPlan>,
    /// Bound function and runtime expressions.
    pub function: BoundTableFunction,
    /// Whether to emit a null-extended row for each unmatched input.
    pub preserve_unmatched: bool,
    /// Join predicate evaluated before deciding which inputs are unmatched.
    pub predicate: Option<Expr>,
    /// Indices into the original function schema.
    pub projection: Vec<usize>,
    /// Qualified schema of the input followed by projected function fields.
    pub schema: DFSchemaRef,
}

impl TableFunctionApply {
    /// Get the bound call represented by this logical plan, if present.
    pub fn from_plan(plan: &LogicalPlan) -> Option<&Self> {
        match plan {
            LogicalPlan::Extension(ext) => ext.node.as_any().downcast_ref(),
            _ => None,
        }
    }
    /// Wrap a bound call as a logical operator.
    pub fn into_plan(self) -> LogicalPlan {
        LogicalPlan::Extension(crate::Extension {
            node: Arc::new(self),
        })
    }
}

impl PartialEq for TableFunctionApply {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.function.source, &other.function.source)
            && self.function.arguments == other.function.arguments
            && self.input == other.input
            && self.schema == other.schema
            && self.projection == other.projection
            && self.predicate == other.predicate
            && self.preserve_unmatched == other.preserve_unmatched
            && self.function_name == other.function_name
    }
}
impl Eq for TableFunctionApply {}
impl PartialOrd for TableFunctionApply {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.eq(other).then_some(std::cmp::Ordering::Equal)
    }
}
impl Hash for TableFunctionApply {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.function_name.hash(state);
        self.input.hash(state);
        self.function.arguments.hash(state);
        self.predicate.hash(state);
        self.preserve_unmatched.hash(state);
        self.projection.hash(state);
        self.schema.hash(state);
    }
}

impl UserDefinedLogicalNodeCore for TableFunctionApply {
    fn name(&self) -> &str {
        "TableFunctionApply"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }
    fn expressions(&self) -> Vec<Expr> {
        self.function
            .arguments
            .iter()
            .chain(&self.predicate)
            .cloned()
            .collect()
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TableFunctionApply: {}({}), projection={:?}",
            self.function_name,
            self.function
                .arguments
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", "),
            self.projection
        )?;
        if self.preserve_unmatched {
            write!(f, ", left")?;
        }
        if let Some(on) = &self.predicate {
            write!(f, ", on={on}")?;
        }
        Ok(())
    }
    fn with_exprs_and_inputs(
        &self,
        mut exprs: Vec<Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        let [input] = inputs.try_into().map_err(|_| {
            datafusion_common::internal_datafusion_err!(
                "Expected one table function input"
            )
        })?;
        if exprs.len()
            != self.function.arguments.len() + usize::from(self.predicate.is_some())
        {
            return internal_err!("Unexpected argument count rebuilding table function");
        }
        let predicate = if self.predicate.is_some() {
            exprs.pop()
        } else {
            None
        };
        Ok(Self {
            input: Arc::new(input),
            function: BoundTableFunction {
                source: Arc::clone(&self.function.source),
                arguments: exprs,
            },
            predicate,
            ..self.clone()
        })
    }
}
