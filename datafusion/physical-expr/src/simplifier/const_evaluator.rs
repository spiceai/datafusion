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

//! Constant expression evaluation for the physical expression simplifier

use std::sync::{Arc, OnceLock};

use arrow::array::new_null_array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, ScalarValue};
use datafusion_expr_common::columnar_value::ColumnarValue;
use datafusion_physical_expr_common::physical_expr::is_volatile;

use crate::PhysicalExpr;
use crate::expressions::{
    BinaryExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NegativeExpr,
    NotExpr, TryCastExpr,
};

/// Simplify expressions that consist only of literals by evaluating them.
///
/// This function checks if all children of the given expression are literals.
/// If so, it evaluates the expression against a dummy RecordBatch and returns
/// the result as a new Literal.
///
/// # Example transformations
/// - `1 + 2` -> `3`
/// - `(1 + 2) * 3` -> `9` (with bottom-up traversal)
/// - `'hello' || ' world'` -> `'hello world'`
pub fn simplify_const_expr(
    expr: &Arc<dyn PhysicalExpr>,
) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
    if is_volatile(expr) || has_column_references(expr) {
        return Ok(Transformed::no(Arc::clone(expr)));
    }

    // Create a 1-row dummy batch for evaluation
    let batch = create_dummy_batch()?;

    // Evaluate the expression
    match expr.evaluate(&batch) {
        Ok(ColumnarValue::Scalar(scalar)) => {
            Ok(Transformed::yes(Arc::new(Literal::new(scalar))))
        }
        Ok(ColumnarValue::Array(arr)) if arr.len() == 1 => {
            // Some operations return an array even for scalar inputs
            let scalar = ScalarValue::try_from_array(&arr, 0)?;
            Ok(Transformed::yes(Arc::new(Literal::new(scalar))))
        }
        Ok(_) => {
            // Unexpected result - keep original expression
            Ok(Transformed::no(Arc::clone(expr)))
        }
        Err(_) => {
            // On error, keep original expression
            // The expression might succeed at runtime due to short-circuit evaluation
            // or other runtime conditions
            Ok(Transformed::no(Arc::clone(expr)))
        }
    }
}

/// Simplify expressions whose immediate children are all literals.
///
/// This function only checks the direct children of the expression,
/// not the entire subtree. It is designed to be used with bottom-up tree
/// traversal, where children are simplified before parents.
///
/// # Example transformations
/// - `1 + 2` -> `3`
/// - `(1 + 2) * 3` -> `9` (with bottom-up traversal, inner expr simplified first)
/// - `'hello' || ' world'` -> `'hello world'`
pub(crate) fn simplify_const_expr_immediate(
    expr: Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
    // Already a literal - nothing to do
    if expr.as_any().is::<Literal>() {
        return Ok(Transformed::no(expr));
    }

    // Column references cannot be evaluated at plan time
    if expr.as_any().is::<Column>() {
        return Ok(Transformed::no(expr));
    }

    // Volatile nodes cannot be evaluated at plan time
    if expr.is_volatile_node() {
        return Ok(Transformed::no(expr));
    }

    // Since transform visits bottom-up, children have already been simplified.
    // If all children are now Literals, this node can be const-evaluated.
    // This is O(k) where k = number of children, instead of O(subtree).
    if !children_are_all_literals(&expr) {
        return Ok(Transformed::no(expr));
    }

    // Evaluate the expression
    match expr.evaluate(batch) {
        Ok(ColumnarValue::Scalar(scalar)) => {
            Ok(Transformed::yes(Arc::new(Literal::new(scalar))))
        }
        Ok(ColumnarValue::Array(arr)) if arr.len() == 1 => {
            // Some operations return an array even for scalar inputs
            let scalar = ScalarValue::try_from_array(&arr, 0)?;
            Ok(Transformed::yes(Arc::new(Literal::new(scalar))))
        }
        Ok(_) => {
            // Unexpected result - keep original expression
            Ok(Transformed::no(expr))
        }
        Err(_) => {
            // On error, keep original expression
            // The expression might succeed at runtime due to short-circuit evaluation
            // or other runtime conditions
            Ok(Transformed::no(expr))
        }
    }
}

/// Returns `true` if every child of `expr` is a [`Literal`].
///
/// Semantically equivalent to
/// `expr.children().iter().all(|c| c.as_any().is::<Literal>())` (including the
/// `true` result for childless nodes), but avoids the per-node `Vec`
/// allocation that [`PhysicalExpr::children`] performs by reading the fields of
/// the common fixed-arity expression types directly. This check runs for every
/// non-leaf node on every simplifier pass, so that allocation is pure overhead
/// on the pruning hot path. Variable-arity and rarer node types fall back to
/// the allocating accessor.
fn children_are_all_literals(expr: &Arc<dyn PhysicalExpr>) -> bool {
    fn is_literal(expr: &Arc<dyn PhysicalExpr>) -> bool {
        expr.as_any().is::<Literal>()
    }

    let any = expr.as_any();
    if let Some(binary) = any.downcast_ref::<BinaryExpr>() {
        is_literal(binary.left()) && is_literal(binary.right())
    } else if let Some(cast) = any.downcast_ref::<CastExpr>() {
        is_literal(cast.expr())
    } else if let Some(try_cast) = any.downcast_ref::<TryCastExpr>() {
        is_literal(try_cast.expr())
    } else if let Some(not) = any.downcast_ref::<NotExpr>() {
        is_literal(not.arg())
    } else if let Some(negative) = any.downcast_ref::<NegativeExpr>() {
        is_literal(negative.arg())
    } else if let Some(is_null) = any.downcast_ref::<IsNullExpr>() {
        is_literal(is_null.arg())
    } else if let Some(is_not_null) = any.downcast_ref::<IsNotNullExpr>() {
        is_literal(is_not_null.arg())
    } else {
        // Variable-arity or rarer node types (CASE, scalar functions, IN-list,
        // ...): fall back to the allocating accessor. `all` is `true` for nodes
        // with no children, matching the original behaviour.
        expr.children().iter().all(|child| is_literal(child))
    }
}

/// Return the shared 1-row dummy [`RecordBatch`] used to evaluate constant
/// expressions.
///
/// The batch is identical and immutable across every call, so it is built once
/// and memoized. The simplifier invokes this on every `simplify` call (often
/// hundreds of thousands of times when opening files for pruning), and building
/// the schema, null array, and batch each time was pure allocation overhead on
/// that hot path.
///
/// # Errors
///
/// Returns an error if the (constant) dummy batch cannot be constructed, which
/// should never happen for a single null column.
pub(crate) fn dummy_batch() -> Result<&'static RecordBatch> {
    static DUMMY_BATCH: OnceLock<RecordBatch> = OnceLock::new();
    if let Some(batch) = DUMMY_BATCH.get() {
        return Ok(batch);
    }
    // `OnceLock::get_or_init` cannot propagate the construction error, so build
    // the batch here (racing threads simply rebuild and discard) and publish it.
    let batch = create_dummy_batch()?;
    Ok(DUMMY_BATCH.get_or_init(|| batch))
}

/// Create a 1-row dummy RecordBatch for evaluating constant expressions.
///
/// The batch is never actually accessed for data - it's just needed because
/// the PhysicalExpr::evaluate API requires a RecordBatch. For expressions
/// that only contain literals, the batch content is irrelevant.
///
/// This is the same approach used in the logical expression `ConstEvaluator`.
fn create_dummy_batch() -> Result<RecordBatch> {
    // RecordBatch requires at least one column
    let dummy_schema = Arc::new(Schema::new(vec![Field::new("_", DataType::Null, true)]));
    let col = new_null_array(&DataType::Null, 1);
    Ok(RecordBatch::try_new(dummy_schema, vec![col])?)
}

/// Check if this expression has any column references.
pub fn has_column_references(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let mut has_columns = false;
    expr.apply(|expr| {
        if expr.as_any().downcast_ref::<Column>().is_some() {
            has_columns = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })
    .expect("apply should not fail");
    has_columns
}
