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

//! [`Unparser`] for converting `Expr` to SQL text

pub mod ast;
mod expr;
mod plan;
mod rewrite;
mod utils;

use self::dialect::{DefaultDialect, Dialect};
use crate::unparser::extension_unparser::UserDefinedLogicalNodeUnparser;
use datafusion_common::DFSchemaRef;
use datafusion_expr::{Expr, ExprSchemable, LogicalPlan};
pub use expr::expr_to_sql;
pub use plan::plan_to_sql;
use std::sync::Arc;
pub mod dialect;
pub mod extension_unparser;

/// Convert a DataFusion [`Expr`] to [`sqlparser::ast::Expr`]
///
/// See [`expr_to_sql`] for background. `Unparser` allows greater control of
/// the conversion, but with a more complicated API.
///
/// To get more human-readable output, see [`Self::with_pretty`]
///
/// # Example
/// ```
/// use datafusion_expr::{col, lit};
/// use datafusion_sql::unparser::Unparser;
/// let expr = col("a").gt(lit(4)); // form an expression `a > 4`
/// let unparser = Unparser::default();
/// let sql = unparser.expr_to_sql(&expr).unwrap();// convert to AST
/// // use the Display impl to convert to SQL text
/// assert_eq!(sql.to_string(), "(a > 4)");
/// // now convert to pretty sql
/// let unparser = unparser.with_pretty(true);
/// let sql = unparser.expr_to_sql(&expr).unwrap();
/// assert_eq!(sql.to_string(), "a > 4"); // note lack of parenthesis
/// ```
///
/// [`Expr`]: datafusion_expr::Expr
pub struct Unparser<'a> {
    dialect: &'a dyn Dialect,
    pretty: bool,
    extension_unparsers: Vec<Arc<dyn UserDefinedLogicalNodeUnparser>>,
    /// The schema the expressions being unparsed resolve against, when the
    /// caller knows it.
    ///
    /// A dialect sometimes needs an operand's type to render it — BigQuery has
    /// no cast from `DATE` to `INT64`, and no common supertype for a civil
    /// timestamp against an instant — and an expression only carries its own type
    /// when it happens to be a cast or a literal. Plan unparsing does know the
    /// schema at each node, so it passes it down rather than leaving expression
    /// unparsing to guess.
    schema: Option<DFSchemaRef>,
    /// Recursive CTEs met below the statement root, waiting to be attached to it.
    ///
    /// A recursive CTE nested inside a larger statement becomes a `WITH` entry on
    /// an enclosing query plus a table reference where it stood. *Which*
    /// enclosing query is the whole question: attaching it to the nearest one
    /// puts `WITH RECURSIVE` inside a derived table, and BigQuery answers "WITH
    /// RECURSIVE is only allowed at the top level of the SELECT". The statement
    /// root drains this, so the entry lands there however deep the plan buried
    /// it.
    pending_recursive_ctes: Arc<std::sync::Mutex<Vec<sqlparser::ast::Cte>>>,
    /// How many derived tables enclose the statement currently being built.
    ///
    /// A derived table is rendered by unparsing its plan as a whole statement,
    /// so "am I a statement?" is not the same question as "am I the top level?".
    /// Only depth zero may take the pending CTEs; draining at any other depth
    /// puts `WITH RECURSIVE` back inside the parentheses it has to escape.
    derived_depth: Arc<std::sync::atomic::AtomicUsize>,
}

impl<'a> Unparser<'a> {
    pub fn new(dialect: &'a dyn Dialect) -> Self {
        Self {
            dialect,
            pretty: false,
            extension_unparsers: vec![],
            schema: None,
            pending_recursive_ctes: Arc::new(std::sync::Mutex::new(Vec::new())),
            derived_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// The same unparser, resolving expression types against `schema`.
    ///
    /// Plan unparsing sets this per node, so a dialect that renders by type sees
    /// the types instead of inferring them from expression shape. Without it the
    /// renderings still apply wherever an expression states its own type.
    #[must_use]
    pub fn with_schema(&self, schema: DFSchemaRef) -> Unparser<'a> {
        Unparser {
            dialect: self.dialect,
            pretty: self.pretty,
            extension_unparsers: self.extension_unparsers.clone(),
            schema: Some(schema),
            // Shared, not recreated: this is called part-way down the walk, and a
            // recursive CTE met below would otherwise be collected into a list
            // that is dropped with the derived unparser.
            pending_recursive_ctes: Arc::clone(&self.pending_recursive_ctes),
            derived_depth: Arc::clone(&self.derived_depth),
        }
    }

    /// The type `expr` resolves to, from the schema when one was supplied and
    /// otherwise from the expression itself.
    ///
    /// `None` means the type is genuinely unknown here, and a rendering that
    /// depends on it must not fire.
    pub(crate) fn resolved_data_type(
        &self,
        expr: &Expr,
    ) -> Option<arrow::datatypes::DataType> {
        if let Some(schema) = &self.schema
            && let Ok(data_type) = expr.get_type(schema.as_ref())
        {
            return Some(data_type);
        }
        utils::provable_data_type(expr)
    }

    /// [`Self::plan_to_sql`] for a plan that is *inside* another statement.
    ///
    /// A derived table and an expression subquery are both rendered by building
    /// a whole statement and embedding it, so "a statement is being built" is not
    /// the same as "this is the top level". Recursive CTEs are hoisted to the top
    /// level, and the depth this keeps is how the drain tells the two apart —
    /// every nested rendering has to go through here, or a CTE met beneath it is
    /// attached inside the parentheses it needed to escape.
    fn plan_to_sql_nested(
        &self,
        plan: &LogicalPlan,
    ) -> datafusion_common::Result<sqlparser::ast::Statement> {
        self.derived_depth
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rendered = self.plan_to_sql(plan);
        self.derived_depth
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        rendered
    }

    /// The field `expr` resolves to, when a schema was supplied.
    ///
    /// [`Self::resolved_data_type`] answers with a `DataType`, which is not
    /// enough for an operand carried by an Arrow **extension type**: the storage
    /// type of `arrow.json` is `Utf8`, and what distinguishes it from an
    /// ordinary string lives in the field's metadata. A dialect that renders a
    /// document differently depending on whether the remote column is a native
    /// JSON type or a string needs the field to tell them apart.
    ///
    /// `None` means the type is genuinely unknown here — there was no schema, or
    /// the expression does not resolve against it — and a rendering that depends
    /// on the distinction must take the branch that is safe when it is unknown,
    /// never guess.
    pub fn resolved_field(&self, expr: &Expr) -> Option<Arc<arrow::datatypes::Field>> {
        let schema = self.schema.as_ref()?;
        expr.to_field(schema.as_ref()).ok().map(|(_, field)| field)
    }

    /// Create pretty SQL output, better suited for human consumption
    ///
    /// See example on the struct level documentation
    ///
    /// # Pretty Output
    ///
    /// By default, `Unparser` generates SQL text that will parse back to the
    /// same parsed [`Expr`], which is useful for creating machine readable
    /// expressions to send to other systems. However, the resulting expressions are
    /// not always nice to read for humans.
    ///
    /// For example
    ///
    /// ```sql
    /// ((a + 4) > 5)
    /// ```
    ///
    /// This method removes parenthesis using to the precedence rules of
    /// DataFusion. If the output is reparsed, the resulting [`Expr`] produces
    /// same value as the original in DataFusion, but with a potentially
    /// different order of operations.
    ///
    /// Note that this setting may create invalid SQL for other SQL query
    /// engines with different precedence rules
    ///
    /// # Example
    /// ```
    /// use datafusion_expr::{col, lit};
    /// use datafusion_sql::unparser::Unparser;
    /// let expr = col("a").gt(lit(4)).and(col("b").lt(lit(5))); // form an expression `a > 4 AND b < 5`
    /// let unparser = Unparser::default().with_pretty(true);
    /// let sql = unparser.expr_to_sql(&expr).unwrap();
    /// assert_eq!(sql.to_string(), "a > 4 AND b < 5"); // note lack of parenthesis
    /// ```
    ///
    /// [`Expr`]: datafusion_expr::Expr
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Add a custom unparser for user defined logical nodes
    ///
    /// DataFusion allows user to define custom logical nodes. This method allows to add custom child unparsers for these nodes.
    /// Implementation of [`UserDefinedLogicalNodeUnparser`] can be added to the root unparser to handle custom logical nodes.
    ///
    /// The child unparsers are called iteratively.
    /// There are two methods in [`Unparser`] will be called:
    /// - `extension_to_statement`: This method is called when the custom logical node is a custom statement.
    ///   If multiple child unparsers return a non-None value, the last unparsing result will be returned.
    /// - `extension_to_sql`: This method is called when the custom logical node is part of a statement.
    ///   If multiple child unparsers are registered for the same custom logical node, all of them will be called in order.
    pub fn with_extension_unparsers(
        mut self,
        extension_unparsers: Vec<Arc<dyn UserDefinedLogicalNodeUnparser>>,
    ) -> Self {
        self.extension_unparsers = extension_unparsers;
        self
    }
}

impl Default for Unparser<'_> {
    fn default() -> Self {
        Self {
            dialect: &DefaultDialect {},
            pretty: false,
            extension_unparsers: vec![],
            schema: None,
            pending_recursive_ctes: Arc::new(std::sync::Mutex::new(Vec::new())),
            derived_depth: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}
