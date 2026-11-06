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

use std::sync::{Arc, OnceLock};

use arrow::array::types::{Int16Type, Int32Type, Int64Type, RunEndIndexType};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Capacities, MutableArrayData, PrimitiveArray,
    RunArray, Scalar, StructArray, UInt32Array, UnionArray, cast::AsArray, make_array,
    make_comparator,
};
use arrow::compute::{SortOptions, take};
use arrow::datatypes::{DataType, Field, FieldRef, UnionMode};
use arrow_buffer::{ArrowNativeType, NullBuffer};

use datafusion_common::cast::{as_map_array, as_struct_array};
use datafusion_common::{
    Result, ScalarValue, exec_datafusion_err, exec_err, internal_err, plan_datafusion_err,
};
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::simplify::ExprSimplifyResult;
use datafusion_expr::{
    ColumnarValue, Documentation, Expr, ExpressionPlacement, ReturnFieldArgs,
    ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_macros::user_doc;

use super::named_struct::NamedStructFunc;
use super::r#struct::StructFunc;

#[user_doc(
    doc_section(label = "Other Functions"),
    description = r#"Returns a field within a map or a struct with the given key.
    Supports nested field access by providing multiple field names.
    Note: most users invoke `get_field` indirectly via field access
    syntax such as `my_struct_col['field_name']` which results in a call to
    `get_field(my_struct_col, 'field_name')`.
    Nested access like `my_struct['a']['b']` is optimized to a single call:
    `get_field(my_struct, 'a', 'b')`."#,
    syntax_example = "get_field(expression, field_name[, field_name2, ...])",
    sql_example = r#"```sql
> -- Access a field from a struct column
> create table test( struct_col) as values
    ({name: 'Alice', age: 30}),
    ({name: 'Bob', age: 25});
> select struct_col from test;
+-----------------------------+
| struct_col                  |
+-----------------------------+
| {name: Alice, age: 30}      |
| {name: Bob, age: 25}        |
+-----------------------------+
> select struct_col['name'] as name from test;
+-------+
| name  |
+-------+
| Alice |
| Bob   |
+-------+

> -- Nested field access with multiple arguments
> create table test(struct_col) as values
    ({outer: {inner_val: 42}});
> select struct_col['outer']['inner_val'] as result from test;
+--------+
| result |
+--------+
| 42     |
+--------+
```"#,
    argument(
        name = "expression",
        description = "The map or struct to retrieve a field from."
    ),
    argument(
        name = "field_name",
        description = "The field name(s) to access, in order for nested access. Must evaluate to strings."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct GetFieldFunc {
    signature: Signature,
}

impl Default for GetFieldFunc {
    fn default() -> Self {
        Self::new()
    }
}

/// Process a map array by finding matching keys and extracting corresponding values.
///
/// This function handles both simple (scalar) and nested key types by using
/// appropriate comparison strategies.
fn process_map_array(
    array: &dyn Array,
    key_array: Arc<dyn Array>,
) -> Result<ColumnarValue> {
    let map_array = as_map_array(array)?;
    let keys = if key_array.data_type().is_nested() {
        let comparator = make_comparator(
            map_array.keys().as_ref(),
            key_array.as_ref(),
            SortOptions::default(),
        )?;
        let len = map_array.keys().len().min(key_array.len());
        let values = (0..len).map(|i| comparator(i, i).is_eq()).collect();
        let nulls = NullBuffer::union(map_array.keys().nulls(), key_array.nulls());
        BooleanArray::new(values, nulls)
    } else {
        let be_compared = Scalar::new(key_array);
        arrow::compute::kernels::cmp::eq(&be_compared, map_array.keys())?
    };

    let original_data = map_array.entries().column(1).to_data();
    let capacity = Capacities::Array(original_data.len());
    let mut mutable =
        MutableArrayData::with_capacities(vec![&original_data], true, capacity);

    let offsets = map_array.value_offsets();
    // Scan the comparison result in place: slicing it per entry would allocate
    // a new array for every row of the map. Map keys are non-null by
    // definition, so the comparison result carries no nulls to check here.
    let matches = keys.values();

    for entry in 0..map_array.len() {
        // A null map slot hides its entries even when the offsets still point
        // at populated key/value rows.
        if map_array.is_null(entry) {
            mutable.try_extend_nulls(1)?;
            continue;
        }
        let start = offsets[entry] as usize;
        let end = offsets[entry + 1] as usize;

        let matched = (start..end).find(|&i| matches.value(i));

        match matched {
            Some(i) => mutable.try_extend(0, i, i + 1)?,
            None => mutable.try_extend_nulls(1)?,
        }
    }

    let data = mutable.freeze();
    let data = make_array(data);
    Ok(ColumnarValue::Array(data))
}

/// Process a map array with a nested key type by iterating through entries
/// and using a comparator for key matching.
///
/// This specialized version is used when the key type is nested (e.g., struct, list).
fn process_map_with_nested_key(
    array: &dyn Array,
    key_array: &dyn Array,
) -> Result<ColumnarValue> {
    let map_array = as_map_array(array)?;

    let comparator =
        make_comparator(map_array.keys().as_ref(), key_array, SortOptions::default())?;

    let original_data = map_array.entries().column(1).to_data();
    let capacity = Capacities::Array(original_data.len());
    let mut mutable =
        MutableArrayData::with_capacities(vec![&original_data], true, capacity);

    for entry in 0..map_array.len() {
        if map_array.is_null(entry) {
            mutable.try_extend_nulls(1)?;
            continue;
        }
        let start = map_array.value_offsets()[entry] as usize;
        let end = map_array.value_offsets()[entry + 1] as usize;

        let mut found_match = false;
        for i in start..end {
            if comparator(i, 0).is_eq() {
                mutable.try_extend(0, i, i + 1)?;
                found_match = true;
                break;
            }
        }

        if !found_match {
            mutable.try_extend_nulls(1)?;
        }
    }

    let data = mutable.freeze();
    let data = make_array(data);
    Ok(ColumnarValue::Array(data))
}

/// Extract a single field from a struct or map array
fn extract_single_field(base: ColumnarValue, name: ScalarValue) -> Result<ColumnarValue> {
    let arrays = ColumnarValue::values_to_arrays(&[base])?;
    let array = Arc::clone(&arrays[0]);

    let string_value = name.try_as_str().flatten().map(|s| s.to_string());

    match (array.data_type(), name, string_value) {
        // Dictionary-encoded struct: extract the field from the dictionary's
        // values (the deduplicated struct array) and rebuild a dictionary with
        // the same keys. This preserves dictionary encoding without expanding.
        (DataType::Dictionary(_, value_type), _, Some(field_name))
            if matches!(value_type.as_ref(), DataType::Struct(_)) =>
        {
            let dict = array.as_any_dictionary();
            let values_struct = dict.values().as_struct();
            let field_col =
                values_struct.column_by_name(&field_name).ok_or_else(|| {
                    exec_datafusion_err!(
                        "Field {field_name} not found in dictionary struct"
                    )
                })?;
            Ok(ColumnarValue::Array(
                dict.with_values(Arc::clone(field_col)),
            ))
        }
        (DataType::Map(_, _), ScalarValue::List(arr), _) => {
            let key_array: Arc<dyn Array> = arr;
            process_map_array(&array, key_array)
        }
        (DataType::Map(_, _), ScalarValue::Struct(arr), _) => {
            process_map_array(&array, arr as Arc<dyn Array>)
        }
        (DataType::Map(_, _), other, _) => {
            let data_type = other.data_type();
            if data_type.is_nested() {
                process_map_with_nested_key(&array, &other.to_array()?)
            } else {
                process_map_array(&array, other.to_array()?)
            }
        }
        (DataType::Struct(_), _, Some(k)) => {
            let as_struct_array = as_struct_array(&array)?;
            match as_struct_array.column_by_name(&k) {
                None => exec_err!("Field {k} not found in struct"),
                Some(col) => Ok(ColumnarValue::Array(apply_parent_struct_nulls(
                    as_struct_array,
                    col,
                )?)),
            }
        }
        (DataType::Struct(_), name, _) => exec_err!(
            "get_field is only possible on struct with utf8 indexes. \
                         Received with {name:?} index"
        ),
        (DataType::Null, _, _) => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        (dt, name, _) => exec_err!(
            "get_field is only possible on maps or structs. Received {dt} with {name:?} index"
        ),
    }
}

/// Apply a parent struct's null bitmap onto an extracted child field.
///
/// Arrow allows null struct slots whose children still hold values; field
/// access must return null for those slots. Not every child type can carry a
/// top-level validity bitmap (`Null`, `Union`, `RunEndEncoded`), so those use
/// layout-aware paths instead of attaching the parent bitmap.
fn apply_parent_struct_nulls(parent: &StructArray, child: &ArrayRef) -> Result<ArrayRef> {
    // Fast path: parent has no null rows — return the child unchanged.
    if parent.null_count() == 0 {
        return Ok(Arc::clone(child));
    }

    match child.data_type() {
        // Null arrays are already all-null and reject a validity bitmap in
        // `ArrayData::build()`.
        DataType::Null => Ok(Arc::clone(child)),
        DataType::Union(_, mode) => mask_union_with_parent_nulls(child, parent, *mode),
        DataType::RunEndEncoded(_, _) => {
            mask_run_end_encoded_with_parent_nulls(child, parent)
        }
        _ => {
            let nulls = NullBuffer::union(parent.nulls(), child.nulls());
            Ok(make_array(
                child.to_data().into_builder().nulls(nulls).build()?,
            ))
        }
    }
}

/// Union top-level nulls onto an array that supports a physical validity bitmap.
///
/// `Null` children are returned unchanged. Callers must not pass `Union` or
/// `RunEndEncoded` here — those need layout-aware masking.
fn union_nulls_onto_array(
    array: &ArrayRef,
    extra_nulls: &NullBuffer,
) -> Result<ArrayRef> {
    if matches!(array.data_type(), DataType::Null) {
        return Ok(Arc::clone(array));
    }
    let nulls = NullBuffer::union(array.nulls(), Some(extra_nulls));
    Ok(make_array(
        array.to_data().into_builder().nulls(nulls).build()?,
    ))
}

/// Null selected positions on an array that supports a physical validity bitmap.
fn null_out_positions(array: &ArrayRef, positions: &[usize]) -> Result<ArrayRef> {
    if positions.is_empty() {
        return Ok(Arc::clone(array));
    }
    if matches!(array.data_type(), DataType::Null) {
        return Ok(Arc::clone(array));
    }
    let mut is_valid: Vec<bool> = match array.nulls() {
        Some(nulls) => (0..array.len()).map(|i| nulls.is_valid(i)).collect(),
        None => vec![true; array.len()],
    };
    for &p in positions {
        is_valid[p] = false;
    }
    let nulls = NullBuffer::from(is_valid);
    Ok(make_array(
        array.to_data().into_builder().nulls(Some(nulls)).build()?,
    ))
}

fn mask_union_with_parent_nulls(
    child: &ArrayRef,
    parent: &StructArray,
    mode: UnionMode,
) -> Result<ArrayRef> {
    let union = child
        .as_any()
        .downcast_ref::<UnionArray>()
        .expect("UnionArray downcast");
    // Parent is known to have nulls (caller checked null_count > 0).
    let parent_nulls = parent
        .nulls()
        .expect("parent with null_count > 0 must have a null buffer");
    let fields = union.fields().clone();

    let children = match mode {
        UnionMode::Sparse => fields
            .iter()
            .map(|(type_id, _)| {
                union_nulls_onto_array(union.child(type_id), parent_nulls)
            })
            .collect::<Result<Vec<_>>>()?,
        UnionMode::Dense => {
            let mut positions: Vec<Vec<usize>> =
                fields.iter().map(|_| Vec::new()).collect();
            let tid_to_idx: std::collections::HashMap<i8, usize> = fields
                .iter()
                .enumerate()
                .map(|(idx, (type_id, _))| (type_id, idx))
                .collect();
            for i in 0..parent.len() {
                if parent_nulls.is_null(i) {
                    let type_id = union.type_id(i);
                    let offset = union.value_offset(i);
                    positions[tid_to_idx[&type_id]].push(offset);
                }
            }
            fields
                .iter()
                .enumerate()
                .map(|(idx, (type_id, _))| {
                    null_out_positions(union.child(type_id), &positions[idx])
                })
                .collect::<Result<Vec<_>>>()?
        }
    };

    Ok(Arc::new(UnionArray::try_new(
        fields,
        union.type_ids().clone(),
        union.offsets().cloned(),
        children,
    )?))
}

fn mask_run_end_encoded_with_parent_nulls(
    child: &ArrayRef,
    parent: &StructArray,
) -> Result<ArrayRef> {
    match child.data_type() {
        DataType::RunEndEncoded(run_ends_field, _) => match run_ends_field.data_type() {
            DataType::Int16 => mask_ree_impl::<Int16Type>(child, parent),
            DataType::Int32 => mask_ree_impl::<Int32Type>(child, parent),
            DataType::Int64 => mask_ree_impl::<Int64Type>(child, parent),
            other => {
                internal_err!("unsupported run-end index type for get_field: {other}")
            }
        },
        other => internal_err!("expected RunEndEncoded child, got {other}"),
    }
}

/// Rebuild a `RunEndEncoded` child so parent-null logical slots become null
/// values. Uses one physical value per logical row so mixed validity within a
/// former run is represented correctly without attaching a top-level bitmap.
fn mask_ree_impl<R: RunEndIndexType>(
    child: &ArrayRef,
    parent: &StructArray,
) -> Result<ArrayRef> {
    let run = child
        .as_any()
        .downcast_ref::<RunArray<R>>()
        .expect("RunArray downcast");
    let len = parent.len();
    let logical: Vec<u32> = (0..len as u32).collect();
    let physical = run.get_physical_indices(&logical)?;
    let value_indices = UInt32Array::from_iter(
        (0..len).map(|i| parent.is_valid(i).then_some(physical[i] as u32)),
    );
    let new_values = take(run.values().as_ref(), &value_indices, None)?;
    let run_ends = PrimitiveArray::<R>::from_iter_values(
        (1..=len).map(|i| R::Native::from_usize(i).expect("run end fits in index type")),
    );
    Ok(Arc::new(RunArray::<R>::try_new(
        &run_ends,
        new_values.as_ref(),
    )?))
}

/// The shared `get_field` UDF, reused whenever simplification needs to build a
/// fresh `get_field` node (e.g. re-wrapping the remaining access path).
fn get_field_udf() -> Arc<ScalarUDF> {
    static GET_FIELD_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
    Arc::clone(
        GET_FIELD_UDF
            .get_or_init(|| Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::new()))),
    )
}

/// Try to simplify a `get_field` call whose base is an inline struct
/// constructor by resolving the field access at plan time.
///
/// Handles both struct constructors:
/// * `named_struct('a', x, 'b', y)` — fields are looked up by name.
/// * `struct(x, y)` — fields are positional and named `c0`, `c1`, ...
///
/// For example:
/// * `get_field(named_struct('min', a, 'max', b), 'max')` => `b`
/// * `get_field(struct(a, b), 'c1')` => `b`
///
/// `args` is the (already flattened) argument list of the `get_field` call:
/// `[base, field_name, rest_of_path...]`. When extra path elements remain
/// after resolving the first one (`get_field(named_struct('s', inner), 's', 'k')`),
/// the resolved value is re-wrapped in a `get_field` call for the remaining
/// path so the simplifier can recurse into it on the next pass.
///
/// Returns `None` — leaving the expression untouched — whenever the rewrite
/// cannot be proven safe, e.g. a non-literal field name, a `named_struct`
/// with a non-literal field name (which might shadow the requested field at
/// runtime), or a field the constructor does not produce.
///
/// Replacing the access with the selected field expression drops the
/// expressions for the other (unaccessed) fields, so they are no longer
/// evaluated — e.g. `get_field(named_struct('a', 1/0, 'b', x), 'b')` becomes
/// `x` and the `1/0` is never evaluated. This is intentional and matches the
/// optimizer's contract for immutable expressions: a simplification may drop
/// sub-expressions whose value is not observed.
fn simplify_get_field_over_struct_constructor(args: &[Expr]) -> Option<Expr> {
    let [base, field_name, rest @ ..] = args else {
        return None;
    };

    // The accessed field name must be a non-empty string literal.
    let Expr::Literal(field_name, _) = field_name else {
        return None;
    };
    let field_name = field_name
        .try_as_str()
        .flatten()
        .filter(|s| !s.is_empty())?;

    let Expr::ScalarFunction(ScalarFunction {
        func,
        args: ctor_args,
    }) = base
    else {
        return None;
    };

    let value = if func.inner().is::<NamedStructFunc>() {
        // named_struct(name1, value1, name2, value2, ...)
        if !ctor_args.len().is_multiple_of(2) {
            return None;
        }
        let mut matched = None;
        for pair in ctor_args.chunks_exact(2) {
            // Every name must be a literal string: a non-literal name appearing
            // *before* the first match could evaluate to `field_name` at runtime
            // and become the real first match (Arrow's `column_by_name` returns
            // the first match), so we cannot resolve the access.
            //
            // We conservatively bail on *any* non-literal name. Once a literal
            // match has been found, a later non-literal name is in fact harmless
            // — it can never precede the first match — so bailing there is a
            // deliberate approximation we accept to keep this check simple, not a
            // correctness requirement.
            let Expr::Literal(name, _) = &pair[0] else {
                return None;
            };
            let name = name.try_as_str().flatten()?;
            // `column_by_name` resolves to the first match, so do the same.
            if matched.is_none() && name == field_name {
                matched = Some(&pair[1]);
            }
        }
        matched?.clone()
    } else if func.inner().is::<StructFunc>() {
        // struct(value0, value1, ...) produces fields named c0, c1, ...
        let index: usize = field_name.strip_prefix('c')?.parse().ok()?;
        // Reject non-canonical spellings (e.g. "c01") that name no real field.
        if format!("c{index}") != field_name {
            return None;
        }
        ctor_args.get(index)?.clone()
    } else {
        return None;
    };

    if rest.is_empty() {
        return Some(value);
    }

    // Remaining path elements: re-wrap as get_field(value, rest...) and let
    // the simplifier resolve the rest on a subsequent pass.
    let mut new_args = Vec::with_capacity(rest.len() + 1);
    new_args.push(value);
    new_args.extend_from_slice(rest);
    Some(Expr::ScalarFunction(ScalarFunction::new_udf(
        get_field_udf(),
        new_args,
    )))
}

impl GetFieldFunc {
    pub fn new() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

// get_field(struct_array, field_name)
impl ScalarUDFImpl for GetFieldFunc {
    fn name(&self) -> &str {
        "get_field"
    }

    fn display_name(&self, args: &[Expr]) -> Result<String> {
        if args.len() < 2 {
            return exec_err!(
                "get_field requires at least 2 arguments, got {}",
                args.len()
            );
        }

        let base = &args[0];
        let field_names: Vec<String> = args[1..]
            .iter()
            .map(|f| match f {
                Expr::Literal(name, _) => name.to_string(),
                other => other.schema_name().to_string(),
            })
            .collect();

        Ok(format!("{}[{}]", base, field_names.join("][")))
    }

    fn schema_name(&self, args: &[Expr]) -> Result<String> {
        if args.len() < 2 {
            return exec_err!(
                "get_field requires at least 2 arguments, got {}",
                args.len()
            );
        }

        let base = &args[0];
        let field_names: Vec<String> = args[1..]
            .iter()
            .map(|f| match f {
                Expr::Literal(name, _) => name.to_string(),
                other => other.schema_name().to_string(),
            })
            .collect();

        Ok(format!(
            "{}[{}]",
            base.schema_name(),
            field_names.join("][")
        ))
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        internal_err!("return_field_from_args should be called instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs) -> Result<FieldRef> {
        // Validate minimum 2 arguments: base expression + at least one field name
        if args.scalar_arguments.len() < 2 {
            return exec_err!(
                "get_field requires at least 2 arguments, got {}",
                args.scalar_arguments.len()
            );
        }

        let mut current_field = Arc::clone(&args.arg_fields[0]);

        // Iterate through each field name (starting from index 1)
        for (i, sv) in args.scalar_arguments.iter().enumerate().skip(1) {
            match current_field.data_type() {
                DataType::Map(map_field, _) => {
                    match map_field.data_type() {
                        DataType::Struct(fields) if fields.len() == 2 => {
                            // Arrow's MapArray is essentially a ListArray of structs with two columns. They are
                            // often named "key", and "value", but we don't require any specific naming here;
                            // instead, we assume that the second column is the "value" column both here and in
                            // execution.
                            let value_field = fields
                                .get(1)
                                .expect("fields should have exactly two members");

                            current_field = Arc::new(
                                value_field.as_ref().clone().with_nullable(true),
                            );
                        }
                        _ => {
                            return exec_err!(
                                "Map fields must contain a Struct with exactly 2 fields"
                            );
                        }
                    }
                }
                // Dictionary-encoded struct: resolve the child field from
                // the underlying struct, then wrap the result back in the
                // same Dictionary type so the promised type matches execution.
                DataType::Dictionary(key_type, value_type)
                    if matches!(value_type.as_ref(), DataType::Struct(_)) =>
                {
                    let DataType::Struct(fields) = value_type.as_ref() else {
                        unreachable!()
                    };
                    let field_name = sv
                        .as_ref()
                        .and_then(|sv| {
                            sv.try_as_str().flatten().filter(|s| !s.is_empty())
                        })
                        .ok_or_else(|| {
                            exec_datafusion_err!("Field name must be a non-empty string")
                        })?;

                    let child_field = fields
                        .iter()
                        .find(|f| f.name() == field_name)
                        .ok_or_else(|| {
                            plan_datafusion_err!("Field {field_name} not found in struct")
                        })?;

                    let dict_type = DataType::Dictionary(
                        key_type.clone(),
                        Box::new(child_field.data_type().clone()),
                    );
                    let mut new_field =
                        child_field.as_ref().clone().with_data_type(dict_type);
                    if current_field.is_nullable() {
                        new_field = new_field.with_nullable(true);
                    }
                    current_field = Arc::new(new_field);
                }
                DataType::Struct(fields) => {
                    let field_name = sv
                        .as_ref()
                        .and_then(|sv| {
                            sv.try_as_str().flatten().filter(|s| !s.is_empty())
                        })
                        .ok_or_else(|| {
                            datafusion_common::DataFusionError::Execution(
                                "Field name must be a non-empty string".to_string(),
                            )
                        })?;

                    let child_field = fields
                        .iter()
                        .find(|f| f.name() == field_name)
                        .ok_or_else(|| {
                            plan_datafusion_err!("Field {field_name} not found in struct")
                        })?;

                    let mut new_field = child_field.as_ref().clone();

                    // If the parent is nullable, then getting the child must be nullable
                    if current_field.is_nullable() {
                        new_field = new_field.with_nullable(true);
                    }
                    current_field = Arc::new(new_field);
                }
                DataType::Null => {
                    return Ok(Field::new(self.name(), DataType::Null, true).into());
                }
                other => {
                    return exec_err!(
                        "Cannot access field at argument {}: type {} is not Struct, Map, or Null",
                        i,
                        other
                    );
                }
            }
        }

        Ok(current_field)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        if args.args.len() < 2 {
            return exec_err!(
                "get_field requires at least 2 arguments, got {}",
                args.args.len()
            );
        }

        let mut current = args.args[0].clone();

        // Early exit for null base
        if current.data_type().is_null() {
            return Ok(ColumnarValue::Scalar(ScalarValue::Null));
        }

        // Iterate through each field name
        for field_name in args.args.iter().skip(1) {
            let field_name_scalar = match field_name {
                ColumnarValue::Scalar(name) => name.clone(),
                _ => {
                    return exec_err!(
                        "get_field function requires all field_name arguments to be scalars"
                    );
                }
            };

            current = extract_single_field(current, field_name_scalar)?;

            // Early exit if we hit null
            if current.data_type().is_null() {
                return Ok(ColumnarValue::Scalar(ScalarValue::Null));
            }
        }

        Ok(current)
    }

    fn simplify(
        &self,
        args: Vec<Expr>,
        _info: &datafusion_expr::simplify::SimplifyContext,
    ) -> Result<ExprSimplifyResult> {
        // Need at least 2 args (base + field)
        if args.len() < 2 {
            return Ok(ExprSimplifyResult::Original(args));
        }

        // Flatten all nested get_field calls in a single pass
        // Pattern: get_field(get_field(get_field(base, a), b), c) => get_field(base, a, b, c)
        //
        // `path_args_stack` collects each level's field-name arguments,
        // outermost first; it is reversed below to restore access order.
        let mut path_args_stack = vec![&args[1..]];
        let mut current_expr = &args[0];

        // Walk down the chain of nested get_field calls
        let base_expr = loop {
            if let Expr::ScalarFunction(ScalarFunction {
                func,
                args: inner_args,
            }) = current_expr
                && func.inner().is::<GetFieldFunc>()
            {
                // Store this level's path arguments (all except the first, which is base/nested call)
                path_args_stack.push(&inner_args[1..]);

                // Move to the next level down
                current_expr = &inner_args[0];
                continue;
            }
            // Not a get_field call, this is the base expression
            break current_expr;
        };

        // Whether any nested get_field calls were collapsed above.
        let did_flatten = path_args_stack.len() > 1;

        // Build merged args: [base, ...all path args in access order].
        // The stack holds path slices outermost-first, so iterate in reverse.
        let mut merged_args = vec![base_expr.clone()];
        for path_slice in path_args_stack.iter().rev() {
            merged_args.extend_from_slice(path_slice);
        }

        // Resolve field accesses against an inline struct constructor:
        //   get_field(named_struct('min', a, 'max', b), 'max') => b
        if let Some(simplified) = simplify_get_field_over_struct_constructor(&merged_args)
        {
            return Ok(ExprSimplifyResult::Simplified(simplified));
        }

        if did_flatten {
            return Ok(ExprSimplifyResult::Simplified(Expr::ScalarFunction(
                ScalarFunction::new_udf(get_field_udf(), merged_args),
            )));
        }

        Ok(ExprSimplifyResult::Original(args))
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        if arg_types.len() < 2 {
            return exec_err!(
                "get_field requires at least 2 arguments, got {}",
                arg_types.len()
            );
        }
        // Accept types as-is, validation happens in return_field_from_args
        Ok(arg_types.to_vec())
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }

    fn placement(&self, args: &[ExpressionPlacement]) -> ExpressionPlacement {
        // get_field can be pushed to leaves if:
        // 1. The base (first arg) is a column or already placeable at leaves
        // 2. All field keys (remaining args) are literals
        if args.is_empty() {
            return ExpressionPlacement::KeepInPlace;
        }

        let base_placement = args[0];
        let base_is_pushable = matches!(
            base_placement,
            ExpressionPlacement::Column | ExpressionPlacement::MoveTowardsLeafNodes
        );

        let all_keys_are_literals = args
            .iter()
            .skip(1)
            .all(|p| *p == ExpressionPlacement::Literal);

        if base_is_pushable && all_keys_are_literals {
            ExpressionPlacement::MoveTowardsLeafNodes
        } else {
            ExpressionPlacement::KeepInPlace
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::types::Int32Type;
    use arrow::array::{
        ArrayRef, Int32Array, NullArray, RunArray, StringArray, StructArray, UnionArray,
    };
    use arrow::buffer::ScalarBuffer;
    use arrow::datatypes::{Fields, UnionFields};

    #[test]
    fn test_get_field_applies_parent_struct_nulls() -> Result<()> {
        // Arrow allows a null struct slot whose children still hold values.
        // Field access must return null for that slot, not the hidden child.
        let a_values = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
        let fields: Fields = vec![Field::new("a", DataType::Int32, true)].into();
        let nulls = NullBuffer::from(vec![true, false, true]);
        let struct_array =
            StructArray::new(fields, vec![Arc::new(a_values) as ArrayRef], Some(nulls));

        let result = extract_single_field(
            ColumnarValue::Array(Arc::new(struct_array)),
            ScalarValue::Utf8(Some("a".to_string())),
        )?;
        let result_array = result.into_array(3)?;
        let expected = Int32Array::from(vec![Some(1), None, Some(3)]);
        assert_eq!(result_array.as_ref(), &expected as &dyn Array);
        Ok(())
    }

    #[test]
    fn test_get_field_parent_nulls_fast_path_no_rebuild() -> Result<()> {
        // When the parent has no null rows, the child Arc must be returned
        // unchanged (no ArrayData rebuild).
        let a_values = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
        let child: ArrayRef = Arc::new(a_values);
        let child_ptr = Arc::as_ptr(&child);
        let fields: Fields = vec![Field::new("a", DataType::Int32, true)].into();
        let struct_array = StructArray::new(fields, vec![Arc::clone(&child)], None);

        let result = extract_single_field(
            ColumnarValue::Array(Arc::new(struct_array)),
            ScalarValue::Utf8(Some("a".to_string())),
        )?;
        let result_array = result.into_array(3)?;
        assert!(
            Arc::ptr_eq(&result_array, &child),
            "child ArrayRef must be Arc-cloned when parent has no nulls"
        );
        let _ = child_ptr;
        Ok(())
    }

    #[test]
    fn test_get_field_null_child_with_nullable_parent() -> Result<()> {
        // Null-typed children cannot carry a validity bitmap; Arrow rejects
        // Null+bitmap in build(). Return the Null child unchanged.
        let null_child: ArrayRef = Arc::new(NullArray::new(3));
        let fields: Fields = vec![Field::new("n", DataType::Null, true)].into();
        let nulls = NullBuffer::from(vec![true, false, true]);
        let struct_array =
            StructArray::new(fields, vec![Arc::clone(&null_child)], Some(nulls));

        let result = extract_single_field(
            ColumnarValue::Array(Arc::new(struct_array)),
            ScalarValue::Utf8(Some("n".to_string())),
        )?;
        let result_array = result.into_array(3)?;
        assert_eq!(result_array.data_type(), &DataType::Null);
        assert_eq!(result_array.len(), 3);
        // NullArray has no physical null buffer; logical nulls cover every slot.
        assert_eq!(result_array.logical_null_count(), 3);
        Ok(())
    }

    #[test]
    fn test_get_field_sparse_union_child_with_nullable_parent() -> Result<()> {
        // Sparse unions cannot carry a top-level validity bitmap. Parent nulls
        // are applied by nulling every child at those slots so logical nulls
        // follow the parent.
        let union_fields = UnionFields::try_new(
            vec![1, 3],
            vec![
                Field::new("str", DataType::Utf8, true),
                Field::new("int", DataType::Int32, true),
            ],
        )?;
        let type_ids = ScalarBuffer::from(vec![3_i8, 1, 3]);
        let children: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("c")])),
            Arc::new(Int32Array::from(vec![Some(10), Some(20), Some(30)])),
        ];
        let union = UnionArray::try_new(union_fields.clone(), type_ids, None, children)?;
        let union: ArrayRef = Arc::new(union);

        let fields: Fields =
            vec![Field::new("u", union.data_type().clone(), true)].into();
        let parent_nulls = NullBuffer::from(vec![true, false, true]);
        let struct_array =
            StructArray::new(fields, vec![Arc::clone(&union)], Some(parent_nulls));

        let result = extract_single_field(
            ColumnarValue::Array(Arc::new(struct_array)),
            ScalarValue::Utf8(Some("u".to_string())),
        )?;
        let result_array = result.into_array(3)?;
        assert!(matches!(result_array.data_type(), DataType::Union(_, _)));
        // Union has no top-level bitmap; parent nulls show up as logical nulls.
        let logical = result_array
            .logical_nulls()
            .expect("union with parent nulls must expose logical nulls");
        assert!(logical.is_valid(0));
        assert!(logical.is_null(1));
        assert!(logical.is_valid(2));
        Ok(())
    }

    #[test]
    fn test_get_field_ree_child_with_nullable_parent() -> Result<()> {
        // RunEndEncoded cannot carry a top-level validity bitmap. Parent nulls
        // are encoded as null values in a rebuilt REE.
        let run_ends = Int32Array::from(vec![2, 3]);
        let values = Int32Array::from(vec![Some(7), Some(8)]);
        let ree = RunArray::<Int32Type>::try_new(&run_ends, &values)?;
        let ree: ArrayRef = Arc::new(ree);

        let fields: Fields = vec![Field::new("r", ree.data_type().clone(), true)].into();
        let parent_nulls = NullBuffer::from(vec![true, false, true]);
        let struct_array =
            StructArray::new(fields, vec![Arc::clone(&ree)], Some(parent_nulls));

        let result = extract_single_field(
            ColumnarValue::Array(Arc::new(struct_array)),
            ScalarValue::Utf8(Some("r".to_string())),
        )?;
        let result_array = result.into_array(3)?;
        assert!(matches!(
            result_array.data_type(),
            DataType::RunEndEncoded(_, _)
        ));
        let logical = result_array
            .logical_nulls()
            .expect("REE with parent nulls must expose logical nulls");
        assert!(logical.is_valid(0));
        assert!(logical.is_null(1));
        assert!(logical.is_valid(2));
        Ok(())
    }

    #[test]
    fn test_get_field_utf8view_key() -> Result<()> {
        // Create a struct array with fields "a" and "b"
        let a_values = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
        let b_values = Int32Array::from(vec![Some(10), Some(20), Some(30)]);

        let fields: Fields = vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ]
        .into();

        let struct_array = StructArray::new(
            fields,
            vec![
                Arc::new(a_values) as ArrayRef,
                Arc::new(b_values) as ArrayRef,
            ],
            None,
        );

        let base = ColumnarValue::Array(Arc::new(struct_array));

        // Use Utf8View key to access field "a"
        let key = ScalarValue::Utf8View(Some("a".to_string()));

        let result = extract_single_field(base, key)?;

        let result_array = result.into_array(3)?;
        let expected = Int32Array::from(vec![Some(1), Some(2), Some(3)]);

        assert_eq!(result_array.as_ref(), &expected as &dyn Array);

        Ok(())
    }

    #[test]
    fn test_get_field_dict_encoded_struct() -> Result<()> {
        use arrow::array::{DictionaryArray, StringArray, UInt32Array};
        use arrow::datatypes::UInt32Type;

        let names = Arc::new(StringArray::from(vec!["main", "foo", "bar"])) as ArrayRef;
        let ids = Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef;

        let struct_fields: Fields = vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("id", DataType::Int32, false),
        ]
        .into();

        let values_struct =
            Arc::new(StructArray::new(struct_fields, vec![names, ids], None)) as ArrayRef;

        let keys = UInt32Array::from(vec![0u32, 1, 2, 0, 1]);
        let dict = DictionaryArray::<UInt32Type>::try_new(keys, values_struct)?;

        let base = ColumnarValue::Array(Arc::new(dict));
        let key = ScalarValue::Utf8(Some("name".to_string()));

        let result = extract_single_field(base, key)?;
        let result_array = result.into_array(5)?;

        assert!(
            matches!(result_array.data_type(), DataType::Dictionary(_, _)),
            "expected dictionary output, got {:?}",
            result_array.data_type()
        );

        let result_dict = result_array
            .as_any()
            .downcast_ref::<DictionaryArray<UInt32Type>>()
            .unwrap();
        assert_eq!(result_dict.values().len(), 3);
        assert_eq!(result_dict.len(), 5);

        let resolved = arrow::compute::cast(&result_array, &DataType::Utf8)?;
        let string_arr = resolved.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(string_arr.value(0), "main");
        assert_eq!(string_arr.value(1), "foo");
        assert_eq!(string_arr.value(2), "bar");
        assert_eq!(string_arr.value(3), "main");
        assert_eq!(string_arr.value(4), "foo");

        Ok(())
    }

    #[test]
    fn test_get_field_nested_dict_struct() -> Result<()> {
        use arrow::array::{DictionaryArray, StringArray, UInt32Array};
        use arrow::datatypes::UInt32Type;

        let func_names = Arc::new(StringArray::from(vec!["main", "foo"])) as ArrayRef;
        let func_files = Arc::new(StringArray::from(vec!["main.c", "foo.c"])) as ArrayRef;
        let func_fields: Fields = vec![
            Field::new("name", DataType::Utf8, false),
            Field::new("file", DataType::Utf8, false),
        ]
        .into();
        let func_struct = Arc::new(StructArray::new(
            func_fields.clone(),
            vec![func_names, func_files],
            None,
        )) as ArrayRef;
        let func_dict = Arc::new(DictionaryArray::<UInt32Type>::try_new(
            UInt32Array::from(vec![0u32, 1, 0]),
            func_struct,
        )?) as ArrayRef;

        let line_nums = Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef;
        let line_fields: Fields = vec![
            Field::new("num", DataType::Int32, false),
            Field::new(
                "function",
                DataType::Dictionary(
                    Box::new(DataType::UInt32),
                    Box::new(DataType::Struct(func_fields)),
                ),
                false,
            ),
        ]
        .into();
        let line_struct = StructArray::new(line_fields, vec![line_nums, func_dict], None);

        let base = ColumnarValue::Array(Arc::new(line_struct));

        let func_result =
            extract_single_field(base, ScalarValue::Utf8(Some("function".to_string())))?;

        let func_array = func_result.into_array(3)?;
        assert!(
            matches!(func_array.data_type(), DataType::Dictionary(_, _)),
            "expected dictionary for function, got {:?}",
            func_array.data_type()
        );

        let name_result = extract_single_field(
            ColumnarValue::Array(func_array),
            ScalarValue::Utf8(Some("name".to_string())),
        )?;
        let name_array = name_result.into_array(3)?;

        assert!(
            matches!(name_array.data_type(), DataType::Dictionary(_, _)),
            "expected dictionary for name, got {:?}",
            name_array.data_type()
        );

        let name_dict = name_array
            .as_any()
            .downcast_ref::<DictionaryArray<UInt32Type>>()
            .unwrap();
        assert_eq!(name_dict.values().len(), 2);
        assert_eq!(name_dict.len(), 3);

        let resolved = arrow::compute::cast(&name_array, &DataType::Utf8)?;
        let strings = resolved.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(strings.value(0), "main");
        assert_eq!(strings.value(1), "foo");
        assert_eq!(strings.value(2), "main");

        Ok(())
    }

    #[test]
    fn test_placement_literal_key() {
        let func = GetFieldFunc::new();

        // get_field(col, 'literal') -> leaf-pushable (static field access)
        let args = vec![ExpressionPlacement::Column, ExpressionPlacement::Literal];
        assert_eq!(
            func.placement(&args),
            ExpressionPlacement::MoveTowardsLeafNodes
        );

        // get_field(col, 'a', 'b') -> leaf-pushable (nested static field access)
        let args = vec![
            ExpressionPlacement::Column,
            ExpressionPlacement::Literal,
            ExpressionPlacement::Literal,
        ];
        assert_eq!(
            func.placement(&args),
            ExpressionPlacement::MoveTowardsLeafNodes
        );

        // get_field(get_field(col, 'a'), 'b') represented as MoveTowardsLeafNodes for base
        let args = vec![
            ExpressionPlacement::MoveTowardsLeafNodes,
            ExpressionPlacement::Literal,
        ];
        assert_eq!(
            func.placement(&args),
            ExpressionPlacement::MoveTowardsLeafNodes
        );
    }

    #[test]
    fn test_placement_column_key() {
        let func = GetFieldFunc::new();

        // get_field(col, other_col) -> NOT leaf-pushable (dynamic per-row lookup)
        let args = vec![ExpressionPlacement::Column, ExpressionPlacement::Column];
        assert_eq!(func.placement(&args), ExpressionPlacement::KeepInPlace);

        // get_field(col, 'a', other_col) -> NOT leaf-pushable (dynamic nested lookup)
        let args = vec![
            ExpressionPlacement::Column,
            ExpressionPlacement::Literal,
            ExpressionPlacement::Column,
        ];
        assert_eq!(func.placement(&args), ExpressionPlacement::KeepInPlace);
    }

    #[test]
    fn test_placement_root() {
        let func = GetFieldFunc::new();

        // get_field(root_expr, 'literal') -> NOT leaf-pushable
        let args = vec![
            ExpressionPlacement::KeepInPlace,
            ExpressionPlacement::Literal,
        ];
        assert_eq!(func.placement(&args), ExpressionPlacement::KeepInPlace);

        // get_field(col, root_expr) -> NOT leaf-pushable
        let args = vec![
            ExpressionPlacement::Column,
            ExpressionPlacement::KeepInPlace,
        ];
        assert_eq!(func.placement(&args), ExpressionPlacement::KeepInPlace);
    }

    #[test]
    fn test_placement_edge_cases() {
        let func = GetFieldFunc::new();

        // Empty args -> NOT leaf-pushable
        assert_eq!(func.placement(&[]), ExpressionPlacement::KeepInPlace);

        // Just base, no key -> MoveTowardsLeafNodes (not a valid call but should handle gracefully)
        let args = vec![ExpressionPlacement::Column];
        assert_eq!(
            func.placement(&args),
            ExpressionPlacement::MoveTowardsLeafNodes
        );

        // Literal base with literal key -> NOT leaf-pushable (would be constant-folded)
        let args = vec![ExpressionPlacement::Literal, ExpressionPlacement::Literal];
        assert_eq!(func.placement(&args), ExpressionPlacement::KeepInPlace);
    }

    // --- get_field over struct constructor simplification --------------------

    use datafusion_common::Column;
    use datafusion_expr::simplify::SimplifyContext;

    /// A non-empty string literal expression.
    fn lit_str(s: &str) -> Expr {
        Expr::Literal(ScalarValue::Utf8(Some(s.to_string())), None)
    }

    /// A column reference expression.
    fn col(name: &str) -> Expr {
        Expr::Column(Column::from_name(name))
    }

    fn scalar_fn(udf: ScalarUDF, args: Vec<Expr>) -> Expr {
        Expr::ScalarFunction(ScalarFunction::new_udf(Arc::new(udf), args))
    }

    /// `named_struct(name1, value1, name2, value2, ...)`.
    fn named_struct(pairs: Vec<(&str, Expr)>) -> Expr {
        let args = pairs
            .into_iter()
            .flat_map(|(name, value)| [lit_str(name), value])
            .collect();
        scalar_fn(ScalarUDF::new_from_impl(NamedStructFunc::new()), args)
    }

    /// `struct(value0, value1, ...)`.
    fn struct_fn(values: Vec<Expr>) -> Expr {
        scalar_fn(ScalarUDF::new_from_impl(StructFunc::new()), values)
    }

    /// `get_field(args...)`.
    fn get_field(args: Vec<Expr>) -> Expr {
        scalar_fn(ScalarUDF::new_from_impl(GetFieldFunc::new()), args)
    }

    /// Run `GetFieldFunc::simplify` once and return the rewritten expression,
    /// panicking if the input was left unchanged.
    fn simplified(args: Vec<Expr>) -> Expr {
        match GetFieldFunc::new()
            .simplify(args, &SimplifyContext::default())
            .unwrap()
        {
            ExprSimplifyResult::Simplified(expr) => expr,
            ExprSimplifyResult::Original(args) => {
                panic!("expected the expression to be simplified, got {args:?}")
            }
        }
    }

    /// Assert that `GetFieldFunc::simplify` leaves the arguments unchanged.
    fn assert_not_simplified(args: Vec<Expr>) {
        match GetFieldFunc::new()
            .simplify(args.clone(), &SimplifyContext::default())
            .unwrap()
        {
            ExprSimplifyResult::Original(unchanged) => assert_eq!(unchanged, args),
            ExprSimplifyResult::Simplified(expr) => {
                panic!("expected no simplification, got {expr:?}")
            }
        }
    }

    #[test]
    fn simplify_get_field_named_struct_returns_matching_value() {
        // get_field(named_struct('min', a, 'max', b), 'max') => b
        let args = vec![
            named_struct(vec![("min", col("a")), ("max", col("b"))]),
            lit_str("max"),
        ];
        assert_eq!(simplified(args), col("b"));
    }

    #[test]
    fn simplify_get_field_named_struct_first_field() {
        // get_field(named_struct('min', a, 'max', b), 'min') => a
        let args = vec![
            named_struct(vec![("min", col("a")), ("max", col("b"))]),
            lit_str("min"),
        ];
        assert_eq!(simplified(args), col("a"));
    }

    #[test]
    fn simplify_get_field_named_struct_duplicate_names_picks_first() {
        // Arrow's `column_by_name` resolves to the first match; mirror that.
        let args = vec![
            named_struct(vec![("k", col("a")), ("k", col("b"))]),
            lit_str("k"),
        ];
        assert_eq!(simplified(args), col("a"));
    }

    #[test]
    fn simplify_get_field_struct_positional() {
        // get_field(struct(a, b), 'c1') => b
        let args = vec![struct_fn(vec![col("a"), col("b")]), lit_str("c1")];
        assert_eq!(simplified(args), col("b"));
    }

    #[test]
    fn simplify_get_field_nested_named_struct() {
        // get_field(named_struct('s', named_struct('k', x)), 's', 'k')
        //   => get_field(named_struct('k', x), 'k')   (first pass)
        //   => x                                      (second pass)
        let args = vec![
            named_struct(vec![("s", named_struct(vec![("k", col("x"))]))]),
            lit_str("s"),
            lit_str("k"),
        ];
        let first_pass = simplified(args);
        let Expr::ScalarFunction(ScalarFunction { args, .. }) = first_pass else {
            panic!("expected a get_field call after the first pass")
        };
        assert_eq!(simplified(args), col("x"));
    }

    #[test]
    fn simplify_get_field_flattens_then_resolves_named_struct() {
        // get_field(get_field(named_struct('s', named_struct('k', x)), 's'), 'k')
        // flattens to get_field(named_struct(...), 's', 'k') and resolves 's'.
        let args = vec![
            get_field(vec![
                named_struct(vec![("s", named_struct(vec![("k", col("x"))]))]),
                lit_str("s"),
            ]),
            lit_str("k"),
        ];
        let expected = get_field(vec![named_struct(vec![("k", col("x"))]), lit_str("k")]);
        assert_eq!(simplified(args), expected);
    }

    #[test]
    fn simplify_get_field_dynamic_field_name_left_alone() {
        // A non-literal field name cannot be resolved at plan time.
        let args = vec![named_struct(vec![("a", col("x"))]), col("field_name")];
        assert_not_simplified(args);
    }

    #[test]
    fn simplify_get_field_null_field_name_left_alone() {
        // A NULL string literal field name resolves to no field, so the
        // `try_as_str().flatten()` guard must leave the expression untouched.
        let null_field_name = Expr::Literal(ScalarValue::Utf8(None), None);
        let args = vec![named_struct(vec![("a", col("x"))]), null_field_name];
        assert_not_simplified(args);
    }

    #[test]
    fn simplify_get_field_dynamic_struct_name_left_alone() {
        // A non-literal name inside named_struct could shadow the requested
        // field at runtime, so the rewrite must bail out entirely.
        let named_struct_with_dynamic_name = scalar_fn(
            ScalarUDF::new_from_impl(NamedStructFunc::new()),
            vec![col("dynamic_name"), col("x")],
        );
        let args = vec![named_struct_with_dynamic_name, lit_str("a")];
        assert_not_simplified(args);
    }

    #[test]
    fn simplify_get_field_missing_field_left_alone() {
        // The named_struct does not produce field 'missing'.
        let args = vec![named_struct(vec![("a", col("x"))]), lit_str("missing")];
        assert_not_simplified(args);
    }

    #[test]
    fn simplify_get_field_non_canonical_struct_field_left_alone() {
        // 'c01' is not a real field name produced by `struct(...)`.
        let args = vec![struct_fn(vec![col("a"), col("b")]), lit_str("c01")];
        assert_not_simplified(args);
    }

    #[test]
    fn simplify_get_field_column_base_left_alone() {
        // A plain column base is not a struct constructor.
        let args = vec![col("s"), lit_str("a")];
        assert_not_simplified(args);
    }
}
