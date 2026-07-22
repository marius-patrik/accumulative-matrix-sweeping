use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, IdentityDType, RangeReader, read_identity_vector};

/// Logical dimensions for one query position of full causal attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullAttentionShape {
    head_count: usize,
    query_key_dimension: usize,
    value_dimension: usize,
    key_count: usize,
    query_position: usize,
}

impl FullAttentionShape {
    /// Validate fixed attention dimensions and the causal query position.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for zero, inconsistent, or overflowing dimensions.
    pub fn new(
        head_count: usize,
        query_key_dimension: usize,
        value_dimension: usize,
        key_count: usize,
        query_position: usize,
    ) -> Result<Self, AmsError> {
        if head_count == 0
            || query_key_dimension == 0
            || value_dimension == 0
            || key_count == 0
            || query_position >= key_count
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "full attention dimensions are invalid",
            ));
        }
        mul(
            head_count,
            query_key_dimension,
            "full attention query elements overflow",
        )?;
        mul(
            head_count,
            value_dimension,
            "full attention output elements overflow",
        )?;
        Ok(Self {
            head_count,
            query_key_dimension,
            value_dimension,
            key_count,
            query_position,
        })
    }
}

/// Immutable packed row-major K/V storage layout for full attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullKvLayout {
    key_offset: u64,
    value_offset: u64,
    key_dtype: IdentityDType,
    value_dtype: IdentityDType,
}

impl FullKvLayout {
    /// Bind base offsets and reviewed uncompressed dtypes for separate K/V objects.
    #[must_use]
    pub const fn new(
        key_offset: u64,
        value_offset: u64,
        key_dtype: IdentityDType,
        value_dtype: IdentityDType,
    ) -> Self {
        Self {
            key_offset,
            value_offset,
            key_dtype,
            value_dtype,
        }
    }
}

/// Context-independent caller-owned scratch for one full attention query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullAttentionScratchRequirements {
    /// Encoded bytes for the larger of one K or V vector.
    pub encoded_bytes: usize,
    /// Decoded FP64 key elements.
    pub key_elements: usize,
    /// Decoded FP64 value elements.
    pub value_elements: usize,
    /// Transactional concatenated-head output elements.
    pub output_elements: usize,
    /// Sum of all simultaneously resident caller-owned scratch bytes.
    pub total_bytes: usize,
}

/// Immutable plan for range-streamed full causal attention at one query position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FullAttentionPlan {
    shape: FullAttentionShape,
    layout: FullKvLayout,
    key_end: u64,
    value_end: u64,
    scratch: FullAttentionScratchRequirements,
}

impl FullAttentionPlan {
    /// Derive exact K/V ranges and reject an arena below the complete legal scratch set.
    ///
    /// # Errors
    ///
    /// Returns a typed planning or capacity error for overflow or insufficient arena bytes.
    pub fn from_arena(
        shape: FullAttentionShape,
        layout: FullKvLayout,
        arena_bytes: usize,
    ) -> Result<Self, AmsError> {
        let key_elements = mul(
            mul(
                shape.key_count,
                shape.head_count,
                "full attention key rows overflow",
            )?,
            shape.query_key_dimension,
            "full attention key elements overflow",
        )?;
        let value_elements = mul(
            mul(
                shape.key_count,
                shape.head_count,
                "full attention value rows overflow",
            )?,
            shape.value_dimension,
            "full attention value elements overflow",
        )?;
        let key_bytes = mul(
            key_elements,
            layout.key_dtype.item_bytes(),
            "full attention key bytes overflow",
        )?;
        let value_bytes = mul(
            value_elements,
            layout.value_dtype.item_bytes(),
            "full attention value bytes overflow",
        )?;
        let key_end = add_u64(
            layout.key_offset,
            usize_to_u64(key_bytes, "full attention key bytes exceed u64")?,
            "full attention key range overflow",
        )?;
        let value_end = add_u64(
            layout.value_offset,
            usize_to_u64(value_bytes, "full attention value bytes exceed u64")?,
            "full attention value range overflow",
        )?;
        let encoded_bytes = mul(
            shape.query_key_dimension,
            layout.key_dtype.item_bytes(),
            "full attention encoded key scratch overflow",
        )?
        .max(mul(
            shape.value_dimension,
            layout.value_dtype.item_bytes(),
            "full attention encoded value scratch overflow",
        )?);
        let output_elements = mul(
            shape.head_count,
            shape.value_dimension,
            "full attention transactional output overflow",
        )?;
        let float_elements = add(
            add(
                shape.query_key_dimension,
                shape.value_dimension,
                "full attention decoded vector elements overflow",
            )?,
            output_elements,
            "full attention output scratch elements overflow",
        )?;
        let total_bytes = add(
            encoded_bytes,
            mul(
                float_elements,
                size_of::<f64>(),
                "full attention FP64 scratch bytes overflow",
            )?,
            "full attention total scratch bytes overflow",
        )?;
        if arena_bytes < total_bytes {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "arena cannot hold full attention scratch",
            ));
        }
        Ok(Self {
            shape,
            layout,
            key_end,
            value_end,
            scratch: FullAttentionScratchRequirements {
                encoded_bytes,
                key_elements: shape.query_key_dimension,
                value_elements: shape.value_dimension,
                output_elements,
                total_bytes,
            },
        })
    }

    /// Exact context-independent caller-owned scratch required by the plan.
    #[must_use]
    pub const fn scratch(self) -> FullAttentionScratchRequirements {
        self.scratch
    }
}

/// Separate immutable key and value storage objects.
pub struct FullAttentionReaders<'a> {
    keys: &'a dyn RangeReader,
    values: &'a dyn RangeReader,
}

impl<'a> FullAttentionReaders<'a> {
    /// Bind key and value objects without reading either object.
    #[must_use]
    pub const fn new(keys: &'a dyn RangeReader, values: &'a dyn RangeReader) -> Self {
        Self { keys, values }
    }
}

/// Caller-owned scratch for range-streamed full attention.
pub struct FullAttentionScratch<'a> {
    encoded: &'a mut [u8],
    key: &'a mut [f64],
    value: &'a mut [f64],
    output: &'a mut [f64],
}

impl<'a> FullAttentionScratch<'a> {
    /// Group all preallocated attention scratch regions.
    #[must_use]
    pub const fn new(
        encoded: &'a mut [u8],
        key: &'a mut [f64],
        value: &'a mut [f64],
        output: &'a mut [f64],
    ) -> Self {
        Self {
            encoded,
            key,
            value,
            output,
        }
    }

    pub(crate) const fn admits(&self, requirement: FullAttentionScratchRequirements) -> bool {
        self.encoded.len() >= requirement.encoded_bytes
            && self.key.len() >= requirement.key_elements
            && self.value.len() >= requirement.value_elements
            && self.output.len() >= requirement.output_elements
    }
}

fn dimension_as_f64(value: usize) -> Result<f64, AmsError> {
    let bounded = u32::try_from(value).map_err(|_| {
        AmsError::new(
            ErrorCode::PlanInvalid,
            "full attention dimension exceeds supported range",
        )
    })?;
    Ok(f64::from(bounded))
}

fn vector_offset(
    base: u64,
    token: usize,
    head: usize,
    head_count: usize,
    dimension: usize,
    item_bytes: usize,
) -> Result<u64, AmsError> {
    let row = add(
        mul(token, head_count, "full attention token row overflow")?,
        head,
        "full attention head row overflow",
    )?;
    let relative = mul(
        mul(row, dimension, "full attention row elements overflow")?,
        item_bytes,
        "full attention row bytes overflow",
    )?;
    add_u64(
        base,
        usize_to_u64(relative, "full attention vector offset exceeds u64")?,
        "full attention absolute vector offset overflow",
    )
}

/// Execute full causal attention with a one-pass online softmax over range-streamed K/V.
///
/// Scratch is independent of causal context length. Caller output is untouched until all
/// heads and causal K/V reads complete successfully.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, or numeric error.
#[allow(clippy::suboptimal_flops, clippy::too_many_lines)]
// Preflight and source-order transactional execution share one resource proof.
pub fn glm_full_attention(
    plan: FullAttentionPlan,
    readers: &FullAttentionReaders<'_>,
    query_heads: &[f64],
    scratch: &mut FullAttentionScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    let shape = plan.shape;
    let query_elements = mul(
        shape.head_count,
        shape.query_key_dimension,
        "full attention query elements overflow",
    )?;
    if query_heads.len() != query_elements || output.len() != plan.scratch.output_elements {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "full attention input or output dimensions differ from the plan",
        ));
    }
    if query_heads.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "full attention query is non-finite",
        ));
    }
    let requirement = plan.scratch;
    if scratch.encoded.len() < requirement.encoded_bytes
        || scratch.key.len() < requirement.key_elements
        || scratch.value.len() < requirement.value_elements
        || scratch.output.len() < requirement.output_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "full attention scratch is smaller than the admitted plan",
        ));
    }
    if plan.key_end > readers.keys.len() || plan.value_end > readers.values.len() {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "full attention K/V range exceeds its storage object",
        ));
    }
    let key = &mut scratch.key[..requirement.key_elements];
    let value = &mut scratch.value[..requirement.value_elements];
    let transactional_output = &mut scratch.output[..requirement.output_elements];
    transactional_output.fill(0.0);
    let scale = 1.0 / dimension_as_f64(shape.query_key_dimension)?.sqrt();
    for head in 0..shape.head_count {
        let query_start = head * shape.query_key_dimension;
        let query = &query_heads[query_start..query_start + shape.query_key_dimension];
        let output_start = head * shape.value_dimension;
        let head_output =
            &mut transactional_output[output_start..output_start + shape.value_dimension];
        let mut running_max = f64::NEG_INFINITY;
        let mut denominator = 0.0f64;
        for token in 0..=shape.query_position {
            let key_offset = vector_offset(
                plan.layout.key_offset,
                token,
                head,
                shape.head_count,
                shape.query_key_dimension,
                plan.layout.key_dtype.item_bytes(),
            )?;
            read_identity_vector(
                readers.keys,
                key_offset,
                plan.layout.key_dtype,
                key,
                scratch.encoded,
            )?;
            let mut dot = 0.0;
            for (query_value, key_value) in query.iter().zip(key.iter()) {
                dot += query_value * key_value;
            }
            let score = dot * scale;
            if !score.is_finite() {
                return Err(AmsError::new(
                    ErrorCode::NumericFailure,
                    "full attention score is non-finite",
                ));
            }
            let next_max = running_max.max(score);
            let previous_scale = if running_max.is_finite() {
                (running_max - next_max).exp()
            } else {
                0.0
            };
            let current_scale = (score - next_max).exp();
            let next_denominator = denominator * previous_scale + current_scale;

            let value_offset = vector_offset(
                plan.layout.value_offset,
                token,
                head,
                shape.head_count,
                shape.value_dimension,
                plan.layout.value_dtype.item_bytes(),
            )?;
            read_identity_vector(
                readers.values,
                value_offset,
                plan.layout.value_dtype,
                value,
                scratch.encoded,
            )?;
            for (destination, source) in head_output.iter_mut().zip(value.iter()) {
                *destination = *destination * previous_scale + current_scale * source;
                if !destination.is_finite() {
                    return Err(AmsError::new(
                        ErrorCode::NumericFailure,
                        "full attention output is non-finite",
                    ));
                }
            }
            running_max = next_max;
            denominator = next_denominator;
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "full attention softmax denominator is invalid",
            ));
        }
        for destination in head_output.iter_mut() {
            *destination /= denominator;
            if !destination.is_finite() {
                return Err(AmsError::new(
                    ErrorCode::NumericFailure,
                    "full attention normalized output is non-finite",
                ));
            }
        }
    }
    output.copy_from_slice(transactional_output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;

    struct CountingReader {
        bytes: Vec<u8>,
        reads: Cell<usize>,
        offsets: RefCell<Vec<u64>>,
    }

    impl CountingReader {
        fn new(values: &[f32]) -> Self {
            Self {
                bytes: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
                reads: Cell::new(0),
                offsets: RefCell::new(Vec::new()),
            }
        }
    }

    impl RangeReader for CountingReader {
        fn len(&self) -> u64 {
            u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
            let start = usize::try_from(offset).map_err(|_| {
                AmsError::new(ErrorCode::IoFailure, "test K/V offset exceeds usize")
            })?;
            let end = start
                .checked_add(destination.len())
                .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "test K/V range overflow"))?;
            let source = self.bytes.get(start..end).ok_or_else(|| {
                AmsError::new(ErrorCode::IoFailure, "test K/V range exceeds object")
            })?;
            destination.copy_from_slice(source);
            self.reads.set(self.reads.get().saturating_add(1));
            self.offsets.borrow_mut().push(offset);
            Ok(())
        }
    }

    #[allow(clippy::suboptimal_flops)] // Mirror the explicit source-order semantic oracle.
    fn reference_attention(
        query: &[f64],
        keys: &[f32],
        values: &[f32],
        causal_count: usize,
        key_dimension: usize,
        value_dimension: usize,
    ) -> Vec<f64> {
        let scale = 1.0 / f64::from(u32::try_from(key_dimension).unwrap_or(u32::MAX)).sqrt();
        let mut scores = Vec::new();
        for token in 0..causal_count {
            let mut dot = 0.0;
            for column in 0..key_dimension {
                dot += query[column] * f64::from(keys[token * key_dimension + column]);
            }
            scores.push(dot * scale);
        }
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator: f64 = scores.iter().map(|score| (*score - maximum).exp()).sum();
        let mut output = vec![0.0; value_dimension];
        for (token, score) in scores.iter().enumerate() {
            let probability = (*score - maximum).exp() / denominator;
            for column in 0..value_dimension {
                output[column] += probability * f64::from(values[token * value_dimension + column]);
            }
        }
        output
    }

    #[test]
    fn full_attention_streams_only_the_causal_prefix_with_constant_scratch() -> Result<(), AmsError>
    {
        let keys = CountingReader::new(&[
            1.0, 0.0, 0.0, 1.0, // token 0, heads 0 and 1
            0.0, 1.0, 1.0, 0.0, // token 1
            2.0, 0.0, 0.0, 2.0, // token 2
            100.0, 100.0, 100.0, 100.0, // future token 3
        ]);
        let values = CountingReader::new(&[
            1.0, 10.0, 2.0, 20.0, // token 0
            9.0, 90.0, 8.0, 80.0, // token 1
            3.0, 30.0, 4.0, 40.0, // token 2
            100.0, 1000.0, 200.0, 2000.0, // future token 3
        ]);
        let shape = FullAttentionShape::new(2, 2, 2, 4, 2)?;
        let layout = FullKvLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32);
        let plan = FullAttentionPlan::from_arena(shape, layout, 72)?;
        assert_eq!(plan.scratch().total_bytes, 72);
        let readers = FullAttentionReaders::new(&keys, &values);
        let mut encoded = [0u8; 8];
        let mut key = [0.0f64; 2];
        let mut value = [0.0f64; 2];
        let mut transactional = [0.0f64; 4];
        let mut scratch =
            FullAttentionScratch::new(&mut encoded, &mut key, &mut value, &mut transactional);
        let mut output = [99.0f64; 4];
        glm_full_attention(
            plan,
            &readers,
            &[1.0, 0.5, 0.25, 1.0],
            &mut scratch,
            &mut output,
        )?;
        assert_eq!(keys.reads.get(), 6);
        assert_eq!(values.reads.get(), 6);
        assert!(keys.offsets.borrow().iter().all(|offset| *offset < 48));
        assert!(values.offsets.borrow().iter().all(|offset| *offset < 48));
        let head_zero = reference_attention(
            &[1.0, 0.5],
            &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0],
            &[1.0, 10.0, 9.0, 90.0, 3.0, 30.0],
            3,
            2,
            2,
        );
        let head_one = reference_attention(
            &[0.25, 1.0],
            &[0.0, 1.0, 1.0, 0.0, 0.0, 2.0],
            &[2.0, 20.0, 8.0, 80.0, 4.0, 40.0],
            3,
            2,
            2,
        );
        for (actual, expected) in output.iter().zip(head_zero.iter().chain(head_one.iter())) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        Ok(())
    }

    #[test]
    fn full_attention_rejects_capacity_and_ranges_before_io() -> Result<(), AmsError> {
        let shape = FullAttentionShape::new(1, 2, 2, 2, 1)?;
        let layout = FullKvLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32);
        let plan = FullAttentionPlan::from_arena(shape, layout, 56)?;
        assert_eq!(
            FullAttentionPlan::from_arena(shape, layout, 55)
                .err()
                .map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        let keys = CountingReader::new(&[1.0, 0.0, 0.0, 1.0]);
        let short_values = CountingReader::new(&[1.0, 2.0]);
        let readers = FullAttentionReaders::new(&keys, &short_values);
        let mut encoded = [0u8; 8];
        let mut key = [0.0f64; 2];
        let mut value = [0.0f64; 2];
        let mut transactional = [0.0f64; 2];
        let mut scratch =
            FullAttentionScratch::new(&mut encoded, &mut key, &mut value, &mut transactional);
        let mut output = [7.0f64; 2];
        let error =
            glm_full_attention(plan, &readers, &[1.0, 0.0], &mut scratch, &mut output).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        assert_eq!(keys.reads.get(), 0);
        assert_eq!(short_values.reads.get(), 0);
        assert_eq!(output.map(f64::to_bits), [7.0f64.to_bits(); 2]);
        Ok(())
    }

    #[test]
    fn online_attention_matches_two_pass_softmax_and_commits_transactionally()
    -> Result<(), AmsError> {
        let key_values = [1.0f32, 0.0, 0.5, 1.0, -1.0, 2.0, 4.0, -2.0];
        let value_values = [1.0f32, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0];
        for query_position in 0..4 {
            let keys = CountingReader::new(&key_values);
            let values = CountingReader::new(&value_values);
            let shape = FullAttentionShape::new(1, 2, 2, 4, query_position)?;
            let layout = FullKvLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32);
            let plan = FullAttentionPlan::from_arena(shape, layout, 56)?;
            let readers = FullAttentionReaders::new(&keys, &values);
            let mut encoded = [0u8; 8];
            let mut key = [0.0f64; 2];
            let mut value = [0.0f64; 2];
            let mut transactional = [0.0f64; 2];
            let mut scratch =
                FullAttentionScratch::new(&mut encoded, &mut key, &mut value, &mut transactional);
            let mut output = [0.0f64; 2];
            glm_full_attention(plan, &readers, &[0.75, -0.25], &mut scratch, &mut output)?;
            let expected = reference_attention(
                &[0.75, -0.25],
                &key_values,
                &value_values,
                query_position + 1,
                2,
                2,
            );
            for (actual, reference) in output.iter().zip(expected.iter()) {
                assert!((actual - reference).abs() <= 1e-12);
            }
        }

        let keys = CountingReader::new(&key_values);
        let bad_values = CountingReader::new(&[1.0, 2.0, f32::NAN, 5.0, 8.0, 13.0, 21.0, 34.0]);
        let shape = FullAttentionShape::new(1, 2, 2, 4, 1)?;
        let layout = FullKvLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32);
        let plan = FullAttentionPlan::from_arena(shape, layout, 56)?;
        let readers = FullAttentionReaders::new(&keys, &bad_values);
        let mut encoded = [0u8; 8];
        let mut key = [0.0f64; 2];
        let mut value = [0.0f64; 2];
        let mut transactional = [0.0f64; 2];
        let mut scratch =
            FullAttentionScratch::new(&mut encoded, &mut key, &mut value, &mut transactional);
        let mut output = [77.0f64; 2];
        let error =
            glm_full_attention(plan, &readers, &[0.75, -0.25], &mut scratch, &mut output).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert_eq!(output.map(f64::to_bits), [77.0f64.to_bits(); 2]);
        Ok(())
    }
}
