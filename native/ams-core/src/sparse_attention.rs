use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, IdentityDType, RangeReader, glm_softmax, read_identity_vector};

/// Logical dimensions for one query position of sparse causal attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseAttentionShape {
    head_count: usize,
    query_key_dimension: usize,
    value_dimension: usize,
    key_count: usize,
    query_position: usize,
    selected_count: usize,
}

impl SparseAttentionShape {
    /// Validate fixed attention dimensions and causal selection capacity.
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
        selected_count: usize,
    ) -> Result<Self, AmsError> {
        if head_count == 0
            || query_key_dimension == 0
            || value_dimension == 0
            || key_count == 0
            || selected_count == 0
            || query_position >= key_count
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "sparse attention dimensions are invalid",
            ));
        }
        let causal_count = add(query_position, 1, "sparse attention causal count overflow")?;
        if selected_count > causal_count {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "sparse attention selection exceeds causal keys",
            ));
        }
        mul(
            head_count,
            query_key_dimension,
            "sparse attention query elements overflow",
        )?;
        mul(
            head_count,
            value_dimension,
            "sparse attention output elements overflow",
        )?;
        Ok(Self {
            head_count,
            query_key_dimension,
            value_dimension,
            key_count,
            query_position,
            selected_count,
        })
    }
}

/// Immutable row-major K/V storage layout for sparse attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseKvLayout {
    key_offset: u64,
    value_offset: u64,
    key_dtype: IdentityDType,
    value_dtype: IdentityDType,
}

impl SparseKvLayout {
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

/// Caller-owned scratch requirement for one sparse attention query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseAttentionScratchRequirements {
    /// Encoded bytes for the larger of one K or V vector.
    pub encoded_bytes: usize,
    /// Decoded FP64 key elements.
    pub key_elements: usize,
    /// Decoded FP64 value elements.
    pub value_elements: usize,
    /// Selected-key score elements.
    pub score_elements: usize,
    /// Selected-key probability elements.
    pub probability_elements: usize,
    /// Transactional concatenated-head output elements.
    pub output_elements: usize,
    /// Sum of all simultaneously resident scratch bytes.
    pub total_bytes: usize,
}

/// Immutable plan for range-streamed sparse causal attention at one query position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseAttentionPlan {
    shape: SparseAttentionShape,
    layout: SparseKvLayout,
    key_end: u64,
    value_end: u64,
    scratch: SparseAttentionScratchRequirements,
}

impl SparseAttentionPlan {
    /// Derive exact K/V ranges and reject an arena below the complete legal scratch set.
    ///
    /// # Errors
    ///
    /// Returns a typed planning or capacity error for overflow or insufficient arena bytes.
    pub fn from_arena(
        shape: SparseAttentionShape,
        layout: SparseKvLayout,
        arena_bytes: usize,
    ) -> Result<Self, AmsError> {
        let key_elements = mul(
            mul(
                shape.key_count,
                shape.head_count,
                "sparse attention key rows overflow",
            )?,
            shape.query_key_dimension,
            "sparse attention key elements overflow",
        )?;
        let value_elements = mul(
            mul(
                shape.key_count,
                shape.head_count,
                "sparse attention value rows overflow",
            )?,
            shape.value_dimension,
            "sparse attention value elements overflow",
        )?;
        let key_bytes = mul(
            key_elements,
            layout.key_dtype.item_bytes(),
            "sparse attention key bytes overflow",
        )?;
        let value_bytes = mul(
            value_elements,
            layout.value_dtype.item_bytes(),
            "sparse attention value bytes overflow",
        )?;
        let key_end = add_u64(
            layout.key_offset,
            usize_to_u64(key_bytes, "sparse attention key bytes exceed u64")?,
            "sparse attention key range overflow",
        )?;
        let value_end = add_u64(
            layout.value_offset,
            usize_to_u64(value_bytes, "sparse attention value bytes exceed u64")?,
            "sparse attention value range overflow",
        )?;
        let encoded_bytes = mul(
            shape.query_key_dimension,
            layout.key_dtype.item_bytes(),
            "sparse attention encoded key scratch overflow",
        )?
        .max(mul(
            shape.value_dimension,
            layout.value_dtype.item_bytes(),
            "sparse attention encoded value scratch overflow",
        )?);
        let output_elements = mul(
            shape.head_count,
            shape.value_dimension,
            "sparse attention transactional output overflow",
        )?;
        let float_elements = add(
            add(
                add(
                    add(
                        shape.query_key_dimension,
                        shape.value_dimension,
                        "sparse attention decoded vector elements overflow",
                    )?,
                    shape.selected_count,
                    "sparse attention score elements overflow",
                )?,
                shape.selected_count,
                "sparse attention probability elements overflow",
            )?,
            output_elements,
            "sparse attention output scratch elements overflow",
        )?;
        let total_bytes = add(
            encoded_bytes,
            mul(
                float_elements,
                size_of::<f64>(),
                "sparse attention FP64 scratch bytes overflow",
            )?,
            "sparse attention total scratch bytes overflow",
        )?;
        if arena_bytes < total_bytes {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "arena cannot hold sparse attention scratch",
            ));
        }
        Ok(Self {
            shape,
            layout,
            key_end,
            value_end,
            scratch: SparseAttentionScratchRequirements {
                encoded_bytes,
                key_elements: shape.query_key_dimension,
                value_elements: shape.value_dimension,
                score_elements: shape.selected_count,
                probability_elements: shape.selected_count,
                output_elements,
                total_bytes,
            },
        })
    }

    /// Exact caller-owned scratch required by the plan.
    #[must_use]
    pub const fn scratch(self) -> SparseAttentionScratchRequirements {
        self.scratch
    }
}

/// Separate immutable key and value storage objects.
pub struct SparseAttentionReaders<'a> {
    keys: &'a dyn RangeReader,
    values: &'a dyn RangeReader,
}

impl<'a> SparseAttentionReaders<'a> {
    /// Bind key and value objects without reading either object.
    #[must_use]
    pub const fn new(keys: &'a dyn RangeReader, values: &'a dyn RangeReader) -> Self {
        Self { keys, values }
    }
}

/// Caller-owned scratch for range-streamed sparse attention.
pub struct SparseAttentionScratch<'a> {
    encoded: &'a mut [u8],
    key: &'a mut [f64],
    value: &'a mut [f64],
    scores: &'a mut [f64],
    probabilities: &'a mut [f64],
    output: &'a mut [f64],
}

impl<'a> SparseAttentionScratch<'a> {
    /// Group all preallocated attention scratch regions.
    #[must_use]
    pub const fn new(
        encoded: &'a mut [u8],
        key: &'a mut [f64],
        value: &'a mut [f64],
        scores: &'a mut [f64],
        probabilities: &'a mut [f64],
        output: &'a mut [f64],
    ) -> Self {
        Self {
            encoded,
            key,
            value,
            scores,
            probabilities,
            output,
        }
    }
}

fn dimension_as_f64(value: usize) -> Result<f64, AmsError> {
    let bounded = u32::try_from(value).map_err(|_| {
        AmsError::new(
            ErrorCode::PlanInvalid,
            "sparse attention dimension exceeds supported range",
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
        mul(token, head_count, "sparse attention token row overflow")?,
        head,
        "sparse attention head row overflow",
    )?;
    let relative = mul(
        mul(row, dimension, "sparse attention row elements overflow")?,
        item_bytes,
        "sparse attention row bytes overflow",
    )?;
    add_u64(
        base,
        usize_to_u64(relative, "sparse attention vector offset exceeds u64")?,
        "sparse attention absolute vector offset overflow",
    )
}

/// Execute sparse causal attention by reading only selected K/V vectors.
///
/// Selected indices may be reused unchanged by an `IndexShare` layer. Caller output is
/// untouched until all heads and selected value reads complete successfully.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, or numeric error.
#[allow(clippy::too_many_lines)] // Preflight and transactional execution share one resource proof.
pub fn glm_sparse_attention(
    plan: SparseAttentionPlan,
    readers: &SparseAttentionReaders<'_>,
    query_heads: &[f64],
    selected_indices: &[usize],
    scratch: &mut SparseAttentionScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    let shape = plan.shape;
    let query_elements = mul(
        shape.head_count,
        shape.query_key_dimension,
        "sparse attention query elements overflow",
    )?;
    if query_heads.len() != query_elements
        || selected_indices.len() != shape.selected_count
        || output.len() != plan.scratch.output_elements
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "sparse attention input or output dimensions differ from the plan",
        ));
    }
    if query_heads.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "sparse attention query is non-finite",
        ));
    }
    for (position, index) in selected_indices.iter().copied().enumerate() {
        if index > shape.query_position || selected_indices[..position].contains(&index) {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "sparse attention indices are noncausal or duplicated",
            ));
        }
    }
    let requirement = plan.scratch;
    if scratch.encoded.len() < requirement.encoded_bytes
        || scratch.key.len() < requirement.key_elements
        || scratch.value.len() < requirement.value_elements
        || scratch.scores.len() < requirement.score_elements
        || scratch.probabilities.len() < requirement.probability_elements
        || scratch.output.len() < requirement.output_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "sparse attention scratch is smaller than the admitted plan",
        ));
    }
    if plan.key_end > readers.keys.len() || plan.value_end > readers.values.len() {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "sparse attention K/V range exceeds its storage object",
        ));
    }
    let key = &mut scratch.key[..requirement.key_elements];
    let value = &mut scratch.value[..requirement.value_elements];
    let scores = &mut scratch.scores[..requirement.score_elements];
    let probabilities = &mut scratch.probabilities[..requirement.probability_elements];
    let transactional_output = &mut scratch.output[..requirement.output_elements];
    transactional_output.fill(0.0);
    let scale = 1.0 / dimension_as_f64(shape.query_key_dimension)?.sqrt();
    for head in 0..shape.head_count {
        let query_start = head * shape.query_key_dimension;
        let query = &query_heads[query_start..query_start + shape.query_key_dimension];
        for (score, token) in scores.iter_mut().zip(selected_indices.iter().copied()) {
            let offset = vector_offset(
                plan.layout.key_offset,
                token,
                head,
                shape.head_count,
                shape.query_key_dimension,
                plan.layout.key_dtype.item_bytes(),
            )?;
            read_identity_vector(
                readers.keys,
                offset,
                plan.layout.key_dtype,
                key,
                scratch.encoded,
            )?;
            let mut dot = 0.0;
            for (query_value, key_value) in query.iter().zip(key.iter()) {
                dot += query_value * key_value;
            }
            *score = dot * scale;
            if !score.is_finite() {
                return Err(AmsError::new(
                    ErrorCode::NumericFailure,
                    "sparse attention score is non-finite",
                ));
            }
        }
        glm_softmax(scores, probabilities)?;
        let output_start = head * shape.value_dimension;
        let head_output =
            &mut transactional_output[output_start..output_start + shape.value_dimension];
        for (probability, token) in probabilities.iter().zip(selected_indices.iter().copied()) {
            let offset = vector_offset(
                plan.layout.value_offset,
                token,
                head,
                shape.head_count,
                shape.value_dimension,
                plan.layout.value_dtype.item_bytes(),
            )?;
            read_identity_vector(
                readers.values,
                offset,
                plan.layout.value_dtype,
                value,
                scratch.encoded,
            )?;
            for (destination, source) in head_output.iter_mut().zip(value.iter()) {
                *destination += probability * source;
                if !destination.is_finite() {
                    return Err(AmsError::new(
                        ErrorCode::NumericFailure,
                        "sparse attention output is non-finite",
                    ));
                }
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

    #[test]
    #[allow(clippy::suboptimal_flops, clippy::too_many_lines)]
    // One fixture proves causal preflight, selective reads, parity, and transactional failure.
    fn sparse_attention_reads_only_selected_causal_kv() -> Result<(), AmsError> {
        let keys = CountingReader::new(&[
            1.0, 0.0, 0.0, 1.0, // token 0
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
        let shape = SparseAttentionShape::new(2, 2, 2, 4, 2, 2)?;
        let layout = SparseKvLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32);
        let plan = SparseAttentionPlan::from_arena(shape, layout, 104)?;
        assert_eq!(plan.scratch().total_bytes, 104);
        let readers = SparseAttentionReaders::new(&keys, &values);
        let mut encoded = [0u8; 8];
        let mut key = [0.0f64; 2];
        let mut value = [0.0f64; 2];
        let mut scores = [0.0f64; 2];
        let mut probabilities = [0.0f64; 2];
        let mut transactional_output = [0.0f64; 4];
        let mut scratch = SparseAttentionScratch::new(
            &mut encoded,
            &mut key,
            &mut value,
            &mut scores,
            &mut probabilities,
            &mut transactional_output,
        );

        let mut rejected_output = [7.0f64; 4];
        let error = glm_sparse_attention(
            plan,
            &readers,
            &[1.0, 0.0, 0.0, 1.0],
            &[3, 0],
            &mut scratch,
            &mut rejected_output,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        assert_eq!(keys.reads.get(), 0);
        assert_eq!(values.reads.get(), 0);
        assert!(
            rejected_output
                .iter()
                .all(|value| value.to_bits() == 7.0f64.to_bits())
        );

        let mut output = [0.0f64; 4];
        glm_sparse_attention(
            plan,
            &readers,
            &[1.0, 0.0, 0.0, 1.0],
            &[2, 0],
            &mut scratch,
            &mut output,
        )?;
        let scale = 1.0 / 2.0f64.sqrt();
        let high = (2.0 * scale).exp();
        let low = scale.exp();
        let denominator = high + low;
        let high_probability = high / denominator;
        let low_probability = low / denominator;
        let expected = [
            high_probability * 3.0 + low_probability,
            high_probability * 30.0 + low_probability * 10.0,
            high_probability * 4.0 + low_probability * 2.0,
            high_probability * 40.0 + low_probability * 20.0,
        ];
        assert!(
            output
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() <= 1e-14)
        );
        assert_eq!(keys.reads.get(), 4);
        assert_eq!(values.reads.get(), 4);
        assert!(keys.offsets.borrow().iter().all(|offset| *offset < 48));
        assert!(values.offsets.borrow().iter().all(|offset| *offset < 48));

        let bad_values = CountingReader::new(&[
            1.0,
            10.0,
            2.0,
            20.0,
            9.0,
            90.0,
            8.0,
            80.0,
            f32::NAN,
            30.0,
            4.0,
            40.0,
            100.0,
            1000.0,
            200.0,
            2000.0,
        ]);
        let bad_readers = SparseAttentionReaders::new(&keys, &bad_values);
        let mut rejected_output = [7.0f64; 4];
        let error = glm_sparse_attention(
            plan,
            &bad_readers,
            &[1.0, 0.0, 0.0, 1.0],
            &[2, 0],
            &mut scratch,
            &mut rejected_output,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert!(
            rejected_output
                .iter()
                .all(|value| value.to_bits() == 7.0f64.to_bits())
        );
        Ok(())
    }

    #[test]
    fn sparse_attention_rejects_subminimum_arena() -> Result<(), AmsError> {
        let shape = SparseAttentionShape::new(2, 2, 2, 4, 2, 2)?;
        let layout = SparseKvLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32);
        let error = SparseAttentionPlan::from_arena(shape, layout, 103).err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        Ok(())
    }
}
