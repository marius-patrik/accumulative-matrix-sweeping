use std::cmp::Ordering;

use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{AmsError, ErrorCode, IdentityDType, RangeReader, read_identity_vector};

/// Logical dimensions for one range-streamed causal DSA selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamedDsaShape {
    head_count: usize,
    head_dimension: usize,
    key_count: usize,
    query_position: usize,
    selected_count: usize,
}

impl StreamedDsaShape {
    /// Validate DSA dimensions and clamp top-k to the causal prefix.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for zero, inconsistent, or overflowing dimensions.
    pub fn new(
        head_count: usize,
        head_dimension: usize,
        key_count: usize,
        query_position: usize,
        top_k: usize,
    ) -> Result<Self, AmsError> {
        if head_count == 0
            || head_dimension == 0
            || key_count == 0
            || top_k == 0
            || query_position >= key_count
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "streamed DSA dimensions are invalid",
            ));
        }
        mul(
            head_count,
            head_dimension,
            "streamed DSA query elements overflow",
        )?;
        let causal_count = add(query_position, 1, "streamed DSA causal count overflow")?;
        Ok(Self {
            head_count,
            head_dimension,
            key_count,
            query_position,
            selected_count: top_k.min(causal_count),
        })
    }
}

/// Immutable row-major index-key storage layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamedDsaLayout {
    key_offset: u64,
    key_dtype: IdentityDType,
}

impl StreamedDsaLayout {
    /// Bind an index-key object base offset and reviewed identity dtype.
    #[must_use]
    pub const fn new(key_offset: u64, key_dtype: IdentityDType) -> Self {
        Self {
            key_offset,
            key_dtype,
        }
    }
}

/// Caller-owned scratch requirement for a range-streamed DSA scan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamedDsaScratchRequirements {
    /// Encoded bytes for one index-key vector.
    pub encoded_bytes: usize,
    /// Decoded FP64 index-key elements.
    pub key_elements: usize,
    /// Retained top-k FP64 scores.
    pub score_elements: usize,
    /// Retained top-k key indices.
    pub index_elements: usize,
    /// Sum of all simultaneously resident scratch bytes.
    pub total_bytes: usize,
}

/// Immutable plan for DSA top-k without context-sized score residency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamedDsaTopKPlan {
    shape: StreamedDsaShape,
    layout: StreamedDsaLayout,
    key_end: u64,
    scratch: StreamedDsaScratchRequirements,
}

impl StreamedDsaTopKPlan {
    /// Derive the complete index-key range and exact bounded scratch requirement.
    ///
    /// # Errors
    ///
    /// Returns a typed planning or capacity error for overflow or insufficient arena bytes.
    pub fn from_arena(
        shape: StreamedDsaShape,
        layout: StreamedDsaLayout,
        arena_bytes: usize,
    ) -> Result<Self, AmsError> {
        let key_elements = mul(
            shape.key_count,
            shape.head_dimension,
            "streamed DSA key elements overflow",
        )?;
        let key_bytes = mul(
            key_elements,
            layout.key_dtype.item_bytes(),
            "streamed DSA key bytes overflow",
        )?;
        let key_end = add_u64(
            layout.key_offset,
            usize_to_u64(key_bytes, "streamed DSA key bytes exceed u64")?,
            "streamed DSA key range overflow",
        )?;
        let encoded_bytes = mul(
            shape.head_dimension,
            layout.key_dtype.item_bytes(),
            "streamed DSA encoded scratch overflow",
        )?;
        let key_scratch_bytes = mul(
            shape.head_dimension,
            size_of::<f64>(),
            "streamed DSA decoded scratch overflow",
        )?;
        let score_bytes = mul(
            shape.selected_count,
            size_of::<f64>(),
            "streamed DSA score scratch overflow",
        )?;
        let index_bytes = mul(
            shape.selected_count,
            size_of::<usize>(),
            "streamed DSA index scratch overflow",
        )?;
        let total_bytes = add(
            add(
                add(
                    encoded_bytes,
                    key_scratch_bytes,
                    "streamed DSA key scratch total overflow",
                )?,
                score_bytes,
                "streamed DSA score total overflow",
            )?,
            index_bytes,
            "streamed DSA index total overflow",
        )?;
        if arena_bytes < total_bytes {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "arena cannot hold streamed DSA scratch",
            ));
        }
        Ok(Self {
            shape,
            layout,
            key_end,
            scratch: StreamedDsaScratchRequirements {
                encoded_bytes,
                key_elements: shape.head_dimension,
                score_elements: shape.selected_count,
                index_elements: shape.selected_count,
                total_bytes,
            },
        })
    }

    /// Exact caller-owned scratch required by the plan.
    #[must_use]
    pub const fn scratch(self) -> StreamedDsaScratchRequirements {
        self.scratch
    }
}

/// Caller-owned scratch for a range-streamed DSA scan.
pub struct StreamedDsaScratch<'a> {
    encoded: &'a mut [u8],
    key: &'a mut [f64],
    scores: &'a mut [f64],
    indices: &'a mut [usize],
}

impl<'a> StreamedDsaScratch<'a> {
    /// Group all preallocated DSA scan scratch regions.
    #[must_use]
    pub const fn new(
        encoded: &'a mut [u8],
        key: &'a mut [f64],
        scores: &'a mut [f64],
        indices: &'a mut [usize],
    ) -> Self {
        Self {
            encoded,
            key,
            scores,
            indices,
        }
    }
}

fn dimension_as_f64(value: usize) -> Result<f64, AmsError> {
    let bounded = u32::try_from(value).map_err(|_| {
        AmsError::new(
            ErrorCode::PlanInvalid,
            "streamed DSA dimension exceeds supported range",
        )
    })?;
    Ok(f64::from(bounded))
}

fn key_offset(plan: StreamedDsaTopKPlan, key_index: usize) -> Result<u64, AmsError> {
    let relative = mul(
        mul(
            key_index,
            plan.shape.head_dimension,
            "streamed DSA key row elements overflow",
        )?,
        plan.layout.key_dtype.item_bytes(),
        "streamed DSA key row bytes overflow",
    )?;
    add_u64(
        plan.layout.key_offset,
        usize_to_u64(relative, "streamed DSA key offset exceeds u64")?,
        "streamed DSA absolute key offset overflow",
    )
}

fn insert_score(
    score: f64,
    key_index: usize,
    filled: usize,
    scores: &mut [f64],
    indices: &mut [usize],
) -> Result<usize, AmsError> {
    let capacity = scores.len();
    let mut insertion = filled;
    for position in 0..filled {
        let ordering = score.partial_cmp(&scores[position]).ok_or_else(|| {
            AmsError::new(
                ErrorCode::NumericFailure,
                "streamed DSA score comparison failed",
            )
        })?;
        if ordering == Ordering::Greater
            || (ordering == Ordering::Equal && key_index < indices[position])
        {
            insertion = position;
            break;
        }
    }
    if insertion >= capacity {
        return Ok(filled);
    }
    let new_filled = add(filled, 1, "streamed DSA retained count overflow")?.min(capacity);
    for position in (insertion + 1..new_filled).rev() {
        scores[position] = scores[position - 1];
        indices[position] = indices[position - 1];
    }
    scores[insertion] = score;
    indices[insertion] = key_index;
    Ok(new_filled)
}

/// Scan causal offloaded index keys while retaining only deterministic top-k state.
///
/// Caller output is untouched until the entire causal key range has been read and ranked.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, or numeric error.
#[allow(clippy::suboptimal_flops)] // Preserve the Python semantic oracle's operation order.
pub fn glm_streamed_dsa_topk(
    plan: StreamedDsaTopKPlan,
    reader: &dyn RangeReader,
    query_heads: &[f64],
    head_weights: &[f64],
    scratch: &mut StreamedDsaScratch<'_>,
    selected: &mut [usize],
) -> Result<(), AmsError> {
    let shape = plan.shape;
    let query_elements = mul(
        shape.head_count,
        shape.head_dimension,
        "streamed DSA query elements overflow",
    )?;
    if query_heads.len() != query_elements
        || head_weights.len() != shape.head_count
        || selected.len() != shape.selected_count
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "streamed DSA input or output dimensions differ from the plan",
        ));
    }
    if query_heads.iter().any(|value| !value.is_finite())
        || head_weights.iter().any(|value| !value.is_finite())
    {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "streamed DSA query or head weight is non-finite",
        ));
    }
    let requirement = plan.scratch;
    if scratch.encoded.len() < requirement.encoded_bytes
        || scratch.key.len() < requirement.key_elements
        || scratch.scores.len() < requirement.score_elements
        || scratch.indices.len() < requirement.index_elements
    {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "streamed DSA scratch is smaller than the admitted plan",
        ));
    }
    if plan.key_end > reader.len() {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "streamed DSA key range exceeds its storage object",
        ));
    }
    let key = &mut scratch.key[..requirement.key_elements];
    let scores = &mut scratch.scores[..requirement.score_elements];
    let indices = &mut scratch.indices[..requirement.index_elements];
    let scale = 1.0 / dimension_as_f64(shape.head_dimension)?.sqrt();
    let mut filled = 0usize;
    for key_index in 0..=shape.query_position {
        read_identity_vector(
            reader,
            key_offset(plan, key_index)?,
            plan.layout.key_dtype,
            key,
            scratch.encoded,
        )?;
        let mut score = 0.0;
        for (head_index, query) in query_heads.chunks_exact(shape.head_dimension).enumerate() {
            let mut similarity = 0.0;
            for (query_value, key_value) in query.iter().zip(key.iter()) {
                similarity += query_value * key_value;
            }
            score += head_weights[head_index] * (similarity * scale).max(0.0);
        }
        if !score.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "streamed DSA score is non-finite",
            ));
        }
        filled = insert_score(score, key_index, filled, scores, indices)?;
    }
    if filled != shape.selected_count {
        return Err(AmsError::new(
            ErrorCode::InternalInvariant,
            "streamed DSA retained count differs from the plan",
        ));
    }
    selected.copy_from_slice(indices);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use super::*;
    use crate::{DsaTopKPlan, glm_dsa_topk};

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
                AmsError::new(ErrorCode::IoFailure, "test DSA offset exceeds usize")
            })?;
            let end = start
                .checked_add(destination.len())
                .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "test DSA range overflow"))?;
            let source = self.bytes.get(start..end).ok_or_else(|| {
                AmsError::new(ErrorCode::IoFailure, "test DSA range exceeds object")
            })?;
            destination.copy_from_slice(source);
            self.reads.set(self.reads.get().saturating_add(1));
            self.offsets.borrow_mut().push(offset);
            Ok(())
        }
    }

    #[test]
    fn streamed_dsa_matches_reference_without_context_sized_scores() -> Result<(), AmsError> {
        let reader = CountingReader::new(&[
            1.0, 1.0, // key 0
            1.0, 1.0, // key 1, tied with key 0
            3.0, 0.0, // key 2
            100.0, 100.0, // future key 3
        ]);
        let shape = StreamedDsaShape::new(2, 2, 4, 2, 3)?;
        let layout = StreamedDsaLayout::new(0, IdentityDType::Float32);
        let plan = StreamedDsaTopKPlan::from_arena(shape, layout, 72)?;
        assert_eq!(plan.scratch().total_bytes, 72);
        let mut encoded = [0u8; 8];
        let mut key = [0.0f64; 2];
        let mut scores = [0.0f64; 3];
        let mut indices = [usize::MAX; 3];
        let mut scratch =
            StreamedDsaScratch::new(&mut encoded, &mut key, &mut scores, &mut indices);
        let mut selected = [usize::MAX; 3];
        glm_streamed_dsa_topk(
            plan,
            &reader,
            &[1.0, 0.0, 0.0, 1.0],
            &[0.5, 0.5],
            &mut scratch,
            &mut selected,
        )?;
        assert_eq!(selected, [2, 0, 1]);
        assert_eq!(reader.reads.get(), 3);
        assert!(reader.offsets.borrow().iter().all(|offset| *offset < 24));
        Ok(())
    }

    #[test]
    fn streamed_dsa_rejects_short_scratch_before_io() -> Result<(), AmsError> {
        let reader = CountingReader::new(&[1.0, 1.0]);
        let shape = StreamedDsaShape::new(1, 2, 1, 0, 1)?;
        let layout = StreamedDsaLayout::new(0, IdentityDType::Float32);
        let plan = StreamedDsaTopKPlan::from_arena(shape, layout, 40)?;
        let mut encoded = [0u8; 7];
        let mut key = [0.0f64; 2];
        let mut scores = [0.0f64; 1];
        let mut indices = [usize::MAX; 1];
        let mut scratch =
            StreamedDsaScratch::new(&mut encoded, &mut key, &mut scores, &mut indices);
        let mut selected = [usize::MAX; 1];
        let error = glm_streamed_dsa_topk(
            plan,
            &reader,
            &[1.0, 0.0],
            &[1.0],
            &mut scratch,
            &mut selected,
        )
        .err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::PreflightNoWorkingSet)
        );
        assert_eq!(reader.reads.get(), 0);
        Ok(())
    }

    #[test]
    fn streamed_dsa_matches_in_memory_oracle_across_causal_prefixes() -> Result<(), AmsError> {
        let query_heads = [
            1.0, -0.5, 0.25, 2.0, -1.0, 0.5, 1.5, -0.25, 0.75, 1.0, -2.0, 0.5,
        ];
        let stored_keys = [
            1.0f32, 0.0, -1.0, 0.5, 0.5, 1.0, 0.0, -0.5, 2.0, -1.0, 0.5, 0.0, -1.0, -1.0, 1.0, 1.0,
            0.25, 0.5, 0.75, 1.0, 4.0, 4.0, 4.0, 4.0,
        ];
        let keys = stored_keys.map(f64::from);
        let reader = CountingReader::new(&stored_keys);
        for query_position in 0..5 {
            let reference_plan = DsaTopKPlan::new(3, 4, 6, query_position, 3)?;
            let selected_count = reference_plan.selected_count();
            let mut reference_scores = [0.0f64; 5];
            let mut reference_selected = [usize::MAX; 3];
            glm_dsa_topk(
                reference_plan,
                &query_heads,
                &keys,
                &[0.5, -0.25, 1.0],
                &mut reference_scores[..reference_plan.score_scratch_len()],
                &mut reference_selected[..selected_count],
            )?;

            let shape = StreamedDsaShape::new(3, 4, 6, query_position, 3)?;
            let layout = StreamedDsaLayout::new(0, IdentityDType::Float32);
            let streamed_plan = StreamedDsaTopKPlan::from_arena(shape, layout, 96)?;
            let mut encoded = [0u8; 16];
            let mut key = [0.0f64; 4];
            let mut scores = [0.0f64; 3];
            let mut indices = [usize::MAX; 3];
            let mut scratch =
                StreamedDsaScratch::new(&mut encoded, &mut key, &mut scores, &mut indices);
            let mut actual = [usize::MAX; 3];
            glm_streamed_dsa_topk(
                streamed_plan,
                &reader,
                &query_heads,
                &[0.5, -0.25, 1.0],
                &mut scratch,
                &mut actual[..selected_count],
            )?;
            assert_eq!(
                actual[..selected_count],
                reference_selected[..selected_count]
            );
        }
        Ok(())
    }
}
