use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{
    AmsError, ErrorCode, IdentityDType, LinearPlan, LinearScratch, LinearScratchRequirements,
    RangeReader, glm_rms_norm, glm_rope_interleaved, read_identity_vector, stream_linear,
};

/// Immutable identity layouts for the two MLA low-rank normalization vectors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4MlaNormLayout {
    q_offset: u64,
    kv_offset: u64,
    q_dtype: IdentityDType,
    kv_dtype: IdentityDType,
}

impl Glm4MlaNormLayout {
    /// Bind the Q and KV normalization vector ranges.
    #[must_use]
    pub const fn new(
        q_offset: u64,
        kv_offset: u64,
        q_dtype: IdentityDType,
        kv_dtype: IdentityDType,
    ) -> Self {
        Self {
            q_offset,
            kv_offset,
            q_dtype,
            kv_dtype,
        }
    }
}

/// Exact caller-owned scratch for one GLM-4-MoE-Lite MLA projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4MlaScratchRequirements {
    /// Reusable mixed-storage linear scratch.
    pub linear: LinearScratchRequirements,
    /// Encoded bytes for the larger normalization vector.
    pub norm_encoded_bytes: usize,
    /// Decoded FP64 normalization-weight elements.
    pub norm_weight_elements: usize,
    /// Q low-rank projection elements.
    pub q_a_elements: usize,
    /// Compressed KV plus shared rotary-key elements.
    pub kv_a_elements: usize,
    /// Reusable normalized low-rank elements.
    pub normalized_elements: usize,
    /// Expanded Q projection elements.
    pub q_projected_elements: usize,
    /// Expanded nonrotary K and V projection elements.
    pub kv_projected_elements: usize,
    /// Transactional concatenated-head query elements.
    pub query_output_elements: usize,
    /// Transactional concatenated-head key elements.
    pub key_output_elements: usize,
    /// Transactional concatenated-head value elements.
    pub value_output_elements: usize,
    /// Sum of all simultaneously resident scratch and nested local bytes.
    pub total_bytes: usize,
}

/// Immutable mixed-storage plan for GLM-4-MoE-Lite MLA projections at one token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm4MlaPlan {
    q_a: LinearPlan,
    q_b: LinearPlan,
    kv_a: LinearPlan,
    kv_b: LinearPlan,
    norm_layout: Glm4MlaNormLayout,
    q_norm_end: u64,
    kv_norm_end: u64,
    hidden_elements: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    head_count: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    value_head_dim: usize,
    rms_norm_epsilon: f64,
    rope_theta: f64,
    scratch: Glm4MlaScratchRequirements,
}

impl Glm4MlaPlan {
    /// Validate the four MLA matrix shapes and derive the complete scratch high-water.
    ///
    /// # Errors
    ///
    /// Returns a typed plan error for inconsistent dimensions, numeric constants, or overflow.
    #[allow(
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    pub fn new(
        q_a: LinearPlan,
        q_b: LinearPlan,
        kv_a: LinearPlan,
        kv_b: LinearPlan,
        norm_layout: Glm4MlaNormLayout,
        head_count: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        value_head_dim: usize,
        rms_norm_epsilon: f64,
        rope_theta: f64,
    ) -> Result<Self, AmsError> {
        if head_count == 0
            || qk_nope_head_dim == 0
            || qk_rope_head_dim == 0
            || qk_rope_head_dim % 2 != 0
            || value_head_dim == 0
            || !rms_norm_epsilon.is_finite()
            || rms_norm_epsilon <= 0.0
            || !rope_theta.is_finite()
            || rope_theta <= 0.0
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 MLA dimensions or numeric constants are invalid",
            ));
        }
        let hidden_elements = q_a.columns();
        let q_lora_rank = q_a.rows();
        if kv_a.columns() != hidden_elements || q_b.columns() != q_lora_rank {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 MLA Q/KV input or Q low-rank dimensions differ",
            ));
        }
        if kv_a.rows() <= qk_rope_head_dim {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 MLA compressed KV projection is too small",
            ));
        }
        let kv_lora_rank = kv_a.rows() - qk_rope_head_dim;
        if kv_b.columns() != kv_lora_rank {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 MLA KV low-rank dimensions differ",
            ));
        }
        let qk_head_dim = add(
            qk_nope_head_dim,
            qk_rope_head_dim,
            "GLM-4 MLA QK head dimension overflow",
        )?;
        let query_output_elements = mul(
            head_count,
            qk_head_dim,
            "GLM-4 MLA query projection elements overflow",
        )?;
        let key_output_elements = query_output_elements;
        let kv_per_head = add(
            qk_nope_head_dim,
            value_head_dim,
            "GLM-4 MLA KV head dimension overflow",
        )?;
        let kv_projected_elements = mul(
            head_count,
            kv_per_head,
            "GLM-4 MLA KV projection elements overflow",
        )?;
        let value_output_elements = mul(
            head_count,
            value_head_dim,
            "GLM-4 MLA value projection elements overflow",
        )?;
        if q_b.rows() != query_output_elements || kv_b.rows() != kv_projected_elements {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 MLA expanded projection dimensions differ from the head layout",
            ));
        }

        let q_norm_bytes = mul(
            q_lora_rank,
            norm_layout.q_dtype.item_bytes(),
            "GLM-4 MLA Q norm bytes overflow",
        )?;
        let kv_norm_bytes = mul(
            kv_lora_rank,
            norm_layout.kv_dtype.item_bytes(),
            "GLM-4 MLA KV norm bytes overflow",
        )?;
        let q_norm_end = add_u64(
            norm_layout.q_offset,
            usize_to_u64(q_norm_bytes, "GLM-4 MLA Q norm bytes exceed u64")?,
            "GLM-4 MLA Q norm range overflow",
        )?;
        let kv_norm_end = add_u64(
            norm_layout.kv_offset,
            usize_to_u64(kv_norm_bytes, "GLM-4 MLA KV norm bytes exceed u64")?,
            "GLM-4 MLA KV norm range overflow",
        )?;
        let linear = q_a
            .scratch()
            .union(q_b.scratch())?
            .union(kv_a.scratch())?
            .union(kv_b.scratch())?;
        let norm_weight_elements = q_lora_rank.max(kv_lora_rank);
        let normalized_elements = norm_weight_elements.max(qk_rope_head_dim);
        let float_elements = [
            norm_weight_elements,
            q_lora_rank,
            kv_a.rows(),
            normalized_elements,
            query_output_elements,
            kv_projected_elements,
            query_output_elements,
            key_output_elements,
            value_output_elements,
        ]
        .into_iter()
        .try_fold(0usize, |total, count| {
            add(total, count, "GLM-4 MLA scratch elements overflow")
        })?;
        let norm_encoded_bytes = q_norm_bytes.max(kv_norm_bytes);
        let total_bytes = add(
            add(
                linear.total_bytes,
                norm_encoded_bytes,
                "GLM-4 MLA encoded scratch overflow",
            )?,
            mul(
                float_elements,
                size_of::<f64>(),
                "GLM-4 MLA FP64 scratch bytes overflow",
            )?,
            "GLM-4 MLA total scratch bytes overflow",
        )?;
        Ok(Self {
            q_a,
            q_b,
            kv_a,
            kv_b,
            norm_layout,
            q_norm_end,
            kv_norm_end,
            hidden_elements,
            q_lora_rank,
            kv_lora_rank,
            head_count,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            value_head_dim,
            rms_norm_epsilon,
            rope_theta,
            scratch: Glm4MlaScratchRequirements {
                linear,
                norm_encoded_bytes,
                norm_weight_elements,
                q_a_elements: q_lora_rank,
                kv_a_elements: kv_a.rows(),
                normalized_elements,
                q_projected_elements: query_output_elements,
                kv_projected_elements,
                query_output_elements,
                key_output_elements,
                value_output_elements,
                total_bytes,
            },
        })
    }

    /// Exact logical caller-owned scratch required by this plan.
    #[must_use]
    pub const fn scratch(&self) -> Glm4MlaScratchRequirements {
        self.scratch
    }

    /// Decoder hidden width consumed by both low-rank projections.
    #[must_use]
    pub const fn hidden_elements(&self) -> usize {
        self.hidden_elements
    }

    /// Number of expanded attention heads.
    #[must_use]
    pub const fn head_count(&self) -> usize {
        self.head_count
    }

    /// Per-head concatenated nonrotary and rotary Q/K width.
    #[must_use]
    pub const fn qk_head_dim(&self) -> usize {
        self.qk_head_dim
    }

    /// Per-head value width.
    #[must_use]
    pub const fn value_head_dim(&self) -> usize {
        self.value_head_dim
    }

    /// Concatenated Q/K elements emitted for one token.
    #[must_use]
    pub const fn query_key_output_elements(&self) -> usize {
        self.scratch.query_output_elements
    }

    /// Concatenated value elements emitted for one token.
    #[must_use]
    pub const fn value_output_elements(&self) -> usize {
        self.scratch.value_output_elements
    }
}

/// Six immutable storage objects used by one MLA projection plan.
pub struct Glm4MlaReaders<'a> {
    q_a: &'a dyn RangeReader,
    q_norm: &'a dyn RangeReader,
    q_b: &'a dyn RangeReader,
    kv_a: &'a dyn RangeReader,
    kv_norm: &'a dyn RangeReader,
    kv_b: &'a dyn RangeReader,
}

impl<'a> Glm4MlaReaders<'a> {
    /// Bind MLA storage readers in execution order without reading them.
    #[must_use]
    pub const fn new(
        q_a: &'a dyn RangeReader,
        q_norm: &'a dyn RangeReader,
        q_b: &'a dyn RangeReader,
        kv_a: &'a dyn RangeReader,
        kv_norm: &'a dyn RangeReader,
        kv_b: &'a dyn RangeReader,
    ) -> Self {
        Self {
            q_a,
            q_norm,
            q_b,
            kv_a,
            kv_norm,
            kv_b,
        }
    }
}

/// Caller-owned buffers for mixed-storage MLA projection.
pub struct Glm4MlaScratch<'a> {
    linear: LinearScratch<'a>,
    norm_encoded: &'a mut [u8],
    norm_weights: &'a mut [f64],
    q_a: &'a mut [f64],
    kv_a: &'a mut [f64],
    normalized: &'a mut [f64],
    q_projected: &'a mut [f64],
    kv_projected: &'a mut [f64],
    query_output: &'a mut [f64],
    key_output: &'a mut [f64],
    value_output: &'a mut [f64],
}

impl<'a> Glm4MlaScratch<'a> {
    /// Group every preallocated MLA scratch region under one borrow.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        linear: LinearScratch<'a>,
        norm_encoded: &'a mut [u8],
        norm_weights: &'a mut [f64],
        q_a: &'a mut [f64],
        kv_a: &'a mut [f64],
        normalized: &'a mut [f64],
        q_projected: &'a mut [f64],
        kv_projected: &'a mut [f64],
        query_output: &'a mut [f64],
        key_output: &'a mut [f64],
        value_output: &'a mut [f64],
    ) -> Self {
        Self {
            linear,
            norm_encoded,
            norm_weights,
            q_a,
            kv_a,
            normalized,
            q_projected,
            kv_projected,
            query_output,
            key_output,
            value_output,
        }
    }

    pub(crate) const fn admits(&self, requirement: Glm4MlaScratchRequirements) -> bool {
        self.linear.admits(requirement.linear)
            && self.norm_encoded.len() >= requirement.norm_encoded_bytes
            && self.norm_weights.len() >= requirement.norm_weight_elements
            && self.q_a.len() >= requirement.q_a_elements
            && self.kv_a.len() >= requirement.kv_a_elements
            && self.normalized.len() >= requirement.normalized_elements
            && self.q_projected.len() >= requirement.q_projected_elements
            && self.kv_projected.len() >= requirement.kv_projected_elements
            && self.query_output.len() >= requirement.query_output_elements
            && self.key_output.len() >= requirement.key_output_elements
            && self.value_output.len() >= requirement.value_output_elements
    }
}

fn readers_admit(plan: &Glm4MlaPlan, readers: &Glm4MlaReaders<'_>) -> bool {
    [
        (readers.q_a, plan.q_a.reader_end()),
        (readers.q_norm, plan.q_norm_end),
        (readers.q_b, plan.q_b.reader_end()),
        (readers.kv_a, plan.kv_a.reader_end()),
        (readers.kv_norm, plan.kv_norm_end),
        (readers.kv_b, plan.kv_b.reader_end()),
    ]
    .into_iter()
    .all(|(reader, end)| end <= reader.len())
}

/// Execute GLM-4-MoE-Lite MLA Q/K/V projection directly from mixed storage.
///
/// The operator applies both low-rank `RMSNorms`, the four matrix projections, and
/// interleaved `RoPE`. Q/K/V outputs remain untouched until the complete projection succeeds.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn glm4_mla_project(
    plan: &Glm4MlaPlan,
    readers: &Glm4MlaReaders<'_>,
    hidden: &[f64],
    position: usize,
    scratch: &mut Glm4MlaScratch<'_>,
    query_output: &mut [f64],
    key_output: &mut [f64],
    value_output: &mut [f64],
) -> Result<(), AmsError> {
    let requirement = plan.scratch;
    if hidden.len() != plan.hidden_elements
        || query_output.len() != requirement.query_output_elements
        || key_output.len() != requirement.key_output_elements
        || value_output.len() != requirement.value_output_elements
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 MLA input or output dimensions differ from the plan",
        ));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "GLM-4 MLA hidden state is non-finite",
        ));
    }
    if !scratch.admits(requirement) {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "GLM-4 MLA scratch is smaller than the admitted plan",
        ));
    }
    if !readers_admit(plan, readers) {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "GLM-4 MLA weight range exceeds its storage object",
        ));
    }

    let q_a = &mut scratch.q_a[..requirement.q_a_elements];
    stream_linear(
        readers.q_a,
        plan.q_a,
        hidden,
        None,
        &mut scratch.linear,
        q_a,
    )?;
    let q_norm_weights = &mut scratch.norm_weights[..plan.q_lora_rank];
    read_identity_vector(
        readers.q_norm,
        plan.norm_layout.q_offset,
        plan.norm_layout.q_dtype,
        q_norm_weights,
        scratch.norm_encoded,
    )?;
    let normalized = &mut scratch.normalized[..plan.q_lora_rank];
    glm_rms_norm(q_a, q_norm_weights, plan.rms_norm_epsilon, normalized)?;
    let q_projected = &mut scratch.q_projected[..requirement.q_projected_elements];
    stream_linear(
        readers.q_b,
        plan.q_b,
        normalized,
        None,
        &mut scratch.linear,
        q_projected,
    )?;

    let kv_a = &mut scratch.kv_a[..requirement.kv_a_elements];
    stream_linear(
        readers.kv_a,
        plan.kv_a,
        hidden,
        None,
        &mut scratch.linear,
        kv_a,
    )?;
    let kv_norm_weights = &mut scratch.norm_weights[..plan.kv_lora_rank];
    read_identity_vector(
        readers.kv_norm,
        plan.norm_layout.kv_offset,
        plan.norm_layout.kv_dtype,
        kv_norm_weights,
        scratch.norm_encoded,
    )?;
    let normalized = &mut scratch.normalized[..plan.kv_lora_rank];
    glm_rms_norm(
        &kv_a[..plan.kv_lora_rank],
        kv_norm_weights,
        plan.rms_norm_epsilon,
        normalized,
    )?;
    let kv_projected = &mut scratch.kv_projected[..requirement.kv_projected_elements];
    stream_linear(
        readers.kv_b,
        plan.kv_b,
        normalized,
        None,
        &mut scratch.linear,
        kv_projected,
    )?;

    let transactional_query = &mut scratch.query_output[..requirement.query_output_elements];
    let transactional_key = &mut scratch.key_output[..requirement.key_output_elements];
    let transactional_value = &mut scratch.value_output[..requirement.value_output_elements];
    let shared_rotary_key = &mut scratch.normalized[..plan.qk_rope_head_dim];
    glm_rope_interleaved(
        &kv_a[plan.kv_lora_rank..],
        position,
        plan.rope_theta,
        shared_rotary_key,
    )?;
    let kv_per_head = plan.qk_nope_head_dim + plan.value_head_dim;
    for head in 0..plan.head_count {
        let q_source_start = head * plan.qk_head_dim;
        let q_target_start = q_source_start;
        transactional_query[q_target_start..q_target_start + plan.qk_nope_head_dim]
            .copy_from_slice(&q_projected[q_source_start..q_source_start + plan.qk_nope_head_dim]);
        glm_rope_interleaved(
            &q_projected[q_source_start + plan.qk_nope_head_dim..q_source_start + plan.qk_head_dim],
            position,
            plan.rope_theta,
            &mut transactional_query
                [q_target_start + plan.qk_nope_head_dim..q_target_start + plan.qk_head_dim],
        )?;

        let kv_source_start = head * kv_per_head;
        let key_target_start = head * plan.qk_head_dim;
        transactional_key[key_target_start..key_target_start + plan.qk_nope_head_dim]
            .copy_from_slice(
                &kv_projected[kv_source_start..kv_source_start + plan.qk_nope_head_dim],
            );
        transactional_key
            [key_target_start + plan.qk_nope_head_dim..key_target_start + plan.qk_head_dim]
            .copy_from_slice(shared_rotary_key);
        let value_target_start = head * plan.value_head_dim;
        transactional_value[value_target_start..value_target_start + plan.value_head_dim]
            .copy_from_slice(
                &kv_projected
                    [kv_source_start + plan.qk_nope_head_dim..kv_source_start + kv_per_head],
            );
    }
    query_output.copy_from_slice(transactional_query);
    key_output.copy_from_slice(transactional_key);
    value_output.copy_from_slice(transactional_value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{IdentityLinearPlan, SliceReader};

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn identity_plan(rows: usize, columns: usize) -> Result<LinearPlan, AmsError> {
        Ok(IdentityLinearPlan::from_arena(rows, columns, 0, 16, IdentityDType::Float32)?.into())
    }

    fn fixture_plan() -> Result<Glm4MlaPlan, AmsError> {
        Glm4MlaPlan::new(
            identity_plan(2, 2)?,
            identity_plan(6, 2)?,
            identity_plan(4, 2)?,
            identity_plan(4, 2)?,
            Glm4MlaNormLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            2,
            1,
            2,
            1,
            1e-5,
            10_000.0,
        )
    }

    #[allow(clippy::type_complexity)]
    fn fixture_payloads() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
        (
            encode_f32(&[1.0, 0.0, 0.0, 1.0]),
            encode_f32(&[1.0, 2.0]),
            encode_f32(&[
                1.0, 0.0, // head 0 nonrotary Q
                0.0, 1.0, // head 0 rotary Q row 0
                1.0, 1.0, // head 0 rotary Q row 1
                -1.0, 0.0, // head 1 nonrotary Q
                1.0, -1.0, // head 1 rotary Q row 0
                0.5, 0.0, // head 1 rotary Q row 1
            ]),
            encode_f32(&[
                1.0, 0.0, // compressed KV 0
                0.0, 1.0, // compressed KV 1
                1.0, 1.0, // shared rotary K 0
                -1.0, 1.0, // shared rotary K 1
            ]),
            encode_f32(&[0.5, 1.5]),
            encode_f32(&[
                1.0, 0.0, // head 0 nonrotary K
                0.0, 1.0, // head 0 V
                1.0, 1.0, // head 1 nonrotary K
                -1.0, 1.0, // head 1 V
            ]),
        )
    }

    fn reference_rms(values: &[f64], weights: &[f64], epsilon: f64) -> Vec<f64> {
        let variance = values.iter().map(|value| value * value).sum::<f64>()
            / f64::from(u32::try_from(values.len()).unwrap_or(u32::MAX));
        let inverse = 1.0 / (variance + epsilon).sqrt();
        values
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * inverse * weight)
            .collect()
    }

    #[allow(clippy::suboptimal_flops)]
    fn reference_matvec(matrix: &[f32], rows: usize, columns: usize, input: &[f64]) -> Vec<f64> {
        let mut output = vec![0.0; rows];
        for row in 0..rows {
            for column in 0..columns {
                output[row] += f64::from(matrix[row * columns + column]) * input[column];
            }
        }
        output
    }

    #[allow(clippy::suboptimal_flops)] // Mirror the explicit source-order RoPE oracle.
    fn reference_rope(values: &[f64], position: usize) -> [f64; 2] {
        let angle = f64::from(u32::try_from(position).unwrap_or(u32::MAX));
        let cosine = angle.cos();
        let sine = angle.sin();
        [
            values[0] * cosine - values[1] * sine,
            values[1] * cosine + values[0] * sine,
        ]
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn mla_projector_matches_source_order_reference_with_exact_scratch() -> Result<(), AmsError> {
        let plan = fixture_plan()?;
        assert_eq!(plan.scratch().total_bytes, 296);
        let (q_a, q_norm, q_b, kv_a, kv_norm, kv_b) = fixture_payloads();
        let q_a_reader = SliceReader::new(&q_a);
        let q_norm_reader = SliceReader::new(&q_norm);
        let q_b_reader = SliceReader::new(&q_b);
        let kv_a_reader = SliceReader::new(&kv_a);
        let kv_norm_reader = SliceReader::new(&kv_norm);
        let kv_b_reader = SliceReader::new(&kv_b);
        let readers = Glm4MlaReaders::new(
            &q_a_reader,
            &q_norm_reader,
            &q_b_reader,
            &kv_a_reader,
            &kv_norm_reader,
            &kv_b_reader,
        );
        let mut linear_encoded = [0u8; 8];
        let mut linear_decoded = [0.0f32; 0];
        let mut linear_accumulators = [0.0f64; 0];
        let linear = LinearScratch::new(
            &mut linear_encoded,
            &mut linear_decoded,
            &mut linear_accumulators,
        );
        let mut norm_encoded = [0u8; 8];
        let mut norm_weights = [0.0f64; 2];
        let mut q_a_scratch = [0.0f64; 2];
        let mut kv_a_scratch = [0.0f64; 4];
        let mut normalized = [0.0f64; 2];
        let mut q_projected = [0.0f64; 6];
        let mut kv_projected = [0.0f64; 4];
        let mut query_transactional = [0.0f64; 6];
        let mut key_transactional = [0.0f64; 6];
        let mut value_transactional = [0.0f64; 2];
        let mut scratch = Glm4MlaScratch::new(
            linear,
            &mut norm_encoded,
            &mut norm_weights,
            &mut q_a_scratch,
            &mut kv_a_scratch,
            &mut normalized,
            &mut q_projected,
            &mut kv_projected,
            &mut query_transactional,
            &mut key_transactional,
            &mut value_transactional,
        );
        let mut query = [99.0f64; 6];
        let mut key = [99.0f64; 6];
        let mut value = [99.0f64; 2];
        glm4_mla_project(
            &plan,
            &readers,
            &[1.0, 2.0],
            3,
            &mut scratch,
            &mut query,
            &mut key,
            &mut value,
        )?;

        let q_low = reference_rms(&[1.0, 2.0], &[1.0, 2.0], 1e-5);
        let q_matrix = [
            1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.5, 0.0,
        ];
        let q_expanded = reference_matvec(&q_matrix, 6, 2, &q_low);
        let kv_raw = [1.0, 2.0, 3.0, 1.0];
        let kv_low = reference_rms(&kv_raw[..2], &[0.5, 1.5], 1e-5);
        let kv_matrix = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
        let kv_expanded = reference_matvec(&kv_matrix, 4, 2, &kv_low);
        let shared_rope = reference_rope(&kv_raw[2..], 3);
        let mut expected_query = Vec::new();
        for head in 0..2 {
            let start = head * 3;
            expected_query.push(q_expanded[start]);
            expected_query.extend(reference_rope(&q_expanded[start + 1..start + 3], 3));
        }
        let expected_key = [
            kv_expanded[0],
            shared_rope[0],
            shared_rope[1],
            kv_expanded[2],
            shared_rope[0],
            shared_rope[1],
        ];
        let expected_value = [kv_expanded[1], kv_expanded[3]];
        for (actual, expected) in query.iter().zip(expected_query) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        for (actual, expected) in key.iter().zip(expected_key) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        for (actual, expected) in value.iter().zip(expected_value) {
            assert!((actual - expected).abs() <= 1e-12);
        }
        Ok(())
    }

    struct CountingReader {
        bytes: Vec<u8>,
        reads: Cell<usize>,
    }

    impl CountingReader {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                bytes,
                reads: Cell::new(0),
            }
        }
    }

    impl RangeReader for CountingReader {
        fn len(&self) -> u64 {
            u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
            self.reads.set(self.reads.get().saturating_add(1));
            SliceReader::new(&self.bytes).read_exact_at(offset, destination)
        }
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn mla_preflights_every_reader_and_commits_outputs_transactionally() -> Result<(), AmsError> {
        let plan = fixture_plan()?;
        let (q_a, q_norm, q_b, kv_a, kv_norm, kv_b) = fixture_payloads();
        let q_a = CountingReader::new(q_a);
        let q_norm = CountingReader::new(q_norm);
        let q_b = CountingReader::new(q_b[..q_b.len() - 1].to_vec());
        let kv_a = CountingReader::new(kv_a);
        let kv_norm = CountingReader::new(kv_norm);
        let kv_b = CountingReader::new(kv_b);
        let readers = Glm4MlaReaders::new(&q_a, &q_norm, &q_b, &kv_a, &kv_norm, &kv_b);
        let mut linear_encoded = [0u8; 8];
        let mut linear_decoded = [0.0f32; 0];
        let mut linear_accumulators = [0.0f64; 0];
        let linear = LinearScratch::new(
            &mut linear_encoded,
            &mut linear_decoded,
            &mut linear_accumulators,
        );
        let mut norm_encoded = [0u8; 8];
        let mut norm_weights = [0.0f64; 2];
        let mut q_a_scratch = [0.0f64; 2];
        let mut kv_a_scratch = [0.0f64; 4];
        let mut normalized = [0.0f64; 2];
        let mut q_projected = [0.0f64; 6];
        let mut kv_projected = [0.0f64; 4];
        let mut query_transactional = [0.0f64; 6];
        let mut key_transactional = [0.0f64; 6];
        let mut value_transactional = [0.0f64; 2];
        let mut scratch = Glm4MlaScratch::new(
            linear,
            &mut norm_encoded,
            &mut norm_weights,
            &mut q_a_scratch,
            &mut kv_a_scratch,
            &mut normalized,
            &mut q_projected,
            &mut kv_projected,
            &mut query_transactional,
            &mut key_transactional,
            &mut value_transactional,
        );
        let mut query = [9.0f64; 6];
        let mut key = [9.0f64; 6];
        let mut value = [9.0f64; 2];
        let error = glm4_mla_project(
            &plan,
            &readers,
            &[1.0, 2.0],
            3,
            &mut scratch,
            &mut query,
            &mut key,
            &mut value,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        assert_eq!(
            [
                q_a.reads.get(),
                q_norm.reads.get(),
                q_b.reads.get(),
                kv_a.reads.get(),
                kv_norm.reads.get(),
                kv_b.reads.get(),
            ],
            [0; 6]
        );
        assert_eq!(query.map(f64::to_bits), [9.0f64.to_bits(); 6]);
        assert_eq!(key.map(f64::to_bits), [9.0f64.to_bits(); 6]);
        assert_eq!(value.map(f64::to_bits), [9.0f64.to_bits(); 2]);
        Ok(())
    }

    #[test]
    fn mla_plan_rejects_projection_shape_drift() -> Result<(), AmsError> {
        let wide_rope = Glm4MlaPlan::new(
            identity_plan(1, 2)?,
            identity_plan(3, 1)?,
            identity_plan(3, 2)?,
            identity_plan(2, 1)?,
            Glm4MlaNormLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            1,
            1,
            2,
            1,
            1e-5,
            10_000.0,
        )?;
        assert_eq!(wide_rope.scratch().normalized_elements, 2);
        let error = Glm4MlaPlan::new(
            identity_plan(2, 2)?,
            identity_plan(5, 2)?,
            identity_plan(4, 2)?,
            identity_plan(4, 2)?,
            Glm4MlaNormLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            2,
            1,
            2,
            1,
            1e-5,
            10_000.0,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        Ok(())
    }
}
