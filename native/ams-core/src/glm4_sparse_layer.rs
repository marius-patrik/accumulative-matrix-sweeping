use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::glm4_mla::readers_admit as mla_readers_admit;
use crate::{
    AmsError, ErrorCode, FullAttentionPlan, FullAttentionReaders, FullAttentionScratch,
    FullAttentionScratchRequirements, FullAttentionShape, FullKvLayout, Glm4MlaPlan,
    Glm4MlaReaders, Glm4MlaScratch, Glm4MlaScratchRequirements, IdentityDType, KvCache,
    KvCachePlan, LinearPlan, LinearScratch, LinearScratchRequirements, RangeReader,
    SparseMoeBindings, SparseMoePlan, SparseMoeScratch, SparseMoeScratchRequirements,
    glm_full_attention, glm_rms_norm, glm_sparse_moe, glm4_mla_project, read_identity_vector,
    stream_linear,
};

/// Identity layouts for the norms and router correction bias of one sparse decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4SparseLayerVectorLayout {
    input_norm_offset: u64,
    post_attention_norm_offset: u64,
    correction_bias_offset: u64,
    input_norm_dtype: IdentityDType,
    post_attention_norm_dtype: IdentityDType,
    correction_bias_dtype: IdentityDType,
}

impl Glm4SparseLayerVectorLayout {
    /// Bind the two decoder norms and the expert-selection correction vector.
    #[must_use]
    pub const fn new(
        input_norm_offset: u64,
        post_attention_norm_offset: u64,
        correction_bias_offset: u64,
        input_norm_dtype: IdentityDType,
        post_attention_norm_dtype: IdentityDType,
        correction_bias_dtype: IdentityDType,
    ) -> Self {
        Self {
            input_norm_offset,
            post_attention_norm_offset,
            correction_bias_offset,
            input_norm_dtype,
            post_attention_norm_dtype,
            correction_bias_dtype,
        }
    }
}

/// Exact caller-owned working set for one sparse GLM-4 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4SparseLayerScratchRequirements {
    /// Nested MLA projection scratch.
    pub mla: Glm4MlaScratchRequirements,
    /// One staged K/V token row.
    pub cache_staging_bytes: usize,
    /// Nested context-independent full-attention scratch.
    pub attention: FullAttentionScratchRequirements,
    /// Output-projection linear scratch.
    pub output_linear: LinearScratchRequirements,
    /// Routed-plus-shared sparse-MoE scratch.
    pub moe: SparseMoeScratchRequirements,
    /// Encoded bytes for the larger decoder normalization vector.
    pub norm_encoded_bytes: usize,
    /// Encoded router correction-bias bytes.
    pub correction_encoded_bytes: usize,
    /// Decoder-width normalization weights.
    pub norm_weight_elements: usize,
    /// Routed-expert correction-bias elements.
    pub correction_bias_elements: usize,
    /// Decoder-width normalized hidden state.
    pub normalized_elements: usize,
    /// Concatenated MLA query elements.
    pub query_elements: usize,
    /// Concatenated MLA key elements.
    pub key_elements: usize,
    /// Concatenated MLA value elements.
    pub value_elements: usize,
    /// Concatenated-head attention output elements.
    pub attention_output_elements: usize,
    /// Output projection elements.
    pub output_projection_elements: usize,
    /// First residual elements.
    pub residual_elements: usize,
    /// Post-attention normalized elements.
    pub post_normalized_elements: usize,
    /// Sparse-MoE output elements.
    pub moe_output_elements: usize,
    /// Transactional final layer output elements.
    pub final_output_elements: usize,
    /// Complete caller-owned working set including nested local bytes.
    pub total_bytes: usize,
}

/// Immutable plan for a sparse-MLP GLM-4-MoE-Lite decoder layer.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm4SparseLayerPlan {
    mla: Glm4MlaPlan,
    output_projection: LinearPlan,
    moe: SparseMoePlan,
    cache: KvCachePlan,
    vector_layout: Glm4SparseLayerVectorLayout,
    input_norm_end: u64,
    post_attention_norm_end: u64,
    correction_bias_end: u64,
    rms_norm_epsilon: f64,
    scratch: Glm4SparseLayerScratchRequirements,
}

impl Glm4SparseLayerPlan {
    /// Validate the complete sparse decoder-layer graph and derive its fixed working set.
    ///
    /// # Errors
    ///
    /// Returns a typed plan error for graph, cache, range, numeric, or size disagreement.
    #[allow(
        clippy::similar_names,
        clippy::suspicious_operation_groupings,
        clippy::too_many_lines
    )]
    pub fn new(
        mla: Glm4MlaPlan,
        output_projection: LinearPlan,
        moe: SparseMoePlan,
        cache: KvCachePlan,
        vector_layout: Glm4SparseLayerVectorLayout,
        rms_norm_epsilon: f64,
    ) -> Result<Self, AmsError> {
        if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 sparse layer RMSNorm epsilon is invalid",
            ));
        }
        let hidden = mla.hidden_elements();
        let expert_count = moe.expert_count();
        if cache.head_count() != mla.head_count()
            || cache.key_head_dimension() != mla.qk_head_dim()
            || cache.value_head_dimension() != mla.value_head_dim()
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 sparse layer cache dimensions differ from MLA",
            ));
        }
        if output_projection.rows() != hidden
            || output_projection.columns() != mla.value_output_elements()
            || moe.hidden_elements() != hidden
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 sparse layer projection or MoE dimensions differ",
            ));
        }
        let input_norm_bytes = mul(
            hidden,
            vector_layout.input_norm_dtype.item_bytes(),
            "GLM-4 sparse input norm bytes overflow",
        )?;
        let post_attention_norm_bytes = mul(
            hidden,
            vector_layout.post_attention_norm_dtype.item_bytes(),
            "GLM-4 sparse post-attention norm bytes overflow",
        )?;
        let correction_bias_bytes = mul(
            expert_count,
            vector_layout.correction_bias_dtype.item_bytes(),
            "GLM-4 sparse correction-bias bytes overflow",
        )?;
        let input_norm_end = add_u64(
            vector_layout.input_norm_offset,
            usize_to_u64(input_norm_bytes, "GLM-4 sparse input norm bytes exceed u64")?,
            "GLM-4 sparse input norm range overflow",
        )?;
        let post_attention_norm_end = add_u64(
            vector_layout.post_attention_norm_offset,
            usize_to_u64(
                post_attention_norm_bytes,
                "GLM-4 sparse post-attention norm bytes exceed u64",
            )?,
            "GLM-4 sparse post-attention norm range overflow",
        )?;
        let correction_bias_end = add_u64(
            vector_layout.correction_bias_offset,
            usize_to_u64(
                correction_bias_bytes,
                "GLM-4 sparse correction-bias bytes exceed u64",
            )?,
            "GLM-4 sparse correction-bias range overflow",
        )?;
        let attention_shape = FullAttentionShape::new(
            mla.head_count(),
            mla.qk_head_dim(),
            mla.value_head_dim(),
            cache.capacity_tokens(),
            cache.capacity_tokens() - 1,
        )?;
        let attention_layout = FullKvLayout::new(0, 0, cache.key_dtype(), cache.value_dtype());
        let attention =
            FullAttentionPlan::from_arena(attention_shape, attention_layout, usize::MAX)?.scratch();
        let mla_scratch = mla.scratch();
        let output_linear = output_projection.scratch();
        let moe_scratch = moe.scratch();
        let cache_staging_bytes = cache.requirements().staging_bytes;
        let norm_encoded_bytes = input_norm_bytes.max(post_attention_norm_bytes);
        let query_elements = mla.query_key_output_elements();
        let key_elements = query_elements;
        let value_elements = mla.value_output_elements();
        let attention_output_elements = value_elements;
        let float_elements = [
            hidden,
            expert_count,
            hidden,
            query_elements,
            key_elements,
            value_elements,
            attention_output_elements,
            hidden,
            hidden,
            hidden,
            hidden,
            hidden,
        ]
        .into_iter()
        .try_fold(0usize, |total, count| {
            add(total, count, "GLM-4 sparse layer scratch elements overflow")
        })?;
        let nested_bytes = [
            mla_scratch.total_bytes,
            cache_staging_bytes,
            attention.total_bytes,
            output_linear.total_bytes,
            moe_scratch.total_bytes,
            norm_encoded_bytes,
            correction_bias_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, count| {
            add(total, count, "GLM-4 sparse layer nested scratch overflow")
        })?;
        let total_bytes = add(
            nested_bytes,
            mul(
                float_elements,
                size_of::<f64>(),
                "GLM-4 sparse layer FP64 scratch overflow",
            )?,
            "GLM-4 sparse layer total scratch overflow",
        )?;
        Ok(Self {
            mla,
            output_projection,
            moe,
            cache,
            vector_layout,
            input_norm_end,
            post_attention_norm_end,
            correction_bias_end,
            rms_norm_epsilon,
            scratch: Glm4SparseLayerScratchRequirements {
                mla: mla_scratch,
                cache_staging_bytes,
                attention,
                output_linear,
                moe: moe_scratch,
                norm_encoded_bytes,
                correction_encoded_bytes: correction_bias_bytes,
                norm_weight_elements: hidden,
                correction_bias_elements: expert_count,
                normalized_elements: hidden,
                query_elements,
                key_elements,
                value_elements,
                attention_output_elements,
                output_projection_elements: hidden,
                residual_elements: hidden,
                post_normalized_elements: hidden,
                moe_output_elements: hidden,
                final_output_elements: hidden,
                total_bytes,
            },
        })
    }

    /// Exact fixed working set for one token through this layer.
    #[must_use]
    pub const fn scratch(&self) -> Glm4SparseLayerScratchRequirements {
        self.scratch
    }
}

/// Immutable weight readers for one sparse decoder layer.
pub struct Glm4SparseLayerReaders<'reader, 'slice> {
    input_norm: &'reader dyn RangeReader,
    mla: Glm4MlaReaders<'reader>,
    output_projection: &'reader dyn RangeReader,
    post_attention_norm: &'reader dyn RangeReader,
    correction_bias: &'reader dyn RangeReader,
    moe: SparseMoeBindings<'reader, 'slice>,
}

impl<'reader, 'slice> Glm4SparseLayerReaders<'reader, 'slice> {
    /// Bind every sparse-layer weight object without reading any object.
    #[must_use]
    pub const fn new(
        input_norm: &'reader dyn RangeReader,
        mla: Glm4MlaReaders<'reader>,
        output_projection: &'reader dyn RangeReader,
        post_attention_norm: &'reader dyn RangeReader,
        correction_bias: &'reader dyn RangeReader,
        moe: SparseMoeBindings<'reader, 'slice>,
    ) -> Self {
        Self {
            input_norm,
            mla,
            output_projection,
            post_attention_norm,
            correction_bias,
            moe,
        }
    }
}

/// Caller-owned buffers for one complete sparse decoder-layer token.
pub struct Glm4SparseLayerScratch<'a> {
    mla: Glm4MlaScratch<'a>,
    cache_staging: &'a mut [u8],
    attention: FullAttentionScratch<'a>,
    output_linear: LinearScratch<'a>,
    moe: SparseMoeScratch<'a>,
    norm_encoded: &'a mut [u8],
    correction_encoded: &'a mut [u8],
    norm_weights: &'a mut [f64],
    correction_bias: &'a mut [f64],
    normalized: &'a mut [f64],
    query: &'a mut [f64],
    key: &'a mut [f64],
    value: &'a mut [f64],
    attention_output: &'a mut [f64],
    output_projection: &'a mut [f64],
    residual: &'a mut [f64],
    post_normalized: &'a mut [f64],
    moe_output: &'a mut [f64],
    final_output: &'a mut [f64],
}

impl<'a> Glm4SparseLayerScratch<'a> {
    /// Group the complete admitted sparse decoder-layer working set.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        mla: Glm4MlaScratch<'a>,
        cache_staging: &'a mut [u8],
        attention: FullAttentionScratch<'a>,
        output_linear: LinearScratch<'a>,
        moe: SparseMoeScratch<'a>,
        norm_encoded: &'a mut [u8],
        correction_encoded: &'a mut [u8],
        norm_weights: &'a mut [f64],
        correction_bias: &'a mut [f64],
        normalized: &'a mut [f64],
        query: &'a mut [f64],
        key: &'a mut [f64],
        value: &'a mut [f64],
        attention_output: &'a mut [f64],
        output_projection: &'a mut [f64],
        residual: &'a mut [f64],
        post_normalized: &'a mut [f64],
        moe_output: &'a mut [f64],
        final_output: &'a mut [f64],
    ) -> Self {
        Self {
            mla,
            cache_staging,
            attention,
            output_linear,
            moe,
            norm_encoded,
            correction_encoded,
            norm_weights,
            correction_bias,
            normalized,
            query,
            key,
            value,
            attention_output,
            output_projection,
            residual,
            post_normalized,
            moe_output,
            final_output,
        }
    }

    const fn admits(&self, plan: &Glm4SparseLayerPlan) -> bool {
        let requirement = &plan.scratch;
        self.mla.admits(requirement.mla)
            && self.cache_staging.len() >= requirement.cache_staging_bytes
            && self.attention.admits(requirement.attention)
            && self.output_linear.admits(requirement.output_linear)
            && self.moe.admits(&plan.moe)
            && self.norm_encoded.len() >= requirement.norm_encoded_bytes
            && self.correction_encoded.len() >= requirement.correction_encoded_bytes
            && self.norm_weights.len() >= requirement.norm_weight_elements
            && self.correction_bias.len() >= requirement.correction_bias_elements
            && self.normalized.len() >= requirement.normalized_elements
            && self.query.len() >= requirement.query_elements
            && self.key.len() >= requirement.key_elements
            && self.value.len() >= requirement.value_elements
            && self.attention_output.len() >= requirement.attention_output_elements
            && self.output_projection.len() >= requirement.output_projection_elements
            && self.residual.len() >= requirement.residual_elements
            && self.post_normalized.len() >= requirement.post_normalized_elements
            && self.moe_output.len() >= requirement.moe_output_elements
            && self.final_output.len() >= requirement.final_output_elements
    }
}

/// Execute one token through a sparse-MLP GLM-4-MoE-Lite decoder layer.
///
/// The complete graph is failure-atomic: the next K/V row remains staged and caller output
/// remains untouched until MLA, full causal attention, sparse `MoE`, and both residuals succeed.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
#[allow(clippy::too_many_lines)]
pub fn glm4_sparse_layer_token(
    plan: &Glm4SparseLayerPlan,
    readers: &Glm4SparseLayerReaders<'_, '_>,
    cache: &mut KvCache<'_>,
    position: usize,
    hidden: &[f64],
    scratch: &mut Glm4SparseLayerScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    let requirement = plan.scratch;
    if cache.plan() != plan.cache || position != cache.committed_tokens() {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 sparse layer cache state disagrees with the plan or position",
        ));
    }
    if hidden.len() != requirement.final_output_elements
        || output.len() != requirement.final_output_elements
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 sparse layer hidden or output dimensions differ from the plan",
        ));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "GLM-4 sparse layer hidden state is non-finite",
        ));
    }
    if !scratch.admits(plan) {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "GLM-4 sparse layer scratch is smaller than the admitted plan",
        ));
    }
    if plan.input_norm_end > readers.input_norm.len()
        || plan.post_attention_norm_end > readers.post_attention_norm.len()
        || plan.correction_bias_end > readers.correction_bias.len()
        || plan.output_projection.reader_end() > readers.output_projection.len()
        || !mla_readers_admit(&plan.mla, &readers.mla)
        || !plan.moe.bindings_admit(&readers.moe)
    {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "GLM-4 sparse layer weight binding or range is incomplete",
        ));
    }

    let correction_bias = &mut scratch.correction_bias[..requirement.correction_bias_elements];
    read_identity_vector(
        readers.correction_bias,
        plan.vector_layout.correction_bias_offset,
        plan.vector_layout.correction_bias_dtype,
        correction_bias,
        scratch.correction_encoded,
    )?;
    let norm_weights = &mut scratch.norm_weights[..requirement.norm_weight_elements];
    read_identity_vector(
        readers.input_norm,
        plan.vector_layout.input_norm_offset,
        plan.vector_layout.input_norm_dtype,
        norm_weights,
        scratch.norm_encoded,
    )?;
    let normalized = &mut scratch.normalized[..requirement.normalized_elements];
    glm_rms_norm(hidden, norm_weights, plan.rms_norm_epsilon, normalized)?;
    let query = &mut scratch.query[..requirement.query_elements];
    let key = &mut scratch.key[..requirement.key_elements];
    let value = &mut scratch.value[..requirement.value_elements];
    glm4_mla_project(
        &plan.mla,
        &readers.mla,
        normalized,
        position,
        &mut scratch.mla,
        query,
        key,
        value,
    )?;
    cache.stage_row(position, key, value, scratch.cache_staging)?;

    let attention_output = &mut scratch.attention_output[..requirement.attention_output_elements];
    {
        let staged = cache.staged_view(position, scratch.cache_staging)?;
        let attention_shape = FullAttentionShape::new(
            plan.cache.head_count(),
            plan.cache.key_head_dimension(),
            plan.cache.value_head_dimension(),
            staged.staged_tokens(),
            position,
        )?;
        let attention_layout =
            FullKvLayout::new(0, 0, plan.cache.key_dtype(), plan.cache.value_dtype());
        let attention_plan = FullAttentionPlan::from_arena(
            attention_shape,
            attention_layout,
            requirement.attention.total_bytes,
        )?;
        let attention_readers =
            FullAttentionReaders::new(staged.key_reader(), staged.value_reader());
        glm_full_attention(
            attention_plan,
            &attention_readers,
            query,
            &mut scratch.attention,
            attention_output,
        )?;
    }

    let projected = &mut scratch.output_projection[..requirement.output_projection_elements];
    stream_linear(
        readers.output_projection,
        plan.output_projection,
        attention_output,
        None,
        &mut scratch.output_linear,
        projected,
    )?;
    let residual = &mut scratch.residual[..requirement.residual_elements];
    for ((destination, original), attention) in
        residual.iter_mut().zip(hidden).zip(projected.iter())
    {
        *destination = *original + *attention;
        if !destination.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "GLM-4 sparse layer attention residual is non-finite",
            ));
        }
    }
    read_identity_vector(
        readers.post_attention_norm,
        plan.vector_layout.post_attention_norm_offset,
        plan.vector_layout.post_attention_norm_dtype,
        norm_weights,
        scratch.norm_encoded,
    )?;
    let post_normalized = &mut scratch.post_normalized[..requirement.post_normalized_elements];
    glm_rms_norm(
        residual,
        norm_weights,
        plan.rms_norm_epsilon,
        post_normalized,
    )?;
    let moe_output = &mut scratch.moe_output[..requirement.moe_output_elements];
    glm_sparse_moe(
        &plan.moe,
        &readers.moe,
        post_normalized,
        correction_bias,
        &mut scratch.moe,
        moe_output,
    )?;
    let final_output = &mut scratch.final_output[..requirement.final_output_elements];
    for ((destination, residual_value), moe_value) in final_output
        .iter_mut()
        .zip(residual.iter())
        .zip(moe_output.iter())
    {
        *destination = *residual_value + *moe_value;
        if !destination.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "GLM-4 sparse layer MoE residual is non-finite",
            ));
        }
    }
    cache.commit_staged(position, scratch.cache_staging)?;
    output.copy_from_slice(final_output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        GatedMlpPlan, GatedMlpReaders, GatedMlpScratch, Glm4MlaNormLayout, GlmRouterPlan,
        GlmRouterScratch, IdentityLinearPlan, SliceReader,
    };

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

        fn reads(&self) -> usize {
            self.reads.get()
        }
    }

    impl RangeReader for CountingReader {
        fn len(&self) -> u64 {
            u64::try_from(self.bytes.len()).unwrap_or(u64::MAX)
        }

        fn read_exact_at(&self, offset: u64, destination: &mut [u8]) -> Result<(), AmsError> {
            let start = usize::try_from(offset).map_err(|_| {
                AmsError::new(ErrorCode::IoFailure, "test reader offset exceeds usize")
            })?;
            let end = start
                .checked_add(destination.len())
                .ok_or_else(|| AmsError::new(ErrorCode::IoFailure, "test reader overflow"))?;
            let source = self.bytes.get(start..end).ok_or_else(|| {
                AmsError::new(ErrorCode::IoFailure, "test reader range exceeds object")
            })?;
            destination.copy_from_slice(source);
            self.reads.set(self.reads.get().saturating_add(1));
            Ok(())
        }
    }

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn linear_plan(rows: usize, columns: usize) -> Result<LinearPlan, AmsError> {
        Ok(IdentityLinearPlan::from_arena(rows, columns, 0, 16, IdentityDType::Float32)?.into())
    }

    fn gated_plan() -> Result<GatedMlpPlan, AmsError> {
        GatedMlpPlan::new(linear_plan(2, 2)?, linear_plan(2, 2)?, linear_plan(2, 2)?)
    }

    fn fixture_plan() -> Result<(Glm4SparseLayerPlan, [GatedMlpPlan; 2]), AmsError> {
        let mla = Glm4MlaPlan::new(
            linear_plan(2, 2)?,
            linear_plan(3, 2)?,
            linear_plan(4, 2)?,
            linear_plan(3, 2)?,
            Glm4MlaNormLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            1,
            1,
            2,
            2,
            1e-5,
            10_000.0,
        )?;
        let expert_plans = [gated_plan()?, gated_plan()?];
        let moe = SparseMoePlan::new(
            linear_plan(2, 2)?,
            GlmRouterPlan::new(2, 1, 1, 1, 1.0)?,
            &expert_plans,
            gated_plan()?,
        )?;
        let cache = KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        let plan = Glm4SparseLayerPlan::new(
            mla,
            linear_plan(2, 2)?,
            moe,
            cache,
            Glm4SparseLayerVectorLayout::new(
                0,
                0,
                0,
                IdentityDType::Float32,
                IdentityDType::Float32,
                IdentityDType::Float32,
            ),
            1e-5,
        )?;
        Ok((plan, expert_plans))
    }

    #[allow(
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn run_fixture(
        plan: &Glm4SparseLayerPlan,
        readers: &Glm4SparseLayerReaders<'_, '_>,
        cache: &mut KvCache<'_>,
        position: usize,
        hidden: &[f64; 2],
        output: &mut [f64; 2],
    ) -> Result<(), AmsError> {
        let mut mla_linear_encoded = [0u8; 8];
        let mut mla_linear_decoded = [0.0f32; 0];
        let mut mla_linear_accumulators = [0.0f64; 0];
        let mla_linear = LinearScratch::new(
            &mut mla_linear_encoded,
            &mut mla_linear_decoded,
            &mut mla_linear_accumulators,
        );
        let mut mla_norm_encoded = [0u8; 8];
        let mut mla_norm_weights = [0.0f64; 2];
        let mut q_a = [0.0f64; 2];
        let mut kv_a = [0.0f64; 4];
        let mut mla_normalized = [0.0f64; 2];
        let mut q_projected = [0.0f64; 3];
        let mut kv_projected = [0.0f64; 3];
        let mut mla_query = [0.0f64; 3];
        let mut mla_key = [0.0f64; 3];
        let mut mla_value = [0.0f64; 2];
        let mla_scratch = Glm4MlaScratch::new(
            mla_linear,
            &mut mla_norm_encoded,
            &mut mla_norm_weights,
            &mut q_a,
            &mut kv_a,
            &mut mla_normalized,
            &mut q_projected,
            &mut kv_projected,
            &mut mla_query,
            &mut mla_key,
            &mut mla_value,
        );
        let mut cache_staging = [0u8; 20];
        let mut attention_encoded = [0u8; 12];
        let mut attention_key = [0.0f64; 3];
        let mut attention_value = [0.0f64; 2];
        let mut attention_transactional = [0.0f64; 2];
        let attention = FullAttentionScratch::new(
            &mut attention_encoded,
            &mut attention_key,
            &mut attention_value,
            &mut attention_transactional,
        );
        let mut output_linear_encoded = [0u8; 8];
        let mut output_linear_decoded = [0.0f32; 0];
        let mut output_linear_accumulators = [0.0f64; 0];
        let output_linear = LinearScratch::new(
            &mut output_linear_encoded,
            &mut output_linear_decoded,
            &mut output_linear_accumulators,
        );
        let mut moe_linear_encoded = [0u8; 8];
        let mut moe_linear_decoded = [0.0f32; 0];
        let mut moe_linear_accumulators = [0.0f64; 0];
        let moe_linear = LinearScratch::new(
            &mut moe_linear_encoded,
            &mut moe_linear_decoded,
            &mut moe_linear_accumulators,
        );
        let mut moe_gate = [0.0f64; 2];
        let mut moe_up = [0.0f64; 2];
        let moe_mlp = GatedMlpScratch::new(moe_linear, &mut moe_gate, &mut moe_up);
        let mut router_logits = [0.0f64; 2];
        let mut probabilities = [0.0f64; 2];
        let mut corrected = [0.0f64; 2];
        let mut group_scores = [0.0f64; 1];
        let mut selected_groups = [usize::MAX; 1];
        let routing = GlmRouterScratch::new(
            &mut probabilities,
            &mut corrected,
            &mut group_scores,
            &mut selected_groups,
        );
        let mut expert_indices = [usize::MAX; 1];
        let mut expert_weights = [0.0f64; 1];
        let mut expert_output = [0.0f64; 2];
        let mut accumulator = [0.0f64; 2];
        let moe = SparseMoeScratch::new(
            moe_mlp,
            &mut router_logits,
            routing,
            &mut expert_indices,
            &mut expert_weights,
            &mut expert_output,
            &mut accumulator,
        );
        let mut norm_encoded = [0u8; 8];
        let mut correction_encoded = [0u8; 8];
        let mut norm_weights = [0.0f64; 2];
        let mut correction_bias = [0.0f64; 2];
        let mut normalized = [0.0f64; 2];
        let mut query = [0.0f64; 3];
        let mut key = [0.0f64; 3];
        let mut value = [0.0f64; 2];
        let mut attention_output = [0.0f64; 2];
        let mut output_projection = [0.0f64; 2];
        let mut residual = [0.0f64; 2];
        let mut post_normalized = [0.0f64; 2];
        let mut moe_output = [0.0f64; 2];
        let mut final_output = [0.0f64; 2];
        let mut scratch = Glm4SparseLayerScratch::new(
            mla_scratch,
            &mut cache_staging,
            attention,
            output_linear,
            moe,
            &mut norm_encoded,
            &mut correction_encoded,
            &mut norm_weights,
            &mut correction_bias,
            &mut normalized,
            &mut query,
            &mut key,
            &mut value,
            &mut attention_output,
            &mut output_projection,
            &mut residual,
            &mut post_normalized,
            &mut moe_output,
            &mut final_output,
        );
        glm4_sparse_layer_token(plan, readers, cache, position, hidden, &mut scratch, output)
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::too_many_lines
    )]
    fn sparse_layer_is_prefix_transactional_and_retryable_after_late_failure()
    -> Result<(), AmsError> {
        let (plan, expert_plans) = fixture_plan()?;
        assert_eq!(plan.scratch().total_bytes, 704);
        let norm = encode_f32(&[1.0, 1.0]);
        let correction = encode_f32(&[0.0, 1.0]);
        let q_a = encode_f32(&[1.0, 0.0, 0.0, 1.0]);
        let q_b = encode_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let kv_a = encode_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let kv_b = encode_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let identity = encode_f32(&[1.0, 0.0, 0.0, 1.0]);
        let zeros = encode_f32(&[0.0; 4]);
        let bad_down = encode_f32(&[f32::NAN, 0.0, 0.0, 0.0]);
        let norm_reader = SliceReader::new(&norm);
        let correction_reader = SliceReader::new(&correction);
        let q_a_reader = SliceReader::new(&q_a);
        let q_b_reader = SliceReader::new(&q_b);
        let kv_a_reader = SliceReader::new(&kv_a);
        let kv_b_reader = SliceReader::new(&kv_b);
        let identity_reader = SliceReader::new(&identity);
        let zero_reader = SliceReader::new(&zeros);
        let bad_down_reader = SliceReader::new(&bad_down);
        let shared_readers = GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader);
        let good_expert_readers = [
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
        ];
        let good_moe = SparseMoeBindings::new(
            &zero_reader,
            &expert_plans,
            &good_expert_readers,
            &shared_readers,
        );
        let good_mla = Glm4MlaReaders::new(
            &q_a_reader,
            &norm_reader,
            &q_b_reader,
            &kv_a_reader,
            &norm_reader,
            &kv_b_reader,
        );
        let good_readers = Glm4SparseLayerReaders::new(
            &norm_reader,
            good_mla,
            &identity_reader,
            &norm_reader,
            &correction_reader,
            good_moe,
        );
        let mut key_storage = [0u8; 24];
        let mut value_storage = [0u8; 16];
        let cache_plan =
            KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        let mut cache = KvCache::new(cache_plan, &mut key_storage, &mut value_storage)?;

        let mut first_output = [99.0f64; 2];
        run_fixture(
            &plan,
            &good_readers,
            &mut cache,
            0,
            &[1.0, 0.0],
            &mut first_output,
        )?;
        let first_norm = 1.0 / (0.5f64 + 1e-5).sqrt();
        let second_norm = first_norm / (first_norm * first_norm / 2.0 + 1e-5).sqrt();
        let cached_second_norm = f64::from(second_norm as f32);
        assert!((first_output[0] - (1.0 + cached_second_norm)).abs() <= 1e-12);
        assert!(first_output[1].abs() <= 1e-12);
        assert_eq!(cache.committed_tokens(), 1);

        let bad_expert_readers = [
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
            GatedMlpReaders::new(&zero_reader, &zero_reader, &bad_down_reader),
        ];
        let bad_moe = SparseMoeBindings::new(
            &zero_reader,
            &expert_plans,
            &bad_expert_readers,
            &shared_readers,
        );
        let bad_mla = Glm4MlaReaders::new(
            &q_a_reader,
            &norm_reader,
            &q_b_reader,
            &kv_a_reader,
            &norm_reader,
            &kv_b_reader,
        );
        let bad_readers = Glm4SparseLayerReaders::new(
            &norm_reader,
            bad_mla,
            &identity_reader,
            &norm_reader,
            &correction_reader,
            bad_moe,
        );
        let mut second_output = [77.0f64; 2];
        let error = run_fixture(
            &plan,
            &bad_readers,
            &mut cache,
            1,
            &[0.0, 1.0],
            &mut second_output,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert_eq!(cache.committed_tokens(), 1);
        assert_eq!(second_output.map(f64::to_bits), [77.0f64.to_bits(); 2]);

        run_fixture(
            &plan,
            &good_readers,
            &mut cache,
            1,
            &[0.0, 1.0],
            &mut second_output,
        )?;
        assert_eq!(cache.committed_tokens(), 2);
        assert!(second_output.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn sparse_layer_rejects_incomplete_expert_bindings_before_any_weight_read()
    -> Result<(), AmsError> {
        let (plan, expert_plans) = fixture_plan()?;
        let norm = CountingReader::new(encode_f32(&[1.0, 1.0]));
        let correction = CountingReader::new(encode_f32(&[0.0, 1.0]));
        let q_a = CountingReader::new(encode_f32(&[1.0, 0.0, 0.0, 1.0]));
        let q_b = CountingReader::new(encode_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]));
        let kv_a = CountingReader::new(encode_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]));
        let kv_b = CountingReader::new(encode_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]));
        let identity = CountingReader::new(encode_f32(&[1.0, 0.0, 0.0, 1.0]));
        let zeros = CountingReader::new(encode_f32(&[0.0; 4]));
        let shared_readers = GatedMlpReaders::new(&zeros, &zeros, &zeros);
        let incomplete_expert_readers = [GatedMlpReaders::new(&zeros, &zeros, &zeros)];
        let moe = SparseMoeBindings::new(
            &zeros,
            &expert_plans,
            &incomplete_expert_readers,
            &shared_readers,
        );
        let mla = Glm4MlaReaders::new(&q_a, &norm, &q_b, &kv_a, &norm, &kv_b);
        let readers = Glm4SparseLayerReaders::new(&norm, mla, &identity, &norm, &correction, moe);
        let mut key_storage = [0u8; 24];
        let mut value_storage = [0u8; 16];
        let cache_plan =
            KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        let mut cache = KvCache::new(cache_plan, &mut key_storage, &mut value_storage)?;
        let mut output = [55.0f64; 2];
        let error = run_fixture(&plan, &readers, &mut cache, 0, &[1.0, 0.0], &mut output).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        assert_eq!(cache.committed_tokens(), 0);
        assert_eq!(output.map(f64::to_bits), [55.0f64.to_bits(); 2]);
        assert_eq!(norm.reads(), 0);
        assert_eq!(correction.reads(), 0);
        assert_eq!(q_a.reads(), 0);
        assert_eq!(q_b.reads(), 0);
        assert_eq!(kv_a.reads(), 0);
        assert_eq!(kv_b.reads(), 0);
        assert_eq!(identity.reads(), 0);
        assert_eq!(zeros.reads(), 0);
        Ok(())
    }
}
