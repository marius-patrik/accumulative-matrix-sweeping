use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{
    AmsError, ErrorCode, FullAttentionPlan, FullAttentionReaders, FullAttentionScratch,
    FullAttentionScratchRequirements, FullAttentionShape, FullKvLayout, GatedMlpPlan,
    GatedMlpReaders, GatedMlpScratch, GatedMlpScratchRequirements, Glm4MlaPlan, Glm4MlaReaders,
    Glm4MlaScratch, Glm4MlaScratchRequirements, IdentityDType, KvCache, KvCachePlan, LinearPlan,
    LinearScratch, LinearScratchRequirements, RangeReader, glm_full_attention, glm_gated_mlp,
    glm_rms_norm, glm4_mla_project, read_identity_vector, stream_linear,
};

/// Identity layouts for the two decoder-layer normalization vectors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4DenseLayerNormLayout {
    input_offset: u64,
    post_attention_offset: u64,
    input_dtype: IdentityDType,
    post_attention_dtype: IdentityDType,
}

impl Glm4DenseLayerNormLayout {
    /// Bind the input and post-attention normalization vectors.
    #[must_use]
    pub const fn new(
        input_offset: u64,
        post_attention_offset: u64,
        input_dtype: IdentityDType,
        post_attention_dtype: IdentityDType,
    ) -> Self {
        Self {
            input_offset,
            post_attention_offset,
            input_dtype,
            post_attention_dtype,
        }
    }
}

/// Exact caller-owned working set for one dense GLM-4 decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4DenseLayerScratchRequirements {
    /// Nested MLA projection scratch.
    pub mla: Glm4MlaScratchRequirements,
    /// One staged K/V token row.
    pub cache_staging_bytes: usize,
    /// Nested context-independent attention scratch.
    pub attention: FullAttentionScratchRequirements,
    /// Output-projection linear scratch.
    pub output_linear: LinearScratchRequirements,
    /// Nested dense gated-MLP scratch.
    pub mlp: GatedMlpScratchRequirements,
    /// Encoded bytes for the larger decoder normalization vector.
    pub norm_encoded_bytes: usize,
    /// Decoder-width normalization weights.
    pub norm_weight_elements: usize,
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
    /// Dense MLP output elements.
    pub mlp_output_elements: usize,
    /// Transactional final layer output elements.
    pub final_output_elements: usize,
    /// Complete caller-owned working set including nested local bytes.
    pub total_bytes: usize,
}

/// Immutable plan for the dense first layer of GLM-4-MoE-Lite.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm4DenseLayerPlan {
    mla: Glm4MlaPlan,
    output_projection: LinearPlan,
    mlp: GatedMlpPlan,
    cache: KvCachePlan,
    norm_layout: Glm4DenseLayerNormLayout,
    input_norm_end: u64,
    post_norm_end: u64,
    rms_norm_epsilon: f64,
    scratch: Glm4DenseLayerScratchRequirements,
}

impl Glm4DenseLayerPlan {
    /// Validate the complete dense decoder-layer graph and derive its fixed working set.
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
        mlp: GatedMlpPlan,
        cache: KvCachePlan,
        norm_layout: Glm4DenseLayerNormLayout,
        rms_norm_epsilon: f64,
    ) -> Result<Self, AmsError> {
        if !rms_norm_epsilon.is_finite() || rms_norm_epsilon <= 0.0 {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 dense layer RMSNorm epsilon is invalid",
            ));
        }
        let hidden = mla.hidden_elements();
        if cache.head_count() != mla.head_count()
            || cache.key_head_dimension() != mla.qk_head_dim()
            || cache.value_head_dimension() != mla.value_head_dim()
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 dense layer cache dimensions differ from MLA",
            ));
        }
        if output_projection.rows() != hidden
            || output_projection.columns() != mla.value_output_elements()
            || mlp.input_elements() != hidden
            || mlp.output_elements() != hidden
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 dense layer projection or MLP dimensions differ",
            ));
        }
        let input_norm_bytes = mul(
            hidden,
            norm_layout.input_dtype.item_bytes(),
            "GLM-4 dense input norm bytes overflow",
        )?;
        let post_norm_bytes = mul(
            hidden,
            norm_layout.post_attention_dtype.item_bytes(),
            "GLM-4 dense post-attention norm bytes overflow",
        )?;
        let input_norm_end = add_u64(
            norm_layout.input_offset,
            usize_to_u64(input_norm_bytes, "GLM-4 dense input norm bytes exceed u64")?,
            "GLM-4 dense input norm range overflow",
        )?;
        let post_norm_end = add_u64(
            norm_layout.post_attention_offset,
            usize_to_u64(post_norm_bytes, "GLM-4 dense post norm bytes exceed u64")?,
            "GLM-4 dense post norm range overflow",
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
        let mlp_scratch = mlp.scratch();
        let cache_staging_bytes = cache.requirements().staging_bytes;
        let norm_encoded_bytes = input_norm_bytes.max(post_norm_bytes);
        let query_elements = mla.query_key_output_elements();
        let key_elements = query_elements;
        let value_elements = mla.value_output_elements();
        let attention_output_elements = value_elements;
        let float_elements = [
            hidden,
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
            add(total, count, "GLM-4 dense layer scratch elements overflow")
        })?;
        let nested_bytes = [
            mla_scratch.total_bytes,
            cache_staging_bytes,
            attention.total_bytes,
            output_linear.total_bytes,
            mlp_scratch.total_bytes,
            norm_encoded_bytes,
        ]
        .into_iter()
        .try_fold(0usize, |total, count| {
            add(total, count, "GLM-4 dense layer nested scratch overflow")
        })?;
        let total_bytes = add(
            nested_bytes,
            mul(
                float_elements,
                size_of::<f64>(),
                "GLM-4 dense layer FP64 scratch overflow",
            )?,
            "GLM-4 dense layer total scratch overflow",
        )?;
        Ok(Self {
            mla,
            output_projection,
            mlp,
            cache,
            norm_layout,
            input_norm_end,
            post_norm_end,
            rms_norm_epsilon,
            scratch: Glm4DenseLayerScratchRequirements {
                mla: mla_scratch,
                cache_staging_bytes,
                attention,
                output_linear,
                mlp: mlp_scratch,
                norm_encoded_bytes,
                norm_weight_elements: hidden,
                normalized_elements: hidden,
                query_elements,
                key_elements,
                value_elements,
                attention_output_elements,
                output_projection_elements: hidden,
                residual_elements: hidden,
                post_normalized_elements: hidden,
                mlp_output_elements: hidden,
                final_output_elements: hidden,
                total_bytes,
            },
        })
    }

    /// Exact fixed working set for one token through this layer.
    #[must_use]
    pub const fn scratch(&self) -> Glm4DenseLayerScratchRequirements {
        self.scratch
    }

    /// Decoder hidden width consumed and produced by this layer.
    #[must_use]
    pub const fn hidden_elements(&self) -> usize {
        self.scratch.final_output_elements
    }

    /// Exact K/V cache geometry bound to this layer.
    #[must_use]
    pub const fn cache_plan(&self) -> KvCachePlan {
        self.cache
    }

    pub(crate) fn preflight(
        &self,
        readers: &Glm4DenseLayerReaders<'_>,
        cache: &KvCache<'_>,
        position: usize,
        scratch: &Glm4DenseLayerScratch<'_>,
    ) -> Result<(), AmsError> {
        if cache.plan() != self.cache || position != cache.committed_tokens() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 dense layer cache state disagrees with the plan or position",
            ));
        }
        if !scratch.admits(&self.scratch) {
            return Err(AmsError::new(
                ErrorCode::PreflightNoWorkingSet,
                "GLM-4 dense layer scratch is smaller than the admitted plan",
            ));
        }
        if self.input_norm_end > readers.input_norm.len()
            || self.post_norm_end > readers.post_attention_norm.len()
            || self.output_projection.reader_end() > readers.output_projection.len()
            || !crate::glm4_mla::readers_admit(&self.mla, &readers.mla)
            || !self.mlp.readers_admit(&readers.mlp)
        {
            return Err(AmsError::new(
                ErrorCode::IoFailure,
                "GLM-4 dense layer weight range exceeds its storage object",
            ));
        }
        Ok(())
    }
}

/// Immutable weight readers for one dense decoder layer.
pub struct Glm4DenseLayerReaders<'a> {
    input_norm: &'a dyn RangeReader,
    mla: Glm4MlaReaders<'a>,
    output_projection: &'a dyn RangeReader,
    post_attention_norm: &'a dyn RangeReader,
    mlp: GatedMlpReaders<'a>,
}

impl<'a> Glm4DenseLayerReaders<'a> {
    /// Bind all dense-layer weight objects without reading any object.
    #[must_use]
    pub const fn new(
        input_norm: &'a dyn RangeReader,
        mla: Glm4MlaReaders<'a>,
        output_projection: &'a dyn RangeReader,
        post_attention_norm: &'a dyn RangeReader,
        mlp: GatedMlpReaders<'a>,
    ) -> Self {
        Self {
            input_norm,
            mla,
            output_projection,
            post_attention_norm,
            mlp,
        }
    }
}

/// Caller-owned buffers for one complete dense decoder-layer token.
pub struct Glm4DenseLayerScratch<'a> {
    mla: Glm4MlaScratch<'a>,
    cache_staging: &'a mut [u8],
    attention: FullAttentionScratch<'a>,
    output_linear: LinearScratch<'a>,
    mlp: GatedMlpScratch<'a>,
    norm_encoded: &'a mut [u8],
    norm_weights: &'a mut [f64],
    normalized: &'a mut [f64],
    query: &'a mut [f64],
    key: &'a mut [f64],
    value: &'a mut [f64],
    attention_output: &'a mut [f64],
    output_projection: &'a mut [f64],
    residual: &'a mut [f64],
    post_normalized: &'a mut [f64],
    mlp_output: &'a mut [f64],
    final_output: &'a mut [f64],
}

impl<'a> Glm4DenseLayerScratch<'a> {
    /// Group the complete admitted decoder-layer working set.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        mla: Glm4MlaScratch<'a>,
        cache_staging: &'a mut [u8],
        attention: FullAttentionScratch<'a>,
        output_linear: LinearScratch<'a>,
        mlp: GatedMlpScratch<'a>,
        norm_encoded: &'a mut [u8],
        norm_weights: &'a mut [f64],
        normalized: &'a mut [f64],
        query: &'a mut [f64],
        key: &'a mut [f64],
        value: &'a mut [f64],
        attention_output: &'a mut [f64],
        output_projection: &'a mut [f64],
        residual: &'a mut [f64],
        post_normalized: &'a mut [f64],
        mlp_output: &'a mut [f64],
        final_output: &'a mut [f64],
    ) -> Self {
        Self {
            mla,
            cache_staging,
            attention,
            output_linear,
            mlp,
            norm_encoded,
            norm_weights,
            normalized,
            query,
            key,
            value,
            attention_output,
            output_projection,
            residual,
            post_normalized,
            mlp_output,
            final_output,
        }
    }

    const fn admits(&self, requirement: &Glm4DenseLayerScratchRequirements) -> bool {
        self.mla.admits(requirement.mla)
            && self.cache_staging.len() >= requirement.cache_staging_bytes
            && self.attention.admits(requirement.attention)
            && self.output_linear.admits(requirement.output_linear)
            && self.mlp.admits(requirement.mlp)
            && self.norm_encoded.len() >= requirement.norm_encoded_bytes
            && self.norm_weights.len() >= requirement.norm_weight_elements
            && self.normalized.len() >= requirement.normalized_elements
            && self.query.len() >= requirement.query_elements
            && self.key.len() >= requirement.key_elements
            && self.value.len() >= requirement.value_elements
            && self.attention_output.len() >= requirement.attention_output_elements
            && self.output_projection.len() >= requirement.output_projection_elements
            && self.residual.len() >= requirement.residual_elements
            && self.post_normalized.len() >= requirement.post_normalized_elements
            && self.mlp_output.len() >= requirement.mlp_output_elements
            && self.final_output.len() >= requirement.final_output_elements
    }
}

/// Execute one token through the dense first GLM-4-MoE-Lite decoder layer.
///
/// The current K/V row is staged and visible to attention but remains uncommitted until
/// MLA, attention, output projection, both residuals, and the dense MLP all succeed.
/// Caller output is copied only after the cache prefix commits.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
#[allow(clippy::too_many_lines)]
pub fn glm4_dense_layer_token(
    plan: &Glm4DenseLayerPlan,
    readers: &Glm4DenseLayerReaders<'_>,
    cache: &mut KvCache<'_>,
    position: usize,
    hidden: &[f64],
    scratch: &mut Glm4DenseLayerScratch<'_>,
    output: &mut [f64],
) -> Result<(), AmsError> {
    let requirement = plan.scratch;
    if hidden.len() != requirement.final_output_elements
        || output.len() != requirement.final_output_elements
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 dense layer hidden or output dimensions differ from the plan",
        ));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "GLM-4 dense layer hidden state is non-finite",
        ));
    }
    plan.preflight(readers, cache, position, scratch)?;

    let norm_weights = &mut scratch.norm_weights[..requirement.norm_weight_elements];
    read_identity_vector(
        readers.input_norm,
        plan.norm_layout.input_offset,
        plan.norm_layout.input_dtype,
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
                "GLM-4 dense layer attention residual is non-finite",
            ));
        }
    }
    read_identity_vector(
        readers.post_attention_norm,
        plan.norm_layout.post_attention_offset,
        plan.norm_layout.post_attention_dtype,
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
    let mlp_output = &mut scratch.mlp_output[..requirement.mlp_output_elements];
    glm_gated_mlp(
        plan.mlp,
        &readers.mlp,
        post_normalized,
        &mut scratch.mlp,
        mlp_output,
    )?;
    let final_output = &mut scratch.final_output[..requirement.final_output_elements];
    for ((destination, residual_value), mlp_value) in final_output
        .iter_mut()
        .zip(residual.iter())
        .zip(mlp_output.iter())
    {
        *destination = *residual_value + *mlp_value;
        if !destination.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "GLM-4 dense layer MLP residual is non-finite",
            ));
        }
    }
    cache.commit_staged(position, scratch.cache_staging)?;
    output.copy_from_slice(final_output);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GatedMlpScratch, Glm4MlaNormLayout, IdentityLinearPlan, SliceReader};

    fn encode_f32(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn linear_plan(rows: usize, columns: usize) -> Result<LinearPlan, AmsError> {
        Ok(IdentityLinearPlan::from_arena(rows, columns, 0, 16, IdentityDType::Float32)?.into())
    }

    fn fixture_plan() -> Result<Glm4DenseLayerPlan, AmsError> {
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
        let mlp = GatedMlpPlan::new(linear_plan(2, 2)?, linear_plan(2, 2)?, linear_plan(2, 2)?)?;
        let cache = KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        Glm4DenseLayerPlan::new(
            mla,
            linear_plan(2, 2)?,
            mlp,
            cache,
            Glm4DenseLayerNormLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            1e-5,
        )
    }

    #[allow(
        clippy::similar_names,
        clippy::too_many_arguments,
        clippy::too_many_lines
    )]
    fn run_fixture(
        plan: &Glm4DenseLayerPlan,
        readers: &Glm4DenseLayerReaders<'_>,
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
        let attention_scratch = FullAttentionScratch::new(
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
        let mut mlp_linear_encoded = [0u8; 8];
        let mut mlp_linear_decoded = [0.0f32; 0];
        let mut mlp_linear_accumulators = [0.0f64; 0];
        let mlp_linear = LinearScratch::new(
            &mut mlp_linear_encoded,
            &mut mlp_linear_decoded,
            &mut mlp_linear_accumulators,
        );
        let mut mlp_gate = [0.0f64; 2];
        let mut mlp_up = [0.0f64; 2];
        let mlp_scratch = GatedMlpScratch::new(mlp_linear, &mut mlp_gate, &mut mlp_up);
        let mut norm_encoded = [0u8; 8];
        let mut norm_weights = [0.0f64; 2];
        let mut normalized = [0.0f64; 2];
        let mut query = [0.0f64; 3];
        let mut key = [0.0f64; 3];
        let mut value = [0.0f64; 2];
        let mut attention_output = [0.0f64; 2];
        let mut output_projection = [0.0f64; 2];
        let mut residual = [0.0f64; 2];
        let mut post_normalized = [0.0f64; 2];
        let mut mlp_output = [0.0f64; 2];
        let mut final_output = [0.0f64; 2];
        let mut scratch = Glm4DenseLayerScratch::new(
            mla_scratch,
            &mut cache_staging,
            attention_scratch,
            output_linear,
            mlp_scratch,
            &mut norm_encoded,
            &mut norm_weights,
            &mut normalized,
            &mut query,
            &mut key,
            &mut value,
            &mut attention_output,
            &mut output_projection,
            &mut residual,
            &mut post_normalized,
            &mut mlp_output,
            &mut final_output,
        );
        glm4_dense_layer_token(plan, readers, cache, position, hidden, &mut scratch, output)
    }

    #[test]
    #[allow(
        clippy::cast_possible_truncation,
        clippy::similar_names,
        clippy::suboptimal_flops,
        clippy::too_many_lines
    )]
    fn dense_layer_is_prefix_transactional_and_retryable_after_late_failure() -> Result<(), AmsError>
    {
        let plan = fixture_plan()?;
        assert_eq!(plan.scratch().total_bytes, 568);
        let norm = encode_f32(&[1.0, 1.0]);
        let q_a = encode_f32(&[1.0, 0.0, 0.0, 1.0]);
        let q_b = encode_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let kv_a = encode_f32(&[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let kv_b = encode_f32(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let identity = encode_f32(&[1.0, 0.0, 0.0, 1.0]);
        let zeros = encode_f32(&[0.0; 4]);
        let bad_down = encode_f32(&[f32::NAN, 0.0, 0.0, 0.0]);
        let norm_reader = SliceReader::new(&norm);
        let q_a_reader = SliceReader::new(&q_a);
        let q_b_reader = SliceReader::new(&q_b);
        let kv_a_reader = SliceReader::new(&kv_a);
        let kv_b_reader = SliceReader::new(&kv_b);
        let identity_reader = SliceReader::new(&identity);
        let zero_reader = SliceReader::new(&zeros);
        let bad_down_reader = SliceReader::new(&bad_down);
        let mla_readers = Glm4MlaReaders::new(
            &q_a_reader,
            &norm_reader,
            &q_b_reader,
            &kv_a_reader,
            &norm_reader,
            &kv_b_reader,
        );
        let good_mlp = GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader);
        let good_readers = Glm4DenseLayerReaders::new(
            &norm_reader,
            mla_readers,
            &identity_reader,
            &norm_reader,
            good_mlp,
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

        let bad_mlp = GatedMlpReaders::new(&zero_reader, &zero_reader, &bad_down_reader);
        let bad_mla_readers = Glm4MlaReaders::new(
            &q_a_reader,
            &norm_reader,
            &q_b_reader,
            &kv_a_reader,
            &norm_reader,
            &kv_b_reader,
        );
        let bad_readers = Glm4DenseLayerReaders::new(
            &norm_reader,
            bad_mla_readers,
            &identity_reader,
            &norm_reader,
            bad_mlp,
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
        let sine = 1.0f64.sin();
        let cosine = 1.0f64.cos();
        let query_left = -second_norm * sine;
        let query_right = second_norm * cosine;
        let cached_first_norm = f64::from(first_norm as f32);
        let cached_key_left = f64::from((-first_norm * sine) as f32);
        let cached_key_right = f64::from((first_norm * cosine) as f32);
        let score_zero = query_left * cached_first_norm / 3.0f64.sqrt();
        let score_one =
            (query_left * cached_key_left + query_right * cached_key_right) / 3.0f64.sqrt();
        let maximum = score_zero.max(score_one);
        let weight_zero = (score_zero - maximum).exp();
        let weight_one = (score_one - maximum).exp();
        let denominator = weight_zero + weight_one;
        let expected = [
            weight_zero / denominator * cached_second_norm,
            1.0 + weight_one / denominator * cached_second_norm,
        ];
        for (actual, reference) in second_output.iter().zip(expected) {
            assert!((actual - reference).abs() <= 1e-12);
        }
        assert_eq!(cache.committed_tokens(), 2);
        Ok(())
    }

    #[test]
    fn dense_layer_plan_rejects_cache_shape_drift() -> Result<(), AmsError> {
        let plan = fixture_plan()?;
        let wrong_cache =
            KvCachePlan::new(2, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        assert_ne!(plan.cache, wrong_cache);
        let error = Glm4DenseLayerPlan::new(
            plan.mla,
            plan.output_projection,
            plan.mlp,
            wrong_cache,
            plan.norm_layout,
            plan.rms_norm_epsilon,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        Ok(())
    }
}
