use crate::{
    AmsError, ErrorCode, Glm4DenseLayerPlan, Glm4DenseLayerReaders, Glm4DenseLayerScratch,
    Glm4DenseLayerScratchRequirements, Glm4SparseLayerPlan, Glm4SparseLayerReaders,
    Glm4SparseLayerScratch, Glm4SparseLayerScratchRequirements, KvCache,
};

/// Immutable first-dense/rest-sparse GLM-4-MoE-Lite decoder stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm4DecoderPlan {
    dense: Glm4DenseLayerPlan,
    sparse: Vec<Glm4SparseLayerPlan>,
    sparse_scratch: Vec<Glm4SparseLayerScratchRequirements>,
    hidden_elements: usize,
}

impl Glm4DecoderPlan {
    /// Validate a complete GLM-4 inference-layer schedule.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` when the stack is missing sparse layers or layer/cache shapes drift.
    pub fn new(
        dense: Glm4DenseLayerPlan,
        sparse: Vec<Glm4SparseLayerPlan>,
    ) -> Result<Self, AmsError> {
        if sparse.is_empty() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 decoder requires one dense layer followed by sparse layers",
            ));
        }
        let hidden_elements = dense.hidden_elements();
        let cache_plan = dense.cache_plan();
        if sparse.iter().any(|layer| {
            layer.hidden_elements() != hidden_elements || layer.cache_plan() != cache_plan
        }) {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 decoder layer hidden or cache geometry differs",
            ));
        }
        let sparse_scratch = sparse.iter().map(Glm4SparseLayerPlan::scratch).collect();
        Ok(Self {
            dense,
            sparse,
            sparse_scratch,
            hidden_elements,
        })
    }

    /// Number of inference decoder layers in execution order.
    #[must_use]
    pub fn layer_count(&self) -> usize {
        self.sparse.len() + 1
    }

    /// Decoder hidden width.
    #[must_use]
    pub const fn hidden_elements(&self) -> usize {
        self.hidden_elements
    }

    /// Fixed token capacity shared by every layer cache.
    #[must_use]
    pub const fn cache_capacity_tokens(&self) -> usize {
        self.dense.cache_plan().capacity_tokens()
    }

    /// Exact working set for the sole dense layer.
    #[must_use]
    pub const fn dense_scratch(&self) -> Glm4DenseLayerScratchRequirements {
        self.dense.scratch()
    }

    /// Per-layer sparse requirements that one reusable sparse allocation must admit.
    #[must_use]
    pub fn sparse_scratch(&self) -> &[Glm4SparseLayerScratchRequirements] {
        &self.sparse_scratch
    }

    pub(crate) fn preflight(
        &self,
        readers: &Glm4DecoderReaders<'_, '_, '_>,
        caches: &[KvCache<'_>],
        position: usize,
        dense_scratch: &Glm4DenseLayerScratch<'_>,
        sparse_scratch: &Glm4SparseLayerScratch<'_>,
    ) -> Result<(), AmsError> {
        if readers.sparse.len() != self.sparse.len() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 decoder reader inventory differs from the plan",
            ));
        }
        if caches.len() != self.layer_count() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 decoder cache inventory differs from the plan",
            ));
        }
        self.dense
            .preflight(&readers.dense, &caches[0], position, dense_scratch)?;
        for ((layer, layer_readers), cache) in
            self.sparse.iter().zip(readers.sparse).zip(&caches[1..])
        {
            layer.preflight(layer_readers, cache, position, sparse_scratch)?;
        }
        Ok(())
    }

    pub(crate) fn rollback(
        &self,
        caches: &mut [KvCache<'_>],
        position: usize,
    ) -> Result<(), AmsError> {
        if caches.len() != self.layer_count() {
            return Err(AmsError::new(
                ErrorCode::InternalInvariant,
                "GLM-4 decoder rollback cache inventory differs from the plan",
            ));
        }
        let committed_layers = caches.len();
        rollback_prefixes(caches, position, committed_layers)
    }
}

/// Weight bindings for one complete decoder stack.
pub struct Glm4DecoderReaders<'reader, 'slice, 'layers> {
    dense: Glm4DenseLayerReaders<'reader>,
    sparse: &'layers [Glm4SparseLayerReaders<'reader, 'slice>],
}

impl<'reader, 'slice, 'layers> Glm4DecoderReaders<'reader, 'slice, 'layers> {
    /// Bind the dense reader followed by every sparse-layer reader.
    #[must_use]
    pub const fn new(
        dense: Glm4DenseLayerReaders<'reader>,
        sparse: &'layers [Glm4SparseLayerReaders<'reader, 'slice>],
    ) -> Self {
        Self { dense, sparse }
    }
}

fn rollback_prefixes(
    caches: &mut [KvCache<'_>],
    position: usize,
    committed_layers: usize,
) -> Result<(), AmsError> {
    for cache in caches[..committed_layers].iter_mut().rev() {
        cache.rollback_last(position)?;
    }
    Ok(())
}

/// Execute one token through the full GLM-4 inference decoder stack transactionally.
///
/// Every layer binding, cache, and reusable scratch set is preflighted before the first weight read.
/// A later failure rolls every already-committed layer cache back to the original token prefix and
/// leaves caller output untouched. The dense and sparse working sets are each reused across their
/// respective layer class rather than multiplied by layer count.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
#[allow(clippy::too_many_arguments)]
pub fn glm4_decoder_token(
    plan: &Glm4DecoderPlan,
    readers: &Glm4DecoderReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    position: usize,
    hidden: &[f64],
    dense_scratch: &mut Glm4DenseLayerScratch<'_>,
    sparse_scratch: &mut Glm4SparseLayerScratch<'_>,
    hidden_a: &mut [f64],
    hidden_b: &mut [f64],
    output: &mut [f64],
) -> Result<(), AmsError> {
    if hidden.len() != plan.hidden_elements
        || hidden_a.len() < plan.hidden_elements
        || hidden_b.len() < plan.hidden_elements
        || output.len() != plan.hidden_elements
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 decoder hidden or output dimensions differ from the plan",
        ));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(AmsError::new(
            ErrorCode::NumericFailure,
            "GLM-4 decoder hidden state is non-finite",
        ));
    }
    plan.preflight(readers, caches, position, dense_scratch, sparse_scratch)?;

    let hidden_a = &mut hidden_a[..plan.hidden_elements];
    let hidden_b = &mut hidden_b[..plan.hidden_elements];
    let mut committed_layers = 0usize;
    let execution = (|| -> Result<bool, AmsError> {
        crate::glm4_dense_layer_token(
            &plan.dense,
            &readers.dense,
            &mut caches[0],
            position,
            hidden,
            dense_scratch,
            hidden_a,
        )?;
        committed_layers = 1;
        let mut current_is_a = true;
        for (index, (layer, layer_readers)) in plan.sparse.iter().zip(readers.sparse).enumerate() {
            if current_is_a {
                crate::glm4_sparse_layer_token(
                    layer,
                    layer_readers,
                    &mut caches[index + 1],
                    position,
                    hidden_a,
                    sparse_scratch,
                    hidden_b,
                )?;
            } else {
                crate::glm4_sparse_layer_token(
                    layer,
                    layer_readers,
                    &mut caches[index + 1],
                    position,
                    hidden_b,
                    sparse_scratch,
                    hidden_a,
                )?;
            }
            committed_layers += 1;
            current_is_a = !current_is_a;
        }
        Ok(current_is_a)
    })();
    let current_is_a = match execution {
        Ok(current_is_a) => current_is_a,
        Err(error) => {
            rollback_prefixes(caches, position, committed_layers)?;
            return Err(error);
        }
    };
    if current_is_a {
        output.copy_from_slice(hidden_a);
    } else {
        output.copy_from_slice(hidden_b);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::{
        FullAttentionScratch, GatedMlpPlan, GatedMlpReaders, GatedMlpScratch,
        Glm4DenseLayerNormLayout, Glm4FinishReason, Glm4GenerationStep, Glm4GreedySession,
        Glm4MlaNormLayout, Glm4MlaPlan, Glm4MlaReaders, Glm4MlaScratch, Glm4ModelPlan,
        Glm4ModelReaders, Glm4ModelScratch, Glm4ModelVectorLayout, Glm4SparseLayerVectorLayout,
        GlmRouterPlan, GlmRouterScratch, IdentityDType, IdentityLinearPlan, KvCachePlan,
        LinearPlan, LinearScratch, RangeReader, SliceReader, SparseMoeBindings, SparseMoePlan,
        SparseMoeScratch, glm4_greedy_advance, glm4_model_next_token,
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
        Ok(IdentityLinearPlan::from_arena(rows, columns, 0, 32, IdentityDType::Float32)?.into())
    }

    fn mla_plan() -> Result<Glm4MlaPlan, AmsError> {
        Glm4MlaPlan::new(
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
        )
    }

    fn gated_plan() -> Result<GatedMlpPlan, AmsError> {
        GatedMlpPlan::new(linear_plan(2, 2)?, linear_plan(2, 2)?, linear_plan(2, 2)?)
    }

    fn fixture_plan() -> Result<(Glm4DecoderPlan, [GatedMlpPlan; 2]), AmsError> {
        let cache = KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        let dense = Glm4DenseLayerPlan::new(
            mla_plan()?,
            linear_plan(2, 2)?,
            gated_plan()?,
            cache,
            Glm4DenseLayerNormLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            1e-5,
        )?;
        let expert_plans = [gated_plan()?, gated_plan()?];
        let moe = SparseMoePlan::new(
            linear_plan(2, 2)?,
            GlmRouterPlan::new(2, 1, 1, 1, 1.0)?,
            &expert_plans,
            gated_plan()?,
        )?;
        let sparse = Glm4SparseLayerPlan::new(
            mla_plan()?,
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
        Ok((Glm4DecoderPlan::new(dense, vec![sparse])?, expert_plans))
    }

    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn with_layer_scratch<T>(
        operation: impl FnOnce(Glm4DenseLayerScratch<'_>, Glm4SparseLayerScratch<'_>) -> T,
    ) -> T {
        let mut dense_mla_linear_encoded = [0u8; 16];
        let mut dense_mla_linear_decoded = [0.0f32; 0];
        let mut dense_mla_linear_accumulators = [0.0f64; 0];
        let dense_mla_linear = LinearScratch::new(
            &mut dense_mla_linear_encoded,
            &mut dense_mla_linear_decoded,
            &mut dense_mla_linear_accumulators,
        );
        let mut dense_mla_norm_encoded = [0u8; 8];
        let mut dense_mla_norm_weights = [0.0f64; 2];
        let mut dense_q_a = [0.0f64; 2];
        let mut dense_kv_a = [0.0f64; 4];
        let mut dense_mla_normalized = [0.0f64; 2];
        let mut dense_q_projected = [0.0f64; 3];
        let mut dense_kv_projected = [0.0f64; 3];
        let mut dense_mla_query = [0.0f64; 3];
        let mut dense_mla_key = [0.0f64; 3];
        let mut dense_mla_value = [0.0f64; 2];
        let dense_mla = Glm4MlaScratch::new(
            dense_mla_linear,
            &mut dense_mla_norm_encoded,
            &mut dense_mla_norm_weights,
            &mut dense_q_a,
            &mut dense_kv_a,
            &mut dense_mla_normalized,
            &mut dense_q_projected,
            &mut dense_kv_projected,
            &mut dense_mla_query,
            &mut dense_mla_key,
            &mut dense_mla_value,
        );
        let mut dense_cache_staging = [0u8; 20];
        let mut dense_attention_encoded = [0u8; 12];
        let mut dense_attention_key = [0.0f64; 3];
        let mut dense_attention_value = [0.0f64; 2];
        let mut dense_attention_transactional = [0.0f64; 2];
        let dense_attention = FullAttentionScratch::new(
            &mut dense_attention_encoded,
            &mut dense_attention_key,
            &mut dense_attention_value,
            &mut dense_attention_transactional,
        );
        let mut dense_output_linear_encoded = [0u8; 16];
        let mut dense_output_linear_decoded = [0.0f32; 0];
        let mut dense_output_linear_accumulators = [0.0f64; 0];
        let dense_output_linear = LinearScratch::new(
            &mut dense_output_linear_encoded,
            &mut dense_output_linear_decoded,
            &mut dense_output_linear_accumulators,
        );
        let mut dense_mlp_linear_encoded = [0u8; 16];
        let mut dense_mlp_linear_decoded = [0.0f32; 0];
        let mut dense_mlp_linear_accumulators = [0.0f64; 0];
        let dense_mlp_linear = LinearScratch::new(
            &mut dense_mlp_linear_encoded,
            &mut dense_mlp_linear_decoded,
            &mut dense_mlp_linear_accumulators,
        );
        let mut dense_mlp_gate = [0.0f64; 2];
        let mut dense_mlp_up = [0.0f64; 2];
        let dense_mlp =
            GatedMlpScratch::new(dense_mlp_linear, &mut dense_mlp_gate, &mut dense_mlp_up);
        let mut dense_norm_encoded = [0u8; 8];
        let mut dense_norm_weights = [0.0f64; 2];
        let mut dense_normalized = [0.0f64; 2];
        let mut dense_query = [0.0f64; 3];
        let mut dense_key = [0.0f64; 3];
        let mut dense_value = [0.0f64; 2];
        let mut dense_attention_output = [0.0f64; 2];
        let mut dense_output_projection = [0.0f64; 2];
        let mut dense_residual = [0.0f64; 2];
        let mut dense_post_normalized = [0.0f64; 2];
        let mut dense_mlp_output = [0.0f64; 2];
        let mut dense_final_output = [0.0f64; 2];
        let dense_scratch = Glm4DenseLayerScratch::new(
            dense_mla,
            &mut dense_cache_staging,
            dense_attention,
            dense_output_linear,
            dense_mlp,
            &mut dense_norm_encoded,
            &mut dense_norm_weights,
            &mut dense_normalized,
            &mut dense_query,
            &mut dense_key,
            &mut dense_value,
            &mut dense_attention_output,
            &mut dense_output_projection,
            &mut dense_residual,
            &mut dense_post_normalized,
            &mut dense_mlp_output,
            &mut dense_final_output,
        );

        let mut sparse_mla_linear_encoded = [0u8; 16];
        let mut sparse_mla_linear_decoded = [0.0f32; 0];
        let mut sparse_mla_linear_accumulators = [0.0f64; 0];
        let sparse_mla_linear = LinearScratch::new(
            &mut sparse_mla_linear_encoded,
            &mut sparse_mla_linear_decoded,
            &mut sparse_mla_linear_accumulators,
        );
        let mut sparse_mla_norm_encoded = [0u8; 8];
        let mut sparse_mla_norm_weights = [0.0f64; 2];
        let mut sparse_q_a = [0.0f64; 2];
        let mut sparse_kv_a = [0.0f64; 4];
        let mut sparse_mla_normalized = [0.0f64; 2];
        let mut sparse_q_projected = [0.0f64; 3];
        let mut sparse_kv_projected = [0.0f64; 3];
        let mut sparse_mla_query = [0.0f64; 3];
        let mut sparse_mla_key = [0.0f64; 3];
        let mut sparse_mla_value = [0.0f64; 2];
        let sparse_mla = Glm4MlaScratch::new(
            sparse_mla_linear,
            &mut sparse_mla_norm_encoded,
            &mut sparse_mla_norm_weights,
            &mut sparse_q_a,
            &mut sparse_kv_a,
            &mut sparse_mla_normalized,
            &mut sparse_q_projected,
            &mut sparse_kv_projected,
            &mut sparse_mla_query,
            &mut sparse_mla_key,
            &mut sparse_mla_value,
        );
        let mut sparse_cache_staging = [0u8; 20];
        let mut sparse_attention_encoded = [0u8; 12];
        let mut sparse_attention_key = [0.0f64; 3];
        let mut sparse_attention_value = [0.0f64; 2];
        let mut sparse_attention_transactional = [0.0f64; 2];
        let sparse_attention = FullAttentionScratch::new(
            &mut sparse_attention_encoded,
            &mut sparse_attention_key,
            &mut sparse_attention_value,
            &mut sparse_attention_transactional,
        );
        let mut sparse_output_linear_encoded = [0u8; 16];
        let mut sparse_output_linear_decoded = [0.0f32; 0];
        let mut sparse_output_linear_accumulators = [0.0f64; 0];
        let sparse_output_linear = LinearScratch::new(
            &mut sparse_output_linear_encoded,
            &mut sparse_output_linear_decoded,
            &mut sparse_output_linear_accumulators,
        );
        let mut sparse_moe_linear_encoded = [0u8; 16];
        let mut sparse_moe_linear_decoded = [0.0f32; 0];
        let mut sparse_moe_linear_accumulators = [0.0f64; 0];
        let sparse_moe_linear = LinearScratch::new(
            &mut sparse_moe_linear_encoded,
            &mut sparse_moe_linear_decoded,
            &mut sparse_moe_linear_accumulators,
        );
        let mut sparse_moe_gate = [0.0f64; 2];
        let mut sparse_moe_up = [0.0f64; 2];
        let sparse_moe_mlp =
            GatedMlpScratch::new(sparse_moe_linear, &mut sparse_moe_gate, &mut sparse_moe_up);
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
        let sparse_moe = SparseMoeScratch::new(
            sparse_moe_mlp,
            &mut router_logits,
            routing,
            &mut expert_indices,
            &mut expert_weights,
            &mut expert_output,
            &mut accumulator,
        );
        let mut sparse_norm_encoded = [0u8; 8];
        let mut sparse_correction_encoded = [0u8; 8];
        let mut sparse_norm_weights = [0.0f64; 2];
        let mut sparse_correction_bias = [0.0f64; 2];
        let mut sparse_normalized = [0.0f64; 2];
        let mut sparse_query = [0.0f64; 3];
        let mut sparse_key = [0.0f64; 3];
        let mut sparse_value = [0.0f64; 2];
        let mut sparse_attention_output = [0.0f64; 2];
        let mut sparse_output_projection = [0.0f64; 2];
        let mut sparse_residual = [0.0f64; 2];
        let mut sparse_post_normalized = [0.0f64; 2];
        let mut sparse_moe_output = [0.0f64; 2];
        let mut sparse_final_output = [0.0f64; 2];
        let sparse_scratch = Glm4SparseLayerScratch::new(
            sparse_mla,
            &mut sparse_cache_staging,
            sparse_attention,
            sparse_output_linear,
            sparse_moe,
            &mut sparse_norm_encoded,
            &mut sparse_correction_encoded,
            &mut sparse_norm_weights,
            &mut sparse_correction_bias,
            &mut sparse_normalized,
            &mut sparse_query,
            &mut sparse_key,
            &mut sparse_value,
            &mut sparse_attention_output,
            &mut sparse_output_projection,
            &mut sparse_residual,
            &mut sparse_post_normalized,
            &mut sparse_moe_output,
            &mut sparse_final_output,
        );
        operation(dense_scratch, sparse_scratch)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_fixture(
        plan: &Glm4DecoderPlan,
        readers: &Glm4DecoderReaders<'_, '_, '_>,
        caches: &mut [KvCache<'_>; 2],
        position: usize,
        hidden: &[f64; 2],
        output: &mut [f64; 2],
    ) -> Result<(), AmsError> {
        with_layer_scratch(|mut dense_scratch, mut sparse_scratch| {
            let mut hidden_a = [0.0f64; 2];
            let mut hidden_b = [0.0f64; 2];
            glm4_decoder_token(
                plan,
                readers,
                caches,
                position,
                hidden,
                &mut dense_scratch,
                &mut sparse_scratch,
                &mut hidden_a,
                &mut hidden_b,
                output,
            )
        })
    }

    fn with_model_scratch<T>(operation: impl FnOnce(&mut Glm4ModelScratch<'_>) -> T) -> T {
        with_layer_scratch(|dense_scratch, sparse_scratch| {
            let mut vector_encoded = [0u8; 8];
            let mut lm_encoded = [0u8; 16];
            let mut lm_decoded = [0.0f32; 0];
            let mut lm_accumulators = [0.0f64; 0];
            let lm_scratch =
                LinearScratch::new(&mut lm_encoded, &mut lm_decoded, &mut lm_accumulators);
            let mut input_hidden = [0.0f64; 2];
            let mut hidden_a = [0.0f64; 2];
            let mut hidden_b = [0.0f64; 2];
            let mut decoder_output = [0.0f64; 2];
            let mut norm_weights = [0.0f64; 2];
            let mut normalized = [0.0f64; 2];
            let mut logits = [0.0f64; 3];
            let mut scratch = Glm4ModelScratch::new(
                dense_scratch,
                sparse_scratch,
                &mut vector_encoded,
                lm_scratch,
                &mut input_hidden,
                &mut hidden_a,
                &mut hidden_b,
                &mut decoder_output,
                &mut norm_weights,
                &mut normalized,
                &mut logits,
            );
            operation(&mut scratch)
        })
    }

    fn run_model_fixture(
        plan: &Glm4ModelPlan,
        readers: &Glm4ModelReaders<'_, '_, '_>,
        caches: &mut [KvCache<'_>; 2],
        position: usize,
        input_token: usize,
    ) -> Result<usize, AmsError> {
        with_model_scratch(|scratch| {
            glm4_model_next_token(plan, readers, caches, position, input_token, scratch)
        })
    }

    fn advance_generation_fixture(
        plan: &Glm4ModelPlan,
        readers: &Glm4ModelReaders<'_, '_, '_>,
        caches: &mut [KvCache<'_>; 2],
        session: &mut Glm4GreedySession<'_>,
        cancelled: bool,
    ) -> Result<Glm4GenerationStep, AmsError> {
        with_model_scratch(|scratch| {
            glm4_greedy_advance(plan, readers, caches, session, scratch, cancelled)
        })
    }

    #[test]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn decoder_preflights_all_layers_and_rolls_back_a_late_failure() -> Result<(), AmsError> {
        let (plan, expert_plans) = fixture_plan()?;
        assert_eq!(plan.layer_count(), 2);
        let norm = CountingReader::new(encode_f32(&[1.0, 1.0]));
        let correction = encode_f32(&[0.0, 1.0]);
        let zeros = encode_f32(&[0.0; 8]);
        let bad = encode_f32(&[f32::NAN, 0.0, 0.0, 0.0]);
        let zero_reader = SliceReader::new(&zeros);
        let correction_reader = SliceReader::new(&correction);
        let bad_reader = SliceReader::new(&bad);
        let dense_mla = Glm4MlaReaders::new(
            &zero_reader,
            &norm,
            &zero_reader,
            &zero_reader,
            &norm,
            &zero_reader,
        );
        let dense_mlp = GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader);
        let dense_readers =
            Glm4DenseLayerReaders::new(&norm, dense_mla, &zero_reader, &norm, dense_mlp);
        let shared = GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader);
        let incomplete_experts = [GatedMlpReaders::new(
            &zero_reader,
            &zero_reader,
            &zero_reader,
        )];
        let incomplete_moe =
            SparseMoeBindings::new(&zero_reader, &expert_plans, &incomplete_experts, &shared);
        let incomplete_sparse_mla = Glm4MlaReaders::new(
            &zero_reader,
            &norm,
            &zero_reader,
            &zero_reader,
            &norm,
            &zero_reader,
        );
        let incomplete_sparse = Glm4SparseLayerReaders::new(
            &norm,
            incomplete_sparse_mla,
            &zero_reader,
            &norm,
            &correction_reader,
            incomplete_moe,
        );
        let incomplete_sparse_readers = [incomplete_sparse];
        let incomplete_readers = Glm4DecoderReaders::new(dense_readers, &incomplete_sparse_readers);
        let cache_plan =
            KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        let mut dense_keys = [0u8; 24];
        let mut dense_values = [0u8; 16];
        let mut sparse_keys = [0u8; 24];
        let mut sparse_values = [0u8; 16];
        let dense_cache = KvCache::new(cache_plan, &mut dense_keys, &mut dense_values)?;
        let sparse_cache = KvCache::new(cache_plan, &mut sparse_keys, &mut sparse_values)?;
        let mut caches = [dense_cache, sparse_cache];
        let mut output = [41.0f64; 2];
        let error = run_fixture(
            &plan,
            &incomplete_readers,
            &mut caches,
            0,
            &[1.0, -1.0],
            &mut output,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        assert_eq!(norm.reads.get(), 0);
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 0));
        assert_eq!(output.map(f64::to_bits), [41.0f64.to_bits(); 2]);

        let bad_experts = [
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
            GatedMlpReaders::new(&zero_reader, &zero_reader, &bad_reader),
        ];
        let bad_moe = SparseMoeBindings::new(&zero_reader, &expert_plans, &bad_experts, &shared);
        let bad_sparse_mla = Glm4MlaReaders::new(
            &zero_reader,
            &norm,
            &zero_reader,
            &zero_reader,
            &norm,
            &zero_reader,
        );
        let bad_sparse = Glm4SparseLayerReaders::new(
            &norm,
            bad_sparse_mla,
            &zero_reader,
            &norm,
            &correction_reader,
            bad_moe,
        );
        let bad_sparse_readers = [bad_sparse];
        let bad_dense_mla = Glm4MlaReaders::new(
            &zero_reader,
            &norm,
            &zero_reader,
            &zero_reader,
            &norm,
            &zero_reader,
        );
        let bad_dense = Glm4DenseLayerReaders::new(
            &norm,
            bad_dense_mla,
            &zero_reader,
            &norm,
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
        );
        let bad_readers = Glm4DecoderReaders::new(bad_dense, &bad_sparse_readers);
        let error = run_fixture(
            &plan,
            &bad_readers,
            &mut caches,
            0,
            &[1.0, -1.0],
            &mut output,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 0));
        assert_eq!(output.map(f64::to_bits), [41.0f64.to_bits(); 2]);

        let good_experts = [
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
        ];
        let good_moe = SparseMoeBindings::new(&zero_reader, &expert_plans, &good_experts, &shared);
        let good_sparse_mla = Glm4MlaReaders::new(
            &zero_reader,
            &norm,
            &zero_reader,
            &zero_reader,
            &norm,
            &zero_reader,
        );
        let good_sparse = Glm4SparseLayerReaders::new(
            &norm,
            good_sparse_mla,
            &zero_reader,
            &norm,
            &correction_reader,
            good_moe,
        );
        let good_sparse_readers = [good_sparse];
        let good_dense_mla = Glm4MlaReaders::new(
            &zero_reader,
            &norm,
            &zero_reader,
            &zero_reader,
            &norm,
            &zero_reader,
        );
        let good_dense = Glm4DenseLayerReaders::new(
            &norm,
            good_dense_mla,
            &zero_reader,
            &norm,
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
        );
        let good_readers = Glm4DecoderReaders::new(good_dense, &good_sparse_readers);
        run_fixture(
            &plan,
            &good_readers,
            &mut caches,
            0,
            &[1.0, -1.0],
            &mut output,
        )?;
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 1));
        assert_eq!(
            output.map(f64::to_bits),
            [1.0f64.to_bits(), (-1.0f64).to_bits()]
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn model_executes_embedding_to_argmax_and_rolls_back_a_bad_lm_head() -> Result<(), AmsError> {
        let (decoder, expert_plans) = fixture_plan()?;
        let model = Glm4ModelPlan::new(
            decoder,
            linear_plan(3, 2)?,
            Glm4ModelVectorLayout::new(0, 0, IdentityDType::Float32, IdentityDType::Float32),
            2,
            1e-5,
        )?;
        assert_eq!(model.model_vocabulary_size(), 3);
        assert_eq!(model.tokenizer_vocabulary_size(), 2);
        assert_eq!(model.scratch().local_bytes, 144);
        let norm = encode_f32(&[1.0, 1.0]);
        let correction = encode_f32(&[0.0, 1.0]);
        let zeros = encode_f32(&[0.0; 8]);
        let embeddings = encode_f32(&[0.0, 0.0, 1.0, -1.0, -1.0, 1.0]);
        // The unmapped third row scores above the second row and must not be selected.
        let lm_head = encode_f32(&[0.0, 0.0, 1.0, 0.0, 1.0, -1.0]);
        let bad_lm_head = encode_f32(&[f32::NAN, 0.0, 1.0, 0.0, 1.0, -1.0]);
        let short_lm_head = lm_head[..20].to_vec();
        let norm_reader = SliceReader::new(&norm);
        let correction_reader = SliceReader::new(&correction);
        let zero_reader = SliceReader::new(&zeros);
        let embedding_reader = CountingReader::new(embeddings);
        let lm_head_reader = CountingReader::new(lm_head);
        let short_lm_head_reader = SliceReader::new(&short_lm_head);
        let bad_lm_head_reader = SliceReader::new(&bad_lm_head);
        let shared = GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader);
        let experts = [
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
            GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
        ];

        let cache_plan =
            KvCachePlan::new(1, 3, 2, 2, IdentityDType::Float32, IdentityDType::Float32)?;
        let mut dense_keys = [0u8; 24];
        let mut dense_values = [0u8; 16];
        let mut sparse_keys = [0u8; 24];
        let mut sparse_values = [0u8; 16];
        let dense_cache = KvCache::new(cache_plan, &mut dense_keys, &mut dense_values)?;
        let sparse_cache = KvCache::new(cache_plan, &mut sparse_keys, &mut sparse_values)?;
        let mut caches = [dense_cache, sparse_cache];

        let make_decoder_readers = || {
            let dense_mla = Glm4MlaReaders::new(
                &zero_reader,
                &norm_reader,
                &zero_reader,
                &zero_reader,
                &norm_reader,
                &zero_reader,
            );
            let dense = Glm4DenseLayerReaders::new(
                &norm_reader,
                dense_mla,
                &zero_reader,
                &norm_reader,
                GatedMlpReaders::new(&zero_reader, &zero_reader, &zero_reader),
            );
            let moe = SparseMoeBindings::new(&zero_reader, &expert_plans, &experts, &shared);
            let sparse_mla = Glm4MlaReaders::new(
                &zero_reader,
                &norm_reader,
                &zero_reader,
                &zero_reader,
                &norm_reader,
                &zero_reader,
            );
            let sparse = Glm4SparseLayerReaders::new(
                &norm_reader,
                sparse_mla,
                &zero_reader,
                &norm_reader,
                &correction_reader,
                moe,
            );
            (dense, sparse)
        };

        let (short_dense, short_sparse) = make_decoder_readers();
        let short_sparse_readers = [short_sparse];
        let short_readers = Glm4ModelReaders::new(
            &embedding_reader,
            Glm4DecoderReaders::new(short_dense, &short_sparse_readers),
            &norm_reader,
            &short_lm_head_reader,
        );
        let error = run_model_fixture(&model, &short_readers, &mut caches, 0, 2).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        assert_eq!(embedding_reader.reads.get(), 0);
        let prompt = [1, 1];
        let eos = [0];
        let mut short_session = Glm4GreedySession::new(&model, &prompt, &eos, 1)?;
        let error = advance_generation_fixture(
            &model,
            &short_readers,
            &mut caches,
            &mut short_session,
            false,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        assert_eq!(embedding_reader.reads.get(), 0);
        assert_eq!(short_session.position(), 0);
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 0));

        let (dense, sparse) = make_decoder_readers();
        let sparse_readers = [sparse];
        let readers = Glm4ModelReaders::new(
            &embedding_reader,
            Glm4DecoderReaders::new(dense, &sparse_readers),
            &norm_reader,
            &lm_head_reader,
        );
        let mut session = Glm4GreedySession::new(&model, &prompt, &eos, 1)?;
        let error =
            advance_generation_fixture(&model, &readers, &mut caches, &mut session, true).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::Cancelled));
        assert_eq!(session.position(), 0);
        assert_eq!(embedding_reader.reads.get(), 0);
        assert_eq!(lm_head_reader.reads.get(), 0);

        let step = advance_generation_fixture(&model, &readers, &mut caches, &mut session, false)?;
        assert_eq!(
            step,
            Glm4GenerationStep::Prefill {
                consumed_tokens: 1,
                total_tokens: 2
            }
        );
        assert_eq!(lm_head_reader.reads.get(), 0);
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 1));

        let mut disagreeing_session = Glm4GreedySession::new(&model, &prompt, &eos, 1)?;
        let reads_before_disagreement = embedding_reader.reads.get();
        let error = advance_generation_fixture(
            &model,
            &readers,
            &mut caches,
            &mut disagreeing_session,
            false,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        assert_eq!(embedding_reader.reads.get(), reads_before_disagreement);
        assert_eq!(lm_head_reader.reads.get(), 0);

        let (bad_dense, bad_sparse) = make_decoder_readers();
        let bad_sparse_readers = [bad_sparse];
        let bad_readers = Glm4ModelReaders::new(
            &embedding_reader,
            Glm4DecoderReaders::new(bad_dense, &bad_sparse_readers),
            &norm_reader,
            &bad_lm_head_reader,
        );
        let error =
            advance_generation_fixture(&model, &bad_readers, &mut caches, &mut session, false)
                .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::NumericFailure));
        assert_eq!(session.position(), 1);
        assert_eq!(session.prompt_consumed(), 1);
        assert_eq!(session.generated_tokens(), 0);
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 1));

        let (retry_dense, retry_sparse) = make_decoder_readers();
        let retry_sparse_readers = [retry_sparse];
        let retry_readers = Glm4ModelReaders::new(
            &embedding_reader,
            Glm4DecoderReaders::new(retry_dense, &retry_sparse_readers),
            &norm_reader,
            &lm_head_reader,
        );
        let step =
            advance_generation_fixture(&model, &retry_readers, &mut caches, &mut session, false)?;
        assert_eq!(
            step,
            Glm4GenerationStep::Finished {
                token_id: Some(1),
                reason: Glm4FinishReason::Length
            }
        );
        assert_eq!(session.position(), 2);
        assert_eq!(session.generated_tokens(), 1);
        assert_eq!(session.pending_input(), Some(1));
        assert_eq!(session.finish_reason(), Some(Glm4FinishReason::Length));
        assert!(lm_head_reader.reads.get() > 0);
        assert!(caches.iter().all(|cache| cache.committed_tokens() == 2));
        Ok(())
    }
}
