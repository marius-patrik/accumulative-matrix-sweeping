use ams_core::{
    ErrorCode, FullAttentionScratch, FullAttentionScratchRequirements, GatedMlpScratch,
    GatedMlpScratchRequirements, Glm4DenseLayerScratch, Glm4DenseLayerScratchRequirements,
    Glm4MlaScratch, Glm4MlaScratchRequirements, Glm4ModelPlan, Glm4ModelScratch,
    Glm4SparseLayerScratch, Glm4SparseLayerScratchRequirements, GlmRouterScratch, KvCache,
    KvCachePlan, LinearScratch, LinearScratchRequirements, SparseMoeScratch,
    SparseMoeScratchRequirements,
};

use crate::RuntimeError;

fn allocate_zeroed<T: Clone + Default>(length: usize) -> Result<Vec<T>, RuntimeError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length).map_err(|_| {
        RuntimeError::new(
            ErrorCode::PreflightNoWorkingSet,
            "native owned working-set allocation failed",
        )
    })?;
    values.resize(length, T::default());
    Ok(values)
}

fn slice_bytes<T>(values: &[T]) -> Result<usize, RuntimeError> {
    values.len().checked_mul(size_of::<T>()).ok_or_else(|| {
        RuntimeError::new(
            ErrorCode::InternalInvariant,
            "owned working-set byte count overflowed",
        )
    })
}

fn checked_total(parts: impl IntoIterator<Item = usize>) -> Result<usize, RuntimeError> {
    parts.into_iter().try_fold(0usize, |total, part| {
        total.checked_add(part).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::InternalInvariant,
                "owned working-set total overflowed",
            )
        })
    })
}

struct LinearStorage {
    encoded: Vec<u8>,
    decoded: Vec<f32>,
    accumulators: Vec<f64>,
}

impl LinearStorage {
    fn new(requirement: LinearScratchRequirements) -> Result<Self, RuntimeError> {
        Self::from_many([requirement])
    }

    fn from_many(
        requirements: impl IntoIterator<Item = LinearScratchRequirements>,
    ) -> Result<Self, RuntimeError> {
        let mut encoded = 0usize;
        let mut decoded = 0usize;
        let mut accumulators = 0usize;
        for requirement in requirements {
            encoded = encoded.max(requirement.encoded_bytes);
            decoded = decoded.max(requirement.decoded_elements);
            accumulators = accumulators.max(requirement.accumulator_elements);
        }
        Ok(Self {
            encoded: allocate_zeroed(encoded)?,
            decoded: allocate_zeroed(decoded)?,
            accumulators: allocate_zeroed(accumulators)?,
        })
    }

    fn scratch(&mut self) -> LinearScratch<'_> {
        LinearScratch::new(&mut self.encoded, &mut self.decoded, &mut self.accumulators)
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            slice_bytes(&self.encoded)?,
            slice_bytes(&self.decoded)?,
            slice_bytes(&self.accumulators)?,
        ])
    }
}

struct AttentionStorage {
    encoded: Vec<u8>,
    key: Vec<f64>,
    value: Vec<f64>,
    output: Vec<f64>,
}

impl AttentionStorage {
    fn new(requirement: FullAttentionScratchRequirements) -> Result<Self, RuntimeError> {
        Self::from_many([requirement])
    }

    fn from_many(
        requirements: impl IntoIterator<Item = FullAttentionScratchRequirements>,
    ) -> Result<Self, RuntimeError> {
        let mut encoded = 0usize;
        let mut key = 0usize;
        let mut value = 0usize;
        let mut output = 0usize;
        for requirement in requirements {
            encoded = encoded.max(requirement.encoded_bytes);
            key = key.max(requirement.key_elements);
            value = value.max(requirement.value_elements);
            output = output.max(requirement.output_elements);
        }
        Ok(Self {
            encoded: allocate_zeroed(encoded)?,
            key: allocate_zeroed(key)?,
            value: allocate_zeroed(value)?,
            output: allocate_zeroed(output)?,
        })
    }

    fn scratch(&mut self) -> FullAttentionScratch<'_> {
        FullAttentionScratch::new(
            &mut self.encoded,
            &mut self.key,
            &mut self.value,
            &mut self.output,
        )
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            slice_bytes(&self.encoded)?,
            slice_bytes(&self.key)?,
            slice_bytes(&self.value)?,
            slice_bytes(&self.output)?,
        ])
    }
}

struct MlaStorage {
    linear: LinearStorage,
    norm_encoded: Vec<u8>,
    norm_weights: Vec<f64>,
    q_a: Vec<f64>,
    kv_a: Vec<f64>,
    normalized: Vec<f64>,
    q_projected: Vec<f64>,
    kv_projected: Vec<f64>,
    query_output: Vec<f64>,
    key_output: Vec<f64>,
    value_output: Vec<f64>,
}

impl MlaStorage {
    fn new(requirement: Glm4MlaScratchRequirements) -> Result<Self, RuntimeError> {
        Self::from_many(&std::iter::once(requirement))
    }

    fn from_many<I>(requirements: &I) -> Result<Self, RuntimeError>
    where
        I: Clone + Iterator<Item = Glm4MlaScratchRequirements>,
    {
        let maximum = |field: fn(&Glm4MlaScratchRequirements) -> usize| {
            requirements
                .clone()
                .map(|requirement| field(&requirement))
                .max()
                .unwrap_or(0)
        };
        Ok(Self {
            linear: LinearStorage::from_many(requirements.clone().map(|value| value.linear))?,
            norm_encoded: allocate_zeroed(maximum(|value| value.norm_encoded_bytes))?,
            norm_weights: allocate_zeroed(maximum(|value| value.norm_weight_elements))?,
            q_a: allocate_zeroed(maximum(|value| value.q_a_elements))?,
            kv_a: allocate_zeroed(maximum(|value| value.kv_a_elements))?,
            normalized: allocate_zeroed(maximum(|value| value.normalized_elements))?,
            q_projected: allocate_zeroed(maximum(|value| value.q_projected_elements))?,
            kv_projected: allocate_zeroed(maximum(|value| value.kv_projected_elements))?,
            query_output: allocate_zeroed(maximum(|value| value.query_output_elements))?,
            key_output: allocate_zeroed(maximum(|value| value.key_output_elements))?,
            value_output: allocate_zeroed(maximum(|value| value.value_output_elements))?,
        })
    }

    fn scratch(&mut self) -> Glm4MlaScratch<'_> {
        Glm4MlaScratch::new(
            self.linear.scratch(),
            &mut self.norm_encoded,
            &mut self.norm_weights,
            &mut self.q_a,
            &mut self.kv_a,
            &mut self.normalized,
            &mut self.q_projected,
            &mut self.kv_projected,
            &mut self.query_output,
            &mut self.key_output,
            &mut self.value_output,
        )
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            self.linear.heap_bytes()?,
            slice_bytes(&self.norm_encoded)?,
            slice_bytes(&self.norm_weights)?,
            slice_bytes(&self.q_a)?,
            slice_bytes(&self.kv_a)?,
            slice_bytes(&self.normalized)?,
            slice_bytes(&self.q_projected)?,
            slice_bytes(&self.kv_projected)?,
            slice_bytes(&self.query_output)?,
            slice_bytes(&self.key_output)?,
            slice_bytes(&self.value_output)?,
        ])
    }
}

struct GatedStorage {
    linear: LinearStorage,
    gate: Vec<f64>,
    up: Vec<f64>,
}

impl GatedStorage {
    fn new(requirement: GatedMlpScratchRequirements) -> Result<Self, RuntimeError> {
        Self::from_many(&std::iter::once(requirement))
    }

    fn from_many<I>(requirements: &I) -> Result<Self, RuntimeError>
    where
        I: Clone + Iterator<Item = GatedMlpScratchRequirements>,
    {
        let intermediate = requirements
            .clone()
            .map(|requirement| requirement.intermediate_elements)
            .max()
            .unwrap_or(0);
        if intermediate % 2 != 0 {
            return Err(RuntimeError::new(
                ErrorCode::InternalInvariant,
                "gated MLP intermediate storage is not two equal vectors",
            ));
        }
        let per_intermediate = intermediate / 2;
        Ok(Self {
            linear: LinearStorage::from_many(requirements.clone().map(|value| value.linear))?,
            gate: allocate_zeroed(per_intermediate)?,
            up: allocate_zeroed(per_intermediate)?,
        })
    }

    fn scratch(&mut self) -> GatedMlpScratch<'_> {
        GatedMlpScratch::new(self.linear.scratch(), &mut self.gate, &mut self.up)
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            self.linear.heap_bytes()?,
            slice_bytes(&self.gate)?,
            slice_bytes(&self.up)?,
        ])
    }
}

struct SparseMoeStorage {
    mlp: GatedStorage,
    router_logits: Vec<f64>,
    probabilities: Vec<f64>,
    corrected: Vec<f64>,
    group_scores: Vec<f64>,
    selected_groups: Vec<usize>,
    expert_indices: Vec<usize>,
    expert_weights: Vec<f64>,
    expert_output: Vec<f64>,
    accumulator: Vec<f64>,
}

impl SparseMoeStorage {
    fn from_many<I>(requirements: &I) -> Result<Self, RuntimeError>
    where
        I: Clone + Iterator<Item = SparseMoeScratchRequirements>,
    {
        let maximum = |field: fn(&SparseMoeScratchRequirements) -> usize| {
            requirements
                .clone()
                .map(|requirement| field(&requirement))
                .max()
                .unwrap_or(0)
        };
        let expert_buffer = maximum(|value| value.expert_buffer_elements);
        if expert_buffer % 2 != 0 {
            return Err(RuntimeError::new(
                ErrorCode::InternalInvariant,
                "sparse MoE expert buffer is not two equal vectors",
            ));
        }
        let hidden = expert_buffer / 2;
        Ok(Self {
            mlp: GatedStorage::from_many(&requirements.clone().map(|value| value.mlp))?,
            router_logits: allocate_zeroed(maximum(|value| value.router_logits_elements))?,
            probabilities: allocate_zeroed(maximum(|value| value.router_probability_elements))?,
            corrected: allocate_zeroed(maximum(|value| value.router_corrected_elements))?,
            group_scores: allocate_zeroed(maximum(|value| value.router_group_score_elements))?,
            selected_groups: allocate_zeroed(maximum(|value| {
                value.router_selected_group_elements
            }))?,
            expert_indices: allocate_zeroed(maximum(|value| value.selected_expert_elements))?,
            expert_weights: allocate_zeroed(maximum(|value| value.selected_expert_elements))?,
            expert_output: allocate_zeroed(hidden)?,
            accumulator: allocate_zeroed(hidden)?,
        })
    }

    fn scratch(&mut self) -> SparseMoeScratch<'_> {
        SparseMoeScratch::new(
            self.mlp.scratch(),
            &mut self.router_logits,
            GlmRouterScratch::new(
                &mut self.probabilities,
                &mut self.corrected,
                &mut self.group_scores,
                &mut self.selected_groups,
            ),
            &mut self.expert_indices,
            &mut self.expert_weights,
            &mut self.expert_output,
            &mut self.accumulator,
        )
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            self.mlp.heap_bytes()?,
            slice_bytes(&self.router_logits)?,
            slice_bytes(&self.probabilities)?,
            slice_bytes(&self.corrected)?,
            slice_bytes(&self.group_scores)?,
            slice_bytes(&self.selected_groups)?,
            slice_bytes(&self.expert_indices)?,
            slice_bytes(&self.expert_weights)?,
            slice_bytes(&self.expert_output)?,
            slice_bytes(&self.accumulator)?,
        ])
    }
}

struct DenseStorage {
    mla: MlaStorage,
    cache_staging: Vec<u8>,
    attention: AttentionStorage,
    output_linear: LinearStorage,
    mlp: GatedStorage,
    norm_encoded: Vec<u8>,
    norm_weights: Vec<f64>,
    normalized: Vec<f64>,
    query: Vec<f64>,
    key: Vec<f64>,
    value: Vec<f64>,
    attention_output: Vec<f64>,
    output_projection: Vec<f64>,
    residual: Vec<f64>,
    post_normalized: Vec<f64>,
    mlp_output: Vec<f64>,
    final_output: Vec<f64>,
}

impl DenseStorage {
    fn new(requirement: &Glm4DenseLayerScratchRequirements) -> Result<Self, RuntimeError> {
        Ok(Self {
            mla: MlaStorage::new(requirement.mla)?,
            cache_staging: allocate_zeroed(requirement.cache_staging_bytes)?,
            attention: AttentionStorage::new(requirement.attention)?,
            output_linear: LinearStorage::new(requirement.output_linear)?,
            mlp: GatedStorage::new(requirement.mlp)?,
            norm_encoded: allocate_zeroed(requirement.norm_encoded_bytes)?,
            norm_weights: allocate_zeroed(requirement.norm_weight_elements)?,
            normalized: allocate_zeroed(requirement.normalized_elements)?,
            query: allocate_zeroed(requirement.query_elements)?,
            key: allocate_zeroed(requirement.key_elements)?,
            value: allocate_zeroed(requirement.value_elements)?,
            attention_output: allocate_zeroed(requirement.attention_output_elements)?,
            output_projection: allocate_zeroed(requirement.output_projection_elements)?,
            residual: allocate_zeroed(requirement.residual_elements)?,
            post_normalized: allocate_zeroed(requirement.post_normalized_elements)?,
            mlp_output: allocate_zeroed(requirement.mlp_output_elements)?,
            final_output: allocate_zeroed(requirement.final_output_elements)?,
        })
    }

    fn scratch(&mut self) -> Glm4DenseLayerScratch<'_> {
        Glm4DenseLayerScratch::new(
            self.mla.scratch(),
            &mut self.cache_staging,
            self.attention.scratch(),
            self.output_linear.scratch(),
            self.mlp.scratch(),
            &mut self.norm_encoded,
            &mut self.norm_weights,
            &mut self.normalized,
            &mut self.query,
            &mut self.key,
            &mut self.value,
            &mut self.attention_output,
            &mut self.output_projection,
            &mut self.residual,
            &mut self.post_normalized,
            &mut self.mlp_output,
            &mut self.final_output,
        )
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            self.mla.heap_bytes()?,
            slice_bytes(&self.cache_staging)?,
            self.attention.heap_bytes()?,
            self.output_linear.heap_bytes()?,
            self.mlp.heap_bytes()?,
            slice_bytes(&self.norm_encoded)?,
            slice_bytes(&self.norm_weights)?,
            slice_bytes(&self.normalized)?,
            slice_bytes(&self.query)?,
            slice_bytes(&self.key)?,
            slice_bytes(&self.value)?,
            slice_bytes(&self.attention_output)?,
            slice_bytes(&self.output_projection)?,
            slice_bytes(&self.residual)?,
            slice_bytes(&self.post_normalized)?,
            slice_bytes(&self.mlp_output)?,
            slice_bytes(&self.final_output)?,
        ])
    }
}

struct SparseStorage {
    mla: MlaStorage,
    cache_staging: Vec<u8>,
    attention: AttentionStorage,
    output_linear: LinearStorage,
    moe: SparseMoeStorage,
    norm_encoded: Vec<u8>,
    correction_encoded: Vec<u8>,
    norm_weights: Vec<f64>,
    correction_bias: Vec<f64>,
    normalized: Vec<f64>,
    query: Vec<f64>,
    key: Vec<f64>,
    value: Vec<f64>,
    attention_output: Vec<f64>,
    output_projection: Vec<f64>,
    residual: Vec<f64>,
    post_normalized: Vec<f64>,
    moe_output: Vec<f64>,
    final_output: Vec<f64>,
}

impl SparseStorage {
    fn new(requirements: &[Glm4SparseLayerScratchRequirements]) -> Result<Self, RuntimeError> {
        let maximum = |field: fn(&Glm4SparseLayerScratchRequirements) -> usize| {
            requirements.iter().map(field).max().unwrap_or(0)
        };
        Ok(Self {
            mla: MlaStorage::from_many(&requirements.iter().map(|value| value.mla))?,
            cache_staging: allocate_zeroed(maximum(|value| value.cache_staging_bytes))?,
            attention: AttentionStorage::from_many(
                requirements.iter().map(|value| value.attention),
            )?,
            output_linear: LinearStorage::from_many(
                requirements.iter().map(|value| value.output_linear),
            )?,
            moe: SparseMoeStorage::from_many(&requirements.iter().map(|value| value.moe))?,
            norm_encoded: allocate_zeroed(maximum(|value| value.norm_encoded_bytes))?,
            correction_encoded: allocate_zeroed(maximum(|value| value.correction_encoded_bytes))?,
            norm_weights: allocate_zeroed(maximum(|value| value.norm_weight_elements))?,
            correction_bias: allocate_zeroed(maximum(|value| value.correction_bias_elements))?,
            normalized: allocate_zeroed(maximum(|value| value.normalized_elements))?,
            query: allocate_zeroed(maximum(|value| value.query_elements))?,
            key: allocate_zeroed(maximum(|value| value.key_elements))?,
            value: allocate_zeroed(maximum(|value| value.value_elements))?,
            attention_output: allocate_zeroed(maximum(|value| value.attention_output_elements))?,
            output_projection: allocate_zeroed(maximum(|value| value.output_projection_elements))?,
            residual: allocate_zeroed(maximum(|value| value.residual_elements))?,
            post_normalized: allocate_zeroed(maximum(|value| value.post_normalized_elements))?,
            moe_output: allocate_zeroed(maximum(|value| value.moe_output_elements))?,
            final_output: allocate_zeroed(maximum(|value| value.final_output_elements))?,
        })
    }

    fn scratch(&mut self) -> Glm4SparseLayerScratch<'_> {
        Glm4SparseLayerScratch::new(
            self.mla.scratch(),
            &mut self.cache_staging,
            self.attention.scratch(),
            self.output_linear.scratch(),
            self.moe.scratch(),
            &mut self.norm_encoded,
            &mut self.correction_encoded,
            &mut self.norm_weights,
            &mut self.correction_bias,
            &mut self.normalized,
            &mut self.query,
            &mut self.key,
            &mut self.value,
            &mut self.attention_output,
            &mut self.output_projection,
            &mut self.residual,
            &mut self.post_normalized,
            &mut self.moe_output,
            &mut self.final_output,
        )
    }

    fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            self.mla.heap_bytes()?,
            slice_bytes(&self.cache_staging)?,
            self.attention.heap_bytes()?,
            self.output_linear.heap_bytes()?,
            self.moe.heap_bytes()?,
            slice_bytes(&self.norm_encoded)?,
            slice_bytes(&self.correction_encoded)?,
            slice_bytes(&self.norm_weights)?,
            slice_bytes(&self.correction_bias)?,
            slice_bytes(&self.normalized)?,
            slice_bytes(&self.query)?,
            slice_bytes(&self.key)?,
            slice_bytes(&self.value)?,
            slice_bytes(&self.attention_output)?,
            slice_bytes(&self.output_projection)?,
            slice_bytes(&self.residual)?,
            slice_bytes(&self.post_normalized)?,
            slice_bytes(&self.moe_output)?,
            slice_bytes(&self.final_output)?,
        ])
    }
}

pub struct ModelScratchStorage {
    dense: DenseStorage,
    sparse: SparseStorage,
    vector_encoded: Vec<u8>,
    lm_head: LinearStorage,
    input_hidden: Vec<f64>,
    hidden_a: Vec<f64>,
    hidden_b: Vec<f64>,
    decoder_output: Vec<f64>,
    norm_weights: Vec<f64>,
    normalized: Vec<f64>,
    logits: Vec<f64>,
}

impl ModelScratchStorage {
    pub fn new(plan: &Glm4ModelPlan) -> Result<Self, RuntimeError> {
        let requirement = plan.scratch();
        let decoder = plan.decoder();
        Ok(Self {
            dense: DenseStorage::new(&decoder.dense_scratch())?,
            sparse: SparseStorage::new(decoder.sparse_scratch())?,
            vector_encoded: allocate_zeroed(requirement.vector_encoded_bytes)?,
            lm_head: LinearStorage::new(requirement.lm_head)?,
            input_hidden: allocate_zeroed(requirement.hidden_elements)?,
            hidden_a: allocate_zeroed(requirement.hidden_elements)?,
            hidden_b: allocate_zeroed(requirement.hidden_elements)?,
            decoder_output: allocate_zeroed(requirement.hidden_elements)?,
            norm_weights: allocate_zeroed(requirement.hidden_elements)?,
            normalized: allocate_zeroed(requirement.hidden_elements)?,
            logits: allocate_zeroed(requirement.logit_elements)?,
        })
    }

    pub fn scratch(&mut self) -> Glm4ModelScratch<'_> {
        Glm4ModelScratch::new(
            self.dense.scratch(),
            self.sparse.scratch(),
            &mut self.vector_encoded,
            self.lm_head.scratch(),
            &mut self.input_hidden,
            &mut self.hidden_a,
            &mut self.hidden_b,
            &mut self.decoder_output,
            &mut self.norm_weights,
            &mut self.normalized,
            &mut self.logits,
        )
    }

    pub fn heap_bytes(&self) -> Result<usize, RuntimeError> {
        checked_total([
            self.dense.heap_bytes()?,
            self.sparse.heap_bytes()?,
            slice_bytes(&self.vector_encoded)?,
            self.lm_head.heap_bytes()?,
            slice_bytes(&self.input_hidden)?,
            slice_bytes(&self.hidden_a)?,
            slice_bytes(&self.hidden_b)?,
            slice_bytes(&self.decoder_output)?,
            slice_bytes(&self.norm_weights)?,
            slice_bytes(&self.normalized)?,
            slice_bytes(&self.logits)?,
        ])
    }
}

pub struct CacheStorage {
    keys: Vec<u8>,
    values: Vec<u8>,
}

impl CacheStorage {
    pub fn allocate_all(plan: KvCachePlan, layer_count: usize) -> Result<Vec<Self>, RuntimeError> {
        let requirement = plan.requirements();
        let mut storage = Vec::new();
        storage.try_reserve_exact(layer_count).map_err(|_| {
            RuntimeError::new(
                ErrorCode::PreflightNoWorkingSet,
                "native cache inventory allocation failed",
            )
        })?;
        for _ in 0..layer_count {
            storage.push(Self {
                keys: allocate_zeroed(requirement.key_storage_bytes)?,
                values: allocate_zeroed(requirement.value_storage_bytes)?,
            });
        }
        Ok(storage)
    }

    pub fn bind_all(
        storage: &mut [Self],
        plan: KvCachePlan,
    ) -> Result<Vec<KvCache<'_>>, RuntimeError> {
        let mut caches = Vec::new();
        caches.try_reserve_exact(storage.len()).map_err(|_| {
            RuntimeError::new(
                ErrorCode::PreflightNoWorkingSet,
                "native cache-handle inventory allocation failed",
            )
        })?;
        for item in storage {
            caches.push(KvCache::new(plan, &mut item.keys, &mut item.values)?);
        }
        Ok(caches)
    }

    pub fn heap_bytes(storage: &[Self]) -> Result<usize, RuntimeError> {
        storage.iter().try_fold(0usize, |total, item| {
            checked_total([total, slice_bytes(&item.keys)?, slice_bytes(&item.values)?])
        })
    }
}
