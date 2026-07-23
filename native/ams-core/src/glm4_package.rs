use std::collections::HashMap;

use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{
    AmsError, ErrorCode, GatedMlpPlan, GatedMlpReaders, Glm4DecoderPlan, Glm4DecoderReaders,
    Glm4DenseLayerNormLayout, Glm4DenseLayerPlan, Glm4DenseLayerReaders, Glm4MlaNormLayout,
    Glm4MlaPlan, Glm4MlaReaders, Glm4ModelPlan, Glm4ModelReaders, Glm4ModelVectorLayout,
    Glm4SparseLayerPlan, Glm4SparseLayerReaders, Glm4SparseLayerVectorLayout, GlmRouterPlan,
    IdentityDType, IdentityLinearPlan, Int4Config, Int4LinearPlan, KvCachePlan, LinearPlan,
    RangeReader, SparseMoeBindings, SparseMoePlan, TernaryConfig, TernaryLinearPlan,
};

/// Base-model tensor roles accepted by the native GLM-4 package planner.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Glm4BindingRole {
    /// Decoder token embedding table.
    Embedding,
    /// Final decoder `RMSNorm`.
    FinalNorm,
    /// Causal-LM output projection.
    LmHead,
    /// Per-layer input `RMSNorm`.
    InputNorm,
    /// Per-layer post-attention `RMSNorm`.
    PostAttentionNorm,
    /// MLA Q low-rank input projection.
    AttentionQaProjection,
    /// MLA Q low-rank `RMSNorm`.
    AttentionQaNorm,
    /// MLA expanded Q projection.
    AttentionQbProjection,
    /// MLA compressed KV plus rotary-key projection.
    AttentionKvAProjection,
    /// MLA KV low-rank `RMSNorm`.
    AttentionKvANorm,
    /// MLA expanded nonrotary-K and V projection.
    AttentionKvBProjection,
    /// Attention output projection.
    AttentionOutputProjection,
    /// Dense MLP gate projection.
    DenseGateProjection,
    /// Dense MLP up projection.
    DenseUpProjection,
    /// Dense MLP down projection.
    DenseDownProjection,
    /// Sparse-MoE router projection.
    RouterWeight,
    /// Sparse-MoE noaux correction bias.
    RouterCorrectionBias,
    /// Routed-expert gate projection.
    RoutedExpertGateProjection,
    /// Routed-expert up projection.
    RoutedExpertUpProjection,
    /// Routed-expert down projection.
    RoutedExpertDownProjection,
    /// Shared-expert gate projection.
    SharedExpertGateProjection,
    /// Shared-expert up projection.
    SharedExpertUpProjection,
    /// Shared-expert down projection.
    SharedExpertDownProjection,
}

/// Exact rank-one or row-major rank-two tensor geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm4BindingShape {
    /// A vector with the declared element count.
    Vector(usize),
    /// A row-major matrix with `(rows, columns)`.
    Matrix(usize, usize),
}

/// Native storage encoding admitted for one GLM-4 binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm4BindingEncoding {
    /// Direct floating-point storage.
    Identity(IdentityDType),
    /// Grouped trit5 ternary storage.
    Ternary(TernaryConfig),
    /// Grouped symmetric signed-INT4 storage.
    Int4(Int4Config),
}

/// One normalized package tensor range supplied to the native planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4TensorBindingSpec {
    role: Glm4BindingRole,
    layer_index: Option<usize>,
    expert_index: Option<usize>,
    shape: Glm4BindingShape,
    encoding: Glm4BindingEncoding,
    reader_index: usize,
    offset: u64,
    encoded_bytes: usize,
}

impl Glm4TensorBindingSpec {
    /// Construct one flat package binding without opening or reading its object.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        role: Glm4BindingRole,
        layer_index: Option<usize>,
        expert_index: Option<usize>,
        shape: Glm4BindingShape,
        encoding: Glm4BindingEncoding,
        reader_index: usize,
        offset: u64,
        encoded_bytes: usize,
    ) -> Self {
        Self {
            role,
            layer_index,
            expert_index,
            shape,
            encoding,
            reader_index,
            offset,
            encoded_bytes,
        }
    }
}

/// Core model dimensions that determine the GLM-4 base graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4ModelDimensions {
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    vocabulary_size: usize,
    hidden_layer_count: usize,
}

impl Glm4ModelDimensions {
    /// Construct the normalized base-model dimensions.
    #[must_use]
    pub const fn new(
        hidden_size: usize,
        intermediate_size: usize,
        moe_intermediate_size: usize,
        vocabulary_size: usize,
        hidden_layer_count: usize,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            moe_intermediate_size,
            vocabulary_size,
            hidden_layer_count,
        }
    }
}

/// MLA dimensions shared by every GLM-4 inference layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4AttentionDimensions {
    head_count: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dimension: usize,
    qk_rope_head_dimension: usize,
    value_head_dimension: usize,
}

impl Glm4AttentionDimensions {
    /// Construct the normalized MLA head and low-rank dimensions.
    #[must_use]
    pub const fn new(
        head_count: usize,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        nonrotary_head_dimension: usize,
        rotary_head_dimension: usize,
        value_head_dimension: usize,
    ) -> Self {
        Self {
            head_count,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dimension: nonrotary_head_dimension,
            qk_rope_head_dimension: rotary_head_dimension,
            value_head_dimension,
        }
    }
}

/// Routed-plus-shared expert dimensions and noaux policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm4ExpertPolicy {
    routed_expert_count: usize,
    shared_expert_count: usize,
    experts_per_token: usize,
    group_count: usize,
    top_groups: usize,
    routed_scaling_factor: f64,
}

impl Glm4ExpertPolicy {
    /// Construct the normalized GLM-4 sparse-MoE policy.
    #[must_use]
    pub const fn new(
        routed_expert_count: usize,
        shared_expert_count: usize,
        experts_per_token: usize,
        group_count: usize,
        top_groups: usize,
        routed_scaling_factor: f64,
    ) -> Self {
        Self {
            routed_expert_count,
            shared_expert_count,
            experts_per_token,
            group_count,
            top_groups,
            routed_scaling_factor,
        }
    }
}

/// Complete architecture semantics consumed by native GLM-4 plan construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm4ArchitecturePlanSpec {
    model: Glm4ModelDimensions,
    attention: Glm4AttentionDimensions,
    experts: Glm4ExpertPolicy,
    max_position_embeddings: usize,
    rms_norm_epsilon: f64,
    rope_theta: f64,
}

impl Glm4ArchitecturePlanSpec {
    /// Construct a complete normalized architecture spec.
    #[must_use]
    pub const fn new(
        model: Glm4ModelDimensions,
        attention: Glm4AttentionDimensions,
        experts: Glm4ExpertPolicy,
        max_position_embeddings: usize,
        rms_norm_epsilon: f64,
        rope_theta: f64,
    ) -> Self {
        Self {
            model,
            attention,
            experts,
            max_position_embeddings,
            rms_norm_epsilon,
            rope_theta,
        }
    }
}

/// Runtime resource and tokenizer policy bound into a native GLM-4 plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4RuntimePlanSpec {
    linear_arena_bytes: usize,
    context_capacity_tokens: usize,
    tokenizer_vocabulary_size: usize,
    cache_key_dtype: IdentityDType,
    cache_value_dtype: IdentityDType,
}

impl Glm4RuntimePlanSpec {
    /// Construct the explicit native runtime policy.
    #[must_use]
    pub const fn new(
        linear_arena_bytes: usize,
        context_capacity_tokens: usize,
        tokenizer_vocabulary_size: usize,
        cache_key_dtype: IdentityDType,
        cache_value_dtype: IdentityDType,
    ) -> Self {
        Self {
            linear_arena_bytes,
            context_capacity_tokens,
            tokenizer_vocabulary_size,
            cache_key_dtype,
            cache_value_dtype,
        }
    }
}

type BindingKey = (Glm4BindingRole, Option<usize>, Option<usize>);

#[derive(Clone, Copy)]
struct MlaBinding {
    q_a: Glm4TensorBindingSpec,
    q_norm: Glm4TensorBindingSpec,
    q_b: Glm4TensorBindingSpec,
    kv_a: Glm4TensorBindingSpec,
    kv_norm: Glm4TensorBindingSpec,
    kv_b: Glm4TensorBindingSpec,
}

#[derive(Clone, Copy)]
struct GatedBinding {
    gate: Glm4TensorBindingSpec,
    up: Glm4TensorBindingSpec,
    down: Glm4TensorBindingSpec,
}

#[derive(Clone, Copy)]
struct DenseBinding {
    input_norm: Glm4TensorBindingSpec,
    mla: MlaBinding,
    output_projection: Glm4TensorBindingSpec,
    post_attention_norm: Glm4TensorBindingSpec,
    mlp: GatedBinding,
}

#[derive(Clone)]
struct SparseBinding {
    input_norm: Glm4TensorBindingSpec,
    mla: MlaBinding,
    output_projection: Glm4TensorBindingSpec,
    post_attention_norm: Glm4TensorBindingSpec,
    correction_bias: Glm4TensorBindingSpec,
    router: Glm4TensorBindingSpec,
    experts: Vec<GatedBinding>,
    shared: GatedBinding,
}

#[derive(Clone)]
struct ModelBinding {
    embedding: Glm4TensorBindingSpec,
    dense: DenseBinding,
    sparse: Vec<SparseBinding>,
    final_norm: Glm4TensorBindingSpec,
    lm_head: Glm4TensorBindingSpec,
}

/// Complete native model plan plus the exact reader-index topology that produced it.
#[derive(Clone)]
pub struct Glm4BoundModelPlan {
    model: Glm4ModelPlan,
    reader_lengths: Vec<u64>,
    binding: ModelBinding,
    sparse_expert_plans: Vec<Vec<GatedMlpPlan>>,
}

impl Glm4BoundModelPlan {
    /// Independently validate a flat package inventory and assemble the complete native model plan.
    ///
    /// No reader is opened or read. `reader_lengths` must correspond exactly to the deduplicated
    /// object registry that a later [`Self::with_readers`] call supplies.
    ///
    /// # Errors
    ///
    /// Returns a typed plan, capability, or working-set error for an incomplete inventory,
    /// shape/codec/range drift, unsupported cache policy, or inconsistent architecture.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        architecture: Glm4ArchitecturePlanSpec,
        runtime: Glm4RuntimePlanSpec,
        reader_lengths: Vec<u64>,
        tensors: Vec<Glm4TensorBindingSpec>,
    ) -> Result<Self, AmsError> {
        validate_top_level(architecture, runtime, &reader_lengths)?;
        let mut inventory = HashMap::with_capacity(tensors.len());
        for tensor in tensors {
            let key = (tensor.role, tensor.layer_index, tensor.expert_index);
            if inventory.insert(key, tensor).is_some() {
                return Err(AmsError::new(
                    ErrorCode::PlanInvalid,
                    "native GLM-4 tensor binding is duplicated",
                ));
            }
        }

        let model = architecture.model;
        let attention = architecture.attention;
        let expert_policy = architecture.experts;
        let embedding = take(&mut inventory, Glm4BindingRole::Embedding, None, None)?;
        validate_identity_matrix(
            embedding,
            model.vocabulary_size,
            model.hidden_size,
            &reader_lengths,
        )?;
        let final_norm = take(&mut inventory, Glm4BindingRole::FinalNorm, None, None)?;
        validate_identity_vector(final_norm, model.hidden_size, &reader_lengths)?;
        let lm_head = take(&mut inventory, Glm4BindingRole::LmHead, None, None)?;
        let lm_head_plan = linear_plan(
            lm_head,
            model.vocabulary_size,
            model.hidden_size,
            runtime.linear_arena_bytes,
            &reader_lengths,
        )?;

        let cache = KvCachePlan::new(
            attention.head_count,
            add(
                attention.qk_nope_head_dimension,
                attention.qk_rope_head_dimension,
                "native GLM-4 cache QK head dimension overflow",
            )?,
            attention.value_head_dimension,
            runtime.context_capacity_tokens,
            runtime.cache_key_dtype,
            runtime.cache_value_dtype,
        )?;
        let (dense_plan, dense_binding) = build_dense(
            &mut inventory,
            architecture,
            runtime,
            cache,
            &reader_lengths,
        )?;
        let router = GlmRouterPlan::new(
            expert_policy.routed_expert_count,
            expert_policy.experts_per_token,
            expert_policy.group_count,
            expert_policy.top_groups,
            expert_policy.routed_scaling_factor,
        )?;
        let mut sparse_plans = Vec::with_capacity(model.hidden_layer_count - 1);
        let mut sparse_bindings = Vec::with_capacity(model.hidden_layer_count - 1);
        let mut sparse_expert_plans = Vec::with_capacity(model.hidden_layer_count - 1);
        for layer_index in 1..model.hidden_layer_count {
            let (plan, binding, expert_plans) = build_sparse(
                &mut inventory,
                architecture,
                runtime,
                cache,
                router,
                layer_index,
                &reader_lengths,
            )?;
            sparse_plans.push(plan);
            sparse_bindings.push(binding);
            sparse_expert_plans.push(expert_plans);
        }
        if !inventory.is_empty() {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "native GLM-4 package contains an unexpected base-model binding",
            ));
        }
        let decoder = Glm4DecoderPlan::new(dense_plan, sparse_plans)?;
        let vector_layout = Glm4ModelVectorLayout::new(
            embedding.offset,
            final_norm.offset,
            identity_dtype(embedding)?,
            identity_dtype(final_norm)?,
        );
        let model_plan = Glm4ModelPlan::new(
            decoder,
            lm_head_plan,
            vector_layout,
            runtime.tokenizer_vocabulary_size,
            architecture.rms_norm_epsilon,
        )?;
        Ok(Self {
            model: model_plan,
            reader_lengths,
            binding: ModelBinding {
                embedding,
                dense: dense_binding,
                sparse: sparse_bindings,
                final_norm,
                lm_head,
            },
            sparse_expert_plans,
        })
    }

    /// Return the complete executable native model plan.
    #[must_use]
    pub const fn model_plan(&self) -> &Glm4ModelPlan {
        &self.model
    }

    /// Return the exact number of immutable object readers required by this binding.
    #[must_use]
    pub fn reader_count(&self) -> usize {
        self.reader_lengths.len()
    }

    /// Assemble the complete borrow-scoped reader topology and invoke one operation.
    ///
    /// Reader construction performs no range reads. The callback cannot retain the topology after
    /// the object registry or plan borrow ends, avoiding self-referential ownership in wrappers.
    ///
    /// # Errors
    ///
    /// Returns `IO_FAILURE` when the registry count or lengths differ from the admitted plan, or
    /// propagates the callback's typed execution error.
    pub fn with_readers<T>(
        &self,
        readers: &[&dyn RangeReader],
        operation: impl FnOnce(&Glm4ModelReaders<'_, '_, '_>) -> Result<T, AmsError>,
    ) -> Result<T, AmsError> {
        if readers.len() != self.reader_lengths.len()
            || readers
                .iter()
                .zip(&self.reader_lengths)
                .any(|(reader, length)| reader.len() != *length)
        {
            return Err(AmsError::new(
                ErrorCode::IoFailure,
                "native GLM-4 reader registry differs from the admitted objects",
            ));
        }
        let dense = dense_readers(&self.binding.dense, readers);
        let expert_readers: Vec<Vec<GatedMlpReaders<'_>>> = self
            .binding
            .sparse
            .iter()
            .map(|layer| {
                layer
                    .experts
                    .iter()
                    .map(|expert| gated_readers(expert, readers))
                    .collect()
            })
            .collect();
        let shared_readers: Vec<GatedMlpReaders<'_>> = self
            .binding
            .sparse
            .iter()
            .map(|layer| gated_readers(&layer.shared, readers))
            .collect();
        let sparse: Vec<Glm4SparseLayerReaders<'_, '_>> = self
            .binding
            .sparse
            .iter()
            .zip(&self.sparse_expert_plans)
            .zip(&expert_readers)
            .zip(&shared_readers)
            .map(|(((layer, expert_plans), layer_expert_readers), shared)| {
                let moe = SparseMoeBindings::new(
                    reader(readers, layer.router),
                    expert_plans,
                    layer_expert_readers,
                    shared,
                );
                Glm4SparseLayerReaders::new(
                    reader(readers, layer.input_norm),
                    mla_readers(&layer.mla, readers),
                    reader(readers, layer.output_projection),
                    reader(readers, layer.post_attention_norm),
                    reader(readers, layer.correction_bias),
                    moe,
                )
            })
            .collect();
        let decoder = Glm4DecoderReaders::new(dense, &sparse);
        let model = Glm4ModelReaders::new(
            reader(readers, self.binding.embedding),
            decoder,
            reader(readers, self.binding.final_norm),
            reader(readers, self.binding.lm_head),
        );
        operation(&model)
    }
}

fn validate_top_level(
    architecture: Glm4ArchitecturePlanSpec,
    runtime: Glm4RuntimePlanSpec,
    reader_lengths: &[u64],
) -> Result<(), AmsError> {
    let model = architecture.model;
    let attention = architecture.attention;
    let experts = architecture.experts;
    if model.hidden_size == 0
        || model.intermediate_size == 0
        || model.moe_intermediate_size == 0
        || model.vocabulary_size == 0
        || model.hidden_layer_count < 2
        || attention.head_count == 0
        || attention.q_lora_rank == 0
        || attention.kv_lora_rank == 0
        || attention.qk_nope_head_dimension == 0
        || attention.qk_rope_head_dimension == 0
        || attention.qk_rope_head_dimension % 2 != 0
        || attention.value_head_dimension == 0
        || experts.shared_expert_count == 0
        || architecture.max_position_embeddings == 0
        || runtime.linear_arena_bytes == 0
        || runtime.context_capacity_tokens == 0
        || runtime.context_capacity_tokens > architecture.max_position_embeddings
        || runtime.tokenizer_vocabulary_size == 0
        || runtime.tokenizer_vocabulary_size > model.vocabulary_size
        || reader_lengths.is_empty()
        || reader_lengths.contains(&0)
        || !architecture.rms_norm_epsilon.is_finite()
        || architecture.rms_norm_epsilon <= 0.0
        || !architecture.rope_theta.is_finite()
        || architecture.rope_theta <= 0.0
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 architecture or runtime policy is invalid",
        ));
    }
    Ok(())
}

fn take(
    inventory: &mut HashMap<BindingKey, Glm4TensorBindingSpec>,
    role: Glm4BindingRole,
    layer_index: Option<usize>,
    expert_index: Option<usize>,
) -> Result<Glm4TensorBindingSpec, AmsError> {
    inventory
        .remove(&(role, layer_index, expert_index))
        .ok_or_else(|| {
            AmsError::new(
                ErrorCode::PlanInvalid,
                "native GLM-4 package is missing a required tensor binding",
            )
        })
}

const fn identity_dtype(binding: Glm4TensorBindingSpec) -> Result<IdentityDType, AmsError> {
    match binding.encoding {
        Glm4BindingEncoding::Identity(dtype) => Ok(dtype),
        Glm4BindingEncoding::Ternary(_) | Glm4BindingEncoding::Int4(_) => Err(AmsError::new(
            ErrorCode::CapabilityMismatch,
            "native GLM-4 vector or embedding requires identity storage",
        )),
    }
}

fn validate_range(
    binding: Glm4TensorBindingSpec,
    calculated_bytes: usize,
    reader_lengths: &[u64],
) -> Result<(), AmsError> {
    if binding.encoded_bytes != calculated_bytes {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 encoded byte count differs from the codec plan",
        ));
    }
    let reader_length = reader_lengths.get(binding.reader_index).ok_or_else(|| {
        AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 tensor references an absent object reader",
        )
    })?;
    let end = add_u64(
        binding.offset,
        usize_to_u64(
            calculated_bytes,
            "native GLM-4 encoded byte count exceeds u64",
        )?,
        "native GLM-4 tensor range overflow",
    )?;
    if end > *reader_length {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "native GLM-4 tensor range exceeds its declared object",
        ));
    }
    Ok(())
}

fn validate_identity_vector(
    binding: Glm4TensorBindingSpec,
    expected_length: usize,
    reader_lengths: &[u64],
) -> Result<(), AmsError> {
    if binding.shape != Glm4BindingShape::Vector(expected_length) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 vector shape differs from the architecture",
        ));
    }
    let dtype = identity_dtype(binding)?;
    validate_range(
        binding,
        mul(
            expected_length,
            dtype.item_bytes(),
            "native GLM-4 vector byte count overflow",
        )?,
        reader_lengths,
    )
}

fn validate_identity_matrix(
    binding: Glm4TensorBindingSpec,
    expected_rows: usize,
    expected_columns: usize,
    reader_lengths: &[u64],
) -> Result<(), AmsError> {
    if binding.shape != Glm4BindingShape::Matrix(expected_rows, expected_columns) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 identity matrix shape differs from the architecture",
        ));
    }
    let dtype = identity_dtype(binding)?;
    let elements = mul(
        expected_rows,
        expected_columns,
        "native GLM-4 identity matrix element count overflow",
    )?;
    validate_range(
        binding,
        mul(
            elements,
            dtype.item_bytes(),
            "native GLM-4 identity matrix byte count overflow",
        )?,
        reader_lengths,
    )
}

fn linear_plan(
    binding: Glm4TensorBindingSpec,
    expected_rows: usize,
    expected_columns: usize,
    arena_bytes: usize,
    reader_lengths: &[u64],
) -> Result<LinearPlan, AmsError> {
    if binding.shape != Glm4BindingShape::Matrix(expected_rows, expected_columns) {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 linear shape differs from the architecture",
        ));
    }
    let plan: LinearPlan = match binding.encoding {
        Glm4BindingEncoding::Identity(dtype) => IdentityLinearPlan::from_arena(
            expected_rows,
            expected_columns,
            binding.offset,
            arena_bytes,
            dtype,
        )?
        .into(),
        Glm4BindingEncoding::Ternary(config) => TernaryLinearPlan::from_arena(
            expected_rows,
            expected_columns,
            binding.offset,
            arena_bytes,
            config,
        )?
        .into(),
        Glm4BindingEncoding::Int4(config) => Int4LinearPlan::from_arena(
            expected_rows,
            expected_columns,
            binding.offset,
            arena_bytes,
            config,
        )?
        .into(),
    };
    let calculated_bytes = usize::try_from(plan.reader_end() - binding.offset).map_err(|_| {
        AmsError::new(
            ErrorCode::PlanInvalid,
            "native GLM-4 linear byte count exceeds usize",
        )
    })?;
    validate_range(binding, calculated_bytes, reader_lengths)?;
    Ok(plan)
}

#[allow(clippy::similar_names, clippy::too_many_lines)]
fn build_mla(
    inventory: &mut HashMap<BindingKey, Glm4TensorBindingSpec>,
    architecture: Glm4ArchitecturePlanSpec,
    runtime: Glm4RuntimePlanSpec,
    layer_index: usize,
    reader_lengths: &[u64],
) -> Result<(Glm4MlaPlan, MlaBinding), AmsError> {
    let model = architecture.model;
    let attention = architecture.attention;
    let qk_head_dimension = add(
        attention.qk_nope_head_dimension,
        attention.qk_rope_head_dimension,
        "native GLM-4 QK head dimension overflow",
    )?;
    let q_a = take(
        inventory,
        Glm4BindingRole::AttentionQaProjection,
        Some(layer_index),
        None,
    )?;
    let q_norm = take(
        inventory,
        Glm4BindingRole::AttentionQaNorm,
        Some(layer_index),
        None,
    )?;
    let q_b = take(
        inventory,
        Glm4BindingRole::AttentionQbProjection,
        Some(layer_index),
        None,
    )?;
    let kv_a = take(
        inventory,
        Glm4BindingRole::AttentionKvAProjection,
        Some(layer_index),
        None,
    )?;
    let kv_norm = take(
        inventory,
        Glm4BindingRole::AttentionKvANorm,
        Some(layer_index),
        None,
    )?;
    let kv_b = take(
        inventory,
        Glm4BindingRole::AttentionKvBProjection,
        Some(layer_index),
        None,
    )?;
    let q_low_rank_plan = linear_plan(
        q_a,
        attention.q_lora_rank,
        model.hidden_size,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    validate_identity_vector(q_norm, attention.q_lora_rank, reader_lengths)?;
    let q_expanded_plan = linear_plan(
        q_b,
        mul(
            attention.head_count,
            qk_head_dimension,
            "native GLM-4 Q output width overflow",
        )?,
        attention.q_lora_rank,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let kv_compressed_plan = linear_plan(
        kv_a,
        add(
            attention.kv_lora_rank,
            attention.qk_rope_head_dimension,
            "native GLM-4 compressed KV width overflow",
        )?,
        model.hidden_size,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    validate_identity_vector(kv_norm, attention.kv_lora_rank, reader_lengths)?;
    let kv_expanded_plan = linear_plan(
        kv_b,
        mul(
            attention.head_count,
            add(
                attention.qk_nope_head_dimension,
                attention.value_head_dimension,
                "native GLM-4 KV head width overflow",
            )?,
            "native GLM-4 KV output width overflow",
        )?,
        attention.kv_lora_rank,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let layout = Glm4MlaNormLayout::new(
        q_norm.offset,
        kv_norm.offset,
        identity_dtype(q_norm)?,
        identity_dtype(kv_norm)?,
    );
    let plan = Glm4MlaPlan::new(
        q_low_rank_plan,
        q_expanded_plan,
        kv_compressed_plan,
        kv_expanded_plan,
        layout,
        attention.head_count,
        attention.qk_nope_head_dimension,
        attention.qk_rope_head_dimension,
        attention.value_head_dimension,
        architecture.rms_norm_epsilon,
        architecture.rope_theta,
    )?;
    Ok((
        plan,
        MlaBinding {
            q_a,
            q_norm,
            q_b,
            kv_a,
            kv_norm,
            kv_b,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn build_gated(
    inventory: &mut HashMap<BindingKey, Glm4TensorBindingSpec>,
    roles: [Glm4BindingRole; 3],
    layer_index: usize,
    expert_index: Option<usize>,
    rows: usize,
    columns: usize,
    arena_bytes: usize,
    reader_lengths: &[u64],
) -> Result<(GatedMlpPlan, GatedBinding), AmsError> {
    let gate = take(inventory, roles[0], Some(layer_index), expert_index)?;
    let up = take(inventory, roles[1], Some(layer_index), expert_index)?;
    let down = take(inventory, roles[2], Some(layer_index), expert_index)?;
    let plan = GatedMlpPlan::new(
        linear_plan(gate, rows, columns, arena_bytes, reader_lengths)?,
        linear_plan(up, rows, columns, arena_bytes, reader_lengths)?,
        linear_plan(down, columns, rows, arena_bytes, reader_lengths)?,
    )?;
    Ok((plan, GatedBinding { gate, up, down }))
}

fn build_dense(
    inventory: &mut HashMap<BindingKey, Glm4TensorBindingSpec>,
    architecture: Glm4ArchitecturePlanSpec,
    runtime: Glm4RuntimePlanSpec,
    cache: KvCachePlan,
    reader_lengths: &[u64],
) -> Result<(Glm4DenseLayerPlan, DenseBinding), AmsError> {
    let layer_index = 0;
    let model = architecture.model;
    let input_norm = take(
        inventory,
        Glm4BindingRole::InputNorm,
        Some(layer_index),
        None,
    )?;
    validate_identity_vector(input_norm, model.hidden_size, reader_lengths)?;
    let (mla, attention_binding) = build_mla(
        inventory,
        architecture,
        runtime,
        layer_index,
        reader_lengths,
    )?;
    let output_projection = take(
        inventory,
        Glm4BindingRole::AttentionOutputProjection,
        Some(layer_index),
        None,
    )?;
    let output_plan = linear_plan(
        output_projection,
        model.hidden_size,
        mul(
            architecture.attention.head_count,
            architecture.attention.value_head_dimension,
            "native GLM-4 attention output width overflow",
        )?,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let post_attention_norm = take(
        inventory,
        Glm4BindingRole::PostAttentionNorm,
        Some(layer_index),
        None,
    )?;
    validate_identity_vector(post_attention_norm, model.hidden_size, reader_lengths)?;
    let (mlp, dense_mlp_binding) = build_gated(
        inventory,
        [
            Glm4BindingRole::DenseGateProjection,
            Glm4BindingRole::DenseUpProjection,
            Glm4BindingRole::DenseDownProjection,
        ],
        layer_index,
        None,
        model.intermediate_size,
        model.hidden_size,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let norm_layout = Glm4DenseLayerNormLayout::new(
        input_norm.offset,
        post_attention_norm.offset,
        identity_dtype(input_norm)?,
        identity_dtype(post_attention_norm)?,
    );
    let plan = Glm4DenseLayerPlan::new(
        mla,
        output_plan,
        mlp,
        cache,
        norm_layout,
        architecture.rms_norm_epsilon,
    )?;
    Ok((
        plan,
        DenseBinding {
            input_norm,
            mla: attention_binding,
            output_projection,
            post_attention_norm,
            mlp: dense_mlp_binding,
        },
    ))
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_sparse(
    inventory: &mut HashMap<BindingKey, Glm4TensorBindingSpec>,
    architecture: Glm4ArchitecturePlanSpec,
    runtime: Glm4RuntimePlanSpec,
    cache: KvCachePlan,
    router_policy: GlmRouterPlan,
    layer_index: usize,
    reader_lengths: &[u64],
) -> Result<(Glm4SparseLayerPlan, SparseBinding, Vec<GatedMlpPlan>), AmsError> {
    let model = architecture.model;
    let experts = architecture.experts;
    let input_norm = take(
        inventory,
        Glm4BindingRole::InputNorm,
        Some(layer_index),
        None,
    )?;
    validate_identity_vector(input_norm, model.hidden_size, reader_lengths)?;
    let (mla, mla_binding) = build_mla(
        inventory,
        architecture,
        runtime,
        layer_index,
        reader_lengths,
    )?;
    let output_projection = take(
        inventory,
        Glm4BindingRole::AttentionOutputProjection,
        Some(layer_index),
        None,
    )?;
    let output_plan = linear_plan(
        output_projection,
        model.hidden_size,
        mul(
            architecture.attention.head_count,
            architecture.attention.value_head_dimension,
            "native GLM-4 attention output width overflow",
        )?,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let post_attention_norm = take(
        inventory,
        Glm4BindingRole::PostAttentionNorm,
        Some(layer_index),
        None,
    )?;
    validate_identity_vector(post_attention_norm, model.hidden_size, reader_lengths)?;
    let router = take(
        inventory,
        Glm4BindingRole::RouterWeight,
        Some(layer_index),
        None,
    )?;
    let router_linear = linear_plan(
        router,
        experts.routed_expert_count,
        model.hidden_size,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let correction_bias = take(
        inventory,
        Glm4BindingRole::RouterCorrectionBias,
        Some(layer_index),
        None,
    )?;
    validate_identity_vector(correction_bias, experts.routed_expert_count, reader_lengths)?;
    let mut expert_plans = Vec::with_capacity(experts.routed_expert_count);
    let mut expert_bindings = Vec::with_capacity(experts.routed_expert_count);
    for expert_index in 0..experts.routed_expert_count {
        let (plan, binding) = build_gated(
            inventory,
            [
                Glm4BindingRole::RoutedExpertGateProjection,
                Glm4BindingRole::RoutedExpertUpProjection,
                Glm4BindingRole::RoutedExpertDownProjection,
            ],
            layer_index,
            Some(expert_index),
            model.moe_intermediate_size,
            model.hidden_size,
            runtime.linear_arena_bytes,
            reader_lengths,
        )?;
        expert_plans.push(plan);
        expert_bindings.push(binding);
    }
    let shared_intermediate = mul(
        model.moe_intermediate_size,
        experts.shared_expert_count,
        "native GLM-4 shared expert width overflow",
    )?;
    let (shared, shared_binding) = build_gated(
        inventory,
        [
            Glm4BindingRole::SharedExpertGateProjection,
            Glm4BindingRole::SharedExpertUpProjection,
            Glm4BindingRole::SharedExpertDownProjection,
        ],
        layer_index,
        None,
        shared_intermediate,
        model.hidden_size,
        runtime.linear_arena_bytes,
        reader_lengths,
    )?;
    let moe = SparseMoePlan::new(router_linear, router_policy, &expert_plans, shared)?;
    let vector_layout = Glm4SparseLayerVectorLayout::new(
        input_norm.offset,
        post_attention_norm.offset,
        correction_bias.offset,
        identity_dtype(input_norm)?,
        identity_dtype(post_attention_norm)?,
        identity_dtype(correction_bias)?,
    );
    let plan = Glm4SparseLayerPlan::new(
        mla,
        output_plan,
        moe,
        cache,
        vector_layout,
        architecture.rms_norm_epsilon,
    )?;
    Ok((
        plan,
        SparseBinding {
            input_norm,
            mla: mla_binding,
            output_projection,
            post_attention_norm,
            correction_bias,
            router,
            experts: expert_bindings,
            shared: shared_binding,
        },
        expert_plans,
    ))
}

fn reader<'a>(
    readers: &'a [&dyn RangeReader],
    binding: Glm4TensorBindingSpec,
) -> &'a dyn RangeReader {
    readers[binding.reader_index]
}

fn mla_readers<'a>(binding: &MlaBinding, readers: &'a [&dyn RangeReader]) -> Glm4MlaReaders<'a> {
    Glm4MlaReaders::new(
        reader(readers, binding.q_a),
        reader(readers, binding.q_norm),
        reader(readers, binding.q_b),
        reader(readers, binding.kv_a),
        reader(readers, binding.kv_norm),
        reader(readers, binding.kv_b),
    )
}

fn gated_readers<'a>(
    binding: &GatedBinding,
    readers: &'a [&dyn RangeReader],
) -> GatedMlpReaders<'a> {
    GatedMlpReaders::new(
        reader(readers, binding.gate),
        reader(readers, binding.up),
        reader(readers, binding.down),
    )
}

fn dense_readers<'a>(
    binding: &DenseBinding,
    readers: &'a [&dyn RangeReader],
) -> Glm4DenseLayerReaders<'a> {
    Glm4DenseLayerReaders::new(
        reader(readers, binding.input_norm),
        mla_readers(&binding.mla, readers),
        reader(readers, binding.output_projection),
        reader(readers, binding.post_attention_norm),
        gated_readers(&binding.mlp, readers),
    )
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct RejectingReader {
        length: u64,
        read_count: Cell<usize>,
    }

    impl RangeReader for RejectingReader {
        fn len(&self) -> u64 {
            self.length
        }

        fn read_exact_at(&self, _offset: u64, _destination: &mut [u8]) -> Result<(), AmsError> {
            self.read_count.set(self.read_count.get() + 1);
            Err(AmsError::new(
                ErrorCode::IoFailure,
                "test reader rejects payload access",
            ))
        }
    }

    struct Fixture {
        architecture: Glm4ArchitecturePlanSpec,
        runtime: Glm4RuntimePlanSpec,
        lengths: Vec<u64>,
        tensors: Vec<Glm4TensorBindingSpec>,
    }

    impl Fixture {
        #[allow(clippy::too_many_lines)]
        fn new() -> Result<Self, AmsError> {
            let architecture = Glm4ArchitecturePlanSpec::new(
                Glm4ModelDimensions::new(4, 6, 3, 8, 2),
                Glm4AttentionDimensions::new(1, 2, 2, 2, 2, 2),
                Glm4ExpertPolicy::new(2, 1, 1, 1, 1, 1.5),
                16,
                1e-5,
                10_000.0,
            );
            let runtime = Glm4RuntimePlanSpec::new(
                64,
                8,
                8,
                IdentityDType::BFloat16,
                IdentityDType::BFloat16,
            );
            let mut fixture = Self {
                architecture,
                runtime,
                lengths: Vec::new(),
                tensors: Vec::new(),
            };
            fixture.push_matrix(Glm4BindingRole::Embedding, None, None, 8, 4, identity())?;
            fixture.push_vector(Glm4BindingRole::FinalNorm, None, 4, identity())?;
            fixture.push_matrix(Glm4BindingRole::LmHead, None, None, 8, 4, identity())?;
            fixture.push_layer_common(0)?;
            fixture.push_matrix(
                Glm4BindingRole::DenseGateProjection,
                Some(0),
                None,
                6,
                4,
                identity(),
            )?;
            fixture.push_matrix(
                Glm4BindingRole::DenseUpProjection,
                Some(0),
                None,
                6,
                4,
                identity(),
            )?;
            fixture.push_matrix(
                Glm4BindingRole::DenseDownProjection,
                Some(0),
                None,
                4,
                6,
                identity(),
            )?;
            fixture.push_layer_common(1)?;
            fixture.push_matrix(
                Glm4BindingRole::RouterWeight,
                Some(1),
                None,
                2,
                4,
                identity(),
            )?;
            fixture.push_vector(
                Glm4BindingRole::RouterCorrectionBias,
                Some(1),
                2,
                identity(),
            )?;
            for expert_index in 0..2 {
                let encoding = if expert_index == 0 {
                    Glm4BindingEncoding::Ternary(TernaryConfig::new(5)?)
                } else {
                    identity()
                };
                fixture.push_matrix(
                    Glm4BindingRole::RoutedExpertGateProjection,
                    Some(1),
                    Some(expert_index),
                    3,
                    4,
                    encoding,
                )?;
                fixture.push_matrix(
                    Glm4BindingRole::RoutedExpertUpProjection,
                    Some(1),
                    Some(expert_index),
                    3,
                    4,
                    encoding,
                )?;
                fixture.push_matrix(
                    Glm4BindingRole::RoutedExpertDownProjection,
                    Some(1),
                    Some(expert_index),
                    4,
                    3,
                    encoding,
                )?;
            }
            for role in [
                Glm4BindingRole::SharedExpertGateProjection,
                Glm4BindingRole::SharedExpertUpProjection,
            ] {
                fixture.push_matrix(role, Some(1), None, 3, 4, identity())?;
            }
            fixture.push_matrix(
                Glm4BindingRole::SharedExpertDownProjection,
                Some(1),
                None,
                4,
                3,
                identity(),
            )?;
            Ok(fixture)
        }

        fn push_layer_common(&mut self, layer: usize) -> Result<(), AmsError> {
            self.push_vector(Glm4BindingRole::InputNorm, Some(layer), 4, identity())?;
            self.push_matrix(
                Glm4BindingRole::AttentionQaProjection,
                Some(layer),
                None,
                2,
                4,
                identity(),
            )?;
            self.push_vector(Glm4BindingRole::AttentionQaNorm, Some(layer), 2, identity())?;
            self.push_matrix(
                Glm4BindingRole::AttentionQbProjection,
                Some(layer),
                None,
                4,
                2,
                if layer == 0 {
                    Glm4BindingEncoding::Int4(Int4Config::new(5)?)
                } else {
                    identity()
                },
            )?;
            self.push_matrix(
                Glm4BindingRole::AttentionKvAProjection,
                Some(layer),
                None,
                4,
                4,
                identity(),
            )?;
            self.push_vector(
                Glm4BindingRole::AttentionKvANorm,
                Some(layer),
                2,
                identity(),
            )?;
            self.push_matrix(
                Glm4BindingRole::AttentionKvBProjection,
                Some(layer),
                None,
                4,
                2,
                identity(),
            )?;
            self.push_matrix(
                Glm4BindingRole::AttentionOutputProjection,
                Some(layer),
                None,
                4,
                2,
                identity(),
            )?;
            self.push_vector(
                Glm4BindingRole::PostAttentionNorm,
                Some(layer),
                4,
                identity(),
            )
        }

        fn push_vector(
            &mut self,
            role: Glm4BindingRole,
            layer: Option<usize>,
            length: usize,
            encoding: Glm4BindingEncoding,
        ) -> Result<(), AmsError> {
            self.push(
                role,
                layer,
                None,
                Glm4BindingShape::Vector(length),
                encoding,
            )
        }

        #[allow(clippy::too_many_arguments)]
        fn push_matrix(
            &mut self,
            role: Glm4BindingRole,
            layer: Option<usize>,
            expert: Option<usize>,
            rows: usize,
            columns: usize,
            encoding: Glm4BindingEncoding,
        ) -> Result<(), AmsError> {
            self.push(
                role,
                layer,
                expert,
                Glm4BindingShape::Matrix(rows, columns),
                encoding,
            )
        }

        fn push(
            &mut self,
            role: Glm4BindingRole,
            layer: Option<usize>,
            expert: Option<usize>,
            shape: Glm4BindingShape,
            encoding: Glm4BindingEncoding,
        ) -> Result<(), AmsError> {
            let elements = match shape {
                Glm4BindingShape::Vector(length) => length,
                Glm4BindingShape::Matrix(rows, columns) => mul(rows, columns, "test shape")?,
            };
            let encoded_bytes = match encoding {
                Glm4BindingEncoding::Identity(dtype) => {
                    mul(elements, dtype.item_bytes(), "test identity bytes")?
                }
                Glm4BindingEncoding::Ternary(config) => config.encoded_size(elements)?,
                Glm4BindingEncoding::Int4(config) => config.encoded_size(elements)?,
            };
            let reader_index = self.lengths.len();
            self.lengths.push(
                u64::try_from(encoded_bytes)
                    .map_err(|_| AmsError::new(ErrorCode::PlanInvalid, "test length"))?,
            );
            self.tensors.push(Glm4TensorBindingSpec::new(
                role,
                layer,
                expert,
                shape,
                encoding,
                reader_index,
                0,
                encoded_bytes,
            ));
            Ok(())
        }
    }

    const fn identity() -> Glm4BindingEncoding {
        Glm4BindingEncoding::Identity(IdentityDType::Float32)
    }

    #[test]
    fn mixed_package_builds_complete_model_and_borrow_scoped_readers() -> Result<(), AmsError> {
        let fixture = Fixture::new()?;
        let bound = Glm4BoundModelPlan::new(
            fixture.architecture,
            fixture.runtime,
            fixture.lengths.clone(),
            fixture.tensors,
        )?;
        assert_eq!(bound.model_plan().decoder().layer_count(), 2);
        assert_eq!(bound.model_plan().context_capacity_tokens(), 8);
        assert_eq!(bound.model_plan().tokenizer_vocabulary_size(), 8);
        assert_eq!(bound.reader_count(), fixture.lengths.len());
        let registry: Vec<RejectingReader> = fixture
            .lengths
            .iter()
            .map(|length| RejectingReader {
                length: *length,
                read_count: Cell::new(0),
            })
            .collect();
        let readers: Vec<&dyn RangeReader> = registry
            .iter()
            .map(|reader| reader as &dyn RangeReader)
            .collect();
        let observed = bound.with_readers(&readers, |_| Ok(17))?;
        assert_eq!(observed, 17);
        assert!(registry.iter().all(|reader| reader.read_count.get() == 0));
        Ok(())
    }

    #[test]
    fn package_builder_rejects_missing_duplicate_and_transposed_bindings() -> Result<(), AmsError> {
        let mut missing = Fixture::new()?;
        missing.tensors.pop();
        let error = Glm4BoundModelPlan::new(
            missing.architecture,
            missing.runtime,
            missing.lengths,
            missing.tensors,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));

        let mut duplicate = Fixture::new()?;
        duplicate.tensors.push(duplicate.tensors[0]);
        let error = Glm4BoundModelPlan::new(
            duplicate.architecture,
            duplicate.runtime,
            duplicate.lengths,
            duplicate.tensors,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));

        let mut transposed = Fixture::new()?;
        let target = transposed
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == Glm4BindingRole::AttentionQaProjection)
            .ok_or_else(|| AmsError::new(ErrorCode::InternalInvariant, "test target missing"))?;
        target.shape = Glm4BindingShape::Matrix(4, 2);
        let error = Glm4BoundModelPlan::new(
            transposed.architecture,
            transposed.runtime,
            transposed.lengths,
            transposed.tensors,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));
        Ok(())
    }

    #[test]
    fn package_builder_rejects_vector_codec_range_and_registry_drift() -> Result<(), AmsError> {
        let mut low_bit_vector = Fixture::new()?;
        let target = low_bit_vector
            .tensors
            .iter_mut()
            .find(|tensor| tensor.role == Glm4BindingRole::FinalNorm)
            .ok_or_else(|| AmsError::new(ErrorCode::InternalInvariant, "test target missing"))?;
        target.encoding = Glm4BindingEncoding::Int4(Int4Config::new(4)?);
        let error = Glm4BoundModelPlan::new(
            low_bit_vector.architecture,
            low_bit_vector.runtime,
            low_bit_vector.lengths,
            low_bit_vector.tensors,
        )
        .err();
        assert_eq!(
            error.map(AmsError::code),
            Some(ErrorCode::CapabilityMismatch)
        );

        let mut bad_range = Fixture::new()?;
        bad_range.tensors[0].encoded_bytes -= 1;
        let error = Glm4BoundModelPlan::new(
            bad_range.architecture,
            bad_range.runtime,
            bad_range.lengths,
            bad_range.tensors,
        )
        .err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::PlanInvalid));

        let fixture = Fixture::new()?;
        let bound = Glm4BoundModelPlan::new(
            fixture.architecture,
            fixture.runtime,
            fixture.lengths.clone(),
            fixture.tensors,
        )?;
        let short_registry: Vec<RejectingReader> = fixture
            .lengths
            .iter()
            .map(|length| RejectingReader {
                length: *length,
                read_count: Cell::new(0),
            })
            .collect();
        short_registry[0].length.checked_sub(1).ok_or_else(|| {
            AmsError::new(
                ErrorCode::InternalInvariant,
                "test object unexpectedly empty",
            )
        })?;
        let wrong_reader = RejectingReader {
            length: fixture.lengths[0] - 1,
            read_count: Cell::new(0),
        };
        let mut readers: Vec<&dyn RangeReader> = short_registry
            .iter()
            .map(|reader| reader as &dyn RangeReader)
            .collect();
        readers[0] = &wrong_reader;
        let error = bound.with_readers(&readers, |_| Ok(())).err();
        assert_eq!(error.map(AmsError::code), Some(ErrorCode::IoFailure));
        assert!(
            short_registry
                .iter()
                .all(|reader| reader.read_count.get() == 0)
        );
        Ok(())
    }
}
