use crate::checked::{add, add_u64, mul, usize_to_u64};
use crate::{
    AmsError, ErrorCode, Glm4DecoderPlan, Glm4DecoderReaders, Glm4DenseLayerScratch,
    Glm4SparseLayerScratch, IdentityDType, KvCache, LinearPlan, LinearScratch,
    LinearScratchRequirements, RangeReader, glm_rms_norm, glm4_decoder_token, read_identity_vector,
    stream_linear,
};

/// Identity-vector layout for the embedding table and final decoder normalization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4ModelVectorLayout {
    embedding_offset: u64,
    final_norm_offset: u64,
    embedding_dtype: IdentityDType,
    final_norm_dtype: IdentityDType,
}

impl Glm4ModelVectorLayout {
    /// Bind the embedding matrix and final normalization vector.
    #[must_use]
    pub const fn new(
        embedding_offset: u64,
        final_norm_offset: u64,
        embedding_dtype: IdentityDType,
        final_norm_dtype: IdentityDType,
    ) -> Self {
        Self {
            embedding_offset,
            final_norm_offset,
            embedding_dtype,
            final_norm_dtype,
        }
    }
}

/// Caller-owned non-layer working set for one model token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm4ModelScratchRequirements {
    /// Reusable bytes for one embedding row or the final norm vector.
    pub vector_encoded_bytes: usize,
    /// Nested mixed-storage LM-head linear scratch.
    pub lm_head: LinearScratchRequirements,
    /// Embedding/decoder/final-normalization hidden-width buffers.
    pub hidden_elements: usize,
    /// Full vocabulary logits retained for deterministic selection.
    pub logit_elements: usize,
    /// Exact non-layer scratch bytes; layer requirements remain exposed by the decoder plan.
    pub local_bytes: usize,
}

/// Immutable one-token GLM-4 causal-LM plan around the transactional decoder stack.
#[derive(Clone, Debug, PartialEq)]
pub struct Glm4ModelPlan {
    decoder: Glm4DecoderPlan,
    lm_head: LinearPlan,
    layout: Glm4ModelVectorLayout,
    model_vocabulary_size: usize,
    tokenizer_vocabulary_size: usize,
    embedding_end: u64,
    final_norm_end: u64,
    rms_norm_epsilon: f64,
    scratch: Glm4ModelScratchRequirements,
}

impl Glm4ModelPlan {
    /// Validate embedding, decoder, final norm, and LM-head geometry.
    ///
    /// # Errors
    ///
    /// Returns `PLAN_INVALID` for inconsistent graph dimensions, ranges, or numeric policy.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        decoder: Glm4DecoderPlan,
        lm_head: LinearPlan,
        layout: Glm4ModelVectorLayout,
        tokenizer_vocabulary_size: usize,
        rms_norm_epsilon: f64,
    ) -> Result<Self, AmsError> {
        let hidden_elements = decoder.hidden_elements();
        let model_vocabulary_size = lm_head.rows();
        if model_vocabulary_size == 0
            || tokenizer_vocabulary_size == 0
            || tokenizer_vocabulary_size > model_vocabulary_size
            || lm_head.columns() != hidden_elements
            || !rms_norm_epsilon.is_finite()
            || rms_norm_epsilon <= 0.0
        {
            return Err(AmsError::new(
                ErrorCode::PlanInvalid,
                "GLM-4 model dimensions or RMSNorm policy are invalid",
            ));
        }
        let embedding_row_bytes = mul(
            hidden_elements,
            layout.embedding_dtype.item_bytes(),
            "GLM-4 embedding row bytes overflow",
        )?;
        let embedding_bytes = mul(
            model_vocabulary_size,
            embedding_row_bytes,
            "GLM-4 embedding bytes overflow",
        )?;
        let final_norm_bytes = mul(
            hidden_elements,
            layout.final_norm_dtype.item_bytes(),
            "GLM-4 final norm bytes overflow",
        )?;
        let embedding_end = add_u64(
            layout.embedding_offset,
            usize_to_u64(embedding_bytes, "GLM-4 embedding bytes exceed u64")?,
            "GLM-4 embedding range overflow",
        )?;
        let final_norm_end = add_u64(
            layout.final_norm_offset,
            usize_to_u64(final_norm_bytes, "GLM-4 final norm bytes exceed u64")?,
            "GLM-4 final norm range overflow",
        )?;
        let lm_head_scratch = lm_head.scratch();
        let hidden_float_elements = mul(
            hidden_elements,
            6,
            "GLM-4 model hidden scratch elements overflow",
        )?;
        let float_elements = add(
            hidden_float_elements,
            model_vocabulary_size,
            "GLM-4 model float scratch elements overflow",
        )?;
        let vector_encoded_bytes = embedding_row_bytes.max(final_norm_bytes);
        let local_bytes = add(
            add(
                vector_encoded_bytes,
                lm_head_scratch.total_bytes,
                "GLM-4 model encoded and linear scratch overflow",
            )?,
            mul(
                float_elements,
                size_of::<f64>(),
                "GLM-4 model FP64 scratch overflow",
            )?,
            "GLM-4 model local scratch overflow",
        )?;
        Ok(Self {
            decoder,
            lm_head,
            layout,
            model_vocabulary_size,
            tokenizer_vocabulary_size,
            embedding_end,
            final_norm_end,
            rms_norm_epsilon,
            scratch: Glm4ModelScratchRequirements {
                vector_encoded_bytes,
                lm_head: lm_head_scratch,
                hidden_elements,
                logit_elements: model_vocabulary_size,
                local_bytes,
            },
        })
    }

    /// Exact non-layer working set for one token.
    #[must_use]
    pub const fn scratch(&self) -> Glm4ModelScratchRequirements {
        self.scratch
    }

    /// Nested decoder plan, including per-layer reusable scratch requirements.
    #[must_use]
    pub const fn decoder(&self) -> &Glm4DecoderPlan {
        &self.decoder
    }

    /// Full model vocabulary size and LM-head row count.
    #[must_use]
    pub const fn model_vocabulary_size(&self) -> usize {
        self.model_vocabulary_size
    }

    /// Tokenizer-mapped prefix eligible for input and output token selection.
    #[must_use]
    pub const fn tokenizer_vocabulary_size(&self) -> usize {
        self.tokenizer_vocabulary_size
    }

    /// Maximum token prefix admitted by every decoder-layer cache.
    #[must_use]
    pub const fn context_capacity_tokens(&self) -> usize {
        self.decoder.cache_capacity_tokens()
    }
}

/// Weight readers for embedding, decoder stack, final norm, and LM head.
pub struct Glm4ModelReaders<'reader, 'slice, 'layers> {
    embedding: &'reader dyn RangeReader,
    decoder: Glm4DecoderReaders<'reader, 'slice, 'layers>,
    final_norm: &'reader dyn RangeReader,
    lm_head: &'reader dyn RangeReader,
}

impl<'reader, 'slice, 'layers> Glm4ModelReaders<'reader, 'slice, 'layers> {
    /// Bind the complete causal-LM weight inventory without reading it.
    #[must_use]
    pub const fn new(
        embedding: &'reader dyn RangeReader,
        decoder: Glm4DecoderReaders<'reader, 'slice, 'layers>,
        final_norm: &'reader dyn RangeReader,
        lm_head: &'reader dyn RangeReader,
    ) -> Self {
        Self {
            embedding,
            decoder,
            final_norm,
            lm_head,
        }
    }
}

/// Caller-owned layer and model buffers for allocation-free one-token execution.
pub struct Glm4ModelScratch<'a> {
    dense: Glm4DenseLayerScratch<'a>,
    sparse: Glm4SparseLayerScratch<'a>,
    vector_encoded: &'a mut [u8],
    lm_head: LinearScratch<'a>,
    input_hidden: &'a mut [f64],
    hidden_a: &'a mut [f64],
    hidden_b: &'a mut [f64],
    decoder_output: &'a mut [f64],
    norm_weights: &'a mut [f64],
    normalized: &'a mut [f64],
    logits: &'a mut [f64],
}

impl<'a> Glm4ModelScratch<'a> {
    /// Group every admitted layer and model scratch region.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        dense: Glm4DenseLayerScratch<'a>,
        sparse: Glm4SparseLayerScratch<'a>,
        vector_encoded: &'a mut [u8],
        lm_head: LinearScratch<'a>,
        input_hidden: &'a mut [f64],
        hidden_a: &'a mut [f64],
        hidden_b: &'a mut [f64],
        decoder_output: &'a mut [f64],
        norm_weights: &'a mut [f64],
        normalized: &'a mut [f64],
        logits: &'a mut [f64],
    ) -> Self {
        Self {
            dense,
            sparse,
            vector_encoded,
            lm_head,
            input_hidden,
            hidden_a,
            hidden_b,
            decoder_output,
            norm_weights,
            normalized,
            logits,
        }
    }

    const fn admits(&self, plan: &Glm4ModelPlan) -> bool {
        let requirement = plan.scratch;
        self.vector_encoded.len() >= requirement.vector_encoded_bytes
            && self.lm_head.admits(requirement.lm_head)
            && self.input_hidden.len() >= requirement.hidden_elements
            && self.hidden_a.len() >= requirement.hidden_elements
            && self.hidden_b.len() >= requirement.hidden_elements
            && self.decoder_output.len() >= requirement.hidden_elements
            && self.norm_weights.len() >= requirement.hidden_elements
            && self.normalized.len() >= requirement.hidden_elements
            && self.logits.len() >= requirement.logit_elements
    }
}

fn select_argmax(logits: &[f64]) -> Result<usize, AmsError> {
    let mut best: Option<(usize, f64)> = None;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(AmsError::new(
                ErrorCode::NumericFailure,
                "GLM-4 model logit is non-finite",
            ));
        }
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index).ok_or_else(|| {
        AmsError::new(
            ErrorCode::InternalInvariant,
            "GLM-4 model produced no logits",
        )
    })
}

fn preflight_model_token(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &[KvCache<'_>],
    position: usize,
    input_token: usize,
    scratch: &Glm4ModelScratch<'_>,
) -> Result<(), AmsError> {
    if input_token >= plan.tokenizer_vocabulary_size {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 model input token is outside the tokenizer vocabulary",
        ));
    }
    if !scratch.admits(plan) {
        return Err(AmsError::new(
            ErrorCode::PreflightNoWorkingSet,
            "GLM-4 model scratch is smaller than the admitted plan",
        ));
    }
    if plan.embedding_end > readers.embedding.len()
        || plan.final_norm_end > readers.final_norm.len()
        || plan.lm_head.reader_end() > readers.lm_head.len()
    {
        return Err(AmsError::new(
            ErrorCode::IoFailure,
            "GLM-4 model weight range exceeds its storage object",
        ));
    }
    plan.decoder.preflight(
        &readers.decoder,
        caches,
        position,
        &scratch.dense,
        &scratch.sparse,
    )
}

fn execute_model_decoder_token(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    position: usize,
    input_token: usize,
    scratch: &mut Glm4ModelScratch<'_>,
) -> Result<(), AmsError> {
    let hidden = plan.scratch.hidden_elements;
    let row_bytes = mul(
        hidden,
        plan.layout.embedding_dtype.item_bytes(),
        "GLM-4 embedding row bytes overflow during execution",
    )?;
    let row_offset = add_u64(
        plan.layout.embedding_offset,
        usize_to_u64(
            mul(
                input_token,
                row_bytes,
                "GLM-4 embedding row offset overflow",
            )?,
            "GLM-4 embedding row offset exceeds u64",
        )?,
        "GLM-4 embedding absolute offset overflow",
    )?;
    let input_hidden = &mut scratch.input_hidden[..hidden];
    read_identity_vector(
        readers.embedding,
        row_offset,
        plan.layout.embedding_dtype,
        input_hidden,
        scratch.vector_encoded,
    )?;
    glm4_decoder_token(
        &plan.decoder,
        &readers.decoder,
        caches,
        position,
        input_hidden,
        &mut scratch.dense,
        &mut scratch.sparse,
        &mut scratch.hidden_a[..hidden],
        &mut scratch.hidden_b[..hidden],
        &mut scratch.decoder_output[..hidden],
    )
}

/// Consume one prompt token into every decoder cache without evaluating the LM head.
///
/// Complete embedding, decoder, final-normalization, and LM-head bindings are still preflighted
/// before the embedding read. This makes a later next-token selection admissible without paying the
/// full vocabulary projection for prompt positions whose logits are discarded.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
pub fn glm4_model_cache_token(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    position: usize,
    input_token: usize,
    scratch: &mut Glm4ModelScratch<'_>,
) -> Result<(), AmsError> {
    preflight_model_token(plan, readers, caches, position, input_token, scratch)?;
    execute_model_decoder_token(plan, readers, caches, position, input_token, scratch)
}

fn execute_model_tail(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    scratch: &mut Glm4ModelScratch<'_>,
) -> Result<usize, AmsError> {
    let hidden = plan.scratch.hidden_elements;
    let norm_weights = &mut scratch.norm_weights[..hidden];
    read_identity_vector(
        readers.final_norm,
        plan.layout.final_norm_offset,
        plan.layout.final_norm_dtype,
        norm_weights,
        scratch.vector_encoded,
    )?;
    let normalized = &mut scratch.normalized[..hidden];
    glm_rms_norm(
        &scratch.decoder_output[..hidden],
        norm_weights,
        plan.rms_norm_epsilon,
        normalized,
    )?;
    let logits = &mut scratch.logits[..plan.model_vocabulary_size];
    stream_linear(
        readers.lm_head,
        plan.lm_head,
        normalized,
        None,
        &mut scratch.lm_head,
        logits,
    )?;
    select_argmax(&logits[..plan.tokenizer_vocabulary_size])
}

fn execute_model_token_with_tail(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    position: usize,
    input_token: usize,
    scratch: &mut Glm4ModelScratch<'_>,
) -> Result<usize, AmsError> {
    preflight_model_token(plan, readers, caches, position, input_token, scratch)?;
    execute_model_decoder_token(plan, readers, caches, position, input_token, scratch)?;
    let tail = execute_model_tail(plan, readers, scratch);
    match tail {
        Ok(token) => Ok(token),
        Err(error) => {
            plan.decoder.rollback(caches, position)?;
            Err(error)
        }
    }
}

/// Execute one autoregressive token and return the deterministic lowest-index argmax token.
///
/// The complete model is preflighted before the embedding read. Decoder caches remain committed only
/// if final normalization, LM-head execution, and token selection also succeed.
///
/// # Errors
///
/// Returns a typed plan, capacity, storage, codec, or numeric error.
pub fn glm4_model_next_token(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    position: usize,
    input_token: usize,
    scratch: &mut Glm4ModelScratch<'_>,
) -> Result<usize, AmsError> {
    execute_model_token_with_tail(plan, readers, caches, position, input_token, scratch)
}

/// Execute one autoregressive token and copy its decoder hidden state and complete logits.
///
/// This is the diagnostic observation boundary for differential qualification. It shares the exact
/// preflight, cache commit/rollback, final normalization, and LM-head path used by next-token
/// selection. Caller-owned outputs remain untouched unless the complete transition succeeds.
///
/// # Errors
///
/// Returns a typed plan, capacity, output-shape, storage, codec, or numeric error.
pub fn glm4_model_observe_token(
    plan: &Glm4ModelPlan,
    readers: &Glm4ModelReaders<'_, '_, '_>,
    caches: &mut [KvCache<'_>],
    position: usize,
    input_token: usize,
    scratch: &mut Glm4ModelScratch<'_>,
    outputs: (&mut [f64], &mut [f64]),
) -> Result<usize, AmsError> {
    let (hidden_output, logits_output) = outputs;
    if hidden_output.len() != plan.scratch.hidden_elements
        || logits_output.len() != plan.model_vocabulary_size
    {
        return Err(AmsError::new(
            ErrorCode::PlanInvalid,
            "GLM-4 observation output dimensions differ from the model",
        ));
    }
    let token =
        execute_model_token_with_tail(plan, readers, caches, position, input_token, scratch)?;
    hidden_output.copy_from_slice(&scratch.decoder_output[..plan.scratch.hidden_elements]);
    logits_output.copy_from_slice(&scratch.logits[..plan.model_vocabulary_size]);
    Ok(token)
}
