//! Strict package-to-native admission for the AMS GLM-4 bring-up runtime.

use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use ams_core::{
    AmsError, ErrorCode, FileRangeReader, Glm4ArchitecturePlanSpec, Glm4AttentionDimensions,
    Glm4BindingEncoding, Glm4BindingRole, Glm4BindingShape, Glm4BoundModelPlan, Glm4ExpertPolicy,
    Glm4ModelDimensions, Glm4ModelReaders, Glm4RuntimePlanSpec, Glm4TensorBindingSpec,
    IdentityDType, Int4Config, RangeReader, TernaryConfig,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_ENVELOPE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_VERIFICATION_BUFFER_BYTES: usize = 64 * 1024 * 1024;
const BINDING_SCHEMA: &str = "ams.native.glm4-binding.v1";
const ENVELOPE_SCHEMA: &str = "ams.native.glm4-envelope.v1";

/// Dynamic error at the serialized package and native-runtime boundary.
#[derive(Debug)]
pub struct RuntimeError {
    code: ErrorCode,
    message: String,
}

impl RuntimeError {
    fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Stable AMS error code.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        self.code
    }

    /// Redaction-safe failure description.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for RuntimeError {}

impl From<AmsError> for RuntimeError {
    fn from(error: AmsError) -> Self {
        Self::new(error.code(), error.context())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingEnvelope {
    schema_id: String,
    binding_hash: String,
    binding_identity_json: String,
    storage_paths: Vec<StoragePath>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StoragePath {
    object_id: String,
    absolute_path: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingIdentity {
    schema_id: String,
    package_id: String,
    manifest_content_root: String,
    architecture: ArchitectureIdentity,
    storage_objects: Vec<StorageIdentity>,
    tensors: Vec<TensorIdentity>,
    linear_arena_bytes: usize,
    context_capacity_tokens: usize,
    cache_key_dtype: String,
    cache_value_dtype: String,
    cache_storage_bytes_per_layer: usize,
    cache_storage_bytes_total: usize,
    cache_staging_bytes_per_layer: usize,
    tokenizer_vocabulary_size: usize,
    eos_token_ids: Vec<usize>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureIdentity {
    content_hash: String,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_nextn_predict_layers: usize,
    first_k_dense_replace: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    num_experts_per_tok: usize,
    n_group: usize,
    topk_group: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    routed_scaling_factor: f64,
    mlp_layer_types: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StorageIdentity {
    object_id: String,
    size_bytes: u64,
    alignment_bytes: usize,
    content_hash: String,
    kind: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorIdentity {
    tensor_name: String,
    role: String,
    layer_index: Option<usize>,
    expert_index: Option<usize>,
    mtp: bool,
    shape: Vec<usize>,
    logical_dtype: String,
    encoding: String,
    storage_object_id: String,
    offset: u64,
    encoded_bytes: usize,
    decoded_bytes: usize,
    codec_group_size: Option<usize>,
    codec_config_hash: Option<String>,
}

#[derive(Clone, Copy)]
struct ExpectedTensor {
    role: &'static str,
    layer_index: Option<usize>,
    expert_index: Option<usize>,
    mtp: bool,
}

/// Auditable result of native descriptor validation and complete object hashing.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Glm4AdmissionEvidence {
    /// Evidence schema.
    pub schema_id: &'static str,
    /// Exact relocation-stable Python binding identity.
    pub binding_hash: String,
    /// Admitted AMS package identifier.
    pub package_id: String,
    /// Manifest content root bound into the descriptor.
    pub manifest_content_root: String,
    /// Complete immutable storage objects verified through retained handles.
    pub verified_object_count: usize,
    /// Total object bytes hashed during admission.
    pub verified_object_bytes: u64,
    /// Maximum temporary hash buffer.
    pub verification_buffer_bytes: usize,
    /// Complete GLM-4 tensor inventory, including MTP.
    pub tensor_count: usize,
    /// Base-model tensors supplied to the executable plan.
    pub executable_tensor_count: usize,
    /// Separately admitted, currently non-executable MTP tensors.
    pub mtp_tensor_count: usize,
    /// Exact fixed K/V storage admitted for every inference layer.
    pub cache_storage_bytes_total: usize,
    /// Model-local logical scratch, excluding reusable layer scratch.
    pub model_local_scratch_bytes: usize,
    /// Dense-layer logical scratch.
    pub dense_layer_scratch_bytes: usize,
    /// Maximum reusable sparse-layer logical scratch.
    pub sparse_layer_scratch_bytes: usize,
}

/// Fully admitted native GLM-4 binding retaining the exact verified file handles.
pub struct AdmittedGlm4Binding {
    plan: Glm4BoundModelPlan,
    readers: Vec<FileRangeReader>,
    evidence: Glm4AdmissionEvidence,
}

impl AdmittedGlm4Binding {
    /// Admission evidence produced before this binding became observable.
    #[must_use]
    pub const fn evidence(&self) -> &Glm4AdmissionEvidence {
        &self.evidence
    }

    /// Execute an operation against the plan and exact retained reader topology.
    ///
    /// # Errors
    ///
    /// Propagates native plan, storage, numeric, or execution failures.
    pub fn with_model<T>(
        &self,
        operation: impl FnOnce(
            &ams_core::Glm4ModelPlan,
            &Glm4ModelReaders<'_, '_, '_>,
        ) -> Result<T, AmsError>,
    ) -> Result<T, RuntimeError> {
        let readers: Vec<&dyn RangeReader> = self
            .readers
            .iter()
            .map(|reader| reader as &dyn RangeReader)
            .collect();
        self.plan
            .with_readers(&readers, |model_readers| {
                operation(self.plan.model_plan(), model_readers)
            })
            .map_err(Into::into)
    }
}

/// Parse, independently validate, and fully hash one serialized GLM-4 binding.
///
/// The returned value is constructed only after every object hash succeeds. The same retained file
/// handles are used by all later range reads.
///
/// # Errors
///
/// Returns a typed package, integrity, capability, plan, resource, or I/O failure.
pub fn admit_glm4_binding_file(
    path: impl AsRef<Path>,
    verification_buffer_bytes: usize,
) -> Result<AdmittedGlm4Binding, RuntimeError> {
    if verification_buffer_bytes == 0 || verification_buffer_bytes > MAX_VERIFICATION_BUFFER_BYTES {
        return Err(RuntimeError::new(
            ErrorCode::PlanInvalid,
            "verification buffer is outside 1..=64 MiB",
        ));
    }
    let path = path.as_ref();
    let metadata = path.symlink_metadata().map_err(|_| {
        RuntimeError::new(
            ErrorCode::IoFailure,
            "binding envelope metadata read failed",
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            "binding envelope is not a nonsymlink regular file",
        ));
    }
    if metadata.len() == 0 || metadata.len() > MAX_ENVELOPE_BYTES {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            "binding envelope size is outside the admitted bound",
        ));
    }
    let payload = fs::read(path)
        .map_err(|_| RuntimeError::new(ErrorCode::IoFailure, "binding envelope read failed"))?;
    admit_glm4_binding_bytes(&payload, verification_buffer_bytes)
}

/// Parse, independently validate, and fully hash an in-memory binding envelope.
///
/// # Errors
///
/// Returns a typed package, integrity, capability, plan, resource, or I/O failure.
#[allow(clippy::too_many_lines)] // Admission is one fail-closed transaction before publication.
pub fn admit_glm4_binding_bytes(
    payload: &[u8],
    verification_buffer_bytes: usize,
) -> Result<AdmittedGlm4Binding, RuntimeError> {
    if payload.is_empty() || u64::try_from(payload.len()).unwrap_or(u64::MAX) > MAX_ENVELOPE_BYTES {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            "binding envelope size is outside the admitted bound",
        ));
    }
    if verification_buffer_bytes == 0 || verification_buffer_bytes > MAX_VERIFICATION_BUFFER_BYTES {
        return Err(RuntimeError::new(
            ErrorCode::PlanInvalid,
            "verification buffer is outside 1..=64 MiB",
        ));
    }
    let envelope: BindingEnvelope = serde_json::from_slice(payload).map_err(|_| {
        RuntimeError::new(
            ErrorCode::InvalidPackage,
            "binding envelope JSON is malformed or contains unreviewed fields",
        )
    })?;
    if envelope.schema_id != ENVELOPE_SCHEMA {
        return Err(RuntimeError::new(
            ErrorCode::CapabilityMismatch,
            "binding envelope schema is unsupported",
        ));
    }
    validate_sha256(&envelope.binding_hash, "binding hash")?;
    let observed_binding_hash = sha256_digest(envelope.binding_identity_json.as_bytes());
    if observed_binding_hash != envelope.binding_hash {
        return Err(RuntimeError::new(
            ErrorCode::IntegrityFailure,
            "binding identity bytes do not match the envelope hash",
        ));
    }
    let identity: BindingIdentity =
        serde_json::from_str(&envelope.binding_identity_json).map_err(|_| {
            RuntimeError::new(
                ErrorCode::InvalidPackage,
                "binding identity JSON is malformed or contains unreviewed fields",
            )
        })?;
    let normalized = normalize_identity(identity, &envelope.storage_paths)?;
    let plan = Glm4BoundModelPlan::new(
        normalized.architecture,
        normalized.runtime,
        normalized.reader_lengths,
        normalized.tensors,
    )?;
    let mut readers = Vec::with_capacity(normalized.objects.len());
    let mut verified_object_bytes = 0u64;
    let mut buffer = vec![0u8; verification_buffer_bytes];
    for object in &normalized.objects {
        let reader = FileRangeReader::open(&object.path)?;
        if reader.len() != object.size_bytes {
            return Err(RuntimeError::new(
                ErrorCode::IntegrityFailure,
                "storage object length differs from the binding identity",
            ));
        }
        verify_reader_sha256(&reader, &object.content_hash, &mut buffer)?;
        verified_object_bytes = verified_object_bytes
            .checked_add(object.size_bytes)
            .ok_or_else(|| {
                RuntimeError::new(
                    ErrorCode::InvalidPackage,
                    "verified object byte count overflowed",
                )
            })?;
        readers.push(reader);
    }
    let model_scratch = plan.model_plan().scratch();
    let decoder = plan.model_plan().decoder();
    let dense_layer_scratch_bytes = decoder.dense_scratch().total_bytes;
    let sparse_layer_scratch_bytes = decoder
        .sparse_scratch()
        .iter()
        .map(|requirement| requirement.total_bytes)
        .max()
        .unwrap_or(0);
    let evidence = Glm4AdmissionEvidence {
        schema_id: "ams.native.glm4-admission.v1",
        binding_hash: envelope.binding_hash,
        package_id: normalized.package_id,
        manifest_content_root: normalized.manifest_content_root,
        verified_object_count: readers.len(),
        verified_object_bytes,
        verification_buffer_bytes: verification_buffer_bytes.min(
            usize::try_from(
                normalized
                    .objects
                    .iter()
                    .map(|object| object.size_bytes)
                    .max()
                    .unwrap_or(0),
            )
            .unwrap_or(usize::MAX),
        ),
        tensor_count: normalized.tensor_count,
        executable_tensor_count: normalized.executable_tensor_count,
        mtp_tensor_count: normalized.mtp_tensor_count,
        cache_storage_bytes_total: normalized.cache_storage_bytes_total,
        model_local_scratch_bytes: model_scratch.local_bytes,
        dense_layer_scratch_bytes,
        sparse_layer_scratch_bytes,
    };
    Ok(AdmittedGlm4Binding {
        plan,
        readers,
        evidence,
    })
}

struct ObjectBinding {
    path: PathBuf,
    size_bytes: u64,
    content_hash: String,
}

struct NormalizedBinding {
    package_id: String,
    manifest_content_root: String,
    architecture: Glm4ArchitecturePlanSpec,
    runtime: Glm4RuntimePlanSpec,
    objects: Vec<ObjectBinding>,
    reader_lengths: Vec<u64>,
    tensors: Vec<Glm4TensorBindingSpec>,
    tensor_count: usize,
    executable_tensor_count: usize,
    mtp_tensor_count: usize,
    cache_storage_bytes_total: usize,
}

#[allow(clippy::too_many_lines)]
fn normalize_identity(
    identity: BindingIdentity,
    paths: &[StoragePath],
) -> Result<NormalizedBinding, RuntimeError> {
    if identity.schema_id != BINDING_SCHEMA {
        return Err(RuntimeError::new(
            ErrorCode::CapabilityMismatch,
            "binding identity schema is unsupported",
        ));
    }
    validate_identifier(&identity.package_id, "package ID")?;
    validate_sha256(&identity.manifest_content_root, "manifest content root")?;
    validate_architecture(&identity.architecture)?;
    validate_token_policy(&identity)?;
    let expected = expected_inventory(&identity.architecture)?;
    if identity.tensors.len() != expected.len()
        || !identity
            .tensors
            .windows(2)
            .all(|window| window[0].tensor_name < window[1].tensor_name)
    {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            "binding tensor inventory is not exact and sorted",
        ));
    }
    for tensor in &identity.tensors {
        let expected_tensor = expected.get(&tensor.tensor_name).ok_or_else(|| {
            RuntimeError::new(
                ErrorCode::CapabilityMismatch,
                "binding contains an unexpected GLM-4 tensor",
            )
        })?;
        if tensor.role != expected_tensor.role
            || tensor.layer_index != expected_tensor.layer_index
            || tensor.expert_index != expected_tensor.expert_index
            || tensor.mtp != expected_tensor.mtp
        {
            return Err(RuntimeError::new(
                ErrorCode::CapabilityMismatch,
                "binding tensor semantics differ from the GLM-4 inventory",
            ));
        }
    }

    if identity.storage_objects.len() != paths.len()
        || identity.storage_objects.is_empty()
        || !identity
            .storage_objects
            .windows(2)
            .all(|window| window[0].object_id < window[1].object_id)
    {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            "binding storage inventory is not exact and sorted",
        ));
    }
    let mut objects = Vec::with_capacity(identity.storage_objects.len());
    let mut reader_lengths = Vec::with_capacity(identity.storage_objects.len());
    let mut storage_indices = HashMap::with_capacity(identity.storage_objects.len());
    for (index, (storage, path)) in identity.storage_objects.iter().zip(paths).enumerate() {
        validate_identifier(&storage.object_id, "storage object ID")?;
        validate_sha256(&storage.content_hash, "storage object hash")?;
        if storage.object_id != path.object_id
            || storage.kind != "tensor_data"
            || storage.size_bytes == 0
            || storage.alignment_bytes == 0
            || !path.absolute_path.is_absolute()
        {
            return Err(RuntimeError::new(
                ErrorCode::InvalidPackage,
                "binding storage identity and local path map disagree",
            ));
        }
        storage_indices.insert(storage.object_id.as_str(), index);
        reader_lengths.push(storage.size_bytes);
        objects.push(ObjectBinding {
            path: path.absolute_path.clone(),
            size_bytes: storage.size_bytes,
            content_hash: storage.content_hash.clone(),
        });
    }

    let mut tensors = Vec::new();
    let mut mtp_tensor_count = 0usize;
    for tensor in &identity.tensors {
        let expected_shape = expected_shape(&identity.architecture, tensor)?;
        if tensor.shape != expected_shape {
            return Err(RuntimeError::new(
                ErrorCode::CapabilityMismatch,
                "binding tensor shape differs from the GLM-4 architecture",
            ));
        }
        let dtype = parse_identity_dtype(&tensor.logical_dtype)?;
        let elements = checked_product(&tensor.shape, "tensor element count")?;
        let decoded_bytes = elements.checked_mul(dtype.item_bytes()).ok_or_else(|| {
            RuntimeError::new(ErrorCode::InvalidPackage, "decoded tensor bytes overflowed")
        })?;
        if decoded_bytes != tensor.decoded_bytes {
            return Err(RuntimeError::new(
                ErrorCode::InvalidPackage,
                "binding decoded tensor bytes differ from shape and dtype",
            ));
        }
        let encoding = parse_encoding(tensor, dtype, elements)?;
        let storage_index = *storage_indices
            .get(tensor.storage_object_id.as_str())
            .ok_or_else(|| {
                RuntimeError::new(
                    ErrorCode::InvalidPackage,
                    "tensor references an absent storage object",
                )
            })?;
        let end = tensor
            .offset
            .checked_add(u64::try_from(tensor.encoded_bytes).map_err(|_| {
                RuntimeError::new(ErrorCode::InvalidPackage, "tensor byte count exceeds u64")
            })?)
            .ok_or_else(|| {
                RuntimeError::new(ErrorCode::InvalidPackage, "tensor range overflowed")
            })?;
        if end > reader_lengths[storage_index] {
            return Err(RuntimeError::new(
                ErrorCode::InvalidPackage,
                "tensor range exceeds its storage object",
            ));
        }
        if tensor.mtp {
            mtp_tensor_count = mtp_tensor_count.checked_add(1).ok_or_else(|| {
                RuntimeError::new(ErrorCode::InvalidPackage, "MTP tensor count overflowed")
            })?;
            continue;
        }
        tensors.push(Glm4TensorBindingSpec::new(
            parse_base_role(&tensor.role)?,
            tensor.layer_index,
            tensor.expert_index,
            match tensor.shape.as_slice() {
                [length] => Glm4BindingShape::Vector(*length),
                [rows, columns] => Glm4BindingShape::Matrix(*rows, *columns),
                _ => {
                    return Err(RuntimeError::new(
                        ErrorCode::InvalidPackage,
                        "native GLM-4 tensor rank is unsupported",
                    ));
                }
            },
            encoding,
            storage_index,
            tensor.offset,
            tensor.encoded_bytes,
        ));
    }
    let cache_key_dtype = parse_cache_dtype(&identity.cache_key_dtype)?;
    let cache_value_dtype = parse_cache_dtype(&identity.cache_value_dtype)?;
    let expected_cache = derive_cache_bytes(
        &identity.architecture,
        identity.context_capacity_tokens,
        cache_key_dtype,
        cache_value_dtype,
    )?;
    if expected_cache.0 != identity.cache_staging_bytes_per_layer
        || expected_cache.1 != identity.cache_storage_bytes_per_layer
        || expected_cache.2 != identity.cache_storage_bytes_total
    {
        return Err(RuntimeError::new(
            ErrorCode::PlanInvalid,
            "binding cache byte declarations differ from the architecture",
        ));
    }
    let architecture = Glm4ArchitecturePlanSpec::new(
        Glm4ModelDimensions::new(
            identity.architecture.hidden_size,
            identity.architecture.intermediate_size,
            identity.architecture.moe_intermediate_size,
            identity.architecture.vocab_size,
            identity.architecture.num_hidden_layers,
        ),
        Glm4AttentionDimensions::new(
            identity.architecture.num_attention_heads,
            identity.architecture.q_lora_rank,
            identity.architecture.kv_lora_rank,
            identity.architecture.qk_nope_head_dim,
            identity.architecture.qk_rope_head_dim,
            identity.architecture.v_head_dim,
        ),
        Glm4ExpertPolicy::new(
            identity.architecture.n_routed_experts,
            identity.architecture.n_shared_experts,
            identity.architecture.num_experts_per_tok,
            identity.architecture.n_group,
            identity.architecture.topk_group,
            identity.architecture.routed_scaling_factor,
        ),
        identity.architecture.max_position_embeddings,
        identity.architecture.rms_norm_eps,
        identity.architecture.rope_theta,
    );
    let runtime = Glm4RuntimePlanSpec::new(
        identity.linear_arena_bytes,
        identity.context_capacity_tokens,
        identity.tokenizer_vocabulary_size,
        cache_key_dtype,
        cache_value_dtype,
    );
    let tensor_count = identity.tensors.len();
    let executable_tensor_count = tensors.len();
    Ok(NormalizedBinding {
        package_id: identity.package_id,
        manifest_content_root: identity.manifest_content_root,
        architecture,
        runtime,
        objects,
        reader_lengths,
        tensors,
        tensor_count,
        executable_tensor_count,
        mtp_tensor_count,
        cache_storage_bytes_total: identity.cache_storage_bytes_total,
    })
}

fn validate_architecture(architecture: &ArchitectureIdentity) -> Result<(), RuntimeError> {
    validate_sha256(&architecture.content_hash, "architecture hash")?;
    let expected_qk = architecture
        .qk_nope_head_dim
        .checked_add(architecture.qk_rope_head_dim)
        .ok_or_else(|| RuntimeError::new(ErrorCode::InvalidPackage, "QK width overflowed"))?;
    let schedule_matches = architecture.mlp_layer_types.len() == architecture.num_hidden_layers
        && architecture
            .mlp_layer_types
            .iter()
            .enumerate()
            .all(|(index, layer_type)| layer_type == if index == 0 { "dense" } else { "sparse" });
    if architecture.hidden_size == 0
        || architecture.intermediate_size == 0
        || architecture.moe_intermediate_size == 0
        || architecture.vocab_size == 0
        || architecture.num_hidden_layers < 2
        || architecture.num_nextn_predict_layers != 1
        || architecture.first_k_dense_replace != 1
        || architecture.n_routed_experts == 0
        || architecture.n_shared_experts == 0
        || architecture.num_experts_per_tok == 0
        || architecture.n_group == 0
        || architecture.n_routed_experts % architecture.n_group != 0
        || architecture.topk_group == 0
        || architecture.topk_group > architecture.n_group
        || architecture.num_attention_heads == 0
        || architecture.num_attention_heads != architecture.num_key_value_heads
        || architecture.q_lora_rank == 0
        || architecture.kv_lora_rank == 0
        || architecture.qk_nope_head_dim == 0
        || architecture.qk_rope_head_dim == 0
        || architecture.qk_rope_head_dim % 2 != 0
        || architecture.qk_head_dim != expected_qk
        || architecture.v_head_dim == 0
        || architecture.max_position_embeddings == 0
        || !architecture.rms_norm_eps.is_finite()
        || architecture.rms_norm_eps <= 0.0
        || !architecture.rope_theta.is_finite()
        || architecture.rope_theta <= 0.0
        || !architecture.routed_scaling_factor.is_finite()
        || architecture.routed_scaling_factor <= 0.0
        || !schedule_matches
    {
        return Err(RuntimeError::new(
            ErrorCode::CapabilityMismatch,
            "binding architecture is outside reviewed GLM-4 semantics",
        ));
    }
    Ok(())
}

fn validate_token_policy(identity: &BindingIdentity) -> Result<(), RuntimeError> {
    if identity.linear_arena_bytes == 0
        || identity.context_capacity_tokens == 0
        || identity.context_capacity_tokens > identity.architecture.max_position_embeddings
        || identity.tokenizer_vocabulary_size == 0
        || identity.tokenizer_vocabulary_size > identity.architecture.vocab_size
        || identity.eos_token_ids.is_empty()
        || !identity
            .eos_token_ids
            .windows(2)
            .all(|window| window[0] < window[1])
        || identity
            .eos_token_ids
            .iter()
            .any(|token| *token >= identity.tokenizer_vocabulary_size)
    {
        return Err(RuntimeError::new(
            ErrorCode::PlanInvalid,
            "binding runtime or token policy is invalid",
        ));
    }
    Ok(())
}

fn expected_inventory(
    architecture: &ArchitectureIdentity,
) -> Result<HashMap<String, ExpectedTensor>, RuntimeError> {
    let mut expected = HashMap::new();
    insert_expected(
        &mut expected,
        "model.embed_tokens.weight".to_owned(),
        ExpectedTensor {
            role: "embedding",
            layer_index: None,
            expert_index: None,
            mtp: false,
        },
    )?;
    insert_expected(
        &mut expected,
        "model.norm.weight".to_owned(),
        ExpectedTensor {
            role: "final_norm",
            layer_index: None,
            expert_index: None,
            mtp: false,
        },
    )?;
    insert_expected(
        &mut expected,
        "lm_head.weight".to_owned(),
        ExpectedTensor {
            role: "lm_head",
            layer_index: None,
            expert_index: None,
            mtp: false,
        },
    )?;
    for layer in 0..architecture.num_hidden_layers {
        add_layer_inventory(&mut expected, architecture, layer, layer == 0, false)?;
    }
    for offset in 0..architecture.num_nextn_predict_layers {
        add_layer_inventory(
            &mut expected,
            architecture,
            architecture.num_hidden_layers + offset,
            false,
            true,
        )?;
    }
    Ok(expected)
}

fn add_layer_inventory(
    expected: &mut HashMap<String, ExpectedTensor>,
    architecture: &ArchitectureIdentity,
    layer: usize,
    dense: bool,
    mtp: bool,
) -> Result<(), RuntimeError> {
    let common = [
        ("input_layernorm.weight", "input_norm"),
        ("post_attention_layernorm.weight", "post_attention_norm"),
        ("self_attn.q_a_proj.weight", "attention_q_a_projection"),
        ("self_attn.q_a_layernorm.weight", "attention_q_a_norm"),
        ("self_attn.q_b_proj.weight", "attention_q_b_projection"),
        (
            "self_attn.kv_a_proj_with_mqa.weight",
            "attention_kv_a_projection",
        ),
        ("self_attn.kv_a_layernorm.weight", "attention_kv_a_norm"),
        ("self_attn.kv_b_proj.weight", "attention_kv_b_projection"),
        ("self_attn.o_proj.weight", "attention_output_projection"),
    ];
    for (suffix, role) in common {
        insert_layer_expected(expected, layer, suffix, role, None, mtp)?;
    }
    if dense {
        for (suffix, role) in [
            ("mlp.gate_proj.weight", "dense_gate_projection"),
            ("mlp.up_proj.weight", "dense_up_projection"),
            ("mlp.down_proj.weight", "dense_down_projection"),
        ] {
            insert_layer_expected(expected, layer, suffix, role, None, mtp)?;
        }
    } else {
        insert_layer_expected(
            expected,
            layer,
            "mlp.gate.weight",
            "router_weight",
            None,
            mtp,
        )?;
        insert_layer_expected(
            expected,
            layer,
            "mlp.gate.e_score_correction_bias",
            "router_correction_bias",
            None,
            mtp,
        )?;
        for expert in 0..architecture.n_routed_experts {
            for (projection, role) in [
                ("gate_proj", "routed_expert_gate_projection"),
                ("up_proj", "routed_expert_up_projection"),
                ("down_proj", "routed_expert_down_projection"),
            ] {
                insert_layer_expected(
                    expected,
                    layer,
                    &format!("mlp.experts.{expert}.{projection}.weight"),
                    role,
                    Some(expert),
                    mtp,
                )?;
            }
        }
        for (projection, role) in [
            ("gate_proj", "shared_expert_gate_projection"),
            ("up_proj", "shared_expert_up_projection"),
            ("down_proj", "shared_expert_down_projection"),
        ] {
            insert_layer_expected(
                expected,
                layer,
                &format!("mlp.shared_experts.{projection}.weight"),
                role,
                None,
                mtp,
            )?;
        }
    }
    if mtp {
        for (suffix, role) in [
            ("enorm.weight", "mtp_embed_norm"),
            ("hnorm.weight", "mtp_hidden_norm"),
            ("eh_proj.weight", "mtp_embed_hidden_projection"),
            ("embed_tokens.weight", "mtp_embedding"),
            ("shared_head.head.weight", "mtp_shared_head"),
            ("shared_head.norm.weight", "mtp_shared_head_norm"),
        ] {
            insert_layer_expected(expected, layer, suffix, role, None, true)?;
        }
    }
    Ok(())
}

fn insert_layer_expected(
    expected: &mut HashMap<String, ExpectedTensor>,
    layer: usize,
    suffix: &str,
    role: &'static str,
    expert_index: Option<usize>,
    mtp: bool,
) -> Result<(), RuntimeError> {
    insert_expected(
        expected,
        format!("model.layers.{layer}.{suffix}"),
        ExpectedTensor {
            role,
            layer_index: Some(layer),
            expert_index,
            mtp,
        },
    )
}

fn insert_expected(
    expected: &mut HashMap<String, ExpectedTensor>,
    name: String,
    tensor: ExpectedTensor,
) -> Result<(), RuntimeError> {
    if expected.insert(name, tensor).is_some() {
        return Err(RuntimeError::new(
            ErrorCode::InternalInvariant,
            "native expected tensor inventory contains a duplicate",
        ));
    }
    Ok(())
}

fn expected_shape(
    architecture: &ArchitectureIdentity,
    tensor: &TensorIdentity,
) -> Result<Vec<usize>, RuntimeError> {
    let hidden = architecture.hidden_size;
    let shared = architecture
        .moe_intermediate_size
        .checked_mul(architecture.n_shared_experts)
        .ok_or_else(|| {
            RuntimeError::new(ErrorCode::InvalidPackage, "shared expert width overflowed")
        })?;
    let shape = match tensor.role.as_str() {
        "embedding" | "lm_head" | "mtp_embedding" | "mtp_shared_head" => {
            vec![architecture.vocab_size, hidden]
        }
        "final_norm"
        | "input_norm"
        | "post_attention_norm"
        | "mtp_embed_norm"
        | "mtp_hidden_norm"
        | "mtp_shared_head_norm" => vec![hidden],
        "attention_q_a_projection" => vec![architecture.q_lora_rank, hidden],
        "attention_q_a_norm" => vec![architecture.q_lora_rank],
        "attention_q_b_projection" => vec![
            checked_mul(
                architecture.num_attention_heads,
                architecture.qk_head_dim,
                "Q projection rows",
            )?,
            architecture.q_lora_rank,
        ],
        "attention_kv_a_projection" => vec![
            architecture
                .kv_lora_rank
                .checked_add(architecture.qk_rope_head_dim)
                .ok_or_else(|| {
                    RuntimeError::new(ErrorCode::InvalidPackage, "KV A rows overflowed")
                })?,
            hidden,
        ],
        "attention_kv_a_norm" => vec![architecture.kv_lora_rank],
        "attention_kv_b_projection" => vec![
            checked_mul(
                architecture.num_attention_heads,
                architecture
                    .qk_nope_head_dim
                    .checked_add(architecture.v_head_dim)
                    .ok_or_else(|| {
                        RuntimeError::new(ErrorCode::InvalidPackage, "KV head width overflowed")
                    })?,
                "KV B rows",
            )?,
            architecture.kv_lora_rank,
        ],
        "attention_output_projection" => vec![
            hidden,
            checked_mul(
                architecture.num_attention_heads,
                architecture.v_head_dim,
                "attention output columns",
            )?,
        ],
        "dense_gate_projection" | "dense_up_projection" => {
            vec![architecture.intermediate_size, hidden]
        }
        "dense_down_projection" => vec![hidden, architecture.intermediate_size],
        "router_weight" => vec![architecture.n_routed_experts, hidden],
        "router_correction_bias" => vec![architecture.n_routed_experts],
        "routed_expert_gate_projection" | "routed_expert_up_projection" => {
            vec![architecture.moe_intermediate_size, hidden]
        }
        "routed_expert_down_projection" => {
            vec![hidden, architecture.moe_intermediate_size]
        }
        "shared_expert_gate_projection" | "shared_expert_up_projection" => {
            vec![shared, hidden]
        }
        "shared_expert_down_projection" => vec![hidden, shared],
        "mtp_embed_hidden_projection" => vec![
            hidden,
            hidden.checked_mul(2).ok_or_else(|| {
                RuntimeError::new(ErrorCode::InvalidPackage, "MTP hidden width overflowed")
            })?,
        ],
        _ => {
            return Err(RuntimeError::new(
                ErrorCode::CapabilityMismatch,
                "binding tensor role is unsupported",
            ));
        }
    };
    Ok(shape)
}

fn parse_base_role(role: &str) -> Result<Glm4BindingRole, RuntimeError> {
    let role = match role {
        "embedding" => Glm4BindingRole::Embedding,
        "final_norm" => Glm4BindingRole::FinalNorm,
        "lm_head" => Glm4BindingRole::LmHead,
        "input_norm" => Glm4BindingRole::InputNorm,
        "post_attention_norm" => Glm4BindingRole::PostAttentionNorm,
        "attention_q_a_projection" => Glm4BindingRole::AttentionQaProjection,
        "attention_q_a_norm" => Glm4BindingRole::AttentionQaNorm,
        "attention_q_b_projection" => Glm4BindingRole::AttentionQbProjection,
        "attention_kv_a_projection" => Glm4BindingRole::AttentionKvAProjection,
        "attention_kv_a_norm" => Glm4BindingRole::AttentionKvANorm,
        "attention_kv_b_projection" => Glm4BindingRole::AttentionKvBProjection,
        "attention_output_projection" => Glm4BindingRole::AttentionOutputProjection,
        "dense_gate_projection" => Glm4BindingRole::DenseGateProjection,
        "dense_up_projection" => Glm4BindingRole::DenseUpProjection,
        "dense_down_projection" => Glm4BindingRole::DenseDownProjection,
        "router_weight" => Glm4BindingRole::RouterWeight,
        "router_correction_bias" => Glm4BindingRole::RouterCorrectionBias,
        "routed_expert_gate_projection" => Glm4BindingRole::RoutedExpertGateProjection,
        "routed_expert_up_projection" => Glm4BindingRole::RoutedExpertUpProjection,
        "routed_expert_down_projection" => Glm4BindingRole::RoutedExpertDownProjection,
        "shared_expert_gate_projection" => Glm4BindingRole::SharedExpertGateProjection,
        "shared_expert_up_projection" => Glm4BindingRole::SharedExpertUpProjection,
        "shared_expert_down_projection" => Glm4BindingRole::SharedExpertDownProjection,
        _ => {
            return Err(RuntimeError::new(
                ErrorCode::CapabilityMismatch,
                "base-model tensor role is unsupported",
            ));
        }
    };
    Ok(role)
}

fn parse_identity_dtype(dtype: &str) -> Result<IdentityDType, RuntimeError> {
    match dtype {
        "float16" => Ok(IdentityDType::Float16),
        "bfloat16" => Ok(IdentityDType::BFloat16),
        "float32" => Ok(IdentityDType::Float32),
        _ => Err(RuntimeError::new(
            ErrorCode::CapabilityMismatch,
            "tensor logical dtype is unsupported",
        )),
    }
}

fn parse_cache_dtype(dtype: &str) -> Result<IdentityDType, RuntimeError> {
    match dtype {
        "bfloat16" => Ok(IdentityDType::BFloat16),
        "float32" => Ok(IdentityDType::Float32),
        _ => Err(RuntimeError::new(
            ErrorCode::CapabilityMismatch,
            "cache dtype is unsupported",
        )),
    }
}

fn parse_encoding(
    tensor: &TensorIdentity,
    dtype: IdentityDType,
    elements: usize,
) -> Result<Glm4BindingEncoding, RuntimeError> {
    match tensor.encoding.as_str() {
        "identity" => {
            if tensor.codec_group_size.is_some()
                || tensor.codec_config_hash.is_some()
                || tensor.encoded_bytes != tensor.decoded_bytes
            {
                return Err(RuntimeError::new(
                    ErrorCode::InvalidPackage,
                    "identity tensor carries codec fields or a transformed byte count",
                ));
            }
            Ok(Glm4BindingEncoding::Identity(dtype))
        }
        "ternary_trit5" => {
            let group_size = tensor.codec_group_size.ok_or_else(|| {
                RuntimeError::new(
                    ErrorCode::InvalidPackage,
                    "ternary tensor has no group size",
                )
            })?;
            let config = TernaryConfig::new(group_size)?;
            let expected_hash = codec_hash(
                format!(
                    "{{\"group_size\":{group_size},\"packing\":\"trit5\",\
                     \"scale_dtype\":\"float32\",\"threshold_denominator\":10,\
                     \"threshold_numerator\":7,\"version\":\"1.0.0\"}}"
                )
                .as_bytes(),
            );
            if tensor.codec_config_hash.as_deref() != Some(expected_hash.as_str())
                || tensor.encoded_bytes != config.encoded_size(elements)?
            {
                return Err(RuntimeError::new(
                    ErrorCode::IntegrityFailure,
                    "ternary tensor codec identity or byte count is inconsistent",
                ));
            }
            Ok(Glm4BindingEncoding::Ternary(config))
        }
        "int4_symmetric" => {
            let group_size = tensor.codec_group_size.ok_or_else(|| {
                RuntimeError::new(ErrorCode::InvalidPackage, "INT4 tensor has no group size")
            })?;
            let config = Int4Config::new(group_size)?;
            let expected_hash = codec_hash(
                format!(
                    "{{\"group_size\":{group_size},\"packing\":\"signed-nibble-low-first\",\
                     \"scale_dtype\":\"float32\",\"version\":\"1.0.0\"}}"
                )
                .as_bytes(),
            );
            if tensor.codec_config_hash.as_deref() != Some(expected_hash.as_str())
                || tensor.encoded_bytes != config.encoded_size(elements)?
            {
                return Err(RuntimeError::new(
                    ErrorCode::IntegrityFailure,
                    "INT4 tensor codec identity or byte count is inconsistent",
                ));
            }
            Ok(Glm4BindingEncoding::Int4(config))
        }
        _ => Err(RuntimeError::new(
            ErrorCode::CapabilityMismatch,
            "tensor encoding is unsupported",
        )),
    }
}

fn derive_cache_bytes(
    architecture: &ArchitectureIdentity,
    context_capacity: usize,
    key_dtype: IdentityDType,
    value_dtype: IdentityDType,
) -> Result<(usize, usize, usize), RuntimeError> {
    let key_elements = checked_mul(
        architecture.num_attention_heads,
        architecture.qk_head_dim,
        "cache key row elements",
    )?;
    let value_elements = checked_mul(
        architecture.num_attention_heads,
        architecture.v_head_dim,
        "cache value row elements",
    )?;
    let key_bytes = checked_mul(key_elements, key_dtype.item_bytes(), "cache key row bytes")?;
    let value_bytes = checked_mul(
        value_elements,
        value_dtype.item_bytes(),
        "cache value row bytes",
    )?;
    let staging = key_bytes
        .checked_add(value_bytes)
        .ok_or_else(|| RuntimeError::new(ErrorCode::InvalidPackage, "cache staging overflowed"))?;
    let per_layer = checked_mul(context_capacity, staging, "cache storage per layer")?;
    let total = checked_mul(
        architecture.num_hidden_layers,
        per_layer,
        "cache storage total",
    )?;
    Ok((staging, per_layer, total))
}

fn checked_product(values: &[usize], name: &str) -> Result<usize, RuntimeError> {
    values
        .iter()
        .try_fold(1usize, |product, value| checked_mul(product, *value, name))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, RuntimeError> {
    left.checked_mul(right).ok_or_else(|| {
        RuntimeError::new(
            ErrorCode::InvalidPackage,
            format!("{name} overflowed native usize"),
        )
    })
}

fn codec_hash(payload: &[u8]) -> String {
    sha256_digest(payload)
}

fn validate_identifier(value: &str, name: &str) -> Result<(), RuntimeError> {
    let mut bytes = value.bytes();
    let first = bytes.next();
    if value.len() > 512
        || !first.is_some_and(|byte| byte.is_ascii_alphanumeric())
        || !bytes.all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(byte, b'.' | b'_' | b':' | b'/' | b'@' | b'+' | b'-')
        })
    {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            format!("{name} is not a valid AMS identifier"),
        ));
    }
    Ok(())
}

fn validate_sha256(value: &str, name: &str) -> Result<(), RuntimeError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            format!("{name} is not a SHA-256 digest"),
        ));
    };
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RuntimeError::new(
            ErrorCode::InvalidPackage,
            format!("{name} is not a lowercase SHA-256 digest"),
        ));
    }
    Ok(())
}

fn sha256_digest(payload: &[u8]) -> String {
    let digest = Sha256::digest(payload);
    format_sha256(&digest)
}

fn format_sha256(digest: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity("sha256:".len() + digest.len() * 2);
    output.push_str("sha256:");
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

fn verify_reader_sha256(
    reader: &FileRangeReader,
    expected: &str,
    buffer: &mut [u8],
) -> Result<(), RuntimeError> {
    let mut hasher = Sha256::new();
    let mut offset = 0u64;
    while offset < reader.len() {
        let remaining = reader.len() - offset;
        let count = usize::try_from(remaining.min(buffer.len() as u64)).map_err(|_| {
            RuntimeError::new(
                ErrorCode::InvalidPackage,
                "verification range exceeds usize",
            )
        })?;
        reader.read_exact_at(offset, &mut buffer[..count])?;
        hasher.update(&buffer[..count]);
        offset = offset.checked_add(count as u64).ok_or_else(|| {
            RuntimeError::new(ErrorCode::InvalidPackage, "verification offset overflowed")
        })?;
    }
    let observed = format_sha256(&hasher.finalize());
    if observed != expected {
        return Err(RuntimeError::new(
            ErrorCode::IntegrityFailure,
            "storage object content hash differs from the binding identity",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_rejects_unknown_fields_before_any_path_is_opened() {
        let payload = br#"{
            "schema_id":"ams.native.glm4-envelope.v1",
            "binding_hash":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "binding_identity_json":"{}",
            "storage_paths":[],
            "unknown":true
        }"#;
        let error = admit_glm4_binding_bytes(payload, 1024).err();
        assert_eq!(
            error.as_ref().map(RuntimeError::code),
            Some(ErrorCode::InvalidPackage)
        );
    }

    #[test]
    fn envelope_hash_mismatch_is_an_integrity_failure_before_identity_parse() {
        let payload = br#"{
            "schema_id":"ams.native.glm4-envelope.v1",
            "binding_hash":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "binding_identity_json":"{}",
            "storage_paths":[]
        }"#;
        let error = admit_glm4_binding_bytes(payload, 1024).err();
        assert_eq!(
            error.as_ref().map(RuntimeError::code),
            Some(ErrorCode::IntegrityFailure)
        );
    }

    #[test]
    fn codec_hashes_match_the_python_v1_canonical_vectors() {
        let ternary = codec_hash(
            br#"{"group_size":128,"packing":"trit5","scale_dtype":"float32","threshold_denominator":10,"threshold_numerator":7,"version":"1.0.0"}"#,
        );
        let int4 = codec_hash(
            br#"{"group_size":128,"packing":"signed-nibble-low-first","scale_dtype":"float32","version":"1.0.0"}"#,
        );
        assert_eq!(
            ternary,
            "sha256:1dc19a60253d029e748a1d652baaa61f66716e9241a2e2ad779412fdf1a6c5d3"
        );
        assert_eq!(
            int4,
            "sha256:3f79c72ebff39734ec3d34b3b0d0551e5c7569c657fb6579ef4f9acf4cabb881"
        );
    }

    #[test]
    fn identifiers_and_hashes_fail_closed() {
        assert!(validate_identifier("tensor:model.0", "test").is_ok());
        assert!(validate_identifier("../escape", "test").is_err());
        assert!(
            validate_sha256(
                "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "test"
            )
            .is_ok()
        );
        assert!(
            validate_sha256(
                "sha256:0123456789ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef",
                "test"
            )
            .is_err()
        );
    }

    #[test]
    fn expected_inventory_includes_exact_base_and_mtp_schedule() {
        let architecture = ArchitectureIdentity {
            content_hash: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                .to_owned(),
            hidden_size: 4,
            intermediate_size: 6,
            moe_intermediate_size: 3,
            vocab_size: 8,
            num_hidden_layers: 2,
            num_nextn_predict_layers: 1,
            first_k_dense_replace: 1,
            n_routed_experts: 2,
            n_shared_experts: 1,
            num_experts_per_tok: 1,
            n_group: 1,
            topk_group: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            q_lora_rank: 2,
            kv_lora_rank: 2,
            qk_nope_head_dim: 1,
            qk_rope_head_dim: 2,
            qk_head_dim: 3,
            v_head_dim: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 0.00001,
            rope_theta: 10_000.0,
            routed_scaling_factor: 1.5,
            mlp_layer_types: vec!["dense".to_owned(), "sparse".to_owned()],
        };
        let expected = expected_inventory(&architecture).unwrap_or_default();
        assert_eq!(expected.len(), 61);
        assert_eq!(
            expected
                .get("model.layers.2.shared_head.head.weight")
                .map(|tensor| (tensor.role, tensor.mtp)),
            Some(("mtp_shared_head", true))
        );
    }
}
