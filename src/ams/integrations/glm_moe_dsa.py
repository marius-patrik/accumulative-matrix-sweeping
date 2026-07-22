"""Fail-closed GLM-MoE-DSA architecture and tensor inventory normalization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ams.checked import checked_positive
from ams.errors import AmsError, ErrorCode
from ams.integrations.huggingface import HuggingFaceShardIndex

_MAX_CONFIG_BYTES = 1024 * 1024

_REQUIRED_CONFIG_FIELDS = {
    "architectures",
    "attention_bias",
    "dtype",
    "first_k_dense_replace",
    "hidden_act",
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_share_for_mtp_iteration",
    "index_topk",
    "indexer_types",
    "intermediate_size",
    "kv_lora_rank",
    "max_position_embeddings",
    "mlp_layer_types",
    "model_type",
    "moe_intermediate_size",
    "moe_router_dtype",
    "n_group",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "q_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_parameters",
    "routed_scaling_factor",
    "scoring_func",
    "tie_word_embeddings",
    "topk_group",
    "topk_method",
    "v_head_dim",
    "vocab_size",
}

_REVIEWED_OPTIONAL_CONFIG_FIELDS = {
    "attention_dropout",
    "eos_token_id",
    "ep_size",
    "head_dim",
    "index_skip_topk_offset",
    "index_topk_freq",
    "index_topk_pattern",
    "indexer_rope_interleave",
    "initializer_range",
    "moe_layer_freq",
    "pad_token_id",
    "pretraining_tp",
    "rope_interleave",
    "transformers_version",
    "use_cache",
}


class _DuplicateConfigKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateConfigKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _positive_integer(raw: dict[str, Any], name: str) -> int:
    return checked_positive(raw[name], name=f"glm_moe_dsa.{name}")


def _finite_positive_number(raw: dict[str, Any], name: str) -> float:
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"GLM {name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"GLM {name} must be finite and positive")
    return normalized


def _string_tuple(raw: dict[str, Any], name: str, expected_length: int) -> tuple[str, ...]:
    values = raw[name]
    if (
        not isinstance(values, list)
        or len(values) != expected_length
        or any(not isinstance(value, str) for value in values)
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"GLM {name} has an invalid shape or value")
    return tuple(values)


@dataclass(frozen=True, slots=True)
class GlmMoeDsaArchitecture:
    """Normalized fields that determine the GLM-MoE-DSA graph and tensor inventory."""

    content_hash: str
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    first_k_dense_replace: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    n_group: int
    topk_group: int
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    qk_head_dim: int
    v_head_dim: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    routed_scaling_factor: float
    index_share_for_mtp_iteration: bool
    mlp_layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]


def parse_glm_moe_dsa_architecture(payload: bytes) -> GlmMoeDsaArchitecture:
    """Parse only the reviewed GLM fields needed to construct an exact graph inventory."""
    if not payload or len(payload) > _MAX_CONFIG_BYTES:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM config size is invalid")
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateConfigKey, ValueError) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM config JSON is invalid") from exc
    allowed = _REQUIRED_CONFIG_FIELDS | _REVIEWED_OPTIONAL_CONFIG_FIELDS
    if (
        not isinstance(raw, dict)
        or not _REQUIRED_CONFIG_FIELDS.issubset(raw)
        or not set(raw).issubset(allowed)
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM config fields are missing or unreviewed")
    if raw["architectures"] != ["GlmMoeDsaForCausalLM"] or raw["model_type"] != "glm_moe_dsa":
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "config is not GLM-MoE-DSA causal LM")
    fixed_values = {
        "attention_bias": False,
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "moe_router_dtype": "float32",
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
    }
    if any(raw[name] != value for name, value in fixed_values.items()):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "GLM execution semantics are unsupported")

    num_hidden_layers = _positive_integer(raw, "num_hidden_layers")
    num_nextn_predict_layers = _positive_integer(raw, "num_nextn_predict_layers")
    first_k_dense_replace = _positive_integer(raw, "first_k_dense_replace")
    if first_k_dense_replace > num_hidden_layers:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM dense layer prefix exceeds decoder depth")
    mlp_layer_types = _string_tuple(raw, "mlp_layer_types", num_hidden_layers)
    expected_mlp_types = ("dense",) * first_k_dense_replace + ("sparse",) * (
        num_hidden_layers - first_k_dense_replace
    )
    if mlp_layer_types != expected_mlp_types:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "GLM MLP layer schedule is unsupported")
    indexer_types = _string_tuple(raw, "indexer_types", num_hidden_layers)
    saw_full_indexer = False
    for indexer_type in indexer_types:
        if indexer_type == "full":
            saw_full_indexer = True
        elif indexer_type != "shared" or not saw_full_indexer:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM indexer sharing schedule is invalid")

    n_routed_experts = _positive_integer(raw, "n_routed_experts")
    num_experts_per_tok = _positive_integer(raw, "num_experts_per_tok")
    n_group = _positive_integer(raw, "n_group")
    topk_group = _positive_integer(raw, "topk_group")
    if (
        num_experts_per_tok > n_routed_experts
        or n_routed_experts % n_group != 0
        or topk_group > n_group
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM expert routing dimensions are invalid")

    qk_nope_head_dim = _positive_integer(raw, "qk_nope_head_dim")
    qk_rope_head_dim = _positive_integer(raw, "qk_rope_head_dim")
    qk_head_dim = _positive_integer(raw, "qk_head_dim")
    index_head_dim = _positive_integer(raw, "index_head_dim")
    if qk_head_dim != qk_nope_head_dim + qk_rope_head_dim or index_head_dim < qk_rope_head_dim:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM attention head dimensions are inconsistent")
    rope_parameters = raw["rope_parameters"]
    if (
        not isinstance(rope_parameters, dict)
        or set(rope_parameters) != {"rope_theta", "rope_type"}
        or rope_parameters["rope_type"] != "default"
    ):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "GLM RoPE configuration is unsupported")
    index_share_for_mtp_iteration = raw["index_share_for_mtp_iteration"]
    if not isinstance(index_share_for_mtp_iteration, bool):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM MTP index-sharing flag must be boolean")

    return GlmMoeDsaArchitecture(
        content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        hidden_size=_positive_integer(raw, "hidden_size"),
        intermediate_size=_positive_integer(raw, "intermediate_size"),
        moe_intermediate_size=_positive_integer(raw, "moe_intermediate_size"),
        vocab_size=_positive_integer(raw, "vocab_size"),
        num_hidden_layers=num_hidden_layers,
        num_nextn_predict_layers=num_nextn_predict_layers,
        first_k_dense_replace=first_k_dense_replace,
        n_routed_experts=n_routed_experts,
        n_shared_experts=_positive_integer(raw, "n_shared_experts"),
        num_experts_per_tok=num_experts_per_tok,
        n_group=n_group,
        topk_group=topk_group,
        num_attention_heads=_positive_integer(raw, "num_attention_heads"),
        num_key_value_heads=_positive_integer(raw, "num_key_value_heads"),
        q_lora_rank=_positive_integer(raw, "q_lora_rank"),
        kv_lora_rank=_positive_integer(raw, "kv_lora_rank"),
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_head_dim,
        v_head_dim=_positive_integer(raw, "v_head_dim"),
        index_head_dim=index_head_dim,
        index_n_heads=_positive_integer(raw, "index_n_heads"),
        index_topk=_positive_integer(raw, "index_topk"),
        max_position_embeddings=_positive_integer(raw, "max_position_embeddings"),
        rms_norm_eps=_finite_positive_number(raw, "rms_norm_eps"),
        rope_theta=_finite_positive_number(rope_parameters, "rope_theta"),
        routed_scaling_factor=_finite_positive_number(raw, "routed_scaling_factor"),
        index_share_for_mtp_iteration=index_share_for_mtp_iteration,
        mlp_layer_types=mlp_layer_types,
        indexer_types=indexer_types,
    )


class GlmTensorRole(StrEnum):
    EMBEDDING = "embedding"
    FINAL_NORM = "final_norm"
    LM_HEAD = "lm_head"
    INPUT_NORM = "input_norm"
    POST_ATTENTION_NORM = "post_attention_norm"
    ATTENTION_Q_A_PROJECTION = "attention_q_a_projection"
    ATTENTION_Q_A_NORM = "attention_q_a_norm"
    ATTENTION_Q_B_PROJECTION = "attention_q_b_projection"
    ATTENTION_KV_A_PROJECTION = "attention_kv_a_projection"
    ATTENTION_KV_A_NORM = "attention_kv_a_norm"
    ATTENTION_KV_B_PROJECTION = "attention_kv_b_projection"
    ATTENTION_OUTPUT_PROJECTION = "attention_output_projection"
    INDEXER_WQ_B_PROJECTION = "indexer_wq_b_projection"
    INDEXER_WK_PROJECTION = "indexer_wk_projection"
    INDEXER_K_NORM_WEIGHT = "indexer_k_norm_weight"
    INDEXER_K_NORM_BIAS = "indexer_k_norm_bias"
    INDEXER_WEIGHTS_PROJECTION = "indexer_weights_projection"
    DENSE_GATE_PROJECTION = "dense_gate_projection"
    DENSE_UP_PROJECTION = "dense_up_projection"
    DENSE_DOWN_PROJECTION = "dense_down_projection"
    ROUTER_WEIGHT = "router_weight"
    ROUTER_CORRECTION_BIAS = "router_correction_bias"
    ROUTED_EXPERT_GATE_PROJECTION = "routed_expert_gate_projection"
    ROUTED_EXPERT_UP_PROJECTION = "routed_expert_up_projection"
    ROUTED_EXPERT_DOWN_PROJECTION = "routed_expert_down_projection"
    SHARED_EXPERT_GATE_PROJECTION = "shared_expert_gate_projection"
    SHARED_EXPERT_UP_PROJECTION = "shared_expert_up_projection"
    SHARED_EXPERT_DOWN_PROJECTION = "shared_expert_down_projection"
    MTP_EMBED_NORM = "mtp_embed_norm"
    MTP_HIDDEN_NORM = "mtp_hidden_norm"
    MTP_EMBED_HIDDEN_PROJECTION = "mtp_embed_hidden_projection"
    MTP_SHARED_HEAD_NORM = "mtp_shared_head_norm"


@dataclass(frozen=True, slots=True)
class GlmTensorSlot:
    tensor_name: str
    role: GlmTensorRole
    layer_index: int | None = None
    expert_index: int | None = None
    mtp: bool = False


@dataclass(frozen=True, slots=True)
class GlmTensorInventory:
    architecture_hash: str
    index_hash: str
    slots: tuple[GlmTensorSlot, ...]


def _add_layer_slots(
    slots: dict[str, GlmTensorSlot],
    architecture: GlmMoeDsaArchitecture,
    layer_index: int,
    *,
    mlp_type: str,
    full_indexer: bool,
    mtp: bool,
) -> None:
    prefix = f"model.layers.{layer_index}"

    def add(suffix: str, role: GlmTensorRole, expert_index: int | None = None) -> None:
        name = f"{prefix}.{suffix}"
        slots[name] = GlmTensorSlot(name, role, layer_index, expert_index, mtp)

    common = {
        "input_layernorm.weight": GlmTensorRole.INPUT_NORM,
        "post_attention_layernorm.weight": GlmTensorRole.POST_ATTENTION_NORM,
        "self_attn.q_a_proj.weight": GlmTensorRole.ATTENTION_Q_A_PROJECTION,
        "self_attn.q_a_layernorm.weight": GlmTensorRole.ATTENTION_Q_A_NORM,
        "self_attn.q_b_proj.weight": GlmTensorRole.ATTENTION_Q_B_PROJECTION,
        "self_attn.kv_a_proj_with_mqa.weight": GlmTensorRole.ATTENTION_KV_A_PROJECTION,
        "self_attn.kv_a_layernorm.weight": GlmTensorRole.ATTENTION_KV_A_NORM,
        "self_attn.kv_b_proj.weight": GlmTensorRole.ATTENTION_KV_B_PROJECTION,
        "self_attn.o_proj.weight": GlmTensorRole.ATTENTION_OUTPUT_PROJECTION,
    }
    for suffix, role in common.items():
        add(suffix, role)
    if full_indexer:
        indexer = {
            "self_attn.indexer.wq_b.weight": GlmTensorRole.INDEXER_WQ_B_PROJECTION,
            "self_attn.indexer.wk.weight": GlmTensorRole.INDEXER_WK_PROJECTION,
            "self_attn.indexer.k_norm.weight": GlmTensorRole.INDEXER_K_NORM_WEIGHT,
            "self_attn.indexer.k_norm.bias": GlmTensorRole.INDEXER_K_NORM_BIAS,
            "self_attn.indexer.weights_proj.weight": GlmTensorRole.INDEXER_WEIGHTS_PROJECTION,
        }
        for suffix, role in indexer.items():
            add(suffix, role)
    if mlp_type == "dense":
        add("mlp.gate_proj.weight", GlmTensorRole.DENSE_GATE_PROJECTION)
        add("mlp.up_proj.weight", GlmTensorRole.DENSE_UP_PROJECTION)
        add("mlp.down_proj.weight", GlmTensorRole.DENSE_DOWN_PROJECTION)
    else:
        add("mlp.gate.weight", GlmTensorRole.ROUTER_WEIGHT)
        add("mlp.gate.e_score_correction_bias", GlmTensorRole.ROUTER_CORRECTION_BIAS)
        for expert_index in range(architecture.n_routed_experts):
            expert_prefix = f"mlp.experts.{expert_index}"
            add(
                f"{expert_prefix}.gate_proj.weight",
                GlmTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
                expert_index,
            )
            add(
                f"{expert_prefix}.up_proj.weight",
                GlmTensorRole.ROUTED_EXPERT_UP_PROJECTION,
                expert_index,
            )
            add(
                f"{expert_prefix}.down_proj.weight",
                GlmTensorRole.ROUTED_EXPERT_DOWN_PROJECTION,
                expert_index,
            )
        add("mlp.shared_experts.gate_proj.weight", GlmTensorRole.SHARED_EXPERT_GATE_PROJECTION)
        add("mlp.shared_experts.up_proj.weight", GlmTensorRole.SHARED_EXPERT_UP_PROJECTION)
        add("mlp.shared_experts.down_proj.weight", GlmTensorRole.SHARED_EXPERT_DOWN_PROJECTION)
    if mtp:
        add("enorm.weight", GlmTensorRole.MTP_EMBED_NORM)
        add("hnorm.weight", GlmTensorRole.MTP_HIDDEN_NORM)
        add("eh_proj.weight", GlmTensorRole.MTP_EMBED_HIDDEN_PROJECTION)
        add("shared_head.norm.weight", GlmTensorRole.MTP_SHARED_HEAD_NORM)


def expected_glm_tensor_slots(
    architecture: GlmMoeDsaArchitecture,
) -> tuple[GlmTensorSlot, ...]:
    """Construct the exact reviewed tensor-name set, including separately marked MTP layers."""
    roots = {
        "model.embed_tokens.weight": GlmTensorSlot(
            "model.embed_tokens.weight", GlmTensorRole.EMBEDDING
        ),
        "model.norm.weight": GlmTensorSlot("model.norm.weight", GlmTensorRole.FINAL_NORM),
        "lm_head.weight": GlmTensorSlot("lm_head.weight", GlmTensorRole.LM_HEAD),
    }
    for layer_index, mlp_type in enumerate(architecture.mlp_layer_types):
        _add_layer_slots(
            roots,
            architecture,
            layer_index,
            mlp_type=mlp_type,
            full_indexer=architecture.indexer_types[layer_index] == "full",
            mtp=False,
        )
    for offset in range(architecture.num_nextn_predict_layers):
        _add_layer_slots(
            roots,
            architecture,
            architecture.num_hidden_layers + offset,
            mlp_type="sparse",
            full_indexer=True,
            mtp=True,
        )
    return tuple(roots[name] for name in sorted(roots))


def validate_glm_tensor_inventory(
    architecture: GlmMoeDsaArchitecture,
    index: HuggingFaceShardIndex,
) -> GlmTensorInventory:
    """Require the provider index to equal the reviewed GLM tensor inventory exactly."""
    expected = expected_glm_tensor_slots(architecture)
    expected_by_name = {slot.tensor_name: slot for slot in expected}
    actual_names = {entry.tensor_name for entry in index.entries}
    if actual_names != set(expected_by_name):
        missing = len(set(expected_by_name) - actual_names)
        unexpected = len(actual_names - set(expected_by_name))
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "Hugging Face index does not match the reviewed GLM tensor inventory",
            evidence={"missing": missing, "unexpected": unexpected},
        )
    return GlmTensorInventory(
        architecture_hash=architecture.content_hash,
        index_hash=index.content_hash,
        slots=expected,
    )
