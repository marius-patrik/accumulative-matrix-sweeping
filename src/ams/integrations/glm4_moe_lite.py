"""Fail-closed GLM-4-MoE-Lite architecture and tensor inventory normalization."""

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
    "attention_dropout",
    "dtype",
    "first_k_dense_replace",
    "hidden_act",
    "hidden_size",
    "intermediate_size",
    "kv_lora_rank",
    "max_position_embeddings",
    "model_type",
    "moe_intermediate_size",
    "n_group",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "partial_rotary_factor",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "routed_scaling_factor",
    "tie_word_embeddings",
    "topk_group",
    "topk_method",
    "v_head_dim",
    "vocab_size",
}

_REVIEWED_OPTIONAL_CONFIG_FIELDS = {
    "bos_token_id",
    "eos_token_id",
    "initializer_range",
    "mlp_layer_types",
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
    return checked_positive(raw[name], name=f"glm4_moe_lite.{name}")


def _finite_positive_number(raw: dict[str, Any], name: str) -> float:
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"GLM-4-MoE-Lite {name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"GLM-4-MoE-Lite {name} must be finite and positive",
        )
    return normalized


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteArchitecture:
    """Normalized fields that determine the GLM-4-MoE-Lite graph and inventory."""

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
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    routed_scaling_factor: float
    mlp_layer_types: tuple[str, ...]


def parse_glm4_moe_lite_architecture(payload: bytes) -> Glm4MoeLiteArchitecture:
    """Parse only reviewed GLM-4-MoE-Lite fields needed for exact graph admission."""
    if not payload or len(payload) > _MAX_CONFIG_BYTES:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM-4-MoE-Lite config size is invalid")
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateConfigKey, ValueError) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM-4-MoE-Lite config JSON is invalid") from exc
    allowed = _REQUIRED_CONFIG_FIELDS | _REVIEWED_OPTIONAL_CONFIG_FIELDS
    if (
        not isinstance(raw, dict)
        or not _REQUIRED_CONFIG_FIELDS.issubset(raw)
        or not set(raw).issubset(allowed)
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "GLM-4-MoE-Lite config fields are missing or unreviewed",
        )
    if raw["architectures"] != ["Glm4MoeLiteForCausalLM"] or raw["model_type"] != "glm4_moe_lite":
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "config is not a GLM-4-MoE-Lite causal LM",
        )
    fixed_values = {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "hidden_act": "silu",
        "norm_topk_prob": True,
        "partial_rotary_factor": 1.0,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "topk_method": "noaux_tc",
    }
    if any(raw[name] != value for name, value in fixed_values.items()):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4-MoE-Lite execution semantics are unsupported",
        )
    optional_fixed_values = {
        "rope_interleave": True,
        "use_cache": True,
    }
    if any(name in raw and raw[name] != value for name, value in optional_fixed_values.items()):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4-MoE-Lite optional execution semantics are unsupported",
        )

    num_hidden_layers = _positive_integer(raw, "num_hidden_layers")
    num_nextn_predict_layers = _positive_integer(raw, "num_nextn_predict_layers")
    first_k_dense_replace = _positive_integer(raw, "first_k_dense_replace")
    if first_k_dense_replace != 1 or first_k_dense_replace > num_hidden_layers:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4-MoE-Lite dense layer prefix is unsupported",
        )
    expected_mlp_types = ("dense",) * first_k_dense_replace + ("sparse",) * (
        num_hidden_layers - first_k_dense_replace
    )
    if "mlp_layer_types" in raw:
        mlp_layer_types = raw["mlp_layer_types"]
        if not isinstance(mlp_layer_types, list) or tuple(mlp_layer_types) != expected_mlp_types:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "GLM-4-MoE-Lite MLP layer schedule is unsupported",
            )

    n_routed_experts = _positive_integer(raw, "n_routed_experts")
    num_experts_per_tok = _positive_integer(raw, "num_experts_per_tok")
    n_group = _positive_integer(raw, "n_group")
    topk_group = _positive_integer(raw, "topk_group")
    experts_per_group = n_routed_experts // n_group if n_group else 0
    if (
        num_experts_per_tok > n_routed_experts
        or n_routed_experts % n_group != 0
        or topk_group > n_group
        or experts_per_group < 2
        or num_experts_per_tok > topk_group * experts_per_group
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "GLM-4-MoE-Lite expert routing dimensions are invalid",
        )

    qk_nope_head_dim = _positive_integer(raw, "qk_nope_head_dim")
    qk_rope_head_dim = _positive_integer(raw, "qk_rope_head_dim")
    num_attention_heads = _positive_integer(raw, "num_attention_heads")
    num_key_value_heads = _positive_integer(raw, "num_key_value_heads")
    if qk_rope_head_dim % 2 or num_attention_heads != num_key_value_heads:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "GLM-4-MoE-Lite attention head dimensions are inconsistent",
        )

    return Glm4MoeLiteArchitecture(
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
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        q_lora_rank=_positive_integer(raw, "q_lora_rank"),
        kv_lora_rank=_positive_integer(raw, "kv_lora_rank"),
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        v_head_dim=_positive_integer(raw, "v_head_dim"),
        max_position_embeddings=_positive_integer(raw, "max_position_embeddings"),
        rms_norm_eps=_finite_positive_number(raw, "rms_norm_eps"),
        rope_theta=_finite_positive_number(raw, "rope_theta"),
        routed_scaling_factor=_finite_positive_number(raw, "routed_scaling_factor"),
        mlp_layer_types=expected_mlp_types,
    )


class Glm4MoeLiteTensorRole(StrEnum):
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
    MTP_EMBEDDING = "mtp_embedding"
    MTP_SHARED_HEAD = "mtp_shared_head"
    MTP_SHARED_HEAD_NORM = "mtp_shared_head_norm"


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteTensorSlot:
    tensor_name: str
    role: Glm4MoeLiteTensorRole
    layer_index: int | None = None
    expert_index: int | None = None
    mtp: bool = False


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteTensorInventory:
    architecture_hash: str
    index_hash: str
    slots: tuple[Glm4MoeLiteTensorSlot, ...]


def expected_glm4_moe_lite_tensor_shape(
    architecture: Glm4MoeLiteArchitecture,
    slot: Glm4MoeLiteTensorSlot,
) -> tuple[int, ...]:
    """Return the sole reviewed row-major shape for one GLM-4 tensor slot."""

    role = slot.role
    hidden = architecture.hidden_size
    shared_intermediate = architecture.moe_intermediate_size * architecture.n_shared_experts
    if role in {
        Glm4MoeLiteTensorRole.EMBEDDING,
        Glm4MoeLiteTensorRole.LM_HEAD,
        Glm4MoeLiteTensorRole.MTP_EMBEDDING,
        Glm4MoeLiteTensorRole.MTP_SHARED_HEAD,
    }:
        return (architecture.vocab_size, hidden)
    if role in {
        Glm4MoeLiteTensorRole.FINAL_NORM,
        Glm4MoeLiteTensorRole.INPUT_NORM,
        Glm4MoeLiteTensorRole.POST_ATTENTION_NORM,
        Glm4MoeLiteTensorRole.MTP_EMBED_NORM,
        Glm4MoeLiteTensorRole.MTP_HIDDEN_NORM,
        Glm4MoeLiteTensorRole.MTP_SHARED_HEAD_NORM,
    }:
        return (hidden,)
    if role is Glm4MoeLiteTensorRole.ATTENTION_Q_A_PROJECTION:
        return (architecture.q_lora_rank, hidden)
    if role is Glm4MoeLiteTensorRole.ATTENTION_Q_A_NORM:
        return (architecture.q_lora_rank,)
    if role is Glm4MoeLiteTensorRole.ATTENTION_Q_B_PROJECTION:
        return (
            architecture.num_attention_heads * architecture.qk_head_dim,
            architecture.q_lora_rank,
        )
    if role is Glm4MoeLiteTensorRole.ATTENTION_KV_A_PROJECTION:
        return (architecture.kv_lora_rank + architecture.qk_rope_head_dim, hidden)
    if role is Glm4MoeLiteTensorRole.ATTENTION_KV_A_NORM:
        return (architecture.kv_lora_rank,)
    if role is Glm4MoeLiteTensorRole.ATTENTION_KV_B_PROJECTION:
        return (
            architecture.num_attention_heads
            * (architecture.qk_nope_head_dim + architecture.v_head_dim),
            architecture.kv_lora_rank,
        )
    if role is Glm4MoeLiteTensorRole.ATTENTION_OUTPUT_PROJECTION:
        return (hidden, architecture.num_attention_heads * architecture.v_head_dim)
    if role in {
        Glm4MoeLiteTensorRole.DENSE_GATE_PROJECTION,
        Glm4MoeLiteTensorRole.DENSE_UP_PROJECTION,
    }:
        return (architecture.intermediate_size, hidden)
    if role is Glm4MoeLiteTensorRole.DENSE_DOWN_PROJECTION:
        return (hidden, architecture.intermediate_size)
    if role is Glm4MoeLiteTensorRole.ROUTER_WEIGHT:
        return (architecture.n_routed_experts, hidden)
    if role is Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS:
        return (architecture.n_routed_experts,)
    if role in {
        Glm4MoeLiteTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
        Glm4MoeLiteTensorRole.ROUTED_EXPERT_UP_PROJECTION,
    }:
        return (architecture.moe_intermediate_size, hidden)
    if role is Glm4MoeLiteTensorRole.ROUTED_EXPERT_DOWN_PROJECTION:
        return (hidden, architecture.moe_intermediate_size)
    if role in {
        Glm4MoeLiteTensorRole.SHARED_EXPERT_GATE_PROJECTION,
        Glm4MoeLiteTensorRole.SHARED_EXPERT_UP_PROJECTION,
    }:
        return (shared_intermediate, hidden)
    if role is Glm4MoeLiteTensorRole.SHARED_EXPERT_DOWN_PROJECTION:
        return (hidden, shared_intermediate)
    if role is Glm4MoeLiteTensorRole.MTP_EMBED_HIDDEN_PROJECTION:
        return (hidden, hidden * 2)
    raise AmsError(
        ErrorCode.INTERNAL_INVARIANT,
        "GLM-4 tensor role has no reviewed shape",
    )


def _add_layer_slots(
    slots: dict[str, Glm4MoeLiteTensorSlot],
    architecture: Glm4MoeLiteArchitecture,
    layer_index: int,
    *,
    mlp_type: str,
    mtp: bool,
) -> None:
    prefix = f"model.layers.{layer_index}"

    def add(
        suffix: str,
        role: Glm4MoeLiteTensorRole,
        expert_index: int | None = None,
    ) -> None:
        name = f"{prefix}.{suffix}"
        slots[name] = Glm4MoeLiteTensorSlot(name, role, layer_index, expert_index, mtp)

    common = {
        "input_layernorm.weight": Glm4MoeLiteTensorRole.INPUT_NORM,
        "post_attention_layernorm.weight": Glm4MoeLiteTensorRole.POST_ATTENTION_NORM,
        "self_attn.q_a_proj.weight": Glm4MoeLiteTensorRole.ATTENTION_Q_A_PROJECTION,
        "self_attn.q_a_layernorm.weight": Glm4MoeLiteTensorRole.ATTENTION_Q_A_NORM,
        "self_attn.q_b_proj.weight": Glm4MoeLiteTensorRole.ATTENTION_Q_B_PROJECTION,
        "self_attn.kv_a_proj_with_mqa.weight": (Glm4MoeLiteTensorRole.ATTENTION_KV_A_PROJECTION),
        "self_attn.kv_a_layernorm.weight": Glm4MoeLiteTensorRole.ATTENTION_KV_A_NORM,
        "self_attn.kv_b_proj.weight": Glm4MoeLiteTensorRole.ATTENTION_KV_B_PROJECTION,
        "self_attn.o_proj.weight": Glm4MoeLiteTensorRole.ATTENTION_OUTPUT_PROJECTION,
    }
    for suffix, role in common.items():
        add(suffix, role)
    if mlp_type == "dense":
        add("mlp.gate_proj.weight", Glm4MoeLiteTensorRole.DENSE_GATE_PROJECTION)
        add("mlp.up_proj.weight", Glm4MoeLiteTensorRole.DENSE_UP_PROJECTION)
        add("mlp.down_proj.weight", Glm4MoeLiteTensorRole.DENSE_DOWN_PROJECTION)
    else:
        add("mlp.gate.weight", Glm4MoeLiteTensorRole.ROUTER_WEIGHT)
        add(
            "mlp.gate.e_score_correction_bias",
            Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS,
        )
        for expert_index in range(architecture.n_routed_experts):
            expert_prefix = f"mlp.experts.{expert_index}"
            add(
                f"{expert_prefix}.gate_proj.weight",
                Glm4MoeLiteTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
                expert_index,
            )
            add(
                f"{expert_prefix}.up_proj.weight",
                Glm4MoeLiteTensorRole.ROUTED_EXPERT_UP_PROJECTION,
                expert_index,
            )
            add(
                f"{expert_prefix}.down_proj.weight",
                Glm4MoeLiteTensorRole.ROUTED_EXPERT_DOWN_PROJECTION,
                expert_index,
            )
        add(
            "mlp.shared_experts.gate_proj.weight",
            Glm4MoeLiteTensorRole.SHARED_EXPERT_GATE_PROJECTION,
        )
        add(
            "mlp.shared_experts.up_proj.weight",
            Glm4MoeLiteTensorRole.SHARED_EXPERT_UP_PROJECTION,
        )
        add(
            "mlp.shared_experts.down_proj.weight",
            Glm4MoeLiteTensorRole.SHARED_EXPERT_DOWN_PROJECTION,
        )
    if mtp:
        add("enorm.weight", Glm4MoeLiteTensorRole.MTP_EMBED_NORM)
        add("hnorm.weight", Glm4MoeLiteTensorRole.MTP_HIDDEN_NORM)
        add("eh_proj.weight", Glm4MoeLiteTensorRole.MTP_EMBED_HIDDEN_PROJECTION)
        add("embed_tokens.weight", Glm4MoeLiteTensorRole.MTP_EMBEDDING)
        add("shared_head.head.weight", Glm4MoeLiteTensorRole.MTP_SHARED_HEAD)
        add("shared_head.norm.weight", Glm4MoeLiteTensorRole.MTP_SHARED_HEAD_NORM)


def expected_glm4_moe_lite_tensor_slots(
    architecture: Glm4MoeLiteArchitecture,
) -> tuple[Glm4MoeLiteTensorSlot, ...]:
    """Construct the exact reviewed tensor-name set, including separate MTP weights."""
    roots = {
        "model.embed_tokens.weight": Glm4MoeLiteTensorSlot(
            "model.embed_tokens.weight",
            Glm4MoeLiteTensorRole.EMBEDDING,
        ),
        "model.norm.weight": Glm4MoeLiteTensorSlot(
            "model.norm.weight",
            Glm4MoeLiteTensorRole.FINAL_NORM,
        ),
        "lm_head.weight": Glm4MoeLiteTensorSlot(
            "lm_head.weight",
            Glm4MoeLiteTensorRole.LM_HEAD,
        ),
    }
    for layer_index, mlp_type in enumerate(architecture.mlp_layer_types):
        _add_layer_slots(
            roots,
            architecture,
            layer_index,
            mlp_type=mlp_type,
            mtp=False,
        )
    for offset in range(architecture.num_nextn_predict_layers):
        _add_layer_slots(
            roots,
            architecture,
            architecture.num_hidden_layers + offset,
            mlp_type="sparse",
            mtp=True,
        )
    return tuple(roots[name] for name in sorted(roots))


def validate_glm4_moe_lite_tensor_inventory(
    architecture: Glm4MoeLiteArchitecture,
    index: HuggingFaceShardIndex,
) -> Glm4MoeLiteTensorInventory:
    """Require the provider index to equal the reviewed tensor inventory exactly."""
    expected = expected_glm4_moe_lite_tensor_slots(architecture)
    expected_by_name = {slot.tensor_name: slot for slot in expected}
    actual_names = {entry.tensor_name for entry in index.entries}
    if actual_names != set(expected_by_name):
        missing = len(set(expected_by_name) - actual_names)
        unexpected = len(actual_names - set(expected_by_name))
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "Hugging Face index does not match the reviewed GLM-4-MoE-Lite inventory",
            evidence={"missing": missing, "unexpected": unexpected},
        )
    return Glm4MoeLiteTensorInventory(
        architecture_hash=architecture.content_hash,
        index_hash=index.content_hash,
        slots=expected,
    )
