"""Composed scalar GLM-MoE-DSA prefill oracle over abstract weight access."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from ams.checked import checked_positive, checked_product
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm_moe_dsa import GlmMoeDsaArchitecture
from ams.ops.glm_moe_dsa import (
    GlmExpertRouting,
    apply_rope_half_split_reference,
    apply_rope_interleaved_reference,
    dsa_topk_reference,
    layer_norm_reference,
    rms_norm_reference,
    route_glm_experts_reference,
    silu_reference,
    softmax_reference,
)


class GlmWeightAccess(Protocol):
    """Minimum semantic weight operations required by the scalar model oracle."""

    def vector(self, tensor_name: str, length: int) -> tuple[float, ...]:
        """Read one complete small vector parameter."""

    def embedding(self, tensor_name: str, index: int, width: int) -> tuple[float, ...]:
        """Read one embedding row."""

    def linear(
        self,
        tensor_name: str,
        values: Sequence[float],
        rows: int,
    ) -> tuple[float, ...]:
        """Multiply one row-major matrix without prescribing its storage encoding."""


@dataclass(frozen=True, slots=True)
class GlmReferenceTensor:
    """Small trusted-fixture tensor; production weights must use bounded package access."""

    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.shape or len(self.shape) > 2:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "reference tensor rank must be one or two")
        expected = checked_product(self.shape, name="reference_tensor.elements")
        if expected != len(self.values):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "reference tensor shape and values differ")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "reference tensor contains non-finite data")


class GlmReferenceWeights:
    """In-memory weight access used only for deliberately miniature trusted fixtures."""

    def __init__(self, tensors: Mapping[str, GlmReferenceTensor]) -> None:
        if not tensors:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "reference weight set must not be empty")
        self._tensors = MappingProxyType(dict(tensors))

    def _tensor(self, tensor_name: str) -> GlmReferenceTensor:
        try:
            return self._tensors[tensor_name]
        except KeyError as exc:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                f"required reference tensor is absent: {tensor_name}",
            ) from exc

    def vector(self, tensor_name: str, length: int) -> tuple[float, ...]:
        checked_positive(length, name="reference_vector.length")
        tensor = self._tensor(tensor_name)
        if tensor.shape != (length,):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "reference vector shape is invalid")
        return tuple(float(value) for value in tensor.values)

    def embedding(self, tensor_name: str, index: int, width: int) -> tuple[float, ...]:
        checked_positive(width, name="reference_embedding.width")
        tensor = self._tensor(tensor_name)
        if len(tensor.shape) != 2 or tensor.shape[1] != width:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "reference embedding shape is invalid")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < tensor.shape[0]
        ):
            raise AmsError(ErrorCode.PLAN_INVALID, "embedding index is outside the vocabulary")
        start = index * width
        return tuple(float(value) for value in tensor.values[start : start + width])

    def linear(
        self,
        tensor_name: str,
        values: Sequence[float],
        rows: int,
    ) -> tuple[float, ...]:
        checked_positive(rows, name="reference_linear.rows")
        tensor = self._tensor(tensor_name)
        if tensor.shape != (rows, len(values)) or not values:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "reference linear shape is invalid")
        normalized = tuple(float(value) for value in values)
        if any(not math.isfinite(value) for value in normalized):
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "reference linear input is non-finite")
        output: list[float] = []
        for row in range(rows):
            accumulator = 0.0
            base = row * len(normalized)
            for column, value in enumerate(normalized):
                accumulator += float(tensor.values[base + column]) * value
            if not math.isfinite(accumulator):
                raise AmsError(ErrorCode.NUMERIC_FAILURE, "reference linear output is non-finite")
            output.append(accumulator)
        return tuple(output)


@dataclass(frozen=True, slots=True)
class GlmReferenceLayerTrace:
    layer_index: int
    indexer_type: str
    dsa_indices: tuple[tuple[int, ...], ...]
    expert_routing: tuple[GlmExpertRouting, ...]


@dataclass(frozen=True, slots=True)
class GlmReferenceOutput:
    hidden_states: tuple[tuple[float, ...], ...]
    logits: tuple[tuple[float, ...], ...]
    layers: tuple[GlmReferenceLayerTrace, ...]


def _add(left: Sequence[float], right: Sequence[float], *, name: str) -> tuple[float, ...]:
    if len(left) != len(right) or not left:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, f"{name} vector dimensions differ")
    return tuple(float(a) + float(b) for a, b in zip(left, right, strict=True))


def _dot(left: Sequence[float], right: Sequence[float], *, name: str) -> float:
    if len(left) != len(right) or not left:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, f"{name} vector dimensions differ")
    accumulator = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        accumulator += float(left_value) * float(right_value)
    if not math.isfinite(accumulator):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, f"{name} output is non-finite")
    return accumulator


def _heads(
    values: Sequence[float], head_count: int, head_dimension: int
) -> tuple[tuple[float, ...], ...]:
    if len(values) != head_count * head_dimension:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "projected attention dimensions differ")
    return tuple(
        tuple(
            float(value) for value in values[index * head_dimension : (index + 1) * head_dimension]
        )
        for index in range(head_count)
    )


def _gated_mlp(
    weights: GlmWeightAccess,
    prefix: str,
    values: Sequence[float],
    intermediate_size: int,
    hidden_size: int,
) -> tuple[float, ...]:
    gate = weights.linear(f"{prefix}.gate_proj.weight", values, intermediate_size)
    up = weights.linear(f"{prefix}.up_proj.weight", values, intermediate_size)
    activated = tuple(
        silu_reference(gate_value) * up_value for gate_value, up_value in zip(gate, up, strict=True)
    )
    return weights.linear(f"{prefix}.down_proj.weight", activated, hidden_size)


def _build_attention_projections(
    architecture: GlmMoeDsaArchitecture,
    weights: GlmWeightAccess,
    prefix: str,
    hidden_states: tuple[tuple[float, ...], ...],
) -> tuple[
    tuple[tuple[tuple[float, ...], ...], ...],
    tuple[tuple[tuple[float, ...], ...], ...],
    tuple[tuple[tuple[float, ...], ...], ...],
    tuple[tuple[float, ...], ...],
]:
    q_a_norm_weight = weights.vector(
        f"{prefix}.self_attn.q_a_layernorm.weight", architecture.q_lora_rank
    )
    kv_a_norm_weight = weights.vector(
        f"{prefix}.self_attn.kv_a_layernorm.weight", architecture.kv_lora_rank
    )
    queries: list[tuple[tuple[float, ...], ...]] = []
    keys: list[tuple[tuple[float, ...], ...]] = []
    values: list[tuple[tuple[float, ...], ...]] = []
    query_residuals: list[tuple[float, ...]] = []
    for position, hidden_state in enumerate(hidden_states):
        q_a = weights.linear(
            f"{prefix}.self_attn.q_a_proj.weight",
            hidden_state,
            architecture.q_lora_rank,
        )
        q_residual = rms_norm_reference(q_a, q_a_norm_weight, architecture.rms_norm_eps)
        query_residuals.append(q_residual)
        q_flat = weights.linear(
            f"{prefix}.self_attn.q_b_proj.weight",
            q_residual,
            architecture.num_attention_heads * architecture.qk_head_dim,
        )
        query_heads: list[tuple[float, ...]] = []
        for head in _heads(q_flat, architecture.num_attention_heads, architecture.qk_head_dim):
            q_pass = head[: architecture.qk_nope_head_dim]
            q_rot = apply_rope_interleaved_reference(
                head[architecture.qk_nope_head_dim :],
                position=position,
                theta=architecture.rope_theta,
            )
            query_heads.append(q_pass + q_rot)
        queries.append(tuple(query_heads))

        compressed = weights.linear(
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            hidden_state,
            architecture.kv_lora_rank + architecture.qk_rope_head_dim,
        )
        kv_pass = rms_norm_reference(
            compressed[: architecture.kv_lora_rank],
            kv_a_norm_weight,
            architecture.rms_norm_eps,
        )
        k_rot = apply_rope_interleaved_reference(
            compressed[architecture.kv_lora_rank :],
            position=position,
            theta=architecture.rope_theta,
        )
        kv_flat = weights.linear(
            f"{prefix}.self_attn.kv_b_proj.weight",
            kv_pass,
            architecture.num_attention_heads
            * (architecture.qk_nope_head_dim + architecture.v_head_dim),
        )
        key_heads: list[tuple[float, ...]] = []
        value_heads: list[tuple[float, ...]] = []
        for head in _heads(
            kv_flat,
            architecture.num_attention_heads,
            architecture.qk_nope_head_dim + architecture.v_head_dim,
        ):
            key_heads.append(head[: architecture.qk_nope_head_dim] + k_rot)
            value_heads.append(head[architecture.qk_nope_head_dim :])
        keys.append(tuple(key_heads))
        values.append(tuple(value_heads))
    return tuple(queries), tuple(keys), tuple(values), tuple(query_residuals)


def _run_full_indexer(
    architecture: GlmMoeDsaArchitecture,
    weights: GlmWeightAccess,
    prefix: str,
    hidden_states: tuple[tuple[float, ...], ...],
    query_residuals: tuple[tuple[float, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    k_norm_weight = weights.vector(
        f"{prefix}.self_attn.indexer.k_norm.weight", architecture.index_head_dim
    )
    k_norm_bias = weights.vector(
        f"{prefix}.self_attn.indexer.k_norm.bias", architecture.index_head_dim
    )
    index_keys: list[tuple[float, ...]] = []
    for position, hidden_state in enumerate(hidden_states):
        key = weights.linear(
            f"{prefix}.self_attn.indexer.wk.weight", hidden_state, architecture.index_head_dim
        )
        key = layer_norm_reference(key, k_norm_weight, k_norm_bias, 1e-6)
        key_rot = apply_rope_half_split_reference(
            key[: architecture.qk_rope_head_dim],
            position=position,
            theta=architecture.rope_theta,
        )
        index_keys.append(key_rot + key[architecture.qk_rope_head_dim :])

    selected: list[tuple[int, ...]] = []
    head_weight_scale = architecture.index_n_heads**-0.5
    for position, (hidden_state, q_residual) in enumerate(
        zip(hidden_states, query_residuals, strict=True)
    ):
        query_flat = weights.linear(
            f"{prefix}.self_attn.indexer.wq_b.weight",
            q_residual,
            architecture.index_n_heads * architecture.index_head_dim,
        )
        query_heads: list[tuple[float, ...]] = []
        for head in _heads(query_flat, architecture.index_n_heads, architecture.index_head_dim):
            q_rot = apply_rope_half_split_reference(
                head[: architecture.qk_rope_head_dim],
                position=position,
                theta=architecture.rope_theta,
            )
            query_heads.append(q_rot + head[architecture.qk_rope_head_dim :])
        head_weights = tuple(
            value * head_weight_scale
            for value in weights.linear(
                f"{prefix}.self_attn.indexer.weights_proj.weight",
                hidden_state,
                architecture.index_n_heads,
            )
        )
        selected.append(
            dsa_topk_reference(
                query_heads,
                index_keys,
                head_weights,
                query_position=position,
                top_k=architecture.index_topk,
            )
        )
    return tuple(selected)


def _run_attention(
    architecture: GlmMoeDsaArchitecture,
    weights: GlmWeightAccess,
    prefix: str,
    hidden_states: tuple[tuple[float, ...], ...],
    previous_indices: tuple[tuple[int, ...], ...] | None,
    indexer_type: str,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[int, ...], ...]]:
    queries, keys, values, query_residuals = _build_attention_projections(
        architecture, weights, prefix, hidden_states
    )
    if indexer_type == "full":
        dsa_indices = _run_full_indexer(
            architecture, weights, prefix, hidden_states, query_residuals
        )
    elif previous_indices is not None:
        dsa_indices = previous_indices
    else:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "shared DSA layer has no prior index")

    outputs: list[tuple[float, ...]] = []
    scale = architecture.qk_head_dim**-0.5
    for query_index, selected_keys in enumerate(dsa_indices):
        concatenated: list[float] = []
        for head_index in range(architecture.num_attention_heads):
            scores = tuple(
                _dot(
                    queries[query_index][head_index],
                    keys[key_index][head_index],
                    name="attention.score",
                )
                * scale
                for key_index in selected_keys
            )
            probabilities = softmax_reference(scores)
            head_output = [0.0] * architecture.v_head_dim
            for probability, key_index in zip(probabilities, selected_keys, strict=True):
                for value_index, value in enumerate(values[key_index][head_index]):
                    head_output[value_index] += probability * value
            concatenated.extend(head_output)
        outputs.append(
            weights.linear(
                f"{prefix}.self_attn.o_proj.weight",
                concatenated,
                architecture.hidden_size,
            )
        )
    return tuple(outputs), dsa_indices


def _run_sparse_mlp(
    architecture: GlmMoeDsaArchitecture,
    weights: GlmWeightAccess,
    prefix: str,
    hidden_states: tuple[tuple[float, ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[GlmExpertRouting, ...]]:
    correction_bias = weights.vector(
        f"{prefix}.mlp.gate.e_score_correction_bias", architecture.n_routed_experts
    )
    outputs: list[tuple[float, ...]] = []
    routes: list[GlmExpertRouting] = []
    shared_intermediate = architecture.moe_intermediate_size * architecture.n_shared_experts
    for hidden_state in hidden_states:
        router_logits = weights.linear(
            f"{prefix}.mlp.gate.weight", hidden_state, architecture.n_routed_experts
        )
        route = route_glm_experts_reference(
            router_logits,
            correction_bias,
            experts_per_token=architecture.num_experts_per_tok,
            group_count=architecture.n_group,
            top_groups=architecture.topk_group,
            routed_scaling_factor=architecture.routed_scaling_factor,
        )
        routes.append(route)
        routed = [0.0] * architecture.hidden_size
        for expert_index, expert_weight in zip(
            route.expert_indices, route.expert_weights, strict=True
        ):
            expert_output = _gated_mlp(
                weights,
                f"{prefix}.mlp.experts.{expert_index}",
                hidden_state,
                architecture.moe_intermediate_size,
                architecture.hidden_size,
            )
            for index, value in enumerate(expert_output):
                routed[index] += expert_weight * value
        shared = _gated_mlp(
            weights,
            f"{prefix}.mlp.shared_experts",
            hidden_state,
            shared_intermediate,
            architecture.hidden_size,
        )
        outputs.append(tuple(value + shared[index] for index, value in enumerate(routed)))
    return tuple(outputs), tuple(routes)


def run_glm_moe_dsa_prefill_reference(
    architecture: GlmMoeDsaArchitecture,
    weights: GlmWeightAccess,
    input_ids: Sequence[int],
    *,
    enable_mtp: bool = False,
) -> GlmReferenceOutput:
    """Execute a batch-one causal prefill without caches or MTP speculation."""
    if enable_mtp:
        raise AmsError(ErrorCode.UNSUPPORTED_OP, "MTP reference execution is not implemented")
    if not input_ids or len(input_ids) > architecture.max_position_embeddings:
        raise AmsError(ErrorCode.PLAN_INVALID, "input token count is empty or exceeds context")
    hidden_states = tuple(
        weights.embedding("model.embed_tokens.weight", token_id, architecture.hidden_size)
        for token_id in input_ids
    )
    traces: list[GlmReferenceLayerTrace] = []
    previous_indices: tuple[tuple[int, ...], ...] | None = None
    for layer_index in range(architecture.num_hidden_layers):
        prefix = f"model.layers.{layer_index}"
        residual = hidden_states
        input_norm_weight = weights.vector(
            f"{prefix}.input_layernorm.weight", architecture.hidden_size
        )
        normalized = tuple(
            rms_norm_reference(values, input_norm_weight, architecture.rms_norm_eps)
            for values in hidden_states
        )
        attention, previous_indices = _run_attention(
            architecture,
            weights,
            prefix,
            normalized,
            previous_indices,
            architecture.indexer_types[layer_index],
        )
        hidden_states = tuple(
            _add(left, right, name="attention.residual")
            for left, right in zip(residual, attention, strict=True)
        )
        residual = hidden_states
        post_norm_weight = weights.vector(
            f"{prefix}.post_attention_layernorm.weight", architecture.hidden_size
        )
        normalized = tuple(
            rms_norm_reference(values, post_norm_weight, architecture.rms_norm_eps)
            for values in hidden_states
        )
        if architecture.mlp_layer_types[layer_index] == "dense":
            mlp = tuple(
                _gated_mlp(
                    weights,
                    f"{prefix}.mlp",
                    values,
                    architecture.intermediate_size,
                    architecture.hidden_size,
                )
                for values in normalized
            )
            routes: tuple[GlmExpertRouting, ...] = ()
        else:
            mlp, routes = _run_sparse_mlp(architecture, weights, prefix, normalized)
        hidden_states = tuple(
            _add(left, right, name="mlp.residual")
            for left, right in zip(residual, mlp, strict=True)
        )
        traces.append(
            GlmReferenceLayerTrace(
                layer_index=layer_index,
                indexer_type=architecture.indexer_types[layer_index],
                dsa_indices=previous_indices,
                expert_routing=routes,
            )
        )
    final_norm_weight = weights.vector("model.norm.weight", architecture.hidden_size)
    hidden_states = tuple(
        rms_norm_reference(values, final_norm_weight, architecture.rms_norm_eps)
        for values in hidden_states
    )
    logits = tuple(
        weights.linear("lm_head.weight", values, architecture.vocab_size)
        for values in hidden_states
    )
    return GlmReferenceOutput(hidden_states, logits, tuple(traces))
