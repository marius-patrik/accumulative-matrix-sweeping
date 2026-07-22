"""Deterministic scalar semantic oracles for GLM-MoE-DSA control operators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ams.checked import checked_positive
from ams.errors import AmsError, ErrorCode


def _finite_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if not values:
        raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must not be empty")
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in normalized):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, f"{name} contains a non-finite value")
    return normalized


def rms_norm_reference(
    values: Sequence[float],
    weight: Sequence[float],
    epsilon: float,
) -> tuple[float, ...]:
    """Apply GLM RMSNorm using a fixed left-to-right square reduction."""
    values = _finite_values(values, name="rms_norm.values")
    weight = _finite_values(weight, name="rms_norm.weight")
    if len(values) != len(weight):
        raise AmsError(ErrorCode.PLAN_INVALID, "RMSNorm value and weight lengths differ")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "RMSNorm epsilon must be finite and positive")
    square_sum = 0.0
    for value in values:
        square_sum += value * value
    inverse_root_mean_square = 1.0 / math.sqrt(square_sum / len(values) + epsilon)
    return tuple(
        value * inverse_root_mean_square * scale
        for value, scale in zip(values, weight, strict=True)
    )


def layer_norm_reference(
    values: Sequence[float],
    weight: Sequence[float],
    bias: Sequence[float],
    epsilon: float,
) -> tuple[float, ...]:
    """Apply the DSA indexer LayerNorm with fixed two-pass reductions."""
    values = _finite_values(values, name="layer_norm.values")
    weight = _finite_values(weight, name="layer_norm.weight")
    bias = _finite_values(bias, name="layer_norm.bias")
    if len(values) != len(weight) or len(values) != len(bias):
        raise AmsError(ErrorCode.PLAN_INVALID, "LayerNorm vector lengths differ")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "LayerNorm epsilon must be finite and positive")
    value_sum = 0.0
    for value in values:
        value_sum += value
    mean = value_sum / len(values)
    square_deviation_sum = 0.0
    for value in values:
        deviation = value - mean
        square_deviation_sum += deviation * deviation
    inverse_standard_deviation = 1.0 / math.sqrt(square_deviation_sum / len(values) + epsilon)
    return tuple(
        (value - mean) * inverse_standard_deviation * scale + offset
        for value, scale, offset in zip(values, weight, bias, strict=True)
    )


def silu_reference(value: float) -> float:
    """Evaluate SiLU without overflowing for large negative inputs."""
    value = float(value)
    if not math.isfinite(value):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "SiLU input is non-finite")
    if value >= 0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def softmax_reference(values: Sequence[float]) -> tuple[float, ...]:
    """Apply max-shifted softmax in declared left-to-right order."""
    values = _finite_values(values, name="softmax.values")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = 0.0
    for value in exponentials:
        denominator += value
    if not math.isfinite(denominator) or denominator <= 0:
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "softmax denominator is invalid")
    return tuple(value / denominator for value in exponentials)


def apply_rope_interleaved_reference(
    values: Sequence[float],
    *,
    position: int,
    theta: float,
) -> tuple[float, ...]:
    """Apply the main MLA interleaved-pair rotary layout."""
    values = _finite_values(values, name="rope_interleaved.values")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise AmsError(ErrorCode.PLAN_INVALID, "RoPE theta must be numeric")
    theta = float(theta)
    if not math.isfinite(theta) or theta <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "RoPE theta must be finite and positive")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "RoPE position must be a nonnegative integer")
    if len(values) % 2:
        raise AmsError(ErrorCode.PLAN_INVALID, "interleaved RoPE dimension must be even")
    output: list[float] = []
    for pair_index in range(len(values) // 2):
        angle = position / (theta ** (2 * pair_index / len(values)))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        left = values[2 * pair_index]
        right = values[2 * pair_index + 1]
        output.extend((left * cosine - right * sine, right * cosine + left * sine))
    return tuple(output)


def apply_rope_half_split_reference(
    values: Sequence[float],
    *,
    position: int,
    theta: float,
) -> tuple[float, ...]:
    """Apply the DSA indexer half-split rotary layout."""
    values = _finite_values(values, name="rope_half_split.values")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise AmsError(ErrorCode.PLAN_INVALID, "RoPE theta must be numeric")
    theta = float(theta)
    if not math.isfinite(theta) or theta <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "RoPE theta must be finite and positive")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "RoPE position must be a nonnegative integer")
    if len(values) % 2:
        raise AmsError(ErrorCode.PLAN_INVALID, "half-split RoPE dimension must be even")
    half = len(values) // 2
    first: list[float] = []
    second: list[float] = []
    for index in range(half):
        angle = position / (theta ** (2 * index / len(values)))
        cosine = math.cos(angle)
        sine = math.sin(angle)
        left = values[index]
        right = values[half + index]
        first.append(left * cosine - right * sine)
        second.append(right * cosine + left * sine)
    return tuple(first + second)


def _dot(left: Sequence[float], right: Sequence[float], *, name: str) -> float:
    if len(left) != len(right) or not left:
        raise AmsError(ErrorCode.PLAN_INVALID, f"{name} vector lengths differ or are empty")
    accumulator = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        left_float = float(left_value)
        right_float = float(right_value)
        if not math.isfinite(left_float) or not math.isfinite(right_float):
            raise AmsError(ErrorCode.NUMERIC_FAILURE, f"{name} contains a non-finite value")
        accumulator += left_float * right_float
    return accumulator


def dsa_topk_reference(
    query_heads: Sequence[Sequence[float]],
    key_vectors: Sequence[Sequence[float]],
    head_weights: Sequence[float],
    *,
    query_position: int,
    top_k: int,
) -> tuple[int, ...]:
    """Rank causal DSA keys with deterministic score and index tie-breaking."""
    checked_positive(top_k, name="dsa.top_k")
    if (
        isinstance(query_position, bool)
        or not isinstance(query_position, int)
        or query_position < 0
    ):
        raise AmsError(ErrorCode.PLAN_INVALID, "DSA query position must be nonnegative")
    if not query_heads or len(query_heads) != len(head_weights) or not key_vectors:
        raise AmsError(ErrorCode.PLAN_INVALID, "DSA head or key dimensions are invalid")
    head_dimension = len(query_heads[0])
    if head_dimension == 0 or any(len(head) != head_dimension for head in query_heads):
        raise AmsError(ErrorCode.PLAN_INVALID, "DSA query head dimensions differ")
    if any(len(key) != head_dimension for key in key_vectors):
        raise AmsError(ErrorCode.PLAN_INVALID, "DSA key dimension differs from query heads")
    if query_position >= len(key_vectors):
        raise AmsError(ErrorCode.PLAN_INVALID, "DSA query position exceeds available keys")
    weights = _finite_values(head_weights, name="dsa.head_weights")
    scale = head_dimension**-0.5
    scores: list[tuple[int, float]] = []
    for key_index in range(query_position + 1):
        score = 0.0
        for head_index, query in enumerate(query_heads):
            similarity = max(
                _dot(query, key_vectors[key_index], name="dsa.similarity") * scale,
                0.0,
            )
            score += weights[head_index] * similarity
        if not math.isfinite(score):
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "DSA index score is non-finite")
        scores.append((key_index, score))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return tuple(index for index, _ in scores[: min(top_k, len(scores))])


@dataclass(frozen=True, slots=True)
class GlmExpertRouting:
    expert_indices: tuple[int, ...]
    expert_weights: tuple[float, ...]


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def route_glm_experts_reference(
    router_logits: Sequence[float],
    correction_bias: Sequence[float],
    *,
    experts_per_token: int,
    group_count: int,
    top_groups: int,
    routed_scaling_factor: float,
) -> GlmExpertRouting:
    """Implement sigmoid/noaux_tc GLM routing with deterministic top-k ties."""
    logits = _finite_values(router_logits, name="router.logits")
    bias = _finite_values(correction_bias, name="router.correction_bias")
    checked_positive(experts_per_token, name="router.experts_per_token")
    checked_positive(group_count, name="router.group_count")
    checked_positive(top_groups, name="router.top_groups")
    if len(logits) != len(bias) or len(logits) % group_count:
        raise AmsError(ErrorCode.PLAN_INVALID, "router expert and group dimensions are invalid")
    if experts_per_token > len(logits) or top_groups > group_count:
        raise AmsError(ErrorCode.PLAN_INVALID, "router top-k exceeds its candidate dimensions")
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "router scaling factor must be finite and positive")
    experts_per_group = len(logits) // group_count
    if experts_per_token > top_groups * experts_per_group:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "router selected groups cannot contain the requested experts",
        )
    probabilities = tuple(_sigmoid(value) for value in logits)
    corrected = tuple(
        probability + offset for probability, offset in zip(probabilities, bias, strict=True)
    )
    group_scores: list[tuple[int, float]] = []
    for group_index in range(group_count):
        start = group_index * experts_per_group
        group_values = sorted(corrected[start : start + experts_per_group], reverse=True)
        group_scores.append((group_index, sum(group_values[: min(2, len(group_values))])))
    group_scores.sort(key=lambda item: (-item[1], item[0]))
    selected_groups = {group_index for group_index, _ in group_scores[:top_groups]}
    candidates = [
        expert_index
        for expert_index in range(len(logits))
        if expert_index // experts_per_group in selected_groups
    ]
    candidates.sort(key=lambda expert_index: (-corrected[expert_index], expert_index))
    selected = tuple(candidates[:experts_per_token])
    denominator = 0.0
    for expert_index in selected:
        denominator += probabilities[expert_index]
    if not math.isfinite(denominator) or denominator <= 0:
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "router weight denominator is invalid")
    weights = tuple(
        probabilities[expert_index] / denominator * routed_scaling_factor
        for expert_index in selected
    )
    return GlmExpertRouting(selected, weights)
