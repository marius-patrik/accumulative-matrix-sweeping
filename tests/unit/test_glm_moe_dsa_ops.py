import math

import pytest

from ams.errors import AmsError, ErrorCode
from ams.ops import (
    apply_rope_half_split_reference,
    apply_rope_interleaved_reference,
    dsa_topk_reference,
    layer_norm_reference,
    rms_norm_reference,
    route_glm_experts_reference,
    silu_reference,
    softmax_reference,
)


def test_rms_and_layer_norm_use_explicit_reference_reductions() -> None:
    values = (1.0, -2.0, 3.0, -4.0)
    rms = rms_norm_reference(values, (1.0, 0.5, 2.0, -1.0), 1e-5)
    inverse_rms = 1.0 / math.sqrt(sum(value * value for value in values) / 4 + 1e-5)
    assert rms == pytest.approx(
        (
            values[0] * inverse_rms,
            values[1] * inverse_rms * 0.5,
            values[2] * inverse_rms * 2.0,
            values[3] * inverse_rms * -1.0,
        ),
        rel=0,
        abs=1e-15,
    )
    normalized = layer_norm_reference(values, (1.0,) * 4, (0.0,) * 4, 1e-6)
    assert sum(normalized) == pytest.approx(0.0, abs=1e-15)
    assert sum(value * value for value in normalized) / 4 == pytest.approx(1.0, abs=1e-6)


def test_rope_layouts_are_distinct_and_preserve_squared_norm() -> None:
    values = (1.0, 2.0, 3.0, 4.0)
    interleaved = apply_rope_interleaved_reference(values, position=7, theta=10000.0)
    half_split = apply_rope_half_split_reference(values, position=7, theta=10000.0)
    assert interleaved != half_split
    expected_norm = sum(value * value for value in values)
    assert sum(value * value for value in interleaved) == pytest.approx(expected_norm, abs=1e-12)
    assert sum(value * value for value in half_split) == pytest.approx(expected_norm, abs=1e-12)


def test_dsa_topk_is_causal_and_breaks_equal_scores_by_key_index() -> None:
    query_heads = ((1.0, 0.0), (0.0, 1.0))
    keys = (
        (1.0, 1.0),
        (1.0, 1.0),
        (3.0, 0.0),
        (100.0, 100.0),
    )
    selected = dsa_topk_reference(
        query_heads,
        keys,
        (0.5, 0.5),
        query_position=2,
        top_k=3,
    )
    assert selected == (2, 0, 1)
    assert 3 not in selected


def test_noaux_router_applies_group_filter_bias_and_unbiased_output_weights() -> None:
    routing = route_glm_experts_reference(
        (4.0, 3.0, 2.0, 1.0, 5.0, 4.0, -1.0, -2.0),
        (0.0, 0.0, 0.0, 0.0, -10.0, -10.0, 10.0, 10.0),
        experts_per_token=2,
        group_count=2,
        top_groups=1,
        routed_scaling_factor=2.5,
    )
    assert routing.expert_indices == (6, 7)
    expected_raw = (1 / (1 + math.exp(1)), 1 / (1 + math.exp(2)))
    denominator = sum(expected_raw)
    assert routing.expert_weights == pytest.approx(
        tuple(value / denominator * 2.5 for value in expected_raw),
        rel=0,
        abs=1e-15,
    )


def test_scalar_activation_and_softmax_are_stable_at_extreme_values() -> None:
    assert silu_reference(-1000.0) == pytest.approx(0.0, abs=1e-300)
    probabilities = softmax_reference((1000.0, 999.0, -1000.0))
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


@pytest.mark.parametrize(
    "operation",
    [
        lambda: rms_norm_reference((1.0,), (1.0, 2.0), 1e-5),
        lambda: dsa_topk_reference(((1.0,),), ((1.0,),), (1.0,), query_position=2, top_k=1),
        lambda: route_glm_experts_reference(
            (1.0, 2.0),
            (0.0, 0.0),
            experts_per_token=3,
            group_count=1,
            top_groups=1,
            routed_scaling_factor=1.0,
        ),
    ],
)
def test_control_operators_reject_invalid_plans(operation) -> None:
    with pytest.raises(AmsError) as caught:
        operation()
    assert caught.value.code is ErrorCode.PLAN_INVALID
