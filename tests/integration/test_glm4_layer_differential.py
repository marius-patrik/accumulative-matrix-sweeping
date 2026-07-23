import hashlib

import pytest

from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4LayerDifferentialStatus,
    Glm4LayerObservation,
    compare_glm4_layer_observations,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _observation(
    runtime: str,
    hidden_states: tuple[tuple[float, ...], ...],
    logits: tuple[tuple[float, ...], ...] | None,
    *,
    input_hash: str | None = None,
) -> Glm4LayerObservation:
    return Glm4LayerObservation(
        runtime_id=runtime,
        runtime_version="1.0.0",
        runtime_code_hash=_digest(runtime),
        input_hash=input_hash or _digest("input"),
        sample_ids=("token-0", "token-1"),
        hidden_states=hidden_states,
        logits=logits,
    )


def test_exact_hidden_and_logit_observations_pass_only_the_layer_gate() -> None:
    reference = _observation(
        "official-reference",
        ((1.0, 2.0), (3.0, 4.0)),
        ((0.0, 2.0, 1.0), (4.0, 3.0, 2.0)),
    )
    candidate = _observation(
        "ams-candidate",
        ((1.0, 2.0), (3.0, 4.0)),
        ((0.0, 2.0, 1.0), (4.0, 3.0, 2.0)),
    )

    evidence = compare_glm4_layer_observations(
        reference,
        candidate,
        expected_hidden_size=2,
        expected_vocabulary_size=3,
        route_agreement=1.0,
    )

    assert evidence.status is Glm4LayerDifferentialStatus.PASSED
    assert evidence.hidden_cosine_similarity == 1.0
    assert evidence.hidden_normalized_rmse == 0.0
    assert evidence.top_token_agreement == 1.0
    assert evidence.full_layer_gate_passed
    assert not evidence.qualifies_precision_policy
    assert evidence.blockers == ()


def test_hidden_only_differential_is_explicitly_blocked_from_promotion() -> None:
    reference = _observation(
        "official-reference",
        ((1.0, 2.0), (3.0, 4.0)),
        None,
    )
    candidate = _observation(
        "ams-candidate",
        ((1.0, 2.0), (3.0, 4.0)),
        None,
    )

    evidence = compare_glm4_layer_observations(
        reference,
        candidate,
        expected_hidden_size=2,
        expected_vocabulary_size=3,
        blockers=("native official-layer observation is absent",),
    )

    assert evidence.status is Glm4LayerDifferentialStatus.BLOCKED
    assert evidence.hidden_state_gate_passed
    assert not evidence.logit_gate_passed
    assert not evidence.full_layer_gate_passed
    assert not evidence.qualifies_precision_policy
    assert evidence.blockers == (
        "native official-layer observation is absent",
        "teacher-forced logits are absent from both observations",
    )


def test_threshold_failure_and_corpus_drift_fail_closed() -> None:
    reference = _observation(
        "official-reference",
        ((1.0, 0.0), (0.0, 1.0)),
        ((3.0, 2.0, 1.0), (1.0, 3.0, 2.0)),
    )
    inaccurate = _observation(
        "ams-candidate",
        ((0.0, 1.0), (1.0, 0.0)),
        ((1.0, 2.0, 3.0), (3.0, 2.0, 1.0)),
    )

    evidence = compare_glm4_layer_observations(
        reference,
        inaccurate,
        expected_hidden_size=2,
        expected_vocabulary_size=3,
    )
    assert evidence.status is Glm4LayerDifferentialStatus.FAILED
    assert not evidence.hidden_state_gate_passed
    assert not evidence.logit_gate_passed
    assert not evidence.full_layer_gate_passed

    drifted = _observation(
        "ams-candidate",
        ((1.0, 0.0), (0.0, 1.0)),
        ((3.0, 2.0, 1.0), (1.0, 3.0, 2.0)),
        input_hash=_digest("different-input"),
    )
    with pytest.raises(AmsError, match="input corpus") as caught:
        compare_glm4_layer_observations(
            reference,
            drifted,
            expected_hidden_size=2,
            expected_vocabulary_size=3,
        )
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
