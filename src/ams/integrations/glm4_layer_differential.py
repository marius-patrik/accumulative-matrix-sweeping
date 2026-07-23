"""Fail-closed evidence contract for one official GLM-4 sparse-layer differential."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import StrEnum

from ams.canonical import canonical_json_bytes
from ams.checked import checked_positive
from ams.descriptors import validate_digest, validate_identifier
from ams.errors import AmsError, ErrorCode

_MAX_SAMPLES = 4096
_MAX_VECTOR_WIDTH = 1_000_000


class Glm4LayerDifferentialStatus(StrEnum):
    """Outcome of the reviewed layer-differential gate."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Glm4LayerDifferentialThresholds:
    """Explicit promotion thresholds for hidden states and teacher-forced logits."""

    minimum_hidden_cosine_similarity: float
    maximum_hidden_normalized_rmse: float
    minimum_top_token_agreement: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_hidden_cosine_similarity,
            self.maximum_hidden_normalized_rmse,
            self.minimum_top_token_agreement,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "GLM-4 layer differential thresholds must be finite numbers",
            )
        if not 0.0 <= self.minimum_hidden_cosine_similarity <= 1.0:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "minimum hidden cosine similarity must be in [0, 1]",
            )
        if self.maximum_hidden_normalized_rmse < 0.0:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "maximum hidden normalized RMSE must be nonnegative",
            )
        if not 0.0 <= self.minimum_top_token_agreement <= 1.0:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "minimum top-token agreement must be in [0, 1]",
            )


OFFICIAL_GLM47_LAYER_THRESHOLDS = Glm4LayerDifferentialThresholds(
    minimum_hidden_cosine_similarity=0.995,
    maximum_hidden_normalized_rmse=0.10,
    minimum_top_token_agreement=0.95,
)


def _normalize_matrix(
    values: tuple[tuple[float, ...], ...],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if not values or len(values) > _MAX_SAMPLES:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            f"{name} sample count is outside the reviewed bound",
        )
    width = len(values[0])
    if width == 0 or width > _MAX_VECTOR_WIDTH:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            f"{name} width is outside the reviewed bound",
        )
    normalized: list[tuple[float, ...]] = []
    for row in values:
        if len(row) != width:
            raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} rows have inconsistent widths")
        if any(isinstance(value, bool) or not isinstance(value, int | float) for value in row):
            raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} contains a nonnumeric value")
        converted = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in converted):
            raise AmsError(ErrorCode.NUMERIC_FAILURE, f"{name} contains a non-finite value")
        normalized.append(converted)
    return tuple(normalized)


def _matrix_hash(values: tuple[tuple[float, ...], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", len(values), len(values[0])))
    for row in values:
        for value in row:
            digest.update(struct.pack("<d", value))
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Glm4LayerObservation:
    """One runtime's outputs for the exact same sparse-layer input corpus."""

    runtime_id: str
    runtime_version: str
    runtime_code_hash: str
    input_hash: str
    sample_ids: tuple[str, ...]
    hidden_states: tuple[tuple[float, ...], ...]
    logits: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.runtime_id, name="glm4_layer_observation.runtime_id")
        if (
            not isinstance(self.runtime_version, str)
            or not self.runtime_version
            or len(self.runtime_version.encode()) > 4096
        ):
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "GLM-4 layer observation runtime version is empty",
            )
        validate_digest(
            self.runtime_code_hash,
            name="glm4_layer_observation.runtime_code_hash",
        )
        validate_digest(self.input_hash, name="glm4_layer_observation.input_hash")
        if (
            not self.sample_ids
            or len(self.sample_ids) > _MAX_SAMPLES
            or len(set(self.sample_ids)) != len(self.sample_ids)
            or any(
                not isinstance(value, str) or not value or len(value.encode()) > 4096
                for value in self.sample_ids
            )
        ):
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "GLM-4 layer observation sample IDs are invalid",
            )
        hidden_states = _normalize_matrix(self.hidden_states, name="hidden states")
        if len(hidden_states) != len(self.sample_ids):
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "GLM-4 layer hidden-state count differs from sample IDs",
            )
        object.__setattr__(self, "hidden_states", hidden_states)
        if self.logits is not None:
            logits = _normalize_matrix(self.logits, name="logits")
            if len(logits) != len(self.sample_ids):
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "GLM-4 layer logit count differs from sample IDs",
                )
            object.__setattr__(self, "logits", logits)

    @property
    def hidden_state_hash(self) -> str:
        """Content hash of the complete normalized hidden-state matrix."""

        return _matrix_hash(self.hidden_states)

    @property
    def logits_hash(self) -> str | None:
        """Content hash of the complete normalized logit matrix, when present."""

        return None if self.logits is None else _matrix_hash(self.logits)

    @property
    def observation_hash(self) -> str:
        """Relocation-independent hash of runtime identity and observed outputs."""

        payload = {
            "runtime_id": self.runtime_id,
            "runtime_version": self.runtime_version,
            "runtime_code_hash": self.runtime_code_hash,
            "input_hash": self.input_hash,
            "sample_ids": self.sample_ids,
            "hidden_state_hash": self.hidden_state_hash,
            "logits_hash": self.logits_hash,
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Glm4LayerDifferentialEvidence:
    """Bounded comparison result; a layer pass never qualifies a precision policy."""

    status: Glm4LayerDifferentialStatus
    sample_count: int
    hidden_size: int
    vocabulary_size: int | None
    hidden_cosine_similarity: float
    hidden_normalized_rmse: float
    top_token_agreement: float | None
    route_agreement: float | None
    hidden_state_gate_passed: bool
    logit_gate_passed: bool
    full_layer_gate_passed: bool
    qualifies_precision_policy: bool
    reference_observation_hash: str
    candidate_observation_hash: str
    thresholds: Glm4LayerDifferentialThresholds
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", Glm4LayerDifferentialStatus(self.status))
        checked_positive(self.sample_count, name="glm4_layer_evidence.sample_count")
        checked_positive(self.hidden_size, name="glm4_layer_evidence.hidden_size")
        if self.vocabulary_size is not None:
            checked_positive(self.vocabulary_size, name="glm4_layer_evidence.vocabulary_size")
        for name, value in (
            ("hidden_cosine_similarity", self.hidden_cosine_similarity),
            ("hidden_normalized_rmse", self.hidden_normalized_rmse),
        ):
            if not math.isfinite(value):
                raise AmsError(ErrorCode.NUMERIC_FAILURE, f"{name} is non-finite")
        for name, value in (
            ("top_token_agreement", self.top_token_agreement),
            ("route_agreement", self.route_agreement),
        ):
            if value is not None and (not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise AmsError(ErrorCode.NUMERIC_FAILURE, f"{name} is outside [0, 1]")
        validate_digest(
            self.reference_observation_hash,
            name="glm4_layer_evidence.reference_observation_hash",
        )
        validate_digest(
            self.candidate_observation_hash,
            name="glm4_layer_evidence.candidate_observation_hash",
        )
        if self.qualifies_precision_policy:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "one layer differential cannot qualify a model precision policy",
            )
        if self.full_layer_gate_passed != (
            self.hidden_state_gate_passed and self.logit_gate_passed
        ):
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "GLM-4 layer aggregate gate disagrees with its component gates",
            )
        if self.status is Glm4LayerDifferentialStatus.PASSED:
            if not self.full_layer_gate_passed or self.blockers:
                raise AmsError(
                    ErrorCode.INTERNAL_INVARIANT,
                    "passed GLM-4 layer evidence still has a failed gate or blocker",
                )
        elif self.status is Glm4LayerDifferentialStatus.BLOCKED:
            if not self.blockers:
                raise AmsError(
                    ErrorCode.INTERNAL_INVARIANT,
                    "blocked GLM-4 layer evidence has no explicit blocker",
                )
        elif self.full_layer_gate_passed:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "failed GLM-4 layer evidence cannot pass the aggregate gate",
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize the stable evidence fields without embedding raw model outputs."""

        return {
            "schema_id": "ams.glm4-layer-differential.v1",
            "status": self.status.value,
            "sample_count": self.sample_count,
            "hidden_size": self.hidden_size,
            "vocabulary_size": self.vocabulary_size,
            "metrics": {
                "hidden_cosine_similarity": self.hidden_cosine_similarity,
                "hidden_normalized_rmse": self.hidden_normalized_rmse,
                "top_token_agreement": self.top_token_agreement,
                "route_agreement": self.route_agreement,
            },
            "gates": {
                "hidden_state_gate_passed": self.hidden_state_gate_passed,
                "logit_gate_passed": self.logit_gate_passed,
                "full_layer_gate_passed": self.full_layer_gate_passed,
                "qualifies_precision_policy": self.qualifies_precision_policy,
            },
            "reference_observation_hash": self.reference_observation_hash,
            "candidate_observation_hash": self.candidate_observation_hash,
            "thresholds": {
                "minimum_hidden_cosine_similarity": (
                    self.thresholds.minimum_hidden_cosine_similarity
                ),
                "maximum_hidden_normalized_rmse": (self.thresholds.maximum_hidden_normalized_rmse),
                "minimum_top_token_agreement": self.thresholds.minimum_top_token_agreement,
            },
            "blockers": list(self.blockers),
        }


def _vector_metrics(
    reference: tuple[tuple[float, ...], ...],
    candidate: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    reference_square = 0.0
    candidate_square = 0.0
    difference_square = 0.0
    dot = 0.0
    count = 0
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        for expected, actual in zip(reference_row, candidate_row, strict=True):
            reference_square += expected * expected
            candidate_square += actual * actual
            difference = actual - expected
            difference_square += difference * difference
            dot += expected * actual
            count += 1
    denominator = math.sqrt(reference_square * candidate_square)
    reference_rms = math.sqrt(reference_square / count)
    if denominator == 0.0 or reference_rms == 0.0:
        raise AmsError(
            ErrorCode.NUMERIC_FAILURE,
            "GLM-4 layer differential cannot normalize a zero-energy reference",
        )
    cosine = max(-1.0, min(1.0, dot / denominator))
    normalized_rmse = math.sqrt(difference_square / count) / reference_rms
    if not math.isfinite(cosine) or not math.isfinite(normalized_rmse):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "GLM-4 layer metrics are non-finite")
    return cosine, normalized_rmse


def _top_token(values: tuple[float, ...]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def compare_glm4_layer_observations(
    reference: Glm4LayerObservation,
    candidate: Glm4LayerObservation,
    *,
    expected_hidden_size: int,
    expected_vocabulary_size: int,
    thresholds: Glm4LayerDifferentialThresholds = OFFICIAL_GLM47_LAYER_THRESHOLDS,
    route_agreement: float | None = None,
    blockers: tuple[str, ...] = (),
) -> Glm4LayerDifferentialEvidence:
    """Compare exact-corpus observations and refuse a full pass without logits."""

    checked_positive(expected_hidden_size, name="glm4_layer.expected_hidden_size")
    checked_positive(expected_vocabulary_size, name="glm4_layer.expected_vocabulary_size")
    if reference.input_hash != candidate.input_hash or reference.sample_ids != candidate.sample_ids:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 layer observations do not share one exact input corpus",
        )
    if (
        len(reference.hidden_states[0]) != expected_hidden_size
        or len(candidate.hidden_states[0]) != expected_hidden_size
        or any(len(row) != expected_hidden_size for row in reference.hidden_states)
        or any(len(row) != expected_hidden_size for row in candidate.hidden_states)
    ):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 layer observation hidden width differs from the architecture",
        )
    cosine, normalized_rmse = _vector_metrics(
        reference.hidden_states,
        candidate.hidden_states,
    )
    hidden_passed = (
        cosine >= thresholds.minimum_hidden_cosine_similarity
        and normalized_rmse <= thresholds.maximum_hidden_normalized_rmse
    )

    agreement: float | None = None
    vocabulary_size: int | None = None
    if (reference.logits is None) != (candidate.logits is None):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "only one GLM-4 layer observation contains logits",
        )
    if reference.logits is not None and candidate.logits is not None:
        if (
            len(reference.logits[0]) != expected_vocabulary_size
            or len(candidate.logits[0]) != expected_vocabulary_size
            or any(len(row) != expected_vocabulary_size for row in reference.logits)
            or any(len(row) != expected_vocabulary_size for row in candidate.logits)
        ):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "GLM-4 layer observation vocabulary differs from the architecture",
            )
        matches = sum(
            _top_token(expected) == _top_token(actual)
            for expected, actual in zip(reference.logits, candidate.logits, strict=True)
        )
        agreement = matches / len(reference.logits)
        vocabulary_size = expected_vocabulary_size

    logit_passed = agreement is not None and agreement >= thresholds.minimum_top_token_agreement
    full_passed = hidden_passed and logit_passed
    normalized_blockers = tuple(blockers)
    if agreement is None:
        normalized_blockers += ("teacher-forced logits are absent from both observations",)
    if normalized_blockers:
        status = Glm4LayerDifferentialStatus.BLOCKED
    elif full_passed:
        status = Glm4LayerDifferentialStatus.PASSED
    else:
        status = Glm4LayerDifferentialStatus.FAILED
    return Glm4LayerDifferentialEvidence(
        status=status,
        sample_count=len(reference.sample_ids),
        hidden_size=expected_hidden_size,
        vocabulary_size=vocabulary_size,
        hidden_cosine_similarity=cosine,
        hidden_normalized_rmse=normalized_rmse,
        top_token_agreement=agreement,
        route_agreement=route_agreement,
        hidden_state_gate_passed=hidden_passed,
        logit_gate_passed=logit_passed,
        full_layer_gate_passed=full_passed,
        qualifies_precision_policy=False,
        reference_observation_hash=reference.observation_hash,
        candidate_observation_hash=candidate.observation_hash,
        thresholds=thresholds,
        blockers=normalized_blockers,
    )
