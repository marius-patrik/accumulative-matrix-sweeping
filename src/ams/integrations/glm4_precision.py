"""Experimental GLM-4 mixed-precision policy and evidence-gated qualification."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_mul, checked_positive, checked_product
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.descriptors import DType, validate_digest
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_moe_lite import (
    Glm4MoeLiteArchitecture,
    Glm4MoeLiteTensorInventory,
    Glm4MoeLiteTensorRole,
    expected_glm4_moe_lite_tensor_shape,
    expected_glm4_moe_lite_tensor_slots,
)
from ams.integrations.huggingface import (
    HuggingFaceCatalogTensor,
    HuggingFaceMixedPolicy,
    HuggingFaceTensorAssignment,
    HuggingFaceTensorEncoding,
    build_huggingface_mixed_policy,
)

_SOURCE_DTYPES = {
    "F16": (DType.FLOAT16, 2),
    "BF16": (DType.BFLOAT16, 2),
    "F32": (DType.FLOAT32, 4),
}

_IDENTITY_ROLES = {
    Glm4MoeLiteTensorRole.EMBEDDING,
    Glm4MoeLiteTensorRole.FINAL_NORM,
    Glm4MoeLiteTensorRole.INPUT_NORM,
    Glm4MoeLiteTensorRole.POST_ATTENTION_NORM,
    Glm4MoeLiteTensorRole.ATTENTION_Q_A_NORM,
    Glm4MoeLiteTensorRole.ATTENTION_KV_A_NORM,
    Glm4MoeLiteTensorRole.ROUTER_WEIGHT,
    Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS,
    Glm4MoeLiteTensorRole.MTP_EMBED_NORM,
    Glm4MoeLiteTensorRole.MTP_HIDDEN_NORM,
    Glm4MoeLiteTensorRole.MTP_EMBEDDING,
    Glm4MoeLiteTensorRole.MTP_SHARED_HEAD_NORM,
}

_TERNARY_ROLES = {
    Glm4MoeLiteTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
    Glm4MoeLiteTensorRole.ROUTED_EXPERT_UP_PROJECTION,
    Glm4MoeLiteTensorRole.ROUTED_EXPERT_DOWN_PROJECTION,
}


class Glm4PrecisionCandidateStatus(StrEnum):
    """A candidate cannot become deployable merely because it was constructed."""

    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class Glm4PrecisionCandidate:
    candidate_hash: str
    architecture_hash: str
    source_index_hash: str
    policy: HuggingFaceMixedPolicy
    encoding_counts: tuple[tuple[HuggingFaceTensorEncoding, int], ...]
    source_bytes: int
    estimated_encoded_bytes: int
    status: Glm4PrecisionCandidateStatus = Glm4PrecisionCandidateStatus.EXPERIMENTAL

    @property
    def assignments(self) -> tuple[HuggingFaceTensorAssignment, ...]:
        return self.policy.assignments


@dataclass(frozen=True, slots=True)
class Glm4PrecisionQualityThresholds:
    """Explicit acceptance criteria derived from a separately recorded baseline."""

    minimum_evaluated_tokens: int
    minimum_evaluated_tasks: int
    maximum_mean_token_nll_delta: float
    minimum_top1_token_agreement: float
    minimum_task_score_retention: float

    def __post_init__(self) -> None:
        checked_positive(
            self.minimum_evaluated_tokens,
            name="glm4_precision.minimum_evaluated_tokens",
        )
        checked_positive(
            self.minimum_evaluated_tasks,
            name="glm4_precision.minimum_evaluated_tasks",
        )
        _validate_nonnegative_finite(
            self.maximum_mean_token_nll_delta,
            name="maximum_mean_token_nll_delta",
        )
        _validate_unit_interval(
            self.minimum_top1_token_agreement,
            name="minimum_top1_token_agreement",
        )
        _validate_unit_interval(
            self.minimum_task_score_retention,
            name="minimum_task_score_retention",
        )


@dataclass(frozen=True, slots=True)
class Glm4PrecisionQualityEvidence:
    """Reproducible comparison of one candidate against one trusted baseline."""

    candidate_hash: str
    source_index_hash: str
    calibration_corpus_hash: str
    evaluation_corpus_hash: str
    evaluator_hash: str
    trusted_baseline_hash: str
    candidate_runtime_hash: str
    evaluated_tokens: int
    evaluated_tasks: int
    mean_token_nll_delta: float
    top1_token_agreement: float
    task_score_retention: float

    def __post_init__(self) -> None:
        for name in (
            "candidate_hash",
            "source_index_hash",
            "calibration_corpus_hash",
            "evaluation_corpus_hash",
            "evaluator_hash",
            "trusted_baseline_hash",
            "candidate_runtime_hash",
        ):
            validate_digest(getattr(self, name), name=f"glm4_precision.{name}")
        checked_positive(self.evaluated_tokens, name="glm4_precision.evaluated_tokens")
        checked_positive(self.evaluated_tasks, name="glm4_precision.evaluated_tasks")
        _validate_nonnegative_finite(
            self.mean_token_nll_delta,
            name="mean_token_nll_delta",
        )
        _validate_unit_interval(self.top1_token_agreement, name="top1_token_agreement")
        _validate_unit_interval(self.task_score_retention, name="task_score_retention")


@dataclass(frozen=True, slots=True)
class Glm4QualifiedPrecisionPolicy:
    """Evidence that one immutable experimental candidate cleared explicit thresholds."""

    qualification_hash: str
    candidate: Glm4PrecisionCandidate
    thresholds: Glm4PrecisionQualityThresholds
    evidence: Glm4PrecisionQualityEvidence


def _validate_nonnegative_finite(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be finite and nonnegative")


def _validate_unit_interval(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be numeric")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be in [0, 1]")


def experimental_glm4_encoding_for_role(
    role: Glm4MoeLiteTensorRole,
) -> HuggingFaceTensorEncoding:
    """Return the exact reviewed role assignment for the experimental candidate."""
    try:
        role = Glm4MoeLiteTensorRole(role)
    except ValueError as exc:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unreviewed GLM-4 precision role") from exc
    if role in _IDENTITY_ROLES:
        return HuggingFaceTensorEncoding.IDENTITY
    if role in _TERNARY_ROLES:
        return HuggingFaceTensorEncoding.TERNARY_TRIT5
    if role in set(Glm4MoeLiteTensorRole):
        return HuggingFaceTensorEncoding.INT4_SYMMETRIC
    raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unreviewed GLM-4 precision role")


def _validated_catalog_by_name(
    architecture: Glm4MoeLiteArchitecture,
    inventory: Glm4MoeLiteTensorInventory,
    tensors: tuple[HuggingFaceCatalogTensor, ...],
) -> dict[str, HuggingFaceCatalogTensor]:
    expected_slots = expected_glm4_moe_lite_tensor_slots(architecture)
    if (
        inventory.architecture_hash != architecture.content_hash
        or inventory.slots != expected_slots
    ):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 precision inventory does not match the reviewed architecture",
        )
    tensor_by_name = {tensor.tensor_name: tensor for tensor in tensors}
    if len(tensor_by_name) != len(tensors) or set(tensor_by_name) != {
        slot.tensor_name for slot in expected_slots
    }:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 precision catalog does not match the reviewed inventory",
        )
    for slot in expected_slots:
        tensor = tensor_by_name[slot.tensor_name]
        expected_source = _SOURCE_DTYPES.get(tensor.source_dtype)
        if expected_source is None or expected_source[0] is not tensor.dtype:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"GLM-4 tensor source dtype is unsupported or inconsistent: {slot.tensor_name}",
            )
        expected_shape = expected_glm4_moe_lite_tensor_shape(architecture, slot)
        element_count = checked_product(expected_shape, name="glm4_precision.elements")
        expected_bytes = checked_mul(
            element_count,
            expected_source[1],
            name="glm4_precision.source_bytes",
        )
        if tensor.shape != expected_shape or tensor.source_length != expected_bytes:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"GLM-4 tensor shape or source length is inconsistent: {slot.tensor_name}",
            )
    return tensor_by_name


def build_experimental_glm4_precision_candidate(
    architecture: Glm4MoeLiteArchitecture,
    inventory: Glm4MoeLiteTensorInventory,
    tensors: tuple[HuggingFaceCatalogTensor, ...],
    *,
    ternary_config: TernaryCodecConfig,
    int4_config: Int4CodecConfig,
) -> Glm4PrecisionCandidate:
    """Build the reviewed metadata-only candidate without reading tensor payloads."""
    tensor_by_name = _validated_catalog_by_name(architecture, inventory, tensors)
    assignments: list[HuggingFaceTensorAssignment] = []
    source_bytes = 0
    estimated_encoded_bytes = 0
    encoding_counts = {encoding: 0 for encoding in HuggingFaceTensorEncoding}
    for slot in inventory.slots:
        tensor = tensor_by_name[slot.tensor_name]
        encoding = experimental_glm4_encoding_for_role(slot.role)
        element_count = checked_product(tensor.shape, name="glm4_precision.elements")
        if encoding is HuggingFaceTensorEncoding.IDENTITY:
            assignment = HuggingFaceTensorAssignment(slot.tensor_name, encoding)
            encoded_bytes = tensor.source_length
        elif encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
            assignment = HuggingFaceTensorAssignment(
                slot.tensor_name,
                encoding,
                ternary_config=ternary_config,
            )
            encoded_bytes = ternary_config.encoded_size(element_count)
        else:
            assignment = HuggingFaceTensorAssignment(
                slot.tensor_name,
                encoding,
                int4_config=int4_config,
            )
            encoded_bytes = int4_config.encoded_size(element_count)
        assignments.append(assignment)
        encoding_counts[encoding] += 1
        source_bytes = checked_add(
            source_bytes,
            tensor.source_length,
            name="glm4_precision.total_source_bytes",
        )
        estimated_encoded_bytes = checked_add(
            estimated_encoded_bytes,
            encoded_bytes,
            name="glm4_precision.total_encoded_bytes",
        )
    policy = build_huggingface_mixed_policy(tensors, tuple(assignments))
    normalized_counts = tuple(
        (encoding, encoding_counts[encoding])
        for encoding in HuggingFaceTensorEncoding
        if encoding_counts[encoding]
    )
    candidate_payload = {
        "architecture_hash": architecture.content_hash,
        "encoding_counts": [
            {"encoding": encoding.value, "tensor_count": count}
            for encoding, count in normalized_counts
        ],
        "estimated_encoded_bytes": estimated_encoded_bytes,
        "policy_hash": policy.policy_hash,
        "source_bytes": source_bytes,
        "source_index_hash": inventory.index_hash,
        "status": Glm4PrecisionCandidateStatus.EXPERIMENTAL.value,
    }
    candidate_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(candidate_payload)).hexdigest()
    return Glm4PrecisionCandidate(
        candidate_hash=candidate_hash,
        architecture_hash=architecture.content_hash,
        source_index_hash=inventory.index_hash,
        policy=policy,
        encoding_counts=normalized_counts,
        source_bytes=source_bytes,
        estimated_encoded_bytes=estimated_encoded_bytes,
    )


def qualify_glm4_precision_candidate(
    candidate: Glm4PrecisionCandidate,
    thresholds: Glm4PrecisionQualityThresholds,
    evidence: Glm4PrecisionQualityEvidence,
) -> Glm4QualifiedPrecisionPolicy:
    """Fail closed unless matching evidence clears every caller-supplied threshold."""
    if (
        candidate.status is not Glm4PrecisionCandidateStatus.EXPERIMENTAL
        or evidence.candidate_hash != candidate.candidate_hash
        or evidence.source_index_hash != candidate.source_index_hash
    ):
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "GLM-4 precision evidence does not identify this exact candidate",
        )
    failures = []
    if evidence.evaluated_tokens < thresholds.minimum_evaluated_tokens:
        failures.append("evaluated_tokens")
    if evidence.evaluated_tasks < thresholds.minimum_evaluated_tasks:
        failures.append("evaluated_tasks")
    if evidence.mean_token_nll_delta > thresholds.maximum_mean_token_nll_delta:
        failures.append("mean_token_nll_delta")
    if evidence.top1_token_agreement < thresholds.minimum_top1_token_agreement:
        failures.append("top1_token_agreement")
    if evidence.task_score_retention < thresholds.minimum_task_score_retention:
        failures.append("task_score_retention")
    if failures:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 precision candidate did not clear the declared quality thresholds",
            evidence={"failed_metrics": ",".join(failures)},
        )
    qualification_payload = {
        "candidate_hash": candidate.candidate_hash,
        "evidence": {name: getattr(evidence, name) for name in evidence.__dataclass_fields__},
        "thresholds": {name: getattr(thresholds, name) for name in thresholds.__dataclass_fields__},
    }
    qualification_hash = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(qualification_payload)).hexdigest()
    )
    return Glm4QualifiedPrecisionPolicy(
        qualification_hash=qualification_hash,
        candidate=candidate,
        thresholds=thresholds,
        evidence=evidence,
    )
