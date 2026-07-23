"""Experimental GLM-MoE-DSA mixed-precision policy construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_product
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm_moe_dsa import (
    GlmMoeDsaArchitecture,
    GlmTensorInventory,
    GlmTensorRole,
    validate_glm_tensor_catalog,
)
from ams.integrations.huggingface import (
    HuggingFaceCatalogTensor,
    HuggingFaceMixedPolicy,
    HuggingFaceTensorAssignment,
    HuggingFaceTensorEncoding,
    build_huggingface_mixed_policy,
)

_IDENTITY_ROLES = {
    GlmTensorRole.FINAL_NORM,
    GlmTensorRole.INPUT_NORM,
    GlmTensorRole.POST_ATTENTION_NORM,
    GlmTensorRole.ATTENTION_Q_A_NORM,
    GlmTensorRole.ATTENTION_KV_A_NORM,
    GlmTensorRole.INDEXER_WQ_B_PROJECTION,
    GlmTensorRole.INDEXER_WK_PROJECTION,
    GlmTensorRole.INDEXER_K_NORM_WEIGHT,
    GlmTensorRole.INDEXER_K_NORM_BIAS,
    GlmTensorRole.INDEXER_WEIGHTS_PROJECTION,
    GlmTensorRole.ROUTER_WEIGHT,
    GlmTensorRole.ROUTER_CORRECTION_BIAS,
    GlmTensorRole.MTP_EMBED_NORM,
    GlmTensorRole.MTP_HIDDEN_NORM,
    GlmTensorRole.MTP_SHARED_HEAD_NORM,
}

_TERNARY_ROLES = {
    GlmTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
    GlmTensorRole.ROUTED_EXPERT_UP_PROJECTION,
    GlmTensorRole.ROUTED_EXPERT_DOWN_PROJECTION,
}


class GlmPrecisionCandidateStatus(StrEnum):
    """A structural candidate is never deployable without separate quality evidence."""

    EXPERIMENTAL = "experimental"


@dataclass(frozen=True, slots=True)
class GlmPrecisionCandidate:
    candidate_hash: str
    architecture_hash: str
    source_index_hash: str
    policy: HuggingFaceMixedPolicy
    encoding_counts: tuple[tuple[HuggingFaceTensorEncoding, int], ...]
    source_bytes: int
    estimated_encoded_bytes: int
    status: GlmPrecisionCandidateStatus = GlmPrecisionCandidateStatus.EXPERIMENTAL

    @property
    def assignments(self) -> tuple[HuggingFaceTensorAssignment, ...]:
        return self.policy.assignments


def experimental_glm_encoding_for_role(
    role: GlmTensorRole,
) -> HuggingFaceTensorEncoding:
    """Return the first storage-feasibility assignment, not a quality recommendation."""

    try:
        role = GlmTensorRole(role)
    except ValueError as exc:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unreviewed GLM precision role") from exc
    if role in _IDENTITY_ROLES:
        return HuggingFaceTensorEncoding.IDENTITY
    if role in _TERNARY_ROLES:
        return HuggingFaceTensorEncoding.TERNARY_TRIT5
    if role in set(GlmTensorRole):
        return HuggingFaceTensorEncoding.INT4_SYMMETRIC
    raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unreviewed GLM precision role")


def build_experimental_glm_precision_candidate(
    architecture: GlmMoeDsaArchitecture,
    inventory: GlmTensorInventory,
    tensors: tuple[HuggingFaceCatalogTensor, ...],
    *,
    ternary_config: TernaryCodecConfig,
    int4_config: Int4CodecConfig,
) -> GlmPrecisionCandidate:
    """Build a deterministic metadata-only GLM candidate after exact catalog validation."""

    validate_glm_tensor_catalog(architecture, inventory, tensors)
    tensor_by_name = {tensor.tensor_name: tensor for tensor in tensors}
    assignments: list[HuggingFaceTensorAssignment] = []
    source_bytes = 0
    estimated_encoded_bytes = 0
    encoding_counts = {encoding: 0 for encoding in HuggingFaceTensorEncoding}
    for slot in inventory.slots:
        tensor = tensor_by_name[slot.tensor_name]
        encoding = experimental_glm_encoding_for_role(slot.role)
        element_count = checked_product(tensor.shape, name="glm_precision.elements")
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
        elif encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC:
            assignment = HuggingFaceTensorAssignment(
                slot.tensor_name,
                encoding,
                int4_config=int4_config,
            )
            encoded_bytes = int4_config.encoded_size(element_count)
        else:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unreviewed GLM tensor encoding")
        assignments.append(assignment)
        encoding_counts[encoding] += 1
        source_bytes = checked_add(
            source_bytes,
            tensor.source_length,
            name="glm_precision.total_source_bytes",
        )
        estimated_encoded_bytes = checked_add(
            estimated_encoded_bytes,
            encoded_bytes,
            name="glm_precision.total_encoded_bytes",
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
        "status": GlmPrecisionCandidateStatus.EXPERIMENTAL.value,
    }
    candidate_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(candidate_payload)).hexdigest()
    return GlmPrecisionCandidate(
        candidate_hash=candidate_hash,
        architecture_hash=architecture.content_hash,
        source_index_hash=inventory.index_hash,
        policy=policy,
        encoding_counts=normalized_counts,
        source_bytes=source_bytes,
        estimated_encoded_bytes=estimated_encoded_bytes,
    )
