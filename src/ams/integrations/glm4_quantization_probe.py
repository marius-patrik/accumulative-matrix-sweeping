"""Bounded diagnostic sampling of the experimental GLM-4 precision policy."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import StrEnum

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_mul, checked_positive, checked_product
from ams.codecs import (
    Int4CodecConfig,
    TernaryCodecConfig,
    decode_int4_group_reference,
    decode_ternary_group_reference,
    encode_int4_group_reference,
    encode_ternary_group_reference,
)
from ams.descriptors import DType, validate_digest, validate_identifier
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_moe_lite import (
    Glm4MoeLiteArchitecture,
    Glm4MoeLiteTensorInventory,
    Glm4MoeLiteTensorSlot,
    expected_glm4_moe_lite_tensor_shape,
    validate_glm4_moe_lite_tensor_inventory,
)
from ams.integrations.glm4_precision import experimental_glm4_encoding_for_role
from ams.integrations.huggingface import (
    HuggingFaceShardIndex,
    HuggingFaceTensorEncoding,
)
from ams.integrations.safetensors import SafetensorsTensor, parse_safetensors_header
from ams.storage import RangeReader, hash_reader_range

_SOURCE_ITEM_BYTES = {
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.FLOAT32: 4,
}
_SAMPLING_STRATEGY = "evenly_spaced_group_indices_v1"


class Glm4QuantizationProbeStatus(StrEnum):
    """Tensor-error sampling is diagnostic evidence, never a quality qualification."""

    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True, slots=True)
class Glm4QuantizationProbeConfig:
    """Hard bounds and deterministic selection rules for one shard probe."""

    groups_per_tensor: int = 64
    hash_buffer_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        checked_positive(self.groups_per_tensor, name="glm4_probe.groups_per_tensor")
        checked_positive(self.hash_buffer_bytes, name="glm4_probe.hash_buffer_bytes")
        if self.groups_per_tensor > 4096:
            raise AmsError(ErrorCode.PLAN_INVALID, "GLM-4 probe groups per tensor exceeds 4096")
        if self.hash_buffer_bytes > 64 * 1024 * 1024:
            raise AmsError(ErrorCode.PLAN_INVALID, "GLM-4 probe hash buffer exceeds 64 MiB")

    @property
    def config_hash(self) -> str:
        payload = {
            "groups_per_tensor": self.groups_per_tensor,
            "hash_buffer_bytes": self.hash_buffer_bytes,
            "sampling_strategy": _SAMPLING_STRATEGY,
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Glm4QuantizationErrorMetrics:
    """Aggregate reconstruction error for one encoding or reviewed tensor role."""

    scope: str
    tensor_count: int
    sampled_group_count: int
    sampled_element_count: int
    sampled_source_bytes: int
    normalized_root_mean_square_error: float
    normalized_mean_absolute_error: float
    mean_absolute_error: float
    cosine_similarity: float
    maximum_absolute_error: float
    reconstructed_zero_fraction: float


@dataclass(frozen=True, slots=True)
class Glm4QuantizationProbeEvidence:
    """Complete, bounded evidence for one exact source shard and precision candidate."""

    schema_id: str
    status: Glm4QuantizationProbeStatus
    qualifies_precision_policy: bool
    architecture_hash: str
    source_index_hash: str
    source_repository: str
    source_revision: str
    candidate_hash: str
    policy_hash: str
    shard_name: str
    shard_content_hash: str
    probe_config_hash: str
    sampling_strategy: str
    groups_per_tensor: int
    hash_buffer_bytes: int
    ternary_config_hash: str
    int4_config_hash: str
    shard_tensor_count: int
    identity_tensor_count: int
    sampled_tensor_count: int
    sampled_group_count: int
    sampled_element_count: int
    source_file_bytes: int
    integrity_bytes_read: int
    prefix_and_header_bytes_read: int
    sampled_source_bytes_read: int
    maximum_sample_read_bytes: int
    encoding_tensor_counts: tuple[tuple[str, int], ...]
    source_dtype_counts: tuple[tuple[str, int], ...]
    encoding_metrics: tuple[Glm4QuantizationErrorMetrics, ...]
    role_metrics: tuple[Glm4QuantizationErrorMetrics, ...]


@dataclass(slots=True)
class _MetricAccumulator:
    tensor_count: int = 0
    sampled_group_count: int = 0
    sampled_element_count: int = 0
    sampled_source_bytes: int = 0
    source_square_total: float = 0.0
    reconstructed_square_total: float = 0.0
    dot_total: float = 0.0
    error_square_total: float = 0.0
    source_absolute_total: float = 0.0
    error_absolute_total: float = 0.0
    maximum_absolute_error: float = 0.0
    reconstructed_zero_count: int = 0

    def add_tensor(self) -> None:
        self.tensor_count += 1

    def add_group(
        self,
        source: list[float],
        reconstructed: list[float],
        source_bytes: int,
    ) -> None:
        if len(source) != len(reconstructed) or not source:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "GLM-4 probe group length mismatch")
        self.sampled_group_count += 1
        self.sampled_element_count += len(source)
        self.sampled_source_bytes += source_bytes
        for source_value, reconstructed_value in zip(source, reconstructed, strict=True):
            error = reconstructed_value - source_value
            absolute_error = abs(error)
            self.source_square_total += source_value * source_value
            self.reconstructed_square_total += reconstructed_value * reconstructed_value
            self.dot_total += source_value * reconstructed_value
            self.error_square_total += error * error
            self.source_absolute_total += abs(source_value)
            self.error_absolute_total += absolute_error
            self.maximum_absolute_error = max(self.maximum_absolute_error, absolute_error)
            if reconstructed_value == 0.0:
                self.reconstructed_zero_count += 1

    def finish(self, scope: str) -> Glm4QuantizationErrorMetrics:
        if self.tensor_count <= 0 or self.sampled_element_count <= 0:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "GLM-4 probe metric is empty")
        if self.source_square_total == 0.0:
            normalized_rmse = 0.0 if self.error_square_total == 0.0 else math.inf
        else:
            normalized_rmse = math.sqrt(self.error_square_total / self.source_square_total)
        if self.source_absolute_total == 0.0:
            normalized_mae = 0.0 if self.error_absolute_total == 0.0 else math.inf
        else:
            normalized_mae = self.error_absolute_total / self.source_absolute_total
        cosine_denominator = math.sqrt(self.source_square_total * self.reconstructed_square_total)
        if cosine_denominator == 0.0:
            cosine_similarity = (
                1.0
                if self.source_square_total == 0.0 and self.reconstructed_square_total == 0.0
                else 0.0
            )
        else:
            cosine_similarity = self.dot_total / cosine_denominator
        metrics = Glm4QuantizationErrorMetrics(
            scope=scope,
            tensor_count=self.tensor_count,
            sampled_group_count=self.sampled_group_count,
            sampled_element_count=self.sampled_element_count,
            sampled_source_bytes=self.sampled_source_bytes,
            normalized_root_mean_square_error=normalized_rmse,
            normalized_mean_absolute_error=normalized_mae,
            mean_absolute_error=self.error_absolute_total / self.sampled_element_count,
            cosine_similarity=max(-1.0, min(1.0, cosine_similarity)),
            maximum_absolute_error=self.maximum_absolute_error,
            reconstructed_zero_fraction=(
                self.reconstructed_zero_count / self.sampled_element_count
            ),
        )
        for value in (
            metrics.normalized_root_mean_square_error,
            metrics.normalized_mean_absolute_error,
            metrics.mean_absolute_error,
            metrics.cosine_similarity,
            metrics.maximum_absolute_error,
            metrics.reconstructed_zero_fraction,
        ):
            if not math.isfinite(value):
                raise AmsError(
                    ErrorCode.NUMERIC_FAILURE,
                    "GLM-4 probe produced a non-finite aggregate metric",
                )
        return metrics


def _sample_group_indices(group_count: int, requested_count: int) -> tuple[int, ...]:
    checked_positive(group_count, name="glm4_probe.group_count")
    checked_positive(requested_count, name="glm4_probe.requested_group_count")
    sample_count = min(group_count, requested_count)
    if sample_count == 1:
        return (0,)
    return tuple(index * (group_count - 1) // (sample_count - 1) for index in range(sample_count))


def _decode_source_values(
    payload: bytearray,
    element_count: int,
    dtype: DType,
) -> list[float]:
    expected_bytes = checked_mul(
        element_count,
        _SOURCE_ITEM_BYTES[dtype],
        name="glm4_probe.source_group_bytes",
    )
    if len(payload) != expected_bytes:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "GLM-4 probe source buffer is inconsistent")
    if dtype is DType.FLOAT16:
        values = [value[0] for value in struct.iter_unpack("<e", payload)]
    elif dtype is DType.FLOAT32:
        values = [value[0] for value in struct.iter_unpack("<f", payload)]
    elif dtype is DType.BFLOAT16:
        values = [
            struct.unpack("<f", struct.pack("<I", value[0] << 16))[0]
            for value in struct.iter_unpack("<H", payload)
        ]
    else:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"GLM-4 probe source dtype is unsupported: {dtype.value}",
        )
    if len(values) != element_count:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "GLM-4 probe decoded length mismatch")
    if not all(math.isfinite(value) for value in values):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "GLM-4 probe source contains NaN or infinity")
    return values


def _validate_tensor(
    architecture: Glm4MoeLiteArchitecture,
    slot: Glm4MoeLiteTensorSlot,
    tensor: SafetensorsTensor,
) -> tuple[int, int]:
    if tensor.shape != expected_glm4_moe_lite_tensor_shape(architecture, slot):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            (
                "GLM-4 shard tensor shape differs from the reviewed architecture: "
                f"{tensor.source_name}"
            ),
        )
    item_bytes = _SOURCE_ITEM_BYTES.get(tensor.dtype)
    if item_bytes is None:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"GLM-4 shard tensor dtype is unsupported: {tensor.source_name}",
        )
    element_count = checked_product(tensor.shape, name="glm4_probe.tensor_elements")
    expected_bytes = checked_mul(
        element_count,
        item_bytes,
        name="glm4_probe.tensor_source_bytes",
    )
    if tensor.data_length != expected_bytes:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"GLM-4 shard tensor byte length is inconsistent: {tensor.source_name}",
        )
    return element_count, item_bytes


def _metric_accumulator(
    mapping: dict[str, _MetricAccumulator],
    scope: str,
) -> _MetricAccumulator:
    accumulator = mapping.get(scope)
    if accumulator is None:
        accumulator = _MetricAccumulator()
        mapping[scope] = accumulator
    return accumulator


def probe_experimental_glm4_quantization_shard(
    architecture: Glm4MoeLiteArchitecture,
    inventory: Glm4MoeLiteTensorInventory,
    index: HuggingFaceShardIndex,
    *,
    source_repository: str,
    source_revision: str,
    shard_name: str,
    reader: RangeReader,
    expected_shard_hash: str,
    candidate_hash: str,
    policy_hash: str,
    ternary_config: TernaryCodecConfig,
    int4_config: Int4CodecConfig,
    config: Glm4QuantizationProbeConfig | None = None,
) -> Glm4QuantizationProbeEvidence:
    """Measure sampled codec error after authenticating one complete official shard.

    The result is deliberately non-qualifying: tensor reconstruction error cannot replace
    end-to-end logit, perplexity, task, latency, or resource evidence.
    """

    config = config or Glm4QuantizationProbeConfig()
    validate_identifier(source_repository, name="glm4_probe.source_repository")
    validate_identifier(source_revision, name="glm4_probe.source_revision")
    for name, digest in (
        ("expected_shard_hash", expected_shard_hash),
        ("candidate_hash", candidate_hash),
        ("policy_hash", policy_hash),
    ):
        validate_digest(digest, name=f"glm4_probe.{name}")
    if not expected_shard_hash.startswith("sha256:"):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "GLM-4 probe requires a SHA-256 shard hash")
    reviewed_inventory = validate_glm4_moe_lite_tensor_inventory(architecture, index)
    if inventory != reviewed_inventory:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 probe inventory does not match the reviewed architecture and index",
        )
    expected_names = {
        entry.tensor_name for entry in index.entries if entry.shard_name == shard_name
    }
    if not expected_names or shard_name not in index.shard_names:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 probe shard is absent from the normalized index",
        )

    actual_hash = hash_reader_range(
        reader,
        0,
        reader.size_bytes,
        buffer_bytes=config.hash_buffer_bytes,
        algorithm="sha256",
    )
    if actual_hash != expected_shard_hash:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4 probe shard hash mismatch")
    header = parse_safetensors_header(reader)
    tensor_by_name = {tensor.source_name: tensor for tensor in header.tensors}
    if set(tensor_by_name) != expected_names:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 shard header does not exactly match its normalized index entries",
            evidence={
                "missing": len(expected_names - set(tensor_by_name)),
                "unexpected": len(set(tensor_by_name) - expected_names),
            },
        )

    slot_by_name = {slot.tensor_name: slot for slot in inventory.slots}
    encoding_counts = {encoding.value: 0 for encoding in HuggingFaceTensorEncoding}
    dtype_counts: dict[str, int] = {}
    encoding_accumulators: dict[str, _MetricAccumulator] = {}
    role_accumulators: dict[str, _MetricAccumulator] = {}
    identity_tensor_count = 0
    sampled_tensor_count = 0
    sampled_source_bytes = 0
    maximum_sample_read = 0

    for tensor_name in sorted(tensor_by_name):
        tensor = tensor_by_name[tensor_name]
        slot = slot_by_name[tensor_name]
        element_count, item_bytes = _validate_tensor(architecture, slot, tensor)
        dtype_counts[tensor.source_dtype] = dtype_counts.get(tensor.source_dtype, 0) + 1
        encoding = experimental_glm4_encoding_for_role(slot.role)
        encoding_counts[encoding.value] += 1
        if encoding is HuggingFaceTensorEncoding.IDENTITY:
            identity_tensor_count += 1
            continue
        sampled_tensor_count += 1
        if encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
            group_size = ternary_config.group_size
        elif encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC:
            group_size = int4_config.group_size
        else:
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "GLM-4 probe encountered an unreviewed compressed encoding",
            )
        group_count = (element_count + group_size - 1) // group_size
        encoding_accumulator = _metric_accumulator(encoding_accumulators, encoding.value)
        role_accumulator = _metric_accumulator(role_accumulators, slot.role.value)
        encoding_accumulator.add_tensor()
        role_accumulator.add_tensor()
        for group_index in _sample_group_indices(group_count, config.groups_per_tensor):
            element_offset = checked_mul(
                group_index,
                group_size,
                name="glm4_probe.group_element_offset",
            )
            group_elements = min(group_size, element_count - element_offset)
            byte_count = checked_mul(
                group_elements,
                item_bytes,
                name="glm4_probe.sample_bytes",
            )
            source_offset = checked_add(
                tensor.absolute_offset,
                checked_mul(
                    element_offset,
                    item_bytes,
                    name="glm4_probe.sample_element_bytes",
                ),
                name="glm4_probe.sample_source_offset",
            )
            payload = bytearray(byte_count)
            reader.read_into(source_offset, payload)
            source_values = _decode_source_values(payload, group_elements, tensor.dtype)
            if encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
                encoded = encode_ternary_group_reference(source_values, ternary_config)
                reconstructed = decode_ternary_group_reference(encoded, group_elements)
            else:
                encoded = encode_int4_group_reference(source_values, int4_config)
                reconstructed = decode_int4_group_reference(encoded, group_elements)
            encoding_accumulator.add_group(source_values, reconstructed, byte_count)
            role_accumulator.add_group(source_values, reconstructed, byte_count)
            sampled_source_bytes = checked_add(
                sampled_source_bytes,
                byte_count,
                name="glm4_probe.sampled_source_bytes",
            )
            maximum_sample_read = max(maximum_sample_read, byte_count)

    encoding_metrics = tuple(
        encoding_accumulators[scope].finish(scope) for scope in sorted(encoding_accumulators)
    )
    role_metrics = tuple(
        role_accumulators[scope].finish(scope) for scope in sorted(role_accumulators)
    )
    sampled_group_count = sum(metric.sampled_group_count for metric in encoding_metrics)
    sampled_element_count = sum(metric.sampled_element_count for metric in encoding_metrics)
    if (
        sampled_source_bytes != sum(metric.sampled_source_bytes for metric in encoding_metrics)
        or sampled_tensor_count != sum(metric.tensor_count for metric in encoding_metrics)
        or sampled_tensor_count + identity_tensor_count != len(header.tensors)
    ):
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "GLM-4 probe evidence totals disagree")

    return Glm4QuantizationProbeEvidence(
        schema_id="ams.glm4.quantization-probe.v1",
        status=Glm4QuantizationProbeStatus.DIAGNOSTIC,
        qualifies_precision_policy=False,
        architecture_hash=architecture.content_hash,
        source_index_hash=index.content_hash,
        source_repository=source_repository,
        source_revision=source_revision,
        candidate_hash=candidate_hash,
        policy_hash=policy_hash,
        shard_name=shard_name,
        shard_content_hash=actual_hash,
        probe_config_hash=config.config_hash,
        sampling_strategy=_SAMPLING_STRATEGY,
        groups_per_tensor=config.groups_per_tensor,
        hash_buffer_bytes=config.hash_buffer_bytes,
        ternary_config_hash=ternary_config.config_hash,
        int4_config_hash=int4_config.config_hash,
        shard_tensor_count=len(header.tensors),
        identity_tensor_count=identity_tensor_count,
        sampled_tensor_count=sampled_tensor_count,
        sampled_group_count=sampled_group_count,
        sampled_element_count=sampled_element_count,
        source_file_bytes=reader.size_bytes,
        integrity_bytes_read=reader.size_bytes,
        prefix_and_header_bytes_read=8 + header.header_bytes,
        sampled_source_bytes_read=sampled_source_bytes,
        maximum_sample_read_bytes=maximum_sample_read,
        encoding_tensor_counts=tuple(
            (encoding, count) for encoding, count in sorted(encoding_counts.items()) if count
        ),
        source_dtype_counts=tuple(sorted(dtype_counts.items())),
        encoding_metrics=encoding_metrics,
        role_metrics=role_metrics,
    )
