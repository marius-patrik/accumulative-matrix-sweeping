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
    Int3DiagnosticCodecConfig,
    Int4CodecConfig,
    TernaryCodecConfig,
    decode_int3_group_reference,
    decode_int4_group_reference,
    decode_ternary_group_reference,
    encode_int3_group_reference,
    encode_int4_group_reference,
    encode_ternary_group_reference,
)
from ams.descriptors import DType, validate_digest, validate_identifier
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_moe_lite import (
    Glm4MoeLiteArchitecture,
    Glm4MoeLiteTensorInventory,
    Glm4MoeLiteTensorRole,
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


class Glm4QuantizationVariantEncoding(StrEnum):
    """Diagnostic encodings; only established codecs may become package assignments."""

    INT2_SYMMETRIC_MIDRISE = "int2_symmetric_midrise"
    INT3_SYMMETRIC = "int3_symmetric"
    TERNARY_TRIT5 = "ternary_trit5"
    TERNARY_RESIDUAL2 = "ternary_residual2"
    TERNARY_SPARSE_BF16_RESIDUAL = "ternary_sparse_bf16_residual"
    INT4_SYMMETRIC = "int4_symmetric"


@dataclass(frozen=True, slots=True)
class Glm4LowBitDiagnosticConfig:
    """Exact simulated semantics for formats not yet admitted to AMS packages."""

    encoding: Glm4QuantizationVariantEncoding
    group_size: int = 128
    threshold_numerator: int | None = None
    threshold_denominator: int | None = None
    residual_count: int = 0
    version: str = "diagnostic-1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "encoding",
            Glm4QuantizationVariantEncoding(self.encoding),
        )
        checked_positive(self.group_size, name="glm4_diagnostic_codec.group_size")
        if self.group_size > 256:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "diagnostic codec group size exceeds uint8 residual positions",
            )
        if self.version != "diagnostic-1":
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "unsupported diagnostic codec version",
            )
        if self.encoding is Glm4QuantizationVariantEncoding.TERNARY_SPARSE_BF16_RESIDUAL:
            if (
                isinstance(self.threshold_numerator, bool)
                or not isinstance(self.threshold_numerator, int)
                or isinstance(self.threshold_denominator, bool)
                or not isinstance(self.threshold_denominator, int)
            ):
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "sparse ternary residual requires an integer threshold ratio",
                )
            checked_positive(
                self.threshold_numerator,
                name="glm4_diagnostic_codec.threshold_numerator",
            )
            checked_positive(
                self.threshold_denominator,
                name="glm4_diagnostic_codec.threshold_denominator",
            )
            if self.threshold_numerator > self.threshold_denominator:
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "sparse ternary residual threshold ratio must not exceed one",
                )
            checked_positive(
                self.residual_count,
                name="glm4_diagnostic_codec.residual_count",
            )
            if self.residual_count > self.group_size:
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "sparse ternary residual count exceeds the group size",
                )
        elif self.encoding not in {
            Glm4QuantizationVariantEncoding.INT2_SYMMETRIC_MIDRISE,
            Glm4QuantizationVariantEncoding.INT3_SYMMETRIC,
        }:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "established or residual-pass codecs cannot use a diagnostic config",
            )
        elif (
            self.threshold_numerator is not None
            or self.threshold_denominator is not None
            or self.residual_count != 0
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "uniform diagnostic codecs cannot declare residual fields",
            )

    @property
    def config_hash(self) -> str:
        if self.encoding is Glm4QuantizationVariantEncoding.INT3_SYMMETRIC:
            return Int3DiagnosticCodecConfig(group_size=self.group_size).config_hash
        payload: dict[str, int | str] = {
            "encoding": self.encoding.value,
            "group_size": self.group_size,
            "scale_dtype": "float32",
            "version": self.version,
        }
        if self.encoding is Glm4QuantizationVariantEncoding.TERNARY_SPARSE_BF16_RESIDUAL:
            assert self.threshold_numerator is not None
            assert self.threshold_denominator is not None
            payload.update(
                {
                    "base_packing": "trit5",
                    "residual_count": self.residual_count,
                    "residual_dtype": "bfloat16",
                    "residual_position_dtype": "uint8",
                    "threshold_denominator": self.threshold_denominator,
                    "threshold_numerator": self.threshold_numerator,
                }
            )
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def group_record_size(self, element_count: int) -> int:
        checked_positive(
            element_count,
            name="glm4_diagnostic_codec.group_element_count",
        )
        if element_count > self.group_size:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "diagnostic codec group exceeds configured group size",
            )
        if self.encoding is Glm4QuantizationVariantEncoding.INT2_SYMMETRIC_MIDRISE:
            bit_count = checked_mul(element_count, 2, name="glm4_int2.bits")
            return 4 + (bit_count + 7) // 8
        if self.encoding is Glm4QuantizationVariantEncoding.INT3_SYMMETRIC:
            return Int3DiagnosticCodecConfig(group_size=self.group_size).group_record_size(
                element_count
            )
        residuals = min(self.residual_count, element_count)
        ternary_bytes = 4 + (element_count + 4) // 5
        return checked_add(
            ternary_bytes,
            checked_mul(residuals, 3, name="glm4_residual.entries"),
            name="glm4_residual.record_bytes",
        )

    def encoded_size(self, element_count: int) -> int:
        checked_positive(element_count, name="glm4_diagnostic_codec.element_count")
        groups = (element_count - 1) // self.group_size + 1
        full_records = checked_mul(
            max(groups - 1, 0),
            self.group_record_size(self.group_size),
            name="glm4_diagnostic_codec.full_records",
        )
        tail_count = element_count - (groups - 1) * self.group_size
        return checked_add(
            full_records,
            self.group_record_size(tail_count),
            name="glm4_diagnostic_codec.encoded_bytes",
        )

    def reconstruct(self, values: list[float]) -> list[float]:
        if not values or len(values) > self.group_size:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "diagnostic codec values must contain one bounded group",
            )
        if not all(
            not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        ):
            raise AmsError(
                ErrorCode.NUMERIC_FAILURE,
                "diagnostic codec source contains a non-finite or nonnumeric value",
            )
        if self.encoding is Glm4QuantizationVariantEncoding.INT2_SYMMETRIC_MIDRISE:
            return _reconstruct_int2_midrise(values)
        if self.encoding is Glm4QuantizationVariantEncoding.INT3_SYMMETRIC:
            config = Int3DiagnosticCodecConfig(group_size=self.group_size)
            payload = encode_int3_group_reference(values, config)
            return decode_int3_group_reference(payload, len(values))
        assert self.threshold_numerator is not None
        assert self.threshold_denominator is not None
        ternary_config = TernaryCodecConfig(
            group_size=self.group_size,
            threshold_numerator=self.threshold_numerator,
            threshold_denominator=self.threshold_denominator,
        )
        payload = encode_ternary_group_reference(values, ternary_config)
        reconstructed = decode_ternary_group_reference(payload, len(values))
        ranked_residuals = sorted(
            (
                (abs(source - approximate), index, source - approximate)
                for index, (source, approximate) in enumerate(
                    zip(values, reconstructed, strict=True)
                )
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for _, index, residual in ranked_residuals[: self.residual_count]:
            reconstructed[index] += _round_bfloat16(residual)
        return reconstructed


def _round_float32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise AmsError(
            ErrorCode.NUMERIC_FAILURE,
            "diagnostic codec scale is not representable as float32",
        ) from exc


def _round_bfloat16(value: float) -> float:
    """Round a finite value to BF16 with round-to-nearest, ties-to-even."""

    bits = struct.unpack("<I", struct.pack("<f", _round_float32(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", rounded & 0xFFFF0000))[0]


def _reconstruct_int2_midrise(values: list[float]) -> list[float]:
    maximum = max(abs(value) for value in values)
    scale = _round_float32(maximum)
    if scale == 0.0:
        return [0.0] * len(values)
    levels = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)
    output = []
    for value in values:
        normalized = value / scale
        level = min(
            levels,
            key=lambda candidate: (
                abs(normalized - candidate),
                0 if candidate > 0.0 and value >= 0.0 else 1,
                abs(candidate),
            ),
        )
        output.append(scale * level)
    return output


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


@dataclass(frozen=True, slots=True)
class Glm4QuantizationCodecVariant:
    """One exact low-bit codec configuration applied to identical sampled source groups."""

    variant_id: str
    encoding: Glm4QuantizationVariantEncoding
    ternary_config: TernaryCodecConfig | None = None
    int4_config: Int4CodecConfig | None = None
    diagnostic_config: Glm4LowBitDiagnosticConfig | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.variant_id, name="glm4_probe.variant_id")
        object.__setattr__(self, "encoding", Glm4QuantizationVariantEncoding(self.encoding))
        if self.diagnostic_config is not None:
            if (
                not isinstance(self.diagnostic_config, Glm4LowBitDiagnosticConfig)
                or self.ternary_config is not None
                or self.int4_config is not None
            ):
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "diagnostic comparison variant requires only a diagnostic config",
                )
            if self.encoding is not self.diagnostic_config.encoding:
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "diagnostic comparison encoding disagrees with its config",
                )
            return
        if self.encoding in {
            Glm4QuantizationVariantEncoding.TERNARY_TRIT5,
            Glm4QuantizationVariantEncoding.TERNARY_RESIDUAL2,
        }:
            if (
                not isinstance(self.ternary_config, TernaryCodecConfig)
                or self.int4_config is not None
            ):
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "ternary comparison variant requires only a ternary config",
                )
        elif self.encoding is Glm4QuantizationVariantEncoding.INT4_SYMMETRIC:
            if not isinstance(self.int4_config, Int4CodecConfig) or self.ternary_config is not None:
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "INT4 comparison variant requires only an INT4 config",
                )
        else:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "comparison variants must use a lossy low-bit encoding",
            )

    @property
    def codec_config_hash(self) -> str:
        if self.diagnostic_config is not None:
            return self.diagnostic_config.config_hash
        if self.ternary_config is not None:
            if self.encoding is Glm4QuantizationVariantEncoding.TERNARY_RESIDUAL2:
                payload = {
                    "base_ternary_config_hash": self.ternary_config.config_hash,
                    "residual_passes": 2,
                    "version": "1.0.0",
                }
                return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
            return self.ternary_config.config_hash
        if self.int4_config is not None:
            return self.int4_config.config_hash
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "comparison variant has no codec config")

    @property
    def group_size(self) -> int:
        if self.diagnostic_config is not None:
            return self.diagnostic_config.group_size
        if self.ternary_config is not None:
            return self.ternary_config.group_size
        if self.int4_config is not None:
            return self.int4_config.group_size
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "comparison variant has no group size")

    @property
    def variant_hash(self) -> str:
        payload = {
            "codec_config_hash": self.codec_config_hash,
            "encoding": self.encoding.value,
            "variant_id": self.variant_id,
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def encoded_size(self, element_count: int) -> int:
        if self.diagnostic_config is not None:
            return self.diagnostic_config.encoded_size(element_count)
        if self.ternary_config is not None:
            encoded_bytes = self.ternary_config.encoded_size(element_count)
            if self.encoding is Glm4QuantizationVariantEncoding.TERNARY_RESIDUAL2:
                return checked_mul(
                    encoded_bytes,
                    2,
                    name="glm4_comparison.residual_ternary_bytes",
                )
            return encoded_bytes
        if self.int4_config is not None:
            return self.int4_config.encoded_size(element_count)
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "comparison variant has no size contract")

    def reconstruct(self, values: list[float]) -> list[float]:
        if self.diagnostic_config is not None:
            return self.diagnostic_config.reconstruct(values)
        if self.ternary_config is not None:
            payload = encode_ternary_group_reference(values, self.ternary_config)
            first = decode_ternary_group_reference(payload, len(values))
            if self.encoding is Glm4QuantizationVariantEncoding.TERNARY_TRIT5:
                return first
            if self.encoding is Glm4QuantizationVariantEncoding.TERNARY_RESIDUAL2:
                residual = [
                    source_value - first_value
                    for source_value, first_value in zip(values, first, strict=True)
                ]
                residual_payload = encode_ternary_group_reference(
                    residual,
                    self.ternary_config,
                )
                second = decode_ternary_group_reference(residual_payload, len(values))
                return [
                    first_value + second_value
                    for first_value, second_value in zip(first, second, strict=True)
                ]
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "ternary comparison variant has an unknown diagnostic encoding",
            )
        if self.int4_config is not None:
            payload = encode_int4_group_reference(values, self.int4_config)
            return decode_int4_group_reference(payload, len(values))
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "comparison variant has no codec")


@dataclass(frozen=True, slots=True)
class Glm4QuantizationVariantMetrics:
    """Same-sample reconstruction and full-selected-tensor storage for one variant."""

    variant_id: str
    variant_hash: str
    encoding: Glm4QuantizationVariantEncoding
    codec_config_hash: str
    group_size: int
    selected_tensor_encoded_bytes: int
    metrics: Glm4QuantizationErrorMetrics


@dataclass(frozen=True, slots=True)
class Glm4QuantizationComparisonEvidence:
    """Non-qualifying same-source comparison of low-bit variants for selected roles."""

    schema_id: str
    status: Glm4QuantizationProbeStatus
    qualifies_precision_policy: bool
    architecture_hash: str
    source_index_hash: str
    source_repository: str
    source_revision: str
    baseline_candidate_hash: str
    baseline_policy_hash: str
    shard_name: str
    shard_content_hash: str
    comparison_hash: str
    sampling_strategy: str
    groups_per_tensor: int
    group_size: int
    hash_buffer_bytes: int
    selected_roles: tuple[str, ...]
    shard_tensor_count: int
    selected_tensor_count: int
    sampled_group_count: int
    sampled_element_count: int
    selected_tensor_source_bytes: int
    source_file_bytes: int
    integrity_bytes_read: int
    prefix_and_header_bytes_read: int
    sampled_source_bytes_read: int
    maximum_sample_read_bytes: int
    source_dtype_counts: tuple[tuple[str, int], ...]
    variants: tuple[Glm4QuantizationVariantMetrics, ...]


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


@dataclass(frozen=True, slots=True)
class _AdmittedProbeShard:
    content_hash: str
    header_bytes: int
    tensors: tuple[SafetensorsTensor, ...]
    tensor_by_name: dict[str, SafetensorsTensor]
    slot_by_name: dict[str, Glm4MoeLiteTensorSlot]


def _admit_probe_shard(
    architecture: Glm4MoeLiteArchitecture,
    inventory: Glm4MoeLiteTensorInventory,
    index: HuggingFaceShardIndex,
    *,
    shard_name: str,
    reader: RangeReader,
    expected_shard_hash: str,
    config: Glm4QuantizationProbeConfig,
) -> _AdmittedProbeShard:
    validate_digest(expected_shard_hash, name="glm4_probe.expected_shard_hash")
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
    return _AdmittedProbeShard(
        content_hash=actual_hash,
        header_bytes=header.header_bytes,
        tensors=header.tensors,
        tensor_by_name=tensor_by_name,
        slot_by_name=slot_by_name,
    )


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
        ("candidate_hash", candidate_hash),
        ("policy_hash", policy_hash),
    ):
        validate_digest(digest, name=f"glm4_probe.{name}")
    admitted = _admit_probe_shard(
        architecture,
        inventory,
        index,
        shard_name=shard_name,
        reader=reader,
        expected_shard_hash=expected_shard_hash,
        config=config,
    )
    tensor_by_name = admitted.tensor_by_name
    slot_by_name = admitted.slot_by_name
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
        or sampled_tensor_count + identity_tensor_count != len(admitted.tensors)
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
        shard_content_hash=admitted.content_hash,
        probe_config_hash=config.config_hash,
        sampling_strategy=_SAMPLING_STRATEGY,
        groups_per_tensor=config.groups_per_tensor,
        hash_buffer_bytes=config.hash_buffer_bytes,
        ternary_config_hash=ternary_config.config_hash,
        int4_config_hash=int4_config.config_hash,
        shard_tensor_count=len(admitted.tensors),
        identity_tensor_count=identity_tensor_count,
        sampled_tensor_count=sampled_tensor_count,
        sampled_group_count=sampled_group_count,
        sampled_element_count=sampled_element_count,
        source_file_bytes=reader.size_bytes,
        integrity_bytes_read=reader.size_bytes,
        prefix_and_header_bytes_read=8 + admitted.header_bytes,
        sampled_source_bytes_read=sampled_source_bytes,
        maximum_sample_read_bytes=maximum_sample_read,
        encoding_tensor_counts=tuple(
            (encoding, count) for encoding, count in sorted(encoding_counts.items()) if count
        ),
        source_dtype_counts=tuple(sorted(dtype_counts.items())),
        encoding_metrics=encoding_metrics,
        role_metrics=role_metrics,
    )


def compare_glm4_quantization_variants(
    architecture: Glm4MoeLiteArchitecture,
    inventory: Glm4MoeLiteTensorInventory,
    index: HuggingFaceShardIndex,
    *,
    source_repository: str,
    source_revision: str,
    shard_name: str,
    reader: RangeReader,
    expected_shard_hash: str,
    baseline_candidate_hash: str,
    baseline_policy_hash: str,
    selected_roles: tuple[Glm4MoeLiteTensorRole, ...],
    variants: tuple[Glm4QuantizationCodecVariant, ...],
    config: Glm4QuantizationProbeConfig | None = None,
) -> Glm4QuantizationComparisonEvidence:
    """Compare exact codecs on identical authenticated source groups for selected roles."""

    config = config or Glm4QuantizationProbeConfig()
    validate_identifier(source_repository, name="glm4_probe.source_repository")
    validate_identifier(source_revision, name="glm4_probe.source_revision")
    validate_digest(
        baseline_candidate_hash,
        name="glm4_probe.baseline_candidate_hash",
    )
    validate_digest(
        baseline_policy_hash,
        name="glm4_probe.baseline_policy_hash",
    )
    if not isinstance(selected_roles, tuple) or not 1 <= len(selected_roles) <= 32:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "GLM-4 comparison requires one to 32 selected roles",
        )
    try:
        normalized_roles = tuple(
            sorted(
                (Glm4MoeLiteTensorRole(role) for role in selected_roles),
                key=lambda role: role.value,
            )
        )
    except ValueError as exc:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "GLM-4 comparison contains an unknown tensor role",
        ) from exc
    if len(set(normalized_roles)) != len(normalized_roles):
        raise AmsError(ErrorCode.PLAN_INVALID, "GLM-4 comparison roles are duplicated")
    if not isinstance(variants, tuple) or not 2 <= len(variants) <= 16:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "GLM-4 comparison requires two to 16 codec variants",
        )
    if not all(isinstance(variant, Glm4QuantizationCodecVariant) for variant in variants):
        raise AmsError(ErrorCode.PLAN_INVALID, "GLM-4 comparison variant has the wrong type")
    normalized_variants = tuple(sorted(variants, key=lambda variant: variant.variant_id))
    if len({variant.variant_id for variant in normalized_variants}) != len(
        normalized_variants
    ) or len({variant.variant_hash for variant in normalized_variants}) != len(normalized_variants):
        raise AmsError(ErrorCode.PLAN_INVALID, "GLM-4 comparison variants are duplicated")
    group_sizes = {variant.group_size for variant in normalized_variants}
    if len(group_sizes) != 1:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "same-sample GLM-4 comparison requires one shared group size",
        )
    group_size = next(iter(group_sizes))

    admitted = _admit_probe_shard(
        architecture,
        inventory,
        index,
        shard_name=shard_name,
        reader=reader,
        expected_shard_hash=expected_shard_hash,
        config=config,
    )
    selected_role_set = set(normalized_roles)
    seen_roles: set[Glm4MoeLiteTensorRole] = set()
    variant_accumulators = {
        variant.variant_id: _MetricAccumulator() for variant in normalized_variants
    }
    variant_encoded_bytes = {variant.variant_id: 0 for variant in normalized_variants}
    dtype_counts: dict[str, int] = {}
    selected_tensor_count = 0
    selected_tensor_source_bytes = 0
    sampled_group_count = 0
    sampled_element_count = 0
    sampled_source_bytes = 0
    maximum_sample_read = 0

    for tensor_name in sorted(admitted.tensor_by_name):
        tensor = admitted.tensor_by_name[tensor_name]
        slot = admitted.slot_by_name[tensor_name]
        if slot.role not in selected_role_set:
            continue
        seen_roles.add(slot.role)
        element_count, item_bytes = _validate_tensor(architecture, slot, tensor)
        selected_tensor_count += 1
        selected_tensor_source_bytes = checked_add(
            selected_tensor_source_bytes,
            tensor.data_length,
            name="glm4_comparison.selected_source_bytes",
        )
        dtype_counts[tensor.source_dtype] = dtype_counts.get(tensor.source_dtype, 0) + 1
        for variant in normalized_variants:
            variant_accumulators[variant.variant_id].add_tensor()
            variant_encoded_bytes[variant.variant_id] = checked_add(
                variant_encoded_bytes[variant.variant_id],
                variant.encoded_size(element_count),
                name="glm4_comparison.encoded_bytes",
            )
        group_count = (element_count + group_size - 1) // group_size
        for group_index in _sample_group_indices(group_count, config.groups_per_tensor):
            element_offset = checked_mul(
                group_index,
                group_size,
                name="glm4_comparison.group_element_offset",
            )
            group_elements = min(group_size, element_count - element_offset)
            byte_count = checked_mul(
                group_elements,
                item_bytes,
                name="glm4_comparison.sample_bytes",
            )
            source_offset = checked_add(
                tensor.absolute_offset,
                checked_mul(
                    element_offset,
                    item_bytes,
                    name="glm4_comparison.sample_element_bytes",
                ),
                name="glm4_comparison.sample_source_offset",
            )
            payload = bytearray(byte_count)
            reader.read_into(source_offset, payload)
            source_values = _decode_source_values(payload, group_elements, tensor.dtype)
            for variant in normalized_variants:
                reconstructed = variant.reconstruct(source_values)
                variant_accumulators[variant.variant_id].add_group(
                    source_values,
                    reconstructed,
                    byte_count,
                )
            sampled_group_count += 1
            sampled_element_count += group_elements
            sampled_source_bytes = checked_add(
                sampled_source_bytes,
                byte_count,
                name="glm4_comparison.sampled_source_bytes",
            )
            maximum_sample_read = max(maximum_sample_read, byte_count)

    if seen_roles != selected_role_set or selected_tensor_count <= 0:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4 comparison shard does not contain every selected role",
            evidence={
                "missing_roles": len(selected_role_set - seen_roles),
                "selected_tensors": selected_tensor_count,
            },
        )
    variant_metrics = []
    for variant in normalized_variants:
        metrics = variant_accumulators[variant.variant_id].finish(variant.variant_id)
        if (
            metrics.tensor_count != selected_tensor_count
            or metrics.sampled_group_count != sampled_group_count
            or metrics.sampled_element_count != sampled_element_count
            or metrics.sampled_source_bytes != sampled_source_bytes
        ):
            raise AmsError(
                ErrorCode.INTERNAL_INVARIANT,
                "GLM-4 comparison variant totals disagree",
            )
        variant_metrics.append(
            Glm4QuantizationVariantMetrics(
                variant_id=variant.variant_id,
                variant_hash=variant.variant_hash,
                encoding=variant.encoding,
                codec_config_hash=variant.codec_config_hash,
                group_size=variant.group_size,
                selected_tensor_encoded_bytes=variant_encoded_bytes[variant.variant_id],
                metrics=metrics,
            )
        )
    comparison_payload = {
        "baseline_candidate_hash": baseline_candidate_hash,
        "baseline_policy_hash": baseline_policy_hash,
        "probe_config_hash": config.config_hash,
        "selected_roles": [role.value for role in normalized_roles],
        "shard_content_hash": admitted.content_hash,
        "source_index_hash": index.content_hash,
        "variants": [variant.variant_hash for variant in normalized_variants],
    }
    comparison_hash = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(comparison_payload)).hexdigest()
    )
    return Glm4QuantizationComparisonEvidence(
        schema_id="ams.glm4.quantization-comparison.v1",
        status=Glm4QuantizationProbeStatus.DIAGNOSTIC,
        qualifies_precision_policy=False,
        architecture_hash=architecture.content_hash,
        source_index_hash=index.content_hash,
        source_repository=source_repository,
        source_revision=source_revision,
        baseline_candidate_hash=baseline_candidate_hash,
        baseline_policy_hash=baseline_policy_hash,
        shard_name=shard_name,
        shard_content_hash=admitted.content_hash,
        comparison_hash=comparison_hash,
        sampling_strategy=_SAMPLING_STRATEGY,
        groups_per_tensor=config.groups_per_tensor,
        group_size=group_size,
        hash_buffer_bytes=config.hash_buffer_bytes,
        selected_roles=tuple(role.value for role in normalized_roles),
        shard_tensor_count=len(admitted.tensors),
        selected_tensor_count=selected_tensor_count,
        sampled_group_count=sampled_group_count,
        sampled_element_count=sampled_element_count,
        selected_tensor_source_bytes=selected_tensor_source_bytes,
        source_file_bytes=reader.size_bytes,
        integrity_bytes_read=reader.size_bytes,
        prefix_and_header_bytes_read=8 + admitted.header_bytes,
        sampled_source_bytes_read=sampled_source_bytes,
        maximum_sample_read_bytes=maximum_sample_read,
        source_dtype_counts=tuple(sorted(dtype_counts.items())),
        variants=tuple(variant_metrics),
    )
