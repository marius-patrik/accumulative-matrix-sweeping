"""Deterministic grouped ternary reference codec.

Format ``ams.ternary.trit5`` version 1.0.0 stores each group as a little-endian FP32
scale followed by five base-3 digits per byte. Digits 0, 1, and 2 mean -scale, zero,
and +scale. Unused tail digits must encode zero. Quantization uses source-order means:
the threshold is ``mean(abs(x)) * numerator / denominator`` and the scale is the
source-order mean absolute value of weights strictly above that threshold.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_mul, checked_positive, checked_product
from ams.descriptors import ByteRange, DType
from ams.errors import AmsError, ErrorCode
from ams.storage import RangeReader

_TRITS_PER_BYTE = 5
_SCALE_BYTES = 4
_TRIT_POWERS = (1, 3, 9, 27, 81)
_SOURCE_ITEM_BYTES = {
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.FLOAT32: 4,
}


class BinarySink(Protocol):
    def write(self, data: bytes | bytearray | memoryview) -> int | None: ...


@dataclass(frozen=True, slots=True)
class TernaryCodecConfig:
    group_size: int = 128
    threshold_numerator: int = 7
    threshold_denominator: int = 10
    scale_dtype: DType = DType.FLOAT32
    packing: str = "trit5"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        checked_positive(self.group_size, name="ternary.group_size")
        if self.group_size > 65_536:
            raise AmsError(ErrorCode.PLAN_INVALID, "ternary group size exceeds 65536")
        checked_positive(self.threshold_numerator, name="ternary.threshold_numerator")
        checked_positive(self.threshold_denominator, name="ternary.threshold_denominator")
        if self.threshold_numerator > self.threshold_denominator:
            raise AmsError(ErrorCode.PLAN_INVALID, "ternary threshold ratio must not exceed one")
        object.__setattr__(self, "scale_dtype", DType(self.scale_dtype))
        if self.scale_dtype is not DType.FLOAT32:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "ternary v1 requires FP32 scales")
        if self.packing != "trit5" or self.version != "1.0.0":
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "unsupported ternary codec variant")

    @property
    def config_hash(self) -> str:
        payload = {
            "group_size": self.group_size,
            "packing": self.packing,
            "scale_dtype": self.scale_dtype.value,
            "threshold_denominator": self.threshold_denominator,
            "threshold_numerator": self.threshold_numerator,
            "version": self.version,
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def group_record_size(self, element_count: int) -> int:
        checked_positive(element_count, name="ternary.group_element_count")
        if element_count > self.group_size:
            raise AmsError(ErrorCode.PLAN_INVALID, "ternary group exceeds configured group size")
        return _SCALE_BYTES + (element_count + _TRITS_PER_BYTE - 1) // _TRITS_PER_BYTE

    def encoded_size(self, element_count: int) -> int:
        checked_positive(element_count, name="ternary.element_count")
        groups = (element_count - 1) // self.group_size + 1
        full_record = self.group_record_size(self.group_size)
        tail = element_count if groups == 1 else element_count - (groups - 1) * self.group_size
        return checked_add(
            checked_mul(max(groups - 1, 0), full_record, name="ternary.full_records"),
            self.group_record_size(tail),
            name="ternary.encoded_bytes",
        )


@dataclass(frozen=True, slots=True)
class TernaryEncodingResult:
    content_hash: str
    source_checksum: str
    element_count: int
    group_count: int
    encoded_bytes: int
    decoded_bytes: int
    maximum_source_read_bytes: int


def _decode_source_group(payload: bytearray, count: int, dtype: DType) -> list[float]:
    values: list[float] = []
    if dtype is DType.FLOAT32:
        values.extend(struct.unpack_from("<f", payload, index * 4)[0] for index in range(count))
    elif dtype is DType.FLOAT16:
        values.extend(struct.unpack_from("<e", payload, index * 2)[0] for index in range(count))
    elif dtype is DType.BFLOAT16:
        for index in range(count):
            word = struct.unpack_from("<H", payload, index * 2)[0]
            values.append(struct.unpack("<f", struct.pack("<I", word << 16))[0])
    else:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, f"unsupported ternary source dtype: {dtype}")
    if not all(math.isfinite(value) for value in values):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "ternary source contains NaN or infinity")
    return values


def _encode_group(values: Sequence[float], config: TernaryCodecConfig) -> bytes:
    absolute_total = 0.0
    for value in values:
        absolute_total += abs(value)
    mean_absolute = absolute_total / len(values)
    threshold = mean_absolute * config.threshold_numerator / config.threshold_denominator
    selected_total = 0.0
    selected_count = 0
    for value in values:
        if abs(value) > threshold:
            selected_total += abs(value)
            selected_count += 1
    scale = selected_total / selected_count if selected_count else 0.0
    digits = [0 if value < -threshold else 2 if value > threshold else 1 for value in values]
    payload = bytearray(struct.pack("<f", scale))
    for start in range(0, len(digits), _TRITS_PER_BYTE):
        packed = 0
        for slot, power in enumerate(_TRIT_POWERS):
            index = start + slot
            digit = digits[index] if index < len(digits) else 1
            packed += digit * power
        payload.append(packed)
    return bytes(payload)


def encode_ternary_group_reference(
    values: Sequence[float],
    config: TernaryCodecConfig | None = None,
) -> bytes:
    """Validate and encode one complete v1 group using the production codec semantics."""
    config = config or TernaryCodecConfig()
    try:
        element_count = len(values)
    except TypeError as exc:
        raise AmsError(ErrorCode.PLAN_INVALID, "ternary group values must be a sequence") from exc
    checked_positive(element_count, name="ternary.group_element_count")
    if element_count > config.group_size:
        raise AmsError(ErrorCode.PLAN_INVALID, "ternary group exceeds configured group size")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AmsError(ErrorCode.PLAN_INVALID, "ternary group values must be numeric")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise AmsError(
                ErrorCode.NUMERIC_FAILURE,
                "ternary source contains an unrepresentable numeric value",
            ) from exc
        if not finite:
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "ternary source contains NaN or infinity")
    return _encode_group(values, config)


def _write_exact(sink: BinarySink, payload: bytes, digest) -> int:
    view = memoryview(payload)
    try:
        written = 0
        while written < len(payload):
            count = sink.write(view[written:])
            if count is None or count == 0:
                raise AmsError(
                    ErrorCode.IO_FAILURE,
                    "short write from ternary encoder",
                    retriable=True,
                )
            written += count
        digest.update(view)
        return written
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "ternary output write failed",
            retriable=True,
        ) from exc
    finally:
        view.release()


def encode_ternary_stream(
    reader: RangeReader,
    source: ByteRange,
    shape: tuple[int, ...],
    source_dtype: DType,
    sink: BinarySink,
    config: TernaryCodecConfig | None = None,
) -> TernaryEncodingResult:
    """Quantize one source tensor without materializing it or its output in full."""
    config = config or TernaryCodecConfig()
    source_dtype = DType(source_dtype)
    if source_dtype not in _SOURCE_ITEM_BYTES:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"ternary source dtype is unsupported: {source_dtype.value}",
        )
    element_count = checked_product(shape, name="ternary.shape")
    checked_positive(element_count, name="ternary.element_count")
    item_bytes = _SOURCE_ITEM_BYTES[source_dtype]
    decoded_bytes = checked_mul(element_count, item_bytes, name="ternary.decoded_bytes")
    if source.length != decoded_bytes:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "ternary source range does not match tensor shape and dtype",
        )
    source.validate_within(reader.size_bytes)
    source_algorithm, source_expected = source.checksum.split(":", 1)
    if source_algorithm not in hashlib.algorithms_available or source_algorithm == "blake3":
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"source hash backend is unavailable: {source_algorithm}",
        )
    source_digest = hashlib.new(source_algorithm)
    output_digest = hashlib.sha256()
    source_buffer = bytearray(config.group_size * item_bytes)
    source_view = memoryview(source_buffer)
    encoded_bytes = 0
    maximum_read = 0
    try:
        completed = 0
        while completed < element_count:
            count = min(config.group_size, element_count - completed)
            byte_count = count * item_bytes
            window = source_view[:byte_count]
            offset = checked_add(
                source.offset,
                checked_mul(completed, item_bytes, name="ternary.source_progress"),
                name="ternary.source_offset",
            )
            reader.read_into(offset, window)
            source_digest.update(window)
            maximum_read = max(maximum_read, byte_count)
            values = _decode_source_group(source_buffer, count, source_dtype)
            encoded_bytes += _write_exact(
                sink,
                encode_ternary_group_reference(values, config),
                output_digest,
            )
            completed += count
    finally:
        source_view.release()
    if source_digest.hexdigest() != source_expected:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "ternary source range hash mismatch")
    expected_encoded = config.encoded_size(element_count)
    if encoded_bytes != expected_encoded:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "ternary encoded byte count is inconsistent")
    return TernaryEncodingResult(
        content_hash="sha256:" + output_digest.hexdigest(),
        source_checksum=source.checksum,
        element_count=element_count,
        group_count=(element_count + config.group_size - 1) // config.group_size,
        encoded_bytes=encoded_bytes,
        decoded_bytes=decoded_bytes,
        maximum_source_read_bytes=maximum_read,
    )


def decode_ternary_reference(
    payload: bytes,
    element_count: int,
    config: TernaryCodecConfig | None = None,
) -> list[float]:
    """Decode a complete test/reference payload with strict canonical padding checks."""
    config = config or TernaryCodecConfig()
    if len(payload) != config.encoded_size(element_count):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "ternary payload length is invalid")
    output: list[float] = []
    cursor = 0
    remaining = element_count
    while remaining:
        count = min(config.group_size, remaining)
        record_size = config.group_record_size(count)
        output.extend(decode_ternary_group_reference(payload[cursor : cursor + record_size], count))
        cursor += record_size
        remaining -= count
    if cursor != len(payload) or len(output) != element_count:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "ternary decoder length mismatch")
    return output


def decode_ternary_group_reference(payload, element_count: int) -> list[float]:
    """Validate and decode one complete v1 group record."""
    checked_positive(element_count, name="ternary.group_element_count")
    expected_size = _SCALE_BYTES + (element_count + _TRITS_PER_BYTE - 1) // _TRITS_PER_BYTE
    view = memoryview(payload).cast("B")
    try:
        if view.nbytes != expected_size:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "ternary group record length is invalid")
        scale = struct.unpack_from("<f", view, 0)[0]
        if not math.isfinite(scale) or scale < 0:
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "ternary scale is invalid")
        output: list[float] = []
        packed_bytes = expected_size - _SCALE_BYTES
        for packed_index in range(packed_bytes):
            packed = view[_SCALE_BYTES + packed_index]
            if packed > 242:
                raise AmsError(ErrorCode.INVALID_PACKAGE, "ternary packed byte exceeds 242")
            for slot in range(_TRITS_PER_BYTE):
                digit = (packed // _TRIT_POWERS[slot]) % 3
                element_index = packed_index * _TRITS_PER_BYTE + slot
                if element_index >= element_count:
                    if digit != 1:
                        raise AmsError(
                            ErrorCode.INVALID_PACKAGE,
                            "ternary tail padding is not canonical zero",
                        )
                    continue
                output.append(-scale if digit == 0 else 0.0 if digit == 1 else scale)
        return output
    finally:
        view.release()


def encode_ternary_bytes_for_test(
    reader: RangeReader,
    source: ByteRange,
    shape: tuple[int, ...],
    source_dtype: DType,
    config: TernaryCodecConfig | None = None,
) -> tuple[bytes, TernaryEncodingResult]:
    """Test helper; production conversion must stream to a bounded file sink."""
    sink = BytesIO()
    result = encode_ternary_stream(reader, source, shape, source_dtype, sink, config)
    return sink.getvalue(), result
