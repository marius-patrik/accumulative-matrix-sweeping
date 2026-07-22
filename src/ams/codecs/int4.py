"""Deterministic grouped symmetric INT4 reference codec.

Format ``ams.int4.symmetric`` version 1.0.0 stores each group as a little-endian FP32
scale followed by two signed four-bit values per byte, low nibble first. Quantized values
are restricted to ``[-7, 7]``; nibble ``8`` is invalid. A tail high nibble must be zero.
The scale is the FP32-rounded group maximum absolute value divided by seven. Quantization
rounds half away from zero against that stored scale and clamps to the supported range.
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

_VALUES_PER_BYTE = 2
_SCALE_BYTES = 4
_SOURCE_ITEM_BYTES = {
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.FLOAT32: 4,
}


class BinarySink(Protocol):
    def write(self, data: bytes | bytearray | memoryview) -> int | None: ...


@dataclass(frozen=True, slots=True)
class Int4CodecConfig:
    group_size: int = 128
    scale_dtype: DType = DType.FLOAT32
    packing: str = "signed-nibble-low-first"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        checked_positive(self.group_size, name="int4.group_size")
        if self.group_size > 65_536:
            raise AmsError(ErrorCode.PLAN_INVALID, "INT4 group size exceeds 65536")
        object.__setattr__(self, "scale_dtype", DType(self.scale_dtype))
        if self.scale_dtype is not DType.FLOAT32:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "INT4 v1 requires FP32 scales")
        if self.packing != "signed-nibble-low-first" or self.version != "1.0.0":
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "unsupported INT4 codec variant")

    @property
    def config_hash(self) -> str:
        payload = {
            "group_size": self.group_size,
            "packing": self.packing,
            "scale_dtype": self.scale_dtype.value,
            "version": self.version,
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def group_record_size(self, element_count: int) -> int:
        checked_positive(element_count, name="int4.group_element_count")
        if element_count > self.group_size:
            raise AmsError(ErrorCode.PLAN_INVALID, "INT4 group exceeds configured group size")
        return _SCALE_BYTES + (element_count + _VALUES_PER_BYTE - 1) // _VALUES_PER_BYTE

    def encoded_size(self, element_count: int) -> int:
        checked_positive(element_count, name="int4.element_count")
        groups = (element_count - 1) // self.group_size + 1
        full_record = self.group_record_size(self.group_size)
        tail = element_count if groups == 1 else element_count - (groups - 1) * self.group_size
        return checked_add(
            checked_mul(max(groups - 1, 0), full_record, name="int4.full_records"),
            self.group_record_size(tail),
            name="int4.encoded_bytes",
        )


@dataclass(frozen=True, slots=True)
class Int4EncodingResult:
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
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, f"unsupported INT4 source dtype: {dtype}")
    if not all(math.isfinite(value) for value in values):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "INT4 source contains NaN or infinity")
    return values


def _quantize(value: float, scale: float) -> int:
    if scale == 0.0:
        return 0
    magnitude = math.floor(abs(value) / scale + 0.5)
    quantized = min(7, magnitude)
    return -quantized if value < 0 else quantized


def _encode_group(values: Sequence[float]) -> bytes:
    maximum = max(abs(value) for value in values)
    scale = struct.unpack("<f", struct.pack("<f", maximum / 7.0))[0] if maximum else 0.0
    quantized = [_quantize(value, scale) for value in values]
    payload = bytearray(struct.pack("<f", scale))
    for start in range(0, len(quantized), _VALUES_PER_BYTE):
        low = quantized[start] & 0xF
        high = quantized[start + 1] & 0xF if start + 1 < len(quantized) else 0
        payload.append(low | (high << 4))
    return bytes(payload)


def _write_exact(sink: BinarySink, payload: bytes, digest) -> int:
    view = memoryview(payload)
    try:
        written = 0
        while written < len(payload):
            count = sink.write(view[written:])
            if count is None or count == 0:
                raise AmsError(
                    ErrorCode.IO_FAILURE, "short write from INT4 encoder", retriable=True
                )
            written += count
        digest.update(view)
        return written
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, "INT4 output write failed", retriable=True) from exc
    finally:
        view.release()


def encode_int4_stream(
    reader: RangeReader,
    source: ByteRange,
    shape: tuple[int, ...],
    source_dtype: DType,
    sink: BinarySink,
    config: Int4CodecConfig | None = None,
) -> Int4EncodingResult:
    """Quantize one source tensor with one source group and one record resident at a time."""
    config = config or Int4CodecConfig()
    source_dtype = DType(source_dtype)
    if source_dtype not in _SOURCE_ITEM_BYTES:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"INT4 source dtype is unsupported: {source_dtype.value}",
        )
    element_count = checked_product(shape, name="int4.shape")
    checked_positive(element_count, name="int4.element_count")
    item_bytes = _SOURCE_ITEM_BYTES[source_dtype]
    decoded_bytes = checked_mul(element_count, item_bytes, name="int4.decoded_bytes")
    if source.length != decoded_bytes:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT4 source range differs from shape and dtype")
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
                checked_mul(completed, item_bytes, name="int4.source_progress"),
                name="int4.source_offset",
            )
            reader.read_into(offset, window)
            source_digest.update(window)
            maximum_read = max(maximum_read, byte_count)
            encoded_bytes += _write_exact(
                sink,
                _encode_group(_decode_source_group(source_buffer, count, source_dtype)),
                output_digest,
            )
            completed += count
    finally:
        source_view.release()
    if source_digest.hexdigest() != source_expected:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "INT4 source range hash mismatch")
    if encoded_bytes != config.encoded_size(element_count):
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 encoded byte count is inconsistent")
    return Int4EncodingResult(
        content_hash="sha256:" + output_digest.hexdigest(),
        source_checksum=source.checksum,
        element_count=element_count,
        group_count=(element_count + config.group_size - 1) // config.group_size,
        encoded_bytes=encoded_bytes,
        decoded_bytes=decoded_bytes,
        maximum_source_read_bytes=maximum_read,
    )


def _signed_nibble(value: int) -> int:
    return value - 16 if value >= 8 else value


def decode_int4_group_reference(payload, element_count: int) -> list[float]:
    """Validate and decode one complete v1 group record."""
    checked_positive(element_count, name="int4.group_element_count")
    expected_size = _SCALE_BYTES + (element_count + 1) // 2
    view = memoryview(payload).cast("B")
    try:
        if view.nbytes != expected_size:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 group record length is invalid")
        scale = struct.unpack_from("<f", view, 0)[0]
        if not math.isfinite(scale) or scale < 0:
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "INT4 scale is invalid")
        output: list[float] = []
        for packed_index, packed in enumerate(view[_SCALE_BYTES:]):
            for slot, nibble in enumerate((packed & 0xF, packed >> 4)):
                index = packed_index * 2 + slot
                if index >= element_count:
                    if nibble != 0:
                        raise AmsError(
                            ErrorCode.INVALID_PACKAGE,
                            "INT4 tail padding is not canonical zero",
                        )
                    continue
                quantized = _signed_nibble(nibble)
                if quantized == -8:
                    raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 reserved value -8 is invalid")
                output.append(scale * quantized)
        return output
    finally:
        view.release()


def decode_int4_reference(
    payload: bytes,
    element_count: int,
    config: Int4CodecConfig | None = None,
) -> list[float]:
    """Decode a complete reference payload with strict length and padding checks."""
    config = config or Int4CodecConfig()
    if len(payload) != config.encoded_size(element_count):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 payload length is invalid")
    output: list[float] = []
    cursor = 0
    remaining = element_count
    while remaining:
        count = min(config.group_size, remaining)
        record_size = config.group_record_size(count)
        output.extend(decode_int4_group_reference(payload[cursor : cursor + record_size], count))
        cursor += record_size
        remaining -= count
    if cursor != len(payload) or len(output) != element_count:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 decoder length mismatch")
    return output


def encode_int4_bytes_for_test(
    reader: RangeReader,
    source: ByteRange,
    shape: tuple[int, ...],
    source_dtype: DType,
    config: Int4CodecConfig | None = None,
) -> tuple[bytes, Int4EncodingResult]:
    sink = BytesIO()
    result = encode_int4_stream(reader, source, shape, source_dtype, sink, config)
    return sink.getvalue(), result
