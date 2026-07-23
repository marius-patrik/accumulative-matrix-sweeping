"""Diagnostic grouped symmetric INT3 reference codec.

This exact group format exists to decide whether a production package/native codec is justified.
It stores one little-endian FP32 scale followed by signed three-bit values in an LSB-first
bitstream.
Values are restricted to ``[-3, 3]``; code ``4`` (signed ``-4``) is reserved. Unused high tail bits
must be zero. The scale is the FP32-rounded group maximum absolute value divided by three, and
quantization rounds half away from zero against that stored scale.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from ams.canonical import canonical_json_bytes
from ams.checked import checked_positive
from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode

_BITS_PER_VALUE = 3
_SCALE_BYTES = 4


@dataclass(frozen=True, slots=True)
class Int3DiagnosticCodecConfig:
    """Exact diagnostic contract; not yet a package-policy capability."""

    group_size: int = 128
    scale_dtype: DType = DType.FLOAT32
    packing: str = "signed-3bit-lsb-first"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        checked_positive(self.group_size, name="int3.group_size")
        if self.group_size > 65_536:
            raise AmsError(ErrorCode.PLAN_INVALID, "INT3 group size exceeds 65536")
        object.__setattr__(self, "scale_dtype", DType(self.scale_dtype))
        if self.scale_dtype is not DType.FLOAT32:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "INT3 diagnostic v1 requires FP32 scales")
        if self.packing != "signed-3bit-lsb-first" or self.version != "1.0.0":
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "unsupported INT3 diagnostic codec variant",
            )

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
        checked_positive(element_count, name="int3.group_element_count")
        if element_count > self.group_size:
            raise AmsError(ErrorCode.PLAN_INVALID, "INT3 group exceeds configured group size")
        return _SCALE_BYTES + (element_count * _BITS_PER_VALUE + 7) // 8

    def encoded_size(self, element_count: int) -> int:
        checked_positive(element_count, name="int3.element_count")
        groups = (element_count - 1) // self.group_size + 1
        full_record = self.group_record_size(self.group_size)
        tail = element_count if groups == 1 else element_count - (groups - 1) * self.group_size
        return (groups - 1) * full_record + self.group_record_size(tail)


def _quantize(value: float, scale: float) -> int:
    if scale == 0.0:
        return 0
    magnitude = math.floor(abs(value) / scale + 0.5)
    quantized = min(3, magnitude)
    return -quantized if value < 0 else quantized


def encode_int3_group_reference(
    values: Sequence[float],
    config: Int3DiagnosticCodecConfig | None = None,
) -> bytes:
    """Validate and encode one diagnostic INT3 group."""

    config = config or Int3DiagnosticCodecConfig()
    try:
        element_count = len(values)
    except TypeError as exc:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT3 group values must be a sequence") from exc
    checked_positive(element_count, name="int3.group_element_count")
    if element_count > config.group_size:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT3 group exceeds configured group size")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AmsError(ErrorCode.PLAN_INVALID, "INT3 group values must be numeric")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise AmsError(
                ErrorCode.NUMERIC_FAILURE,
                "INT3 source contains an unrepresentable numeric value",
            ) from exc
        if not finite:
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "INT3 source contains NaN or infinity")

    maximum = max(abs(value) for value in values)
    scale = struct.unpack("<f", struct.pack("<f", maximum / 3.0))[0] if maximum else 0.0
    packed_bytes = (element_count * _BITS_PER_VALUE + 7) // 8
    packed = bytearray(packed_bytes)
    for index, value in enumerate(values):
        code = _quantize(value, scale) & 0x7
        bit_offset = index * _BITS_PER_VALUE
        byte_index, shift = divmod(bit_offset, 8)
        packed[byte_index] |= (code << shift) & 0xFF
        if shift > 5:
            packed[byte_index + 1] |= code >> (8 - shift)
    return struct.pack("<f", scale) + packed


def decode_int3_group_reference(payload, element_count: int) -> list[float]:
    """Validate and decode one complete diagnostic INT3 group record."""

    checked_positive(element_count, name="int3.group_element_count")
    expected_size = _SCALE_BYTES + (element_count * _BITS_PER_VALUE + 7) // 8
    view = memoryview(payload).cast("B")
    try:
        if view.nbytes != expected_size:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "INT3 group record length is invalid")
        scale = struct.unpack_from("<f", view, 0)[0]
        if not math.isfinite(scale) or scale < 0:
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "INT3 scale is invalid")
        used_tail_bits = (element_count * _BITS_PER_VALUE) % 8
        if used_tail_bits and view[-1] >> used_tail_bits:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "INT3 tail padding is not canonical zero",
            )
        output = []
        packed = view[_SCALE_BYTES:]
        for index in range(element_count):
            bit_offset = index * _BITS_PER_VALUE
            byte_index, shift = divmod(bit_offset, 8)
            word = packed[byte_index]
            if shift > 5:
                word |= packed[byte_index + 1] << 8
            code = (word >> shift) & 0x7
            quantized = code - 8 if code >= 4 else code
            if quantized == -4:
                raise AmsError(ErrorCode.INVALID_PACKAGE, "INT3 reserved value -4 is invalid")
            output.append(scale * quantized)
        return output
    finally:
        view.release()
