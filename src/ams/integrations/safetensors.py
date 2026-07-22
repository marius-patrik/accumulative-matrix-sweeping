"""Strict, metadata-only normalization of the safetensors file boundary."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

from ams.checked import (
    checked_add,
    checked_mul,
    checked_product,
    checked_uint,
)
from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode
from ams.storage import RangeReader

_PREFIX_BYTES = 8
_DTYPES: dict[str, tuple[DType, int]] = {
    "BOOL": (DType.BOOL, 1),
    "U8": (DType.UINT8, 1),
    "I8": (DType.INT8, 1),
    "U16": (DType.UINT16, 2),
    "I16": (DType.INT16, 2),
    "U32": (DType.UINT32, 4),
    "I32": (DType.INT32, 4),
    "U64": (DType.UINT64, 8),
    "I64": (DType.INT64, 8),
    "F8_E4M3": (DType.FLOAT8_E4M3FN, 1),
    "F8_E5M2": (DType.FLOAT8_E5M2, 1),
    "F16": (DType.FLOAT16, 2),
    "BF16": (DType.BFLOAT16, 2),
    "F32": (DType.FLOAT32, 4),
    "F64": (DType.FLOAT64, 8),
}


@dataclass(frozen=True, slots=True)
class SafetensorsLimits:
    max_header_bytes: int = 100_000_000
    max_tensors: int = 1_000_000
    max_rank: int = 32
    max_name_bytes: int = 4096
    max_metadata_entries: int = 4096
    max_metadata_value_bytes: int = 65_536

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SafetensorsTensor:
    source_name: str
    dtype: DType
    source_dtype: str
    shape: tuple[int, ...]
    data_offset: int
    data_length: int
    absolute_offset: int


@dataclass(frozen=True, slots=True)
class SafetensorsHeader:
    header_bytes: int
    data_offset: int
    data_bytes: int
    tensors: tuple[SafetensorsTensor, ...]
    metadata: tuple[tuple[str, str], ...]


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _read_exact(reader: RangeReader, offset: int, length: int) -> bytearray:
    destination = bytearray(length)
    reader.read_into(offset, destination)
    return destination


def _parse_json_header(payload: bytes) -> dict[str, Any]:
    if not payload or payload[0] != ord("{"):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors header must begin with '{'")
    try:
        text = payload.decode("utf-8", errors="strict")
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
        value, end = decoder.raw_decode(text)
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors header JSON is invalid") from exc
    if any(character != " " for character in text[end:]):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "safetensors header padding contains non-space bytes",
        )
    if not isinstance(value, dict):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors header must be an object")
    return value


def _normalize_metadata(value: Any, limits: SafetensorsLimits) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict) or len(value) > limits.max_metadata_entries:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors metadata is not a bounded object")
    normalized: list[tuple[str, str]] = []
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise AmsError(
                ErrorCode.INVALID_PACKAGE, "safetensors metadata must map strings to strings"
            )
        if len(key.encode("utf-8")) > limits.max_name_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors metadata key is too long")
        if len(item.encode("utf-8")) > limits.max_metadata_value_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors metadata value is too long")
        normalized.append((key, item))
    return tuple(sorted(normalized))


def _normalize_tensor(
    name: str,
    value: Any,
    *,
    data_offset: int,
    limits: SafetensorsLimits,
) -> SafetensorsTensor:
    if not name or len(name.encode("utf-8")) > limits.max_name_bytes:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors tensor name is empty or too long")
    if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"safetensors tensor metadata has unexpected fields: {name}",
        )
    source_dtype = value["dtype"]
    if not isinstance(source_dtype, str):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"safetensors dtype is not a string: {name}")
    if source_dtype not in _DTYPES:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"safetensors dtype is unsupported: {source_dtype}",
            subsystem="safetensors",
        )
    dtype, item_bytes = _DTYPES[source_dtype]

    raw_shape = value["shape"]
    if not isinstance(raw_shape, list) or len(raw_shape) > limits.max_rank:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"safetensors shape rank is invalid: {name}")
    shape: list[int] = []
    for axis, dimension in enumerate(raw_shape):
        shape.append(checked_uint(dimension, name=f"safetensors.{name}.shape[{axis}]"))

    offsets = value["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"safetensors data offsets are invalid: {name}")
    begin = checked_uint(offsets[0], name=f"safetensors.{name}.begin")
    end = checked_uint(offsets[1], name=f"safetensors.{name}.end")
    if begin > end:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"safetensors data offsets are reversed: {name}")
    length = end - begin
    element_count = checked_product(shape, name=f"safetensors.{name}.elements")
    expected_length = checked_mul(element_count, item_bytes, name=f"safetensors.{name}.bytes")
    if length != expected_length:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"safetensors tensor byte length does not match shape and dtype: {name}",
            evidence={"actual": length, "expected": expected_length},
        )
    return SafetensorsTensor(
        source_name=name,
        dtype=dtype,
        source_dtype=source_dtype,
        shape=tuple(shape),
        data_offset=begin,
        data_length=length,
        absolute_offset=checked_add(data_offset, begin, name=f"safetensors.{name}.absolute"),
    )


def _validate_complete_coverage(tensors: list[SafetensorsTensor], data_bytes: int) -> None:
    nonempty = sorted(
        (tensor for tensor in tensors if tensor.data_length > 0),
        key=lambda tensor: (tensor.data_offset, tensor.source_name),
    )
    cursor = 0
    for tensor in nonempty:
        if tensor.data_offset != cursor:
            relation = "overlap" if tensor.data_offset < cursor else "gap"
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                f"safetensors data buffer contains a tensor {relation}",
            )
        cursor = checked_add(tensor.data_offset, tensor.data_length, name="safetensors.coverage")
    if cursor != data_bytes:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors data buffer is not fully indexed")


def parse_safetensors_header(
    reader: RangeReader,
    limits: SafetensorsLimits | None = None,
) -> SafetensorsHeader:
    """Read and normalize safetensors metadata without importing executable objects."""
    limits = limits or SafetensorsLimits()
    file_size = checked_uint(reader.size_bytes, name="safetensors.file_size")
    if file_size < _PREFIX_BYTES:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors file is shorter than its prefix")
    prefix = _read_exact(reader, 0, _PREFIX_BYTES)
    header_bytes = struct.unpack("<Q", prefix)[0]
    if header_bytes == 0 or header_bytes > limits.max_header_bytes:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "safetensors header length is outside the configured limit",
            evidence={"header_bytes": header_bytes, "limit": limits.max_header_bytes},
        )
    data_offset = checked_add(_PREFIX_BYTES, header_bytes, name="safetensors.data_offset")
    if data_offset > file_size:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors header exceeds the file")
    header = _parse_json_header(_read_exact(reader, _PREFIX_BYTES, header_bytes))
    metadata = _normalize_metadata(header.pop("__metadata__", None), limits)
    if len(header) > limits.max_tensors:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors tensor count exceeds the limit")
    tensors = [
        _normalize_tensor(name, value, data_offset=data_offset, limits=limits)
        for name, value in header.items()
    ]
    data_bytes = file_size - data_offset
    for tensor in tensors:
        if (
            checked_add(
                tensor.data_offset,
                tensor.data_length,
                name=f"safetensors.{tensor.source_name}.end",
            )
            > data_bytes
        ):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "safetensors tensor exceeds the data buffer")
    _validate_complete_coverage(tensors, data_bytes)
    return SafetensorsHeader(
        header_bytes=header_bytes,
        data_offset=data_offset,
        data_bytes=data_bytes,
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.source_name)),
        metadata=metadata,
    )
