"""Immutable descriptors for AMS package storage and conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from ams.checked import checked_add, checked_positive, checked_range_end, checked_uint
from ams.errors import AmsError, ErrorCode

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,511}$")
_DIGEST = re.compile(r"^(sha256|sha512|blake3):([0-9a-f]+)$")
_DIGEST_LENGTH = {"sha256": 64, "sha512": 128, "blake3": 64}


def validate_identifier(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} is not a valid AMS identifier")
    return value


def validate_digest(value: str, *, name: str) -> str:
    match = _DIGEST.fullmatch(value) if isinstance(value, str) else None
    if match is None or len(match.group(2)) != _DIGEST_LENGTH[match.group(1)]:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} is not a supported content digest")
    return value


def validate_semver(value: str, *, name: str) -> str:
    pattern = (
        r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
        r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
    )
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} is not semantic version syntax")
    return value


class DType(StrEnum):
    BOOL = "bool"
    UINT8 = "uint8"
    INT8 = "int8"
    UINT16 = "uint16"
    INT16 = "int16"
    UINT32 = "uint32"
    INT32 = "int32"
    UINT64 = "uint64"
    INT64 = "int64"
    FLOAT8_E4M3FN = "float8_e4m3fn"
    FLOAT8_E5M2 = "float8_e5m2"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    COMPLEX64 = "complex64"
    COMPLEX128 = "complex128"
    INT2_GROUPED = "int2_grouped"
    INT3_GROUPED = "int3_grouped"
    INT4_GROUPED = "int4_grouped"
    INT8_GROUPED = "int8_grouped"
    CUSTOM = "custom"


class QuantizationKind(StrEnum):
    TERNARY = "ternary"
    INT2 = "int2"
    INT3 = "int3"
    INT4 = "int4"
    INT8 = "int8"


class JournalEntryState(StrEnum):
    PLANNED = "planned"
    WRITTEN = "written"
    VERIFIED = "verified"
    PUBLISHED = "published"


@dataclass(frozen=True, slots=True)
class ByteRange:
    object_id: str
    offset: int
    length: int
    checksum: str

    def __post_init__(self) -> None:
        validate_identifier(self.object_id, name="byte_range.object_id")
        checked_range_end(self.offset, self.length, name="byte_range")
        validate_digest(self.checksum, name="byte_range.checksum")

    @property
    def end(self) -> int:
        return checked_range_end(self.offset, self.length, name="byte_range")

    def validate_within(self, size_bytes: int) -> None:
        checked_positive(size_bytes, name="storage_object.size_bytes")
        if self.end > size_bytes:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "byte range exceeds its storage object",
                evidence={"range_end": self.end, "object_size": size_bytes},
            )


@dataclass(frozen=True, slots=True)
class StorageObject:
    object_id: str
    uri: str
    size_bytes: int
    alignment_bytes: int
    content_hash: str
    immutable: bool = True
    kind: str = "file"

    def __post_init__(self) -> None:
        validate_identifier(self.object_id, name="storage_object.object_id")
        if not isinstance(self.uri, str) or not 1 <= len(self.uri) <= 8192:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "storage_object.uri is invalid")
        checked_positive(self.size_bytes, name="storage_object.size_bytes")
        checked_positive(self.alignment_bytes, name="storage_object.alignment_bytes")
        validate_digest(self.content_hash, name="storage_object.content_hash")
        validate_identifier(self.kind, name="storage_object.kind")


@dataclass(frozen=True, slots=True)
class CodecDescriptor:
    name: str
    version: str
    lossless: bool
    max_decoded_bytes: int

    def __post_init__(self) -> None:
        validate_identifier(self.name, name="codec.name")
        validate_semver(self.version, name="codec.version")
        checked_positive(self.max_decoded_bytes, name="codec.max_decoded_bytes")


@dataclass(frozen=True, slots=True)
class QuantizationSpec:
    kind: QuantizationKind
    group_size: int
    axis: int
    scale_dtype: DType = DType.FLOAT16
    zero_point: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", QuantizationKind(self.kind))
        object.__setattr__(self, "scale_dtype", DType(self.scale_dtype))
        checked_positive(self.group_size, name="quantization.group_size")
        checked_uint(self.axis, name="quantization.axis")
        if self.scale_dtype not in {DType.FLOAT16, DType.FLOAT32}:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "quantization scales must be float16 or float32",
            )
        if self.kind is QuantizationKind.TERNARY and self.zero_point:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "symmetric ternary encoding cannot declare a zero point",
            )


@dataclass(frozen=True, slots=True)
class ChunkDescriptor:
    chunk_id: str
    byte_range: ByteRange
    logical_origin: tuple[int, ...]
    logical_extent: tuple[int, ...]
    encoded_bytes: int
    decoded_bytes: int
    padding_bytes: int = 0

    def __post_init__(self) -> None:
        validate_identifier(self.chunk_id, name="chunk.chunk_id")
        if len(self.logical_origin) != len(self.logical_extent):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "chunk origin and extent ranks differ")
        if len(self.logical_origin) > 32:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "chunk rank exceeds 32")
        for index, value in enumerate(self.logical_origin):
            checked_uint(value, name=f"chunk.logical_origin[{index}]")
        for index, value in enumerate(self.logical_extent):
            checked_positive(value, name=f"chunk.logical_extent[{index}]")
        checked_positive(self.encoded_bytes, name="chunk.encoded_bytes")
        checked_positive(self.decoded_bytes, name="chunk.decoded_bytes")
        checked_uint(self.padding_bytes, name="chunk.padding_bytes")
        occupied = checked_add(
            self.encoded_bytes,
            self.padding_bytes,
            name="chunk.occupied_bytes",
        )
        if occupied != self.byte_range.length:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "chunk encoded and padding bytes do not equal its byte range length",
            )


@dataclass(frozen=True, slots=True)
class TensorLayout:
    layout_id: str
    layout_version: str
    complete: bool
    tile_shape: tuple[int, ...]
    alignment_bytes: int
    storage_dtype: DType
    chunks: tuple[ChunkDescriptor, ...]
    codec: CodecDescriptor | None = None
    quantization: QuantizationSpec | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.layout_id, name="layout.layout_id")
        validate_semver(self.layout_version, name="layout.layout_version")
        object.__setattr__(self, "storage_dtype", DType(self.storage_dtype))
        checked_positive(self.alignment_bytes, name="layout.alignment_bytes")
        if len(self.tile_shape) > 32:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "layout tile rank exceeds 32")
        for index, value in enumerate(self.tile_shape):
            checked_positive(value, name=f"layout.tile_shape[{index}]")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "layout chunk IDs are not unique")


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    tensor_id: str
    tensor_class: str
    shape: tuple[int, ...]
    logical_dtype: DType
    immutable: bool
    layouts: tuple[TensorLayout, ...]
    byte_order: str = "little"

    def __post_init__(self) -> None:
        validate_identifier(self.tensor_id, name="tensor.tensor_id")
        validate_identifier(self.tensor_class, name="tensor.tensor_class")
        object.__setattr__(self, "logical_dtype", DType(self.logical_dtype))
        if self.byte_order not in {"little", "big", "not_applicable"}:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor.byte_order is invalid")
        if len(self.shape) > 32:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor rank exceeds 32")
        for index, value in enumerate(self.shape):
            checked_uint(value, name=f"tensor.shape[{index}]")
        if not self.layouts:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor must declare at least one layout")
        layout_ids = [layout.layout_id for layout in self.layouts]
        if len(set(layout_ids)) != len(layout_ids):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor layout IDs are not unique")
        for layout in self.layouts:
            if len(layout.tile_shape) != len(self.shape):
                raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor and tile ranks differ")
            for chunk in layout.chunks:
                if len(chunk.logical_origin) != len(self.shape):
                    raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor and chunk ranks differ")
                for axis, (origin, extent, size) in enumerate(
                    zip(chunk.logical_origin, chunk.logical_extent, self.shape, strict=True)
                ):
                    if checked_add(origin, extent, name=f"chunk.axis[{axis}].end") > size:
                        raise AmsError(ErrorCode.INVALID_PACKAGE, "chunk exceeds tensor shape")


@dataclass(frozen=True, slots=True)
class ConversionJournalEntry:
    source_range: ByteRange
    target_chunk_id: str
    state: JournalEntryState
    target_hash: str | None = None
    encoded_bytes: int | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.target_chunk_id, name="journal.target_chunk_id")
        object.__setattr__(self, "state", JournalEntryState(self.state))
        completed = self.state is not JournalEntryState.PLANNED
        if completed != (self.target_hash is not None and self.encoded_bytes is not None):
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "written journal entries require target hash and encoded byte count",
            )
        if self.target_hash is not None:
            validate_digest(self.target_hash, name="journal.target_hash")
        if self.encoded_bytes is not None:
            checked_positive(self.encoded_bytes, name="journal.encoded_bytes")


@dataclass(frozen=True, slots=True)
class ConversionJournal:
    journal_version: str
    source_root: str
    configuration_hash: str
    entries: tuple[ConversionJournalEntry, ...]

    def __post_init__(self) -> None:
        validate_semver(self.journal_version, name="journal.version")
        validate_digest(self.source_root, name="journal.source_root")
        validate_digest(self.configuration_hash, name="journal.configuration_hash")
        chunk_ids = [entry.target_chunk_id for entry in self.entries]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "journal target chunk IDs are not unique")
