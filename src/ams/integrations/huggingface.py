"""Strict normalization of Hugging Face sharded safetensors repositories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_product, checked_uint
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.conversion import ConversionItem, ConversionPlan
from ams.descriptors import ByteRange, DType, validate_digest, validate_identifier
from ams.errors import AmsError, ErrorCode
from ams.integrations.safetensors import SafetensorsHeader, parse_safetensors_header
from ams.storage import RangeReader, hash_reader_range


@dataclass(frozen=True, slots=True)
class HuggingFaceIndexLimits:
    max_index_bytes: int = 64 * 1024 * 1024
    max_tensors: int = 1_000_000
    max_shards: int = 100_000
    max_name_bytes: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AmsError(ErrorCode.PLAN_INVALID, f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class HuggingFaceShardIndexEntry:
    tensor_name: str
    shard_name: str


@dataclass(frozen=True, slots=True)
class HuggingFaceShardIndex:
    total_size: int
    content_hash: str
    metadata_hash: str
    entries: tuple[HuggingFaceShardIndexEntry, ...]
    shard_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HuggingFaceShardSource:
    shard_name: str
    object_id: str
    content_hash: str
    reader: RangeReader

    def __post_init__(self) -> None:
        _validate_shard_name(self.shard_name, max_bytes=4096)
        validate_identifier(self.object_id, name="huggingface.object_id")
        validate_digest(self.content_hash, name="huggingface.content_hash")


@dataclass(frozen=True, slots=True)
class HuggingFaceCatalogTensor:
    tensor_name: str
    shard_name: str
    object_id: str
    dtype: DType
    source_dtype: str
    shape: tuple[int, ...]
    source_offset: int
    source_length: int


@dataclass(frozen=True, slots=True)
class HuggingFaceCatalog:
    source_root: str
    index_content_hash: str
    index_metadata_hash: str
    total_size: int
    tensors: tuple[HuggingFaceCatalogTensor, ...]
    sources: tuple[HuggingFaceShardSource, ...]


@dataclass(frozen=True, slots=True)
class HuggingFaceHeaderAudit:
    """Structural evidence obtained without reading or hashing tensor payloads."""

    index_content_hash: str
    shard_count: int
    tensor_count: int
    declared_total_size: int
    tensor_elements: int
    tensor_bytes: int
    source_file_bytes: int
    prefix_and_header_bytes: int
    dtype_counts: tuple[tuple[str, int], ...]


class HuggingFaceTotalSizeSemantics(StrEnum):
    TENSOR_BYTES = "tensor_bytes"
    TENSOR_ELEMENTS = "tensor_elements"


@dataclass(frozen=True, slots=True)
class HuggingFaceCatalogPolicy:
    """Interpret provider metadata, pinning every nonstandard interpretation to one index."""

    total_size_semantics: HuggingFaceTotalSizeSemantics = HuggingFaceTotalSizeSemantics.TENSOR_BYTES
    expected_index_content_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "total_size_semantics",
            HuggingFaceTotalSizeSemantics(self.total_size_semantics),
        )
        if self.total_size_semantics is HuggingFaceTotalSizeSemantics.TENSOR_BYTES:
            if self.expected_index_content_hash is not None:
                raise AmsError(
                    ErrorCode.PLAN_INVALID,
                    "standard Hugging Face size semantics cannot carry an exception pin",
                )
            return
        if self.expected_index_content_hash is None:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "nonstandard Hugging Face size semantics require an exact index hash",
            )
        validate_digest(
            self.expected_index_content_hash,
            name="huggingface.expected_index_content_hash",
        )


@dataclass(frozen=True, slots=True)
class HuggingFaceHeaderCatalog:
    """An exact structural catalog whose expected shard hashes are not yet payload-verified."""

    source_root: str
    index_content_hash: str
    index_metadata_hash: str
    total_size: int
    tensors: tuple[HuggingFaceCatalogTensor, ...]
    sources: tuple[HuggingFaceShardSource, ...]
    audit: HuggingFaceHeaderAudit


@dataclass(frozen=True, slots=True)
class HuggingFacePlannedTensor:
    tensor: HuggingFaceCatalogTensor
    target_chunk_id: str | None
    source_checksum: str


@dataclass(frozen=True, slots=True)
class HuggingFaceIdentityPlan:
    conversion: ConversionPlan
    tensors: tuple[HuggingFacePlannedTensor, ...]


class HuggingFaceTensorEncoding(StrEnum):
    IDENTITY = "identity"
    TERNARY_TRIT5 = "ternary_trit5"
    INT4_SYMMETRIC = "int4_symmetric"


def _codec_config_identity_fields(
    encoding: HuggingFaceTensorEncoding,
    ternary_config: TernaryCodecConfig | None,
    int4_config: Int4CodecConfig | None,
) -> dict[str, str]:
    """Enforce and identify the one codec configuration selected by an encoding."""
    if ternary_config is not None and not isinstance(ternary_config, TernaryCodecConfig):
        raise AmsError(ErrorCode.PLAN_INVALID, "ternary config has the wrong type")
    if int4_config is not None and not isinstance(int4_config, Int4CodecConfig):
        raise AmsError(ErrorCode.PLAN_INVALID, "INT4 config has the wrong type")
    if encoding is HuggingFaceTensorEncoding.IDENTITY:
        if ternary_config is not None or int4_config is not None:
            raise AmsError(ErrorCode.PLAN_INVALID, "identity assignment cannot have codec config")
        return {}
    if encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
        if ternary_config is None or int4_config is not None:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "ternary assignment requires only a ternary codec config",
            )
        return {"ternary_config_hash": ternary_config.config_hash}
    if encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC:
        if int4_config is None or ternary_config is not None:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "INT4 assignment requires only an INT4 codec config",
            )
        return {"int4_config_hash": int4_config.config_hash}
    raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unknown Hugging Face tensor encoding")


@dataclass(frozen=True, slots=True)
class HuggingFaceTensorAssignment:
    tensor_name: str
    encoding: HuggingFaceTensorEncoding
    ternary_config: TernaryCodecConfig | None = None
    int4_config: Int4CodecConfig | None = None

    def __post_init__(self) -> None:
        _validate_external_name(self.tensor_name, field="tensor name", max_bytes=4096)
        object.__setattr__(self, "encoding", HuggingFaceTensorEncoding(self.encoding))
        _codec_config_identity_fields(self.encoding, self.ternary_config, self.int4_config)


@dataclass(frozen=True, slots=True)
class HuggingFaceMixedPolicy:
    policy_hash: str
    assignments: tuple[HuggingFaceTensorAssignment, ...]


@dataclass(frozen=True, slots=True)
class HuggingFaceProgressiveShardPlan:
    shard_name: str
    object_id: str
    content_hash: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class HuggingFaceProgressiveTensorPlan:
    tensor: HuggingFaceCatalogTensor
    target_chunk_id: str
    encoding: HuggingFaceTensorEncoding
    ternary_config: TernaryCodecConfig | None = None
    int4_config: Int4CodecConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoding", HuggingFaceTensorEncoding(self.encoding))
        _codec_config_identity_fields(self.encoding, self.ternary_config, self.int4_config)


@dataclass(frozen=True, slots=True)
class HuggingFaceProgressiveMixedPlan:
    source_root: str
    index_content_hash: str
    index_metadata_hash: str
    policy_hash: str
    plan_hash: str
    total_size: int
    shards: tuple[HuggingFaceProgressiveShardPlan, ...]
    tensors: tuple[HuggingFaceProgressiveTensorPlan, ...]


@dataclass(frozen=True, slots=True)
class HuggingFaceMixedPlannedTensor:
    tensor: HuggingFaceCatalogTensor
    target_chunk_id: str
    source_checksum: str
    encoding: HuggingFaceTensorEncoding
    ternary_config: TernaryCodecConfig | None = None
    int4_config: Int4CodecConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "encoding", HuggingFaceTensorEncoding(self.encoding))
        _codec_config_identity_fields(self.encoding, self.ternary_config, self.int4_config)


@dataclass(frozen=True, slots=True)
class HuggingFaceMixedPlan:
    conversion: ConversionPlan
    policy_hash: str
    tensors: tuple[HuggingFaceMixedPlannedTensor, ...]


class _DuplicateIndexKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateIndexKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_external_name(value: Any, *, field: str, max_bytes: int) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > max_bytes:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"Hugging Face {field} is empty or too long")
    return value


def _validate_shard_name(value: Any, *, max_bytes: int) -> str:
    name = _validate_external_name(value, field="shard name", max_bytes=max_bytes)
    if (
        name in {".", ".."}
        or "/" in name
        or "\\" in name
        or PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
        or not name.endswith(".safetensors")
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face shard name is unsafe")
    return name


def parse_huggingface_shard_index(
    payload: bytes,
    limits: HuggingFaceIndexLimits | None = None,
) -> HuggingFaceShardIndex:
    """Normalize a bounded `model.safetensors.index.json` payload."""
    limits = limits or HuggingFaceIndexLimits()
    if not payload or len(payload) > limits.max_index_bytes:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face shard index size is invalid")
    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateIndexKey, ValueError) as exc:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE, "Hugging Face shard index JSON is invalid"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {"metadata", "weight_map"}:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face shard index fields are invalid")
    metadata = raw["metadata"]
    weight_map = raw["weight_map"]
    if not isinstance(metadata, dict) or "total_size" not in metadata:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face index metadata is invalid")
    total_size = checked_uint(metadata["total_size"], name="huggingface.total_size")
    if not isinstance(weight_map, dict) or not 1 <= len(weight_map) <= limits.max_tensors:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face weight map is invalid")
    entries: list[HuggingFaceShardIndexEntry] = []
    for tensor_name, shard_name in weight_map.items():
        entries.append(
            HuggingFaceShardIndexEntry(
                tensor_name=_validate_external_name(
                    tensor_name,
                    field="tensor name",
                    max_bytes=limits.max_name_bytes,
                ),
                shard_name=_validate_shard_name(
                    shard_name,
                    max_bytes=limits.max_name_bytes,
                ),
            )
        )
    shard_names = tuple(sorted({entry.shard_name for entry in entries}))
    if len(shard_names) > limits.max_shards:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face shard count exceeds the limit")
    return HuggingFaceShardIndex(
        total_size=total_size,
        content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        metadata_hash="sha256:" + hashlib.sha256(canonical_json_bytes(metadata)).hexdigest(),
        entries=tuple(sorted(entries, key=lambda entry: entry.tensor_name)),
        shard_names=shard_names,
    )


def _verify_shard(source: HuggingFaceShardSource, *, buffer_bytes: int) -> SafetensorsHeader:
    algorithm, _ = source.content_hash.split(":", 1)
    actual = hash_reader_range(
        source.reader,
        0,
        source.reader.size_bytes,
        buffer_bytes=buffer_bytes,
        algorithm=algorithm,
    )
    if actual != source.content_hash:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            f"Hugging Face shard hash mismatch: {source.shard_name}",
        )
    return parse_safetensors_header(source.reader)


def _inspect_huggingface_headers(
    index: HuggingFaceShardIndex,
    sources: tuple[HuggingFaceShardSource, ...],
    *,
    verify_hashes: bool,
    buffer_bytes: int,
) -> tuple[tuple[HuggingFaceCatalogTensor, ...], HuggingFaceHeaderAudit]:
    source_by_name = {source.shard_name: source for source in sources}
    if len(source_by_name) != len(sources) or set(source_by_name) != set(index.shard_names):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face shard source set is incomplete")
    index_by_tensor = {entry.tensor_name: entry.shard_name for entry in index.entries}
    tensors: list[HuggingFaceCatalogTensor] = []
    observed_names: set[str] = set()
    tensor_bytes = 0
    tensor_elements = 0
    source_file_bytes = 0
    prefix_and_header_bytes = 0
    dtype_counts: dict[str, int] = {}
    for shard_name in index.shard_names:
        source = source_by_name[shard_name]
        header = (
            _verify_shard(source, buffer_bytes=buffer_bytes)
            if verify_hashes
            else parse_safetensors_header(source.reader)
        )
        source_file_bytes = checked_add(
            source_file_bytes,
            source.reader.size_bytes,
            name="huggingface.source_file_bytes",
        )
        prefix_and_header_bytes = checked_add(
            prefix_and_header_bytes,
            header.data_offset,
            name="huggingface.prefix_and_header_bytes",
        )
        for tensor in header.tensors:
            if tensor.source_name in observed_names:
                raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor appears in more than one shard")
            if index_by_tensor.get(tensor.source_name) != shard_name:
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "Hugging Face index and shard tensor mapping disagree",
                )
            observed_names.add(tensor.source_name)
            tensor_bytes = checked_add(
                tensor_bytes,
                tensor.data_length,
                name="huggingface.tensor_bytes",
            )
            tensor_elements = checked_add(
                tensor_elements,
                checked_product(
                    tensor.shape,
                    name=f"huggingface.{tensor.source_name}.elements",
                ),
                name="huggingface.tensor_elements",
            )
            dtype_counts[tensor.source_dtype] = dtype_counts.get(tensor.source_dtype, 0) + 1
            tensors.append(
                HuggingFaceCatalogTensor(
                    tensor_name=tensor.source_name,
                    shard_name=shard_name,
                    object_id=source.object_id,
                    dtype=tensor.dtype,
                    source_dtype=tensor.source_dtype,
                    shape=tensor.shape,
                    source_offset=tensor.absolute_offset,
                    source_length=tensor.data_length,
                )
            )
    if observed_names != set(index_by_tensor):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face index references absent tensors")
    normalized_tensors = tuple(sorted(tensors, key=lambda tensor: tensor.tensor_name))
    audit = HuggingFaceHeaderAudit(
        index_content_hash=index.content_hash,
        shard_count=len(sources),
        tensor_count=len(normalized_tensors),
        declared_total_size=index.total_size,
        tensor_elements=tensor_elements,
        tensor_bytes=tensor_bytes,
        source_file_bytes=source_file_bytes,
        prefix_and_header_bytes=prefix_and_header_bytes,
        dtype_counts=tuple(sorted(dtype_counts.items())),
    )
    return normalized_tensors, audit


def audit_huggingface_headers(
    index: HuggingFaceShardIndex,
    sources: tuple[HuggingFaceShardSource, ...],
) -> HuggingFaceHeaderAudit:
    """Cross-check every shard header and index mapping without claiming payload integrity."""
    _, audit = _inspect_huggingface_headers(
        index,
        sources,
        verify_hashes=False,
        buffer_bytes=1,
    )
    return audit


def _preflight_total_size_semantics(
    index: HuggingFaceShardIndex,
    policy: HuggingFaceCatalogPolicy,
) -> None:
    if (
        policy.total_size_semantics is HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS
        and policy.expected_index_content_hash != index.content_hash
    ):
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "Hugging Face index does not match the pinned size-semantics exception",
        )


def _validate_total_size_semantics(
    index: HuggingFaceShardIndex,
    audit: HuggingFaceHeaderAudit,
    policy: HuggingFaceCatalogPolicy,
) -> None:
    observed = (
        audit.tensor_elements
        if policy.total_size_semantics is HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS
        else audit.tensor_bytes
    )
    if observed != index.total_size:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "Hugging Face total_size does not match tensor storage",
            evidence={"actual": observed, "declared": index.total_size},
        )


def _huggingface_source_root(
    index: HuggingFaceShardIndex,
    sources: tuple[HuggingFaceShardSource, ...],
) -> str:
    payload = {
        "index": index.content_hash,
        "shards": [
            {"name": source.shard_name, "content_hash": source.content_hash}
            for source in sorted(sources, key=lambda item: item.shard_name)
        ],
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_huggingface_header_catalog(
    index: HuggingFaceShardIndex,
    sources: tuple[HuggingFaceShardSource, ...],
    *,
    policy: HuggingFaceCatalogPolicy | None = None,
) -> HuggingFaceHeaderCatalog:
    """Build a structural catalog without reading or claiming integrity for tensor payloads."""
    policy = policy or HuggingFaceCatalogPolicy()
    _preflight_total_size_semantics(index, policy)
    tensors, audit = _inspect_huggingface_headers(
        index,
        sources,
        verify_hashes=False,
        buffer_bytes=1,
    )
    _validate_total_size_semantics(index, audit, policy)
    return HuggingFaceHeaderCatalog(
        source_root=_huggingface_source_root(index, sources),
        index_content_hash=index.content_hash,
        index_metadata_hash=index.metadata_hash,
        total_size=audit.tensor_bytes,
        tensors=tensors,
        sources=tuple(sorted(sources, key=lambda source: source.shard_name)),
        audit=audit,
    )


def build_huggingface_catalog(
    index: HuggingFaceShardIndex,
    sources: tuple[HuggingFaceShardSource, ...],
    *,
    buffer_bytes: int = 1024 * 1024,
    policy: HuggingFaceCatalogPolicy | None = None,
) -> HuggingFaceCatalog:
    """Cross-check the provider index, immutable shard hashes, and normalized headers."""
    policy = policy or HuggingFaceCatalogPolicy()
    _preflight_total_size_semantics(index, policy)
    tensors, audit = _inspect_huggingface_headers(
        index,
        sources,
        verify_hashes=True,
        buffer_bytes=buffer_bytes,
    )
    _validate_total_size_semantics(index, audit, policy)
    return HuggingFaceCatalog(
        source_root=_huggingface_source_root(index, sources),
        index_content_hash=index.content_hash,
        index_metadata_hash=index.metadata_hash,
        total_size=audit.tensor_bytes,
        tensors=tensors,
        sources=tuple(sorted(sources, key=lambda source: source.shard_name)),
    )


def build_huggingface_shard_catalog(
    index: HuggingFaceShardIndex,
    source: HuggingFaceShardSource,
    *,
    buffer_bytes: int = 1024 * 1024,
) -> HuggingFaceCatalog:
    """Authenticate and catalog one exact shard while retaining the full-index identity."""
    expected_names = {
        entry.tensor_name for entry in index.entries if entry.shard_name == source.shard_name
    }
    if source.shard_name not in index.shard_names or not expected_names:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "Hugging Face shard is absent from the full index",
        )
    header = _verify_shard(source, buffer_bytes=buffer_bytes)
    observed_names = {tensor.source_name for tensor in header.tensors}
    if observed_names != expected_names:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "Hugging Face shard header does not match its full-index tensor set",
            evidence={
                "missing": len(expected_names - observed_names),
                "unexpected": len(observed_names - expected_names),
            },
        )
    tensors = tuple(
        sorted(
            (
                HuggingFaceCatalogTensor(
                    tensor_name=tensor.source_name,
                    shard_name=source.shard_name,
                    object_id=source.object_id,
                    dtype=tensor.dtype,
                    source_dtype=tensor.source_dtype,
                    shape=tensor.shape,
                    source_offset=tensor.absolute_offset,
                    source_length=tensor.data_length,
                )
                for tensor in header.tensors
            ),
            key=lambda tensor: tensor.tensor_name,
        )
    )
    total_size = 0
    for tensor in tensors:
        total_size = checked_add(
            total_size,
            tensor.source_length,
            name="huggingface.shard_catalog.total_size",
        )
    return HuggingFaceCatalog(
        source_root=_huggingface_source_root(index, (source,)),
        index_content_hash=index.content_hash,
        index_metadata_hash=index.metadata_hash,
        total_size=total_size,
        tensors=tensors,
        sources=(source,),
    )


def build_huggingface_identity_plan(
    catalog: HuggingFaceCatalog,
    configuration_hash: str,
    *,
    buffer_bytes: int = 1024 * 1024,
) -> HuggingFaceIdentityPlan:
    """Pre-hash each source tensor and construct a deterministic multi-shard copy plan."""
    validate_digest(configuration_hash, name="conversion.configuration_hash")
    source_by_id = {source.object_id: source for source in catalog.sources}
    if len(source_by_id) != len(catalog.sources):
        raise AmsError(ErrorCode.PLAN_INVALID, "Hugging Face source object IDs are not unique")
    planned: list[HuggingFacePlannedTensor] = []
    items: list[ConversionItem] = []
    for tensor in catalog.tensors:
        checksum = _hash_catalog_tensor(source_by_id, tensor, buffer_bytes=buffer_bytes)
        target_chunk_id = None
        if tensor.source_length > 0:
            target_chunk_id = (
                "tensor:" + hashlib.sha256(tensor.tensor_name.encode("utf-8")).hexdigest()
            )
            items.append(
                ConversionItem(
                    target_chunk_id=target_chunk_id,
                    source_range=ByteRange(
                        object_id=tensor.object_id,
                        offset=tensor.source_offset,
                        length=tensor.source_length,
                        checksum=checksum,
                    ),
                )
            )
        planned.append(
            HuggingFacePlannedTensor(
                tensor=tensor,
                target_chunk_id=target_chunk_id,
                source_checksum=checksum,
            )
        )
    if not items:
        raise AmsError(ErrorCode.PLAN_INVALID, "Hugging Face model contains no tensor payload")
    return HuggingFaceIdentityPlan(
        conversion=ConversionPlan(
            source_root=catalog.source_root,
            configuration_hash=configuration_hash,
            items=tuple(items),
        ),
        tensors=tuple(planned),
    )


def _hash_catalog_tensor(
    source_by_id: dict[str, HuggingFaceShardSource],
    tensor: HuggingFaceCatalogTensor,
    *,
    buffer_bytes: int,
) -> str:
    """Apply the one canonical checked range-hash rule used by all HF planners."""
    source = source_by_id[tensor.object_id]
    return hash_reader_range(
        source.reader,
        tensor.source_offset,
        tensor.source_length,
        buffer_bytes=buffer_bytes,
    )


def build_huggingface_mixed_policy(
    tensors: tuple[HuggingFaceCatalogTensor, ...],
    assignments: tuple[HuggingFaceTensorAssignment, ...],
) -> HuggingFaceMixedPolicy:
    """Normalize one complete mixed policy for both eager and progressive conversion."""
    assignment_by_name = {assignment.tensor_name: assignment for assignment in assignments}
    tensor_by_name = {tensor.tensor_name: tensor for tensor in tensors}
    if (
        len(assignment_by_name) != len(assignments)
        or len(tensor_by_name) != len(tensors)
        or set(assignment_by_name) != set(tensor_by_name)
    ):
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "mixed policy must assign every catalog tensor exactly once",
        )
    float_dtypes = {DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32}
    normalized: list[HuggingFaceTensorAssignment] = []
    policy_payload = []
    for tensor_name in sorted(tensor_by_name):
        tensor = tensor_by_name[tensor_name]
        assignment = assignment_by_name[tensor_name]
        if tensor.source_length == 0 or 0 in tensor.shape:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "mixed package v1 cannot encode zero-sized tensors",
            )
        if (
            assignment.encoding
            in {
                HuggingFaceTensorEncoding.TERNARY_TRIT5,
                HuggingFaceTensorEncoding.INT4_SYMMETRIC,
            }
            and tensor.dtype not in float_dtypes
        ):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"low-bit assignment requires a supported float source: {tensor.tensor_name}",
            )
        normalized.append(assignment)
        policy_payload.append(
            {
                "tensor_name": tensor_name,
                "encoding": assignment.encoding.value,
                **_codec_config_identity_fields(
                    assignment.encoding,
                    assignment.ternary_config,
                    assignment.int4_config,
                ),
            }
        )
    policy_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(policy_payload)).hexdigest()
    return HuggingFaceMixedPolicy(policy_hash, tuple(normalized))


def _huggingface_mixed_target_chunk_id(
    policy_hash: str,
    assignment: HuggingFaceTensorAssignment,
) -> str:
    identity = {
        "encoding": assignment.encoding.value,
        "policy_hash": policy_hash,
        "tensor_name": assignment.tensor_name,
    }
    return "tensor:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def build_huggingface_progressive_mixed_plan(
    catalog: HuggingFaceHeaderCatalog,
    assignments: tuple[HuggingFaceTensorAssignment, ...],
) -> HuggingFaceProgressiveMixedPlan:
    """Plan exact mixed outputs from headers without reading any tensor payload."""
    policy = build_huggingface_mixed_policy(catalog.tensors, assignments)
    assignment_by_name = {assignment.tensor_name: assignment for assignment in policy.assignments}
    source_by_id = {source.object_id: source for source in catalog.sources}
    if len(source_by_id) != len(catalog.sources):
        raise AmsError(ErrorCode.PLAN_INVALID, "Hugging Face source object IDs are not unique")
    shards = tuple(
        HuggingFaceProgressiveShardPlan(
            source.shard_name,
            source.object_id,
            source.content_hash,
            source.reader.size_bytes,
        )
        for source in catalog.sources
    )
    tensors = tuple(
        HuggingFaceProgressiveTensorPlan(
            tensor=tensor,
            target_chunk_id=_huggingface_mixed_target_chunk_id(
                policy.policy_hash,
                assignment_by_name[tensor.tensor_name],
            ),
            encoding=assignment_by_name[tensor.tensor_name].encoding,
            ternary_config=assignment_by_name[tensor.tensor_name].ternary_config,
            int4_config=assignment_by_name[tensor.tensor_name].int4_config,
        )
        for tensor in catalog.tensors
    )
    plan_payload = {
        "source_root": catalog.source_root,
        "index_content_hash": catalog.index_content_hash,
        "index_metadata_hash": catalog.index_metadata_hash,
        "policy_hash": policy.policy_hash,
        "total_size": catalog.total_size,
        "shards": [
            {
                "shard_name": shard.shard_name,
                "object_id": shard.object_id,
                "content_hash": shard.content_hash,
                "size_bytes": shard.size_bytes,
            }
            for shard in shards
        ],
        "tensors": [
            {
                "tensor_name": planned.tensor.tensor_name,
                "shard_name": planned.tensor.shard_name,
                "object_id": planned.tensor.object_id,
                "dtype": planned.tensor.dtype.value,
                "source_dtype": planned.tensor.source_dtype,
                "shape": list(planned.tensor.shape),
                "source_offset": planned.tensor.source_offset,
                "source_length": planned.tensor.source_length,
                "target_chunk_id": planned.target_chunk_id,
                "encoding": planned.encoding.value,
                **_codec_config_identity_fields(
                    planned.encoding,
                    planned.ternary_config,
                    planned.int4_config,
                ),
            }
            for planned in tensors
        ],
    }
    plan_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(plan_payload)).hexdigest()
    return HuggingFaceProgressiveMixedPlan(
        source_root=catalog.source_root,
        index_content_hash=catalog.index_content_hash,
        index_metadata_hash=catalog.index_metadata_hash,
        policy_hash=policy.policy_hash,
        plan_hash=plan_hash,
        total_size=catalog.total_size,
        shards=shards,
        tensors=tensors,
    )


def build_huggingface_mixed_plan(
    catalog: HuggingFaceCatalog,
    assignments: tuple[HuggingFaceTensorAssignment, ...],
    *,
    buffer_bytes: int = 1024 * 1024,
) -> HuggingFaceMixedPlan:
    """Build a plan only when every tensor has one explicit storage encoding."""
    policy = build_huggingface_mixed_policy(catalog.tensors, assignments)
    assignment_by_name = {assignment.tensor_name: assignment for assignment in policy.assignments}
    source_by_id = {source.object_id: source for source in catalog.sources}
    if len(source_by_id) != len(catalog.sources):
        raise AmsError(ErrorCode.PLAN_INVALID, "Hugging Face source object IDs are not unique")
    planned: list[HuggingFaceMixedPlannedTensor] = []
    items: list[ConversionItem] = []
    for tensor in catalog.tensors:
        assignment = assignment_by_name[tensor.tensor_name]
        checksum = _hash_catalog_tensor(source_by_id, tensor, buffer_bytes=buffer_bytes)
        target_chunk_id = _huggingface_mixed_target_chunk_id(
            policy.policy_hash,
            assignment,
        )
        source_range = ByteRange(
            object_id=tensor.object_id,
            offset=tensor.source_offset,
            length=tensor.source_length,
            checksum=checksum,
        )
        items.append(ConversionItem(target_chunk_id, source_range))
        planned.append(
            HuggingFaceMixedPlannedTensor(
                tensor=tensor,
                target_chunk_id=target_chunk_id,
                source_checksum=checksum,
                encoding=assignment.encoding,
                ternary_config=assignment.ternary_config,
                int4_config=assignment.int4_config,
            )
        )
    return HuggingFaceMixedPlan(
        conversion=ConversionPlan(
            source_root=catalog.source_root,
            configuration_hash=policy.policy_hash,
            items=tuple(items),
        ),
        policy_hash=policy.policy_hash,
        tensors=tuple(planned),
    )
