"""Strict normalization of Hugging Face sharded safetensors repositories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_uint
from ams.codecs import TernaryCodecConfig
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


@dataclass(frozen=True, slots=True)
class HuggingFaceTensorAssignment:
    tensor_name: str
    encoding: HuggingFaceTensorEncoding
    ternary_config: TernaryCodecConfig | None = None

    def __post_init__(self) -> None:
        _validate_external_name(self.tensor_name, field="tensor name", max_bytes=4096)
        object.__setattr__(self, "encoding", HuggingFaceTensorEncoding(self.encoding))
        if self.encoding is HuggingFaceTensorEncoding.IDENTITY and self.ternary_config is not None:
            raise AmsError(ErrorCode.PLAN_INVALID, "identity assignment cannot have ternary config")
        if self.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5 and self.ternary_config is None:
            raise AmsError(ErrorCode.PLAN_INVALID, "ternary assignment requires codec config")


@dataclass(frozen=True, slots=True)
class HuggingFaceMixedPlannedTensor:
    tensor: HuggingFaceCatalogTensor
    target_chunk_id: str
    source_checksum: str
    encoding: HuggingFaceTensorEncoding
    ternary_config: TernaryCodecConfig | None = None


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


def build_huggingface_catalog(
    index: HuggingFaceShardIndex,
    sources: tuple[HuggingFaceShardSource, ...],
    *,
    buffer_bytes: int = 1024 * 1024,
) -> HuggingFaceCatalog:
    """Cross-check the provider index, immutable shard hashes, and normalized headers."""
    source_by_name = {source.shard_name: source for source in sources}
    if len(source_by_name) != len(sources) or set(source_by_name) != set(index.shard_names):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face shard source set is incomplete")
    index_by_tensor = {entry.tensor_name: entry.shard_name for entry in index.entries}
    tensors: list[HuggingFaceCatalogTensor] = []
    observed_names: set[str] = set()
    total_size = 0
    for shard_name in index.shard_names:
        source = source_by_name[shard_name]
        header = _verify_shard(source, buffer_bytes=buffer_bytes)
        for tensor in header.tensors:
            if tensor.source_name in observed_names:
                raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor appears in more than one shard")
            if index_by_tensor.get(tensor.source_name) != shard_name:
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "Hugging Face index and shard tensor mapping disagree",
                )
            observed_names.add(tensor.source_name)
            total_size = checked_add(total_size, tensor.data_length, name="huggingface.total_size")
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
    if total_size != index.total_size:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "Hugging Face total_size does not match tensor storage",
            evidence={"actual": total_size, "declared": index.total_size},
        )
    source_root_payload = {
        "index": index.content_hash,
        "shards": [
            {"name": source.shard_name, "content_hash": source.content_hash}
            for source in sorted(sources, key=lambda item: item.shard_name)
        ],
    }
    source_root = "sha256:" + hashlib.sha256(canonical_json_bytes(source_root_payload)).hexdigest()
    return HuggingFaceCatalog(
        source_root=source_root,
        index_content_hash=index.content_hash,
        index_metadata_hash=index.metadata_hash,
        total_size=total_size,
        tensors=tuple(sorted(tensors, key=lambda tensor: tensor.tensor_name)),
        sources=tuple(sorted(sources, key=lambda source: source.shard_name)),
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


def build_huggingface_mixed_plan(
    catalog: HuggingFaceCatalog,
    assignments: tuple[HuggingFaceTensorAssignment, ...],
    *,
    buffer_bytes: int = 1024 * 1024,
) -> HuggingFaceMixedPlan:
    """Build a plan only when every tensor has one explicit storage encoding."""
    assignment_by_name = {assignment.tensor_name: assignment for assignment in assignments}
    catalog_names = {tensor.tensor_name for tensor in catalog.tensors}
    if len(assignment_by_name) != len(assignments) or set(assignment_by_name) != catalog_names:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "mixed policy must assign every catalog tensor exactly once",
        )
    source_by_id = {source.object_id: source for source in catalog.sources}
    if len(source_by_id) != len(catalog.sources):
        raise AmsError(ErrorCode.PLAN_INVALID, "Hugging Face source object IDs are not unique")
    policy_payload = []
    for tensor_name in sorted(assignment_by_name):
        assignment = assignment_by_name[tensor_name]
        policy_payload.append(
            {
                "tensor_name": tensor_name,
                "encoding": assignment.encoding.value,
                **(
                    {"ternary_config_hash": assignment.ternary_config.config_hash}
                    if assignment.ternary_config is not None
                    else {}
                ),
            }
        )
    policy_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(policy_payload)).hexdigest()
    planned: list[HuggingFaceMixedPlannedTensor] = []
    items: list[ConversionItem] = []
    float_dtypes = {DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32}
    for tensor in catalog.tensors:
        assignment = assignment_by_name[tensor.tensor_name]
        if tensor.source_length == 0 or 0 in tensor.shape:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "mixed package v1 cannot encode zero-sized tensors",
            )
        if (
            assignment.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5
            and tensor.dtype not in float_dtypes
        ):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"ternary assignment requires a supported float source: {tensor.tensor_name}",
            )
        checksum = _hash_catalog_tensor(source_by_id, tensor, buffer_bytes=buffer_bytes)
        target_identity = {
            "encoding": assignment.encoding.value,
            "policy_hash": policy_hash,
            "tensor_name": tensor.tensor_name,
        }
        target_chunk_id = (
            "tensor:" + hashlib.sha256(canonical_json_bytes(target_identity)).hexdigest()
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
            )
        )
    return HuggingFaceMixedPlan(
        conversion=ConversionPlan(
            source_root=catalog.source_root,
            configuration_hash=policy_hash,
            items=tuple(items),
        ),
        policy_hash=policy_hash,
        tensors=tuple(planned),
    )
