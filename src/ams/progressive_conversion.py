"""Shard-transactional mixed conversion with durable, granular progress records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_positive, checked_product
from ams.conversion import ConversionItem, ConversionPlan
from ams.descriptors import (
    ByteRange,
    ConversionJournal,
    ConversionJournalEntry,
    JournalEntryState,
    StorageObject,
    validate_digest,
)
from ams.errors import AmsError, ErrorCode
from ams.int4_conversion import Int4ChunkSpec, publish_int4_chunk_atomic
from ams.integrations.huggingface import (
    HuggingFaceCatalog,
    HuggingFaceHeaderCatalog,
    HuggingFaceMixedPlan,
    HuggingFaceMixedPlannedTensor,
    HuggingFaceProgressiveMixedPlan,
    HuggingFaceProgressiveShardPlan,
    HuggingFaceProgressiveTensorPlan,
    HuggingFaceShardSource,
    HuggingFaceTensorEncoding,
)
from ams.integrations.huggingface_staging import (
    release_huggingface_shard_source,
    stage_huggingface_shard,
    validate_huggingface_shard_cache_empty,
)
from ams.integrations.safetensors import parse_safetensors_header
from ams.storage import FileRangeStore, copy_range_atomic, hash_reader_range
from ams.ternary_conversion import TernaryChunkSpec, publish_ternary_chunk_atomic

_PLAN_SCHEMA = "ams.huggingface.progressive-plan"
_SHARD_SCHEMA = "ams.huggingface.progressive-shard"
_TENSOR_SCHEMA = "ams.huggingface.progressive-tensor"
_FORMAT_VERSION = {"major": 1, "minor": 0}
_MAX_RECORD_BYTES = 128 * 1024


@dataclass(frozen=True, slots=True)
class ProgressiveShardRecord:
    shard_name: str
    object_id: str
    content_hash: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ProgressiveTensorRecord:
    tensor_name: str
    shard_name: str
    target_chunk_id: str
    encoding: HuggingFaceTensorEncoding
    source_checksum: str
    target_hash: str
    encoded_bytes: int


@dataclass(frozen=True, slots=True)
class ProgressiveConversionSnapshot:
    plan_hash: str
    shards: tuple[ProgressiveShardRecord, ...]
    tensors: tuple[ProgressiveTensorRecord, ...]


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _read_record(path: Path, *, max_bytes: int) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise AmsError(ErrorCode.INVALID_PACKAGE, "progressive record is not a regular file")
        size = path.stat().st_size
        if size == 0 or size > max_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "progressive record size is invalid")
        value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    except AmsError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "progressive record is malformed") from exc
    if not isinstance(value, dict):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "progressive record must be an object")
    return value


def _write_all(handle, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = handle.write(payload[written:])
        if count is None or count == 0:
            raise AmsError(ErrorCode.IO_FAILURE, "short write to progressive record")
        written += count


def _publish_record(path: Path, value: dict[str, Any], *, max_bytes: int) -> None:
    payload = canonical_json_bytes(value)
    if len(payload) > max_bytes:
        raise AmsError(ErrorCode.TRANSACTION_FAILURE, "progressive record exceeds its limit")
    if path.exists():
        if _read_record(path, max_bytes=max_bytes) != value:
            raise AmsError(
                ErrorCode.PLAN_INVALID, "progressive record disagrees with durable state"
            )
        return
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb", buffering=0) as handle:
            _write_all(handle, payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        if _read_record(path, max_bytes=max_bytes) != value:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "published progressive record changed")
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.TRANSACTION_FAILURE,
            "progressive record publication failed",
            retriable=True,
        ) from exc
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _key_path(directory: Path, key: str) -> Path:
    name = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return directory / f"{name}.json"


def _plan_record(plan: HuggingFaceProgressiveMixedPlan) -> dict[str, Any]:
    return {
        "schema_id": _PLAN_SCHEMA,
        "format_version": _FORMAT_VERSION,
        "plan_hash": plan.plan_hash,
        "source_root": plan.source_root,
        "index_content_hash": plan.index_content_hash,
        "index_metadata_hash": plan.index_metadata_hash,
        "policy_hash": plan.policy_hash,
        "total_size": plan.total_size,
        "shard_count": len(plan.shards),
        "tensor_count": len(plan.tensors),
    }


class ProgressiveConversionJournalStore:
    """One immutable plan marker plus one atomic record per verified shard and output tensor."""

    def __init__(self, root: Path, *, max_record_bytes: int = _MAX_RECORD_BYTES) -> None:
        self.root = root.resolve(strict=False)
        self.max_record_bytes = checked_positive(
            max_record_bytes,
            name="progressive.max_record_bytes",
        )
        self.plan_path = self.root / "plan.json"
        self.shard_directory = self.root / "shards"
        self.tensor_directory = self.root / "tensors"

    def load_or_create(self, plan: HuggingFaceProgressiveMixedPlan) -> None:
        try:
            if self.root.is_symlink():
                raise AmsError(ErrorCode.INVALID_PACKAGE, "progressive journal root is a symlink")
            self.root.mkdir(parents=True, exist_ok=True)
            self.shard_directory.mkdir(exist_ok=True)
            self.tensor_directory.mkdir(exist_ok=True)
            if self.shard_directory.is_symlink() or self.tensor_directory.is_symlink():
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "progressive journal state directory is a symlink",
                )
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "progressive journal directories could not be prepared",
                retriable=True,
            ) from exc
        _publish_record(
            self.plan_path,
            _plan_record(plan),
            max_bytes=self.max_record_bytes,
        )

    def load_existing(self, plan: HuggingFaceProgressiveMixedPlan) -> None:
        try:
            if (
                self.root.is_symlink()
                or not self.root.is_dir()
                or self.shard_directory.is_symlink()
                or not self.shard_directory.is_dir()
                or self.tensor_directory.is_symlink()
                or not self.tensor_directory.is_dir()
            ):
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "progressive journal structure is missing or invalid",
                )
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "progressive journal structure could not be inspected",
                retriable=True,
            ) from exc
        if _read_record(self.plan_path, max_bytes=self.max_record_bytes) != _plan_record(plan):
            raise AmsError(
                ErrorCode.PLAN_INVALID, "progressive plan marker disagrees with the plan"
            )

    def shard_record(
        self,
        plan: HuggingFaceProgressiveMixedPlan,
        shard: HuggingFaceProgressiveShardPlan,
    ) -> ProgressiveShardRecord | None:
        path = _key_path(self.shard_directory, shard.object_id)
        if not path.exists():
            return None
        expected = {
            "schema_id": _SHARD_SCHEMA,
            "format_version": _FORMAT_VERSION,
            "plan_hash": plan.plan_hash,
            "state": "verified",
            "shard_name": shard.shard_name,
            "object_id": shard.object_id,
            "content_hash": shard.content_hash,
            "size_bytes": shard.size_bytes,
        }
        if _read_record(path, max_bytes=self.max_record_bytes) != expected:
            raise AmsError(ErrorCode.PLAN_INVALID, "verified shard record disagrees with the plan")
        return ProgressiveShardRecord(
            shard.shard_name,
            shard.object_id,
            shard.content_hash,
            shard.size_bytes,
        )

    def mark_shard_verified(
        self,
        plan: HuggingFaceProgressiveMixedPlan,
        shard: HuggingFaceProgressiveShardPlan,
    ) -> ProgressiveShardRecord:
        value = {
            "schema_id": _SHARD_SCHEMA,
            "format_version": _FORMAT_VERSION,
            "plan_hash": plan.plan_hash,
            "state": "verified",
            "shard_name": shard.shard_name,
            "object_id": shard.object_id,
            "content_hash": shard.content_hash,
            "size_bytes": shard.size_bytes,
        }
        _publish_record(
            _key_path(self.shard_directory, shard.object_id),
            value,
            max_bytes=self.max_record_bytes,
        )
        record = self.shard_record(plan, shard)
        if record is None:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "verified shard record was not published")
        return record

    def tensor_record(
        self,
        plan: HuggingFaceProgressiveMixedPlan,
        tensor: HuggingFaceProgressiveTensorPlan,
    ) -> ProgressiveTensorRecord | None:
        path = _key_path(self.tensor_directory, tensor.target_chunk_id)
        if not path.exists():
            return None
        value = _read_record(path, max_bytes=self.max_record_bytes)
        required = {
            "schema_id",
            "format_version",
            "plan_hash",
            "state",
            "tensor_name",
            "shard_name",
            "target_chunk_id",
            "encoding",
            "source_checksum",
            "target_hash",
            "encoded_bytes",
        }
        if set(value) != required or any(
            (
                value["schema_id"] != _TENSOR_SCHEMA,
                value["format_version"] != _FORMAT_VERSION,
                value["plan_hash"] != plan.plan_hash,
                value["state"] != "published",
                value["tensor_name"] != tensor.tensor.tensor_name,
                value["shard_name"] != tensor.tensor.shard_name,
                value["target_chunk_id"] != tensor.target_chunk_id,
                value["encoding"] != tensor.encoding.value,
            )
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID, "published tensor record disagrees with the plan"
            )
        source_checksum = value["source_checksum"]
        target_hash = value["target_hash"]
        encoded_bytes = value["encoded_bytes"]
        validate_digest(source_checksum, name="progressive.source_checksum")
        validate_digest(target_hash, name="progressive.target_hash")
        checked_positive(encoded_bytes, name="progressive.encoded_bytes")
        if tensor.encoding is HuggingFaceTensorEncoding.IDENTITY and (
            target_hash != source_checksum or encoded_bytes != tensor.tensor.source_length
        ):
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE, "identity progressive record is inconsistent"
            )
        if tensor.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
            if tensor.ternary_config is None:
                raise AmsError(ErrorCode.INTERNAL_INVARIANT, "ternary plan has no codec config")
            element_count = checked_product(
                tensor.tensor.shape,
                name=f"progressive.{tensor.tensor.tensor_name}.elements",
            )
            if encoded_bytes != tensor.ternary_config.encoded_size(element_count):
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE, "ternary progressive size is inconsistent"
                )
        if tensor.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC:
            if tensor.int4_config is None:
                raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 plan has no codec config")
            element_count = checked_product(
                tensor.tensor.shape,
                name=f"progressive.{tensor.tensor.tensor_name}.elements",
            )
            if encoded_bytes != tensor.int4_config.encoded_size(element_count):
                raise AmsError(ErrorCode.INTEGRITY_FAILURE, "INT4 progressive size is inconsistent")
        return ProgressiveTensorRecord(
            tensor.tensor.tensor_name,
            tensor.tensor.shard_name,
            tensor.target_chunk_id,
            tensor.encoding,
            source_checksum,
            target_hash,
            encoded_bytes,
        )

    def mark_tensor_published(
        self,
        plan: HuggingFaceProgressiveMixedPlan,
        tensor: HuggingFaceProgressiveTensorPlan,
        *,
        source_checksum: str,
        target_hash: str,
        encoded_bytes: int,
    ) -> ProgressiveTensorRecord:
        validate_digest(source_checksum, name="progressive.source_checksum")
        validate_digest(target_hash, name="progressive.target_hash")
        checked_positive(encoded_bytes, name="progressive.encoded_bytes")
        value = {
            "schema_id": _TENSOR_SCHEMA,
            "format_version": _FORMAT_VERSION,
            "plan_hash": plan.plan_hash,
            "state": "published",
            "tensor_name": tensor.tensor.tensor_name,
            "shard_name": tensor.tensor.shard_name,
            "target_chunk_id": tensor.target_chunk_id,
            "encoding": tensor.encoding.value,
            "source_checksum": source_checksum,
            "target_hash": target_hash,
            "encoded_bytes": encoded_bytes,
        }
        _publish_record(
            _key_path(self.tensor_directory, tensor.target_chunk_id),
            value,
            max_bytes=self.max_record_bytes,
        )
        record = self.tensor_record(plan, tensor)
        if record is None:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "tensor record was not published")
        return record

    def completed_snapshot(
        self,
        plan: HuggingFaceProgressiveMixedPlan,
    ) -> ProgressiveConversionSnapshot:
        self.load_existing(plan)
        shards: list[ProgressiveShardRecord] = []
        for shard in plan.shards:
            record = self.shard_record(plan, shard)
            if record is None:
                raise AmsError(ErrorCode.TRANSACTION_FAILURE, "source shard is not verified")
            shards.append(record)
        tensors: list[ProgressiveTensorRecord] = []
        for tensor in plan.tensors:
            record = self.tensor_record(plan, tensor)
            if record is None:
                raise AmsError(ErrorCode.TRANSACTION_FAILURE, "tensor output is not published")
            tensors.append(record)
        return ProgressiveConversionSnapshot(plan.plan_hash, tuple(shards), tuple(tensors))


def _validate_catalog_plan(
    catalog: HuggingFaceHeaderCatalog,
    plan: HuggingFaceProgressiveMixedPlan,
) -> None:
    if (
        catalog.source_root != plan.source_root
        or catalog.index_content_hash != plan.index_content_hash
        or catalog.index_metadata_hash != plan.index_metadata_hash
        or catalog.total_size != plan.total_size
        or tuple(planned.tensor for planned in plan.tensors) != catalog.tensors
    ):
        raise AmsError(ErrorCode.PLAN_INVALID, "progressive plan and header catalog disagree")
    source_by_id = {source.object_id: source for source in catalog.sources}
    if len(source_by_id) != len(catalog.sources):
        raise AmsError(ErrorCode.PLAN_INVALID, "progressive source IDs are duplicated")
    expected_shards = tuple(
        (
            source.shard_name,
            source.object_id,
            source.content_hash,
            source.reader.size_bytes,
        )
        for source in catalog.sources
    )
    actual_shards = tuple(
        (shard.shard_name, shard.object_id, shard.content_hash, shard.size_bytes)
        for shard in plan.shards
    )
    if expected_shards != actual_shards:
        raise AmsError(ErrorCode.PLAN_INVALID, "progressive source plan disagrees with catalog")


def _validate_staged_header(
    source: HuggingFaceShardSource,
    tensors: tuple[HuggingFaceProgressiveTensorPlan, ...],
) -> None:
    header = parse_safetensors_header(source.reader)
    expected = tuple(
        (
            tensor.tensor.tensor_name,
            tensor.tensor.dtype,
            tensor.tensor.source_dtype,
            tensor.tensor.shape,
            tensor.tensor.source_offset,
            tensor.tensor.source_length,
        )
        for tensor in tensors
    )
    actual = tuple(
        (
            tensor.source_name,
            tensor.dtype,
            tensor.source_dtype,
            tensor.shape,
            tensor.absolute_offset,
            tensor.data_length,
        )
        for tensor in header.tensors
    )
    if expected != actual:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "staged shard header differs from the plan")


def _verify_published_chunk(
    destination_root: Path,
    record: ProgressiveTensorRecord,
    *,
    buffer_bytes: int,
) -> None:
    algorithm, hexdigest = record.target_hash.split(":", 1)
    path = destination_root / "chunks" / f"{algorithm}-{hexdigest}.bin"
    try:
        if path.is_symlink() or not path.is_file():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "progressive output chunk is missing or not a regular file",
            )
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "progressive output chunk could not be inspected",
            retriable=True,
        ) from exc
    descriptor = StorageObject(
        f"progressive:{hexdigest}",
        path.name,
        record.encoded_bytes,
        1,
        record.target_hash,
    )
    reader = FileRangeStore(path, descriptor)
    actual = hash_reader_range(
        reader,
        0,
        record.encoded_bytes,
        buffer_bytes=buffer_bytes,
        algorithm=algorithm,
    )
    if actual != record.target_hash:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "progressive output chunk hash mismatch")


def _publish_tensor(
    reader,
    planned: HuggingFaceProgressiveTensorPlan,
    destination_root: Path,
    source_checksum: str,
    *,
    buffer_bytes: int,
) -> tuple[str, int]:
    source_range = ByteRange(
        planned.tensor.object_id,
        planned.tensor.source_offset,
        planned.tensor.source_length,
        source_checksum,
    )
    if planned.encoding is HuggingFaceTensorEncoding.IDENTITY:
        copy_range_atomic(
            reader,
            source_range,
            destination_root,
            planned.target_chunk_id,
            buffer_bytes=buffer_bytes,
        )
        return source_checksum, source_range.length
    if planned.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
        if planned.ternary_config is None:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "ternary plan has no codec config")
        publication = publish_ternary_chunk_atomic(
            reader,
            TernaryChunkSpec(
                planned.target_chunk_id,
                source_range,
                planned.tensor.shape,
                planned.tensor.dtype,
                planned.ternary_config,
            ),
            destination_root,
            verification_buffer_bytes=buffer_bytes,
        )
        return publication.target_hash, publication.encoded_bytes
    if planned.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC:
        if planned.int4_config is None:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 plan has no codec config")
        publication = publish_int4_chunk_atomic(
            reader,
            Int4ChunkSpec(
                planned.target_chunk_id,
                source_range,
                planned.tensor.shape,
                planned.tensor.dtype,
                planned.int4_config,
            ),
            destination_root,
            verification_buffer_bytes=buffer_bytes,
        )
        return publication.target_hash, publication.encoded_bytes
    raise AmsError(ErrorCode.INTERNAL_INVARIANT, "progressive tensor encoding is unknown")


def execute_progressive_huggingface_mixed_conversion(
    catalog: HuggingFaceHeaderCatalog,
    plan: HuggingFaceProgressiveMixedPlan,
    destination_root: Path,
    journal_root: Path,
    cache_root: Path,
    *,
    buffer_bytes: int = 1024 * 1024,
) -> ProgressiveConversionSnapshot:
    """Verify, convert, and release one source shard at a time with durable progress."""
    checked_positive(buffer_bytes, name="progressive.buffer_bytes")
    _validate_catalog_plan(catalog, plan)
    destination = destination_root.resolve(strict=False)
    cache = cache_root.resolve(strict=False)
    journal = journal_root.resolve(strict=False)
    try:
        cache.relative_to(destination)
        overlaps = True
    except ValueError:
        try:
            destination.relative_to(cache)
            overlaps = True
        except ValueError:
            overlaps = False
    if overlaps:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "progressive source cache and destination must be disjoint",
        )
    try:
        journal.relative_to(cache)
        journal_overlaps_cache = True
    except ValueError:
        try:
            cache.relative_to(journal)
            journal_overlaps_cache = True
        except ValueError:
            journal_overlaps_cache = False
    if journal_overlaps_cache:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "progressive journal and source cache must be disjoint",
        )
    store = ProgressiveConversionJournalStore(journal_root)
    store.load_or_create(plan)
    source_by_id = {source.object_id: source for source in catalog.sources}
    grouped_tensors: dict[str, list[HuggingFaceProgressiveTensorPlan]] = {
        shard.shard_name: [] for shard in plan.shards
    }
    for tensor in plan.tensors:
        if tensor.tensor.shard_name not in grouped_tensors:
            raise AmsError(ErrorCode.PLAN_INVALID, "tensor references an unknown source shard")
        grouped_tensors[tensor.tensor.shard_name].append(tensor)
    tensors_by_shard = {
        shard_name: tuple(tensors) for shard_name, tensors in grouped_tensors.items()
    }
    for shard in plan.shards:
        source = source_by_id[shard.object_id]
        shard_tensors = tensors_by_shard[shard.shard_name]
        prior_shard = store.shard_record(plan, shard)
        prior_tensors = tuple(store.tensor_record(plan, tensor) for tensor in shard_tensors)
        if prior_shard is not None and all(record is not None for record in prior_tensors):
            for record in prior_tensors:
                if record is None:
                    raise AmsError(ErrorCode.INTERNAL_INVARIANT, "complete shard lost a record")
                _verify_published_chunk(destination, record, buffer_bytes=buffer_bytes)
            release_huggingface_shard_source(source, cache)
            continue

        staged = stage_huggingface_shard(source, cache, buffer_bytes=buffer_bytes)
        _validate_staged_header(staged.source, shard_tensors)
        store.mark_shard_verified(plan, shard)
        for tensor in shard_tensors:
            existing = store.tensor_record(plan, tensor)
            if existing is not None:
                _verify_published_chunk(destination, existing, buffer_bytes=buffer_bytes)
                continue
            source_checksum = hash_reader_range(
                staged.source.reader,
                tensor.tensor.source_offset,
                tensor.tensor.source_length,
                buffer_bytes=buffer_bytes,
            )
            target_hash, encoded_bytes = _publish_tensor(
                staged.source.reader,
                tensor,
                destination,
                source_checksum,
                buffer_bytes=buffer_bytes,
            )
            store.mark_tensor_published(
                plan,
                tensor,
                source_checksum=source_checksum,
                target_hash=target_hash,
                encoded_bytes=encoded_bytes,
            )
        if any(store.tensor_record(plan, tensor) is None for tensor in shard_tensors):
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "shard outputs are incomplete at release")
        release_huggingface_shard_source(staged.source, cache, declared_path=staged.path)
    validate_huggingface_shard_cache_empty(cache)
    return store.completed_snapshot(plan)


def finalize_progressive_huggingface_mixed_conversion(
    catalog: HuggingFaceHeaderCatalog,
    plan: HuggingFaceProgressiveMixedPlan,
    journal_root: Path,
) -> tuple[HuggingFaceCatalog, HuggingFaceMixedPlan, ConversionJournal]:
    """Promote only complete durable progressive state into the established manifest contracts."""
    _validate_catalog_plan(catalog, plan)
    snapshot = ProgressiveConversionJournalStore(journal_root).completed_snapshot(plan)
    shard_records = {
        (record.shard_name, record.object_id, record.content_hash, record.size_bytes)
        for record in snapshot.shards
    }
    expected_shards = {
        (shard.shard_name, shard.object_id, shard.content_hash, shard.size_bytes)
        for shard in plan.shards
    }
    if shard_records != expected_shards or len(snapshot.shards) != len(plan.shards):
        raise AmsError(ErrorCode.PLAN_INVALID, "progressive verified shard set is incomplete")
    record_by_target = {record.target_chunk_id: record for record in snapshot.tensors}
    if len(record_by_target) != len(snapshot.tensors) or set(record_by_target) != {
        tensor.target_chunk_id for tensor in plan.tensors
    }:
        raise AmsError(ErrorCode.PLAN_INVALID, "progressive tensor record set is incomplete")

    items: list[ConversionItem] = []
    mixed_tensors: list[HuggingFaceMixedPlannedTensor] = []
    for tensor in plan.tensors:
        record = record_by_target[tensor.target_chunk_id]
        if (
            record.tensor_name != tensor.tensor.tensor_name
            or record.shard_name != tensor.tensor.shard_name
            or record.encoding is not tensor.encoding
        ):
            raise AmsError(ErrorCode.PLAN_INVALID, "progressive tensor record identity changed")
        source_range = ByteRange(
            tensor.tensor.object_id,
            tensor.tensor.source_offset,
            tensor.tensor.source_length,
            record.source_checksum,
        )
        items.append(ConversionItem(tensor.target_chunk_id, source_range))
        mixed_tensors.append(
            HuggingFaceMixedPlannedTensor(
                tensor=tensor.tensor,
                target_chunk_id=tensor.target_chunk_id,
                source_checksum=record.source_checksum,
                encoding=tensor.encoding,
                ternary_config=tensor.ternary_config,
                int4_config=tensor.int4_config,
            )
        )
    conversion = ConversionPlan(plan.source_root, plan.policy_hash, tuple(items))
    mixed_plan = HuggingFaceMixedPlan(conversion, plan.policy_hash, tuple(mixed_tensors))
    journal_entries_by_target = {
        item.target_chunk_id: ConversionJournalEntry(
            source_range=item.source_range,
            target_chunk_id=item.target_chunk_id,
            state=JournalEntryState.PUBLISHED,
            target_hash=record_by_target[item.target_chunk_id].target_hash,
            encoded_bytes=record_by_target[item.target_chunk_id].encoded_bytes,
        )
        for item in conversion.items
    }
    journal = ConversionJournal(
        "1.0.0",
        plan.source_root,
        plan.policy_hash,
        tuple(journal_entries_by_target[item.target_chunk_id] for item in conversion.items),
    )
    verified_catalog = HuggingFaceCatalog(
        source_root=catalog.source_root,
        index_content_hash=catalog.index_content_hash,
        index_metadata_hash=catalog.index_metadata_hash,
        total_size=catalog.total_size,
        tensors=catalog.tensors,
        sources=catalog.sources,
    )
    return verified_catalog, mixed_plan, journal
