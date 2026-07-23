"""Crash-recoverable publication of deterministic symmetric INT4 tensor chunks.

A completed record is authoritative only when it matches the exact source, shape,
dtype, and codec configuration in the requested conversion and its content-addressed
chunk verifies. Partial output without such a record is never treated as published.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_product
from ams.codecs import Int4CodecConfig, Int4EncodingResult, encode_int4_stream
from ams.conversion import ConversionJournalStore, ConversionPlan
from ams.descriptors import (
    ByteRange,
    DType,
    JournalEntryState,
    StorageObject,
    validate_digest,
    validate_identifier,
)
from ams.errors import AmsError, ErrorCode
from ams.storage import FileRangeStore, RangeReader, hash_reader_range

_RECORD_SCHEMA = "ams.int4.publication"
_RECORD_VERSION = {"major": 1, "minor": 0}
_MAX_RECORD_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class Int4ChunkSpec:
    publication_key: str
    source: ByteRange
    shape: tuple[int, ...]
    source_dtype: DType
    config: Int4CodecConfig

    def __post_init__(self) -> None:
        validate_identifier(self.publication_key, name="int4.publication_key")
        object.__setattr__(self, "source_dtype", DType(self.source_dtype))


@dataclass(frozen=True, slots=True)
class Int4Publication:
    publication_key: str
    source_checksum: str
    configuration_hash: str
    target_hash: str
    encoded_bytes: int
    decoded_bytes: int
    element_count: int
    group_count: int
    path: Path


class _DuplicateRecordKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateRecordKey(key)
        result[key] = value
    return result


def _record_id(spec: Int4ChunkSpec) -> str:
    identity = {
        "configuration_hash": spec.config.config_hash,
        "publication_key": spec.publication_key,
        "shape": list(spec.shape),
        "source": {
            "checksum": spec.source.checksum,
            "length": spec.source.length,
            "object_id": spec.source.object_id,
            "offset": spec.source.offset,
        },
        "source_dtype": spec.source_dtype.value,
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def _record_dict(spec: Int4ChunkSpec, result: Int4EncodingResult) -> dict[str, Any]:
    if result.source_checksum != spec.source.checksum:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 encoder source identity changed")
    return {
        "schema_id": _RECORD_SCHEMA,
        "format_version": _RECORD_VERSION,
        "publication_key": spec.publication_key,
        "source_checksum": result.source_checksum,
        "configuration_hash": spec.config.config_hash,
        "target_hash": result.content_hash,
        "encoded_bytes": result.encoded_bytes,
        "decoded_bytes": result.decoded_bytes,
        "element_count": result.element_count,
        "group_count": result.group_count,
    }


def _load_record(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 publication record is not a file")
        size = path.stat().st_size
        if size == 0 or size > _MAX_RECORD_BYTES:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 publication record size is invalid")
        payload = path.read_bytes()
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except AmsError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _DuplicateRecordKey) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 publication record is malformed") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_id",
        "format_version",
        "publication_key",
        "source_checksum",
        "configuration_hash",
        "target_hash",
        "encoded_bytes",
        "decoded_bytes",
        "element_count",
        "group_count",
    }:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 publication record fields are invalid")
    return value


def _validate_record(record: dict[str, Any], spec: Int4ChunkSpec) -> None:
    try:
        expected_elements = checked_product(spec.shape, name="int4.publication.shape")
        expected_encoded = spec.config.encoded_size(expected_elements)
        expected_groups = (expected_elements + spec.config.group_size - 1) // spec.config.group_size
        valid = (
            record["schema_id"] == _RECORD_SCHEMA
            and record["format_version"] == _RECORD_VERSION
            and record["publication_key"] == spec.publication_key
            and record["source_checksum"] == spec.source.checksum
            and record["configuration_hash"] == spec.config.config_hash
            and isinstance(record["encoded_bytes"], int)
            and not isinstance(record["encoded_bytes"], bool)
            and record["encoded_bytes"] == expected_encoded
            and isinstance(record["decoded_bytes"], int)
            and not isinstance(record["decoded_bytes"], bool)
            and record["decoded_bytes"] == spec.source.length
            and isinstance(record["element_count"], int)
            and not isinstance(record["element_count"], bool)
            and record["element_count"] == expected_elements
            and isinstance(record["group_count"], int)
            and not isinstance(record["group_count"], bool)
            and record["group_count"] == expected_groups
        )
        validate_digest(record["target_hash"], name="int4.target_hash")
    except (AmsError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AmsError):
            raise
        raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 publication record is invalid") from exc
    if not valid:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT4 publication record does not match the plan")


def _write_record(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if len(payload) > _MAX_RECORD_BYTES:
        raise AmsError(ErrorCode.TRANSACTION_FAILURE, "INT4 publication record is too large")
    try:
        with path.open("wb", buffering=0) as handle:
            written = 0
            while written < len(payload):
                count = handle.write(payload[written:])
                if count is None or count == 0:
                    raise AmsError(
                        ErrorCode.IO_FAILURE,
                        "short write to INT4 publication record",
                        retriable=True,
                    )
                written += count
            handle.flush()
            os.fsync(handle.fileno())
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.TRANSACTION_FAILURE,
            "INT4 publication record write failed",
            retriable=True,
        ) from exc


def _verify_chunk(
    path: Path,
    target_hash: str,
    encoded_bytes: int,
    *,
    buffer_bytes: int,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "INT4 chunk is not a regular file")
    descriptor = StorageObject(
        object_id="int4:verified",
        uri=path.name,
        size_bytes=encoded_bytes,
        alignment_bytes=1,
        content_hash=target_hash,
    )
    reader = FileRangeStore(path, descriptor)
    algorithm = target_hash.split(":", 1)[0]
    actual = hash_reader_range(
        reader,
        0,
        encoded_bytes,
        buffer_bytes=buffer_bytes,
        algorithm=algorithm,
    )
    if actual != target_hash:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "INT4 chunk content hash mismatch")


def _publication_from_record(record: dict[str, Any], path: Path) -> Int4Publication:
    return Int4Publication(
        publication_key=record["publication_key"],
        source_checksum=record["source_checksum"],
        configuration_hash=record["configuration_hash"],
        target_hash=record["target_hash"],
        encoded_bytes=record["encoded_bytes"],
        decoded_bytes=record["decoded_bytes"],
        element_count=record["element_count"],
        group_count=record["group_count"],
        path=path,
    )


def publish_int4_chunk_atomic(
    reader: RangeReader,
    spec: Int4ChunkSpec,
    destination_root: Path,
    *,
    verification_buffer_bytes: int = 1024 * 1024,
    stream_encoder=encode_int4_stream,
) -> Int4Publication:
    """Encode or recover one INT4 chunk without rereading a completed transform."""
    if verification_buffer_bytes <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "verification buffer size must be positive")
    root = destination_root.resolve(strict=False)
    chunks = root / "chunks"
    staging = root / ".staging"
    records = root / ".records"
    try:
        chunks.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=True)
        records.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "INT4 publication directories could not be created",
            retriable=True,
        ) from exc
    record_id = _record_id(spec)
    record_path = records / f"{record_id}.json"
    pending_path = records / f"{record_id}.pending.json"
    pending_temporary = records / f"{record_id}.pending.tmp"
    staging_path = staging / f"{record_id}.part"

    if record_path.exists():
        record = _load_record(record_path)
        _validate_record(record, spec)
        algorithm, hexdigest = record["target_hash"].split(":", 1)
        final_path = chunks / f"{algorithm}-{hexdigest}.bin"
        _verify_chunk(
            final_path,
            record["target_hash"],
            record["encoded_bytes"],
            buffer_bytes=verification_buffer_bytes,
        )
        return _publication_from_record(record, final_path)

    if pending_path.exists():
        record = _load_record(pending_path)
        _validate_record(record, spec)
        algorithm, hexdigest = record["target_hash"].split(":", 1)
        final_path = chunks / f"{algorithm}-{hexdigest}.bin"
        if final_path.exists():
            _verify_chunk(
                final_path,
                record["target_hash"],
                record["encoded_bytes"],
                buffer_bytes=verification_buffer_bytes,
            )
        elif staging_path.exists():
            _verify_chunk(
                staging_path,
                record["target_hash"],
                record["encoded_bytes"],
                buffer_bytes=verification_buffer_bytes,
            )
            try:
                os.replace(staging_path, final_path)
            except OSError as exc:
                raise AmsError(
                    ErrorCode.TRANSACTION_FAILURE,
                    "INT4 pending chunk could not be published",
                    retriable=True,
                ) from exc
        else:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "INT4 pending record has no corresponding chunk",
            )
        try:
            os.replace(pending_path, record_path)
        except OSError as exc:
            raise AmsError(
                ErrorCode.TRANSACTION_FAILURE,
                "INT4 publication record could not be finalized",
                retriable=True,
            ) from exc
        return _publication_from_record(record, final_path)

    try:
        with staging_path.open("wb", buffering=0) as handle:
            result = stream_encoder(
                reader,
                spec.source,
                spec.shape,
                spec.source_dtype,
                handle,
                spec.config,
            )
            handle.flush()
            os.fsync(handle.fileno())
        record = _record_dict(spec, result)
        _write_record(pending_temporary, record)
        os.replace(pending_temporary, pending_path)
        algorithm, hexdigest = result.content_hash.split(":", 1)
        final_path = chunks / f"{algorithm}-{hexdigest}.bin"
        os.replace(staging_path, final_path)
        os.replace(pending_path, record_path)
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.TRANSACTION_FAILURE,
            "INT4 chunk publication failed",
            retriable=True,
        ) from exc
    return _publication_from_record(record, final_path)


def execute_int4_conversion(
    readers: dict[str, RangeReader],
    plan: ConversionPlan,
    specs: tuple[Int4ChunkSpec, ...],
    destination_root: Path,
    journal_path: Path,
    *,
    verification_buffer_bytes: int = 1024 * 1024,
):
    """Execute and durably journal an explicitly symmetric-INT4-only plan."""
    spec_by_key = {spec.publication_key: spec for spec in specs}
    if len(spec_by_key) != len(specs) or set(spec_by_key) != {
        item.target_chunk_id for item in plan.items
    }:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT4 specs do not match conversion items")
    for item in plan.items:
        spec = spec_by_key[item.target_chunk_id]
        if spec.source != item.source_range:
            raise AmsError(ErrorCode.PLAN_INVALID, "INT4 spec source differs from the plan")
        if spec.config.config_hash != plan.configuration_hash:
            raise AmsError(ErrorCode.PLAN_INVALID, "INT4 codec configuration differs from the plan")
    store = ConversionJournalStore(journal_path)
    journal = store.load_or_create(plan)
    entries = {entry.target_chunk_id: entry for entry in journal.entries}
    for item in plan.items:
        spec = spec_by_key[item.target_chunk_id]
        if item.source_range.object_id not in readers:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "INT4 source reader is missing")
        publication = publish_int4_chunk_atomic(
            readers[item.source_range.object_id],
            spec,
            destination_root,
            verification_buffer_bytes=verification_buffer_bytes,
        )
        entry = entries[item.target_chunk_id]
        if entry.state is JournalEntryState.PUBLISHED:
            if (
                entry.target_hash != publication.target_hash
                or entry.encoded_bytes != publication.encoded_bytes
            ):
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "journal and INT4 publication record disagree",
                )
            continue
        entries[item.target_chunk_id] = replace(
            entry,
            state=JournalEntryState.PUBLISHED,
            target_hash=publication.target_hash,
            encoded_bytes=publication.encoded_bytes,
        )
        journal = replace(
            journal,
            entries=tuple(entries[planned.target_chunk_id] for planned in plan.items),
        )
        store.write(journal)
    return journal
