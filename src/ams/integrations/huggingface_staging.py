"""Verified, restart-safe ephemeral staging for one Hugging Face shard."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from ams.descriptors import ByteRange, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations.huggingface import HuggingFaceShardSource
from ams.storage import FileRangeStore, copy_range_atomic

_MARKER_NAME = ".ams-hf-shard-cache-v1"
_MARKER_PAYLOAD = b"ams.huggingface.shard-cache\nversion=1\n"


@dataclass(frozen=True, slots=True)
class StagedHuggingFaceShard:
    """A fully hash-verified local lease on one immutable source shard."""

    cache_root: Path
    path: Path
    source: HuggingFaceShardSource


def _write_all(handle, payload: bytes) -> None:
    written = 0
    while written < len(payload):
        count = handle.write(payload[written:])
        if count is None or count == 0:
            raise AmsError(ErrorCode.IO_FAILURE, "short write to shard-cache marker")
        written += count


def _validate_marker(root: Path) -> None:
    marker = root / _MARKER_NAME
    try:
        if marker.is_symlink() or not marker.is_file() or marker.read_bytes() != _MARKER_PAYLOAD:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "Hugging Face shard-cache marker is missing or invalid",
            )
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "Hugging Face shard-cache marker could not be read",
            retriable=True,
        ) from exc


def _prepare_cache_root(cache_root: Path) -> Path:
    try:
        root = cache_root.resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        marker = root / _MARKER_NAME
        if marker.exists():
            _validate_marker(root)
            return root
        with marker.open("xb", buffering=0) as handle:
            _write_all(handle, _MARKER_PAYLOAD)
            handle.flush()
            os.fsync(handle.fileno())
        _validate_marker(root)
        return root
    except AmsError:
        raise
    except FileExistsError:
        root = cache_root.resolve(strict=True)
        _validate_marker(root)
        return root
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "Hugging Face shard-cache root could not be prepared",
            retriable=True,
        ) from exc


def stage_huggingface_shard(
    source: HuggingFaceShardSource,
    cache_root: Path,
    *,
    buffer_bytes: int = 1024 * 1024,
) -> StagedHuggingFaceShard:
    """Transfer one shard once, verify its full hash, and expose a local range reader.

    An interrupted or failed transfer is never exposed as a staged source. A successful retry reuses
    the content-addressed file after independently verifying it, without reading the remote source.
    """
    root = _prepare_cache_root(cache_root)
    publication_key = f"hf-shard-stage:{source.object_id}"
    algorithm, hexdigest = source.content_hash.split(":", 1)
    expected_chunk = root / "chunks" / f"{algorithm}-{hexdigest}.bin"
    staging_name = hashlib.sha256(
        f"{publication_key}\0{source.object_id}\0{0}\0{source.reader.size_bytes}".encode()
    ).hexdigest()
    expected_staging = root / ".staging" / f"{staging_name}.part"
    for directory, expected in (
        (root / "chunks", expected_chunk),
        (root / ".staging", expected_staging),
    ):
        try:
            if directory.is_symlink():
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "Hugging Face shard-cache state directory is a symlink",
                )
            if directory.exists():
                unexpected = [entry for entry in directory.iterdir() if entry != expected]
                if unexpected:
                    raise AmsError(
                        ErrorCode.BROKER_VIOLATION,
                        "Hugging Face shard cache already contains another source lease",
                    )
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "Hugging Face shard-cache state could not be inspected",
                retriable=True,
            ) from exc
    published = copy_range_atomic(
        source.reader,
        ByteRange(
            object_id=source.object_id,
            offset=0,
            length=source.reader.size_bytes,
            checksum=source.content_hash,
        ),
        root,
        publication_key,
        buffer_bytes=buffer_bytes,
    )
    descriptor = StorageObject(
        object_id=source.object_id,
        uri=published.name,
        size_bytes=source.reader.size_bytes,
        alignment_bytes=1,
        content_hash=source.content_hash,
    )
    local_source = HuggingFaceShardSource(
        shard_name=source.shard_name,
        object_id=source.object_id,
        content_hash=source.content_hash,
        reader=FileRangeStore(published, descriptor),
    )
    return StagedHuggingFaceShard(root, published, local_source)


def validate_huggingface_shard_cache_empty(cache_root: Path) -> None:
    """Prove a marked ephemeral cache retains no source or partial source object."""
    if not cache_root.exists():
        return
    root = cache_root.resolve(strict=True)
    _validate_marker(root)
    try:
        for directory in (root / "chunks", root / ".staging"):
            if directory.is_symlink():
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "Hugging Face shard-cache state directory is a symlink",
                )
            if directory.exists() and next(directory.iterdir(), None) is not None:
                raise AmsError(
                    ErrorCode.BROKER_VIOLATION,
                    "Hugging Face shard cache retained a source object",
                )
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "Hugging Face shard-cache state could not be inspected",
            retriable=True,
        ) from exc


def release_huggingface_shard_source(
    source: HuggingFaceShardSource,
    cache_root: Path,
    *,
    declared_path: Path | None = None,
) -> None:
    """Idempotently remove one exact expected source object from a marked cache root."""
    try:
        if not cache_root.exists():
            return
        root = cache_root.resolve(strict=True)
        _validate_marker(root)
        algorithm, hexdigest = source.content_hash.split(":", 1)
        expected = (root / "chunks" / f"{algorithm}-{hexdigest}.bin").resolve(strict=False)
        if declared_path is not None and declared_path.resolve(strict=False) != expected:
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "staged Hugging Face shard path is outside its exact cache slot",
            )
        if not expected.exists():
            return
        if expected.is_symlink() or not expected.is_file():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "staged Hugging Face shard is not a regular file",
            )
        stat = expected.stat()
        if stat.st_size != source.reader.size_bytes:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "staged Hugging Face shard size changed before release",
            )
        expected.unlink()
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "staged Hugging Face shard could not be released",
            retriable=True,
        ) from exc


def release_huggingface_shard(stage: StagedHuggingFaceShard) -> None:
    """Release a concrete staged lease through the exact-source cleanup boundary."""
    release_huggingface_shard_source(
        stage.source,
        stage.cache_root,
        declared_path=stage.path,
    )
