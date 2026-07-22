"""Bounded, idempotent identity publication for content-addressed ranges."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ams.checked import checked_add
from ams.descriptors import ByteRange, validate_identifier
from ams.errors import AmsError, ErrorCode
from ams.storage.file import RangeReader


def _hash_file(path: Path, *, expected_size: int, buffer_bytes: int, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    buffer = bytearray(min(buffer_bytes, max(expected_size, 1)))
    try:
        if path.is_symlink() or not path.is_file():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "published chunk is not a regular file",
            )
        if path.stat().st_size != expected_size:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "published chunk has an unexpected size")
        with path.open("rb", buffering=0) as handle:
            while True:
                count = handle.readinto(buffer)
                if not count:
                    break
                digest.update(memoryview(buffer)[:count])
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "published chunk verification failed",
            retriable=True,
        ) from exc
    return digest.hexdigest()


def copy_range_atomic(
    reader: RangeReader,
    source: ByteRange,
    destination_root: Path,
    publication_key: str,
    *,
    buffer_bytes: int = 1024 * 1024,
) -> Path:
    """Copy one verified range and atomically publish it by content hash.

    A retry revalidates an already-published chunk without reading the source. An interrupted
    staging file is safely overwritten because it is never a visible package root.
    """
    validate_identifier(publication_key, name="publication_key")
    if isinstance(buffer_bytes, bool) or not isinstance(buffer_bytes, int) or buffer_bytes <= 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "copy buffer size must be a positive integer")
    source.validate_within(reader.size_bytes)
    algorithm, expected = source.checksum.split(":", 1)
    if algorithm not in hashlib.algorithms_available or algorithm == "blake3":
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"content hash backend is unavailable: {algorithm}",
        )

    try:
        root = destination_root.resolve(strict=False)
        chunks = root / "chunks"
        staging = root / ".staging"
        chunks.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "conversion output directories could not be created",
            retriable=True,
        ) from exc

    final_path = chunks / f"{algorithm}-{expected}.bin"
    if final_path.exists():
        actual = _hash_file(
            final_path,
            expected_size=source.length,
            buffer_bytes=buffer_bytes,
            algorithm=algorithm,
        )
        if actual != expected:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "published chunk content hash mismatch")
        return final_path

    staging_name = hashlib.sha256(
        f"{publication_key}\0{source.object_id}\0{source.offset}\0{source.length}".encode()
    ).hexdigest()
    staging_path = staging / f"{staging_name}.part"
    digest = hashlib.new(algorithm)
    buffer = bytearray(min(buffer_bytes, source.length))
    view = memoryview(buffer)
    try:
        with staging_path.open("wb", buffering=0) as handle:
            completed = 0
            while completed < source.length:
                count = min(len(buffer), source.length - completed)
                window = view[:count]
                source_offset = checked_add(source.offset, completed, name="copy.source_offset")
                reader.read_into(source_offset, window)
                digest.update(window)
                written = 0
                while written < count:
                    result = handle.write(window[written:])
                    if result is None or result == 0:
                        raise AmsError(
                            ErrorCode.IO_FAILURE,
                            "short write to conversion staging file",
                            retriable=True,
                        )
                    written += result
                completed += count
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "source range content hash mismatch")
        os.replace(staging_path, final_path)
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "conversion range publication failed",
            retriable=True,
        ) from exc
    finally:
        view.release()
    return final_path
