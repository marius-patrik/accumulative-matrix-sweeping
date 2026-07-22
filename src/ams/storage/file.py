"""Synchronous, bounded range reads for the Phase 0 reference runtime."""

from __future__ import annotations

import hashlib
from collections.abc import Buffer
from pathlib import Path
from typing import Protocol

from ams.checked import checked_range_end, checked_uint
from ams.descriptors import StorageObject
from ams.errors import AmsError, ErrorCode


class RangeReader(Protocol):
    size_bytes: int

    def read_into(self, offset: int, destination: Buffer) -> None: ...


class FileRangeStore:
    """An immutable file object whose reads never allocate the requested payload."""

    def __init__(self, path: Path, descriptor: StorageObject):
        self.path = path.resolve(strict=True)
        self.descriptor = descriptor
        self.size_bytes = descriptor.size_bytes
        actual_size = self.path.stat().st_size
        if actual_size != self.size_bytes:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "storage object size does not match its descriptor",
                evidence={"actual_size": actual_size, "declared_size": self.size_bytes},
            )

    def read_into(self, offset: int, destination: Buffer) -> None:
        view = memoryview(destination).cast("B")
        try:
            if view.readonly:
                raise AmsError(ErrorCode.IO_FAILURE, "read destination is read-only")
            length = view.nbytes
            if length == 0:
                checked_uint(offset, name="read.offset")
                if offset > self.size_bytes:
                    raise AmsError(ErrorCode.IO_FAILURE, "zero-length read begins past object end")
                return
            end = checked_range_end(offset, length, name="read")
            if end > self.size_bytes:
                raise AmsError(
                    ErrorCode.IO_FAILURE,
                    "read exceeds storage object",
                    evidence={"offset": offset, "length": length, "size": self.size_bytes},
                )
            with self.path.open("rb", buffering=0) as handle:
                handle.seek(offset)
                completed = 0
                while completed < length:
                    count = handle.readinto(view[completed:])
                    if count is None or count == 0:
                        raise AmsError(
                            ErrorCode.IO_FAILURE,
                            "short read from storage object",
                            retriable=True,
                            evidence={"completed": completed, "requested": length},
                        )
                    completed += count
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "file range read failed",
                retriable=True,
                evidence={"os_error": type(exc).__name__},
            ) from exc
        finally:
            view.release()

    def verify_content_hash(self, *, buffer_bytes: int = 1024 * 1024) -> None:
        if buffer_bytes <= 0:
            raise AmsError(ErrorCode.PLAN_INVALID, "hash buffer size must be positive")
        algorithm, expected = self.descriptor.content_hash.split(":", 1)
        if algorithm == "blake3":
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "BLAKE3 verification requires an optional backend",
            )
        digest = hashlib.new(algorithm)
        buffer = bytearray(min(buffer_bytes, self.size_bytes))
        with self.path.open("rb", buffering=0) as handle:
            while True:
                count = handle.readinto(buffer)
                if not count:
                    break
                digest.update(memoryview(buffer)[:count])
        if digest.hexdigest() != expected:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "storage object hash mismatch")
