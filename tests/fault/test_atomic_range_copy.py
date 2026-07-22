import hashlib
from pathlib import Path

import pytest

from ams.descriptors import ByteRange
from ams.errors import AmsError, ErrorCode
from ams.storage import copy_range_atomic


class MemoryReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.size_bytes = len(payload)
        self.reads = 0

    def read_into(self, offset: int, destination) -> None:
        self.reads += 1
        view = memoryview(destination).cast("B")
        view[:] = self.payload[offset : offset + view.nbytes]


class FailingReader(MemoryReader):
    def read_into(self, offset: int, destination) -> None:
        if self.reads == 1:
            raise AmsError(ErrorCode.IO_FAILURE, "injected read failure", retriable=True)
        super().read_into(offset, destination)


def source_range(payload: bytes) -> ByteRange:
    return ByteRange(
        "source-shard",
        0,
        len(payload),
        "sha256:" + hashlib.sha256(payload).hexdigest(),
    )


def test_interrupted_copy_is_retryable_and_idempotent(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 8
    source = source_range(payload)
    failing = FailingReader(payload)
    with pytest.raises(AmsError) as caught:
        copy_range_atomic(failing, source, tmp_path, "tensor-0", buffer_bytes=127)
    assert caught.value.code is ErrorCode.IO_FAILURE
    assert caught.value.retriable
    assert not list((tmp_path / "chunks").glob("*.bin"))

    healthy = MemoryReader(payload)
    published = copy_range_atomic(healthy, source, tmp_path, "tensor-0", buffer_bytes=127)
    assert published.read_bytes() == payload
    assert healthy.reads > 1

    already_published = MemoryReader(payload)
    assert (
        copy_range_atomic(
            already_published,
            source,
            tmp_path,
            "tensor-0",
            buffer_bytes=31,
        )
        == published
    )
    assert already_published.reads == 0


def test_corrupt_published_chunk_fails_without_source_retry(tmp_path: Path) -> None:
    payload = b"correct payload"
    source = source_range(payload)
    published = copy_range_atomic(MemoryReader(payload), source, tmp_path, "tensor-0")
    published.write_bytes(b"corrupt payload")
    reader = MemoryReader(payload)
    with pytest.raises(AmsError) as caught:
        copy_range_atomic(reader, source, tmp_path, "tensor-0")
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE
    assert reader.reads == 0
