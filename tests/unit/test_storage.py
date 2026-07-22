import hashlib
from pathlib import Path

import pytest

from ams.descriptors import StorageObject
from ams.errors import AmsError, ErrorCode
from ams.storage import FileRangeStore


def descriptor(payload: bytes) -> StorageObject:
    digest = hashlib.sha256(payload).hexdigest()
    return StorageObject(
        object_id="fixture",
        uri="fixture.bin",
        size_bytes=len(payload),
        alignment_bytes=1,
        content_hash=f"sha256:{digest}",
    )


def test_file_store_reads_exact_range_without_return_allocation(tmp_path: Path) -> None:
    payload = bytes(range(32))
    path = tmp_path / "fixture.bin"
    path.write_bytes(payload)
    store = FileRangeStore(path, descriptor(payload))
    destination = bytearray(7)
    store.read_into(11, destination)
    assert destination == payload[11:18]
    store.verify_content_hash(buffer_bytes=5)


def test_file_store_rejects_declared_size_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"1234")
    wrong = StorageObject("fixture", "fixture.bin", 3, 1, "sha256:" + "0" * 64)
    with pytest.raises(AmsError, match="size"):
        FileRangeStore(path, wrong)


def test_file_store_rejects_out_of_bounds_read(tmp_path: Path) -> None:
    payload = b"1234"
    path = tmp_path / "fixture.bin"
    path.write_bytes(payload)
    store = FileRangeStore(path, descriptor(payload))
    with pytest.raises(AmsError) as caught:
        store.read_into(2, bytearray(3))
    assert caught.value.code is ErrorCode.IO_FAILURE
