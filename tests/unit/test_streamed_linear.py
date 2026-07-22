import hashlib
import struct
from pathlib import Path

import pytest

from ams.descriptors import StorageObject
from ams.errors import AmsError, ErrorCode
from ams.ops import StreamedLinearPlan, stream_linear_f32
from ams.storage import FileRangeStore


class ObservedReader:
    def __init__(self, inner: FileRangeStore):
        self.inner = inner
        self.size_bytes = inner.size_bytes
        self.maximum_read = 0
        self.total_read = 0

    def read_into(self, offset: int, destination) -> None:
        size = memoryview(destination).nbytes
        self.maximum_read = max(self.maximum_read, size)
        self.total_read += size
        self.inner.read_into(offset, destination)


def write_weights(tmp_path: Path, values: list[float]) -> FileRangeStore:
    payload = struct.pack(f"<{len(values)}f", *values)
    path = tmp_path / "weights.bin"
    path.write_bytes(payload)
    storage = StorageObject(
        "weights",
        "weights.bin",
        len(payload),
        4,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
    )
    return FileRangeStore(path, storage)


@pytest.mark.parametrize("arena_bytes", [12, 20, 64])
def test_streamed_linear_matches_source_order_reference(tmp_path: Path, arena_bytes: int) -> None:
    rows, columns = 5, 17
    weights = [((index % 13) - 6) / 7 for index in range(rows * columns)]
    vector = [((index % 5) - 2) / 3 for index in range(columns)]
    reader = ObservedReader(write_weights(tmp_path, weights))
    plan = StreamedLinearPlan.create(
        rows=rows,
        columns=columns,
        weight_offset=0,
        arena_bytes=arena_bytes,
    )
    outputs: list[float] = []
    stream_linear_f32(reader, plan, vector, lambda _, value: outputs.append(value))
    reference = [
        sum(
            struct.unpack("<f", struct.pack("<f", weights[row * columns + column]))[0]
            * vector[column]
            for column in range(columns)
        )
        for row in range(rows)
    ]
    assert outputs == pytest.approx(reference, rel=0, abs=1e-12)
    assert reader.size_bytes > arena_bytes
    assert reader.maximum_read + 8 <= arena_bytes
    assert reader.total_read == reader.size_bytes


def test_streamed_linear_rejects_subminimum_working_set() -> None:
    with pytest.raises(AmsError) as caught:
        StreamedLinearPlan.create(rows=1, columns=1, weight_offset=0, arena_bytes=11)
    assert caught.value.code is ErrorCode.PREFLIGHT_NO_WORKING_SET
