import hashlib
import struct
from pathlib import Path

from ams.descriptors import StorageObject
from ams.ops import StreamedLinearPlan, stream_linear_f32
from ams.storage import FileRangeStore


class HighWaterReader:
    def __init__(self, inner: FileRangeStore):
        self._inner = inner
        self.size_bytes = inner.size_bytes
        self.maximum_read_bytes = 0

    def read_into(self, offset: int, destination) -> None:
        requested = memoryview(destination).nbytes
        self.maximum_read_bytes = max(self.maximum_read_bytes, requested)
        self._inner.read_into(offset, destination)


def test_weight_tensor_larger_than_arena_has_source_order_parity(tmp_path: Path) -> None:
    rows, columns = 127, 131
    source_weights = [((index % 29) - 14) / 17 for index in range(rows * columns)]
    payload = struct.pack(f"<{len(source_weights)}f", *source_weights)
    weights = struct.unpack(f"<{len(source_weights)}f", payload)
    vector = [((index % 11) - 5) / 13 for index in range(columns)]
    path = tmp_path / "oversize-weights.bin"
    path.write_bytes(payload)
    descriptor = StorageObject(
        "oversize-weights",
        "oversize-weights.bin",
        len(payload),
        4,
        "sha256:" + hashlib.sha256(payload).hexdigest(),
    )
    reader = HighWaterReader(FileRangeStore(path, descriptor))
    plan = StreamedLinearPlan.create(
        rows=rows,
        columns=columns,
        weight_offset=0,
        arena_bytes=28,
    )
    actual: list[float] = []
    stream_linear_f32(reader, plan, vector, lambda _, value: actual.append(value))

    expected = []
    for row in range(rows):
        accumulator = 0.0
        for column in range(columns):
            accumulator += weights[row * columns + column] * vector[column]
        expected.append(accumulator)

    assert actual == expected
    assert reader.size_bytes > plan.arena_bytes * 2_000
    assert reader.maximum_read_bytes + 8 == plan.working_set_bytes
    assert plan.working_set_bytes <= plan.arena_bytes
