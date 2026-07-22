import hashlib
import struct
from pathlib import Path

import pytest

from ams.descriptors import DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.ops import StreamedLinearPlan, stream_linear_f32, stream_linear_identity
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


class MemoryReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.size_bytes = len(payload)
        self.maximum_read = 0

    def read_into(self, offset: int, destination) -> None:
        view = memoryview(destination).cast("B")
        try:
            self.maximum_read = max(self.maximum_read, view.nbytes)
            view[:] = self.payload[offset : offset + view.nbytes]
        finally:
            view.release()


def encode_identity(values: list[float], dtype: DType) -> tuple[bytes, list[float]]:
    if dtype is DType.FLOAT32:
        payload = struct.pack(f"<{len(values)}f", *values)
        decoded = list(struct.unpack(f"<{len(values)}f", payload))
    elif dtype is DType.FLOAT16:
        payload = struct.pack(f"<{len(values)}e", *values)
        decoded = list(struct.unpack(f"<{len(values)}e", payload))
    else:
        words = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
        payload = struct.pack(f"<{len(words)}H", *words)
        decoded = [struct.unpack("<f", struct.pack("<I", word << 16))[0] for word in words]
    return payload, decoded


@pytest.mark.parametrize("dtype", [DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32])
def test_streamed_identity_linear_supports_official_glm_float_dtypes(dtype: DType) -> None:
    rows, columns = 3, 7
    values = [((index % 11) - 5) / 7 for index in range(rows * columns)]
    vector_values = [((index % 5) - 2) / 3 for index in range(columns)]
    payload, decoded = encode_identity(values, dtype)
    reader = MemoryReader(payload)
    plan = StreamedLinearPlan.create(
        rows=rows,
        columns=columns,
        weight_offset=0,
        arena_bytes=12,
        storage_dtype=dtype,
    )
    actual: list[float] = []
    stream_linear_identity(reader, plan, vector_values, lambda _, value: actual.append(value))
    expected = []
    for row in range(rows):
        accumulator = 0.0
        for column in range(columns):
            accumulator += decoded[row * columns + column] * vector_values[column]
        expected.append(accumulator)
    assert actual == expected
    assert reader.maximum_read + 8 <= plan.arena_bytes


def test_streamed_identity_linear_rejects_nonfinite_input() -> None:
    payload, _ = encode_identity([1.0], DType.FLOAT32)
    reader = MemoryReader(payload)
    plan = StreamedLinearPlan.create(rows=1, columns=1, weight_offset=0, arena_bytes=12)
    with pytest.raises(AmsError) as caught:
        stream_linear_identity(reader, plan, [float("nan")], lambda *_: None)
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE
    assert reader.maximum_read == 0
