import hashlib
import struct
from io import BytesIO

import pytest

from ams.codecs import TernaryCodecConfig, decode_ternary_reference, encode_ternary_stream
from ams.descriptors import ByteRange, DType
from ams.errors import AmsError, ErrorCode
from ams.ops import TernaryStreamedLinearPlan, stream_linear_ternary


class MemoryReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.size_bytes = len(payload)
        self.maximum_read = 0

    def read_into(self, offset: int, destination) -> None:
        view = memoryview(destination).cast("B")
        self.maximum_read = max(self.maximum_read, view.nbytes)
        view[:] = self.payload[offset : offset + view.nbytes]


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def encode_weights(values: list[float], config: TernaryCodecConfig) -> bytes:
    source_payload = struct.pack(f"<{len(values)}f", *values)
    sink = BytesIO()
    encode_ternary_stream(
        MemoryReader(source_payload),
        ByteRange("source", 0, len(source_payload), digest(source_payload)),
        (len(values),),
        DType.FLOAT32,
        sink,
        config,
    )
    return sink.getvalue()


@pytest.mark.parametrize("arena_bytes", [70, 86, 256])
def test_ternary_linear_streams_oversize_weights_with_dequantized_parity(
    arena_bytes: int,
) -> None:
    rows, columns = 101, 103
    values = [float((index % 29) - 14) / 7 for index in range(rows * columns)]
    vector = [float((index % 13) - 6) / 5 for index in range(columns)]
    bias = [float((index % 7) - 3) / 11 for index in range(rows)]
    config = TernaryCodecConfig(group_size=7)
    encoded = encode_weights(values, config)
    decoded = decode_ternary_reference(encoded, rows * columns, config)
    expected = []
    for row in range(rows):
        accumulator = bias[row]
        for column in range(columns):
            accumulator += decoded[row * columns + column] * vector[column]
        expected.append(accumulator)

    reader = MemoryReader(encoded)
    plan = TernaryStreamedLinearPlan.create(
        rows=rows,
        columns=columns,
        weight_offset=0,
        arena_bytes=arena_bytes,
        config=config,
    )
    actual: list[float] = []
    stream_linear_ternary(
        reader,
        plan,
        vector,
        lambda _, value: actual.append(value),
        bias=bias,
    )
    assert actual == expected
    assert len(encoded) > arena_bytes * 30
    assert reader.maximum_read <= config.group_record_size(config.group_size)
    assert plan.working_set_bytes <= arena_bytes


def test_ternary_linear_rejects_corrupt_tail_padding() -> None:
    config = TernaryCodecConfig(group_size=5)
    encoded = bytearray(encode_weights([1.0], config))
    encoded[-1] = 2 + 2 * 3 + 1 * 9 + 1 * 27 + 1 * 81
    plan = TernaryStreamedLinearPlan.create(
        rows=1,
        columns=1,
        weight_offset=0,
        arena_bytes=53,
        config=config,
    )
    with pytest.raises(AmsError) as caught:
        stream_linear_ternary(MemoryReader(bytes(encoded)), plan, [1.0], lambda *_: None)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


def test_ternary_linear_rejects_subminimum_arena() -> None:
    with pytest.raises(AmsError) as caught:
        TernaryStreamedLinearPlan.create(
            rows=1,
            columns=1,
            weight_offset=0,
            arena_bytes=52,
            config=TernaryCodecConfig(group_size=5),
        )
    assert caught.value.code is ErrorCode.PREFLIGHT_NO_WORKING_SET
