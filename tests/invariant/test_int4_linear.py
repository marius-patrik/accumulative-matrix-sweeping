import hashlib
import struct
from io import BytesIO

import pytest

from ams.codecs import Int4CodecConfig, decode_int4_reference, encode_int4_stream
from ams.descriptors import ByteRange, DType
from ams.errors import AmsError, ErrorCode
from ams.ops import Int4StreamedLinearPlan, stream_linear_int4


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


def encode_weights(values: list[float], config: Int4CodecConfig) -> bytes:
    source_payload = struct.pack(f"<{len(values)}f", *values)
    sink = BytesIO()
    encode_int4_stream(
        MemoryReader(source_payload),
        ByteRange("source", 0, len(source_payload), digest(source_payload)),
        (len(values),),
        DType.FLOAT32,
        sink,
        config,
    )
    return sink.getvalue()


@pytest.mark.parametrize("arena_bytes", [72, 88, 256])
def test_int4_linear_streams_oversize_weights_with_dequantized_parity(
    arena_bytes: int,
) -> None:
    rows, columns = 101, 103
    values = [float((index % 29) - 14) / 7 for index in range(rows * columns)]
    vector = [float((index % 13) - 6) / 5 for index in range(columns)]
    bias = [float((index % 7) - 3) / 11 for index in range(rows)]
    config = Int4CodecConfig(group_size=7)
    encoded = encode_weights(values, config)
    decoded = decode_int4_reference(encoded, rows * columns, config)
    expected = []
    for row in range(rows):
        accumulator = bias[row]
        for column in range(columns):
            accumulator += decoded[row * columns + column] * vector[column]
        expected.append(accumulator)

    reader = MemoryReader(encoded)
    plan = Int4StreamedLinearPlan.create(
        rows=rows,
        columns=columns,
        weight_offset=0,
        arena_bytes=arena_bytes,
        config=config,
    )
    actual: list[float] = []
    stream_linear_int4(
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


@pytest.mark.parametrize("packed", [0x08, 0x10])
def test_int4_linear_rejects_reserved_codes_and_noncanonical_padding(packed: int) -> None:
    config = Int4CodecConfig(group_size=5)
    encoded = bytearray(encode_weights([1.0], config))
    encoded[-1] = packed
    plan = Int4StreamedLinearPlan.create(
        rows=1,
        columns=1,
        weight_offset=0,
        arena_bytes=55,
        config=config,
    )
    with pytest.raises(AmsError) as caught:
        stream_linear_int4(MemoryReader(bytes(encoded)), plan, [1.0], lambda *_: None)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


def test_int4_linear_rejects_subminimum_arena() -> None:
    with pytest.raises(AmsError) as caught:
        Int4StreamedLinearPlan.create(
            rows=1,
            columns=1,
            weight_offset=0,
            arena_bytes=54,
            config=Int4CodecConfig(group_size=5),
        )
    assert caught.value.code is ErrorCode.PREFLIGHT_NO_WORKING_SET


def test_int4_linear_rejects_nonfinite_input_before_storage_reads() -> None:
    config = Int4CodecConfig(group_size=5)
    reader = MemoryReader(encode_weights([1.0], config))
    plan = Int4StreamedLinearPlan.create(
        rows=1,
        columns=1,
        weight_offset=0,
        arena_bytes=55,
        config=config,
    )
    with pytest.raises(AmsError) as caught:
        stream_linear_int4(reader, plan, [float("inf")], lambda *_: None)
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE
    assert reader.maximum_read == 0
