import hashlib
import math
import struct
from io import BytesIO

import pytest

from ams.codecs import (
    Int4CodecConfig,
    decode_int4_group_reference,
    decode_int4_reference,
    encode_int4_group_reference,
    encode_int4_stream,
    encode_int4_stream_numpy,
)
from ams.descriptors import ByteRange, DType
from ams.errors import AmsError, ErrorCode


class ObservedMemoryReader:
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


def source(payload: bytes) -> ByteRange:
    return ByteRange("source", 0, len(payload), digest(payload))


def encode(values: list[float], dtype: DType, config: Int4CodecConfig):
    if dtype is DType.FLOAT32:
        payload = struct.pack(f"<{len(values)}f", *values)
    elif dtype is DType.FLOAT16:
        payload = struct.pack(f"<{len(values)}e", *values)
    elif dtype is DType.BFLOAT16:
        words = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
        payload = struct.pack(f"<{len(words)}H", *words)
    else:
        raise AssertionError(dtype)
    reader = ObservedMemoryReader(payload)
    sink = BytesIO()
    result = encode_int4_stream(reader, source(payload), (len(values),), dtype, sink, config)
    return sink.getvalue(), result, reader


def encode_numpy(
    values: list[float],
    dtype: DType,
    config: Int4CodecConfig,
    *,
    maximum_source_read_bytes: int,
):
    if dtype is DType.FLOAT32:
        payload = struct.pack(f"<{len(values)}f", *values)
    elif dtype is DType.FLOAT16:
        payload = struct.pack(f"<{len(values)}e", *values)
    elif dtype is DType.BFLOAT16:
        words = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
        payload = struct.pack(f"<{len(words)}H", *words)
    else:
        raise AssertionError(dtype)
    reader = ObservedMemoryReader(payload)
    sink = BytesIO()
    result = encode_int4_stream_numpy(
        reader,
        source(payload),
        (len(values),),
        dtype,
        sink,
        config,
        maximum_source_read_bytes=maximum_source_read_bytes,
    )
    return sink.getvalue(), result, reader


def test_known_group_has_stable_signed_nibble_encoding() -> None:
    config = Int4CodecConfig(group_size=7)
    payload, result, _ = encode([-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0], DType.FLOAT32, config)
    scale = struct.unpack("<f", struct.pack("<f", 8.0 / 7.0))[0]
    assert payload == struct.pack("<f4B", scale, 0xD9, 0x0F, 0x31, 0x07)
    assert result.content_hash == digest(payload)
    decoded = decode_int4_reference(payload, 7, config)
    expected = [-7 * scale, -3 * scale, -scale, 0.0, scale, 3 * scale, 7 * scale]
    assert decoded == expected


def test_rounding_is_half_away_from_zero_against_the_stored_scale() -> None:
    config = Int4CodecConfig(group_size=7)
    payload, _, _ = encode([0.5, -0.5, 1.49, -1.49, 1.5, -1.5, 7.0], DType.FLOAT32, config)
    assert payload == struct.pack("<f4B", 1.0, 0xF1, 0xF1, 0xE2, 0x07)


def test_public_group_encoder_is_exact_and_rejects_invalid_values() -> None:
    config = Int4CodecConfig(group_size=3)
    values = [-1.0, 0.0, 1.0]
    payload, _, _ = encode(values, DType.FLOAT32, config)
    group_payload = encode_int4_group_reference(values, config)
    assert group_payload == payload
    assert decode_int4_group_reference(group_payload, len(values)) == decode_int4_reference(
        payload, len(values), config
    )
    with pytest.raises(AmsError) as caught:
        encode_int4_group_reference([0.0, float("nan")], config)
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE
    with pytest.raises(AmsError) as caught:
        encode_int4_group_reference([0.0] * 4, config)
    assert caught.value.code is ErrorCode.PLAN_INVALID
    with pytest.raises(AmsError) as caught:
        encode_int4_group_reference([10**10_000], config)
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE


@pytest.mark.parametrize("dtype", [DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32])
def test_supported_source_dtypes_produce_identical_output(dtype: DType) -> None:
    config = Int4CodecConfig(group_size=5)
    payload, _, _ = encode([-2.0, -1.0, 0.0, 1.0, 2.0], dtype, config)
    expected, _, _ = encode([-2.0, -1.0, 0.0, 1.0, 2.0], DType.FLOAT32, config)
    assert payload == expected


def test_large_tensor_has_group_bounded_source_reads() -> None:
    values = [float((index % 17) - 8) for index in range(1001)]
    config = Int4CodecConfig(group_size=7)
    payload, result, reader = encode(values, DType.BFLOAT16, config)
    assert len(payload) == config.encoded_size(len(values))
    assert result.group_count == math.ceil(len(values) / config.group_size)
    assert result.maximum_source_read_bytes == config.group_size * 2
    assert reader.maximum_read == config.group_size * 2
    assert reader.size_bytes > reader.maximum_read * 100


@pytest.mark.parametrize("dtype", [DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32])
@pytest.mark.parametrize("group_size", [3, 4, 7, 128])
def test_numpy_bulk_encoder_is_byte_exact_with_scalar_oracle(
    dtype: DType,
    group_size: int,
) -> None:
    values = [
        0.0 if index % 29 == 0 else ((index * 104_729) % 20_003 - 10_001) / 997.0
        for index in range(301)
    ]
    config = Int4CodecConfig(group_size=group_size)
    expected, scalar, _ = encode(values, dtype, config)
    actual, bulk, _ = encode_numpy(
        values,
        dtype,
        config,
        maximum_source_read_bytes=group_size * (4 if dtype is DType.FLOAT32 else 2) * 5,
    )

    assert actual == expected
    assert bulk.content_hash == scalar.content_hash
    assert bulk.source_checksum == scalar.source_checksum
    assert bulk.element_count == scalar.element_count
    assert bulk.group_count == scalar.group_count
    assert bulk.encoded_bytes == scalar.encoded_bytes
    assert bulk.decoded_bytes == scalar.decoded_bytes


def test_numpy_bulk_encoder_obeys_read_bound_and_rejects_tamper() -> None:
    values = [float((index % 17) - 8) for index in range(1_001)]
    config = Int4CodecConfig(group_size=7)
    maximum_read = config.group_size * 2 * 5
    payload, result, reader = encode_numpy(
        values,
        DType.BFLOAT16,
        config,
        maximum_source_read_bytes=maximum_read,
    )
    assert len(payload) == config.encoded_size(len(values))
    assert result.maximum_source_read_bytes == maximum_read
    assert reader.maximum_read == maximum_read

    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    with pytest.raises(AmsError) as caught:
        encode_int4_stream_numpy(
            ObservedMemoryReader(raw),
            ByteRange("source", 0, len(raw), "sha256:" + "0" * 64),
            (4,),
            DType.FLOAT32,
            BytesIO(),
            Int4CodecConfig(group_size=2),
            maximum_source_read_bytes=16,
        )
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_integrity_numeric_reserved_and_padding_fail_closed() -> None:
    payload = struct.pack("<2f", 1.0, -1.0)
    wrong = ByteRange("source", 0, len(payload), "sha256:" + "0" * 64)
    with pytest.raises(AmsError) as caught:
        encode_int4_stream(
            ObservedMemoryReader(payload),
            wrong,
            (2,),
            DType.FLOAT32,
            BytesIO(),
        )
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE

    nonfinite = struct.pack("<f", float("nan"))
    with pytest.raises(AmsError) as caught:
        encode_int4_stream(
            ObservedMemoryReader(nonfinite),
            source(nonfinite),
            (1,),
            DType.FLOAT32,
            BytesIO(),
        )
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE

    config = Int4CodecConfig(group_size=2)
    with pytest.raises(AmsError, match="reserved"):
        decode_int4_reference(struct.pack("<fB", 1.0, 0x08), 2, config)
    with pytest.raises(AmsError, match="padding"):
        decode_int4_reference(struct.pack("<fB", 1.0, 0x10), 1, config)
