import hashlib
import math
import struct
from io import BytesIO

import pytest

from ams.codecs.ternary import (
    TernaryCodecConfig,
    decode_ternary_reference,
    encode_ternary_stream,
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


def encode(values: list[float], dtype: DType, config: TernaryCodecConfig):
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
    result = encode_ternary_stream(
        reader,
        source(payload),
        (len(values),),
        dtype,
        sink,
        config,
    )
    return sink.getvalue(), result, reader


def test_known_group_has_stable_trit5_encoding_and_reconstruction() -> None:
    config = TernaryCodecConfig(group_size=5)
    payload, result, _ = encode([-2.0, -1.0, 0.0, 1.0, 2.0], DType.FLOAT32, config)
    assert payload == struct.pack("<fB", 1.5, 225)
    assert result.content_hash == digest(payload)
    assert decode_ternary_reference(payload, 5, config) == [-1.5, -1.5, 0.0, 1.5, 1.5]


@pytest.mark.parametrize("dtype", [DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32])
def test_supported_source_dtypes_produce_identical_output(dtype: DType) -> None:
    config = TernaryCodecConfig(group_size=5)
    payload, _, _ = encode([-2.0, -1.0, 0.0, 1.0, 2.0], dtype, config)
    assert payload == struct.pack("<fB", 1.5, 225)


def test_source_tensor_larger_than_group_is_read_with_bounded_residency() -> None:
    values = [float((index % 17) - 8) for index in range(1001)]
    config = TernaryCodecConfig(group_size=7)
    payload, result, reader = encode(values, DType.BFLOAT16, config)
    assert len(payload) == config.encoded_size(len(values))
    assert result.group_count == math.ceil(len(values) / config.group_size)
    assert result.maximum_source_read_bytes == config.group_size * 2
    assert reader.maximum_read == config.group_size * 2
    assert reader.size_bytes > reader.maximum_read * 100


def test_source_checksum_mismatch_is_integrity_failure() -> None:
    payload = struct.pack("<2f", 1.0, -1.0)
    wrong = ByteRange("source", 0, len(payload), "sha256:" + "0" * 64)
    with pytest.raises(AmsError) as caught:
        encode_ternary_stream(
            ObservedMemoryReader(payload),
            wrong,
            (2,),
            DType.FLOAT32,
            BytesIO(),
        )
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_source_is_numeric_failure(bad_value: float) -> None:
    payload = struct.pack("<f", bad_value)
    with pytest.raises(AmsError) as caught:
        encode_ternary_stream(
            ObservedMemoryReader(payload),
            source(payload),
            (1,),
            DType.FLOAT32,
            BytesIO(),
        )
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE


def test_decoder_rejects_reserved_byte_and_noncanonical_padding() -> None:
    config = TernaryCodecConfig(group_size=5)
    with pytest.raises(AmsError, match="exceeds"):
        decode_ternary_reference(struct.pack("<fB", 1.0, 243), 5, config)
    noncanonical_padding = 2 + 2 * 3 + 1 * 9 + 1 * 27 + 1 * 81
    with pytest.raises(AmsError, match="padding"):
        decode_ternary_reference(struct.pack("<fB", 1.0, noncanonical_padding), 1, config)
