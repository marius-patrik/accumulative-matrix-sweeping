import struct

import pytest

from ams.codecs import (
    Int3DiagnosticCodecConfig,
    decode_int3_group_reference,
    encode_int3_group_reference,
)
from ams.errors import AmsError, ErrorCode


def test_known_group_has_stable_signed_three_bit_encoding() -> None:
    config = Int3DiagnosticCodecConfig(group_size=7)
    values = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    payload = encode_int3_group_reference(values, config)
    assert payload == struct.pack("<f3B", 1.0, 0xF5, 0x11, 0x0D)
    assert decode_int3_group_reference(payload, len(values)) == values
    assert config.group_record_size(len(values)) == 7
    assert config.encoded_size(len(values)) == len(payload)


def test_group_size_128_has_exact_packable_storage_bound() -> None:
    config = Int3DiagnosticCodecConfig(group_size=128)
    assert config.group_record_size(128) == 52
    assert config.encoded_size(129) == 57


def test_group_encoder_rejects_nonfinite_oversized_and_unrepresentable_values() -> None:
    config = Int3DiagnosticCodecConfig(group_size=2)
    with pytest.raises(AmsError) as caught:
        encode_int3_group_reference([0.0, float("nan")], config)
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE
    with pytest.raises(AmsError) as caught:
        encode_int3_group_reference([0.0] * 3, config)
    assert caught.value.code is ErrorCode.PLAN_INVALID
    with pytest.raises(AmsError) as caught:
        encode_int3_group_reference([10**10_000], config)
    assert caught.value.code is ErrorCode.NUMERIC_FAILURE


def test_decoder_rejects_reserved_code_and_noncanonical_tail_bits() -> None:
    with pytest.raises(AmsError, match="reserved") as caught:
        decode_int3_group_reference(struct.pack("<fB", 1.0, 0x04), 1)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE
    with pytest.raises(AmsError, match="padding") as caught:
        decode_int3_group_reference(struct.pack("<fB", 1.0, 0x08), 1)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE
