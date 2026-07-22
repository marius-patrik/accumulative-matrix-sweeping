import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ams.descriptors import DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import SafetensorsLimits, parse_safetensors_header
from ams.storage import FileRangeStore


class MemoryReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.size_bytes = len(payload)

    def read_into(self, offset: int, destination) -> None:
        view = memoryview(destination).cast("B")
        view[:] = self.payload[offset : offset + view.nbytes]


def build_file(header: dict, data: bytes, *, padding: int = 0) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode() + b" " * padding
    return struct.pack("<Q", len(encoded)) + encoded + data


def test_safetensors_header_normalizes_valid_external_metadata() -> None:
    header = {
        "z.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [4, 12]},
        "a.bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "__metadata__": {"format": "pt"},
    }
    parsed = parse_safetensors_header(MemoryReader(build_file(header, bytes(12), padding=3)))
    assert [tensor.source_name for tensor in parsed.tensors] == ["a.bias", "z.weight"]
    assert parsed.tensors[1].dtype is DType.BFLOAT16
    assert parsed.tensors[1].absolute_offset == parsed.data_offset + 4
    assert parsed.metadata == (("format", "pt"),)


def test_safetensors_boundary_rejects_duplicate_tensor_names() -> None:
    encoded = (
        b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
        b'"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    )
    payload = struct.pack("<Q", len(encoded)) + encoded + bytes(4)
    with pytest.raises(AmsError) as caught:
        parse_safetensors_header(MemoryReader(payload))
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


@pytest.mark.parametrize(
    "header,data",
    [
        (
            {"x": {"dtype": "F32", "shape": [1], "data_offsets": [1, 5]}},
            bytes(5),
        ),
        (
            {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
            bytes(4),
        ),
        (
            {"x": {"dtype": "UNKNOWN", "shape": [1], "data_offsets": [0, 4]}},
            bytes(4),
        ),
    ],
)
def test_safetensors_boundary_rejects_gap_size_and_dtype_drift(header, data) -> None:
    with pytest.raises(AmsError):
        parse_safetensors_header(MemoryReader(build_file(header, data)))


def test_safetensors_header_limit_is_checked_before_allocation() -> None:
    payload = struct.pack("<Q", 1024) + b"{}"
    with pytest.raises(AmsError, match="configured limit"):
        parse_safetensors_header(
            MemoryReader(payload),
            SafetensorsLimits(max_header_bytes=16),
        )


def test_parser_matches_a_file_emitted_by_official_safetensors(tmp_path: Path) -> None:
    path = tmp_path / "official.safetensors"
    save_file(
        {
            "weight": np.arange(12, dtype=np.float32).reshape(3, 4),
            "bias": np.arange(3, dtype=np.int64),
        },
        path,
        metadata={"format": "numpy"},
    )
    payload_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = StorageObject(
        "official-fixture",
        "official.safetensors",
        path.stat().st_size,
        1,
        f"sha256:{payload_hash}",
    )
    parsed = parse_safetensors_header(FileRangeStore(path, descriptor))
    assert [(tensor.source_name, tensor.dtype, tensor.shape) for tensor in parsed.tensors] == [
        ("bias", DType.INT64, (3,)),
        ("weight", DType.FLOAT32, (3, 4)),
    ]
    assert parsed.metadata == (("format", "numpy"),)
