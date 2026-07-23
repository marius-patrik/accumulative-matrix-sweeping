import hashlib
from pathlib import Path

import pytest

from ams.errors import AmsError, ErrorCode
from ci import verify_glm4_official_layer as verifier


def _open_fixture(path: Path, payload: bytes):
    return verifier._open_authenticated_shard(
        path,
        expected_name=path.name,
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        object_id="hf:test-layer",
        label="test layer shard",
        buffer_bytes=7,
    )


def test_authenticated_layer_shard_accepts_the_exact_pinned_payload(
    tmp_path: Path,
) -> None:
    payload = b"authenticated GLM layer fixture"
    path = tmp_path / "model-layer.safetensors"
    path.write_bytes(payload)

    reader = _open_fixture(path, payload)

    destination = bytearray(len(payload))
    reader.read_into(0, destination)
    assert destination == payload


def test_authenticated_layer_shard_rejects_same_size_content_corruption(
    tmp_path: Path,
) -> None:
    payload = b"authenticated GLM layer fixture"
    path = tmp_path / "model-layer.safetensors"
    path.write_bytes(b"x" * len(payload))

    with pytest.raises(AmsError, match="hash mismatch") as caught:
        _open_fixture(path, payload)

    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_authenticated_layer_shard_rejects_truncated_source(
    tmp_path: Path,
) -> None:
    payload = b"authenticated GLM layer fixture"
    path = tmp_path / "model-layer.safetensors"
    path.write_bytes(payload[:-1])

    with pytest.raises(AmsError, match="regular-file size") as caught:
        _open_fixture(path, payload)

    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE
