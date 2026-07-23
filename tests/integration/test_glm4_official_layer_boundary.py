import hashlib
from pathlib import Path

import pytest

from ams.errors import AmsError, ErrorCode
from ci import verify_glm4_official_layer as verifier


def _pin_fixture(monkeypatch: pytest.MonkeyPatch, path: Path, payload: bytes) -> None:
    monkeypatch.setattr(verifier, "_SHARD_NAME", path.name)
    monkeypatch.setattr(verifier, "_SHARD_BYTES", len(payload))
    monkeypatch.setattr(verifier, "_SHARD_SHA256", hashlib.sha256(payload).hexdigest())


def test_authenticated_layer_shard_accepts_the_exact_pinned_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"authenticated GLM layer fixture"
    path = tmp_path / "model-layer.safetensors"
    path.write_bytes(payload)
    _pin_fixture(monkeypatch, path, payload)

    reader = verifier._open_authenticated_layer_shard(path, buffer_bytes=7)

    destination = bytearray(len(payload))
    reader.read_into(0, destination)
    assert destination == payload


def test_authenticated_layer_shard_rejects_same_size_content_corruption(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"authenticated GLM layer fixture"
    path = tmp_path / "model-layer.safetensors"
    path.write_bytes(b"x" * len(payload))
    _pin_fixture(monkeypatch, path, payload)

    with pytest.raises(AmsError, match="hash mismatch") as caught:
        verifier._open_authenticated_layer_shard(path, buffer_bytes=7)

    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_authenticated_layer_shard_rejects_truncated_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"authenticated GLM layer fixture"
    path = tmp_path / "model-layer.safetensors"
    path.write_bytes(payload[:-1])
    _pin_fixture(monkeypatch, path, payload)

    with pytest.raises(AmsError, match="regular-file size") as caught:
        verifier._open_authenticated_layer_shard(path, buffer_bytes=7)

    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE
