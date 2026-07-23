from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ams.errors import AmsError
from ci import verify_glm4_official_model_native as verifier


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


class _ByteView:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def numpy(self):
        return self

    def tobytes(self, *, order: str) -> bytes:
        assert order == "C"
        return self.payload


class _Finite:
    def all(self) -> bool:
        return True


class _FakeTorch:
    bfloat16 = object()
    uint8 = object()

    @staticmethod
    def isfinite(_hidden) -> _Finite:
        return _Finite()


class _FakeTensor:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.shape = (1, 1, 4)
        self.dtype = _FakeTorch.bfloat16

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def view(self, dtype):
        assert dtype is _FakeTorch.uint8
        return _ByteView(self.payload)


class _SafeHandle:
    def __init__(self, metadata: dict[str, str]) -> None:
        self._metadata = metadata

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def metadata(self) -> dict[str, str]:
        return self._metadata

    def keys(self) -> list[str]:
        return ["hidden"]


def _architecture():
    return SimpleNamespace(
        content_hash=_digest("architecture"),
        hidden_size=4,
        num_hidden_layers=2,
    )


def _runtime_identity(**overrides: str) -> dict[str, str]:
    identity = {
        "schema_id": "ams.glm47-streaming-reference-runtime.v1",
        "runtime_id": "transformers.glm4_moe_lite.complete_streaming_bf16",
        "modeling_source_hash": _digest("modeling"),
        "layer_verifier_source_hash": _digest("layer-verifier"),
        "model_verifier_source_hash": _digest("model-verifier"),
        "runtime_code_hash": _digest("runtime"),
        "torch_version": "2.13.0+cpu",
        "transformers_version": "5.12.0",
        "safetensors_version": "0.8.0",
    }
    identity.update(overrides)
    return identity


def _install_safetensors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hidden: _FakeTensor,
    save_file=None,
) -> None:
    parent = ModuleType("safetensors")
    parent.__path__ = []
    child = ModuleType("safetensors.torch")
    child.load_file = lambda _path, device: {"hidden": hidden}
    if save_file is not None:
        child.save_file = save_file
    parent.torch = child
    monkeypatch.setitem(sys.modules, "safetensors", parent)
    monkeypatch.setitem(sys.modules, "safetensors.torch", child)


def test_reference_checkpoint_valid_resume_round_trips_atomic_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture = _architecture()
    source_root = _digest("source")
    input_hash = _digest("input")
    runtime_identity = _runtime_identity()
    hidden = _FakeTensor(b"\x01\x00\x02\x00\x03\x00\x04\x00")
    captured: dict[str, dict[str, str]] = {}

    def save_file(tensors, path, *, metadata) -> None:
        assert tensors == {"hidden": hidden}
        captured["metadata"] = metadata
        Path(path).write_bytes(b"bounded checkpoint fixture")

    _install_safetensors(monkeypatch, hidden=hidden, save_file=save_file)

    def safe_open(_path, *, framework: str, device: str) -> _SafeHandle:
        assert (framework, device) == ("pt", "cpu")
        return _SafeHandle(captured["metadata"])

    verifier._save_reference_checkpoint(
        tmp_path,
        architecture,
        source_root,
        input_hash,
        runtime_identity,
        1,
        hidden,
        _FakeTorch,
        safe_open,
    )
    resumed, next_layer = verifier._load_reference_checkpoint(
        tmp_path,
        architecture,
        source_root,
        input_hash,
        runtime_identity,
        1,
        _FakeTorch,
        safe_open,
    )

    assert resumed is hidden
    assert next_layer == 2
    assert (tmp_path / "layer-01.safetensors").is_file()
    assert not list(tmp_path.glob("*.tmp"))
    assert captured["metadata"]["schema_id"] == ("ams.glm47-streaming-reference-checkpoint.v2")
    assert captured["metadata"]["source_root"] == source_root
    assert captured["metadata"]["hidden_payload_hash"] == verifier._hidden_payload_hash(
        hidden, _FakeTorch
    )


def test_reference_checkpoint_rejects_stale_source_or_runtime_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture = _architecture()
    source_root = _digest("source")
    input_hash = _digest("input")
    runtime_identity = _runtime_identity()
    hidden = _FakeTensor(b"\x01\x00\x02\x00\x03\x00\x04\x00")
    payload_hash = verifier._hidden_payload_hash(hidden, _FakeTorch)
    _install_safetensors(monkeypatch, hidden=hidden)

    cases = (
        (
            "stale-source",
            _digest("stale-source"),
            runtime_identity,
        ),
        (
            "stale-runtime",
            source_root,
            _runtime_identity(runtime_code_hash=_digest("stale-runtime")),
        ),
    )
    for name, checkpoint_source, checkpoint_runtime in cases:
        root = tmp_path / name
        root.mkdir()
        (root / "layer-00.safetensors").write_bytes(b"valid older fixture")
        (root / "layer-01.safetensors").write_bytes(b"stale latest fixture")
        metadata_by_name = {
            "layer-00.safetensors": verifier._checkpoint_metadata(
                architecture,
                source_root,
                input_hash,
                runtime_identity,
                0,
                1,
                payload_hash,
            ),
            "layer-01.safetensors": verifier._checkpoint_metadata(
                architecture,
                checkpoint_source,
                input_hash,
                checkpoint_runtime,
                1,
                1,
                payload_hash,
            ),
        }

        def safe_open(
            path,
            *,
            framework: str,
            device: str,
            current_metadata=metadata_by_name,
        ) -> _SafeHandle:
            assert (framework, device) == ("pt", "cpu")
            return _SafeHandle(current_metadata[Path(path).name])

        with pytest.raises(AmsError, match="checkpoint identity is invalid"):
            verifier._load_reference_checkpoint(
                root,
                architecture,
                source_root,
                input_hash,
                runtime_identity,
                1,
                _FakeTorch,
                safe_open,
            )


def test_reference_checkpoint_rejects_finite_payload_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture = _architecture()
    source_root = _digest("source")
    input_hash = _digest("input")
    runtime_identity = _runtime_identity()
    original = _FakeTensor(b"\x01\x00\x02\x00\x03\x00\x04\x00")
    tampered = _FakeTensor(b"\x01\x00\x02\x00\x03\x00\x05\x00")
    metadata = verifier._checkpoint_metadata(
        architecture,
        source_root,
        input_hash,
        runtime_identity,
        1,
        1,
        verifier._hidden_payload_hash(original, _FakeTorch),
    )
    (tmp_path / "layer-01.safetensors").write_bytes(b"finite tampered fixture")
    _install_safetensors(monkeypatch, hidden=tampered)

    def safe_open(_path, *, framework: str, device: str) -> _SafeHandle:
        assert (framework, device) == ("pt", "cpu")
        return _SafeHandle(metadata)

    with pytest.raises(AmsError, match="payload hash mismatch"):
        verifier._load_reference_checkpoint(
            tmp_path,
            architecture,
            source_root,
            input_hash,
            runtime_identity,
            1,
            _FakeTorch,
            safe_open,
        )
