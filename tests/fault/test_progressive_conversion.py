from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import ams.progressive_conversion as progressive_module
from ams.codecs import TernaryCodecConfig
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    HuggingFaceTensorAssignment,
    HuggingFaceTensorEncoding,
    build_huggingface_header_catalog,
    build_huggingface_progressive_mixed_plan,
    parse_huggingface_shard_index,
)
from ams.progressive_conversion import (
    ProgressiveConversionJournalStore,
    execute_progressive_huggingface_mixed_conversion,
)


class MemoryReader:
    def __init__(self, payload: bytes, *, fail_on_read: bool = False) -> None:
        self.payload = payload
        self.size_bytes = len(payload)
        self.fail_on_read = fail_on_read
        self.reads = 0

    def read_into(self, offset: int, destination) -> None:
        if self.fail_on_read:
            raise AssertionError("completed progressive conversion read its remote source")
        self.reads += 1
        view = memoryview(destination).cast("B")
        view[:] = self.payload[offset : offset + view.nbytes]


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def fixture(tmp_path: Path):
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "model.layers.0.mlp.weight": np.arange(20, dtype=np.float32).reshape(4, 5) - 10,
    }
    save_file(tensors, shard_path)
    payload = shard_path.read_bytes()
    reader = MemoryReader(payload)
    source = HuggingFaceShardSource(
        shard_path.name,
        "source:00001",
        digest(payload),
        reader,
    )
    index = parse_huggingface_shard_index(
        json.dumps(
            {
                "metadata": {"total_size": sum(tensor.nbytes for tensor in tensors.values())},
                "weight_map": {name: shard_path.name for name in tensors},
            },
            separators=(",", ":"),
        ).encode()
    )
    catalog = build_huggingface_header_catalog(index, (source,))
    assignments = (
        HuggingFaceTensorAssignment(
            "model.embed.weight",
            HuggingFaceTensorEncoding.IDENTITY,
        ),
        HuggingFaceTensorAssignment(
            "model.layers.0.mlp.weight",
            HuggingFaceTensorEncoding.TERNARY_TRIT5,
            TernaryCodecConfig(group_size=5),
        ),
    )
    plan = build_huggingface_progressive_mixed_plan(catalog, assignments)
    return catalog, plan, payload


def unavailable_catalog(catalog, payload: bytes):
    source = catalog.sources[0]
    unavailable = MemoryReader(payload, fail_on_read=True)
    return replace(
        catalog,
        sources=(
            HuggingFaceShardSource(
                source.shard_name,
                source.object_id,
                source.content_hash,
                unavailable,
            ),
        ),
    ), unavailable


def two_shard_fixture(tmp_path: Path):
    arrays = (
        {"model.embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4)},
        {"model.layers.0.mlp.weight": np.arange(20, dtype=np.float32).reshape(4, 5) - 10},
    )
    sources = []
    weight_map = {}
    for index, tensors in enumerate(arrays, 1):
        path = tmp_path / f"model-{index:05d}-of-00002.safetensors"
        save_file(tensors, path)
        payload = path.read_bytes()
        sources.append(
            HuggingFaceShardSource(
                path.name,
                f"source:{index:05d}",
                digest(payload),
                MemoryReader(payload),
            )
        )
        weight_map.update({name: path.name for name in tensors})
    index = parse_huggingface_shard_index(
        json.dumps(
            {
                "metadata": {
                    "total_size": sum(
                        tensor.nbytes for tensors in arrays for tensor in tensors.values()
                    )
                },
                "weight_map": weight_map,
            },
            separators=(",", ":"),
        ).encode()
    )
    catalog = build_huggingface_header_catalog(index, tuple(sources))
    assignments = (
        HuggingFaceTensorAssignment(
            "model.embed.weight",
            HuggingFaceTensorEncoding.IDENTITY,
        ),
        HuggingFaceTensorAssignment(
            "model.layers.0.mlp.weight",
            HuggingFaceTensorEncoding.TERNARY_TRIT5,
            TernaryCodecConfig(group_size=5),
        ),
    )
    return catalog, build_huggingface_progressive_mixed_plan(catalog, assignments)


def test_progressive_conversion_resumes_mid_shard_without_remote_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, plan, payload = fixture(tmp_path)
    destination = tmp_path / "package"
    journal = tmp_path / "journal"
    cache = tmp_path / "source-cache"
    real_publish = progressive_module._publish_tensor
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AmsError(ErrorCode.IO_FAILURE, "injected tensor failure", retriable=True)
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(progressive_module, "_publish_tensor", interrupt_second)
    with pytest.raises(AmsError, match="injected tensor failure"):
        execute_progressive_huggingface_mixed_conversion(
            catalog,
            plan,
            destination,
            journal,
            cache,
            buffer_bytes=17,
        )
    store = ProgressiveConversionJournalStore(journal)
    assert store.shard_record(plan, plan.shards[0]) is not None
    assert [store.tensor_record(plan, tensor) is not None for tensor in plan.tensors] == [
        True,
        False,
    ]
    assert list((cache / "chunks").glob("*.bin"))

    monkeypatch.setattr(progressive_module, "_publish_tensor", real_publish)
    offline_catalog, offline = unavailable_catalog(catalog, payload)
    snapshot = execute_progressive_huggingface_mixed_conversion(
        offline_catalog,
        plan,
        destination,
        journal,
        cache,
        buffer_bytes=13,
    )
    assert len(snapshot.shards) == 1
    assert len(snapshot.tensors) == 2
    assert offline.reads == 0

    repeated = execute_progressive_huggingface_mixed_conversion(
        offline_catalog,
        plan,
        destination,
        journal,
        cache,
        buffer_bytes=11,
    )
    assert repeated == snapshot
    assert offline.reads == 0


def test_progressive_conversion_releases_each_source_before_staging_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, plan = two_shard_fixture(tmp_path)
    destination = tmp_path / "package"
    journal = tmp_path / "journal"
    cache = tmp_path / "source-cache"
    real_stage = progressive_module.stage_huggingface_shard
    staged_names = []

    def track_stage(source, root, **kwargs):
        assert not list((root / "chunks").glob("*.bin"))
        staged = real_stage(source, root, **kwargs)
        assert len(list((root / "chunks").glob("*.bin"))) == 1
        staged_names.append(source.shard_name)
        return staged

    monkeypatch.setattr(progressive_module, "stage_huggingface_shard", track_stage)
    snapshot = execute_progressive_huggingface_mixed_conversion(
        catalog,
        plan,
        destination,
        journal,
        cache,
        buffer_bytes=13,
    )
    assert staged_names == [shard.shard_name for shard in plan.shards]
    assert len(snapshot.shards) == 2
    assert len(snapshot.tensors) == 2
    assert not list((cache / "chunks").glob("*.bin"))


def test_complete_journal_recovers_release_without_restaging_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, plan, payload = fixture(tmp_path)
    destination = tmp_path / "package"
    journal = tmp_path / "journal"
    cache = tmp_path / "source-cache"
    real_release = progressive_module.release_huggingface_shard_source

    def fail_release(*_args, **_kwargs):
        raise AmsError(ErrorCode.IO_FAILURE, "injected release failure", retriable=True)

    monkeypatch.setattr(progressive_module, "release_huggingface_shard_source", fail_release)
    with pytest.raises(AmsError, match="injected release failure"):
        execute_progressive_huggingface_mixed_conversion(
            catalog,
            plan,
            destination,
            journal,
            cache,
            buffer_bytes=19,
        )
    completed = ProgressiveConversionJournalStore(journal).completed_snapshot(plan)
    assert len(completed.tensors) == 2
    assert list((cache / "chunks").glob("*.bin"))

    monkeypatch.setattr(progressive_module, "release_huggingface_shard_source", real_release)
    offline_catalog, offline = unavailable_catalog(catalog, payload)
    recovered = execute_progressive_huggingface_mixed_conversion(
        offline_catalog,
        plan,
        destination,
        journal,
        cache,
        buffer_bytes=7,
    )
    assert recovered == completed
    assert offline.reads == 0
    assert not list((cache / "chunks").glob("*.bin"))


def test_progressive_journal_rejects_plan_and_published_record_disagreement(
    tmp_path: Path,
) -> None:
    catalog, plan, payload = fixture(tmp_path)
    destination = tmp_path / "package"
    journal = tmp_path / "journal"
    cache = tmp_path / "source-cache"
    execute_progressive_huggingface_mixed_conversion(
        catalog,
        plan,
        destination,
        journal,
        cache,
        buffer_bytes=23,
    )

    changed_assignments = (
        HuggingFaceTensorAssignment(
            "model.embed.weight",
            HuggingFaceTensorEncoding.IDENTITY,
        ),
        HuggingFaceTensorAssignment(
            "model.layers.0.mlp.weight",
            HuggingFaceTensorEncoding.TERNARY_TRIT5,
            TernaryCodecConfig(group_size=10),
        ),
    )
    changed_plan = build_huggingface_progressive_mixed_plan(catalog, changed_assignments)
    offline_catalog, offline = unavailable_catalog(catalog, payload)
    with pytest.raises(AmsError, match="durable state"):
        execute_progressive_huggingface_mixed_conversion(
            offline_catalog,
            changed_plan,
            destination,
            journal,
            cache,
            buffer_bytes=5,
        )
    assert offline.reads == 0

    tensor_record = next((journal / "tensors").glob("*.json"))
    value = json.loads(tensor_record.read_text(encoding="utf-8"))
    value["target_hash"] = "sha256:" + "0" * 64
    tensor_record.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AmsError):
        ProgressiveConversionJournalStore(journal).completed_snapshot(plan)
