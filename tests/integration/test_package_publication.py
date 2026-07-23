import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema.validators import Draft202012Validator
from safetensors.numpy import save_file

import ams.package as package_module
from ams.conversion import execute_multi_source_identity_conversion
from ams.descriptors import DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    build_huggingface_catalog,
    build_huggingface_identity_plan,
    parse_huggingface_shard_index,
)
from ams.package import (
    GraphArtifact,
    OperatorRequirement,
    build_huggingface_identity_manifest,
    publish_manifest_last,
    verify_manifest_content_root,
)
from ams.storage import FileRangeStore


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def prepare_conversion(tmp_path: Path, *, include_empty: bool = False):
    source_directory = tmp_path / "source"
    source_directory.mkdir()
    shard_path = source_directory / "model-00001-of-00001.safetensors"
    tensors = {
        "model.embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "model.layers.0.weight": np.arange(15, dtype=np.int16).reshape(3, 5),
    }
    if include_empty:
        tensors["model.empty"] = np.empty((0, 4), dtype=np.float32)
    save_file(tensors, shard_path)
    shard_payload = shard_path.read_bytes()
    shard_hash = digest(shard_payload)
    source = HuggingFaceShardSource(
        shard_name=shard_path.name,
        object_id="source:00001",
        content_hash=shard_hash,
        reader=FileRangeStore(
            shard_path,
            StorageObject(
                "source:00001",
                shard_path.name,
                len(shard_payload),
                1,
                shard_hash,
            ),
        ),
    )
    index_payload = json.dumps(
        {
            "metadata": {"total_size": sum(tensor.nbytes for tensor in tensors.values())},
            "weight_map": {name: shard_path.name for name in tensors},
        },
        separators=(",", ":"),
    ).encode()
    index = parse_huggingface_shard_index(index_payload)
    catalog = build_huggingface_catalog(index, (source,), buffer_bytes=17)
    plan = build_huggingface_identity_plan(catalog, digest(b"identity-config"), buffer_bytes=13)
    package_root = tmp_path / "package"
    journal = execute_multi_source_identity_conversion(
        {source.object_id: source.reader},
        plan.conversion,
        package_root,
        package_root / "conversion.journal.json",
        buffer_bytes=11,
    )
    graph_payload = b'{"entry_points":["causal_lm"],"ir_version":"0.1.0"}'
    graph_path = package_root / "graph" / "ir.json"
    graph_path.parent.mkdir()
    graph_path.write_bytes(graph_payload)
    graph = GraphArtifact(
        uri="graph/ir.json",
        size_bytes=len(graph_payload),
        content_hash=digest(graph_payload),
        ir_version="0.1.0",
        entry_points=("causal_lm",),
        required_operators=(OperatorRequirement("linear", "1.0.0"),),
    )
    return catalog, plan, journal, graph, package_root


def build_manifest(catalog, plan, journal, graph, *, architecture: str = "SyntheticDecoder"):
    return build_huggingface_identity_manifest(
        catalog,
        plan,
        journal,
        graph,
        architecture=architecture,
        model_configuration={"hidden_size": 4, "num_hidden_layers": 1},
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
    )


def test_manifest_is_schema_valid_deterministic_and_published_last(tmp_path: Path) -> None:
    catalog, plan, journal, graph, package_root = prepare_conversion(tmp_path)
    manifest = build_manifest(catalog, plan, journal, graph)
    assert manifest == build_manifest(catalog, plan, journal, graph)
    verify_manifest_content_root(manifest)
    schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "manifest.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(manifest)
    manifest_path = package_root / "manifest.json"
    assert not manifest_path.exists()
    assert publish_manifest_last(package_root, manifest, buffer_bytes=7) == manifest_path
    assert json.loads(manifest_path.read_bytes()) == manifest
    assert publish_manifest_last(package_root, manifest, buffer_bytes=5) == manifest_path


def test_registry_manifest_resolves_shared_cas_without_duplicate_package_chunks(
    tmp_path: Path,
) -> None:
    catalog, plan, journal, graph, package_root = prepare_conversion(tmp_path)
    store_root = tmp_path / "model-store"
    cas_root = store_root / "cas"
    cas_root.mkdir(parents=True)
    (package_root / "chunks").replace(cas_root / "chunks")
    graph_payload = (package_root / graph.uri).read_bytes()
    graph_uri = "manifests/glm47-int4/graph/ir.json"
    graph_path = store_root / graph_uri
    graph_path.parent.mkdir(parents=True)
    graph_path.write_bytes(graph_payload)
    registry_graph = GraphArtifact(
        uri=graph_uri,
        size_bytes=len(graph_payload),
        content_hash=digest(graph_payload),
        ir_version=graph.ir_version,
        entry_points=graph.entry_points,
        required_operators=graph.required_operators,
    )
    manifest = build_huggingface_identity_manifest(
        catalog,
        plan,
        journal,
        registry_graph,
        architecture="SyntheticDecoder",
        model_configuration={"hidden_size": 4, "num_hidden_layers": 1},
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
        storage_uri_prefix="cas/chunks",
    )
    manifest_uri = "manifests/glm47-int4/manifest.json"

    published = publish_manifest_last(
        store_root,
        manifest,
        manifest_uri=manifest_uri,
        buffer_bytes=7,
    )

    assert published == store_root / manifest_uri
    assert all(
        storage["uri"].startswith(("cas/chunks/", "manifests/glm47-int4/graph/"))
        for storage in manifest["storage_objects"]
    )
    assert not (store_root / "chunks").exists()


def test_corrupt_chunk_prevents_manifest_visibility(tmp_path: Path) -> None:
    catalog, plan, journal, graph, package_root = prepare_conversion(tmp_path)
    manifest = build_manifest(catalog, plan, journal, graph)
    chunk = next((package_root / "chunks").glob("*.bin"))
    payload = bytearray(chunk.read_bytes())
    payload[0] ^= 0xFF
    chunk.write_bytes(payload)
    with pytest.raises(AmsError) as caught:
        publish_manifest_last(package_root, manifest, buffer_bytes=3)
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE
    assert not (package_root / "manifest.json").exists()


def test_published_manifest_is_immutable(tmp_path: Path) -> None:
    catalog, plan, journal, graph, package_root = prepare_conversion(tmp_path)
    first = build_manifest(catalog, plan, journal, graph)
    publish_manifest_last(package_root, first)
    different = build_manifest(
        catalog,
        plan,
        journal,
        graph,
        architecture="DifferentSyntheticDecoder",
    )
    with pytest.raises(AmsError) as caught:
        publish_manifest_last(package_root, different)
    assert caught.value.code is ErrorCode.TRANSACTION_FAILURE


def test_zero_sized_tensor_is_rejected_instead_of_silently_omitted(tmp_path: Path) -> None:
    catalog, plan, journal, graph, _ = prepare_conversion(tmp_path, include_empty=True)
    with pytest.raises(AmsError) as caught:
        build_manifest(catalog, plan, journal, graph)
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_atomic_replace_failure_keeps_manifest_invisible_and_is_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog, plan, journal, graph, package_root = prepare_conversion(tmp_path)
    manifest = build_manifest(catalog, plan, journal, graph)
    real_replace = package_module.os.replace

    def fail_replace(*_args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(package_module.os, "replace", fail_replace)
    with pytest.raises(AmsError) as caught:
        publish_manifest_last(package_root, manifest)
    assert caught.value.code is ErrorCode.TRANSACTION_FAILURE
    assert caught.value.retriable
    assert not (package_root / "manifest.json").exists()

    monkeypatch.setattr(package_module.os, "replace", real_replace)
    assert publish_manifest_last(package_root, manifest).is_file()


def test_missing_declared_object_is_typed_and_keeps_manifest_invisible(tmp_path: Path) -> None:
    catalog, plan, journal, graph, package_root = prepare_conversion(tmp_path)
    manifest = build_manifest(catalog, plan, journal, graph)
    (package_root / graph.uri).unlink()
    with pytest.raises(AmsError) as caught:
        publish_manifest_last(package_root, manifest)
    assert caught.value.code is ErrorCode.IO_FAILURE
    assert caught.value.retriable
    assert not (package_root / "manifest.json").exists()
