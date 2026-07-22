import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from jsonschema.validators import Draft202012Validator
from safetensors.numpy import save_file

from ams.codecs import (
    Int4CodecConfig,
    TernaryCodecConfig,
    decode_int4_reference,
    decode_ternary_reference,
)
from ams.descriptors import DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    HuggingFaceTensorAssignment,
    HuggingFaceTensorEncoding,
    build_huggingface_catalog,
    build_huggingface_header_catalog,
    build_huggingface_mixed_plan,
    build_huggingface_progressive_mixed_plan,
    parse_huggingface_shard_index,
)
from ams.mixed_conversion import execute_huggingface_mixed_conversion
from ams.package import (
    GraphArtifact,
    OperatorRequirement,
    build_huggingface_mixed_manifest,
    publish_manifest_last,
)
from ams.progressive_conversion import (
    execute_progressive_huggingface_mixed_conversion,
    finalize_progressive_huggingface_mixed_conversion,
)
from ams.storage import FileRangeStore


class FailOnRead:
    def __init__(self, size_bytes: int):
        self.size_bytes = size_bytes
        self.reads = 0

    def read_into(self, _offset: int, _destination) -> None:
        self.reads += 1
        raise AmsError(ErrorCode.IO_FAILURE, "source must not be read on restart")


class HeaderOnlyReader:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.size_bytes = inner.size_bytes
        self.reads: list[tuple[int, int]] = []

    def read_into(self, offset: int, destination) -> None:
        length = memoryview(destination).nbytes
        if (offset, length) == (0, 8) or offset == 8:
            self.reads.append((offset, length))
            self.inner.read_into(offset, destination)
            return
        raise AssertionError("structural planning attempted to read tensor payload")


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def prepare_source(tmp_path: Path):
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    values = {
        "model.embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "model.layers.0.mlp.experts.0.weight": np.arange(20, dtype=np.float32).reshape(4, 5) - 10,
        "model.layers.0.self_attn.q_proj.weight": np.arange(15, dtype=np.float32).reshape(3, 5) - 7,
    }
    save_file(values, shard_path)
    payload = shard_path.read_bytes()
    content_hash = digest(payload)
    source = HuggingFaceShardSource(
        shard_name=shard_path.name,
        object_id="source:00001",
        content_hash=content_hash,
        reader=FileRangeStore(
            shard_path,
            StorageObject(
                "source:00001",
                shard_path.name,
                len(payload),
                1,
                content_hash,
            ),
        ),
    )
    index = parse_huggingface_shard_index(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.nbytes for value in values.values())},
                "weight_map": {name: shard_path.name for name in values},
            },
            separators=(",", ":"),
        ).encode()
    )
    return index, source


def prepare_catalog(tmp_path: Path):
    index, source = prepare_source(tmp_path)
    return build_huggingface_catalog(index, (source,), buffer_bytes=19), source


def assignments(
    config: TernaryCodecConfig,
    int4_config: Int4CodecConfig | None = None,
):
    int4_config = int4_config or Int4CodecConfig(group_size=4)
    return (
        HuggingFaceTensorAssignment(
            "model.embed.weight",
            HuggingFaceTensorEncoding.IDENTITY,
        ),
        HuggingFaceTensorAssignment(
            "model.layers.0.mlp.experts.0.weight",
            HuggingFaceTensorEncoding.TERNARY_TRIT5,
            config,
        ),
        HuggingFaceTensorAssignment(
            "model.layers.0.self_attn.q_proj.weight",
            HuggingFaceTensorEncoding.INT4_SYMMETRIC,
            int4_config=int4_config,
        ),
    )


def graph_for(package_root: Path) -> GraphArtifact:
    payload = b'{"entry_points":["causal_lm"],"operators":["linear"]}'
    path = package_root / "graph" / "ir.json"
    path.parent.mkdir()
    path.write_bytes(payload)
    return GraphArtifact(
        "graph/ir.json",
        len(payload),
        digest(payload),
        "0.1.0",
        ("causal_lm",),
        (OperatorRequirement("linear", "1.0.0"),),
    )


def test_progressive_plan_uses_headers_only_and_matches_eager_policy_identity(
    tmp_path: Path,
) -> None:
    index, source = prepare_source(tmp_path)
    header_reader = HeaderOnlyReader(source.reader)
    structural_source = HuggingFaceShardSource(
        source.shard_name,
        source.object_id,
        source.content_hash,
        header_reader,
    )
    header_catalog = build_huggingface_header_catalog(index, (structural_source,))
    config = TernaryCodecConfig(group_size=5)
    progressive = build_huggingface_progressive_mixed_plan(
        header_catalog,
        assignments(config),
    )
    repeated = build_huggingface_progressive_mixed_plan(
        header_catalog,
        assignments(config),
    )
    assert header_reader.reads == [(0, 8), (8, header_catalog.audit.prefix_and_header_bytes - 8)]
    assert repeated == progressive

    verified_catalog = build_huggingface_catalog(index, (source,), buffer_bytes=19)
    eager = build_huggingface_mixed_plan(
        verified_catalog,
        assignments(config),
        buffer_bytes=17,
    )
    assert progressive.policy_hash == eager.policy_hash
    assert [tensor.target_chunk_id for tensor in progressive.tensors] == [
        tensor.target_chunk_id for tensor in eager.tensors
    ]


def test_progressive_finalization_matches_the_eager_manifest_contract(tmp_path: Path) -> None:
    index, source = prepare_source(tmp_path)
    config = TernaryCodecConfig(group_size=5)
    policy = assignments(config)

    header_catalog = build_huggingface_header_catalog(index, (source,))
    progressive_plan = build_huggingface_progressive_mixed_plan(header_catalog, policy)
    progressive_root = tmp_path / "progressive-package"
    journal_root = tmp_path / "progressive-journal"
    execute_progressive_huggingface_mixed_conversion(
        header_catalog,
        progressive_plan,
        progressive_root,
        journal_root,
        tmp_path / "progressive-source-cache",
        buffer_bytes=13,
    )
    promoted_catalog, promoted_plan, promoted_journal = (
        finalize_progressive_huggingface_mixed_conversion(
            header_catalog,
            progressive_plan,
            journal_root,
        )
    )

    eager_catalog = build_huggingface_catalog(index, (source,), buffer_bytes=19)
    eager_plan = build_huggingface_mixed_plan(eager_catalog, policy, buffer_bytes=17)
    eager_root = tmp_path / "eager-package"
    eager_journal = execute_huggingface_mixed_conversion(
        eager_catalog,
        eager_plan,
        eager_root,
        eager_root / "conversion.journal.json",
        verification_buffer_bytes=11,
    )
    assert promoted_catalog == eager_catalog
    assert promoted_plan == eager_plan
    assert promoted_journal == eager_journal

    progressive_graph = graph_for(progressive_root)
    eager_graph = graph_for(eager_root)
    progressive_manifest = build_huggingface_mixed_manifest(
        promoted_catalog,
        promoted_plan,
        promoted_journal,
        progressive_graph,
        architecture="SyntheticSparseDecoder",
        model_configuration={"hidden_size": 5, "num_experts": 1},
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
    )
    eager_manifest = build_huggingface_mixed_manifest(
        eager_catalog,
        eager_plan,
        eager_journal,
        eager_graph,
        architecture="SyntheticSparseDecoder",
        model_configuration={"hidden_size": 5, "num_experts": 1},
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
    )
    assert progressive_manifest == eager_manifest
    assert publish_manifest_last(progressive_root, progressive_manifest, buffer_bytes=7).is_file()
    assert publish_manifest_last(eager_root, eager_manifest, buffer_bytes=7).is_file()


def test_explicit_mixed_policy_converts_publishes_and_restarts_without_source(
    tmp_path: Path,
) -> None:
    catalog, source = prepare_catalog(tmp_path)
    config = TernaryCodecConfig(group_size=5)
    plan = build_huggingface_mixed_plan(catalog, assignments(config), buffer_bytes=17)
    package_root = tmp_path / "package"
    journal_path = package_root / "conversion.journal.json"
    journal = execute_huggingface_mixed_conversion(
        catalog,
        plan,
        package_root,
        journal_path,
        verification_buffer_bytes=7,
    )
    by_id = {entry.target_chunk_id: entry for entry in journal.entries}
    identity = next(
        tensor for tensor in plan.tensors if tensor.encoding is HuggingFaceTensorEncoding.IDENTITY
    )
    ternary = next(
        tensor
        for tensor in plan.tensors
        if tensor.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5
    )
    int4 = next(
        tensor
        for tensor in plan.tensors
        if tensor.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC
    )
    assert by_id[identity.target_chunk_id].target_hash == identity.source_checksum
    assert by_id[ternary.target_chunk_id].target_hash != ternary.source_checksum
    assert by_id[int4.target_chunk_id].target_hash != int4.source_checksum
    ternary_hash = by_id[ternary.target_chunk_id].target_hash
    algorithm, hexdigest = ternary_hash.split(":", 1)
    ternary_payload = (package_root / "chunks" / f"{algorithm}-{hexdigest}.bin").read_bytes()
    decoded = decode_ternary_reference(ternary_payload, 20, config)
    assert len(decoded) == 20
    int4_hash = by_id[int4.target_chunk_id].target_hash
    algorithm, hexdigest = int4_hash.split(":", 1)
    int4_payload = (package_root / "chunks" / f"{algorithm}-{hexdigest}.bin").read_bytes()
    int4_decoded = decode_int4_reference(int4_payload, 15, int4.int4_config)
    assert len(int4_decoded) == 15

    graph = graph_for(package_root)
    manifest = build_huggingface_mixed_manifest(
        catalog,
        plan,
        journal,
        graph,
        architecture="SyntheticSparseDecoder",
        model_configuration={"hidden_size": 5, "num_experts": 1},
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
    )
    schema = json.loads(
        (Path(__file__).parents[2] / "schemas" / "manifest.schema.json").read_text()
    )
    Draft202012Validator(schema).validate(manifest)
    assert "ams.identity-layout.v1" in manifest["required_features"]
    assert "ams.codec.ternary.trit5.v1" in manifest["required_features"]
    assert "ams.codec.int4.symmetric.v1" in manifest["required_features"]
    tensors_by_name = {
        tensor["extensions"]["hf.source-name"]: tensor for tensor in manifest["tensors"]
    }
    ternary_layout = tensors_by_name["model.layers.0.mlp.experts.0.weight"]["layouts"][0]
    assert ternary_layout["storage_dtype"] == "custom"
    assert ternary_layout["codec"]["name"] == "ams.ternary.trit5"
    assert ternary_layout["codec"]["parameters"]["ams.config-hash"] == config.config_hash
    int4_layout = tensors_by_name["model.layers.0.self_attn.q_proj.weight"]["layouts"][0]
    assert int4_layout["storage_dtype"] == "custom"
    assert int4_layout["codec"]["name"] == "ams.int4.symmetric"
    assert int4_layout["codec"]["parameters"]["ams.config-hash"] == int4.int4_config.config_hash
    assert publish_manifest_last(package_root, manifest, buffer_bytes=5).is_file()

    fail_reader = FailOnRead(source.reader.size_bytes)
    restart_source = HuggingFaceShardSource(
        source.shard_name,
        source.object_id,
        source.content_hash,
        fail_reader,
    )
    restart_catalog = replace(catalog, sources=(restart_source,))
    restarted = execute_huggingface_mixed_conversion(
        restart_catalog,
        plan,
        package_root,
        journal_path,
        verification_buffer_bytes=3,
    )
    assert restarted == journal
    assert fail_reader.reads == 0


def test_mixed_policy_rejects_missing_assignment(tmp_path: Path) -> None:
    catalog, _ = prepare_catalog(tmp_path)
    incomplete = assignments(TernaryCodecConfig(group_size=5))[:1]
    with pytest.raises(AmsError) as caught:
        build_huggingface_mixed_plan(catalog, incomplete)
    assert caught.value.code is ErrorCode.PLAN_INVALID


def test_assignments_require_exactly_the_selected_codec_config() -> None:
    ternary = TernaryCodecConfig(group_size=5)
    int4 = Int4CodecConfig(group_size=4)
    with pytest.raises(AmsError) as caught:
        HuggingFaceTensorAssignment(
            "tensor.weight",
            HuggingFaceTensorEncoding.IDENTITY,
            int4_config=int4,
        )
    assert caught.value.code is ErrorCode.PLAN_INVALID

    with pytest.raises(AmsError) as caught:
        HuggingFaceTensorAssignment(
            "tensor.weight",
            HuggingFaceTensorEncoding.INT4_SYMMETRIC,
        )
    assert caught.value.code is ErrorCode.PLAN_INVALID

    with pytest.raises(AmsError) as caught:
        HuggingFaceTensorAssignment(
            "tensor.weight",
            HuggingFaceTensorEncoding.INT4_SYMMETRIC,
            ternary_config=ternary,
            int4_config=int4,
        )
    assert caught.value.code is ErrorCode.PLAN_INVALID


def test_mixed_policy_rejects_ternary_for_non_float_source(tmp_path: Path) -> None:
    catalog, _ = prepare_catalog(tmp_path)
    changed = replace(
        catalog,
        tensors=tuple(
            replace(tensor, dtype=DType.INT16)
            if tensor.tensor_name == "model.layers.0.mlp.experts.0.weight"
            else tensor
            for tensor in catalog.tensors
        ),
    )
    with pytest.raises(AmsError) as caught:
        build_huggingface_mixed_plan(
            changed,
            assignments(TernaryCodecConfig(group_size=5)),
        )
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_policy_hash_and_chunk_ids_change_with_ternary_configuration(tmp_path: Path) -> None:
    catalog, _ = prepare_catalog(tmp_path)
    first = build_huggingface_mixed_plan(
        catalog,
        assignments(TernaryCodecConfig(group_size=5)),
    )
    second = build_huggingface_mixed_plan(
        catalog,
        assignments(TernaryCodecConfig(group_size=10)),
    )
    assert first.policy_hash != second.policy_hash
    assert [tensor.target_chunk_id for tensor in first.tensors] != [
        tensor.target_chunk_id for tensor in second.tensors
    ]


def test_policy_hash_and_chunk_ids_change_with_int4_configuration(tmp_path: Path) -> None:
    catalog, _ = prepare_catalog(tmp_path)
    ternary_config = TernaryCodecConfig(group_size=5)
    first = build_huggingface_mixed_plan(
        catalog,
        assignments(ternary_config, Int4CodecConfig(group_size=4)),
    )
    second = build_huggingface_mixed_plan(
        catalog,
        assignments(ternary_config, Int4CodecConfig(group_size=8)),
    )
    assert first.policy_hash != second.policy_hash
    assert [tensor.target_chunk_id for tensor in first.tensors] != [
        tensor.target_chunk_id for tensor in second.tensors
    ]
