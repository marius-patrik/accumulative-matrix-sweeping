import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ams.conversion import execute_multi_source_identity_conversion
from ams.descriptors import StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceCatalogPolicy,
    HuggingFaceIndexLimits,
    HuggingFaceShardSource,
    HuggingFaceTotalSizeSemantics,
    audit_huggingface_headers,
    build_huggingface_catalog,
    build_huggingface_identity_plan,
    parse_huggingface_shard_index,
)
from ams.storage import FileRangeStore


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def make_source(path: Path, object_id: str) -> HuggingFaceShardSource:
    payload = path.read_bytes()
    content_hash = digest(payload)
    descriptor = StorageObject(
        object_id=object_id,
        uri=path.name,
        size_bytes=len(payload),
        alignment_bytes=1,
        content_hash=content_hash,
    )
    return HuggingFaceShardSource(
        shard_name=path.name,
        object_id=object_id,
        content_hash=content_hash,
        reader=FileRangeStore(path, descriptor),
    )


def create_sharded_fixture(tmp_path: Path):
    first_path = tmp_path / "model-00001-of-00002.safetensors"
    second_path = tmp_path / "model-00002-of-00002.safetensors"
    first = {
        "model.embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "model.empty": np.empty((0, 4), dtype=np.float32),
    }
    second = {
        "model.layers.0.weight": np.arange(15, dtype=np.int16).reshape(3, 5),
    }
    save_file(first, first_path)
    save_file(second, second_path)
    total_size = sum(tensor.nbytes for tensor in first.values()) + sum(
        tensor.nbytes for tensor in second.values()
    )
    index_payload = json.dumps(
        {
            "metadata": {"total_size": total_size, "producer": "test"},
            "weight_map": {
                "model.layers.0.weight": second_path.name,
                "model.empty": first_path.name,
                "model.embed.weight": first_path.name,
            },
        },
        separators=(",", ":"),
    ).encode()
    sources = (
        make_source(first_path, "source:00001"),
        make_source(second_path, "source:00002"),
    )
    return index_payload, sources, total_size


def test_sharded_index_catalog_plan_and_multi_source_conversion(tmp_path: Path) -> None:
    index_payload, sources, total_size = create_sharded_fixture(tmp_path)
    index = parse_huggingface_shard_index(index_payload)
    assert index.total_size == total_size
    assert index.shard_names == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )
    catalog = build_huggingface_catalog(index, sources, buffer_bytes=13)
    assert catalog.total_size == total_size
    assert [tensor.tensor_name for tensor in catalog.tensors] == [
        "model.embed.weight",
        "model.empty",
        "model.layers.0.weight",
    ]
    plan = build_huggingface_identity_plan(
        catalog,
        digest(b"identity-config"),
        buffer_bytes=11,
    )
    empty = next(item for item in plan.tensors if item.tensor.tensor_name == "model.empty")
    assert empty.target_chunk_id is None
    assert empty.source_checksum == digest(b"")
    output = tmp_path / "ams-package"
    journal = execute_multi_source_identity_conversion(
        {source.object_id: source.reader for source in sources},
        plan.conversion,
        output,
        output / "conversion.journal.json",
        buffer_bytes=7,
    )
    assert len(journal.entries) == 2
    assert all(entry.state.value == "published" for entry in journal.entries)
    for tensor in plan.tensors:
        if tensor.target_chunk_id is None:
            continue
        algorithm, hexdigest = tensor.source_checksum.split(":", 1)
        published = output / "chunks" / f"{algorithm}-{hexdigest}.bin"
        assert published.is_file()
        assert digest(published.read_bytes()) == tensor.source_checksum


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"metadata": {"total_size": 4}},
        {
            "metadata": {"total_size": 4},
            "weight_map": {"weight": "../model.safetensors"},
        },
        {
            "metadata": {"total_size": 4},
            "weight_map": {"weight": None},
        },
        {
            "metadata": {"total_size": 4},
            "weight_map": {"weight": "model.safetensors"},
            "unexpected": True,
        },
    ],
)
def test_sharded_index_rejects_null_missing_unsafe_and_drifted_fields(value) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    with pytest.raises(AmsError) as caught:
        parse_huggingface_shard_index(payload)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


def test_sharded_index_rejects_duplicate_weight_names() -> None:
    payload = (
        b'{"metadata":{"total_size":8},"weight_map":{'
        b'"weight":"model-1.safetensors","weight":"model-2.safetensors"}}'
    )
    with pytest.raises(AmsError, match="JSON"):
        parse_huggingface_shard_index(payload)


def test_sharded_index_limit_is_checked_before_json_allocation() -> None:
    with pytest.raises(AmsError, match="size"):
        parse_huggingface_shard_index(
            b"{}" * 16,
            HuggingFaceIndexLimits(max_index_bytes=8),
        )


def test_catalog_rejects_index_to_header_mapping_drift(tmp_path: Path) -> None:
    index_payload, sources, total_size = create_sharded_fixture(tmp_path)
    raw = json.loads(index_payload)
    raw["metadata"]["total_size"] = total_size
    raw["weight_map"]["model.embed.weight"] = "model-00002-of-00002.safetensors"
    index = parse_huggingface_shard_index(json.dumps(raw).encode())
    with pytest.raises(AmsError, match="disagree"):
        build_huggingface_catalog(index, sources)


def test_catalog_rejects_shard_hash_mismatch_before_header_use(tmp_path: Path) -> None:
    index_payload, sources, _ = create_sharded_fixture(tmp_path)
    index = parse_huggingface_shard_index(index_payload)
    first = sources[0]
    corrupted_identity = HuggingFaceShardSource(
        first.shard_name,
        first.object_id,
        "sha256:" + "0" * 64,
        first.reader,
    )
    with pytest.raises(AmsError) as caught:
        build_huggingface_catalog(index, (corrupted_identity, sources[1]))
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_header_audit_proves_structure_without_claiming_payload_hashes(tmp_path: Path) -> None:
    index_payload, sources, total_size = create_sharded_fixture(tmp_path)
    index = parse_huggingface_shard_index(index_payload)
    first = sources[0]
    unverified_identity = HuggingFaceShardSource(
        first.shard_name,
        first.object_id,
        "sha256:" + "0" * 64,
        first.reader,
    )
    audit = audit_huggingface_headers(index, (unverified_identity, sources[1]))

    assert audit.shard_count == 2
    assert audit.tensor_count == 3
    assert audit.tensor_elements == 27
    assert audit.tensor_bytes == total_size
    assert audit.source_file_bytes == sum(source.reader.size_bytes for source in sources)
    assert audit.prefix_and_header_bytes == audit.source_file_bytes - total_size
    assert audit.dtype_counts == (("F32", 2), ("I16", 1))


def test_nonstandard_element_count_semantics_require_the_exact_index_pin(
    tmp_path: Path,
) -> None:
    index_payload, sources, _ = create_sharded_fixture(tmp_path)
    raw = json.loads(index_payload)
    raw["metadata"]["total_size"] = 27
    index = parse_huggingface_shard_index(json.dumps(raw, separators=(",", ":")).encode())

    with pytest.raises(AmsError, match="total_size"):
        build_huggingface_catalog(index, sources)
    with pytest.raises(AmsError, match="require an exact index hash"):
        HuggingFaceCatalogPolicy(HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS)
    wrong_pin = HuggingFaceCatalogPolicy(
        HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS,
        digest(b"different-index"),
    )
    forbidden_sources = tuple(
        HuggingFaceShardSource(
            source.shard_name,
            source.object_id,
            source.content_hash,
            ForbiddenReader(source.reader.size_bytes),
        )
        for source in sources
    )
    with pytest.raises(AmsError, match="does not match the pinned"):
        build_huggingface_catalog(index, forbidden_sources, policy=wrong_pin)

    pinned = HuggingFaceCatalogPolicy(
        HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS,
        index.content_hash,
    )
    catalog = build_huggingface_catalog(index, sources, policy=pinned)
    assert catalog.total_size == 78


class ForbiddenReader:
    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = size_bytes

    def read_into(self, _offset: int, _destination) -> None:
        raise AssertionError("preflight failure performed source I/O")
