import hashlib
import json
import struct
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ams.canonical import canonical_json_bytes
from ams.codecs import (
    Int4CodecConfig,
    TernaryCodecConfig,
    decode_int4_reference,
    decode_ternary_reference,
    encode_int4_stream,
    encode_ternary_stream,
)
from ams.descriptors import ByteRange, DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    HuggingFaceTensorAssignment,
    HuggingFaceTensorEncoding,
    build_huggingface_catalog,
    build_huggingface_mixed_plan,
    expected_glm4_moe_lite_tensor_shape,
    expected_glm4_moe_lite_tensor_slots,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
)
from ams.mixed_conversion import execute_huggingface_mixed_conversion
from ams.ops import (
    GlmPackageWeights,
    GlmReferenceTensor,
    GlmReferenceWeights,
    GlmWeightAccess,
    run_glm4_moe_lite_prefill_reference,
)
from ams.package import (
    GraphArtifact,
    OperatorRequirement,
    build_huggingface_mixed_manifest,
    publish_manifest_last,
)
from ams.storage import FileRangeStore


def tiny_glm4_config():
    return {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 4,
        "intermediate_size": 6,
        "kv_lora_rank": 2,
        "max_position_embeddings": 16,
        "model_type": "glm4_moe_lite",
        "moe_intermediate_size": 3,
        "n_group": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 1,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "num_nextn_predict_layers": 1,
        "partial_rotary_factor": 1.0,
        "q_lora_rank": 2,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10_000.0,
        "routed_scaling_factor": 1.5,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "v_head_dim": 2,
        "vocab_size": 8,
    }


def tiny_glm4_architecture():
    return parse_glm4_moe_lite_architecture(canonical_json_bytes(tiny_glm4_config()))


def fixture_values(count: int, salt: int, *, center: float = 0.0, scale: float = 0.08):
    return tuple(
        center + (((index * (salt * 2 + 1) + salt) % 19) - 9) * scale for index in range(count)
    )


def matrix(tensors, name: str, rows: int, columns: int, salt: int) -> None:
    tensors[name] = GlmReferenceTensor((rows, columns), fixture_values(rows * columns, salt))


def vector(tensors, name: str, length: int, salt: int, *, norm: bool = False) -> None:
    values = (
        fixture_values(length, salt, center=1.0, scale=0.01)
        if norm
        else fixture_values(length, salt)
    )
    tensors[name] = GlmReferenceTensor((length,), values)


def add_attention(tensors, architecture, prefix: str, salt: int) -> None:
    matrix(
        tensors,
        f"{prefix}.self_attn.q_a_proj.weight",
        architecture.q_lora_rank,
        architecture.hidden_size,
        salt,
    )
    vector(
        tensors,
        f"{prefix}.self_attn.q_a_layernorm.weight",
        architecture.q_lora_rank,
        salt + 1,
        norm=True,
    )
    matrix(
        tensors,
        f"{prefix}.self_attn.q_b_proj.weight",
        architecture.num_attention_heads * architecture.qk_head_dim,
        architecture.q_lora_rank,
        salt + 2,
    )
    matrix(
        tensors,
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
        architecture.kv_lora_rank + architecture.qk_rope_head_dim,
        architecture.hidden_size,
        salt + 3,
    )
    vector(
        tensors,
        f"{prefix}.self_attn.kv_a_layernorm.weight",
        architecture.kv_lora_rank,
        salt + 4,
        norm=True,
    )
    matrix(
        tensors,
        f"{prefix}.self_attn.kv_b_proj.weight",
        architecture.num_attention_heads
        * (architecture.qk_nope_head_dim + architecture.v_head_dim),
        architecture.kv_lora_rank,
        salt + 5,
    )
    matrix(
        tensors,
        f"{prefix}.self_attn.o_proj.weight",
        architecture.hidden_size,
        architecture.num_attention_heads * architecture.v_head_dim,
        salt + 6,
    )


def add_sparse_mlp(tensors, architecture, prefix: str, salt: int) -> None:
    matrix(
        tensors,
        f"{prefix}.mlp.gate.weight",
        architecture.n_routed_experts,
        architecture.hidden_size,
        salt,
    )
    tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = GlmReferenceTensor(
        (architecture.n_routed_experts,), (10.0, -10.0)
    )
    for expert_index in range(architecture.n_routed_experts):
        expert = f"{prefix}.mlp.experts.{expert_index}"
        matrix(
            tensors,
            f"{expert}.gate_proj.weight",
            architecture.moe_intermediate_size,
            architecture.hidden_size,
            salt + 1 + expert_index,
        )
        matrix(
            tensors,
            f"{expert}.up_proj.weight",
            architecture.moe_intermediate_size,
            architecture.hidden_size,
            salt + 3 + expert_index,
        )
        matrix(
            tensors,
            f"{expert}.down_proj.weight",
            architecture.hidden_size,
            architecture.moe_intermediate_size,
            salt + 5 + expert_index,
        )
    shared = f"{prefix}.mlp.shared_experts"
    shared_size = architecture.moe_intermediate_size * architecture.n_shared_experts
    matrix(tensors, f"{shared}.gate_proj.weight", shared_size, architecture.hidden_size, salt + 7)
    matrix(tensors, f"{shared}.up_proj.weight", shared_size, architecture.hidden_size, salt + 8)
    matrix(tensors, f"{shared}.down_proj.weight", architecture.hidden_size, shared_size, salt + 9)


def glm4_fixture_tensor_map(architecture):
    tensors = {}
    matrix(
        tensors,
        "model.embed_tokens.weight",
        architecture.vocab_size,
        architecture.hidden_size,
        1,
    )
    vector(tensors, "model.norm.weight", architecture.hidden_size, 2, norm=True)
    matrix(tensors, "lm_head.weight", architecture.vocab_size, architecture.hidden_size, 3)
    for layer_index, mlp_type in enumerate(architecture.mlp_layer_types):
        prefix = f"model.layers.{layer_index}"
        vector(
            tensors,
            f"{prefix}.input_layernorm.weight",
            architecture.hidden_size,
            10 + layer_index,
            norm=True,
        )
        vector(
            tensors,
            f"{prefix}.post_attention_layernorm.weight",
            architecture.hidden_size,
            20 + layer_index,
            norm=True,
        )
        add_attention(tensors, architecture, prefix, 30 + layer_index * 10)
        if mlp_type == "dense":
            matrix(
                tensors,
                f"{prefix}.mlp.gate_proj.weight",
                architecture.intermediate_size,
                architecture.hidden_size,
                100,
            )
            matrix(
                tensors,
                f"{prefix}.mlp.up_proj.weight",
                architecture.intermediate_size,
                architecture.hidden_size,
                101,
            )
            matrix(
                tensors,
                f"{prefix}.mlp.down_proj.weight",
                architecture.hidden_size,
                architecture.intermediate_size,
                102,
            )
        else:
            add_sparse_mlp(tensors, architecture, prefix, 110)
    for offset in range(architecture.num_nextn_predict_layers):
        layer_index = architecture.num_hidden_layers + offset
        prefix = f"model.layers.{layer_index}"
        vector(
            tensors,
            f"{prefix}.input_layernorm.weight",
            architecture.hidden_size,
            200,
            norm=True,
        )
        vector(
            tensors,
            f"{prefix}.post_attention_layernorm.weight",
            architecture.hidden_size,
            201,
            norm=True,
        )
        add_attention(tensors, architecture, prefix, 210)
        add_sparse_mlp(tensors, architecture, prefix, 220)
        vector(tensors, f"{prefix}.enorm.weight", architecture.hidden_size, 230, norm=True)
        vector(tensors, f"{prefix}.hnorm.weight", architecture.hidden_size, 231, norm=True)
        matrix(
            tensors,
            f"{prefix}.eh_proj.weight",
            architecture.hidden_size,
            architecture.hidden_size * 2,
            232,
        )
        matrix(
            tensors,
            f"{prefix}.embed_tokens.weight",
            architecture.vocab_size,
            architecture.hidden_size,
            233,
        )
        matrix(
            tensors,
            f"{prefix}.shared_head.head.weight",
            architecture.vocab_size,
            architecture.hidden_size,
            234,
        )
        vector(
            tensors,
            f"{prefix}.shared_head.norm.weight",
            architecture.hidden_size,
            235,
            norm=True,
        )
    return tensors


class RejectingExpertWeights(GlmWeightAccess):
    def __init__(self, inner: GlmWeightAccess, rejected_fragment: str) -> None:
        self.inner = inner
        self.rejected_fragment = rejected_fragment

    def _check(self, tensor_name: str) -> None:
        if self.rejected_fragment in tensor_name:
            raise AssertionError(f"unselected expert was fetched: {tensor_name}")

    def vector(self, tensor_name: str, length: int) -> tuple[float, ...]:
        self._check(tensor_name)
        return self.inner.vector(tensor_name, length)

    def embedding(self, tensor_name: str, index: int, width: int) -> tuple[float, ...]:
        self._check(tensor_name)
        return self.inner.embedding(tensor_name, index, width)

    def linear(self, tensor_name: str, values: Sequence[float], rows: int) -> tuple[float, ...]:
        self._check(tensor_name)
        return self.inner.linear(tensor_name, values, rows)


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class MemoryReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.size_bytes = len(payload)

    def read_into(self, offset: int, destination) -> None:
        view = memoryview(destination).cast("B")
        try:
            view[:] = self.payload[offset : offset + view.nbytes]
        finally:
            view.release()


def stored_f32_tensor(tensor: GlmReferenceTensor) -> GlmReferenceTensor:
    payload = struct.pack(f"<{len(tensor.values)}f", *tensor.values)
    return GlmReferenceTensor(
        tensor.shape,
        struct.unpack(f"<{len(tensor.values)}f", payload),
    )


def encoded_reference_tensor(
    tensor: GlmReferenceTensor,
    config: TernaryCodecConfig | Int4CodecConfig,
) -> GlmReferenceTensor:
    payload = struct.pack(f"<{len(tensor.values)}f", *tensor.values)
    sink = BytesIO()
    byte_range = ByteRange("fixture", 0, len(payload), digest(payload))
    if isinstance(config, TernaryCodecConfig):
        encode_ternary_stream(
            MemoryReader(payload), byte_range, tensor.shape, DType.FLOAT32, sink, config
        )
        decoded = decode_ternary_reference(sink.getvalue(), len(tensor.values), config)
    else:
        encode_int4_stream(
            MemoryReader(payload), byte_range, tensor.shape, DType.FLOAT32, sink, config
        )
        decoded = decode_int4_reference(sink.getvalue(), len(tensor.values), config)
    return GlmReferenceTensor(tensor.shape, tuple(decoded))


def build_mini_glm4_package(tmp_path: Path):
    architecture = tiny_glm4_architecture()
    tensors = {
        name: stored_f32_tensor(tensor)
        for name, tensor in glm4_fixture_tensor_map(architecture).items()
    }
    assert set(tensors) == {
        slot.tensor_name for slot in expected_glm4_moe_lite_tensor_slots(architecture)
    }
    arrays = {
        name: np.asarray(tensor.values, dtype=np.float32).reshape(tensor.shape)
        for name, tensor in tensors.items()
    }
    shard_path = tmp_path / "model-00001-of-00001.safetensors"
    save_file(arrays, shard_path)
    shard_payload = shard_path.read_bytes()
    shard_hash = digest(shard_payload)
    source = HuggingFaceShardSource(
        shard_path.name,
        "source:mini-glm4",
        shard_hash,
        FileRangeStore(
            shard_path,
            StorageObject(
                "source:mini-glm4",
                shard_path.name,
                len(shard_payload),
                1,
                shard_hash,
            ),
        ),
    )
    index = parse_huggingface_shard_index(
        json.dumps(
            {
                "metadata": {"total_size": sum(array.nbytes for array in arrays.values())},
                "weight_map": {name: shard_path.name for name in arrays},
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    catalog = build_huggingface_catalog(index, (source,), buffer_bytes=97)
    ternary_config = TernaryCodecConfig(group_size=5)
    int4_config = Int4CodecConfig(group_size=5)
    ternary_names = {
        f"model.layers.1.mlp.experts.0.{projection}_proj.weight"
        for projection in ("gate", "up", "down")
    }
    int4_names = {
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    }
    assignments = tuple(
        HuggingFaceTensorAssignment(
            name,
            HuggingFaceTensorEncoding.TERNARY_TRIT5
            if name in ternary_names
            else HuggingFaceTensorEncoding.INT4_SYMMETRIC
            if name in int4_names
            else HuggingFaceTensorEncoding.IDENTITY,
            ternary_config=ternary_config if name in ternary_names else None,
            int4_config=int4_config if name in int4_names else None,
        )
        for name in sorted(tensors)
    )
    plan = build_huggingface_mixed_plan(catalog, assignments, buffer_bytes=89)
    package_root = tmp_path / "package"
    journal = execute_huggingface_mixed_conversion(
        catalog,
        plan,
        package_root,
        package_root / "conversion.journal.json",
        verification_buffer_bytes=83,
    )
    graph_payload = b'{"entry_points":["causal_lm"],"operators":["glm4_moe_lite"]}'
    graph_path = package_root / "graph" / "ir.json"
    graph_path.parent.mkdir()
    graph_path.write_bytes(graph_payload)
    graph = GraphArtifact(
        "graph/ir.json",
        len(graph_payload),
        digest(graph_payload),
        "0.1.0",
        ("causal_lm",),
        (OperatorRequirement("glm4_moe_lite", "1.0.0"),),
    )
    manifest = build_huggingface_mixed_manifest(
        catalog,
        plan,
        journal,
        graph,
        architecture="Glm4MoeLiteForCausalLM",
        model_configuration=tiny_glm4_config(),
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
    )
    publish_manifest_last(package_root, manifest, buffer_bytes=79)
    expected_tensors = dict(tensors)
    for name in ternary_names:
        expected_tensors[name] = encoded_reference_tensor(tensors[name], ternary_config)
    for name in int4_names:
        expected_tensors[name] = encoded_reference_tensor(tensors[name], int4_config)
    return package_root, architecture, GlmReferenceWeights(expected_tensors)


def test_mini_glm4_prefill_executes_full_attention_dense_and_sparse_layers() -> None:
    architecture = tiny_glm4_architecture()
    weights = RejectingExpertWeights(
        GlmReferenceWeights(glm4_fixture_tensor_map(architecture)),
        ".mlp.experts.1.",
    )
    first = run_glm4_moe_lite_prefill_reference(architecture, weights, (1, 3, 5))
    second = run_glm4_moe_lite_prefill_reference(architecture, weights, (1, 3, 5))
    assert first == second
    causal = ((0,), (0, 1), (0, 1, 2))
    assert all(layer.causal_key_indices == causal for layer in first.layers)
    assert first.layers[0].expert_routing == ()
    assert all(route.expert_indices == (0,) for route in first.layers[1].expert_routing)
    assert len(first.logits) == 3
    assert all(len(logits) == architecture.vocab_size for logits in first.logits)
    assert all(all(value == value for value in logits) for logits in first.logits)
    assert [max(range(len(logits)), key=logits.__getitem__) for logits in first.logits] == [7, 7, 7]
    assert tuple(logits[0] for logits in first.logits) == pytest.approx(
        (-1.143865917592693, -1.1638312723196396, -1.4368080033224389),
        rel=0,
        abs=1e-14,
    )


def test_mini_glm4_fixture_matches_every_reviewed_tensor_shape() -> None:
    architecture = tiny_glm4_architecture()
    tensors = glm4_fixture_tensor_map(architecture)
    for slot in expected_glm4_moe_lite_tensor_slots(architecture):
        assert tensors[slot.tensor_name].shape == expected_glm4_moe_lite_tensor_shape(
            architecture,
            slot,
        )


def test_mini_glm4_mtp_is_explicitly_unsupported() -> None:
    architecture = tiny_glm4_architecture()
    with pytest.raises(AmsError) as caught:
        run_glm4_moe_lite_prefill_reference(
            architecture,
            GlmReferenceWeights(glm4_fixture_tensor_map(architecture)),
            (1,),
            enable_mtp=True,
        )
    assert caught.value.code is ErrorCode.UNSUPPORTED_OP


def test_mini_glm4_mixed_package_matches_full_decode_and_skips_unselected_expert(
    tmp_path: Path,
) -> None:
    package_root, architecture, expected_weights = build_mini_glm4_package(tmp_path)
    package_weights = GlmPackageWeights.open(
        package_root,
        linear_arena_bytes=64,
        verification_buffer_bytes=31,
    )
    assert package_weights.architecture == architecture
    actual = run_glm4_moe_lite_prefill_reference(
        architecture,
        RejectingExpertWeights(package_weights, ".mlp.experts.1."),
        (1, 3, 5),
    )
    expected = run_glm4_moe_lite_prefill_reference(
        architecture,
        expected_weights,
        (1, 3, 5),
    )
    assert actual == expected
    assert package_weights.read_evidence.verified_objects > 0
    assert package_weights.read_evidence.maximum_read_bytes <= 64


def test_mini_glm4_package_architecture_field_is_authoritative(tmp_path: Path) -> None:
    package_root, _, _ = build_mini_glm4_package(tmp_path)
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["model"]["architecture"] = "GlmMoeDsaForCausalLM"
    del manifest["content_root"]
    manifest["content_root"] = digest(canonical_json_bytes(manifest))
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(AmsError) as caught:
        GlmPackageWeights.open(package_root, linear_arena_bytes=64)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


def test_mini_glm4_package_rejects_a_transposed_tensor_shape(tmp_path: Path) -> None:
    package_root, _, _ = build_mini_glm4_package(tmp_path)
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    target_name = "model.layers.0.self_attn.q_a_proj.weight"
    target = next(
        tensor
        for tensor in manifest["tensors"]
        if tensor["extensions"]["hf.source-name"] == target_name
    )
    transposed = list(reversed(target["shape"]))
    assert transposed != target["shape"]
    target["shape"] = transposed
    target["layouts"][0]["tile_shape"] = transposed
    target["layouts"][0]["chunks"][0]["logical_extent"] = transposed
    del manifest["content_root"]
    manifest["content_root"] = digest(canonical_json_bytes(manifest))
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(AmsError) as caught:
        GlmPackageWeights.open(package_root, linear_arena_bytes=64)
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
