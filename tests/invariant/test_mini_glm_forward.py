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
from ams.codecs import TernaryCodecConfig, decode_ternary_reference, encode_ternary_stream
from ams.descriptors import ByteRange, DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    HuggingFaceTensorAssignment,
    HuggingFaceTensorEncoding,
    build_huggingface_catalog,
    build_huggingface_mixed_plan,
    expected_glm_tensor_slots,
    parse_glm_moe_dsa_architecture,
    parse_huggingface_shard_index,
)
from ams.mixed_conversion import execute_huggingface_mixed_conversion
from ams.ops import (
    GlmPackageWeights,
    GlmReferenceTensor,
    GlmReferenceWeights,
    GlmWeightAccess,
    run_glm_moe_dsa_prefill_reference,
)
from ams.package import (
    GraphArtifact,
    OperatorRequirement,
    build_huggingface_mixed_manifest,
    publish_manifest_last,
)
from ams.storage import FileRangeStore


def tiny_config():
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "attention_bias": False,
        "dtype": "bfloat16",
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 8,
        "index_head_dim": 4,
        "index_n_heads": 2,
        "index_share_for_mtp_iteration": True,
        "index_topk": 2,
        "indexer_types": ["full", "shared"],
        "intermediate_size": 12,
        "kv_lora_rank": 4,
        "max_position_embeddings": 32,
        "mlp_layer_types": ["dense", "sparse"],
        "model_type": "glm_moe_dsa",
        "moe_intermediate_size": 6,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 2,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_nextn_predict_layers": 1,
        "q_lora_rank": 4,
        "qk_head_dim": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "routed_scaling_factor": 1.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "v_head_dim": 3,
        "vocab_size": 16,
    }


def tiny_architecture():
    payload = json.dumps(
        tiny_config(),
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return parse_glm_moe_dsa_architecture(payload)


def fixture_values(count: int, salt: int, *, center: float = 0.0, scale: float = 0.08):
    return tuple(
        center + (((index * (salt * 2 + 1) + salt) % 19) - 9) * scale for index in range(count)
    )


def matrix(tensors, name: str, rows: int, columns: int, salt: int) -> None:
    tensors[name] = GlmReferenceTensor((rows, columns), fixture_values(rows * columns, salt))


def vector(tensors, name: str, length: int, salt: int, *, norm: bool = False) -> None:
    if norm:
        values = fixture_values(length, salt, center=1.0, scale=0.01)
    else:
        values = fixture_values(length, salt)
    tensors[name] = GlmReferenceTensor((length,), values)


def fixture_tensor_map(architecture):
    tensors = {}
    matrix(
        tensors, "model.embed_tokens.weight", architecture.vocab_size, architecture.hidden_size, 1
    )
    vector(tensors, "model.norm.weight", architecture.hidden_size, 2, norm=True)
    matrix(tensors, "lm_head.weight", architecture.vocab_size, architecture.hidden_size, 3)
    for layer_index in range(architecture.num_hidden_layers):
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
        matrix(
            tensors,
            f"{prefix}.self_attn.q_a_proj.weight",
            architecture.q_lora_rank,
            architecture.hidden_size,
            30 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.q_a_layernorm.weight",
            architecture.q_lora_rank,
            40 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.q_b_proj.weight",
            architecture.num_attention_heads * architecture.qk_head_dim,
            architecture.q_lora_rank,
            50 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            architecture.kv_lora_rank + architecture.qk_rope_head_dim,
            architecture.hidden_size,
            60 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.kv_a_layernorm.weight",
            architecture.kv_lora_rank,
            70 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.kv_b_proj.weight",
            architecture.num_attention_heads
            * (architecture.qk_nope_head_dim + architecture.v_head_dim),
            architecture.kv_lora_rank,
            80 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.o_proj.weight",
            architecture.hidden_size,
            architecture.num_attention_heads * architecture.v_head_dim,
            90 + layer_index,
        )
        if architecture.indexer_types[layer_index] == "full":
            matrix(
                tensors,
                f"{prefix}.self_attn.indexer.wq_b.weight",
                architecture.index_n_heads * architecture.index_head_dim,
                architecture.q_lora_rank,
                100 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.self_attn.indexer.wk.weight",
                architecture.index_head_dim,
                architecture.hidden_size,
                110 + layer_index,
            )
            vector(
                tensors,
                f"{prefix}.self_attn.indexer.k_norm.weight",
                architecture.index_head_dim,
                120 + layer_index,
                norm=True,
            )
            vector(
                tensors,
                f"{prefix}.self_attn.indexer.k_norm.bias",
                architecture.index_head_dim,
                130 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.self_attn.indexer.weights_proj.weight",
                architecture.index_n_heads,
                architecture.hidden_size,
                140 + layer_index,
            )
        if architecture.mlp_layer_types[layer_index] == "dense":
            matrix(
                tensors,
                f"{prefix}.mlp.gate_proj.weight",
                architecture.intermediate_size,
                architecture.hidden_size,
                150 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.mlp.up_proj.weight",
                architecture.intermediate_size,
                architecture.hidden_size,
                160 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.mlp.down_proj.weight",
                architecture.hidden_size,
                architecture.intermediate_size,
                170 + layer_index,
            )
        else:
            matrix(
                tensors,
                f"{prefix}.mlp.gate.weight",
                architecture.n_routed_experts,
                architecture.hidden_size,
                180 + layer_index,
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
                    190 + expert_index,
                )
                matrix(
                    tensors,
                    f"{expert}.up_proj.weight",
                    architecture.moe_intermediate_size,
                    architecture.hidden_size,
                    200 + expert_index,
                )
                matrix(
                    tensors,
                    f"{expert}.down_proj.weight",
                    architecture.hidden_size,
                    architecture.moe_intermediate_size,
                    210 + expert_index,
                )
            shared = f"{prefix}.mlp.shared_experts"
            shared_size = architecture.moe_intermediate_size * architecture.n_shared_experts
            matrix(
                tensors, f"{shared}.gate_proj.weight", shared_size, architecture.hidden_size, 220
            )
            matrix(tensors, f"{shared}.up_proj.weight", shared_size, architecture.hidden_size, 221)
            matrix(
                tensors, f"{shared}.down_proj.weight", architecture.hidden_size, shared_size, 222
            )
    for offset in range(architecture.num_nextn_predict_layers):
        layer_index = architecture.num_hidden_layers + offset
        prefix = f"model.layers.{layer_index}"
        vector(
            tensors,
            f"{prefix}.input_layernorm.weight",
            architecture.hidden_size,
            300 + layer_index,
            norm=True,
        )
        vector(
            tensors,
            f"{prefix}.post_attention_layernorm.weight",
            architecture.hidden_size,
            310 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.q_a_proj.weight",
            architecture.q_lora_rank,
            architecture.hidden_size,
            320 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.q_a_layernorm.weight",
            architecture.q_lora_rank,
            330 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.q_b_proj.weight",
            architecture.num_attention_heads * architecture.qk_head_dim,
            architecture.q_lora_rank,
            340 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            architecture.kv_lora_rank + architecture.qk_rope_head_dim,
            architecture.hidden_size,
            350 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.kv_a_layernorm.weight",
            architecture.kv_lora_rank,
            360 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.kv_b_proj.weight",
            architecture.num_attention_heads
            * (architecture.qk_nope_head_dim + architecture.v_head_dim),
            architecture.kv_lora_rank,
            370 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.o_proj.weight",
            architecture.hidden_size,
            architecture.num_attention_heads * architecture.v_head_dim,
            380 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.indexer.wq_b.weight",
            architecture.index_n_heads * architecture.index_head_dim,
            architecture.q_lora_rank,
            390 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.indexer.wk.weight",
            architecture.index_head_dim,
            architecture.hidden_size,
            400 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.indexer.k_norm.weight",
            architecture.index_head_dim,
            410 + layer_index,
            norm=True,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.indexer.k_norm.bias",
            architecture.index_head_dim,
            420 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.indexer.weights_proj.weight",
            architecture.index_n_heads,
            architecture.hidden_size,
            430 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.mlp.gate.weight",
            architecture.n_routed_experts,
            architecture.hidden_size,
            440 + layer_index,
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
                450 + expert_index,
            )
            matrix(
                tensors,
                f"{expert}.up_proj.weight",
                architecture.moe_intermediate_size,
                architecture.hidden_size,
                460 + expert_index,
            )
            matrix(
                tensors,
                f"{expert}.down_proj.weight",
                architecture.hidden_size,
                architecture.moe_intermediate_size,
                470 + expert_index,
            )
        shared = f"{prefix}.mlp.shared_experts"
        shared_size = architecture.moe_intermediate_size * architecture.n_shared_experts
        matrix(tensors, f"{shared}.gate_proj.weight", shared_size, architecture.hidden_size, 480)
        matrix(tensors, f"{shared}.up_proj.weight", shared_size, architecture.hidden_size, 481)
        matrix(tensors, f"{shared}.down_proj.weight", architecture.hidden_size, shared_size, 482)
        vector(tensors, f"{prefix}.enorm.weight", architecture.hidden_size, 490, norm=True)
        vector(tensors, f"{prefix}.hnorm.weight", architecture.hidden_size, 491, norm=True)
        matrix(
            tensors,
            f"{prefix}.eh_proj.weight",
            architecture.hidden_size,
            architecture.hidden_size * 2,
            492,
        )
        vector(
            tensors,
            f"{prefix}.shared_head.norm.weight",
            architecture.hidden_size,
            493,
            norm=True,
        )
    return tensors


def fixture_weights(architecture):
    return GlmReferenceWeights(fixture_tensor_map(architecture))


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


def ternary_reference_tensor(
    tensor: GlmReferenceTensor,
    config: TernaryCodecConfig,
) -> GlmReferenceTensor:
    payload = struct.pack(f"<{len(tensor.values)}f", *tensor.values)
    sink = BytesIO()
    encode_ternary_stream(
        MemoryReader(payload),
        ByteRange("fixture", 0, len(payload), digest(payload)),
        tensor.shape,
        DType.FLOAT32,
        sink,
        config,
    )
    decoded = decode_ternary_reference(sink.getvalue(), len(tensor.values), config)
    return GlmReferenceTensor(tensor.shape, tuple(decoded))


def build_mini_glm_package(tmp_path: Path):
    architecture = tiny_architecture()
    tensors = {
        name: stored_f32_tensor(tensor) for name, tensor in fixture_tensor_map(architecture).items()
    }
    assert set(tensors) == {slot.tensor_name for slot in expected_glm_tensor_slots(architecture)}
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
        "source:mini-glm",
        shard_hash,
        FileRangeStore(
            shard_path,
            StorageObject(
                "source:mini-glm",
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
    ternary_names = {
        f"model.layers.1.mlp.experts.0.{projection}_proj.weight"
        for projection in ("gate", "up", "down")
    }
    assignments = tuple(
        HuggingFaceTensorAssignment(
            name,
            HuggingFaceTensorEncoding.TERNARY_TRIT5
            if name in ternary_names
            else HuggingFaceTensorEncoding.IDENTITY,
            ternary_config if name in ternary_names else None,
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
    graph_payload = b'{"entry_points":["causal_lm"],"operators":["glm_moe_dsa"]}'
    graph_path = package_root / "graph" / "ir.json"
    graph_path.parent.mkdir()
    graph_path.write_bytes(graph_payload)
    graph = GraphArtifact(
        "graph/ir.json",
        len(graph_payload),
        digest(graph_payload),
        "0.1.0",
        ("causal_lm",),
        (OperatorRequirement("glm_moe_dsa", "1.0.0"),),
    )
    manifest = build_huggingface_mixed_manifest(
        catalog,
        plan,
        journal,
        graph,
        architecture="GlmMoeDsaForCausalLM",
        model_configuration=tiny_config(),
        default_dtype=DType.FLOAT32,
        licenses=("test-only",),
    )
    publish_manifest_last(package_root, manifest, buffer_bytes=79)
    expected_tensors = dict(tensors)
    for name in ternary_names:
        expected_tensors[name] = ternary_reference_tensor(tensors[name], ternary_config)
    return package_root, architecture, GlmReferenceWeights(expected_tensors)


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


def test_mini_glm_prefill_executes_dense_indexshare_and_sparse_experts() -> None:
    architecture = tiny_architecture()
    weights = RejectingExpertWeights(fixture_weights(architecture), ".mlp.experts.1.")
    first = run_glm_moe_dsa_prefill_reference(architecture, weights, (1, 5, 9))
    second = run_glm_moe_dsa_prefill_reference(architecture, weights, (1, 5, 9))
    assert first == second
    assert len(first.logits) == 3
    assert all(len(logits) == architecture.vocab_size for logits in first.logits)
    assert first.layers[0].indexer_type == "full"
    assert first.layers[1].indexer_type == "shared"
    assert first.layers[1].dsa_indices == first.layers[0].dsa_indices
    assert first.layers[0].dsa_indices[0] == (0,)
    assert all(len(indices) <= architecture.index_topk for indices in first.layers[0].dsa_indices)
    assert first.layers[0].expert_routing == ()
    assert all(route.expert_indices == (0,) for route in first.layers[1].expert_routing)
    assert all(all(value == value for value in logits) for logits in first.logits)
    assert [max(range(len(logits)), key=logits.__getitem__) for logits in first.logits] == [
        14,
        14,
        14,
    ]
    assert tuple(logits[0] for logits in first.logits) == pytest.approx(
        (1.1646461163222221, 0.8460455459844063, 0.8805485467331877),
        rel=0,
        abs=1e-14,
    )


def test_mini_glm_mtp_remains_an_explicit_unsupported_feature() -> None:
    architecture = tiny_architecture()
    with pytest.raises(AmsError) as caught:
        run_glm_moe_dsa_prefill_reference(
            architecture,
            fixture_weights(architecture),
            (1,),
            enable_mtp=True,
        )
    assert caught.value.code is ErrorCode.UNSUPPORTED_OP


def test_mini_glm_mixed_package_matches_full_decode_and_skips_unselected_expert(
    tmp_path: Path,
) -> None:
    package_root, architecture, expected_weights = build_mini_glm_package(tmp_path)
    package_weights = GlmPackageWeights.open(
        package_root,
        linear_arena_bytes=64,
        verification_buffer_bytes=31,
    )
    assert package_weights.architecture == architecture
    guarded = RejectingExpertWeights(package_weights, ".mlp.experts.1.")
    actual = run_glm_moe_dsa_prefill_reference(architecture, guarded, (1, 5, 9))
    expected = run_glm_moe_dsa_prefill_reference(architecture, expected_weights, (1, 5, 9))
    assert actual == expected
    evidence = package_weights.read_evidence
    assert evidence.verified_objects > 0
    assert evidence.verification_bytes > 0
    assert evidence.range_read_bytes > 0
    assert evidence.maximum_read_bytes <= 64


def test_mini_glm_package_detects_lazy_object_tampering(tmp_path: Path) -> None:
    package_root, architecture, _ = build_mini_glm_package(tmp_path)
    manifest = json.loads((package_root / "manifest.json").read_bytes())
    embedding = next(
        tensor
        for tensor in manifest["tensors"]
        if tensor["extensions"]["hf.source-name"] == "model.embed_tokens.weight"
    )
    object_id = embedding["layouts"][0]["chunks"][0]["range"]["object_id"]
    storage = next(item for item in manifest["storage_objects"] if item["object_id"] == object_id)
    object_path = package_root / storage["uri"]
    with object_path.open("r+b", buffering=0) as handle:
        original = handle.read(1)
        handle.seek(0)
        handle.write(bytes([original[0] ^ 0x01]))
    package_weights = GlmPackageWeights.open(package_root, linear_arena_bytes=64)
    with pytest.raises(AmsError) as caught:
        run_glm_moe_dsa_prefill_reference(architecture, package_weights, (1,))
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_mini_glm_package_rejects_a_canonically_rewritten_partial_inventory(
    tmp_path: Path,
) -> None:
    package_root, _, _ = build_mini_glm_package(tmp_path)
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["tensors"].pop()
    del manifest["content_root"]
    manifest["content_root"] = digest(canonical_json_bytes(manifest))
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(AmsError) as caught:
        GlmPackageWeights.open(package_root, linear_arena_bytes=64)
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
    assert caught.value.evidence == {"missing": 1, "unexpected": 0}


def test_mini_glm_package_rejects_encoding_feature_drift(tmp_path: Path) -> None:
    package_root, _, _ = build_mini_glm_package(tmp_path)
    manifest_path = package_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["required_features"].remove("ams.codec.ternary.trit5.v1")
    del manifest["content_root"]
    manifest["content_root"] = digest(canonical_json_bytes(manifest))
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(AmsError) as caught:
        GlmPackageWeights.open(package_root, linear_arena_bytes=64)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE
