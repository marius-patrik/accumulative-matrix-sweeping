from __future__ import annotations

import json
from pathlib import Path

from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.descriptors import DType
from ams.integrations import (
    GlmTensorRole,
    HuggingFaceCatalogTensor,
    HuggingFaceTensorEncoding,
    build_experimental_glm_precision_candidate,
    expected_glm_tensor_shape,
    expected_glm_tensor_slots,
    parse_glm_moe_dsa_architecture,
    parse_huggingface_shard_index,
    validate_glm_tensor_inventory,
)


def _architecture():
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "attention_bias": False,
        "dtype": "bfloat16",
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 16,
        "index_head_dim": 4,
        "index_n_heads": 2,
        "index_share_for_mtp_iteration": True,
        "index_topk": 4,
        "indexer_types": ["full", "shared"],
        "intermediate_size": 32,
        "kv_lora_rank": 4,
        "max_position_embeddings": 128,
        "mlp_layer_types": ["dense", "sparse"],
        "model_type": "glm_moe_dsa",
        "moe_intermediate_size": 8,
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
        "rope_parameters": {"rope_theta": 10_000, "rope_type": "default"},
        "routed_scaling_factor": 1.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "v_head_dim": 4,
        "vocab_size": 32,
    }
    return parse_glm_moe_dsa_architecture(
        json.dumps(config, separators=(",", ":"), sort_keys=True).encode()
    )


def _fixture():
    architecture = _architecture()
    slots = expected_glm_tensor_slots(architecture)
    shard_name = "model-00001-of-00001.safetensors"
    index = parse_huggingface_shard_index(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {slot.tensor_name: shard_name for slot in slots},
            },
            separators=(",", ":"),
        ).encode()
    )
    inventory = validate_glm_tensor_inventory(architecture, index)
    tensors = []
    offset = 8
    for slot in slots:
        shape = expected_glm_tensor_shape(architecture, slot)
        is_f32 = slot.role is GlmTensorRole.ROUTER_CORRECTION_BIAS
        elements = 1
        for dimension in shape:
            elements *= dimension
        source_length = elements * (4 if is_f32 else 2)
        tensors.append(
            HuggingFaceCatalogTensor(
                tensor_name=slot.tensor_name,
                shard_name=shard_name,
                object_id="fixture:source",
                dtype=DType.FLOAT32 if is_f32 else DType.BFLOAT16,
                source_dtype="F32" if is_f32 else "BF16",
                shape=shape,
                source_offset=offset,
                source_length=source_length,
            )
        )
        offset += source_length
    return architecture, inventory, tuple(tensors)


def test_glm_precision_candidate_is_complete_deterministic_and_nonqualifying() -> None:
    architecture, inventory, tensors = _fixture()
    ternary = TernaryCodecConfig(group_size=8)
    int4 = Int4CodecConfig(group_size=8)
    candidate = build_experimental_glm_precision_candidate(
        architecture,
        inventory,
        tensors,
        ternary_config=ternary,
        int4_config=int4,
    )
    repeated = build_experimental_glm_precision_candidate(
        architecture,
        inventory,
        tensors,
        ternary_config=ternary,
        int4_config=int4,
    )
    assert candidate == repeated
    assert candidate.candidate_hash.startswith("sha256:")
    assert candidate.status.value == "experimental"
    assert len(candidate.assignments) == len(tensors)
    assert candidate.source_bytes == sum(tensor.source_length for tensor in tensors)
    assert candidate.estimated_encoded_bytes < candidate.source_bytes

    assignments = {assignment.tensor_name: assignment for assignment in candidate.assignments}
    slots_by_role = {slot.role: slot for slot in inventory.slots}
    assert (
        assignments[slots_by_role[GlmTensorRole.ROUTER_WEIGHT].tensor_name].encoding
        is HuggingFaceTensorEncoding.IDENTITY
    )
    assert (
        assignments[slots_by_role[GlmTensorRole.INDEXER_WQ_B_PROJECTION].tensor_name].encoding
        is HuggingFaceTensorEncoding.IDENTITY
    )
    expert_assignment = assignments[
        slots_by_role[GlmTensorRole.ROUTED_EXPERT_GATE_PROJECTION].tensor_name
    ]
    assert expert_assignment.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5
    assert expert_assignment.ternary_config == ternary
    embedding_assignment = assignments[slots_by_role[GlmTensorRole.EMBEDDING].tensor_name]
    assert embedding_assignment.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC
    assert embedding_assignment.int4_config == int4


def test_glm_precision_candidate_identity_covers_every_sensitive_index_tensor() -> None:
    architecture, inventory, tensors = _fixture()
    candidate = build_experimental_glm_precision_candidate(
        architecture,
        inventory,
        tensors,
        ternary_config=TernaryCodecConfig(group_size=8),
        int4_config=Int4CodecConfig(group_size=8),
    )
    assignment_by_name = {
        assignment.tensor_name: assignment for assignment in candidate.assignments
    }
    sensitive_roles = {
        GlmTensorRole.INDEXER_WQ_B_PROJECTION,
        GlmTensorRole.INDEXER_WK_PROJECTION,
        GlmTensorRole.INDEXER_K_NORM_WEIGHT,
        GlmTensorRole.INDEXER_K_NORM_BIAS,
        GlmTensorRole.INDEXER_WEIGHTS_PROJECTION,
        GlmTensorRole.ROUTER_WEIGHT,
        GlmTensorRole.ROUTER_CORRECTION_BIAS,
    }
    assert all(
        assignment_by_name[slot.tensor_name].encoding is HuggingFaceTensorEncoding.IDENTITY
        for slot in inventory.slots
        if slot.role in sensitive_roles
    )


def test_committed_glm52_precision_candidate_is_structural_and_nonqualifying() -> None:
    evidence_path = (
        Path(__file__).parents[2] / "docs" / "evidence" / "glm52_precision_candidate.json"
    )
    evidence = json.loads(evidence_path.read_bytes())
    assert evidence["schema_id"] == "ams.glm.precision-candidate.v1"
    assert evidence["status"] == "experimental"
    assert evidence["qualifies_precision_policy"] is False
    assert evidence["source_audit_hash"] == (
        "sha256:20114c227ceb45a137991bac191635dcb485e67252bf2a39a5580459dbae0f5c"
    )
    assert evidence["encoding_counts"] == {
        "identity": 582,
        "int4_symmetric": 635,
        "ternary_trit5": 58_368,
    }
    assert evidence["source_bytes"] == 1_506_659_919_872
    assert evidence["estimated_encoded_bytes"] == 182_650_058_752
    assert evidence["compression_ratio"] == (
        evidence["source_bytes"] / evidence["estimated_encoded_bytes"]
    )
