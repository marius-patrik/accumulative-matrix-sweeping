import json
from dataclasses import replace
from pathlib import Path

import pytest

from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    GlmTensorRole,
    HuggingFaceCatalogTensor,
    expected_glm_tensor_shape,
    expected_glm_tensor_slots,
    parse_glm_moe_dsa_architecture,
    parse_huggingface_shard_index,
    validate_glm_tensor_catalog,
    validate_glm_tensor_inventory,
)


def tiny_glm_config(**overrides):
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
        "rope_parameters": {"rope_theta": 10000, "rope_type": "default"},
        "routed_scaling_factor": 1.5,
        "scoring_func": "sigmoid",
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "v_head_dim": 4,
        "vocab_size": 32,
    }
    config.update(overrides)
    return config


def parse_config(config):
    return parse_glm_moe_dsa_architecture(
        json.dumps(config, separators=(",", ":"), sort_keys=True).encode()
    )


def index_for_names(names):
    payload = json.dumps(
        {
            "metadata": {"total_size": len(names) * 2},
            "weight_map": {name: "model-00001-of-00001.safetensors" for name in names},
        },
        separators=(",", ":"),
    ).encode()
    return parse_huggingface_shard_index(payload)


def test_tiny_glm_inventory_marks_sparse_experts_shared_index_and_mtp() -> None:
    architecture = parse_config(tiny_glm_config())
    slots = expected_glm_tensor_slots(architecture)
    assert len(slots) == 69
    assert sum(slot.role is GlmTensorRole.ROUTED_EXPERT_GATE_PROJECTION for slot in slots) == 4
    assert sum(slot.role is GlmTensorRole.INDEXER_WEIGHTS_PROJECTION for slot in slots) == 2
    mtp_slots = [slot for slot in slots if slot.mtp]
    assert len(mtp_slots) == 29
    assert {slot.layer_index for slot in mtp_slots} == {2}
    slot_by_role = {slot.role: slot for slot in slots}
    assert expected_glm_tensor_shape(architecture, slot_by_role[GlmTensorRole.EMBEDDING]) == (
        32,
        16,
    )
    assert expected_glm_tensor_shape(
        architecture, slot_by_role[GlmTensorRole.ATTENTION_Q_B_PROJECTION]
    ) == (8, 4)
    assert expected_glm_tensor_shape(
        architecture, slot_by_role[GlmTensorRole.INDEXER_WQ_B_PROJECTION]
    ) == (8, 4)
    assert expected_glm_tensor_shape(
        architecture, slot_by_role[GlmTensorRole.ROUTED_EXPERT_DOWN_PROJECTION]
    ) == (16, 8)
    assert expected_glm_tensor_shape(
        architecture, slot_by_role[GlmTensorRole.MTP_EMBED_HIDDEN_PROJECTION]
    ) == (16, 32)
    inventory = validate_glm_tensor_inventory(
        architecture,
        index_for_names([slot.tensor_name for slot in slots]),
    )
    assert inventory.architecture_hash == architecture.content_hash
    assert len(inventory.slots) == len(slots)


def test_tiny_glm_catalog_binds_every_shape_dtype_and_byte_length() -> None:
    architecture = parse_config(tiny_glm_config())
    slots = expected_glm_tensor_slots(architecture)
    index = index_for_names([slot.tensor_name for slot in slots])
    inventory = validate_glm_tensor_inventory(architecture, index)
    tensors = []
    offset = 8
    for slot in slots:
        shape = expected_glm_tensor_shape(architecture, slot)
        is_f32 = slot.role is GlmTensorRole.ROUTER_CORRECTION_BIAS
        length = 1
        for dimension in shape:
            length *= dimension
        length *= 4 if is_f32 else 2
        tensors.append(
            HuggingFaceCatalogTensor(
                tensor_name=slot.tensor_name,
                shard_name="model-00001-of-00001.safetensors",
                object_id="fixture:source",
                dtype=DType.FLOAT32 if is_f32 else DType.BFLOAT16,
                source_dtype="F32" if is_f32 else "BF16",
                shape=shape,
                source_offset=offset,
                source_length=length,
            )
        )
        offset += length
    catalog = tuple(tensors)
    validate_glm_tensor_catalog(architecture, inventory, catalog)

    projection_index = next(
        index
        for index, tensor in enumerate(catalog)
        if tensor.tensor_name.endswith("self_attn.q_a_proj.weight")
    )
    projection = catalog[projection_index]
    transposed = (
        *catalog[:projection_index],
        replace(projection, shape=tuple(reversed(projection.shape))),
        *catalog[projection_index + 1 :],
    )
    with pytest.raises(AmsError, match="shape, dtype, or byte length") as caught:
        validate_glm_tensor_catalog(architecture, inventory, transposed)
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_official_glm52_dimensions_imply_the_observed_59585_tensor_names() -> None:
    architecture = parse_config(
        tiny_glm_config(
            num_hidden_layers=78,
            num_nextn_predict_layers=1,
            first_k_dense_replace=3,
            mlp_layer_types=["dense"] * 3 + ["sparse"] * 75,
            indexer_types=[
                "full" if index < 3 or (index - 2) % 4 == 0 else "shared" for index in range(78)
            ],
            n_routed_experts=256,
            num_experts_per_tok=8,
        )
    )
    slots = expected_glm_tensor_slots(architecture)
    assert len(slots) == 59_585
    assert sum(slot.role is GlmTensorRole.ROUTED_EXPERT_UP_PROJECTION for slot in slots) == 19_456
    assert sum(slot.role is GlmTensorRole.INDEXER_WK_PROJECTION for slot in slots) == 22


def test_official_glm52_source_audit_is_pinned_and_structural_only() -> None:
    evidence_path = Path(__file__).parents[2] / "docs" / "evidence" / "glm52_source_audit.json"
    evidence = json.loads(evidence_path.read_bytes())
    assert evidence["repository"] == "zai-org/GLM-5.2"
    assert evidence["revision"] == "b4734de4facf877f85769a911abafc5283eab3d9"
    assert evidence["architecture_hash"] == (
        "sha256:185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a"
    )
    assert evidence["index_hash"] == (
        "sha256:5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e"
    )
    assert evidence["shard_inventory_hash"] == (
        "sha256:a7ed6dcbd48c7740d354d723a2e428ae74daf5e269d5da020b05389f40aab512"
    )
    assert evidence["shard_count"] == 282
    assert evidence["tensor_count"] == 59_585
    assert evidence["tensor_bytes"] == evidence["declared_total_size"] == 1_506_659_919_872
    assert evidence["tensor_elements"] == 753_329_940_480
    assert evidence["header_bytes_read"] == 7_467_536
    assert evidence["weight_payload_bytes_read"] == 0
    assert evidence["status"] == "structural_headers_only"
    assert evidence["qualifies_precision_policy"] is False


@pytest.mark.parametrize(
    "change",
    [
        {"unknown_execution_flag": True},
        {"mlp_layer_types": ["sparse", "sparse"]},
        {"indexer_types": ["shared", "full"]},
        {"qk_head_dim": 5},
        {"qk_rope_head_dim": 3, "qk_nope_head_dim": 1},
        {"num_key_value_heads": 3},
        {"dtype": "float16"},
    ],
)
def test_glm_config_rejects_unreviewed_or_inconsistent_semantics(change) -> None:
    with pytest.raises(AmsError) as caught:
        parse_config(tiny_glm_config(**change))
    assert caught.value.code in {ErrorCode.INVALID_PACKAGE, ErrorCode.CAPABILITY_MISMATCH}


def test_glm_inventory_rejects_missing_and_unexpected_tensors() -> None:
    architecture = parse_config(tiny_glm_config())
    names = [slot.tensor_name for slot in expected_glm_tensor_slots(architecture)]
    names.pop()
    names.append("model.layers.99.unreviewed.weight")
    with pytest.raises(AmsError) as caught:
        validate_glm_tensor_inventory(architecture, index_for_names(names))
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
    assert caught.value.evidence == {"missing": 1, "unexpected": 1}


def test_glm_config_rejects_duplicate_keys() -> None:
    with pytest.raises(AmsError, match="JSON"):
        parse_glm_moe_dsa_architecture(b'{"model_type":"glm_moe_dsa","model_type":"other"}')
