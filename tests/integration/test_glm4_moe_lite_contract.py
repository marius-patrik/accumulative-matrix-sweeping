import hashlib
import json

import pytest

from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4MoeLiteTensorRole,
    expected_glm4_moe_lite_tensor_slots,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    validate_glm4_moe_lite_tensor_inventory,
)


def tiny_glm4_config(**overrides):
    config = {
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "first_k_dense_replace": 1,
        "hidden_act": "silu",
        "hidden_size": 16,
        "intermediate_size": 32,
        "kv_lora_rank": 4,
        "max_position_embeddings": 128,
        "model_type": "glm4_moe_lite",
        "moe_intermediate_size": 8,
        "n_group": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 2,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_nextn_predict_layers": 1,
        "partial_rotary_factor": 1.0,
        "q_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "rms_norm_eps": 1e-5,
        "rope_scaling": None,
        "rope_theta": 10000,
        "routed_scaling_factor": 1.5,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "v_head_dim": 4,
        "vocab_size": 32,
    }
    config.update(overrides)
    return config


def parse_config(config):
    return parse_glm4_moe_lite_architecture(
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


def test_tiny_inventory_marks_dense_sparse_expert_and_mtp_weights() -> None:
    architecture = parse_config(tiny_glm4_config())
    slots = expected_glm4_moe_lite_tensor_slots(architecture)
    assert len(slots) == 61
    assert (
        sum(slot.role is Glm4MoeLiteTensorRole.ROUTED_EXPERT_GATE_PROJECTION for slot in slots) == 4
    )
    mtp_slots = [slot for slot in slots if slot.mtp]
    assert len(mtp_slots) == 26
    assert {slot.layer_index for slot in mtp_slots} == {2}
    assert {
        slot.role
        for slot in mtp_slots
        if slot.role
        in {
            Glm4MoeLiteTensorRole.MTP_EMBEDDING,
            Glm4MoeLiteTensorRole.MTP_SHARED_HEAD,
            Glm4MoeLiteTensorRole.MTP_SHARED_HEAD_NORM,
        }
    } == {
        Glm4MoeLiteTensorRole.MTP_EMBEDDING,
        Glm4MoeLiteTensorRole.MTP_SHARED_HEAD,
        Glm4MoeLiteTensorRole.MTP_SHARED_HEAD_NORM,
    }
    inventory = validate_glm4_moe_lite_tensor_inventory(
        architecture,
        index_for_names([slot.tensor_name for slot in slots]),
    )
    assert inventory.architecture_hash == architecture.content_hash
    assert len(inventory.slots) == len(slots)


def test_official_glm47_flash_dimensions_match_pinned_inventory_digest() -> None:
    architecture = parse_config(
        tiny_glm4_config(
            hidden_size=2048,
            intermediate_size=10240,
            moe_intermediate_size=1536,
            vocab_size=154880,
            num_hidden_layers=47,
            n_routed_experts=64,
            num_experts_per_tok=4,
            num_attention_heads=20,
            num_key_value_heads=20,
            q_lora_rank=768,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            max_position_embeddings=202752,
            rope_theta=1000000,
            routed_scaling_factor=1.8,
        )
    )
    slots = expected_glm4_moe_lite_tensor_slots(architecture)
    names = "\n".join(slot.tensor_name for slot in slots).encode()
    assert len(slots) == 9_703
    assert hashlib.sha256(names).hexdigest() == (
        "23321d795f0b797ab951613b86cf4d02008e4057b446055fcc2b0265b1f3db3d"
    )
    assert sum(slot.role is Glm4MoeLiteTensorRole.ROUTER_WEIGHT for slot in slots) == 47
    assert (
        sum(slot.role is Glm4MoeLiteTensorRole.ROUTED_EXPERT_UP_PROJECTION for slot in slots)
        == 3_008
    )


@pytest.mark.parametrize(
    "change",
    [
        {"unknown_execution_flag": True},
        {"dtype": "float16"},
        {"first_k_dense_replace": 2},
        {"mlp_layer_types": ["sparse", "sparse"]},
        {"n_routed_experts": 3, "n_group": 2},
        {"n_routed_experts": 1},
        {"n_routed_experts": 4, "n_group": 2, "topk_group": 1, "num_experts_per_tok": 3},
        {"num_key_value_heads": 3},
        {"qk_rope_head_dim": 3},
        {"rope_interleave": False},
        {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
    ],
)
def test_config_rejects_unreviewed_or_inconsistent_semantics(change) -> None:
    with pytest.raises(AmsError) as caught:
        parse_config(tiny_glm4_config(**change))
    assert caught.value.code in {ErrorCode.INVALID_PACKAGE, ErrorCode.CAPABILITY_MISMATCH}


def test_inventory_rejects_missing_and_unexpected_tensors() -> None:
    architecture = parse_config(tiny_glm4_config())
    names = [slot.tensor_name for slot in expected_glm4_moe_lite_tensor_slots(architecture)]
    names.pop()
    names.append("model.layers.99.unreviewed.weight")
    with pytest.raises(AmsError) as caught:
        validate_glm4_moe_lite_tensor_inventory(architecture, index_for_names(names))
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
    assert caught.value.evidence == {"missing": 1, "unexpected": 1}


def test_config_rejects_duplicate_keys() -> None:
    with pytest.raises(AmsError, match="JSON"):
        parse_glm4_moe_lite_architecture(b'{"model_type":"glm4_moe_lite","model_type":"other"}')
