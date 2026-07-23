import hashlib
import json
from dataclasses import replace

import pytest

from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4MoeLiteTensorRole,
    Glm4PrecisionCandidateStatus,
    Glm4PrecisionProfile,
    Glm4PrecisionQualityEvidence,
    Glm4PrecisionQualityThresholds,
    HuggingFaceCatalogTensor,
    HuggingFaceTensorEncoding,
    build_accuracy_first_glm4_precision_candidate,
    build_experimental_glm4_precision_candidate,
    expected_glm4_moe_lite_tensor_shape,
    expected_glm4_moe_lite_tensor_slots,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    qualify_glm4_precision_candidate,
    validate_glm4_moe_lite_tensor_inventory,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _architecture():
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
    return parse_glm4_moe_lite_architecture(
        json.dumps(config, separators=(",", ":"), sort_keys=True).encode()
    )


def _candidate_fixture():
    architecture = _architecture()
    slots = expected_glm4_moe_lite_tensor_slots(architecture)
    index_payload = json.dumps(
        {
            "metadata": {"total_size": len(slots) * 2},
            "weight_map": {slot.tensor_name: "model-00001-of-00001.safetensors" for slot in slots},
        },
        separators=(",", ":"),
    ).encode()
    index = parse_huggingface_shard_index(index_payload)
    inventory = validate_glm4_moe_lite_tensor_inventory(architecture, index)
    tensors = []
    offset = 0
    for slot in slots:
        shape = expected_glm4_moe_lite_tensor_shape(architecture, slot)
        length = 2
        for dimension in shape:
            length *= dimension
        tensors.append(
            HuggingFaceCatalogTensor(
                tensor_name=slot.tensor_name,
                shard_name="model-00001-of-00001.safetensors",
                object_id="hf:model-00001-of-00001.safetensors",
                dtype=DType.BFLOAT16,
                source_dtype="BF16",
                shape=shape,
                source_offset=offset,
                source_length=length,
            )
        )
        offset += length
    ternary = TernaryCodecConfig(group_size=16)
    int4 = Int4CodecConfig(group_size=16)
    candidate = build_experimental_glm4_precision_candidate(
        architecture,
        inventory,
        tuple(tensors),
        ternary_config=ternary,
        int4_config=int4,
    )
    return architecture, inventory, tuple(tensors), slots, ternary, int4, candidate


def test_candidate_assigns_every_reviewed_role_and_estimates_exact_bytes() -> None:
    _, _, tensors, slots, ternary, int4, candidate = _candidate_fixture()
    assignment_by_name = {
        assignment.tensor_name: assignment for assignment in candidate.assignments
    }
    ternary_roles = {
        Glm4MoeLiteTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
        Glm4MoeLiteTensorRole.ROUTED_EXPERT_UP_PROJECTION,
        Glm4MoeLiteTensorRole.ROUTED_EXPERT_DOWN_PROJECTION,
    }
    identity_roles = {
        Glm4MoeLiteTensorRole.EMBEDDING,
        Glm4MoeLiteTensorRole.FINAL_NORM,
        Glm4MoeLiteTensorRole.INPUT_NORM,
        Glm4MoeLiteTensorRole.POST_ATTENTION_NORM,
        Glm4MoeLiteTensorRole.ATTENTION_Q_A_NORM,
        Glm4MoeLiteTensorRole.ATTENTION_KV_A_NORM,
        Glm4MoeLiteTensorRole.ROUTER_WEIGHT,
        Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS,
        Glm4MoeLiteTensorRole.MTP_EMBED_NORM,
        Glm4MoeLiteTensorRole.MTP_HIDDEN_NORM,
        Glm4MoeLiteTensorRole.MTP_EMBEDDING,
        Glm4MoeLiteTensorRole.MTP_SHARED_HEAD_NORM,
    }
    expected_bytes = 0
    tensor_by_name = {tensor.tensor_name: tensor for tensor in tensors}
    for slot in slots:
        assignment = assignment_by_name[slot.tensor_name]
        elements = 1
        for dimension in tensor_by_name[slot.tensor_name].shape:
            elements *= dimension
        if slot.role in identity_roles:
            assert assignment.encoding is HuggingFaceTensorEncoding.IDENTITY
            expected_bytes += tensor_by_name[slot.tensor_name].source_length
        elif slot.role in ternary_roles:
            assert assignment.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5
            expected_bytes += ternary.encoded_size(elements)
        else:
            assert assignment.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC
            expected_bytes += int4.encoded_size(elements)
    assert candidate.status is Glm4PrecisionCandidateStatus.EXPERIMENTAL
    assert candidate.source_bytes == sum(tensor.source_length for tensor in tensors)
    assert candidate.estimated_encoded_bytes == expected_bytes
    assert sum(count for _, count in candidate.encoding_counts) == len(slots)


def test_candidate_is_deterministic_and_rejects_catalog_drift() -> None:
    architecture, inventory, tensors, _, ternary, int4, candidate = _candidate_fixture()
    repeated = build_experimental_glm4_precision_candidate(
        architecture,
        inventory,
        tuple(reversed(tensors)),
        ternary_config=ternary,
        int4_config=int4,
    )
    assert repeated.candidate_hash == candidate.candidate_hash
    assert repeated.policy.policy_hash == candidate.policy.policy_hash

    missing = tensors[:-1]
    with pytest.raises(AmsError, match="catalog") as caught:
        build_experimental_glm4_precision_candidate(
            architecture,
            inventory,
            missing,
            ternary_config=ternary,
            int4_config=int4,
        )
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH

    transposed = replace(tensors[0], shape=tuple(reversed(tensors[0].shape)))
    with pytest.raises(AmsError, match="shape") as caught:
        build_experimental_glm4_precision_candidate(
            architecture,
            inventory,
            (transposed, *tensors[1:]),
            ternary_config=ternary,
            int4_config=int4,
        )
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH

    wrong_dtype = replace(tensors[0], dtype=DType.FLOAT16)
    with pytest.raises(AmsError, match="dtype") as caught:
        build_experimental_glm4_precision_candidate(
            architecture,
            inventory,
            (wrong_dtype, *tensors[1:]),
            ternary_config=ternary,
            int4_config=int4,
        )
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH


def test_accuracy_first_candidate_changes_only_compressed_roles_to_int4() -> None:
    architecture, inventory, tensors, slots, _, int4, capacity = _candidate_fixture()
    accuracy = build_accuracy_first_glm4_precision_candidate(
        architecture,
        inventory,
        tensors,
        int4_config=int4,
    )
    repeated = build_accuracy_first_glm4_precision_candidate(
        architecture,
        inventory,
        tuple(reversed(tensors)),
        int4_config=int4,
    )

    capacity_by_name = {assignment.tensor_name: assignment for assignment in capacity.assignments}
    accuracy_by_name = {assignment.tensor_name: assignment for assignment in accuracy.assignments}
    tensor_by_name = {tensor.tensor_name: tensor for tensor in tensors}
    expected_bytes = 0
    identity_count = 0
    for slot in slots:
        before = capacity_by_name[slot.tensor_name]
        after = accuracy_by_name[slot.tensor_name]
        if before.encoding is HuggingFaceTensorEncoding.IDENTITY:
            assert after.encoding is HuggingFaceTensorEncoding.IDENTITY
            expected_bytes += tensor_by_name[slot.tensor_name].source_length
            identity_count += 1
        else:
            assert after.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC
            assert after.int4_config == int4
            assert after.ternary_config is None
            elements = 1
            for dimension in tensor_by_name[slot.tensor_name].shape:
                elements *= dimension
            expected_bytes += int4.encoded_size(elements)

    assert dict(accuracy.encoding_counts) == {
        HuggingFaceTensorEncoding.IDENTITY: identity_count,
        HuggingFaceTensorEncoding.INT4_SYMMETRIC: len(slots) - identity_count,
    }
    assert accuracy.estimated_encoded_bytes == expected_bytes
    assert accuracy.candidate_hash != capacity.candidate_hash
    assert accuracy.policy.policy_hash != capacity.policy.policy_hash
    assert repeated.candidate_hash == accuracy.candidate_hash
    assert repeated.policy.policy_hash == accuracy.policy.policy_hash
    assert Glm4PrecisionProfile.INT4_BRINGUP.value == "int4_bringup_v1"


def _evidence(candidate, **overrides):
    values = {
        "candidate_hash": candidate.candidate_hash,
        "source_index_hash": candidate.source_index_hash,
        "calibration_corpus_hash": _digest("calibration"),
        "evaluation_corpus_hash": _digest("evaluation"),
        "evaluator_hash": _digest("evaluator"),
        "trusted_baseline_hash": _digest("baseline"),
        "candidate_runtime_hash": _digest("runtime"),
        "evaluated_tokens": 10_000,
        "evaluated_tasks": 20,
        "mean_token_nll_delta": 0.04,
        "top1_token_agreement": 0.92,
        "task_score_retention": 0.97,
    }
    values.update(overrides)
    return Glm4PrecisionQualityEvidence(**values)


def test_qualification_requires_matching_identity_and_every_explicit_threshold() -> None:
    *_, candidate = _candidate_fixture()
    thresholds = Glm4PrecisionQualityThresholds(
        minimum_evaluated_tokens=10_000,
        minimum_evaluated_tasks=20,
        maximum_mean_token_nll_delta=0.05,
        minimum_top1_token_agreement=0.90,
        minimum_task_score_retention=0.95,
    )
    qualified = qualify_glm4_precision_candidate(candidate, thresholds, _evidence(candidate))
    assert qualified.candidate is candidate
    assert qualified.qualification_hash.startswith("sha256:")

    with pytest.raises(AmsError, match="exact candidate") as caught:
        qualify_glm4_precision_candidate(
            candidate,
            thresholds,
            _evidence(candidate, candidate_hash=_digest("other")),
        )
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE

    with pytest.raises(AmsError, match="quality thresholds") as caught:
        qualify_glm4_precision_candidate(
            candidate,
            thresholds,
            _evidence(
                candidate,
                evaluated_tokens=9_999,
                evaluated_tasks=19,
                mean_token_nll_delta=0.06,
                top1_token_agreement=0.89,
                task_score_retention=0.94,
            ),
        )
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
    assert caught.value.evidence["failed_metrics"] == (
        "evaluated_tokens,evaluated_tasks,mean_token_nll_delta,"
        "top1_token_agreement,task_score_retention"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_mean_token_nll_delta", -0.01),
        ("minimum_top1_token_agreement", 1.01),
        ("minimum_task_score_retention", float("nan")),
    ],
)
def test_quality_thresholds_reject_invalid_numeric_contract(field, value) -> None:
    values = {
        "minimum_evaluated_tokens": 1,
        "minimum_evaluated_tasks": 1,
        "maximum_mean_token_nll_delta": 0.1,
        "minimum_top1_token_agreement": 0.8,
        "minimum_task_score_retention": 0.9,
    }
    values[field] = value
    with pytest.raises(AmsError) as caught:
        Glm4PrecisionQualityThresholds(**values)
    assert caught.value.code is ErrorCode.PLAN_INVALID
