import hashlib
import json
import struct
from pathlib import Path

import pytest

from ams.canonical import canonical_json_bytes
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4QuantizationProbeConfig,
    Glm4QuantizationProbeStatus,
    expected_glm4_moe_lite_tensor_slots,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    probe_experimental_glm4_quantization_shard,
    validate_glm4_moe_lite_tensor_inventory,
)

_SHARD = "model-00001-of-00002.safetensors"
_OTHER_SHARD = "model-00002-of-00002.safetensors"
_TERNARY_TENSOR = "model.layers.1.mlp.experts.0.gate_proj.weight"
_INT4_TENSOR = "model.layers.1.self_attn.q_a_proj.weight"
_IDENTITY_TENSOR = "model.layers.1.input_layernorm.weight"


class ObservedMemoryReader:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.size_bytes = len(payload)
        self.reads: list[tuple[int, int]] = []

    def read_into(self, offset: int, destination) -> None:
        view = memoryview(destination).cast("B")
        self.reads.append((offset, view.nbytes))
        view[:] = self.payload[offset : offset + view.nbytes]


def _digest(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _architecture():
    config = {
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
    return parse_glm4_moe_lite_architecture(canonical_json_bytes(config))


def _bfloat16(values: list[float]) -> bytes:
    words = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
    return struct.pack(f"<{len(words)}H", *words)


def _safetensors_payload() -> bytes:
    tensors = (
        (_TERNARY_TENSOR, (3, 4), [float(index - 5) for index in range(12)]),
        (_INT4_TENSOR, (2, 4), [-7.0, -4.0, -1.0, 0.0, 1.0, 2.0, 4.0, 7.0]),
        (_IDENTITY_TENSOR, (4,), [1.0, 1.0, 1.0, 1.0]),
    )
    header = {}
    data = bytearray()
    for name, shape, values in tensors:
        start = len(data)
        data.extend(_bfloat16(values))
        header[name] = {
            "data_offsets": [start, len(data)],
            "dtype": "BF16",
            "shape": list(shape),
        }
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    return struct.pack("<Q", len(header_bytes)) + header_bytes + data


def _index(*, identity_in_probe_shard: bool = True):
    architecture = _architecture()
    selected = {_TERNARY_TENSOR, _INT4_TENSOR}
    if identity_in_probe_shard:
        selected.add(_IDENTITY_TENSOR)
    weight_map = {
        slot.tensor_name: _SHARD if slot.tensor_name in selected else _OTHER_SHARD
        for slot in expected_glm4_moe_lite_tensor_slots(architecture)
    }
    payload = canonical_json_bytes(
        {
            "metadata": {"total_size": len(weight_map) * 2},
            "weight_map": weight_map,
        }
    )
    index = parse_huggingface_shard_index(payload)
    inventory = validate_glm4_moe_lite_tensor_inventory(architecture, index)
    return architecture, index, inventory


def _probe(
    reader: ObservedMemoryReader,
    *,
    identity_in_probe_shard: bool = True,
    expected_shard_hash: str | None = None,
):
    architecture, index, inventory = _index(identity_in_probe_shard=identity_in_probe_shard)
    return probe_experimental_glm4_quantization_shard(
        architecture,
        inventory,
        index,
        source_repository="fixture/glm4",
        source_revision="fixture-revision",
        shard_name=_SHARD,
        reader=reader,
        expected_shard_hash=expected_shard_hash or _digest(reader.payload),
        candidate_hash=_digest("candidate"),
        policy_hash=_digest("policy"),
        ternary_config=TernaryCodecConfig(group_size=4),
        int4_config=Int4CodecConfig(group_size=4),
        config=Glm4QuantizationProbeConfig(groups_per_tensor=2, hash_buffer_bytes=7),
    )


def test_probe_is_deterministic_bounded_and_explicitly_nonqualifying() -> None:
    payload = _safetensors_payload()
    first_reader = ObservedMemoryReader(payload)
    first = _probe(first_reader)
    second = _probe(ObservedMemoryReader(payload))

    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.status is Glm4QuantizationProbeStatus.DIAGNOSTIC
    assert first.qualifies_precision_policy is False
    assert first.source_repository == "fixture/glm4"
    assert first.source_revision == "fixture-revision"
    assert first.shard_tensor_count == 3
    assert first.identity_tensor_count == 1
    assert first.sampled_tensor_count == 2
    assert first.sampled_group_count == 4
    assert first.sampled_element_count == 16
    assert first.sampled_source_bytes_read == 32
    assert first.maximum_sample_read_bytes == 8
    assert first.integrity_bytes_read == len(payload)
    assert first.encoding_tensor_counts == (
        ("identity", 1),
        ("int4_symmetric", 1),
        ("ternary_trit5", 1),
    )
    assert {metric.scope for metric in first.encoding_metrics} == {
        "int4_symmetric",
        "ternary_trit5",
    }
    metrics = {metric.scope: metric for metric in first.encoding_metrics}
    assert metrics["int4_symmetric"].mean_absolute_error == 0
    assert metrics["ternary_trit5"].mean_absolute_error > 0
    assert all(0 < metric.cosine_similarity <= 1 for metric in first.encoding_metrics)
    assert any(length == 7 for _, length in first_reader.reads)
    assert not any(length in {16, 24} for _, length in first_reader.reads)


def test_probe_rejects_wrong_full_shard_hash_before_emitting_evidence() -> None:
    reader = ObservedMemoryReader(_safetensors_payload())
    with pytest.raises(AmsError, match="hash mismatch") as caught:
        _probe(reader, expected_shard_hash="sha256:" + "0" * 64)
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_probe_rejects_header_and_normalized_index_set_drift() -> None:
    reader = ObservedMemoryReader(_safetensors_payload())
    with pytest.raises(AmsError, match="exactly match") as caught:
        _probe(reader, identity_in_probe_shard=False)
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
    assert caught.value.evidence == {"missing": 0, "unexpected": 1}


def test_committed_official_shard_evidence_is_nonqualifying_and_self_consistent() -> None:
    evidence_path = (
        Path(__file__).parents[2] / "docs" / "evidence" / "glm47_shard2_quantization_probe.json"
    )
    evidence = json.loads(evidence_path.read_bytes())
    assert evidence["schema_id"] == "ams.glm4.quantization-probe.v1"
    assert evidence["status"] == "diagnostic"
    assert evidence["qualifies_precision_policy"] is False
    assert evidence["candidate_hash"] == _digest(
        canonical_json_bytes(
            {
                "architecture_hash": evidence["architecture_hash"],
                "encoding_counts": [
                    {"encoding": encoding, "tensor_count": count}
                    for encoding, count in (
                        ("identity", 292),
                        ("ternary_trit5", 9024),
                        ("int4_symmetric", 387),
                    )
                ],
                "estimated_encoded_bytes": 9_100_218_112,
                "policy_hash": evidence["policy_hash"],
                "source_bytes": 62_442_983_168,
                "source_index_hash": evidence["source_index_hash"],
                "status": "experimental",
            }
        )
    )
    assert (
        evidence["sampled_tensor_count"] + evidence["identity_tensor_count"]
        == evidence["shard_tensor_count"]
    )
    assert (
        sum(metric["sampled_group_count"] for metric in evidence["encoding_metrics"])
        == evidence["sampled_group_count"]
    )
    assert (
        sum(metric["sampled_element_count"] for metric in evidence["encoding_metrics"])
        == evidence["sampled_element_count"]
    )
    assert (
        sum(metric["sampled_source_bytes"] for metric in evidence["encoding_metrics"])
        == evidence["sampled_source_bytes_read"]
    )
    assert evidence["maximum_sample_read_bytes"] == 256
