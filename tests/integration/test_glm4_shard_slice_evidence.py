from __future__ import annotations

import json
from pathlib import Path


def test_committed_expert_zero_slice_is_authenticated_nonqualifying_int4() -> None:
    evidence_root = Path(__file__).parents[2] / "docs" / "evidence"
    candidate = json.loads((evidence_root / "glm47_int4_bringup_candidate.json").read_bytes())
    comparison = json.loads(
        (evidence_root / "glm47_shard2_expert_codec_comparison.json").read_bytes()
    )
    conversion = json.loads(
        (evidence_root / "glm47_shard2_expert0_int4_conversion.json").read_bytes()
    )

    assert conversion["schema_id"] == "ams.glm4.shard-slice-conversion.v1"
    assert conversion["status"] == "diagnostic"
    assert conversion["qualifies_precision_policy"] is False
    assert conversion["publishes_model_manifest"] is False
    assert conversion["candidate_hash"] == candidate["candidate_hash"]
    assert conversion["full_policy_hash"] == candidate["policy_hash"]
    assert conversion["source_index_hash"] == candidate["source_index_hash"]
    assert conversion["shard_content_hash"] == comparison["shard_content_hash"]
    assert conversion["tensor_count"] == 3
    assert conversion["source_tensor_bytes"] == 18_874_368
    assert conversion["encoded_tensor_bytes"] == 5_013_504
    assert conversion["maximum_codec_source_read_bytes"] == 256

    outputs = {output["tensor_name"]: output for output in conversion["outputs"]}
    assert set(outputs) == {
        "model.layers.1.mlp.experts.0.down_proj.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.layers.1.mlp.experts.0.up_proj.weight",
    }
    assert {output["encoding"] for output in outputs.values()} == {"int4_symmetric"}
    assert {output["source_bytes"] for output in outputs.values()} == {6_291_456}
    assert {output["encoded_bytes"] for output in outputs.values()} == {1_671_168}
    assert len({output["source_checksum"] for output in outputs.values()}) == 3
    assert len({output["target_hash"] for output in outputs.values()}) == 3
