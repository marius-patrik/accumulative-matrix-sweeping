from __future__ import annotations

import json
from pathlib import Path


def test_committed_expert_codec_comparison_selects_staged_ab_path() -> None:
    evidence_root = Path(__file__).parents[2] / "docs" / "evidence"
    comparison = json.loads(
        (evidence_root / "glm47_shard2_expert_codec_comparison.json").read_bytes()
    )
    candidate = json.loads((evidence_root / "glm47_precision_candidate.json").read_bytes())

    assert comparison["schema_id"] == "ams.glm4.expert-codec-comparison-summary.v1"
    assert comparison["status"] == "diagnostic"
    assert comparison["qualifies_precision_policy"] is False
    assert comparison["decision_status"] == "int4_bringup_then_residual2_ab"
    assert comparison["baseline_candidate_hash"] == candidate["candidate_hash"]
    assert comparison["baseline_policy_hash"] == candidate["policy_hash"]
    assert comparison["sampled_group_count"] == 12_288
    assert comparison["sampled_element_count"] == 1_572_864
    assert comparison["maximum_sample_read_bytes"] == 256
    assert comparison["shard_content_hash"] == (
        "sha256:8c51e2434efe609cbe652014a924e088a5ea97be35ca29cfa893a1a9a90304b1"
    )

    variants = {variant["variant_id"]: variant for variant in comparison["variants"]}
    assert len(variants) == 7
    int2 = variants["int2-symmetric-midrise"]
    int3 = variants["int3-symmetric"]
    int4 = variants["int4-symmetric"]
    residual2 = variants["residual2-ternary-threshold-08-of-10"]
    sparse4 = variants["ternary-threshold-08-sparse-bf16-k04"]
    sparse8 = variants["ternary-threshold-08-sparse-bf16-k08"]
    sparse16 = variants["ternary-threshold-08-sparse-bf16-k16"]

    assert int2["bits_per_weight"] == 2.25
    assert int2["normalized_root_mean_square_error"] > 0.5
    assert int3["bits_per_weight"] == 3.25
    assert int3["normalized_root_mean_square_error"] == 0.27406993486967834
    assert residual2["bits_per_weight"] == 3.75
    assert residual2["normalized_root_mean_square_error"] == 0.20045466131651093
    assert int4["bits_per_weight"] == 4.25
    assert int4["normalized_root_mean_square_error"] == 0.1175447379580027

    assert int3["selected_tensor_encoded_bytes"] < residual2["selected_tensor_encoded_bytes"]
    assert (
        int3["normalized_root_mean_square_error"] > residual2["normalized_root_mean_square_error"]
    )
    assert residual2["selected_tensor_encoded_bytes"] < int4["selected_tensor_encoded_bytes"]
    assert (
        residual2["normalized_root_mean_square_error"] > int4["normalized_root_mean_square_error"]
    )
    assert sparse8["selected_tensor_encoded_bytes"] > int3["selected_tensor_encoded_bytes"]
    assert sparse8["normalized_root_mean_square_error"] > int3["normalized_root_mean_square_error"]
    assert sparse16["selected_tensor_encoded_bytes"] > residual2["selected_tensor_encoded_bytes"]
    assert (
        sparse16["normalized_root_mean_square_error"]
        > residual2["normalized_root_mean_square_error"]
    )
    assert int2["selected_tensor_encoded_bytes"] < sparse4["selected_tensor_encoded_bytes"]
    assert int2["normalized_root_mean_square_error"] > sparse4["normalized_root_mean_square_error"]
