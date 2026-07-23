from __future__ import annotations

import json
from pathlib import Path


def test_committed_routed_expert_comparison_rejects_threshold_only_ternary() -> None:
    evidence_root = Path(__file__).parents[2] / "docs" / "evidence"
    comparison = json.loads(
        (evidence_root / "glm47_shard2_routed_expert_comparison.json").read_bytes()
    )
    candidate = json.loads((evidence_root / "glm47_precision_candidate.json").read_bytes())
    assert comparison["schema_id"] == "ams.glm4.quantization-comparison-summary.v1"
    assert comparison["status"] == "diagnostic"
    assert comparison["qualifies_precision_policy"] is False
    assert comparison["decision_status"] == "requires_higher_capacity_candidate"
    assert comparison["baseline_candidate_hash"] == candidate["candidate_hash"]
    assert comparison["baseline_policy_hash"] == candidate["policy_hash"]
    assert comparison["sampled_group_count"] == 12_288
    assert comparison["sampled_element_count"] == 1_572_864
    assert comparison["maximum_sample_read_bytes"] == 256

    variants = {variant["variant_id"]: variant for variant in comparison["variants"]}
    assert len(variants) == 9
    ternary = [variant for variant in variants.values() if variant["encoding"] == "ternary_trit5"]
    assert {variant["variant_id"] for variant in ternary} == {
        f"ternary-threshold-{numerator:02d}-of-10" for numerator in range(3, 11)
    }
    assert {variant["selected_tensor_encoded_bytes"] for variant in ternary} == {141_557_760}
    best_ternary = min(
        ternary,
        key=lambda variant: variant["normalized_root_mean_square_error"],
    )
    assert best_ternary["variant_id"] == "ternary-threshold-08-of-10"
    assert best_ternary["cosine_similarity"] == 0.9008458699259669
    assert best_ternary["normalized_root_mean_square_error"] == 0.43413905449406004

    int4 = variants["int4-symmetric"]
    assert int4["selected_tensor_encoded_bytes"] == 320_864_256
    assert int4["cosine_similarity"] > best_ternary["cosine_similarity"]
    assert (
        int4["normalized_root_mean_square_error"]
        < best_ternary["normalized_root_mean_square_error"] / 3
    )
