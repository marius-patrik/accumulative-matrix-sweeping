from __future__ import annotations

import json
from pathlib import Path


def test_committed_int4_bringup_candidate_is_distinct_and_nonqualifying() -> None:
    evidence_root = Path(__file__).parents[2] / "docs" / "evidence"
    capacity = json.loads((evidence_root / "glm47_precision_candidate.json").read_bytes())
    bringup = json.loads((evidence_root / "glm47_int4_bringup_candidate.json").read_bytes())

    assert capacity["profile"] == "ternary_capacity_v1"
    assert bringup["profile"] == "int4_bringup_v1"
    assert bringup["status"] == "experimental"
    assert bringup["candidate_hash"] != capacity["candidate_hash"]
    assert bringup["policy_hash"] != capacity["policy_hash"]
    assert bringup["source_index_hash"] == capacity["source_index_hash"]
    assert bringup["source_bytes"] == capacity["source_bytes"] == 62_442_983_168
    assert bringup["tensor_count"] == capacity["tensor_count"] == 9_703
    assert bringup["encoding_counts"] == {
        "identity": 292,
        "int4_symmetric": 9_411,
    }
    assert bringup["estimated_encoded_bytes"] == 17_527_623_424
    assert bringup["int4_config_hash"] == capacity["int4_config_hash"]
    assert bringup["ternary_config_hash"] is None
