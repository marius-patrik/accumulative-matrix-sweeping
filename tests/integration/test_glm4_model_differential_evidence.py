import hashlib
import json
from pathlib import Path

from jsonschema.validators import Draft202012Validator

from ams.canonical import canonical_json_bytes


def test_complete_native_glm47_evidence_is_authenticated_and_passes() -> None:
    root = Path(__file__).parents[2]
    evidence = json.loads(
        (root / "docs" / "evidence" / "glm47_complete_bf16_differential.json").read_bytes()
    )
    schema = json.loads((root / "schemas" / "glm4-model-differential.schema.json").read_bytes())
    Draft202012Validator(schema).validate(evidence)

    source = evidence["source"]
    storage = source["storage"]
    source_identity = {
        "schema_id": "ams.glm47-complete-source.v1",
        "repository": source["repository"],
        "revision": source["revision"],
        "architecture_hash": source["architecture_hash"],
        "source_index_hash": source["source_index_hash"],
        "storage": storage,
    }
    source_root = "sha256:" + hashlib.sha256(canonical_json_bytes(source_identity)).hexdigest()

    assert evidence["status"] == "passed"
    assert evidence["blockers"] == []
    assert source["source_root"] == (
        "sha256:8bff00fd6d3d5b6066b0d16b8f89302ab003c7558b2ce639aeab641be77f8fec"
    )
    assert source_root == source["source_root"]
    assert source["shard_count"] == len(storage) == 48
    assert len({item["object_id"] for item in storage}) == 48
    assert source["source_storage_bytes"] == sum(item["size_bytes"] for item in storage)
    assert source["source_storage_bytes"] == 62_444_175_504
    assert source["tensor_count"] == 9_703
    assert source["base_layer_count"] == 47
    assert source["mtp_layer_index"] == 47
    assert source["mtp_admitted_not_executed"] is True
    assert source["full_hash_authenticated"] is True
    assert source["teacher_forced_full_model"] is True

    assert (
        evidence["candidate"]["runtime_code_hash"]
        == evidence["native_execution"]["native_binary_hash"]
        == "sha256:9df828424f1e93082218e5b94d4a8caefbd1da472f9dc7074ed1e9a652eea995"
    )
    assert evidence["metrics"] == {
        "hidden_cosine_similarity": 0.9999629578504667,
        "hidden_normalized_rmse": 0.009282564453363888,
        "route_agreement": None,
        "top_token_agreement": 1.0,
    }
    assert evidence["thresholds"] == {
        "maximum_hidden_normalized_rmse": 0.1,
        "minimum_hidden_cosine_similarity": 0.995,
        "minimum_top_token_agreement": 0.95,
    }
    assert evidence["teacher_forced"] == {
        "candidate_top_token_ids": [3_422, 712, 198, 99_381, 220, 220, 271, 271],
        "input_token_ids": [117_736, 67_585, 17_434, 122_163, 72_012, 21_861, 126_590, 76_439],
        "reference_top_token_ids": [3_422, 712, 198, 99_381, 220, 220, 271, 271],
    }
    assert evidence["gates"] == {
        "complete_model_gate_passed": True,
        "hidden_state_gate_passed": True,
        "logit_gate_passed": True,
        "qualifies_precision_policy": False,
    }
    assert evidence["resources"] == {
        "full_model_materialized": False,
        "reference_checkpoint_tensor_bytes": 32_768,
        "reference_streaming_layer_source_payload_bound_bytes": 1_270_622_976,
        "torch_threads": 8,
    }
    assert evidence["native_execution"]["cache_heap_bytes"] == 7_700_480
    assert evidence["native_execution"]["scratch_heap_bytes"] == 2_839_888
    assert evidence["native_execution"]["raw_observation_bytes"] == 10_043_392
