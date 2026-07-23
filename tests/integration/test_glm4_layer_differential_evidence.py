import json
from pathlib import Path

from jsonschema.validators import Draft202012Validator


def test_official_layer_evidence_is_authenticated_but_not_qualified() -> None:
    root = Path(__file__).parents[2]
    evidence = json.loads(
        (root / "docs" / "evidence" / "glm47_layer1_bf16_differential.json").read_bytes()
    )
    schema = json.loads((root / "schemas" / "glm4-layer-differential.schema.json").read_bytes())
    Draft202012Validator(schema).validate(evidence)

    assert evidence["source"] == {
        "architecture_hash": (
            "sha256:dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb"
        ),
        "layer_index": 1,
        "logit_readout": {
            "final_norm_tensor": "model.norm.weight",
            "kind": "isolated_final_norm_lm_head",
            "lm_head_tensor": "lm_head.weight",
            "shard_name": "model-00047-of-00048.safetensors",
            "shard_sha256": (
                "sha256:1bcc5d06065d2a564894657945ccfe9411762421c2c60acf91de31050cd4d84d"
            ),
            "shard_size_bytes": 2_539_429_936,
            "teacher_forced_full_model": False,
        },
        "repository": "zai-org/GLM-4.7-Flash",
        "revision": "7dd20894a642a0aa287e9827cb1a1f7f91386b67",
        "shard_name": "model-00002-of-00048.safetensors",
        "shard_sha256": ("sha256:8c51e2434efe609cbe652014a924e088a5ea97be35ca29cfa893a1a9a90304b1"),
        "shard_size_bytes": 1_270_648_128,
        "shard_source_root": (
            "sha256:1b297252e41f3e7e2fb6cd0f52dfa43e4d036b59bfe96be5688af1938d4be45f"
        ),
        "source_index_hash": (
            "sha256:91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
        ),
        "tensor_count": 206,
        "tensor_payload_bytes": 1_270_622_976,
    }
    assert evidence["status"] == "blocked"
    assert evidence["metrics"]["hidden_cosine_similarity"] >= 0.995
    assert evidence["metrics"]["hidden_normalized_rmse"] <= 0.10
    assert evidence["metrics"]["route_agreement"] == 1.0
    assert evidence["metrics"]["top_token_agreement"] == 1.0
    assert evidence["gates"] == {
        "full_layer_gate_passed": True,
        "hidden_state_gate_passed": True,
        "logit_gate_passed": True,
        "qualifies_precision_policy": False,
    }
    assert evidence["blockers"] == [
        "candidate runtime is the AMS Python semantic oracle, not native ams-core execution",
        "isolated final-head readout is not a complete-model teacher-forced execution",
    ]
