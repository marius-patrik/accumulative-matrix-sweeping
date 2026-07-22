import json
from collections.abc import Sequence

import pytest

from ams.errors import AmsError, ErrorCode
from ams.integrations import parse_glm_moe_dsa_architecture
from ams.ops import (
    GlmReferenceTensor,
    GlmReferenceWeights,
    GlmWeightAccess,
    run_glm_moe_dsa_prefill_reference,
)


def tiny_architecture():
    payload = json.dumps(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "attention_bias": False,
            "dtype": "bfloat16",
            "first_k_dense_replace": 1,
            "hidden_act": "silu",
            "hidden_size": 8,
            "index_head_dim": 4,
            "index_n_heads": 2,
            "index_share_for_mtp_iteration": True,
            "index_topk": 2,
            "indexer_types": ["full", "shared"],
            "intermediate_size": 12,
            "kv_lora_rank": 4,
            "max_position_embeddings": 32,
            "mlp_layer_types": ["dense", "sparse"],
            "model_type": "glm_moe_dsa",
            "moe_intermediate_size": 6,
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
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "routed_scaling_factor": 1.5,
            "scoring_func": "sigmoid",
            "tie_word_embeddings": False,
            "topk_group": 1,
            "topk_method": "noaux_tc",
            "v_head_dim": 3,
            "vocab_size": 16,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return parse_glm_moe_dsa_architecture(payload)


def fixture_values(count: int, salt: int, *, center: float = 0.0, scale: float = 0.08):
    return tuple(
        center + (((index * (salt * 2 + 1) + salt) % 19) - 9) * scale for index in range(count)
    )


def matrix(tensors, name: str, rows: int, columns: int, salt: int) -> None:
    tensors[name] = GlmReferenceTensor((rows, columns), fixture_values(rows * columns, salt))


def vector(tensors, name: str, length: int, salt: int, *, norm: bool = False) -> None:
    if norm:
        values = fixture_values(length, salt, center=1.0, scale=0.01)
    else:
        values = fixture_values(length, salt)
    tensors[name] = GlmReferenceTensor((length,), values)


def fixture_weights(architecture):
    tensors = {}
    matrix(
        tensors, "model.embed_tokens.weight", architecture.vocab_size, architecture.hidden_size, 1
    )
    vector(tensors, "model.norm.weight", architecture.hidden_size, 2, norm=True)
    matrix(tensors, "lm_head.weight", architecture.vocab_size, architecture.hidden_size, 3)
    for layer_index in range(architecture.num_hidden_layers):
        prefix = f"model.layers.{layer_index}"
        vector(
            tensors,
            f"{prefix}.input_layernorm.weight",
            architecture.hidden_size,
            10 + layer_index,
            norm=True,
        )
        vector(
            tensors,
            f"{prefix}.post_attention_layernorm.weight",
            architecture.hidden_size,
            20 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.q_a_proj.weight",
            architecture.q_lora_rank,
            architecture.hidden_size,
            30 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.q_a_layernorm.weight",
            architecture.q_lora_rank,
            40 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.q_b_proj.weight",
            architecture.num_attention_heads * architecture.qk_head_dim,
            architecture.q_lora_rank,
            50 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight",
            architecture.kv_lora_rank + architecture.qk_rope_head_dim,
            architecture.hidden_size,
            60 + layer_index,
        )
        vector(
            tensors,
            f"{prefix}.self_attn.kv_a_layernorm.weight",
            architecture.kv_lora_rank,
            70 + layer_index,
            norm=True,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.kv_b_proj.weight",
            architecture.num_attention_heads
            * (architecture.qk_nope_head_dim + architecture.v_head_dim),
            architecture.kv_lora_rank,
            80 + layer_index,
        )
        matrix(
            tensors,
            f"{prefix}.self_attn.o_proj.weight",
            architecture.hidden_size,
            architecture.num_attention_heads * architecture.v_head_dim,
            90 + layer_index,
        )
        if architecture.indexer_types[layer_index] == "full":
            matrix(
                tensors,
                f"{prefix}.self_attn.indexer.wq_b.weight",
                architecture.index_n_heads * architecture.index_head_dim,
                architecture.q_lora_rank,
                100 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.self_attn.indexer.wk.weight",
                architecture.index_head_dim,
                architecture.hidden_size,
                110 + layer_index,
            )
            vector(
                tensors,
                f"{prefix}.self_attn.indexer.k_norm.weight",
                architecture.index_head_dim,
                120 + layer_index,
                norm=True,
            )
            vector(
                tensors,
                f"{prefix}.self_attn.indexer.k_norm.bias",
                architecture.index_head_dim,
                130 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.self_attn.indexer.weights_proj.weight",
                architecture.index_n_heads,
                architecture.hidden_size,
                140 + layer_index,
            )
        if architecture.mlp_layer_types[layer_index] == "dense":
            matrix(
                tensors,
                f"{prefix}.mlp.gate_proj.weight",
                architecture.intermediate_size,
                architecture.hidden_size,
                150 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.mlp.up_proj.weight",
                architecture.intermediate_size,
                architecture.hidden_size,
                160 + layer_index,
            )
            matrix(
                tensors,
                f"{prefix}.mlp.down_proj.weight",
                architecture.hidden_size,
                architecture.intermediate_size,
                170 + layer_index,
            )
        else:
            matrix(
                tensors,
                f"{prefix}.mlp.gate.weight",
                architecture.n_routed_experts,
                architecture.hidden_size,
                180 + layer_index,
            )
            tensors[f"{prefix}.mlp.gate.e_score_correction_bias"] = GlmReferenceTensor(
                (architecture.n_routed_experts,), (10.0, -10.0)
            )
            for expert_index in range(architecture.n_routed_experts):
                expert = f"{prefix}.mlp.experts.{expert_index}"
                matrix(
                    tensors,
                    f"{expert}.gate_proj.weight",
                    architecture.moe_intermediate_size,
                    architecture.hidden_size,
                    190 + expert_index,
                )
                matrix(
                    tensors,
                    f"{expert}.up_proj.weight",
                    architecture.moe_intermediate_size,
                    architecture.hidden_size,
                    200 + expert_index,
                )
                matrix(
                    tensors,
                    f"{expert}.down_proj.weight",
                    architecture.hidden_size,
                    architecture.moe_intermediate_size,
                    210 + expert_index,
                )
            shared = f"{prefix}.mlp.shared_experts"
            shared_size = architecture.moe_intermediate_size * architecture.n_shared_experts
            matrix(
                tensors, f"{shared}.gate_proj.weight", shared_size, architecture.hidden_size, 220
            )
            matrix(tensors, f"{shared}.up_proj.weight", shared_size, architecture.hidden_size, 221)
            matrix(
                tensors, f"{shared}.down_proj.weight", architecture.hidden_size, shared_size, 222
            )
    return GlmReferenceWeights(tensors)


class RejectingExpertWeights(GlmWeightAccess):
    def __init__(self, inner: GlmWeightAccess, rejected_fragment: str) -> None:
        self.inner = inner
        self.rejected_fragment = rejected_fragment

    def _check(self, tensor_name: str) -> None:
        if self.rejected_fragment in tensor_name:
            raise AssertionError(f"unselected expert was fetched: {tensor_name}")

    def vector(self, tensor_name: str, length: int) -> tuple[float, ...]:
        self._check(tensor_name)
        return self.inner.vector(tensor_name, length)

    def embedding(self, tensor_name: str, index: int, width: int) -> tuple[float, ...]:
        self._check(tensor_name)
        return self.inner.embedding(tensor_name, index, width)

    def linear(self, tensor_name: str, values: Sequence[float], rows: int) -> tuple[float, ...]:
        self._check(tensor_name)
        return self.inner.linear(tensor_name, values, rows)


def test_mini_glm_prefill_executes_dense_indexshare_and_sparse_experts() -> None:
    architecture = tiny_architecture()
    weights = RejectingExpertWeights(fixture_weights(architecture), ".mlp.experts.1.")
    first = run_glm_moe_dsa_prefill_reference(architecture, weights, (1, 5, 9))
    second = run_glm_moe_dsa_prefill_reference(architecture, weights, (1, 5, 9))
    assert first == second
    assert len(first.logits) == 3
    assert all(len(logits) == architecture.vocab_size for logits in first.logits)
    assert first.layers[0].indexer_type == "full"
    assert first.layers[1].indexer_type == "shared"
    assert first.layers[1].dsa_indices == first.layers[0].dsa_indices
    assert first.layers[0].dsa_indices[0] == (0,)
    assert all(len(indices) <= architecture.index_topk for indices in first.layers[0].dsa_indices)
    assert first.layers[0].expert_routing == ()
    assert all(route.expert_indices == (0,) for route in first.layers[1].expert_routing)
    assert all(all(value == value for value in logits) for logits in first.logits)
    assert [max(range(len(logits)), key=logits.__getitem__) for logits in first.logits] == [
        14,
        14,
        14,
    ]
    assert tuple(logits[0] for logits in first.logits) == pytest.approx(
        (1.1646461163222221, 0.8460455459844063, 0.8805485467331877),
        rel=0,
        abs=1e-14,
    )


def test_mini_glm_mtp_remains_an_explicit_unsupported_feature() -> None:
    architecture = tiny_architecture()
    with pytest.raises(AmsError) as caught:
        run_glm_moe_dsa_prefill_reference(
            architecture,
            fixture_weights(architecture),
            (1,),
            enable_mtp=True,
        )
    assert caught.value.code is ErrorCode.UNSUPPORTED_OP
