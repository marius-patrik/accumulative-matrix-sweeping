"""Authenticate and differentially execute the pinned official GLM-4.7 sparse layer."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.descriptors import DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4LayerObservation,
    Glm4MoeLiteTensorRole,
    HuggingFaceShardSource,
    build_huggingface_shard_catalog,
    compare_glm4_layer_observations,
    expected_glm4_moe_lite_tensor_shape,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    validate_glm4_moe_lite_tensor_inventory,
)
from ams.ops import GlmWeightAccess, run_glm4_moe_lite_sparse_layer_reference
from ams.storage import FileRangeStore
from ams.version import __version__ as ams_version

_REPOSITORY = "zai-org/GLM-4.7-Flash"
_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
_LAYER_INDEX = 1
_SHARD_NAME = "model-00002-of-00048.safetensors"
_SHARD_BYTES = 1_270_648_128
_SHARD_SHA256 = "8c51e2434efe609cbe652014a924e088a5ea97be35ca29cfa893a1a9a90304b1"
_CONFIG_SHA256 = "dc9b97c7c9bed726a2e6939da4234d5c43abb3edec8812068c9a1af1dbc13acb"
_INDEX_SHA256 = "91e6e95ca21700f50904a680c8c4212f5aa16dc7c10a013f01c906957c889791"
_EXPECTED_TENSOR_COUNT = 206
_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_EXPECTED_TRANSFORMERS_VERSION = "5.12.0"
_EXPECTED_SAFETENSORS_VERSION = "0.8.0"
_EXPECTED_TORCH_BASE_VERSION = "2.13.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-hash the pinned GLM-4.7 layer-1 shard, run the official Transformers "
            "BF16 layer and the independent AMS semantic oracle on one exact corpus, and "
            "emit non-qualifying differential evidence"
        )
    )
    parser.add_argument("asset_root", type=Path, help="Pinned config and full shard index")
    parser.add_argument("shard", type=Path, help=f"Exact local {_SHARD_NAME}")
    parser.add_argument("--samples", type=int, default=4, help="Deterministic token positions")
    parser.add_argument("--buffer-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Authenticate source assets without loading PyTorch weights",
    )
    return parser


def _read_bounded_regular(path: Path, maximum_bytes: int, *, label: str) -> bytes:
    try:
        if path.is_symlink():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is a symbolic link")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is not a regular file")
        size = resolved.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                f"{label} size is outside the reviewed bound",
                evidence={"size_bytes": size, "maximum_bytes": maximum_bytes},
            )
        return resolved.read_bytes()
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, f"{label} could not be read") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _open_authenticated_layer_shard(
    shard_path: Path,
    *,
    buffer_bytes: int,
) -> FileRangeStore:
    """Open the exact pinned layer shard only after hashing its complete payload."""

    if buffer_bytes <= 0 or buffer_bytes > 64 * 1024 * 1024:
        raise AmsError(ErrorCode.PLAN_INVALID, "verification buffer is outside the reviewed bound")
    if shard_path.name != _SHARD_NAME:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM-4.7 layer shard name is not pinned")
    try:
        if shard_path.is_symlink():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "GLM-4.7 layer shard is a symbolic link",
            )
        resolved_shard = shard_path.resolve(strict=True)
        if not resolved_shard.is_file() or resolved_shard.stat().st_size != _SHARD_BYTES:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "GLM-4.7 layer shard is not the pinned regular-file size",
            )
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, "GLM-4.7 layer shard metadata failed") from exc

    descriptor = StorageObject(
        object_id=f"hf:{_SHARD_NAME}",
        uri="file:local-pinned-glm47-layer1",
        size_bytes=_SHARD_BYTES,
        alignment_bytes=1,
        content_hash=f"sha256:{_SHARD_SHA256}",
    )
    reader = FileRangeStore(resolved_shard, descriptor)
    reader.verify_content_hash(buffer_bytes=buffer_bytes)
    return reader


def _admit_assets(asset_root: Path, shard_path: Path, *, buffer_bytes: int):
    config_payload = _read_bounded_regular(
        asset_root / "config.json",
        _MAX_CONFIG_BYTES,
        label="GLM-4.7 config",
    )
    index_payload = _read_bounded_regular(
        asset_root / "model.safetensors.index.json",
        _MAX_INDEX_BYTES,
        label="GLM-4.7 shard index",
    )
    if _sha256(config_payload) != _CONFIG_SHA256:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 config hash is not pinned")
    if _sha256(index_payload) != _INDEX_SHA256:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 shard-index hash is not pinned")

    architecture = parse_glm4_moe_lite_architecture(config_payload)
    index = parse_huggingface_shard_index(index_payload)
    inventory = validate_glm4_moe_lite_tensor_inventory(architecture, index)
    reader = _open_authenticated_layer_shard(shard_path, buffer_bytes=buffer_bytes)
    source = HuggingFaceShardSource(
        shard_name=_SHARD_NAME,
        object_id=reader.descriptor.object_id,
        content_hash=reader.descriptor.content_hash,
        reader=reader,
    )
    catalog = build_huggingface_shard_catalog(index, source, buffer_bytes=buffer_bytes)

    expected_slots = tuple(slot for slot in inventory.slots if slot.layer_index == _LAYER_INDEX)
    expected_by_name = {slot.tensor_name: slot for slot in expected_slots}
    observed_by_name = {tensor.tensor_name: tensor for tensor in catalog.tensors}
    if (
        len(expected_slots) != _EXPECTED_TENSOR_COUNT
        or len(observed_by_name) != _EXPECTED_TENSOR_COUNT
        or set(expected_by_name) != set(observed_by_name)
    ):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "official GLM-4.7 shard is not exactly one complete sparse layer",
        )
    for tensor_name, slot in expected_by_name.items():
        tensor = observed_by_name[tensor_name]
        expected_dtype = (
            DType.FLOAT32
            if slot.role is Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS
            else DType.BFLOAT16
        )
        expected_source_dtype = (
            "F32" if slot.role is Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS else "BF16"
        )
        if (
            tensor.shape != expected_glm4_moe_lite_tensor_shape(architecture, slot)
            or tensor.dtype is not expected_dtype
            or tensor.source_dtype != expected_source_dtype
        ):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"official GLM-4.7 layer tensor semantics drifted: {tensor_name}",
            )
    return architecture, catalog, observed_by_name


def _source_evidence(architecture, catalog) -> dict[str, object]:
    return {
        "repository": _REPOSITORY,
        "revision": _REVISION,
        "architecture_hash": architecture.content_hash,
        "source_index_hash": catalog.index_content_hash,
        "layer_index": _LAYER_INDEX,
        "shard_name": _SHARD_NAME,
        "shard_size_bytes": _SHARD_BYTES,
        "shard_sha256": f"sha256:{_SHARD_SHA256}",
        "shard_source_root": catalog.source_root,
        "tensor_count": len(catalog.tensors),
        "tensor_payload_bytes": catalog.total_size,
    }


def _deterministic_input(
    sample_count: int, hidden_size: int, torch
) -> tuple[tuple[tuple[float, ...], ...], str]:
    if isinstance(sample_count, bool) or not 1 <= sample_count <= 16:
        raise AmsError(ErrorCode.PLAN_INVALID, "sample count must be in [1, 16]")
    source = [
        [
            ((((sample + 1) * 104_729 + (column + 1) * 13_007) % 2_003) - 1_001) / 1_001.0
            for column in range(hidden_size)
        ]
        for sample in range(sample_count)
    ]
    rounded = torch.tensor(source, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    values = tuple(tuple(float(value) for value in row) for row in rounded.tolist())
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"generator": "ams.glm47-layer-input.v1", "values": values}))
    return values, "sha256:" + digest.hexdigest()


def _module_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((path.resolve(strict=True) for path in paths), key=str):
        payload = path.read_bytes()
        digest.update(path.name.encode())
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def _require_reference_toolchain():
    try:
        import safetensors
        import torch
        import transformers
        from safetensors import safe_open
        from transformers import Glm4MoeLiteConfig
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteDecoderLayer,
            Glm4MoeLiteRotaryEmbedding,
        )
    except ImportError as exc:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "official layer differential requires torch, transformers, and safetensors",
        ) from exc
    if transformers.__version__ != _EXPECTED_TRANSFORMERS_VERSION:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "Transformers version differs from the pinned reference",
        )
    if safetensors.__version__ != _EXPECTED_SAFETENSORS_VERSION:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "safetensors version differs from the pinned reference",
        )
    if torch.__version__.split("+", 1)[0] != _EXPECTED_TORCH_BASE_VERSION:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "PyTorch base version differs from the pinned reference",
        )
    return (
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRotaryEmbedding,
    )


def _load_official_layer(
    config_path: Path,
    shard_path: Path,
    torch,
    safe_open,
    Glm4MoeLiteConfig,
    Glm4MoeLiteDecoderLayer,
):
    config = Glm4MoeLiteConfig.from_json_file(config_path)
    config._attn_implementation = "eager"
    prefix = f"model.layers.{_LAYER_INDEX}."
    with torch.device("meta"):
        layer = Glm4MoeLiteDecoderLayer(config, _LAYER_INDEX)
    parameter_names = tuple(layer.state_dict())
    state: dict[str, Any] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for local_name in parameter_names:
            if local_name in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}:
                continue
            state[local_name] = handle.get_tensor(prefix + local_name)

        expert_count = config.n_routed_experts
        intermediate = config.moe_intermediate_size
        hidden = config.hidden_size
        gate_up = torch.empty(
            (expert_count, intermediate * 2, hidden),
            dtype=torch.bfloat16,
        )
        down = torch.empty(
            (expert_count, hidden, intermediate),
            dtype=torch.bfloat16,
        )
        for expert_index in range(expert_count):
            expert_prefix = f"{prefix}mlp.experts.{expert_index}."
            gate_up[expert_index, :intermediate].copy_(
                handle.get_tensor(expert_prefix + "gate_proj.weight")
            )
            gate_up[expert_index, intermediate:].copy_(
                handle.get_tensor(expert_prefix + "up_proj.weight")
            )
            down[expert_index].copy_(handle.get_tensor(expert_prefix + "down_proj.weight"))
        state["mlp.experts.gate_up_proj"] = gate_up
        state["mlp.experts.down_proj"] = down
    if set(state) != set(parameter_names):
        raise AmsError(
            ErrorCode.INTERNAL_INVARIANT,
            "official Transformers layer mapping is incomplete",
        )
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval()
    return config, layer


def _route_rows(indices, weights) -> tuple[dict[int, float], ...]:
    return tuple(
        {int(expert): float(weight) for expert, weight in zip(index_row, weight_row, strict=True)}
        for index_row, weight_row in zip(indices.tolist(), weights.tolist(), strict=True)
    )


def _run_official_reference(
    asset_root: Path,
    shard_path: Path,
    input_values: tuple[tuple[float, ...], ...],
    input_hash: str,
    torch,
    transformers,
    safetensors,
    safe_open,
    Glm4MoeLiteConfig,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteRotaryEmbedding,
):
    config, layer = _load_official_layer(
        asset_root / "config.json",
        shard_path,
        torch,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
    )
    hidden = torch.tensor(input_values, dtype=torch.bfloat16).unsqueeze(0)
    sample_count = len(input_values)
    positions = torch.arange(sample_count, dtype=torch.long).unsqueeze(0)
    rotary = Glm4MoeLiteRotaryEmbedding(config)
    position_embeddings = rotary(hidden, positions)
    causal = torch.zeros((1, 1, sample_count, sample_count), dtype=hidden.dtype)
    causal.masked_fill_(
        torch.triu(torch.ones((sample_count, sample_count), dtype=torch.bool), diagonal=1),
        torch.finfo(hidden.dtype).min,
    )
    captured: list[Any] = []

    def capture_moe_input(_module, arguments):
        captured.append(arguments[0].detach())

    hook = layer.mlp.register_forward_pre_hook(capture_moe_input)
    with torch.no_grad():
        output = layer(
            hidden,
            attention_mask=causal,
            position_ids=positions,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    hook.remove()
    if len(captured) != 1:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "official MoE input capture failed")
    with torch.no_grad():
        router_logits = layer.mlp.gate(captured[0])
        route_indices, route_weights = layer.mlp.route_tokens_to_experts(router_logits)
    rows = tuple(tuple(float(value) for value in row) for row in output[0].float().tolist())
    route_rows = _route_rows(route_indices, route_weights)
    modeling_path = Path(inspect.getsourcefile(type(layer)) or "")
    observation = Glm4LayerObservation(
        runtime_id="transformers.glm4_moe_lite",
        runtime_version=(
            f"transformers/{transformers.__version__};"
            f"torch/{torch.__version__};safetensors/{safetensors.__version__}"
        ),
        runtime_code_hash=_module_hash((modeling_path, Path(__file__))),
        input_hash=input_hash,
        sample_ids=tuple(f"position-{index}" for index in range(sample_count)),
        hidden_states=rows,
        logits=None,
    )
    return observation, route_rows


class _TorchSafeTensorWeightAccess(GlmWeightAccess):
    def __init__(self, handle, tensor_by_name: dict[str, object], torch) -> None:
        self._handle = handle
        self._tensor_by_name = tensor_by_name
        self._torch = torch

    def _tensor(self, tensor_name: str):
        if tensor_name not in self._tensor_by_name:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                f"required official layer tensor is absent: {tensor_name}",
            )
        return self._handle.get_tensor(tensor_name)

    def vector(self, tensor_name: str, length: int) -> tuple[float, ...]:
        tensor = self._tensor(tensor_name)
        if tuple(tensor.shape) != (length,):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "official vector shape is invalid")
        return tuple(float(value) for value in tensor.to(self._torch.float32).tolist())

    def embedding(self, tensor_name: str, index: int, width: int) -> tuple[float, ...]:
        raise AmsError(
            ErrorCode.UNSUPPORTED_OP,
            f"layer-only weight access cannot read embeddings: {tensor_name}/{index}/{width}",
        )

    def linear(
        self,
        tensor_name: str,
        values: Sequence[float],
        rows: int,
    ) -> tuple[float, ...]:
        tensor = self._tensor(tensor_name)
        if tuple(tensor.shape) != (rows, len(values)):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "official matrix shape is invalid")
        with self._torch.no_grad():
            source = tensor.to(self._torch.float32)
            vector = self._torch.tensor(values, dtype=self._torch.float32)
            output = self._torch.mv(source, vector)
        return tuple(float(value) for value in output.tolist())


def _run_ams_oracle(
    architecture,
    shard_path: Path,
    tensor_by_name: dict[str, object],
    input_values: tuple[tuple[float, ...], ...],
    input_hash: str,
    torch,
    safe_open,
):
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weights = _TorchSafeTensorWeightAccess(handle, tensor_by_name, torch)
        result = run_glm4_moe_lite_sparse_layer_reference(
            architecture,
            weights,
            _LAYER_INDEX,
            input_values,
        )
    module_root = Path(__file__).parents[1] / "src" / "ams"
    code_hash = _module_hash(
        (
            Path(__file__),
            module_root / "ops" / "glm_moe_dsa.py",
            module_root / "ops" / "glm_moe_dsa_model.py",
        )
    )
    observation = Glm4LayerObservation(
        runtime_id="ams.python.glm4_sparse_layer_reference",
        runtime_version=f"ams/{ams_version};torch/{torch.__version__}",
        runtime_code_hash=code_hash,
        input_hash=input_hash,
        sample_ids=tuple(f"position-{index}" for index in range(len(input_values))),
        hidden_states=result.hidden_states,
        logits=None,
    )
    routes = tuple(
        {
            int(expert): float(weight)
            for expert, weight in zip(
                route.expert_indices,
                route.expert_weights,
                strict=True,
            )
        }
        for route in result.expert_routing
    )
    return observation, routes


def _route_agreement(
    reference: tuple[dict[int, float], ...],
    candidate: tuple[dict[int, float], ...],
) -> float:
    if len(reference) != len(candidate) or not reference:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "route sample counts differ")
    matches = 0
    for expected, actual in zip(reference, candidate, strict=True):
        if set(expected) != set(actual):
            continue
        if all(
            math.isclose(expected[key], actual[key], rel_tol=2e-3, abs_tol=2e-3) for key in expected
        ):
            matches += 1
    return matches / len(reference)


def _preflight_output(source: dict[str, object]) -> dict[str, object]:
    return {
        "schema_id": "ams.glm4-layer-source-preflight.v1",
        "status": "blocked",
        "source": source,
        "gates": {
            "source_admitted": True,
            "hidden_state_gate_passed": False,
            "logit_gate_passed": False,
            "full_layer_gate_passed": False,
            "qualifies_precision_policy": False,
        },
        "blockers": [
            "official and AMS layer observations were not requested",
            "native official-layer observation is absent",
            "teacher-forced logits are absent",
        ],
    }


def main() -> int:
    arguments = _parser().parse_args()
    architecture, catalog, tensor_by_name = _admit_assets(
        arguments.asset_root,
        arguments.shard,
        buffer_bytes=arguments.buffer_bytes,
    )
    source = _source_evidence(architecture, catalog)
    if arguments.preflight_only:
        print(json.dumps(_preflight_output(source), indent=2, sort_keys=True))
        return 2

    (
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRotaryEmbedding,
    ) = _require_reference_toolchain()
    input_values, input_hash = _deterministic_input(
        arguments.samples,
        architecture.hidden_size,
        torch,
    )
    reference, reference_routes = _run_official_reference(
        arguments.asset_root,
        arguments.shard,
        input_values,
        input_hash,
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRotaryEmbedding,
    )
    candidate, candidate_routes = _run_ams_oracle(
        architecture,
        arguments.shard,
        tensor_by_name,
        input_values,
        input_hash,
        torch,
        safe_open,
    )
    route_agreement = _route_agreement(reference_routes, candidate_routes)
    comparison = compare_glm4_layer_observations(
        reference,
        candidate,
        expected_hidden_size=architecture.hidden_size,
        expected_vocabulary_size=architecture.vocab_size,
        route_agreement=route_agreement,
        blockers=(
            "candidate runtime is the AMS Python semantic oracle, not native ams-core execution",
            (
                "official LM-head shard model-00047-of-00048.safetensors is absent; "
                "teacher-forced logits were not produced"
            ),
        ),
    )
    output = comparison.to_dict()
    output["source"] = source
    output["input"] = {
        "schema_id": "ams.glm47-layer-input.v1",
        "content_hash": input_hash,
        "sample_count": len(input_values),
        "hidden_size": architecture.hidden_size,
    }
    output["reference"] = {
        "runtime_id": reference.runtime_id,
        "runtime_version": reference.runtime_version,
        "runtime_code_hash": reference.runtime_code_hash,
        "observation_hash": reference.observation_hash,
        "hidden_state_hash": reference.hidden_state_hash,
        "logits_hash": reference.logits_hash,
    }
    output["candidate"] = {
        "runtime_id": candidate.runtime_id,
        "runtime_version": candidate.runtime_version,
        "runtime_code_hash": candidate.runtime_code_hash,
        "observation_hash": candidate.observation_hash,
        "hidden_state_hash": candidate.hidden_state_hash,
        "logits_hash": candidate.logits_hash,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if comparison.full_layer_gate_passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AmsError as error:
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
