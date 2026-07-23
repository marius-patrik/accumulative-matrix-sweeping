"""Execute an authenticated two-layer GLM-4.7 probe through native ams-core."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4LayerDifferentialStatus,
    Glm4LayerObservation,
    Glm4MoeLiteArchitecture,
    HuggingFaceShardSource,
    build_huggingface_shard_catalog,
    compare_glm4_layer_observations,
    expected_glm4_moe_lite_tensor_shape,
    expected_glm4_moe_lite_tensor_slots,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    validate_glm4_moe_lite_tensor_inventory,
)
from ams.ops import (
    Glm4NativeBindingPlan,
    Glm4NativeStorageBinding,
    Glm4NativeTensorBinding,
    serialize_glm4_native_binding_plan,
)
from ams.version import __version__ as ams_version

if __package__:
    from .verify_glm4_official_layer import (
        _CONFIG_SHA256,
        _EXPECTED_SAFETENSORS_VERSION,
        _EXPECTED_TORCH_BASE_VERSION,
        _EXPECTED_TRANSFORMERS_VERSION,
        _HEAD_SHARD_BYTES,
        _HEAD_SHARD_NAME,
        _HEAD_SHARD_SHA256,
        _INDEX_SHA256,
        _LAYER_INDEX,
        _MAX_CONFIG_BYTES,
        _MAX_INDEX_BYTES,
        _SHARD_BYTES,
        _SHARD_NAME,
        _SHARD_SHA256,
        _module_hash,
        _open_authenticated_shard,
        _read_bounded_regular,
        _sha256,
        _source_evidence,
    )
else:
    from verify_glm4_official_layer import (
        _CONFIG_SHA256,
        _EXPECTED_SAFETENSORS_VERSION,
        _EXPECTED_TORCH_BASE_VERSION,
        _EXPECTED_TRANSFORMERS_VERSION,
        _HEAD_SHARD_BYTES,
        _HEAD_SHARD_NAME,
        _HEAD_SHARD_SHA256,
        _INDEX_SHA256,
        _LAYER_INDEX,
        _MAX_CONFIG_BYTES,
        _MAX_INDEX_BYTES,
        _SHARD_BYTES,
        _SHARD_NAME,
        _SHARD_SHA256,
        _module_hash,
        _open_authenticated_shard,
        _read_bounded_regular,
        _sha256,
        _source_evidence,
    )

_EMBEDDING_SHARD_NAME = "model-00001-of-00048.safetensors"
_EMBEDDING_SHARD_BYTES = 1_438_134_344
_EMBEDDING_SHARD_SHA256 = "90abe0d075755853145c96906a1300f57c167fcc9aa67221239b448abf54933c"
_MTP_SHARD_NAME = "model-00048-of-00048.safetensors"
_MTP_SHARD_BYTES = 1_287_438_264
_MTP_SHARD_SHA256 = "35fff90a30ca808d86dc24f9e3eda119832ab69fb1f88ae4cccfbf0e5ee409a1"
_SOURCE_MTP_LAYER = 47
_BOUND_MTP_LAYER = 2
_LINEAR_ARENA_BYTES = 1024 * 1024
_NATIVE_TIMEOUT_SECONDS = 300


@dataclass(frozen=True, slots=True)
class _ShardPin:
    name: str
    size_bytes: int
    sha256: str
    label: str

    @property
    def object_id(self) -> str:
        return f"hf:{self.name}"


_EMBEDDING_PIN = _ShardPin(
    _EMBEDDING_SHARD_NAME,
    _EMBEDDING_SHARD_BYTES,
    _EMBEDDING_SHARD_SHA256,
    "GLM-4.7 embedding/dense-layer shard",
)
_LAYER_PIN = _ShardPin(
    _SHARD_NAME,
    _SHARD_BYTES,
    _SHARD_SHA256,
    "GLM-4.7 sparse-layer shard",
)
_HEAD_PIN = _ShardPin(
    _HEAD_SHARD_NAME,
    _HEAD_SHARD_BYTES,
    _HEAD_SHARD_SHA256,
    "GLM-4.7 final-norm/LM-head shard",
)
_MTP_PIN = _ShardPin(
    _MTP_SHARD_NAME,
    _MTP_SHARD_BYTES,
    _MTP_SHARD_SHA256,
    "GLM-4.7 MTP shard",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate the pinned GLM-4.7 source shards, execute layers 0 and 1 plus the "
            "final readout through native ams-core, and compare with Transformers BF16"
        )
    )
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("embedding_shard", type=Path)
    parser.add_argument("layer_shard", type=Path)
    parser.add_argument("head_shard", type=Path)
    parser.add_argument("mtp_shard", type=Path)
    parser.add_argument("native_binary", type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--buffer-bytes", type=int, default=4 * 1024 * 1024)
    return parser


def _authenticate_catalog(index, path: Path, pin: _ShardPin, *, buffer_bytes: int):
    reader = _open_authenticated_shard(
        path,
        expected_name=pin.name,
        expected_size_bytes=pin.size_bytes,
        expected_sha256=pin.sha256,
        object_id=pin.object_id,
        label=pin.label,
        buffer_bytes=buffer_bytes,
    )
    source = HuggingFaceShardSource(
        shard_name=pin.name,
        object_id=pin.object_id,
        content_hash=f"sha256:{pin.sha256}",
        reader=reader,
    )
    return reader, build_huggingface_shard_catalog(index, source, buffer_bytes=buffer_bytes)


def _load_source(
    asset_root: Path,
    shard_paths: dict[str, Path],
    *,
    buffer_bytes: int,
):
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
    validate_glm4_moe_lite_tensor_inventory(architecture, index)
    readers = {}
    tensors = {}
    catalogs = {}
    for pin in (_EMBEDDING_PIN, _LAYER_PIN, _HEAD_PIN, _MTP_PIN):
        reader, catalog = _authenticate_catalog(
            index,
            shard_paths[pin.name],
            pin,
            buffer_bytes=buffer_bytes,
        )
        readers[pin.object_id] = reader
        catalogs[pin.name] = catalog
        for tensor in catalog.tensors:
            if tensor.tensor_name in tensors:
                raise AmsError(
                    ErrorCode.INTERNAL_INVARIANT,
                    "native probe source tensor appears in multiple admitted shards",
                )
            tensors[tensor.tensor_name] = tensor
    return architecture, index, readers, tensors, catalogs


def _probe_architecture(
    source: Glm4MoeLiteArchitecture,
) -> tuple[Glm4MoeLiteArchitecture, dict[str, object]]:
    identity = {
        "schema_id": "ams.glm47-two-layer-native-probe.v1",
        "source_architecture_hash": source.content_hash,
        "executable_source_layers": [0, 1],
        "source_mtp_layer": _SOURCE_MTP_LAYER,
        "bound_mtp_layer": _BOUND_MTP_LAYER,
        "complete_model": False,
    }
    content_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return (
        replace(
            source,
            content_hash=content_hash,
            num_hidden_layers=2,
            num_nextn_predict_layers=1,
            mlp_layer_types=("dense", "sparse"),
        ),
        identity,
    )


def _source_tensor_name(binding_name: str, *, mtp: bool) -> str:
    if not mtp:
        return binding_name
    prefix = f"model.layers.{_BOUND_MTP_LAYER}."
    if not binding_name.startswith(prefix):
        raise AmsError(
            ErrorCode.INTERNAL_INVARIANT,
            "native probe MTP binding has an unexpected layer name",
        )
    return f"model.layers.{_SOURCE_MTP_LAYER}." + binding_name.removeprefix(prefix)


def _cache_bytes(
    architecture: Glm4MoeLiteArchitecture,
    context_capacity: int,
) -> tuple[int, int, int]:
    key_row = architecture.num_attention_heads * architecture.qk_head_dim * 2
    value_row = architecture.num_attention_heads * architecture.v_head_dim * 2
    staging = key_row + value_row
    per_layer = context_capacity * staging
    return staging, per_layer, architecture.num_hidden_layers * per_layer


def _build_native_binding(
    source_architecture: Glm4MoeLiteArchitecture,
    readers: dict[str, Any],
    source_tensors: dict[str, Any],
    *,
    context_capacity: int,
) -> tuple[Glm4NativeBindingPlan, dict[str, object]]:
    architecture, adaptation = _probe_architecture(source_architecture)
    object_ids = tuple(sorted(readers))
    storage_indices = {object_id: index for index, object_id in enumerate(object_ids)}
    storage_objects = tuple(
        Glm4NativeStorageBinding(
            object_id=object_id,
            absolute_path=str(readers[object_id].path),
            size_bytes=readers[object_id].descriptor.size_bytes,
            alignment_bytes=readers[object_id].descriptor.alignment_bytes,
            content_hash=readers[object_id].descriptor.content_hash,
            kind="tensor_data",
        )
        for object_id in object_ids
    )
    tensor_bindings = []
    mapping = []
    for slot in expected_glm4_moe_lite_tensor_slots(architecture):
        source_name = _source_tensor_name(slot.tensor_name, mtp=slot.mtp)
        try:
            source = source_tensors[source_name]
        except KeyError as exc:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"native probe source tensor is absent: {source_name}",
            ) from exc
        expected_shape = expected_glm4_moe_lite_tensor_shape(architecture, slot)
        if source.shape != expected_shape:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"native probe source tensor shape drifted: {source_name}",
            )
        tensor_bindings.append(
            Glm4NativeTensorBinding(
                tensor_name=slot.tensor_name,
                role=slot.role,
                layer_index=slot.layer_index,
                expert_index=slot.expert_index,
                mtp=slot.mtp,
                shape=source.shape,
                logical_dtype=source.dtype,
                encoding="identity",
                storage_index=storage_indices[source.object_id],
                offset=source.source_offset,
                encoded_bytes=source.source_length,
                decoded_bytes=source.source_length,
                codec_group_size=None,
                codec_config_hash=None,
            )
        )
        mapping.append(
            {
                "binding_name": slot.tensor_name,
                "source_name": source_name,
                "storage_object_id": source.object_id,
                "offset": source.source_offset,
                "length": source.source_length,
            }
        )
    tensors = tuple(tensor_bindings)
    manifest_identity = {
        "schema_id": "ams.glm47-two-layer-native-binding-source.v1",
        "adaptation": adaptation,
        "storage": [
            {
                "object_id": value.object_id,
                "content_hash": value.content_hash,
                "size_bytes": value.size_bytes,
            }
            for value in storage_objects
        ],
        "mapping": mapping,
    }
    manifest_content_root = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(manifest_identity)).hexdigest()
    )
    staging, cache_per_layer, cache_total = _cache_bytes(architecture, context_capacity)
    identity_payload = {
        "schema_id": "ams.native.glm4-binding.v1",
        "package_id": "glm47-two-layer-native-probe",
        "manifest_content_root": manifest_content_root,
        "architecture": architecture,
        "storage_objects": [
            {
                "object_id": value.object_id,
                "size_bytes": value.size_bytes,
                "alignment_bytes": value.alignment_bytes,
                "content_hash": value.content_hash,
                "kind": value.kind,
            }
            for value in storage_objects
        ],
        "tensors": [
            {
                "tensor_name": value.tensor_name,
                "role": value.role,
                "layer_index": value.layer_index,
                "expert_index": value.expert_index,
                "mtp": value.mtp,
                "shape": value.shape,
                "logical_dtype": value.logical_dtype,
                "encoding": value.encoding,
                "storage_object_id": storage_objects[value.storage_index].object_id,
                "offset": value.offset,
                "encoded_bytes": value.encoded_bytes,
                "decoded_bytes": value.decoded_bytes,
                "codec_group_size": value.codec_group_size,
                "codec_config_hash": value.codec_config_hash,
            }
            for value in tensors
        ],
        "linear_arena_bytes": _LINEAR_ARENA_BYTES,
        "context_capacity_tokens": context_capacity,
        "cache_key_dtype": DType.BFLOAT16,
        "cache_value_dtype": DType.BFLOAT16,
        "cache_storage_bytes_per_layer": cache_per_layer,
        "cache_storage_bytes_total": cache_total,
        "cache_staging_bytes_per_layer": staging,
        "tokenizer_vocabulary_size": architecture.vocab_size,
        "eos_token_ids": [0],
    }
    identity_json = canonical_json_bytes(identity_payload)
    binding_hash = "sha256:" + hashlib.sha256(identity_json).hexdigest()
    return (
        Glm4NativeBindingPlan(
            schema_id="ams.native.glm4-binding.v1",
            binding_hash=binding_hash,
            binding_identity_json=identity_json,
            package_id="glm47-two-layer-native-probe",
            manifest_content_root=manifest_content_root,
            architecture=architecture,
            storage_objects=storage_objects,
            tensors=tensors,
            linear_arena_bytes=_LINEAR_ARENA_BYTES,
            context_capacity_tokens=context_capacity,
            cache_key_dtype=DType.BFLOAT16,
            cache_value_dtype=DType.BFLOAT16,
            cache_storage_bytes_per_layer=cache_per_layer,
            cache_storage_bytes_total=cache_total,
            cache_staging_bytes_per_layer=staging,
            tokenizer_vocabulary_size=architecture.vocab_size,
            eos_token_ids=(0,),
        ),
        adaptation,
    )


def _require_toolchain():
    try:
        import safetensors
        import torch
        import transformers
        from safetensors import safe_open
        from transformers import Glm4MoeLiteConfig
        from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
            Glm4MoeLiteDecoderLayer,
            Glm4MoeLiteRMSNorm,
            Glm4MoeLiteRotaryEmbedding,
        )
    except ImportError as exc:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "native layer differential requires torch, transformers, and safetensors",
        ) from exc
    if (
        transformers.__version__ != _EXPECTED_TRANSFORMERS_VERSION
        or safetensors.__version__ != _EXPECTED_SAFETENSORS_VERSION
        or torch.__version__.split("+", 1)[0] != _EXPECTED_TORCH_BASE_VERSION
    ):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "native layer differential reference toolchain is not pinned",
        )
    return (
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRMSNorm,
        Glm4MoeLiteRotaryEmbedding,
    )


def _load_transformers_layer(
    config,
    layer_index: int,
    shard_path: Path,
    torch,
    safe_open,
    Glm4MoeLiteDecoderLayer,
):
    prefix = f"model.layers.{layer_index}."
    with torch.device("meta"):
        layer = Glm4MoeLiteDecoderLayer(config, layer_index)
    parameter_names = tuple(layer.state_dict())
    state = {}
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        for local_name in parameter_names:
            if local_name in {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}:
                continue
            state[local_name] = handle.get_tensor(prefix + local_name)
        if "mlp.experts.gate_up_proj" in parameter_names:
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
            f"Transformers layer {layer_index} mapping is incomplete",
        )
    layer.load_state_dict(state, strict=True, assign=True)
    layer.eval()
    return layer


def _deterministic_token_ids(sample_count: int, vocabulary_size: int) -> tuple[int, ...]:
    if isinstance(sample_count, bool) or not 1 <= sample_count <= 8:
        raise AmsError(ErrorCode.PLAN_INVALID, "native probe sample count must be in [1, 8]")
    return tuple(
        ((index + 1) * 104_729 + 13_007) % vocabulary_size for index in range(sample_count)
    )


def _run_transformers_reference(
    asset_root: Path,
    embedding_shard: Path,
    layer_shard: Path,
    head_shard: Path,
    token_ids: tuple[int, ...],
    input_hash: str,
    torch,
    transformers,
    safetensors,
    safe_open,
    Glm4MoeLiteConfig,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteRotaryEmbedding,
):
    config = Glm4MoeLiteConfig.from_json_file(asset_root / "config.json")
    config._attn_implementation = "eager"
    dense = _load_transformers_layer(
        config,
        0,
        embedding_shard,
        torch,
        safe_open,
        Glm4MoeLiteDecoderLayer,
    )
    sparse = _load_transformers_layer(
        config,
        _LAYER_INDEX,
        layer_shard,
        torch,
        safe_open,
        Glm4MoeLiteDecoderLayer,
    )
    with safe_open(embedding_shard, framework="pt", device="cpu") as handle:
        embedding = handle.get_tensor("model.embed_tokens.weight")
        hidden = embedding[torch.tensor(token_ids, dtype=torch.long)].unsqueeze(0)
    sample_count = len(token_ids)
    positions = torch.arange(sample_count, dtype=torch.long).unsqueeze(0)
    rotary = Glm4MoeLiteRotaryEmbedding(config)
    position_embeddings = rotary(hidden, positions)
    causal = torch.zeros((1, 1, sample_count, sample_count), dtype=hidden.dtype)
    causal.masked_fill_(
        torch.triu(torch.ones((sample_count, sample_count), dtype=torch.bool), diagonal=1),
        torch.finfo(hidden.dtype).min,
    )
    with torch.no_grad():
        hidden = dense(
            hidden,
            attention_mask=causal,
            position_ids=positions,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
        hidden = sparse(
            hidden,
            attention_mask=causal,
            position_ids=positions,
            use_cache=False,
            position_embeddings=position_embeddings,
        )
    with safe_open(head_shard, framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor("model.norm.weight")
        lm_head = handle.get_tensor("lm_head.weight")
        norm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(norm_weight, requires_grad=False)
        norm.eval()
        with torch.no_grad():
            logits = torch.nn.functional.linear(norm(hidden), lm_head)
    hidden_rows = tuple(tuple(float(value) for value in row) for row in hidden[0].float().tolist())
    logit_rows = tuple(tuple(float(value) for value in row) for row in logits[0].float().tolist())
    modeling_path = Path(inspect.getsourcefile(type(sparse)) or "")
    observation = Glm4LayerObservation(
        runtime_id="transformers.glm4_moe_lite.two_layer_probe",
        runtime_version=(
            f"transformers/{transformers.__version__};"
            f"torch/{torch.__version__};safetensors/{safetensors.__version__}"
        ),
        runtime_code_hash=_module_hash((modeling_path, Path(__file__))),
        input_hash=input_hash,
        sample_ids=tuple(f"position-{index}" for index in range(sample_count)),
        hidden_states=hidden_rows,
        logits=logit_rows,
    )
    return observation


def _hash_regular_file(path: Path, *, buffer_bytes: int) -> tuple[Path, str]:
    if buffer_bytes <= 0 or buffer_bytes > 64 * 1024 * 1024:
        raise AmsError(ErrorCode.PLAN_INVALID, "native binary hash buffer is invalid")
    try:
        if path.is_symlink():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "native binary is a symbolic link")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "native binary is not a regular file")
        digest = hashlib.sha256()
        with resolved.open("rb", buffering=0) as handle:
            buffer = bytearray(buffer_bytes)
            while count := handle.readinto(buffer):
                digest.update(memoryview(buffer)[:count])
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, "native binary authentication failed") from exc
    return resolved, "sha256:" + digest.hexdigest()


def _validate_native_output(
    payload: object,
    *,
    binding_hash: str,
    sample_count: int,
    hidden_size: int,
    vocabulary_size: int,
) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_id",
        "binding_hash",
        "hidden_states",
        "logits",
        "selected_token_ids",
        "committed_cache_tokens",
        "cache_heap_bytes",
        "scratch_heap_bytes",
    }:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "native observation fields are invalid")
    if (
        payload["schema_id"] != "ams.native.glm4-observation.v1"
        or payload["binding_hash"] != binding_hash
        or payload["committed_cache_tokens"] != sample_count
    ):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "native observation identity disagrees")
    hidden = payload["hidden_states"]
    logits = payload["logits"]
    selected = payload["selected_token_ids"]
    if (
        not isinstance(hidden, list)
        or not isinstance(logits, list)
        or not isinstance(selected, list)
        or len(hidden) != sample_count
        or len(logits) != sample_count
        or len(selected) != sample_count
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "native observation row counts are invalid")
    for rows, width, label in (
        (hidden, hidden_size, "hidden"),
        (logits, vocabulary_size, "logit"),
    ):
        if any(
            not isinstance(row, list)
            or len(row) != width
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in row
            )
            for row in rows
        ):
            raise AmsError(
                ErrorCode.NUMERIC_FAILURE,
                f"native observation {label} rows are invalid",
            )
    expected_selected = [
        max(range(vocabulary_size), key=lambda index: (row[index], -index)) for row in logits
    ]
    if selected != expected_selected:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "native selected tokens disagree with logits")
    return payload


def _run_native_observation(
    binary: Path,
    plan: Glm4NativeBindingPlan,
    token_ids: tuple[int, ...],
    *,
    buffer_bytes: int,
) -> tuple[dict[str, object], str]:
    resolved_binary, binary_hash = _hash_regular_file(binary, buffer_bytes=buffer_bytes)
    with tempfile.TemporaryDirectory(prefix="ams-glm47-native-probe-") as temporary:
        root = Path(temporary)
        envelope = root / "binding.json"
        request = root / "request.json"
        envelope.write_bytes(serialize_glm4_native_binding_plan(plan))
        request.write_bytes(
            canonical_json_bytes(
                {
                    "schema_id": "ams.native.glm4-observation-request.v1",
                    "input_token_ids": token_ids,
                }
            )
        )
        try:
            result = subprocess.run(
                [
                    resolved_binary,
                    "observe",
                    envelope,
                    request,
                    str(buffer_bytes),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=_NATIVE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AmsError(ErrorCode.IO_FAILURE, "native observation process failed") from exc
    if result.returncode != 0:
        raise AmsError(
            ErrorCode.NUMERIC_FAILURE,
            "native observation process returned a failure",
            evidence={"exit_code": result.returncode, "stderr": result.stderr[-4096:]},
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "native observation JSON is invalid") from exc
    return (
        _validate_native_output(
            payload,
            binding_hash=plan.binding_hash,
            sample_count=len(token_ids),
            hidden_size=plan.architecture.hidden_size,
            vocabulary_size=plan.architecture.vocab_size,
        ),
        binary_hash,
    )


def main() -> int:
    arguments = _parser().parse_args()
    paths = {
        _EMBEDDING_SHARD_NAME: arguments.embedding_shard,
        _SHARD_NAME: arguments.layer_shard,
        _HEAD_SHARD_NAME: arguments.head_shard,
        _MTP_SHARD_NAME: arguments.mtp_shard,
    }
    architecture, _index, readers, tensors, catalogs = _load_source(
        arguments.asset_root,
        paths,
        buffer_bytes=arguments.buffer_bytes,
    )
    token_ids = _deterministic_token_ids(arguments.samples, architecture.vocab_size)
    input_identity = {
        "schema_id": "ams.glm47-two-layer-token-input.v1",
        "source_architecture_hash": architecture.content_hash,
        "token_ids": token_ids,
    }
    input_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(input_identity)).hexdigest()
    plan, adaptation = _build_native_binding(
        architecture,
        readers,
        tensors,
        context_capacity=len(token_ids),
    )
    (
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRMSNorm,
        Glm4MoeLiteRotaryEmbedding,
    ) = _require_toolchain()
    reference = _run_transformers_reference(
        arguments.asset_root,
        arguments.embedding_shard,
        arguments.layer_shard,
        arguments.head_shard,
        token_ids,
        input_hash,
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRMSNorm,
        Glm4MoeLiteRotaryEmbedding,
    )
    native, binary_hash = _run_native_observation(
        arguments.native_binary,
        plan,
        token_ids,
        buffer_bytes=arguments.buffer_bytes,
    )
    candidate = Glm4LayerObservation(
        runtime_id="ams.core.glm47_two_layer_probe",
        runtime_version=f"ams/{ams_version};native/ams-runtime",
        runtime_code_hash=binary_hash,
        input_hash=input_hash,
        sample_ids=tuple(f"position-{index}" for index in range(len(token_ids))),
        hidden_states=tuple(
            tuple(float(value) for value in row) for row in native["hidden_states"]
        ),
        logits=tuple(tuple(float(value) for value in row) for row in native["logits"]),
    )
    comparison = compare_glm4_layer_observations(
        reference,
        candidate,
        expected_hidden_size=architecture.hidden_size,
        expected_vocabulary_size=architecture.vocab_size,
        blockers=("two-layer native probe is not a complete-model teacher-forced execution",),
    )
    output = comparison.to_dict()
    output["source"] = _source_evidence(
        architecture,
        catalogs[_SHARD_NAME],
        readers[_HEAD_PIN.object_id],
    )
    output["input"] = {
        "schema_id": "ams.glm47-layer-input.v1",
        "content_hash": input_hash,
        "sample_count": len(token_ids),
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
    output["native_probe"] = {
        **adaptation,
        "binding_hash": plan.binding_hash,
        "native_binary_hash": binary_hash,
        "input_token_ids": list(token_ids),
        "selected_token_ids": native["selected_token_ids"],
        "committed_cache_tokens": native["committed_cache_tokens"],
        "cache_heap_bytes": native["cache_heap_bytes"],
        "scratch_heap_bytes": native["scratch_heap_bytes"],
        "storage": [
            {
                "object_id": value.object_id,
                "content_hash": value.content_hash,
                "size_bytes": value.size_bytes,
            }
            for value in plan.storage_objects
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if comparison.status is Glm4LayerDifferentialStatus.PASSED else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AmsError as error:
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
