"""Execute the complete authenticated GLM-4.7 model through native ams-core."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.descriptors import DType, StorageObject
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
)
from ams.storage import FileRangeStore
from ams.version import __version__ as ams_version

if __package__:
    from . import audit_glm4_precision as source_audit
    from . import verify_glm4_official_layer_native as layer_probe
else:
    import audit_glm4_precision as source_audit
    import verify_glm4_official_layer_native as layer_probe

_REPOSITORY = "zai-org/GLM-4.7-Flash"
_REVISION = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"
_SHARD_COUNT = 48
_BASE_LAYER_COUNT = 47
_MTP_LAYER_INDEX = 47
_LINEAR_ARENA_BYTES = 1024 * 1024
_CHECKPOINT_SCHEMA = "ams.glm47-streaming-reference-checkpoint.v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Full-hash all 48 pinned GLM-4.7 shards, execute the complete 47-layer base model "
            "through streaming Transformers BF16 and native ams-core, and compare teacher-forced "
            "hidden states and logits"
        )
    )
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("native_binary", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--buffer-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--resume-reference",
        action="store_true",
        help=(
            "Resume the streaming reference from a validated local checkpoint. "
            "Final evidence should also match a fresh run without this flag."
        ),
    )
    return parser


def _read_pinned_assets(asset_root: Path):
    config_payload = layer_probe._read_bounded_regular(
        asset_root / "config.json",
        layer_probe._MAX_CONFIG_BYTES,
        label="GLM-4.7 config",
    )
    index_payload = layer_probe._read_bounded_regular(
        asset_root / "model.safetensors.index.json",
        layer_probe._MAX_INDEX_BYTES,
        label="GLM-4.7 shard index",
    )
    if layer_probe._sha256(config_payload) != layer_probe._CONFIG_SHA256:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 config hash is not pinned")
    if layer_probe._sha256(index_payload) != layer_probe._INDEX_SHA256:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 shard-index hash is not pinned")
    architecture = parse_glm4_moe_lite_architecture(config_payload)
    index = parse_huggingface_shard_index(index_payload)
    validate_glm4_moe_lite_tensor_inventory(architecture, index)
    if (
        architecture.num_hidden_layers != _BASE_LAYER_COUNT
        or architecture.num_nextn_predict_layers != 1
        or len(index.shard_names) != _SHARD_COUNT
    ):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "GLM-4.7 complete-model architecture or shard count drifted",
        )
    return config_payload, architecture, index


def _validate_anchor_pins(siblings: dict[str, tuple[int, str]]) -> None:
    for pin in (
        layer_probe._EMBEDDING_PIN,
        layer_probe._LAYER_PIN,
        layer_probe._HEAD_PIN,
        layer_probe._MTP_PIN,
    ):
        if siblings.get(pin.name) != (pin.size_bytes, pin.sha256):
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                f"pinned GLM-4.7 metadata disagrees with the embedded anchor: {pin.name}",
            )


def _open_catalog_source(
    shard_root: Path,
    shard_name: str,
    size_bytes: int,
    sha256: str,
) -> tuple[FileRangeStore, HuggingFaceShardSource]:
    path = shard_root / shard_name
    try:
        if path.is_symlink():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                f"GLM-4.7 shard is a symbolic link: {shard_name}",
            )
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size != size_bytes:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                f"GLM-4.7 shard is not the pinned regular-file size: {shard_name}",
            )
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            f"GLM-4.7 shard metadata failed: {shard_name}",
        ) from exc
    object_id = f"hf:{shard_name}"
    descriptor = StorageObject(
        object_id=object_id,
        uri=f"file:local-pinned-{shard_name}",
        size_bytes=size_bytes,
        alignment_bytes=1,
        content_hash=f"sha256:{sha256}",
    )
    reader = FileRangeStore(resolved, descriptor)
    return (
        reader,
        HuggingFaceShardSource(
            shard_name=shard_name,
            object_id=object_id,
            content_hash=descriptor.content_hash,
            reader=reader,
        ),
    )


def _authenticate_complete_source(
    architecture: Glm4MoeLiteArchitecture,
    index,
    shard_root: Path,
    *,
    buffer_bytes: int,
):
    if buffer_bytes <= 0 or buffer_bytes > 64 * 1024 * 1024:
        raise AmsError(ErrorCode.PLAN_INVALID, "verification buffer is outside the reviewed bound")
    metadata = source_audit._fetch_model_metadata(_REPOSITORY, _REVISION)
    siblings = source_audit._normalize_siblings(metadata)
    _validate_anchor_pins(siblings)
    if any(name not in siblings for name in index.shard_names):
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "pinned Hugging Face metadata omits a GLM-4.7 shard",
        )

    readers: dict[str, FileRangeStore] = {}
    tensors: dict[str, Any] = {}
    storage = []
    for position, shard_name in enumerate(index.shard_names, start=1):
        size_bytes, sha256 = siblings[shard_name]
        reader, source = _open_catalog_source(
            shard_root,
            shard_name,
            size_bytes,
            sha256,
        )
        print(
            f"[authenticate {position:02d}/{_SHARD_COUNT}] {shard_name}",
            file=sys.stderr,
            flush=True,
        )
        catalog = build_huggingface_shard_catalog(index, source, buffer_bytes=buffer_bytes)
        readers[source.object_id] = reader
        storage.append(
            {
                "object_id": source.object_id,
                "content_hash": source.content_hash,
                "size_bytes": size_bytes,
            }
        )
        for tensor in catalog.tensors:
            if tensor.tensor_name in tensors:
                raise AmsError(
                    ErrorCode.INTERNAL_INVARIANT,
                    "complete-model tensor appears in multiple authenticated shards",
                )
            tensors[tensor.tensor_name] = tensor
    expected_names = {entry.tensor_name for entry in index.entries}
    if set(tensors) != expected_names:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "authenticated complete-model tensor inventory disagrees with the index",
        )
    for slot in expected_glm4_moe_lite_tensor_slots(architecture):
        tensor = tensors[slot.tensor_name]
        if tensor.shape != expected_glm4_moe_lite_tensor_shape(architecture, slot):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                f"authenticated complete-model tensor shape drifted: {slot.tensor_name}",
            )

    source_identity = {
        "schema_id": "ams.glm47-complete-source.v1",
        "repository": _REPOSITORY,
        "revision": _REVISION,
        "architecture_hash": architecture.content_hash,
        "source_index_hash": index.content_hash,
        "storage": storage,
    }
    source_root = "sha256:" + hashlib.sha256(canonical_json_bytes(source_identity)).hexdigest()
    return readers, tensors, tuple(storage), source_root


def _build_full_native_binding(
    architecture: Glm4MoeLiteArchitecture,
    readers: dict[str, FileRangeStore],
    source_tensors: dict[str, Any],
    storage_evidence: tuple[dict[str, object], ...],
    source_root: str,
    *,
    context_capacity: int,
) -> Glm4NativeBindingPlan:
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
        source = source_tensors[slot.tensor_name]
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
                "tensor_name": slot.tensor_name,
                "storage_object_id": source.object_id,
                "offset": source.source_offset,
                "length": source.source_length,
            }
        )
    tensors = tuple(tensor_bindings)
    manifest_identity = {
        "schema_id": "ams.glm47-complete-native-binding-source.v1",
        "source_root": source_root,
        "storage": storage_evidence,
        "mapping": mapping,
    }
    manifest_content_root = (
        "sha256:" + hashlib.sha256(canonical_json_bytes(manifest_identity)).hexdigest()
    )
    staging, cache_per_layer, cache_total = layer_probe._cache_bytes(
        architecture,
        context_capacity,
    )
    identity_payload = {
        "schema_id": "ams.native.glm4-binding.v1",
        "package_id": "glm47-complete-bf16-native-probe",
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
    return Glm4NativeBindingPlan(
        schema_id="ams.native.glm4-binding.v1",
        binding_hash=binding_hash,
        binding_identity_json=identity_json,
        package_id="glm47-complete-bf16-native-probe",
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
    )


def _checkpoint_metadata(
    architecture: Glm4MoeLiteArchitecture,
    input_hash: str,
    completed_layer: int,
    sample_count: int,
) -> dict[str, str]:
    return {
        "schema_id": _CHECKPOINT_SCHEMA,
        "architecture_hash": architecture.content_hash,
        "input_hash": input_hash,
        "completed_layer": str(completed_layer),
        "sample_count": str(sample_count),
        "hidden_size": str(architecture.hidden_size),
    }


def _prepare_checkpoint_root(path: Path) -> Path:
    try:
        if path.exists() and path.is_symlink():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "streaming reference checkpoint root is a symbolic link",
            )
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "streaming reference checkpoint root is not a directory",
            )
        return resolved
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "streaming reference checkpoint root could not be prepared",
        ) from exc


def _load_reference_checkpoint(
    root: Path,
    architecture: Glm4MoeLiteArchitecture,
    input_hash: str,
    sample_count: int,
    torch,
    safe_open,
):
    from safetensors.torch import load_file

    for completed_layer in range(architecture.num_hidden_layers - 1, -1, -1):
        path = root / f"layer-{completed_layer:02d}.safetensors"
        if not path.exists():
            continue
        try:
            if path.is_symlink() or not path.is_file():
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "streaming reference checkpoint is not a nonsymlink regular file",
                )
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                keys = tuple(handle.keys())
            expected = _checkpoint_metadata(
                architecture,
                input_hash,
                completed_layer,
                sample_count,
            )
            if metadata != expected or keys != ("hidden",):
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "streaming reference checkpoint identity is invalid",
                )
            hidden = load_file(path, device="cpu")["hidden"]
        except AmsError:
            raise
        except (OSError, RuntimeError, KeyError) as exc:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "streaming reference checkpoint could not be admitted",
            ) from exc
        if (
            tuple(hidden.shape) != (1, sample_count, architecture.hidden_size)
            or hidden.dtype != torch.bfloat16
            or not bool(torch.isfinite(hidden).all())
        ):
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "streaming reference checkpoint tensor is invalid",
            )
        print(
            f"[reference resume] completed layer {completed_layer}",
            file=sys.stderr,
            flush=True,
        )
        return hidden, completed_layer + 1
    return None, 0


def _save_reference_checkpoint(
    root: Path,
    architecture: Glm4MoeLiteArchitecture,
    input_hash: str,
    completed_layer: int,
    hidden,
) -> None:
    from safetensors.torch import save_file

    target = root / f"layer-{completed_layer:02d}.safetensors"
    temporary = root / f".layer-{completed_layer:02d}-{os.getpid()}.tmp"
    try:
        if target.exists() and target.is_symlink():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "streaming reference checkpoint target is a symbolic link",
            )
        save_file(
            {"hidden": hidden.detach().cpu().contiguous()},
            temporary,
            metadata=_checkpoint_metadata(
                architecture,
                input_hash,
                completed_layer,
                hidden.shape[1],
            ),
        )
        temporary.replace(target)
    except AmsError:
        raise
    except (OSError, RuntimeError) as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "streaming reference checkpoint could not be published",
        ) from exc


def _layer_shard_path(index, shard_root: Path, layer_index: int) -> Path:
    prefix = f"model.layers.{layer_index}."
    shards = {entry.shard_name for entry in index.entries if entry.tensor_name.startswith(prefix)}
    if len(shards) != 1:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            f"base layer {layer_index} does not map to one pinned shard",
        )
    return shard_root / next(iter(shards))


def _run_streaming_reference(
    asset_root: Path,
    shard_root: Path,
    index,
    architecture: Glm4MoeLiteArchitecture,
    token_ids: tuple[int, ...],
    input_hash: str,
    checkpoint_root: Path,
    torch,
    transformers,
    safetensors,
    safe_open,
    Glm4MoeLiteConfig,
    Glm4MoeLiteDecoderLayer,
    Glm4MoeLiteRMSNorm,
    Glm4MoeLiteRotaryEmbedding,
    *,
    resume_reference: bool,
):
    config = Glm4MoeLiteConfig.from_json_file(asset_root / "config.json")
    config._attn_implementation = "eager"
    hidden, start_layer = (None, 0)
    if resume_reference:
        hidden, start_layer = _load_reference_checkpoint(
            checkpoint_root,
            architecture,
            input_hash,
            len(token_ids),
            torch,
            safe_open,
        )
    if hidden is None:
        embedding_path = _layer_shard_path(index, shard_root, 0)
        with safe_open(embedding_path, framework="pt", device="cpu") as handle:
            embedding = handle.get_tensor("model.embed_tokens.weight")
            hidden = embedding[torch.tensor(token_ids, dtype=torch.long)].unsqueeze(0)
        del embedding

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
        for layer_index in range(start_layer, architecture.num_hidden_layers):
            print(
                f"[reference layer {layer_index:02d}/{architecture.num_hidden_layers - 1:02d}]",
                file=sys.stderr,
                flush=True,
            )
            layer = layer_probe._load_transformers_layer(
                config,
                layer_index,
                _layer_shard_path(index, shard_root, layer_index),
                torch,
                safe_open,
                Glm4MoeLiteDecoderLayer,
            )
            hidden = layer(
                hidden,
                attention_mask=causal,
                position_ids=positions,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            if not bool(torch.isfinite(hidden).all()):
                raise AmsError(
                    ErrorCode.NUMERIC_FAILURE,
                    f"streaming Transformers reference failed at layer {layer_index}",
                )
            _save_reference_checkpoint(
                checkpoint_root,
                architecture,
                input_hash,
                layer_index,
                hidden,
            )
            del layer
            gc.collect()

    decoder_hidden = tuple(
        tuple(float(value) for value in row) for row in hidden[0].float().tolist()
    )
    head_path = shard_root / layer_probe._HEAD_SHARD_NAME
    with safe_open(head_path, framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor("model.norm.weight")
        lm_head = handle.get_tensor("lm_head.weight")
        norm = Glm4MoeLiteRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight = torch.nn.Parameter(norm_weight, requires_grad=False)
        norm.eval()
        with torch.no_grad():
            logits = torch.nn.functional.linear(norm(hidden), lm_head)
    logit_rows = tuple(tuple(float(value) for value in row) for row in logits[0].float().tolist())
    modeling_path = Path(inspect.getsourcefile(Glm4MoeLiteDecoderLayer) or "")
    runtime_code_hash = layer_probe._module_hash(
        (modeling_path, Path(layer_probe.__file__), Path(__file__))
    )
    return Glm4LayerObservation(
        runtime_id="transformers.glm4_moe_lite.complete_streaming_bf16",
        runtime_version=(
            f"transformers/{transformers.__version__};"
            f"torch/{torch.__version__};safetensors/{safetensors.__version__}"
        ),
        runtime_code_hash=runtime_code_hash,
        input_hash=input_hash,
        sample_ids=tuple(f"position-{index}" for index in range(sample_count)),
        hidden_states=decoder_hidden,
        logits=logit_rows,
    )


def _maximum_base_layer_payload(
    architecture: Glm4MoeLiteArchitecture,
    tensors: dict[str, Any],
) -> int:
    totals = []
    for layer_index in range(architecture.num_hidden_layers):
        prefix = f"model.layers.{layer_index}."
        totals.append(
            sum(tensor.source_length for name, tensor in tensors.items() if name.startswith(prefix))
        )
    return max(totals)


def main() -> int:
    arguments = _parser().parse_args()
    if isinstance(arguments.torch_threads, bool) or not 1 <= arguments.torch_threads <= 64:
        raise AmsError(ErrorCode.PLAN_INVALID, "torch thread count must be in [1, 64]")
    _config, architecture, index = _read_pinned_assets(arguments.asset_root)
    token_ids = layer_probe._deterministic_token_ids(arguments.samples, architecture.vocab_size)
    input_identity = {
        "schema_id": "ams.glm47-complete-token-input.v1",
        "source_architecture_hash": architecture.content_hash,
        "token_ids": token_ids,
    }
    input_hash = "sha256:" + hashlib.sha256(canonical_json_bytes(input_identity)).hexdigest()
    readers, tensors, storage, source_root = _authenticate_complete_source(
        architecture,
        index,
        arguments.shard_root,
        buffer_bytes=arguments.buffer_bytes,
    )
    plan = _build_full_native_binding(
        architecture,
        readers,
        tensors,
        storage,
        source_root,
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
    ) = layer_probe._require_toolchain()
    torch.set_num_threads(arguments.torch_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint_root = _prepare_checkpoint_root(
        arguments.checkpoint_dir
        or (
            arguments.shard_root
            / ".qualification"
            / "glm47-streaming-reference"
            / input_hash.removeprefix("sha256:")
        )
    )
    reference = _run_streaming_reference(
        arguments.asset_root,
        arguments.shard_root,
        index,
        architecture,
        token_ids,
        input_hash,
        checkpoint_root,
        torch,
        transformers,
        safetensors,
        safe_open,
        Glm4MoeLiteConfig,
        Glm4MoeLiteDecoderLayer,
        Glm4MoeLiteRMSNorm,
        Glm4MoeLiteRotaryEmbedding,
        resume_reference=arguments.resume_reference,
    )
    gc.collect()
    layer_probe._NATIVE_TIMEOUT_SECONDS = 30 * 60
    native, binary_hash = layer_probe._run_native_observation(
        arguments.native_binary,
        plan,
        token_ids,
        buffer_bytes=arguments.buffer_bytes,
    )
    candidate = Glm4LayerObservation(
        runtime_id="ams.core.glm47_complete_bf16",
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
    )
    reference_top_tokens = [
        max(range(architecture.vocab_size), key=row.__getitem__) for row in reference.logits or ()
    ]
    candidate_top_tokens = [
        max(range(architecture.vocab_size), key=row.__getitem__) for row in candidate.logits or ()
    ]
    if candidate_top_tokens != native["selected_token_ids"]:
        raise AmsError(
            ErrorCode.INTERNAL_INVARIANT,
            "native selected tokens disagree with the complete candidate logits",
        )
    output = comparison.to_dict()
    output["schema_id"] = "ams.glm47-model-differential.v1"
    output["source"] = {
        "repository": _REPOSITORY,
        "revision": _REVISION,
        "architecture_hash": architecture.content_hash,
        "source_index_hash": index.content_hash,
        "source_root": source_root,
        "shard_count": len(storage),
        "tensor_count": len(tensors),
        "source_storage_bytes": sum(item["size_bytes"] for item in storage),
        "storage": storage,
        "base_layer_count": architecture.num_hidden_layers,
        "mtp_layer_index": _MTP_LAYER_INDEX,
        "mtp_admitted_not_executed": True,
        "full_hash_authenticated": True,
        "teacher_forced_full_model": True,
    }
    output["input"] = {
        "schema_id": "ams.glm47-model-input.v1",
        "content_hash": input_hash,
        "sample_count": len(token_ids),
        "token_ids": list(token_ids),
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
    output["gates"] = {
        "hidden_state_gate_passed": comparison.hidden_state_gate_passed,
        "logit_gate_passed": comparison.logit_gate_passed,
        "complete_model_gate_passed": comparison.full_layer_gate_passed,
        "qualifies_precision_policy": False,
    }
    output["native_execution"] = {
        "binding_hash": plan.binding_hash,
        "native_binary_hash": binary_hash,
        "selected_token_ids": native["selected_token_ids"],
        "committed_cache_tokens": native["committed_cache_tokens"],
        "cache_heap_bytes": native["cache_heap_bytes"],
        "scratch_heap_bytes": native["scratch_heap_bytes"],
        "raw_observation_bytes": (
            len(token_ids) * (architecture.hidden_size + architecture.vocab_size) * 8
        ),
    }
    output["teacher_forced"] = {
        "input_token_ids": list(token_ids),
        "reference_top_token_ids": reference_top_tokens,
        "candidate_top_token_ids": candidate_top_tokens,
    }
    output["resources"] = {
        "full_model_materialized": False,
        "reference_streaming_layer_source_payload_bound_bytes": _maximum_base_layer_payload(
            architecture,
            tensors,
        ),
        "reference_checkpoint_tensor_bytes": (len(token_ids) * architecture.hidden_size * 2),
        "torch_threads": arguments.torch_threads,
    }
    if any(
        not math.isfinite(value)
        for value in (
            comparison.hidden_cosine_similarity,
            comparison.hidden_normalized_rmse,
            comparison.top_token_agreement or 0.0,
        )
    ):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "complete-model metrics are non-finite")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if comparison.status is Glm4LayerDifferentialStatus.PASSED else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AmsError as error:
        print(json.dumps(error.to_dict(), sort_keys=True), file=sys.stderr)
        raise SystemExit(1) from error
