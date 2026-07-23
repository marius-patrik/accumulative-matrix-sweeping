"""Convert the complete authenticated GLM-4.7 package into restartable AMS INT4 CAS state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_mul, checked_product
from ams.codecs import Int4CodecConfig, encode_int4_stream_numpy
from ams.descriptors import DType, validate_digest
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4PrecisionProfile,
    HuggingFaceCatalogPolicy,
    HuggingFaceTensorEncoding,
    HuggingFaceTotalSizeSemantics,
    build_accuracy_first_glm4_precision_candidate,
    build_huggingface_header_catalog,
    build_huggingface_progressive_mixed_plan,
)
from ams.package import (
    GraphArtifact,
    build_huggingface_mixed_manifest,
    publish_manifest_last,
)
from ams.progressive_conversion import (
    execute_progressive_huggingface_mixed_conversion,
    finalize_progressive_huggingface_mixed_conversion,
)

if __package__:
    from . import verify_glm4_official_model_native as bf16_probe
else:
    import verify_glm4_official_model_native as bf16_probe

_SOURCE_RECEIPT = (
    Path(__file__).parents[1] / "docs" / "evidence" / "glm47_complete_bf16_differential.json"
)
_CANDIDATE_RECEIPT = (
    Path(__file__).parents[1] / "docs" / "evidence" / "glm47_int4_bringup_candidate.json"
)
_EXPECTED_NUMPY_VERSION = "2.5.1"
_EXPECTED_CANDIDATE_HASH = "sha256:0b09bf971fe0ecce07d9f3801518c31cd2d574d18a98fb47faf132407b876850"
_EXPECTED_POLICY_HASH = "sha256:c6af0d95cedb6b602196159cd0b420e14bda8a4612daabf42d101af07faa7e77"
_EXPECTED_SOURCE_FILE_BYTES = 62_444_175_504
_EXPECTED_SOURCE_TENSOR_BYTES = 62_442_983_168
_EXPECTED_TARGET_BYTES = 17_527_623_424
_EXPECTED_IDENTITY_COUNT = 292
_EXPECTED_INT4_COUNT = 9_411
_EXPECTED_TENSOR_COUNT = 9_703
_DEFAULT_BUFFER_BYTES = 8 * 1024 * 1024
_DEFAULT_SAFETY_MARGIN_BYTES = 4 * 1024 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024


def _default_model_store() -> Path:
    override = os.environ.get("ANDROMEDA_MODEL_STORE")
    return Path(override) if override else Path.home() / ".agents" / "store" / "models"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute the complete pinned GLM-4.7 accuracy-first INT4 conversion. "
            "Outputs are content addressed and restartable; source shards are never deleted."
        )
    )
    parser.add_argument("asset_root", type=Path)
    parser.add_argument("shard_root", type=Path)
    parser.add_argument("--model-store", type=Path, default=_default_model_store())
    parser.add_argument("--buffer-bytes", type=int, default=_DEFAULT_BUFFER_BYTES)
    parser.add_argument("--int4-block-bytes", type=int, default=_DEFAULT_BUFFER_BYTES)
    parser.add_argument(
        "--safety-margin-bytes",
        type=int,
        default=_DEFAULT_SAFETY_MARGIN_BYTES,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Authenticate headers and print exact capacity/identity bounds without writing state.",
    )
    return parser


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is not a regular file")
        size = path.stat().st_size
        if size <= 0 or size > _MAX_JSON_BYTES:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} size is invalid")
        value = json.loads(path.read_bytes())
    except AmsError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is malformed") from exc
    if not isinstance(value, dict):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} must contain one object")
    return value


def _source_pins(index) -> dict[str, tuple[int, str]]:
    receipt = _read_json_object(_SOURCE_RECEIPT, label="complete BF16 source receipt")
    try:
        source = receipt["source"]
        storage = source["storage"]
    except KeyError as exc:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "BF16 source receipt is incomplete") from exc
    if (
        receipt.get("status") != "passed"
        or not isinstance(source, dict)
        or source.get("repository") != bf16_probe._REPOSITORY
        or source.get("revision") != bf16_probe._REVISION
        or source.get("source_index_hash") != index.content_hash
        or source.get("shard_count") != bf16_probe._SHARD_COUNT
        or source.get("tensor_count") != _EXPECTED_TENSOR_COUNT
        or source.get("source_storage_bytes") != _EXPECTED_SOURCE_FILE_BYTES
        or not isinstance(storage, list)
        or len(storage) != bf16_probe._SHARD_COUNT
    ):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "BF16 source receipt identity drifted")
    pins: dict[str, tuple[int, str]] = {}
    for item in storage:
        if not isinstance(item, dict) or set(item) != {
            "object_id",
            "content_hash",
            "size_bytes",
        }:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "BF16 source storage row is invalid")
        object_id = item["object_id"]
        content_hash = item["content_hash"]
        size_bytes = item["size_bytes"]
        if (
            not isinstance(object_id, str)
            or not object_id.startswith("hf:model-")
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "BF16 source storage identity is invalid")
        validate_digest(content_hash, name="glm47_int4.source_content_hash")
        shard_name = object_id.removeprefix("hf:")
        if shard_name in pins:
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "BF16 source storage is duplicated")
        pins[shard_name] = (size_bytes, content_hash.removeprefix("sha256:"))
    if set(pins) != set(index.shard_names):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "BF16 source shard set drifted")
    bf16_probe._validate_anchor_pins(pins)
    return pins


def _build_structural_catalog(asset_root: Path, shard_root: Path):
    config_payload, architecture, index = bf16_probe._read_pinned_assets(asset_root)
    pins = _source_pins(index)
    sources = tuple(
        bf16_probe._open_catalog_source(
            shard_root,
            shard_name,
            pins[shard_name][0],
            pins[shard_name][1],
        )[1]
        for shard_name in index.shard_names
    )
    catalog = build_huggingface_header_catalog(
        index,
        sources,
        policy=HuggingFaceCatalogPolicy(
            total_size_semantics=HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS,
            expected_index_content_hash=index.content_hash,
        ),
    )
    if (
        catalog.audit.source_file_bytes != _EXPECTED_SOURCE_FILE_BYTES
        or catalog.audit.tensor_bytes != _EXPECTED_SOURCE_TENSOR_BYTES
        or catalog.audit.tensor_count != _EXPECTED_TENSOR_COUNT
    ):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 structural byte counts drifted")
    return config_payload, architecture, index, catalog


def _build_candidate(architecture, index, catalog):
    inventory = bf16_probe.validate_glm4_moe_lite_tensor_inventory(architecture, index)
    int4_config = Int4CodecConfig(group_size=128)
    candidate = build_accuracy_first_glm4_precision_candidate(
        architecture,
        inventory,
        catalog.tensors,
        int4_config=int4_config,
    )
    checked = _read_json_object(_CANDIDATE_RECEIPT, label="INT4 candidate receipt")
    expected_counts = {
        HuggingFaceTensorEncoding.IDENTITY: _EXPECTED_IDENTITY_COUNT,
        HuggingFaceTensorEncoding.INT4_SYMMETRIC: _EXPECTED_INT4_COUNT,
    }
    if (
        checked.get("status") != "experimental"
        or checked.get("profile") != Glm4PrecisionProfile.INT4_BRINGUP.value
        or checked.get("candidate_hash") != _EXPECTED_CANDIDATE_HASH
        or checked.get("policy_hash") != _EXPECTED_POLICY_HASH
        or checked.get("estimated_encoded_bytes") != _EXPECTED_TARGET_BYTES
        or checked.get("source_index_hash") != index.content_hash
        or candidate.candidate_hash != _EXPECTED_CANDIDATE_HASH
        or candidate.policy.policy_hash != _EXPECTED_POLICY_HASH
        or candidate.estimated_encoded_bytes != _EXPECTED_TARGET_BYTES
        or dict(candidate.encoding_counts) != expected_counts
        or int4_config.config_hash != checked.get("int4_config_hash")
    ):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 INT4 candidate receipt drifted")
    return candidate, int4_config


def _target_bytes(plan) -> tuple[int, int, int]:
    total = 0
    identity = 0
    int4 = 0
    for tensor in plan.tensors:
        if tensor.encoding is HuggingFaceTensorEncoding.IDENTITY:
            encoded = tensor.tensor.source_length
            identity = checked_add(identity, encoded, name="glm47_int4.identity_bytes")
        elif tensor.encoding is HuggingFaceTensorEncoding.INT4_SYMMETRIC:
            if tensor.int4_config is None:
                raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 tensor has no codec config")
            encoded = tensor.int4_config.encoded_size(
                checked_product(tensor.tensor.shape, name="glm47_int4.elements")
            )
            int4 = checked_add(int4, encoded, name="glm47_int4.int4_bytes")
        else:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "INT4 plan contains another encoding")
        total = checked_add(total, encoded, name="glm47_int4.target_bytes")
    if total != _EXPECTED_TARGET_BYTES:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-4.7 INT4 target byte count drifted")
    return total, identity, int4


def _nearest_existing_parent(path: Path) -> Path:
    current = path.resolve(strict=False)
    while not current.exists():
        if current.parent == current:
            raise AmsError(ErrorCode.IO_FAILURE, "model-store volume is unavailable")
        current = current.parent
    return current


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve(strict=False)
    second = second.resolve(strict=False)
    try:
        first.relative_to(second)
        return True
    except ValueError:
        try:
            second.relative_to(first)
            return True
        except ValueError:
            return False


def _capacity_evidence(
    model_store: Path,
    shard_root: Path,
    plan,
    target_bytes: int,
    safety_margin_bytes: int,
) -> dict[str, int]:
    if isinstance(safety_margin_bytes, bool) or safety_margin_bytes < 0:
        raise AmsError(ErrorCode.PLAN_INVALID, "conversion safety margin is invalid")
    if _paths_overlap(model_store, shard_root):
        raise AmsError(ErrorCode.PLAN_INVALID, "model store and source shard root overlap")
    maximum_source_shard_bytes = max(shard.size_bytes for shard in plan.shards)
    durable_record_bound_bytes = checked_mul(
        len(plan.tensors),
        128 * 1024 + 64 * 1024,
        name="glm47_int4.record_bound_bytes",
    )
    manifest_bound_bytes = 64 * 1024 * 1024
    required_free_bytes = sum(
        (
            target_bytes,
            maximum_source_shard_bytes,
            durable_record_bound_bytes,
            manifest_bound_bytes,
            safety_margin_bytes,
        )
    )
    volume = shutil.disk_usage(_nearest_existing_parent(model_store))
    if volume.free < required_free_bytes:
        raise AmsError(
            ErrorCode.PREFLIGHT_NO_BACKING,
            "insufficient free storage for complete GLM-4.7 INT4 conversion",
            evidence={
                "available_bytes": volume.free,
                "required_bytes": required_free_bytes,
            },
        )
    return {
        "available_bytes": volume.free,
        "durable_record_bound_bytes": durable_record_bound_bytes,
        "manifest_bound_bytes": manifest_bound_bytes,
        "maximum_source_shard_bytes": maximum_source_shard_bytes,
        "required_free_bytes": required_free_bytes,
        "safety_margin_bytes": safety_margin_bytes,
        "target_bytes": target_bytes,
    }


def _prepare_directory(path: Path, *, label: str) -> Path:
    try:
        if path.exists() and path.is_symlink():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is a symbolic link")
        path.mkdir(parents=True, exist_ok=True)
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            raise AmsError(ErrorCode.INVALID_PACKAGE, f"{label} is not a directory")
        return resolved
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, f"{label} could not be prepared") from exc


def _publish_immutable(path: Path, payload: bytes, *, label: str) -> Path:
    if not payload:
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, f"{label} payload is empty")
    _prepare_directory(path.parent, label=f"{label} parent")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    if path.exists():
        try:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise AmsError(
                    ErrorCode.TRANSACTION_FAILURE, f"{label} disagrees with durable state"
                )
            return path
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(ErrorCode.IO_FAILURE, f"{label} could not be verified") from exc
    try:
        with temporary.open("xb", buffering=0) as handle:
            written = 0
            while written < len(payload):
                count = handle.write(payload[written:])
                if count is None or count == 0:
                    raise AmsError(ErrorCode.IO_FAILURE, f"{label} write was short")
                written += count
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.TRANSACTION_FAILURE,
            f"{label} could not be published",
            retriable=True,
        ) from exc
    finally:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    return path


class _ProgressReporter:
    def __init__(self, journal_root: Path, total_tensors: int) -> None:
        self.journal_root = journal_root
        self.total_tensors = total_tensors
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args) -> None:
        self.stop.set()
        self.thread.join(timeout=5)

    def _run(self) -> None:
        while not self.stop.wait(30):
            tensor_directory = self.journal_root / "tensors"
            shard_directory = self.journal_root / "shards"
            tensors = (
                sum(1 for _ in tensor_directory.glob("*.json")) if tensor_directory.is_dir() else 0
            )
            shards = (
                sum(1 for _ in shard_directory.glob("*.json")) if shard_directory.is_dir() else 0
            )
            print(
                f"[progress] verified_shards={shards}/48 "
                f"published_tensors={tensors}/{self.total_tensors}",
                file=sys.stderr,
                flush=True,
            )


def _graph_artifact(
    model_store: Path,
    manifest_base_uri: str,
    architecture,
    plan,
) -> GraphArtifact:
    graph_value = {
        "architecture_hash": architecture.content_hash,
        "base_layer_count": architecture.num_hidden_layers,
        "entry_points": ["causal_lm"],
        "mtp_admitted_not_executed": True,
        "mtp_layer_index": architecture.num_hidden_layers,
        "plan_hash": plan.plan_hash,
        "schema_id": "ams.glm47-int4-native-graph.v1",
        "tensor_count": len(plan.tensors),
    }
    payload = canonical_json_bytes(graph_value)
    uri = f"{manifest_base_uri}/graph.json"
    _publish_immutable(model_store / uri, payload, label="GLM-4.7 INT4 graph")
    return GraphArtifact(
        uri=uri,
        size_bytes=len(payload),
        content_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        ir_version="1.0.0",
        entry_points=("causal_lm",),
    )


def _output_layout(model_store: Path, plan_hash: str) -> dict[str, Any]:
    plan_id = plan_hash.removeprefix("sha256:")
    qualification_uri = f"qualification/glm47-int4/{plan_id}"
    manifest_base_uri = f"manifests/glm47-int4/{plan_id}"
    return {
        "cas_root": model_store / "cas",
        "cache_root": model_store / ".staging" / "glm47-int4" / plan_id,
        "journal_root": model_store / qualification_uri / "journal",
        "manifest_base_uri": manifest_base_uri,
        "manifest_uri": f"{manifest_base_uri}/manifest.json",
        "receipt_uri": f"{qualification_uri}/conversion.json",
    }


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    for name in ("buffer_bytes", "int4_block_bytes"):
        value = getattr(arguments, name)
        if isinstance(value, bool) or not 1 <= value <= 64 * 1024 * 1024:
            raise AmsError(ErrorCode.PLAN_INVALID, f"{name} is outside the reviewed bound")
    config_payload, architecture, index, catalog = _build_structural_catalog(
        arguments.asset_root,
        arguments.shard_root,
    )
    candidate, int4_config = _build_candidate(architecture, index, catalog)
    plan = build_huggingface_progressive_mixed_plan(catalog, candidate.assignments)
    if plan.policy_hash != _EXPECTED_POLICY_HASH:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "progressive policy hash drifted")
    target_bytes, identity_bytes, int4_bytes = _target_bytes(plan)
    model_store = arguments.model_store.resolve(strict=False)
    capacity = _capacity_evidence(
        model_store,
        arguments.shard_root,
        plan,
        target_bytes,
        arguments.safety_margin_bytes,
    )
    layout = _output_layout(model_store, plan.plan_hash)
    result: dict[str, Any] = {
        "schema_id": "ams.glm47-int4-conversion.v1",
        "status": "planned" if arguments.dry_run else "converted",
        "repository": bf16_probe._REPOSITORY,
        "revision": bf16_probe._REVISION,
        "architecture_hash": architecture.content_hash,
        "source_index_hash": index.content_hash,
        "source_root": plan.source_root,
        "source_file_bytes": catalog.audit.source_file_bytes,
        "source_tensor_bytes": catalog.audit.tensor_bytes,
        "candidate_hash": candidate.candidate_hash,
        "policy_hash": candidate.policy.policy_hash,
        "plan_hash": plan.plan_hash,
        "int4_config_hash": int4_config.config_hash,
        "numpy_version": _EXPECTED_NUMPY_VERSION,
        "encoding_counts": {encoding.value: count for encoding, count in candidate.encoding_counts},
        "identity_bytes": identity_bytes,
        "int4_bytes": int4_bytes,
        "target_bytes": target_bytes,
        "tensor_count": len(plan.tensors),
        "capacity": capacity,
        "model_store": str(model_store),
        "cas_uri": "cas",
        "manifest_uri": layout["manifest_uri"],
        "qualification_receipt_uri": layout["receipt_uri"],
        "source_deleted": False,
    }
    if arguments.dry_run:
        return result
    try:
        import numpy as np
    except ImportError as exc:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "complete INT4 conversion requires the conversion extra",
        ) from exc
    if np.__version__ != _EXPECTED_NUMPY_VERSION:
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH,
            "complete INT4 conversion requires the pinned NumPy version",
        )
    _prepare_directory(model_store, label="Andromeda model store")
    _prepare_directory(layout["cas_root"], label="Andromeda model CAS")
    encoder = partial(
        encode_int4_stream_numpy,
        maximum_source_read_bytes=arguments.int4_block_bytes,
    )
    with _ProgressReporter(layout["journal_root"], len(plan.tensors)):
        snapshot = execute_progressive_huggingface_mixed_conversion(
            catalog,
            plan,
            layout["cas_root"],
            layout["journal_root"],
            layout["cache_root"],
            buffer_bytes=arguments.buffer_bytes,
            int4_stream_encoder=encoder,
        )
    promoted_catalog, promoted_plan, promoted_journal = (
        finalize_progressive_huggingface_mixed_conversion(
            catalog,
            plan,
            layout["journal_root"],
        )
    )
    graph = _graph_artifact(
        model_store,
        layout["manifest_base_uri"],
        architecture,
        plan,
    )
    model_configuration = json.loads(config_payload)
    manifest = build_huggingface_mixed_manifest(
        promoted_catalog,
        promoted_plan,
        promoted_journal,
        graph,
        architecture="Glm4MoeLiteForCausalLM",
        model_configuration=model_configuration,
        default_dtype=DType.BFLOAT16,
        licenses=("MIT",),
        storage_uri_prefix="cas/chunks",
    )
    manifest_path = publish_manifest_last(
        model_store,
        manifest,
        manifest_uri=layout["manifest_uri"],
        buffer_bytes=arguments.buffer_bytes,
    )
    result.update(
        {
            "status": "converted",
            "verified_shard_count": len(snapshot.shards),
            "published_tensor_count": len(snapshot.tensors),
            "manifest_content_root": manifest["content_root"],
            "manifest_package_id": manifest["package_id"],
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": ("sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()),
        }
    )
    receipt_path = model_store / layout["receipt_uri"]
    _publish_immutable(
        receipt_path,
        canonical_json_bytes(result),
        label="GLM-4.7 INT4 conversion receipt",
    )
    return result


def main() -> int:
    arguments = _parser().parse_args()
    result = _run(arguments)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
