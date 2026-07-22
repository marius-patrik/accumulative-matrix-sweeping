"""Canonical AMS directory manifests and manifest-last publication."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_positive
from ams.descriptors import (
    DType,
    JournalEntryState,
    StorageObject,
    validate_digest,
    validate_identifier,
    validate_semver,
)
from ams.errors import AmsError, ErrorCode
from ams.integrations.huggingface import HuggingFaceCatalog, HuggingFaceIdentityPlan
from ams.storage import FileRangeStore, hash_reader_range

_MANIFEST_NAME = "manifest.json"
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_ROOT_FEATURE = "ams.content-root.manifest-minus-root.v1"


def _validate_uri(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 8192:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{field} is invalid")
    return value


def _validate_relative_package_uri(value: str, *, field: str) -> str:
    _validate_uri(value, field=field)
    path = Path(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{field} is not a safe relative path")
    return value


@dataclass(frozen=True, slots=True)
class OperatorRequirement:
    op_name: str
    schema_version: str
    plugin_id: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.op_name, name="graph.operator.op_name")
        validate_semver(self.schema_version, name="graph.operator.schema_version")
        if self.plugin_id is not None:
            validate_identifier(self.plugin_id, name="graph.operator.plugin_id")


@dataclass(frozen=True, slots=True)
class GraphArtifact:
    uri: str
    size_bytes: int
    content_hash: str
    ir_version: str
    entry_points: tuple[str, ...]
    required_operators: tuple[OperatorRequirement, ...] = ()

    def __post_init__(self) -> None:
        _validate_relative_package_uri(self.uri, field="graph.uri")
        checked_positive(self.size_bytes, name="graph.size_bytes")
        validate_digest(self.content_hash, name="graph.content_hash")
        validate_semver(self.ir_version, name="graph.ir_version")
        if not self.entry_points or len(set(self.entry_points)) != len(self.entry_points):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "graph entry points are empty or duplicated")
        for entry_point in self.entry_points:
            validate_identifier(entry_point, name="graph.entry_point")


def _operator_dict(requirement: OperatorRequirement) -> dict[str, str]:
    value = {
        "op_name": requirement.op_name,
        "schema_version": requirement.schema_version,
    }
    if requirement.plugin_id is not None:
        value["plugin_id"] = requirement.plugin_id
    return value


def _manifest_root(manifest_without_root: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(manifest_without_root)).hexdigest()


def verify_manifest_content_root(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict) or "content_root" not in manifest:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "AMS manifest has no content root")
    expected = manifest["content_root"]
    validate_digest(expected, name="manifest.content_root")
    preimage = dict(manifest)
    del preimage["content_root"]
    if _manifest_root(preimage) != expected:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "AMS manifest content root mismatch")


def build_huggingface_identity_manifest(
    catalog: HuggingFaceCatalog,
    plan: HuggingFaceIdentityPlan,
    journal,
    graph: GraphArtifact,
    *,
    architecture: str,
    model_configuration: dict[str, Any],
    default_dtype: DType,
    licenses: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a schema-shaped manifest only after every planned chunk is published."""
    if plan.conversion.source_root != catalog.source_root:
        raise AmsError(ErrorCode.PLAN_INVALID, "catalog and conversion source roots differ")
    if (
        journal.source_root != plan.conversion.source_root
        or journal.configuration_hash != plan.conversion.configuration_hash
    ):
        raise AmsError(ErrorCode.PLAN_INVALID, "conversion journal does not match the plan")
    if not isinstance(architecture, str) or not 1 <= len(architecture) <= 512:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "model architecture is invalid")
    default_dtype = DType(default_dtype)
    normalized_configuration = json.loads(canonical_json_bytes(model_configuration))
    if not isinstance(normalized_configuration, dict):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "model configuration must be an object")
    for license_name in licenses:
        if not isinstance(license_name, str) or len(license_name) > 1024:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "model license value is invalid")

    entry_by_id = {entry.target_chunk_id: entry for entry in journal.entries}
    if len(entry_by_id) != len(journal.entries):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal chunk IDs are duplicated")
    expected_ids = {item.target_chunk_id for item in plan.conversion.items}
    if set(entry_by_id) != expected_ids:
        raise AmsError(ErrorCode.PLAN_INVALID, "conversion journal chunk set differs from the plan")

    storage_by_hash: dict[str, dict[str, Any]] = {}
    tensors: list[dict[str, Any]] = []
    for planned in plan.tensors:
        tensor = planned.tensor
        if planned.target_chunk_id is None or tensor.source_length == 0 or 0 in tensor.shape:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "identity manifest v1 cannot encode zero-sized tensors",
                evidence={"tensor_name": tensor.tensor_name},
            )
        entry = entry_by_id[planned.target_chunk_id]
        if (
            entry.state is not JournalEntryState.PUBLISHED
            or entry.target_hash != planned.source_checksum
            or entry.encoded_bytes != tensor.source_length
        ):
            raise AmsError(
                ErrorCode.TRANSACTION_FAILURE,
                "tensor chunk has not reached a verified published state",
                evidence={"tensor_name": tensor.tensor_name},
            )
        algorithm, hexdigest = planned.source_checksum.split(":", 1)
        object_id = f"object:{hexdigest}"
        chunk_uri = f"chunks/{algorithm}-{hexdigest}.bin"
        storage_by_hash.setdefault(
            planned.source_checksum,
            {
                "object_id": object_id,
                "uri": chunk_uri,
                "size_bytes": tensor.source_length,
                "alignment_bytes": 1,
                "content_hash": planned.source_checksum,
                "immutable": True,
                "kind": "tensor_data",
            },
        )
        tensors.append(
            {
                "tensor_id": planned.target_chunk_id,
                "tensor_class": "parameter",
                "shape": list(tensor.shape),
                "logical_dtype": tensor.dtype.value,
                "byte_order": "little",
                "immutable": True,
                "layouts": [
                    {
                        "layout_id": "layout:identity.v1",
                        "layout_version": "1.0.0",
                        "complete": True,
                        "tile_shape": list(tensor.shape),
                        "alignment_bytes": 1,
                        "storage_dtype": tensor.dtype.value,
                        "chunks": [
                            {
                                "chunk_id": planned.target_chunk_id,
                                "range": {
                                    "object_id": object_id,
                                    "offset": 0,
                                    "length": tensor.source_length,
                                    "checksum": planned.source_checksum,
                                },
                                "logical_origin": [0] * len(tensor.shape),
                                "logical_extent": list(tensor.shape),
                                "encoded_bytes": tensor.source_length,
                                "decoded_bytes": tensor.source_length,
                            }
                        ],
                        "extensions": {"ams.encoding": "identity"},
                    }
                ],
                "extensions": {
                    "hf.shard-name": tensor.shard_name,
                    "hf.source-dtype": tensor.source_dtype,
                    "hf.source-name": tensor.tensor_name,
                },
            }
        )

    graph_hexdigest = graph.content_hash.split(":", 1)[1]
    graph_object = {
        "object_id": f"graph:{graph_hexdigest}",
        "uri": graph.uri,
        "size_bytes": graph.size_bytes,
        "alignment_bytes": 1,
        "content_hash": graph.content_hash,
        "immutable": True,
        "kind": "graph",
    }
    identity = {
        "source_root": catalog.source_root,
        "configuration_hash": plan.conversion.configuration_hash,
        "graph_hash": graph.content_hash,
        "architecture": architecture,
        "model_configuration": normalized_configuration,
        "tensors": [
            {
                "tensor_id": tensor["tensor_id"],
                "checksum": tensor["layouts"][0]["chunks"][0]["range"]["checksum"],
            }
            for tensor in sorted(tensors, key=lambda item: item["tensor_id"])
        ],
    }
    package_id = "package:" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    manifest_without_root: dict[str, Any] = {
        "schema_id": "ams.model.manifest",
        "format_version": {"major": 1, "minor": 0},
        "package_id": package_id,
        "required_features": [_ROOT_FEATURE, "ams.identity-layout.v1"],
        "optional_features": [],
        "graph": {
            "uri": graph.uri,
            "content_hash": graph.content_hash,
            "ir_version": graph.ir_version,
            "entry_points": list(graph.entry_points),
            "required_operators": [
                _operator_dict(requirement) for requirement in graph.required_operators
            ],
        },
        "model": {
            "architecture": architecture,
            "configuration": normalized_configuration,
            "default_dtype": default_dtype.value,
        },
        "storage_objects": [
            graph_object,
            *[storage_by_hash[key] for key in sorted(storage_by_hash)],
        ],
        "tensors": sorted(tensors, key=lambda item: item["tensor_id"]),
        "integrity": {
            "hash_algorithm": "sha256",
            "canonicalization": "ams.canonical-json.v1",
        },
        "provenance": {
            "tool": "ams.convert.huggingface",
            "tool_version": "0.1.0-dev.0",
            "source_artifacts": [
                {
                    "uri": "model.safetensors.index.json",
                    "content_hash": catalog.index_content_hash,
                },
                *[
                    {"uri": source.shard_name, "content_hash": source.content_hash}
                    for source in catalog.sources
                ],
            ],
            "configuration_hash": plan.conversion.configuration_hash,
            "licenses": list(licenses),
        },
        "extensions": {
            "ams.root-preimage": "manifest-minus-content-root",
            "ams.source-root": catalog.source_root,
            "hf.index-metadata-hash": catalog.index_metadata_hash,
        },
    }
    return {
        **manifest_without_root,
        "content_root": _manifest_root(manifest_without_root),
    }


def _resolve_package_file(root: Path, uri: str) -> Path:
    _validate_relative_package_uri(uri, field="storage_object.uri")
    try:
        unresolved = root / Path(uri)
        current = root
        for part in Path(uri).parts:
            current /= part
            if current.is_symlink():
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE,
                    "storage object path contains a symlink",
                )
        candidate = unresolved.resolve(strict=True)
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "declared package storage object is unavailable",
            retriable=True,
        ) from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE, "storage object escapes the package root"
        ) from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise AmsError(ErrorCode.INVALID_PACKAGE, "storage object is not a regular file")
    return candidate


def publish_manifest_last(
    package_root: Path,
    manifest: dict[str, Any],
    *,
    buffer_bytes: int = 1024 * 1024,
    max_manifest_bytes: int = _MAX_MANIFEST_BYTES,
) -> Path:
    """Verify every declared local object, then atomically expose the immutable manifest."""
    if (
        isinstance(max_manifest_bytes, bool)
        or not isinstance(max_manifest_bytes, int)
        or max_manifest_bytes <= 0
    ):
        raise AmsError(ErrorCode.PLAN_INVALID, "manifest size limit must be positive")
    verify_manifest_content_root(manifest)
    try:
        if package_root.is_symlink():
            raise AmsError(ErrorCode.INVALID_PACKAGE, "package root cannot be a symlink")
        root = package_root.resolve(strict=True)
        if not root.is_dir():
            raise AmsError(ErrorCode.INVALID_PACKAGE, "package root is not a directory")
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, "package root is unavailable", retriable=True) from exc
    storage_objects = manifest.get("storage_objects")
    if not isinstance(storage_objects, list) or not storage_objects:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "manifest storage object list is invalid")
    for storage_object in storage_objects:
        if not isinstance(storage_object, dict):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "storage object descriptor is invalid")
        try:
            uri = storage_object["uri"]
            size_bytes = storage_object["size_bytes"]
            content_hash = storage_object["content_hash"]
        except KeyError as exc:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "storage object fields are missing") from exc
        checked_positive(size_bytes, name="storage_object.size_bytes")
        validate_digest(content_hash, name="storage_object.content_hash")
        path = _resolve_package_file(root, uri)
        try:
            if path.stat().st_size != size_bytes:
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "storage object size changed before manifest publication",
                    evidence={"uri": uri},
                )
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "storage object changed during manifest verification",
                retriable=True,
            ) from exc
        algorithm = content_hash.split(":", 1)[0]
        reader = FileRangeStore(
            path,
            StorageObject(
                object_id=storage_object.get("object_id"),
                uri=uri,
                size_bytes=size_bytes,
                alignment_bytes=storage_object.get("alignment_bytes"),
                content_hash=content_hash,
                immutable=True,
                kind=storage_object.get("kind", "other"),
            ),
        )
        actual = hash_reader_range(
            reader,
            0,
            size_bytes,
            buffer_bytes=buffer_bytes,
            algorithm=algorithm,
        )
        if actual != content_hash:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "storage object hash changed before manifest publication",
                evidence={"uri": uri},
            )

    payload = canonical_json_bytes(manifest)
    if len(payload) > max_manifest_bytes:
        raise AmsError(ErrorCode.TRANSACTION_FAILURE, "manifest exceeds its size limit")
    final_path = root / _MANIFEST_NAME
    temporary = root / f"{_MANIFEST_NAME}.tmp"
    if final_path.exists():
        try:
            if final_path.is_symlink() or not final_path.is_file():
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "published manifest is not a regular file",
                )
            if final_path.stat().st_size > max_manifest_bytes:
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "published manifest exceeds its size limit",
                )
            existing = final_path.read_bytes()
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "published manifest could not be verified",
                retriable=True,
            ) from exc
        if existing != payload:
            raise AmsError(
                ErrorCode.TRANSACTION_FAILURE,
                "an immutable package manifest is already published with different content",
            )
        return final_path
    try:
        with temporary.open("wb", buffering=0) as handle:
            written = 0
            while written < len(payload):
                count = handle.write(payload[written:])
                if count is None or count == 0:
                    raise AmsError(
                        ErrorCode.IO_FAILURE,
                        "short write to package manifest",
                        retriable=True,
                    )
                written += count
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final_path)
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.TRANSACTION_FAILURE,
            "package manifest publication failed",
            retriable=True,
        ) from exc
    return final_path
