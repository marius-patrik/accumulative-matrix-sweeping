"""Build one metadata-only GLM-5.2 precision candidate from pinned source evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    GlmTensorRole,
    HuggingFaceCatalogTensor,
    build_experimental_glm_precision_candidate,
    expected_glm_tensor_shape,
    parse_glm_moe_dsa_architecture,
    parse_huggingface_shard_index,
    validate_glm_tensor_inventory,
)

_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024
_SOURCE_EVIDENCE_FIELDS = {
    "architecture",
    "architecture_hash",
    "assets",
    "declared_total_size",
    "dtype_counts",
    "header_bytes_read",
    "index_hash",
    "index_metadata_hash",
    "qualifies_precision_policy",
    "repository",
    "revision",
    "shard_count",
    "shard_inventory_hash",
    "source_file_bytes",
    "source_root",
    "status",
    "tensor_bytes",
    "tensor_count",
    "tensor_elements",
    "weight_payload_bytes_read",
}


class _DuplicateKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive a deterministic, explicitly non-qualified GLM-5.2 storage candidate"
    )
    parser.add_argument("root", type=Path, help="Pinned GLM-5.2 control-asset directory")
    parser.add_argument(
        "--source-evidence",
        type=Path,
        required=True,
        help="Committed GLM-5.2 structural source audit",
    )
    parser.add_argument("--group-size", type=int, default=128)
    return parser


def _read_bounded(path: Path, maximum_bytes: int, *, label: str) -> bytes:
    try:
        if path.is_symlink():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is a symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"{label} is not a regular file")
        size = resolved.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, f"{label} size is outside its bound")
        return resolved.read_bytes()
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(ErrorCode.IO_FAILURE, f"{label} could not be read", retriable=True) from exc


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, ValueError) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{label} must be a JSON object")
    return value


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    source_evidence = _strict_json_object(
        _read_bounded(
            arguments.source_evidence,
            _MAX_EVIDENCE_BYTES,
            label="GLM-5.2 source evidence",
        ),
        label="GLM-5.2 source evidence",
    )
    if set(source_evidence) != _SOURCE_EVIDENCE_FIELDS:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "GLM-5.2 source evidence fields changed")
    if (
        source_evidence["status"] != "structural_headers_only"
        or source_evidence["qualifies_precision_policy"] is not False
        or source_evidence["weight_payload_bytes_read"] != 0
    ):
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "GLM-5.2 source evidence status is invalid")

    root = arguments.root.resolve(strict=True)
    config_bytes = _read_bounded(
        root / "config.json",
        _MAX_CONFIG_BYTES,
        label="GLM-5.2 config",
    )
    index_bytes = _read_bounded(
        root / "model.safetensors.index.json",
        _MAX_INDEX_BYTES,
        label="GLM-5.2 index",
    )
    architecture = parse_glm_moe_dsa_architecture(config_bytes)
    index = parse_huggingface_shard_index(index_bytes)
    inventory = validate_glm_tensor_inventory(architecture, index)
    if (
        source_evidence["architecture_hash"] != architecture.content_hash
        or source_evidence["index_hash"] != index.content_hash
        or source_evidence["tensor_count"] != len(inventory.slots)
    ):
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "GLM-5.2 source evidence does not identify these assets",
        )

    shard_by_tensor = {entry.tensor_name: entry.shard_name for entry in index.entries}
    tensors = []
    for slot in inventory.slots:
        shape = expected_glm_tensor_shape(architecture, slot)
        is_f32 = slot.role is GlmTensorRole.ROUTER_CORRECTION_BIAS
        elements = 1
        for dimension in shape:
            elements *= dimension
        shard_name = shard_by_tensor[slot.tensor_name]
        tensors.append(
            HuggingFaceCatalogTensor(
                tensor_name=slot.tensor_name,
                shard_name=shard_name,
                object_id=f"hf:{shard_name}",
                dtype=DType.FLOAT32 if is_f32 else DType.BFLOAT16,
                source_dtype="F32" if is_f32 else "BF16",
                shape=shape,
                source_offset=0,
                source_length=elements * (4 if is_f32 else 2),
            )
        )
    ternary_config = TernaryCodecConfig(group_size=arguments.group_size)
    int4_config = Int4CodecConfig(group_size=arguments.group_size)
    candidate = build_experimental_glm_precision_candidate(
        architecture,
        inventory,
        tuple(tensors),
        ternary_config=ternary_config,
        int4_config=int4_config,
    )
    if candidate.source_bytes != source_evidence["tensor_bytes"]:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "GLM-5.2 candidate source bytes disagree with the audited headers",
        )
    return {
        "architecture_hash": candidate.architecture_hash,
        "candidate_hash": candidate.candidate_hash,
        "compression_ratio": candidate.source_bytes / candidate.estimated_encoded_bytes,
        "encoding_counts": {encoding.value: count for encoding, count in candidate.encoding_counts},
        "estimated_encoded_bytes": candidate.estimated_encoded_bytes,
        "group_size": arguments.group_size,
        "int4_config_hash": int4_config.config_hash,
        "policy_hash": candidate.policy.policy_hash,
        "qualifies_precision_policy": False,
        "repository": source_evidence["repository"],
        "revision": source_evidence["revision"],
        "schema_id": "ams.glm.precision-candidate.v1",
        "shard_inventory_hash": source_evidence["shard_inventory_hash"],
        "source_audit_hash": "sha256:"
        + hashlib.sha256(canonical_json_bytes(source_evidence)).hexdigest(),
        "source_bytes": candidate.source_bytes,
        "source_index_hash": candidate.source_index_hash,
        "source_root": source_evidence["source_root"],
        "status": candidate.status.value,
        "tensor_count": len(candidate.assignments),
        "ternary_config_hash": ternary_config.config_hash,
    }


def main() -> int:
    try:
        evidence = _run(_parser().parse_args())
    except (AmsError, OSError) as exc:
        if isinstance(exc, AmsError):
            payload = exc.to_dict()
        else:
            payload = {
                "error": {
                    "code": ErrorCode.IO_FAILURE.value,
                    "message": "GLM-5.2 precision audit failed",
                    "retriable": True,
                }
            }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
