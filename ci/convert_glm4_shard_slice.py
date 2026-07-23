"""Convert an explicit authenticated GLM-4.7 shard tensor subset under one candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from ams.checked import checked_add
from ams.codecs import Int4CodecConfig
from ams.descriptors import DType, JournalEntryState, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4PrecisionProfile,
    HuggingFaceShardSource,
    HuggingFaceTensorEncoding,
    build_accuracy_first_glm4_precision_candidate,
    build_huggingface_mixed_plan,
    build_huggingface_shard_catalog,
    derive_expected_glm4_catalog_tensors,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    validate_glm4_moe_lite_tensor_inventory,
)
from ams.mixed_conversion import execute_huggingface_mixed_conversion
from ams.storage import FileRangeStore

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024
_MAX_SELECTED_TENSORS = 256
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_FIELDS = {
    "architecture_hash",
    "candidate_hash",
    "compression_ratio",
    "encoding_counts",
    "estimated_encoded_bytes",
    "group_size",
    "header_bytes_read",
    "int4_config_hash",
    "policy_hash",
    "profile",
    "repository",
    "revision",
    "source_bytes",
    "source_index_hash",
    "status",
    "tensor_count",
    "ternary_config_hash",
}
_SOURCE_ITEM_BYTES = {
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.FLOAT32: 4,
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
        description=(
            "Authenticate one official GLM-4 shard and convert only explicitly named "
            "tensors under the reviewed INT4 bring-up candidate"
        )
    )
    parser.add_argument("asset_root", type=Path, help="Pinned config and full shard index")
    parser.add_argument("shard", type=Path, help="One local official safetensors shard")
    parser.add_argument("output_root", type=Path, help="Restart-safe slice output directory")
    parser.add_argument(
        "--candidate-evidence",
        required=True,
        type=Path,
        help="Committed int4_bringup_v1 candidate evidence",
    )
    parser.add_argument("--repository", required=True, help="Exact owner/model identity")
    parser.add_argument("--revision", required=True, help="Exact 40-character commit")
    parser.add_argument("--expected-sha256", required=True, help="Exact shard SHA-256")
    parser.add_argument(
        "--tensor-name",
        action="append",
        required=True,
        help="Exact indexed tensor name; repeat for an explicit bounded subset",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--buffer-bytes", type=int, default=1024 * 1024)
    return parser


def _read_bounded(path: Path, maximum_bytes: int, *, label: str) -> bytes:
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
                f"{label} size is outside the configured bound",
                evidence={"size_bytes": size, "maximum_bytes": maximum_bytes},
            )
        return resolved.read_bytes()
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            f"{label} could not be read",
            retriable=True,
        ) from exc


def _parse_candidate_evidence(path: Path) -> dict[str, Any]:
    payload = _read_bounded(path, _MAX_EVIDENCE_BYTES, label="candidate evidence")
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, ValueError) as exc:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "candidate evidence is not strict JSON",
        ) from exc
    if not isinstance(value, dict) or set(value) != _CANDIDATE_FIELDS:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "candidate evidence fields do not match the reviewed schema",
        )
    return value


def _require_candidate_evidence(
    evidence: dict[str, Any],
    *,
    repository: str,
    revision: str,
    group_size: int,
    candidate,
) -> None:
    expected = {
        "architecture_hash": candidate.architecture_hash,
        "candidate_hash": candidate.candidate_hash,
        "compression_ratio": candidate.source_bytes / candidate.estimated_encoded_bytes,
        "encoding_counts": {encoding.value: count for encoding, count in candidate.encoding_counts},
        "estimated_encoded_bytes": candidate.estimated_encoded_bytes,
        "group_size": group_size,
        "int4_config_hash": next(
            assignment.int4_config.config_hash
            for assignment in candidate.assignments
            if assignment.int4_config is not None
        ),
        "policy_hash": candidate.policy.policy_hash,
        "profile": Glm4PrecisionProfile.INT4_BRINGUP.value,
        "repository": repository,
        "revision": revision,
        "source_bytes": candidate.source_bytes,
        "source_index_hash": candidate.source_index_hash,
        "status": candidate.status.value,
        "tensor_count": len(candidate.assignments),
        "ternary_config_hash": None,
    }
    for field, expected_value in expected.items():
        if evidence.get(field) != expected_value:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                f"candidate evidence field is inconsistent: {field}",
                evidence={
                    "declared": str(evidence.get(field)),
                    "derived": str(expected_value),
                },
            )
    header_bytes_read = evidence.get("header_bytes_read")
    if (
        isinstance(header_bytes_read, bool)
        or not isinstance(header_bytes_read, int)
        or header_bytes_read <= 0
    ):
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            "candidate header audit byte count is invalid",
        )


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    if _REPOSITORY.fullmatch(arguments.repository) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "repository must be one exact owner/model ID")
    if _REVISION.fullmatch(arguments.revision) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "revision must be one exact lowercase commit")
    if _SHA256.fullmatch(arguments.expected_sha256) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "expected shard SHA-256 is invalid")
    if (
        isinstance(arguments.buffer_bytes, bool)
        or not isinstance(arguments.buffer_bytes, int)
        or arguments.buffer_bytes <= 0
        or arguments.buffer_bytes > 64 * 1024 * 1024
    ):
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "buffer bytes must be in [1, 67108864]",
        )
    selected_names = tuple(arguments.tensor_name)
    if (
        not selected_names
        or len(selected_names) > _MAX_SELECTED_TENSORS
        or len(set(selected_names)) != len(selected_names)
    ):
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "selected tensor names must be unique and contain at most 256 entries",
        )

    config_bytes = _read_bounded(
        arguments.asset_root / "config.json",
        _MAX_CONFIG_BYTES,
        label="GLM-4 config",
    )
    index_bytes = _read_bounded(
        arguments.asset_root / "model.safetensors.index.json",
        _MAX_INDEX_BYTES,
        label="GLM-4 shard index",
    )
    architecture = parse_glm4_moe_lite_architecture(config_bytes)
    index = parse_huggingface_shard_index(index_bytes)
    inventory = validate_glm4_moe_lite_tensor_inventory(architecture, index)
    expected_tensors = derive_expected_glm4_catalog_tensors(
        architecture,
        inventory,
        index,
    )
    int4_config = Int4CodecConfig(group_size=arguments.group_size)
    candidate = build_accuracy_first_glm4_precision_candidate(
        architecture,
        inventory,
        expected_tensors,
        int4_config=int4_config,
    )
    candidate_evidence = _parse_candidate_evidence(arguments.candidate_evidence)
    _require_candidate_evidence(
        candidate_evidence,
        repository=arguments.repository,
        revision=arguments.revision,
        group_size=arguments.group_size,
        candidate=candidate,
    )

    try:
        if arguments.shard.is_symlink():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "source shard is a symbolic link")
        shard_path = arguments.shard.resolve(strict=True)
        if not shard_path.is_file():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, "source shard is not a regular file")
        shard_size = shard_path.stat().st_size
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "source shard could not be inspected",
            retriable=True,
        ) from exc
    expected_hash = f"sha256:{arguments.expected_sha256}"
    object_id = f"hf-slice:{shard_path.name}"
    reader = FileRangeStore(
        shard_path,
        StorageObject(
            object_id=object_id,
            uri=str(shard_path),
            size_bytes=shard_size,
            alignment_bytes=1,
            content_hash=expected_hash,
        ),
    )
    source = HuggingFaceShardSource(
        shard_name=shard_path.name,
        object_id=object_id,
        content_hash=expected_hash,
        reader=reader,
    )
    shard_catalog = build_huggingface_shard_catalog(
        index,
        source,
        buffer_bytes=arguments.buffer_bytes,
    )
    expected_by_name = {tensor.tensor_name: tensor for tensor in expected_tensors}
    for tensor in shard_catalog.tensors:
        expected = expected_by_name[tensor.tensor_name]
        if (
            tensor.shard_name != expected.shard_name
            or tensor.dtype is not expected.dtype
            or tensor.source_dtype != expected.source_dtype
            or tensor.shape != expected.shape
            or tensor.source_length != expected.source_length
        ):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "authenticated shard tensor differs from the reviewed inventory: "
                f"{tensor.tensor_name}",
            )

    shard_by_name = {tensor.tensor_name: tensor for tensor in shard_catalog.tensors}
    missing = set(selected_names) - set(shard_by_name)
    if missing:
        raise AmsError(
            ErrorCode.PLAN_INVALID,
            "selected tensor is absent from the authenticated shard",
            evidence={"missing": len(missing)},
        )
    selected_set = set(selected_names)
    selected_tensors = tuple(
        tensor for tensor in shard_catalog.tensors if tensor.tensor_name in selected_set
    )
    selected_catalog = replace(
        shard_catalog,
        total_size=sum(tensor.source_length for tensor in selected_tensors),
        tensors=selected_tensors,
    )
    full_assignment_by_name = {
        assignment.tensor_name: assignment for assignment in candidate.assignments
    }
    assignments = tuple(full_assignment_by_name[tensor.tensor_name] for tensor in selected_tensors)
    plan = build_huggingface_mixed_plan(
        selected_catalog,
        assignments,
        buffer_bytes=arguments.buffer_bytes,
    )
    output_root = arguments.output_root.resolve(strict=False)
    journal = execute_huggingface_mixed_conversion(
        selected_catalog,
        plan,
        output_root,
        output_root / "conversion.journal.json",
        verification_buffer_bytes=arguments.buffer_bytes,
    )
    journal_by_id = {entry.target_chunk_id: entry for entry in journal.entries}
    source_bytes = 0
    encoded_bytes = 0
    maximum_codec_source_read_bytes = 0
    outputs = []
    for planned in plan.tensors:
        entry = journal_by_id[planned.target_chunk_id]
        if (
            entry.state is not JournalEntryState.PUBLISHED
            or entry.target_hash is None
            or entry.encoded_bytes is None
        ):
            raise AmsError(
                ErrorCode.TRANSACTION_FAILURE,
                "slice conversion did not publish every selected tensor",
            )
        source_bytes = checked_add(
            source_bytes,
            planned.tensor.source_length,
            name="glm4_slice.source_bytes",
        )
        encoded_bytes = checked_add(
            encoded_bytes,
            entry.encoded_bytes,
            name="glm4_slice.encoded_bytes",
        )
        if planned.encoding is not HuggingFaceTensorEncoding.IDENTITY:
            maximum_codec_source_read_bytes = max(
                maximum_codec_source_read_bytes,
                int4_config.group_size * _SOURCE_ITEM_BYTES[planned.tensor.dtype],
            )
        outputs.append(
            {
                "tensor_name": planned.tensor.tensor_name,
                "encoding": planned.encoding.value,
                "source_bytes": planned.tensor.source_length,
                "source_checksum": planned.source_checksum,
                "encoded_bytes": entry.encoded_bytes,
                "target_hash": entry.target_hash,
            }
        )
    return {
        "schema_id": "ams.glm4.shard-slice-conversion.v1",
        "status": "diagnostic",
        "qualifies_precision_policy": False,
        "publishes_model_manifest": False,
        "execution_scope": "authenticated_shard_tensor_subset",
        "repository": arguments.repository,
        "revision": arguments.revision,
        "source_index_hash": index.content_hash,
        "candidate_hash": candidate.candidate_hash,
        "full_policy_hash": candidate.policy.policy_hash,
        "slice_policy_hash": plan.policy_hash,
        "shard_name": source.shard_name,
        "shard_content_hash": source.content_hash,
        "shard_file_bytes": source.reader.size_bytes,
        "source_root": selected_catalog.source_root,
        "tensor_count": len(outputs),
        "source_tensor_bytes": source_bytes,
        "encoded_tensor_bytes": encoded_bytes,
        "compression_ratio": source_bytes / encoded_bytes,
        "group_size": int4_config.group_size,
        "buffer_bytes": arguments.buffer_bytes,
        "maximum_codec_source_read_bytes": maximum_codec_source_read_bytes,
        "outputs": outputs,
    }


def main() -> int:
    try:
        evidence = _run(_parser().parse_args())
    except AmsError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
