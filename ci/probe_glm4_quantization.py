"""Authenticate and sample one local GLM-4 shard against a committed precision candidate."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_mul, checked_product
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.descriptors import DType, StorageObject
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    Glm4LowBitDiagnosticConfig,
    Glm4MoeLiteTensorRole,
    Glm4QuantizationCodecVariant,
    Glm4QuantizationProbeConfig,
    HuggingFaceCatalogTensor,
    build_experimental_glm4_precision_candidate,
    compare_glm4_quantization_variants,
    expected_glm4_moe_lite_tensor_shape,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    probe_experimental_glm4_quantization_shard,
    validate_glm4_moe_lite_tensor_inventory,
)
from ams.storage import FileRangeStore

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024
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
    "repository",
    "revision",
    "source_bytes",
    "source_index_hash",
    "status",
    "ternary_config_hash",
    "tensor_count",
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
            "Full-hash one local official GLM-4 shard, then sample bounded groups with "
            "the exact experimental ternary/INT4 codecs"
        )
    )
    parser.add_argument("asset_root", type=Path, help="Pinned directory with config and index")
    parser.add_argument("shard", type=Path, help="One local official safetensors shard")
    parser.add_argument(
        "--candidate-evidence",
        required=True,
        type=Path,
        help="Committed structural precision-candidate JSON",
    )
    parser.add_argument("--repository", required=True, help="Exact owner/model identity")
    parser.add_argument("--revision", required=True, help="Exact 40-character commit")
    parser.add_argument("--expected-sha256", required=True, help="Exact shard SHA-256")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--groups-per-tensor", type=int, default=64)
    parser.add_argument("--hash-buffer-bytes", type=int, default=1024 * 1024)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-routed-experts",
        action="store_true",
        help=(
            "Compare fixed ternary thresholds, two-pass ternary residuals, and INT4 "
            "on identical routed-expert groups"
        ),
    )
    comparison.add_argument(
        "--compare-routed-expert-codecs",
        action="store_true",
        help=(
            "Compare diagnostic INT2/INT3 and sparse-residual ternary candidates, "
            "two-pass ternary, and INT4 on identical routed-expert groups"
        ),
    )
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


def _require_exact_evidence(
    evidence: dict[str, Any],
    *,
    repository: str,
    revision: str,
    candidate,
    group_size: int,
    header_tensor_count: int,
) -> None:
    expected_counts = {encoding.value: count for encoding, count in candidate.encoding_counts}
    expected = {
        "architecture_hash": candidate.architecture_hash,
        "encoding_counts": expected_counts,
        "estimated_encoded_bytes": candidate.estimated_encoded_bytes,
        "group_size": group_size,
        "int4_config_hash": next(
            assignment.int4_config.config_hash
            for assignment in candidate.assignments
            if assignment.int4_config is not None
        ),
        "policy_hash": candidate.policy.policy_hash,
        "repository": repository,
        "revision": revision,
        "source_bytes": candidate.source_bytes,
        "source_index_hash": candidate.source_index_hash,
        "status": candidate.status.value,
        "ternary_config_hash": next(
            assignment.ternary_config.config_hash
            for assignment in candidate.assignments
            if assignment.ternary_config is not None
        ),
        "tensor_count": header_tensor_count,
        "compression_ratio": candidate.source_bytes / candidate.estimated_encoded_bytes,
        "candidate_hash": candidate.candidate_hash,
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


def _run(arguments: argparse.Namespace):
    if _REPOSITORY.fullmatch(arguments.repository) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "repository must be one exact owner/model ID")
    if _REVISION.fullmatch(arguments.revision) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "revision must be one exact lowercase commit")
    if _SHA256.fullmatch(arguments.expected_sha256) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "expected shard SHA-256 is invalid")

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
    ternary_config = TernaryCodecConfig(group_size=arguments.group_size)
    int4_config = Int4CodecConfig(group_size=arguments.group_size)

    entry_by_name = {entry.tensor_name: entry for entry in index.entries}
    catalog_tensors = []
    source_offset = 0
    for slot in inventory.slots:
        shape = expected_glm4_moe_lite_tensor_shape(architecture, slot)
        if slot.role is Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS:
            dtype = DType.FLOAT32
            source_dtype = "F32"
            item_bytes = 4
        else:
            dtype = DType.BFLOAT16
            source_dtype = "BF16"
            item_bytes = 2
        source_length = checked_mul(
            checked_product(shape, name="glm4_probe_cli.tensor_elements"),
            item_bytes,
            name="glm4_probe_cli.tensor_bytes",
        )
        shard_name = entry_by_name[slot.tensor_name].shard_name
        catalog_tensors.append(
            HuggingFaceCatalogTensor(
                tensor_name=slot.tensor_name,
                shard_name=shard_name,
                object_id=f"hf:{shard_name}",
                dtype=dtype,
                source_dtype=source_dtype,
                shape=shape,
                source_offset=source_offset,
                source_length=source_length,
            )
        )
        source_offset += source_length
    candidate = build_experimental_glm4_precision_candidate(
        architecture,
        inventory,
        tuple(catalog_tensors),
        ternary_config=ternary_config,
        int4_config=int4_config,
    )
    candidate_evidence = _parse_candidate_evidence(arguments.candidate_evidence)
    _require_exact_evidence(
        candidate_evidence,
        repository=arguments.repository,
        revision=arguments.revision,
        candidate=candidate,
        group_size=arguments.group_size,
        header_tensor_count=len(inventory.slots),
    )

    try:
        shard_path = arguments.shard.resolve(strict=True)
        if arguments.shard.is_symlink() or not shard_path.is_file():
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                "probe shard is not a regular nonsymlink file",
            )
        shard_size = shard_path.stat().st_size
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "probe shard could not be inspected",
            retriable=True,
        ) from exc
    expected_hash = f"sha256:{arguments.expected_sha256}"
    reader = FileRangeStore(
        shard_path,
        StorageObject(
            object_id=f"hf-probe:{shard_path.name}",
            uri=str(shard_path),
            size_bytes=shard_size,
            alignment_bytes=1,
            content_hash=expected_hash,
        ),
    )
    probe_config = Glm4QuantizationProbeConfig(
        groups_per_tensor=arguments.groups_per_tensor,
        hash_buffer_bytes=arguments.hash_buffer_bytes,
    )
    if arguments.compare_routed_experts or arguments.compare_routed_expert_codecs:
        if arguments.compare_routed_expert_codecs:
            variants = (
                Glm4QuantizationCodecVariant(
                    variant_id="int2-symmetric-midrise",
                    encoding="int2_symmetric_midrise",
                    diagnostic_config=Glm4LowBitDiagnosticConfig(
                        encoding="int2_symmetric_midrise",
                        group_size=arguments.group_size,
                    ),
                ),
                Glm4QuantizationCodecVariant(
                    variant_id="int3-symmetric",
                    encoding="int3_symmetric",
                    diagnostic_config=Glm4LowBitDiagnosticConfig(
                        encoding="int3_symmetric",
                        group_size=arguments.group_size,
                    ),
                ),
                *(
                    Glm4QuantizationCodecVariant(
                        variant_id=f"ternary-threshold-08-sparse-bf16-k{residual_count:02d}",
                        encoding="ternary_sparse_bf16_residual",
                        diagnostic_config=Glm4LowBitDiagnosticConfig(
                            encoding="ternary_sparse_bf16_residual",
                            group_size=arguments.group_size,
                            threshold_numerator=8,
                            threshold_denominator=10,
                            residual_count=residual_count,
                        ),
                    )
                    for residual_count in (4, 8, 16)
                ),
                Glm4QuantizationCodecVariant(
                    variant_id="residual2-ternary-threshold-08-of-10",
                    encoding="ternary_residual2",
                    ternary_config=TernaryCodecConfig(
                        group_size=arguments.group_size,
                        threshold_numerator=8,
                        threshold_denominator=10,
                    ),
                ),
                Glm4QuantizationCodecVariant(
                    variant_id="int4-symmetric",
                    encoding="int4_symmetric",
                    int4_config=int4_config,
                ),
            )
        else:
            variants = (
                *(
                    Glm4QuantizationCodecVariant(
                        variant_id=f"ternary-threshold-{numerator:02d}-of-10",
                        encoding="ternary_trit5",
                        ternary_config=TernaryCodecConfig(
                            group_size=arguments.group_size,
                            threshold_numerator=numerator,
                            threshold_denominator=10,
                        ),
                    )
                    for numerator in range(3, 11)
                ),
                *(
                    Glm4QuantizationCodecVariant(
                        variant_id=f"residual2-ternary-threshold-{numerator:02d}-of-10",
                        encoding="ternary_residual2",
                        ternary_config=TernaryCodecConfig(
                            group_size=arguments.group_size,
                            threshold_numerator=numerator,
                            threshold_denominator=10,
                        ),
                    )
                    for numerator in (7, 8)
                ),
                Glm4QuantizationCodecVariant(
                    variant_id="int4-symmetric",
                    encoding="int4_symmetric",
                    int4_config=int4_config,
                ),
            )
        return compare_glm4_quantization_variants(
            architecture,
            inventory,
            index,
            source_repository=arguments.repository,
            source_revision=arguments.revision,
            shard_name=shard_path.name,
            reader=reader,
            expected_shard_hash=expected_hash,
            baseline_candidate_hash=candidate.candidate_hash,
            baseline_policy_hash=candidate.policy.policy_hash,
            selected_roles=(
                Glm4MoeLiteTensorRole.ROUTED_EXPERT_GATE_PROJECTION,
                Glm4MoeLiteTensorRole.ROUTED_EXPERT_UP_PROJECTION,
                Glm4MoeLiteTensorRole.ROUTED_EXPERT_DOWN_PROJECTION,
            ),
            variants=variants,
            config=probe_config,
        )
    return probe_experimental_glm4_quantization_shard(
        architecture,
        inventory,
        index,
        source_repository=arguments.repository,
        source_revision=arguments.revision,
        shard_name=shard_path.name,
        reader=reader,
        expected_shard_hash=expected_hash,
        candidate_hash=candidate.candidate_hash,
        policy_hash=candidate.policy.policy_hash,
        ternary_config=ternary_config,
        int4_config=int4_config,
        config=probe_config,
    )


def main() -> int:
    try:
        evidence = _run(_parser().parse_args())
    except AmsError as exc:
        print(json.dumps(exc.to_dict(), sort_keys=True), file=sys.stderr)
        return 2
    payload = json.loads(canonical_json_bytes(evidence))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
