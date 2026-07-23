"""Audit one pinned GLM-4 precision candidate from remote safetensors headers only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.integrations import (
    Glm4PrecisionProfile,
    HuggingFaceCatalogPolicy,
    HuggingFaceShardSource,
    HuggingFaceTotalSizeSemantics,
    build_accuracy_first_glm4_precision_candidate,
    build_experimental_glm4_precision_candidate,
    build_huggingface_header_catalog,
    parse_glm4_moe_lite_architecture,
    parse_huggingface_shard_index,
    validate_glm4_moe_lite_tensor_inventory,
)
from ams.storage import HttpRangeReader

_MAX_API_BYTES = 16 * 1024 * 1024
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Range-read only safetensors prefixes/headers and emit a structural mixed-policy audit"
        )
    )
    parser.add_argument("root", type=Path, help="Pinned GLM-4 asset directory")
    parser.add_argument("--repository", required=True, help="Exact Hugging Face owner/model ID")
    parser.add_argument("--revision", required=True, help="Exact 40-character commit")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in Glm4PrecisionProfile),
        default=Glm4PrecisionProfile.TERNARY_CAPACITY.value,
        help="Exact reviewed role-assignment profile",
    )
    return parser


def _fetch_model_metadata(repository: str, revision: str) -> dict[str, Any]:
    if _REPOSITORY.fullmatch(repository) is None:
        raise RuntimeError("repository must be one exact owner/model ID")
    if _REVISION.fullmatch(revision) is None:
        raise RuntimeError("revision must be one exact lowercase commit")
    repository_path = quote(repository, safe="/")
    revision_path = quote(revision, safe="")
    url = f"https://huggingface.co/api/models/{repository_path}/revision/{revision_path}?blobs=true"
    request = Request(url, headers={"Accept-Encoding": "identity"})
    with urlopen(request, timeout=30) as response:
        payload = response.read(_MAX_API_BYTES + 1)
    if len(payload) > _MAX_API_BYTES:
        raise RuntimeError("Hugging Face metadata response exceeds the audit bound")
    try:
        metadata = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Hugging Face metadata response is invalid JSON") from exc
    if not isinstance(metadata, dict) or metadata.get("sha") != revision:
        raise RuntimeError("Hugging Face metadata does not identify the pinned revision")
    return metadata


def _normalize_siblings(metadata: dict[str, Any]) -> dict[str, tuple[int, str]]:
    raw_siblings = metadata.get("siblings")
    if not isinstance(raw_siblings, list):
        raise RuntimeError("Hugging Face metadata has no sibling inventory")
    siblings: dict[str, tuple[int, str]] = {}
    for item in raw_siblings:
        if not isinstance(item, dict):
            raise RuntimeError("Hugging Face sibling metadata is malformed")
        name = item.get("rfilename")
        lfs = item.get("lfs")
        if not isinstance(name, str) or not isinstance(lfs, dict):
            continue
        size = lfs.get("size")
        digest = lfs.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or name in siblings
        ):
            raise RuntimeError("Hugging Face LFS sibling metadata is malformed")
        siblings[name] = (size, digest)
    return siblings


def main() -> int:
    arguments = _parser().parse_args()
    index_path = arguments.root / "model.safetensors.index.json"
    config_path = arguments.root / "config.json"
    index = parse_huggingface_shard_index(index_path.read_bytes())
    architecture = parse_glm4_moe_lite_architecture(config_path.read_bytes())
    inventory = validate_glm4_moe_lite_tensor_inventory(architecture, index)

    metadata = _fetch_model_metadata(arguments.repository, arguments.revision)
    siblings = _normalize_siblings(metadata)
    repository_path = quote(arguments.repository, safe="/")
    revision_path = quote(arguments.revision, safe="")
    sources = []
    for shard_name in index.shard_names:
        if shard_name not in siblings:
            raise RuntimeError(f"pinned shard is absent from Hugging Face metadata: {shard_name}")
        size_bytes, digest = siblings[shard_name]
        shard_path = quote(shard_name, safe="")
        url = f"https://huggingface.co/{repository_path}/resolve/{revision_path}/{shard_path}"
        sources.append(
            HuggingFaceShardSource(
                shard_name=shard_name,
                object_id=f"hf:{shard_name}",
                content_hash=f"sha256:{digest}",
                reader=HttpRangeReader(url, size_bytes),
            )
        )
    catalog = build_huggingface_header_catalog(
        index,
        tuple(sources),
        policy=HuggingFaceCatalogPolicy(
            total_size_semantics=HuggingFaceTotalSizeSemantics.TENSOR_ELEMENTS,
            expected_index_content_hash=index.content_hash,
        ),
    )
    ternary_config = TernaryCodecConfig(group_size=arguments.group_size)
    int4_config = Int4CodecConfig(group_size=arguments.group_size)
    profile = Glm4PrecisionProfile(arguments.profile)
    if profile is Glm4PrecisionProfile.INT4_BRINGUP:
        candidate = build_accuracy_first_glm4_precision_candidate(
            architecture,
            inventory,
            catalog.tensors,
            int4_config=int4_config,
        )
    else:
        candidate = build_experimental_glm4_precision_candidate(
            architecture,
            inventory,
            catalog.tensors,
            ternary_config=ternary_config,
            int4_config=int4_config,
        )
    output = {
        "architecture_hash": candidate.architecture_hash,
        "candidate_hash": candidate.candidate_hash,
        "compression_ratio": candidate.source_bytes / candidate.estimated_encoded_bytes,
        "encoding_counts": {encoding.value: count for encoding, count in candidate.encoding_counts},
        "estimated_encoded_bytes": candidate.estimated_encoded_bytes,
        "group_size": arguments.group_size,
        "header_bytes_read": catalog.audit.prefix_and_header_bytes,
        "int4_config_hash": int4_config.config_hash,
        "policy_hash": candidate.policy.policy_hash,
        "profile": profile.value,
        "repository": arguments.repository,
        "revision": arguments.revision,
        "source_bytes": candidate.source_bytes,
        "source_index_hash": candidate.source_index_hash,
        "status": candidate.status.value,
        "ternary_config_hash": (
            ternary_config.config_hash if profile is Glm4PrecisionProfile.TERNARY_CAPACITY else None
        ),
        "tensor_count": len(candidate.assignments),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
