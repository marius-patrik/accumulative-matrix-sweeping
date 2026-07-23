"""Audit one pinned GLM-5.2 source revision without downloading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from ams.canonical import canonical_json_bytes
from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    build_huggingface_header_catalog,
    parse_glm_moe_dsa_architecture,
    parse_huggingface_shard_index,
    validate_glm_tensor_inventory,
)
from ams.storage import HttpRangeReader

_MAX_API_BYTES = 16 * 1024 * 1024
_MAX_ASSET_BYTES = {
    "LICENSE": 1024 * 1024,
    "README.md": 4 * 1024 * 1024,
    "chat_template.jinja": 1024 * 1024,
    "config.json": 1024 * 1024,
    "generation_config.json": 1024 * 1024,
    "model.safetensors.index.json": 64 * 1024 * 1024,
    "tokenizer.json": 64 * 1024 * 1024,
    "tokenizer_config.json": 4 * 1024 * 1024,
}
_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _strict_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey, ValueError) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{label} is not strict JSON") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate the exact GLM-5.2 control files and Hugging Face LFS inventory, "
            "then range-read only safetensors prefixes and headers"
        )
    )
    parser.add_argument("root", type=Path, help="Pinned GLM-5.2 control-asset directory")
    parser.add_argument("--repository", required=True, help="Exact Hugging Face owner/model ID")
    parser.add_argument("--revision", required=True, help="Exact 40-character commit")
    return parser


def _read_asset(root: Path, name: str, maximum_bytes: int) -> bytes:
    unresolved = root / name
    try:
        if unresolved.is_symlink():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"GLM-5.2 asset is a symlink: {name}")
        path = unresolved.resolve(strict=True)
        if not path.is_file():
            raise AmsError(ErrorCode.INTEGRITY_FAILURE, f"GLM-5.2 asset is not a file: {name}")
        size = path.stat().st_size
        if size <= 0 or size > maximum_bytes:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                f"GLM-5.2 asset size is outside its bound: {name}",
            )
        return path.read_bytes()
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            f"GLM-5.2 asset could not be read: {name}",
            retriable=True,
        ) from exc


def _fetch_model_metadata(repository: str, revision: str) -> dict[str, Any]:
    if _REPOSITORY.fullmatch(repository) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "repository must be one exact owner/model ID")
    if _REVISION.fullmatch(revision) is None:
        raise AmsError(ErrorCode.PLAN_INVALID, "revision must be one exact lowercase commit")
    repository_path = quote(repository, safe="/")
    revision_path = quote(revision, safe="")
    url = f"https://huggingface.co/api/models/{repository_path}/revision/{revision_path}?blobs=true"
    request = Request(
        url,
        headers={"Accept-Encoding": "identity", "User-Agent": "ams-runtime/0.1"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read(_MAX_API_BYTES + 1)
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE,
            "Hugging Face metadata request failed",
            retriable=True,
        ) from exc
    if len(payload) > _MAX_API_BYTES:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face metadata exceeds its bound")
    metadata = _strict_json(payload, label="Hugging Face metadata")
    if not isinstance(metadata, dict) or metadata.get("sha") != revision:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "Hugging Face metadata does not identify the pinned revision",
        )
    return metadata


def _normalize_lfs_siblings(metadata: dict[str, Any]) -> dict[str, tuple[int, str]]:
    raw_siblings = metadata.get("siblings")
    if not isinstance(raw_siblings, list):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face metadata has no file inventory")
    siblings: dict[str, tuple[int, str]] = {}
    for item in raw_siblings:
        if not isinstance(item, dict):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face file metadata is malformed")
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
            raise AmsError(ErrorCode.INVALID_PACKAGE, "Hugging Face LFS metadata is malformed")
        siblings[name] = (size, digest)
    return siblings


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    root = arguments.root.resolve(strict=True)
    assets = {
        name: _read_asset(root, name, maximum_bytes)
        for name, maximum_bytes in _MAX_ASSET_BYTES.items()
    }
    architecture = parse_glm_moe_dsa_architecture(assets["config.json"])
    index = parse_huggingface_shard_index(assets["model.safetensors.index.json"])
    inventory = validate_glm_tensor_inventory(architecture, index)

    metadata = _fetch_model_metadata(arguments.repository, arguments.revision)
    siblings = _normalize_lfs_siblings(metadata)
    repository_path = quote(arguments.repository, safe="/")
    revision_path = quote(arguments.revision, safe="")
    shard_lock = []
    sources = []
    for shard_name in index.shard_names:
        entry = siblings.get(shard_name)
        if entry is None:
            raise AmsError(
                ErrorCode.INTEGRITY_FAILURE,
                f"pinned GLM-5.2 shard is absent from LFS metadata: {shard_name}",
            )
        size_bytes, digest = entry
        shard_lock.append(
            {
                "name": shard_name,
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        shard_path = quote(shard_name, safe="")
        sources.append(
            HuggingFaceShardSource(
                shard_name=shard_name,
                object_id=f"hf:{shard_name}",
                content_hash=f"sha256:{digest}",
                reader=HttpRangeReader(
                    "https://huggingface.co/"
                    f"{repository_path}/resolve/{revision_path}/{shard_path}",
                    size_bytes,
                ),
            )
        )
    catalog = build_huggingface_header_catalog(index, tuple(sources))
    if len(catalog.tensors) != len(inventory.slots):
        raise AmsError(ErrorCode.INTERNAL_INVARIANT, "GLM-5.2 inventory changed after header audit")

    asset_evidence = {
        name: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for name, payload in sorted(assets.items())
    }
    return {
        "architecture": {
            "first_dense_layers": architecture.first_k_dense_replace,
            "hidden_size": architecture.hidden_size,
            "index_topk": architecture.index_topk,
            "max_position_embeddings": architecture.max_position_embeddings,
            "num_attention_heads": architecture.num_attention_heads,
            "num_experts_per_token": architecture.num_experts_per_tok,
            "num_hidden_layers": architecture.num_hidden_layers,
            "num_nextn_predict_layers": architecture.num_nextn_predict_layers,
            "num_routed_experts": architecture.n_routed_experts,
            "vocab_size": architecture.vocab_size,
        },
        "architecture_hash": architecture.content_hash,
        "assets": asset_evidence,
        "declared_total_size": catalog.audit.declared_total_size,
        "dtype_counts": dict(catalog.audit.dtype_counts),
        "header_bytes_read": catalog.audit.prefix_and_header_bytes,
        "index_hash": catalog.index_content_hash,
        "index_metadata_hash": catalog.index_metadata_hash,
        "qualifies_precision_policy": False,
        "repository": arguments.repository,
        "revision": arguments.revision,
        "shard_count": catalog.audit.shard_count,
        "shard_inventory_hash": "sha256:"
        + hashlib.sha256(canonical_json_bytes(shard_lock)).hexdigest(),
        "source_file_bytes": catalog.audit.source_file_bytes,
        "source_root": catalog.source_root,
        "status": "structural_headers_only",
        "tensor_bytes": catalog.audit.tensor_bytes,
        "tensor_count": catalog.audit.tensor_count,
        "tensor_elements": catalog.audit.tensor_elements,
        "weight_payload_bytes_read": 0,
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
                    "message": "GLM-5.2 source audit failed",
                    "retriable": True,
                }
            }
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
