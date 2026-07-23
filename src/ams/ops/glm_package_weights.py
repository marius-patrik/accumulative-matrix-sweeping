"""Bounded GLM weight operations over a published AMS directory package."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.checked import checked_add, checked_mul, checked_positive, checked_product, checked_uint
from ams.codecs import Int4CodecConfig, TernaryCodecConfig
from ams.descriptors import DType, StorageObject, validate_digest, validate_identifier
from ams.errors import AmsError, ErrorCode
from ams.integrations.glm4_moe_lite import (
    Glm4MoeLiteArchitecture,
    Glm4MoeLiteTensorRole,
    expected_glm4_moe_lite_tensor_shape,
    expected_glm4_moe_lite_tensor_slots,
    parse_glm4_moe_lite_architecture,
)
from ams.integrations.glm_moe_dsa import (
    GlmMoeDsaArchitecture,
    expected_glm_tensor_slots,
    parse_glm_moe_dsa_architecture,
)
from ams.ops.glm_moe_dsa_model import GlmWeightAccess
from ams.ops.reference import (
    Int4StreamedLinearPlan,
    StreamedLinearPlan,
    TernaryStreamedLinearPlan,
    stream_linear_identity,
    stream_linear_int4,
    stream_linear_ternary,
)
from ams.package import resolve_package_file, resolve_package_root, verify_manifest_content_root
from ams.storage import FileRangeStore

_MAX_MANIFEST_BYTES = 64 * 1024 * 1024
_IDENTITY_FEATURE = "ams.identity-layout.v1"
_ROOT_FEATURE = "ams.content-root.manifest-minus-root.v1"
_INT4_FEATURE = "ams.codec.int4.symmetric.v1"
_TERNARY_FEATURE = "ams.codec.ternary.trit5.v1"
_SUPPORTED_REQUIRED_FEATURES = {
    _IDENTITY_FEATURE,
    _INT4_FEATURE,
    _ROOT_FEATURE,
    _TERNARY_FEATURE,
}
_ITEM_BYTES = {DType.FLOAT16: 2, DType.BFLOAT16: 2, DType.FLOAT32: 4}


class _DuplicateManifestKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateManifestKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} must be an object")
    return value


def _array(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} must be an array")
    return value


def _exact_fields(value: dict[str, Any], fields: set[str], *, name: str) -> None:
    if set(value) != fields:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} fields are missing or unreviewed")


def _dtype(value: Any, *, name: str) -> DType:
    try:
        return DType(value)
    except (TypeError, ValueError) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} is unsupported") from exc


class _VerifiedObject:
    def __init__(self, reader: FileRangeStore, *, verification_buffer_bytes: int) -> None:
        self.reader = reader
        self.size_bytes = reader.size_bytes
        self._verification_buffer_bytes = verification_buffer_bytes
        self._verified = False
        self.verification_bytes = 0
        self.range_read_bytes = 0
        self.maximum_read_bytes = 0

    def _verify(self) -> None:
        if not self._verified:
            self.reader.verify_content_hash(buffer_bytes=self._verification_buffer_bytes)
            self._verified = True
            self.verification_bytes = self.size_bytes
            self.maximum_read_bytes = max(
                self.maximum_read_bytes,
                min(self._verification_buffer_bytes, self.size_bytes),
            )

    def read_into(self, offset: int, destination) -> None:
        self._verify()
        view = memoryview(destination)
        try:
            read_bytes = view.nbytes
        finally:
            view.release()
        self.range_read_bytes += read_bytes
        self.maximum_read_bytes = max(self.maximum_read_bytes, read_bytes)
        self.reader.read_into(offset, destination)


@dataclass(frozen=True, slots=True)
class _PackageTensor:
    source_name: str
    shape: tuple[int, ...]
    logical_dtype: DType
    encoding: str
    reader: _VerifiedObject
    offset: int
    encoded_bytes: int
    decoded_bytes: int
    ternary_config: TernaryCodecConfig | None
    int4_config: Int4CodecConfig | None


@dataclass(frozen=True, slots=True)
class GlmPackageReadEvidence:
    """Bounded process-local I/O counters for package weight operations."""

    verified_objects: int
    verification_bytes: int
    range_read_bytes: int
    maximum_read_bytes: int


@dataclass(frozen=True, slots=True)
class Glm4NativeStorageBinding:
    """One admitted immutable object available to a native GLM-4 reader registry."""

    object_id: str
    absolute_path: str
    size_bytes: int
    alignment_bytes: int
    content_hash: str
    kind: str


@dataclass(frozen=True, slots=True)
class Glm4NativeTensorBinding:
    """One exact GLM-4 tensor range and its native codec parameters."""

    tensor_name: str
    role: Glm4MoeLiteTensorRole
    layer_index: int | None
    expert_index: int | None
    mtp: bool
    shape: tuple[int, ...]
    logical_dtype: DType
    encoding: str
    storage_index: int
    offset: int
    encoded_bytes: int
    decoded_bytes: int
    codec_group_size: int | None
    codec_config_hash: str | None


@dataclass(frozen=True, slots=True)
class Glm4NativeBindingPlan:
    """Immutable package-to-native GLM-4 handoff with explicit resource policy."""

    schema_id: str
    binding_hash: str
    binding_identity_json: bytes
    package_id: str
    manifest_content_root: str
    architecture: Glm4MoeLiteArchitecture
    storage_objects: tuple[Glm4NativeStorageBinding, ...]
    tensors: tuple[Glm4NativeTensorBinding, ...]
    linear_arena_bytes: int
    context_capacity_tokens: int
    cache_key_dtype: DType
    cache_value_dtype: DType
    cache_storage_bytes_per_layer: int
    cache_storage_bytes_total: int
    cache_staging_bytes_per_layer: int
    tokenizer_vocabulary_size: int
    eos_token_ids: tuple[int, ...]

    @property
    def executable_tensor_count(self) -> int:
        """Return the base-model inventory, excluding separately marked MTP tensors."""
        return sum(not tensor.mtp for tensor in self.tensors)

    @property
    def mtp_tensor_count(self) -> int:
        """Return the separately admitted but not yet executable MTP inventory."""
        return sum(tensor.mtp for tensor in self.tensors)

    @property
    def encoding_counts(self) -> tuple[tuple[str, int], ...]:
        """Return deterministic encoding counts for audit and qualification output."""
        encodings = {tensor.encoding for tensor in self.tensors}
        return tuple(
            (encoding, sum(tensor.encoding == encoding for tensor in self.tensors))
            for encoding in sorted(encodings)
        )


def serialize_glm4_native_binding_plan(plan: Glm4NativeBindingPlan) -> bytes:
    """Serialize one exact binding identity plus its machine-local object-path map."""
    if not isinstance(plan, Glm4NativeBindingPlan):
        raise AmsError(ErrorCode.PLAN_INVALID, "native GLM-4 binding has the wrong type")
    if plan.schema_id != "ams.native.glm4-binding.v1":
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "native GLM-4 binding schema is unsupported")
    expected_hash = "sha256:" + hashlib.sha256(plan.binding_identity_json).hexdigest()
    if expected_hash != plan.binding_hash:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "native GLM-4 binding identity does not match its declared hash",
        )
    try:
        identity = json.loads(
            plan.binding_identity_json,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateManifestKey, ValueError) as exc:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "native GLM-4 binding identity JSON is invalid",
        ) from exc
    if canonical_json_bytes(identity) != plan.binding_identity_json:
        raise AmsError(
            ErrorCode.INTEGRITY_FAILURE,
            "native GLM-4 binding identity JSON is not canonical",
        )
    object_ids = tuple(binding.object_id for binding in plan.storage_objects)
    if len(set(object_ids)) != len(object_ids):
        raise AmsError(
            ErrorCode.INTERNAL_INVARIANT,
            "native GLM-4 storage object identity is duplicated",
        )
    envelope = {
        "binding_hash": plan.binding_hash,
        "binding_identity_json": plan.binding_identity_json.decode("utf-8"),
        "schema_id": "ams.native.glm4-envelope.v1",
        "storage_paths": [
            {
                "absolute_path": binding.absolute_path,
                "object_id": binding.object_id,
            }
            for binding in plan.storage_objects
        ],
    }
    return canonical_json_bytes(envelope)


def _parse_manifest_payload(path: Path, max_manifest_bytes: int) -> dict[str, Any]:
    checked_positive(max_manifest_bytes, name="package.max_manifest_bytes")
    try:
        size = path.stat().st_size
        if not 1 <= size <= max_manifest_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "published manifest size is invalid")
        payload = path.read_bytes()
    except AmsError:
        raise
    except OSError as exc:
        raise AmsError(
            ErrorCode.IO_FAILURE, "published manifest read failed", retriable=True
        ) from exc
    try:
        manifest = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateManifestKey, ValueError) as exc:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "published manifest JSON is invalid") from exc
    manifest = _object(manifest, name="manifest")
    if canonical_json_bytes(manifest) != payload:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "published manifest is not canonical JSON")
    verify_manifest_content_root(manifest)
    return manifest


def _parse_storage_objects(
    root: Path,
    manifest: dict[str, Any],
    *,
    verification_buffer_bytes: int,
) -> dict[str, _VerifiedObject]:
    checked_positive(verification_buffer_bytes, name="package.verification_buffer_bytes")
    objects: dict[str, _VerifiedObject] = {}
    uris: set[str] = set()
    for raw_value in _array(manifest.get("storage_objects"), name="manifest.storage_objects"):
        raw = _object(raw_value, name="storage object")
        _exact_fields(
            raw,
            {
                "object_id",
                "uri",
                "size_bytes",
                "alignment_bytes",
                "content_hash",
                "immutable",
                "kind",
            },
            name="storage object",
        )
        if raw["immutable"] is not True:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "mutable package objects are unsupported")
        descriptor = StorageObject(
            object_id=raw["object_id"],
            uri=raw["uri"],
            size_bytes=raw["size_bytes"],
            alignment_bytes=raw["alignment_bytes"],
            content_hash=raw["content_hash"],
            immutable=True,
            kind=raw["kind"],
        )
        if descriptor.object_id in objects or descriptor.uri in uris:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "storage object IDs or URIs are duplicated")
        path = resolve_package_file(root, descriptor.uri)
        objects[descriptor.object_id] = _VerifiedObject(
            FileRangeStore(path, descriptor),
            verification_buffer_bytes=verification_buffer_bytes,
        )
        uris.add(descriptor.uri)
    if not objects:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "package has no storage objects")
    return objects


def _parse_ternary_config(codec: dict[str, Any], decoded_bytes: int) -> TernaryCodecConfig:
    _exact_fields(
        codec,
        {"name", "version", "lossless", "max_decoded_bytes", "parameters"},
        name="ternary codec",
    )
    if (
        codec["name"] != "ams.ternary.trit5"
        or codec["version"] != "1.0.0"
        or codec["lossless"] is not False
        or codec["max_decoded_bytes"] != decoded_bytes
    ):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "ternary codec declaration is unsupported")
    parameters = _object(codec["parameters"], name="ternary parameters")
    _exact_fields(
        parameters,
        {
            "ams.config-hash",
            "ams.group-size",
            "ams.packing",
            "ams.scale-dtype",
            "ams.threshold-denominator",
            "ams.threshold-numerator",
        },
        name="ternary parameters",
    )
    config = TernaryCodecConfig(
        group_size=parameters["ams.group-size"],
        threshold_numerator=parameters["ams.threshold-numerator"],
        threshold_denominator=parameters["ams.threshold-denominator"],
        scale_dtype=_dtype(parameters["ams.scale-dtype"], name="ternary scale dtype"),
        packing=parameters["ams.packing"],
        version=codec["version"],
    )
    if parameters["ams.config-hash"] != config.config_hash:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "ternary configuration hash mismatch")
    return config


def _parse_int4_config(codec: dict[str, Any], decoded_bytes: int) -> Int4CodecConfig:
    _exact_fields(
        codec,
        {"name", "version", "lossless", "max_decoded_bytes", "parameters"},
        name="INT4 codec",
    )
    if (
        codec["name"] != "ams.int4.symmetric"
        or codec["version"] != "1.0.0"
        or codec["lossless"] is not False
        or codec["max_decoded_bytes"] != decoded_bytes
    ):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "INT4 codec declaration is unsupported")
    parameters = _object(codec["parameters"], name="INT4 parameters")
    _exact_fields(
        parameters,
        {"ams.config-hash", "ams.group-size", "ams.packing", "ams.scale-dtype"},
        name="INT4 parameters",
    )
    config = Int4CodecConfig(
        group_size=parameters["ams.group-size"],
        scale_dtype=_dtype(parameters["ams.scale-dtype"], name="INT4 scale dtype"),
        packing=parameters["ams.packing"],
        version=codec["version"],
    )
    if parameters["ams.config-hash"] != config.config_hash:
        raise AmsError(ErrorCode.INTEGRITY_FAILURE, "INT4 configuration hash mismatch")
    return config


def _parse_tensor(
    raw_value: Any,
    objects: dict[str, _VerifiedObject],
) -> _PackageTensor:
    raw = _object(raw_value, name="tensor")
    _exact_fields(
        raw,
        {
            "tensor_id",
            "tensor_class",
            "shape",
            "logical_dtype",
            "byte_order",
            "immutable",
            "layouts",
            "extensions",
        },
        name="tensor",
    )
    validate_identifier(raw["tensor_id"], name="tensor.tensor_id")
    if (
        raw["tensor_class"] != "parameter"
        or raw["byte_order"] != "little"
        or raw["immutable"] is not True
    ):
        raise AmsError(
            ErrorCode.CAPABILITY_MISMATCH, "tensor mutability or representation is unsupported"
        )
    shape_values = _array(raw["shape"], name="tensor.shape")
    if not shape_values or len(shape_values) > 2:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "GLM package tensor rank must be one or two")
    shape = tuple(checked_positive(value, name="tensor.shape") for value in shape_values)
    logical_dtype = _dtype(raw["logical_dtype"], name="tensor logical dtype")
    if logical_dtype not in _ITEM_BYTES:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "GLM package tensor dtype is unsupported")
    extensions = _object(raw["extensions"], name="tensor.extensions")
    _exact_fields(
        extensions,
        {"hf.shard-name", "hf.source-dtype", "hf.source-name"},
        name="tensor.extensions",
    )
    source_name = extensions["hf.source-name"]
    if not isinstance(source_name, str) or not source_name:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor source name is invalid")
    layouts = _array(raw["layouts"], name="tensor.layouts")
    if len(layouts) != 1:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "exactly one GLM tensor layout is required")
    layout = _object(layouts[0], name="tensor.layout")
    allowed_layout_fields = {
        "layout_id",
        "layout_version",
        "complete",
        "tile_shape",
        "alignment_bytes",
        "storage_dtype",
        "chunks",
        "extensions",
        "codec",
    }
    if not set(layout).issubset(allowed_layout_fields) or not (
        allowed_layout_fields - {"codec"}
    ).issubset(layout):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor layout fields are missing or unreviewed")
    if (
        layout["layout_version"] != "1.0.0"
        or layout["complete"] is not True
        or layout["tile_shape"] != list(shape)
        or layout["alignment_bytes"] != 1
    ):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "tensor layout geometry is unsupported")
    layout_extensions = _object(layout["extensions"], name="layout.extensions")
    _exact_fields(layout_extensions, {"ams.encoding"}, name="layout.extensions")
    encoding = layout_extensions["ams.encoding"]
    chunks = _array(layout["chunks"], name="layout.chunks")
    if len(chunks) != 1:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "exactly one tensor chunk is required")
    chunk = _object(chunks[0], name="tensor.chunk")
    _exact_fields(
        chunk,
        {
            "chunk_id",
            "range",
            "logical_origin",
            "logical_extent",
            "encoded_bytes",
            "decoded_bytes",
        },
        name="tensor.chunk",
    )
    if chunk["logical_origin"] != [0] * len(shape) or chunk["logical_extent"] != list(shape):
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "partial tensor chunks are unsupported")
    encoded_bytes = checked_positive(chunk["encoded_bytes"], name="chunk.encoded_bytes")
    decoded_bytes = checked_positive(chunk["decoded_bytes"], name="chunk.decoded_bytes")
    expected_decoded = checked_mul(
        checked_product(shape, name="tensor.elements"),
        _ITEM_BYTES[logical_dtype],
        name="tensor.decoded_bytes",
    )
    if decoded_bytes != expected_decoded:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor decoded size is inconsistent")
    byte_range = _object(chunk["range"], name="chunk.range")
    _exact_fields(byte_range, {"object_id", "offset", "length", "checksum"}, name="chunk.range")
    object_id = byte_range["object_id"]
    if object_id not in objects:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor references an absent storage object")
    offset = checked_uint(byte_range["offset"], name="chunk.range.offset")
    length = checked_positive(byte_range["length"], name="chunk.range.length")
    validate_digest(byte_range["checksum"], name="chunk.range.checksum")
    reader = objects[object_id]
    if (
        reader.reader.descriptor.kind != "tensor_data"
        or length != encoded_bytes
        or checked_add(offset, length, name="chunk.range.end") > reader.size_bytes
        or byte_range["checksum"] != reader.reader.descriptor.content_hash
    ):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "tensor chunk range is inconsistent")
    if encoding == "identity":
        if (
            layout["layout_id"] != "layout:identity.v1"
            or layout["storage_dtype"] != logical_dtype.value
            or "codec" in layout
            or encoded_bytes != decoded_bytes
        ):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "identity tensor layout is inconsistent")
        ternary_config = None
        int4_config = None
    elif encoding == "ternary_trit5":
        if layout["layout_id"] != "layout:ternary.trit5.v1" or layout["storage_dtype"] != "custom":
            raise AmsError(ErrorCode.INVALID_PACKAGE, "ternary tensor layout is inconsistent")
        ternary_config = _parse_ternary_config(
            _object(layout.get("codec"), name="tensor.codec"), decoded_bytes
        )
        element_count = checked_product(shape, name="ternary.elements")
        if ternary_config.encoded_size(element_count) != encoded_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "ternary encoded size is inconsistent")
        int4_config = None
    elif encoding == "int4_symmetric":
        if layout["layout_id"] != "layout:int4.symmetric.v1" or layout["storage_dtype"] != "custom":
            raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 tensor layout is inconsistent")
        int4_config = _parse_int4_config(
            _object(layout.get("codec"), name="tensor.codec"), decoded_bytes
        )
        element_count = checked_product(shape, name="int4.elements")
        if int4_config.encoded_size(element_count) != encoded_bytes:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "INT4 encoded size is inconsistent")
        ternary_config = None
    else:
        raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "tensor encoding is unsupported")
    return _PackageTensor(
        source_name,
        shape,
        logical_dtype,
        encoding,
        reader,
        offset,
        encoded_bytes,
        decoded_bytes,
        ternary_config,
        int4_config,
    )


class GlmPackageWeights(GlmWeightAccess):
    """GLM weight operations backed by verified AMS object ranges."""

    def __init__(
        self,
        architecture: GlmMoeDsaArchitecture | Glm4MoeLiteArchitecture,
        tensors: dict[str, _PackageTensor],
        *,
        package_id: str,
        manifest_content_root: str,
        linear_arena_bytes: int,
    ) -> None:
        checked_positive(linear_arena_bytes, name="package.linear_arena_bytes")
        self.architecture = architecture
        self._tensors = tensors
        self._package_id = package_id
        self._manifest_content_root = manifest_content_root
        self._linear_arena_bytes = linear_arena_bytes

    @property
    def read_evidence(self) -> GlmPackageReadEvidence:
        """Return process-local object verification and range-read counters."""
        objects = {id(tensor.reader): tensor.reader for tensor in self._tensors.values()}
        return GlmPackageReadEvidence(
            verified_objects=sum(reader.verification_bytes > 0 for reader in objects.values()),
            verification_bytes=sum(reader.verification_bytes for reader in objects.values()),
            range_read_bytes=sum(reader.range_read_bytes for reader in objects.values()),
            maximum_read_bytes=max(
                (reader.maximum_read_bytes for reader in objects.values()), default=0
            ),
        )

    @classmethod
    def open(
        cls,
        package_root: Path,
        *,
        linear_arena_bytes: int = 1024 * 1024,
        verification_buffer_bytes: int = 1024 * 1024,
        max_manifest_bytes: int = _MAX_MANIFEST_BYTES,
    ) -> GlmPackageWeights:
        root = resolve_package_root(package_root)
        manifest_path = resolve_package_file(root, "manifest.json")
        manifest = _parse_manifest_payload(manifest_path, max_manifest_bytes)
        _exact_fields(
            manifest,
            {
                "schema_id",
                "format_version",
                "package_id",
                "required_features",
                "optional_features",
                "graph",
                "model",
                "storage_objects",
                "tensors",
                "integrity",
                "provenance",
                "extensions",
                "content_root",
            },
            name="manifest",
        )
        if manifest["schema_id"] != "ams.model.manifest" or manifest["format_version"] != {
            "major": 1,
            "minor": 0,
        }:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "AMS manifest version is unsupported")
        package_id = validate_identifier(manifest["package_id"], name="manifest.package_id")
        required_features = _array(manifest["required_features"], name="required_features")
        if (
            any(not isinstance(feature, str) for feature in required_features)
            or len(set(required_features)) != len(required_features)
            or not {_ROOT_FEATURE, _IDENTITY_FEATURE}.issubset(required_features)
            or not set(required_features).issubset(_SUPPORTED_REQUIRED_FEATURES)
        ):
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "AMS required features are unsupported")
        optional_features = _array(manifest["optional_features"], name="optional_features")
        if any(not isinstance(feature, str) for feature in optional_features) or len(
            set(optional_features)
        ) != len(optional_features):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "AMS optional features are invalid")
        model = _object(manifest["model"], name="manifest.model")
        _exact_fields(
            model, {"architecture", "configuration", "default_dtype"}, name="manifest.model"
        )
        _dtype(model["default_dtype"], name="model default dtype")
        configuration_payload = canonical_json_bytes(model["configuration"])
        expected_glm4_slots = None
        if model["architecture"] == "GlmMoeDsaForCausalLM":
            architecture = parse_glm_moe_dsa_architecture(configuration_payload)
            expected_names = {slot.tensor_name for slot in expected_glm_tensor_slots(architecture)}
        elif model["architecture"] == "Glm4MoeLiteForCausalLM":
            architecture = parse_glm4_moe_lite_architecture(configuration_payload)
            expected_glm4_slots = {
                slot.tensor_name: slot for slot in expected_glm4_moe_lite_tensor_slots(architecture)
            }
            expected_names = set(expected_glm4_slots)
        else:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "AMS package architecture is not a reviewed GLM causal LM",
            )
        objects = _parse_storage_objects(
            root,
            manifest,
            verification_buffer_bytes=verification_buffer_bytes,
        )
        tensors: dict[str, _PackageTensor] = {}
        tensor_ids: set[str] = set()
        for raw_tensor in _array(manifest["tensors"], name="manifest.tensors"):
            parsed = _parse_tensor(raw_tensor, objects)
            raw = _object(raw_tensor, name="tensor")
            tensor_id = raw["tensor_id"]
            if parsed.source_name in tensors or tensor_id in tensor_ids:
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE, "tensor IDs or source names are duplicated"
                )
            if expected_glm4_slots is not None and parsed.source_name in expected_glm4_slots:
                expected_shape = expected_glm4_moe_lite_tensor_shape(
                    architecture,
                    expected_glm4_slots[parsed.source_name],
                )
                if parsed.shape != expected_shape:
                    raise AmsError(
                        ErrorCode.CAPABILITY_MISMATCH,
                        "GLM-4 package tensor shape differs from the reviewed graph",
                    )
            tensors[parsed.source_name] = parsed
            tensor_ids.add(tensor_id)
        if set(tensors) != expected_names:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "AMS package does not contain its architecture's exact reviewed tensor inventory",
                evidence={
                    "missing": len(expected_names - set(tensors)),
                    "unexpected": len(set(tensors) - expected_names),
                },
            )
        expected_features = {_ROOT_FEATURE}
        encodings = {tensor.encoding for tensor in tensors.values()}
        if "identity" in encodings:
            expected_features.add(_IDENTITY_FEATURE)
        if "ternary_trit5" in encodings:
            expected_features.add(_TERNARY_FEATURE)
        if "int4_symmetric" in encodings:
            expected_features.add(_INT4_FEATURE)
        if set(required_features) != expected_features:
            raise AmsError(
                ErrorCode.INVALID_PACKAGE,
                "AMS required features do not match the tensor encodings",
            )
        return cls(
            architecture,
            tensors,
            package_id=package_id,
            manifest_content_root=manifest["content_root"],
            linear_arena_bytes=linear_arena_bytes,
        )

    def native_glm4_binding_plan(
        self,
        *,
        context_capacity_tokens: int,
        tokenizer_vocabulary_size: int,
        eos_token_ids: Sequence[int],
        cache_key_dtype: DType = DType.BFLOAT16,
        cache_value_dtype: DType = DType.BFLOAT16,
    ) -> Glm4NativeBindingPlan:
        """Normalize this package into a deterministic, zero-weight-I/O native handoff."""
        architecture = self.architecture
        if not isinstance(architecture, Glm4MoeLiteArchitecture):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "native GLM-4 binding requires a GLM-4-MoE-Lite package",
            )
        if (
            isinstance(context_capacity_tokens, bool)
            or not isinstance(context_capacity_tokens, int)
            or not 1 <= context_capacity_tokens <= architecture.max_position_embeddings
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "native GLM-4 context capacity is outside the model limit",
            )
        if (
            isinstance(tokenizer_vocabulary_size, bool)
            or not isinstance(tokenizer_vocabulary_size, int)
            or not 1 <= tokenizer_vocabulary_size <= architecture.vocab_size
        ):
            raise AmsError(
                ErrorCode.PLAN_INVALID,
                "native GLM-4 tokenizer vocabulary is outside the model vocabulary",
            )
        if isinstance(eos_token_ids, str | bytes) or not isinstance(eos_token_ids, Sequence):
            raise AmsError(ErrorCode.PLAN_INVALID, "native GLM-4 EOS policy is invalid")
        normalized_eos = tuple(eos_token_ids)
        if (
            not normalized_eos
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or not 0 <= token_id < tokenizer_vocabulary_size
                for token_id in normalized_eos
            )
            or len(set(normalized_eos)) != len(normalized_eos)
        ):
            raise AmsError(ErrorCode.PLAN_INVALID, "native GLM-4 EOS policy is invalid")
        normalized_eos = tuple(sorted(normalized_eos))

        try:
            key_dtype = DType(cache_key_dtype)
            value_dtype = DType(cache_value_dtype)
        except (TypeError, ValueError) as exc:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "native GLM-4 cache dtype is unsupported",
            ) from exc
        admitted_cache_dtypes = {DType.BFLOAT16, DType.FLOAT32}
        if key_dtype not in admitted_cache_dtypes or value_dtype not in admitted_cache_dtypes:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "native GLM-4 cache dtype is unsupported",
            )

        key_row_elements = checked_mul(
            architecture.num_attention_heads,
            architecture.qk_head_dim,
            name="native_glm4.cache_key_row_elements",
        )
        value_row_elements = checked_mul(
            architecture.num_attention_heads,
            architecture.v_head_dim,
            name="native_glm4.cache_value_row_elements",
        )
        key_row_bytes = checked_mul(
            key_row_elements,
            _ITEM_BYTES[key_dtype],
            name="native_glm4.cache_key_row_bytes",
        )
        value_row_bytes = checked_mul(
            value_row_elements,
            _ITEM_BYTES[value_dtype],
            name="native_glm4.cache_value_row_bytes",
        )
        cache_staging_bytes = checked_add(
            key_row_bytes,
            value_row_bytes,
            name="native_glm4.cache_staging_bytes",
        )
        cache_storage_bytes_per_layer = checked_mul(
            context_capacity_tokens,
            cache_staging_bytes,
            name="native_glm4.cache_storage_bytes_per_layer",
        )
        cache_storage_bytes_total = checked_mul(
            architecture.num_hidden_layers,
            cache_storage_bytes_per_layer,
            name="native_glm4.cache_storage_bytes_total",
        )

        slots = expected_glm4_moe_lite_tensor_slots(architecture)
        identity_only_roles = {
            Glm4MoeLiteTensorRole.EMBEDDING,
            Glm4MoeLiteTensorRole.FINAL_NORM,
            Glm4MoeLiteTensorRole.INPUT_NORM,
            Glm4MoeLiteTensorRole.POST_ATTENTION_NORM,
            Glm4MoeLiteTensorRole.ATTENTION_Q_A_NORM,
            Glm4MoeLiteTensorRole.ATTENTION_KV_A_NORM,
            Glm4MoeLiteTensorRole.ROUTER_CORRECTION_BIAS,
        }
        if any(
            not slot.mtp
            and slot.role in identity_only_roles
            and self._tensors[slot.tensor_name].encoding != "identity"
            for slot in slots
        ):
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "native GLM-4 vector and embedding bindings require identity storage",
            )
        referenced_readers = {
            tensor.reader.reader.descriptor.object_id: tensor.reader
            for tensor in self._tensors.values()
        }
        object_ids = tuple(sorted(referenced_readers))
        storage_indices = {object_id: index for index, object_id in enumerate(object_ids)}
        storage_objects = tuple(
            Glm4NativeStorageBinding(
                object_id=object_id,
                absolute_path=str(referenced_readers[object_id].reader.path),
                size_bytes=referenced_readers[object_id].reader.descriptor.size_bytes,
                alignment_bytes=referenced_readers[object_id].reader.descriptor.alignment_bytes,
                content_hash=referenced_readers[object_id].reader.descriptor.content_hash,
                kind=referenced_readers[object_id].reader.descriptor.kind,
            )
            for object_id in object_ids
        )
        tensor_bindings: list[Glm4NativeTensorBinding] = []
        for slot in slots:
            tensor = self._tensors[slot.tensor_name]
            descriptor = tensor.reader.reader.descriptor
            codec = tensor.ternary_config or tensor.int4_config
            tensor_bindings.append(
                Glm4NativeTensorBinding(
                    tensor_name=slot.tensor_name,
                    role=slot.role,
                    layer_index=slot.layer_index,
                    expert_index=slot.expert_index,
                    mtp=slot.mtp,
                    shape=tensor.shape,
                    logical_dtype=tensor.logical_dtype,
                    encoding=tensor.encoding,
                    storage_index=storage_indices[descriptor.object_id],
                    offset=tensor.offset,
                    encoded_bytes=tensor.encoded_bytes,
                    decoded_bytes=tensor.decoded_bytes,
                    codec_group_size=codec.group_size if codec is not None else None,
                    codec_config_hash=codec.config_hash if codec is not None else None,
                )
            )
        tensors = tuple(tensor_bindings)
        identity_payload = {
            "schema_id": "ams.native.glm4-binding.v1",
            "package_id": self._package_id,
            "manifest_content_root": self._manifest_content_root,
            "architecture": architecture,
            "storage_objects": [
                {
                    "object_id": binding.object_id,
                    "size_bytes": binding.size_bytes,
                    "alignment_bytes": binding.alignment_bytes,
                    "content_hash": binding.content_hash,
                    "kind": binding.kind,
                }
                for binding in storage_objects
            ],
            "tensors": [
                {
                    "tensor_name": binding.tensor_name,
                    "role": binding.role,
                    "layer_index": binding.layer_index,
                    "expert_index": binding.expert_index,
                    "mtp": binding.mtp,
                    "shape": binding.shape,
                    "logical_dtype": binding.logical_dtype,
                    "encoding": binding.encoding,
                    "storage_object_id": storage_objects[binding.storage_index].object_id,
                    "offset": binding.offset,
                    "encoded_bytes": binding.encoded_bytes,
                    "decoded_bytes": binding.decoded_bytes,
                    "codec_group_size": binding.codec_group_size,
                    "codec_config_hash": binding.codec_config_hash,
                }
                for binding in tensors
            ],
            "linear_arena_bytes": self._linear_arena_bytes,
            "context_capacity_tokens": context_capacity_tokens,
            "cache_key_dtype": key_dtype,
            "cache_value_dtype": value_dtype,
            "cache_storage_bytes_per_layer": cache_storage_bytes_per_layer,
            "cache_storage_bytes_total": cache_storage_bytes_total,
            "cache_staging_bytes_per_layer": cache_staging_bytes,
            "tokenizer_vocabulary_size": tokenizer_vocabulary_size,
            "eos_token_ids": normalized_eos,
        }
        binding_identity_json = canonical_json_bytes(identity_payload)
        binding_hash = "sha256:" + hashlib.sha256(binding_identity_json).hexdigest()
        return Glm4NativeBindingPlan(
            schema_id="ams.native.glm4-binding.v1",
            binding_hash=binding_hash,
            binding_identity_json=binding_identity_json,
            package_id=self._package_id,
            manifest_content_root=self._manifest_content_root,
            architecture=architecture,
            storage_objects=storage_objects,
            tensors=tensors,
            linear_arena_bytes=self._linear_arena_bytes,
            context_capacity_tokens=context_capacity_tokens,
            cache_key_dtype=key_dtype,
            cache_value_dtype=value_dtype,
            cache_storage_bytes_per_layer=cache_storage_bytes_per_layer,
            cache_storage_bytes_total=cache_storage_bytes_total,
            cache_staging_bytes_per_layer=cache_staging_bytes,
            tokenizer_vocabulary_size=tokenizer_vocabulary_size,
            eos_token_ids=normalized_eos,
        )

    def _tensor(self, tensor_name: str) -> _PackageTensor:
        try:
            return self._tensors[tensor_name]
        except KeyError as exc:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "required package tensor is absent") from exc

    @staticmethod
    def _require_identity(tensor: _PackageTensor) -> None:
        if tensor.encoding != "identity":
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "vector and embedding access require identity storage",
            )

    @staticmethod
    def _decode_identity(payload: bytearray, count: int, dtype: DType) -> tuple[float, ...]:
        if dtype is DType.FLOAT32:
            values = struct.unpack(f"<{count}f", payload)
        elif dtype is DType.FLOAT16:
            values = struct.unpack(f"<{count}e", payload)
        elif dtype is DType.BFLOAT16:
            values = tuple(
                struct.unpack(
                    "<f",
                    struct.pack("<I", struct.unpack_from("<H", payload, index * 2)[0] << 16),
                )[0]
                for index in range(count)
            )
        else:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "identity dtype changed after loading")
        if any(not math.isfinite(value) for value in values):
            raise AmsError(ErrorCode.NUMERIC_FAILURE, "identity tensor contains non-finite data")
        return tuple(values)

    @classmethod
    def _read_identity(
        cls,
        tensor: _PackageTensor,
        offset: int,
        count: int,
    ) -> tuple[float, ...]:
        item_bytes = _ITEM_BYTES[tensor.logical_dtype]
        payload = bytearray(checked_mul(count, item_bytes, name="package_identity.bytes"))
        tensor.reader.read_into(offset, payload)
        return cls._decode_identity(payload, count, tensor.logical_dtype)

    def vector(self, tensor_name: str, length: int) -> tuple[float, ...]:
        checked_positive(length, name="package_vector.length")
        tensor = self._tensor(tensor_name)
        self._require_identity(tensor)
        if tensor.shape != (length,):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "package vector shape is invalid")
        return self._read_identity(tensor, tensor.offset, length)

    def embedding(self, tensor_name: str, index: int, width: int) -> tuple[float, ...]:
        checked_positive(width, name="package_embedding.width")
        tensor = self._tensor(tensor_name)
        self._require_identity(tensor)
        if len(tensor.shape) != 2 or tensor.shape[1] != width:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "package embedding shape is invalid")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < tensor.shape[0]
        ):
            raise AmsError(ErrorCode.PLAN_INVALID, "package embedding index is invalid")
        item_bytes = _ITEM_BYTES[tensor.logical_dtype]
        row_bytes = checked_mul(width, item_bytes, name="package_embedding.row_bytes")
        offset = checked_add(
            tensor.offset,
            checked_mul(index, row_bytes, name="package_embedding.row_offset"),
            name="package_embedding.offset",
        )
        return self._read_identity(tensor, offset, width)

    def linear(
        self,
        tensor_name: str,
        values: Sequence[float],
        rows: int,
    ) -> tuple[float, ...]:
        checked_positive(rows, name="package_linear.rows")
        tensor = self._tensor(tensor_name)
        if tensor.shape != (rows, len(values)) or not values:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "package linear shape is invalid")
        output: list[float] = []
        if tensor.encoding == "identity":
            plan = StreamedLinearPlan.create(
                rows=rows,
                columns=len(values),
                weight_offset=tensor.offset,
                arena_bytes=self._linear_arena_bytes,
                storage_dtype=tensor.logical_dtype,
            )
            stream_linear_identity(
                tensor.reader,
                plan,
                values,
                lambda _, value: output.append(value),
            )
        elif tensor.encoding == "ternary_trit5" and tensor.ternary_config is not None:
            plan = TernaryStreamedLinearPlan.create(
                rows=rows,
                columns=len(values),
                weight_offset=tensor.offset,
                arena_bytes=self._linear_arena_bytes,
                config=tensor.ternary_config,
            )
            stream_linear_ternary(
                tensor.reader,
                plan,
                values,
                lambda _, value: output.append(value),
            )
        elif tensor.encoding == "int4_symmetric" and tensor.int4_config is not None:
            plan = Int4StreamedLinearPlan.create(
                rows=rows,
                columns=len(values),
                weight_offset=tensor.offset,
                arena_bytes=self._linear_arena_bytes,
                config=tensor.int4_config,
            )
            stream_linear_int4(
                tensor.reader,
                plan,
                values,
                lambda _, value: output.append(value),
            )
        else:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "package linear encoding is unsupported")
        return tuple(output)
