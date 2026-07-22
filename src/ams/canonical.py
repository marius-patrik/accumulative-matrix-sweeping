"""Canonical JSON encoding for hashed AMS control-plane artifacts."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

from ams.errors import AmsError, ErrorCode


def _normalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalize(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "canonical JSON keys must be strings")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple | list):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "canonical JSON forbids non-finite numbers")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise AmsError(
        ErrorCode.INVALID_PACKAGE,
        f"unsupported canonical JSON value: {type(value).__name__}",
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode *value* as stable UTF-8 JSON without insignificant whitespace."""
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
