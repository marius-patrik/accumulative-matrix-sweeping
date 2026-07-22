"""Stable, serializable AMS error taxonomy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any


class ErrorCategory(StrEnum):
    PREFLIGHT = "preflight"
    PACKAGE = "package"
    INTEGRITY = "integrity"
    CAPABILITY = "capability"
    PLAN = "plan"
    RESOURCE = "resource"
    IO = "io"
    BACKEND = "backend"
    NUMERIC = "numeric"
    TRANSACTION = "transaction"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class ErrorCode(StrEnum):
    PREFLIGHT_NO_BACKING = "PREFLIGHT_NO_BACKING"
    PREFLIGHT_NO_WORKING_SET = "PREFLIGHT_NO_WORKING_SET"
    UNSUPPORTED_OP = "UNSUPPORTED_OP"
    INVALID_PACKAGE = "INVALID_PACKAGE"
    INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
    SIGNATURE_FAILURE = "SIGNATURE_FAILURE"
    CAPABILITY_MISMATCH = "CAPABILITY_MISMATCH"
    PLAN_INVALID = "PLAN_INVALID"
    RESERVATION_LOST = "RESERVATION_LOST"
    BROKER_VIOLATION = "BROKER_VIOLATION"
    IO_FAILURE = "IO_FAILURE"
    BACKEND_FAILURE = "BACKEND_FAILURE"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"
    TRANSACTION_FAILURE = "TRANSACTION_FAILURE"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    CANCELLED = "CANCELLED"
    INTERNAL_INVARIANT = "INTERNAL_INVARIANT"


_CATEGORY_BY_CODE: dict[ErrorCode, ErrorCategory] = {
    ErrorCode.PREFLIGHT_NO_BACKING: ErrorCategory.PREFLIGHT,
    ErrorCode.PREFLIGHT_NO_WORKING_SET: ErrorCategory.PREFLIGHT,
    ErrorCode.UNSUPPORTED_OP: ErrorCategory.CAPABILITY,
    ErrorCode.INVALID_PACKAGE: ErrorCategory.PACKAGE,
    ErrorCode.INTEGRITY_FAILURE: ErrorCategory.INTEGRITY,
    ErrorCode.SIGNATURE_FAILURE: ErrorCategory.INTEGRITY,
    ErrorCode.CAPABILITY_MISMATCH: ErrorCategory.CAPABILITY,
    ErrorCode.PLAN_INVALID: ErrorCategory.PLAN,
    ErrorCode.RESERVATION_LOST: ErrorCategory.RESOURCE,
    ErrorCode.BROKER_VIOLATION: ErrorCategory.RESOURCE,
    ErrorCode.IO_FAILURE: ErrorCategory.IO,
    ErrorCode.BACKEND_FAILURE: ErrorCategory.BACKEND,
    ErrorCode.NUMERIC_FAILURE: ErrorCategory.NUMERIC,
    ErrorCode.TRANSACTION_FAILURE: ErrorCategory.TRANSACTION,
    ErrorCode.DEADLINE_EXCEEDED: ErrorCategory.DEADLINE,
    ErrorCode.CANCELLED: ErrorCategory.CANCELLED,
    ErrorCode.INTERNAL_INVARIANT: ErrorCategory.INTERNAL,
}


class AmsError(RuntimeError):
    """An error with stable machine-readable identity and redaction-safe context."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        retriable: bool = False,
        phase: str | None = None,
        subsystem: str | None = None,
        correlations: Mapping[str, str] | None = None,
        evidence: Mapping[str, int | str | bool] | None = None,
        fallback_attempts: Sequence[str] = (),
        cause_code: ErrorCode | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code)
        self.category = _CATEGORY_BY_CODE[self.code]
        self.message = message
        self.retriable = bool(retriable)
        self.phase = phase
        self.subsystem = subsystem
        self.correlations = MappingProxyType(dict(correlations or {}))
        self.evidence = MappingProxyType(dict(evidence or {}))
        self.fallback_attempts = tuple(fallback_attempts)
        self.cause_code = ErrorCode(cause_code) if cause_code is not None else None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category.value,
            "code": self.code.value,
            "message": self.message,
            "retriable": self.retriable,
        }
        optional: tuple[tuple[str, Any], ...] = (
            ("phase", self.phase),
            ("subsystem", self.subsystem),
            ("cause_code", self.cause_code.value if self.cause_code else None),
        )
        payload.update((key, value) for key, value in optional if value is not None)
        if self.correlations:
            payload["correlations"] = dict(sorted(self.correlations.items()))
        if self.evidence:
            payload["evidence"] = dict(sorted(self.evidence.items()))
        if self.fallback_attempts:
            payload["fallback_attempts"] = list(self.fallback_attempts)
        return payload
