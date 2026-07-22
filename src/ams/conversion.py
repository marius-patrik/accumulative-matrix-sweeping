"""Deterministic, restart-safe orchestration for bounded identity conversion."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ams.canonical import canonical_json_bytes
from ams.descriptors import (
    ByteRange,
    ConversionJournal,
    ConversionJournalEntry,
    JournalEntryState,
    validate_digest,
    validate_identifier,
)
from ams.errors import AmsError, ErrorCode
from ams.storage import RangeReader, copy_range_atomic

_JOURNAL_VERSION = "1.0.0"
_MAX_JOURNAL_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ConversionItem:
    target_chunk_id: str
    source_range: ByteRange

    def __post_init__(self) -> None:
        validate_identifier(self.target_chunk_id, name="conversion.target_chunk_id")


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    source_root: str
    configuration_hash: str
    items: tuple[ConversionItem, ...]

    def __post_init__(self) -> None:
        validate_digest(self.source_root, name="conversion.source_root")
        validate_digest(self.configuration_hash, name="conversion.configuration_hash")
        items = tuple(sorted(self.items, key=lambda item: item.target_chunk_id))
        if not items:
            raise AmsError(ErrorCode.PLAN_INVALID, "conversion plan has no items")
        identifiers = [item.target_chunk_id for item in items]
        if len(set(identifiers)) != len(identifiers):
            raise AmsError(ErrorCode.PLAN_INVALID, "conversion target chunk IDs are not unique")
        object.__setattr__(self, "items", items)

    def planned_journal(self) -> ConversionJournal:
        return ConversionJournal(
            journal_version=_JOURNAL_VERSION,
            source_root=self.source_root,
            configuration_hash=self.configuration_hash,
            entries=tuple(
                ConversionJournalEntry(
                    source_range=item.source_range,
                    target_chunk_id=item.target_chunk_id,
                    state=JournalEntryState.PLANNED,
                )
                for item in self.items
            ),
        )


def _journal_to_dict(journal: ConversionJournal) -> dict[str, Any]:
    return {
        "journal_version": journal.journal_version,
        "source_root": journal.source_root,
        "configuration_hash": journal.configuration_hash,
        "entries": [
            {
                "source_range": {
                    "object_id": entry.source_range.object_id,
                    "offset": entry.source_range.offset,
                    "length": entry.source_range.length,
                    "checksum": entry.source_range.checksum,
                },
                "target_chunk_id": entry.target_chunk_id,
                "state": entry.state.value,
                **({"target_hash": entry.target_hash} if entry.target_hash is not None else {}),
                **(
                    {"encoded_bytes": entry.encoded_bytes}
                    if entry.encoded_bytes is not None
                    else {}
                ),
            }
            for entry in journal.entries
        ],
    }


class _DuplicateJournalKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJournalKey(key)
        result[key] = value
    return result


def _require_exact_fields(
    value: dict[str, Any], required: set[str], optional: set[str] | None = None
) -> None:
    optional = optional or set()
    if not required <= set(value) or not set(value) <= required | optional:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal fields are invalid")


def _journal_from_dict(value: Any) -> ConversionJournal:
    if not isinstance(value, dict):
        raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal must be an object")
    _require_exact_fields(
        value,
        {"journal_version", "source_root", "configuration_hash", "entries"},
    )
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) > 1_000_000:
        raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal entries are invalid")
    entries: list[ConversionJournalEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal entry must be an object")
        _require_exact_fields(
            raw_entry,
            {"source_range", "target_chunk_id", "state"},
            {"target_hash", "encoded_bytes"},
        )
        raw_range = raw_entry["source_range"]
        if not isinstance(raw_range, dict):
            raise AmsError(ErrorCode.INVALID_PACKAGE, "journal source range must be an object")
        _require_exact_fields(raw_range, {"object_id", "offset", "length", "checksum"})
        entries.append(
            ConversionJournalEntry(
                source_range=ByteRange(
                    object_id=raw_range["object_id"],
                    offset=raw_range["offset"],
                    length=raw_range["length"],
                    checksum=raw_range["checksum"],
                ),
                target_chunk_id=raw_entry["target_chunk_id"],
                state=JournalEntryState(raw_entry["state"]),
                target_hash=raw_entry.get("target_hash"),
                encoded_bytes=raw_entry.get("encoded_bytes"),
            )
        )
    return ConversionJournal(
        journal_version=value["journal_version"],
        source_root=value["source_root"],
        configuration_hash=value["configuration_hash"],
        entries=tuple(entries),
    )


class ConversionJournalStore:
    """A bounded journal whose visibility point is an atomic file replacement."""

    def __init__(self, path: Path, *, max_bytes: int = _MAX_JOURNAL_BYTES):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise AmsError(ErrorCode.PLAN_INVALID, "journal size limit must be positive")
        self.path = path
        self.max_bytes = max_bytes

    def load(self) -> ConversionJournal:
        try:
            if self.path.is_symlink() or not self.path.is_file():
                raise AmsError(
                    ErrorCode.INVALID_PACKAGE, "conversion journal is not a regular file"
                )
            size = self.path.stat().st_size
            if size == 0 or size > self.max_bytes:
                raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal size is invalid")
            payload = self.path.read_bytes()
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.IO_FAILURE,
                "conversion journal could not be read",
                retriable=True,
            ) from exc
        try:
            value = json.loads(payload, object_pairs_hook=_unique_object)
            return _journal_from_dict(value)
        except AmsError:
            raise
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJournalKey,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise AmsError(ErrorCode.INVALID_PACKAGE, "conversion journal is malformed") from exc

    def write(self, journal: ConversionJournal) -> None:
        payload = canonical_json_bytes(_journal_to_dict(journal))
        if len(payload) > self.max_bytes:
            raise AmsError(
                ErrorCode.TRANSACTION_FAILURE, "conversion journal exceeds its size limit"
            )
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb", buffering=0) as handle:
                written = 0
                while written < len(payload):
                    count = handle.write(payload[written:])
                    if count is None or count == 0:
                        raise AmsError(
                            ErrorCode.IO_FAILURE,
                            "short write to conversion journal",
                            retriable=True,
                        )
                    written += count
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except AmsError:
            raise
        except OSError as exc:
            raise AmsError(
                ErrorCode.TRANSACTION_FAILURE,
                "conversion journal publication failed",
                retriable=True,
            ) from exc

    def load_or_create(self, plan: ConversionPlan) -> ConversionJournal:
        if self.path.exists():
            journal = self.load()
            expected = plan.planned_journal()
            if (
                journal.journal_version != expected.journal_version
                or journal.source_root != expected.source_root
                or journal.configuration_hash != expected.configuration_hash
                or tuple((entry.target_chunk_id, entry.source_range) for entry in journal.entries)
                != tuple((entry.target_chunk_id, entry.source_range) for entry in expected.entries)
            ):
                raise AmsError(ErrorCode.PLAN_INVALID, "conversion journal does not match the plan")
            return journal
        journal = plan.planned_journal()
        self.write(journal)
        return journal


def execute_identity_conversion(
    reader: RangeReader,
    plan: ConversionPlan,
    destination_root: Path,
    journal_path: Path,
    *,
    buffer_bytes: int = 1024 * 1024,
) -> ConversionJournal:
    """Publish every planned range, recording a durable per-chunk visibility decision."""
    store = ConversionJournalStore(journal_path)
    journal = store.load_or_create(plan)
    entries = {entry.target_chunk_id: entry for entry in journal.entries}
    for item in plan.items:
        entry = entries[item.target_chunk_id]
        copy_range_atomic(
            reader,
            item.source_range,
            destination_root,
            item.target_chunk_id,
            buffer_bytes=buffer_bytes,
        )
        if entry.state is not JournalEntryState.PUBLISHED:
            entry = replace(
                entry,
                state=JournalEntryState.PUBLISHED,
                target_hash=item.source_range.checksum,
                encoded_bytes=item.source_range.length,
            )
            entries[item.target_chunk_id] = entry
            journal = replace(
                journal,
                entries=tuple(entries[planned.target_chunk_id] for planned in plan.items),
            )
            store.write(journal)
    return journal
