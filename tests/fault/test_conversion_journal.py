import hashlib
import json
from pathlib import Path

import pytest

from ams.conversion import (
    ConversionItem,
    ConversionJournalStore,
    ConversionPlan,
    execute_identity_conversion,
)
from ams.descriptors import ByteRange, JournalEntryState
from ams.errors import AmsError, ErrorCode


class RecordingReader:
    def __init__(self, payload: bytes, *, fail_at_or_after: int | None = None):
        self.payload = payload
        self.size_bytes = len(payload)
        self.fail_at_or_after = fail_at_or_after
        self.offsets: list[int] = []

    def read_into(self, offset: int, destination) -> None:
        if self.fail_at_or_after is not None and offset >= self.fail_at_or_after:
            raise AmsError(ErrorCode.IO_FAILURE, "injected conversion failure", retriable=True)
        self.offsets.append(offset)
        view = memoryview(destination).cast("B")
        view[:] = self.payload[offset : offset + view.nbytes]


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def build_plan(payload: bytes, split: int) -> ConversionPlan:
    return ConversionPlan(
        source_root=digest(payload),
        configuration_hash=digest(b"identity-v1"),
        items=(
            ConversionItem(
                "chunk-a",
                ByteRange("source", 0, split, digest(payload[:split])),
            ),
            ConversionItem(
                "chunk-b",
                ByteRange("source", split, len(payload) - split, digest(payload[split:])),
            ),
        ),
    )


def test_journal_resumes_after_process_boundary_without_rereading_published_chunk(
    tmp_path: Path,
) -> None:
    payload = bytes(range(251)) * 10
    split = 1_234
    plan = build_plan(payload, split)
    output = tmp_path / "package"
    journal_path = output / "conversion.journal.json"
    interrupted = RecordingReader(payload, fail_at_or_after=split)
    with pytest.raises(AmsError) as caught:
        execute_identity_conversion(
            interrupted,
            plan,
            output,
            journal_path,
            buffer_bytes=97,
        )
    assert caught.value.retriable
    partial = ConversionJournalStore(journal_path).load()
    assert [entry.state for entry in partial.entries] == [
        JournalEntryState.PUBLISHED,
        JournalEntryState.PLANNED,
    ]

    resumed = RecordingReader(payload)
    complete = execute_identity_conversion(
        resumed,
        plan,
        output,
        journal_path,
        buffer_bytes=89,
    )
    assert all(entry.state is JournalEntryState.PUBLISHED for entry in complete.entries)
    assert resumed.offsets
    assert min(resumed.offsets) >= split

    no_source_reads = RecordingReader(payload, fail_at_or_after=0)
    execute_identity_conversion(no_source_reads, plan, output, journal_path, buffer_bytes=31)
    assert no_source_reads.offsets == []


def test_malformed_journal_is_a_permanent_package_error(tmp_path: Path) -> None:
    path = tmp_path / "conversion.journal.json"
    path.write_text('{"source_root":"duplicate","source_root":"value"}', encoding="utf-8")
    with pytest.raises(AmsError) as caught:
        ConversionJournalStore(path).load()
    assert caught.value.code is ErrorCode.INVALID_PACKAGE
    assert not caught.value.retriable


def test_bad_journal_enum_type_is_normalized_to_package_error(tmp_path: Path) -> None:
    path = tmp_path / "conversion.journal.json"
    path.write_text(
        json.dumps(
            {
                "journal_version": "1.0.0",
                "source_root": digest(b"source"),
                "configuration_hash": digest(b"config"),
                "entries": [
                    {
                        "source_range": {
                            "object_id": "source",
                            "offset": 0,
                            "length": 1,
                            "checksum": digest(b"x"),
                        },
                        "target_chunk_id": "target",
                        "state": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AmsError) as caught:
        ConversionJournalStore(path).load()
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


def test_existing_journal_cannot_be_reused_for_a_different_plan(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    first = build_plan(payload, 4)
    journal_path = tmp_path / "conversion.journal.json"
    store = ConversionJournalStore(journal_path)
    store.load_or_create(first)
    changed = ConversionPlan(
        source_root=first.source_root,
        configuration_hash=digest(b"different-config"),
        items=first.items,
    )
    with pytest.raises(AmsError) as caught:
        store.load_or_create(changed)
    assert caught.value.code is ErrorCode.PLAN_INVALID
