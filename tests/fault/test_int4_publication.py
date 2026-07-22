import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

import ams.int4_conversion as conversion_module
from ams.codecs import Int4CodecConfig, decode_int4_reference
from ams.conversion import ConversionItem, ConversionPlan
from ams.descriptors import ByteRange, DType, JournalEntryState
from ams.errors import AmsError, ErrorCode
from ams.int4_conversion import (
    Int4ChunkSpec,
    execute_int4_conversion,
    publish_int4_chunk_atomic,
)


class MemoryReader:
    def __init__(self, payload: bytes, *, fail_on_read: bool = False):
        self.payload = payload
        self.size_bytes = len(payload)
        self.fail_on_read = fail_on_read
        self.reads = 0

    def read_into(self, offset: int, destination) -> None:
        if self.fail_on_read:
            raise AmsError(ErrorCode.IO_FAILURE, "unexpected source reread")
        self.reads += 1
        view = memoryview(destination).cast("B")
        view[:] = self.payload[offset : offset + view.nbytes]


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def make_spec(payload: bytes) -> Int4ChunkSpec:
    return Int4ChunkSpec(
        publication_key="tensor:weight",
        source=ByteRange("source", 0, len(payload), digest(payload)),
        shape=(len(payload) // 4,),
        source_dtype=DType.FLOAT32,
        config=Int4CodecConfig(group_size=5),
    )


def make_plan(spec: Int4ChunkSpec) -> ConversionPlan:
    return ConversionPlan(
        source_root=digest(b"source-root"),
        configuration_hash=spec.config.config_hash,
        items=(ConversionItem(spec.publication_key, spec.source),),
    )


def test_completed_publication_updates_new_journal_without_source_reread(tmp_path: Path) -> None:
    values = [-8.0, -4.0, -1.0, 0.0, 1.0]
    payload = struct.pack("<5f", *values)
    spec = make_spec(payload)
    publication = publish_int4_chunk_atomic(MemoryReader(payload), spec, tmp_path)
    decoded = decode_int4_reference(publication.path.read_bytes(), 5, spec.config)
    assert decoded[0] == pytest.approx(-8.0)
    assert decoded[-1] == pytest.approx(8.0 / 7.0)

    no_reread = MemoryReader(payload, fail_on_read=True)
    journal = execute_int4_conversion(
        {"source": no_reread},
        make_plan(spec),
        (spec,),
        tmp_path,
        tmp_path / "conversion.journal.json",
    )
    assert no_reread.reads == 0
    assert journal.entries[0].state is JournalEntryState.PUBLISHED
    assert journal.entries[0].target_hash == publication.target_hash


def test_crash_between_chunk_and_record_finalization_recovers_without_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = struct.pack("<10f", *[float(index - 5) for index in range(10)])
    spec = make_spec(payload)
    real_replace = conversion_module.os.replace
    calls = 0

    def fail_third_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected record finalization failure")
        return real_replace(source, destination)

    monkeypatch.setattr(conversion_module.os, "replace", fail_third_replace)
    with pytest.raises(AmsError) as caught:
        publish_int4_chunk_atomic(MemoryReader(payload), spec, tmp_path)
    assert caught.value.code is ErrorCode.TRANSACTION_FAILURE
    assert list((tmp_path / ".records").glob("*.pending.json"))
    assert list((tmp_path / "chunks").glob("*.bin"))

    monkeypatch.setattr(conversion_module.os, "replace", real_replace)
    no_reread = MemoryReader(payload, fail_on_read=True)
    recovered = publish_int4_chunk_atomic(no_reread, spec, tmp_path)
    assert recovered.path.is_file()
    assert no_reread.reads == 0
    assert not list((tmp_path / ".records").glob("*.pending.json"))


def test_pending_record_without_staged_or_final_chunk_fails_closed(tmp_path: Path) -> None:
    payload = struct.pack("<5f", -2.0, -1.0, 0.0, 1.0, 2.0)
    spec = make_spec(payload)
    real_replace = conversion_module.os.replace
    calls = 0

    def fail_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected chunk publication failure")
        return real_replace(source, destination)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(conversion_module.os, "replace", fail_second_replace)
        with pytest.raises(AmsError) as caught:
            publish_int4_chunk_atomic(MemoryReader(payload), spec, tmp_path)
    assert caught.value.code is ErrorCode.TRANSACTION_FAILURE
    next((tmp_path / ".staging").glob("*.part")).unlink()

    with pytest.raises(AmsError) as caught:
        publish_int4_chunk_atomic(MemoryReader(payload, fail_on_read=True), spec, tmp_path)
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE


def test_corrupt_publication_record_is_permanent_failure(tmp_path: Path) -> None:
    payload = struct.pack("<5f", -2.0, -1.0, 0.0, 1.0, 2.0)
    spec = make_spec(payload)
    publish_int4_chunk_atomic(MemoryReader(payload), spec, tmp_path)
    record = next((tmp_path / ".records").glob("*.json"))
    value = json.loads(record.read_text())
    value["element_count"] += 1
    record.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(AmsError) as caught:
        publish_int4_chunk_atomic(MemoryReader(payload, fail_on_read=True), spec, tmp_path)
    assert caught.value.code is ErrorCode.PLAN_INVALID


def test_int4_specs_and_configuration_must_exactly_match_plan(tmp_path: Path) -> None:
    payload = struct.pack("<f", 1.0)
    spec = make_spec(payload)
    with pytest.raises(AmsError) as caught:
        execute_int4_conversion(
            {"source": MemoryReader(payload)},
            make_plan(spec),
            (),
            tmp_path,
            tmp_path / "journal.json",
        )
    assert caught.value.code is ErrorCode.PLAN_INVALID

    other_config = replace(spec, config=Int4CodecConfig(group_size=7))
    with pytest.raises(AmsError) as caught:
        execute_int4_conversion(
            {"source": MemoryReader(payload)},
            make_plan(spec),
            (other_config,),
            tmp_path,
            tmp_path / "journal.json",
        )
    assert caught.value.code is ErrorCode.PLAN_INVALID
