from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from ams.errors import AmsError, ErrorCode
from ams.integrations import (
    HuggingFaceShardSource,
    release_huggingface_shard,
    stage_huggingface_shard,
    validate_huggingface_shard_cache_empty,
)


class RecordingReader:
    def __init__(self, payload: bytes, *, fail_on_read: bool = False) -> None:
        self.payload = payload
        self.size_bytes = len(payload)
        self.fail_on_read = fail_on_read
        self.reads = 0

    def read_into(self, offset: int, destination) -> None:
        if self.fail_on_read:
            raise AssertionError("a verified staged shard reread its remote source")
        self.reads += 1
        view = memoryview(destination).cast("B")
        view[:] = self.payload[offset : offset + view.nbytes]


def digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def source_for(reader: RecordingReader, *, content_hash: str | None = None):
    return HuggingFaceShardSource(
        "model-00001-of-00001.safetensors",
        "source:00001",
        content_hash or digest(reader.payload),
        reader,
    )


def test_verified_shard_stage_restarts_without_remote_io_and_releases_exact_file(
    tmp_path,
) -> None:
    payload = bytes(range(251)) * 20
    remote = RecordingReader(payload)
    cache_root = tmp_path / "ephemeral-source"
    staged = stage_huggingface_shard(source_for(remote), cache_root, buffer_bytes=97)
    assert remote.reads > 1
    local = bytearray(31)
    staged.source.reader.read_into(113, local)
    assert local == payload[113:144]

    unavailable = RecordingReader(payload, fail_on_read=True)
    restarted = stage_huggingface_shard(source_for(unavailable), cache_root, buffer_bytes=53)
    assert restarted.path == staged.path
    assert unavailable.reads == 0

    release_huggingface_shard(restarted)
    assert not restarted.path.exists()
    release_huggingface_shard(restarted)
    validate_huggingface_shard_cache_empty(cache_root)


def test_bad_source_hash_never_publishes_a_staged_shard(tmp_path) -> None:
    payload = b"not-the-declared-object"
    remote = RecordingReader(payload)
    with pytest.raises(AmsError) as caught:
        stage_huggingface_shard(
            source_for(remote, content_hash=digest(b"different-object")),
            tmp_path / "ephemeral-source",
            buffer_bytes=7,
        )
    assert caught.value.code is ErrorCode.INTEGRITY_FAILURE
    assert not list((tmp_path / "ephemeral-source" / "chunks").glob("*.bin"))


def test_release_refuses_path_or_cache_marker_drift(tmp_path) -> None:
    payload = b"verified-source"
    staged = stage_huggingface_shard(
        source_for(RecordingReader(payload)),
        tmp_path / "ephemeral-source",
        buffer_bytes=5,
    )
    outside = tmp_path / "unrelated.bin"
    outside.write_bytes(b"preserve me")
    with pytest.raises(AmsError) as caught:
        release_huggingface_shard(replace(staged, path=outside))
    assert caught.value.code is ErrorCode.PLAN_INVALID
    assert outside.read_bytes() == b"preserve me"
    assert staged.path.exists()

    marker = staged.cache_root / ".ams-hf-shard-cache-v1"
    marker.write_bytes(b"tampered")
    with pytest.raises(AmsError) as caught:
        release_huggingface_shard(staged)
    assert caught.value.code is ErrorCode.INVALID_PACKAGE
    assert staged.path.exists()


def test_cache_refuses_a_second_source_lease_until_the_first_is_released(tmp_path) -> None:
    cache = tmp_path / "ephemeral-source"
    first = stage_huggingface_shard(
        source_for(RecordingReader(b"first-source")),
        cache,
        buffer_bytes=5,
    )
    second_reader = RecordingReader(b"second-source")
    second = HuggingFaceShardSource(
        "model-00002-of-00002.safetensors",
        "source:00002",
        digest(second_reader.payload),
        second_reader,
    )
    with pytest.raises(AmsError) as caught:
        stage_huggingface_shard(second, cache, buffer_bytes=5)
    assert caught.value.code is ErrorCode.BROKER_VIOLATION
    assert second_reader.reads == 0
    assert first.path.exists()
