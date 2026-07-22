"""Explicit per-tensor identity/ternary conversion for Hugging Face catalogs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ams.conversion import ConversionJournalStore
from ams.descriptors import JournalEntryState
from ams.errors import AmsError, ErrorCode
from ams.integrations.huggingface import (
    HuggingFaceCatalog,
    HuggingFaceMixedPlan,
    HuggingFaceTensorEncoding,
)
from ams.storage import copy_range_atomic
from ams.ternary_conversion import TernaryChunkSpec, publish_ternary_chunk_atomic


def execute_huggingface_mixed_conversion(
    catalog: HuggingFaceCatalog,
    plan: HuggingFaceMixedPlan,
    destination_root: Path,
    journal_path: Path,
    *,
    verification_buffer_bytes: int = 1024 * 1024,
):
    """Execute an exact, fully assigned mixed storage policy and journal every chunk."""
    if plan.conversion.source_root != catalog.source_root:
        raise AmsError(ErrorCode.PLAN_INVALID, "mixed plan and catalog source roots differ")
    source_by_id = {source.object_id: source for source in catalog.sources}
    planned_by_id = {tensor.target_chunk_id: tensor for tensor in plan.tensors}
    if len(planned_by_id) != len(plan.tensors) or set(planned_by_id) != {
        item.target_chunk_id for item in plan.conversion.items
    }:
        raise AmsError(ErrorCode.PLAN_INVALID, "mixed tensor set differs from conversion items")
    item_by_id = {item.target_chunk_id: item for item in plan.conversion.items}
    store = ConversionJournalStore(journal_path)
    journal = store.load_or_create(plan.conversion)
    entries = {entry.target_chunk_id: entry for entry in journal.entries}
    for planned in plan.tensors:
        item = item_by_id[planned.target_chunk_id]
        if item.source_range.object_id not in source_by_id:
            raise AmsError(ErrorCode.CAPABILITY_MISMATCH, "mixed conversion source is missing")
        reader = source_by_id[item.source_range.object_id].reader
        if planned.encoding is HuggingFaceTensorEncoding.IDENTITY:
            copy_range_atomic(
                reader,
                item.source_range,
                destination_root,
                item.target_chunk_id,
                buffer_bytes=verification_buffer_bytes,
            )
            target_hash = item.source_range.checksum
            encoded_bytes = item.source_range.length
        elif planned.encoding is HuggingFaceTensorEncoding.TERNARY_TRIT5:
            if planned.ternary_config is None:
                raise AmsError(ErrorCode.INTERNAL_INVARIANT, "ternary mixed tensor has no config")
            publication = publish_ternary_chunk_atomic(
                reader,
                TernaryChunkSpec(
                    publication_key=item.target_chunk_id,
                    source=item.source_range,
                    shape=planned.tensor.shape,
                    source_dtype=planned.tensor.dtype,
                    config=planned.ternary_config,
                ),
                destination_root,
                verification_buffer_bytes=verification_buffer_bytes,
            )
            target_hash = publication.target_hash
            encoded_bytes = publication.encoded_bytes
        else:
            raise AmsError(ErrorCode.INTERNAL_INVARIANT, "unknown mixed tensor encoding")
        entry = entries[item.target_chunk_id]
        if entry.state is JournalEntryState.PUBLISHED:
            if entry.target_hash != target_hash or entry.encoded_bytes != encoded_bytes:
                raise AmsError(
                    ErrorCode.INTEGRITY_FAILURE,
                    "mixed conversion journal and published chunk disagree",
                )
            continue
        entries[item.target_chunk_id] = replace(
            entry,
            state=JournalEntryState.PUBLISHED,
            target_hash=target_hash,
            encoded_bytes=encoded_bytes,
        )
        journal = replace(
            journal,
            entries=tuple(entries[item.target_chunk_id] for item in plan.conversion.items),
        )
        store.write(journal)
    return journal
