from dataclasses import replace

import pytest

from ams.canonical import canonical_json_bytes
from ams.descriptors import (
    ByteRange,
    ConversionJournal,
    ConversionJournalEntry,
    JournalEntryState,
    QuantizationKind,
    QuantizationSpec,
)
from ams.errors import AmsError

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def test_byte_range_accepts_exact_object_boundary() -> None:
    byte_range = ByteRange("weights-0", offset=16, length=16, checksum=HASH_A)
    byte_range.validate_within(32)
    assert byte_range.end == 32


def test_byte_range_rejects_overrun() -> None:
    byte_range = ByteRange("weights-0", offset=17, length=16, checksum=HASH_A)
    with pytest.raises(AmsError, match="exceeds"):
        byte_range.validate_within(32)


def test_ternary_contract_is_symmetric_and_canonical() -> None:
    specification = QuantizationSpec(QuantizationKind.TERNARY, group_size=128, axis=1)
    assert canonical_json_bytes(specification) == (
        b'{"axis":1,"group_size":128,"kind":"ternary","scale_dtype":"float16","zero_point":false}'
    )
    with pytest.raises(AmsError, match="zero point"):
        replace(specification, zero_point=True)


def test_conversion_journal_requires_content_addressed_completion() -> None:
    source = ByteRange("source-0", offset=0, length=8, checksum=HASH_A)
    planned = ConversionJournalEntry(source, "target-0", JournalEntryState.PLANNED)
    journal = ConversionJournal("1.0.0", HASH_A, HASH_B, (planned,))
    assert journal.entries == (planned,)
    with pytest.raises(AmsError, match="require target hash"):
        replace(planned, state=JournalEntryState.VERIFIED)


def test_conversion_journal_rejects_duplicate_target_chunks() -> None:
    source = ByteRange("source-0", offset=0, length=8, checksum=HASH_A)
    entry = ConversionJournalEntry(source, "target-0", JournalEntryState.PLANNED)
    with pytest.raises(AmsError, match="not unique"):
        ConversionJournal("1.0.0", HASH_A, HASH_B, (entry, entry))
