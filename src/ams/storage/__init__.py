"""Bounded storage adapters."""

from ams.storage.copy import copy_range_atomic
from ams.storage.file import FileRangeStore, RangeReader

__all__ = ["FileRangeStore", "RangeReader", "copy_range_atomic"]
