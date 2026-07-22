"""Scalar semantic oracles with bounded weight and output residency."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ams.checked import checked_add, checked_mul, checked_positive, checked_uint
from ams.codecs import (
    Int4CodecConfig,
    TernaryCodecConfig,
    decode_int4_group_reference,
    decode_ternary_group_reference,
)
from ams.descriptors import DType
from ams.errors import AmsError, ErrorCode
from ams.storage import RangeReader

_ACCUMULATOR_BYTES = 8
_IDENTITY_ITEM_BYTES = {
    DType.FLOAT16: 2,
    DType.BFLOAT16: 2,
    DType.FLOAT32: 4,
}


@dataclass(frozen=True, slots=True)
class StreamedLinearPlan:
    rows: int
    columns: int
    weight_offset: int
    arena_bytes: int
    reduction_tile: int
    working_set_bytes: int
    storage_dtype: DType
    item_bytes: int

    @classmethod
    def create(
        cls,
        *,
        rows: int,
        columns: int,
        weight_offset: int,
        arena_bytes: int,
        storage_dtype: DType = DType.FLOAT32,
    ) -> StreamedLinearPlan:
        checked_positive(rows, name="linear.rows")
        checked_positive(columns, name="linear.columns")
        checked_uint(weight_offset, name="linear.weight_offset")
        checked_positive(arena_bytes, name="linear.arena_bytes")
        try:
            storage_dtype = DType(storage_dtype)
            item_bytes = _IDENTITY_ITEM_BYTES[storage_dtype]
        except (KeyError, TypeError, ValueError) as exc:
            raise AmsError(
                ErrorCode.CAPABILITY_MISMATCH,
                "linear identity storage dtype is unsupported",
            ) from exc
        minimum = item_bytes + _ACCUMULATOR_BYTES
        if arena_bytes < minimum:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "linear arena cannot hold one weight and its accumulator",
                evidence={"available": arena_bytes, "minimum": minimum},
            )
        reduction_tile = min(columns, (arena_bytes - _ACCUMULATOR_BYTES) // item_bytes)
        working_set_bytes = checked_add(
            checked_mul(reduction_tile, item_bytes, name="linear.weight_buffer"),
            _ACCUMULATOR_BYTES,
            name="linear.working_set",
        )
        weight_elements = checked_mul(rows, columns, name="linear.weight_elements")
        checked_add(
            weight_offset,
            checked_mul(weight_elements, item_bytes, name="linear.weight_bytes"),
            name="linear.weight_end",
        )
        return cls(
            rows=rows,
            columns=columns,
            weight_offset=weight_offset,
            arena_bytes=arena_bytes,
            reduction_tile=reduction_tile,
            working_set_bytes=working_set_bytes,
            storage_dtype=storage_dtype,
            item_bytes=item_bytes,
        )


def stream_linear_f32(
    store: RangeReader,
    plan: StreamedLinearPlan,
    vector: Sequence[float],
    emit: Callable[[int, float], None],
    *,
    bias: Sequence[float] | None = None,
) -> None:
    """Compute row-major FP32 weights through the typed identity implementation."""
    if plan.storage_dtype is not DType.FLOAT32:
        raise AmsError(ErrorCode.PLAN_INVALID, "FP32 linear received a non-FP32 plan")
    stream_linear_identity(store, plan, vector, emit, bias=bias)


def _decode_identity_value(buffer: bytearray, offset: int, dtype: DType) -> float:
    if dtype is DType.FLOAT32:
        return struct.unpack_from("<f", buffer, offset)[0]
    if dtype is DType.FLOAT16:
        return struct.unpack_from("<e", buffer, offset)[0]
    if dtype is DType.BFLOAT16:
        word = struct.unpack_from("<H", buffer, offset)[0]
        return struct.unpack("<f", struct.pack("<I", word << 16))[0]
    raise AmsError(ErrorCode.INTERNAL_INVARIANT, "identity linear dtype changed after planning")


def stream_linear_identity(
    store: RangeReader,
    plan: StreamedLinearPlan,
    vector: Sequence[float],
    emit: Callable[[int, float], None],
    *,
    bias: Sequence[float] | None = None,
) -> None:
    """Compute row-major FP16/BF16/FP32 weights one output scalar at a time."""
    if len(vector) != plan.columns:
        raise AmsError(ErrorCode.PLAN_INVALID, "linear input length does not match columns")
    if bias is not None and len(bias) != plan.rows:
        raise AmsError(ErrorCode.PLAN_INVALID, "linear bias length does not match rows")
    if any(not math.isfinite(float(value)) for value in vector) or (
        bias is not None and any(not math.isfinite(float(value)) for value in bias)
    ):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "identity linear input or bias is non-finite")
    weight_bytes = checked_mul(
        checked_mul(plan.rows, plan.columns, name="linear.weight_elements"),
        plan.item_bytes,
        name="linear.weight_bytes",
    )
    weight_end = checked_add(plan.weight_offset, weight_bytes, name="linear.weight_end")
    if weight_end > store.size_bytes:
        raise AmsError(ErrorCode.IO_FAILURE, "linear weights exceed the storage object")

    buffer = bytearray(plan.reduction_tile * plan.item_bytes)
    view = memoryview(buffer)
    try:
        for row in range(plan.rows):
            accumulator = float(bias[row]) if bias is not None else 0.0
            row_base = checked_add(
                plan.weight_offset,
                checked_mul(
                    checked_mul(row, plan.columns, name="linear.row_elements"),
                    plan.item_bytes,
                    name="linear.row_bytes",
                ),
                name="linear.row_offset",
            )
            for start in range(0, plan.columns, plan.reduction_tile):
                count = min(plan.reduction_tile, plan.columns - start)
                byte_count = count * plan.item_bytes
                offset = checked_add(
                    row_base,
                    start * plan.item_bytes,
                    name="linear.tile_offset",
                )
                store.read_into(offset, view[:byte_count])
                for index in range(count):
                    weight = _decode_identity_value(
                        buffer,
                        index * plan.item_bytes,
                        plan.storage_dtype,
                    )
                    if not math.isfinite(weight):
                        raise AmsError(
                            ErrorCode.NUMERIC_FAILURE,
                            "identity linear weight is non-finite",
                        )
                    accumulator += weight * float(vector[start + index])
            if not math.isfinite(accumulator):
                raise AmsError(ErrorCode.NUMERIC_FAILURE, "identity linear output is non-finite")
            emit(row, accumulator)
    finally:
        view.release()


@dataclass(frozen=True, slots=True)
class TernaryStreamedLinearPlan:
    rows: int
    columns: int
    weight_offset: int
    arena_bytes: int
    output_row_tile: int
    working_set_bytes: int
    config: TernaryCodecConfig

    @classmethod
    def create(
        cls,
        *,
        rows: int,
        columns: int,
        weight_offset: int,
        arena_bytes: int,
        config: TernaryCodecConfig | None = None,
    ) -> TernaryStreamedLinearPlan:
        config = config or TernaryCodecConfig()
        checked_positive(rows, name="ternary_linear.rows")
        checked_positive(columns, name="ternary_linear.columns")
        checked_uint(weight_offset, name="ternary_linear.weight_offset")
        checked_positive(arena_bytes, name="ternary_linear.arena_bytes")
        record_bytes = config.group_record_size(config.group_size)
        decoded_group_bytes = checked_mul(
            config.group_size,
            _ACCUMULATOR_BYTES,
            name="ternary_linear.decoded_group",
        )
        fixed_bytes = checked_add(
            record_bytes,
            decoded_group_bytes,
            name="ternary_linear.fixed_working_set",
        )
        minimum = checked_add(
            fixed_bytes,
            _ACCUMULATOR_BYTES,
            name="ternary_linear.minimum_working_set",
        )
        if arena_bytes < minimum:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "ternary linear arena cannot hold one group and output accumulator",
                evidence={"available": arena_bytes, "minimum": minimum},
            )
        output_row_tile = min(rows, (arena_bytes - fixed_bytes) // _ACCUMULATOR_BYTES)
        working_set_bytes = checked_add(
            fixed_bytes,
            checked_mul(
                output_row_tile,
                _ACCUMULATOR_BYTES,
                name="ternary_linear.output_accumulators",
            ),
            name="ternary_linear.working_set",
        )
        element_count = checked_mul(rows, columns, name="ternary_linear.elements")
        checked_add(
            weight_offset,
            config.encoded_size(element_count),
            name="ternary_linear.weight_end",
        )
        return cls(
            rows=rows,
            columns=columns,
            weight_offset=weight_offset,
            arena_bytes=arena_bytes,
            output_row_tile=output_row_tile,
            working_set_bytes=working_set_bytes,
            config=config,
        )


def stream_linear_ternary(
    store: RangeReader,
    plan: TernaryStreamedLinearPlan,
    vector: Sequence[float],
    emit: Callable[[int, float], None],
    *,
    bias: Sequence[float] | None = None,
) -> None:
    """Multiply directly from grouped ternary storage with bounded row/group tiles."""
    if len(vector) != plan.columns:
        raise AmsError(ErrorCode.PLAN_INVALID, "ternary linear input length is invalid")
    if bias is not None and len(bias) != plan.rows:
        raise AmsError(ErrorCode.PLAN_INVALID, "ternary linear bias length is invalid")
    if any(not math.isfinite(float(value)) for value in vector) or (
        bias is not None and any(not math.isfinite(float(value)) for value in bias)
    ):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "ternary linear input or bias is non-finite")
    element_count = checked_mul(plan.rows, plan.columns, name="ternary_linear.elements")
    encoded_bytes = plan.config.encoded_size(element_count)
    if (
        checked_add(plan.weight_offset, encoded_bytes, name="ternary_linear.weight_end")
        > store.size_bytes
    ):
        raise AmsError(ErrorCode.IO_FAILURE, "ternary linear weights exceed the storage object")
    full_record_bytes = plan.config.group_record_size(plan.config.group_size)
    record_buffer = bytearray(full_record_bytes)
    record_view = memoryview(record_buffer)
    try:
        for row_start in range(0, plan.rows, plan.output_row_tile):
            row_count = min(plan.output_row_tile, plan.rows - row_start)
            accumulators = [
                float(bias[row_start + index]) if bias is not None else 0.0
                for index in range(row_count)
            ]
            flat_start = checked_mul(row_start, plan.columns, name="ternary_linear.flat_start")
            flat_end = checked_mul(
                row_start + row_count,
                plan.columns,
                name="ternary_linear.flat_end",
            )
            first_group = flat_start // plan.config.group_size
            final_group = (flat_end - 1) // plan.config.group_size
            for group_index in range(first_group, final_group + 1):
                group_flat_start = group_index * plan.config.group_size
                count = min(plan.config.group_size, element_count - group_flat_start)
                record_size = plan.config.group_record_size(count)
                record_offset = checked_add(
                    plan.weight_offset,
                    checked_mul(
                        group_index,
                        full_record_bytes,
                        name="ternary_linear.group_record_offset",
                    ),
                    name="ternary_linear.record_offset",
                )
                store.read_into(record_offset, record_view[:record_size])
                values = decode_ternary_group_reference(record_view[:record_size], count)
                local_start = max(flat_start, group_flat_start) - group_flat_start
                local_end = min(flat_end, group_flat_start + count) - group_flat_start
                for local_index in range(local_start, local_end):
                    flat_index = group_flat_start + local_index
                    output_index = flat_index // plan.columns - row_start
                    input_index = flat_index % plan.columns
                    accumulators[output_index] += values[local_index] * float(vector[input_index])
            for index, value in enumerate(accumulators):
                if not math.isfinite(value):
                    raise AmsError(ErrorCode.NUMERIC_FAILURE, "ternary linear output is non-finite")
                emit(row_start + index, value)
    finally:
        record_view.release()


@dataclass(frozen=True, slots=True)
class Int4StreamedLinearPlan:
    rows: int
    columns: int
    weight_offset: int
    arena_bytes: int
    output_row_tile: int
    working_set_bytes: int
    config: Int4CodecConfig

    @classmethod
    def create(
        cls,
        *,
        rows: int,
        columns: int,
        weight_offset: int,
        arena_bytes: int,
        config: Int4CodecConfig | None = None,
    ) -> Int4StreamedLinearPlan:
        config = config or Int4CodecConfig()
        checked_positive(rows, name="int4_linear.rows")
        checked_positive(columns, name="int4_linear.columns")
        checked_uint(weight_offset, name="int4_linear.weight_offset")
        checked_positive(arena_bytes, name="int4_linear.arena_bytes")
        record_bytes = config.group_record_size(config.group_size)
        decoded_group_bytes = checked_mul(
            config.group_size,
            _ACCUMULATOR_BYTES,
            name="int4_linear.decoded_group",
        )
        fixed_bytes = checked_add(
            record_bytes,
            decoded_group_bytes,
            name="int4_linear.fixed_working_set",
        )
        minimum = checked_add(
            fixed_bytes,
            _ACCUMULATOR_BYTES,
            name="int4_linear.minimum_working_set",
        )
        if arena_bytes < minimum:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "INT4 linear arena cannot hold one group and output accumulator",
                evidence={"available": arena_bytes, "minimum": minimum},
            )
        output_row_tile = min(rows, (arena_bytes - fixed_bytes) // _ACCUMULATOR_BYTES)
        working_set_bytes = checked_add(
            fixed_bytes,
            checked_mul(
                output_row_tile,
                _ACCUMULATOR_BYTES,
                name="int4_linear.output_accumulators",
            ),
            name="int4_linear.working_set",
        )
        element_count = checked_mul(rows, columns, name="int4_linear.elements")
        checked_add(
            weight_offset,
            config.encoded_size(element_count),
            name="int4_linear.weight_end",
        )
        return cls(
            rows=rows,
            columns=columns,
            weight_offset=weight_offset,
            arena_bytes=arena_bytes,
            output_row_tile=output_row_tile,
            working_set_bytes=working_set_bytes,
            config=config,
        )


def stream_linear_int4(
    store: RangeReader,
    plan: Int4StreamedLinearPlan,
    vector: Sequence[float],
    emit: Callable[[int, float], None],
    *,
    bias: Sequence[float] | None = None,
) -> None:
    """Multiply directly from grouped symmetric INT4 with bounded row/group tiles."""
    if len(vector) != plan.columns:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT4 linear input length is invalid")
    if bias is not None and len(bias) != plan.rows:
        raise AmsError(ErrorCode.PLAN_INVALID, "INT4 linear bias length is invalid")
    if any(not math.isfinite(float(value)) for value in vector) or (
        bias is not None and any(not math.isfinite(float(value)) for value in bias)
    ):
        raise AmsError(ErrorCode.NUMERIC_FAILURE, "INT4 linear input or bias is non-finite")
    element_count = checked_mul(plan.rows, plan.columns, name="int4_linear.elements")
    encoded_bytes = plan.config.encoded_size(element_count)
    if (
        checked_add(plan.weight_offset, encoded_bytes, name="int4_linear.weight_end")
        > store.size_bytes
    ):
        raise AmsError(ErrorCode.IO_FAILURE, "INT4 linear weights exceed the storage object")
    full_record_bytes = plan.config.group_record_size(plan.config.group_size)
    record_buffer = bytearray(full_record_bytes)
    record_view = memoryview(record_buffer)
    try:
        for row_start in range(0, plan.rows, plan.output_row_tile):
            row_count = min(plan.output_row_tile, plan.rows - row_start)
            accumulators = [
                float(bias[row_start + index]) if bias is not None else 0.0
                for index in range(row_count)
            ]
            flat_start = checked_mul(row_start, plan.columns, name="int4_linear.flat_start")
            flat_end = checked_mul(
                row_start + row_count,
                plan.columns,
                name="int4_linear.flat_end",
            )
            first_group = flat_start // plan.config.group_size
            final_group = (flat_end - 1) // plan.config.group_size
            for group_index in range(first_group, final_group + 1):
                group_flat_start = group_index * plan.config.group_size
                count = min(plan.config.group_size, element_count - group_flat_start)
                record_size = plan.config.group_record_size(count)
                record_offset = checked_add(
                    plan.weight_offset,
                    checked_mul(
                        group_index,
                        full_record_bytes,
                        name="int4_linear.group_record_offset",
                    ),
                    name="int4_linear.record_offset",
                )
                store.read_into(record_offset, record_view[:record_size])
                values = decode_int4_group_reference(record_view[:record_size], count)
                local_start = max(flat_start, group_flat_start) - group_flat_start
                local_end = min(flat_end, group_flat_start + count) - group_flat_start
                for local_index in range(local_start, local_end):
                    flat_index = group_flat_start + local_index
                    output_index = flat_index // plan.columns - row_start
                    input_index = flat_index % plan.columns
                    accumulators[output_index] += values[local_index] * float(vector[input_index])
            for index, value in enumerate(accumulators):
                if not math.isfinite(value):
                    raise AmsError(ErrorCode.NUMERIC_FAILURE, "INT4 linear output is non-finite")
                emit(row_start + index, value)
    finally:
        record_view.release()
