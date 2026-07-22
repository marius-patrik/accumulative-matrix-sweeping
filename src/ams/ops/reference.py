"""Scalar semantic oracles with bounded weight and output residency."""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ams.checked import checked_add, checked_mul, checked_positive, checked_uint
from ams.codecs import TernaryCodecConfig, decode_ternary_group_reference
from ams.errors import AmsError, ErrorCode
from ams.storage import RangeReader

_F32_BYTES = 4
_ACCUMULATOR_BYTES = 8


@dataclass(frozen=True, slots=True)
class StreamedLinearPlan:
    rows: int
    columns: int
    weight_offset: int
    arena_bytes: int
    reduction_tile: int
    working_set_bytes: int

    @classmethod
    def create(
        cls,
        *,
        rows: int,
        columns: int,
        weight_offset: int,
        arena_bytes: int,
    ) -> StreamedLinearPlan:
        checked_positive(rows, name="linear.rows")
        checked_positive(columns, name="linear.columns")
        checked_uint(weight_offset, name="linear.weight_offset")
        checked_positive(arena_bytes, name="linear.arena_bytes")
        minimum = _F32_BYTES + _ACCUMULATOR_BYTES
        if arena_bytes < minimum:
            raise AmsError(
                ErrorCode.PREFLIGHT_NO_WORKING_SET,
                "linear arena cannot hold one weight and its accumulator",
                evidence={"available": arena_bytes, "minimum": minimum},
            )
        reduction_tile = min(columns, (arena_bytes - _ACCUMULATOR_BYTES) // _F32_BYTES)
        working_set_bytes = checked_add(
            checked_mul(reduction_tile, _F32_BYTES, name="linear.weight_buffer"),
            _ACCUMULATOR_BYTES,
            name="linear.working_set",
        )
        weight_elements = checked_mul(rows, columns, name="linear.weight_elements")
        checked_add(
            weight_offset,
            checked_mul(weight_elements, _F32_BYTES, name="linear.weight_bytes"),
            name="linear.weight_end",
        )
        return cls(
            rows=rows,
            columns=columns,
            weight_offset=weight_offset,
            arena_bytes=arena_bytes,
            reduction_tile=reduction_tile,
            working_set_bytes=working_set_bytes,
        )


def stream_linear_f32(
    store: RangeReader,
    plan: StreamedLinearPlan,
    vector: Sequence[float],
    emit: Callable[[int, float], None],
    *,
    bias: Sequence[float] | None = None,
) -> None:
    """Compute row-major FP32 weights one output scalar at a time.

    The input vector is part of the runtime base. Weight residency is bounded by
    ``plan.reduction_tile * 4`` and output residency is a single accumulator.
    """
    if len(vector) != plan.columns:
        raise AmsError(ErrorCode.PLAN_INVALID, "linear input length does not match columns")
    if bias is not None and len(bias) != plan.rows:
        raise AmsError(ErrorCode.PLAN_INVALID, "linear bias length does not match rows")
    weight_bytes = checked_mul(
        checked_mul(plan.rows, plan.columns, name="linear.weight_elements"),
        _F32_BYTES,
        name="linear.weight_bytes",
    )
    weight_end = checked_add(plan.weight_offset, weight_bytes, name="linear.weight_end")
    if weight_end > store.size_bytes:
        raise AmsError(ErrorCode.IO_FAILURE, "linear weights exceed the storage object")

    buffer = bytearray(plan.reduction_tile * _F32_BYTES)
    view = memoryview(buffer)
    try:
        for row in range(plan.rows):
            accumulator = float(bias[row]) if bias is not None else 0.0
            row_base = checked_add(
                plan.weight_offset,
                checked_mul(
                    checked_mul(row, plan.columns, name="linear.row_elements"),
                    _F32_BYTES,
                    name="linear.row_bytes",
                ),
                name="linear.row_offset",
            )
            for start in range(0, plan.columns, plan.reduction_tile):
                count = min(plan.reduction_tile, plan.columns - start)
                byte_count = count * _F32_BYTES
                offset = checked_add(row_base, start * _F32_BYTES, name="linear.tile_offset")
                store.read_into(offset, view[:byte_count])
                for index in range(count):
                    weight = struct.unpack_from("<f", buffer, index * _F32_BYTES)[0]
                    accumulator += weight * float(vector[start + index])
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
                emit(row_start + index, value)
    finally:
        record_view.release()
