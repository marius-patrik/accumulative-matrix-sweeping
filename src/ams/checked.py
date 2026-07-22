"""Checked integer and byte-range arithmetic used at all storage boundaries."""

from collections.abc import Iterable

from ams.errors import AmsError, ErrorCode

MAX_SERIALIZED_UINT = (1 << 63) - 1


def checked_uint(value: int, *, name: str) -> int:
    """Return a schema-compatible unsigned integer or raise a typed package error."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} must be an integer")
    if value < 0 or value > MAX_SERIALIZED_UINT:
        raise AmsError(
            ErrorCode.INVALID_PACKAGE,
            f"{name} is outside [0, {MAX_SERIALIZED_UINT}]",
        )
    return value


def checked_positive(value: int, *, name: str) -> int:
    checked_uint(value, name=name)
    if value == 0:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} must be positive")
    return value


def checked_add(left: int, right: int, *, name: str = "sum") -> int:
    checked_uint(left, name=f"{name}.left")
    checked_uint(right, name=f"{name}.right")
    if left > MAX_SERIALIZED_UINT - right:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} overflows uint63")
    return left + right


def checked_mul(left: int, right: int, *, name: str = "product") -> int:
    checked_uint(left, name=f"{name}.left")
    checked_uint(right, name=f"{name}.right")
    if left != 0 and right > MAX_SERIALIZED_UINT // left:
        raise AmsError(ErrorCode.INVALID_PACKAGE, f"{name} overflows uint63")
    return left * right


def checked_product(values: Iterable[int], *, name: str = "product") -> int:
    result = 1
    for index, value in enumerate(values):
        result = checked_mul(result, value, name=f"{name}[{index}]")
    return result


def checked_range_end(offset: int, length: int, *, name: str = "range") -> int:
    checked_uint(offset, name=f"{name}.offset")
    checked_positive(length, name=f"{name}.length")
    return checked_add(offset, length, name=f"{name}.end")
