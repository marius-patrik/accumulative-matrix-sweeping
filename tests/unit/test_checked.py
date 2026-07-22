import pytest

from ams.checked import MAX_SERIALIZED_UINT, checked_add, checked_mul, checked_product
from ams.errors import AmsError, ErrorCode


def test_checked_arithmetic_accepts_exact_boundary() -> None:
    assert checked_add(MAX_SERIALIZED_UINT - 1, 1) == MAX_SERIALIZED_UINT
    assert checked_mul(MAX_SERIALIZED_UINT, 1) == MAX_SERIALIZED_UINT


@pytest.mark.parametrize(
    "operation",
    [
        lambda: checked_add(MAX_SERIALIZED_UINT, 1),
        lambda: checked_mul(MAX_SERIALIZED_UINT, 2),
        lambda: checked_product((1 << 32, 1 << 32)),
    ],
)
def test_checked_arithmetic_rejects_overflow(operation) -> None:
    with pytest.raises(AmsError) as caught:
        operation()
    assert caught.value.code is ErrorCode.INVALID_PACKAGE


def test_checked_arithmetic_rejects_boolean_as_integer() -> None:
    with pytest.raises(AmsError, match="must be an integer"):
        checked_add(True, 1)
