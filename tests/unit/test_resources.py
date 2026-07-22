import pytest

from ams.errors import AmsError, ErrorCode
from ams.resources import ResourceAmount, ResourceBroker, ResourceVector


def vector(device_bytes: int, io_slots: int = 1) -> ResourceVector:
    return ResourceVector(
        memory=(ResourceAmount("device:0", device_bytes),),
        capacities=(ResourceAmount("io:weights", io_slots),),
    )


def test_atomic_reservation_and_idempotent_release() -> None:
    broker = ResourceBroker(vector(64, 2))
    reservation = broker.reserve(vector(64, 1))
    assert broker.used() == vector(64, 1)
    reservation.release()
    reservation.release()
    assert broker.used() == vector(0, 0)


def test_failed_multi_resource_reservation_changes_nothing() -> None:
    broker = ResourceBroker(vector(64, 1))
    with pytest.raises(AmsError) as caught:
        broker.reserve(vector(32, 2))
    assert caught.value.code is ErrorCode.PREFLIGHT_NO_WORKING_SET
    assert broker.used() == vector(0, 0)


def test_unknown_resource_is_capability_error() -> None:
    broker = ResourceBroker(vector(64))
    request = ResourceVector(memory=(ResourceAmount("host:0", 1),))
    with pytest.raises(AmsError) as caught:
        broker.reserve(request)
    assert caught.value.code is ErrorCode.CAPABILITY_MISMATCH
