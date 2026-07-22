"""Atomic resource-vector admission for bounded reference execution."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

from ams.checked import checked_add, checked_uint
from ams.descriptors import validate_identifier
from ams.errors import AmsError, ErrorCode


@dataclass(frozen=True, slots=True, order=True)
class ResourceAmount:
    resource_id: str
    amount: int

    def __post_init__(self) -> None:
        validate_identifier(self.resource_id, name="resource.resource_id")
        checked_uint(self.amount, name=f"resource[{self.resource_id}].amount")


@dataclass(frozen=True, slots=True)
class ResourceVector:
    memory: tuple[ResourceAmount, ...] = ()
    capacities: tuple[ResourceAmount, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("memory", "capacities"):
            values = tuple(sorted(getattr(self, field_name)))
            if len({item.resource_id for item in values}) != len(values):
                raise AmsError(ErrorCode.PLAN_INVALID, f"duplicate {field_name} resource ID")
            object.__setattr__(self, field_name, values)

    def as_dict(self) -> dict[tuple[str, str], int]:
        result = {("memory", item.resource_id): item.amount for item in self.memory}
        result.update((("capacity", item.resource_id), item.amount) for item in self.capacities)
        return result

    @classmethod
    def from_dict(cls, values: dict[tuple[str, str], int]) -> ResourceVector:
        memory = tuple(
            ResourceAmount(resource_id, amount)
            for (kind, resource_id), amount in values.items()
            if kind == "memory"
        )
        capacities = tuple(
            ResourceAmount(resource_id, amount)
            for (kind, resource_id), amount in values.items()
            if kind == "capacity"
        )
        return cls(memory=memory, capacities=capacities)


class Reservation:
    """An idempotently releasable broker reservation."""

    def __init__(self, broker: ResourceBroker, reservation_id: str, resources: ResourceVector):
        self._broker = broker
        self.reservation_id = reservation_id
        self.resources = resources
        self._released = False
        self._release_lock = threading.Lock()

    @property
    def released(self) -> bool:
        with self._release_lock:
            return self._released

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._broker._release(self.reservation_id, self.resources)
            self._released = True

    def __enter__(self) -> Reservation:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class ResourceBroker:
    """Reserve complete vectors under one lock; partial acquisition is impossible."""

    def __init__(self, capacity: ResourceVector):
        self._capacity = capacity.as_dict()
        self._used = {key: 0 for key in self._capacity}
        self._active: set[str] = set()
        self._lock = threading.Lock()

    def reserve(self, request: ResourceVector) -> Reservation:
        requested = request.as_dict()
        with self._lock:
            for key, amount in requested.items():
                if key not in self._capacity:
                    raise AmsError(
                        ErrorCode.CAPABILITY_MISMATCH,
                        f"resource is not configured: {key[0]}:{key[1]}",
                    )
                available = self._capacity[key] - self._used[key]
                if amount > available:
                    raise AmsError(
                        ErrorCode.PREFLIGHT_NO_WORKING_SET,
                        f"insufficient {key[0]} resource: {key[1]}",
                        evidence={"available": available, "requested": amount},
                    )
            reservation_id = f"reservation:{uuid.uuid4()}"
            for key, amount in requested.items():
                self._used[key] = checked_add(
                    self._used[key],
                    amount,
                    name=f"broker.used[{key[0]}:{key[1]}]",
                )
            self._active.add(reservation_id)
        return Reservation(self, reservation_id, request)

    def _release(self, reservation_id: str, resources: ResourceVector) -> None:
        with self._lock:
            if reservation_id not in self._active:
                raise AmsError(
                    ErrorCode.RESERVATION_LOST,
                    "reservation is unknown or was already released",
                )
            for key, amount in resources.as_dict().items():
                if amount > self._used[key]:
                    raise AmsError(ErrorCode.BROKER_VIOLATION, "resource usage became negative")
            for key, amount in resources.as_dict().items():
                self._used[key] -= amount
            self._active.remove(reservation_id)

    def used(self) -> ResourceVector:
        with self._lock:
            return ResourceVector.from_dict(dict(self._used))
