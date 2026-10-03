"""Component-owned node-ID reservations with a detached staging fork."""

from dataclasses import dataclass
from numbers import Integral
from threading import RLock

from .errors import MeshError


@dataclass(frozen=True)
class ReservationSnapshot:
    node_ids: tuple[int, ...]
    reserved_ids: tuple[int, ...]


class ComponentNodeReservationPool:
    """Reserve IDs against one mutable mesh without reusing failed reserves."""

    def __init__(self, mesh, *, _reserved=(), _origin=None):
        self._mesh = mesh
        self._reserved = set()
        for node_id in _reserved:
            if isinstance(node_id, bool) or not isinstance(node_id, Integral) or node_id < 1:
                raise MeshError("component reservation IDs must be positive integers")
            self._reserved.add(int(node_id))
        self._origin = _origin
        self._lock = RLock()

    @property
    def mesh(self):
        return self._mesh

    @property
    def origin(self):
        return self._origin

    def release_origin(self):
        """Drop the old component after its staged fork is published."""
        with self._lock:
            self._origin = None

    def snapshot(self) -> ReservationSnapshot:
        with self._lock:
            return ReservationSnapshot(
                tuple(sorted(int(node) for node in self._mesh.nodes)),
                tuple(sorted(self._reserved)),
            )

    def allocate(self) -> int:
        with self._lock:
            node_id = max((*self._mesh.nodes, *self._reserved), default=0) + 1
            self._reserved.add(node_id)
            return node_id

    def fork_detached(self, mesh, *, expected: ReservationSnapshot):
        """Clone all outstanding reservations against a different mesh."""
        with self._lock:
            if mesh is self._mesh or self.snapshot() != expected:
                raise MeshError("component reservation source changed before staging")
            if tuple(sorted(int(node) for node in mesh.nodes)) != expected.node_ids:
                raise MeshError("staged mesh does not preserve published node IDs")
            return ComponentNodeReservationPool(
                mesh, _reserved=expected.reserved_ids, _origin=self,
            )
