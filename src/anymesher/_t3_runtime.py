"""Private geometric records and deterministic, incrementally refreshed work.

Location deliberately retains ordered exact-predicate scanning. Geometric AABB
rejection is not used: the existing tolerance can admit points outside a cell.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable

import numpy as np


Cell = tuple[int, int, int]


@dataclass(slots=True)
class CellGeometry:
    coordinates: np.ndarray
    location_tolerance: float
    angles: tuple[float, float, float] | None = None

    @classmethod
    def prepare(cls, points: np.ndarray, cell: Cell) -> CellGeometry:
        coordinates = np.ascontiguousarray(points[np.asarray(cell, dtype=np.int64)])
        coordinates.setflags(write=False)
        scale = max(float(np.ptp(coordinates, axis=0).max()), 1.0)
        return cls(coordinates, 64.0 * np.finfo(float).eps * scale * scale)


@dataclass(slots=True)
class _Qualification:
    geometry: CellGeometry
    stamp: bytes
    protected_corner: bool
    severity: float
    geometry_limited: bool


class TriangleWorkQueue:
    """Refresh qualifications; reuse live entries without rebuilding the heap.

    Consumed surviving cells are reactivated each round, as in the reference
    rebuild. Implementation-stale entries never consume a meshing operation.
    """

    def __init__(self, cancellation_check=None, cancellation_interval=4096):
        self._records: dict[Cell, _Qualification] = {}
        self._active: dict[Cell, tuple[float, int, Cell, int]] = {}
        self._heap: list[tuple[float, int, Cell, int]] = []
        self._seen: set[Cell] = set()
        self._token = 0
        self._cancel = cancellation_check
        self._interval = cancellation_interval
        self.evaluations = 0
        self.cache_hits = 0

    def _checkpoint(self, number: int) -> None:
        if self._cancel is not None and number % self._interval == 0:
            self._cancel("native-v2 queue maintenance")

    def begin_round(self) -> None:
        self._seen = set()

    def refresh(
        self,
        identity: Cell,
        geometry: CellGeometry,
        local_tensors: np.ndarray,
        protected_corner: bool,
        evaluate: Callable[[], tuple[float, bool]],
    ) -> tuple[float, bool]:
        stamp = local_tensors.tobytes(order="C")
        previous = self._records.get(identity)
        if (previous is None or previous.geometry is not geometry
                or previous.stamp != stamp
                or previous.protected_corner != protected_corner):
            severity, limited = evaluate()
            record = _Qualification(geometry, stamp, protected_corner, severity, limited)
            self._records[identity] = record
            self.evaluations += 1
        else:
            record = previous
            self.cache_hits += 1
        self._seen.add(identity)
        if record.geometry_limited or not (record.severity > 1.0 + 1.0e-12):
            self._active.pop(identity, None)
        else:
            current = self._active.get(identity)
            if current is None or current[0] != -record.severity:
                self._token += 1
                entry = (-record.severity, 1, identity, self._token)
                heapq.heappush(self._heap, entry)
                self._active[identity] = entry
        return record.severity, record.geometry_limited

    def finish_round(self) -> None:
        removed = []
        for number, identity in enumerate(self._records):
            self._checkpoint(number)
            if identity not in self._seen:
                removed.append(identity)
        for number, identity in enumerate(removed):
            self._checkpoint(number)
            del self._records[identity]
            self._active.pop(identity, None)
        if len(self._heap) > max(64, 2 * len(self._active)):
            retained = []
            for number, entry in enumerate(self._active.values()):
                self._checkpoint(number)
                retained.append(entry)
            self._checkpoint(0)
            heapq.heapify(retained)
            self._heap = retained

    def _prune(self) -> None:
        number = 0
        while self._heap:
            entry = self._heap[0]
            if self._active.get(entry[2]) == entry:
                return
            self._checkpoint(number)
            heapq.heappop(self._heap)
            number += 1

    def __bool__(self) -> bool:
        self._prune()
        return bool(self._heap)

    def pop(self) -> tuple[float, int, Cell, int]:
        self._prune()
        entry = heapq.heappop(self._heap)
        del self._active[entry[2]]
        return entry
