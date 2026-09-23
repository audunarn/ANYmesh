"""Find affected frontal queue cells without revisiting all cell geometry."""
from __future__ import annotations

import numpy as np

from .errors import MeshError


class QueueRefreshIndex:
    """Invocation-local append-only point binding and node/cell incidence."""

    def __init__(self):
        self._index = None
        self._points = None
        self._tensors = None
        self._protected = frozenset()
        self._incident = {}

    def prepare(self, index, points, tensors, protected, checkpoint):
        checkpoint(0)
        points = np.ascontiguousarray(points, dtype=np.float64)
        tensors = np.ascontiguousarray(tensors, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2
                or tensors.shape != (len(points), 2, 2)
                or not np.all(np.isfinite(points))
                or not np.all(np.isfinite(tensors))):
            raise MeshError("incremental queue requires finite point-bound 2D metrics")
        if self._points is not None and (
            len(points) < len(self._points)
            or not np.array_equal(points[:len(self._points)].view(np.uint64),
                                  self._points.view(np.uint64))
        ):
            raise MeshError("incremental frontal queue requires append-only coordinates")
        current = index._row_by_cell
        previous = {} if self._index is None else self._index._row_by_cell
        removed = tuple(sorted(previous.keys() - current.keys()))
        added = tuple(sorted(current.keys() - previous.keys()))
        members = self._incident.copy()
        changes = {}
        work = 0
        for adding, cells in ((False, removed), (True, added)):
            for cell in cells:
                checkpoint(work)
                work += 1
                if any(node < 0 or node >= len(points) for node in cell):
                    raise MeshError("queue connectivity is outside its point binding")
                for node in cell:
                    if node not in changes:
                        changes[node] = set(members.get(node, ()))
                    if adding:
                        changes[node].add(cell)
                    else:
                        changes[node].remove(cell)
        for node, cells in changes.items():
            if cells:
                members[node] = cells
            else:
                members.pop(node, None)
        touched = set(added)
        protected = frozenset(protected)
        if self._tensors is not None:
            count = len(self._tensors)
            changed = np.flatnonzero(np.any(
                tensors[:count].view(np.uint64) != self._tensors.view(np.uint64),
                axis=(1, 2),
            ))
            for number, node in enumerate(changed):
                checkpoint(number)
                touched.update(members.get(int(node), ()))
            # New nodes only belong to added cells, already in touched.
            for number, edge in enumerate(sorted(protected ^ self._protected)):
                checkpoint(number)
                for node in edge:
                    touched.update(members.get(int(node), ()))
        checkpoint(0)
        next_points = points.copy()
        next_points.setflags(write=False)
        next_tensors = tensors.copy()
        next_tensors.setflags(write=False)
        rows = np.asarray(sorted(touched), dtype=np.int64).reshape((-1, 3))
        # Publish private preparation only after all fallible staging completes.
        self._incident = members
        self._points = next_points
        self._tensors = next_tensors
        self._protected = protected
        self._index = index
        return rows, removed
