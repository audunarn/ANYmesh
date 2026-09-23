"""Incremental, defensive canonical exports for qualified frontal topology."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .errors import MeshError


_KEY = np.dtype([("a", np.int64), ("b", np.int64), ("c", np.int64)])


def _keys(rows):
    return rows.view(_KEY).reshape(-1)


def _rows(values):
    return np.asarray(sorted(values), dtype=np.int64).reshape(-1, 3)


def _replace_rows(old, removed, added):
    """Move canonical rows in NumPy; never renormalize surviving triangles."""
    if not removed and not added:
        return old
    if removed:
        deleted = _rows(removed)
        positions = np.searchsorted(_keys(old), _keys(deleted))
        if (np.any(positions >= len(old))
                or not np.array_equal(old[positions], deleted)):
            raise MeshError("canonical export deletion is inconsistent")
        keep = np.ones(len(old), dtype=bool)
        keep[positions] = False
        survivors = old[keep]
    else:
        survivors = old
    if added:
        inserted = _rows(added)
        if len(inserted) > 1 and np.any(np.all(inserted[1:] == inserted[:-1], axis=1)):
            raise MeshError("canonical export contains duplicate triangles")
        positions = np.searchsorted(_keys(survivors), _keys(inserted))
        present = positions < len(survivors)
        if np.any(np.all(survivors[positions[present]] == inserted[present], axis=1)):
            raise MeshError("canonical export contains duplicate triangles")
        positions = positions + np.arange(len(inserted))
        result = np.empty((len(survivors) + len(inserted), 3), dtype=np.int64)
        keep = np.ones(len(result), dtype=bool)
        keep[positions] = False
        result[positions] = inserted
        result[keep] = survivors
    else:
        result = survivors
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CanonicalTriangleExport:
    points: np.ndarray
    by_cell: object
    rows: np.ndarray
    source_keys: object = None

    @classmethod
    def prepare(cls, previous, points, cells, canonicalize, check):
        from ._t3_edge_membership import EdgeMembershipMap
        from ._t3_row_lookup import freeze_row_keys

        check("native-v2 canonical export prepare")
        source_keys = freeze_row_keys(cells)
        points = np.ascontiguousarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise MeshError("canonical export requires finite 2D points")
        reusable = previous is not None and len(points) >= len(previous.points)
        if reusable:
            reusable = np.array_equal(
                points[:len(previous.points)].view(np.uint64),
                previous.points.view(np.uint64),
            )
        if reusable:
            before = previous.by_cell
            before_keys = (previous.source_keys
                           if source_keys is not None and previous.source_keys is not None
                           else before.keys())
            after_keys = source_keys if source_keys is not None else cells.keys()
            removed = sorted(before_keys - after_keys)
            added = sorted(after_keys - before_keys)
            if not removed and not added and len(points) == len(previous.points):
                check("native-v2 canonical export commit")
                return previous
            mapping = before.copy() if removed or added else before
            removed_rows = [before[cell] for cell in removed]
            for number, cell in enumerate(removed):
                if number % 4096 == 0:
                    check("native-v2 canonical export cells")
                del mapping[cell]
            old_rows = previous.rows
        else:
            added = sorted(cells)
            mapping = {}
            removed_rows = []
            old_rows = np.empty((0, 3), dtype=np.int64)
        new_rows = []
        for number, cell in enumerate(added):
            if number % 4096 == 0:
                check("native-v2 canonical export cells")
            if len(cell) != 3 or min(cell) < 0 or max(cell) >= len(points):
                raise MeshError("canonical export connectivity is out of range")
            canonical = canonicalize(cell, points)
            mapping[cell] = canonical
            new_rows.append(canonical)
        rows = _replace_rows(old_rows, removed_rows, new_rows)
        # Both arrays and the connectivity map belong to the immutable snapshot.
        saved_points = points.copy()
        saved_points.setflags(write=False)
        rows.setflags(write=False)
        if isinstance(mapping, MappingProxyType):
            immutable_mapping = mapping
        else:
            if not isinstance(mapping, EdgeMembershipMap):
                mapping = EdgeMembershipMap.from_items(mapping.items())
            immutable_mapping = MappingProxyType(mapping)
        result = cls(saved_points, immutable_mapping, rows, source_keys)
        check("native-v2 canonical export commit")
        return result


def canonical_frontal_export(topology, *, cancellation_check=None):
    """Keep the general full export as the independent compatibility oracle."""
    from .native_v2 import _canonical_triangle

    def check(phase):
        if cancellation_check is not None:
            cancellation_check(phase)

    points = topology.points
    index = topology._topology_index
    previous = getattr(index, "_canonical_export_snapshot", None)
    snapshot = CanonicalTriangleExport.prepare(
        previous, points, index._row_by_cell, _canonical_triangle, check,
    )
    triangles = snapshot.rows.copy()
    check("native-v2 canonical export commit")
    index._canonical_export_snapshot = snapshot
    return points, triangles
