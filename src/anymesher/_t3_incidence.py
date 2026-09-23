"""Detached incidence snapshots keyed by connectivity, not export row numbers.

This first incremental layer retains full row lookup construction and shallow
map copies. Only changed cells rebuild edge membership; native persistent
storage and incremental geometric qualification are deliberately separate.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from .errors import MeshError


Cell = tuple[int, int, int]
Edge = tuple[int, int]


def _edges(cell: Cell) -> tuple[Edge, Edge, Edge]:
    a, b, c = cell
    return (
        (min(a, b), max(a, b)),
        (min(b, c), max(b, c)),
        (min(c, a), max(c, a)),
    )


class T3IncidenceIndex:
    """Private read-only snapshot; updates never change their source snapshot."""

    def __init__(self, triangles: Iterable[Iterable[int]]) -> None:
        self._row_by_cell = self._read_rows(triangles, None)
        from ._t3_correspondence import row_storage
        self._row_cells, self._row_matrix = row_storage(self._row_by_cell)
        from ._t3_row_lookup import canonical_row_lookup

        ranked = canonical_row_lookup(self._row_cells, self._row_matrix)
        if ranked is not None:
            self._row_by_cell = ranked
        from ._t3_geometry_cache import CellGeometryCache

        self._geometry = CellGeometryCache()
        memberships: dict[Edge, list[Cell]] = {}
        for cell in self._row_by_cell:
            for edge in _edges(cell):
                memberships.setdefault(edge, []).append(cell)
        from ._t3_edge_membership import EdgeMembershipMap

        self._edge_cells = EdgeMembershipMap.from_items(
            (edge, tuple(sorted(cells))) for edge, cells in memberships.items()
        )
        self._edge_rows = np.asarray(sorted(self._edge_cells), dtype=np.int64).reshape(-1, 2)
        self._edge_rows.setflags(write=False)
        from .native_cpp import native_t3_incidence

        self._native_state = native_t3_incidence(self._row_by_cell)

    @staticmethod
    def _read_rows(
        triangles: Iterable[Iterable[int]],
        cancellation_check: Callable[[str], None] | None,
    ) -> dict[Cell, int]:
        result: dict[Cell, int] = {}
        for row, triangle in enumerate(triangles):
            if cancellation_check is not None and row % 4096 == 0:
                cancellation_check("native-v2 incidence scan")
            cell = tuple(map(int, triangle))
            if len(cell) != 3 or len(set(cell)) != 3 or cell in result:
                raise MeshError("mutable T3 incidence requires distinct triangle rows")
            result[cell] = row
        return result

    def __contains__(self, edge: Edge) -> bool:
        return edge in self._edge_cells

    def attached(self, edge: Edge) -> tuple[int, ...]:
        """Return current export rows in the reference scan's exact order."""
        return tuple(sorted(
            self._row_by_cell[cell] for cell in self._edge_cells.get(edge, ())
        ))


    def canonical_edges(self, *, cancellation_check=None):
        """Return a defensive canonical edge matrix without scanning cells."""
        if cancellation_check is not None:
            cancellation_check("native-v2 topology-edge scan")
        result = self._edge_rows.copy()
        if cancellation_check is not None:
            cancellation_check("native-v2 topology-edge scan")
        return result

    def _updated_edge_rows(self, changed, cancellation_check):
        """Apply only added/deleted edge keys; array movement stays in NumPy."""
        if cancellation_check is not None:
            cancellation_check("native-v2 incidence update")
        removed = sorted(edge for edge, cells in changed.items() if not cells)
        added = sorted(edge for edge, cells in changed.items()
                       if cells and edge not in self._edge_cells)
        if not removed and not added:
            return self._edge_rows
        old = self._edge_rows
        key_dtype = np.dtype([("first", np.int64), ("second", np.int64)])
        if removed:
            deleted = np.asarray(removed, dtype=np.int64).reshape(-1, 2)
            locations = np.searchsorted(
                old.view(key_dtype).reshape(-1), deleted.view(key_dtype).reshape(-1))
            if (np.any(locations >= len(old))
                    or not np.array_equal(old[locations], deleted)):
                raise MeshError("canonical incidence edge deletion is inconsistent")
            keep = np.ones(len(old), dtype=bool)
            keep[locations] = False
            survivors = old[keep]
        else:
            survivors = old
        if added:
            inserted = np.asarray(added, dtype=np.int64).reshape(-1, 2)
            locations = np.searchsorted(
                survivors.view(key_dtype).reshape(-1),
                inserted.view(key_dtype).reshape(-1),
            ) + np.arange(len(inserted))
            result = np.empty((len(survivors) + len(inserted), 2), dtype=np.int64)
            keep = np.ones(len(result), dtype=bool)
            keep[locations] = False
            result[locations] = inserted
            result[keep] = survivors
        else:
            result = survivors
        result.setflags(write=False)
        if cancellation_check is not None:
            cancellation_check("native-v2 incidence update")
        return result

    def updated(
        self,
        triangles: Iterable[Iterable[int]],
        *,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> T3IncidenceIndex:
        from ._t3_correspondence import reuse_row_storage, row_storage
        reused = reuse_row_storage(self, triangles, cancellation_check)
        if reused is None:
            rows = self._read_rows(triangles, cancellation_check)
            row_cells, row_matrix = row_storage(rows)
        else:
            rows, row_cells, row_matrix = reused
        removed = sorted(self._row_by_cell.keys() - rows.keys())
        added = sorted(rows.keys() - self._row_by_cell.keys())
        changed: dict[Edge, set[Cell]] = {}
        for number, cell in enumerate(removed):
            if cancellation_check is not None and number % 4096 == 0:
                cancellation_check("native-v2 incidence update")
            for edge in _edges(cell):
                if edge not in changed:
                    changed[edge] = set(self._edge_cells[edge])
                changed[edge].remove(cell)
        for number, cell in enumerate(added):
            if cancellation_check is not None and number % 4096 == 0:
                cancellation_check("native-v2 incidence update")
            for edge in _edges(cell):
                if edge not in changed:
                    changed[edge] = set(self._edge_cells.get(edge, ()))
                changed[edge].add(cell)
        memberships = self._edge_cells.copy()
        for number, (edge, cells) in enumerate(changed.items()):
            if cancellation_check is not None and number % 4096 == 0:
                cancellation_check("native-v2 incidence update")
            if cells:
                memberships[edge] = tuple(sorted(cells))
            else:
                del memberships[edge]
        result = object.__new__(type(self))
        result._row_by_cell = rows
        result._row_cells, result._row_matrix = row_cells, row_matrix
        result._geometry = self._geometry.without(removed)
        result._edge_cells = memberships
        result._edge_rows = self._updated_edge_rows(changed, cancellation_check)
        from .native_cpp import native_t3_incidence

        result._native_state = native_t3_incidence(rows, self._native_state)
        result._canonical_export_snapshot = getattr(self, "_canonical_export_snapshot", None)
        if cancellation_check is not None:
            cancellation_check("native-v2 native incidence commit")
        return result

    def geometry(self, points, cell):
        """Reuse geometry only while the exact cell survives an append-only edit."""
        from ._t3_runtime import CellGeometry

        identity = tuple(map(int, cell))
        record = self._geometry.get(identity)
        if record is None:
            record = CellGeometry.prepare(points, identity)
            self._geometry[identity] = record
        return record
