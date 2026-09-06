"""Detached incidence snapshots keyed by connectivity, not export row numbers.

This first incremental layer retains full row lookup construction and shallow
map copies. Only changed cells rebuild edge membership; native persistent
storage and incremental geometric qualification are deliberately separate.
"""

from __future__ import annotations

from typing import Callable, Iterable

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
        self._geometry = {}
        memberships: dict[Edge, list[Cell]] = {}
        for cell in self._row_by_cell:
            for edge in _edges(cell):
                memberships.setdefault(edge, []).append(cell)
        self._edge_cells = {
            edge: tuple(sorted(cells)) for edge, cells in memberships.items()
        }
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

    def updated(
        self,
        triangles: Iterable[Iterable[int]],
        *,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> T3IncidenceIndex:
        rows = self._read_rows(triangles, cancellation_check)
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
        result._geometry = {
            cell: record for cell, record in self._geometry.items() if cell in rows
        }
        result._edge_cells = memberships
        from .native_cpp import native_t3_incidence

        result._native_state = native_t3_incidence(rows, self._native_state)
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
