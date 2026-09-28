"""Bounded minimum-angle improvement of residual T3 after the quad front.

Residual triangles are closures of the seed triangulation left where the
front could not pair them.  A diagonal flip between two adjacent residual T3
keeps coverage, protected segments and the Q4/T3 front unchanged (the four
outer edges keep the same residual incidence), so it is a purely local,
transactional improvement.  A flip is committed only when it strictly raises
the pair's minimum corner angle (Lawson's min-angle criterion), which bounds
the number of flips; passes are additionally capped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .front import body_edges, edge_key
from .quality_gate import RELATIVE_EPS, corner_metrics
from .state import QuadMeshState

__all__ = ["ResidualFlipReport", "improve_residual_triangles"]

_MAX_PASSES = 8
_MIN_GAIN_DEGREES = 1.0e-9


@dataclass(frozen=True)
class ResidualFlipReport:
    passes: int
    flips: int
    min_t3_angle_before: float | None
    min_t3_angle_after: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passes": self.passes,
            "flips": self.flips,
            "min_t3_angle_before": self.min_t3_angle_before,
            "min_t3_angle_after": self.min_t3_angle_after,
        }


def _points(state: QuadMeshState, body: tuple[int, ...]) -> list[tuple[float, float]]:
    return [tuple(map(float, state.position(node))) for node in body]


def _min_angle(state: QuadMeshState, body: tuple[int, ...]) -> float:
    return corner_metrics(_points(state, body))[0]


def _thinnest(state: QuadMeshState) -> float | None:
    angles = [
        _min_angle(state, tuple(int(x) for x in state.cell(cid)))
        for cid in sorted(int(c) for c in state.cells)
        if state.cell_kind(cid) == "T3"
    ]
    return min(angles) if angles else None


def _apex(body: tuple[int, ...], edge: tuple[int, int]) -> int:
    return next(node for node in body if node not in edge)


def _ccw_from(body: tuple[int, ...], start: int) -> tuple[int, int, int]:
    index = body.index(start)
    return body[index], body[(index + 1) % 3], body[(index + 2) % 3]


def _flip(state: QuadMeshState, edge: tuple[int, int]) -> bool:
    cells = tuple(sorted(int(c) for c in state.edge_cells(edge)))
    if len(cells) != 2 or any(state.cell_kind(c) != "T3" for c in cells):
        return False
    first = tuple(int(x) for x in state.cell(cells[0]))
    second = tuple(int(x) for x in state.cell(cells[1]))
    c = _apex(first, edge)
    d = _apex(second, edge)
    # CCW first = (c, a, b) with shared edge a->b; second then runs b->a.
    _, a, b = _ccw_from(first, c)
    # The flipped pair (c, a, d) and (d, b, c) is valid only if the quad
    # c-a-d-b is strictly convex at a and b.
    candidates = ((c, a, d), (d, b, c))
    for body in candidates:
        pts = _points(state, body)
        lengths = [
            ((pts[(i + 1) % 3][0] - pts[i][0]) ** 2 + (pts[(i + 1) % 3][1] - pts[i][1]) ** 2) ** 0.5
            for i in range(3)
        ]
        doubled = ((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
                   - (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0]))
        if doubled <= RELATIVE_EPS * max(lengths) ** 2:
            return False
    before = min(_min_angle(state, first), _min_angle(state, second))
    after = min(_min_angle(state, body) for body in candidates)
    if after <= before + _MIN_GAIN_DEGREES:
        return False
    with state.transaction() as tx:
        tx.remove_cell(cells[0])
        tx.remove_cell(cells[1])
        for body in candidates:
            tx.allocate_cell(body, "T3")
        tx.commit()
    return True


def improve_residual_triangles(
    state: QuadMeshState,
    *,
    cancellation_check: Callable[[str], None] | None = None,
    max_passes: int = _MAX_PASSES,
) -> ResidualFlipReport:
    """Flip unprotected T3/T3 diagonals that raise the local minimum angle.

    Deterministic (sorted edge order per pass), bounded (``max_passes``) and
    cancellable between passes.  Protected edges and every edge touching a Q4
    are never changed.
    """
    before = _thinnest(state)
    flips = passes = 0
    for passes in range(1, max_passes + 1):
        if cancellation_check is not None:
            cancellation_check("quad-first:residual-flips")
        edges = sorted({
            edge_key(*edge)
            for cid in state.cells
            if state.cell_kind(cid) == "T3"
            for edge in body_edges(tuple(int(x) for x in state.cell(cid)))
        })
        changed = 0
        for edge in edges:
            if state.is_protected_edge(edge) or state.is_front_edge(edge):
                continue
            try:
                incident = state.edge_cells(edge)
            except Exception:  # removed by an earlier flip in this pass
                continue
            if len(incident) != 2:
                continue
            if _flip(state, edge):
                changed += 1
        flips += changed
        if not changed:
            break
    return ResidualFlipReport(passes, flips, before, _thinnest(state))
