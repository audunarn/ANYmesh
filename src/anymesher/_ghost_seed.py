"""Complete Delaunay seed with a symbolic vertex at infinity (reference).

The finite super-triangle insertion in ``triangulation._bowyer_watson`` loses
hull cells when hull stations are almost collinear: a finite super vertex lies
inside the huge circumcircle of such a flat hull triangle, so the triangle is
never Delaunay in the augmented set and its removal leaves a pocket.  Here the
super vertex is replaced by one symbolic vertex at infinity (``GHOST``), so the
result covers the convex hull by construction.

Conflict rules (exact predicates only):

* a finite counter-clockwise cell ``(a, b, c)`` conflicts with ``p`` when
  ``incircle(a, b, c, p) > 0``;
* a ghost cell ``(a, b, GHOST)`` stands for the outside of the directed hull
  edge ``a -> b`` (outside on its left); it conflicts when
  ``orient2d(a, b, p) > 0``, or when ``p`` lies on the open segment ``ab``.

Points are inserted in lexicographic ``(x, y, row)`` order, so each new point
is outside the closed hull of the inserted ones and a conflicting ghost cell
always exists.  The cavity is the conflict set grown from that cell across
shared edges; every cavity boundary edge is joined to the new point.  The cell
set after each insertion depends only on the insertion order and the predicate
signs, never on storage order, so the native mirror reproduces it exactly.

Result contract: the seed is published only when every interior edge is
STRICTLY locally Delaunay (``incircle < 0`` across it).  Then every cell's
circumcircle has all other stations strictly outside, the Delaunay
triangulation of the stations is unique, and the result equals the one any
other exact Delaunay construction returns -- in particular the finite-hull fan
plus Lawson flips of ``triangulation._finite_hull_seed`` -- cell for cell.
Exactly cocircular ties (``incircle == 0`` across an interior edge) admit
several Delaunay triangulations; the seed then refuses (``reason ==
"cocircular"``) so the caller keeps its finite-hull route and its tie choice.

Anything else outside these rules (duplicate stations, an all-collinear input,
a zero-area new cell, a non-Delaunay result) also raises ``GhostSeedRefusal``.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

GHOST = -1


class GhostSeedRefusal(Exception):
    """The symbolic-infinity seed could not certify a result."""

    def __init__(self, message: str, reason: str = "refused") -> None:
        super().__init__(message)
        self.reason = reason


def ghost_bowyer_watson(
    points,
    orient2d: Callable[..., float],
    incircle: Callable[..., float],
    cancellation_check: Callable[[str], None] | None = None,
) -> list[tuple[int, int, int]]:
    """Return the sorted canonical cells of a complete Delaunay triangulation."""
    count = len(points)
    if count < 3:
        raise GhostSeedRefusal("ghost seed needs at least three points")
    coords = [(float(row[0]), float(row[1])) for row in points.tolist()]
    order = sorted(range(count), key=lambda row: (coords[row][0], coords[row][1], row))
    for first, second in zip(order, order[1:]):
        if coords[first] == coords[second]:
            raise GhostSeedRefusal("ghost seed refuses duplicate stations")

    # Directed edge (u, v) -> cell key that contains it.  Cells are tuples:
    # finite CCW (a, b, c) or ghost (a, b, GHOST).
    owner: dict[tuple[int, int], tuple[int, int, int]] = {}
    cells: set[tuple[int, int, int]] = set()
    ghosts: set[tuple[int, int, int]] = set()

    def add(cell: tuple[int, int, int]) -> None:
        a, b, c = cell
        for edge in ((a, b), (b, c), (c, a)):
            if edge in owner:
                raise GhostSeedRefusal("ghost seed created a nonmanifold edge")
            owner[edge] = cell
        cells.add(cell)
        if c == GHOST:
            ghosts.add(cell)

    def remove(cell: tuple[int, int, int]) -> None:
        a, b, c = cell
        for edge in ((a, b), (b, c), (c, a)):
            del owner[edge]
        cells.discard(cell)
        ghosts.discard(cell)

    def between(a: int, b: int, p: int) -> bool:
        # p is collinear with a and b here; open-segment test by lexicographic order.
        low, high = sorted((coords[a], coords[b]))
        return low < coords[p] < high

    def conflict(cell: tuple[int, int, int], p: int) -> bool:
        a, b, c = cell
        if c == GHOST:
            turn = orient2d(coords[a], coords[b], coords[p])
            return turn > 0.0 or (turn == 0.0 and between(a, b, p))
        return incircle(coords[a], coords[b], coords[c], coords[p]) > 0.0

    def finite(a: int, b: int, c: int) -> tuple[int, int, int]:
        turn = orient2d(coords[a], coords[b], coords[c])
        if turn <= 0.0:
            raise GhostSeedRefusal("ghost seed created a non-positive cell")
        return (a, b, c)

    # Initial fan over the leading collinear run.
    apex = 2
    while apex < count and orient2d(coords[order[0]], coords[order[1]], coords[order[apex]]) == 0.0:
        apex += 1
    if apex == count:
        raise GhostSeedRefusal("ghost seed input is collinear")
    top = order[apex]
    side = orient2d(coords[order[0]], coords[order[1]], coords[top])
    for index in range(apex - 1):
        u, v = order[index], order[index + 1]
        add(finite(u, v, top) if side > 0.0 else finite(v, u, top))
    for (u, v) in list(owner):
        if (v, u) not in owner:
            add((v, u, GHOST))

    for position in range(apex + 1, count):
        if cancellation_check is not None and position % 64 == 0:
            cancellation_check("python triangulation ghost seed insertion")
        p = order[position]
        start = next((cell for cell in sorted(ghosts) if conflict(cell, p)), None)
        if start is None:
            raise GhostSeedRefusal("ghost seed could not locate a point")
        cavity = {start}
        pending = [start]
        while pending:
            a, b, c = pending.pop()
            for u, v in ((a, b), (b, c), (c, a)):
                neighbour = owner.get((v, u))
                if neighbour is None:
                    raise GhostSeedRefusal("ghost seed lost an edge twin")
                if neighbour not in cavity and conflict(neighbour, p):
                    cavity.add(neighbour)
                    pending.append(neighbour)
        boundary = []
        for a, b, c in cavity:
            for u, v in ((a, b), (b, c), (c, a)):
                if owner[(v, u)] not in cavity:
                    boundary.append((u, v))
        for cell in cavity:
            remove(cell)
        for u, v in sorted(boundary):
            if v == GHOST:
                add((p, u, GHOST))
            elif u == GHOST:
                add((v, p, GHOST))
            else:
                add(finite(u, v, p))

    result = sorted(_canonical(cell) for cell in cells if cell[2] != GHOST)
    # Fail closed: every interior edge must be strictly locally Delaunay.
    tie = False
    if cancellation_check is not None:
        cancellation_check("python triangulation ghost seed certificate")
    for (u, v), cell in owner.items():
        if u > v or GHOST in (u, v) or cell[2] == GHOST:
            continue
        other = owner[(v, u)]
        if other[2] == GHOST:
            continue
        opposite = next(node for node in other if node not in (u, v))
        value = incircle(*(coords[node] for node in cell), coords[opposite])
        if value > 0.0:
            raise GhostSeedRefusal("ghost seed result is not Delaunay")
        tie = tie or value == 0.0
    if tie:
        raise GhostSeedRefusal("ghost seed has cocircular ties", reason="cocircular")
    return result


def _canonical(cell: Sequence[int]) -> tuple[int, int, int]:
    a, b, c = cell
    if b < a and b < c:
        return (b, c, a)
    if c < a and c < b:
        return (c, a, b)
    return (a, b, c)
