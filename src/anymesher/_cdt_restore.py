"""Local Lawson restoration for an existing constrained chart triangulation."""

from dataclasses import dataclass
import heapq

import numpy as np

from .errors import MeshError
from .triangulation import incircle, orient2d


@dataclass(frozen=True)
class CDTRestoration:
    triangles: np.ndarray
    flips: int
    queue_visits: int
    converged: bool


def restore_constrained_delaunay(points, triangles, protected_edges, *, max_flips,
                                 cancellation_check=None):
    """Keep coordinates/rows detached and update only two cells per flip."""
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if isinstance(max_flips, bool) or not isinstance(max_flips, int) or max_flips < 0:
        raise MeshError("CDT restoration requires a nonnegative flip budget")
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise MeshError("CDT restoration needs finite chart coordinates")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise MeshError("CDT restoration needs valid triangle indices")

    def pair(a, b):
        return (a, b) if a < b else (b, a)

    def edges(row):
        return tuple(pair(row[i], row[(i + 1) % 3]) for i in range(3))

    def canonical(row):
        row = tuple(int(n) for n in row)
        if orient2d(*(points[n] for n in row)) <= 0:
            raise MeshError("CDT restoration requires positive triangle orientation")
        i = row.index(min(row))
        return row[i:] + row[:i]

    work = [canonical(row) for row in triangles]
    if len(set(work)) != len(work):
        raise MeshError("CDT restoration refuses duplicate cells")
    incidence = {}
    for row, cell in enumerate(work):
        for edge in edges(cell):
            incidence.setdefault(edge, set()).add(row)
    if any(len(rows) > 2 for rows in incidence.values()):
        raise MeshError("CDT restoration refuses non-manifold incidence")
    protected = {pair(int(a), int(b)) for a, b in np.asarray(protected_edges).reshape((-1, 2))}
    if not protected.issubset(incidence):
        raise MeshError("CDT restoration is missing a protected edge")
    queue, queued = [], set()

    def enqueue(edge):
        if edge not in protected and len(incidence.get(edge, ())) == 2 and edge not in queued:
            heapq.heappush(queue, edge)
            queued.add(edge)

    for edge in sorted(incidence):
        enqueue(edge)
    flips = visits = 0
    converged = True
    while queue:
        if cancellation_check is not None and visits % 256 == 0:
            cancellation_check("quadratic constrained Delaunay restoration")
        edge = heapq.heappop(queue)
        queued.remove(edge)
        visits += 1
        rows = sorted(incidence.get(edge, ()))
        if len(rows) != 2 or edge in protected:
            continue
        first, second = rows
        a, b = edge
        c = next(n for n in work[first] if n not in edge)
        d = next(n for n in work[second] if n not in edge)
        # A concave two-cell patch cannot use the alternative diagonal.
        ca = orient2d(points[c], points[d], points[a])
        cb = orient2d(points[c], points[d], points[b])
        if ca == 0 or cb == 0 or (ca > 0) == (cb > 0):
            continue
        if pair(c, d) in incidence:
            continue
        if incircle(*(points[n] for n in work[first]), points[d]) <= 0:
            continue
        if flips >= max_flips:
            converged = False
            break
        proposed = ((c, d, a), (d, c, b)) if ca > 0 else ((d, c, a), (c, d, b))
        replacements = [canonical(row) for row in proposed]
        for row in rows:
            for old_edge in edges(work[row]):
                incidence[old_edge].remove(row)
                if not incidence[old_edge]:
                    del incidence[old_edge]
        for row, cell in zip(rows, replacements):
            work[row] = cell
            for new_edge in edges(cell):
                incidence.setdefault(new_edge, set()).add(row)
        flips += 1
        for cell in replacements:
            for new_edge in edges(cell):
                enqueue(new_edge)
    if cancellation_check is not None:
        cancellation_check("quadratic constrained Delaunay restoration complete")
    return CDTRestoration(np.asarray(sorted(work), dtype=np.int64).reshape((-1, 3)), flips, visits, converged)
