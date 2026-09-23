"""Bounded, transactional two-flip repair of constrained triangle transitions."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .triangulation import orient2d


@dataclass(frozen=True)
class TransitionPatch:
    points: np.ndarray
    triangles: np.ndarray
    attempts: int
    committed_flips: int
    improved: bool


def repair_transition_patch(points, triangles, protected_edges, poor_triangle_ids,
                            *, score: Callable, relax: Callable | None = None,
                            max_attempts=64, cancellation_check=None,
                            target_met: Callable | None = None,
                            work_available: Callable | None = None):
    """Try at most two local flips per candidate; never publish an intermediate.

    ``score`` is the caller's whole-candidate policy score (smaller is better).
    ``relax`` optionally repairs a detached candidate and returns its coordinates.
    Protected coordinates and all boundary segments remain exact. Row numbers in
    ``poor_triangle_ids`` are zero-based in the supplied connectivity.
    """
    x = np.array(points, dtype=np.float64, copy=True)
    initial = np.array(triangles, dtype=np.int64, copy=True)
    if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
            or initial.ndim != 2 or initial.shape[1] != 3
            or np.any(initial < 0) or np.any(initial >= len(x))
            or isinstance(max_attempts, bool) or not isinstance(max_attempts, int)
            or max_attempts < 0):
        raise ValueError("invalid transition-patch input")

    def edge(a, b):
        return tuple(sorted((int(a), int(b))))

    def incidence(cells):
        result = {}
        for i, row in enumerate(cells):
            for j in range(3):
                result.setdefault(edge(row[j], row[(j + 1) % 3]), []).append(i)
        if any(len(rows) > 2 for rows in result.values()):
            raise ValueError("non-manifold transition-patch input")
        return result

    def canonical(cells):
        rows = []
        for row in cells:
            if orient2d(x[row[0]], x[row[1]], x[row[2]]) <= 0:
                raise ValueError("non-positive transition-patch triangle")
            k = int(np.argmin(row))
            rows.append(tuple(int(v) for v in np.roll(row, -k)))
        if len(set(rows)) != len(rows):
            raise ValueError("duplicate transition-patch triangle")
        return np.asarray(sorted(rows), dtype=np.int64).reshape((-1, 3))

    adjacency = incidence(initial)
    protected = {edge(*pair) for pair in protected_edges}
    if not protected.issubset(adjacency):
        raise ValueError("missing protected transition edge")
    protected.update(pair for pair, rows in adjacency.items() if len(rows) == 1)
    fixed = sorted({v for pair in protected for v in pair})
    affected = set()
    direct_edges = set()
    for i in poor_triangle_ids:
        if isinstance(i, bool) or int(i) != i or not 0 <= int(i) < len(initial):
            raise ValueError("invalid poor triangle index")
        affected.update(int(v) for v in initial[int(i)])
        row = initial[int(i)]
        direct_edges.update(edge(row[j], row[(j + 1) % 3]) for j in range(3))
    cells = canonical(initial)
    best_x, best_cells = x.copy(), initial.copy()
    best_score = score(best_x.copy(), best_cells.copy())
    attempts = committed = 0
    seen = {cells.tobytes()}

    def checkpoint():
        if cancellation_check is not None:
            cancellation_check()

    def flips(current, vertices):
        links = incidence(current)
        for pair, rows in sorted(links.items(), key=lambda item: (item[0] not in direct_edges, item[0])):
            if (pair in protected or len(rows) != 2
                    or not vertices.intersection(pair)):
                continue
            a, b = pair
            c = next(int(v) for v in current[rows[0]] if v not in pair)
            d = next(int(v) for v in current[rows[1]] if v not in pair)
            if edge(c, d) in links:
                continue
            ca = orient2d(x[c], x[d], x[a])
            cb = orient2d(x[c], x[d], x[b])
            if ca == 0 or cb == 0 or (ca > 0) == (cb > 0):
                continue
            replacement = ((c, d, a), (d, c, b)) if ca > 0 else ((d, c, a), (c, d, b))
            candidate = current.copy()
            candidate[rows] = replacement
            yield canonical(candidate), {a, b, c, d}

    def consider(candidate, depth):
        nonlocal attempts, best_x, best_cells, best_score, committed
        key = candidate.tobytes()
        if (key in seen or attempts >= max_attempts
                or (work_available is not None and not work_available())):
            return
        seen.add(key)
        checkpoint()
        attempts += 1
        candidate_x = x.copy()
        if relax is not None:
            candidate_x = np.asarray(relax(candidate_x, candidate.copy()), dtype=np.float64)
        if (candidate_x.shape != x.shape or not np.isfinite(candidate_x).all()
                or candidate_x[fixed].tobytes() != x[fixed].tobytes()):
            raise ValueError("transition relaxation changed protected coordinates")
        if any(orient2d(candidate_x[a], candidate_x[b], candidate_x[c]) <= 0
               for a, b, c in candidate):
            return
        value = score(candidate_x.copy(), candidate.copy())
        if value < best_score:
            best_score = value
            best_x, best_cells = candidate_x.copy(), candidate.copy()
            committed = depth

    checkpoint()
    if target_met is not None and target_met(best_x.copy(), best_cells.copy()):
        return TransitionPatch(best_x, best_cells, 0, 0, False)
    for first, vertices in flips(cells, affected):
        if attempts >= max_attempts or (work_available is not None and not work_available()):
            break
        consider(first, 1)
        if committed and target_met is not None and target_met(best_x.copy(), best_cells.copy()):
            break
        for second, _ in flips(first, vertices):
            if attempts >= max_attempts or (work_available is not None and not work_available()):
                break
            consider(second, 2)
            if committed and target_met is not None and target_met(best_x.copy(), best_cells.copy()):
                break
        if committed and target_met is not None and target_met(best_x.copy(), best_cells.copy()):
            break
    checkpoint()
    return TransitionPatch(best_x, best_cells, attempts, committed, committed != 0)
