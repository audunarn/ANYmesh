"""Bounded, detached insertion beside an over-constrained boundary fan."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryFanRepair:
    points: np.ndarray
    triangles: np.ndarray
    attempts: int
    added_points: int
    improved: bool


def _areas(points, triangles):
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])


def _canonical(triangles):
    rows = [tuple(np.roll(row, -int(np.argmin(row)))) for row in triangles]
    return np.asarray(sorted(rows), dtype=np.int64)


def repair_boundary_fan(points, triangles, protected, poor_triangles, *, score,
                        max_candidates=6, max_insertions=1, relax=None,
                        consume_work=None, cancellation_check=None, target_met=None):
    """Replace a three-cell cavity with a five-cell fan, at most once.

    Only an interior node incident on at least three protected boundary
    intervals is considered. The candidate point belongs to the adjacent
    cavity, never a protected segment. Publication remains the caller's job.
    """
    original = np.array(points, dtype=np.float64, copy=True)
    cells = np.array(triangles, dtype=np.int64, copy=True)
    if (original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all()
            or cells.ndim != 2 or cells.shape[1] != 3
            or np.any(cells < 0) or np.any(cells >= len(original))):
        raise ValueError("invalid boundary-fan input")
    if (not isinstance(max_candidates, int) or max_candidates < 0
            or not isinstance(max_insertions, int) or max_insertions not in (0, 1)):
        raise ValueError("invalid boundary-fan budget")
    if np.any(_areas(original, cells) <= 0):
        raise ValueError("non-positive boundary-fan input")
    protected_edges = {tuple(sorted(map(int, edge))) for edge in protected}
    incidence = defaultdict(list)
    for index, row in enumerate(cells):
        for a, b in zip(row, np.roll(row, -1)):
            incidence[tuple(sorted((int(a), int(b))))].append(index)
    if any(len(rows) > 2 for rows in incidence.values()) or not protected_edges.issubset(incidence):
        raise ValueError("invalid boundary-fan incidence")
    boundary = {edge for edge, rows in incidence.items() if len(rows) == 1}
    fixed = sorted({v for edge in protected_edges | boundary for v in edge})
    poor = sorted(set(map(int, poor_triangles)))
    if any(i < 0 or i >= len(cells) for i in poor):
        raise ValueError("invalid boundary-fan defect index")
    best_points, best_cells = original.copy(), cells.copy()
    best_score = score(original.copy(), cells.copy())
    attempts = 0

    def finish():
        added = int(len(best_points) > len(original))
        return BoundaryFanRepair(best_points, best_cells, attempts, added, bool(added))

    if not max_candidates or not max_insertions or not poor:
        return finish()
    active = sorted({int(v) for i in poor for v in cells[i]} - set(fixed))
    seen_cavities = set()
    for node in active:
        fan = []
        for index in np.flatnonzero(np.any(cells == node, axis=1)):
            edge = tuple(sorted(int(v) for v in cells[index] if v != node))
            if edge in boundary and edge in protected_edges:
                fan.append((int(index), edge))
        if len(fan) < 3:
            continue
        degrees = defaultdict(int)
        for _, edge in fan:
            for v in edge:
                degrees[v] += 1
        if any(degree > 2 for degree in degrees.values()):
            continue
        for endpoint in sorted(v for v, degree in degrees.items() if degree == 1):
            first = [i for i, edge in fan if endpoint in edge]
            edge = tuple(sorted((node, endpoint)))
            if len(first) != 1 or edge in protected_edges or len(incidence[edge]) != 2:
                continue
            second = next(i for i in incidence[edge] if i != first[0])
            opposite = next(int(v) for v in cells[second] if v not in edge)
            next_edge = tuple(sorted((node, opposite)))
            if next_edge in protected_edges or len(incidence[next_edge]) != 2:
                continue
            third = next(i for i in incidence[next_edge] if i != second)
            cavity = tuple(sorted((first[0], second, third)))
            if len(set(cavity)) != 3 or cavity in seen_cavities:
                continue
            seen_cavities.add(cavity)
            local = defaultdict(int)
            for row in cells[list(cavity)]:
                for a, b in zip(row, np.roll(row, -1)):
                    local[tuple(sorted((int(a), int(b))))] += 1
            if any(count == 2 and e in protected_edges for e, count in local.items()):
                continue
            rim = [e for e, count in local.items() if count == 1]
            adjacent = defaultdict(list)
            for a, b in rim:
                adjacent[a].append(b)
                adjacent[b].append(a)
            if len(rim) != 5 or len(adjacent) != 5 or any(len(v) != 2 for v in adjacent.values()):
                continue
            polygon = [min(adjacent)]
            previous = None
            while len(polygon) < 5:
                choices = sorted(v for v in adjacent[polygon[-1]] if v != previous)
                following = choices[0]
                if following in polygon:
                    break
                previous = polygon[-1]
                polygon.append(following)
            if len(polygon) != 5 or polygon[0] not in adjacent[polygon[-1]]:
                continue
            xy = original[polygon]
            area = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
            if area < 0:
                polygon = [polygon[0]] + list(reversed(polygon[1:]))
            weights = _areas(original, cells[list(cavity)])
            centers = (
                np.mean(original[polygon], axis=0),
                np.average(np.mean(original[cells[list(cavity)]], axis=1), axis=0, weights=weights),
                np.mean(original[cells[second]], axis=0),
            )
            seen_centers = set()
            for center in centers:
                if attempts >= max_candidates:
                    return finish()
                if cancellation_check is not None:
                    cancellation_check()
                if center.tobytes() in seen_centers:
                    continue
                seen_centers.add(center.tobytes())
                if consume_work is not None and not consume_work():
                    return finish()
                attempts += 1
                trial = np.vstack((original, center))
                replacement = np.asarray([(a, b, len(original)) for a, b in
                                          zip(polygon, polygon[1:] + polygon[:1])], dtype=np.int64)
                if np.any(_areas(trial, replacement) <= 0):
                    continue
                if not np.isclose(np.sum(_areas(trial, replacement)), np.sum(weights),
                                  rtol=64 * np.finfo(float).eps, atol=0):
                    continue
                trial_cells = _canonical(np.vstack((np.delete(cells, cavity, axis=0), replacement)))
                if relax is not None:
                    trial = np.array(relax(trial.copy(), trial_cells.copy()), dtype=np.float64, copy=True)
                if (trial.shape != (len(original) + 1, 2) or not np.isfinite(trial).all()
                        or trial[fixed].tobytes() != original[fixed].tobytes()
                        or np.any(_areas(trial, trial_cells) <= 0)):
                    raise ValueError("invalid relaxed boundary-fan candidate")
                candidate_score = score(trial.copy(), trial_cells.copy())
                if candidate_score < best_score:
                    best_points, best_cells, best_score = trial.copy(), trial_cells.copy(), candidate_score
                    if target_met is not None and target_met(trial.copy(), trial_cells.copy()):
                        return finish()
    return finish()
