"""Bounded deterministic optimization for native UV meshes."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from operator import index
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .errors import MeshError
from .triangulation import orient2d

__all__ = [
    "EdgeFlipResult",
    "SmoothingResult",
    "constrained_smoothing",
    "local_edge_flip",
]


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    made = np.ascontiguousarray(value, dtype=dtype)
    made.setflags(write=False)
    return made


def _points(value: Any) -> np.ndarray:
    made = np.asarray(value, dtype=np.float64)
    if made.ndim != 2 or made.shape[1] != 2:
        raise MeshError("optimization points must have shape (n, 2)")
    if not np.all(np.isfinite(made)):
        raise MeshError("optimization points must be finite")
    return made


def _node(value: Any, count: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MeshError(f"{name} must contain integer node rows")
    try:
        made = int(index(value))
    except TypeError as error:
        raise MeshError(f"{name} must contain integer node rows") from error
    if made < 0 or made >= count:
        raise MeshError(f"{name} references an invalid node row")
    return made


def _edge(value: Sequence[int], count: int, name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise MeshError(f"{name} needs two node rows")
    first = _node(value[0], count, name)
    second = _node(value[1], count, name)
    if first == second:
        raise MeshError(f"{name} cannot have zero length")
    return (first, second) if first < second else (second, first)


def _cell_edges(cell: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (min(int(cell[i]), int(cell[(i + 1) % len(cell)])),
         max(int(cell[i]), int(cell[(i + 1) % len(cell)])))
        for i in range(len(cell))
    )


def _cells(value: Any, count: int) -> tuple[tuple[int, ...], ...]:
    array = np.asarray(value)
    raw: Iterable[Any]
    if array.ndim == 2 and array.shape[1] in (3, 4):
        raw = array
    else:
        try:
            raw = tuple(value)
        except TypeError as error:
            raise MeshError("cells must be an iterable of T3 or Q4 connectivity") from error
    result: list[tuple[int, ...]] = []
    for number, item in enumerate(raw):
        values = tuple(item)
        if len(values) not in (3, 4):
            raise MeshError("cells must contain only T3 or Q4 connectivity")
        made = tuple(_node(node, count, f"cell {number}") for node in values)
        if len(set(made)) != len(made):
            raise MeshError(f"cell {number} repeats a node row")
        result.append(made)
    return tuple(result)


def _turns(points: np.ndarray, cell: Sequence[int]) -> tuple[float, ...]:
    return tuple(
        orient2d(
            points[cell[(i - 1) % len(cell)]],
            points[cell[i]],
            points[cell[(i + 1) % len(cell)]],
        )
        for i in range(len(cell))
    )


def _cell_sign(points: np.ndarray, cell: Sequence[int], number: int) -> int:
    turns = _turns(points, cell)
    if all(value > 0.0 for value in turns):
        return 1
    if all(value < 0.0 for value in turns):
        return -1
    raise MeshError(f"cell {number} is degenerate, concave, or self-intersecting")


def _valid_cell(points: np.ndarray, cell: Sequence[int], sign: int) -> bool:
    return all(sign * value > 0.0 for value in _turns(points, cell))


def _metric_values(metric: Any, points: np.ndarray) -> np.ndarray:
    value = metric(points.copy()) if callable(metric) else metric
    if value is None:
        return np.repeat(np.eye(2, dtype=np.float64)[None, :, :], len(points), axis=0)
    made = np.asarray(value, dtype=np.float64)
    if made.shape == (2, 2):
        made = np.repeat(made[None, :, :], len(points), axis=0)
    if made.shape != (len(points), 2, 2) or not np.all(np.isfinite(made)):
        raise MeshError("metric must have shape (2, 2) or (n, 2, 2)")
    made = 0.5 * (made + np.swapaxes(made, 1, 2))
    if np.any(np.linalg.eigvalsh(made) <= 0.0):
        raise MeshError("metric tensors must be positive definite")
    return made


def _project_candidate(
    projector: Callable[[np.ndarray], Any] | None,
    point: np.ndarray,
) -> np.ndarray:
    if projector is None:
        return point
    value = projector(point.reshape(1, 2).copy())
    if hasattr(value, "uv"):
        value = value.uv
    made = _points(value)
    if made.shape != (1, 2):
        raise MeshError("smoothing projector must return one UV row")
    return made[0]


@dataclass(frozen=True)
class SmoothingResult:
    """Result and bounded-work diagnostics for constrained smoothing."""

    points: np.ndarray
    moved_nodes: np.ndarray
    iterations: int
    accepted_moves: int
    rejected_moves: int
    converged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _readonly(_points(self.points), np.float64))
        moved = np.asarray(self.moved_nodes, dtype=np.int64)
        if moved.ndim != 1:
            raise MeshError("moved_nodes must be one-dimensional")
        object.__setattr__(self, "moved_nodes", _readonly(moved, np.int64))


def constrained_smoothing(
    points: Any,
    cells: Any,
    *,
    fixed_nodes: Iterable[int] = (),
    constrained_edges: Iterable[Sequence[int]] = (),
    preserve_boundary: bool = True,
    metric: Any = None,
    projector: Callable[[np.ndarray], Any] | None = None,
    iterations: int = 8,
    relaxation: float = 0.5,
) -> SmoothingResult:
    """Metric Laplacian smoothing with fixed constraints and inversion guards.

    Inputs and outputs are UV node rows.  Boundary nodes are fixed by default;
    every endpoint of a constrained edge is fixed as well.  Candidate moves
    are applied in node-row order and rejected if any incident T3/Q4 would
    lose its original convex orientation.
    """

    coordinates = _points(points).copy()
    topology = _cells(cells, len(coordinates))
    try:
        iteration_limit = int(index(iterations))
    except TypeError as error:
        raise MeshError("iterations must be a non-negative integer") from error
    if iteration_limit < 0:
        raise MeshError("iterations must be a non-negative integer")
    made_relaxation = float(relaxation)
    if not np.isfinite(made_relaxation) or not 0.0 < made_relaxation <= 1.0:
        raise MeshError("relaxation must be finite and in (0, 1]")

    neighbors = [set() for _ in range(len(coordinates))]
    incident_cells = [set() for _ in range(len(coordinates))]
    incidence: dict[tuple[int, int], int] = {}
    signs = tuple(
        _cell_sign(coordinates, cell, number)
        for number, cell in enumerate(topology)
    )
    for cell_number, cell in enumerate(topology):
        for node in cell:
            incident_cells[node].add(cell_number)
        for edge in _cell_edges(cell):
            incidence[edge] = incidence.get(edge, 0) + 1
            neighbors[edge[0]].add(edge[1])
            neighbors[edge[1]].add(edge[0])

    fixed = {
        _node(value, len(coordinates), "fixed_nodes") for value in fixed_nodes
    }
    constraints = {
        _edge(value, len(coordinates), "constrained edge")
        for value in constrained_edges
    }
    fixed.update(node for edge in constraints for node in edge)
    if preserve_boundary:
        fixed.update(
            node for edge, attached in incidence.items() if attached == 1 for node in edge
        )

    accepted = 0
    rejected = 0
    moved: set[int] = set()
    iterations_run = 0
    converged = not topology or iteration_limit == 0
    for iteration_number in range(iteration_limit):
        metrics = _metric_values(metric, coordinates)
        accepted_this_iteration = 0
        for node in range(len(coordinates)):
            if node in fixed or not neighbors[node]:
                continue
            system = np.zeros((2, 2), dtype=np.float64)
            right = np.zeros(2, dtype=np.float64)
            for other in sorted(neighbors[node]):
                edge_metric = 0.5 * (metrics[node] + metrics[other])
                system += edge_metric
                right += edge_metric @ coordinates[other]
            try:
                target = np.linalg.solve(system, right)
            except np.linalg.LinAlgError:
                target = np.mean(coordinates[sorted(neighbors[node])], axis=0)
            candidate = coordinates[node] + made_relaxation * (
                target - coordinates[node]
            )
            candidate = _project_candidate(projector, candidate)
            if np.array_equal(candidate, coordinates[node]):
                continue
            previous = coordinates[node].copy()
            coordinates[node] = candidate
            valid = all(
                _valid_cell(coordinates, topology[cell_number], signs[cell_number])
                for cell_number in sorted(incident_cells[node])
            )
            if not valid:
                coordinates[node] = previous
                rejected += 1
                continue
            accepted += 1
            accepted_this_iteration += 1
            moved.add(node)
        iterations_run = iteration_number + 1
        if accepted_this_iteration == 0:
            converged = True
            break

    return SmoothingResult(
        coordinates,
        np.asarray(sorted(moved), dtype=np.int64),
        iterations_run,
        accepted,
        rejected,
        converged,
    )


def _triangle_quality(
    points: np.ndarray,
    triangle: Sequence[int],
    metrics: np.ndarray,
) -> float:
    rows = np.asarray(triangle, dtype=np.int64)
    coordinates = points[rows]
    metric = np.mean(metrics[rows], axis=0)
    determinant = float(np.linalg.det(metric))
    if determinant <= 0.0:
        return 0.0
    edges = (
        coordinates[1] - coordinates[0],
        coordinates[2] - coordinates[1],
        coordinates[0] - coordinates[2],
    )
    squared = sum(float(edge @ metric @ edge) for edge in edges)
    area_twice = abs(orient2d(coordinates[0], coordinates[1], coordinates[2]))
    if squared <= 0.0 or area_twice <= 0.0:
        return 0.0
    return float(2.0 * np.sqrt(3.0) * area_twice * np.sqrt(determinant) / squared)


def _ccw(points: np.ndarray, triangle: Sequence[int]) -> tuple[int, int, int] | None:
    made = tuple(int(value) for value in triangle)
    orientation = orient2d(points[made[0]], points[made[1]], points[made[2]])
    if orientation == 0.0:
        return None
    return made if orientation > 0.0 else (made[0], made[2], made[1])


def _triangle_incidence(
    triangles: np.ndarray,
) -> dict[tuple[int, int], set[int]]:
    result: dict[tuple[int, int], set[int]] = {}
    for row, triangle in enumerate(triangles):
        for edge in _cell_edges(triangle):
            result.setdefault(edge, set()).add(row)
    return result


@dataclass(frozen=True)
class EdgeFlipResult:
    """Connectivity and work-queue diagnostics from local edge flipping."""

    triangles: np.ndarray
    flip_count: int
    queue_visits: int
    converged: bool

    def __post_init__(self) -> None:
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise MeshError("edge-flip triangles must have shape (m, 3)")
        object.__setattr__(self, "triangles", _readonly(triangles, np.int64))


def local_edge_flip(
    points: Any,
    triangles: Any,
    *,
    protected_edges: Iterable[Sequence[int]] = (),
    metric: Any = None,
    max_flips: int | None = None,
) -> EdgeFlipResult:
    """Improve local metric triangle quality with a deterministic edge queue.

    Triangle rows stay in place so external element IDs remain aligned.
    Protected edges are never removed.  A flip is accepted only when the
    minimum metric mean-ratio quality of its two triangles strictly improves.
    """

    coordinates = _points(points)
    connectivity = np.asarray(triangles, dtype=np.int64)
    if connectivity.ndim == 1 and connectivity.size == 0:
        connectivity = np.empty((0, 3), dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 3:
        raise MeshError("triangles must have shape (m, 3)")
    if connectivity.size and (
        int(np.min(connectivity)) < 0
        or int(np.max(connectivity)) >= len(coordinates)
    ):
        raise MeshError("triangle connectivity references an invalid node row")
    work = np.empty_like(connectivity)
    for row, triangle in enumerate(connectivity):
        if len(set(map(int, triangle))) != 3:
            raise MeshError(f"triangle {row} repeats a node row")
        made = _ccw(coordinates, triangle)
        if made is None:
            raise MeshError(f"triangle {row} has zero area")
        work[row] = made

    protected = {
        _edge(value, len(coordinates), "protected edge")
        for value in protected_edges
    }
    metrics = _metric_values(metric, coordinates)
    if max_flips is None:
        flip_limit = max(16, 8 * len(work))
    else:
        try:
            flip_limit = int(index(max_flips))
        except TypeError as error:
            raise MeshError("max_flips must be a non-negative integer") from error
        if flip_limit < 0:
            raise MeshError("max_flips must be a non-negative integer")

    incidence = _triangle_incidence(work)
    queue: list[tuple[int, int]] = []
    queued: set[tuple[int, int]] = set()

    def enqueue(edge: tuple[int, int]) -> None:
        if edge not in protected and len(incidence.get(edge, ())) == 2 and edge not in queued:
            heappush(queue, edge)
            queued.add(edge)

    for edge in sorted(incidence):
        enqueue(edge)

    visits = 0
    flips = 0
    converged = True
    while queue:
        if flips >= flip_limit:
            converged = False
            break
        edge = heappop(queue)
        queued.discard(edge)
        visits += 1
        attached = sorted(incidence.get(edge, ()))
        if edge in protected or len(attached) != 2:
            continue
        first_row, second_row = attached
        first = tuple(int(value) for value in work[first_row])
        second = tuple(int(value) for value in work[second_row])
        first_opposite = next((node for node in first if node not in edge), None)
        second_opposite = next((node for node in second if node not in edge), None)
        if first_opposite is None or second_opposite is None or first_opposite == second_opposite:
            continue
        a, b = edge
        c, d = int(first_opposite), int(second_opposite)
        if (
            orient2d(coordinates[a], coordinates[b], coordinates[c])
            * orient2d(coordinates[a], coordinates[b], coordinates[d])
            >= 0.0
            or orient2d(coordinates[c], coordinates[d], coordinates[a])
            * orient2d(coordinates[c], coordinates[d], coordinates[b])
            >= 0.0
        ):
            continue
        new_edge = (c, d) if c < d else (d, c)
        if incidence.get(new_edge, set()).difference(attached):
            continue
        candidate_first = _ccw(coordinates, (c, d, a))
        candidate_second = _ccw(coordinates, (d, c, b))
        if candidate_first is None or candidate_second is None:
            continue
        candidates = sorted((candidate_first, candidate_second))
        current_quality = min(
            _triangle_quality(coordinates, first, metrics),
            _triangle_quality(coordinates, second, metrics),
        )
        candidate_quality = min(
            _triangle_quality(coordinates, candidates[0], metrics),
            _triangle_quality(coordinates, candidates[1], metrics),
        )
        if candidate_quality <= current_quality:
            continue

        old_edges = set(_cell_edges(first)) | set(_cell_edges(second))
        for old_edge in old_edges:
            rows = incidence[old_edge]
            rows.discard(first_row)
            rows.discard(second_row)
            if not rows:
                del incidence[old_edge]
        work[first_row] = candidates[0]
        work[second_row] = candidates[1]
        new_edges = set(_cell_edges(candidates[0])) | set(_cell_edges(candidates[1]))
        for made_edge in new_edges:
            rows = incidence.setdefault(made_edge, set())
            if made_edge in _cell_edges(candidates[0]):
                rows.add(first_row)
            if made_edge in _cell_edges(candidates[1]):
                rows.add(second_row)
        flips += 1
        for affected in sorted(old_edges | new_edges):
            enqueue(affected)

    return EdgeFlipResult(work, flips, visits, converged)
