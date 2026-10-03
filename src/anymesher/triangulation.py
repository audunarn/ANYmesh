"""Deterministic constrained planar triangulation without optional geometry wheels."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from math import floor, fsum, hypot, isfinite, sqrt
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .errors import MeshError
from .native import (
    NativeBoundary,
    NativeBoundarySelection,
    run_native_triangulation,
    snapshot_native_boundary,
)

__all__ = [
    "PlanarTriangulation",
    "TriangulationResult",
    "constrained_planar_triangulation",
    "constrained_triangulate",
    "incircle",
    "orient2d",
    "triangulate_polygon",
]


_FLOAT_EPSILON = np.finfo(np.float64).eps


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def orient2d(first: Sequence[float], second: Sequence[float], third: Sequence[float]) -> float:
    """Adaptive orientation predicate; positive means counter-clockwise."""

    ax = float(first[0]) - float(third[0])
    ay = float(first[1]) - float(third[1])
    bx = float(second[0]) - float(third[0])
    by = float(second[1]) - float(third[1])
    determinant = ax * by - ay * bx
    error = 8.0 * _FLOAT_EPSILON * (abs(ax * by) + abs(ay * bx))
    if abs(determinant) > error:
        return determinant
    with localcontext() as context:
        context.prec = 80
        exact_ax = _decimal(first[0]) - _decimal(third[0])
        exact_ay = _decimal(first[1]) - _decimal(third[1])
        exact_bx = _decimal(second[0]) - _decimal(third[0])
        exact_by = _decimal(second[1]) - _decimal(third[1])
        exact = (exact_ax * exact_by) - (exact_ay * exact_bx)
    return float(exact)


def incircle(
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
    point: Sequence[float],
) -> float:
    """Adaptive in-circle predicate, positive for an interior point."""

    ax, ay = float(first[0]) - float(point[0]), float(first[1]) - float(point[1])
    bx, by = float(second[0]) - float(point[0]), float(second[1]) - float(point[1])
    cx, cy = float(third[0]) - float(point[0]), float(third[1]) - float(point[1])
    alift = ax * ax + ay * ay
    blift = bx * bx + by * by
    clift = cx * cx + cy * cy
    determinant = fsum((
        alift * (bx * cy - by * cx),
        -blift * (ax * cy - ay * cx),
        clift * (ax * by - ay * bx),
    ))
    scale = (
        abs(alift * (bx * cy - by * cx))
        + abs(blift * (ax * cy - ay * cx))
        + abs(clift * (ax * by - ay * bx))
    )
    if abs(determinant) <= 32.0 * _FLOAT_EPSILON * scale:
        with localcontext() as context:
            context.prec = 80
            point_x, point_y = _decimal(point[0]), _decimal(point[1])
            dax = _decimal(first[0]) - point_x
            day = _decimal(first[1]) - point_y
            dbx = _decimal(second[0]) - point_x
            dby = _decimal(second[1]) - point_y
            dcx = _decimal(third[0]) - point_x
            dcy = _decimal(third[1]) - point_y
            da = dax * dax + day * day
            db = dbx * dbx + dby * dby
            dc = dcx * dcx + dcy * dcy
            exact = (
                da * (dbx * dcy - dby * dcx)
                - db * (dax * dcy - day * dcx)
                + dc * (dax * dby - day * dbx)
            )
        determinant = float(exact)
    if orient2d(first, second, third) < 0.0:
        determinant = -determinant
    return determinant


def _ring_area(points: np.ndarray, ring: Sequence[int]) -> float:
    return 0.5 * fsum(
        float(points[ring[index], 0] * points[ring[(index + 1) % len(ring)], 1]
              - points[ring[(index + 1) % len(ring)], 0] * points[ring[index], 1])
        for index in range(len(ring))
    )


def _point_on_segment(point: np.ndarray, first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    first_x, first_y = float(first[0]), float(first[1])
    second_x, second_y = float(second[0]), float(second[1])
    point_x, point_y = float(point[0]), float(point[1])
    if not (
        min(first_x, second_x) - tolerance <= point_x <= max(first_x, second_x) + tolerance
        and min(first_y, second_y) - tolerance <= point_y <= max(first_y, second_y) + tolerance
    ):
        return False
    length = hypot(second_x - first_x, second_y - first_y)
    return abs(orient2d(first, second, point)) <= tolerance * max(1.0, length)


def _ring_arrays(points: np.ndarray, ring: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(ring, dtype=np.int64)
    coordinates = np.asarray(points, dtype=np.float64)
    return coordinates[ids], coordinates[np.roll(ids, -1)]


def _near_ring_edges(point: np.ndarray, first: np.ndarray, second: np.ndarray, tolerance: float) -> list[int]:
    """Edges whose tolerance-expanded bounding box holds ``point``.

    Uses exactly the comparisons ``_point_on_segment`` starts with, so every
    edge it could accept is returned.
    """
    x, y = float(point[0]), float(point[1])
    fx, fy, sx, sy = first[:, 0], first[:, 1], second[:, 0], second[:, 1]
    near = (
        (np.minimum(fx, sx) - tolerance <= x) & (x <= np.maximum(fx, sx) + tolerance)
        & (np.minimum(fy, sy) - tolerance <= y) & (y <= np.maximum(fy, sy) + tolerance)
    )
    return np.flatnonzero(near).tolist()


def _point_on_ring(
    point: np.ndarray,
    points: np.ndarray,
    ring: Sequence[int],
    tolerance: float,
    edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """``any(_point_on_segment(...))`` over the ring edges, bbox-prefiltered.

    ``edges`` may carry ``_ring_arrays(points, ring)`` precomputed by a caller
    that queries the same ring many times.
    """
    if len(ring) == 0:
        return False
    first, second = _ring_arrays(points, ring) if edges is None else edges
    return any(
        _point_on_segment(point, first[index], second[index], tolerance)
        for index in _near_ring_edges(point, first, second, tolerance)
    )


def _point_in_ring(
    point: np.ndarray,
    points: np.ndarray,
    ring: Sequence[int],
    tolerance: float,
    edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Even-odd containment, ``True`` on the ring within ``tolerance``.

    Vectorized but decision-identical to the original scalar loop: on-ring
    detection keeps the exact ``_point_on_segment`` test behind the same
    bounding-box comparisons, and each crossing abscissa is evaluated with the
    same operation order.  ``edges`` is as for ``_point_on_ring``.
    """
    if len(ring) == 0:
        return False
    first, second = _ring_arrays(points, ring) if edges is None else edges
    if any(
        _point_on_segment(point, first[index], second[index], tolerance)
        for index in _near_ring_edges(point, first, second, tolerance)
    ):
        return True
    x, y = float(point[0]), float(point[1])
    fx, fy, sx, sy = first[:, 0], first[:, 1], second[:, 0], second[:, 1]
    straddle = (fy > y) != (sy > y)
    if not straddle.any():
        return False
    fx, fy, sx, sy = fx[straddle], fy[straddle], sx[straddle], sy[straddle]
    crossing = fx + (y - fy) * (sx - fx) / (sy - fy)
    return bool(np.count_nonzero(crossing > x) % 2)


def _proper_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    first = orient2d(a, b, c)
    second = orient2d(a, b, d)
    third = orient2d(c, d, a)
    fourth = orient2d(c, d, b)
    return ((first > 0.0 and second < 0.0) or (first < 0.0 and second > 0.0)) and (
        (third > 0.0 and fourth < 0.0) or (third < 0.0 and fourth > 0.0)
    )


def _intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> tuple[float, np.ndarray]:
    direction = b - a
    other = d - c
    denominator = direction[0] * other[1] - direction[1] * other[0]
    if denominator == 0.0:
        raise MeshError("cannot intersect parallel segments")
    delta = c - a
    parameter = float((delta[0] * other[1] - delta[1] * other[0]) / denominator)
    return parameter, a + parameter * direction


def _normal_edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _normal_ring(raw: Sequence[int], count: int, name: str) -> list[int]:
    ring = [int(value) for value in raw]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    collapsed: list[int] = []
    for value in ring:
        if value < 0 or value >= count:
            raise MeshError(f"{name} references invalid point row {value}")
        if not collapsed or collapsed[-1] != value:
            collapsed.append(value)
    if len(collapsed) > 1 and collapsed[0] == collapsed[-1]:
        collapsed.pop()
    if len(collapsed) < 3 or len(set(collapsed)) < 3:
        raise MeshError(f"{name} needs at least three distinct vertices")
    return collapsed


def _canonical_triangle(triangle: Sequence[int], points: np.ndarray) -> tuple[int, int, int]:
    values = [int(value) for value in triangle]
    if orient2d(points[values[0]], points[values[1]], points[values[2]]) < 0.0:
        values[1], values[2] = values[2], values[1]
    start = min(range(3), key=values.__getitem__)
    values = values[start:] + values[:start]
    return values[0], values[1], values[2]


def _deduplicate(
    points: np.ndarray,
    outer: Sequence[int],
    holes: Sequence[Sequence[int]],
    constraints: Sequence[Sequence[int]],
    tolerance: float,
) -> tuple[np.ndarray, list[int], list[list[int]], list[tuple[int, int]], np.ndarray]:
    # A linear scan of every prior unique point made PSLG preparation
    # quadratic before the triangulator even started. Tolerance-sized buckets
    # preserve the same earliest-match semantics: any point within tolerance
    # must be in its own or one of the eight neighbouring cells.
    unique: list[np.ndarray] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    remap = np.empty(len(points), dtype=np.int64)
    tolerance_squared = tolerance * tolerance
    for index, point in enumerate(points):
        cell = (
            floor(float(point[0]) / tolerance),
            floor(float(point[1]) / tolerance),
        )
        match = -1
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                for row in buckets.get(
                    (cell[0] + x_offset, cell[1] + y_offset), ()
                ):
                    delta_x = float(point[0]) - float(unique[row][0])
                    delta_y = float(point[1]) - float(unique[row][1])
                    if (
                        delta_x * delta_x + delta_y * delta_y
                        <= tolerance_squared
                        and (match < 0 or row < match)
                    ):
                        match = row
        if match < 0:
            unique.append(np.array(point, copy=True))
            match = len(unique) - 1
            buckets.setdefault(cell, []).append(match)
        remap[index] = match
    mapped_outer = _normal_ring([int(remap[index]) for index in outer], len(unique), "outer loop")
    mapped_holes = [
        _normal_ring([int(remap[index]) for index in ring], len(unique), f"hole {number}")
        for number, ring in enumerate(holes)
    ]
    mapped_constraints: list[tuple[int, int]] = []
    for raw in constraints:
        if len(raw) != 2:
            raise MeshError("each mandatory constraint needs two point rows")
        first, second = int(raw[0]), int(raw[1])
        if first < 0 or first >= len(points) or second < 0 or second >= len(points):
            raise MeshError("mandatory constraint references an invalid point row")
        mapped = int(remap[first]), int(remap[second])
        if mapped[0] == mapped[1]:
            raise MeshError("mandatory constraint has zero length")
        mapped_constraints.append(mapped)
    return np.asarray(unique, dtype=np.float64), mapped_outer, mapped_holes, mapped_constraints, remap


def _ring_segments(ring: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(ring[index]), int(ring[(index + 1) % len(ring)])) for index in range(len(ring))]


def _segment_candidate_pairs(
    points: np.ndarray,
    segments: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return deterministic AABB candidates without an all-pairs scan."""

    if len(segments) < 2:
        return []
    minimum = np.min(points, axis=0)
    extent = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    bucket_count = max(1, int(len(segments) ** 0.5))
    cell_size = extent / bucket_count if extent > 0.0 else 1.0
    boxes: list[tuple[float, float, float, float]] = []
    buckets: dict[tuple[int, int], list[int]] = {}
    candidates: set[tuple[int, int]] = set()

    for index, (first, second) in enumerate(segments):
        first_point, second_point = points[first], points[second]
        min_x = min(float(first_point[0]), float(second_point[0]))
        max_x = max(float(first_point[0]), float(second_point[0]))
        min_y = min(float(first_point[1]), float(second_point[1]))
        max_y = max(float(first_point[1]), float(second_point[1]))
        box = (min_x, max_x, min_y, max_y)
        boxes.append(box)
        first_x = floor((min_x - float(minimum[0])) / cell_size)
        last_x = floor((max_x - float(minimum[0])) / cell_size)
        first_y = floor((min_y - float(minimum[1])) / cell_size)
        last_y = floor((max_y - float(minimum[1])) / cell_size)
        cells = [
            (x_cell, y_cell)
            for x_cell in range(first_x, last_x + 1)
            for y_cell in range(first_y, last_y + 1)
        ]
        for cell in cells:
            candidates.update((prior, index) for prior in buckets.get(cell, ()))
        for cell in cells:
            buckets.setdefault(cell, []).append(index)

    return [
        (first, second)
        for first, second in sorted(candidates)
        if boxes[first][0] <= boxes[second][1]
        and boxes[second][0] <= boxes[first][1]
        and boxes[first][2] <= boxes[second][3]
        and boxes[second][2] <= boxes[first][3]
    ]


def _validate_ring(points: np.ndarray, ring: Sequence[int], name: str) -> None:
    segments = _ring_segments(ring)
    for first_index, second_index in _segment_candidate_pairs(points, segments):
        a, b = segments[first_index]
        c, d = segments[second_index]
        if len({a, b, c, d}) < 4:
            continue
        if _proper_intersection(points[a], points[b], points[c], points[d]):
            raise MeshError(f"{name} is self-intersecting")


@dataclass(frozen=True)
class _PreparedPSLG:
    points: np.ndarray
    outer: np.ndarray
    holes: tuple[np.ndarray, ...]
    segments: np.ndarray
    boundary_segments: np.ndarray
    mandatory_segments: np.ndarray
    tolerance: float
    protected_node_rows: tuple[tuple[int, int], ...] = ()


def _prepare_pslg(
    raw_points: Any,
    raw_outer: Sequence[int] | None,
    raw_holes: Sequence[Sequence[int]],
    raw_constraints: Sequence[Sequence[int]],
    tolerance: float | None,
    *,
    compiled_kernels: bool = False,
    protected_node_ids: Mapping[int, int] | None = None,
) -> _PreparedPSLG:
    points = np.asarray(raw_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise MeshError("planar points must have shape (n, 2), n >= 3")
    if not np.all(np.isfinite(points)):
        raise MeshError("planar points must be finite")
    extent = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    epsilon = extent * 1.0e-12 if tolerance is None else float(tolerance)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise MeshError("tolerance must be positive and finite")
    outer = list(range(len(points))) if raw_outer is None else list(raw_outer)
    required_protected_input_rows: set[int] = set()
    if protected_node_ids is not None:
        required_protected_input_rows.update(outer)
        required_protected_input_rows.update(row for ring in raw_holes for row in ring)
        required_protected_input_rows.update(row for edge in raw_constraints for row in edge)
    original_points = points
    points, outer, holes, constraints, remap = _deduplicate(
        points, outer, raw_holes, raw_constraints, epsilon
    )
    protected_rows: tuple[tuple[int, int], ...] = ()
    if protected_node_ids is not None:
        if not isinstance(protected_node_ids, Mapping):
            raise MeshError("protected_node_ids must map input rows to registry IDs")
        by_id: dict[int, int] = {}
        by_row: dict[int, int] = {}
        supplied_rows: set[int] = set()
        for raw_row, registry_id in protected_node_ids.items():
            if isinstance(raw_row, (bool, np.bool_)) or not isinstance(raw_row, (int, np.integer)):
                raise MeshError("protected input row must be an integer")
            if isinstance(registry_id, (bool, np.bool_)) or not isinstance(registry_id, (int, np.integer)) or registry_id < 0:
                raise MeshError("protected registry ID must be a nonnegative integer")
            row = int(raw_row)
            node_id = int(registry_id)
            if row < 0 or row >= len(original_points):
                raise MeshError("protected input row is outside the PSLG")
            mapped = int(remap[row])
            if not np.array_equal(original_points[row], points[mapped]):
                raise MeshError("protected station coordinate changed during deduplication")
            if node_id in by_id and by_id[node_id] != mapped:
                raise MeshError("one protected registry ID maps to multiple point rows")
            if mapped in by_row and by_row[mapped] != node_id:
                raise MeshError("distinct protected registry IDs collapsed to one point row")
            by_id[node_id] = mapped
            by_row[mapped] = node_id
            supplied_rows.add(row)
        if not required_protected_input_rows <= supplied_rows:
            raise MeshError("boundary or mandatory constraint is missing a protected registry ID")
        protected_rows = tuple(sorted(by_id.items()))
    _validate_ring(points, outer, "outer loop")
    for number, ring in enumerate(holes):
        _validate_ring(points, ring, f"hole {number}")
    if abs(_ring_area(points, outer)) <= epsilon * epsilon:
        raise MeshError("outer loop has zero area")
    if _ring_area(points, outer) < 0.0:
        outer.reverse()
    for ring in holes:
        if abs(_ring_area(points, ring)) <= epsilon * epsilon:
            raise MeshError("hole loop has zero area")
        if _ring_area(points, ring) > 0.0:
            ring.reverse()
        if not _point_in_ring(points[ring[0]], points, outer, epsilon):
            raise MeshError("a hole lies outside the outer loop")

    boundary = _ring_segments(outer)
    for ring in holes:
        boundary.extend(_ring_segments(ring))
    records: list[tuple[int, int, str]] = [(*edge, "boundary") for edge in boundary]
    records.extend((*edge, "mandatory") for edge in constraints)

    # Constraints may cross each other.  Turn every such crossing into an
    # explicit vertex before Delaunay construction; crossings with a domain
    # boundary are invalid rather than silently clipping the constraint.
    mutable_points = [np.array(point, copy=True) for point in points]
    segment_rows = [(first, second) for first, second, _kind in records]
    for first_index, second_index in _segment_candidate_pairs(points, segment_rows):
        a, b, first_kind = records[first_index]
        c, d, second_kind = records[second_index]
        if len({a, b, c, d}) < 4:
            continue
        if not _proper_intersection(
            mutable_points[a], mutable_points[b], mutable_points[c], mutable_points[d]
        ):
            continue
        if first_kind == "boundary" or second_kind == "boundary":
            raise MeshError("a mandatory constraint crosses the domain boundary")
        _, crossing = _intersection(
            mutable_points[a], mutable_points[b], mutable_points[c], mutable_points[d]
        )
        if not any(np.linalg.norm(crossing - candidate) <= epsilon for candidate in mutable_points):
            mutable_points.append(crossing)
    points = np.asarray(mutable_points, dtype=np.float64)

    compiled_memberships = None
    if compiled_kernels:
        from .native_cpp import pslg_segment_memberships

        compiled_memberships = pslg_segment_memberships(
            points,
            np.asarray([(first, second) for first, second, _ in records], dtype=np.int64),
            epsilon,
        )
    split_records: list[tuple[int, int, str]] = []
    boundary_chains: dict[tuple[int, int], list[int]] = {}
    point_x, point_y = points[:, 0], points[:, 1]
    for record_index, (first, second, kind) in enumerate(records):
        start, end = points[first], points[second]
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if compiled_memberships is None:
            members: list[tuple[float, int]] = []
            # The same bounding-box comparisons ``_point_on_segment`` starts
            # with, vectorized; every row it could accept stays a candidate
            # and is still decided by the scalar test, in ascending row order.
            start_x, start_y = float(start[0]), float(start[1])
            end_x, end_y = float(end[0]), float(end[1])
            near = (
                (min(start_x, end_x) - epsilon <= point_x)
                & (point_x <= max(start_x, end_x) + epsilon)
                & (min(start_y, end_y) - epsilon <= point_y)
                & (point_y <= max(start_y, end_y) + epsilon)
            )
            for row in np.flatnonzero(near).tolist():
                point = points[row]
                if _point_on_segment(point, start, end, epsilon):
                    parameter = float(np.dot(point - start, direction) / denominator)
                    if -epsilon <= parameter <= 1.0 + epsilon:
                        members.append((min(1.0, max(0.0, parameter)), row))
            members.sort(key=lambda item: (item[0], item[1]))
            ordered = [row for _, row in members]
        else:
            ordered = list(compiled_memberships[record_index])
        # Tolerance-qualified memberships outside an endpoint are not boundary
        # stations. Clamping and ID tie breaking must never reorder endpoints.
        ordered = [first] + [row for row in ordered if row not in (first, second)
                             and 0.0 < float(np.dot(points[row]-start, direction)/denominator) < 1.0] + [second]
        if kind == "boundary":
            boundary_chains[first, second] = ordered
        for a, b in zip(ordered, ordered[1:]):
            if a != b:
                split_records.append((a, b, kind))

    boundary_set = {_normal_edge(a, b) for a, b, kind in split_records if kind == "boundary"}
    mandatory_set = {_normal_edge(a, b) for a, b, kind in split_records if kind == "mandatory"}
    all_segments = sorted(boundary_set | mandatory_set)

    # Domain filtering follows the actual split boundary, including stations
    # inserted on a ring edge. Keep its material side on the left.
    def expanded_ring(ring: list[int]) -> list[int]:
        result = [node for first, second in zip(ring, ring[1:] + ring[:1])
                  for node in boundary_chains[first, second][:-1]]
        if len(set(result)) != len(result):
            raise MeshError("expanded boundary has ambiguous repeated stations")
        return result

    outer = expanded_ring(outer)
    holes = [expanded_ring(ring) for ring in holes]
    _validate_ring(points, outer, "expanded outer loop")
    for number, ring in enumerate(holes):
        _validate_ring(points, ring, f"expanded hole {number}")

    compiled_domain = None
    if compiled_kernels:
        from .native_cpp import pslg_domain_classification

        compiled_domain = pslg_domain_classification(
            points,
            np.asarray(outer, dtype=np.int64),
            tuple(np.asarray(ring, dtype=np.int64) for ring in holes),
            epsilon,
        )
    outer_edges = _ring_arrays(points, outer) if outer else None
    hole_edges = [_ring_arrays(points, ring) if ring else None for ring in holes]
    for row, point in enumerate(points):
        if compiled_domain is None:
            on_outer = _point_on_ring(point, points, outer, epsilon, outer_edges)
            inside = _point_in_ring(point, points, outer, epsilon, outer_edges)
            in_hole = any(
                _point_in_ring(point, points, ring, epsilon, edges)
                and not _point_on_ring(point, points, ring, epsilon, edges)
                for ring, edges in zip(holes, hole_edges)
            )
        else:
            on_outer, inside, in_hole = map(bool, compiled_domain[row])
        if not inside or (in_hole and not on_outer):
            is_hole_vertex = any(row in ring for ring in holes)
            if not is_hole_vertex:
                raise MeshError("a point or constraint endpoint lies outside the meshed domain")

    def array_of(values: Iterable[tuple[int, int]]) -> np.ndarray:
        return np.asarray(sorted(values), dtype=np.int64).reshape((-1, 2))

    frozen_points = np.ascontiguousarray(points, dtype=np.float64)
    frozen_points.setflags(write=False)
    return _PreparedPSLG(
        points=frozen_points,
        outer=np.asarray(outer, dtype=np.int64),
        holes=tuple(np.asarray(ring, dtype=np.int64) for ring in holes),
        segments=array_of(all_segments),
        boundary_segments=array_of(boundary_set),
        mandatory_segments=array_of(mandatory_set),
        tolerance=epsilon,
        protected_node_rows=protected_rows,
    )


def _circumcircle(work: np.ndarray, triangle: tuple[int, int, int]) -> tuple[float, float, float, float, float, float]:
    """``(ax, ay, ux, uy, r2, delta)`` for the Bowyer-Watson pre-filter.

    ``u`` is the circumcentre offset from vertex ``a`` and ``delta`` a
    generous forward bound on the rounding error of ``u``.  Needle triangles
    (e.g. two close points and a far super-triangle vertex) cancel badly in the
    denominator; their large ``delta`` widens the filter instead of risking a
    false exclusion.  Degenerate triangles get an infinite radius.
    """
    ax, ay = float(work[triangle[0]][0]), float(work[triangle[0]][1])
    bx, by = float(work[triangle[1]][0]) - ax, float(work[triangle[1]][1]) - ay
    cx, cy = float(work[triangle[2]][0]) - ax, float(work[triangle[2]][1]) - ay
    denominator = 2.0 * (bx * cy - by * cx)
    denominator_error = 16.0 * _FLOAT_EPSILON * (abs(bx * cy) + abs(by * cx))
    if abs(denominator) <= 2.0 * denominator_error:
        return ax, ay, 0.0, 0.0, float("inf"), 0.0
    blift = bx * bx + by * by
    clift = cx * cx + cy * cy
    numerator_x = cy * blift - by * clift
    numerator_y = bx * clift - cx * blift
    ux = numerator_x / denominator
    uy = numerator_y / denominator
    error_x = 16.0 * _FLOAT_EPSILON * (abs(cy * blift) + abs(by * clift))
    error_y = 16.0 * _FLOAT_EPSILON * (abs(bx * clift) + abs(cx * blift))
    delta = 16.0 * (
        (error_x + abs(ux) * denominator_error) + (error_y + abs(uy) * denominator_error)
    ) / abs(denominator)
    radius2 = ux * ux + uy * uy
    if not (np.isfinite(radius2) and np.isfinite(delta)):
        return ax, ay, 0.0, 0.0, float("inf"), 0.0
    return ax, ay, ux, uy, radius2, delta


# Relative slack for rounding in the distance evaluation itself.
_CIRCLE_FILTER_SLACK = 1.0e-8

# Cavity-search grid margins (``_bowyer_watson``).  The registered disk must
# contain every point the pre-filter keeps; these margins exceed the filter's
# own slack by orders of magnitude and only cost a few extra candidates.
_GRID_RADIUS_MARGIN = 1.0e-6
_GRID_ABSOLUTE_MARGIN = 1.0e-9
# Disks covering more cells than this are checked on every insertion instead.
_GRID_GLOBAL_CELLS = 64


def _incircle_candidates(data: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Rows (ascending) whose circumcircle may strictly contain ``point``.

    ``data`` holds one ``_circumcircle`` row per live triangle.  A row is
    excluded only when the point is outside the circle by more than the
    propagated circumcentre error plus a relative slack.
    """

    px = float(point[0]) - data[:, 0]
    py = float(point[1]) - data[:, 1]
    dx = px - data[:, 2]
    dy = py - data[:, 3]
    distance2 = dx * dx + dy * dy
    radius2 = data[:, 4]
    delta = data[:, 5]
    with np.errstate(invalid="ignore", over="ignore"):
        bound = (
            radius2
            + 2.0 * delta * (np.sqrt(distance2) + np.sqrt(radius2))
            + 2.0 * delta * delta
            + _CIRCLE_FILTER_SLACK * (radius2 + px * px + py * py)
            + 1.0e-300
        )
        outside = distance2 > bound
    # NaN or infinite bounds compare False and therefore stay candidates.
    return np.flatnonzero(~outside)


def _bowyer_watson(points: np.ndarray) -> list[tuple[int, int, int]]:
    count = len(points)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = 0.5 * (minimum + maximum)
    span = max(float(np.max(maximum - minimum)), 1.0)
    super_points = np.array(
        [
            center + np.array((-32.0 * span, -16.0 * span)),
            center + np.array((32.0 * span, -16.0 * span)),
            center + np.array((0.0, 32.0 * span)),
        ]
    )
    work = np.vstack((points, super_points))
    # Plain float tuples: the scalar predicates read the same values, faster
    # than indexing numpy rows.
    coords = [tuple(row) for row in work.tolist()]
    # Triangles live in reusable slots so an insertion costs O(cavity) work
    # instead of rebuilding every list and array.  ``slots[row]`` and
    # ``circles[row]`` stay aligned; ``alive`` marks occupied rows.  The live
    # triangles form a set: which slot a triangle occupies never affects which
    # triangles exist.
    capacity = 2 * count + 16
    slots: list[tuple[int, int, int] | None] = [None] * capacity
    circles = np.zeros((capacity, 6), dtype=np.float64)
    alive = np.zeros(capacity, dtype=bool)
    free: list[int] = list(range(capacity - 1, -1, -1))

    # Every query point is an input point, so a uniform grid over the input
    # bounding box locates candidates.  Each triangle is registered in the
    # cells met by the bounding box of a disk that contains every point the
    # circumcircle pre-filter would keep: the computed radius widened by a
    # relative margin (far above the filter's 1e-8 slack), twice the
    # circumcentre error bound and an absolute rounding margin.  Unbounded or
    # very large disks (super-triangle fans, needles) stay on a short global
    # list that every query also checks.  Candidates therefore remain a
    # superset of the full-scan pre-filter's, and ``incircle`` decides each.
    low_x, low_y = float(minimum[0]), float(minimum[1])
    extent_x = float(maximum[0]) - low_x
    extent_y = float(maximum[1]) - low_y
    per_axis = max(1, int(np.ceil(np.sqrt(count))))
    cell_size = max(extent_x, extent_y) / per_axis
    if not (np.isfinite(cell_size) and cell_size > 0.0):
        cell_size = 1.0
    columns = int(extent_x / cell_size) + 1
    grid_rows = int(extent_y / cell_size) + 1
    buckets: dict[int, set[int]] = {}
    registered: dict[int, list[int]] = {}
    global_rows: set[int] = set()

    def column_of(x: float) -> int:
        return min(columns - 1, max(0, floor((x - low_x) / cell_size)))

    def row_of(y: float) -> int:
        return min(grid_rows - 1, max(0, floor((y - low_y) / cell_size)))

    def register(
        row: int,
        triangle: tuple[int, int, int],
        circle: tuple[float, float, float, float, float, float],
    ) -> None:
        if triangle[0] >= count or triangle[1] >= count or triangle[2] >= count:
            # Super-triangle fans have enormous disks; skip the arithmetic.
            global_rows.add(row)
            return
        ax, ay, ux, uy, radius2, delta = circle
        radius = (
            sqrt(radius2) * (1.0 + _GRID_RADIUS_MARGIN)
            + 4.0 * delta
            + _GRID_ABSOLUTE_MARGIN * (abs(ax) + abs(ay) + abs(ux) + abs(uy) + span)
        )
        center_x, center_y = ax + ux, ay + uy
        if not (isfinite(radius) and isfinite(center_x) and isfinite(center_y)):
            global_rows.add(row)
            return
        first_column, last_column = column_of(center_x - radius), column_of(center_x + radius)
        first_row, last_row = row_of(center_y - radius), row_of(center_y + radius)
        if (last_column - first_column + 1) * (last_row - first_row + 1) > _GRID_GLOBAL_CELLS:
            global_rows.add(row)
            return
        keys = [
            grid_row * columns + column
            for grid_row in range(first_row, last_row + 1)
            for column in range(first_column, last_column + 1)
        ]
        for key in keys:
            buckets.setdefault(key, set()).add(row)
        registered[row] = keys

    def release(row: int) -> None:
        keys = registered.pop(row, None)
        if keys is None:
            global_rows.discard(row)
        else:
            for key in keys:
                buckets[key].discard(row)
        slots[row] = None
        alive[row] = False
        free.append(row)

    def store(triangle: tuple[int, int, int]) -> None:
        nonlocal capacity, circles, alive
        if not free:
            slots.extend([None] * capacity)
            circles = np.vstack((circles, np.zeros((capacity, 6), dtype=np.float64)))
            alive = np.concatenate((alive, np.zeros(capacity, dtype=bool)))
            free.extend(range(2 * capacity - 1, capacity - 1, -1))
            capacity *= 2
        row = free.pop()
        slots[row] = triangle
        circle = _circumcircle(coords, triangle)
        circles[row] = circle
        alive[row] = True
        register(row, triangle, circle)

    store((count, count + 1, count + 2))
    insertion_order = sorted(range(count), key=lambda row: (float(points[row, 0]), float(points[row, 1]), row))
    for point_id in insertion_order:
        # Exactly the triangles a full scan with the adaptive ``incircle``
        # predicate would select: the grid and the conservative circumcircle
        # pre-filter only skip triangles the point is clearly outside of, and
        # every remaining candidate is still decided by ``incircle``.  The
        # cavity is a set, so neither the row order nor the triangle list order
        # affects the result.
        point = coords[point_id]
        nearby = buckets.get(row_of(float(point[1])) * columns + column_of(float(point[0])))
        rows = np.asarray(
            sorted(global_rows.union(nearby) if nearby else global_rows), dtype=np.int64
        )
        bad = [
            row
            for row in rows[_incircle_candidates(circles[rows], point)].tolist()
            if incircle(
                coords[slots[row][0]], coords[slots[row][1]], coords[slots[row][2]], point
            ) > 0.0
        ]
        if not bad:
            bad = [
                row
                for row in np.flatnonzero(alive).tolist()
                if all(
                    orient2d(coords[slots[row][index]], coords[slots[row][(index + 1) % 3]], point) >= 0.0
                    for index in range(3)
                )
            ]
        if not bad:
            raise MeshError("Delaunay insertion could not locate a point")
        edge_counts: dict[tuple[int, int], int] = {}
        for row in bad:
            triangle = slots[row]
            for index in range(3):
                edge = _normal_edge(triangle[index], triangle[(index + 1) % 3])
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        for row in bad:
            release(row)
        for first, second in sorted(edge for edge, frequency in edge_counts.items() if frequency == 1):
            determinant = orient2d(coords[first], coords[second], point)
            if determinant > 0.0:
                store((first, second, point_id))
            elif determinant < 0.0:
                store((second, first, point_id))
    triangles = [triangle for triangle in slots if triangle is not None]
    result = {
        _canonical_triangle(triangle, work)
        for triangle in triangles
        if all(node < count for node in triangle)
        and orient2d(coords[triangle[0]], coords[triangle[1]], coords[triangle[2]]) != 0.0
    }
    return sorted(result)


def _edge_incidence(triangles: Sequence[tuple[int, int, int]]) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for row, triangle in enumerate(triangles):
        for index in range(3):
            edge = _normal_edge(triangle[index], triangle[(index + 1) % 3])
            result.setdefault(edge, []).append(row)
    return result


def _convex_hull(points: np.ndarray, *, retain_collinear: bool = False,
                 cancellation_check: Callable[[str], None] | None = None) -> list[int]:
    """Return the finite hull in CCW order using the adaptive predicate."""
    order = sorted(range(len(points)), key=lambda node: (*points[node], node))
    chains: list[list[int]] = []
    for sequence in (order, reversed(order)):
        chain: list[int] = []
        for node in sequence:
            if cancellation_check is not None:
                cancellation_check("python triangulation convex hull")
            while len(chain) > 1:
                turn = orient2d(points[chain[-2]], points[chain[-1]], points[node])
                if turn > 0.0 or (retain_collinear and turn == 0.0):
                    break
                chain.pop()
            chain.append(node)
        chains.append(chain[:-1])
    return chains[0] + chains[1]


def _complete_hull_seed(points: np.ndarray, triangles: Sequence[tuple[int, int, int]],
                        cancellation_check: Callable[[str], None] | None = None) -> bool:
    """Certify a connected positive disk with every original hull station."""
    hull = _convex_hull(points, retain_collinear=True, cancellation_check=cancellation_check)
    incidence = _edge_incidence(triangles)
    if not triangles or len(set(hull)) != len(hull):
        return False
    expected = {_normal_edge(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))}
    if {edge for edge, rows in incidence.items() if len(rows) == 1} != expected:
        return False
    if any(len(rows) > 2 for rows in incidence.values()):
        return False
    if {node for triangle in triangles for node in triangle} != set(range(len(points))):
        return False
    if len(points) - len(incidence) + len(triangles) != 1:
        return False
    neighbors: list[list[int]] = [[] for _ in triangles]
    for rows in incidence.values():
        if len(rows) == 2:
            neighbors[rows[0]].append(rows[1])
            neighbors[rows[1]].append(rows[0])
    reached, pending = set(), [0]
    while pending:
        if cancellation_check is not None:
            cancellation_check("python triangulation hull validation")
        row = pending.pop()
        if row in reached:
            continue
        reached.add(row)
        a, b, c = triangles[row]
        if orient2d(points[a], points[b], points[c]) <= 0.0:
            return False
        pending.extend(neighbor for neighbor in neighbors[row] if neighbor not in reached)
    return len(reached) == len(triangles)


def _finite_hull_seed(points: np.ndarray,
                      cancellation_check: Callable[[str], None] | None = None) -> list[tuple[int, int, int]]:
    """Seed a complete finite disk when removal of the super triangle loses cells.

    A strict hull fan covers the domain before station insertion. Exact on-edge
    insertion splits both incident cells; Lawson flips subsequently restore
    Delaunay legality without changing any input coordinate or identifier.
    """
    hull = _convex_hull(points, cancellation_check=cancellation_check)
    if len(hull) < 3:
        raise MeshError("finite triangulation hull has zero area")
    triangles = [_canonical_triangle((hull[0], hull[i], hull[i + 1]), points)
                 for i in range(1, len(hull) - 1)]
    installed = set(hull)
    for node in sorted(set(range(len(points))) - installed, key=lambda index: (*points[index], index)):
        if cancellation_check is not None:
            cancellation_check("python triangulation finite hull insertion")
        containing = []
        edge = None
        for row, triangle in enumerate(triangles):
            turns = [orient2d(points[triangle[i]], points[triangle[(i + 1) % 3]], points[node])
                     for i in range(3)]
            if min(turns) >= 0.0:
                containing.append(row)
                for i, turn in enumerate(turns):
                    if turn == 0.0:
                        edge = _normal_edge(triangle[i], triangle[(i + 1) % 3])
                if edge is None:
                    break
        if not containing:
            raise MeshError("finite hull insertion could not locate a point")
        if edge is not None:
            containing = _edge_incidence(triangles)[edge]
            replacements = []
            for row in containing:
                opposite = next(vertex for vertex in triangles[row] if vertex not in edge)
                replacements.extend(((edge[0], node, opposite), (node, edge[1], opposite)))
        else:
            a, b, c = triangles[containing[0]]
            replacements = [(a, b, node), (b, c, node), (c, a, node)]
        removed = set(containing)
        triangles = [triangle for row, triangle in enumerate(triangles) if row not in removed]
        triangles.extend(_canonical_triangle(triangle, points) for triangle in replacements)
    # Strict positivity prevents cocircular flip cycles. The work bound is an
    # algorithmic convergence guard, not an operand-count restriction.
    limit = max(64, 32 * len(triangles) ** 2)
    for _ in range(limit):
        if cancellation_check is not None:
            cancellation_check("python triangulation finite hull legality")
        incidence = _edge_incidence(triangles)
        changed = False
        for edge, rows in sorted(incidence.items()):
            if cancellation_check is not None:
                cancellation_check("python triangulation finite hull edge")
            if len(rows) != 2:
                continue
            first, second = (triangles[row] for row in rows)
            a = next(node for node in first if node not in edge)
            b = next(node for node in second if node not in edge)
            if (_normal_edge(a, b) not in incidence
                    and _proper_intersection(points[edge[0]], points[edge[1]], points[a], points[b])
                    and incircle(*(points[node] for node in first), points[b]) > 0.0):
                triangles[rows[0]] = _canonical_triangle((a, b, edge[0]), points)
                triangles[rows[1]] = _canonical_triangle((b, a, edge[1]), points)
                changed = True
                break
        if not changed:
            result = sorted(triangles)
            if not _complete_hull_seed(points, result, cancellation_check):
                raise MeshError("finite hull triangulation is incomplete")
            return result
    raise MeshError("finite hull triangulation legality did not converge")


def _point_in_closed_triangle(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
) -> bool:
    return bool(
        orient2d(first, second, point) >= 0.0
        and orient2d(second, third, point) >= 0.0
        and orient2d(third, first, point) >= 0.0
    )


def _triangulate_cavity_chain(
    points: np.ndarray,
    raw_chain: Sequence[int],
) -> list[tuple[int, int, int]]:
    """Ear-clip one deterministic side of a recovered constraint cavity."""

    chain = [int(node) for node in raw_chain]
    if len(chain) < 3:
        raise MeshError("mandatory-segment cavity side has no area")
    area = _ring_area(points, chain)
    if area == 0.0:
        raise MeshError("mandatory-segment cavity side has zero area")
    if area < 0.0:
        chain.reverse()
    result: list[tuple[int, int, int]] = []
    while len(chain) > 3:
        clipped = False
        for position in range(len(chain)):
            previous = chain[(position - 1) % len(chain)]
            current = chain[position]
            following = chain[(position + 1) % len(chain)]
            if orient2d(points[previous], points[current], points[following]) <= 0.0:
                continue
            if any(
                _point_in_closed_triangle(
                    points[node],
                    points[previous],
                    points[current],
                    points[following],
                )
                for node in chain
                if node not in (previous, current, following)
            ):
                continue
            result.append(_canonical_triangle((previous, current, following), points))
            del chain[position]
            clipped = True
            break
        if not clipped:
            raise MeshError("mandatory-segment cavity is ambiguous")
    result.append(_canonical_triangle(chain, points))
    return result


def _recover_segment_by_cavity(
    points: np.ndarray,
    triangles: list[tuple[int, int, int]],
    target: tuple[int, int],
    protected: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Recover a stalled constraint without adding or moving point rows."""

    incidence = _edge_incidence(triangles)
    start, end = points[target[0]], points[target[1]]
    crossed = [
        edge
        for edge, attached in incidence.items()
        if len(attached) == 2
        and edge not in protected
        and _proper_intersection(start, end, points[edge[0]], points[edge[1]])
    ]
    removed = {row for edge in crossed for row in incidence[edge]}
    if not crossed or not removed:
        raise MeshError(f"could not recover mandatory segment {target}")
    if any(
        edge != target
        and len(incidence.get(edge, ())) == 2
        and all(row in removed for row in incidence[edge])
        for edge in protected
    ):
        raise MeshError("mandatory-segment cavity would remove a protected edge")

    cavity_counts: dict[tuple[int, int], int] = {}
    for row in sorted(removed):
        triangle = triangles[row]
        for index in range(3):
            edge = _normal_edge(int(triangle[index]), int(triangle[(index + 1) % 3]))
            cavity_counts[edge] = cavity_counts.get(edge, 0) + 1
    boundary_edges = {edge for edge, count in cavity_counts.items() if count == 1}
    adjacency: dict[int, set[int]] = {}
    for first, second in boundary_edges:
        adjacency.setdefault(first, set()).add(second)
        adjacency.setdefault(second, set()).add(first)
    if target[0] not in adjacency or target[1] not in adjacency:
        raise MeshError("mandatory-segment cavity does not contain both endpoints")
    if any(len(neighbors) != 2 for neighbors in adjacency.values()):
        raise MeshError("mandatory-segment cavity boundary is not a simple cycle")

    def trace(first_neighbor: int) -> list[int]:
        path = [target[0], first_neighbor]
        previous, current = target[0], first_neighbor
        visited = {target[0]}
        while current != target[1]:
            if current in visited:
                raise MeshError("mandatory-segment cavity path repeats a node")
            visited.add(current)
            choices = sorted(adjacency[current].difference((previous,)))
            if len(choices) != 1:
                raise MeshError("mandatory-segment cavity path is ambiguous")
            previous, current = current, choices[0]
            path.append(current)
        return path

    neighbors = sorted(adjacency[target[0]])
    first_chain = trace(neighbors[0])
    second_chain = trace(neighbors[1])
    if set(first_chain[1:-1]).intersection(second_chain[1:-1]):
        raise MeshError("mandatory-segment cavity sides overlap")
    traced_edges = {
        _normal_edge(chain[index], chain[index + 1])
        for chain in (first_chain, second_chain)
        for index in range(len(chain) - 1)
    }
    if traced_edges != boundary_edges:
        raise MeshError("mandatory-segment cavity traversal lost boundary edges")

    replacement = [triangle for row, triangle in enumerate(triangles) if row not in removed]
    replacement.extend(_triangulate_cavity_chain(points, first_chain))
    replacement.extend(_triangulate_cavity_chain(points, second_chain))
    recovered = sorted(set(replacement))
    recovered_incidence = _edge_incidence(recovered)
    if len(recovered_incidence.get(target, ())) != 2:
        raise MeshError(f"could not recover mandatory segment {target}")
    if any(edge not in recovered_incidence for edge in protected):
        raise MeshError("mandatory-segment cavity lost a protected edge")
    return recovered


def _recover_segment(
    points: np.ndarray,
    triangles: list[tuple[int, int, int]],
    segment: tuple[int, int],
    protected: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    target = _normal_edge(*segment)
    limit = max(64, 32 * len(triangles))
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for _ in range(limit):
        incidence = _edge_incidence(triangles)
        if target in incidence:
            return triangles
        state = tuple(triangles)
        if state in seen:
            break
        seen.add(state)
        crossings: list[tuple[float, tuple[int, int]]] = []
        start, end = points[target[0]], points[target[1]]
        for edge, attached in incidence.items():
            if len(attached) != 2 or edge in protected:
                continue
            if _proper_intersection(start, end, points[edge[0]], points[edge[1]]):
                parameter, _ = _intersection(start, end, points[edge[0]], points[edge[1]])
                crossings.append((parameter, edge))
        flipped = False
        for _, edge in sorted(crossings, key=lambda item: (item[0], item[1])):
            attached = incidence[edge]
            first_triangle, second_triangle = triangles[attached[0]], triangles[attached[1]]
            first_opposite = next(node for node in first_triangle if node not in edge)
            second_opposite = next(node for node in second_triangle if node not in edge)
            new_edge = _normal_edge(first_opposite, second_opposite)
            if new_edge in incidence and new_edge != target:
                continue
            if not _proper_intersection(
                points[edge[0]], points[edge[1]], points[first_opposite], points[second_opposite]
            ):
                continue
            first_new = _canonical_triangle((first_opposite, second_opposite, edge[0]), points)
            second_new = _canonical_triangle((second_opposite, first_opposite, edge[1]), points)
            if orient2d(points[first_new[0]], points[first_new[1]], points[first_new[2]]) <= 0.0:
                continue
            replacement = [
                triangle for row, triangle in enumerate(triangles) if row not in set(attached)
            ]
            replacement.extend((first_new, second_new))
            triangles = sorted(set(replacement))
            flipped = True
            break
        if not flipped:
            break
    return _recover_segment_by_cavity(points, triangles, target, protected)


def _domain_ring_edges(
    prepared: _PreparedPSLG,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, ...]:
    """``_ring_arrays`` of the outer ring and each hole, for ``_inside_domain``."""
    return tuple(
        _ring_arrays(prepared.points, ring) if len(ring) else None
        for ring in (prepared.outer, *prepared.holes)
    )


def _inside_domain(
    point: np.ndarray,
    prepared: _PreparedPSLG,
    ring_edges: tuple[tuple[np.ndarray, np.ndarray] | None, ...] | None = None,
) -> bool:
    """``ring_edges`` may carry ``_domain_ring_edges(prepared)``."""
    if ring_edges is None:
        ring_edges = (None,) * (1 + len(prepared.holes))
    if not _point_in_ring(point, prepared.points, prepared.outer, prepared.tolerance, ring_edges[0]):
        return False
    return not any(
        _point_in_ring(point, prepared.points, ring, prepared.tolerance, edges)
        for ring, edges in zip(prepared.holes, ring_edges[1:])
    )


def _finish_triangles(
    points: np.ndarray,
    triangles: Any,
    prepared: _PreparedPSLG,
    cancellation_check: Callable[[str], None] | None = None,
) -> np.ndarray:
    raw = np.asarray(triangles, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise MeshError("triangulation must contain three-node triangles")
    canonical: set[tuple[int, int, int]] = set()
    for triangle in raw:
        if cancellation_check is not None:
            cancellation_check("python triangulation domain cells")
        if np.any(triangle < 0) or np.any(triangle >= len(points)) or len(set(map(int, triangle))) != 3:
            raise MeshError("triangulation contains invalid connectivity")
        candidate = _canonical_triangle(triangle, points)
        if orient2d(points[candidate[0]], points[candidate[1]], points[candidate[2]]) <= 0.0:
            raise MeshError("triangulation contains a zero-area triangle")
        canonical.add(candidate)
    cells = sorted(canonical)
    incidence = _edge_incidence(cells)
    boundary = {tuple(map(int, edge)) for edge in prepared.boundary_segments}
    rejected: set[int] = set()
    pending: list[int] = []
    for ring in (prepared.outer, *prepared.holes):
        for first, second in zip(ring, np.roll(ring, -1)):
            if cancellation_check is not None:
                cancellation_check("python triangulation domain boundary")
            edge = _normal_edge(int(first), int(second))
            if edge not in incidence:
                raise MeshError("triangulation lost a domain boundary")
            for row in incidence[edge]:
                opposite = next(node for node in cells[row] if node not in edge)
                if orient2d(points[first], points[second], points[opposite]) < 0.0:
                    pending.append(row)
    while pending:
        if cancellation_check is not None:
            cancellation_check("python triangulation domain connectivity")
        row = pending.pop()
        if row in rejected:
            continue
        rejected.add(row)
        cell = cells[row]
        for index in range(3):
            edge = _normal_edge(cell[index], cell[(index + 1) % 3])
            if edge not in boundary:
                pending.extend(neighbor for neighbor in incidence[edge] if neighbor not in rejected)
    result = np.asarray([cell for row, cell in enumerate(cells) if row not in rejected], dtype=np.int64).reshape((-1, 3))
    edges = set(_edge_incidence([tuple(map(int, row)) for row in result]).keys())
    missing = [tuple(map(int, edge)) for edge in prepared.segments if tuple(map(int, edge)) not in edges]
    if missing:
        raise MeshError(f"triangulation lost mandatory segments: {missing[:3]}")
    return result


@dataclass(frozen=True)
class PlanarTriangulation:
    points: np.ndarray
    triangles: np.ndarray
    segments: np.ndarray
    boundary_segments: np.ndarray
    mandatory_segments: np.ndarray
    outer_loop: np.ndarray
    hole_loops: tuple[np.ndarray, ...]
    backend: str = "python"
    requested_backend: str = "python"
    selected_backend: str = "python"
    actual_backend: str = "python"
    fallback_reason: str | None = None
    native_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    protected_node_rows: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        for name in ("points", "triangles", "segments", "boundary_segments", "mandatory_segments", "outer_loop"):
            array = np.ascontiguousarray(getattr(self, name))
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        holes = tuple(np.ascontiguousarray(ring, dtype=np.int64) for ring in self.hole_loops)
        for ring in holes:
            ring.setflags(write=False)
        object.__setattr__(self, "hole_loops", holes)
        object.__setattr__(self, "native_diagnostics", dict(self.native_diagnostics))
        object.__setattr__(self, "protected_node_rows", tuple(self.protected_node_rows))

    @property
    def constraint_edges(self) -> np.ndarray:
        return self.mandatory_segments

    @property
    def edges(self) -> np.ndarray:
        values = {
            _normal_edge(int(triangle[index]), int(triangle[(index + 1) % 3]))
            for triangle in self.triangles
            for index in range(3)
        }
        result = np.asarray(sorted(values), dtype=np.int64).reshape((-1, 2))
        result.setflags(write=False)
        return result

    def __iter__(self):
        yield self.points
        yield self.triangles


TriangulationResult = PlanarTriangulation


def _strict_native_triangles(
    result_points: np.ndarray,
    triangles: np.ndarray,
    prepared: _PreparedPSLG,
) -> np.ndarray:
    if (
        result_points.dtype != np.dtype(np.float64)
        or not result_points.dtype.isnative
        or result_points.shape != prepared.points.shape
        or not result_points.flags.c_contiguous
        or result_points.tobytes(order="C") != prepared.points.tobytes(order="C")
    ):
        raise MeshError(
            "native triangulation changed prepared PSLG point rows or binary64 values"
        )
    if (
        triangles.dtype != np.dtype(np.int64)
        or not triangles.dtype.isnative
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not triangles.flags.c_contiguous
    ):
        raise MeshError(
            "native triangulation connectivity must be C-contiguous native int64 T3 rows"
        )
    if not len(triangles):
        raise MeshError("native triangulation returned no cells")

    ring_edges = _domain_ring_edges(prepared)
    canonical: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    incidence: dict[tuple[int, int], list[int]] = {}
    area = 0.0
    for row, raw in enumerate(triangles):
        made = tuple(int(value) for value in raw)
        if min(made) < 0 or max(made) >= len(result_points) or len(set(made)) != 3:
            raise MeshError("native triangulation returned invalid connectivity")
        candidate = _canonical_triangle(made, result_points)
        determinant = orient2d(
            result_points[candidate[0]],
            result_points[candidate[1]],
            result_points[candidate[2]],
        )
        if determinant <= 0.0:
            raise MeshError("native triangulation returned a zero-area cell")
        if candidate in seen:
            raise MeshError("native triangulation returned duplicate cells")
        seen.add(candidate)
        centroid = np.mean(result_points[np.asarray(candidate)], axis=0)
        if not _inside_domain(centroid, prepared, ring_edges):
            raise MeshError("native triangulation returned a cell outside the domain")
        canonical.append(candidate)
        area += 0.5 * determinant
        for local in range(3):
            edge = _normal_edge(candidate[local], candidate[(local + 1) % 3])
            incidence.setdefault(edge, []).append(row)

    if any(len(rows) > 2 for rows in incidence.values()):
        raise MeshError("native triangulation returned nonmanifold incidence")
    boundary = {tuple(map(int, edge)) for edge in prepared.boundary_segments}
    mandatory = {tuple(map(int, edge)) for edge in prepared.mandatory_segments}
    required = {tuple(map(int, edge)) for edge in prepared.segments}
    missing = sorted(required.difference(incidence))
    if missing:
        raise MeshError(f"native triangulation omitted mandatory segments: {missing[:5]}")
    wrong_boundary = sorted(
        edge for edge in boundary if len(incidence.get(edge, ())) != 1
    )
    if wrong_boundary:
        raise MeshError(
            f"native triangulation returned invalid boundary incidence: {wrong_boundary[:5]}"
        )
    open_interior = sorted(
        edge
        for edge, rows in incidence.items()
        if len(rows) == 1 and edge not in boundary
    )
    if open_interior:
        raise MeshError(
            f"native triangulation left open interior edges: {open_interior[:5]}"
        )
    if any(edge not in incidence for edge in mandatory):
        raise MeshError("native triangulation omitted a mandatory interior constraint")

    edges = sorted(incidence)
    records = sorted(
        (
            float(min(result_points[a, 0], result_points[b, 0])),
            float(max(result_points[a, 0], result_points[b, 0])),
            float(min(result_points[a, 1], result_points[b, 1])),
            float(max(result_points[a, 1], result_points[b, 1])),
            a,
            b,
        )
        for a, b in edges
    )
    active: list[tuple[float, float, float, float, int, int]] = []
    tolerance = prepared.tolerance
    for record in records:
        minimum_x, maximum_x, minimum_y, maximum_y, a, b = record
        active = [item for item in active if item[1] >= minimum_x - tolerance]
        for other in active:
            _, _, other_minimum_y, other_maximum_y, c, d = other
            if maximum_y < other_minimum_y - tolerance or other_maximum_y < minimum_y - tolerance:
                continue
            shared = {a, b}.intersection((c, d))
            first, second = result_points[a], result_points[b]
            third, fourth = result_points[c], result_points[d]
            crossing = _proper_intersection(first, second, third, fourth)
            touching = False
            if not shared:
                touching = any(
                    _point_on_segment(point, start, end, tolerance)
                    for point, start, end in (
                        (first, third, fourth),
                        (second, third, fourth),
                        (third, first, second),
                        (fourth, first, second),
                    )
                )
            if crossing or touching:
                raise MeshError(
                    f"native triangulation returned crossing or overlapping edges {(a, b)} and {(c, d)}"
                )
        active.append(record)

    expected_area = abs(_ring_area(result_points, prepared.outer)) - sum(
        abs(_ring_area(result_points, hole)) for hole in prepared.holes
    )
    area_tolerance = max(
        prepared.tolerance * max(1.0, expected_area) * max(16, len(boundary)),
        128.0 * np.finfo(float).eps * max(1.0, expected_area),
    )
    if abs(area - expected_area) > area_tolerance:
        raise MeshError(
            "native triangulation coverage area does not match the prepared domain"
        )
    return np.ascontiguousarray(sorted(canonical), dtype=np.int64)


def constrained_planar_triangulation(
    points: Any,
    outer: Sequence[int] | None = None,
    *,
    boundary: Sequence[int] | None = None,
    holes: Sequence[Sequence[int]] = (),
    constraints: Sequence[Sequence[int]] = (),
    mandatory_constraints: Sequence[Sequence[int]] | None = None,
    tolerance: float | None = None,
    backend: str | NativeBoundary | None = "auto",
    cancellation_check: Callable[[str], None] | None = None,
    protected_node_ids: Mapping[int, int] | None = None,
) -> PlanarTriangulation:
    """Triangulate a planar straight-line graph.

    ``points`` are never reordered.  ``outer``/``holes`` and ``constraints``
    contain point-row indices.  Crossing mandatory constraints are split at a
    newly appended point.  Domain boundaries and mandatory constraints are
    guaranteed to occur as output edges or the call raises ``MeshError``.
    """

    if outer is not None and boundary is not None:
        raise MeshError("provide outer or boundary, not both")
    if mandatory_constraints is not None:
        if constraints:
            raise MeshError("provide constraints or mandatory_constraints, not both")
        constraints = mandatory_constraints
    chosen = backend
    if chosen is None:
        chosen = "python"
    explicit_boundary: NativeBoundary | None = None
    if not isinstance(chosen, str):
        explicit_boundary = chosen
        chosen = "native"
    if chosen not in ("python", "native", "auto"):
        raise MeshError("backend must be 'python', 'native', 'auto', or a NativeBoundary")

    requested_backend = (
        str(getattr(explicit_boundary, "name", "native"))
        if explicit_boundary is not None
        else str(chosen)
    )
    selection: NativeBoundarySelection | None = None
    if chosen in ("native", "auto"):
        selection = snapshot_native_boundary(explicit_boundary)
    if chosen == "native" and selection is None:
        raise MeshError("no native triangulation boundary is registered")

    prepared = _prepare_pslg(
        points,
        outer if outer is not None else boundary,
        holes,
        constraints,
        tolerance,
        compiled_kernels=(
            selection is not None
            and selection.name == "anymesher-cpp17"
        ),
        protected_node_ids=protected_node_ids,
    )

    used_backend = "python"
    selected_backend = "python"
    fallback_reason = (
        "native_capability_absent"
        if chosen == "auto" and selection is None
        else None
    )
    native_diagnostics: Mapping[str, Any] = {}
    result_points = prepared.points
    result_triangles: np.ndarray | None = None
    if selection is not None:
        selected_backend = selection.name
        native_result = run_native_triangulation(
            prepared.points,
            prepared.segments,
            prepared.outer,
            prepared.holes,
            boundary=selection.boundary,
            cancellation_check=cancellation_check,
        )
        result_points = native_result.points
        validated = None
        if selection.name == "anymesher-cpp17":
            from .native_cpp import validate_native_triangulation

            validated = validate_native_triangulation(
                result_points,
                native_result.triangles,
                prepared.segments,
                prepared.boundary_segments,
                prepared.mandatory_segments,
                prepared.outer,
                prepared.holes,
                prepared.tolerance,
            )
        if validated is None:
            strict_triangles = _strict_native_triangles(
                result_points, native_result.triangles, prepared
            )
            result_triangles = _finish_triangles(
                result_points, strict_triangles, prepared, cancellation_check
            )
        else:
            result_triangles = validated
        used_backend = selection.name
        native_diagnostics = native_result.diagnostics

    if result_triangles is None:
        if cancellation_check is not None:
            cancellation_check("python triangulation insertion start")
        triangles = _bowyer_watson(prepared.points)
        if not _complete_hull_seed(prepared.points, triangles, cancellation_check):
            triangles = _finite_hull_seed(prepared.points, cancellation_check)
        protected: set[tuple[int, int]] = set()
        # ``_recover_segment`` returns the triangles unchanged when the segment
        # is already an edge; test that against one edge set, rebuilt only
        # after a recovery changed the triangulation, instead of rebuilding
        # the full incidence map for every segment.
        present = set(_edge_incidence(triangles))
        for raw_segment in prepared.segments:
            segment = tuple(map(int, raw_segment))
            if _normal_edge(*segment) not in present:
                triangles = _recover_segment(prepared.points, triangles, segment, protected)
                present = set(_edge_incidence(triangles))
            protected.add(_normal_edge(*segment))
        result_triangles = _finish_triangles(prepared.points, triangles, prepared, cancellation_check)
        if cancellation_check is not None:
            cancellation_check("python triangulation complete")

    if prepared.protected_node_rows and (
        result_points.shape != prepared.points.shape
        or result_points.tobytes(order="C") != prepared.points.tobytes(order="C")
    ):
        raise MeshError("triangulation changed protected station point rows or binary64 values")

    return PlanarTriangulation(
        points=result_points,
        triangles=result_triangles,
        segments=prepared.segments,
        boundary_segments=prepared.boundary_segments,
        mandatory_segments=prepared.mandatory_segments,
        outer_loop=prepared.outer,
        hole_loops=prepared.holes,
        backend=str(used_backend),
        requested_backend=requested_backend,
        selected_backend=selected_backend,
        actual_backend=str(used_backend),
        fallback_reason=fallback_reason,
        native_diagnostics=native_diagnostics,
        protected_node_rows=prepared.protected_node_rows,
    )


constrained_triangulate = constrained_planar_triangulation


def triangulate_polygon(
    outer: Any,
    holes: Sequence[Any] = (),
    constraints: Sequence[Any] = (),
    *,
    interior_points: Any | None = None,
    tolerance: float | None = None,
    backend: str | NativeBoundary | None = "auto",
    cancellation_check: Callable[[str], None] | None = None,
    protected_node_ids: Mapping[int, int] | None = None,
) -> PlanarTriangulation:
    """Coordinate-oriented wrapper around ``constrained_planar_triangulation``."""

    outer_points = np.asarray(outer, dtype=np.float64)
    if outer_points.ndim != 2 or outer_points.shape[1] != 2:
        raise MeshError("outer polygon must have shape (n, 2)")
    hole_points = [np.asarray(hole, dtype=np.float64) for hole in holes]
    if any(hole.ndim != 2 or hole.shape[1] != 2 for hole in hole_points):
        raise MeshError("every hole must have shape (n, 2)")
    constraint_points = [np.asarray(segment, dtype=np.float64) for segment in constraints]
    if any(segment.shape != (2, 2) for segment in constraint_points):
        raise MeshError("coordinate constraints must each have shape (2, 2)")
    extra = np.empty((0, 2), dtype=np.float64) if interior_points is None else np.asarray(interior_points, dtype=np.float64)
    if extra.ndim == 1 and extra.size == 0:
        extra = np.empty((0, 2), dtype=np.float64)
    if extra.ndim != 2 or extra.shape[1] != 2:
        raise MeshError("interior_points must have shape (n, 2)")

    blocks = [outer_points, *hole_points, extra, *(segment for segment in constraint_points)]
    points = np.vstack([block for block in blocks if len(block)])
    cursor = 0
    outer_ids = list(range(cursor, cursor + len(outer_points)))
    cursor += len(outer_points)
    hole_ids: list[list[int]] = []
    for hole in hole_points:
        hole_ids.append(list(range(cursor, cursor + len(hole))))
        cursor += len(hole)
    cursor += len(extra)
    constraint_ids: list[tuple[int, int]] = []
    for segment in constraint_points:
        constraint_ids.append((cursor, cursor + 1))
        cursor += 2
    return constrained_planar_triangulation(
        points,
        outer_ids,
        holes=hole_ids,
        constraints=constraint_ids,
        tolerance=tolerance,
        backend=backend,
        cancellation_check=cancellation_check,
        protected_node_ids=protected_node_ids,
    )
