"""Deterministic constrained planar triangulation without optional geometry wheels."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from math import fsum
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


def _decimal(value: float) -> Decimal:
    return Decimal.from_float(float(value))


def orient2d(first: Sequence[float], second: Sequence[float], third: Sequence[float]) -> float:
    """Adaptive orientation predicate; positive means counter-clockwise."""

    ax = float(first[0]) - float(third[0])
    ay = float(first[1]) - float(third[1])
    bx = float(second[0]) - float(third[0])
    by = float(second[1]) - float(third[1])
    determinant = ax * by - ay * bx
    error = 8.0 * np.finfo(float).eps * (abs(ax * by) + abs(ay * bx))
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
    if abs(determinant) <= 32.0 * np.finfo(float).eps * scale:
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
    if abs(orient2d(first, second, point)) > tolerance * max(1.0, np.linalg.norm(second - first)):
        return False
    return bool(
        np.all(point >= np.minimum(first, second) - tolerance)
        and np.all(point <= np.maximum(first, second) + tolerance)
    )


def _point_in_ring(point: np.ndarray, points: np.ndarray, ring: Sequence[int], tolerance: float) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    for index, first_id in enumerate(ring):
        second_id = ring[(index + 1) % len(ring)]
        first, second = points[first_id], points[second_id]
        if _point_on_segment(point, first, second, tolerance):
            return True
        y1, y2 = float(first[1]), float(second[1])
        if (y1 > y) != (y2 > y):
            crossing = float(first[0]) + (y - y1) * float(second[0] - first[0]) / (y2 - y1)
            if crossing > x:
                inside = not inside
    return inside


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
) -> tuple[np.ndarray, list[int], list[list[int]], list[tuple[int, int]]]:
    unique: list[np.ndarray] = []
    remap = np.empty(len(points), dtype=np.int64)
    for index, point in enumerate(points):
        match = next(
            (row for row, candidate in enumerate(unique) if np.linalg.norm(point - candidate) <= tolerance),
            -1,
        )
        if match < 0:
            unique.append(np.array(point, copy=True))
            match = len(unique) - 1
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
    return np.asarray(unique, dtype=np.float64), mapped_outer, mapped_holes, mapped_constraints


def _ring_segments(ring: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(ring[index]), int(ring[(index + 1) % len(ring)])) for index in range(len(ring))]


def _validate_ring(points: np.ndarray, ring: Sequence[int], name: str) -> None:
    segments = _ring_segments(ring)
    for first_index, (a, b) in enumerate(segments):
        for second_index in range(first_index + 1, len(segments)):
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


def _prepare_pslg(
    raw_points: Any,
    raw_outer: Sequence[int] | None,
    raw_holes: Sequence[Sequence[int]],
    raw_constraints: Sequence[Sequence[int]],
    tolerance: float | None,
    *,
    compiled_kernels: bool = False,
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
    points, outer, holes, constraints = _deduplicate(
        points, outer, raw_holes, raw_constraints, epsilon
    )
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
    for first_index, (a, b, first_kind) in enumerate(records):
        for c, d, second_kind in records[first_index + 1:]:
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
    for record_index, (first, second, kind) in enumerate(records):
        start, end = points[first], points[second]
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if compiled_memberships is None:
            members: list[tuple[float, int]] = []
            for row, point in enumerate(points):
                if _point_on_segment(point, start, end, epsilon):
                    parameter = float(np.dot(point - start, direction) / denominator)
                    if -epsilon <= parameter <= 1.0 + epsilon:
                        members.append((min(1.0, max(0.0, parameter)), row))
            members.sort(key=lambda item: (item[0], item[1]))
            ordered = [row for _, row in members]
        else:
            ordered = list(compiled_memberships[record_index])
        for a, b in zip(ordered, ordered[1:]):
            if a != b:
                split_records.append((a, b, kind))

    boundary_set = {_normal_edge(a, b) for a, b, kind in split_records if kind == "boundary"}
    mandatory_set = {_normal_edge(a, b) for a, b, kind in split_records if kind == "mandatory"}
    all_segments = sorted(boundary_set | mandatory_set)

    compiled_domain = None
    if compiled_kernels:
        from .native_cpp import pslg_domain_classification

        compiled_domain = pslg_domain_classification(
            points,
            np.asarray(outer, dtype=np.int64),
            tuple(np.asarray(ring, dtype=np.int64) for ring in holes),
            epsilon,
        )
    for row, point in enumerate(points):
        if compiled_domain is None:
            on_outer = any(_point_on_segment(point, points[a], points[b], epsilon) for a, b in _ring_segments(outer))
            inside = _point_in_ring(point, points, outer, epsilon)
            in_hole = any(
                _point_in_ring(point, points, ring, epsilon)
                and not any(_point_on_segment(point, points[a], points[b], epsilon) for a, b in _ring_segments(ring))
                for ring in holes
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
    )


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
    triangles: list[tuple[int, int, int]] = [(count, count + 1, count + 2)]
    insertion_order = sorted(range(count), key=lambda row: (float(points[row, 0]), float(points[row, 1]), row))
    for point_id in insertion_order:
        bad = [
            row
            for row, triangle in enumerate(triangles)
            if incircle(work[triangle[0]], work[triangle[1]], work[triangle[2]], work[point_id]) > 0.0
        ]
        if not bad:
            bad = [
                row
                for row, triangle in enumerate(triangles)
                if all(
                    orient2d(work[triangle[index]], work[triangle[(index + 1) % 3]], work[point_id]) >= 0.0
                    for index in range(3)
                )
            ]
        if not bad:
            raise MeshError("Delaunay insertion could not locate a point")
        edge_counts: dict[tuple[int, int], int] = {}
        for row in bad:
            triangle = triangles[row]
            for index in range(3):
                edge = _normal_edge(triangle[index], triangle[(index + 1) % 3])
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        bad_set = set(bad)
        triangles = [triangle for row, triangle in enumerate(triangles) if row not in bad_set]
        for first, second in sorted(edge for edge, frequency in edge_counts.items() if frequency == 1):
            determinant = orient2d(work[first], work[second], work[point_id])
            if determinant > 0.0:
                triangles.append((first, second, point_id))
            elif determinant < 0.0:
                triangles.append((second, first, point_id))
        triangles.sort()
    result = {
        _canonical_triangle(triangle, work)
        for triangle in triangles
        if all(node < count for node in triangle)
        and orient2d(work[triangle[0]], work[triangle[1]], work[triangle[2]]) != 0.0
    }
    return sorted(result)


def _edge_incidence(triangles: Sequence[tuple[int, int, int]]) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for row, triangle in enumerate(triangles):
        for index in range(3):
            edge = _normal_edge(triangle[index], triangle[(index + 1) % 3])
            result.setdefault(edge, []).append(row)
    return result


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


def _inside_domain(point: np.ndarray, prepared: _PreparedPSLG) -> bool:
    if not _point_in_ring(point, prepared.points, prepared.outer, prepared.tolerance):
        return False
    return not any(
        _point_in_ring(point, prepared.points, ring, prepared.tolerance)
        for ring in prepared.holes
    )


def _finish_triangles(
    points: np.ndarray,
    triangles: Any,
    prepared: _PreparedPSLG,
) -> np.ndarray:
    raw = np.asarray(triangles, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 3:
        raise MeshError("triangulation must contain three-node triangles")
    canonical: set[tuple[int, int, int]] = set()
    for triangle in raw:
        if np.any(triangle < 0) or np.any(triangle >= len(points)) or len(set(map(int, triangle))) != 3:
            raise MeshError("triangulation contains invalid connectivity")
        candidate = _canonical_triangle(triangle, points)
        if orient2d(points[candidate[0]], points[candidate[1]], points[candidate[2]]) <= 0.0:
            raise MeshError("triangulation contains a zero-area triangle")
        centroid = np.mean(points[np.asarray(candidate)], axis=0)
        if _inside_domain(centroid, prepared):
            canonical.add(candidate)
    result = np.asarray(sorted(canonical), dtype=np.int64).reshape((-1, 3))
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
        if not _inside_domain(centroid, prepared):
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
                result_points, strict_triangles, prepared
            )
        else:
            result_triangles = validated
        used_backend = selection.name
        native_diagnostics = native_result.diagnostics

    if result_triangles is None:
        if cancellation_check is not None:
            cancellation_check("python triangulation insertion start")
        triangles = _bowyer_watson(prepared.points)
        protected: set[tuple[int, int]] = set()
        for raw_segment in prepared.segments:
            segment = tuple(map(int, raw_segment))
            triangles = _recover_segment(prepared.points, triangles, segment, protected)
            protected.add(_normal_edge(*segment))
        result_triangles = _finish_triangles(prepared.points, triangles, prepared)
        if cancellation_check is not None:
            cancellation_check("python triangulation complete")

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
    )
