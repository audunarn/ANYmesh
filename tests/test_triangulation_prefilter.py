"""Triangulation speed-ups must not change results.

``_reference_bowyer_watson`` is the original full-scan implementation and
``_reference_point_in_ring`` the original scalar ring loop, both kept verbatim
as oracles.  The production ``_bowyer_watson`` only skips triangles whose
circumcircle provably excludes the inserted point, and the ring predicates are
vectorized with identical comparisons, so results must agree exactly,
including on cocircular lattices (incircle exactly zero), rings, tiny and
large coordinate scales, and needle triangles against the super-triangle.
"""
from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.triangulation import (
    _bowyer_watson,
    _canonical_triangle,
    _normal_edge,
    _point_in_ring,
    _point_on_ring,
    _point_on_segment,
    _ring_segments,
    incircle,
    orient2d,
)


def _reference_bowyer_watson(points: np.ndarray) -> list[tuple[int, int, int]]:
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


@lru_cache(maxsize=1)
def _cases():
    rng = random.Random(20260928)
    cases = []
    for k in range(120):
        n = rng.randint(3, 60)
        kind = k % 6
        scale = rng.choice([1e-9, 1e-3, 1.0, 1e3, 1e7])
        offset = rng.choice([0.0, 1.0, 1e4]) * scale
        if kind == 0:
            points = [(rng.uniform(0, 1), rng.uniform(0, 1)) for _ in range(n)]
        elif kind == 1:
            m = max(2, int(n ** 0.5))
            points = [(i, j) for i in range(m) for j in range(m)]
        elif kind == 2:
            points = [(i / n, 0.0) for i in range(n)] + [(0.5, 0.3), (0.2, -0.4)]
        elif kind == 3:
            m = max(2, int(n ** 0.5))
            points = [(i + rng.uniform(-1e-12, 1e-12), j) for i in range(m) for j in range(m)]
        elif kind == 4:
            points = [(np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)) for i in range(n)]
            points.append((0.1, 0.05))
        else:
            points = [(rng.gauss(0, 1e-3), rng.gauss(0, 1e-3)) for _ in range(n)]
            points += [(1.0, 1.0), (-1.0, 0.5)]
        cases.append(np.unique(np.asarray(points, dtype=float) * scale + offset, axis=0))
    # First counterexample for a fixed-slack filter: 75 cocircular points at
    # 1e-9 against the unit-scale super-triangle (needle circumcircles).
    ring = [(np.cos(2 * np.pi * i / 75), np.sin(2 * np.pi * i / 75)) for i in range(75)]
    cases.append(np.asarray(ring + [(0.1, 0.05)], dtype=float) * 1e-9)
    return tuple(cases)


def _outcome(function, points):
    try:
        return function(points)
    except MeshError as exc:  # both must fail identically, if at all
        return ("MeshError", str(exc))


@pytest.mark.parametrize("index", range(len(_cases())))
def test_prefiltered_bowyer_watson_matches_full_scan(index: int) -> None:
    points = _cases()[index]
    assert _outcome(_bowyer_watson, points) == _outcome(_reference_bowyer_watson, points)


def _reference_point_in_ring(point, points, ring, tolerance):
    """The original scalar even-odd loop, kept verbatim as an oracle."""
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


def test_vectorized_ring_predicates_match_scalar_loops() -> None:
    rng = random.Random(5)
    for case in range(120):
        count = rng.randint(3, 30)
        scale = rng.choice([1e-6, 1.0, 1e4])
        if case % 3 == 0:
            ring_points = [
                (np.cos(2 * np.pi * i / count) * (1 + 0.3 * rng.random()),
                 np.sin(2 * np.pi * i / count) * (1 + 0.3 * rng.random()))
                for i in range(count)
            ]
        elif case % 3 == 1:
            ring_points = [(i, 0) for i in range(count)] + [(count - 1, 1), (0, 1)]
        else:
            ring_points = [(0, 0), (4, 0), (4, 1.2), (2.2, 1.2), (2.2, 2.6), (0, 2.6)]
        ring_points = np.asarray(ring_points, dtype=float) * scale
        span = scale * (count if case % 3 == 1 else 1)
        queries = [(rng.uniform(-1.5, 1.5) * span, rng.uniform(-1.5, 3) * scale) for _ in range(20)]
        queries += [tuple(p) for p in ring_points]
        queries += [tuple((ring_points[i] + ring_points[(i + 1) % len(ring_points)]) / 2)
                    for i in range(len(ring_points))]
        points = np.vstack((ring_points, np.asarray(queries, dtype=float)))
        ring = list(range(len(ring_points)))
        for tolerance in (0.0, 1e-12 * scale, 1e-6 * scale):
            for row in range(len(ring_points), len(points)):
                point = points[row]
                assert _point_in_ring(point, points, ring, tolerance) == _reference_point_in_ring(
                    point, points, ring, tolerance)
                assert _point_on_ring(point, points, ring, tolerance) == any(
                    _point_on_segment(point, points[a], points[b], tolerance)
                    for a, b in _ring_segments(ring))
