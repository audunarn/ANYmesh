"""Differential regression for the incremental finite-hull legalization.

The production loop maintains edge incidence incrementally and re-checks only
the edges a flip can affect. This file keeps a verbatim copy of the previous
full-rescan legalization as an independent reference and requires identical
triangulations, identical cancellation phase labels, and unchanged inputs.
"""
from __future__ import annotations

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.triangulation import (
    _canonical_triangle,
    _complete_hull_seed,
    _convex_hull,
    _edge_incidence,
    _finite_hull_seed,
    _normal_edge,
    _proper_intersection,
    incircle,
    orient2d,
)


def _reference_finite_hull_seed(points, cancellation_check=None):
    """The original full-rescan legalization, preserved verbatim."""
    hull = _convex_hull(points, cancellation_check=cancellation_check)
    if len(hull) < 3:
        raise MeshError("finite triangulation hull has zero area")
    triangles = [_canonical_triangle((hull[0], hull[i], hull[i + 1]), points)
                 for i in range(1, len(hull) - 1)]
    installed = set(hull)
    for node in sorted(set(range(len(points))) - installed,
                       key=lambda index: (*points[index], index)):
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


def _compare(points):
    points = np.asarray(points, dtype=float)
    original = points.copy()
    reference_phases: list[str] = []
    actual_phases: list[str] = []
    expected = _reference_finite_hull_seed(points, reference_phases.append)
    actual = _finite_hull_seed(points, actual_phases.append)
    assert actual == expected
    assert set(actual_phases) == set(reference_phases)
    np.testing.assert_array_equal(points, original)
    assert _complete_hull_seed(points, actual)
    return len(reference_phases), len(actual_phases)


def test_incremental_legality_matches_rescan_on_random_points():
    rng = np.random.default_rng(20261008)
    for size in (4, 9, 16, 25, 40, 60):
        _compare(rng.random((size, 2)))


def test_incremental_legality_matches_rescan_on_structured_points():
    _compare([(x, y) for x in range(5) for y in range(5)])
    _compare([(x, y) for x in range(7) for y in range(7)])
    angle = 0.37
    rotation = np.asarray(((np.cos(angle), -np.sin(angle)),
                          (np.sin(angle), np.cos(angle))))
    grid = np.asarray([(x, y) for x in range(6) for y in range(6)]) @ rotation.T
    _compare(grid)


def test_incremental_legality_matches_rescan_on_edge_stations():
    # Stations exactly on prior edges exercise the on-edge insertion split
    # and the alternate-edge presence rule during legalization.
    _compare([
        (0., 0.), (4., 0.), (4., 4.), (0., 4.),
        (2., 0.), (4., 2.), (2., 4.), (0., 2.),
        (2., 2.), (1., 1.), (3., 3.), (1., 3.), (3., 1.),
        (2., 1.), (2., 3.),
    ])
    _compare([
        (0., 0.), (3., 0.), (3., 3.), (0., 3.),
        (1.5, 0.), (3., 1.5), (1.5, 3.), (0., 1.5),
        (0.75, 0.75), (2.25, 2.25), (1., 2.), (2., 1.),
    ])


def test_incremental_legality_matches_rescan_on_collinear_hull():
    _compare([
        (0., 0.), (1., 0.), (2., 0.), (3., 0.), (4., 0.),
        (4., 2.), (2., 1.), (0., 2.), (1., 1.), (3., 2.),
    ])


def test_incremental_legality_avoids_rescanning_every_edge():
    rng = np.random.default_rng(20261008)
    points = rng.random((60, 2))
    reference_count, actual_count = _compare(points)
    # The rescan loop re-examines every interior edge once per flip; the
    # incremental loop only re-checks edges a flip can affect.
    assert actual_count < reference_count / 4


@pytest.mark.parametrize("phase", (
    "python triangulation convex hull",
    "python triangulation finite hull insertion",
    "python triangulation finite hull edge",
    "python triangulation finite hull legality",
    "python triangulation hull validation",
))
def test_incremental_legality_propagates_cancellation(phase):
    points = np.random.default_rng(517).random((30, 2))
    original = points.copy()
    class Cancelled(Exception):
        pass
    def cancel(current):
        if current == phase:
            raise Cancelled(phase)
    with pytest.raises(Cancelled, match=phase):
        _finite_hull_seed(points, cancel)
    np.testing.assert_array_equal(points, original)
