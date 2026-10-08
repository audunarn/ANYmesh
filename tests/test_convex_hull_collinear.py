"""Exact boundary of a point set with exactly and nearly collinear stations.

``_convex_hull(..., retain_collinear=True)`` is the boundary used by the finite-hull
completeness check.  Stations lying exactly on a hull edge are boundary stations, and
the monotone chain alone dropped some of them.  The oracle below computes the boundary
in rational arithmetic from the stored floating coordinates: a point is on the boundary
exactly when it lies on a supporting line of the set, that is a line through two stations
with every station on one closed side.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from anymesher.triangulation import _convex_hull


def _exact_boundary(points) -> set[int]:
    exact = [(Fraction(float(x)), Fraction(float(y))) for x, y in points]
    count = len(exact)

    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    boundary: set[int] = set()
    for first in range(count):
        for second in range(first + 1, count):
            p, q = exact[first], exact[second]
            if p == q:
                continue
            signs = {side(p, q, exact[k]) for k in range(count) if k not in (first, second)}
            if not (all(value >= 0 for value in signs) or all(value <= 0 for value in signs)):
                continue
            on_line = [k for k in range(count) if side(p, q, exact[k]) == 0]
            boundary.update(on_line)
    return boundary


def _random_near_collinear(rng, count: int) -> np.ndarray:
    """Stations on a few lines, rounded into floats so that they are only nearly collinear."""

    points = []
    lines = [((0.0, 0.0), (1.0, 0.0)), ((1.0, 0.0), (0.3, 1.0)), ((0.3, 1.0), (0.0, 0.0))]
    for _ in range(count):
        if rng.random() < 0.6:
            (ax, ay), (bx, by) = lines[int(rng.integers(len(lines)))]
            t = float(rng.random())
            points.append((ax + t * (bx - ax), ay + t * (by - ay)))
        else:
            points.append((float(rng.random()), float(rng.random())))
    points = np.asarray(points, dtype=float)
    # Duplicate stations are excluded by the triangulator before the hull is used.
    _, keep = np.unique(points, axis=0, return_index=True)
    return points[np.sort(keep)]


@pytest.mark.parametrize("seed", range(40))
def test_retained_hull_is_the_exact_boundary_of_near_collinear_stations(seed):
    rng = np.random.default_rng(seed)
    points = _random_near_collinear(rng, count=int(rng.integers(5, 16)))
    hull = _convex_hull(points, retain_collinear=True)
    assert len(set(hull)) == len(hull)
    assert set(hull) == _exact_boundary(points)


def test_exactly_collinear_bottom_stations_are_all_retained():
    # Stations 0, 1, 2 and 4 are exactly collinear; station 3 lies 1e-17 above the chord.
    points = np.array([(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 1.0e-17),
                       (1.0, 0.0), (0.5, 1.0)])
    hull = _convex_hull(points, retain_collinear=True)
    assert set(hull) == {0, 1, 2, 4, 5}
    assert len(hull) == 5


def test_strict_hull_still_has_only_strict_vertices():
    points = np.array([(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    assert set(_convex_hull(points)) == {0, 2, 3, 4}
