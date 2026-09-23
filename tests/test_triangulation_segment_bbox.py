"""The cheap segment box must preserve the robust predicate's old decision."""

from math import hypot

import numpy as np
import pytest

from anymesher import triangulation


def _orientation_first(point, first, second, tolerance):
    a, b = float(first[0]), float(first[1])
    c, d = float(second[0]), float(second[1])
    x, y = float(point[0]), float(point[1])
    length = hypot(c - a, d - b)
    if abs(triangulation.orient2d(first, second, point)) > tolerance * max(1., length):
        return False
    return (
        min(a, c) - tolerance <= x <= max(a, c) + tolerance
        and min(b, d) - tolerance <= y <= max(b, d) + tolerance
    )


@pytest.mark.parametrize("first,second", (
    ((0., 0.), (2., 0.)),
    ((-3., 1.), (2., 4.)),
    ((1., 2.), (1., 2.)),
    ((1.e-9, -1.e-9), (2.e-9, 3.e-9)),
))
def test_segment_box_keeps_the_orientation_first_decision(first, second):
    start, end = np.asarray(first), np.asarray(second)
    rng = np.random.default_rng(1729)
    points = [start, end, (start + end) * .5]
    points.extend(rng.uniform(-4., 5., size=(300, 2)))
    points.extend((start + (end - start) * phase) + np.asarray((0., offset))
                  for phase in (-1.e-12, 0., .5, 1., 1. + 1.e-12)
                  for offset in (-1.e-12, 0., 1.e-12))
    for tolerance in (1.e-12, 1.e-9, 1.e-4):
        for point in points:
            point = np.asarray(point, dtype=np.float64)
            assert triangulation._point_on_segment(point, start, end, tolerance) == (
                _orientation_first(point, start, end, tolerance)
            )


def test_far_outside_box_never_calls_orientation(monkeypatch):
    calls = []
    original = triangulation.orient2d

    def counted(*points):
        calls.append(points)
        return original(*points)

    monkeypatch.setattr(triangulation, "orient2d", counted)
    assert not triangulation._point_on_segment(
        np.asarray((100., 100.)), np.asarray((0., 0.)),
        np.asarray((1., 1.)), 1.e-12,
    )
    assert not calls
