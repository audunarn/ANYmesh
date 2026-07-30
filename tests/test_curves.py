"""Curve geometry: arcs must be circles, not approximations of them."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.geometry import GeometryModel
from anymesher.geometry.curves import DegenerateArcError


def test_straight_line_length_and_sampling():
    model = GeometryModel()
    start = model.add_point(1.0, 2.0, 3.0)
    end = model.add_point(4.0, 6.0, 3.0)
    edge = model.add_line(start, end)

    assert model.edge_length(edge) == pytest.approx(5.0)
    points = model.sample_edge(edge, np.linspace(0.0, 1.0, 5))
    assert points[0] == pytest.approx([1.0, 2.0, 3.0])
    assert points[-1] == pytest.approx([4.0, 6.0, 3.0])
    # Uniform parameter gives uniform spacing.
    spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert spacing == pytest.approx(np.full(4, 1.25))


def test_quarter_arc_is_exactly_circular():
    radius = 2.0
    model = GeometryModel()
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    edge = model.add_arc(start, via, end)

    assert model.edge_length(edge) == pytest.approx(0.5 * np.pi * radius)
    points = model.sample_edge(edge, np.linspace(0.0, 1.0, 17))
    assert np.linalg.norm(points, axis=1) == pytest.approx(radius)
    # Uniform parameter gives uniform arc length.
    spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert spacing == pytest.approx(np.full(16, spacing[0]))


def test_via_point_selects_the_major_arc():
    """An arc through a via point on the far side sweeps more than 180 deg."""

    radius = 1.0
    model = GeometryModel()
    start = model.add_point(radius, 0.0, 0.0)
    end = model.add_point(-radius, 0.0, 0.0)
    # Via below the axis: the arc must travel clockwise the long way round.
    via = model.add_point(0.0, -radius, 0.0)
    edge = model.add_arc(start, via, end)

    assert model.edge_length(edge) == pytest.approx(np.pi * radius)
    midpoint = model.sample_edge(edge, np.array([0.5]))[0]
    assert midpoint == pytest.approx([0.0, -radius, 0.0])


def test_arc_out_of_plane():
    """Arcs are not restricted to a global plane."""

    model = GeometryModel()
    start = model.add_point(1.0, 0.0, 0.0)
    via = model.add_point(0.0, 0.0, 1.0)
    end = model.add_point(-1.0, 0.0, 0.0)
    edge = model.add_arc(start, via, end)

    points = model.sample_edge(edge, np.linspace(0.0, 1.0, 9))
    assert np.linalg.norm(points, axis=1) == pytest.approx(1.0)
    assert points[:, 1] == pytest.approx(np.zeros(9))


def test_collinear_arc_is_rejected_at_modelling_time():
    model = GeometryModel()
    start = model.add_point(0.0, 0.0, 0.0)
    via = model.add_point(1.0, 0.0, 0.0)
    end = model.add_point(2.0, 0.0, 0.0)

    with pytest.raises(DegenerateArcError):
        model.add_arc(start, via, end)


def test_arc_follows_a_moved_point():
    """Curves hold no coordinates, so moving a point moves the curve."""

    model = GeometryModel()
    start = model.add_point(1.0, 0.0, 0.0)
    via = model.add_point(0.0, 1.0, 0.0)
    end = model.add_point(-1.0, 0.0, 0.0)
    edge = model.add_arc(start, via, end)
    assert model.edge_length(edge) == pytest.approx(np.pi)

    model.move_point(start, 2.0, 0.0, 0.0)
    model.move_point(via, 0.0, 2.0, 0.0)
    model.move_point(end, -2.0, 0.0, 0.0)
    assert model.edge_length(edge) == pytest.approx(2.0 * np.pi)


def test_edge_tangents():
    model = GeometryModel()
    start = model.add_point(1.0, 0.0, 0.0)
    via = model.add_point(0.0, 1.0, 0.0)
    end = model.add_point(-1.0, 0.0, 0.0)
    edge = model.add_arc(start, via, end)

    # Counter-clockwise quarter turns: tangent at the start points along +y.
    assert model.edge_tangent(edge, 0.0) == pytest.approx([0.0, 1.0, 0.0])
    assert model.edge_tangent(edge, 1.0) == pytest.approx([0.0, -1.0, 0.0])
