"""Topology: loop ordering, corner detection, extrusion, and identity."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.geometry import GeometryError, GeometryModel


def unit_square(model: GeometryModel):
    points = model.add_points([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    return points, edges


def test_loop_is_ordered_from_unordered_edges():
    model = GeometryModel()
    points, edges = unit_square(model)
    # Hand the edges over shuffled and partly reversed.
    shuffled = [edges[2], edges[0], edges[3], edges[1]]
    face = model.add_face(shuffled)

    loop = model.faces[face].loop
    assert len(loop) == 4
    # Consecutive oriented edges must join end-to-start all the way round.
    for current, following in zip(loop, loop[1:] + loop[:1]):
        assert model.oriented_end_vertex(current) == model.oriented_start_vertex(
            following
        )


def test_open_chain_is_rejected():
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=False)

    with pytest.raises(GeometryError, match="closed"):
        model.add_face(edges)


def test_four_edge_face_uses_its_own_corners():
    model = GeometryModel()
    _, edges = unit_square(model)
    face = model.add_face(edges)
    assert model.faces[face].corners == (0, 1, 2, 3)
    assert all(len(model.faces[face].side(k)) == 1 for k in range(4))


def test_corner_detection_ignores_a_split_side():
    """A five-edge face is still mappable: the split side is one chain."""

    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (0.4, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)

    sides = model.faces[face].sides()
    assert model.faces[face].corners == (0, 2, 3, 4)
    assert len(sides[0]) == 2  # the split bottom
    assert [len(side) for side in sides[1:]] == [1, 1, 1]


def test_neutral_triangle_is_visible_as_not_mappable():
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (1, 0, 0), (0.5, 1, 0)])
    edges = model.add_polyline(points, close=True)

    face = model.add_face(edges)
    from anymesher import check_mappable

    report = check_mappable(model, face)
    assert not report.ok
    assert report.side_edge_counts == (0, 0, 0, 0)


def test_corner_override_must_be_in_loop_order():
    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (0.4, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    )
    edges = model.add_polyline(points, close=True)

    with pytest.raises(GeometryError, match="loop order"):
        model.add_face(edges, corners=(0, 3, 2, 4))


def test_extrusion_of_a_chain_shares_its_swept_lines():
    """Extruding a chain gives a conformal strip, not detached patches."""

    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    edges = model.add_polyline(points)
    faces = model.extrude(edges, (0.0, 0.0, 1.0))

    assert len(faces) == 2
    first = {item.edge for item in model.faces[faces[0]].loop}
    second = {item.edge for item in model.faces[faces[1]].loop}
    # The swept line at the shared point belongs to both faces.
    assert len(first & second) == 1


def test_extrusion_of_an_arc_gives_an_exact_cylinder():
    radius = 2.0
    model = GeometryModel()
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    arc = model.add_arc(start, via, end)
    faces = model.extrude([arc], (0.0, 0.0, 3.0))

    assert len(faces) == 1
    top = model.faces[faces[0]].loop[2].edge
    points = model.sample_edge(top, np.linspace(0.0, 1.0, 9))
    assert np.linalg.norm(points[:, :2], axis=1) == pytest.approx(radius)
    assert points[:, 2] == pytest.approx(np.full(9, 3.0))


def test_extrusion_rejects_a_zero_vector():
    model = GeometryModel()
    start = model.add_point(0, 0, 0)
    end = model.add_point(1, 0, 0)
    edge = model.add_line(start, end)

    with pytest.raises(GeometryError, match="non-zero"):
        model.extrude([edge], (0.0, 0.0, 0.0))


def test_ids_are_never_reused():
    model = GeometryModel()
    first = model.add_point(0, 0, 0)
    second = model.add_point(1, 0, 0)
    third = model.add_point(2, 0, 0)
    assert [first, second, third] == [1, 2, 3]
    # Identity is per kind, and independent of the other kinds.
    edge = model.add_line(first, second)
    assert edge == 1


def test_entity_ref_checks_existence():
    model = GeometryModel()
    vertex = model.add_point(0, 0, 0)
    assert model.entity_ref("vertex", vertex).id == vertex

    with pytest.raises(GeometryError, match="no face"):
        model.entity_ref("face", 99)
    with pytest.raises(GeometryError, match="unknown entity kind"):
        model.entity_ref("solid", 1)
