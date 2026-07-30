"""Decomposition: turning regions that cannot be mapped into ones that can.

The property that matters for all of these is the same: after the operation, the
model contains only mappable faces and the mesher can actually mesh it.  A split
that produced a topologically valid but unmeshable model would be worse than one
that refused, because the failure would surface later and further away.

The ANYfem tests that also drive a project, a solve or a section stayed with
ANYfem; these exercise the operations through the geometry API alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import (
    GeometryError,
    GeometryModel,
    check_mappable,
    generate_mesh,
    punch_circular_hole,
    split_face_at,
    split_face_between,
    strip_face,
    surface_point,
    triangle_to_quads,
)


def _rectangle(length: float = 2.0, width: float = 1.0):
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (length, 0, 0), (length, width, 0), (0, width, 0)])
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)
    return model, face, points, edges


def _every_face_meshes(model: GeometryModel, target_size: float = 0.25) -> None:
    for face_id in model.faces:
        assert check_mappable(model, face_id).ok, check_mappable(model, face_id).messages
    mesh = generate_mesh(model, target_size=target_size)
    assert mesh.quads
    # Conformity: a node shared between faces is the same node object, so the
    # total is less than the sum of the parts wherever faces meet.
    assert mesh.num_nodes <= sum(
        len(mesh.nodes_on(model.entity_ref("face", face_id))) for face_id in model.faces
    )


def test_a_plain_rectangle_is_already_mappable() -> None:
    model, face, _points, _edges = _rectangle()
    report = check_mappable(model, face)

    assert report.ok
    assert report.edge_count == 4
    assert report.side_edge_counts == (1, 1, 1, 1)
    assert "mappable" in str(report)


def test_surface_point_follows_the_mesher_convention() -> None:
    model, face, _points, _edges = _rectangle(2.0, 1.0)
    face_object = model.faces[face]

    assert surface_point(model, face_object, 0.0, 0.0) == pytest.approx([0.0, 0.0, 0.0])
    assert surface_point(model, face_object, 1.0, 0.0) == pytest.approx([2.0, 0.0, 0.0])
    assert surface_point(model, face_object, 1.0, 1.0) == pytest.approx([2.0, 1.0, 0.0])
    assert surface_point(model, face_object, 0.5, 0.5) == pytest.approx([1.0, 0.5, 0.0])


def test_splitting_across_a_parametric_direction_leaves_two_mappable_faces() -> None:
    model, face, _points, _edges = _rectangle()
    new_edge, (left, right) = split_face_at(model, face, 0, 0.5)

    assert len(model.faces) == 2
    assert {left, right} <= set(model.faces)
    assert new_edge in model.edges
    _every_face_meshes(model)

    # The cut lands where it was asked to: the new edge is at x = 1.
    positions = model.sample_edge(new_edge, np.linspace(0.0, 1.0, 5))
    assert positions[:, 0] == pytest.approx(1.0)


def test_splitting_the_other_axis_cuts_the_other_way() -> None:
    model, face, _points, _edges = _rectangle()
    new_edge, _faces = split_face_at(model, face, 1, 0.25)

    positions = model.sample_edge(new_edge, np.linspace(0.0, 1.0, 5))
    assert positions[:, 1] == pytest.approx(0.25)
    _every_face_meshes(model)


def test_splitting_between_two_vertices_uses_them_as_they_are() -> None:
    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
    )
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)

    _new_edge, (first, second) = split_face_between(model, face, points[1], points[4])

    assert len(model.faces) == 2
    assert first != second
    _every_face_meshes(model)


def test_stripping_a_face_produces_the_requested_number_of_bands() -> None:
    model, face, _points, _edges = _rectangle(4.0, 1.0)
    strips, dividers = strip_face(model, face, 0, 4)

    assert len(strips) == 4
    # One fewer divider than strip, and each divider is shared by the plates
    # either side -- which is what makes a beam on it a stiffener coupled by
    # shared nodes rather than by an extra constraint.
    assert len(dividers) == 3
    assert len(model.faces) == 4
    _every_face_meshes(model, target_size=0.25)

    for divider in dividers:
        assert len(model.faces_using_edge(divider)) == 2


def test_stripping_needs_at_least_two_strips() -> None:
    model, face, _points, _edges = _rectangle(4.0, 1.0)

    with pytest.raises(GeometryError, match="at least 2"):
        strip_face(model, face, 0, 1)
    with pytest.raises(GeometryError, match="axis must be 0 or 1"):
        strip_face(model, face, 2, 3)


def test_a_triangle_becomes_three_mappable_quads() -> None:
    model = GeometryModel()
    edges = model.add_polyline(
        model.add_points([(0, 0, 0), (1, 0, 0), (0.4, 0.9, 0)]), close=True
    )
    faces = triangle_to_quads(model, edges)

    # Three real quadrilaterals rather than one degenerate quad with two
    # coincident corners, which is what a solver would otherwise be handed.
    assert len(faces) == 3
    assert len(model.faces) == 3
    _every_face_meshes(model, target_size=0.15)
    mesh = generate_mesh(model, target_size=0.15)
    for nodes in mesh.quads.values():
        assert len(set(nodes)) == 4


def test_punching_a_hole_leaves_a_meshable_ring() -> None:
    model, face, _points, _edges = _rectangle(2.0, 2.0)
    faces, arcs = punch_circular_hole(model, face, (1.0, 1.0, 0.0), 0.3)

    # The butterfly decomposition: four four-sided patches around four arcs, so
    # the hole edge itself stays addressable for a load or a support.
    assert len(faces) == 4
    assert len(arcs) == 4
    assert face not in model.faces
    _every_face_meshes(model, target_size=0.15)

    # No node ends up inside the hole.
    mesh = generate_mesh(model, target_size=0.15)
    radii = np.linalg.norm(mesh.node_positions()[:, :2] - np.array([1.0, 1.0]), axis=1)
    assert radii.min() == pytest.approx(0.3, rel=1.0e-6)


def test_a_face_with_too_few_edges_is_refused_at_construction() -> None:
    model = GeometryModel()
    edges = model.add_polyline(model.add_points([(0, 0, 0), (1, 0, 0), (0.4, 0.9, 0)]), close=True)

    # The geometry model enforces mappability when a face is made, so a
    # three-sided region never becomes an unmeshable face in the first place.
    with pytest.raises(GeometryError, match="at least four edges"):
        model.add_face(edges)


def test_operations_on_a_missing_face_say_which() -> None:
    model, _face, _points, _edges = _rectangle()

    with pytest.raises(GeometryError, match="no face 99"):
        check_mappable(model, 99)
    with pytest.raises(GeometryError, match="no face 99"):
        split_face_at(model, 99, 0, 0.5)
