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
    Arc,
    GeometryError,
    GeometryModel,
    Straight,
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


def _quarter_cylinder(radius: float = 2.0, height: float = 3.0):
    model = GeometryModel()
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    arc = model.add_arc(start, via, end)
    face = model.extrude([arc], (0.0, 0.0, height))[0]
    return model, arc, face


def _assert_no_degenerate_quads(mesh) -> None:
    for nodes in mesh.quads.values():
        corners = np.array([mesh.nodes[node] for node in nodes])
        area = 0.5 * (
            np.linalg.norm(
                np.cross(corners[1] - corners[0], corners[2] - corners[0])
            )
            + np.linalg.norm(
                np.cross(corners[2] - corners[0], corners[3] - corners[0])
            )
        )
        assert len(set(nodes)) == 4
        assert area > 1.0e-12


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


def test_splitting_an_edge_updates_faces_lengths_and_corners() -> None:
    model, face, _points, edges = _rectangle(4.0, 2.0)
    before = model.face_side_lengths(face)
    original = edges[0]

    _vertex, halves = model.split_edge(original, 0.3)

    assert len(model.faces[face].loop) == 5
    assert [len(side) for side in model.faces[face].sides()] == [2, 1, 1, 1]
    assert set(halves) <= set(model.edges)
    assert original not in model.edges
    assert model.face_side_lengths(face) == pytest.approx(before)

    other_model, other_face, _points, other_edges = _rectangle()
    other_model.split_edge(other_edges[2], 0.5)
    assert other_model.faces[other_face].corners == (0, 1, 2, 4)


def test_splitting_an_arc_keeps_both_halves_on_the_circle() -> None:
    radius = 2.0
    model, arc, _face = _quarter_cylinder(radius=radius)
    total = model.edge_length(arc)

    _vertex, (first, second) = model.split_edge(arc, 0.5)

    assert isinstance(model.edges[first].curve, Arc)
    assert isinstance(model.edges[second].curve, Arc)
    assert model.edge_length(first) + model.edge_length(second) == pytest.approx(total)
    for half in (first, second):
        points = model.sample_edge(half, np.linspace(0.0, 1.0, 9))
        assert np.linalg.norm(points, axis=1) == pytest.approx(radius)


def test_split_edge_requires_an_interior_parameter() -> None:
    model, _face, _points, edges = _rectangle()

    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(GeometryError, match="strictly between"):
            model.split_edge(edges[0], bad)


def test_face_corners_can_be_reassigned_and_are_validated() -> None:
    model, face, _points, edges = _rectangle()
    model.split_edge(edges[0], 0.5)

    model.set_face_corners(face, (0, 1, 2, 3))
    assert model.faces[face].corners == (0, 1, 2, 3)
    assert [len(side) for side in model.faces[face].sides()] == [1, 1, 1, 2]

    with pytest.raises(GeometryError, match="distinct"):
        model.set_face_corners(face, (0, 0, 1, 2))
    with pytest.raises(GeometryError, match="outside"):
        model.set_face_corners(face, (0, 1, 2, 9))


def test_mappability_report_points_at_sliver_geometry() -> None:
    model, face, _points, _edges = _rectangle()
    assert check_mappable(model, face).ok

    points = model.add_points([(0, 5, 0), (10, 5, 0), (10, 5.01, 0), (0, 5.2, 0)])
    sliver = model.add_face(model.add_polyline(points, close=True))
    report = check_mappable(model, sliver)

    assert not report.ok
    assert "splitting" in " ".join(report.messages)


def test_revolve_distinguishes_closed_and_partial_sweeps() -> None:
    model = GeometryModel()
    start = model.add_point(2.0, 0.0, 0.0)
    end = model.add_point(2.0, 0.0, 3.0)
    edge = model.add_line(start, end)

    faces = model.revolve([edge], (0, 0, 0), (0, 0, 1), 2.0 * np.pi)
    assert len(faces) == 4
    mesh = generate_mesh(model, target_size=0.5)
    positions = mesh.node_positions()
    assert np.linalg.norm(positions[:, :2], axis=1) == pytest.approx(2.0)
    assert len({tuple(np.round(point, 9)) for point in positions}) == mesh.num_nodes

    partial = GeometryModel()
    start = partial.add_point(2.0, 0.0, 0.0)
    end = partial.add_point(2.0, 0.0, 3.0)
    edge = partial.add_line(start, end)
    faces = partial.revolve([edge], (0, 0, 0), (0, 0, 1), np.pi)
    assert len(faces) == 2
    partial_positions = generate_mesh(partial, target_size=0.5).node_positions()
    angles = np.arctan2(
        partial_positions[:, 1],
        partial_positions[:, 0],
    )
    assert angles.min() == pytest.approx(0.0, abs=1.0e-9)
    assert angles.max() == pytest.approx(np.pi, abs=1.0e-9)


def test_revolve_validates_profile_axis_and_angle() -> None:
    model = GeometryModel()
    on_axis = model.add_point(0.0, 0.0, 0.0)
    off_axis = model.add_point(2.0, 0.0, 0.0)
    crossing = model.add_line(on_axis, off_axis)
    with pytest.raises(GeometryError, match="lies on the revolve axis"):
        model.revolve([crossing], (0, 0, 0), (0, 0, 1), np.pi / 2)

    profile = GeometryModel()
    start = profile.add_point(2.0, 0.0, 0.0)
    end = profile.add_point(2.0, 0.0, 1.0)
    edge = profile.add_line(start, end)
    with pytest.raises(GeometryError, match="non-zero"):
        profile.revolve([edge], (0, 0, 0), (0, 0, 0), np.pi)
    with pytest.raises(GeometryError, match="angle must be non-zero"):
        profile.revolve([edge], (0, 0, 0), (0, 0, 1), 0.0)


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
    assert isinstance(model.edges[new_edge].curve, Straight)
    assert model.edge_length(new_edge) == pytest.approx(1.0)
    assert sorted(model.face_side_lengths(right)) == pytest.approx(
        [1.0, 1.0, 1.0, 1.0]
    )
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


def test_curved_surface_operations_remain_exact_and_conformal() -> None:
    radius = 2.0
    model, _arc, face_id = _quarter_cylinder(radius=radius, height=3.0)
    face = model.faces[face_id]

    for u in (0.0, 0.25, 0.5, 1.0):
        for v in (0.0, 0.5, 1.0):
            point = surface_point(model, face, u, v)
            assert float(np.linalg.norm(point[:2])) == pytest.approx(radius)

    hoop, (first, second) = split_face_at(model, face_id, axis=1, fraction=0.5)
    assert isinstance(model.edges[hoop].curve, Arc)
    points = model.sample_edge(hoop, np.linspace(0.0, 1.0, 9))
    assert np.linalg.norm(points[:, :2], axis=1) == pytest.approx(radius)
    assert points[:, 2] == pytest.approx(np.full(9, 1.5))

    mesh = generate_mesh(model, target_size=0.4)
    assert np.linalg.norm(mesh.node_positions()[:, :2], axis=1) == pytest.approx(radius)
    shared = set(mesh.nodes_on(model.entity_ref("edge", hoop)))
    left = set(mesh.nodes_on(model.entity_ref("face", first)))
    right = set(mesh.nodes_on(model.entity_ref("face", second)))
    assert left & right == shared

    generator_model, _arc, generator_face = _quarter_cylinder(radius=radius)
    generator, _faces = split_face_at(
        generator_model, generator_face, axis=0, fraction=0.5
    )
    assert isinstance(generator_model.edges[generator].curve, Straight)


def test_face_split_validates_parameter_axis_and_side_pair() -> None:
    model, face, points, _edges = _rectangle()

    with pytest.raises(GeometryError, match="strictly between"):
        split_face_at(model, face, axis=0, fraction=0.0)
    with pytest.raises(GeometryError, match="axis must be"):
        split_face_at(model, face, axis=2, fraction=0.5)
    with pytest.raises(GeometryError, match="opposite sides"):
        split_face_between(model, face, points[0], points[1])


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

    for strip in strips:
        assert sorted(model.face_side_lengths(strip)) == pytest.approx(
            [1.0, 1.0, 1.0, 1.0]
        )
    for divider in dividers:
        assert len(model.faces_using_edge(divider)) == 2

    mesh = generate_mesh(model, target_size=0.25, beam_edges=dividers)
    beam_nodes = {node for pair in mesh.beams.values() for node in pair}
    plate_nodes = {
        node
        for strip in strips
        for node in mesh.nodes_on(model.entity_ref("face", strip))
    }
    assert beam_nodes
    assert beam_nodes <= plate_nodes


def test_stripping_needs_at_least_two_strips() -> None:
    model, face, _points, _edges = _rectangle(4.0, 1.0)

    with pytest.raises(GeometryError, match="at least 2"):
        strip_face(model, face, 0, 1)
    with pytest.raises(GeometryError, match="axis must be 0 or 1"):
        strip_face(model, face, 2, 3)


def test_stripping_a_cylinder_keeps_the_surface_exact() -> None:
    radius = 2.0
    model, _arc, face = _quarter_cylinder(radius=radius, height=4.0)

    strips, _dividers = strip_face(model, face, axis=1, count=4)

    assert len(strips) == 4
    mesh = generate_mesh(model, target_size=0.4)
    assert np.linalg.norm(mesh.node_positions()[:, :2], axis=1) == pytest.approx(radius)


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
    _assert_no_degenerate_quads(mesh)


def test_triangle_decomposition_validates_its_input_edges() -> None:
    rectangle, _face, _points, rectangle_edges = _rectangle()
    with pytest.raises(GeometryError, match="exactly three edges"):
        triangle_to_quads(rectangle, rectangle_edges)

    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)
    diagonal = model.add_line(points[2], points[0])
    with pytest.raises(GeometryError, match="already bounds a face"):
        triangle_to_quads(model, [edges[0], edges[1], diagonal])


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
    ring = np.vstack(
        [
            [mesh.nodes[node] for node in mesh.nodes_on(model.entity_ref("edge", arc))]
            for arc in arcs
        ]
    )
    assert np.linalg.norm(ring[:, :2] - np.array([1.0, 1.0]), axis=1) == pytest.approx(
        0.3
    )
    _assert_no_degenerate_quads(mesh)


def test_punching_a_hole_validates_radius_fit_and_planarity() -> None:
    model, face, _points, _edges = _rectangle(4.0, 3.0)
    with pytest.raises(GeometryError, match="does not fit"):
        punch_circular_hole(model, face, (2.0, 1.5, 0.0), 5.0)
    with pytest.raises(GeometryError, match="radius must be positive"):
        punch_circular_hole(model, face, (2.0, 1.5, 0.0), 0.0)

    curved, _arc, curved_face = _quarter_cylinder()
    with pytest.raises(GeometryError, match="not planar"):
        punch_circular_hole(curved, curved_face, (1.4, 1.4, 1.5), 0.2)


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
