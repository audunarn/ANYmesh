"""Mapped meshing: conformity, boundary fidelity, and association."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.geometry import GeometryModel
from anymesher.mapped import generate_mesh


def test_rectangle_mesh_shape_and_counts():
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)

    mesh = generate_mesh(model, target_size=0.25)
    assert mesh.grid_of_face[face].shape == (8 + 1, 4 + 1)
    assert len(mesh.quads) == 8 * 4
    assert mesh.num_nodes == 9 * 5


def test_shared_edge_nodes_are_the_same_nodes():
    """Conformity comes from shared topology, not tolerance matching."""

    model = GeometryModel()
    points = model.add_points(
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0)]
    )
    left_bottom = model.add_line(points[0], points[1])
    right_bottom = model.add_line(points[1], points[2])
    right = model.add_line(points[2], points[3])
    right_top = model.add_line(points[3], points[4])
    left_top = model.add_line(points[4], points[5])
    left = model.add_line(points[5], points[0])
    shared = model.add_line(points[1], points[4])

    first = model.add_face([left_bottom, shared, left_top, left])
    second = model.add_face([right_bottom, right, right_top, shared])

    mesh = generate_mesh(model, target_size=0.25)
    first_nodes = set(mesh.nodes_on(model.entity_ref("face", first)))
    second_nodes = set(mesh.nodes_on(model.entity_ref("face", second)))
    shared_nodes = set(mesh.nodes_on(model.entity_ref("edge", shared)))

    assert first_nodes & second_nodes == shared_nodes
    assert len(shared_nodes) == mesh.seeding[shared] + 1


def test_boundary_nodes_lie_exactly_on_their_curve():
    """The Coons blend must not move a boundary node off its edge."""

    radius = 2.0
    model = GeometryModel()
    start = model.add_point(radius, 0.0, 0.0)
    via = model.add_point(radius / np.sqrt(2), radius / np.sqrt(2), 0.0)
    end = model.add_point(0.0, radius, 0.0)
    arc = model.add_arc(start, via, end)
    model.extrude([arc], (0.0, 0.0, 3.0))

    mesh = generate_mesh(model, target_size=0.4)
    positions = mesh.node_positions()
    # Every node of a ruled cylindrical patch lies on the cylinder, interior
    # nodes included, because a Coons patch reduces to the ruled surface.
    assert np.linalg.norm(positions[:, :2], axis=1) == pytest.approx(radius)
    assert positions[:, 2].min() == pytest.approx(0.0)
    assert positions[:, 2].max() == pytest.approx(3.0)


def test_quads_are_not_degenerate():
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (4, 0, 0), (3, 2, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)

    mesh = generate_mesh(model, target_size=0.5)
    for nodes in mesh.quads.values():
        corners = np.array([mesh.nodes[node] for node in nodes])
        assert len(set(nodes)) == 4
        # Non-zero area, computed as two triangles.
        area = 0.5 * (
            np.linalg.norm(
                np.cross(corners[1] - corners[0], corners[2] - corners[0])
            )
            + np.linalg.norm(
                np.cross(corners[2] - corners[0], corners[3] - corners[0])
            )
        )
        assert area > 1.0e-12


def test_beam_edges_are_meshed_into_segments():
    model = GeometryModel()
    start = model.add_point(0, 0, 0)
    end = model.add_point(3, 0, 0)
    edge = model.add_line(start, end)

    mesh = generate_mesh(model, target_size=0.5, beam_edges=[edge])
    assert len(mesh.beams) == 6
    assert len(mesh.nodes_on(model.entity_ref("edge", edge))) == 7
    assert mesh.elements_on(model.entity_ref("edge", edge)) == sorted(mesh.beams)


def test_beam_on_a_plate_edge_shares_the_plate_nodes():
    """A stiffener on a plate boundary is coupled by sharing nodes."""

    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    face = model.add_face(edges)

    mesh = generate_mesh(model, target_size=0.25, beam_edges=[edges[0]])
    beam_nodes = {node for pair in mesh.beams.values() for node in pair}
    face_nodes = set(mesh.nodes_on(model.entity_ref("face", face)))
    assert beam_nodes <= face_nodes


def test_association_survives_a_remesh():
    """The same geometry reference resolves in every mesh of the model."""

    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)
    ref = model.entity_ref("edge", edges[0])

    coarse = generate_mesh(model, target_size=0.5)
    fine = generate_mesh(model, target_size=0.125)

    assert len(coarse.nodes_on(ref)) == 5
    assert len(fine.nodes_on(ref)) == 17
    # Both meshes still cover the whole line, end points included.
    for mesh in (coarse, fine):
        positions = np.array([mesh.nodes[node] for node in mesh.nodes_on(ref)])
        assert positions[:, 0].min() == pytest.approx(0.0)
        assert positions[:, 0].max() == pytest.approx(2.0)


def test_element_ids_are_unique_across_shells_and_beams():
    model = GeometryModel()
    points = model.add_points([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    edges = model.add_polyline(points, close=True)
    model.add_face(edges)

    mesh = generate_mesh(model, target_size=0.5, beam_edges=[edges[0]])
    assert not set(mesh.quads) & set(mesh.beams)
    assert mesh.num_elements == len(mesh.quads) + len(mesh.beams)
