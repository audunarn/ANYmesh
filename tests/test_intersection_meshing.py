"""Automatic working-copy imprint for crossing structural plates."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    EntityRef,
    GeometryModel,
    IntersectionKind,
    apply_imprint,
    fragment_coplanar_overlaps,
    plan_imprint,
    query_intersection,
)
from anygeometry.serialization import to_dict
from anymesher import generate_hybrid_mesh, generate_mesh_with_intersections
from anymesher.errors import MeshError
from anymesher.preparation import prepare_structural_closure
from anymesher.serialize import mesh_from_dict, mesh_to_dict


def _plate(geometry: GeometryModel, points) -> int:
    vertices = [geometry.add_point(*point) for point in points]
    return geometry.add_plate(vertices)


def test_crossing_planar_faces_are_imprinted_and_share_mesh_nodes():
    geometry = GeometryModel()
    horizontal = _plate(
        geometry,
        ((-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)),
    )
    vertical = _plate(
        geometry,
        ((0, 0, -1), (0, 2, -1), (0, 2, 1), (0, 0, 1)),
    )
    before = to_dict(geometry)

    mesh = generate_mesh_with_intersections(geometry, target_size=0.5)

    # Meshing works on a clone: design topology and IDs stay untouched.
    assert to_dict(geometry) == before
    assert mesh.automatic_intersections == 1
    assert set(mesh.elements_of_face) == {horizontal, vertical}

    horizontal_nodes = set(mesh.nodes_on(geometry.entity_ref("face", horizontal)))
    vertical_nodes = set(mesh.nodes_on(geometry.entity_ref("face", vertical)))
    shared = horizontal_nodes & vertical_nodes
    assert len(shared) == 5
    assert all(
        abs(float(mesh.nodes[node][0])) < 1.0e-12
        and abs(float(mesh.nodes[node][2])) < 1.0e-12
        for node in shared
    )

    horizontal_elements = set(mesh.elements_of_face[horizontal])
    vertical_elements = set(mesh.elements_of_face[vertical])
    for node in shared:
        assert any(node in mesh.shells[element] for element in horizontal_elements)
        assert any(node in mesh.shells[element] for element in vertical_elements)


def test_structured_crossing_plates_accept_only_declared_junction_edges():
    geometry = GeometryModel()
    horizontal = _plate(
        geometry,
        ((-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)),
    )
    vertical = _plate(
        geometry,
        ((0, 0, -1), (0, 2, -1), (0, 2, 1), (0, 0, 1)),
    )

    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.5,
        strategy="mapped",
    )

    assert mesh.automatic_intersections == 1
    assert mesh.declared_plate_junction_edges
    assert set(mesh.elements_of_face) == {horizontal, vertical}
    incidence: dict[tuple[int, int], list[int]] = {}
    for element_id, connectivity in mesh.shells.items():
        corners = mesh.corners_of(element_id)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge = (min(first, second), max(first, second))
            incidence.setdefault(edge, []).append(element_id)
    assert all(
        len(incidence[edge]) == 4
        for edge in mesh.declared_plate_junction_edges
    )

    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.declared_plate_junction_edges == mesh.declared_plate_junction_edges


def test_structured_t_junction_accepts_upstream_imprinted_transverse_edge():
    geometry = GeometryModel()
    lower_left = geometry.add_point(0.0, 0.0, 0.0)
    lower_right = geometry.add_point(2.0, 0.0, 0.0)
    upper_right = geometry.add_point(2.0, 2.0, 0.0)
    upper_left = geometry.add_point(0.0, 2.0, 0.0)
    geometry.add_plate((lower_left, lower_right, upper_right, upper_left))
    diagonal = geometry.add_line(lower_right, upper_left)
    geometry.extrude((diagonal,), (0.0, 0.0, 1.0))

    prepared, upstream_report = prepare_structural_closure(geometry)
    assert upstream_report.declared_face_connection_edges == (diagonal,)

    mesh = generate_hybrid_mesh(
        prepared,
        target_size=0.25,
        strategy="auto",
    )

    assert len(mesh.elements_of_face) == 3
    assert mesh.declared_plate_junction_edges
    incidence: dict[tuple[int, int], list[int]] = {}
    for element_id in mesh.shells:
        corners = mesh.corners_of(element_id)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge = (min(first, second), max(first, second))
            incidence.setdefault(edge, []).append(element_id)
    assert all(
        len(incidence[edge]) == 3
        for edge in mesh.declared_plate_junction_edges
    )


def test_intersection_diagnostic_survives_mesh_round_trip():
    geometry = GeometryModel()
    _plate(geometry, ((-1, 0, 0), (1, 0, 0), (1, 1, 0), (-1, 1, 0)))
    _plate(geometry, ((0, 0, -1), (0, 1, -1), (0, 1, 1), (0, 0, 1)))
    mesh = generate_mesh_with_intersections(geometry, target_size=0.5)

    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.automatic_intersections == 1
    assert (
        restored.automatic_beam_connections
        == mesh.automatic_beam_connections
    )
    assert (
        restored.automatic_shell_connections
        == mesh.automatic_shell_connections
    )
    np.testing.assert_allclose(restored.node_positions(), mesh.node_positions())


def test_multiple_crossings_are_discovered_after_the_first_face_is_fragmented():
    geometry = GeometryModel()
    horizontal = _plate(
        geometry,
        ((-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)),
    )
    verticals = (
        _plate(geometry, ((0, 0, -1), (0, 2, -1), (0, 2, 1), (0, 0, 1))),
        _plate(
            geometry,
            ((0.5, 0, -1), (0.5, 2, -1), (0.5, 2, 1), (0.5, 0, 1)),
        ),
    )

    mesh = generate_mesh_with_intersections(geometry, target_size=0.5)

    assert mesh.automatic_intersections == 2
    horizontal_nodes = set(
        mesh.nodes_on(geometry.entity_ref("face", horizontal))
    )
    assert [
        len(
            horizontal_nodes
            & set(mesh.nodes_on(geometry.entity_ref("face", vertical)))
        )
        for vertical in verticals
    ] == [5, 5]


def _beam(geometry: GeometryModel, start, end) -> int:
    first = geometry.add_point(*start)
    second = geometry.add_point(*end)
    return geometry.add_line(first, second)


def test_independently_drawn_coincident_beam_ends_share_a_mesh_node():
    geometry = GeometryModel()
    first = _beam(geometry, (0, 0, 0), (1, 0, 0))
    second = _beam(geometry, (1, 0, 0), (2, 0, 0))

    mesh = generate_mesh_with_intersections(
        geometry,
        target_size=0.5,
        beam_edges=(first, second),
    )

    shared = set(mesh.nodes_of_edge[first]) & set(mesh.nodes_of_edge[second])
    assert len(shared) == 1
    np.testing.assert_allclose(mesh.nodes[shared.pop()], (1, 0, 0))
    assert mesh.automatic_beam_connections >= 1


def test_beam_ending_on_another_beam_splits_and_connects_the_receiver():
    geometry = GeometryModel()
    receiver = _beam(geometry, (0, 0, 0), (2, 0, 0))
    branch = _beam(geometry, (1, 1, 0), (1, 0, 0))

    mesh = generate_mesh_with_intersections(
        geometry,
        target_size=0.5,
        beam_edges=(receiver, branch),
    )

    shared = set(mesh.nodes_of_edge[receiver]) & set(mesh.nodes_of_edge[branch])
    assert len(shared) == 1
    np.testing.assert_allclose(mesh.nodes[shared.pop()], (1, 0, 0))
    assert len(mesh.elements_of_edge[receiver]) == 4
    assert mesh.automatic_beam_connections >= 1


def test_crossing_beam_interiors_are_split_and_connected():
    geometry = GeometryModel()
    horizontal = _beam(geometry, (0, 0, 0), (2, 0, 0))
    vertical = _beam(geometry, (1, -1, 0), (1, 1, 0))

    mesh = generate_mesh_with_intersections(
        geometry,
        target_size=0.5,
        beam_edges=(horizontal, vertical),
    )

    shared = set(mesh.nodes_of_edge[horizontal]) & set(mesh.nodes_of_edge[vertical])
    assert len(shared) == 1
    np.testing.assert_allclose(mesh.nodes[shared.pop()], (1, 0, 0))
    assert len(mesh.elements_of_edge[horizontal]) == 4
    assert len(mesh.elements_of_edge[vertical]) == 4


def test_beam_crossing_shell_interior_is_split_and_interpolation_coupled():
    geometry = GeometryModel()
    plate = _plate(
        geometry,
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
    )
    beam = _beam(geometry, (0.3, 0.4, -1), (0.3, 0.4, 1))

    mesh = generate_mesh_with_intersections(
        geometry,
        target_size=0.5,
        beam_edges=(beam,),
    )

    crossing_nodes = [
        node
        for node in mesh.nodes_of_edge[beam]
        if np.linalg.norm(mesh.nodes[node] - np.asarray((0.3, 0.4, 0.0))) < 1.0e-10
    ]
    assert len(crossing_nodes) == 1
    coupling = next(
        item for item in mesh.couplings.values() if item.beam_node == crossing_nodes[0]
    )
    interpolated = sum(
        weight * mesh.nodes[node]
        for node, weight in zip(coupling.plate_nodes, coupling.weights)
    )
    np.testing.assert_allclose(interpolated, (0.3, 0.4, 0.0), atol=1.0e-10)
    assert set(coupling.plate_nodes).issubset(
        set(mesh.nodes_on(geometry.entity_ref("face", plate)))
    )
    assert mesh.automatic_beam_connections >= 1


def test_coplanar_beam_inside_shell_is_coupled_at_each_mesh_station():
    geometry = GeometryModel()
    _plate(
        geometry,
        ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)),
    )
    beam = _beam(geometry, (0.2, 0.4, 0), (1.8, 0.4, 0))

    mesh = generate_mesh_with_intersections(
        geometry,
        target_size=0.4,
        beam_edges=(beam,),
    )

    shell_nodes = {node for values in mesh.shells.values() for node in values}
    coupled = {item.beam_node for item in mesh.couplings.values()}
    assert set(mesh.nodes_of_edge[beam]).issubset(shell_nodes | coupled)
    assert mesh.automatic_beam_connections >= 1


def test_shell_boundary_terminating_inside_plate_gets_interpolation_tie():
    geometry = GeometryModel()
    support = _plate(
        geometry,
        ((0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)),
    )
    wall = _plate(
        geometry,
        ((1, 1, 0), (2, 1, 0), (2, 1, 1), (1, 1, 1)),
    )

    mesh = generate_mesh_with_intersections(geometry, target_size=0.5)

    assert mesh.automatic_intersections == 0
    assert mesh.automatic_shell_connections == 3
    support_nodes = set(mesh.nodes_on(geometry.entity_ref("face", support)))
    wall_nodes = set(mesh.nodes_on(geometry.entity_ref("face", wall)))
    attachments = [
        item for item in mesh.couplings.values() if item.beam_node in wall_nodes
    ]
    assert len(attachments) == 3
    assert all(set(item.plate_nodes).issubset(support_nodes) for item in attachments)


def test_two_adjacent_edges_extruded_separately_form_three_connected_plates():
    """Separate modelling commands duplicate the elevated corner topology.

    The geometric intersection is nevertheless an existing boundary, not an
    interior cut.  Meshing must couple it without attempting a zero-width
    imprint face.
    """

    geometry = GeometryModel()
    base = _plate(
        geometry,
        ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)),
    )
    edges = [item.edge for item in geometry.faces[base].loop]
    first_wall = geometry.extrude((edges[0],), (0, 0, 1))[0]
    second_wall = geometry.extrude((edges[1],), (0, 0, 1))[0]

    mesh = generate_mesh_with_intersections(geometry, target_size=0.25)

    assert mesh.automatic_intersections == 0
    assert set(mesh.elements_of_face) == {base, first_wall, second_wall}
    assert mesh.automatic_shell_connections >= 1
    dependency_pairs = {
        (coupling.beam_node, master)
        for coupling in mesh.couplings.values()
        for master, weight in zip(coupling.plate_nodes, coupling.weights)
        if abs(weight) > 1.0e-12
    }
    assert not any((master, slave) in dependency_pairs for slave, master in dependency_pairs)
    first_nodes = set(mesh.nodes_on(EntityRef("face", first_wall)))
    second_nodes = set(mesh.nodes_on(EntityRef("face", second_wall)))
    shared_positions = {
        tuple(np.round(mesh.nodes[node], 12))
        for node in first_nodes
    } & {
        tuple(np.round(mesh.nodes[node], 12))
        for node in second_nodes
    }
    assert (2.0, 0.0, 1.0) in shared_positions


def test_positive_area_coplanar_overlap_is_blocked_before_double_stiffness():
    geometry = GeometryModel()
    _plate(geometry, ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)))
    _plate(geometry, ((1, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0)))

    with pytest.raises(MeshError, match=r"positive-area.*overlap.*1 m\^2"):
        generate_mesh_with_intersections(geometry, target_size=0.25)

    result = fragment_coplanar_overlaps(geometry, tuple(geometry.faces))
    mesh = generate_mesh_with_intersections(geometry, target_size=0.25)
    assert len(result.outputs) == 3
    assert set(mesh.elements_of_face) == set(geometry.faces)
    assert mesh.automatic_intersections == 0


def test_hybrid_mesh_consumes_nonplanar_boundary_connect_by_shared_node_identity():
    geometry = GeometryModel()
    support = _plate(
        geometry,
        ((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)),
    )
    start, control, end = geometry.add_points(
        ((0.5, 0.5, 0.0), (1.5, 1.5, 0.0), (2.5, 0.5, 0.0))
    )
    spline = geometry.add_spline(start, (control,), end)
    wall = geometry.extrude((spline,), (0.0, 0.0, 1.0))[0]
    geometry.add_sheet((support,))
    geometry.add_sheet((wall,))
    result = query_intersection(
        geometry,
        geometry.handle("face", support),
        geometry.handle("face", wall),
    )
    application = apply_imprint(
        geometry,
        plan_imprint(geometry, result, policy="connect"),
        policy="connect",
    )
    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.25,
        strategy="auto",
        native_backend="python",
    )

    assert result.kind is IntersectionKind.CONTAINED
    assert application.face_intersection is not None
    assert application.face_intersection.edge.id == spline
    shared_nodes = set(mesh.nodes_on(EntityRef("edge", spline)))
    assert len(shared_nodes) >= 2
    incident_faces = {
        geometry.face_uses[face_use_id].face_id
        for face_use_id in geometry.face_uses_using_edge(spline)
    }
    assert len(incident_faces) >= 2
    assert all(
        shared_nodes <= set(mesh.nodes_on(EntityRef("face", face_id)))
        for face_id in incident_faces
    )
