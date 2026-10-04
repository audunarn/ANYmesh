"""Exact current linear associations and material coverage, without routing."""

from collections import defaultdict
from fractions import Fraction
import numpy as np
import pytest

from test_authored_component_binding import prepared
from anymesher._authored_component_binding import bind_authored_sheet_joint_component
from anymesher._authored_component_cells import validate_authored_component_cells
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView


def candidate():
    geometry, roots, correspondences, edge = prepared()
    component = bind_authored_sheet_joint_component(geometry, edge, correspondences, roots)
    mesh = Mesh(
        geometry_model_id=geometry.model_id, geometry_revision=geometry.revision,
        nodes={vid: np.asarray(geometry.vertex_position(vid), dtype=float)
               for vid in geometry.vertices},
    )
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(geometry))
    edges = {
        edge for correspondence in correspondences
        for loop in correspondence.exterior_loops
        for _source, _forward, current in loop for edge in current
    } | {edge for correspondence in correspondences
         for edge, _uses in correspondence.interior_incidence}
    for edge_id in sorted(edges):
        current = geometry.edges[edge_id]
        registry.register(edge_id, 0., node_id=current.start)
        registry.register(edge_id, 1., node_id=current.end)
        mesh.nodes_of_edge[edge_id] = [current.start, current.end]
    mesh.node_of_vertex = {
        row["id"]: row["id"] for row in component.owner_receipt.current_records["vertices"]
    }
    cell_faces = {}
    identifier = 100
    for face_id in component.current_face_ids:
        vertices = tuple(geometry.oriented_start_vertex(use)
                         for use in geometry.faces[face_id].loop)
        assert len(vertices) == 4
        pairs = ((vertices[0], vertices[1], vertices[2]),
                 (vertices[0], vertices[2], vertices[3]))
        mesh.elements_of_face[face_id] = []
        for tri in pairs:
            mesh.tris[identifier] = tri
            mesh.elements_of_face[face_id].append(identifier)
            cell_faces[identifier] = face_id
            identifier += 1
    uses = {row["id"]: row for row in component.owner_receipt.current_records["face_uses"]}
    sheet_cells = defaultdict(set)
    for sheet, _root, _original_use, current_uses in component.occurrence_correspondence:
        for use in current_uses:
            sheet_cells[sheet].update(mesh.elements_of_face[uses[use]["face_id"]])
    mesh.elements_of_sheet = {sheet: sorted(ids) for sheet, ids in sheet_cells.items()}
    mesh.declared_plate_junction_edges = tuple(sorted({
        (min(a, b), max(a, b))
        for joint in component.owner_receipt.joint_edge_ids
        for a, b in zip(mesh.nodes_of_edge[joint], mesh.nodes_of_edge[joint][1:])
    }))
    return geometry, component, mesh, registry, cell_faces


def prove(value):
    return validate_authored_component_cells(*value)


def test_full_current_component_buckets_and_exact_material_partition():
    geometry, component, mesh, registry, cells = value = candidate()
    before = {node: xyz.copy() for node, xyz in mesh.nodes.items()}
    evidence = prove(value)
    assert len(evidence.current_face_cells) == 8
    assert len(evidence.cell_current_faces) == 16
    assert len(evidence.sheet_cells) == 2
    assert len(evidence.face_use_cells) == 8
    assert len(evidence.joint_chains) == 1
    assert evidence.joint_chains[0][0] == component.joint_edge_id
    assert evidence.publication_qualified is False
    assert all(np.array_equal(mesh.nodes[node], xyz) for node, xyz in before.items())


def test_missing_current_cell_edge_vertex_and_joint_associations_refuse():
    value = candidate()
    value[2].elements_of_face.pop(next(iter(value[2].elements_of_face)))
    with pytest.raises(MeshError, match="face cell buckets"):
        prove(value)
    value = candidate()
    edge = value[1].joint_edge_id
    value[2].nodes_of_edge[edge].reverse()
    with pytest.raises(MeshError, match="station order"):
        prove(value)
    value = candidate()
    value[2].node_of_vertex.pop(next(iter(value[2].node_of_vertex)))
    with pytest.raises(MeshError, match="vertex nodes"):
        prove(value)
    value = candidate()
    value[2].declared_plate_junction_edges = ()
    with pytest.raises(MeshError, match="joint declaration"):
        prove(value)


def test_coincident_duplicate_joint_node_cannot_fake_shared_root_incidence():
    geometry, component, mesh, registry, cell_faces = candidate()
    joint = tuple(mesh.nodes_of_edge[component.joint_edge_id])
    second_root = set(component.boundary_correspondences[1].descendants)
    adjacent = [cell for cell, nodes in mesh.tris.items()
                if set(joint) <= set(nodes) and cell_faces[cell] in second_root]
    assert adjacent
    duplicate = max(mesh.nodes) + 1
    mesh.nodes[duplicate] = mesh.nodes[joint[0]].copy()
    for cell in adjacent:
        mesh.tris[cell] = tuple(duplicate if node == joint[0] else node
                                for node in mesh.tris[cell])
    with pytest.raises(MeshError, match="joint lacks shell incidence"):
        validate_authored_component_cells(
            geometry, component, mesh, registry, cell_faces,
        )


def test_unsupported_offset_or_altered_material_coordinate_refuses():
    value = candidate()
    value[2].offset_nodes_of_edge[value[1].joint_edge_id] = [1]
    with pytest.raises(MeshError, match="unsupported associations"):
        prove(value)
    value = candidate()
    node = value[2].nodes_of_edge[value[1].joint_edge_id][0]
    value[2].nodes[node] = value[2].nodes[node] + np.array((0., 0., .01))
    with pytest.raises(MeshError, match="vertex coordinate changed"):
        prove(value)


def created_candidate():
    from anygeometry import evaluate_prepared_authored_face
    from anymesher._authored_planar_stations import plan_authored_planar_stations

    geometry, component, mesh, registry, cell_faces = candidate()
    old_cell = min(mesh.tris)
    face = cell_faces.pop(old_cell)
    corners = mesh.tris.pop(old_cell)
    correspondence = next(item for item in component.boundary_correspondences
                          if face in item.descendants)
    source_uv = dict(plan_authored_planar_stations(
        geometry, correspondence, mesh, registry,
    ).node_material_uv)
    centre = tuple(sum(source_uv[node][axis] for node in corners) / 3
                   for axis in (0, 1))
    assert all(type(value) is Fraction for value in centre)
    new_node = max(mesh.nodes) + 1
    mesh.nodes[new_node] = evaluate_prepared_authored_face(
        geometry, correspondence, [[float(value) for value in centre]],
    )[0]
    new_cells = tuple(range(max(mesh.tris) + 1, max(mesh.tris) + 4))
    for identifier, triangle in zip(new_cells,
                                    ((corners[0], corners[1], new_node),
                                     (corners[1], corners[2], new_node),
                                     (corners[2], corners[0], new_node))):
        mesh.tris[identifier] = triangle
        cell_faces[identifier] = face
    mesh.elements_of_face[face] = sorted(
        (*[cell for cell in mesh.elements_of_face[face] if cell != old_cell],
         *new_cells)
    )
    for sheet, cells in mesh.elements_of_sheet.items():
        if old_cell in cells:
            mesh.elements_of_sheet[sheet] = sorted(
                (*[cell for cell in cells if cell != old_cell], *new_cells)
            )
    roots = {root: {} for root in component.authored_face_ids}
    roots[correspondence.authored_definition.face_id][new_node] = centre
    return geometry, component, mesh, registry, cell_faces, roots, new_node


def test_created_interior_node_requires_exact_root_uv_and_owner_xyz():
    geometry, component, mesh, registry, cell_faces, roots, new_node = created_candidate()
    validated = validate_authored_component_cells(
        geometry, component, mesh, registry, cell_faces,
        created_material_uv_by_root=roots,
    )
    assert len(validated.cell_current_faces) == 18
    with pytest.raises(MeshError, match="created UV"):
        validate_authored_component_cells(
            geometry, component, mesh, registry, cell_faces,
        )
    mesh.nodes[new_node] = mesh.nodes[new_node] + np.array((0., 0., .01))
    with pytest.raises(MeshError, match="left owner support"):
        validate_authored_component_cells(
            geometry, component, mesh, registry, cell_faces,
            created_material_uv_by_root=roots,
        )
