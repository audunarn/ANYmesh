"""Literal child membership is separate from complete mesh admission."""

import numpy as np
import pytest

import anygeometry as owner
from anygeometry.entities import EntityRef
from anyfem import Project
from anyfem.model.attributes import LineLoad
from anyfem.model.regions import ManualRegion, Region, RegionDomain, RegionRef
from anyfem.prepared_reference_scope import (
    PreparedReferenceScopeError, query_prepared_project_reference_scope,
)
from anymesher._authored_component_stage import AuthoredComponentPublication
from anymesher._authored_project_references import validate_authored_child_project_cells
from anymesher._authored_scope_binding import BoundAuthoredRootInputs
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView
from anymesher.native_v2 import ComponentSeedRegistry


def child_pressure_candidate():
    geometry = owner.GeometryModel()
    root = geometry.add_plate(geometry.add_points(
        ((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    geometry.set_face_surface(root, owner.Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    geometry.add_plate(geometry.add_points(
        ((3, -1, -1), (3, 5, -1), (3, 5, 1), (3, -1, 1))))
    owner.apply_intersections(geometry,
        owner.plan_intersections(geometry, tuple(geometry.faces), policy="connect"),
        policy="connect")
    correspondence = owner.query_prepared_authored_boundary_correspondence(
        geometry, root,
    )
    left, right = sorted(correspondence.descendants, key=lambda face: min(
        geometry.vertex_position(geometry.oriented_start_vertex(use))[0]
        for use in geometry.faces[face].loop))
    project = Project(geometry=geometry)
    region = Region("Left", RegionDomain.GEOMETRY, "face",
                    ManualRegion((EntityRef("face", left),)))
    project.regions.add(region)
    project.load_case().add_pressure(EntityRef("face", left), 1200,
                                     region=RegionRef(region.id))
    manifest = query_prepared_project_reference_scope(project, geometry)
    scope = owner.query_prepared_model_scope(geometry)
    bound = BoundAuthoredRootInputs(root, correspondence.descendants, scope)

    vertex_at = {
        tuple(float(value) for value in geometry.vertex_position(vid)[:2]): vid
        for vid in geometry.vertices
        if float(geometry.vertex_position(vid)[2]) == 0.
    }
    a, b, c, d, e, f = (vertex_at[xy] for xy in (
        (0., 0.), (3., 0.), (3., 4.), (0., 4.), (4., 0.), (4., 4.)))
    mesh = Mesh(nodes={vid: np.asarray(geometry.vertex_position(vid), dtype=float)
                       for vid in (a, b, c, d, e, f)},
                tris={101: (a, b, c), 102: (a, c, d),
                      103: (b, e, f), 104: (b, f, c)},
                elements_of_face={root: [101, 102, 103, 104]})
    uv = {vid: np.asarray(geometry.vertex_position(vid)[:2], dtype=float)
          for vid in (a, b, c, d, e, f)}
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(geometry))
    for edge_id in manifest.required_boundary_edge_ids:
        edge = geometry.edges[edge_id]
        registry.register(edge_id, 0., node_id=edge.start)
        registry.register(edge_id, 1., node_id=edge.end)
    return project, bound, manifest, correspondence, left, right, mesh, registry, uv


def prove(candidate, assigned=None):
    project, bound, manifest, correspondence, left, right, mesh, registry, uv = candidate
    return validate_authored_child_project_cells(
        project, project.geometry, bound, manifest, correspondence, mesh, registry,
        ({101: left, 102: left, 103: right, 104: right}
         if assigned is None else assigned), uv,
    )


def test_selected_left_cells_have_literal_owner_proof_without_mesh_admission():
    candidate = child_pressure_candidate()
    before = owner.to_dict(candidate[0].geometry)
    stations = tuple(candidate[7].entries())
    node_xyz = {node: point.copy() for node, point in candidate[6].nodes.items()}
    evidence = prove(candidate)
    assert evidence.selected_children == (candidate[4],)
    assert evidence.cell_children == (
        (101, candidate[4]), (102, candidate[4]),
        (103, candidate[5]), (104, candidate[5]))
    assert evidence.literal_partition_complete is True
    assert evidence.publication_qualified is False
    assert owner.to_dict(candidate[0].geometry) == before
    assert tuple(candidate[7].entries()) == stations
    assert all(np.array_equal(candidate[6].nodes[node], point)
               for node, point in node_xyz.items())


def test_crossing_or_wrong_child_refuses_even_inside_original_root():
    project, bound, manifest, correspondence, left, right, mesh, registry, uv = (
        child_pressure_candidate())
    with pytest.raises(MeshError, match="no declared shell cells"):
        validate_authored_child_project_cells(
            project, project.geometry, bound, manifest, correspondence, mesh,
            registry, {101: right, 102: right, 103: right, 104: right}, uv,
        )
    with pytest.raises(owner.GeometryError):
        validate_authored_child_project_cells(
            project, project.geometry, bound, manifest, correspondence, mesh,
            registry, {101: right, 102: left, 103: right, 104: right}, uv,
        )
    mesh.nodes.update({100: np.array((2., .5, 0.)),
                       110: np.array((3.5, .5, 0.)),
                       120: np.array((2., 1.5, 0.))})
    uv.update({node: xyz[:2] for node, xyz in mesh.nodes.items() if node in (100, 110, 120)})
    mesh.tris[105] = (100, 110, 120)
    mesh.elements_of_face[bound.authored_face] = [101, 102, 103, 104, 105]
    with pytest.raises(owner.GeometryError):
        validate_authored_child_project_cells(
            project, project.geometry, bound, manifest, correspondence, mesh,
            registry, {101: left, 102: left, 103: right, 104: right, 105: left}, uv,
        )


def test_changed_node_and_quadratic_request_refuse():
    candidate = child_pressure_candidate()
    mesh = candidate[6]
    first_node = next(iter(mesh.nodes))
    mesh.nodes[first_node] = mesh.nodes[first_node] + np.array((0., 0., .01))
    with pytest.raises(MeshError, match="does not support"):
        prove(candidate)
    candidate = child_pressure_candidate()
    candidate[6].order = "quadratic"
    with pytest.raises(MeshError, match="linear shell"):
        prove(candidate)


def test_missing_selected_child_boundary_stations_refuse():
    project, bound, manifest, correspondence, left, right, mesh, _registry, uv = (
        child_pressure_candidate())
    empty = GlobalEdgeBoundaryRegistry(GeometryMeshingView(project.geometry))
    with pytest.raises(MeshError, match="lacks complete stations"):
        validate_authored_child_project_cells(
            project, project.geometry, bound, manifest, correspondence, mesh,
            empty, {101: left, 102: left, 103: right, 104: right}, uv,
        )


def test_incomplete_duplicate_and_overlapping_child_partitions_refuse():
    candidate = child_pressure_candidate()
    mesh = candidate[6]
    mesh.tris.pop(103)
    mesh.tris.pop(104)
    mesh.elements_of_face[candidate[1].authored_face] = [101, 102]
    active = {node for connection in mesh.tris.values() for node in connection}
    for node in tuple(candidate[8]):
        if node not in active:
            candidate[8].pop(node)
    with pytest.raises(owner.GeometryError):
        prove(candidate, {101: candidate[4], 102: candidate[4]})

    candidate = child_pressure_candidate()
    mesh = candidate[6]
    mesh.tris.pop(104)
    mesh.elements_of_face[candidate[1].authored_face] = [101, 102, 103]
    active = {node for connection in mesh.tris.values() for node in connection}
    for node in tuple(candidate[8]):
        if node not in active:
            candidate[8].pop(node)
    with pytest.raises(owner.GeometryError):
        prove(candidate, {101: candidate[4], 102: candidate[4],
                          103: candidate[5]})

    candidate = child_pressure_candidate()
    mesh = candidate[6]
    mesh.tris[104] = mesh.tris[103]
    with pytest.raises(owner.GeometryError):
        prove(candidate)

    candidate = child_pressure_candidate()
    mesh = candidate[6]
    right_lower = mesh.tris[103]
    right_upper = mesh.tris[104]
    mesh.tris[104] = (right_lower[0], right_lower[1], right_upper[2])
    with pytest.raises(owner.GeometryError):
        prove(candidate)


def test_stale_project_during_private_publication_discards_stage():
    candidate = child_pressure_candidate()
    project, bound, manifest, correspondence, left, right, mesh, registry, uv = candidate
    holder = AuthoredComponentPublication(
        mesh, registry, ComponentSeedRegistry(max(mesh.nodes) + 1),
    )
    stage = holder.begin()
    first_edge = manifest.required_boundary_edge_ids[0]
    project.load_case().line_loads.append(
        LineLoad(EntityRef("edge", first_edge), np.array((0., 0., 1.))))

    def preflight_only(candidate_mesh, stations, _seeds):
        validate_authored_child_project_cells(
            project, project.geometry, bound, manifest, correspondence,
            candidate_mesh, stations,
            {101: left, 102: left, 103: right, 104: right}, uv,
        )
        return True

    with pytest.raises(PreparedReferenceScopeError):
        holder.publish(stage, preflight_only)
    assert holder.snapshot()[0] == 0
