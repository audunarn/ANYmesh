"""Exterior and physical interior stations retain owner original-UV truth."""

import numpy as np
import pytest

import anygeometry as owner
from anymesher._authored_planar_stations import plan_authored_planar_stations
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView


def prepared(*, cutter=False):
    model = owner.GeometryModel()
    root = model.add_plate(model.add_points(
        ((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))))
    model.set_face_surface(root, owner.Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    if cutter:
        model.add_plate(model.add_points(
            ((3, -1, -1), (3, 5, -1), (3, 5, 1), (3, -1, 1))))
    owner.apply_intersections(model,
        owner.plan_intersections(model, tuple(model.faces), policy="connect"),
        policy="connect")
    correspondence = owner.query_prepared_authored_boundary_correspondence(
        model, root,
    )
    mesh = Mesh(nodes={vid: np.array(model.vertex_position(vid), dtype=float)
                       for vid in model.vertices})
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    exterior = (edge_id for loop in correspondence.exterior_loops
                for _source, _forward, edges in loop for edge_id in edges)
    interior = (edge_id for edge_id, _uses in correspondence.interior_incidence)
    for edge_id in (*exterior, *interior):
        edge = model.edges[edge_id]
        registry.register(edge_id, 0., node_id=edge.start)
        registry.register(edge_id, 1., node_id=edge.end)
    return model, correspondence, mesh, registry


def test_exterior_node_ids_and_original_uv_are_retained_without_edits():
    model, correspondence, mesh, registry = prepared()
    before = owner.to_dict(model)
    plan = plan_authored_planar_stations(model, correspondence, mesh, registry)
    assert plan.authored_face == correspondence.authored_definition.face_id
    assert len(plan.exterior_receipts) == 4
    assert plan.interior_receipts == ()
    assert all(len(receipt.authored_uv) == 2 for receipt in plan.exterior_receipts)
    assert plan.vertex_preimages.scope.face_preimages == correspondence.face_preimages
    assert plan.publication_qualified is False
    assert owner.to_dict(model) == before


def test_changed_global_xyz_or_stale_binding_refuses():
    model, correspondence, mesh, registry = prepared()
    first = next(iter(mesh.nodes))
    mesh.nodes[first] = mesh.nodes[first] + np.array((0., 0., .01))
    with pytest.raises(owner.GeometryError):
        plan_authored_planar_stations(model, correspondence, mesh, registry)
    model, correspondence, mesh, registry = prepared()
    model.add_point(10, 10, 10)
    with pytest.raises(owner.GeometryError):
        plan_authored_planar_stations(model, correspondence, mesh, registry)


def test_physical_internal_edge_uses_owner_original_uv_station_contract():
    model, correspondence, mesh, registry = prepared(cutter=True)
    edge = correspondence.interior_incidence[0][0]
    before = owner.to_dict(model)
    plan = plan_authored_planar_stations(model, correspondence, mesh, registry)
    assert tuple(receipt.edge_id for receipt in plan.interior_receipts) == (edge,)
    receipt = plan.interior_receipts[0]
    assert receipt.endpoint_ids == (model.edges[edge].start, model.edges[edge].end)
    assert len(receipt.authored_uv) == len(registry.entries(edge))
    assert dict(plan.vertex_preimages.current_to_authored)[model.edges[edge].start] == ()
    assert plan.publication_qualified is False
    assert owner.to_dict(model) == before


def test_missing_or_changed_internal_stations_refuse():
    model, correspondence, mesh, registry = prepared(cutter=True)
    edge = correspondence.interior_incidence[0][0]
    incomplete = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    for entry in registry.entries():
        if entry.key.edge_id != edge:
            incomplete.register(entry.key.edge_id, entry.key.parameter,
                                node_id=entry.node_id)
    with pytest.raises(MeshError, match=f"edge {edge} lacks complete stations"):
        plan_authored_planar_stations(model, correspondence, mesh, incomplete)
    mesh.nodes[model.edges[edge].start] = mesh.nodes[model.edges[edge].start] + np.array((0., 0., .01))
    with pytest.raises(owner.GeometryError):
        plan_authored_planar_stations(model, correspondence, mesh, registry)
