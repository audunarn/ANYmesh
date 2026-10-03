"""Exterior station identity is owner-backed; physical interiors fail closed."""

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
    if not cutter:
        for loop in correspondence.exterior_loops:
            for _source, _forward, edges in loop:
                for edge_id in edges:
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
    assert all(len(receipt.authored_uv) == 2 for receipt in plan.exterior_receipts)
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


def test_physical_internal_edge_needs_owner_original_uv_station_contract():
    model, correspondence, mesh, registry = prepared(cutter=True)
    edge = correspondence.interior_incidence[0][0]
    with pytest.raises(MeshError, match=f"interior edge {edge} needs owner original-UV"):
        plan_authored_planar_stations(model, correspondence, mesh, registry)
