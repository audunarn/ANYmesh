"""Original exterior stations retain current mesh IDs and coordinates."""

import numpy as np
import pytest

import anygeometry as owner
from anymesher._authored_boundary_binding import bind_authored_exterior_stations
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView


@pytest.fixture(scope="module")
def prepared_owner():
    model = owner.GeometryModel()
    vertices = model.add_points(((0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)))
    curve = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
    wall = model.extrude((curve,), (.25, 0, 1.5))[0]
    cutter = model.add_plate(model.add_points(((1.5, -3, -1), (1.5, 3, -1),
                                               (1.5, 3, 3), (1.5, -3, 3))))
    plan = owner.plan_intersections(model, (wall, cutter), policy="connect")
    owner.apply_intersections(model, plan, policy="connect")
    correspondence = owner.query_prepared_authored_boundary_correspondence(model, wall)
    edge = correspondence.exterior_loops[0][0][2][0]
    return model, correspondence, edge


def registered(prepared_owner):
    model, correspondence, edge = prepared_owner
    mesh = Mesh(geometry_model_id=model.model_id, geometry_revision=model.revision)
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    nodes = (101, 102, 103)
    parameters = (0., .5, 1.)
    positions = model.sample_edge(edge, np.asarray(parameters))
    mesh.nodes_of_edge[edge] = list(nodes)
    for node, point in zip(nodes, positions):
        mesh.nodes[node] = np.asarray(point, dtype=float).copy()
    registry.register_many(edge, parameters, points=positions, node_ids=nodes,
                           owner=model.handle("edge", edge))
    return model, correspondence, edge, mesh, registry, nodes


def test_existing_exterior_nodes_bind_original_uv_without_mutation(prepared_owner):
    model, correspondence, edge, mesh, registry, nodes = registered(prepared_owner)
    before = owner.to_dict(model)
    xyz = {node: mesh.nodes[node].copy() for node in nodes}
    result = bind_authored_exterior_stations(model, correspondence, mesh, registry, edge)
    assert result.node_ids == nodes
    assert result.parameters == (0., .5, 1.)
    assert len(result.authored_uv) == len(nodes)
    assert result.coordinates == tuple(tuple(xyz[node]) for node in nodes)
    assert result.publication_qualified is False
    assert owner.to_dict(model) == before
    for node in nodes:
        np.testing.assert_array_equal(mesh.nodes[node], xyz[node])


def test_changed_node_or_missing_registry_station_refuses(prepared_owner):
    model, correspondence, edge, mesh, registry, nodes = registered(prepared_owner)
    mesh.nodes[nodes[1]][0] += 1.e-5
    with pytest.raises(MeshError, match="XYZ changed"):
        bind_authored_exterior_stations(model, correspondence, mesh, registry, edge)
    model, correspondence, edge, mesh, registry, nodes = registered(prepared_owner)
    mesh.nodes_of_edge[edge].pop()
    with pytest.raises(MeshError, match="incomplete"):
        bind_authored_exterior_stations(model, correspondence, mesh, registry, edge)


def test_reversed_edge_sequence_retains_canonical_source_parameters(prepared_owner):
    model, correspondence, edge, mesh, registry, nodes = registered(prepared_owner)
    mesh.nodes_of_edge[edge].reverse()
    result = bind_authored_exterior_stations(model, correspondence, mesh, registry, edge)
    assert result.node_ids == nodes
    assert result.parameters == (0., .5, 1.)


def test_wrong_owner_and_stale_model_refuse(prepared_owner):
    model, correspondence, edge, mesh, registry, _ = registered(prepared_owner)
    other = model.clone(preserve_identity=False)
    with pytest.raises(MeshError, match="another geometry owner"):
        bind_authored_exterior_stations(other, correspondence, mesh, registry, edge)
    mesh.geometry_revision = model.revision - 1
    with pytest.raises(MeshError, match="stale"):
        bind_authored_exterior_stations(model, correspondence, mesh, registry, edge)
    mesh.geometry_revision = model.revision
    with pytest.raises(MeshError, match="positive source edge ID"):
        bind_authored_exterior_stations(model, correspondence, mesh, registry, True)
