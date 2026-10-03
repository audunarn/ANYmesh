"""Owner-ordered authored-root inputs without starting a native mesher."""

import numpy as np
import pytest

import anygeometry as owner
from test_authored_component_cells import candidate
from anymesher._authored_route_boundary import (
    plan_authored_component_boundaries, triangulate_authored_root_boundary,
)
from anymesher.errors import MeshError


def test_both_roots_keep_global_ids_and_exact_material_uv_separate():
    geometry, component, mesh, registry, _cells = candidate()
    before = owner.to_dict(geometry)
    coordinates = {node: xyz.copy() for node, xyz in mesh.nodes.items()}
    packets = plan_authored_component_boundaries(geometry, component, mesh, registry)
    assert tuple(packet.authored_face_id for packet in packets) == (1, 2)
    assert all(packet.publication_qualified is False for packet in packets)
    assert all(len(packet.outer_node_ids) == len(packet.outer_metric)
               for packet in packets)
    assert all(packet.constraint_node_pairs for packet in packets)
    joint_chain = mesh.nodes_of_edge[component.joint_edge_id]
    assert (joint_chain[0], joint_chain[1]) in packets[0].constraint_node_pairs
    assert (joint_chain[0], joint_chain[1]) in packets[1].constraint_node_pairs
    assert all(np.all(np.isfinite(packet.outer_metric)) for packet in packets)
    assert owner.to_dict(geometry) == before
    assert all(np.array_equal(mesh.nodes[node], xyz) for node, xyz in coordinates.items())


def test_missing_or_changed_protected_station_refuses():
    geometry, component, mesh, registry, _cells = candidate()
    joint = component.joint_edge_id
    node = mesh.nodes_of_edge[joint][0]
    mesh.nodes[node] = mesh.nodes[node] + np.array((0., 0., .01))
    with pytest.raises(owner.GeometryError):
        plan_authored_component_boundaries(geometry, component, mesh, registry)
    geometry, component, mesh, registry, _cells = candidate()
    geometry.add_point(20, 20, 20)
    with pytest.raises(MeshError, match="stale"):
        plan_authored_component_boundaries(geometry, component, mesh, registry)


def test_detached_triangulation_retains_both_roots_global_station_ids_and_uv():
    geometry, component, mesh, registry, _cells = candidate()
    packets = plan_authored_component_boundaries(geometry, component, mesh, registry)
    results = [triangulate_authored_root_boundary(packet) for packet in packets]
    assert len(results) == 2
    for packet, result in zip(packets, results):
        assert result.publication_qualified is False
        expected = dict(packet.material_uv_by_node)
        assert {node for node, _row in result.triangulation.protected_node_rows} == set(expected)
        assert len(result.original_uv_by_row) == len(result.triangulation.points)
        for node, row in result.triangulation.protected_node_rows:
            assert result.original_uv_by_row[row] == expected[node]
        assert result.triangulation.points.flags.writeable is False
    joint_ids = set(mesh.nodes_of_edge[component.joint_edge_id])
    assert joint_ids <= {node for node, _row in results[0].triangulation.protected_node_rows}
    assert joint_ids <= {node for node, _row in results[1].triangulation.protected_node_rows}


def test_detached_triangulation_refuses_stale_owner_binding():
    geometry, component, mesh, registry, _cells = candidate()
    packet = plan_authored_component_boundaries(geometry, component, mesh, registry)[0]
    geometry.add_point(20, 20, 20)
    with pytest.raises(MeshError, match="stale"):
        triangulate_authored_root_boundary(packet)


def test_detached_triangulation_refuses_changed_registry_or_node_xyz():
    geometry, component, mesh, registry, _cells = candidate()
    packet = plan_authored_component_boundaries(geometry, component, mesh, registry)[0]
    node = packet.outer_node_ids[0]
    mesh.nodes[node] = mesh.nodes[node] + np.array((0., 0., .01))
    with pytest.raises(MeshError, match="coordinates changed"):
        triangulate_authored_root_boundary(packet)
    geometry, component, mesh, registry, _cells = candidate()
    packet = plan_authored_component_boundaries(geometry, component, mesh, registry)[0]
    registry.register(packet._edge_ids[0], 0.12345)
    with pytest.raises(MeshError, match="registry changed"):
        triangulate_authored_root_boundary(packet)


def test_created_interior_row_keeps_original_chart_uv_provenance():
    geometry, component, mesh, registry, _cells = candidate()
    packet = plan_authored_component_boundaries(geometry, component, mesh, registry)[0]
    seed_uv = np.array(((0.231, 0.317),))
    result = triangulate_authored_root_boundary(
        packet, interior_metric=packet.chart.to_metric(seed_uv)
    )
    protected_rows = {row for _node, row in result.triangulation.protected_node_rows}
    created_rows = set(range(len(result.triangulation.points))) - protected_rows
    assert len(created_rows) == 1
    created = np.array([[float(v) for v in result.original_uv_by_row[row]]
                        for row in created_rows])
    np.testing.assert_allclose(created, seed_uv, rtol=0, atol=1e-14)
