"""Owner-ordered authored-root inputs without starting a native mesher."""

from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest

import anygeometry as owner
from test_authored_component_cells import candidate
from anymesher._authored_route_boundary import (
    bind_authored_root_triangles_to_children, plan_authored_component_boundaries,
    refine_authored_root_with_ledger, triangulate_authored_root_boundary,
)
from anymesher._authored_work_ledger import AuthoredWorkLedger
from anymesher.errors import MeshError
from anymesher.native_v2 import NativeMeshingOptions


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
        correspondence = next(item for item in component.boundary_correspondences
                              if item.authored_definition.face_id == packet.authored_face_id)
        child_binding = bind_authored_root_triangles_to_children(
            packet, result, correspondence,
        )
        assert child_binding.publication_qualified is False
        assert len(child_binding.triangle_current_faces) == len(result.triangulation.triangles)
        assert set(child_binding.triangle_current_faces) == set(correspondence.descendants)
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


def test_child_binding_refuses_changed_uv_or_different_packet():
    geometry, component, mesh, registry, _cells = candidate()
    packets = plan_authored_component_boundaries(geometry, component, mesh, registry)
    result = triangulate_authored_root_boundary(packets[0])
    correspondence = component.boundary_correspondences[0]
    with pytest.raises(MeshError, match="mismatched inputs"):
        bind_authored_root_triangles_to_children(packets[1], result,
                                                component.boundary_correspondences[1])
    altered = ((Fraction(100), Fraction(100)), *result.original_uv_by_row[1:])
    with pytest.raises(MeshError, match="ambiguous current child"):
        bind_authored_root_triangles_to_children(
            packets[0], replace(result, original_uv_by_row=altered), correspondence,
        )


def test_detached_refinement_charges_only_original_remaining_allowance():
    geometry, component, mesh, registry, _cells = candidate()
    packet = plan_authored_component_boundaries(geometry, component, mesh, registry)[0]
    seed = triangulate_authored_root_boundary(packet)
    options = NativeMeshingOptions(point_placement="frontal_delaunay",
                                   max_insertions=8, max_topology_operations=2000)
    original = dict(selected_route="frontal_delaunay", cancelled=False,
                    insertion_budget=8, topology_budget=2000, insertions=1,
                    topology_operations=2, shared_segment_splits=0, shared_nodes=[])
    ledger = AuthoredWorkLedger.from_native_report(options, original)
    result, charged, report = refine_authored_root_with_ledger(
        packet, seed, ledger, options, target_size=2.0,
    )
    assert report["insertion_budget"] == ledger.remaining_insertions
    assert report["topology_budget"] == ledger.remaining_operations
    assert report["insertions"] > 0
    assert report["topology_operations"] > 0
    assert charged.further_insertions == report["insertions"]
    assert charged.further_operations == report["topology_operations"]
    assert result.triangulation.protected_node_rows == seed.triangulation.protected_node_rows
    assert len(result.original_uv_by_row) == len(result.triangulation.points)
    child_binding = bind_authored_root_triangles_to_children(
        packet, result, component.boundary_correspondences[0],
    )
    assert len(child_binding.triangle_current_faces) == len(result.triangulation.triangles)
    with pytest.raises(MeshError, match="original native budget binding"):
        refine_authored_root_with_ledger(
            packet, seed, ledger, replace(options, max_insertions=9), target_size=2.0,
        )
