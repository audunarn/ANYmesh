"""Owner-ordered authored-root inputs without starting a native mesher."""

import numpy as np
import pytest

import anygeometry as owner
from test_authored_component_cells import candidate
from anymesher._authored_route_boundary import plan_authored_component_boundaries
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
