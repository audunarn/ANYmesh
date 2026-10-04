"""Detached two-root assembly preserves registry identity and owner proof."""

import numpy as np
import pytest

from test_authored_component_cells import candidate
from anymesher._authored_route_boundary import (
    bind_authored_root_triangles_to_children,
    plan_authored_component_boundaries,
    triangulate_authored_root_boundary,
)
from anymesher._authored_staged_pair import (
    refine_authored_pair_with_one_ledger, stage_authored_root_pair,
)
from anymesher._authored_component_stage import _mesh_digest, _registry_receipt
from anymesher._authored_work_ledger import AuthoredWorkLedger
from anymesher.errors import MeshError
from anymesher.native_v2 import ComponentSeedRegistry, NativeMeshingOptions
from anymesher.prepared_current_associations import (
    PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA,
    PREPARED_CURRENT_ASSOCIATIONS_SCHEMA,
)


def inputs():
    geometry, component, mesh, registry, _cells = candidate()
    packets = plan_authored_component_boundaries(geometry, component, mesh, registry)
    roots = tuple(triangulate_authored_root_boundary(packet) for packet in packets)
    children = tuple(bind_authored_root_triangles_to_children(packet, result, correspondence)
                     for packet, result, correspondence in zip(
                         packets, roots, component.boundary_correspondences))
    seeds = ComponentSeedRegistry(max(mesh.nodes) + 1)
    return geometry, component, mesh, registry, seeds, packets, roots, children


def test_pair_stages_exact_mesh_without_changing_source():
    args = inputs()
    source, registry, seeds = args[2:5]
    before = (_mesh_digest(source), _registry_receipt(registry), seeds.committed_snapshot())
    result = stage_authored_root_pair(*args)
    assert result.publication_qualified is False
    assert result.solver_admitted is False
    assert result.current_receipt.schema == PREPARED_CURRENT_ASSOCIATIONS_SCHEMA
    assert len(result.mesh.tris) == 16
    assert set(result.core.node_ids) == set(source.nodes)
    assert result.created_material_uv_by_root == ((1, ()), (2, ()))
    for node in source.nodes:
        np.testing.assert_array_equal(result.mesh.nodes[node], source.nodes[node])
    assert result.mesh.nodes_of_edge == source.nodes_of_edge
    assert result.mesh.elements_of_sheet.keys() == source.elements_of_sheet.keys()
    assert (_mesh_digest(source), _registry_receipt(registry), seeds.committed_snapshot()) == before


def test_pair_refuses_cancel_stale_and_mismatched_child_without_source_change():
    args = inputs()
    source, registry, seeds = args[2:5]
    before = (_mesh_digest(source), _registry_receipt(registry), seeds.committed_snapshot())

    def cancel(phase):
        if phase == "authored pair before validation":
            raise LookupError("cancelled")

    with pytest.raises(LookupError, match="cancelled"):
        stage_authored_root_pair(*args, cancellation_check=cancel)
    assert (_mesh_digest(source), _registry_receipt(registry), seeds.committed_snapshot()) == before
    with pytest.raises(MeshError, match="mismatched root output"):
        stage_authored_root_pair(*args[:7], args[7][::-1])
    with pytest.raises(MeshError, match="both original roots"):
        stage_authored_root_pair(*args[:5], ("invalid", args[5][1]), *args[6:])
    assert (_mesh_digest(source), _registry_receipt(registry), seeds.committed_snapshot()) == before
    args[0].add_point(20, 20, 20)
    with pytest.raises(MeshError, match="stale"):
        stage_authored_root_pair(*args)


def test_pair_created_rows_get_unique_ids_and_fresh_v2_receipt():
    geometry, component, mesh, registry, seeds, packets, _roots, _children = inputs()
    source_digest = _mesh_digest(mesh)
    points = (((0.231, 0.317),), ((0.231, 0.317),))
    roots = tuple(triangulate_authored_root_boundary(
        packet, interior_metric=packet.chart.to_metric(np.asarray(uv)),
    ) for packet, uv in zip(packets, points))
    children = tuple(bind_authored_root_triangles_to_children(packet, root, correspondence)
                     for packet, root, correspondence in zip(
                         packets, roots, component.boundary_correspondences))
    result = stage_authored_root_pair(
        geometry, component, mesh, registry, seeds, packets, roots, children,
    )
    assert result.current_receipt.schema == PREPARED_CURRENT_ASSOCIATIONS_CREATED_UV_SCHEMA
    created = [node for _root, values in result.created_material_uv_by_root
               for node, _uv in values]
    assert len(created) == len(set(created)) == 2
    assert all(node > max(mesh.nodes) for node in created)
    assert all(node in result.mesh.nodes for node in created)
    assert len(result.mesh.tris) > 16
    assert _mesh_digest(mesh) == source_digest


def test_pair_refinement_debits_one_ledger_and_limited_output_cannot_stage():
    args = inputs()
    source_digest = _mesh_digest(args[2])
    options = NativeMeshingOptions(point_placement="frontal_delaunay",
                                   max_insertions=30, max_topology_operations=1000)
    original = dict(selected_route="frontal_delaunay", cancelled=False,
                    insertion_budget=30, topology_budget=1000, insertions=1,
                    topology_operations=2, shared_segment_splits=0, shared_nodes=[])
    ledger = AuthoredWorkLedger.from_native_report(options, original)
    refined = refine_authored_pair_with_one_ledger(
        args[5], args[6], ledger, options, target_size=2.0,
    )
    first, second = refined.reports
    assert second["insertion_budget"] == first["insertion_budget"] - first["insertions"]
    assert second["topology_budget"] == first["topology_budget"] - first["topology_operations"]
    assert refined.ledger.further_insertions == first["insertions"] + second["insertions"]
    assert refined.ledger.further_operations == (first["topology_operations"]
                                                 + second["topology_operations"])
    assert ledger.further_insertions == ledger.further_operations == 0
    assert refined.layout_eligible is False
    with pytest.raises(MeshError, match="unaccepted limited root output"):
        stage_authored_root_pair(*args[:6], refined.triangulations, args[7])
    assert _mesh_digest(args[2]) == source_digest
