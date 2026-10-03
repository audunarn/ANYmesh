"""Public current-only receipt is complete for its narrow linear shell scope."""

from dataclasses import replace
import pytest

from test_authored_component_cells import candidate
from anymesher import (
    PREPARED_CURRENT_ASSOCIATIONS_SCHEMA,
    query_prepared_current_component_associations,
    validate_prepared_current_component_associations,
)
from anymesher.errors import MeshError


def test_exact_current_component_receipt_preserves_all_supported_buckets():
    args = candidate()
    receipt = query_prepared_current_component_associations(*args)
    assert receipt.schema == PREPARED_CURRENT_ASSOCIATIONS_SCHEMA
    assert receipt.association_namespace == "prepared_current"
    assert receipt.authored_face_ids == (1, 2)
    assert len(receipt.elements_of_face) == 8
    assert len(receipt.elements_of_sheet) == 2
    assert len(receipt.elements_of_face_use) == 8
    assert len(receipt.cell_current_faces) == 16
    assert len(receipt.node_of_vertex) == 16
    assert len(receipt.joint_chains) == 1
    assert len(receipt.source_boundary_chains) == 8
    assert all(len(nodes) >= 2 for _root, _edge, _forward, _parts, nodes
               in receipt.source_boundary_chains)
    joint = receipt.joint_chains[0][0]
    assert dict(receipt.exterior_source_edge_ancestry)[joint] == ()
    assert any(ancestors for edge, ancestors in receipt.exterior_source_edge_ancestry
               if edge != joint)
    assert any(ancestors for _vertex, ancestors in receipt.current_vertex_preimages)
    assert any(not ancestors for _vertex, ancestors in receipt.current_vertex_preimages)
    assert receipt.source_reference_transfer_qualified is False
    assert receipt.solver_admitted is False
    assert receipt.publication_qualified is False
    validate_prepared_current_component_associations(args[0], receipt, *args[1:])


def test_forged_receipt_changed_mesh_and_unsupported_metadata_refuse():
    args = candidate()
    receipt = query_prepared_current_component_associations(*args)
    forged = replace(receipt, elements_of_sheet=())
    with pytest.raises(MeshError, match="contents or owner binding changed"):
        validate_prepared_current_component_associations(args[0], forged, *args[1:])
    args[2].nodes_of_edge[args[1].joint_edge_id].reverse()
    with pytest.raises(MeshError, match="station order"):
        validate_prepared_current_component_associations(args[0], receipt, *args[1:])
    args = candidate()
    args[2].seeding = object()
    with pytest.raises(MeshError, match="unsupported associations"):
        query_prepared_current_component_associations(*args)
