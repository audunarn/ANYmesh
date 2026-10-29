"""Actual rounded full-mesh gates; root supervises numerical execution."""
import json
import os
from pathlib import Path

import pytest
import anygeometry as owner
from anymesher import generate_hybrid_mesh_result, verify_mesh_quality
from test_local_curved_mesh import curved_loop_model
from test_nurbs_curvilinear_consumer import options


@pytest.mark.parametrize('local', (False,), ids=('fixed_0025',))  # automatic_local removed: >=300 s native run (owner decision 2026-10-28)
def test_rounded_closed_loop_single_span_fixed_and_auto(monkeypatch, local):
    model, face, edges, _ = curved_loop_model()
    before = owner.to_dict(model)
    receipts = []
    issuance = []
    actual = owner.validate_prepared_material_curvilinear_cells

    def observe(*args, **kwargs):
        issuer, regions = args[:2]
        issuer_before = owner.to_dict(issuer)
        receipt = actual(*args, **kwargs)
        owner.validate_prepared_material_curvilinear_cells_binding(
            issuer, regions, receipt, expected_revision=issuer.revision)
        receipts.append(receipt)
        issuance.append((issuer, receipt, issuer_before))
        return receipt

    monkeypatch.setattr(owner, 'validate_prepared_material_curvilinear_cells', observe)
    request = options()
    request.update(native_backend='auto', certification_mode='interactive',
                   mutation_policy='read_only', qualified_s3=True)
    if local:
        request.pop('geometry_approximation_error')
        request['geometry_approximation_policy'] = {'adaptation': 'local'}
    else:
        request['geometry_approximation_error'] = .0025
    result = generate_hybrid_mesh_result(model, **request)
    mesh = result.mesh
    quality = verify_mesh_quality(mesh)
    physical = [row['material_region_final_physical_quality']
                for row in result.triangulation_backend_by_face.values()
                if 'material_region_final_physical_quality' in row]
    edge_intervals = {edge: len(mesh.nodes_of_edge[edge]) - 1 for edge in edges}
    interval_count = sum(edge_intervals.values())
    # This unchanged-clone route publishes identity face/edge maps and retains
    # global node IDs/connectivity/XYZ. Prepared proof issuers may differ from
    # the original source model; compare each receipt with its actual issuer.
    final_cells = {tuple(cell): tuple(tuple(map(float, mesh.nodes[node])) for node in cell)
                   for cell in mesh.shells.values()}
    receipt_audit = []
    for issuer, receipt, document in issuance:
        proved = {cell.node_ids: cell.xyz for cell in receipt.cells}
        receipt_audit.append({
            'issuer_model_id': document['model_id'], 'issuer_revision': document['revision'],
            'issuer_checksum': document['checksum']['value'],
            'receipt_model_id': str(receipt.model_id), 'receipt_revision': receipt.revision,
            'receipt_checksum': receipt.source_checksum, 'cells': len(receipt.cells),
            'surface_error': receipt.surface_error,
            'has_local_field_fingerprint': bool(receipt.local_field_fingerprint),
            'issuer_faces': sorted(int(row['id']) for row in document['faces']),
            'receipt_faces': sorted({cell.face.id for cell in receipt.cells}),
            'matches_final_cells_and_xyz': len(receipt.cells) == len(final_cells) and proved == final_cells,
            'binding_validated_at_issuance': True,
        })
    audit = {
        'mode': 'automatic_local' if local else 'fixed_0025',
        'nodes': len(mesh.nodes), 'cells': len(mesh.shells),
        'edge_intervals': edge_intervals, 'boundary_intervals': interval_count,
        'boundary_168_192_hypothesis_met': 168 <= interval_count <= 192,
        'owner_receipts': len(receipts), 'min_angle': quality.min_angle,
        'geometry_approximation': mesh.hybrid_diagnostics.get('geometry_approximation'),
        'receipt_issuers': receipt_audit,
        'physical': [{'accepted': row['accepted'],
                      'min_angle': row['worst']['minimum_angle'],
                      'maximum_element_growth': row['maximum_element_growth'],
                      'violation_counts': row['violation_counts'],
                      'poor_element_count': len(row['poor_element_ids'])}
                     for row in physical],
    }
    print('ROUNDED_FULL_ACCEPTANCE', json.dumps(audit, sort_keys=True), flush=True)
    audit_path = os.environ.get('ANY_ROUNDED_ACCEPTANCE_AUDIT')
    if audit_path:
        # Scalar diagnostics only: this file grants no serialized proof authority.
        Path(audit_path).write_text(json.dumps(audit, sort_keys=True, indent=2), encoding='utf-8')

    assert mesh.shells and mesh.order == 'linear'
    assert mesh.geometry_model_id == model.model_id
    assert mesh.geometry_revision == model.revision
    assert set(mesh.elements_of_face) == set(model.faces) == {face}
    owned = [cell for cells in mesh.elements_of_face.values() for cell in cells]
    assert len(owned) == len(set(owned)) and set(owned) == set(mesh.shells)
    assert mesh.elements_of_face[face]
    assert set(edges) <= set(mesh.nodes_of_edge)
    used_nodes = {node for cell in mesh.shells.values() for node in cell}
    assert all(set(mesh.nodes_of_edge[edge]) <= used_nodes for edge in edges)
    assert owner.to_dict(model) == before
    admission = mesh.structural_preparation['qualified_s3']
    assert admission['status'] == 'ADMITTED'
    assert admission['authority_model']['source_model_id'] == str(model.model_id)

    assert receipts
    assert all(type(receipt) is owner.PreparedMaterialCurvilinearCells for receipt in receipts)
    assert all(str(receipt.model_id) == document['model_id']
               and receipt.revision == document['revision']
               and receipt.source_checksum == document['checksum']['value']
               and all(str(cell.face.model_id) == document['model_id']
                       and cell.face.id in {int(row['id']) for row in document['faces']}
                       for cell in receipt.cells)
               for issuer, receipt, document in issuance)
    assert any(row['matches_final_cells_and_xyz'] and row['receipt_faces'] == [face]
               for row in receipt_audit)
    if local:
        diagnostic = mesh.hybrid_diagnostics['geometry_approximation']
        assert diagnostic['mode'] == 'automatic' and diagnostic['adaptation'] == 'local'
        field = diagnostic['local_field']
        assert field['definition_checksum'] and field['version']
        assert field['source_model_id'] == diagnostic['source']['model_id'] == str(model.model_id)
        assert field['source_checksum'] == diagnostic['source']['source_checksum'] == before['checksum']['value']
        assert set(diagnostic['face_ids']) <= set(diagnostic['clearance_face_ids']) <= set(model.faces)
        assert all(receipt.local_field_fingerprint and receipt.local_footprints_by_cell
                   and receipt.surface_errors_by_cell for receipt in receipts)
        assert all(0 < error <= receipt.surface_error for receipt in receipts
                   for error in receipt.surface_errors_by_cell)
        assert all(receipt.surface_error == diagnostic['effective_error'] for receipt in receipts)
    else:
        assert all(receipt.surface_error == .0025 and not receipt.local_field_fingerprint
                   for receipt in receipts)
    # These are the actual default physical gates, not chart-only metrics.
    assert quality.min_angle >= 30.
    assert physical
    assert all(row['accepted'] and row['worst']['minimum_angle'] >= 30.
               and row['maximum_element_growth'] <= 1.5
               and all(count == 0 for count in row['violation_counts'].values())
               and not row['poor_element_ids'] for row in physical)
