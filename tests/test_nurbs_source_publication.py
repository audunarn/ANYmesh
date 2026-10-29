"""Actual source publication consumes live owner receipts, never projections."""
from copy import deepcopy
import pytest
import anygeometry as owner
from anymesher.errors import MeshError
from anymesher.material_source_transfer import (
    _publish_identity_source_transfer, _validate_identity_source_publication,
)
from test_material_source_transfer_consumer import prepared_mesh


def test_identity_clone_source_publication_preserves_literal_stations(monkeypatch):
    source, prepared, mesh, face = prepared_mesh()
    before = owner.to_dict(source), owner.to_dict(prepared)
    registry = mesh.boundary_registry
    stations = tuple((entry.key.edge_id, entry.key.parameter, entry.node_id,
                      tuple(entry.point)) for entry in registry.entries())
    def refuse_projection(*args, **kwargs):
        pytest.fail('source station identity must not use closest-point projection')
    monkeypatch.setattr(source, 'closest_edge_point', refuse_projection)
    made = _publish_identity_source_transfer(source, prepared, mesh, registry)
    receipt = _validate_identity_source_publication(source, prepared, mesh, made)
    assert receipt.source_to_prepared_faces == (
        (source.handle('face', face), (prepared.handle('face', face),)),)
    assert mesh.geometry_model_id == source.model_id
    assert mesh.geometry_revision == source.revision
    assert made.view.source is source
    assert tuple((entry.key.edge_id, entry.key.parameter, entry.node_id,
                  tuple(entry.point)) for entry in made.entries()) == stations
    assert (owner.to_dict(source), owner.to_dict(prepared)) == before


@pytest.mark.parametrize('change', ['copy', 'xyz', 'registry'])
def test_published_source_transfer_cannot_move_to_other_mesh_or_changed_cells(change):
    source, prepared, mesh, _ = prepared_mesh()
    registry = _publish_identity_source_transfer(source, prepared, mesh, mesh.boundary_registry)
    if change == 'copy':
        mesh = deepcopy(mesh)
    elif change == 'xyz':
        node = next(iter(mesh.nodes))
        xyz = tuple(mesh.nodes[node])
        mesh.nodes[node] = (xyz[0]+.01, xyz[1], xyz[2])
    else:
        entry = next(iter(registry.entries()))
        object.__setattr__(entry, 'point', tuple(float(v)+.01 for v in entry.point))
    with pytest.raises(MeshError, match='authority|changed'):
        _validate_identity_source_publication(source, prepared, mesh, registry)


def test_public_source_final_callback_cannot_fractionally_edit_sheet_assignment(monkeypatch):
    from anymesher import hybrid
    from test_nurbs_curvilinear_consumer import patch_model, options
    # The fractional engineering-ID attack is independent of curvature.
    # Retain real NURBS owner proof with a small exact flat material mesh.
    source, _, _ = patch_model(warp=0.)
    before = owner.to_dict(source)
    captured = []
    actual = hybrid._finalize_hybrid_candidate
    def capture(candidate, **kwargs):
        captured.append(candidate)
        return actual(candidate, **kwargs)
    monkeypatch.setattr(hybrid, '_finalize_hybrid_candidate', capture)
    attacked = False
    def callback(phase):
        nonlocal attacked
        if phase == 'hybrid generation complete' and not attacked:
            attacked = True
            mesh = captured[0].mesh
            sheet = next(iter(mesh.elements_of_sheet))
            cells = tuple(mesh.elements_of_sheet[sheet])
            mesh.elements_of_sheet[sheet] = [cells[0]+.5, *cells[1:]]
        return False
    with pytest.raises(MeshError, match='associations changed|final mesh or registry changed'):
        request = options()
        request['target_size'] = 2.
        hybrid.generate_hybrid_mesh_result(source, cancellation_check=callback, **request)
    assert attacked and owner.to_dict(source) == before
