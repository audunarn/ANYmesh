"""S4 (D1/D2): the curvilinear recovery policy chooses the tensor route for eligible NURBS faces, native otherwise."""
from __future__ import annotations

import pytest

from anymesher.recovery import generate_automatic_mesh_result, select_curvilinear_recovery_policy

from test_nurbs_curvilinear_consumer import patch_model
from test_qf_nurbs_s0_characterisation import nurbs_patch


def select(model, requested='quad_first'):
    return select_curvilinear_recovery_policy(model, requested_strategy=requested, order='linear',
        certification_mode='interactive', geometry_approximation_error=.01, explicit_controls=False)


@pytest.mark.parametrize('requested', ['quad_first', 'auto'])
def test_eligible_face_selects_the_tensor_route(requested):
    model, face, _ = patch_model()
    policy, record = select(model, requested)
    assert policy is not None and record['nurbs_tensor_faces'] == [face]
    assert 'tensor' in record['reason'] and 'native' in record['reason']


def test_ineligible_face_keeps_the_native_recipe_reason():
    model, face = nurbs_patch(triangle=True)
    policy, record = select(model)
    assert record['nurbs_tensor_faces'] == []
    assert record['reason'] == 'new owner-qualified NURBS material uses existing automatic native recipe'


def test_recovery_run_publishes_through_the_tensor_route():
    model, face, _ = patch_model()
    policy, selection = select(model)
    result = generate_automatic_mesh_result(model, automation=policy, curvilinear_recovery_selection=selection,
        target_size=.5, strategy='quad_first', order='linear', certification_mode='interactive',
        geometry_approximation_error=.01, structural_preparation=False, recombine=True, native_backend='python')
    mesh = result.mesh
    assert any('parametric_tensor' in d for d in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values())
    assert mesh.hybrid_diagnostics['curvilinear_recovery']['nurbs_tensor_faces'] == [face]
    assert len(mesh.tris) == 0
