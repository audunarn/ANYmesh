"""S2: eligible NURBS faces reach the tensor route through quad_options and strategy='auto'; ineligible stay refused."""
from __future__ import annotations

import pytest

from anygeometry import to_dict
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported

from test_qf_nurbs_s0_characterisation import nurbs_patch, planar_model

WARP = .5  # tensor cells of this patch miss the final quality gate (probe: work/warp_probe.py)


def request(**extra):
    base = dict(target_size=.5, native_backend='python', order='linear', certification_mode='none',
                geometry_approximation_error=.01, structural_preparation=False)
    base.update(extra)
    return base


def test_explicit_quad_options_reach_the_tensor_route():
    model, face = nurbs_patch()
    before = to_dict(model)
    result = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request())
    mesh = result.mesh
    assert len(mesh.tris) == 0 and len(mesh.quads) >= 4
    diagnostics = [v for v in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values()]
    assert any('parametric_tensor' in d for d in diagnostics)
    assert to_dict(model) == before


def test_auto_strategy_uses_the_tensor_route_for_eligible_faces():
    model, face = nurbs_patch()
    mesh = generate_hybrid_mesh_result(model, strategy='auto', **request()).mesh
    assert len(mesh.tris) == 0
    assert any('parametric_tensor' in d for d in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values())


def test_explicit_native_strategy_keeps_the_native_route():
    model, face = nurbs_patch()
    mesh = generate_hybrid_mesh_result(model, strategy='native',
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial'),
        **request()).mesh
    assert not any('parametric_tensor' in d for d in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values())


def test_ineligible_nurbs_with_quad_options_is_refused_with_reason_code():
    model, face = nurbs_patch(triangle=True)
    with pytest.raises(QuadPublicUnsupported, match=r'NURBS.*not_four_edges'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request())


def test_planar_only_selection_keeps_its_ordinary_quad_first_route():
    model, face = nurbs_patch()
    plane_model, plane_face = planar_model()
    mesh = generate_hybrid_mesh_result(plane_model, face_ids=(plane_face,), strategy='native',
        quad_options=QuadMeshingOptions(), **request(geometry_approximation_error=None)).mesh
    assert len(mesh.quads) > 0


def test_gate_failure_inside_the_tensor_is_a_typed_refusal_not_a_native_fallback():
    # strong warp: tensor cells miss the warpage gate, explicit quad_options must refuse (no silent native)
    model, face = nurbs_patch(warp=WARP)
    with pytest.raises(QuadPublicUnsupported, match='tensor'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request())


def test_auto_falls_back_to_native_when_the_tensor_refuses(monkeypatch):
    # (a warp large enough to miss the tensor gates also stalls the native route, so the refusal is injected)
    import anymesher._parametric_tensor as tensor
    def refuse(*args, **kwargs):
        raise tensor.ParametricTensorUnsupported('injected refusal')
    monkeypatch.setattr(tensor, 'parametric_tensor_core', refuse)
    model, face = nurbs_patch()
    mesh = generate_hybrid_mesh_result(model, strategy='auto', **request()).mesh
    refusals = [d['parametric_tensor_refusal'] for d in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values()
                if 'parametric_tensor_refusal' in d]
    assert refusals == ['injected refusal'] and len(mesh.quads) + len(mesh.tris) > 0
