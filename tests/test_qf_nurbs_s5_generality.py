"""S5: non-translation NURBS patches either pass every tensor gate or refuse with a typed reason (no silent mis-mesh).

Outcomes were measured with work/s5_probe.py (target 0.3, explicit quad_options); this freezes them.
Each patch ~2-22 s; whole file ~90 s.
"""
import pytest

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher._parametric_tensor import nurbs_tensor_admission

from nurbs_patch_fixtures import patch_model
from test_qf_nurbs_s2_route import request

QUALITY = 'tensor candidate misses the final quality or growth gate'
COUNTS = 'opposite boundary sides have unequal station counts'
EXPECTED = {
    'warp_0.1': 'pass', 'saddle_mild': 'pass', 'bump_mild': 'pass',
    'warp_0.5': COUNTS, 'trapezoid_planar': COUNTS, 'quarter_cone_rational': COUNTS,
    'twisted_pm_0.3': QUALITY, 'saddle_biquadratic': QUALITY, 'bump_biquadratic': QUALITY,
}


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_patch_passes_all_gates_or_refuses_with_reason(name):
    model, face = patch_model(name)
    assert nurbs_tensor_admission(model, face).eligible  # geometry-level admission is not the quality verdict
    expected = EXPECTED[name]
    if expected == 'pass':
        mesh = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                           **request(target_size=.3)).mesh
        assert len(mesh.tris) == 0 and len(mesh.quads) >= 4
        diagnostics = next(iter(mesh.hybrid_diagnostics['triangulation_backend_by_face'].values()))
        worst, growth = diagnostics['parametric_tensor_quality']['worst'], diagnostics['parametric_tensor_quality']['maximum_growth']
        assert worst['minimum_angle'] >= 15. and worst['maximum_angle'] <= 150.
        assert worst['minimum_scaled_jacobian'] >= .2 and worst['maximum_aspect_ratio'] <= 4.
        assert worst['maximum_warpage'] <= .1 and growth <= 1.5
    else:
        with pytest.raises(QuadPublicUnsupported, match=expected):
            generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                        **request(target_size=.3))
