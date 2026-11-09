"""Cheap NURBS route contracts (gap audit T1-T3, T5): refusals, recorded auto fallback, tensor sizing band.

These pin existing behaviour; each runs on the 4-edge ``warp_0.1`` patch in about 2 s.
"""
from __future__ import annotations

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported

from nurbs_patch_fixtures import patch_model

TARGET = .3


def request(**extra):
    base = dict(target_size=TARGET, native_backend='python', order='linear', certification_mode='none',
                geometry_approximation_error=.01, structural_preparation=False)
    base.update(extra)
    return base


def test_mapped_refuses_nurbs_face():  # T1
    model, _ = patch_model('warp_0.1')
    with pytest.raises(MeshError, match='generic mapped route'):
        generate_hybrid_mesh_result(model, strategy='mapped', **request())


def test_smart_mapped_refuses_nurbs_face():  # T2
    model, _ = patch_model('warp_0.1')
    with pytest.raises(QuadPublicUnsupported, match='smart-mapped request'):
        generate_hybrid_mesh_result(model, strategy='smart_mapped', **request())


def test_auto_tensor_refusal_falls_back_to_native_and_records_reason():  # T3
    model, face = patch_model('warp_0.1')
    result = generate_hybrid_mesh_result(model, strategy='auto', recombine=False, **request())
    assert len(result.mesh.tris) > 0
    assert dict(result.strategy_by_face) == {face: 'native'}
    reason = result.mesh.hybrid_diagnostics['triangulation_backend_by_face'][face]['parametric_tensor_refusal']
    assert 'recombine=False' in reason


def test_tensor_edge_length_to_target_band():  # T5
    """Measured 1.111-1.117x target (coarser: station-count rounding). Band pins it so a change is visible."""
    model, _ = patch_model('warp_0.1')
    mesh = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                       **request()).mesh
    assert len(mesh.tris) == 0 and len(mesh.quads) == 9
    edges = np.array([np.linalg.norm(mesh.nodes[q[i]] - mesh.nodes[q[(i + 1) % 4]])
                      for q in mesh.quads.values() for i in range(4)])
    assert 0.9 <= edges.mean() / TARGET <= 1.35
    assert edges.max() / edges.min() <= 1.05
