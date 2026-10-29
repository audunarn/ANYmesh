"""S0 characterisation: default outputs stay bit-identical; ineligible NURBS + quad_options stays refused.

The golden digests were produced by the baseline copy (``integration/``), read-only, with
``work/s0_hashes.py``; the same digests must hold in ``integration-qf``. Ordinary (non-NURBS) quad-first
and explicit-native NURBS results must not change when the NURBS quad-first route is added.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported

from test_curved_native_qualification import _model_face


def digest(mesh) -> str:
    """sha256 over sorted node ids+coordinates and sorted element ids+connectivity."""
    h = hashlib.sha256()
    for key in sorted(mesh.nodes):
        h.update(np.int64(key).tobytes() + np.asarray(mesh.nodes[key], dtype=np.float64).tobytes())
    for name in ('quads', 'tris'):
        table = getattr(mesh, name)
        h.update(name.encode())
        for key in sorted(table):
            h.update(np.int64(key).tobytes() + np.asarray(table[key], dtype=np.int64).tobytes())
    return h.hexdigest()


def planar_model():
    model = GeometryModel()
    pts = [model.add_point(*p) for p in ((0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.))]
    edges = [model.add_line(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    return model, model.add_face(edges)


def quad_first(model, face, h):
    return generate_hybrid_mesh_result(
        model, face_ids=(face,), target_size=h, strategy='native', native_backend='python',
        recombine=True, order='linear', quad_options=QuadMeshingOptions()).mesh


def nurbs_patch(*, warp=.1, triangle=False):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., warp))))
    corners = ((0., 0.), (1., 0.), (0., 1.)) if triangle else ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    n = len(corners)
    curves = tuple(CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.),
        (corners[i], corners[(i + 1) % n]))) for i in range(n))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % n], curve) for i, curve in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face


GOLDEN = {
    'planar': '357065ccbca528c3d08905e9f38473141781069f9fd8db668cada0731565af67',
    'cylinder': '1b7a9cdfdfb3563f4ac95817ff63f77d510050c436f60476e108ea3c211d8949',
    'cone': '843a32d901a5aa5ebb7108e0b2870058afa648f6124050b297da3de2fd0f278f',
    'ruled': '0a3a26e9df46528c148a5eebfe5ec21f4610cc4442a0d69494fd334b9e8c6e31',
    'coons': '0a3a26e9df46528c148a5eebfe5ec21f4610cc4442a0d69494fd334b9e8c6e31',
    'native_nurbs': 'eb02e4af2fc2683b4e1864667dc4e1011bbe10978826ccc10c22abad16dfc84b',
}


def build(name):
    if name == 'planar':
        model, face = planar_model()
        return quad_first(model, face, .4)
    if name == 'cylinder':
        from test_cylindrical_atlas_binding import _sector_model
        model, selected = _sector_model(False)
        return quad_first(model, int(model.face_uses[selected[0].id].face_id), .5)
    if name in ('cone', 'ruled', 'coons'):
        model, face, _ = _model_face(name)
        return quad_first(model, face, .5)
    model, face = nurbs_patch()
    return generate_hybrid_mesh_result(
        model, target_size=.5, strategy='native', native_backend='python', order='linear',
        certification_mode='none', geometry_approximation_error=.01,
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial'),
        structural_preparation=False).mesh


@pytest.mark.parametrize('name', sorted(GOLDEN))
def test_default_outputs_are_bit_identical_to_baseline(name):
    assert digest(build(name)) == GOLDEN[name]


@pytest.mark.parametrize('kind', ['triangle'])
def test_ineligible_nurbs_with_quad_options_stays_refused(kind):
    model, face = nurbs_patch(triangle=True)
    with pytest.raises(QuadPublicUnsupported, match='NURBS'):
        generate_hybrid_mesh_result(
            model, target_size=.5, strategy='native', native_backend='python', order='linear',
            certification_mode='none', geometry_approximation_error=.01, quad_options=QuadMeshingOptions())
