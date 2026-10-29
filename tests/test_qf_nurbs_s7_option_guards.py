"""Fixes-1: the tensor route refuses whenever a caller option it cannot honour is present (Codex findings 1-4).

Each test was run against the pre-fix code and failed there (see PLAN 'Build log fixes-1').
"""
import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher.refinement import Refinement
from anymesher._parametric_tensor import nurbs_tensor_admission

from test_qf_nurbs_s0_characterisation import nurbs_patch
from test_qf_nurbs_s2_route import request


def tensor_used(mesh):
    return any('parametric_tensor' in d for d in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values())


def refusal_reasons(mesh):
    return [d['parametric_tensor_refusal'] for d in mesh.hybrid_diagnostics['triangulation_backend_by_face'].values()
            if 'parametric_tensor_refusal' in d]


CENTRE_ZONE = dict(size=.01, radius=.1, center=(.5, .5, 0.), growth=10.)


def test_nonuniform_size_field_is_refused_for_explicit_quad_options():
    model, face = nurbs_patch(warp=0.)
    with pytest.raises(QuadPublicUnsupported, match='size field|refinements'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
            refinements=(Refinement(**CENTRE_ZONE),), **request())


def test_nonuniform_size_field_is_never_served_by_the_tensor_under_auto():
    model, face = nurbs_patch(warp=0.)
    mesh = generate_hybrid_mesh_result(model, strategy='auto',
        refinements=(Refinement(size=.1, radius=.1, center=(.5, .5, 0.), growth=4.),), **request()).mesh
    assert not tensor_used(mesh) and any(('size field' in r or 'refinements' in r) for r in refusal_reasons(mesh))


def test_recombine_false_is_refused_by_the_tensor():
    model, face = nurbs_patch()
    mesh = generate_hybrid_mesh_result(model, strategy='auto', recombine=False, **request()).mesh
    assert not tensor_used(mesh) and len(mesh.tris) > 0
    assert any('recombine' in r for r in refusal_reasons(mesh))


def test_recombine_false_with_explicit_quad_options_is_a_typed_refusal():
    model, face = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported, match='recombine'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), recombine=False,
                                    **request())


def test_quad_face_ids_covering_the_selection_is_served():
    model, face = nurbs_patch()
    mesh = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                       quad_face_ids=(face,), **request()).mesh
    assert tensor_used(mesh) and len(mesh.tris) == 0


def test_quad_face_ids_that_do_not_cover_the_selection_is_a_typed_refusal():
    model, face = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported, match='quad_face_ids'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                    quad_face_ids=(face + 100,), **request())


def inset_face(lo=.2, hi=.8):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    corners = ((lo, lo), (hi, lo), (hi, hi), (lo, hi))
    curves = tuple(CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.), (corners[i], corners[(i + 1) % 4])))
                   for i in range(4))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], c) for i, c in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face


def test_inset_trim_is_not_the_unit_box():
    model, face = inset_face()
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'not_iso_parametric_rectangle'


def wavy_edge_face(amplitude=.15):
    """Bottom side is piecewise quadratic: v is 0 at t=0, .5, 1 but positive in between."""
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    line = lambda a, b: NurbsCurve(1, (0., 0., 1., 1.), (a, b))
    wavy = NurbsCurve(2, (0., 0., 0., .5, .5, 1., 1., 1.),
                      ((0., 0.), (.25, amplitude), (.5, 0.), (.75, amplitude), (1., 0.)))
    pcurves = (wavy, line((1., 0.), (1., 1.)), line((1., 1.), (0., 1.)), line((0., 1.), (0., 0.)))
    curves = tuple(CurveOnSurface(support, p) for p in pcurves)
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], c) for i, c in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face


def test_edge_deviating_between_samples_is_not_iso_parametric():
    model, face = wavy_edge_face()
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'not_iso_parametric_rectangle'


def test_customised_quad_options_are_refused_not_dropped():
    model, face = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported, match='quad_options'):
        generate_hybrid_mesh_result(model, strategy='native',
            quad_options=QuadMeshingOptions(quality_model='shape_jacobian'), **request())


def test_beam_edges_with_explicit_quad_options_are_refused():
    model, face = nurbs_patch()
    edge = model.faces[face].loop[0].edge
    with pytest.raises(QuadPublicUnsupported, match='beam'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                    beam_edges=(edge,), **request())
