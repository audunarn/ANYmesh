"""Single-face admission uses only the geometry owner's canonical read view."""
from dataclasses import replace

import pytest

from anygeometry import GeometryError, GeometryModel, to_dict
from anygeometry.entities import OrientedEdge
from anygeometry.nurbs import NurbsCurve, NurbsSurface
from anygeometry.nurbs_sweep_chart import source_sweep_chart_model
from anygeometry.surfaces import CoonsSurface
from anymesher._parametric_tensor import nurbs_tensor_admission


def raw_sweep(*, rotation=0, translated=True):
    model = GeometryModel()
    profile = NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
                         ((0., 0., 0.), (0., .5, 1.), (0., 1., 0.)))
    controls = tuple((x+5., y, z) for x, y, z in profile.controls)
    if not translated:
        controls = (controls[0], (5., .5, 1.25), controls[2])
    opposite = replace(profile, controls=controls)
    a, b, c, d = model.add_points(((0., 0., 0.), (0., 1., 0.),
                                  (5., 1., 0.), (5., 0., 0.)))
    bottom = model.add_curve(a, b, profile)
    right = model.add_line(b, c)
    top = model.add_curve(d, c, opposite)
    left = model.add_line(d, a)
    loop = tuple(OrientedEdge(edge, forward) for edge, forward in
                 ((bottom, True), (right, True), (top, False), (left, True)))
    loop = loop[rotation:] + loop[:rotation]
    face = model.add_face_from_loop(loop, (0, 1, 2, 3), surface=CoonsSurface())
    return model, face


@pytest.mark.parametrize('rotation', range(4))
def test_canonical_raw_sweep_is_eligible_without_publishing_chart(rotation):
    model, face = raw_sweep(rotation=rotation)
    before = to_dict(model)
    literal = model.faces[face]
    view = source_sweep_chart_model(model, face)
    assert view is not None
    assert isinstance(view.faces[face].surface, NurbsSurface)
    assert nurbs_tensor_admission(model, face).eligible
    assert model.faces[face] is literal
    assert literal.parameterization is None
    assert to_dict(model) == before


def test_nontranslation_coons_stays_ineligible_and_read_only():
    model, face = raw_sweep(translated=False)
    before = to_dict(model)
    assert source_sweep_chart_model(model, face) is None
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'not_nurbs_surface'
    assert to_dict(model) == before


def test_changed_boundary_cannot_reuse_previously_derived_view():
    model, face = raw_sweep()
    assert source_sweep_chart_model(model, face) is not None
    top = model.faces[face].loop[2].edge
    edge = model.edges[top]
    controls = list(edge.curve.controls)
    controls[1] = (5., .5, 1.25)
    model._put_entity('edge', replace(edge, curve=replace(edge.curve, controls=tuple(controls))))
    before = to_dict(model)
    assert source_sweep_chart_model(model, face) is None
    assert not nurbs_tensor_admission(model, face).eligible
    assert to_dict(model) == before


def test_canonical_view_still_requires_boundary_uv(monkeypatch):
    model, face = raw_sweep()
    before = to_dict(model)
    monkeypatch.setattr(model, 'face_boundary_uv', lambda *args: None)
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'no_boundary_uv'
    assert to_dict(model) == before


@pytest.mark.parametrize('error_type', (GeometryError, KeyboardInterrupt))
def test_owner_errors_propagate_without_alternate_admission(monkeypatch, error_type):
    import anygeometry.nurbs_sweep_chart as owner
    model, face = raw_sweep()
    before = to_dict(model)
    sentinel = error_type('owner sentinel')
    def refuse(*args):
        raise sentinel
    monkeypatch.setattr(owner, 'source_sweep_chart_model', refuse)
    with pytest.raises(error_type) as caught:
        nurbs_tensor_admission(model, face)
    assert caught.value is sentinel
    assert to_dict(model) == before


def test_consumed_quality_parameter_has_only_none_contract():
    from anymesher import _tensor_contract as contract
    assert contract.PARAMETERS['quality_options'] == ('none',)
    assert contract.request_refusal({'quality_options': None}, None, None) is None
    assert 'future_parameter' in contract.request_refusal({'future_parameter': None}, None, None)


@pytest.mark.parametrize('policy', ({}, object(), {'quality_policy': {'minimum_angle': 1.}}))
def test_underlying_unconsumed_quality_is_refused(policy):
    from anymesher._tensor_contract import request_refusal
    assert 'quality_options' in request_refusal({'quality_options': policy}, None, None)


def test_public_tensor_request_still_requires_final_quality_gate(monkeypatch):
    import anymesher.hybrid as owner
    from anymesher.errors import MeshError
    from anymesher.quad.options import QuadMeshingOptions
    from test_qf_nurbs_s0_characterisation import nurbs_patch
    from test_qf_nurbs_s2_route import request
    model, face = nurbs_patch()
    before = to_dict(model)
    actual = owner.validate_mesh_quality
    sentinel = MeshError('final quality sentinel')
    calls = []
    def refuse_after_actual_report(geometry, mesh, **kwargs):
        report = actual(geometry, mesh, **kwargs)
        calls.append((mesh, kwargs['quality_options'], report))
        raise sentinel
    monkeypatch.setattr(owner, 'validate_mesh_quality', refuse_after_actual_report)
    with pytest.raises(MeshError) as caught:
        owner.generate_hybrid_mesh_result(model, strategy='native',
            quad_options=QuadMeshingOptions(), quality_options={'quality_policy': {'minimum_angle': 1.}},
            **request())
    assert caught.value is sentinel
    assert len(calls) == 1 and calls[0][2]['accepted']
    assert calls[0][1].quality_policy.minimum_angle == 1.
    assert to_dict(model) == before
