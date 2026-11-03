"""Strong pair proof obeys intent and the existing explicit record budgets."""
import pytest

import anygeometry as owner
from anymesher.errors import MeshError
from anymesher.preparation import prepare_structural_closure
from test_effective_tensor_admission import raw_sweep
from test_strong_raw_sweep_preparation import pair


def test_isolated_canonical_sweep_never_requests_pair_proof(monkeypatch):
    model, face = raw_sweep()
    before = owner.to_dict(model)
    def forbidden(*args, **kwargs):
        pytest.fail('isolated face requested whole-pair proof')
    monkeypatch.setattr(owner, 'query_translational_profile_interfaces', forbidden)
    working, report = prepare_structural_closure(model, options={'declare_missing_owners': True})
    assert len(working.faces) == 1 and report.candidate_queries == 0
    assert report.applications == 0
    assert owner.to_dict(model) == before


def test_member_only_intent_does_not_request_face_pair_proof(monkeypatch):
    import anymesher.preparation as preparation
    model = pair()
    before = owner.to_dict(model)
    def forbidden(*args, **kwargs):
        pytest.fail('disabled automatic face connections requested pair proof')
    monkeypatch.setattr(owner, 'query_translational_profile_interfaces', forbidden)
    class OriginalRouteObserved(Exception): pass
    def observe_original_route(working, operands, *, policy):
        assert policy.face_connections is False
        assert len(operands) == 2
        raise OriginalRouteObserved
    monkeypatch.setattr(preparation, 'plan_intersections', observe_original_route)
    with pytest.raises(OriginalRouteObserved):
        prepare_structural_closure(model, options={
            'declare_missing_owners': True, 'automatic_face_connections': False,
            'automatic_member_connections': True, 'automatic_member_sheet_connections': False})
    assert owner.to_dict(model) == before


@pytest.mark.parametrize('name', ('maximum_face_records', 'maximum_edge_records'))
def test_record_limits_refuse_before_pair_query_with_original_message(monkeypatch, name):
    model = pair()
    limit = 1 if name == 'maximum_face_records' else len(model.edges)-1
    before = owner.to_dict(model)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(1)
        pytest.fail('record overflow reached expensive pair proof')
    monkeypatch.setattr(owner, 'query_translational_profile_interfaces', forbidden)
    with pytest.raises(MeshError) as caught:
        prepare_structural_closure(model, options={'declare_missing_owners': True, name: limit})
    assert str(caught.value) == f'structural preparation exceeds explicit {name}={limit}'
    assert calls == [] and owner.to_dict(model) == before


def test_candidate_pair_limit_keeps_original_preproof_refusal(monkeypatch):
    from anygeometry.nurbs import NurbsCurve
    model = pair()
    start = next(v for v in model.vertices if tuple(model.vertex_position(v)) == (0., 4., 0.))
    end = model.add_point(0., 6., 0.)
    curve = NurbsCurve(2, (0., 0., 0., 1., 1., 1.),
                       ((0., 4., 0.), (0., 5., 1.), (0., 6., 0.)))
    model.extrude((model.add_curve(start, end, curve),), (3., 0., 0.))
    def forbidden(*args, **kwargs):
        pytest.fail('candidate-pair overflow reached expensive proof')
    monkeypatch.setattr(owner, 'query_translational_profile_interfaces', forbidden)
    with pytest.raises(MeshError, match='^strong sweep interfaces exceed explicit maximum_candidate_pairs$'):
        prepare_structural_closure(model, options={'declare_missing_owners': True,
            'maximum_candidate_pairs': 1, 'maximum_face_records': 3,
            'maximum_edge_records': len(model.edges)})


def test_missing_strong_proof_never_falls_back_to_acceptance(monkeypatch):
    model = pair()
    before = owner.to_dict(model)
    monkeypatch.setattr(owner, 'query_translational_profile_interfaces', lambda *a, **k: None)
    with pytest.raises(MeshError, match='not strongly certified'):
        prepare_structural_closure(model, options={'declare_missing_owners': True})
    assert owner.to_dict(model) == before
