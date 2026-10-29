"""NURBS closure preparation must retain the geometry owner's refusal boundary."""
import pytest
from anygeometry import GeometryModel, GeometryError, to_dict
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anymesher import preparation
from anymesher.errors import MeshError


def _crossing_patch():
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., .2))))
    corners = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    curves = tuple(CurveOnSurface(support, NurbsCurve(1, (0., 0., 1., 1.),
        (corners[i], corners[(i + 1) % 4]))) for i in range(4))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], curve)
                  for i, curve in enumerate(curves))
    patch = model.add_face(edges, surface=support)
    plate = model.add_plate(model.add_points(
        ((.5, -.2, -.5), (.5, 1.2, -.5), (.5, 1.2, .5), (.5, -.2, .5))))
    return model, patch, plate


def test_incomplete_nurbs_owner_classification_preserves_source(monkeypatch):
    model, patch, plate = _crossing_patch()
    before = to_dict(model)
    # Isolate the batch refusal/cancellation seam. Real NURBS chart admission
    # is an independent owner contract and is not asserted by this test.
    monkeypatch.setattr(preparation, '_face_chart_errors', lambda *args, **kwargs: {})
    calls = []
    def refuse(working, operands, *, policy):
        assert working is not model
        assert isinstance(working.faces[patch].surface, NurbsSurface)
        calls.append(tuple((item.kind, item.id) for item in operands))
        raise GeometryError('unclassified NURBS material arrangement')
    def never_apply(*args, **kwargs):
        pytest.fail('an incomplete plan must not be applied')
    monkeypatch.setattr(preparation, 'plan_intersections', refuse)
    monkeypatch.setattr(preparation, 'apply_intersections', never_apply)
    with pytest.raises(MeshError, match='unclassified NURBS'):
        preparation.prepare_structural_closure(model, face_ids=(patch, plate))
    assert len(calls) == 1
    assert to_dict(model) == before


def test_nurbs_planning_cancellation_preserves_source(monkeypatch):
    model, patch, plate = _crossing_patch()
    before = to_dict(model)
    # Isolate the batch refusal/cancellation seam. Real NURBS chart admission
    # is an independent owner contract and is not asserted by this test.
    monkeypatch.setattr(preparation, '_face_chart_errors', lambda *args, **kwargs: {})
    failure = RuntimeError('cancel NURBS material planning')
    planning = False
    def cancel(phase):
        if planning:
            raise failure
    def plan(working, operands, *, policy):
        nonlocal planning
        planning = True
        policy.cancellation_check()
        pytest.fail('owner cancellation must abort planning')
    monkeypatch.setattr(preparation, 'plan_intersections', plan)
    with pytest.raises(RuntimeError) as caught:
        preparation.prepare_structural_closure(model, face_ids=(patch, plate),
                                               cancellation_check=cancel)
    assert caught.value is failure
    assert planning
    assert to_dict(model) == before

def test_real_nurbs_chart_admission_returns_typed_owner_rows():
    """Actual owner path: either a qualified chart or an explicit refusal row."""
    model, patch, plate = _crossing_patch()
    before = to_dict(model)
    try:
        rows = preparation._face_chart_errors(
            model, (patch, plate), cancellation_check=None,
            expected_revision=model.revision)
        assert set(rows) == {patch, plate}
        assert all(value is None or isinstance(value, str) for value in rows.values())
    finally:
        assert to_dict(model) == before
