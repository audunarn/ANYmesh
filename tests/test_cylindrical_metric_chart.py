"""First-increment physical chart tests; no cylinder meshing activation."""

from dataclasses import replace

import anygeometry as owner
import numpy as np
import pytest
from anygeometry.generators import cylinder

from anymesher._cylindrical_chart import CylindricalMetricChart
from anymesher.charts import FaceChart
from anymesher.errors import MeshError
from anymesher.metric import pullback_metric
from anymesher.native_v2 import _pullback_spatial_metrics


def make_chart(radius=.5, height=2.):
    model = cylinder(
        radius=radius, height=height, circumferential_segments=12,
        origin=(0., 0., 0.), axis=(0., 0., 1.), radial_direction=(1., 0., 0.),
        longitudinal_spacing=height, ring_spacing=height,
    )
    face = tuple(model.group("shell"))[0]
    return model, CylindricalMetricChart.from_geometry(model, face.id)


@pytest.mark.parametrize("radius,height", [(.5, 2.), (2., .5), (.125, 4.)])
def test_physical_chart_and_owner_jacobians(radius, height, monkeypatch):
    model, chart = make_chart(radius, height)
    revision = model.revision
    uv = np.array(((0., 0.), (.25, .4), (.75, .6), (1., 1.)))
    points = chart.to_chart(uv)
    np.testing.assert_allclose(chart.to_parameters(points), uv, rtol=0., atol=1.e-15)
    np.testing.assert_array_equal(chart.evaluate(points), chart.face_chart.evaluate(uv))
    analytic = owner.face_derivatives_many
    calls = []

    def observe_actual_owner(geometry, face_id, parameters):
        du, dv = analytic(geometry, face_id, parameters)
        calls.append((geometry, face_id, parameters.copy(), du.copy(), dv.copy()))
        return du, dv

    def forbid_fallback(*args, **kwargs):
        raise AssertionError("cylinder derivatives must use the analytic owner API")

    monkeypatch.setattr(owner, "face_derivatives_many", observe_actual_owner)
    monkeypatch.setattr(FaceChart, "derivatives", forbid_fallback)
    jacobians = chart.jacobians(points)
    assert len(calls) == 1
    geometry, face_id, parameters, du, dv = calls[0]
    assert geometry is model and face_id == chart.face_chart.face_id
    assert du.shape == dv.shape == (len(uv), 3)
    np.testing.assert_array_equal(parameters, chart.to_parameters(points))
    np.testing.assert_array_equal(
        jacobians,
        np.stack((du / chart.circumferential_length, dv / chart.axial_length), axis=2),
    )
    gram = np.einsum("nki,nkj->nij", jacobians, jacobians)
    np.testing.assert_allclose(gram, np.broadcast_to(np.eye(2), gram.shape), rtol=1.e-12, atol=1.e-12)
    assert not np.array_equal(jacobians[0], jacobians[-1])
    assert model.revision == revision


def test_stale_chart_rejects_without_owner_evaluation():
    model, chart = make_chart()
    with model.transaction():
        model.add_points([(3., 3., 3.)])
    with pytest.raises(MeshError, match="stale"):
        chart.evaluate([[0., 0.]])


def test_distinct_owner_parameterization_rejected_before_metric_use(monkeypatch):
    model, chart = make_chart()
    face_id = chart.face_chart.face_id
    face = model.faces[face_id]
    # Even geometrically equivalent but separately owned parameterizations are
    # outside this first increment's deliberately narrow supported family.
    alternate = replace(face.surface)
    assert alternate is not face.surface
    uv = np.array(((.25, .4), (.75, .6)))
    with model.transaction():
        model._put_entity("face", replace(face, parameterization=alternate))
    assert model.faces[face_id].parameterization is alternate
    owner_chart = FaceChart(model, face_id)
    np.testing.assert_allclose(
        owner_chart.evaluate(uv), np.array([alternate.evaluate(*row) for row in uv]),
        rtol=1.e-12, atol=1.e-12,
    )
    consumed = []

    def forbid_derivatives(*args, **kwargs):
        consumed.append(True)
        raise AssertionError("a rejected chart must not consume a metric")

    monkeypatch.setattr(FaceChart, "derivatives", forbid_derivatives)
    monkeypatch.setattr(owner, "face_derivatives_many", forbid_derivatives)
    with pytest.raises(MeshError, match="distinct face parameterization"):
        CylindricalMetricChart.from_geometry(model, face_id)
    assert consumed == []


@pytest.mark.parametrize("points", [[[-1., 0.]], [[np.inf, 0.]], [[0., 0., 0.]]])
def test_invalid_chart_coordinates_fail_closed(points):
    _, chart = make_chart()
    with pytest.raises(MeshError):
        chart.evaluate(points)


def test_constant_and_batched_pullback_preserve_planar_oracle_bytes():
    points = np.zeros((3, 2))
    tensors = np.array([np.diag((1., 2., 3.)), np.eye(3) * 4., np.eye(3)])
    jacobian = np.array(((2., 0.), (0., 3.), (0., 0.)))
    expected = np.array([pullback_metric(tensor, jacobian) for tensor in tensors])
    constant = _pullback_spatial_metrics(tensors, points, jacobian)
    batched = _pullback_spatial_metrics(tensors, points, np.broadcast_to(jacobian, (3, 3, 2)))
    assert constant.tobytes() == expected.tobytes() == batched.tobytes()


def test_position_dependent_cylinder_pullback_matches_owner_rows():
    _, chart = make_chart()
    points = chart.to_chart([[.1, .2], [.8, .9]])
    tensors = np.array([np.diag((1., 2., 3.)), np.diag((4., 2., 1.))])
    expected = np.array([pullback_metric(t, j) for t, j in zip(tensors, chart.jacobians(points))])
    result = _pullback_spatial_metrics(tensors, points, chart.jacobians)
    assert result.tobytes() == expected.tobytes()


def test_callable_jacobian_batches_are_bounded_and_ordered():
    points = np.column_stack((np.arange(4097, dtype=float), np.zeros(4097)))
    tensors = np.broadcast_to(np.eye(3), (len(points), 3, 3))
    calls = []

    def binding(rows):
        calls.append(rows[:, 0].copy())
        return np.broadcast_to(np.eye(3)[:, :2], (len(rows), 3, 2))

    result = _pullback_spatial_metrics(tensors, points, binding)
    assert [len(rows) for rows in calls] == [4096, 1]
    np.testing.assert_array_equal(np.concatenate(calls), points[:, 0])
    np.testing.assert_array_equal(result, np.broadcast_to(np.eye(2), result.shape))


@pytest.mark.parametrize("binding", [
    lambda points: np.zeros((len(points), 2, 2)),
    lambda points: np.full((len(points), 3, 2), np.nan),
    lambda points: np.zeros((len(points), 3, 2)),
])
def test_malformed_or_singular_jacobians_are_rejected(binding):
    with pytest.raises(MeshError):
        _pullback_spatial_metrics(np.eye(3)[None], np.zeros((1, 2)), binding)


def test_binding_error_and_cancellation_propagate():
    failure = RuntimeError("owner failure")

    def fail(*args):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        _pullback_spatial_metrics(np.eye(3)[None], np.zeros((1, 2)), fail)
    assert caught.value is failure
    with pytest.raises(RuntimeError) as caught:
        _pullback_spatial_metrics(
            np.eye(3)[None], np.zeros((1, 2)), np.eye(3)[:, :2],
            cancellation_check=fail, cancellation_interval=1,
        )
    assert caught.value is failure
