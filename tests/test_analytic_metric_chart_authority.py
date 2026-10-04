"""Small owner-bound analytic reference checks; no native triangulation."""

import numpy as np
import pytest
from dataclasses import replace

from anygeometry import (BezierDirectrix, ExtrudedSurface, GeometryModel,
                         query_trimmed_surface_charts,
                         validate_trimmed_surface_charts_binding)
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher.errors import MeshError


def owner():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (.5, 0, 0), (1, 1, 0)))
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    face = model.extrude((edge,), (0, 0, 2))[0]
    model.set_face_surface(face, ExtrudedSurface(
        BezierDirectrix(((0, 0, 0), (.5, 0, 0), (1, 1, 0))), (0, 0, 2)))
    return model, face


def test_reference_derivative_transform_and_inverse_are_owner_bound():
    model, face = owner()
    chart = AnalyticMetricChart(model, face)
    assert chart.validate_reference() is chart
    derivative = chart._reference_derivative
    np.testing.assert_allclose(chart.transform @ chart.transform.T,
                               derivative.T @ derivative, rtol=2e-14, atol=0.)
    chart.inverse[0, 0] *= 2
    with pytest.raises(MeshError, match="reference metric or inverse changed"):
        chart.validate_reference()
    chart = AnalyticMetricChart(model, face)
    chart._reference_derivative.setflags(write=True)
    chart._reference_derivative[0, 0] += .125
    with pytest.raises(MeshError, match="reference metric or inverse changed"):
        chart.validate_reference()


@pytest.mark.parametrize("operation", ("evaluate", "jacobians"))
def test_caller_rows_are_copied_before_owner_callbacks(operation):
    model, face = owner()
    submitted = np.asarray(((.2, .3),))
    armed = False

    def callback(_stage):
        if armed:
            submitted[0, 0] = .9

    chart = AnalyticMetricChart(model, face, callback)
    original = submitted.copy()
    baseline = getattr(chart, operation)(original @ chart.transform)
    submitted = original @ chart.transform
    armed = True
    measured = getattr(chart, operation)(submitted)
    np.testing.assert_array_equal(measured, baseline)
    assert submitted[0, 0] == .9


@pytest.mark.parametrize("operation", ("evaluate", "jacobians"))
def test_copied_rows_and_callback_rebase_refuse(operation):
    model, face = owner()
    holder = {}
    input_rows = np.asarray(((.2, .3), (.8, .7)))

    def callback(_stage):
        chart = holder.get("chart")
        if chart is None or holder.get("changed"):
            return
        holder["changed"] = True
        input_rows[0, 0] = .9
        transform = chart.transform.copy()
        transform[0, 0] *= 2
        chart.transform = transform
        chart.inverse = np.linalg.inv(transform)
        old = chart._authority
        chart._authority = (*old[:-2], chart.transform.tobytes(),
                            chart.inverse.tobytes())

    chart = AnalyticMetricChart(model, face, callback)
    holder["chart"] = chart
    submitted = input_rows.copy()
    with pytest.raises(MeshError, match="authority changed"):
        getattr(chart, operation)(input_rows)
    assert holder["changed"]
    assert not np.array_equal(input_rows, submitted)


@pytest.mark.parametrize("values", ("nonfinite", "singular"))
def test_invalid_public_reference_derivative_refuses(monkeypatch, values):
    import anymesher._analytic_metric_chart as module
    model, face = owner()
    first = np.asarray(((np.nan, 0., 0.),)) if values == "nonfinite" else np.asarray(((1., 0., 0.),))
    second = np.asarray(((0., 0., 2.),)) if values == "nonfinite" else np.asarray(((2., 0., 0.),))
    monkeypatch.setattr(module, "face_derivatives_many",
                        lambda *_args: (first, second))
    with pytest.raises(MeshError, match="invalid|singular"):
        AnalyticMetricChart(model, face)


def test_collection_rebase_during_reference_derivative_refuses(monkeypatch):
    import anymesher._analytic_metric_chart as module
    model, face = owner()
    original_query = module.query_trimmed_surface_charts
    original_derivatives = module.face_derivatives_many
    selected = {}

    def query(*args, **kwargs):
        result = original_query(*args, **kwargs)
        selected["binding"] = result
        return result

    def derivatives(*args, **kwargs):
        object.__setattr__(selected["binding"], "source_checksum", "rebound")
        return original_derivatives(*args, **kwargs)

    monkeypatch.setattr(module, "query_trimmed_surface_charts", query)
    monkeypatch.setattr(module, "face_derivatives_many", derivatives)
    with pytest.raises(MeshError, match="collection changed during reference selection"):
        AnalyticMetricChart(model, face)


@pytest.mark.parametrize("region_case", (False, True))
def test_same_revision_support_and_same_binding_rebase_refuse(region_case):
    if region_case:
        from anymesher._material_region_binding import prepare_material_regions
        from test_material_region_integration import controls, split_model
        model, wall, descendants, _ = split_model()
        bindings = prepare_material_regions(
            model, descendants, {wall: descendants}, order="linear",
            native_options=controls())
        region_binding = bindings[min(descendants)]
        face = region_binding.representative
        chart = AnalyticMetricChart(model, face, region_binding=region_binding)
        faces = descendants
    else:
        model, face = owner()
        chart = AnalyticMetricChart(model, face)
        faces = (face,)
    revision = model.revision
    old_reference = chart._reference_derivative.copy()
    for face_id in faces:
        original = model.faces[face_id]
        surface = original.surface
        changed = replace(surface,
                          vector=tuple(2 * np.asarray(surface.vector)))
        model._faces[face_id] = replace(original, surface=changed)
    assert model.revision == revision
    if region_case:
        refreshed = prepare_material_regions(
            model, descendants, {wall: descendants}, order="linear",
            native_options=controls())[face]
        old_binding = chart.binding
        object.__setattr__(old_binding, "source", refreshed.collection.source)
        object.__setattr__(old_binding, "regions", refreshed.collection.regions)
        object.__setattr__(region_binding, "region", refreshed.region)
        region_binding.validate()
    else:
        refreshed = query_trimmed_surface_charts(model, (face,))
        old_binding = chart.binding
        object.__setattr__(old_binding, "source_checksum", refreshed.source_checksum)
        object.__setattr__(old_binding, "charts", refreshed.charts)
        validate_trimmed_surface_charts_binding(model, old_binding)
    assert chart.binding is old_binding
    from anygeometry import face_derivatives_many
    du, dv = face_derivatives_many(model, face, ((.5, .5),))
    assert not np.array_equal(old_reference, np.column_stack((du[0], dv[0])))
    with pytest.raises(MeshError, match="original collection or region changed"):
        chart.validate_reference()
    for operation in (chart.evaluate, chart.jacobians):
        with pytest.raises(MeshError, match="original collection or region changed"):
            operation(np.asarray(((.4, .5),)) @ chart.transform)
