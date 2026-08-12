from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel
from anymesher.charts import FaceChart, MetricField, metric_edge_lengths


def _unit_chart() -> FaceChart:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    return FaceChart(geometry, geometry.add_plate(vertices))


def test_scalar_geometry_chart_evaluates_differentiates_and_projects() -> None:
    chart = _unit_chart()
    uv = np.asarray(((0.25, 0.75), (1.0, 0.0)))

    assert chart.evaluate(uv) == pytest.approx(
        np.asarray(((0.25, 0.75, 0.0), (1.0, 0.0, 0.0)))
    )
    du, dv = chart.derivatives(uv)
    assert du == pytest.approx(np.tile((1.0, 0.0, 0.0), (2, 1)))
    assert dv == pytest.approx(np.tile((0.0, 1.0, 0.0), (2, 1)))
    assert chart.normals(uv) == pytest.approx(
        np.tile((0.0, 0.0, 1.0), (2, 1))
    )

    projection = chart.project(((0.25, 0.75, 2.0),))
    assert projection.points == pytest.approx(np.asarray(((0.25, 0.75, 0.0),)))
    assert projection.uv == pytest.approx(np.asarray(((0.25, 0.75),)))
    assert projection.distances == pytest.approx((2.0,))


def test_chart_prefers_forthcoming_paired_batch_apis() -> None:
    class BatchGeometry:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def evaluate_faces(self, face_ids, uv):
            self.calls.append("evaluate")
            assert np.all(face_ids == 7)
            return np.column_stack((uv, np.full(len(uv), 7.0)))

        def face_derivatives(self, face_ids, uv):
            self.calls.append("derivatives")
            assert np.all(face_ids == 7)
            return (
                np.tile((1.0, 0.0, 0.0), (len(uv), 1)),
                np.tile((0.0, 1.0, 0.0), (len(uv), 1)),
            )

        def face_normals(self, face_ids, uv):
            self.calls.append("normals")
            assert np.all(face_ids == 7)
            return np.tile((0.0, 0.0, 1.0), (len(uv), 1))

        def project_faces(self, face_ids, points):
            self.calls.append("project")
            assert np.all(face_ids == 7)
            projected = points.copy()
            projected[:, 2] = 7.0
            return projected, points[:, :2], np.abs(points[:, 2] - 7.0)

    geometry = BatchGeometry()
    chart = FaceChart(geometry, 7)
    uv = np.asarray(((0.1, 0.2), (0.8, 0.6)))

    assert chart.evaluate(uv)[:, 2] == pytest.approx((7.0, 7.0))
    chart.derivatives(uv)
    chart.normals(uv)
    projection = chart.project(((0.1, 0.2, 9.0), (0.8, 0.6, 5.0)))

    assert projection.distances == pytest.approx((2.0, 2.0))
    assert geometry.calls == ["evaluate", "derivatives", "normals", "project"]


def test_size_direction_field_builds_metric_and_measures_edges() -> None:
    chart = _unit_chart()
    uv = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    field = MetricField(
        lambda parameters, xyz: np.full(len(xyz), 0.5),
        direction=(1.0, 0.0, 0.0),
        aspect_ratio=2.0,
    )

    metrics = field(chart, uv)
    assert metrics == pytest.approx(np.tile(np.diag((1.0, 4.0)), (3, 1, 1)))
    lengths = metric_edge_lengths(uv, ((0, 1), (0, 2)), metrics)
    assert lengths == pytest.approx((1.0, 2.0))
    assert not metrics.flags.writeable
