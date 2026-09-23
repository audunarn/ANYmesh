from types import SimpleNamespace

import numpy as np
import pytest

from anymesher._physical_chart_quality import physical_chart_quality
from anymesher.surface_mesh import SurfaceMeshOptions


def test_chart_accepted_triangle_can_fail_physical_cylinder_angle():
    points = np.array([[0., 0.], [0., .4], [.2312, .4]])
    triangles = np.array([[0, 2, 1]])
    def evaluate(p):
        return np.column_stack((np.cos(p[:, 0]), np.sin(p[:, 0]), p[:, 1]))
    report = physical_chart_quality(points, triangles, SurfaceMeshOptions(),
                                    SimpleNamespace(evaluate=evaluate), points, evaluate(points))
    assert report["chart_quality"]["quality_violation_count"] == 0
    assert report["physical_quality"]["violation_counts"]["minimum_angle"] == 1
    assert report["poor_element_ids"] == [1]
    assert report["min_angle"] < 30


def test_unchanged_nodes_use_reference_coordinates_exactly():
    points = np.array([[0., 0.], [1., 0.], [.5, .8]])
    triangles = np.array([[0, 1, 2]])
    reference = np.column_stack((points, np.zeros(3)))
    chart = SimpleNamespace(evaluate=lambda p: np.column_stack((100*p[:, 0], p[:, 1], np.zeros(len(p)))))
    report = physical_chart_quality(points, triangles, SurfaceMeshOptions(), chart, points, reference)
    assert report["physical_quality"]["accepted"]
    assert report["quality_violation_count"] == 0


def test_physical_policy_does_not_apply_chart_angle_as_second_quality_gate():
    points = np.array([[0., 0.], [.4, 0.], [.2, .1152]])
    triangles = np.array([[0, 1, 2]])
    def evaluate(p):
        return np.column_stack((np.cos(p[:, 0]), np.sin(p[:, 0]), p[:, 1]))
    report = physical_chart_quality(points, triangles, SurfaceMeshOptions(),
                                    SimpleNamespace(evaluate=evaluate), points, evaluate(points))
    assert report["chart_quality"]["min_angle"] < 30
    assert report["min_angle"] >= 30 and report["physical_quality"]["accepted"]
    assert report["quality_violation_count"] == 0
    assert report["quality_scope"] == "physical_with_chart_validity"


def test_growth_is_measured_from_physical_triangle_perimeters():
    points = np.array([[0., 0.], [1., 0.], [.5, np.sqrt(3)/2], [.5, -2.]])
    triangles = np.array([[0, 1, 2], [1, 0, 3]])
    evaluate = lambda p: np.column_stack((p, np.zeros(len(p))))
    report = physical_chart_quality(points, triangles, SurfaceMeshOptions(min_angle=20),
                                    SimpleNamespace(evaluate=evaluate), points, evaluate(points))
    assert report["max_element_growth"] == pytest.approx((1 + 2*np.sqrt(4.25))/3)
    assert not report["physical_quality"]["accepted"] and report["poor_element_ids"] == [1, 2]


def test_chart_inversion_remains_a_hard_failure():
    points = np.array([[0., 0.], [0., 1.], [1., 0.]])
    with pytest.raises(ValueError, match="non-positive chart"):
        physical_chart_quality(points, np.array([[0, 1, 2]]), SurfaceMeshOptions(),
                               SimpleNamespace(evaluate=lambda p: np.column_stack((p, np.zeros(len(p))))),
                               points, np.column_stack((points, np.zeros(len(points)))))
