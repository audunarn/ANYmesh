"""Evaluate a detached chart candidate in both chart and owner coordinates."""
import numpy as np

from .core import MeshCore
from .quality_v2 import evaluate_quality
from .surface_mesh import _make_candidate, _quality_threshold_report


def physical_chart_quality(points, triangles, settings, chart,
                           reference_points, reference_coordinates):
    report = dict(_make_candidate(points, triangles, settings=settings).report)
    chart_report = dict(report)
    if chart_report["invalid_element_count"]:
        raise ValueError("non-positive chart topology in physical quality evaluation")
    p = np.ascontiguousarray(points, dtype=np.float64)
    reference = np.ascontiguousarray(reference_points, dtype=np.float64)
    xyz = np.asarray(chart.evaluate(p), dtype=np.float64).copy()
    if xyz.shape != (len(p), 3) or not np.isfinite(xyz).all():
        raise ValueError("physical chart quality received an invalid owner evaluation")
    if reference.shape != p.shape:
        raise ValueError("physical chart quality changed point identities")
    unchanged = np.all(p.view(np.uint64) == reference.view(np.uint64), axis=1)
    xyz[unchanged] = np.asarray(reference_coordinates, dtype=np.float64)[unchanged]
    core = MeshCore(xyz, triangles,
                    triangle_ids=np.arange(1, len(triangles) + 1, dtype=np.int64))
    physical = _quality_threshold_report(evaluate_quality(core), settings)
    cells = np.asarray(triangles, dtype=np.int64)
    lengths = np.linalg.norm(np.roll(xyz[cells], -1, axis=1) - xyz[cells], axis=2)
    perimeters = lengths.sum(axis=1)
    incidence = {}
    for i, row in enumerate(cells):
        for a, b in zip(row, np.roll(row, -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    growth_poor = set()
    maximum_growth = 1.
    for adjacent in incidence.values():
        if len(adjacent) == 2:
            a, b = adjacent
            growth = max(perimeters[a] / perimeters[b], perimeters[b] / perimeters[a])
            maximum_growth = max(maximum_growth, float(growth))
            if growth > settings.max_element_growth:
                growth_poor.update((a + 1, b + 1))
    poor = sorted(set(physical["poor_element_ids"]) | growth_poor)
    physical["poor_element_ids"] = poor
    physical["accepted"] = physical["accepted"] and not growth_poor
    physical["thresholds"]["maximum_element_growth"] = settings.max_element_growth
    physical["violation_counts"]["element_growth"] = len(growth_poor)
    physical["worst"]["maximum_element_growth"] = maximum_growth
    report["poor_element_ids"] = poor
    report["repair_element_ids"] = poor
    report["quality_violation_count"] = len(poor)
    worst = physical["worst"]
    report["max_aspect_ratio"] = worst["maximum_aspect_ratio"]
    report["min_scaled_jacobian"] = worst["minimum_scaled_jacobian"]
    report["min_angle"] = worst["minimum_angle"]
    report["max_angle"] = worst["maximum_angle"]
    report["max_element_growth"] = maximum_growth
    report["elements_above_aspect_ratio_5"] = int(np.count_nonzero(lengths.max(axis=1) / lengths.min(axis=1) > 5))
    report["elements_below_minimum_angle"] = physical["violation_counts"]["minimum_angle"]
    report["elements_above_maximum_angle"] = physical["violation_counts"]["maximum_angle"]
    report["elements_below_minimum_scaled_jacobian"] = physical["violation_counts"]["scaled_jacobian"]
    report["elements_above_maximum_growth"] = len(growth_poor)
    report["quality_scope"] = "physical_with_chart_validity"
    report["chart_quality"] = chart_report
    report["physical_quality"] = physical
    return report
