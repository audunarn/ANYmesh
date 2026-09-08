"""Actual seam-trim chart cavities, not a tolerance-relaxation regression."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from anymesher import MetricFieldSpec, NativeMeshingOptions
from anymesher import _frontal_transition_quality as repair
from anymesher.errors import MeshError
from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate


@pytest.fixture(params=json.loads(
    Path(__file__).with_name("frontal_transition_cavities.json").read_text(encoding="utf-8")
), ids=("notched-angle", "regular-growth", "notched-growth"))
def cavity(request):
    row = request.param
    settings = SurfaceMeshOptions(
        target_size=.4, prefer_quality_policy=True,
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
            metric_field=MetricFieldSpec.uniform(.4),
            max_topology_operations=4096,
        ), **row["settings"],
    )
    candidate = _make_candidate(
        np.asarray(row["points"], dtype=np.float64),
        np.asarray(row["triangles"], dtype=np.int64), settings=settings,
    )
    return candidate, np.asarray(row["protected"], dtype=np.int64), settings


def test_real_cavity_repair_is_bounded_deterministic_and_protected(cavity):
    candidate, protected, settings = cavity
    points, cells = candidate.points.tobytes(), candidate.triangles.tobytes()
    assert candidate.report["poor_element_ids"]
    report = {"topology_operations": 17, "insertions": 6}
    result, receipt = repair.repair_frontal_transition(candidate, protected, settings, report)
    repeated, again = repair.repair_frontal_transition(candidate, protected, settings, report)
    assert candidate.points.tobytes() == points and candidate.triangles.tobytes() == cells
    assert result.triangles.tobytes() == cells
    fixed = np.unique(protected)
    assert result.points[fixed].tobytes() == candidate.points[fixed].tobytes()
    assert result.points.tobytes() == repeated.points.tobytes() and receipt == again
    assert result.report["invalid_element_count"] == 0
    assert not result.report["poor_element_ids"]
    entry = receipt["chart_transition_repair"]
    assert entry["accepted"] and 0 < entry["trials"] <= 2048
    assert receipt["topology_operations"] == 17 + entry["trials"]
    assert report == {"topology_operations": 17, "insertions": 6}


def test_no_budget_cannot_publish_or_consume_work(cavity):
    candidate, protected, settings = cavity
    report = {"topology_operations": settings.native_options.max_topology_operations}
    result, receipt = repair.repair_frontal_transition(candidate, protected, settings, report)
    assert result is candidate
    assert receipt["topology_operations"] == report["topology_operations"]
    entry = receipt["chart_transition_repair"]
    assert entry["trials"] == 0 and entry["budget_exhausted"] and not entry["accepted"]
    assert entry["candidate_moved_nodes"] == []


def test_cancellation_propagates_without_candidate_mutation(cavity):
    candidate, protected, settings = cavity
    original = candidate.points.tobytes(), candidate.triangles.tobytes()
    error = RuntimeError("cancel chart repair")
    calls = []
    def cancel(phase):
        calls.append(phase)
        raise error
    with pytest.raises(RuntimeError) as caught:
        repair.repair_frontal_transition(
            candidate, protected, settings, {"topology_operations": 0}, cancel)
    assert caught.value is error and calls == ["native-v2 chart transition repair"]
    assert (candidate.points.tobytes(), candidate.triangles.tobytes()) == original


def test_malformed_repair_cannot_change_a_protected_node(cavity, monkeypatch):
    candidate, protected, settings = cavity
    original = repair.repair_joint_triangle_quality
    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        points = result.points.copy()
        points[int(protected[0, 0]), 0] += .01
        return replace(result, points=points)
    monkeypatch.setattr(repair, "repair_joint_triangle_quality", malformed)
    before = candidate.points.tobytes()
    with pytest.raises(MeshError, match="invalid frontal chart-transition"):
        repair.repair_frontal_transition(
            candidate, protected, settings, {"topology_operations": 0})
    assert candidate.points.tobytes() == before
