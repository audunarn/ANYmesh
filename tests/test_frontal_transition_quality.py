"""Actual seam-trim chart cavities, not a tolerance-relaxation regression."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from anymesher import MetricFieldSpec, NativeMeshingOptions
from anymesher import _frontal_transition_quality as repair
from anymesher.errors import MeshError
from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate, _physical_quality_candidate


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
    assert entry['candidate_adopted'] and entry['quality_satisfied']
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
    assert not entry['candidate_adopted'] and not entry['quality_satisfied']


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


def small_owner_candidate(*, trials=31):
    points = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.2, .35), (.75, .6)))
    cells = np.asarray(((0, 1, 4), (1, 5, 4), (1, 2, 5), (2, 3, 5), (3, 4, 5), (3, 0, 4)))
    protected = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)))
    owner = lambda rows: np.column_stack((rows, .25*rows[:, 0]**2 + .125*rows[:, 1]**2))
    settings = SurfaceMeshOptions(min_angle=30., prefer_quality_policy=True,
        native_options=NativeMeshingOptions(max_topology_operations=trials))
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings), settings, owner)
    return candidate, protected, settings, owner


def test_real_partial_owner_progress_is_opt_in_and_still_unsatisfied():
    candidate, protected, settings, owner = small_owner_candidate()
    report = dict(topology_operations=0)
    default, default_report = repair.repair_frontal_transition(candidate, protected, settings, report,
                                                               evaluate_coordinates=owner)
    result, updated = repair.repair_frontal_transition(candidate, protected, settings, report,
        evaluate_coordinates=owner, allow_partial_progress=True)
    before, after = candidate.report, updated['chart_transition_repair']['proposed_quality']
    print('PARTIAL_OWNER', json.dumps({stage: {name: row[name] for name in
          ('violation_counts', 'quality_violation_count', 'elements_above_maximum_growth', 'max_element_growth')}
          for stage, row in (('before', before), ('after', after))}, sort_keys=True))
    assert default is candidate and not default_report['chart_transition_repair']['candidate_adopted']
    entry = updated['chart_transition_repair']
    assert entry['candidate_adopted'] and not entry['accepted'] and not entry['quality_satisfied']
    assert result is not candidate and result.report == entry['proposed_quality'] == entry['final_quality']
    assert result.report['poor_element_ids'] and result.report['invalid_element_count'] == 0
    assert entry['budget_exhausted'] and entry['trials'] == updated['topology_operations'] == 31
    assert entry['initial_penalty'] > entry['final_penalty']
    assert default_report['chart_transition_repair']['proposed_quality'] == entry['proposed_quality']
    assert report == dict(topology_operations=0)


def test_increased_other_physical_threshold_refuses_proposal_but_keeps_evidence(monkeypatch):
    from anymesher._joint_triangle_repair import JointTriangleRepair
    candidate, protected, settings, owner = small_owner_candidate()
    points = candidate.points.copy()
    points[4] = (.02, .35)
    # A controlled optimizer receipt tests admission independently of its merit
    # search; the helper must certify this proposal with the actual evaluator.
    proposal = JointTriangleRepair(points, 7, (4,), .75, .5, False)
    monkeypatch.setattr(repair, 'repair_joint_triangle_quality', lambda *args, **kwargs: proposal)
    result, report = repair.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=3), evaluate_coordinates=owner, allow_partial_progress=True)
    entry = report['chart_transition_repair']
    assert entry['proposed_quality']['violation_counts']['scaled_jacobian'] > candidate.report['violation_counts']['scaled_jacobian']
    assert result is candidate and entry['final_quality'] == candidate.report
    assert not entry['candidate_adopted'] and not entry['quality_satisfied'] and not entry['accepted']
    assert entry['proposed_moved_nodes'] == [4] and entry['candidate_moved_nodes'] == []
    assert entry['initial_penalty'] == entry['final_penalty'] == .75
    assert entry['proposed_penalty'] == .5 and entry['selected_penalty'] is None
    assert entry['trials'] == 7 and report['topology_operations'] == 10


def test_pinned_source_row_is_fixed_in_every_owner_probe_and_result():
    candidate, protected, settings, owner = small_owner_candidate()
    seen = []
    def evaluate(rows):
        assert rows[4].tobytes() == candidate.points[4].tobytes()
        seen.append(rows.copy())
        return owner(rows)
    result, report = repair.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), evaluate_coordinates=evaluate, allow_partial_progress=True,
        pinned_nodes=(np.int64(4), 4))
    assert seen and result.points[4].tobytes() == candidate.points[4].tobytes()
    entry = report['chart_transition_repair']
    assert 4 not in entry['candidate_moved_nodes'] and 4 not in entry['proposed_moved_nodes']
    assert report['topology_operations'] == entry['trials'] <= 31


@pytest.mark.parametrize('node', (-1, 6, True, np.bool_(False), 1.0, np.nan, np.inf, '4'))
def test_invalid_pin_rejects_before_early_return_or_owner_work(node):
    from anymesher._joint_triangle_repair import repair_joint_triangle_quality
    candidate, protected, settings, owner = small_owner_candidate()
    settings = replace(settings, native_options=NativeMeshingOptions(max_topology_operations=1))
    with pytest.raises(ValueError, match='invalid pinned'):
        repair.repair_frontal_transition(candidate, protected, settings,
            dict(topology_operations=1), pinned_nodes=(node,), evaluate_coordinates=owner)
    with pytest.raises(ValueError, match='invalid pinned'):
        repair_joint_triangle_quality(candidate.points, candidate.triangles, protected, range(6),
            min_angle=30., max_growth=1.5, max_trials=0, pinned_nodes=(node,))


def test_malformed_optimizer_cannot_move_pin(monkeypatch):
    candidate, protected, settings, owner = small_owner_candidate()
    original = repair.repair_joint_triangle_quality
    def malformed(*args, **kwargs):
        result = original(*args, **kwargs)
        points = result.points.copy()
        points[4, 0] += .01
        return replace(result, points=points)
    monkeypatch.setattr(repair, 'repair_joint_triangle_quality', malformed)
    with pytest.raises(MeshError, match='invalid frontal chart-transition'):
        repair.repair_frontal_transition(candidate, protected, settings, dict(topology_operations=0),
            evaluate_coordinates=owner, pinned_nodes=(4,), allow_partial_progress=True)


def test_pins_are_removed_before_twelve_node_selection_cap():
    from anymesher._joint_triangle_repair import repair_joint_triangle_quality
    points = np.asarray([(x/5., y/5.) for y in range(6) for x in range(6)])
    points[7] = (.01, .01)
    cells = np.asarray([cell for y in range(5) for x in range(5)
                        for cell in ((6*y+x, 6*y+x+1, 6*y+x+7),
                                     (6*y+x, 6*y+x+7, 6*y+x+6))])
    pinned = (7, 8, 9, 10, 13, 14, 15, 16, 19, 20, 21, 22)
    seen = []
    def owner(rows):
        assert rows[list(pinned)].tobytes() == points[list(pinned)].tobytes()
        seen.append(rows.copy())
        return np.column_stack((rows, np.zeros(len(rows))))
    result = repair_joint_triangle_quality(points, cells, (), range(len(cells)),
        min_angle=30., max_growth=1.5, max_trials=1, pinned_nodes=pinned, evaluate_coordinates=owner)
    assert result.trials == 1 and result.budget_exhausted
    assert result.initial_penalty > 0
    assert any(row[25].tobytes() != points[25].tobytes() for row in seen)
    assert result.points[list(pinned)].tobytes() == points[list(pinned)].tobytes()


def test_no_owner_opt_in_keeps_generic_complete_success_behavior(cavity):
    candidate, protected, settings = cavity
    first, one = repair.repair_frontal_transition(candidate, protected, settings, dict(topology_operations=0))
    second, two = repair.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), allow_partial_progress=True)
    assert first.points.tobytes() == second.points.tobytes() and one == two
    assert one['chart_transition_repair']['accepted']


def test_clean_early_receipt_is_satisfied_without_adoption():
    candidate, protected, settings, _ = small_owner_candidate()
    settings = replace(settings, min_angle=15.)
    candidate = _make_candidate(candidate.points, candidate.triangles, settings=settings)
    assert not candidate.report['poor_element_ids']
    result, report = repair.repair_frontal_transition(candidate, protected, settings, dict(topology_operations=0))
    entry = report['chart_transition_repair']
    assert result is candidate and entry['quality_satisfied'] and not entry['candidate_adopted']
    assert not entry['accepted'] and entry['trials'] == 0 and entry['proposed_quality'] is None


def test_owner_opt_in_scores_missing_baseline_before_no_budget_receipt():
    physical, protected, settings, owner = small_owner_candidate()
    chart = _make_candidate(physical.points, physical.triangles, settings=settings)
    assert 'violation_counts' not in chart.report
    calls = []
    def evaluate(rows):
        calls.append(rows.copy())
        return owner(rows)
    result, report = repair.repair_frontal_transition(chart, protected, settings,
        dict(topology_operations=31), evaluate_coordinates=evaluate, allow_partial_progress=True)
    entry = report['chart_transition_repair']
    assert len(calls) == 1 and result.report == physical.report
    assert entry['initial_quality'] == entry['final_quality'] == physical.report
    assert not entry['quality_satisfied'] and not entry['candidate_adopted']
    assert entry['trials'] == 0 and report['topology_operations'] == 31


@pytest.mark.parametrize('failure', ('cancel', 'stale'))
def test_owner_partial_path_propagates_errors_without_point_publication(failure):
    from anygeometry import GeometryError
    candidate, protected, settings, owner = small_owner_candidate()
    error = GeometryError('stale owner proposal') if failure == 'stale' else RuntimeError('cancel partial owner')
    before = candidate.points.tobytes(), candidate.triangles.tobytes()
    def evaluate(rows):
        if failure == 'stale' and rows.tobytes() != before[0]:
            raise error
        return owner(rows)
    def cancel(phase):
        if failure == 'cancel':
            raise error
    with pytest.raises(type(error)) as caught:
        repair.repair_frontal_transition(candidate, protected, settings, dict(topology_operations=0),
            cancel, evaluate_coordinates=evaluate, allow_partial_progress=True)
    assert caught.value is error and (candidate.points.tobytes(), candidate.triangles.tobytes()) == before


@pytest.mark.parametrize('analytic', (False, True))
def test_surface_opts_in_only_bound_analytic_evaluate_and_forwards_pins(monkeypatch, analytic):
    import anymesher.surface_mesh as surface
    import anymesher._physical_t3_refinement as physical
    from test_joint_triangle_owner_batches import analytic_chart
    from anygeometry import to_dict
    model, chart = analytic_chart()
    before = to_dict(model)
    point = np.asarray(((.37, .43),))
    seen = []
    def capture(candidate, protected, settings, report, *args, **kwargs):
        pins = kwargs['pinned_nodes']
        seen.append(kwargs['allow_partial_progress'])
        assert len(pins) == 1 and candidate.points[list(pins)].tobytes() == point.tobytes()
        return candidate, dict(report, chart_transition_repair={})
    monkeypatch.setattr(repair, 'repair_frontal_transition', capture)
    monkeypatch.setattr(physical, 'refine_physical_candidate',
                        lambda candidate, triangulation, settings, report, *args:
                        (candidate, triangulation, report))
    surface.mesh_planar_surface(((0., 0.), (1., 0.), (1., 1.), (0., 1.)), interior_points=point,
        options=SurfaceMeshOptions(target_size=1., recombine=False, backend='python',
            min_angle=45., native_options=NativeMeshingOptions(point_placement='frontal_delaunay',
                metric_mode='isotropic_spatial', max_insertions=1, max_topology_operations=8)),
        _metric_to_physical=chart.evaluate if analytic else lambda rows:chart.evaluate(rows),
        _metric_jacobian=np.asarray(((1., 0.), (0., 1.), (0., 0.))),
        _preserve_spatial_refinement=True)
    assert seen == [analytic] and to_dict(model) == before


@pytest.mark.parametrize('allow_partial,has_owner', ((False, False), (False, True), (True, False), (True, True)))
def test_helper_enables_physical_priority_only_for_owner_partial_opt_in(monkeypatch, allow_partial, has_owner):
    candidate, protected, settings, owner = small_owner_candidate(trials=1)
    seen = []
    original = repair.repair_joint_triangle_quality
    def capture(*args, **kwargs):
        seen.append(kwargs['physical_priority'])
        return original(*args, **kwargs)
    monkeypatch.setattr(repair, 'repair_joint_triangle_quality', capture)
    _, report = repair.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), evaluate_coordinates=owner if has_owner else None,
        allow_partial_progress=allow_partial)
    entry = report['chart_transition_repair']
    enabled = allow_partial and has_owner
    assert seen == [enabled]
    assert entry['priority_mode'] == ('physical_severity' if enabled else 'node_id')
    assert entry['selected_nodes'] == entry['root_nodes'] + entry['neighbour_nodes']
    assert entry['selected_nodes'] and entry['trials'] == report['topology_operations'] == 1
