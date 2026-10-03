"""Guarded advancement changes only explicitly admitted owner line searches."""
import numpy as np
import pytest

import anymesher._frontal_transition_quality as frontal
from anymesher._joint_triangle_repair import JointTriangleRepair, repair_joint_triangle_quality
from anymesher._physical_t3_refinement import _alternative_progress
from anymesher.surface_mesh import _make_candidate, _physical_quality_candidate_from_xyz
from test_joint_triangle_owner_batches import fixture, physical, RecordedOwner, assert_same_result
from test_frontal_transition_quality import small_owner_candidate


def search(owner, callback=None, *, budget=31, batch=1, guarded=True, cancel=None, pins=()):
    points, cells, protected = fixture()
    return repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
        min_angle=30., max_growth=1.5, max_trials=budget,
        evaluate_coordinates=owner, candidate_callback=callback,
        coordinate_batch_size=batch, line_search_admission=guarded,
        cancellation_check=cancel, pinned_nodes=pins)


@pytest.mark.parametrize('batch', (1, 8))
def test_refused_armijo_continues_to_admissible_backtrack(batch):
    points, _, _ = fixture()
    seen = []
    def admit(x, xyz, merit, ordinal):
        admitted = np.max(np.linalg.norm(x - points, axis=1)) <= .06
        seen.append((ordinal, x.copy(), merit, bool(admitted)))
        return bool(admitted)
    result = search(physical, admit, batch=batch)
    # Eight initial gradient probes precede the scalar line-search powers.
    line = seen[8:]
    assert not line[0][3] and line[0][2] < result.initial_penalty
    assert result.line_search_refusals >= 1 and result.line_search_admissions >= 1
    first_admitted = next(i for i, entry in enumerate(line) if entry[3])
    assert first_admitted > 0
    first_delta = line[0][1] - points
    assert np.allclose(line[first_admitted][1] - points, first_delta * .5 ** first_admitted,
                       rtol=0., atol=2e-16)
    assert result.line_search_advancements >= 1
    assert result.admission_checks == len(seen)
    assert result.admission_refusals == sum(not row[3] for row in seen)
    assert result.points[:4].tobytes() == points[:4].tobytes()


@pytest.mark.parametrize('budget', (1, 3, 7, 8, 9, 11, 31))
def test_scalar_batch_same_ordinals_coordinates_and_budget(budget):
    events = []
    results = []
    owners = []
    for batch in (1, 8):
        trace = []
        owner = RecordedOwner(6)
        def observe(x, xyz, merit, ordinal):
            trace.append((ordinal, x.tobytes(), xyz.tobytes(), merit))
            assert xyz.tobytes() == physical(x).tobytes()
            return ordinal % 3 != 0
        results.append(search(owner, observe, budget=budget, batch=batch))
        events.append(trace)
        owners.append(owner)
    assert_same_result(*results)
    assert events[0] == events[1]
    assert results[0].trials <= budget
    assert results[0].admission_checks == len(events[0])
    assert [x.tobytes() for x in owners[0].candidates] == [x.tobytes() for x in owners[1].candidates]


@pytest.mark.parametrize('budget', (9, 10, 19, 20))
def test_all_refused_respects_existing_trial_and_backtrack_limits(budget):
    result = search(physical, lambda *args: False, budget=budget)
    assert result.line_search_admissions == result.line_search_advancements == 0
    assert result.admission_refusals == result.admission_checks
    assert result.trials == min(budget, 20)  # Eight probes plus twelve backtracks.
    assert result.budget_exhausted == (budget <= 20)


def test_admission_is_independent_of_retained_best_merit():
    rim = np.column_stack((np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)))
    points = np.vstack((rim, (.6, .1)))
    cells = np.asarray([(i, (i+1) % 6, 6) for i in range(6)])
    edges = np.asarray([(i, (i+1) % 6) for i in range(6)])
    merits = []
    def owner(rows):
        xyz = np.column_stack((rows, np.zeros(len(rows))))
        for node in range(6, len(rows), 7):
            x = rows[node, 0]
            if .6 < x < .6001:
                xyz[node, 0] = .5  # Best admitted gradient probe.
            elif x >= .6001:
                xyz[node, 0] = .55  # Feasible Armijo step, worse than that probe.
        return xyz
    def observe(x, xyz, merit, ordinal):
        merits.append(merit)
        return True
    result = repair_joint_triangle_quality(points, cells, edges, range(6), min_angle=40.,
        max_growth=2., max_trials=10, evaluate_coordinates=owner,
        candidate_callback=observe, line_search_admission=True)
    assert 0. < min(merits[:4]) < merits[4] < result.initial_penalty
    assert result.final_penalty == min(merits[:4])
    assert result.line_search_admissions >= 1 and result.line_search_advancements >= 1


@pytest.mark.parametrize('batch', (1, 8))
def test_refused_zero_gradient_does_not_suppress_line_search(batch):
    rim = np.column_stack((np.cos(np.arange(6) * np.pi / 3), np.sin(np.arange(6) * np.pi / 3)))
    points = np.vstack((rim, (.6, .1)))
    cells = np.asarray([(i, (i+1) % 6, 6) for i in range(6)])
    edges = np.asarray([(i, (i+1) % 6) for i in range(6)])
    def owner(rows):
        xyz = np.column_stack((rows, np.zeros(len(rows))))
        # A discontinuous independent fixture creates an exact zero only at
        # plus-x gradient probes; its physical admission is deliberately false.
        centers = np.arange(6, len(rows), 7)
        selected = centers[(rows[centers, 0] > .6) & (rows[centers, 0] < .6001)]
        xyz[selected, :2] = 0.
        return xyz
    def run(guarded):
        return repair_joint_triangle_quality(points, cells, edges, range(6),
            min_angle=30., max_growth=2., max_trials=8, evaluate_coordinates=owner,
            candidate_callback=lambda *args: False, line_search_admission=guarded,
            coordinate_batch_size=batch)
    legacy, guarded = run(False), run(True)
    assert legacy.final_penalty == guarded.final_penalty == 0.
    assert legacy.trials == 4 and guarded.trials == 8
    assert guarded.refused_zero_trials >= 1 and guarded.line_search_trials > 0
    assert not guarded.line_search_admissions and guarded.budget_exhausted


def test_initial_raw_zero_does_not_skip_guarded_observations():
    points, cells, protected = fixture()
    def run(guarded):
        return repair_joint_triangle_quality(points, cells, protected, range(6), min_angle=1.,
            max_growth=10., max_trials=9, evaluate_coordinates=physical,
            candidate_callback=lambda *args: False, line_search_admission=guarded)
    legacy, guarded = run(False), run(True)
    assert legacy.initial_penalty == guarded.initial_penalty == 0.
    assert legacy.trials == 0 and guarded.trials == 8
    assert guarded.refused_zero_trials == guarded.admission_refusals == 8
    assert not guarded.line_search_advancements


@pytest.mark.parametrize('batch', (1, 8))
def test_invalid_charged_rows_leave_ordinal_gaps(batch):
    points, cells, edges = fixture()
    points[4] = (1e-10, .35)
    trace = []
    def observe(x, xyz, merit, ordinal):
        trace.append(ordinal)
        return True
    result = repair_joint_triangle_quality(points, cells, edges, range(6), min_angle=30.,
        max_growth=1.5, max_trials=3, evaluate_coordinates=physical,
        candidate_callback=observe, coordinate_batch_size=batch, line_search_admission=True)
    assert trace == [1, 3] and result.trials == 3 and result.admission_checks == 2


@pytest.mark.parametrize('mode', (None, 0, 1, 'yes'))
def test_invalid_mode_rejected_before_owner_query(mode):
    owner = RecordedOwner(6)
    with pytest.raises(ValueError, match='invalid joint'):
        search(owner, lambda *args: True, guarded=mode)
    assert not owner.row_counts


@pytest.mark.parametrize('owner, callback', ((None, lambda *args: True), (physical, None)))
def test_admission_requires_owner_and_callback(owner, callback):
    with pytest.raises(ValueError, match='invalid joint'):
        search(owner, callback)


@pytest.mark.parametrize('answer', (None, 0, 1, np.bool_(True)))
def test_admission_receipt_requires_python_bool(answer):
    with pytest.raises(ValueError, match='admission receipt'):
        search(physical, lambda *args: answer)


def test_default_callback_returns_ignored_and_cancellation_sequence_unchanged():
    results, owners, cancellations = [], [], []
    for answer in (None, False, True, {'unexpected': 'ignored'}):
        owner = RecordedOwner(6)
        checkpoints = []
        def cancel():
            checkpoints.append(len(owner.candidates))
        results.append(search(owner, lambda *args: answer, guarded=False, cancel=cancel))
        owners.append(owner)
        cancellations.append(checkpoints)
    for result in results[1:]:
        assert_same_result(results[0], result)
    assert all(c == cancellations[0] for c in cancellations)
    assert all([x.tobytes() for x in o.candidates] == [x.tobytes() for x in owners[0].candidates] for o in owners)
    assert not results[0].line_search_admission and not results[0].admission_checks


def test_all_admitted_preserves_raw_search_owner_queries_and_detached_callback_storage():
    baseline_owner, guarded_owner = RecordedOwner(6), RecordedOwner(6)
    baseline = search(baseline_owner, lambda *args: None, guarded=False)
    def admit(x, xyz, merit, ordinal):
        x.setflags(write=True)
        xyz.setflags(write=True)
        x[:] = -100
        xyz[:] = -100
        return True
    guarded = search(guarded_owner, admit)
    assert baseline.points.tobytes() == guarded.points.tobytes()
    for name in ('trials', 'moved_nodes', 'initial_penalty', 'final_penalty', 'budget_exhausted',
                 'selected_nodes', 'root_nodes', 'neighbour_nodes', 'priority_mode'):
        assert getattr(baseline, name) == getattr(guarded, name)
    assert baseline_owner.row_counts == guarded_owner.row_counts
    assert [x.tobytes() for x in baseline_owner.candidates] == [x.tobytes() for x in guarded_owner.candidates]


@pytest.mark.parametrize('failure', ('observer', 'cancel', 'owner'))
def test_exception_after_admission_propagates_identity_without_input_mutation(failure):
    points, cells, edges = fixture()
    before = points.tobytes(), cells.tobytes(), edges.tobytes()
    error = RuntimeError('failure after retained trial')
    retained = []
    def observe(x, xyz, merit, ordinal):
        if retained and failure == 'observer':
            raise error
        retained.append(x.copy())
        # Copies must remain detached even if the callback deliberately writes.
        x.setflags(write=True)
        xyz.setflags(write=True)
        x[:] = -99
        xyz[:] = -99
        return True
    def cancel():
        if retained and failure == 'cancel':
            raise error
    def owner(rows):
        if retained and failure == 'owner':
            raise error
        return physical(rows)
    with pytest.raises(RuntimeError) as caught:
        search(owner, observe, cancel=cancel)
    assert caught.value is error and retained
    assert (points.tobytes(), cells.tobytes(), edges.tobytes()) == before


def test_real_physical_guard_scores_every_finite_trial_once_and_retains_earliest_tie(monkeypatch):
    candidate, protected, settings, owner = small_owner_candidate(trials=4)
    good, bad = candidate.points.copy(), candidate.points.copy()
    good[4] = (.35, .4)
    bad[4] = (.02, .35)
    good_report = _physical_quality_candidate_from_xyz(
        _make_candidate(good, candidate.triangles, settings=settings), settings, owner(good))
    bad_report = _physical_quality_candidate_from_xyz(
        _make_candidate(bad, candidate.triangles, settings=settings), settings, owner(bad))
    assert _alternative_progress(candidate.report, good_report.report)
    assert not _alternative_progress(candidate.report, bad_report.report)
    calls, answers, scored = [], [], []
    import anymesher.surface_mesh as surface
    original_score = surface._physical_quality_candidate_from_xyz
    def score(*args):
        scored.append(args[2].copy())
        return original_score(*args)
    monkeypatch.setattr(surface, '_physical_quality_candidate_from_xyz', score)
    def counted_owner(rows):
        calls.append(rows.copy())
        return owner(rows)
    def controlled_search(*args, **kwargs):
        assert kwargs['line_search_admission']
        for ordinal, x, merit in ((1, good, 1.), (2, good, 2.), (3, bad, .5), (4, good, 1.)):
            xyz = counted_owner(x)
            answers.append(kwargs['candidate_callback'](x.copy(), xyz, merit, ordinal))
            xyz.setflags(write=True)
            xyz[:] = -999  # Retained physical report cannot alias receipt input.
        return JointTriangleRepair(bad.copy(), 4, (4,), 3., .5, True,
            line_search_admission=True, admission_checks=4, admission_refusals=1)
    monkeypatch.setattr(frontal, 'repair_joint_triangle_quality', controlled_search)
    result, report = frontal.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), evaluate_coordinates=counted_owner,
        allow_partial_progress=True, line_search_admission=True)
    entry = report['chart_transition_repair']
    assert answers == [True, True, False, True]  # Above-retained merit still feasible.
    assert result.points.tobytes() == good.tobytes() and result.report == good_report.report
    assert entry['selected_trial'] == 1 and entry['selected_penalty'] == 1.
    assert entry['proposed_penalty'] == .5 and entry['proposed_quality'] == bad_report.report
    assert entry['candidate_observations'] == entry['candidate_scorings'] == 4
    assert entry['candidate_refusals'] == entry['candidate_retentions'] == 1
    assert entry['admission_checks'] == 4 and entry['admission_refusals'] == 1
    assert entry['trials'] == report['topology_operations'] == 4 and entry['budget_exhausted']
    assert not entry['quality_satisfied'] and not entry['accepted']
    # Exactly four existing trial XYZ queries plus the unchanged final owner check.
    assert len(calls) == len(scored) == 5


def test_real_helper_guarded_mode_spends_same_pool_and_keeps_pins():
    candidate, protected, settings, physical_owner = small_owner_candidate(trials=31)
    calls = []
    def owner(rows):
        assert rows[4].tobytes() == candidate.points[4].tobytes()
        calls.append(rows.copy())
        return physical_owner(rows)
    before = candidate.points.tobytes(), candidate.triangles.tobytes(), dict(candidate.report)
    result, report = frontal.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), evaluate_coordinates=owner, pinned_nodes=(4,),
        allow_partial_progress=True, line_search_admission=True)
    entry = report['chart_transition_repair']
    assert entry['line_search_admission'] and entry['admission_checks'] == entry['candidate_scorings']
    assert entry['candidate_scorings'] == entry['candidate_observations']
    assert report['topology_operations'] == entry['trials'] <= 31
    assert result.points[4].tobytes() == candidate.points[4].tobytes()
    assert (candidate.points.tobytes(), candidate.triangles.tobytes(), candidate.report) == before
    assert calls


def test_later_full_scoring_error_after_actual_retention_propagates_identity(monkeypatch):
    from anygeometry import GeometryError
    import anymesher.surface_mesh as surface
    candidate, protected, settings, owner = small_owner_candidate(trials=2)
    good = candidate.points.copy()
    good[4] = (.35, .4)
    before = candidate.points.tobytes(), candidate.triangles.tobytes(), dict(candidate.report)
    error = GeometryError('cached scorer failure after retained admissible trial')
    calls = []
    original_score = surface._physical_quality_candidate_from_xyz
    def score(*args):
        calls.append(True)
        if len(calls) == 2:
            raise error
        return original_score(*args)
    monkeypatch.setattr(surface, '_physical_quality_candidate_from_xyz', score)
    def controlled_search(*args, **kwargs):
        assert kwargs['candidate_callback'](good.copy(), owner(good), 1., 1) is True
        # This higher-merit trial must still be physically scored in admission
        # mode, and its failure must not publish the earlier retained candidate.
        kwargs['candidate_callback'](good.copy(), owner(good), 2., 2)
        pytest.fail('scoring exception was swallowed')
    monkeypatch.setattr(frontal, 'repair_joint_triangle_quality', controlled_search)
    with pytest.raises(GeometryError) as caught:
        frontal.repair_frontal_transition(candidate, protected, settings,
            dict(topology_operations=0), evaluate_coordinates=owner,
            allow_partial_progress=True, line_search_admission=True)
    assert caught.value is error and len(calls) == 2
    assert (candidate.points.tobytes(), candidate.triangles.tobytes(), candidate.report) == before


def test_default_helper_diagnostics_do_not_include_admission_fields():
    candidate, protected, settings, owner = small_owner_candidate(trials=1)
    _, report = frontal.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), evaluate_coordinates=owner, allow_partial_progress=True)
    entry = report['chart_transition_repair']
    assert not any(key.startswith(('admission_', 'line_search_', 'refused_zero_')) for key in entry)


@pytest.mark.parametrize('analytic', (False, True))
def test_surface_admission_opt_in_only_exact_bound_owner(monkeypatch, analytic):
    import anymesher.surface_mesh as surface
    import anymesher._physical_t3_refinement as refinement
    from anymesher.surface_mesh import SurfaceMeshOptions
    from anymesher.native_v2 import NativeMeshingOptions
    from test_joint_triangle_owner_batches import analytic_chart
    from anygeometry import to_dict
    model, chart = analytic_chart()
    before = to_dict(model)
    seen = []
    def capture(candidate, protected, settings, report, *args, **kwargs):
        seen.append((kwargs['allow_partial_progress'], kwargs['line_search_admission']))
        return candidate, dict(report, chart_transition_repair={})
    monkeypatch.setattr(frontal, 'repair_frontal_transition', capture)
    monkeypatch.setattr(refinement, 'refine_physical_candidate',
                        lambda candidate, triangulation, settings, report, *args, **kwargs:
                        (candidate, triangulation, report))
    surface.mesh_planar_surface(((0., 0.), (1., 0.), (1., 1.), (0., 1.)),
        options=SurfaceMeshOptions(target_size=1., recombine=False, backend='python',
            min_angle=45., native_options=NativeMeshingOptions(point_placement='frontal_delaunay',
                metric_mode='isotropic_spatial', max_insertions=1, max_topology_operations=8)),
        _metric_to_physical=chart.evaluate if analytic else lambda rows: chart.evaluate(rows),
        _metric_jacobian=np.asarray(((1., 0.), (0., 1.), (0., 0.))), _preserve_spatial_refinement=True)
    assert seen == [(analytic, analytic)] and to_dict(model) == before


@pytest.mark.parametrize('mode, partial, owner', ((1, True, physical), (True, False, physical), (True, True, None)))
def test_helper_requires_explicit_owner_partial_mode(mode, partial, owner):
    from anymesher.errors import MeshError
    candidate, protected, settings, _ = small_owner_candidate()
    with pytest.raises(MeshError, match='line-search admission'):
        frontal.repair_frontal_transition(candidate, protected, settings, dict(topology_operations=0),
            evaluate_coordinates=owner, allow_partial_progress=partial, line_search_admission=mode)
