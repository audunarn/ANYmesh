"""Trial observation cannot change search; admission has its own incumbent."""
import numpy as np
import pytest

import anymesher._frontal_transition_quality as frontal
from anymesher._joint_triangle_repair import JointTriangleRepair, repair_joint_triangle_quality
from anymesher._physical_t3_refinement import _alternative_progress
from anymesher.surface_mesh import (_make_candidate, _physical_quality_candidate,
                                    _physical_quality_candidate_from_xyz)
from test_frontal_transition_quality import small_owner_candidate
from test_joint_triangle_owner_batches import RecordedOwner, assert_same_result, fixture, physical


def run(points, cells, protected, owner, *, callback=None, budget=9, batch=1, cancel=None):
    return repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
        min_angle=30., max_growth=1.5, max_trials=budget, evaluate_coordinates=owner,
        coordinate_batch_size=batch, candidate_callback=callback, cancellation_check=cancel)


@pytest.mark.parametrize('budget', (0, 1, 7, 8, 9, 31))
def test_observation_preserves_default_search_owner_calls_and_odd_budget(budget):
    points, cells, protected = fixture()
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    events = []
    results = []
    for batch in (1, 8):
        default_owner, observer_owner = RecordedOwner(len(points)), RecordedOwner(len(points))
        expected = run(points, cells, protected, default_owner, budget=budget, batch=batch)
        recorded = []
        def observe(x, xyz, merit, ordinal):
            assert not x.flags.writeable and not xyz.flags.writeable
            with pytest.raises(ValueError):
                x[0, 0] += 1
            with pytest.raises(ValueError):
                xyz[0, 0] += 1
            assert xyz.tobytes() == physical(x).tobytes()
            assert x[:4].tobytes() == points[:4].tobytes()
            recorded.append((ordinal, merit, x.tobytes(), xyz.tobytes()))
        actual = run(points, cells, protected, observer_owner,
                     callback=observe, budget=budget, batch=batch)
        assert_same_result(actual, expected)
        assert observer_owner.row_counts == default_owner.row_counts
        assert [row[0] for row in recorded] == list(range(1, actual.trials + 1))
        results.append(actual)
        events.append(recorded)
    assert_same_result(*results)
    assert events[0] == events[1]
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before


def test_invalid_charged_row_has_no_observation_and_does_not_shift_ordinals():
    points = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (1e-8, .5)))
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
    protected = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)))
    all_events = []
    for batch in (1, 8):
        events = []
        owner = RecordedOwner(len(points))
        result = run(points, cells, protected, owner, budget=3, batch=batch,
            callback=lambda x, xyz, merit, ordinal: events.append((ordinal, x.tobytes(), xyz.tobytes())))
        assert result.trials == 3 and [row[0] for row in events] == [1, 3]
        all_events.append(events)
    assert all_events[0] == all_events[1]


def test_observer_backing_is_detached_even_when_callback_reenables_writes():
    points, cells, protected = fixture()
    expected = run(points, cells, protected, physical)
    retained = []
    def observe(x, xyz, merit, ordinal):
        retained.append((x, xyz))
        x.setflags(write=True)
        xyz.setflags(write=True)
        x[:] = -100
        xyz[:] = -100
    actual = run(points, cells, protected, physical, callback=observe, batch=8)
    assert_same_result(expected, actual)
    for x, xyz in retained:
        x[:] = 100
        xyz[:] = 100
    assert_same_result(expected, actual)


@pytest.mark.parametrize('callback', (False, True, 1, 'callback'))
def test_invalid_callback_rejected_before_owner_call(callback):
    points, cells, protected = fixture()
    owner = RecordedOwner(len(points))
    with pytest.raises(ValueError, match='invalid joint'):
        run(points, cells, protected, owner, callback=callback)
    assert not owner.row_counts


def test_callback_requires_owner_coordinates():
    points, cells, protected = fixture()
    with pytest.raises(ValueError, match='invalid joint'):
        run(points, cells, protected, None, callback=lambda *args: None)


@pytest.mark.parametrize('batch', (1, 8))
def test_pinned_coordinate_is_exact_in_all_observations(batch):
    points, cells, protected = fixture()
    rows = []
    result = repair_joint_triangle_quality(points, cells, protected, range(len(cells)),
        min_angle=30., max_growth=1.5, max_trials=7, evaluate_coordinates=physical,
        coordinate_batch_size=batch, pinned_nodes=(4,),
        candidate_callback=lambda x, xyz, merit, ordinal: rows.append(x[4].tobytes()))
    assert rows and all(row == points[4].tobytes() for row in rows)
    assert result.points[4].tobytes() == points[4].tobytes()


def test_nonfinite_trial_merit_is_charged_but_not_observed():
    points, cells, protected = fixture()
    calls, observed = [], []
    def owner(rows):
        calls.append(len(rows))
        return physical(rows) if len(calls) <= 2 else np.zeros((len(rows), 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        result = run(points, cells, protected, owner, budget=1,
                     callback=lambda *args: observed.append(args))
    assert result.trials == 1 and result.budget_exhausted and not observed
    assert result.points.tobytes() == points.tobytes() and len(calls) == 3


@pytest.mark.parametrize('batch', (1, 8))
def test_callback_error_after_prior_observation_propagates_identity(batch):
    points, cells, protected = fixture()
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    error = RuntimeError('observer stale or cancelled')
    events = []
    def observe(x, xyz, merit, ordinal):
        events.append(ordinal)
        if ordinal == 2:
            raise error
    with pytest.raises(RuntimeError) as caught:
        run(points, cells, protected, physical, callback=observe, batch=batch)
    assert caught.value is error and events == [1, 2]
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before


@pytest.mark.parametrize('batch', (1, 8))
def test_cancellation_immediately_after_observation_does_not_return_partial(batch):
    points, cells, protected = fixture()
    before = points.tobytes()
    observed = []
    error = RuntimeError('cancel after observation')
    def cancel():
        if observed:
            raise error
    with pytest.raises(RuntimeError) as caught:
        run(points, cells, protected, physical, batch=batch, cancel=cancel,
            callback=lambda x, xyz, merit, ordinal: observed.append(ordinal))
    assert caught.value is error and observed == [1] and points.tobytes() == before


def test_cached_scorer_matches_owner_wrapper_without_query_or_mutation():
    candidate, _, settings, owner = small_owner_candidate()
    chart = _make_candidate(candidate.points, candidate.triangles, settings=settings)
    xyz = owner(chart.points)
    before = chart.points.tobytes(), xyz.tobytes()
    calls = []
    def evaluate(rows):
        calls.append(rows.copy())
        return owner(rows)
    queried = _physical_quality_candidate(chart, settings, evaluate)
    cached = _physical_quality_candidate_from_xyz(chart, settings, xyz)
    assert len(calls) == 1 and cached.report == queried.report and cached.score == queried.score
    assert cached.aspect_ratios.tobytes() == queried.aspect_ratios.tobytes()
    assert (chart.points.tobytes(), xyz.tobytes()) == before


@pytest.mark.parametrize('xyz', (np.zeros((6, 2)), np.full((6, 3), np.nan)))
def test_cached_scorer_rejects_invalid_coordinates(xyz):
    from anymesher.errors import MeshError
    candidate, _, settings, _ = small_owner_candidate()
    with pytest.raises(MeshError, match='finite'):
        _physical_quality_candidate_from_xyz(candidate, settings, xyz)


@pytest.mark.parametrize('batch', (1, 8))
def test_admissible_intermediate_survives_inadmissible_global_merit_winner(monkeypatch, batch):
    candidate, protected, settings, physical_owner = small_owner_candidate(trials=3)
    before = candidate.points.tobytes(), candidate.triangles.tobytes(), dict(candidate.report)
    good, bad = candidate.points.copy(), candidate.points.copy()
    good[4] = (.35, .4)
    bad[4] = (.02, .35)
    good_report = _physical_quality_candidate_from_xyz(
        _make_candidate(good, candidate.triangles, settings=settings), settings, physical_owner(good))
    bad_report = _physical_quality_candidate_from_xyz(
        _make_candidate(bad, candidate.triangles, settings=settings), settings, physical_owner(bad))
    assert _alternative_progress(candidate.report, good_report.report)
    assert not _alternative_progress(candidate.report, bad_report.report)
    calls = []
    def owner(rows):
        calls.append(rows.copy())
        return physical_owner(rows)
    def search(*args, **kwargs):
        # An independent controlled search stream, with actual physical scoring.
        # The later equal-merit admissible trial cannot displace the first.
        for ordinal, points, merit in ((1, good, 1.), (2, good, 1.), (3, bad, .5)):
            x, xyz = points.copy(), owner(points)
            x.setflags(write=False)
            xyz.setflags(write=False)
            kwargs['candidate_callback'](x, xyz, merit, ordinal)
            # The observer's retained candidate must not alias its input backing.
            x.setflags(write=True)
            x[:] = -999
        return JointTriangleRepair(bad.copy(), 3, (4,), 2., .5, True)
    monkeypatch.setattr(frontal, 'repair_joint_triangle_quality', search)
    result, report = frontal.repair_frontal_transition(candidate, protected, settings,
        dict(topology_operations=0), evaluate_coordinates=owner,
        coordinate_batch_size=batch, allow_partial_progress=True)
    entry = report['chart_transition_repair']
    assert result.points.tobytes() == good.tobytes()
    assert result.report == good_report.report == entry['final_quality'] == entry['admissible_quality']
    assert entry['proposed_quality'] == bad_report.report and entry['proposed_penalty'] == .5
    assert entry['final_penalty'] == 1.
    assert entry['selected_penalty'] == 1. and entry['selected_trial'] == entry['admissible_trial'] == 1
    assert entry['candidate_adopted'] and not entry['accepted'] and not entry['quality_satisfied']
    assert (entry['candidate_observations'], entry['candidate_scorings'],
            entry['candidate_refusals'], entry['candidate_retentions']) == (3, 2, 1, 1)
    assert entry['candidate_observation_seconds'] >= entry['candidate_scoring_seconds'] >= 0
    assert entry['trials'] == report['topology_operations'] == 3 and entry['budget_exhausted']
    # Three supplied trials and the unchanged final owner certification only.
    assert len(calls) == 4 and calls[-1].tobytes() == bad.tobytes()
    assert result.points[:4].tobytes() == candidate.points[:4].tobytes()
    assert (candidate.points.tobytes(), candidate.triangles.tobytes(), candidate.report) == before


@pytest.mark.parametrize('failure_stage', ('owner', 'cancel'))
def test_failure_after_admissible_retention_returns_no_partial(monkeypatch, failure_stage):
    candidate, protected, settings, physical_owner = small_owner_candidate(trials=2)
    good = candidate.points.copy()
    good[4] = (.35, .4)
    before = candidate.points.tobytes(), dict(candidate.report)
    error = RuntimeError('stale owner or cancellation after retention')
    retained = []
    def owner(rows):
        if retained and failure_stage == 'owner':
            raise error
        return physical_owner(rows)
    def cancel(phase):
        if retained and failure_stage == 'cancel':
            raise error
    def search(*args, **kwargs):
        kwargs['candidate_callback'](good.copy(), physical_owner(good), 1., 1)
        retained.append(True)
        return JointTriangleRepair(good.copy(), 2, (4,), 2., 1., True)
    monkeypatch.setattr(frontal, 'repair_joint_triangle_quality', search)
    with pytest.raises(RuntimeError) as caught:
        frontal.repair_frontal_transition(candidate, protected, settings,
            dict(topology_operations=0), cancel, evaluate_coordinates=owner, allow_partial_progress=True)
    assert caught.value is error and retained
    assert (candidate.points.tobytes(), candidate.report) == before
