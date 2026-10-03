"""A bounded interior cavity can make progress that neither single flip admits."""
from fractions import Fraction
import json

import numpy as np
import pytest

from anymesher.native_v2 import MutableT3Topology, ComponentSeedRegistry, _GeometryLimited
from anymesher._physical_t3_refinement import _alternative_progress
from anymesher._physical_t3_cavity import two_flip_cavity
from anymesher.surface_mesh import (SurfaceMeshOptions, _make_candidate,
    _physical_quality_candidate_from_xyz)


def pentagon():
    points = np.array([(-1., 0.), (0., -.2), (1., 0.), (1., 8.), (-1., 1.7)])
    cells = np.array([(0, 1, 2), (0, 2, 3), (0, 3, 4)])
    topology = MutableT3Topology(points, cells, [(0, 1), (1, 2), (2, 3), (3, 4)],
        splittable_edges={(0, 4): (71, 1, 0)}, seed_registry=ComponentSeedRegistry(100),
        node_owners=np.arange(5)+30, triangle_owners=[17]*3)
    topology._shared_node_ids = {0: (91, 71, Fraction(1)), 4: (92, 71, Fraction(0))}
    topology.quality_cache[(0, 1, 2)] = (1.,)
    settings = SurfaceMeshOptions(min_angle=15., max_element_growth=1.5)
    xyz = np.column_stack((points, np.zeros(5)))
    def score(p, t):
        return _physical_quality_candidate_from_xyz(_make_candidate(p, t, settings=settings), settings, xyz)
    return topology, score, settings


def exact_area(points, cells):
    values = []
    for cell in cells:
        a, b, c = [[Fraction(float(x)) for x in points[node]] for node in cell]
        area = ((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])) / 2
        assert area > 0
        values.append(area)
    return sum(values)


def summary(candidate):
    return {name: candidate.report[name] for name in ('violation_counts',
        'invalid_element_count', 'quality_violation_count',
        'elements_above_maximum_growth', 'max_element_growth')}


def test_exact_pentagon_singles_refuse_but_complete_pair_progresses():
    topology, score, _ = pentagon()
    initial = score(topology.points, topology.triangles)
    area = exact_area(topology.points, topology.triangles)
    receipts = {'baseline': summary(initial)}
    for edge in ((0, 2), (0, 3)):
        trial, _, _ = pentagon()
        assert trial.flip_edge(edge)
        single = score(trial.points, trial.triangles)
        assert not _alternative_progress(initial.report, single.report)
        assert exact_area(trial.points, trial.triangles) == area
        receipts[str(edge)] = summary(single)
    assert topology.flip_edge((0, 3))
    assert topology.flip_edge((0, 2))
    complete = score(topology.points, topology.triangles)
    assert _alternative_progress(initial.report, complete.report)
    assert complete.report['violation_counts'] == dict.fromkeys(initial.report['violation_counts'], 0)
    assert complete.report['elements_above_maximum_growth'] == 2
    assert complete.report['poor_element_ids']  # Progress is not full acceptance.
    assert topology.triangles.tolist() == [[0, 1, 4], [1, 2, 4], [2, 3, 4]]
    assert exact_area(topology.points, topology.triangles) == area
    receipts['complete'] = summary(complete)
    print('PENTAGON_PHYSICAL_PROOF', json.dumps(receipts, sort_keys=True))


def state(topology):
    return (topology.points.tobytes(), topology.triangles.tobytes(),
        topology.node_owners.tobytes(), topology.triangle_owners.tobytes(),
        topology.constraint_edges.tobytes(), dict(topology._splittable_intervals),
        dict(topology.shared_node_ids), topology.epoch, dict(topology.quality_cache),
        tuple(topology.free_triangle_ids), id(topology._points), id(topology._triangles),
        id(topology._topology_index), id(topology.quality_cache),
        dict(topology._seed_registry._values), topology._seed_registry._next)


def run(topology, score, *, budget=2, first_edges=((0, 3),),
        checkpoint=lambda: None, cancellation=None):
    baseline = score(topology.points, topology.triangles)
    stats = dict(first_attempts=0, second_attempts=0, first_refusals=0,
        second_refusals=0, commits=0, stop_reason=None, used=0)
    def charge(kind):
        if stats['used'] >= budget:
            return False
        stats['used'] += 1
        stats[kind+'_attempts'] += 1
        return True
    result = two_flip_cavity(topology, baseline, first_edges, score=score,
        progress=_alternative_progress, owner_checkpoint=checkpoint,
        charge_attempt=charge, stats=stats, cancellation_check=cancellation)
    return result, stats


def test_detached_complete_pair_preserves_material_stations_and_owners(monkeypatch):
    topology, score, _ = pentagon()
    before = state(topology)
    def forbidden(*args, **kwargs):
        raise AssertionError('interior closure cannot resolve any station')
    monkeypatch.setattr(topology._seed_registry, 'resolve', forbidden)
    result, stats = run(topology, score)
    assert result is not None
    proposal, quality = result
    assert stats == dict(first_attempts=1, second_attempts=1, first_refusals=0,
        second_refusals=0, commits=1, stop_reason='committed', used=2)
    assert state(topology) == before
    assert proposal.points.tobytes() == topology.points.tobytes()
    assert proposal.node_owners.tobytes() == topology.node_owners.tobytes()
    assert proposal.triangle_owners.tolist() == [17]*3
    assert proposal.constraint_edges.tobytes() == topology.constraint_edges.tobytes()
    assert proposal.shared_node_ids == topology.shared_node_ids
    assert proposal._splittable_intervals == topology._splittable_intervals
    assert proposal._seed_registry is topology._seed_registry
    assert proposal.epoch == topology.epoch + 2 and not proposal.quality_cache
    assert exact_area(proposal.points, proposal.triangles) == exact_area(topology.points, topology.triangles)
    assert quality.triangles.tobytes() == proposal.triangles.tobytes()
    assert quality.points.tobytes() == proposal.points.tobytes()
    assert _alternative_progress(score(topology.points, topology.triangles).report, quality.report)


@pytest.mark.parametrize('budget', [0, 1, 2])
def test_partial_budget_never_publishes_first_stage(budget):
    topology, score, _ = pentagon()
    before = state(topology)
    result, stats = run(topology, score, budget=budget)
    assert stats['used'] == budget
    assert state(topology) == before
    assert (result is not None) == (budget == 2)
    assert stats['stop_reason'] == ('committed' if budget == 2 else 'topology_budget')


def test_all_refused_and_duplicate_first_edge_charge_only_actual_attempts():
    topology, score, _ = pentagon()
    before = state(topology)
    stats = dict(first_attempts=0, second_attempts=0, first_refusals=0,
        second_refusals=0, commits=0, stop_reason=None, used=0)
    def charge(kind):
        stats['used'] += 1
        stats[kind+'_attempts'] += 1
        return True
    assert two_flip_cavity(topology, score(topology.points, topology.triangles),
        ((0, 3), (3, 0)), score=score, progress=lambda *_: False,
        owner_checkpoint=lambda: None, charge_attempt=charge, stats=stats) is None
    assert stats['first_attempts'] == stats['second_attempts'] == stats['second_refusals'] == 1
    assert stats['commits'] == 0 and stats['stop_reason'] == 'no_progress_for_local_two_flip_cavities'
    assert state(topology) == before


@pytest.mark.parametrize('where', ['native-v2 physical cavity first attempt',
    'native-v2 physical cavity second attempt', 'native-v2 physical cavity complete admission'])
@pytest.mark.parametrize('error_type', [RuntimeError, _GeometryLimited])
def test_cancellation_identity_after_attempt_and_retention_no_publication(where, error_type):
    topology, score, _ = pentagon()
    before = state(topology)
    error = error_type('cancel cavity')
    def cancel(phase):
        if phase == where:
            raise error
    with pytest.raises(error_type) as caught:
        run(topology, score, cancellation=cancel)
    assert caught.value is error and state(topology) == before


def test_stale_final_binding_checkpoint_cannot_publish_pair():
    topology, score, _ = pentagon()
    before = state(topology)
    calls = []
    error = RuntimeError('stale exact owner')
    def checkpoint():
        calls.append(None)
        if len(calls) == 2:
            raise error
    with pytest.raises(RuntimeError) as caught:
        run(topology, score, checkpoint=checkpoint)
    assert caught.value is error and len(calls) == 2 and state(topology) == before


@pytest.mark.parametrize('which', [1, 2])
def test_scorer_private_exception_propagates_without_local_refusal(which):
    from anymesher.native_v2 import _GeometryLimited
    topology, score, _ = pentagon()
    before = state(topology)
    error = _GeometryLimited('owner/scorer failure')
    calls = []
    def failure(p, t):
        calls.append(None)
        if len(calls) == which+1:  # Initial baseline is not a charged attempt.
            raise error
        return score(p, t)
    with pytest.raises(_GeometryLimited) as caught:
        run(topology, failure)
    assert caught.value is error and state(topology) == before


def test_owner_correspondence_and_cell_input_order_are_preserved():
    first, score, _ = pentagon()
    points, cells = first.points, first.triangles
    permutation = [2, 0, 1]
    other = MutableT3Topology(points, cells[permutation], tuple(first.protected_edges),
        node_owners=first.node_owners, triangle_owners=first.triangle_owners[permutation],
        splittable_edges=first._splittable_intervals, seed_registry=first._seed_registry)
    other._shared_node_ids = dict(first.shared_node_ids)
    result, stats = run(first, score)
    reordered, reordered_stats = run(other, score)
    assert result[0].triangles.tobytes() == reordered[0].triangles.tobytes()
    assert result[0].triangle_owners.tobytes() == reordered[0].triangle_owners.tobytes()
    assert stats == reordered_stats


def test_different_cell_owners_refuse_first_primitive_exactly():
    topology, score, _ = pentagon()
    topology.triangle_owners[:] = [17, 17, 19]
    before = state(topology)
    result, stats = run(topology, score)
    assert result is None and stats['first_attempts'] == stats['first_refusals'] == 1
    assert stats['second_attempts'] == 0 and state(topology) == before


@pytest.mark.parametrize('kind', ['protected', 'splittable'])
def test_mandatory_internal_first_edge_is_excluded_without_attempt(kind):
    topology, score, _ = pentagon()
    if kind == 'protected':
        topology._protected_edges.add((0, 3))
    else:
        topology._splittable_intervals[(0, 3)] = (72, Fraction(0), Fraction(1))
    before = state(topology)
    result, stats = run(topology, score)
    assert result is None and stats['used'] == 0
    assert stats['stop_reason'] == 'no_progress_for_local_two_flip_cavities'
    assert state(topology) == before


def test_local_perimeter_constraint_leaves_staged_first_flip_unpublished():
    topology, score, _ = pentagon()
    topology._protected_edges.add((0, 2))
    before = state(topology)
    result, stats = run(topology, score)
    assert result is None and stats['first_attempts'] == 1 and stats['second_attempts'] == 0
    assert stats['stop_reason'] == 'no_progress_for_local_two_flip_cavities'
    assert state(topology) == before


def test_authored_rational_pentagon_has_exact_area_and_positive_patches():
    rational = [(Fraction(-1), Fraction(0)), (Fraction(0), Fraction(-1, 5)),
        (Fraction(1), Fraction(0)), (Fraction(1), Fraction(8)),
        (Fraction(-1), Fraction(17, 10))]
    for cells in ([(0, 1, 2), (0, 2, 3), (0, 3, 4)],
                  [(0, 1, 2), (0, 2, 4), (2, 3, 4)],
                  [(0, 1, 4), (1, 2, 4), (2, 3, 4)]):
        area = Fraction()
        for cell in cells:
            a, b, c = [rational[node] for node in cell]
            twice = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            assert twice > 0
            area += twice / 2
        assert area == Fraction(99, 10)


def test_invalid_intermediate_refuses_before_second_stage():
    from dataclasses import replace
    topology, score, _ = pentagon()
    before = state(topology)
    calls = []
    def invalid(p, t):
        trial = score(p, t)
        calls.append(None)
        return trial if len(calls) == 1 else replace(trial, report=dict(trial.report, invalid_element_count=1))
    result, stats = run(topology, invalid)
    assert result is None and stats['first_refusals'] == stats['first_attempts'] == 1
    assert stats['second_attempts'] == 0 and state(topology) == before


def test_complete_guard_uses_original_report_and_private_partial_commit_cancel_is_charged():
    topology, score, _ = pentagon()
    before = state(topology)
    baseline = score(topology.points, topology.triangles)
    stats = dict(first_attempts=0, second_attempts=0, first_refusals=0,
        second_refusals=0, commits=0, stop_reason=None)
    charged = []
    error = _GeometryLimited('second private commit cancellation')
    commits = []
    def cancel(phase):
        if phase == 'native-v2 edge flip commit':
            commits.append(None)
            if len(commits) == 2:
                raise error
    def charge(kind):
        charged.append(kind)
        stats[kind+'_attempts'] += 1
        return True
    def guard(original, proposed):
        assert original is baseline.report
        return _alternative_progress(original, proposed)
    with pytest.raises(_GeometryLimited) as caught:
        two_flip_cavity(topology, baseline, ((0, 3),), score=score, progress=guard,
            owner_checkpoint=lambda: None, charge_attempt=charge, stats=stats,
            cancellation_check=cancel)
    assert caught.value is error and charged == ['first', 'second']
    assert stats['commits'] == stats['second_refusals'] == 0 and state(topology) == before


def analytic_owner_fixture(monkeypatch, *, budget=4):
    from dataclasses import replace
    from anygeometry import GeometryModel, BezierDirectrix, ExtrudedSurface
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.triangulation import PlanarTriangulation
    topology, _, settings = pentagon()
    model = GeometryModel()
    # A real nonplanar analytic support, not a forbidden line-degenerate Bezier.
    controls = ((0., 0., 0.), (1., 0., .1), (2., 0., 0.))
    vertices = model.add_points(controls)
    edge = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
    face = model.extrude((edge,), (0., 8.2, 0.))[0]
    model.set_face_surface(face, ExtrudedSurface(BezierDirectrix(controls), (0., 8.2, 0.)))
    chart = AnalyticMetricChart(model, face)
    calls = []
    original = type(model).evaluate_face_many
    def evaluate(owner, *args, **kwargs):
        calls.append(None)
        return original(owner, *args, **kwargs)
    monkeypatch.setattr(type(model), 'evaluate_face_many', evaluate)
    points = topology.points + (1., .2)
    settings = replace(settings, native_options=NativeMeshingOptions(
        max_topology_operations=budget, max_insertions=1))
    xyz = chart.evaluate(points)
    candidate = _physical_quality_candidate_from_xyz(
        _make_candidate(points, topology.triangles, settings=settings), settings, xyz)
    calls.clear()
    boundary = topology.constraint_edges
    triangulation = PlanarTriangulation(points, topology.triangles, boundary, boundary,
        np.empty((0, 2), dtype=int), np.arange(5), ())
    return model, chart, candidate, triangulation, settings, calls


@pytest.mark.parametrize('budget', [2, 3, 4])
def test_exact_owner_route_exhausts_singles_then_detached_pair_in_same_pool(monkeypatch, budget):
    from anygeometry import to_dict
    from anymesher._physical_t3_refinement import refine_physical_candidate
    model, chart, candidate, triangulation, settings, calls = analytic_owner_fixture(monkeypatch, budget=budget)
    source = to_dict(model)
    before = candidate.points.tobytes(), candidate.triangles.tobytes(), triangulation.segments.tobytes()
    registry = ComponentSeedRegistry(100)
    result, output, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=1, shared_segment_splits=0, shared_nodes=[]),
        chart.evaluate, {}, registry, allow_physical_flips=True)
    print('OWNER_CAVITY_RECEIPT', budget, json.dumps(receipt, sort_keys=True))
    assert len(calls) == 1  # Initial public owner evaluation; all flips reuse XYZ.
    assert to_dict(model) == source and not registry.assigned_node_ids
    assert (candidate.points.tobytes(), candidate.triangles.tobytes(), triangulation.segments.tobytes()) == before
    assert receipt['topology_operations'] == receipt['physical_quality_attempts'] == budget
    assert receipt['physical_quality_flip_attempts'] == receipt['physical_quality_flip_refused_attempts'] == 2
    assert receipt['physical_quality_cavity_first_attempts'] == int(budget > 2)
    assert receipt['physical_quality_cavity_second_attempts'] == int(budget > 3)
    assert receipt['physical_quality_flips'] == 2 * int(budget == 4)
    assert receipt['physical_quality_cavity_commits'] == int(budget == 4)
    assert receipt['insertions'] == 1  # Already spent before physical refinement.
    assert receipt['shared_segment_splits'] == receipt['physical_quality_bisections'] == 0
    assert result.points.tobytes() == candidate.points.tobytes()
    assert output.segments.tobytes() == triangulation.segments.tobytes()
    if budget == 4:
        assert _alternative_progress(candidate.report, result.report)
        assert result.report['poor_element_ids'] and receipt['selected_route'] == 'frontal_delaunay_budget_limited'
    else:
        assert result.triangles.tobytes() == candidate.triangles.tobytes()
        assert not receipt['physical_quality_candidate_cells_exhausted']
    assert receipt['physical_quality_stop_reason'] == 'topology_budget'


def test_wrapped_generic_evaluator_keeps_old_single_edit_route(monkeypatch):
    from anymesher._physical_t3_refinement import refine_physical_candidate
    _, chart, candidate, triangulation, settings, calls = analytic_owner_fixture(monkeypatch)
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=1, shared_segment_splits=0, shared_nodes=[]),
        lambda rows: chart.evaluate(rows), {}, ComponentSeedRegistry(100), allow_physical_flips=True)
    assert not any('cavity' in key for key in receipt)
    assert receipt['topology_operations'] == 2 and receipt['physical_quality_flips'] == 0
    assert result.triangles.tobytes() == candidate.triangles.tobytes() and len(calls) == 1


def test_exact_owner_all_flip_families_refused_with_spent_insertion_allowance_is_budget_limited(monkeypatch):
    import anymesher._physical_t3_refinement as physical
    from anygeometry import to_dict
    model, chart, candidate, triangulation, settings, calls = analytic_owner_fixture(monkeypatch, budget=7)
    source = to_dict(model)
    before = candidate.points.tobytes(), candidate.triangles.tobytes(), triangulation.segments.tobytes()
    # Exercise the receipt after every scored single/pair is refused. Actual
    # owner XYZ and full reports still run; this controlled strict refusal is
    # independent of the pentagon's separate actual-progress proof.
    monkeypatch.setattr(physical, '_alternative_progress', lambda *_: False)
    registry = ComponentSeedRegistry(100)
    result, output, receipt = physical.refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=1, shared_segment_splits=0, shared_nodes=[]),
        chart.evaluate, {}, registry, allow_physical_flips=True)
    assert receipt['physical_quality_flip_attempts'] == receipt['physical_quality_flip_refused_attempts'] == 2
    assert receipt['physical_quality_cavity_first_attempts'] == receipt['physical_quality_cavity_second_attempts'] == 2
    assert receipt['physical_quality_cavity_second_refusals'] == 2
    assert receipt['physical_quality_cavity_first_refusals'] == receipt['physical_quality_cavity_commits'] == 0
    assert receipt['topology_operations'] == receipt['physical_quality_attempts'] == 6
    assert receipt['topology_operations'] < settings.native_options.max_topology_operations
    assert receipt['insertions'] == settings.native_options.max_insertions
    assert receipt['physical_quality_cavity_stop_reason'] == 'no_progress_for_local_two_flip_cavities'
    assert receipt['physical_quality_stop_reason'] == 'no_progress_for_flip_candidates_insertion_budget'
    assert not receipt['physical_quality_candidate_cells_exhausted']
    assert receipt['selected_route'] == 'frontal_delaunay_budget_limited'
    assert result.triangles.tobytes() == before[1] and result.points.tobytes() == before[0]
    assert output.segments.tobytes() == before[2]
    assert (candidate.points.tobytes(), candidate.triangles.tobytes(), triangulation.segments.tobytes()) == before
    assert to_dict(model) == source and len(calls) == 1 and not registry.assigned_node_ids


def test_real_stale_owner_at_complete_admission_cannot_return_mesh(monkeypatch):
    from anygeometry import GeometryError
    from anymesher._physical_t3_refinement import refine_physical_candidate
    model, chart, candidate, triangulation, settings, calls = analytic_owner_fixture(monkeypatch)
    before = candidate.points.tobytes(), candidate.triangles.tobytes(), triangulation.segments.tobytes()
    registry = ComponentSeedRegistry(100)
    def mutate(phase):
        if phase == 'native-v2 physical cavity complete admission':
            model.add_point(3., 9., 1.)
    with pytest.raises(GeometryError, match='stale'):
        refine_physical_candidate(candidate, triangulation, settings,
            dict(topology_operations=0, insertions=1, shared_segment_splits=0, shared_nodes=[]),
            chart.evaluate, {}, registry, mutate, allow_physical_flips=True)
    assert (candidate.points.tobytes(), candidate.triangles.tobytes(), triangulation.segments.tobytes()) == before
    assert len(calls) == 1 and not registry.assigned_node_ids


def test_native_incidence_and_python_oracle_parity(monkeypatch):
    import anymesher.native_cpp as native
    topology, score, _ = pentagon()
    if topology._topology_index._native_state is None:
        pytest.skip('compiled incidence backend unavailable; source Python coverage is separate')
    compiled, compiled_stats = run(topology, score)
    monkeypatch.setattr(native, 'native_t3_incidence', lambda *_: None)
    python_topology, _, _ = pentagon()
    assert python_topology._topology_index._native_state is None
    python, python_stats = run(python_topology, score)
    assert compiled_stats == python_stats
    assert compiled[0].points.tobytes() == python[0].points.tobytes()
    assert compiled[0].triangles.tobytes() == python[0].triangles.tobytes()
    assert summary(compiled[1]) == summary(python[1])
