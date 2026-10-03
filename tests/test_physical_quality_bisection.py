import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from anymesher import ComponentSeedRegistry, MutableT3Topology, NativeMeshingOptions
from anymesher.errors import MeshError
from anymesher._physical_t3_refinement import refine_physical_candidate
from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate, _physical_quality_candidate
from anymesher.triangulation import PlanarTriangulation


def square():
    return MutableT3Topology(np.array([(0.,0.),(1.,0.),(1.,1.),(0.,1.)]),
        np.array([(0,1,2),(0,2,3)]), [(0,1),(1,2),(2,3),(0,3)],
        triangle_owners=np.array([17,23]))


def test_interior_bisection_preserves_material_and_each_cell_owner():
    topology = square()
    points = topology.points.copy()
    topology.bisect_interior_edge((0,2))
    np.testing.assert_array_equal(topology.points[:4], points)
    np.testing.assert_array_equal(topology.points[4], [.5,.5])
    assert sorted(topology.triangle_owners.tolist()) == [17,17,23,23]
    assert len(topology.triangles) == 4
    for owner in (17,23):
        rows = topology.points[topology.triangles[topology.triangle_owners == owner]]
        a,b = rows[:,1]-rows[:,0], rows[:,2]-rows[:,0]
        twice_area = a[:,0]*b[:,1]-a[:,1]*b[:,0]
        assert sum(twice_area) == pytest.approx(1.)
    assert set(map(tuple, topology.constraint_edges)) == {(0,1),(1,2),(2,3),(0,3)}


@pytest.mark.parametrize('phase', ['validator', 'native-v2 interior bisection commit'])
def test_bisection_rejection_leaves_exact_topology_and_index(phase):
    topology = square()
    points, triangles = topology.canonical_export()
    index = topology._topology_index
    def reject(*args):
        raise RuntimeError('reject physical split')
    def cancel(where):
        if where == phase:
            reject()
    with pytest.raises(RuntimeError, match='reject physical split'):
        topology.bisect_interior_edge((0,2), cancellation_check=cancel,
                                     _validate_candidate=reject if phase == 'validator' else None)
    actual_points, actual_cells = topology.canonical_export()
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_cells, triangles)
    assert topology.epoch == 0
    assert topology._topology_index is index


def test_shared_split_rejection_does_not_allocate_station():
    registry = ComponentSeedRegistry(100)
    topology = MutableT3Topology(np.array([(0.,0.),(1.,0.),(0.,1.)]),
        np.array([(0,1,2)]), [(0,2),(1,2)],
        splittable_edges={(0,1):(7,1,0)}, seed_registry=registry)
    def reject(*args):
        raise RuntimeError('reject physical split')
    with pytest.raises(RuntimeError, match='reject physical split'):
        topology.split_segment((0,1), _validate_candidate=reject)
    assert registry.resolve(7,1,2) == 100
    assert topology.epoch == 0
    assert len(topology.points) == 3


def cavity():
    raw = json.loads(Path(__file__).with_name('analytic_growth_cavity.json').read_text())
    points, cells = np.array(raw['points']), np.array(raw['triangles'])
    inverse = np.linalg.inv(raw['transform'])
    def evaluate(rows):
        uv = rows @ inverse
        t = uv[:,0]*raw['u_range'][1]
        v = uv[:,1]*raw['v_range'][1]
        # Independent polynomial support evaluator, no display samples.
        weights = np.column_stack(((1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3))
        return weights @ np.array([(0,0,0),(1,2,0),(2,-1,0),(3,1,0)]) + v[:,None]*[.25,0,1.5]
    incidence = {}
    for cell in cells:
        for a,b in zip(cell,np.roll(cell,-1)):
            edge = tuple(sorted((int(a),int(b))))
            incidence[edge] = incidence.get(edge,0)+1
    boundary = np.array([edge for edge,count in sorted(incidence.items()) if count==1])
    triangulation = PlanarTriangulation(points,cells,boundary,boundary,
                                      np.empty((0,2),dtype=int), np.arange(9), ())
    return points,cells,evaluate,boundary,triangulation


def test_cubic_trim_physical_growth_is_repaired_without_moving_source_stations():
    points,cells,evaluate,boundary,triangulation = cavity()
    settings = SurfaceMeshOptions(min_angle=15.,prefer_quality_policy=True)
    candidate = _physical_quality_candidate(_make_candidate(points,cells,settings=settings),settings,evaluate)
    assert candidate.report['max_element_growth'] > 2.
    report = dict(topology_operations=0,insertions=0,shared_segment_splits=0,shared_nodes=[])
    intervals = {tuple(edge):(100+i,0,1) for i,edge in enumerate(boundary)}
    calls = []
    def observed_evaluation(rows):
        calls.append(len(rows))
        return evaluate(rows)
    result,_,updated = refine_physical_candidate(candidate,triangulation,settings,report,
        observed_evaluation,intervals,ComponentSeedRegistry(1000))
    assert result.report['invalid_element_count'] == 0
    assert result.report['poor_element_ids'] == []
    assert result.report['max_element_growth'] <= settings.max_element_growth
    assert updated['physical_quality_bisections'] > 0
    assert len(calls) == updated['physical_quality_bisections'] + 1
    assert updated['physical_quality_alternative_attempts'] == updated['physical_quality_alternative_bisections'] == 0
    assert updated['physical_quality_attempts'] == updated['physical_quality_bisections']
    np.testing.assert_array_equal(result.points[:len(points)],points)
    def area(p,t):
        rows=p[t]
        a,b = rows[:,1]-rows[:,0], rows[:,2]-rows[:,0]
        return sum(a[:,0]*b[:,1]-a[:,1]*b[:,0])*.5
    assert area(result.points,result.triangles) == pytest.approx(area(points,cells),abs=1e-14)
    exhausted = replace(settings,native_options=NativeMeshingOptions(max_topology_operations=1,max_insertions=1))
    result,_,updated = refine_physical_candidate(candidate,triangulation,exhausted,
        dict(report,topology_operations=1,insertions=1),evaluate,intervals,ComponentSeedRegistry(1000))
    np.testing.assert_array_equal(result.points,points)
    assert updated['physical_quality_bisections'] == 0
    assert updated['selected_route'] == 'frontal_delaunay_budget_limited'


def test_physical_invalid_candidate_is_scored_and_evaluator_errors_propagate():
    settings = SurfaceMeshOptions()
    topology = square()
    candidate = _make_candidate(topology.points,topology.triangles,settings=settings)
    invalid = _physical_quality_candidate(candidate,settings,lambda p:np.zeros((len(p),3)))
    assert invalid.report['invalid_element_count'] > 0
    valid = _physical_quality_candidate(candidate,settings,lambda p:np.column_stack((p,np.zeros(len(p)))))
    assert valid.score < invalid.score
    def stale(_):
        raise RuntimeError('stale owner')
    with pytest.raises(RuntimeError,match='stale owner'):
        _physical_quality_candidate(candidate,settings,stale)


def test_contradictory_station_identity_receipts_fail_before_evaluation():
    points,cells,evaluate,boundary,triangulation = cavity()
    settings = SurfaceMeshOptions(min_angle=15.)
    candidate = _make_candidate(points,cells,settings=settings)
    report = dict(topology_operations=0,insertions=0,shared_segment_splits=0,
        shared_nodes=[dict(local_node_id=0,node_id=100,edge_id=7,station=[0,1]),
                      dict(local_node_id=0,node_id=101,edge_id=7,station=[0,1])])
    def unexpected(_):
        pytest.fail('contradictory receipts reached owner evaluation')
    with pytest.raises(MeshError,match='conflicting shared identities'):
        refine_physical_candidate(candidate,triangulation,settings,report,
                                  unexpected,{},ComponentSeedRegistry(1000))


def test_protected_internal_winner_against_independent_alternative_splits(monkeypatch):
    """Discriminate a first-edge refusal from absence of a useful legal split."""
    points, cells, evaluate, boundary, triangulation = cavity()
    source_points, source_cells = points.copy(), cells.copy()
    before = points.tobytes(), cells.tobytes(), boundary.tobytes()
    settings = SurfaceMeshOptions(min_angle=15., prefer_quality_policy=True)
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings),
                                             settings, evaluate)
    intervals = {tuple(edge): (100 + i, 0, 1) for i, edge in enumerate(boundary)}
    original_report = dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])
    captured = {}
    class InteriorAttempt(Exception):
        pass
    def capture_interior(topology, edge, **kwargs):
        trial_points, trial_cells = topology.canonical_export()
        captured.update(points=trial_points, cells=trial_cells,
            segments=topology.constraint_edges, edge=edge, shared=topology.shared_node_ids)
        raise InteriorAttempt
    # Preserve the cavity and follow its existing legal boundary work to the
    # first attempted interior edge. Limit this fixture-only preparation to eight
    # operations; no protected edge or quality threshold is changed.
    with monkeypatch.context() as patch:
        patch.setattr(MutableT3Topology, 'bisect_interior_edge', capture_interior)
        with pytest.raises(InteriorAttempt):
            refine_physical_candidate(candidate, triangulation,
                replace(settings, native_options=NativeMeshingOptions(
                    max_topology_operations=8, max_insertions=8)), original_report,
                evaluate, intervals, ComponentSeedRegistry(1000))
    points, cells = captured['points'], captured['cells']
    np.testing.assert_array_equal(points[:len(source_points)], source_points)
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings),
                                             settings, evaluate)
    xyz = evaluate(points)
    lengths = np.linalg.norm(np.roll(xyz[cells], -1, axis=1) - xyz[cells], axis=2).mean(axis=1)
    incidence = {}
    for row, cell in enumerate(cells):
        for a, b in zip(cell, np.roll(cell, -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(row)
    growth = []
    for owners in incidence.values():
        if len(owners) == 2:
            large, small = sorted(owners, key=lambda row: (lengths[row], row), reverse=True)
            ratio = lengths[large] / lengths[small]
            if ratio > settings.max_element_growth:
                growth.append((-ratio, large))
    selected = min(growth)[1] if growth else int(candidate.report['poor_element_ids'][0]) - 1
    choices = [tuple(sorted((int(a), int(b)))) for a, b in zip(cells[selected], np.roll(cells[selected], -1))]
    winner = max(choices, key=lambda edge: (float(np.linalg.norm(xyz[edge[1]] - xyz[edge[0]])), edge))
    print('PROTECTED_WINNER ' + json.dumps(dict(nodes=len(points), cells=len(cells),
        boundary_preparations=len(points)-len(source_points), row=selected,
        edge=winner, incidence=len(incidence[winner])), sort_keys=True))
    assert len(incidence[winner]) == 2
    assert winner == captured['edge']
    constrained = replace(triangulation, points=points, triangles=cells,
                          segments=np.vstack((captured['segments'], winner)),
                          mandatory_segments=np.asarray((winner,)))
    report = dict(topology_operations=0, insertions=0, shared_segment_splits=0,
        shared_nodes=[dict(local_node_id=int(node), node_id=int(value[0]), edge_id=int(value[1]),
                           station=(value[2].numerator, value[2].denominator))
                      for node, value in captured['shared'].items()])
    bounded_settings = replace(settings, native_options=NativeMeshingOptions(max_topology_operations=2))
    result, _, receipt = refine_physical_candidate(candidate, constrained, bounded_settings, report,
        evaluate, intervals, ComponentSeedRegistry(1000))
    assert receipt['physical_quality_blocked_edge'] is None
    assert receipt['selected_route'] == 'frontal_delaunay_budget_limited'
    assert receipt['physical_quality_stop_reason'] == 'topology_budget'
    assert not receipt['physical_quality_selected_cell_exhausted']
    assert receipt['topology_operations'] == receipt['physical_quality_attempts'] == 2
    assert receipt['physical_quality_bisections'] == receipt['physical_quality_alternative_bisections'] == 1
    assert receipt['physical_quality_alternative_attempts'] == receipt['physical_quality_refused_attempts'] == 1
    np.testing.assert_array_equal(result.points[:len(points)], points)
    assert len([cell for cell in result.triangles if set(winner).issubset(cell)]) == 2
    assert result.report['max_element_growth'] < candidate.report['max_element_growth']
    assert result.score > candidate.score  # Growth progress must not use this old score alone.
    assert result.report['violation_counts'] == candidate.report['violation_counts']
    assert result.report['quality_violation_count'] == candidate.report['quality_violation_count']
    assert result.report['elements_above_maximum_growth'] == candidate.report['elements_above_maximum_growth']

    def metrics(value):
        names = ('invalid_element_count', 'quality_violation_count', 'max_element_growth',
                 'max_aspect_ratio', 'min_scaled_jacobian', 'min_angle', 'max_angle')
        return {name: value.report[name] for name in names}

    def owner_areas(topology):
        rows = topology.points[topology.triangles]
        forward, backward = rows[:, 1] - rows[:, 0], rows[:, 2] - rows[:, 0]
        areas = (forward[:, 0] * backward[:, 1] - forward[:, 1] * backward[:, 0]) * .5
        return np.bincount(topology.triangle_owners, weights=areas, minlength=len(cells))

    alternatives = []
    nodes = {}
    for ends, (owner, lower, upper) in intervals.items():
        nodes.setdefault(owner, {}).update(zip(ends, (lower, upper)))
    for record in report['shared_nodes']:
        from fractions import Fraction
        nodes.setdefault(record['edge_id'], {})[record['local_node_id']] = Fraction(*record['station'])
    current_intervals = {}
    for a, b in captured['segments']:
        owners = [edge for edge, values in nodes.items() if int(a) in values and int(b) in values]
        assert len(owners) == 1
        owner = owners[0]
        current_intervals[(int(a), int(b))] = (owner, nodes[owner][int(a)], nodes[owner][int(b)])
    for edge in sorted(set(choices) - {winner}):
        topology = MutableT3Topology(points, cells, (winner,), splittable_edges=current_intervals,
            seed_registry=ComponentSeedRegistry(1000), triangle_owners=np.arange(len(cells)))
        original_areas = owner_areas(topology)
        def validate(trial_points, trial_cells):
            trial = _physical_quality_candidate(_make_candidate(trial_points, trial_cells, settings=settings),
                                                  settings, evaluate)
            assert trial.report['invalid_element_count'] == 0
        if edge in topology.splittable_edges:
            topology.split_segment(edge, _validate_candidate=validate)
        else:
            topology.bisect_interior_edge(edge, _validate_candidate=validate)
        altered_points, altered_cells = topology.canonical_export()
        trial = _physical_quality_candidate(_make_candidate(altered_points, altered_cells, settings=settings),
                                              settings, evaluate)
        np.testing.assert_array_equal(altered_points[:len(points)], points)
        np.testing.assert_allclose(owner_areas(topology), original_areas, rtol=0., atol=1e-14)
        assert winner in topology.protected_edges
        assert len([row for row in altered_cells if set(winner).issubset(row)]) == 2
        alternatives.append(dict(edge=edge, metrics=metrics(trial),
            selection_improves=trial.score < candidate.score,
            growth_improves=trial.report['max_element_growth'] < candidate.report['max_element_growth'],
            individual_gates_pass=(trial.report['min_angle'] >= settings.min_angle
                and trial.report['max_angle'] <= settings.max_angle
                and trial.report['min_scaled_jacobian'] >= settings.min_scaled_jacobian
                and trial.report['max_aspect_ratio'] <= settings.max_aspect_ratio)))
    print('PROTECTED_ALTERNATIVES ' + json.dumps(dict(initial=metrics(candidate),
        repaired=metrics(result), alternatives=alternatives), sort_keys=True))
    assert alternatives
    assert any(item['growth_improves'] and item['individual_gates_pass'] for item in alternatives)
    assert (source_points.tobytes(), source_cells.tobytes(), boundary.tobytes()) == before
    assert original_report == dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])
    assert report['topology_operations'] == report['insertions'] == report['shared_segment_splits'] == 0


def protected_diamond():
    points = np.asarray(((-1., 0.), (1., 0.), (0., .1), (0., -.1)))
    cells = np.asarray(((0, 1, 2), (0, 3, 1)))
    boundary = np.asarray(((0, 2), (1, 2), (0, 3), (1, 3)))
    winner = (0, 1)
    evaluate = lambda rows: np.column_stack((rows, np.zeros(len(rows))))
    settings = SurfaceMeshOptions(min_angle=15., prefer_quality_policy=True,
        native_options=NativeMeshingOptions(max_topology_operations=3, max_insertions=3))
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings),
                                             settings, evaluate)
    triangulation = PlanarTriangulation(points, cells, np.vstack((boundary, winner)), boundary,
        np.asarray((winner,)), np.asarray((0, 3, 1, 2)), ())
    intervals = {tuple(edge): (100 + i, 0, 1) for i, edge in enumerate(boundary)}
    return candidate, triangulation, settings, evaluate, intervals


@pytest.mark.parametrize('remaining,already_used', ((0, 1), (1, 5), (2, 7), (3, 9), (4, 11), (5, 13)))
def test_alternative_attempt_budget_order_and_nonprogress_rollback(monkeypatch, remaining, already_used):
    from anymesher.native_v2 import _GeometryLimited
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    settings = replace(settings, native_options=NativeMeshingOptions(
        max_topology_operations=already_used + remaining))
    registry = ComponentSeedRegistry(500)
    report = dict(topology_operations=already_used, insertions=0, shared_segment_splits=0, shared_nodes=[])
    original = MutableT3Topology.split_segment
    attempts = []
    def inspect_rollback(topology, edge, **kwargs):
        points, cells = topology.canonical_export()
        before = points.tobytes(), cells.tobytes(), topology.epoch, topology._topology_index,
        attempts.append(edge)
        with pytest.raises(_GeometryLimited, match='no admissible progress'):
            original(topology, edge, **kwargs)
        after_points, after_cells = topology.canonical_export()
        assert (after_points.tobytes(), after_cells.tobytes(), topology.epoch, topology._topology_index) == before
        assert not topology.shared_node_ids and not registry.assigned_node_ids
        raise _GeometryLimited('nonprogress refused before publication')
    monkeypatch.setattr(MutableT3Topology, 'split_segment', inspect_rollback)
    points, cells = candidate.points.tobytes(), candidate.triangles.tobytes()
    result, updated_triangles, receipt = refine_physical_candidate(candidate, triangulation,
        settings, report, evaluate, intervals, registry)
    charged = min(remaining, 5)
    assert receipt['topology_operations'] == already_used + charged
    assert receipt['physical_quality_attempts'] == receipt['physical_quality_refused_attempts'] == charged
    assert receipt['physical_quality_alternative_attempts'] == max(0, charged - 1)
    assert attempts == [(1, 2), (0, 2), (1, 3), (0, 3)][:max(0, charged - 1)]
    assert receipt['physical_quality_bisections'] == receipt['physical_quality_alternative_bisections'] == 0
    assert receipt['insertions'] == 0 and not registry.assigned_node_ids
    assert result.points.tobytes() == points and result.triangles.tobytes() == cells
    assert tuple(sorted((0, 1))) in set(map(tuple, updated_triangles.segments))
    assert receipt['physical_quality_selected_cell_exhausted'] == (remaining >= 3)
    assert receipt['physical_quality_candidate_cells_exhausted'] == (remaining >= 5)
    assert receipt['selected_route'] == ('frontal_delaunay_geometry_limited' if remaining >= 5
                                         else 'frontal_delaunay_budget_limited')
    assert receipt['physical_quality_stop_reason'] == ('no_progress_for_candidate_cells' if remaining >= 5
                                                       else 'topology_budget')
    assert report == dict(topology_operations=already_used, insertions=0, shared_segment_splits=0, shared_nodes=[])
    assert registry.resolve(100, 1, 2) == 500


def test_insertion_allowance_is_not_renewed_for_alternatives():
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    settings = replace(settings, native_options=NativeMeshingOptions(max_insertions=2, max_topology_operations=8))
    registry = ComponentSeedRegistry(500)
    report = dict(topology_operations=3, insertions=1, reserved_node_reuses=1,
                  shared_segment_splits=0, shared_nodes=[])
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings, report,
                                                   evaluate, intervals, registry)
    assert receipt['topology_operations'] == 3 and receipt['insertions'] == 1
    assert receipt['physical_quality_attempts'] == receipt['physical_quality_alternative_attempts'] == 0
    assert receipt['physical_quality_stop_reason'] == 'insertion_budget'
    assert not receipt['physical_quality_selected_cell_exhausted'] and not registry.assigned_node_ids
    assert result.points.tobytes() == candidate.points.tobytes()


@pytest.mark.parametrize('failure', ('cancel', 'stale', 'malformed', 'owner_geometry_limited'))
def test_alternative_cancellation_and_owner_errors_leave_no_publication(monkeypatch, failure):
    from anygeometry import GeometryError
    from anymesher.native_v2 import _GeometryLimited
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    registry = ComponentSeedRegistry(500)
    report = dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])
    topology_seen = []
    original = MutableT3Topology.split_segment
    def capture(topology, edge, **kwargs):
        topology_seen.append((topology, topology._topology_index))
        return original(topology, edge, **kwargs)
    monkeypatch.setattr(MutableT3Topology, 'split_segment', capture)
    error = (GeometryError('stale owner in alternative') if failure == 'stale'
             else _GeometryLimited('owner failure is not a local refusal') if failure == 'owner_geometry_limited'
             else RuntimeError('cancel alternative'))
    def cancel(phase):
        if failure == 'cancel' and phase == 'native-v2 shared segment split start':
            raise error
    def owner(rows):
        if len(rows) > len(candidate.points):
            if failure in ('stale', 'owner_geometry_limited'):
                raise error
            if failure == 'malformed':
                return np.zeros((len(rows)-1, 3))
        return evaluate(rows)
    expected = MeshError if failure == 'malformed' else type(error)
    with pytest.raises(expected) as caught:
        refine_physical_candidate(candidate, triangulation, settings, report,
                                  owner, intervals, registry, cancel)
    if failure != 'malformed':
        assert caught.value is error
    assert len(topology_seen) == 1 and not registry.assigned_node_ids
    topology, index = topology_seen[0]
    points, cells = topology.canonical_export()
    assert points.tobytes() == candidate.points.tobytes() and cells.tobytes() == candidate.triangles.tobytes()
    assert topology.epoch == 0 and topology._topology_index is index
    assert not topology.shared_node_ids
    assert report == dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])


def test_physical_report_binds_truthful_threshold_counts_with_one_owner_call():
    from anymesher.core import MeshCore
    from anymesher.quality_v2 import evaluate_quality
    from anymesher.surface_mesh import _quality_threshold_report
    candidate, _, settings, evaluate, _ = protected_diamond()
    calls = []
    def owner(rows):
        calls.append(rows.copy())
        return evaluate(rows)
    result = _physical_quality_candidate(candidate, settings, owner)
    truth = _quality_threshold_report(evaluate_quality(MeshCore(evaluate(candidate.points),
                                                              candidate.triangles)), settings)
    assert len(calls) == 1
    assert result.report['violation_counts'] == truth['violation_counts']
    assert any(result.report['violation_counts'].values())


@pytest.mark.parametrize('kind', ('shared segment split', 'interior bisection'))
@pytest.mark.parametrize('stage', ('start', 'commit'))
def test_private_type_cancellation_propagates_without_publication(monkeypatch, kind, stage):
    from anymesher.native_v2 import _GeometryLimited
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    if kind == 'shared segment split':
        candidate = _make_candidate(candidate.points[:3], np.array([(0, 1, 2)]), settings=settings)
        boundary = np.array([(0, 1), (1, 2), (0, 2)])
        triangulation = PlanarTriangulation(candidate.points, candidate.triangles, boundary,
                                           boundary, np.empty((0, 2), dtype=int), np.array([0, 1, 2]), ())
        intervals = {(0, 1): (100, 0, 1), (1, 2): (101, 0, 1), (0, 2): (102, 0, 1)}
        method = 'split_segment'
    else:
        triangulation = replace(triangulation, segments=triangulation.boundary_segments,
                                mandatory_segments=np.empty((0, 2), dtype=int))
        method = 'bisect_interior_edge'
    registry = ComponentSeedRegistry(500)
    report = dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])
    snapshots = []
    original = getattr(MutableT3Topology, method)
    def capture(topology, edge, **kwargs):
        snapshots.append((topology, topology._topology_index, topology.canonical_export(),
                          tuple(topology.splittable_edges), tuple(topology.protected_edges)))
        return original(topology, edge, **kwargs)
    monkeypatch.setattr(MutableT3Topology, method, capture)
    phase = f'native-v2 {kind} {stage}'
    error = _GeometryLimited('cancellation is not a local geometry refusal')
    seen = []
    def cancel(where):
        seen.append(where)
        if where == phase:
            raise error
    with pytest.raises(_GeometryLimited) as caught:
        refine_physical_candidate(candidate, triangulation, settings, report,
                                  evaluate, intervals, registry, cancel)
    assert caught.value is error and seen[-1] == phase
    assert len(snapshots) == 1
    topology, index, (points, cells), splittable, protected = snapshots[0]
    actual_points, actual_cells = topology.canonical_export()
    assert actual_points.tobytes() == points.tobytes() == candidate.points.tobytes()
    assert actual_cells.tobytes() == cells.tobytes() == candidate.triangles.tobytes()
    assert topology.epoch == 0 and topology._topology_index is index
    assert tuple(topology.splittable_edges) == splittable and tuple(topology.protected_edges) == protected
    assert not topology.shared_node_ids and not registry.assigned_node_ids
    assert report == dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])
    assert registry.resolve(100, 1, 2) == 500


def test_alternatives_compare_to_same_committed_baseline(monkeypatch):
    import anymesher._physical_t3_refinement as physical
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    original = physical._alternative_progress
    baselines = []
    def capture(before, after):
        baselines.append((id(before), json.dumps(before, sort_keys=True)))
        return original(before, after)
    monkeypatch.setattr(physical, '_alternative_progress', capture)
    refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
        evaluate, intervals, ComponentSeedRegistry(500))
    assert len(baselines) == 2 and baselines[0] == baselines[1]


def test_geometry_limited_primary_triggers_only_selected_cell_alternatives(monkeypatch):
    from anymesher.native_v2 import _GeometryLimited
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    triangulation = replace(triangulation, segments=triangulation.boundary_segments,
                            mandatory_segments=np.empty((0, 2), dtype=int))
    attempts = []
    original = MutableT3Topology.split_segment
    def refuse_primary(topology, edge, **kwargs):
        attempts.append(edge)
        raise _GeometryLimited('local longest-edge geometry refusal')
    def capture_alternative(topology, edge, **kwargs):
        attempts.append(edge)
        return original(topology, edge, **kwargs)
    monkeypatch.setattr(MutableT3Topology, 'bisect_interior_edge', refuse_primary)
    monkeypatch.setattr(MutableT3Topology, 'split_segment', capture_alternative)
    registry = ComponentSeedRegistry(500)
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
        evaluate, intervals, registry)
    assert attempts == [(0, 1), (1, 2), (0, 2)]
    assert receipt['physical_quality_attempts'] == receipt['physical_quality_refused_attempts'] == 3
    assert receipt['physical_quality_alternative_attempts'] == 2
    assert receipt['physical_quality_selected_cell_exhausted'] and not registry.assigned_node_ids
    assert result.points.tobytes() == candidate.points.tobytes()


def test_valid_ordinary_longest_edge_retains_prior_behavior():
    candidate, triangulation, settings, evaluate, intervals = protected_diamond()
    triangulation = replace(triangulation, segments=triangulation.boundary_segments,
                            mandatory_segments=np.empty((0, 2), dtype=int))
    settings = replace(settings, native_options=NativeMeshingOptions(max_topology_operations=1))
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
        evaluate, intervals, ComponentSeedRegistry(500))
    assert receipt['physical_quality_attempts'] == receipt['physical_quality_bisections'] == 1
    assert receipt['physical_quality_alternative_attempts'] == 0
    assert len(result.points) == len(candidate.points) + 1
    assert result.report['quality_violation_count'] > candidate.report['quality_violation_count']
    np.testing.assert_array_equal(result.points[:len(candidate.points)], candidate.points)


def later_defect_fixture(*, insertion_limit=1, sharp_angle=9.9):
    candidate, _, settings, evaluate, intervals = protected_diamond()
    # Angles 20, 9.9, 150.1 degrees: bisecting the long base removes the excessive
    # maximum angle while only one child retains the original sharp corner.
    a, b = np.deg2rad((20., sharp_angle))
    side = np.sin(b) / np.sin(a + b)
    points = np.vstack((candidate.points, (3., 0.), (4., 0.),
                        (3. + side*np.cos(a), side*np.sin(a))))
    cells = np.vstack((candidate.triangles, (4, 5, 6)))
    boundary = np.asarray(((0, 2), (1, 2), (0, 3), (1, 3), (4, 5), (5, 6), (4, 6)))
    settings = replace(settings, native_options=NativeMeshingOptions(
        max_topology_operations=6, max_insertions=insertion_limit))
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings), settings, evaluate)
    triangulation = PlanarTriangulation(points, cells, np.vstack((boundary, (0, 1))), boundary,
        np.asarray(((0, 1),)), np.asarray((0, 3, 1, 2)), ())
    intervals.update({tuple(edge): (200 + i, 0, 1) for i, edge in enumerate(boundary[4:])})
    return candidate, triangulation, settings, evaluate, intervals


def test_later_defective_cell_progress_is_guarded_and_repeatable(monkeypatch):
    import anymesher._physical_t3_refinement as physical
    candidate, triangulation, settings, evaluate, intervals = later_defect_fixture()
    guard = physical._alternative_progress
    records = []
    def inspect_guard(before, after):
        records.append((id(before), before.copy(), after.copy(), guard(before, after)))
        return records[-1][-1]
    monkeypatch.setattr(physical, '_alternative_progress', inspect_guard)
    original = MutableT3Topology.split_segment
    attempts = []
    def capture(topology, edge, **kwargs):
        attempts.append(edge)
        return original(topology, edge, **kwargs)
    monkeypatch.setattr(MutableT3Topology, 'split_segment', capture)
    results = []
    for _ in range(2):
        start = len(records)
        result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
            dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
            evaluate, intervals, ComponentSeedRegistry(500))
        assert receipt['physical_quality_attempts'] == 6
        assert receipt['physical_quality_refused_attempts'] == 5
        assert receipt['physical_quality_alternative_attempts'] == 5
        assert receipt['physical_quality_alternative_bisections'] == 1
        assert receipt['physical_quality_selected_cell_exhausted']
        assert not receipt['physical_quality_candidate_cells_exhausted']
        assert receipt['physical_quality_exhausted_cell_count'] == 2
        assert receipt['physical_quality_candidate_cells_visited'] == 3
        assert result.report['violation_counts']['maximum_angle'] < candidate.report['violation_counts']['maximum_angle']
        assert all(result.report['violation_counts'][name] <= value
                   for name, value in candidate.report['violation_counts'].items())
        assert result.report['quality_violation_count'] <= candidate.report['quality_violation_count']
        assert result.points[:len(candidate.points)].tobytes() == candidate.points.tobytes()
        current_records = records[start:]
        assert len(current_records) == 5 and len({record[0] for record in current_records}) == 1
        assert [record[-1] for record in current_records] == [False]*4 + [True]
        results.append((result.points.tobytes(), result.triangles.tobytes(), receipt))
    assert results[0] == results[1]
    expected = [(1, 2), (0, 2), (1, 3), (0, 3), (4, 5)]
    assert attempts == expected + expected  # Shared refused edge (0, 1) is charged once per baseline.


def test_later_cell_primary_cannot_bypass_nonprogress_guard():
    candidate, triangulation, settings, evaluate, intervals = later_defect_fixture(sharp_angle=9.)
    registry = ComponentSeedRegistry(500)
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
        evaluate, intervals, registry)
    assert receipt['physical_quality_attempts'] == receipt['physical_quality_refused_attempts'] == 6
    assert receipt['physical_quality_alternative_attempts'] == 5
    assert receipt['physical_quality_candidate_cells_visited'] == 3
    assert receipt['physical_quality_blocked_edge'] == (4, 5)
    assert receipt['physical_quality_bisections'] == 0 and not registry.assigned_node_ids
    assert result.points.tobytes() == candidate.points.tobytes()
    assert result.triangles.tobytes() == candidate.triangles.tobytes()


def test_refusal_cache_clears_after_later_cell_commit(monkeypatch):
    candidate, triangulation, settings, evaluate, intervals = later_defect_fixture(insertion_limit=2)
    settings = replace(settings, native_options=NativeMeshingOptions(
        max_topology_operations=9, max_insertions=2))
    attempts = []
    original = MutableT3Topology.split_segment
    def capture(topology, edge, **kwargs):
        attempts.append((topology.epoch, edge))
        return original(topology, edge, **kwargs)
    monkeypatch.setattr(MutableT3Topology, 'split_segment', capture)
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
        evaluate, intervals, ComponentSeedRegistry(500))
    assert attempts == [(0, (1, 2)), (0, (0, 2)), (0, (1, 3)), (0, (0, 3)), (0, (4, 5)),
                        (1, (1, 2)), (1, (0, 2))]
    assert receipt['physical_quality_attempts'] == 9 and receipt['physical_quality_bisections'] == 1
    assert not receipt['physical_quality_candidate_cells_exhausted']
    assert result.points[:len(candidate.points)].tobytes() == candidate.points.tobytes()


def test_growth_priority_rows_and_ties_precede_physical_poor_rows():
    points = np.asarray(((0., 0.), (2., 0.), (0., 2.), (0., -.1),
                         (10., 0.), (12., 0.), (10., 2.), (10., -.1)))
    cells = np.asarray(((0, 1, 2), (0, 3, 1), (4, 5, 6), (4, 7, 5)))
    segments = np.asarray(((0, 1), (0, 2), (1, 2), (0, 3), (1, 3),
                           (4, 5), (4, 6), (5, 6), (4, 7), (5, 7)))
    settings = SurfaceMeshOptions(min_angle=15., prefer_quality_policy=True,
        native_options=NativeMeshingOptions(max_topology_operations=4, max_insertions=1))
    evaluate = lambda rows: np.column_stack((rows, np.zeros(len(rows))))
    candidate = _physical_quality_candidate(_make_candidate(points, cells, settings=settings), settings, evaluate)
    triangulation = PlanarTriangulation(points, cells, segments, segments,
        np.empty((0, 2), dtype=int), np.arange(len(points)), ())
    result, _, receipt = refine_physical_candidate(candidate, triangulation, settings,
        dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
        evaluate, {}, ComponentSeedRegistry(500))
    # Equal growth ratios choose row 0 then row 2, ahead of physical-poor row 1.
    assert receipt['physical_quality_candidate_cells_visited'] == 2
    assert receipt['physical_quality_exhausted_cell_count'] == 1
    assert receipt['physical_quality_attempts'] == 4
    assert receipt['physical_quality_blocked_edge'] == (5, 6)
    assert receipt['physical_quality_stop_reason'] == 'topology_budget'
    assert result.points.tobytes() == points.tobytes() and result.triangles.tobytes() == cells.tobytes()


@pytest.mark.parametrize('phase', ('native-v2 physical growth priority',
                                 'native-v2 physical quality candidate priority'))
def test_candidate_priority_construction_is_cancellable(phase):
    from anymesher.native_v2 import _GeometryLimited
    candidate, triangulation, settings, evaluate, intervals = later_defect_fixture()
    registry = ComponentSeedRegistry(500)
    error = _GeometryLimited('cancel while constructing candidate priorities')
    report = dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])
    def cancel(where):
        if where == phase:
            raise error
    with pytest.raises(_GeometryLimited) as caught:
        refine_physical_candidate(candidate, triangulation, settings, report,
                                  evaluate, intervals, registry, cancel)
    assert caught.value is error and not registry.assigned_node_ids
    assert report == dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[])


def test_cancellation_in_later_cell_preserves_refused_baseline(monkeypatch):
    from anymesher.native_v2 import _GeometryLimited
    candidate, triangulation, settings, evaluate, intervals = later_defect_fixture()
    topology_seen = []
    original = MutableT3Topology.split_segment
    def capture(topology, edge, **kwargs):
        topology_seen.append((topology, topology._topology_index))
        return original(topology, edge, **kwargs)
    monkeypatch.setattr(MutableT3Topology, 'split_segment', capture)
    registry = ComponentSeedRegistry(500)
    error = _GeometryLimited('cancel later cell before any publication')
    visits = 0
    def cancel(phase):
        nonlocal visits
        if phase == 'native-v2 physical quality candidate cell':
            visits += 1
            if visits == 3:
                raise error
    with pytest.raises(_GeometryLimited) as caught:
        refine_physical_candidate(candidate, triangulation, settings,
            dict(topology_operations=0, insertions=0, shared_segment_splits=0, shared_nodes=[]),
            evaluate, intervals, registry, cancel)
    assert caught.value is error and visits == 3 and len(topology_seen) == 4
    assert not registry.assigned_node_ids
    for topology, index in topology_seen:
        points, cells = topology.canonical_export()
        assert points.tobytes() == candidate.points.tobytes()
        assert cells.tobytes() == candidate.triangles.tobytes()
        assert topology.epoch == 0 and topology._topology_index is index and not topology.shared_node_ids


@pytest.mark.parametrize('change,expected', (
    ({}, False),
    ({'max_element_growth': 1.9}, True),
    ({'max_element_growth': 1.9, 'violation_counts': {'aspect_ratio': 1}}, False),
    ({'max_element_growth': 1.9, 'quality_violation_count': 3}, False),
    ({'max_element_growth': 1.9, 'elements_above_maximum_growth': 2}, False),
    ({'max_element_growth': 1.9, 'invalid_element_count': 1}, False),
    ({'max_element_growth': 2.1, 'violation_counts': {'minimum_angle': 1}}, True),
))
def test_physical_progress_guard_exact_count_and_growth_contract(change, expected):
    """Exercise guard arithmetic separately from owner-quality production."""
    from anymesher._physical_t3_refinement import _alternative_progress
    before = dict(invalid_element_count=0, quality_violation_count=2,
        elements_above_maximum_growth=1, max_element_growth=2.,
        violation_counts=dict(scaled_jacobian=0, aspect_ratio=0, minimum_angle=2,
                              maximum_angle=0, warpage=0))
    after = dict(before, violation_counts=dict(before['violation_counts']))
    for name, value in change.items():
        if name == 'violation_counts':
            after[name].update(value)
        else:
            after[name] = value
    snapshot = json.dumps(before, sort_keys=True)
    assert _alternative_progress(before, after) is expected
    assert json.dumps(before, sort_keys=True) == snapshot
