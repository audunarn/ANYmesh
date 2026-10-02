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
