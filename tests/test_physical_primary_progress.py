"""Owner opt-in primary splits must preserve the physical progress guard."""
from dataclasses import replace
import json

import numpy as np
import pytest

from anymesher.native_v2 import MutableT3Topology, ComponentSeedRegistry, NativeMeshingOptions
from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate, _physical_quality_candidate
from anymesher.triangulation import PlanarTriangulation
from anymesher._physical_t3_refinement import refine_physical_candidate, _alternative_progress


def cavity(points, cells, *, budget=1):
    points, cells=np.asarray(points,dtype=float),np.asarray(cells,dtype=np.int64)
    incidence={}
    for row in cells:
        for a,b in zip(row,np.roll(row,-1)):
            edge=tuple(sorted((int(a),int(b))))
            incidence[edge]=incidence.get(edge,0)+1
    boundary=np.asarray([edge for edge,count in sorted(incidence.items()) if count==1])
    triangulation=PlanarTriangulation(points,cells,boundary,boundary,np.empty((0,2),dtype=np.int64),
                                      np.arange(len(points)),())
    calls=[]
    def evaluate(rows):
        calls.append(rows.copy())
        return np.column_stack((rows,np.zeros(len(rows))))
    settings=SurfaceMeshOptions(min_angle=15.,prefer_quality_policy=True,
        native_options=NativeMeshingOptions(max_topology_operations=budget,max_insertions=2))
    candidate=_physical_quality_candidate(_make_candidate(points,cells,settings=settings),settings,evaluate)
    calls.clear()
    intervals={tuple(edge):(100+number,0,1) for number,edge in enumerate(boundary)}
    return candidate,triangulation,settings,evaluate,intervals,calls


def degrading_cavity(*, budget=1):
    return cavity(((-1.,0.),(1.,0.),(0.,.1),(0.,-.1)),((0,1,2),(0,3,1)),budget=budget)


def progress_cavity(*, budget=1):
    return cavity(((0.,0.),(1.,0.),(.5,1.8),(.5,-.3)),((0,1,2),(0,3,1)),budget=budget)


def work_report():
    return dict(topology_operations=0,insertions=0,shared_segment_splits=0,shared_nodes=[])


def physical_summary(report):
    return {key:report[key] for key in ('violation_counts','invalid_element_count',
        'quality_violation_count','elements_above_maximum_growth','max_element_growth')}


def test_real_degrading_primary_is_refused_on_owner_opt_in():
    candidate,triangulation,settings,evaluate,intervals,calls=degrading_cavity()
    primitive=MutableT3Topology(candidate.points,candidate.triangles,triangulation.segments)
    primitive.bisect_interior_edge((0,1))
    proposal=_physical_quality_candidate(_make_candidate(primitive.points,primitive.triangles,settings=settings),
                                         settings,evaluate)
    print('DEGRADING_PRIMARY',json.dumps({'before':physical_summary(candidate.report),
        'proposed':physical_summary(proposal.report)},sort_keys=True))
    assert proposal.report['invalid_element_count']==0
    assert proposal.report['quality_violation_count']>candidate.report['quality_violation_count']
    assert proposal.report['violation_counts']['minimum_angle']>candidate.report['violation_counts']['minimum_angle']
    assert not _alternative_progress(candidate.report,proposal.report)
    calls.clear()
    registry=ComponentSeedRegistry(500)
    before=candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()
    result,output,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        evaluate,intervals,registry,allow_physical_flips=True)
    assert receipt['physical_quality_bisections']==0
    assert receipt['physical_quality_attempts']==receipt['physical_quality_refused_attempts']==1
    assert receipt['physical_quality_alternative_attempts']==0
    assert receipt['topology_operations']==1 and receipt['insertions']==0
    assert receipt['physical_quality_stop_reason']=='topology_budget'
    assert result.points.tobytes()==before[0] and result.triangles.tobytes()==before[1]
    assert output.segments.tobytes()==before[2]
    assert len(calls)==2 and not registry.assigned_node_ids


def test_real_progress_primary_passes_guard_without_extra_owner_work():
    candidate,triangulation,settings,evaluate,intervals,calls=progress_cavity()
    report=work_report()
    default,default_tri,default_receipt=refine_physical_candidate(candidate,triangulation,settings,report,
        evaluate,intervals,ComponentSeedRegistry(500))
    assert len(calls)==2
    calls.clear()
    registry=ComponentSeedRegistry(500)
    result,output,receipt=refine_physical_candidate(candidate,triangulation,settings,report,
        evaluate,intervals,registry,allow_physical_flips=True)
    print('PROGRESS_PRIMARY',json.dumps({'before':physical_summary(candidate.report),
        'accepted':physical_summary(result.report)},sort_keys=True))
    assert _alternative_progress(candidate.report,result.report)
    assert all(value==0 for value in candidate.report['violation_counts'].values())
    assert all(value==0 for value in result.report['violation_counts'].values())
    assert candidate.report['elements_above_maximum_growth']==2
    assert result.report['elements_above_maximum_growth']==0
    assert result.report['max_element_growth']<candidate.report['max_element_growth']
    assert receipt['physical_quality_bisections']==receipt['topology_operations']==receipt['insertions']==1
    assert receipt['physical_quality_alternative_attempts']==receipt['physical_quality_alternative_bisections']==0
    assert receipt['physical_quality_flip_attempts']==receipt['physical_quality_flips']==0
    assert result.points.tobytes()==default.points.tobytes()
    assert result.triangles.tobytes()==default.triangles.tobytes() and result.report==default.report
    assert output.segments.tobytes()==default_tri.segments.tobytes()
    assert all(receipt[key]==value for key,value in default_receipt.items())
    assert len(calls)==2 and registry.assigned_node_ids==(500,)
    assert result.points[:len(candidate.points)].tobytes()==candidate.points.tobytes()
    assert report==work_report()


def test_generic_degrading_primary_retains_existing_behavior():
    candidate,triangulation,settings,evaluate,intervals,calls=degrading_cavity()
    result,_,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        evaluate,intervals,ComponentSeedRegistry(500))
    assert receipt['physical_quality_bisections']==1 and len(result.points)==len(candidate.points)+1
    assert result.report['quality_violation_count']==4>candidate.report['quality_violation_count']
    assert not _alternative_progress(candidate.report,result.report)
    assert not any(key.startswith('physical_quality_flip') for key in receipt)
    assert len(calls)==2


def test_degrading_primary_rolls_back_entire_native_state_before_refusal(monkeypatch):
    from test_physical_quality_flips import state
    candidate,triangulation,settings,evaluate,intervals,calls=degrading_cavity()
    observed=[]
    original=MutableT3Topology.bisect_interior_edge
    def inspect(topology,edge,**kwargs):
        from anymesher.native_v2 import _GeometryLimited
        before=state(topology)
        with pytest.raises(_GeometryLimited,match='no admissible progress'):
            original(topology,edge,**kwargs)
        assert state(topology)==before
        observed.append(edge)
        raise _GeometryLimited('independent confirmed physical refusal')
    monkeypatch.setattr(MutableT3Topology,'bisect_interior_edge',inspect)
    registry=ComponentSeedRegistry(500)
    result,_,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        evaluate,intervals,registry,allow_physical_flips=True)
    assert observed==[(0,1)] and receipt['physical_quality_refused_attempts']==1
    assert result.points.tobytes()==candidate.points.tobytes()
    assert result.triangles.tobytes()==candidate.triangles.tobytes()
    assert not registry.assigned_node_ids and len(calls)==2


@pytest.mark.parametrize('budget', (1,9,10))
def test_real_primary_refusals_then_guarded_flip_share_existing_pool(budget):
    candidate,triangulation,settings,evaluate,_,calls=cavity(
        ((-1.,0.),(1.,0.),(0.,.1),(0.,-.1),
         (3.,0.),(5.,0.),(5.,1.),(3.,.55)),
        ((0,1,2),(0,3,1),(4,5,6),(4,6,7)),budget=budget)
    registry=ComponentSeedRegistry(500)
    before=candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()
    result,output,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        evaluate,{},registry,allow_physical_flips=True)
    assert receipt['physical_quality_bisections']==receipt['insertions']==0
    assert receipt['topology_operations']==receipt['physical_quality_attempts']==budget
    assert receipt['physical_quality_flip_attempts']==max(0,budget-8)
    assert receipt['physical_quality_flips']==int(budget==10)
    assert receipt['physical_quality_refused_attempts']==budget-int(budget==10)
    assert receipt['physical_quality_stop_reason']=='topology_budget'
    assert result.points.tobytes()==before[0] and output.segments.tobytes()==before[2]
    assert not registry.assigned_node_ids
    if budget==10:
        assert _alternative_progress(candidate.report,result.report)
        assert result.report['poor_element_ids']
        assert len(calls)==3
    else:
        assert result.triangles.tobytes()==before[1]
        assert len(calls)==(2 if budget==1 else 3)
    assert (candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes())==before
