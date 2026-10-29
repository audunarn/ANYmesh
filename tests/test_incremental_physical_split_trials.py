import numpy as np
import pytest
from anymesher.native_v2 import MutableT3Topology
from anymesher.surface_mesh import SurfaceMeshOptions,_make_candidate,_physical_quality_candidate_from_xyz
from anymesher._physical_split_trials import PhysicalSplitTrials
from anymesher._physical_t3_refinement import _alternative_progress


def score(points,cells,xyz,settings):
    return _physical_quality_candidate_from_xyz(_make_candidate(points,cells,settings=settings),settings,xyz)


@pytest.mark.parametrize('warp',(0.,.2))
@pytest.mark.parametrize('reverse_rows',(False,True))
def test_actual_split_full_report_score_and_decision_are_exact(warp,reverse_rows):
    points=np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    cells=np.array(((0,1,2),(0,2,3)),dtype=np.int64)
    evaluate=lambda rows:np.column_stack((rows,warp*rows[:,0]*rows[:,1]))
    settings=SurfaceMeshOptions();current=score(points,cells,evaluate(points),settings)
    trials=PhysicalSplitTrials(settings);trials.rebase(current,evaluate(points))
    topology=MutableT3Topology(points,cells);topology.bisect_interior_edge((0,2))
    new_points,new_cells=topology.canonical_export()
    if reverse_rows:new_cells=new_cells[::-1].copy()
    expected=score(new_points,new_cells,evaluate(new_points),settings)
    actual=trials.trial(new_points,new_cells,evaluate(new_points))
    assert actual is not None and actual.report==expected.report and actual.score==expected.score
    np.testing.assert_array_equal(actual.aspect_ratios,expected.aspect_ratios)
    assert _alternative_progress(current.report,actual.report)==_alternative_progress(current.report,expected.report)
    assert trials.current is current
    trials.rebase(actual,evaluate(new_points));assert trials.current is actual


@pytest.mark.parametrize('point',((.5,.5),(.5,0.),(0.,0.)))
def test_uv_validity_orientation_and_degenerate_reports_match(point):
    points=np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    cells=np.array(((0,1,2),(0,2,3)),dtype=np.int64)
    evaluate=lambda rows:np.column_stack((rows,np.zeros(len(rows))))
    settings=SurfaceMeshOptions();current=score(points,cells,evaluate(points),settings)
    trials=PhysicalSplitTrials(settings);trials.rebase(current,evaluate(points))
    made=np.vstack((points,point));split=np.array(((0,1,4),(1,2,4),(0,4,3),(4,2,3)),dtype=np.int64)
    expected=score(made,split,evaluate(made),settings);actual=trials.trial(made,split,evaluate(made))
    assert actual is not None and actual.report==expected.report and actual.score==expected.score


def test_changed_prefix_or_global_bbox_falls_back_without_committing_state():
    points=np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    cells=np.array(((0,1,2),(0,2,3)),dtype=np.int64);xyz=np.column_stack((points,np.zeros(4)))
    settings=SurfaceMeshOptions();current=score(points,cells,xyz,settings);trials=PhysicalSplitTrials(settings);trials.rebase(current,xyz)
    made=np.vstack((points,(2.,2.)));split=np.array(((0,1,4),(1,2,4),(0,4,3),(4,2,3)),dtype=np.int64)
    assert trials.trial(made,split,np.column_stack((made,np.zeros(5)))) is None
    made[-1]=(.5,.5);made[0,0]=.1
    assert trials.trial(made,split,np.column_stack((made,np.zeros(5)))) is None
    assert trials.current is current and not trials.used
