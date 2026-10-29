"""Compare the combined route to the unchanged full quality composition."""
import numpy as np
import pytest
import importlib
sm=importlib.import_module('anymesher.surface_mesh')
from anymesher import _frontal_transition_quality as transition
from test_frontal_transition_quality import small_owner_candidate


def composed(points,cells,settings,xyz,*,_candidate_maker=None,_physical_scorer=None,**metadata):
    scorer=sm._physical_quality_candidate_from_xyz if _physical_scorer is None else _physical_scorer
    maker=sm._make_candidate if _candidate_maker is None else _candidate_maker
    return scorer(maker(points,cells,
        settings=settings,**metadata),settings,xyz)


def equivalent(a,b):
    assert a.report==b.report and a.score==b.score
    np.testing.assert_array_equal(a.points,b.points)
    np.testing.assert_array_equal(a.triangles,b.triangles)
    np.testing.assert_array_equal(a.aspect_ratios,b.aspect_ratios)
    assert (a.flips,a.moved_nodes,a.added_points,a.rounds)==(b.flips,b.moved_nodes,b.added_points,b.rounds)


def inputs(kind):
    points=np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    cells=np.array(((0,1,2),(0,2,3)),dtype=np.int64)
    if kind=='inverted':cells=cells[:,::-1].copy()
    elif kind=='degenerate':points[2]=points[1]
    elif kind=='nan':points[2,0]=np.nan
    elif kind=='infinite':points[2,0]=np.inf
    elif kind=='empty':cells=np.empty((0,3),dtype=np.int64)
    elif kind=='nonmanifold':cells=np.array(((0,1,2),(0,1,3),(0,1,2)),dtype=np.int64)
    elif kind=='index':cells[0,2]=5
    elif kind=='negative_index':cells[0,2]=-1
    elif kind=='float32':points=points.astype(np.float32)
    elif kind=='strided':points=np.repeat(points,2,axis=0)[::2]
    elif kind=='scaled':points*=1.e4
    elif kind=='tiny':points*=1.e-9
    elif kind=='overflow':points*=1.e200
    elif kind in ('threshold_below','threshold_above'):
        threshold=np.finfo(np.float64).eps*32.
        points=np.array(((0.,0.),(1.,0.),(0.,np.nextafter(threshold,0. if kind.endswith('below') else np.inf))))
        cells=np.array(((0,1,2),),dtype=np.int64)
    xyz=np.column_stack((points,.125*points[:,0]**2+.25*points[:,1]**2))
    if kind=='bad_xyz':xyz=xyz[:,:2]
    elif kind=='nonfinite_xyz':xyz[0,2]=np.nan
    return points,cells,xyz


@pytest.mark.parametrize('kind',('valid','inverted','degenerate','nan','infinite','empty',
    'nonmanifold','index','negative_index','float32','strided','scaled','tiny','overflow','threshold_below',
    'threshold_above','bad_xyz','nonfinite_xyz'))
def test_complete_composition_equivalence_or_same_refusal(kind):
    points,cells,xyz=inputs(kind);settings=sm.SurfaceMeshOptions(prefer_quality_policy=True)
    metadata=dict(flips=3,moved_nodes=(2,),added_points=4,rounds=5)
    before=points.tobytes(),cells.tobytes(),xyz.tobytes();old_stats={};new_stats={}
    try:expected=composed(points,cells,settings,xyz,statistics=old_stats,**metadata)
    except Exception as error:
        with pytest.raises(type(error)) as caught:
            sm._make_physical_candidate(points,cells,settings,xyz,statistics=new_stats,**metadata)
        assert str(caught.value)==str(error)
    else:equivalent(sm._make_physical_candidate(points,cells,settings,xyz,statistics=new_stats,**metadata),expected)
    assert old_stats==new_stats=={'quality_evaluations':1}
    assert before==(points.tobytes(),cells.tobytes(),xyz.tobytes())


def test_combined_valid_route_omits_discarded_chart_call(monkeypatch):
    points,cells,xyz=inputs('valid');settings=sm.SurfaceMeshOptions()
    # Profile actual canonical function calls without replacing/pinning methods.
    import sys
    chart_calls=[];original=sys.getprofile()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is sm._triangle_quality.__code__:chart_calls.append(True)
    try:
        sys.setprofile(profile)
        sm._make_physical_candidate(points,cells,settings,xyz)
    finally:sys.setprofile(original)
    assert chart_calls==[]


@pytest.mark.parametrize('changed',('quality','native','numpy'))
def test_unknown_quality_context_keeps_original_calls(monkeypatch,changed):
    points,cells,xyz=inputs('valid');settings=sm.SurfaceMeshOptions();calls=[]
    original=sm._triangle_quality
    def observe(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    if changed=='quality':monkeypatch.setattr(sm,'_triangle_quality',observe)
    elif changed=='native':
        native=sm._native_cpp.native_element_quality
        monkeypatch.setattr(sm._native_cpp,'native_element_quality',lambda *a,**kw:native(*a,**kw))
    else:
        norm=np.linalg.norm
        monkeypatch.setattr(np.linalg,'norm',lambda *a,**kw:norm(*a,**kw))
    # The chart scorer itself can be observed by profile without changing the
    # context whose fallback is under test.
    import sys
    old=sys.getprofile();seen=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is original.__code__:seen.append(True)
    try:
        sys.setprofile(profile)
        got=sm._make_physical_candidate(points,cells,settings,xyz)
    finally:sys.setprofile(old)
    assert seen
    equivalent(got,composed(points,cells,settings,xyz))


def test_real_observer_admission_and_metadata_equal_original_composition(monkeypatch):
    candidate,protected,settings,owner=small_owner_candidate(trials=31)
    before=candidate.points.tobytes(),candidate.triangles.tobytes()
    kwargs=dict(evaluate_coordinates=owner,allow_partial_progress=True,line_search_admission=True)
    events=[]
    def checkpoint(phase):events.append(phase)
    actual,report=transition.repair_frontal_transition(candidate,protected,settings,dict(topology_operations=0),checkpoint,**kwargs)
    assert report['chart_transition_repair']['candidate_scorings']>0
    actual_events=tuple(events);events.clear()
    monkeypatch.setattr(sm,'_make_physical_candidate',composed)
    expected,reference=transition.repair_frontal_transition(candidate,protected,settings,dict(topology_operations=0),checkpoint,**kwargs)
    assert tuple(events)==actual_events
    equivalent(actual,expected)
    for record in (report,reference):
        for key in ('candidate_observation_seconds','candidate_scoring_seconds'):
            record['chart_transition_repair'].pop(key)
    assert report==reference
    assert before==(candidate.points.tobytes(),candidate.triangles.tobytes())


def test_custom_physical_scorer_receives_complete_chart_report(monkeypatch):
    points,cells,xyz=inputs('valid');settings=sm.SurfaceMeshOptions();original=sm._physical_quality_candidate_from_xyz
    reports=[]
    def observe(candidate,settings,xyz):
        reports.append(dict(candidate.report))
        assert 'max_element_growth' in candidate.report and 'min_angle' in candidate.report
        return original(candidate,settings,xyz)
    monkeypatch.setattr(sm,'_physical_quality_candidate_from_xyz',observe)
    equivalent(sm._make_physical_candidate(points,cells,settings,xyz),composed(points,cells,settings,xyz))
    assert reports[0]==reports[1]


@pytest.mark.parametrize('patch',('quality_code','native_wrapper','error_mode'))
def test_in_place_context_changes_use_original_chart_path(monkeypatch,patch):
    points,cells,xyz=inputs('valid');settings=sm.SurfaceMeshOptions()
    if patch=='quality_code':
        def refused(*args,**kwargs):raise RuntimeError('original custom chart refusal')
        monkeypatch.setattr(sm._triangle_quality,'__code__',refused.__code__)
        with pytest.raises(RuntimeError,match='original custom chart refusal'):
            sm._make_physical_candidate(points,cells,settings,xyz)
    elif patch=='native_wrapper':
        # A replaced native service remains a callback-bearing old-path operand.
        original=sm._native_cpp.native_element_quality;events=[]
        def observe(*args,**kwargs):events.append(True);return original(*args,**kwargs)
        monkeypatch.setattr(sm._native_cpp,'native_element_quality',observe)
        equivalent(sm._make_physical_candidate(points,cells,settings,xyz),composed(points,cells,settings,xyz))
        if sm._native_cpp.COMPILED_QUALITY_PIPELINE_AVAILABLE:assert events
    else:
        with np.errstate(divide='raise'):
            equivalent(sm._make_physical_candidate(points,cells,settings,xyz),composed(points,cells,settings,xyz))


@pytest.mark.parametrize('replacement',('late_maker','maker_changes_scorer'))
def test_actual_observer_preserves_old_callable_capture_order(monkeypatch,replacement):
    candidate,protected,settings,owner=small_owner_candidate(trials=1)
    maker=sm._make_candidate;events=[]
    class Observed(Exception):pass
    def forbidden(*args,**kwargs):raise AssertionError('late replacement was incorrectly invoked')
    if replacement=='maker_changes_scorer':
        def custom_maker(*args,**kwargs):
            events.append('maker')
            result=maker(*args,**kwargs)
            monkeypatch.setattr(sm,'_physical_quality_candidate_from_xyz',forbidden)
            return result
        monkeypatch.setattr(sm,'_make_candidate',custom_maker)
    def checkpoint(phase):
        if replacement=='late_maker':monkeypatch.setattr(sm,'_make_candidate',forbidden)
    def drive(points,triangles,segments,poor,**kwargs):
        # Exercise the actual observer with genuine scoring, then stop before
        # unrelated final-result evaluation can observe the deliberate patch.
        kwargs['candidate_callback'](points.copy(),owner(points),0.,1)
        events.append('observer_returned')
        raise Observed()
    monkeypatch.setattr(transition,'repair_joint_triangle_quality',drive)
    with pytest.raises(Observed):
        transition.repair_frontal_transition(candidate,protected,settings,dict(topology_operations=0),
            checkpoint,evaluate_coordinates=owner,allow_partial_progress=True,line_search_admission=True)
    assert events==(['maker','observer_returned'] if replacement=='maker_changes_scorer' else ['observer_returned'])


def test_standalone_composition_captures_physical_scorer_before_custom_maker(monkeypatch):
    points,cells,xyz=inputs('valid');settings=sm.SurfaceMeshOptions()
    expected=composed(points,cells,settings,xyz);maker=sm._make_candidate
    def custom_maker(*args,**kwargs):
        result=maker(*args,**kwargs)
        def forbidden(*args,**kwargs):raise AssertionError('late scorer replacement')
        monkeypatch.setattr(sm,'_physical_quality_candidate_from_xyz',forbidden)
        return result
    monkeypatch.setattr(sm,'_make_candidate',custom_maker)
    equivalent(sm._make_physical_candidate(points,cells,settings,xyz),expected)
