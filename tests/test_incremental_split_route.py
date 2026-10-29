import sys
from dataclasses import dataclass,replace
import numpy as np
import pytest
from anygeometry._detached_geometry_read import MaterialNumericalRead
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher._material_region_binding import material_read_scope
from anymesher._physical_t3_refinement import refine_physical_candidate
from anymesher._physical_split_trials import PhysicalSplitTrials
from anymesher.surface_mesh import SurfaceMeshOptions,_make_candidate,_physical_quality_candidate
from anymesher.native_v2 import NativeMeshingOptions,ComponentSeedRegistry
from test_material_numerical_read import fixture


@dataclass(frozen=True)
class Packet:
    points:object
    triangles:object
    segments:object


@pytest.mark.parametrize('wrapped',(False,True,'instance_rows'))
def test_actual_split_route_matches_full_and_keeps_wrapper_batch_calls(monkeypatch,wrapped):
    model,face,binding,options=fixture();chart=AnalyticMetricChart(model,face,region_binding=binding)
    points=np.array(((.1,.5),(.9,.5),(.5,.9),(.5,.4)))@chart.transform
    cells=np.array(((0,1,2),(0,3,1)),dtype=np.int64)
    segments=np.array(((0,2),(1,2),(0,3),(1,3)),dtype=np.int64)
    settings=SurfaceMeshOptions(native_options=NativeMeshingOptions(max_topology_operations=2,max_insertions=2))
    candidate=_physical_quality_candidate(_make_candidate(points,cells,settings=settings),settings,chart.evaluate)
    original=MaterialNumericalRead.evaluate
    if wrapped is True:
        def delegate(self,parameters,**kwargs):
            return original(self,parameters,**kwargs)
        monkeypatch.setattr(MaterialNumericalRead,'evaluate',delegate)
    elif wrapped=='instance_rows':
        original_rows=chart._rows
        chart._rows=lambda parameters:original_rows(parameters)+np.array((.001*len(parameters),0.))
    outcomes=[];rows=[];hits=[]
    for enabled in (True,False):
        sizes=[];used=[]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is original.__code__:sizes.append(len(frame.f_locals['parameters']))
            if event=='return' and frame.f_code is PhysicalSplitTrials.trial.__code__ and arg is not None:used.append(1)
        with monkeypatch.context() as patch:
            if not enabled:patch.setattr(PhysicalSplitTrials,'available',lambda self:False)
            with material_read_scope(binding,options):
                sys.setprofile(profile)
                try:
                    output=refine_physical_candidate(candidate,Packet(points,cells,segments),settings,
                        dict(topology_operations=0,insertions=0,shared_segment_splits=0,shared_nodes=[]),
                        chart.evaluate,{},ComponentSeedRegistry(20),allow_physical_flips=True)
                finally:sys.setprofile(None)
            outcomes.append(output);rows.append(sizes);hits.append(used)
    assert not hits[1]
    if wrapped!='instance_rows':assert hits[0]
    first,last=outcomes
    np.testing.assert_array_equal(first[0].points,last[0].points)
    np.testing.assert_array_equal(first[0].triangles,last[0].triangles)
    assert first[0].report==last[0].report and first[2]==last[2]
    assert first[0].score==last[0].score
    if wrapped:
        assert rows[0]==rows[1] and 1 not in rows[0]
    else:
        assert 1 in rows[0] and 1 not in rows[1]


@pytest.mark.parametrize('target',('validate_mesh','mesh_quality','evaluate_quality','surface_alias'))
@pytest.mark.parametrize('failure',(False,True))
def test_unknown_full_quality_service_keeps_original_call_order_before_reuse(monkeypatch,target,failure):
    from anymesher import quality_v2
    from anymesher import surface_mesh as surface
    model,face,binding,options=fixture();chart=AnalyticMetricChart(model,face,region_binding=binding)
    points=np.array(((.1,.5),(.9,.5),(.5,.9),(.5,.4)))@chart.transform
    cells=np.array(((0,1,2),(0,3,1)),dtype=np.int64)
    segments=np.array(((0,2),(1,2),(0,3),(1,3)),dtype=np.int64)
    settings=SurfaceMeshOptions(native_options=NativeMeshingOptions(max_topology_operations=2,max_insertions=2))
    candidate=_physical_quality_candidate(_make_candidate(points,cells,settings=settings),settings,chart.evaluate)
    owner = surface if target=='surface_alias' else quality_v2
    name = 'evaluate_quality' if target=='surface_alias' else target
    original=getattr(owner,name);error=RuntimeError('original quality service error')
    outcomes=[];calls=[]
    for enabled in (True,False):
        order=[]
        def delegate(*args,**kwargs):
            order.append('service')
            if failure:raise error
            return original(*args,**kwargs)
        def profile(frame,event,arg):
            if event=='call' and frame.f_code is PhysicalSplitTrials.rebase.__code__:order.append('rebase')
            if event=='call' and frame.f_code is PhysicalSplitTrials.trial.__code__:order.append('trial')
        with monkeypatch.context() as patch:
            patch.setattr(owner,name,delegate)
            if not enabled:patch.setattr(PhysicalSplitTrials,'available',lambda self:False)
            with material_read_scope(binding,options):
                sys.setprofile(profile)
                try:
                    if failure and target in ('validate_mesh','surface_alias'):
                        with pytest.raises(RuntimeError) as caught:
                            refine_physical_candidate(candidate,Packet(points,cells,segments),settings,
                                dict(topology_operations=0,insertions=0,shared_segment_splits=0,shared_nodes=[]),
                                chart.evaluate,{},ComponentSeedRegistry(20),allow_physical_flips=True)
                        assert caught.value is error;outcomes.append(None)
                    else:
                        outcomes.append(refine_physical_candidate(candidate,Packet(points,cells,segments),settings,
                            dict(topology_operations=0,insertions=0,shared_segment_splits=0,shared_nodes=[]),
                            chart.evaluate,{},ComponentSeedRegistry(20),allow_physical_flips=True))
                finally:sys.setprofile(None)
        assert 'rebase' not in order and 'trial' not in order
        calls.append(order)
    assert calls[0]==calls[1]
    if outcomes[0] is not None:
        assert outcomes[0][0].report==outcomes[1][0].report and outcomes[0][2]==outcomes[1][2]


@pytest.mark.parametrize('mutation',('code','defaults'))
def test_full_quality_service_code_and_effective_defaults_disable_cache(monkeypatch,mutation):
    from anymesher import quality_v2
    trials=PhysicalSplitTrials(SurfaceMeshOptions())
    assert trials.available()
    if mutation=='defaults':
        changed=dict(quality_v2.mesh_quality.__kwdefaults__,declared_plate_junction_edges=((1,2),))
        monkeypatch.setattr(quality_v2.mesh_quality,'__kwdefaults__',changed)
    else:
        def unavailable(*args,**kwargs):
            raise AssertionError('changed service must not be invoked by eligibility')
        monkeypatch.setattr(quality_v2.mesh_quality,'__code__',unavailable.__code__)
    assert not trials.available()
