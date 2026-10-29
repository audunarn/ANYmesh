"""Actual source-bound mixed boundary-star phase, not a proof receipt fixture."""
from pathlib import Path
from dataclasses import replace
import json,sys
import numpy as np
import pytest
import anygeometry as owner
from anymesher.core import MeshCore
from anymesher.recombine import RecombinationReport
from anymesher.surface_mesh import SurfaceMeshOptions
from anymesher.errors import MeshError
from anymesher import _physical_shape_patches as shape,_physical_mixed_repair as mixed
from test_physical_mixed_sizing import owner_chart,scope


def packet(chart):
    folder=Path(__file__).parent/'fixtures'
    with np.load(folder/'shape_patch_boundary_core.npz',allow_pickle=False) as d:
        points=d['points'].copy();mesh=MeshCore(d['xyz'],d['triangles'],d['quads'],node_ids=d['node_ids'],triangle_ids=d['triangle_ids'],quad_ids=d['quad_ids'],node_active=d['node_active'],triangle_active=d['triangle_active'],quad_active=d['quad_active'],node_owner_handles=d['node_owner_handles'],triangle_owner_handles=d['triangle_owner_handles'],quad_owner_handles=d['quad_owner_handles'])
        meta=json.loads((folder/'shape_patch_boundary_metadata.json').read_text());meta['exchange_work']=4833
        report=RecombinationReport(mesh,d['paired_triangle_ids'],d['created_quad_ids'],d['scores'],**meta)
        fixed=tuple(sorted(set(map(int,d['segments'].ravel()))))
    return points,report,fixed


def snapshot(points,report):
    return points.tobytes(),tuple(getattr(report.mesh,n).tobytes() for n in ('node_coordinates','triangle_connectivity','quad_connectivity','node_ids','triangle_ids','quad_ids','triangle_active','quad_active')),report.paired_triangle_ids.tobytes(),report.scores.tobytes(),report.exchange_work


def test_actual_boundary_star_owned_phase_global_progress_and_work(owner_chart):
    model,chart=owner_chart;points,report,fixed=packet(chart);before=snapshot(points,report)
    settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work+128)
    with scope(model,chart,settings):
        assert chart.evaluate(points).tobytes()==report.mesh.node_coordinates.tobytes()
        made,result,facts=shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
    assert facts['enabled'] and facts['commits']>=1 and facts['work']==128
    assert facts['patches'][0]['local_truncated'] is False and facts['patches'][0]['global_exhausted']
    assert facts['final_quality']['elements_above_maximum_growth']<facts['initial_quality']['elements_above_maximum_growth']
    assert facts['final_quality']['elements_below_minimum_scaled_jacobian']<facts['initial_quality']['elements_below_minimum_scaled_jacobian']
    assert shape._alternative_progress(facts['initial_quality'],facts['final_quality'])
    assert all(facts['final_quality']['violation_counts'][key]<=value for key,value in facts['initial_quality']['violation_counts'].items())
    assert result.exchange_work==report.exchange_work+128 and snapshot(points,report)==before
    assert np.array_equal(made[list(fixed)],points[list(fixed)])
    for name in ('triangle_connectivity','quad_connectivity','node_ids','triangle_ids','quad_ids','triangle_active','quad_active','node_owner_handles','triangle_owner_handles','quad_owner_handles'):
        np.testing.assert_array_equal(getattr(result.mesh,name),getattr(report.mesh,name))
    np.testing.assert_array_equal(result.paired_triangle_ids,report.paired_triangle_ids)


def test_owned_original_star_operands_are_immutable(owner_chart):
    _,chart=owner_chart;points,report,fixed=packet(chart);guard=shape._OriginalStars(points,report.mesh,(539,583))
    with pytest.raises(AttributeError):guard._data=()
    data=object.__getattribute__(guard,'_data')
    with pytest.raises(ValueError):np.frombuffer(data[1],dtype=float).setflags(write=True)
    assert guard(points)


@pytest.mark.parametrize('mutation',['service','guard','data','descriptor'])
def test_owned_core_change_refuses_before_unknown_numeric_observer(owner_chart,monkeypatch,mutation):
    from anymesher import quality_v2
    model,chart=owner_chart;points,report,fixed=packet(chart);before=snapshot(points,report);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work+1);entered=[]
    def opaque(*args,**kwargs):entered.append(True);raise AssertionError('unknown observer entered')
    code=mixed.repair_mixed_core_coordinates.__code__
    def mutate(current,offset):
        frame=sys._getframe(1)
        if mutation=='service':monkeypatch.setattr(quality_v2,'evaluate_quality',opaque)
        elif mutation=='guard':monkeypatch.setattr(shape._OriginalStars,'__call__',opaque)
        elif mutation=='descriptor':monkeypatch.setattr(shape._OriginalStars,'_data',property(opaque))
        else:object.__setattr__(frame.f_locals['_trial_guard'],'_data',())
    sys.monitoring.use_tool_id(3,'owned core mutation');sys.monitoring.register_callback(3,sys.monitoring.events.PY_START,mutate);sys.monitoring.set_local_events(3,code,sys.monitoring.events.PY_START)
    try:
        with pytest.raises(MeshError):
            with scope(model,chart,settings):shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
    finally:sys.monitoring.free_tool_id(3)
    assert not entered and snapshot(points,report)==before


def test_no_budget_or_unknown_callback_preserves_input(owner_chart):
    model,chart=owner_chart;points,report,fixed=packet(chart);before=snapshot(points,report);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work)
    with scope(model,chart,settings):
        made,result,facts=shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
    assert facts['work']==0 and result is report and np.array_equal(made,points)
    made,result,facts=shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed,cancellation_check=lambda phase:None)
    assert not facts['enabled'] and result is report and snapshot(points,report)==before


def test_phase_service_inventory_is_known_without_unknown_callbacks():
    from anymesher import surface_mesh
    missing=[];code=surface_mesh._physical_pairing_services.__code__
    def observed(current,offset,value):
        frame=sys._getframe(1)
        if 'functions' in frame.f_locals and 'pins' in frame.f_locals:
            missing.extend(getattr(function,'__qualname__',str(type(function))) for function,pin in zip(frame.f_locals['functions'],frame.f_locals['pins']) if pin is None)
    sys.monitoring.use_tool_id(3,'finite inventory diagnostics');sys.monitoring.register_callback(3,sys.monitoring.events.PY_RETURN,observed);sys.monitoring.set_local_events(3,code,sys.monitoring.events.PY_RETURN)
    try:context=surface_mesh._physical_pairing_services()
    finally:sys.monitoring.free_tool_id(3)
    assert context is not None,missing


def test_actual_neighbor_worse_counterexample_is_not_outer_progress():
    data=json.loads((Path(__file__).parent/'fixtures/shape_patch_neighbor_worse.json').read_text())
    assert data['after']['elements_below_minimum_angle']<data['before']['elements_below_minimum_angle']
    assert data['after']['elements_above_maximum_angle']>data['before']['elements_above_maximum_angle']
    assert not shape._admissible(data['before'],data['after'])


def test_exhausted_partial_trial_remains_charged_without_input_mutation(owner_chart):
    model,chart=owner_chart;points,report,fixed=packet(chart);before=snapshot(points,report);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work+1)
    with scope(model,chart,settings):made,result,facts=shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
    assert facts['work']==1 and result.exchange_work==report.exchange_work+1
    assert snapshot(points,report)==before
    assert facts['commits']==0
    np.testing.assert_array_equal(made,points)
    assert result.mesh is report.mesh


@pytest.mark.parametrize('mutation',['source','construction','original_exception'])
def test_owner_mutation_or_original_failure_cannot_publish(owner_chart,monkeypatch,mutation):
    model,chart=owner_chart;points,report,fixed=packet(chart);before=snapshot(points,report);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work+1);marker=RuntimeError('original failure');region=chart.region_binding.region;area=region.material_area;units=model._units
    def changed(code,offset,value):
        if mutation=='source':model._units='mm'
        elif mutation=='construction':object.__setattr__(region,'material_area',area*2.)
        else:raise marker
    sys.monitoring.use_tool_id(3,'owner final mutation');sys.monitoring.register_callback(3,sys.monitoring.events.PY_RETURN,changed);sys.monitoring.set_local_events(3,mixed.repair_mixed_core_coordinates.__code__,sys.monitoring.events.PY_RETURN)
    try:
        with pytest.raises(RuntimeError if mutation=='original_exception' else (owner.GeometryError,MeshError)) as caught:
            with scope(model,chart,settings):shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
        if mutation=='original_exception':assert caught.value is marker
    finally:
        sys.monitoring.free_tool_id(3);model._units=units;object.__setattr__(region,'material_area',area)
    assert snapshot(points,report)==before


def test_hostile_defaults_never_execute_repr_or_equality(monkeypatch):
    from anymesher import surface_mesh
    entered=[]
    class Hostile:
        def __repr__(self):entered.append('repr');raise AssertionError('repr executed')
        def __eq__(self,other):entered.append('eq');raise AssertionError('equality executed')
    monkeypatch.setattr(mixed._growth_descent,'__kwdefaults__',{'unused':Hostile()})
    assert surface_mesh._physical_pairing_services() is None
    assert mixed._constraint_operation_pin(mixed._growth_descent) is None
    assert entered==[]


def test_unknown_guard_kind_cannot_run_class_lookup_descriptor(monkeypatch):
    from anymesher import surface_mesh
    entered=[]
    class Unknown:
        @property
        def __dict__(self):entered.append(True);raise AssertionError('unknown class lookup entered')
    monkeypatch.setattr(shape,'_OriginalStars',Unknown())
    assert surface_mesh._physical_pairing_services() is None
    assert not entered


@pytest.mark.parametrize('mutation',['payload','tuple_subclass','shape_leaf','raw_subclass','frozen_iterable','frozen_leaf','cell_iterable','cell_leaf','cell_bound','raw_mismatch','fixed_mismatch','cells_mismatch'])
def test_forged_owned_data_refuses_before_any_user_operand_callback(owner_chart,mutation):
    model,chart=owner_chart;points,report,_=packet(chart);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work);entered=[]
    class Hostile:
        def __iter__(self):entered.append('iter');raise AssertionError('iter executed')
        def __len__(self):entered.append('len');raise AssertionError('len executed')
        def __eq__(self,other):entered.append('eq');raise AssertionError('eq executed')
        def __repr__(self):entered.append('repr');raise AssertionError('repr executed')
        def __int__(self):entered.append('int');raise AssertionError('int executed')
    class TupleSubclass(tuple):
        def __iter__(self):entered.append('subclass iter');raise AssertionError('tuple subclass iter')
    class BytesSubclass(bytes):
        def __len__(self):entered.append('subclass len');raise AssertionError('bytes subclass len')
    genuine=shape._OriginalStars(points,report.mesh,(539,583));data=object.__getattribute__(genuine,'_data');dimensions,raw,frozen,cells=data
    if mutation=='payload':data=Hostile()
    elif mutation=='tuple_subclass':data=TupleSubclass(data)
    elif mutation=='shape_leaf':data=((Hostile(),2),raw,frozen,cells)
    elif mutation=='raw_subclass':data=(dimensions,BytesSubclass(raw),frozen,cells)
    elif mutation=='frozen_iterable':data=(dimensions,raw,Hostile(),cells)
    elif mutation=='frozen_leaf':data=(dimensions,raw,frozen+(Hostile(),),cells)
    elif mutation=='cell_iterable':data=(dimensions,raw,frozen,(Hostile(),))
    elif mutation=='cell_leaf':data=(dimensions,raw,frozen,((Hostile(),0,1),))
    elif mutation=='cell_bound':data=(dimensions,raw,frozen,((len(points),0,1),))
    elif mutation=='raw_mismatch':data=(dimensions,bytes([raw[0]^1])+raw[1:],frozen,cells)
    elif mutation=='fixed_mismatch':data=(dimensions,raw,frozen[:-1],cells)
    else:data=(dimensions,raw,frozen,cells[:-1])
    guard=object.__new__(shape._OriginalStars);object.__setattr__(guard,'_data',data)
    with pytest.raises(MeshError):
        with scope(model,chart,settings):
            context=shape._capture_context();assert context is not None
            mixed.repair_mixed_core_coordinates(points,report.mesh.triangle_connectivity,report.mesh,settings,chart.evaluate,frozen,exchange_work=report.exchange_work,_trial_guard=guard,_owned_context=context+(data,))
    assert entered==[]


@pytest.mark.parametrize('index',[2,5])
def test_forged_service_signature_never_calls_rich_equality(owner_chart,index):
    model,chart=owner_chart;points,report,_=packet(chart);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work);entered=[]
    class Hostile:
        def __eq__(self,other):entered.append(True);raise AssertionError('signature equality entered')
        def __repr__(self):entered.append(True);raise AssertionError('signature repr entered')
    guard=shape._OriginalStars(points,report.mesh,(539,583));data=object.__getattribute__(guard,'_data')
    with pytest.raises(MeshError):
        with scope(model,chart,settings):
            context=list(shape._capture_context());context[index]=Hostile()
            mixed.repair_mixed_core_coordinates(points,report.mesh.triangle_connectivity,report.mesh,settings,chart.evaluate,data[2],exchange_work=report.exchange_work,_trial_guard=guard,_owned_context=tuple(context)+(data,))
    assert entered==[]


def test_direct_guard_rejects_unissued_iterable_before_unpack(owner_chart):
    _,chart=owner_chart;points,report,_=packet(chart);entered=[]
    class Data:
        def __iter__(self):entered.append(True);raise AssertionError('unissued data unpacked')
    guard=object.__new__(shape._OriginalStars);object.__setattr__(guard,'_data',Data())
    with pytest.raises(MeshError):shape._STAR_CALL(guard,points)
    assert not entered


@pytest.mark.parametrize('field',['triangle_connectivity','triangle_active'])
def test_original_mesh_correspondence_never_casts_caller_object_leaf(owner_chart,field):
    import copy
    model,chart=owner_chart;points,report,_=packet(chart);settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work);entered=[]
    class Hostile:
        def __int__(self):entered.append('int');raise AssertionError('mesh leaf cast')
        def __iter__(self):entered.append('iter');raise AssertionError('mesh flag iteration')
        def __eq__(self,other):entered.append('eq');raise AssertionError('mesh leaf equality')
    fake=copy.copy(report.mesh)
    object.__setattr__(fake,field,np.asarray([[Hostile(),0,1]],dtype=object) if field=='triangle_connectivity' else Hostile())
    guard=shape._OriginalStars(points,report.mesh,(539,583));data=object.__getattribute__(guard,'_data')
    with pytest.raises(MeshError):
        with scope(model,chart,settings):
            context=shape._capture_context()
            mixed.repair_mixed_core_coordinates(points,report.mesh.triangle_connectivity,fake,settings,chart.evaluate,data[2],exchange_work=report.exchange_work,_trial_guard=guard,_owned_context=context+(data,))
    assert entered==[]


def test_rejected_real_closed_patch_does_not_starve_later_admissible_patch(owner_chart):
    model,chart=owner_chart
    origin=np.asarray([.5,.5])@chart.transform;j=chart.jacobians(origin[None,:])[0]
    ex=j[:,0]/np.linalg.norm(j[:,0]);ey=j[:,1]-ex*np.dot(ex,j[:,1]);ey/=np.linalg.norm(ey)
    basis=np.linalg.lstsq(j,np.column_stack((ex,ey)),rcond=None)[0]
    a=np.array([0.,0.]);b=np.array([1.,0.]);c=np.array([np.cos(np.radians(10.)),np.sin(np.radians(10.))])
    first=np.asarray([a,b,c,(a+b+c)/3.])*.001
    second=(np.asarray([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[.18,.18]])+np.array([2.,0.]))*.001
    # A fixed shape-admitted Q/Q competitor keeps the actual worst growth
    # unchanged: continuous growth merit alone cannot admit the first patch.
    remote=np.asarray([[0.,0.],[1.,0.],[1.,1.],[0.,1.],[101.,0.],[101.,1.]])*.00001+np.array([.004,.004])
    points=origin+np.vstack((first,second,remote))@basis.T
    triangles=np.asarray([[0,1,3],[1,2,3],[2,0,3],[4,5,8],[5,6,8],[6,7,8],[7,4,8],[9,10,11],[9,11,12],[10,13,14],[10,14,11]],dtype=np.int64)
    quads=np.asarray([[9,10,11,12],[10,13,14,11]],dtype=np.int64)
    mesh=MeshCore(chart.evaluate(points),triangles,quads,triangle_active=np.asarray([True]*7+[False]*4))
    report=RecombinationReport(mesh,np.asarray([[8,9],[10,11]],dtype=np.int64),mesh.quad_ids,np.ones(2),0,0)
    fixed=sorted(set(range(len(points)))-{3,8});settings=SurfaceMeshOptions(min_angle=15.,max_angle=179.99,min_scaled_jacobian=0.,max_aspect_ratio=1.e6,max_element_growth=1.e6,max_recombination_work=512)
    before=snapshot(points,report)
    with scope(model,chart,settings):made,result,facts=shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
    assert facts['attempts']>=2 and facts['rejections']>=1 and facts['commits']>=1
    assert facts['patches'][0]['accepted'] is False
    assert any(item['accepted'] for item in facts['patches'][1:])
    assert facts['final_quality']['elements_below_minimum_angle']<facts['initial_quality']['elements_below_minimum_angle']
    assert result.exchange_work==sum(item['work'] for item in facts['patches'])>sum(item['work'] for item in facts['patches'] if item['accepted'])
    assert 0<result.exchange_work<=512 and snapshot(points,report)==before
    assert made[:4].tobytes()==points[:4].tobytes() # rejected component rolls back unconditionally
    assert made[fixed].tobytes()==points[fixed].tobytes() and not np.array_equal(made[8],points[8])


def test_local128_truncation_is_not_global_exhaustion_and_all_work_survives(owner_chart):
    model,chart=owner_chart;points,report,fixed=packet(chart);before=snapshot(points,report)
    settings=SurfaceMeshOptions(min_angle=15.,max_recombination_work=report.exchange_work+129)
    with scope(model,chart,settings):made,result,facts=shape.repair_shape_patches(points,report,settings,chart.evaluate,fixed)
    first=facts['patches'][0]
    assert first['work']==128 and first['local_truncated'] and not first['global_exhausted']
    assert facts['work']==129 and result.exchange_work==report.exchange_work+129
    assert snapshot(points,report)==before and made[list(fixed)].tobytes()==points[list(fixed)].tobytes()
