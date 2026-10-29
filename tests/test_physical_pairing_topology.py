"""Physical fixed-corner fan proposals retain UV/source row identity."""
import numpy as np
import pytest
from anymesher.surface_mesh import SurfaceMeshOptions
from anymesher._physical_pairing_topology import repair_two_triangle_fans


def packet():
    # A sharp protected corner split into two inadmissible triangles. Flipping
    # the unprotected interior spoke restores its single, admissible cap.
    uv=np.array([[0.,0.],[1.,0.],[1.8,.6],[.8,.6]])
    xyz=np.column_stack((uv,np.zeros(4)))
    rows=np.array([[0,1,2],[0,2,3]])
    protected=np.array([[0,1],[0,3],[1,2],[2,3]])
    return uv,rows,xyz,protected


def test_cap_changes_connectivity_only_and_full_world_quality_improves():
    uv,rows,xyz,segments=packet();before=(uv.copy(),rows.copy(),xyz.copy())
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),
        {0,1,3},segments,topology_remaining=2)
    assert 1<=work<=2 and facts['commits']==1
    assert np.count_nonzero(np.any(made==0,axis=1))==1
    assert not facts['final_quality']['invalid_element_count']
    assert facts['final_quality']['quality_violation_count']==0
    for actual,original in zip((uv,rows,xyz),before):assert np.array_equal(actual,original)


@pytest.mark.parametrize('remaining',[0,1])
def test_budget_is_charged_and_bounded(remaining):
    uv,rows,xyz,segments=packet()
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),
        {0,1,3},segments,topology_remaining=remaining)
    assert work<=remaining
    if remaining==0:assert np.array_equal(made,rows) and facts['commits']==0


def test_protected_diagonal_cannot_be_replaced():
    uv,rows,xyz,segments=packet()
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),
        {0,1,3},np.vstack((segments,[0,2])),topology_remaining=2)
    assert work==0 and facts['commits']==0 and np.array_equal(made,rows)


def test_unprotected_spokes_do_not_define_a_boundary_corner():
    uv,rows,xyz,segments=packet()
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),
        {0},segments[1:],topology_remaining=2)
    assert work==0 and np.array_equal(made,rows)


def test_nonconvex_uv_cannot_be_repaired_by_world_shape():
    uv,rows,xyz,segments=packet();uv[2]=[.2,.1]
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),
        {0},segments,topology_remaining=1)
    assert work==1 and facts['commits']==0 and np.array_equal(made,rows)

def mixed_packet(*,owners=False,inactive=False):
    from anymesher.core import MeshCore
    from anymesher.recombine import RecombinationReport
    uv,rows,xyz,segments=packet()
    kw=dict(owner_table=('first','second'),triangle_owner_handles=np.array([0,1])) if owners else {}
    if inactive:kw['triangle_active']=np.zeros(2,dtype=bool)
    core=MeshCore(xyz,rows,**kw)
    report=RecombinationReport(core,np.empty((0,2),dtype=np.int64),np.empty(0,dtype=np.int64),np.empty(0),0,0)
    return uv,report,segments


def test_active_mixed_flip_preserves_all_ids_activity_xyz_and_source_arrays():
    from anymesher._physical_pairing_topology import repair_active_triangle_flips
    uv,report,segments=mixed_packet();old=report.mesh
    updated,work,facts=repair_active_triangle_flips(uv,report,SurfaceMeshOptions(),segments,topology_remaining=4)
    assert facts['commits']==1 and work<=4
    assert facts['final_quality']['quality_violation_count']==0
    for name in ('node_ids','triangle_ids','quad_ids','triangle_active','quad_active',
                 'node_coordinates','node_owner_handles','triangle_owner_handles'):
        np.testing.assert_array_equal(getattr(updated.mesh,name),getattr(old,name))
    np.testing.assert_array_equal(report.mesh.triangle_connectivity,packet()[1])


@pytest.mark.parametrize('mode',['owners','inactive','protected','zero','negative_uv'])
def test_mixed_refusal_never_changes_operands(mode):
    from anymesher._physical_pairing_topology import repair_active_triangle_flips
    uv,report,segments=mixed_packet(owners=mode=='owners',inactive=mode=='inactive')
    if mode=='protected':segments=np.vstack((segments,[0,2]))
    if mode=='negative_uv':uv=uv.copy();uv[[1,3]]=uv[[3,1]]
    old=report.mesh.triangle_connectivity.copy()
    updated,work,facts=repair_active_triangle_flips(uv,report,SurfaceMeshOptions(),segments,
        topology_remaining=0 if mode=='zero' else 4)
    assert facts['commits']==0
    np.testing.assert_array_equal(updated.mesh.triangle_connectivity,old)
    np.testing.assert_array_equal(report.mesh.triangle_connectivity,old)


@pytest.mark.parametrize('mode',['callback','foreign_quality'])
def test_real_owner_preview_preserves_unknown_or_callback_route(chart_fixture,monkeypatch,mode):
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface
    from anymesher import _physical_pairing_topology as topology
    from test_physical_corner_cavity import operands
    _model,chart=chart_fixture
    uv,report,settings,edges=operands(chart)
    marker=RuntimeError('original downstream operation')
    def forbidden(*args,**kwargs):raise AssertionError('unqualified topology dispatch')
    def downstream(*args,**kwargs):raise marker
    monkeypatch.setattr(topology,'repair_two_triangle_fans',forbidden)
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    if mode=='foreign_quality':
        original=surface.evaluate_quality
        monkeypatch.setattr(surface,'evaluate_quality',lambda *a,**kw:original(*a,**kw))
    check=(lambda phase:None) if mode=='callback' else None
    with pytest.raises(RuntimeError) as caught:
        surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
            SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,check,topology_remaining=4)
    assert caught.value is marker


from test_physical_mixed_repair_batch import chart_fixture

@pytest.mark.parametrize('scale',[-1.,1.e-16])
def test_world_validity_cannot_override_invalid_or_degenerate_uv(scale):
    uv,rows,xyz,segments=packet();uv=uv.copy();uv[:,0]*=scale
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),
        {0,1,3},segments,topology_remaining=4)
    assert work==0 and facts['commits']==0
    np.testing.assert_array_equal(made,rows)


@pytest.mark.parametrize('service',['repair_two_triangle_fans','repair_active_triangle_flips','_core','_score'])
def test_unknown_pairing_service_never_enters_detached_owner_window(chart_fixture,monkeypatch,service):
    import anygeometry as owner
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface
    from anymesher import _physical_pairing_topology as topology
    from anymesher import _physical_corner_cavity as cavity
    from test_physical_corner_cavity import operands
    model,chart=chart_fixture;source=owner.to_dict(model)
    uv,report,settings,edges=operands(chart);entered=[]
    target=topology if service.startswith('repair_') else cavity
    original=getattr(target,service)
    def wrapper(*args,**kwargs):
        entered.append(True)
        raise AssertionError('unknown external observer was entered')
    marker=RuntimeError('original recombination exception')
    def downstream(*args,**kwargs):raise marker
    monkeypatch.setattr(target,service,wrapper)
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    with pytest.raises(RuntimeError) as caught:
        surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
            SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,None,topology_remaining=4)
    assert caught.value is marker and not entered
    assert owner.to_dict(model)==source


def test_changed_helper_defaults_disable_new_dispatch(chart_fixture,monkeypatch):
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface
    from anymesher import _physical_pairing_topology as topology
    from test_physical_corner_cavity import operands
    _model,chart=chart_fixture;uv,report,settings,edges=operands(chart)
    original=topology.repair_two_triangle_fans
    monkeypatch.setattr(original,'__kwdefaults__',{'topology_remaining':1})
    marker=RuntimeError('strict original downstream')
    def downstream(*args,**kwargs):raise marker
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    with pytest.raises(RuntimeError) as caught:
        surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
            SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,None,topology_remaining=4)
    assert caught.value is marker


def test_changed_service_before_second_dispatch_refuses_without_wrapper_call(chart_fixture,monkeypatch):
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface
    from anymesher import _physical_pairing_topology as topology
    from anymesher.errors import MeshError
    from test_physical_corner_cavity import operands
    _model,chart=chart_fixture;uv,report,settings,edges=operands(chart);entered=[]
    def wrapper(*args,**kwargs):entered.append(True);raise AssertionError('foreign helper')
    def downstream(*args,**kwargs):
        monkeypatch.setattr(topology,'repair_active_triangle_flips',wrapper)
        return report,{}
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    with pytest.raises(MeshError,match='calculation changed before mixed recovery'):
        surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
            SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,None,topology_remaining=4)
    assert not entered


@pytest.mark.parametrize('service',['evaluate_quality','validate_mesh'])
def test_quality_rebinding_after_matching_refuses_before_new_helper_observer(chart_fixture,monkeypatch,service):
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface,quality_v2
    from anymesher.errors import MeshError
    from test_physical_corner_cavity import operands
    _model,chart=chart_fixture;uv,report,settings,edges=operands(chart);entered=[]
    def wrapper(*args,**kwargs):entered.append(True);raise AssertionError('opaque quality observer')
    def downstream(*args,**kwargs):
        monkeypatch.setattr(quality_v2,service,wrapper)
        return report,{}
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    with pytest.raises(MeshError,match='calculation changed before mixed recovery'):
        surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
            SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,None,topology_remaining=4)
    assert not entered


def test_remote_existing_opposite_diagonal_is_refused_by_full_manifold_check():
    uv,rows,xyz,segments=packet()
    uv=np.vstack((uv,[[2.,-1.],[2.,1.]]));xyz=np.column_stack((uv,np.zeros(6)))
    rows=np.vstack((rows,[[1,4,3],[3,1,5]]))
    made,work,facts=repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),{0},segments,topology_remaining=4)
    assert work==1 and facts['commits']==0
    np.testing.assert_array_equal(made,rows)


def test_active_quad_and_its_inactive_covered_triangles_are_untouched():
    from anymesher.core import MeshCore
    from anymesher.recombine import RecombinationReport
    from anymesher._physical_pairing_topology import repair_active_triangle_flips
    uv,rows,xyz,segments=packet()
    uv=np.vstack((uv,[[3.,0.],[4.,0.],[4.,1.],[3.,1.]]));xyz=np.column_stack((uv,np.zeros(8)))
    rows=np.vstack((rows,[[4,5,6],[4,6,7]]));quads=np.array([[4,5,6,7]])
    core=MeshCore(xyz,rows,quads,triangle_active=np.array([True,True,False,False]))
    report=RecombinationReport(core,np.array([[3,4]]),np.array([5]),np.array([1.]),0,0)
    made,work,facts=repair_active_triangle_flips(uv,report,SurfaceMeshOptions(),segments,topology_remaining=4)
    assert facts['commits']==1
    np.testing.assert_array_equal(made.mesh.triangle_connectivity[2:],rows[2:])
    np.testing.assert_array_equal(made.mesh.quad_connectivity,quads)
    np.testing.assert_array_equal(made.mesh.triangle_active,core.triangle_active)


def test_scoring_exception_after_staging_preserves_original_error_and_input_bytes(monkeypatch):
    from anymesher import quality_v2
    uv,rows,xyz,segments=packet();before=(uv.tobytes(),rows.tobytes(),xyz.tobytes())
    original=quality_v2.evaluate_quality;calls=[];marker=RuntimeError('quality service failure')
    def score(*args,**kwargs):
        calls.append(True)
        if len(calls)==2:raise marker
        return original(*args,**kwargs)
    monkeypatch.setattr(quality_v2,'evaluate_quality',score)
    with pytest.raises(RuntimeError) as caught:
        repair_two_triangle_fans(uv,rows,xyz,SurfaceMeshOptions(),{0},segments,topology_remaining=4)
    assert caught.value is marker and len(calls)==2
    assert before==(uv.tobytes(),rows.tobytes(),xyz.tobytes())


@pytest.mark.parametrize('builder_name',['_physical_pairing_services','_physical_full_quality_services'])
@pytest.mark.parametrize('mutation',['wrapper','code','defaults','kwdefaults'])
def test_builder_is_checked_before_any_observer_entry(chart_fixture,monkeypatch,builder_name,mutation):
    import sys
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface
    from anymesher._material_region_binding import material_read_scope
    from anymesher.refinement import SizeField
    from test_physical_corner_cavity import operands
    model,chart=chart_fixture;uv,report,settings,edges=operands(chart)
    original=getattr(surface,builder_name);entered=[];marker=RuntimeError('original strict downstream')
    def opaque():raise AssertionError('opaque builder executed')
    if mutation=='wrapper':
        def wrapper():entered.append(True);return original()
        monkeypatch.setattr(surface,builder_name,wrapper)
    elif mutation=='code':monkeypatch.setattr(original,'__code__',opaque.__code__)
    elif mutation=='defaults':monkeypatch.setattr(original,'__defaults__',(None,))
    else:monkeypatch.setattr(original,'__kwdefaults__',{'unused':None})
    def downstream(*args,**kwargs):raise marker
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    def observe(code,offset):entered.append(True)
    sys.monitoring.use_tool_id(3,'builder call observer')
    sys.monitoring.register_callback(3,sys.monitoring.events.PY_START,observe)
    sys.monitoring.set_local_events(3,original.__code__,sys.monitoring.events.PY_START)
    try:
        with pytest.raises(RuntimeError) as caught:
            with material_read_scope(chart.region_binding,dict(native_options=settings.native_options,
                    native_backend='python',size_field=SizeField(model,.5),cancellation_check=None)):
                surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
                    SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,None,topology_remaining=4)
        assert caught.value is marker
    finally:sys.monitoring.free_tool_id(3)
    assert not entered


@pytest.mark.parametrize('builder_name,ref_name,code_name',[
    ('_physical_pairing_services','_PHYSICAL_PAIRING_BUILDER_REF','_PHYSICAL_PAIRING_BUILDER_CODE'),
    ('_physical_full_quality_services','_PHYSICAL_FULL_BUILDER_REF','_PHYSICAL_FULL_BUILDER_CODE')])
@pytest.mark.parametrize('raises',[False,True])
def test_operation_cannot_renew_builder_declarations_or_mask_original_error(chart_fixture,monkeypatch,builder_name,ref_name,code_name,raises):
    from types import SimpleNamespace
    from anymesher import surface_mesh as surface
    from anymesher.errors import MeshError
    from anymesher._material_region_binding import material_read_scope
    from anymesher.refinement import SizeField
    from test_physical_corner_cavity import operands
    model,chart=chart_fixture;uv,report,settings,edges=operands(chart);entered=[]
    original=getattr(surface,builder_name);marker=RuntimeError('original operation error')
    def opaque():entered.append(True);return original()
    def downstream(*args,**kwargs):
        monkeypatch.setattr(surface,builder_name,opaque)
        monkeypatch.setattr(surface,ref_name,opaque)
        monkeypatch.setattr(surface,code_name,opaque.__code__)
        if raises:raise marker
        return report,{}
    monkeypatch.setattr(surface,'_qualified_recombination',downstream)
    with pytest.raises(RuntimeError if raises else MeshError) as caught:
        with material_read_scope(chart.region_binding,dict(native_options=settings.native_options,
                native_backend='python',size_field=SizeField(model,.5),cancellation_check=None)):
            surface._curvilinear_mixed_preview(SimpleNamespace(points=uv,triangles=report.mesh.triangle_connectivity),
                SimpleNamespace(segments=np.asarray(edges,dtype=np.int64)),settings,chart.evaluate,None,None,topology_remaining=4)
    if raises:assert caught.value is marker
    else:assert 'calculation changed before mixed recovery' in str(caught.value)
    assert not entered
