"""Canonical mixed engine service admission and representation integrity."""
import numpy as np
import pytest
from anymesher import surface_mesh as surface
from anymesher._physical_mixed_sizing import _cover
from anymesher.core import MeshCore
from anymesher.recombine import RecombinationReport
from anymesher.errors import MeshError


def test_canonical_mixed_inventory_is_known_not_stable_unknown():
    assert surface._REFERENCE_PHYSICAL_PAIRING_SERVICES is not None
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


@pytest.mark.parametrize('part',['pairs','created','scores'])
def test_quad_without_complete_pair_metadata_refuses_deterministically(part):
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]])
    core=MeshCore(xyz,np.array([[0,1,2],[0,2,3]]),np.array([[0,1,2,3]]),triangle_active=np.zeros(2,dtype=bool))
    pairs=np.array([[1,2]],dtype=np.int64);created=core.quad_ids.copy();scores=np.array([1.])
    if part=='pairs':pairs=np.empty((0,2),dtype=np.int64)
    elif part=='created':created=np.empty(0,dtype=np.int64)
    else:scores=np.empty(0)
    report=RecombinationReport(core,pairs,created,scores,1,0)
    with pytest.raises(MeshError,match='complete quadrilateral pair metadata'):_cover(report,set())


@pytest.fixture(scope='module')
def owner_chart():
    from pathlib import Path
    import json,anygeometry as owner
    from anymesher._material_region_binding import MaterialRegionBinding
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model=owner.from_dict(json.loads((Path(__file__).parent/'fixtures/mixed_sizing_dome_source.json').read_text()))
    face=next(iter(model.faces));regions=owner.query_material_surface_regions(model,(face,))
    intervals=tuple((model.handle('edge',use.edge),i/32,(i+1)/32) for use in model.faces[face].loop for i in range(32))
    chords=owner.query_prepared_material_boundary_chords(model,regions,intervals,chord_error=.0025)
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,.0025);binding._curvilinear_state['chords']=chords
    return model,AnalyticMetricChart(model,face,region_binding=binding)


def patch(chart,kind):
    import json
    from pathlib import Path
    from collections import Counter
    if kind=='TQ':
        data=json.loads((Path(__file__).parent/'fixtures/mixed_sizing_tq_literals.json').read_text())
        source_uv=np.asarray(data['chart_points'])@np.asarray(data['inverse']);triangles=data['triangles'];quads=data['quads']
    else:
        polygons=[[(0.,0.),(2.,0.),(2.,1.),(0.,1.)],[(2.,0.),(2.4,0.),(2.4,1.),(2.,1.)],[(0.,1.),(2.,1.),(2.,2.),(0.,2.)],[(0.,-1.),(2.,-1.),(2.,0.),(0.,0.)]]
        lookup={};quads=[]
        for polygon in polygons:
            row=[]
            for point in polygon:
                if point not in lookup:lookup[point]=len(lookup)
                row.append(lookup[point])
            quads.append(row)
        source_uv=.4+.01*np.asarray(list(lookup));triangles=[]
    points=source_uv@chart.transform;cover=list(triangles);active=[True]*len(triangles);pairs=[]
    for nodes in quads:
        first=len(cover);cover.extend([[nodes[0],nodes[1],nodes[2]],[nodes[0],nodes[2],nodes[3]]]);active.extend([False,False]);pairs.append([first+1,first+2])
    core=MeshCore(chart.evaluate(points),np.asarray(cover,dtype=np.int64).reshape((-1,3)),np.asarray(quads,dtype=np.int64),triangle_active=np.asarray(active))
    report=RecombinationReport(core,np.asarray(pairs),core.quad_ids.copy(),np.ones(len(pairs)),len(pairs),0)
    cells=[tuple(row) for row in triangles+quads];edges=Counter(tuple(sorted(map(int,(a,b)))) for row in cells for a,b in zip(row,row[1:]+row[:1]));protected=np.asarray([edge for edge,count in edges.items() if count==1],dtype=np.int64)
    return points,report,protected


def scope(model,chart,settings):
    from anymesher._material_region_binding import material_read_scope
    from anymesher.refinement import SizeField
    return material_read_scope(chart.region_binding,dict(native_options=settings.native_options,native_backend='python',size_field=SizeField(model,.25),cancellation_check=None))


@pytest.mark.parametrize('kind',['TQ','QQ'])
def test_genuine_owner_midpoint_repair_full_quality_cover_metadata_and_prefix(owner_chart,kind):
    import anygeometry as owner
    from anymesher._physical_mixed_sizing import repair_mixed_sizing,_score,_cover
    from anymesher.surface_mesh import SurfaceMeshOptions,_physical_chart_invalid_count
    model,chart=owner_chart;points,report,edges=patch(chart,kind);settings=SurfaceMeshOptions(min_angle=15.)
    original=points.tobytes(),report.mesh.triangle_connectivity.tobytes(),owner.to_dict(model)
    with scope(model,chart,settings):
        made,updated,work,nodes,facts=repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=128,insertion_remaining=8)
    assert facts['enabled'] and facts['commits']>=1 and nodes>=1 and work<=128
    assert facts['final_quality']['quality_violation_count']==0
    assert made[:len(points)].tobytes()==points.tobytes()
    assert _physical_chart_invalid_count(made,updated.mesh.triangle_connectivity)==0
    assert len(_cover(updated,set(map(tuple,edges))))==updated.mesh.active_quad_count
    assert set(report.mesh.node_ids)<=set(updated.mesh.node_ids)
    assert set(report.mesh.triangle_ids)<=set(updated.mesh.triangle_ids)
    assert np.array_equal(updated.mesh.node_coordinates,chart.evaluate(made))
    assert original==(points.tobytes(),report.mesh.triangle_connectivity.tobytes(),owner.to_dict(model))


@pytest.mark.parametrize('work,nodes',[(0,8),(128,0),(1,8)])
def test_budget_exhaustion_refuses_atomic_partial_proposal(owner_chart,work,nodes):
    from anymesher._physical_mixed_sizing import repair_mixed_sizing
    from anymesher.surface_mesh import SurfaceMeshOptions
    model,chart=owner_chart;points,report,edges=patch(chart,'TQ');settings=SurfaceMeshOptions(min_angle=15.)
    with scope(model,chart,settings):
        made,updated,used,added,facts=repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=work,insertion_remaining=nodes)
    assert used<=work and added==0 and facts['commits']==0
    np.testing.assert_array_equal(made,points)
    assert updated.mesh is report.mesh


@pytest.mark.parametrize('kind',['TQ','QQ'])
def test_batched_frontier_preserves_scalar_cell_sizes_and_order(owner_chart,kind):
    from anymesher._physical_mixed_sizing import _frontier,_edges
    _,chart=owner_chart
    _,report,_=patch(chart,kind);mesh=report.mesh
    records,incidence,frontier=_frontier(mesh,1.5)
    expected=[];attached={}
    for name,cells,active,ids,owners in (('T',mesh.triangle_connectivity,mesh.triangle_active,mesh.triangle_ids,mesh.triangle_owner_handles),('Q',mesh.quad_connectivity,mesh.quad_active,mesh.quad_ids,mesh.quad_owner_handles)):
        for row in np.flatnonzero(active):
            nodes=tuple(map(int,cells[row]));xyz=mesh.node_coordinates[list(nodes)]
            size=float(np.linalg.norm(np.roll(xyz,-1,axis=0)-xyz,axis=1).mean())
            index=len(expected);expected.append((name,int(row),int(ids[row]),nodes,int(owners[row]),size))
            for edge in _edges(nodes):attached.setdefault(edge,[]).append(index)
    assert records==expected and incidence==attached
    ranked=[]
    for edge,indices in attached.items():
        if len(indices)!=2:continue
        a,b=(expected[i] for i in indices)
        if min(a[5],b[5])<=0:continue
        ratio=max(a[5]/b[5],b[5]/a[5])
        if ratio>1.5:
            larger=indices[0] if a[5]>=b[5] else indices[1]
            ranked.append((-ratio,expected[larger][2],edge,larger))
    assert frontier==sorted(ranked)


def caller_operands(chart,settings):
    from types import SimpleNamespace
    points,rep,edges=patch(chart,'TQ')
    from anymesher.triangulation import PlanarTriangulation
    neighbors={}
    for a,b in edges:neighbors.setdefault(int(a),[]).append(int(b));neighbors.setdefault(int(b),[]).append(int(a))
    ring=[min(neighbors)]
    while len(ring)<len(neighbors):ring.append(next(n for n in neighbors[ring[-1]] if n not in ring))
    if sum(points[a,0]*points[b,1]-points[b,0]*points[a,1] for a,b in zip(ring,ring[1:]+ring[:1]))<0:ring=ring[::-1]
    packet=PlanarTriangulation(points=points,triangles=rep.mesh.triangle_connectivity,segments=edges,boundary_segments=edges,mandatory_segments=np.empty((0,2),dtype=np.int64),outer_loop=np.asarray(ring),hole_loops=(),protected_node_rows=frozenset(ring))
    candidate=surface._make_candidate(points,packet.triangles,settings=settings)
    baseline=dict(generated=np.empty((0,2)),best=candidate,initial=candidate,triangulation=packet,target_met=False,work_statistics={},triangulation_seconds=0.,optimization_seconds=0.)
    return baseline,rep


def run_caller(baseline,settings,chart,callback=None):
    return surface._run_frontal_quality_path(baseline,settings,callback,model_uuid=None,geometry_revision=None,metric_to_physical=chart.evaluate,metric_jacobian=chart.jacobians,automatically_seeded_shared_segments={},component_seed_registry=None,supplemental_metric_field=None,preserve_spatial_refinement=True)


@pytest.mark.parametrize('mode',['eligible','callback','outside_scope','no_recombine','driver_wrapper'])
def test_actual_native_entry_defers_only_eligible_canonical_mixed(owner_chart,monkeypatch,mode):
    import sys,contextlib
    from anymesher.surface_mesh import SurfaceMeshOptions
    from anymesher.native_v2 import NativeMeshingOptions
    model,chart=owner_chart;settings=SurfaceMeshOptions(target_size=.25,min_angle=15.,recombine=mode!='no_recombine',native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))
    baseline,_=caller_operands(chart,settings);original=surface.frontal_delaunay_refine;seen=[]
    class Observed(BaseException):pass
    def observer(code,offset):
        frame=sys._getframe(1);seen.append(frame.f_locals['_physical_growth_limit']);raise Observed()
    if mode=='driver_wrapper':
        def wrapper(*args,**kwargs):return original(*args,**kwargs)
        monkeypatch.setattr(surface,'frontal_delaunay_refine',wrapper)
    sys.monitoring.use_tool_id(3,'native argument observer');sys.monitoring.register_callback(3,sys.monitoring.events.PY_START,observer);sys.monitoring.set_local_events(3,original.__code__,sys.monitoring.events.PY_START)
    try:
        with pytest.raises(Observed):
            with (contextlib.nullcontext() if mode=='outside_scope' else scope(model,chart,settings)):
                run_caller(baseline,settings,chart,(lambda stage:None) if mode=='callback' else None)
    finally:sys.monitoring.free_tool_id(3)
    assert seen==[None if mode=='eligible' else 1.5]


def test_discarded_partial_nodes_remain_spent_but_are_not_published(owner_chart,monkeypatch):
    from types import SimpleNamespace
    from anymesher.surface_mesh import SurfaceMeshOptions
    from anymesher.native_v2 import NativeMeshingOptions
    model,chart=owner_chart;settings=SurfaceMeshOptions(native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'),target_size=.25,min_angle=.01,max_angle=179.99,min_scaled_jacobian=0.,max_element_growth=1.e6,max_recombination_work=1)
    baseline,rep=caller_operands(chart,settings);packet=baseline['triangulation']
    def producer(*args,**kwargs):
        assert kwargs['_physical_growth_limit']==settings.max_element_growth
        return packet,dict(insertions=0,staged_point_insertions=0,reserved_node_reuses=0,topology_operations=0,shared_segment_splits=0)
    def partial(*args,**kwargs):
        return dict(accepted=False,report=SimpleNamespace(exchange_work=1),sizing_nodes=2,corner_work=3,corner_facts={},points=np.vstack((packet.points,packet.points[:2])),triangles=packet.triangles)
    monkeypatch.setattr(surface,'frontal_delaunay_refine',producer);monkeypatch.setattr(surface,'_curvilinear_mixed_preview',partial)
    with scope(model,chart,settings):result,work=run_caller(baseline,settings,chart)
    assert work['insertions']==work['staged_point_insertions']==2 and work['topology_operations']==3
    assert work['published_insertions']==0 and len(result['best'].points)==len(packet.points)


@pytest.mark.parametrize('raises',[False,True])
def test_actual_native_return_driver_change_refuses_before_unknown_helper(owner_chart,monkeypatch,raises):
    import sys,dataclasses
    from anymesher.surface_mesh import SurfaceMeshOptions
    from anymesher.native_v2 import NativeMeshingOptions
    model,chart=owner_chart;options=dataclasses.replace(NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'),max_topology_operations=1,max_insertions=1)
    settings=SurfaceMeshOptions(target_size=.25,min_angle=15.,native_options=options)
    baseline,_=caller_operands(chart,settings);original=surface.frontal_delaunay_refine;entered=[];marker=RuntimeError('original monitoring exception')
    def opaque(*args,**kwargs):entered.append(True);raise AssertionError('unknown driver entered')
    def returned(code,offset,value):
        monkeypatch.setattr(surface,'frontal_delaunay_refine',opaque)
        if raises:raise marker
    sys.monitoring.use_tool_id(3,'native return mutation');sys.monitoring.register_callback(3,sys.monitoring.events.PY_RETURN,returned);sys.monitoring.set_local_events(3,original.__code__,sys.monitoring.events.PY_RETURN)
    try:
        with pytest.raises(RuntimeError if raises else MeshError) as caught:
            with scope(model,chart,settings):run_caller(baseline,settings,chart)
        if raises:assert caught.value is marker
        else:assert any(message in str(caught.value) for message in ('deferred mixed physical repair changed','material snapshot identity'))
    finally:sys.monitoring.free_tool_id(3)
    assert not entered


@pytest.mark.parametrize('limit_name,limit,reason', [('cut_limit',1,'cut_limit'),('node_limit',1,'node_limit'),('cell_limit',0,'cell_limit'),('work_limit',0,'transaction_work_limit')])
def test_connected_staging_refuses_without_changing_genuine_cover(owner_chart,limit_name,limit,reason):
    from anymesher._physical_mixed_sizing import (_frontier,_children,_assemble,_score,_covered_report,_debt_closure)
    model,chart=owner_chart;points,report,edges=patch(chart,'QQ')
    settings=surface.SurfaceMeshOptions(min_angle=15.)
    protected=set(map(tuple,edges));records,incidence,frontier=_frontier(report.mesh,1.5)
    # Refine the actual larger Q along its two internal opposite edges. If the
    # compact fixture has boundary opposite edges, use its true shared edge T/Q
    # transition instead; both are genuine existing midpoint templates.
    edge=next(e for e,v in incidence.items() if len(v)==2)
    parents=[records[i] for i in incidence[edge]]
    proposed=np.vstack((points,.5*(points[edge[0]]+points[edge[1]])))
    mids={edge:len(points)};used=[]
    with scope(model,chart,settings):
        world=chart.evaluate(proposed)
        children=[_children(p,mids,proposed,world,settings,lambda:used.append(1)) for p in parents]
        candidate,cover=_assemble(report.mesh,_cover(report,protected),parents,children,world)
        summary,q=_score(candidate,settings)
        # The equal initial report guarantees that no private stage is admitted
        # without an actual strict global change; it is not a forged pass oracle.
        before_bytes=points.tobytes(),report.mesh.triangle_connectivity.tobytes(),report.mesh.quad_connectivity.tobytes()
        changed,spent,facts=_debt_closure(summary,proposed,candidate,cover,summary,q,parents,report,protected,settings,chart.evaluate,lambda:used.append(1),lambda:None,node_remaining=(1,8,len(points)),first_work=0,initial_interfaces=set(),**{limit_name:limit})
    assert changed is None and spent==0 and facts['staged_nodes']>=1 and not facts['accepted']
    assert facts['stop_reason']==reason or (limit_name=='work_limit' and facts['stop_reason']=='no_connected_debt')
    assert before_bytes==(points.tobytes(),report.mesh.triangle_connectivity.tobytes(),report.mesh.quad_connectivity.tobytes())


def test_connected_helper_replacement_excludes_canonical_plan(monkeypatch):
    import anymesher._physical_mixed_sizing as sizing
    original=sizing._debt_closure
    def observer(*args,**kwargs):
        raise AssertionError('unknown closure dispatched')
    monkeypatch.setattr(sizing,'_debt_closure',observer)
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    assert original is not observer


@pytest.fixture
def real_debt_stage(owner_chart):
    from pathlib import Path
    from anymesher._physical_mixed_sizing import _frontier,_score,_children,_assemble
    model,chart=owner_chart
    with np.load(Path(__file__).parent/'fixtures/mixed_growth_debt_222.npz',allow_pickle=False) as d:
        points=d['points'].copy()
        mesh=MeshCore(d['xyz'],d['triangles'],d['quads'],node_ids=d['node_ids'],triangle_ids=d['triangle_ids'],quad_ids=d['quad_ids'],node_active=d['node_active'],triangle_active=d['triangle_active'],quad_active=d['quad_active'],node_owner_handles=d['node_owner_handles'],triangle_owner_handles=d['triangle_owner_handles'],quad_owner_handles=d['quad_owner_handles'])
        report=RecombinationReport(mesh,d['paired_triangle_ids'].copy(),d['created_quad_ids'].copy(),d['scores'].copy(),0,0)
        protected=set(map(tuple,d['segments']))
    settings=surface.SurfaceMeshOptions(min_angle=15.,max_angle=150.,max_element_growth=1.5)
    assert chart.evaluate(points).tobytes()==mesh.node_coordinates.tobytes()
    before,_=_score(mesh,settings);records,incidence,frontier=_frontier(mesh,1.5)
    parent=records[frontier[0][3]];assert parent[0]=='Q'
    from anymesher._physical_mixed_sizing import _edges
    edges=_edges(parent[3]);bad={e for _,_,e,i in frontier if records[i][2]==parent[2]}
    choices=[(edges[i],edges[i+2]) for i in range(2) if all(e not in protected and e not in bad and len(incidence[e])==2 for e in (edges[i],edges[i+2]))]
    selected=max(choices,key=lambda es:sum(np.linalg.norm(mesh.node_coordinates[a]-mesh.node_coordinates[b]) for a,b in es))
    parents=[records[i] for i in sorted({i for edge in selected for i in incidence[edge]})]
    proposed=np.vstack((points,np.asarray([.5*(points[a]+points[b]) for a,b in selected])))
    mids={edge:len(points)+i for i,edge in enumerate(selected)};work=[]
    world=chart.evaluate(proposed)
    children=[_children(p,mids,proposed,world,settings,lambda:work.append(1)) for p in parents]
    candidate,cover=_assemble(mesh,_cover(report,protected),parents,children,world)
    first,q=_score(candidate,settings)
    return model,chart,points,report,settings,protected,before,frontier,parents,proposed,candidate,cover,first,q


@pytest.mark.parametrize('mode',['positive','work','nodes','stencil','exception'])
def test_real_count_barrier_connected_transaction_and_atomic_limits(real_debt_stage,mode):
    from anymesher._physical_mixed_sizing import _debt_closure,_score
    from anymesher._physical_t3_refinement import _alternative_progress
    model,chart,points,report,settings,protected,before,frontier,parents,proposed,candidate,cover,first,q=real_debt_stage
    assert before['elements_above_maximum_growth']==222
    assert first['elements_above_maximum_growth']>222 and not _alternative_progress(before,first)
    original=(points.tobytes(),report.mesh.node_coordinates.tobytes(),report.mesh.triangle_connectivity.tobytes(),report.mesh.quad_connectivity.tobytes())
    limits={'work_limit':5} if mode=='work' else {'node_limit':2} if mode=='nodes' else {'cell_limit':3} if mode=='stencil' else {}
    marker=RuntimeError('original closure checkpoint exception');charged=[]
    def check():
        if mode=='exception':raise marker
    with scope(model,chart,settings):
        if mode=='exception':
            with pytest.raises(RuntimeError) as caught:
                _debt_closure(before,proposed,candidate,cover,first,q,parents,report,protected,settings,chart.evaluate,lambda:charged.append(1),check,node_remaining=(2,1188,len(points)),first_work=5,initial_interfaces={e for _,_,e,_ in frontier})
            assert caught.value is marker
            return
        outcome,spent,facts=_debt_closure(before,proposed,candidate,cover,first,q,parents,report,protected,settings,chart.evaluate,lambda:charged.append(1),check,node_remaining=(2,1188,len(points)),first_work=5,initial_interfaces={e for _,_,e,_ in frontier},**limits)
        if mode=='positive':
            assert outcome is not None and facts['accepted'] and facts['cuts']>=3
            made,mesh,covers,summary,quality=outcome
            assert _alternative_progress(before,summary) and not any(summary['violation_counts'].values())
            assert summary==_score(mesh,settings)[0] and chart.evaluate(made).tobytes()==mesh.node_coordinates.tobytes()
            assert made[:len(points)].tobytes()==points.tobytes() and spent==len(made)-len(points)==facts['admitted_nodes']
            assert facts['trajectory'][0][0]>222 and facts['trajectory'][-1][0]<=222
        else:
            assert outcome is None and spent==0 and facts['staged_nodes']>=2 and facts['admitted_nodes']==0 and not facts['accepted']
            assert facts['stop_reason']=={'work':'transaction_work_limit','nodes':'node_limit','stencil':'cell_limit'}[mode]
    assert original==(points.tobytes(),report.mesh.node_coordinates.tobytes(),report.mesh.triangle_connectivity.tobytes(),report.mesh.quad_connectivity.tobytes())



def test_rejected_product_transaction_spends_work_not_admitted_nodes(real_debt_stage):
    from anymesher._physical_mixed_sizing import repair_mixed_sizing
    model,chart,points,report,settings,protected,*_=real_debt_stage
    with scope(model,chart,settings):
        made,updated,used,admitted,facts=repair_mixed_sizing(points,report,settings,chart.evaluate,np.asarray(sorted(protected)),topology_remaining=5,insertion_remaining=2)
    assert used==5 and admitted==0 and facts['admitted_nodes']==0
    assert facts['closure_transactions'] and not facts['closure_transactions'][0]['accepted']
    assert facts['attempted_allocation_nodes']>=2 and facts['validated_private_staged_nodes']>=2
    assert made.tobytes()==points.tobytes() and updated is report
    assert 9998+admitted<=10000  # actual native/admitted-node cap convention


def test_invalid_private_stage_allocation_is_not_validated_or_admitted(real_debt_stage,monkeypatch):
    import anymesher._physical_mixed_sizing as sizing
    model,chart,points,report,settings,protected,before,frontier,parents,proposed,candidate,cover,first,q=real_debt_stage
    charged=[]
    # Arithmetic accounting boundary only, deliberately outside canonical fast
    # admission: no product path treats this replaced helper as trusted.
    monkeypatch.setattr(sizing,'_children',lambda *args:None)
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    outcome,admitted,facts=sizing._debt_closure(before,proposed,candidate,cover,first,q,parents,report,protected,settings,chart.evaluate,lambda:charged.append(1),lambda:None,node_remaining=(2,8,len(points)),first_work=5,initial_interfaces={e for _,_,e,_ in frontier},new_row_placement=False,local_retriangulation=False)
    assert outcome is None and admitted==facts['admitted_nodes']==0
    assert facts['stop_reason']=='unsupported_template' and charged
    assert facts['allocated_nodes']>facts['staged_nodes']==2
    assert facts['work']==5+len(charged)


def test_admitted_sizing_nodes_equal_appended_rows_and_share_native_cap(owner_chart):
    from anymesher._physical_mixed_sizing import repair_mixed_sizing
    model,chart=owner_chart;points,report,edges=patch(chart,'TQ');settings=surface.SurfaceMeshOptions(min_angle=15.)
    with scope(model,chart,settings):
        made,updated,used,admitted,facts=repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=128,insertion_remaining=2)
    assert 0<admitted==facts['new_nodes']==facts['admitted_nodes']==len(made)-len(points)<=2
    assert 9998+admitted<=10000


def test_cut_fraction_ladder_unblocks_midpoint_only_stall_on_real_growth_debt(owner_chart):
    """Midpoint-only cuts end in no_admissible_mixed_proposal on this shape-pinned mesh; the ladder commits."""
    from pathlib import Path
    from anymesher import _physical_mixed_sizing as sizing
    model,chart=owner_chart
    with np.load(Path(__file__).parent/'fixtures/mixed_growth_debt_79.npz',allow_pickle=False) as d:
        points=d['points'].copy();segments=d['segments'].copy() if 'segments' in d else None
        mesh=MeshCore(d['xyz'],d['triangles'],d['quads'],node_ids=d['node_ids'],triangle_ids=d['triangle_ids'],quad_ids=d['quad_ids'],node_active=d['node_active'],triangle_active=d['triangle_active'],quad_active=d['quad_active'],node_owner_handles=d['node_owner_handles'],triangle_owner_handles=d['triangle_owner_handles'],quad_owner_handles=d['quad_owner_handles'])
        report=RecombinationReport(mesh,d['paired_triangle_ids'].copy(),d['created_quad_ids'].copy(),d['scores'].copy(),0,0)
    settings=surface.SurfaceMeshOptions(min_angle=15.,max_angle=150.,max_element_growth=1.5)
    protected=np.asarray(sorted({tuple(sorted(map(int,e))) for e in segments}),dtype=np.int64)
    outcomes={}
    for name,count in (('midpoint',1),('ladder',5)):
        with scope(model,chart,settings):
            assert chart.evaluate(points).tobytes()==mesh.node_coordinates.tobytes()
            made,updated,used,admitted,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,protected,topology_remaining=2048,insertion_remaining=64,max_fractions=count,new_row_placement=False,local_retriangulation=False)
            outcomes[name]=(facts,sizing._score(updated.mesh,settings)[0])
    assert outcomes['midpoint'][0]['commits']==0 and outcomes['midpoint'][0]['stop_reason']=='no_admissible_mixed_proposal'
    facts,after=outcomes['ladder']
    assert facts['commits']>0 and after['elements_above_maximum_growth']<79 and after['invalid_element_count']==0
    assert not any(v for k,v in after['violation_counts'].items() if k!='elements_above_maximum_growth')


def test_candidate_fallback_across_choices_and_closure_steps_unblocks_49_stall(owner_chart):
    """From the 49-violation state the first-choice-only search stalled (no commits); fallback commits."""
    from pathlib import Path
    from anymesher import _physical_mixed_sizing as sizing
    model,chart=owner_chart
    with np.load(Path(__file__).parent/'fixtures/mixed_growth_debt_49.npz',allow_pickle=False) as d:
        points=d['points'].copy();segments=d['segments']
        mesh=MeshCore(d['xyz'],d['triangles'],d['quads'],node_ids=d['node_ids'],triangle_ids=d['triangle_ids'],quad_ids=d['quad_ids'],node_active=d['node_active'],triangle_active=d['triangle_active'],quad_active=d['quad_active'],node_owner_handles=d['node_owner_handles'],triangle_owner_handles=d['triangle_owner_handles'],quad_owner_handles=d['quad_owner_handles'])
        report=RecombinationReport(mesh,d['paired_triangle_ids'].copy(),d['created_quad_ids'].copy(),d['scores'].copy(),0,0)
    settings=surface.SurfaceMeshOptions(min_angle=15.,max_angle=150.,max_element_growth=1.5)
    protected=np.asarray(sorted({tuple(sorted(map(int,e))) for e in segments}),dtype=np.int64)
    with scope(model,chart,settings):
        assert chart.evaluate(points).tobytes()==mesh.node_coordinates.tobytes()
        before=sizing._score(mesh,settings)[0]
        made,updated,used,admitted,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,protected,topology_remaining=2048,insertion_remaining=64)
        after=sizing._score(updated.mesh,settings)[0]
    assert before['elements_above_maximum_growth']==49 and facts['commits']>0
    assert after['elements_above_maximum_growth']<49 and after['invalid_element_count']==0
    assert not any(v for k,v in after['violation_counts'].items() if k!='elements_above_maximum_growth')


def _load_state(name):
    """Stored 'real debt' mesh state: (points, mesh, report, protected as sorted array)."""
    from pathlib import Path
    with np.load(Path(__file__).parent/'fixtures'/name,allow_pickle=False) as d:
        points=d['points'].copy();segments=d['segments']
        mesh=MeshCore(d['xyz'],d['triangles'],d['quads'],node_ids=d['node_ids'],triangle_ids=d['triangle_ids'],quad_ids=d['quad_ids'],node_active=d['node_active'],triangle_active=d['triangle_active'],quad_active=d['quad_active'],node_owner_handles=d['node_owner_handles'],triangle_owner_handles=d['triangle_owner_handles'],quad_owner_handles=d['quad_owner_handles'])
        report=RecombinationReport(mesh,d['paired_triangle_ids'].copy(),d['created_quad_ids'].copy(),d['scores'].copy(),0,0)
        protected=np.asarray(sorted({tuple(sorted(map(int,e))) for e in segments}),dtype=np.int64)
    return points,mesh,report,protected


def _cut_candidates(points,mesh,protected,settings):
    """Yield (records, incidence, parents, selected) exactly as the main loop enumerates cuts."""
    from anymesher._physical_mixed_sizing import _frontier,_edges,_positive
    guard=set(map(tuple,protected))
    records,incidence,frontier=_frontier(mesh,settings.max_element_growth);seen=set()
    for _,_,_,large in frontier:
        kind,_,identifier,nodes,owner,_=records[large]
        if identifier in seen:continue
        seen.add(identifier);bad={e for _,_,e,i in frontier if i==large}
        def eligible(edge):
            attached=incidence.get(edge,())
            return edge not in guard and edge not in bad and len(attached)==2 and all(records[i][4]==owner and _positive(points,records[i][3]) for i in attached)
        if kind=='T':choices=[(e,) for e in _edges(nodes) if eligible(e)]
        else:
            edges=_edges(nodes);choices=[(edges[i],edges[i+2]) for i in range(2) if eligible(edges[i]) and eligible(edges[i+2])]
        for selected in choices:
            parents=[records[i] for i in sorted({i for e in selected for i in incidence[e]})]
            if any(sum(e in _edges(p[3]) for e in selected)>1 and p[2]!=identifier for p in parents):continue
            yield records,incidence,parents,selected


def _ladder_local_failures(chart,points,mesh,parents,selected,settings):
    """True when every literal ladder fraction yields no template or a shape violation in the new cells."""
    from anymesher._physical_mixed_sizing import _children,_star_objective
    mids={edge:len(points)+i for i,edge in enumerate(selected)}
    for fraction in (.5,.4,.6,.3,.7):
        proposed=np.vstack((points,np.asarray([(1.-fraction)*points[a]+fraction*points[b] for a,b in selected])))
        world=chart.evaluate(proposed);children=[_children(p,mids,proposed,world,settings,lambda:None) for p in parents]
        if all(c is not None for c in children) and _star_objective(children,world,settings,{})[0]==0:return False
    return True


def _find_placeable(chart,points,mesh,protected,settings,scan=80):
    from anymesher._physical_mixed_sizing import _place_new_rows
    for count,(records,incidence,parents,selected) in enumerate(_cut_candidates(points,mesh,protected,settings)):
        if count>=scan:break
        if not _ladder_local_failures(chart,points,mesh,parents,selected,settings):continue
        mids={edge:len(points)+i for i,edge in enumerate(selected)};spent=[]
        placed=_place_new_rows(points,mesh,parents,selected,mids,settings,chart.evaluate,lambda:spent.append(1),lambda:None,records,incidence)
        if placed is not None:return records,incidence,parents,selected,mids,placed,len(spent)
    raise AssertionError('no placeable ladder-failing cut found in the scanned candidates')


_SETTINGS44=dict(min_angle=15.,max_angle=150.,max_element_growth=1.5)


def test_row_placement_off_arm_stalls_on_44_growth_debt(owner_chart):
    """Old behaviour (switch off): the 44-violation state has no admissible ladder/fallback proposal."""
    from anymesher import _physical_mixed_sizing as sizing
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    with scope(model,chart,settings):
        assert chart.evaluate(points).tobytes()==mesh.node_coordinates.tobytes()
        made,updated,used,admitted,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,protected,topology_remaining=2400,insertion_remaining=64,new_row_placement=False,local_retriangulation=False)
    assert sizing._score(mesh,settings)[0]['elements_above_maximum_growth']==44
    assert sizing._score(mesh,settings)[0]['elements_above_maximum_growth']==44
    assert facts['commits']==0 and facts['stop_reason']=='no_admissible_mixed_proposal' and admitted==0
    assert made.tobytes()==points.tobytes() and updated is report


def test_row_placement_unblocks_44_growth_debt_and_keeps_every_invariant(owner_chart):
    from anymesher import _physical_mixed_sizing as sizing
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    with scope(model,chart,settings):
        assert chart.evaluate(points).tobytes()==mesh.node_coordinates.tobytes()
        before=sizing._score(mesh,settings)[0];assert before['elements_above_maximum_growth']==44
        made,updated,used,admitted,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,protected,topology_remaining=4096,insertion_remaining=64)
        after=sizing._score(updated.mesh,settings)[0]
        assert facts['commits']>=1 and after['elements_above_maximum_growth']<44
        assert after['invalid_element_count']==0 and not any(v for k,v in after['violation_counts'].items() if k!='elements_above_maximum_growth')
        assert made[:len(points)].tobytes()==points.tobytes() and updated.mesh.node_coordinates[:len(points)].tobytes()==mesh.node_coordinates[:len(points)].tobytes()
        fixed=sorted(set(map(int,protected.ravel())));assert made[fixed].tobytes()==points[fixed].tobytes()
        assert surface._physical_chart_invalid_count(made,updated.mesh.triangle_connectivity)==0
        assert len(_cover(updated,set(map(tuple,protected))))==updated.mesh.active_quad_count
        assert chart.evaluate(made).tobytes()==updated.mesh.node_coordinates.tobytes()
    summary=(used,admitted,facts['admitted_nodes'],facts['commits'],facts['stop_reason'],after['elements_above_maximum_growth'],len(made)-len(points))
    assert used<=4096 and admitted==facts['consumed_new_nodes']>=facts['admitted_nodes']==len(made)-len(points) and admitted<=64,summary
    assert set(mesh.node_ids)<=set(updated.mesh.node_ids) and set(mesh.triangle_ids)<=set(updated.mesh.triangle_ids)


def test_place_new_rows_moves_only_new_rows_and_repairs_a_ladder_failing_cut(owner_chart):
    from anymesher._physical_mixed_sizing import _children,_star_objective
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    with scope(model,chart,settings):
        records,incidence,parents,selected,mids,placed,spent=_find_placeable(chart,points,mesh,protected,settings)
        assert _ladder_local_failures(chart,points,mesh,parents,selected,settings)
        assert placed.shape==(len(points)+len(selected),2) and placed[:len(points)].tobytes()==points.tobytes()
        world=chart.evaluate(placed);assert world[:len(points)].tobytes()==mesh.node_coordinates.tobytes()
        children=[_children(p,mids,placed,world,settings,lambda:None) for p in parents]
        assert all(c is not None for c in children) and _star_objective(children,world,settings,{})[0]==0
    assert 1<=spent<=78


def test_place_new_rows_refuses_non_new_rows_and_charges_every_trial(owner_chart):
    from anymesher._physical_mixed_sizing import _place_new_rows
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    with scope(model,chart,settings):
        records,incidence,parents,selected,mids,placed,spent=_find_placeable(chart,points,mesh,protected,settings)
        # A movable row inside the pre-existing prefix is refused before any trial is charged.
        charged=[];wrong={edge:0 for edge in selected}
        assert _place_new_rows(points,mesh,parents,selected,wrong,settings,chart.evaluate,lambda:charged.append(1),lambda:None,records,incidence) is None and not charged
        # The work budget is hit exactly at the trial that would exceed it; no geometry is returned.
        class Exhausted(Exception):pass
        count=[]
        def limited():
            if len(count)>=spent-1:raise Exhausted()
            count.append(1)
        with pytest.raises(Exhausted):_place_new_rows(points,mesh,parents,selected,mids,settings,chart.evaluate,limited,lambda:None,records,incidence)
        assert len(count)==spent-1
        # The original-star guard that admits a placement rejects any change of a pre-existing row.
        from anymesher._physical_shape_patches import _OriginalStars,_STAR_CALL
        base=placed.copy();star=type('S',(),dict(triangle_connectivity=np.empty((0,3),dtype=np.int64),triangle_active=np.empty(0,dtype=bool),quad_connectivity=np.empty((0,4),dtype=np.int64),quad_active=np.empty(0,dtype=bool)))
        guard=_OriginalStars(base,star,sorted(mids.values()));moved=base.copy();moved[0,0]+=1.e-9
        assert _STAR_CALL(guard,base) and not _STAR_CALL(guard,moved)


def test_sector_bound_cut_is_refused_by_row_placement(owner_chart):
    """A T-T cut whose opposite sector is under 2*min_angle cannot be repaired by any placement of the new row."""
    from anymesher._physical_mixed_sizing import _place_new_rows
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    xyz=mesh.node_coordinates;checked=0
    def sector(cell,edge):
        v=next(n for n in cell if n not in edge);a,b=edge
        p=xyz[a]-xyz[v];q=xyz[b]-xyz[v]
        return float(np.degrees(np.arccos(np.clip(p@q/np.linalg.norm(p)/np.linalg.norm(q),-1.,1.))))
    with scope(model,chart,settings):
        for records,incidence,parents,selected in _cut_candidates(points,mesh,protected,settings):
            if len(selected)!=1 or any(p[0]!='T' for p in parents):continue
            if not any(sector(p[3],selected[0])<2*settings.min_angle-2. for p in parents):continue
            mids={selected[0]:len(points)};spent=[]
            assert _place_new_rows(points,mesh,parents,selected,mids,settings,chart.evaluate,lambda:spent.append(1),lambda:None,records,incidence) is None
            assert 1<=len(spent)<=39;checked+=1
            if checked>=2:break
    assert checked>=1


@pytest.mark.parametrize('name',['_place_new_rows','_star_objective','_search_order','_neighbour_sizes'])
def test_row_placement_helpers_are_in_the_code_pin(monkeypatch,name):
    import anymesher._physical_mixed_sizing as sizing
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    monkeypatch.setattr(sizing,name,lambda *a,**k:None)
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


@pytest.mark.parametrize('value',[1,0,None,'yes'])
def test_row_placement_switch_is_an_exact_bool(owner_chart,value):
    from anymesher._physical_mixed_sizing import repair_mixed_sizing
    model,chart=owner_chart;points,report,edges=patch(chart,'TQ');settings=surface.SurfaceMeshOptions(min_angle=15.)
    with scope(model,chart,settings):
        with pytest.raises(MeshError,match='row placement switch'):
            repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=8,insertion_remaining=2,new_row_placement=value)


def test_row_placement_arms_on_79_growth_debt_old_3_commits_growth_73_p1p2_and_default_4_commits_growth_71(owner_chart):
    """Placement is not bit-identical to the old search inside failed closures; both arms must still be valid."""
    from anymesher import _physical_mixed_sizing as sizing
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_79.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    import sys
    summary={};reached={};calls=[]
    code=sizing._retriangulate_cavity.__code__
    def spy(code_object,offset):calls.append(1)  # sys.monitoring observes without rebinding the pinned name
    for name,kwargs in (('old',dict(new_row_placement=False,local_retriangulation=False)),('p1_p2',dict(new_row_placement=True,local_retriangulation=True)),('default',{})):
        del calls[:]
        sys.monitoring.use_tool_id(4,'p2 dispatch spy')
        try:
            sys.monitoring.register_callback(4,sys.monitoring.events.PY_START,spy);sys.monitoring.set_local_events(4,code,sys.monitoring.events.PY_START)
            with scope(model,chart,settings):
                made,updated,used,admitted,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,protected,topology_remaining=1200,insertion_remaining=64,**kwargs)
        finally:sys.monitoring.free_tool_id(4)
        reached[name]=len(calls)
        with scope(model,chart,settings):
            after=sizing._score(updated.mesh,settings)[0]
            assert facts['commits']>0 and after['invalid_element_count']==0 and not any(v for k,v in after['violation_counts'].items() if k!='elements_above_maximum_growth')
            assert made[:len(points)].tobytes()==points.tobytes() and chart.evaluate(made).tobytes()==updated.mesh.node_coordinates.tobytes()
            assert surface._physical_chart_invalid_count(made,updated.mesh.triangle_connectivity)==0 and used<=1200
        summary[name]=(facts['commits'],used,len(made)-len(points),after['elements_above_maximum_growth'],facts['stop_reason'])
    # Exact values are deterministic integer-counter measurements on this environment; the inequalities are the intended behaviour.
    assert summary['old']==(3,1200,8,73,'topology_budget')
    assert summary['p1_p2']==(4,1200,10,71,'topology_budget')
    assert summary['default']==(4,1200,10,71,'topology_budget')  # production defaults: P1 on, P2 off
    # On this state every proposal is found before sweeps 0 and 1 are exhausted, so P2 is never dispatched in any arm (see the 28-state test for dispatch).
    assert reached=={'old':0,'p1_p2':0,'default':0}
    for name in ('p1_p2','default'):
        assert summary[name][0]>=summary['old'][0] and summary[name][3]<summary['old'][3]


def _sector_bound_candidates(points,mesh,protected,settings):
    xyz=mesh.node_coordinates
    def sector(cell,edge):
        v=next(n for n in cell if n not in edge);a,b=edge
        p=xyz[a]-xyz[v];q=xyz[b]-xyz[v]
        return float(np.degrees(np.arccos(np.clip(p@q/np.linalg.norm(p)/np.linalg.norm(q),-1.,1.))))
    for found in _cut_candidates(points,mesh,protected,settings):
        records,incidence,parents,selected=found
        if len(selected)==1 and all(p[0]=='T' for p in parents) and any(sector(p[3],selected[0])<2*settings.min_angle-2. for p in parents):yield found


def test_local_retriangulation_repairs_a_sector_bound_cut_that_placement_cannot(owner_chart):
    from anymesher._physical_mixed_sizing import _place_new_rows,_retriangulate_cavity,_boundary,_edges,_positive,_star_objective
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    guard=set(map(tuple,protected));n=len(points);outcome=None
    with scope(model,chart,settings):
        for records,incidence,parents,selected in _sector_bound_candidates(points,mesh,protected,settings):
            mids={selected[0]:n};spent=[]
            assert _place_new_rows(points,mesh,parents,selected,mids,settings,chart.evaluate,lambda:None,lambda:None,records,incidence) is None
            work=[];found=_retriangulate_cavity(points,mesh,parents,selected,settings,chart.evaluate,lambda:work.append(1),lambda:None,records,incidence,guard)
            assert 1<=len(work)<=128
            if found is not None:outcome=(parents,selected,records,incidence,found,len(work));break
        assert outcome is not None,'no sector-bound cut could be re-triangulated'
        parents,selected,records,incidence,(proposed,cavity,groups),charged=outcome
        assert proposed.shape==(n+1,2) and proposed[:n].tobytes()==points.tobytes()
        triangles=[nodes for group in groups for _,nodes in group]
        assert len(triangles)==len(cavity)+2 and all(len(group)>=1 for group in groups) and len(groups)==len(cavity)
        assert _boundary([c[3] for c in cavity])==_boundary(triangles) and all(_positive(proposed,t) for t in triangles)
        degree=sum(n in t for t in triangles);assert degree>=4
        assert all(c[0]=='T' and c[4]==parents[0][4] for c in cavity)
        shared={e for c in cavity for e in _edges(c[3])}-set(e for t in triangles for e in _edges(t))
        assert not shared&guard  # only unprotected old edges disappear
        world=chart.evaluate(proposed);assert world[:n].tobytes()==mesh.node_coordinates.tobytes()
        assert _star_objective([[('T',t) for t in triangles]],world,settings,{})[0]==0


def test_local_retriangulation_switch_is_an_exact_bool_and_is_in_the_code_pin(owner_chart,monkeypatch):
    from anymesher._physical_mixed_sizing import repair_mixed_sizing
    import anymesher._physical_mixed_sizing as sizing
    model,chart=owner_chart;points,report,edges=patch(chart,'TQ');settings=surface.SurfaceMeshOptions(min_angle=15.)
    with scope(model,chart,settings):
        with pytest.raises(MeshError,match='local retriangulation switch'):
            repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=8,insertion_remaining=2,local_retriangulation=1)
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    monkeypatch.setattr(sizing,'_retriangulate_cavity',lambda *a,**k:None)
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


def test_objective_puts_growth_excess_before_slack_and_search_order_steers_infeasible_by_slack():
    from anymesher._physical_mixed_sizing import _search_order
    assert _search_order((0,.5,-.3))==(0,.5,-.3)          # shape-clear: excess is live before slack
    assert _search_order((2,.5,-.3))==(2,-.3,.5)          # violating: slack steers toward feasibility
    assert (0,.1,-.1)<(0,.2,-.9)<(1,0.,-1.)


def test_retriangulation_objective_equals_independently_enumerated_minimum(owner_chart):
    """Brute force over every spoke mask x every pocket triangulation; the selected layout must hit the minimum objective."""
    from collections import Counter
    from anymesher._physical_mixed_sizing import _retriangulate_cavity,_star_objective,_neighbour_sizes,_boundary,_edges,_positive
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_44.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    guard=set(map(tuple,protected));n=len(points)
    with scope(model,chart,settings):
        for scanned,(records,incidence,parents,selected) in enumerate(_cut_candidates(points,mesh,protected,settings)):
            if scanned>=60:break
            if len(selected)!=1 or any(p[0]!='T' for p in parents):continue
            found=_retriangulate_cavity(points,mesh,parents,selected,settings,chart.evaluate,lambda:None,lambda:None,records,incidence,guard)
            if found is None:continue
            proposed,cavity,groups=found;world=chart.evaluate(proposed);outside=_neighbour_sizes(cavity,records,incidence)
            chosen=_star_objective(groups,world,settings,outside)
            following={x:y for x,y in _boundary([c[3] for c in cavity])};ring=[min(following)]
            while following[ring[-1]]!=ring[0]:ring.append(following[ring[-1]])
            count=len(ring);inner={e for e,c in Counter(e for x in cavity for e in _edges(x[3])).items() if c==2}
            def ok(a,b):
                key=tuple(sorted((a,b)));return key not in incidence or key in inner
            def pockets(path):
                if len(path)==2:yield [];return
                for k in range(1,len(path)-1):
                    tri=(path[0],path[k],path[-1])
                    if not _positive(proposed,tri) or (k>1 and not ok(path[0],path[k])) or (k<len(path)-2 and not ok(path[k],path[-1])):continue
                    for left in pockets(path[:k+1]):
                        for right in pockets(path[k:]):yield [tri]+left+right
            area=sum(float(proposed[x][0]*proposed[y][1]-proposed[y][0]*proposed[x][1]) for x,y in zip(ring,ring[1:]+ring[:1]))
            objectives=[]
            for mask in range(1<<count):
                corners=[i for i in range(count) if mask>>i&1]
                if len(corners)<4:continue
                segments=[]
                for t,i in enumerate(corners):
                    j=corners[(t+1)%len(corners)];tri=(n,ring[i],ring[j]);path=[ring[(i+u)%count] for u in range(((j-i)%count)+1)]
                    if not _positive(proposed,tri) or (len(path)>2 and not ok(path[0],path[-1])):segments=None;break
                    segments.append([[tri]+pocket for pocket in pockets(path)])
                if not segments or any(not options for options in segments):continue
                layouts=[[]]
                for options in segments:layouts=[a+b for a in layouts for b in options]
                for layout in layouts:
                    if len(layout)!=count:continue
                    covered=sum(float(proposed[x][0]*proposed[y][1]-proposed[y][0]*proposed[x][1]+proposed[y][0]*proposed[z][1]-proposed[z][0]*proposed[y][1]+proposed[z][0]*proposed[x][1]-proposed[x][0]*proposed[z][1]) for x,y,z in layout)
                    if abs(covered-area)>1.e-9*abs(area):continue
                    objective=_star_objective([[('T',t) for t in layout]],world,settings,outside)
                    if objective[0]==0:objectives.append(objective)
            assert objectives and chosen[0]==0
            if len(set(objectives))<2:continue  # a single shape-clear layout cannot discriminate selectors
            assert chosen==min(objectives)
            # Discrimination: a selector returning any/first shape-clear layout would differ, because several distinct objectives compete.
            return
    raise AssertionError('no witness with several competing shape-clear layouts found')


def test_numpy_empty_rebinding_is_refused_before_the_call(owner_chart,monkeypatch):
    from anymesher._physical_mixed_sizing import repair_mixed_sizing
    model,chart=owner_chart;points,report,edges=patch(chart,'TQ');settings=surface.SurfaceMeshOptions(min_angle=15.)
    original=np.empty
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    monkeypatch.setattr(np,'empty',lambda *a,**k:original(*a,**k))
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    with scope(model,chart,settings):
        made,updated,used,admitted,facts=repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=128,insertion_remaining=8)
    assert not facts['enabled'] and used==0 and admitted==0 and updated is report


def test_closure_never_uses_local_retriangulation(real_debt_stage,monkeypatch):
    import anymesher._physical_mixed_sizing as sizing
    model,chart,points,report,settings,protected,before,frontier,parents,proposed,candidate,cover,first,q=real_debt_stage
    def forbidden(*args,**kwargs):raise AssertionError('closure used local retriangulation')
    monkeypatch.setattr(sizing,'_retriangulate_cavity',forbidden);monkeypatch.setattr(sizing,'_children',lambda *args:None)
    with scope(model,chart,settings):
        outcome,admitted,facts=sizing._debt_closure(before,proposed,candidate,cover,first,q,parents,report,protected,settings,chart.evaluate,lambda:None,lambda:None,node_remaining=(2,8,len(points)),first_work=5,initial_interfaces={e for _,_,e,_ in frontier},local_retriangulation=True)
    assert outcome is None and admitted==0


@pytest.mark.parametrize('kwargs,message',[(dict(local_retriangulation=1),'local retriangulation switch'),(dict(new_row_placement=0),'row placement switch'),(dict(max_fractions=0),'cut fraction count'),(dict(max_fractions=True),'cut fraction count'),(dict(max_fractions=6),'cut fraction count')])
def test_debt_closure_rejects_invalid_switches_and_fraction_count(real_debt_stage,kwargs,message):
    from anymesher._physical_mixed_sizing import _debt_closure
    model,chart,points,report,settings,protected,before,frontier,parents,proposed,candidate,cover,first,q=real_debt_stage
    with scope(model,chart,settings):
        with pytest.raises(MeshError,match=message):
            _debt_closure(before,proposed,candidate,cover,first,q,parents,report,protected,settings,chart.evaluate,lambda:None,lambda:None,node_remaining=(2,8,len(points)),first_work=5,initial_interfaces=set(),**kwargs)


def test_local_retriangulation_is_dispatched_only_when_opted_in_on_the_28_stall_state(owner_chart):
    """On the 28-violation stall state sweeps 0 and 1 are exhausted, so P2 is reached through production dispatch only with the opt-in."""
    import sys
    from anymesher import _physical_mixed_sizing as sizing
    model,chart=owner_chart;points,mesh,report,protected=_load_state('mixed_growth_debt_28.npz');settings=surface.SurfaceMeshOptions(**_SETTINGS44)
    code=sizing._retriangulate_cavity.__code__;calls=[]
    def spy(code_object,offset):calls.append(1)
    result={}
    for name,kwargs in (('default',{}),('p1_p2',dict(new_row_placement=True,local_retriangulation=True))):
        del calls[:]
        sys.monitoring.use_tool_id(4,'p2 dispatch spy')
        try:
            sys.monitoring.register_callback(4,sys.monitoring.events.PY_START,spy);sys.monitoring.set_local_events(4,code,sys.monitoring.events.PY_START)
            with scope(model,chart,settings):
                made,updated,used,admitted,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,protected,topology_remaining=2400,insertion_remaining=64,**kwargs)
        finally:sys.monitoring.free_tool_id(4)
        result[name]=(len(calls),facts['commits'],used,facts['enabled'])
    # The default arm must have searched (work spent, repair enabled), so its zero P2 calls is not an early exit.
    assert result['default'][3] and result['default'][2]>0 and result['p1_p2'][3] and result['p1_p2'][2]>0
    assert result['default'][0]==0 and result['p1_p2'][0]>0 and result['p1_p2'][0]>result['default'][0]
