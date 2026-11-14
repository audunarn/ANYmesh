"""Local (incremental) mixed-sizing score equals the full score, cell for cell."""
import numpy as np
import pytest
from test_physical_mixed_sizing import owner_chart  # noqa: F401  (fixture)
from anymesher import _physical_mixed_sizing as sizing
from anymesher import surface_mesh as surface
from anymesher.core import MeshCore
from anymesher.recombine import RecombinationReport

SETTINGS=surface.SurfaceMeshOptions(min_angle=15.,max_angle=150.,max_element_growth=1.5)
FIELDS=('element_ids','area','aspect_ratio','minimum_angle','maximum_angle','scaled_jacobian','warpage')


def grid(nx=5,ny=4):
    """Flat quad grid (every quad covered by two inactive triangles) plus a triangle strip.

    Column widths grow geometrically so growth debt exists; one row of cells is triangles only.
    """
    xs=np.cumsum([0.]+[1.4**i for i in range(nx)]);ys=np.arange(ny+1,dtype=float)
    ids={};pts=[]
    for j,y in enumerate(ys):
        for i,x in enumerate(xs):ids[i,j]=len(pts);pts.append((x+.03*((i*7+j*3)%4),y+.04*((i*5+j)%3),0.))
    quads=[];tris=[];active=[];pairs=[]
    for j in range(ny):
        for i in range(nx):
            a,b,c,d=ids[i,j],ids[i+1,j],ids[i+1,j+1],ids[i,j+1]
            if j==ny-1:tris.extend([[a,b,c],[a,c,d]]);active.extend([True,True])
            else:
                first=len(tris);tris.extend([[a,b,c],[a,c,d]]);active.extend([False,False]);quads.append([a,b,c,d]);pairs.append([first+1,first+2])
    mesh=MeshCore(np.asarray(pts),np.asarray(tris),np.asarray(quads),triangle_active=np.asarray(active))
    report=RecombinationReport(mesh,np.asarray(pairs),mesh.quad_ids.copy(),np.ones(len(pairs)),len(pairs),0)
    from collections import Counter
    cells=[tuple(r) for r,a in zip(tris,active) if a]+[tuple(q) for q in quads]
    edges=Counter(tuple(sorted(map(int,(a,b)))) for r in cells for a,b in zip(r,r[1:]+r[:1]))
    protected={e for e,n in edges.items() if n==1}
    return mesh,report,protected


def candidates(mesh,report,protected,limit=40):
    """Yield (selected edges, candidate, cover) for every interior cut that assembles."""
    covers=sizing._cover(report,protected);records,incidence,_=sizing._frontier(mesh,1.5)
    count=0
    for edge,attached in sorted(incidence.items()):
        if edge in protected or len(attached)!=2:continue
        for selected in ((edge,),):
            parents=[records[i] for i in sorted({i for e in selected for i in incidence[e]})]
            xyz0=mesh.node_coordinates;mids={e:len(xyz0)+k for k,e in enumerate(selected)}
            xyz=np.vstack((xyz0,[.5*(xyz0[a]+xyz0[b]) for a,b in selected]))
            children=[sizing._children(p,mids,xyz[:,:2],xyz,SETTINGS,lambda:None) for p in parents]
            if any(c is None for c in children):continue
            made=sizing._assemble(mesh,covers,parents,children,xyz)
            if made is None:continue
            yield selected,made[0],made[1];count+=1
            if count>=limit:return


def equal(local,full):
    (lr,lq),(fr,fq)=local,full
    assert lr==fr
    for group in ('triangles','quadrilaterals'):
        for name in FIELDS:
            a,b=getattr(getattr(lq,group),name),getattr(getattr(fq,group),name)
            assert a.dtype==b.dtype and a.tobytes()==b.tobytes(),(group,name)
    assert lq.validity.errors==fq.validity.errors


@pytest.fixture(scope='module')
def base():
    mesh,report,protected=grid();return mesh,report,protected,sizing._score(mesh,SETTINGS)[1]


def test_local_score_equals_full_score_for_every_candidate_and_chained_commits(base):
    mesh,report,protected,q=base;seen=0
    for _,candidate,cover in candidates(mesh,report,protected):
        equal(sizing._score_local(mesh,q,candidate,SETTINGS),sizing._score(candidate,SETTINGS));seen+=1
    assert seen>=15
    # chain: adopt a candidate as the next base, twice
    current,cq=mesh,q;cur_report,cur_protected=report,protected
    for _ in range(3):
        selected,candidate,cover=next(candidates(current,cur_report,cur_protected))
        local=sizing._score_local(current,cq,candidate,SETTINGS);equal(local,sizing._score(candidate,SETTINGS))
        cur_report=sizing._covered_report(cur_report,candidate,cover,local[1],cur_protected);current,cq=candidate,local[1]


def test_local_score_counts_only_affected_cells(base):
    mesh,report,protected,q=base;total=int(mesh.triangle_active.sum()+mesh.quad_active.sum())
    _,candidate,_=next(candidates(mesh,report,protected))
    stats=sizing._ScoreStats();sizing._score_local(mesh,q,candidate,SETTINGS,stats)
    assert stats.local_calls==1 and stats.local_fallbacks==0
    evaluated=stats.cells_evaluated
    assert 0<evaluated<=8 and evaluated*4<total


@pytest.mark.parametrize('drop',['all','last'])
def test_wrong_affected_set_is_detected(base,monkeypatch,drop):
    mesh,report,protected,q=base
    _,candidate,_=next(candidates(mesh,report,protected))
    real=sizing._recompute_rows
    def wrong(a,b):
        t,qq=real(a,b)
        return (t[:0],qq[:0]) if drop=='all' else (t[:-1],qq[:-1])
    monkeypatch.setattr(sizing,'_recompute_rows',wrong)
    stats=sizing._ScoreStats()
    try:equal(sizing._score_local(mesh,q,candidate,SETTINGS,stats),sizing._score(candidate,SETTINGS));differs=False
    except AssertionError:differs=True
    # An incomplete set must never go unnoticed: it either changes the result
    # (this test fails) or is refused by the certifier and re-scored in full.
    assert differs or stats.local_fallbacks>0


def test_correct_affected_set_never_needs_the_fallback(base):
    mesh,report,protected,q=base;stats=sizing._ScoreStats()
    for _,candidate,_ in candidates(mesh,report,protected):sizing._score_local(mesh,q,candidate,SETTINGS,stats)
    assert stats.local_fallbacks==0 and stats.local_calls>=15


def test_changed_node_prefix_or_invalid_candidate_falls_back_to_full_score(base):
    mesh,report,protected,q=base
    _,candidate,_=next(candidates(mesh,report,protected))
    moved=candidate.node_coordinates.copy();moved[0,0]+=.01
    changed=MeshCore(moved,candidate.triangle_connectivity,candidate.quad_connectivity,node_ids=candidate.node_ids,triangle_ids=candidate.triangle_ids,quad_ids=candidate.quad_ids,node_active=candidate.node_active,triangle_active=candidate.triangle_active,quad_active=candidate.quad_active,node_owner_handles=candidate.node_owner_handles,triangle_owner_handles=candidate.triangle_owner_handles,quad_owner_handles=candidate.quad_owner_handles)
    stats=sizing._ScoreStats()
    equal(sizing._score_local(mesh,q,changed,SETTINGS,stats),sizing._score(changed,SETTINGS))
    assert stats.local_fallbacks==1
    inverted=candidate.node_coordinates.copy();inverted[-1]=inverted[0]   # new node on a base node -> degenerate
    bad=MeshCore(inverted,candidate.triangle_connectivity,candidate.quad_connectivity,node_ids=candidate.node_ids,triangle_ids=candidate.triangle_ids,quad_ids=candidate.quad_ids,node_active=candidate.node_active,triangle_active=candidate.triangle_active,quad_active=candidate.quad_active,node_owner_handles=candidate.node_owner_handles,triangle_owner_handles=candidate.triangle_owner_handles,quad_owner_handles=candidate.quad_owner_handles)
    stats=sizing._ScoreStats()
    equal(sizing._score_local(mesh,q,bad,SETTINGS,stats),sizing._score(bad,SETTINGS))
    assert stats.local_fallbacks==1


@pytest.mark.parametrize('name',['_score_local','_recompute_rows','_spliced'])
def test_local_score_helpers_are_in_the_code_pin(monkeypatch,name):
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    monkeypatch.setattr(sizing,name,lambda *a,**k:None)
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


def test_certified_validity_helper_is_in_the_code_pin(monkeypatch):
    from anymesher import quality_v2
    monkeypatch.setattr(quality_v2,'_certified_linear_validity',lambda *a,**k:None)
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


@pytest.mark.parametrize('name',['_score_local','_recompute_rows','_spliced'])
def test_replaced_local_score_helper_refuses_before_it_is_entered(owner_chart,monkeypatch,name):
    from test_physical_mixed_sizing import patch,scope
    model,chart=owner_chart;points,report,edges=patch(chart,'QQ');settings=surface.SurfaceMeshOptions(min_angle=15.);entered=[]
    def stub(*a,**k):entered.append(name);raise AssertionError('replaced helper entered')
    monkeypatch.setattr(sizing,name,stub)
    with scope(model,chart,settings):
        made,updated,work,nodes,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=128,insertion_remaining=8)
    # A changed calculation helper makes the whole repair ineligible: nothing is entered or changed.
    assert not entered and facts['enabled'] is False and facts['stop_reason']=='ineligible'
    assert made is points and updated is report and (work,nodes)==(0,0)


_QUALITY_NAMES=['np','ElementQuality','ValidityReport','MeshQualityV2','MeshCore','corner_edges','MeshError','acos','atan2','degrees','pi',
    'quad_quality','_quad_values','_corner_data','_normal','_frozen','native_element_quality','triangle_quality']


@pytest.mark.parametrize('name',_QUALITY_NAMES)
def test_every_quality_name_read_by_the_local_score_path_is_in_the_code_pin(monkeypatch,name):
    from anymesher import quality_v2
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    monkeypatch.setattr(quality_v2,name,object() if name in ('np','ElementQuality','ValidityReport','MeshQualityV2','MeshCore','MeshError','pi') else (lambda *a,**k:None))
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


def test_frozen_defaults_and_code_are_in_the_code_pin(monkeypatch):
    from anymesher import quality_v2
    original=quality_v2._frozen.__defaults__
    monkeypatch.setattr(quality_v2._frozen,'__defaults__',(np.float32,))
    assert surface._physical_pairing_services()!=surface._REFERENCE_PHYSICAL_PAIRING_SERVICES
    monkeypatch.setattr(quality_v2._frozen,'__defaults__',original)
    assert surface._physical_pairing_services()==surface._REFERENCE_PHYSICAL_PAIRING_SERVICES


def _globals_read(function):
    import dis
    found=set()
    def walk(code):
        found.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
        for constant in code.co_consts:
            if hasattr(constant,'co_code'):walk(constant)
    walk(function.__code__);return found


def test_local_score_helpers_read_no_unpinned_module_state():
    """No module-level constant/tuple is read: only pinned helpers, numpy (id-pinned) and builtins.

    Builtins and their shadowing are outside the admission contract (see _physical_pairing_services).
    """
    import builtins
    allowed={'_recompute_rows','_spliced','_score','np'}
    for name in ('_score_local','_recompute_rows','_spliced'):
        extra={g for g in _globals_read(getattr(sizing,name)) if g not in allowed and not hasattr(builtins,g)}
        assert not extra,(name,extra)
    assert not hasattr(sizing,'_FIELDS')


def test_replaced_quad_quality_makes_the_repair_ineligible_before_it_is_entered(owner_chart,monkeypatch):
    from anymesher import quality_v2
    from test_physical_mixed_sizing import patch,scope
    model,chart=owner_chart;points,report,edges=patch(chart,'QQ');settings=surface.SurfaceMeshOptions(min_angle=15.);entered=[]
    def stub(*a,**k):entered.append(1);raise AssertionError('replaced quality helper entered')
    monkeypatch.setattr(quality_v2,'quad_quality',stub)
    with scope(model,chart,settings):
        made,updated,work,nodes,facts=sizing.repair_mixed_sizing(points,report,settings,chart.evaluate,edges,topology_remaining=128,insertion_remaining=8)
    assert not entered and facts['enabled'] is False and updated is report
