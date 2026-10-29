"""Exact legacy oracles for operation-local arithmetic data, not acceptance substitutes."""
from dataclasses import replace
import numpy as np
import pytest
from anymesher.core import MeshCore
from anymesher.quality_v2 import evaluate_quality
from anymesher.surface_mesh import SurfaceMeshOptions,_qualified_recombination
import anymesher._physical_mixed_repair as mixed
import anymesher._physical_matching as matching
from test_physical_growth_retention import case
from test_targeted_physical_growth import jacobians


@pytest.mark.parametrize('scale',(2.**-12,1.,2.**20))
@pytest.mark.parametrize('inactive',(False,True))
def test_constraint_vector_exact_legacy_order_mixed_ids_signed_zero(scale,inactive):
    points,cells,evaluate,report,settings=case()
    old=report.mesh;xyz=old.node_coordinates.copy()*scale;xyz[0,0]=-0.
    active=old.triangle_active.copy()
    if inactive:active[np.flatnonzero(active)[0]]=False
    mesh=MeshCore(xyz,old.triangle_connectivity,old.quad_connectivity,
        node_ids=old.node_ids+100,triangle_ids=old.triangle_ids+1000,quad_ids=old.quad_ids+1000,
        triangle_active=active,quad_active=old.quad_active)
    quality=evaluate_quality(mesh,check_validity=False);layout=mixed._prepare_constraint_layout(mesh)
    assert layout is not None
    legacy=mixed._physical_constraint_values(mesh,quality,settings)
    fast=mixed._physical_constraint_values(mesh,quality,settings,_layout=layout)
    assert np.array_equal(fast,legacy) and fast.tobytes()==legacy.tobytes()


def test_layout_is_immutable_and_stale_connectivity_uses_fresh_oracle():
    _,_,_,report,settings=case();mesh=report.mesh
    layout=mixed._prepare_constraint_layout(mesh)
    assert type(layout[1][0][1]) is bytes and type(layout[2]) is bytes
    changed=MeshCore(mesh.node_coordinates,np.roll(mesh.triangle_connectivity,1,axis=1),
        mesh.quad_connectivity,triangle_active=mesh.triangle_active,quad_active=mesh.quad_active)
    assert mixed._constraint_growth_vector(changed,settings,layout) is None
    quality=evaluate_quality(changed,check_validity=False)
    assert np.array_equal(mixed._physical_constraint_values(changed,quality,settings,_layout=layout),
        mixed._physical_constraint_values(changed,quality,settings))
    # Settings are read live, not retained in the topology data.
    tightened=replace(settings,max_element_growth=1.01)
    assert np.array_equal(mixed._physical_constraint_values(mesh,evaluate_quality(mesh),tightened,_layout=layout),
        mixed._physical_constraint_values(mesh,evaluate_quality(mesh),tightened))


def test_unknown_norm_context_retains_original_route(monkeypatch):
    _,_,_,report,settings=case();layout=mixed._prepare_constraint_layout(report.mesh)
    norm=np.linalg.norm
    monkeypatch.setattr(np.linalg,'norm',lambda *args,**kwargs:norm(*args,**kwargs))
    assert mixed._constraint_growth_vector(report.mesh,settings,layout) is None
    assert mixed._prepare_constraint_layout(report.mesh) is None


@pytest.mark.parametrize('mutate',(False,True))
def test_connected_worker_matches_full_dynamic_oracle_with_external_mutation(monkeypatch,mutate):
    def run():
        points,cells,evaluate,report,settings=case();settings=replace(settings,max_recombination_work=48)
        mesh=report.mesh;calls=[];diagnostic={}
        def actual(rows):
            calls.append(len(rows))
            if mutate and len(calls)==1:
                # Same oriented carrier, changed live topology literal; no
                # source authority is fabricated by this numerical fixture.
                mesh.triangle_connectivity=np.roll(mesh.triangle_connectivity,1,axis=1)
                object.__setattr__(settings,'max_element_growth',1.09)
            return evaluate(rows)
        made,used,truncated=mixed.repair_mixed_core_coordinates(points,cells,mesh,settings,actual,(0,2,3,5),
            evaluate_jacobians=jacobians,_diagnostics=diagnostic)
        return made,used,truncated,calls,diagnostic
    fast=run()
    monkeypatch.setattr(mixed,'_prepare_constraint_layout',lambda mesh:None)
    legacy=run()
    np.testing.assert_array_equal(fast[0],legacy[0]);assert fast[1:]==legacy[1:]


def grid():
    xyz=np.array([(x,y,0.) for y in (0.,1.,2.) for x in (0.,.8,1.9,3.)])
    triangles=[]
    for j in range(2):
        for i in range(3):
            a=j*4+i;triangles.extend(((a,a+1,a+5),(a,a+5,a+4)))
    return MeshCore(xyz,np.asarray(triangles)),np.empty((0,2),dtype=np.int64)


@pytest.mark.parametrize('work',(1,64,256))
def test_connected_matching_exact_selected_rows_work_and_callbacks(monkeypatch,work):
    core,protected=grid();settings=SurfaceMeshOptions(max_element_growth=1.05,max_recombination_work=work)
    first_events=[];fast,fq=_qualified_recombination(core,protected,settings,first_events.append,physical_growth=True)
    monkeypatch.setattr(matching,'_MATCHING_NUMPY',())
    second_events=[];legacy,lq=_qualified_recombination(core,protected,settings,second_events.append,physical_growth=True)
    for name in ('triangle_connectivity','quad_connectivity','triangle_active','quad_active','node_coordinates'):
        np.testing.assert_array_equal(getattr(fast.mesh,name),getattr(legacy.mesh,name))
    for name in ('paired_triangle_ids','created_quad_ids','scores'):
        np.testing.assert_array_equal(getattr(fast,name),getattr(legacy,name))
    assert (fast.exchange_work,fast.exchange_count,fast.exchange_truncated)==(legacy.exchange_work,legacy.exchange_count,legacy.exchange_truncated)
    assert fq==lq and first_events==second_events


def test_matching_callback_exception_identity_and_input_immutability():
    core,protected=grid();before=core.triangle_connectivity.tobytes();error=RuntimeError('stop matching')
    def check(phase):
        if phase=='physical recombination alternating path':raise error
    with pytest.raises(RuntimeError) as caught:
        _qualified_recombination(core,protected,SurfaceMeshOptions(max_element_growth=1.05),check,physical_growth=True)
    assert caught.value is error and core.triangle_connectivity.tobytes()==before


@pytest.mark.parametrize('empty',(False,True))
def test_empty_and_degenerate_rows_retain_exact_constraint_vector(empty):
    if empty:mesh=MeshCore(np.empty((0,3)),np.empty((0,3),dtype=np.int64))
    else:
        xyz=np.array(((0.,0.,0.),(0.,0.,0.),(1.,0.,0.),(1.,1.,0.)))
        mesh=MeshCore(xyz,((0,1,2),(1,3,2)))
    quality=evaluate_quality(mesh,check_validity=False);settings=SurfaceMeshOptions()
    with np.errstate(all='ignore'):
        old=mixed._physical_constraint_values(mesh,quality,settings)
        new=mixed._physical_constraint_values(mesh,quality,settings,_layout=mixed._prepare_constraint_layout(mesh))
    assert old.tobytes()==new.tobytes()
