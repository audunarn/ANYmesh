"""Scheduling/work/carrier regressions on a genuine owner-bound component."""
from dataclasses import replace
from importlib import import_module
import numpy as np
import pytest
from test_physical_mixed_repair_batch import chart_fixture
from test_physical_corner_cavity import operands
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.triangulation import PlanarTriangulation

surface=import_module('anymesher.surface_mesh')


def run_component(chart,monkeypatch,check=None,evaluate=None,baseline_extra=None):
    uv,mixed,settings,edges=operands(chart)
    settings=replace(settings,min_angle=1.,min_scaled_jacobian=1e-8,max_aspect_ratio=100.,
        max_angle=179.,max_element_growth=100.,max_recombination_work=10000,target_size=.1,
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))
    protected=np.asarray(edges,dtype=np.int64)
    tri=PlanarTriangulation(uv,mixed.mesh.triangle_connectivity,protected,protected,
        np.empty((0,2),dtype=np.int64),uv[[0,1,5,4]],())
    best=surface._make_candidate(uv,tri.triangles,settings=settings)
    baseline=dict(best=best,triangulation=tri,target_met=False,generated=None,
        triangulation_seconds=0.,work_statistics={})
    if baseline_extra: baseline.update(baseline_extra)
    initial_report=dict(topology_operations=7,insertions=0,shared_segment_splits=0)
    monkeypatch.setattr(surface,'frontal_delaunay_refine',lambda *args,**kwargs:(tri,dict(initial_report)))
    monkeypatch.setattr(surface,'_optimize_candidate',lambda candidate,*args,**kwargs:candidate)
    path,report=surface._run_frontal_quality_path(baseline,settings,check,model_uuid=None,
        geometry_revision=None,metric_to_physical=chart.evaluate if evaluate is None else evaluate,
        metric_jacobian=None,automatically_seeded_shared_segments=None,component_seed_registry=None,
        supplemental_metric_field=None,preserve_spatial_refinement=True)
    return path,report,settings,tri


def test_complete_early_pair_preview_carrier_and_cache_are_consistent(chart_fixture,monkeypatch):
    _model,chart=chart_fixture
    from anymesher._physical_corner_cavity import repair_corner_cavities
    import anymesher._physical_corner_cavity as corner
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1); return repair_corner_cavities(*args,**kwargs)
    monkeypatch.setattr(corner,'repair_corner_cavities',counted)
    path,report,settings,original=run_component(chart,monkeypatch)
    preview=path['_curvilinear_mixed_preview']
    assert report['early_pair_mixed_preview']['accepted'] and preview['accepted']
    assert len(calls)==1
    assert path['_curvilinear_mixed_spent']==0
    np.testing.assert_array_equal(path['best'].triangles,preview['report'].mesh.triangle_connectivity)
    np.testing.assert_array_equal(path['triangulation'].triangles,path['best'].triangles)
    assert preview['triangles_key']==path['best'].triangles.tobytes()
    class Plane:
        def lift(self,points): return np.column_stack((points,np.zeros(len(points))))
    published=surface._prepare_recombined_path(path,Plane(),original.outer_loop,(),settings,None,
                                               physical_evaluator=chart.evaluate)
    assert published['report'] is preview['report'] and len(calls)==1


def test_rejected_early_pair_preview_preserves_consumed_work_for_later_stage(chart_fixture,monkeypatch):
    _model,chart=chart_fixture
    original=surface._curvilinear_mixed_preview; allowances=[]; charged=[]
    def refused(*args,**kwargs):
        allowances.append(args[2].max_recombination_work)
        preview=original(*args,**kwargs)
        charged.append(preview['report'].exchange_work)
        preview['accepted']=False  # exercise refusal scheduling, never admit a fake proof
        return preview
    monkeypatch.setattr(surface,'_curvilinear_mixed_preview',refused)
    path,report,settings,_tri=run_component(chart,monkeypatch)
    assert len(allowances)==2
    assert allowances[1]==settings.max_recombination_work-charged[0]
    assert path['_curvilinear_mixed_spent']==sum(charged)
    assert report['topology_operations']>=7


@pytest.mark.parametrize('callback',(False,True))
def test_forwarder_keeps_old_schedule_but_callback_preserves_early_physical_gate(chart_fixture,monkeypatch,callback):
    _model,chart=chart_fixture
    original=surface._curvilinear_mixed_preview; calls=[]
    def measured(*args,**kwargs):
        calls.append(kwargs.get('topology_remaining'))
        return original(*args,**kwargs)
    monkeypatch.setattr(surface,'_curvilinear_mixed_preview',measured)
    check=(lambda phase:False) if callback else None
    evaluate=chart.evaluate if callback else lambda points:chart.evaluate(points)
    _path,report,_settings,_tri=run_component(chart,monkeypatch,check,evaluate)
    if callback:
        assert calls and calls[0] is not None
        assert 'early_pair_mixed_preview' in report
    else:
        assert calls==[None] and 'early_pair_mixed_preview' not in report


@pytest.mark.parametrize('carried',({'_curvilinear_mixed_spent':1},
    {'_curvilinear_mixed_spent':False},{'_curvilinear_mixed_preview':{'accepted':True}}))
def test_early_pair_rejects_nonfresh_baseline_before_physical_or_preview_work(chart_fixture,monkeypatch,carried):
    from anymesher.errors import MeshError
    _model,chart=chart_fixture
    def forbidden(*args,**kwargs):
        raise AssertionError('nonfresh baseline started physical work')
    monkeypatch.setattr(surface,'_physical_quality_candidate',forbidden)
    monkeypatch.setattr(surface,'_curvilinear_mixed_preview',forbidden)
    with pytest.raises(MeshError,match='fresh baseline'):
        run_component(chart,monkeypatch,baseline_extra=carried)
