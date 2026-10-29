"""Real mixed/T3 quality gates; retention never waives the physical policy."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
sm=importlib.import_module('anymesher.surface_mesh')
from anymesher.core import MeshCore
from anymesher.recombine import recombine_triangles_with_report
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher.errors import MeshError,StructuredQualityRejected


def case(right=2.,limit=1.1):
    uv=np.array(((0.,0.),(1.,0.),(right,0.),(0.,1.),(1.,1.),(right,1.)))
    cells=np.array(((0,1,4),(0,4,3),(1,2,5),(1,5,4)),dtype=np.int64)
    def evaluate(rows):return np.column_stack((rows,np.zeros(len(rows))))
    core=MeshCore(evaluate(uv),cells)
    # Protect every edge except the left-square diagonal: only one Q4 can form.
    edges={tuple(sorted((int(a),int(b)))) for cell in cells for a,b in zip(cell,np.roll(cell,-1))}
    protected=np.array(sorted(edges-{(0,4)}),dtype=np.int64)
    report=recombine_triangles_with_report(core,protected_edges=protected)
    assert report.pair_count==1
    return uv,cells,evaluate,report,sm.SurfaceMeshOptions(max_element_growth=limit)


def final_chart():
    # This final scorer reads no owner authority: it only checks this physical
    # MeshCore. No fake material/chord/cell proof is supplied or claimed.
    chart=object.__new__(AnalyticMetricChart);chart.check=None;chart.face_id=1
    return chart


def test_growth_only_mixed_failure_retains_genuinely_admissible_triangles():
    uv,cells,evaluate,report,settings=case()
    mixed_quality=sm._quality_threshold_report(sm.evaluate_quality(report.mesh),settings)
    assert mixed_quality['accepted']
    mixed_growth,poor=sm._published_growth(report.mesh,settings.max_element_growth)
    assert poor and mixed_growth>settings.max_element_growth
    with pytest.raises(StructuredQualityRejected):final_chart().certify_physical_core(report.mesh,settings)
    report=replace(report,exchange_work=17,exchange_count=3,exchange_truncated=True)
    before=uv.tobytes(),cells.tobytes();events=[]
    retained,evidence=sm._qualified_physical_triangle_retention(uv,cells,report,settings,evaluate,events.append)
    assert retained is not None and evidence['retained']
    assert retained.pair_count==0 and len(retained.paired_triangle_ids)==0 and len(retained.scores)==0
    assert retained.exchange_work==17 and retained.exchange_count==3 and retained.exchange_truncated
    assert evidence['discarded_pair_count']==1 and evidence['consumed_exchange_work']==17
    np.testing.assert_array_equal(retained.mesh.triangle_connectivity,cells)
    np.testing.assert_array_equal(retained.mesh.node_coordinates,evaluate(uv))
    assert retained.mesh.num_quads==0 and before==(uv.tobytes(),cells.tobytes())
    final=final_chart().certify_physical_core(retained.mesh,settings)
    assert final['maximum_element_growth']==evidence['triangle_growth']<=settings.max_element_growth
    assert events==['physical triangle retention start','physical triangle retention complete']


def test_both_physical_topologies_fail_without_triangle_waiver():
    uv,cells,evaluate,report,settings=case(right=2.1,limit=1.01)
    retained,evidence=sm._qualified_physical_triangle_retention(uv,cells,report,settings,evaluate,None)
    assert retained is None and not evidence['retained']
    assert evidence['triangle_growth']>settings.max_element_growth
    with pytest.raises(StructuredQualityRejected):final_chart().certify_physical_core(MeshCore(evaluate(uv),cells),settings)
    assert report.pair_count==1


@pytest.mark.parametrize('stage',('owner','start','complete','mutation','policy'))
def test_retention_preserves_owner_cancellation_and_row_mutation_refusal(stage):
    uv,cells,evaluate,report,settings=case();error=RuntimeError('exact owned failure')
    def owner(rows):
        if stage=='owner':raise error
        return evaluate(rows)
    def check(phase):
        if stage in ('start','complete') and phase.endswith(stage):raise error
        if stage=='mutation' and phase.endswith('complete'):uv[0,0]+=.1
        if stage=='policy' and phase.endswith('complete'):object.__setattr__(settings,'max_element_growth',2.)
    if stage in ('mutation','policy'):
        with pytest.raises(MeshError,match='rows or policy changed'):
            sm._qualified_physical_triangle_retention(uv,cells,report,settings,owner,check)
    else:
        with pytest.raises(RuntimeError) as caught:
            sm._qualified_physical_triangle_retention(uv,cells,report,settings,owner,check)
        assert caught.value is error
    assert report.pair_count==1


def selection_row(*,admissible,mandatory,alignment):
    uv,cells,evaluate,report,settings=case()
    return dict(mandatory_physical_quality=mandatory,physical_admissible=admissible,
        quality_eligible=True,alignment_qualified=True,
        outer_alignment={'maximum_normal_error_degrees':alignment,'mean_normal_error_degrees':alignment},
        hole_alignment={'maximum_normal_error_degrees':0.,'mean_normal_error_degrees':0.},
        quality_policy=sm._quality_threshold_report(sm.evaluate_quality(report.mesh),settings),
        path={},core=report.mesh)


def test_mandatory_physical_admission_precedes_alignment_but_generic_order_is_unchanged():
    failed=selection_row(admissible=False,mandatory=True,alignment=0.)
    accepted=selection_row(admissible=True,mandatory=True,alignment=10.)
    assert sm._recombined_path_key(accepted)<sm._recombined_path_key(failed)
    failed['mandatory_physical_quality']=accepted['mandatory_physical_quality']=False
    assert sm._recombined_path_key(failed)<sm._recombined_path_key(accepted)


def test_already_admissible_mixed_candidate_has_no_mandatory_penalty():
    uv,cells,evaluate,report,settings=case(limit=1.5)
    final=final_chart().certify_physical_core(report.mesh,settings)
    assert final['accepted'] and final['maximum_element_growth']<=settings.max_element_growth
    accepted=selection_row(admissible=True,mandatory=True,alignment=0.)
    alternative=selection_row(admissible=True,mandatory=True,alignment=10.)
    assert sm._recombined_path_key(accepted)<sm._recombined_path_key(alternative)
    assert report.pair_count==1


@pytest.mark.parametrize('limit,retained',( (1.1,True),(1.5,False) ))
def test_real_owner_recombined_caller_retains_only_when_mixed_fails(monkeypatch,limit,retained):
    import anygeometry as owner
    from anymesher._material_region_binding import MaterialRegionBinding
    from anymesher.triangulation import PlanarTriangulation
    import anymesher._physical_mixed_repair as mixed
    from test_nurbs_curvilinear_consumer import patch_model
    model,face,edges=patch_model(warp=0.)
    regions=owner.query_material_surface_regions(model,(face,))
    chords=owner.query_prepared_material_boundary_chords(model,regions,
        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=.01)
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,.01)
    binding._curvilinear_state['chords']=chords
    chart=AnalyticMetricChart(model,face,region_binding=binding)
    uv,cells,_,_,settings=case(limit=limit);uv=uv/2
    core=MeshCore(chart.evaluate(uv),cells)
    all_edges={tuple(sorted((int(a),int(b)))) for cell in cells for a,b in zip(cell,np.roll(cell,-1))}
    protected=np.array(sorted(all_edges-{(0,4)}),dtype=np.int64)
    actual_report=recombine_triangles_with_report(core,protected_edges=protected)
    assert actual_report.pair_count==1
    quality=sm._quality_threshold_report(sm.evaluate_quality(actual_report.mesh),settings)
    # Controlled unsuccessful repair proposal; all quality, owner evaluation,
    # selection and retained T3 qualification below are real production calls.
    monkeypatch.setattr(sm,'_qualified_recombination',lambda *args,**kwargs:(actual_report,quality))
    monkeypatch.setattr(mixed,'repair_mixed_coordinates',lambda points,triangles,report,*args,**kwargs:(points,report))
    boundary=np.array(((0,1),(1,2),(2,5),(5,4),(4,3),(3,0)),dtype=np.int64)
    triangulation=PlanarTriangulation(uv,cells,boundary,boundary,np.empty((0,2),dtype=np.int64),
        np.array((0,1,2,5,4,3),dtype=np.int64),())
    candidate=sm._physical_quality_candidate(sm._make_candidate(uv,cells,settings=settings),settings,chart.evaluate)
    path=dict(name='native',best=candidate,triangulation=triangulation)
    plane=sm._Plane(np.zeros(3),np.array((1.,0.,0.)),np.array((0.,1.,0.)),2)
    result=sm._prepare_recombined_path(path,plane,uv[[0,2,5,3]],(),settings,None,chart.evaluate)
    assert result['mandatory_physical_quality'] and result['physical_admissible']
    assert (result['report'].pair_count==0)==retained
    if retained:
        assert result['physical_triangle_retention']['retained']
        assert result['path']['best'] is candidate and result['path']['triangulation'] is triangulation
        np.testing.assert_array_equal(result['core'].triangle_connectivity,cells)
    else:assert result['physical_triangle_retention'] is None
    final=chart.certify_physical_core(result['report'].mesh,settings)
    assert final['maximum_element_growth']<=limit


@pytest.mark.parametrize('stage',('start','complete'))
def test_retention_truthy_stage_cancellation_refuses(stage):
    uv,cells,evaluate,report,settings=case();events=[]
    def cancel(phase):events.append(phase);return phase.endswith(stage)
    with pytest.raises(MeshError,match='cancelled'):
        sm._qualified_physical_triangle_retention(uv,cells,report,settings,evaluate,cancel)
    assert len(events)==(1 if stage=='start' else 2)
    assert report.pair_count==1
