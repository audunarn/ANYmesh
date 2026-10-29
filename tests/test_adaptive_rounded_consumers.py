import pytest
from anymesher._material_region_binding import _SourceCellResolutionRequired,MaterialRegionBinding
from anymesher._analytic_metric_chart import AnalyticMetricChart
from test_source_cell_resolution import state,assert_restored,provisional_mutation
from test_local_curved_mesh import curved_loop_model
import anygeometry as owner
from importlib import import_module
surface=import_module('anymesher.surface_mesh')


def test_owner_boundary_limited_refusal_rolls_back_and_escapes_local_retry(monkeypatch):
    from anymesher import hybrid
    mesh,options,aliases=state();calls=[]
    error=_SourceCellResolutionRequired('boundary needs shared station refinement')
    error.boundary_intervals=((0,0,True),)
    error.boundary_failure_kind="boundary_limited"
    def attempt(*args,**kwargs):
        calls.append(1);provisional_mutation(mesh,options);raise error
    monkeypatch.setattr(hybrid,'_mesh_native_face_once',attempt)
    with pytest.raises(_SourceCellResolutionRequired) as caught:hybrid._mesh_native_face(None,mesh,7,**options)
    assert caught.value is error and calls==[1]
    assert_restored(mesh,options,aliases)


def test_rounded_chart_preview_eligibility_is_separate_from_nn_batching():
    from anymesher._physical_mixed_repair import _batch_chart
    model,face,edges,curves=curved_loop_model()
    regions=owner.query_material_surface_regions(model,(face,))
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=.0025)
    sampling=owner.query_prepared_material_boundary_sampling(model,regions,
        tuple((model.handle('edge',edge),(0.,1.)) for edge in edges),chord_error=.0025,
        error_measure='parameterized',proposal_fraction=.25)
    binding._curvilinear_state['chords']=sampling.chords
    binding._curvilinear_state['sampling']=sampling
    chart=AnalyticMetricChart(model,face,region_binding=binding)
    assert _batch_chart(chart.evaluate,None) is None
    assert surface._analytic_preview_chart(chart.evaluate) is chart
    assert surface._analytic_preview_chart(lambda rows:chart.evaluate(rows)) is None


def test_boundary_incidence_with_interior_only_refusal_keeps_local_retry(monkeypatch):
    from anymesher import hybrid
    mesh,options,aliases=state();calls=[]
    error=_SourceCellResolutionRequired('actual owner interior witness')
    error.boundary_intervals=((0,0,True),);error.boundary_failure_kind='interior_only'
    def attempt(*args,**kwargs):
        calls.append(1);assert_restored(mesh,options,aliases)
        if len(calls)==1:provisional_mutation(mesh,options);raise error
        return {}
    monkeypatch.setattr(hybrid,'_mesh_native_face_once',attempt)
    result=hybrid._mesh_native_face(None,mesh,7,**options)
    assert len(calls)==2 and result['source_cell_resolution']['refinement_retries']==1
