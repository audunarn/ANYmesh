"""Owner-derived spacing proposals, not physical-growth or source certificates."""
import numpy as np
import pytest
import anygeometry as owner
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.metric import (MetricFieldSpec, IsotropicMetricControl,
    FeatureDistanceMetricControl, SpatialMetricField)
from test_local_curved_mesh import curved_loop_model


@pytest.fixture
def bound():
    model,face,edges,_=curved_loop_model()
    regions=owner.query_material_surface_regions(model,(face,))
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,
        geometry_approximation_error=.0025)
    sampling=owner.query_prepared_material_boundary_sampling(model,regions,
        tuple((model.handle('edge',edge),(0.,1.)) for edge in edges),
        chord_error=.0025,error_measure='parameterized',proposal_fraction=.25)
    binding._curvilinear_state.update(chords=sampling.chords,sampling=sampling)
    return model,binding,sampling.chords


@pytest.mark.parametrize('cap',(.5,.001))
def test_current_chords_refine_only_preserve_controls_and_cover_segments(bound,cap):
    model,binding,chords=bound;before=owner.to_dict(model)
    user=FeatureDistanceMetricControl(((2.,2.,0.),),.0005,.1,1.25,'user')
    spec=MetricFieldSpec(IsotropicMetricControl(cap),feature_controls=(user,),maximum_gradation=1.25)
    result=binding.boundary_size_metric(spec)
    assert result.global_control is spec.global_control
    assert user in result.feature_controls and result.maximum_gradation==1.25
    assert result.imported_samples==spec.imported_samples
    added=[c for c in result.feature_controls if c.name=='owner chord spacing proposal']
    assert all(c.target_size<=cap and c.growth==1.25 for c in added)
    query_rows=[];limits=[]
    for interval in chords.intervals:
        a,b=np.asarray(interval.xyz,float);length=float(np.linalg.norm(b-a))
        if length<cap:
            matching=[c for c in added if c.target_size==length
                and c.influence_distance==length*.5
                and tuple(a+(b-a)*.5) in c.feature_points]
            assert matching
        query_rows.extend((a,(a+b)*.5,b));limits.extend((min(cap,length),)*3)
    tensors=SpatialMetricField(result).evaluate(np.asarray(query_rows))
    sizes=1/np.sqrt(tensors[:,0,0])
    assert np.all(sizes<=np.asarray(limits)*(1+1e-12))
    assert owner.to_dict(model)==before
    assert binding._curvilinear_state['chords'] is chords


def test_owner_receipt_mutation_during_callback_refuses(bound):
    _,binding,chords=bound;old=chords.chord_error;calls=[]
    def mutate(stage):
        calls.append(stage);object.__setattr__(chords,'chord_error',old*2)
    try:
        with pytest.raises(owner.GeometryError):binding.boundary_size_metric(MetricFieldSpec.uniform(.5),mutate)
        assert calls
    finally:object.__setattr__(chords,'chord_error',old)


def test_external_callback_exception_identity_is_preserved(bound):
    _,binding,_=bound
    class Stop(Exception):pass
    error=Stop('cancel')
    def cancel(stage):raise error
    with pytest.raises(Stop) as caught:
        binding.boundary_size_metric(MetricFieldSpec.uniform(.5),cancel)
    assert caught.value is error


def test_composed_chord_metric_reaches_actual_native_entry(monkeypatch):
    from anymesher import hybrid,generate_hybrid_mesh_result
    from test_nurbs_curvilinear_consumer import options
    model,_,_,_=curved_loop_model();before=owner.to_dict(model);seen=[]
    class Reached(Exception):pass
    def inspect(*args,**kwargs):
        spec=kwargs['native_options'].metric_field
        controls=[c for c in spec.feature_controls if c.name=='owner chord spacing proposal']
        assert controls and all(c.target_size<=spec.global_control.target_size for c in controls)
        assert all(c.growth==spec.maximum_gradation for c in controls)
        seen.append(spec);raise Reached()
    monkeypatch.setattr(hybrid,'mesh_planar_surface',inspect)
    request=options();request.pop('geometry_approximation_error')
    with pytest.raises(Reached):generate_hybrid_mesh_result(model,**request)
    assert len(seen)==1 and owner.to_dict(model)==before
