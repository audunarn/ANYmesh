"""Explicit graded proposal is separate from actual affine-map qualification."""
import pytest
import anygeometry as owner
from anymesher.errors import MeshError
from test_affine_boundary_graded_metric import binding_case


@pytest.mark.parametrize('kind',('affine','rotated','curved','nonlinear_plane','rational','multispan'))
def test_explicit_graded_nonaffine_proposal_preserves_affinity_and_legacy(kind):
    model,binding=binding_case(kind)
    old=binding.interior_size_proposal(.5)
    actual_affine=binding._has_globally_affine_support()
    assert binding.interior_size_proposal(.5,_boundary_graded=True,_with_affine=True)==(.5 if actual_affine else old,actual_affine)
    proposed,affine,graded=binding.interior_size_proposal(.5,_boundary_graded=True,_with_graded=True)
    assert proposed==.5 and affine is actual_affine and graded
    assert binding.interior_size_proposal(.5,_with_graded=True)==(old,False,False)


def test_nonaffine_graded_callback_failure_is_original_and_source_change_refuses():
    model,binding=binding_case('curved');error=RuntimeError('cancel')
    def cancel(*args):raise error
    with pytest.raises(RuntimeError) as caught:
        binding.interior_size_proposal(.5,cancel,_boundary_graded=True,_with_graded=True)
    assert caught.value is error
    def mutate(*args):model.add_point(4.,5.,6.)
    with pytest.raises((MeshError,owner.GeometryError)):
        binding.interior_size_proposal(.5,mutate,_boundary_graded=True,_with_graded=True)

@pytest.mark.parametrize('route',('canonical','legacy','experimental'))
def test_actual_native_entry_forwards_distinct_proposal_optin(monkeypatch,route):
    import sys
    from dataclasses import replace
    import numpy as np
    import anymesher.hybrid as hybrid
    from anymesher._material_region_binding import MaterialRegionBinding
    from test_local_curved_mesh import curved_loop_model
    from test_nurbs_curvilinear_consumer import options
    model,*_=curved_loop_model();request=options();request['mutation_policy']='read_only'
    if route=='legacy':request['native_options']=replace(request['native_options'],metric_mode='legacy')
    elif route=='experimental':
        from anymesher.metric import ExperimentalMetricProvider
        def foreign(points):raise AssertionError('provider not needed before native entry')
        request['native_options']=replace(request['native_options'],experimental_metric_provider=ExperimentalMetricProvider(foreign))
    calls=[]
    def entered(code,offset):
        frame=sys._getframe(1);calls.append({name:frame.f_locals[name] for name in ('_boundary_graded','_with_affine','_with_graded')})
    class NativeReached(BaseException):pass
    def native(*args,**kwargs):raise NativeReached()
    monkeypatch.setattr(hybrid,'mesh_planar_surface',native)
    tool=next(i for i in range(5,-1,-1) if sys.monitoring.get_tool(i) is None)
    code=MaterialRegionBinding.interior_size_proposal.__code__;sys.monitoring.use_tool_id(tool,'proposal-routing-test')
    sys.monitoring.register_callback(tool,sys.monitoring.events.PY_START,entered);sys.monitoring.set_local_events(tool,code,sys.monitoring.events.PY_START)
    try:
        with pytest.raises(NativeReached):hybrid.generate_hybrid_mesh_result(model,**request)
    finally:sys.monitoring.set_local_events(tool,code,0);sys.monitoring.free_tool_id(tool)
    if route=='legacy':
        # Existing legacy route bypasses the isotropic material-sizing block.
        assert not calls
        return
    assert calls and calls[-1]['_boundary_graded']
    assert calls[-1]['_with_graded'] is (route=='canonical')
    assert calls[-1]['_with_affine'] is (route!='canonical')

