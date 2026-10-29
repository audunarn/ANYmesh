import contextvars
import sys
import threading
import numpy as np
import pytest
import anygeometry as owner
from anymesher.errors import MeshError
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher._material_region_binding import material_read_scope,_material_read_view,_MATERIAL_READ_SCOPE
from test_material_numerical_read import fixture


def chart_fixture():
    model,face,binding,options=fixture()
    chart=AnalyticMetricChart(model,face,region_binding=binding)
    points=np.array(((.2,.3),(.7,.6)))@chart.transform
    return model,binding,options,chart,points


def test_snapshot_repeats_exact_numerics_without_construction_fingerprints():
    model,binding,options,chart,points=chart_fixture()
    expected=chart.evaluate(points);derivatives=chart.jacobians(points)
    code=owner.chart_definition_fingerprint.__code__;calls=[]
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is code:calls.append(1)
    with material_read_scope(binding,options):
        chart.evaluate(points)
        sys.setprofile(profile)
        try:
            np.testing.assert_array_equal(chart.evaluate(points),expected)
            np.testing.assert_array_equal(chart.jacobians(points),derivatives)
            binding._require_construction()
        finally:sys.setprofile(None)
        assert calls==[]


@pytest.mark.parametrize('mutation',('construction','source','inverse'))
def test_live_nested_mutation_uses_snapshot_then_refuses_before_scope_returns(mutation):
    model,binding,options,chart,points=chart_fixture();expected=chart.evaluate(points)
    with pytest.raises((owner.GeometryError,MeshError)):
        with material_read_scope(binding,options):
            chart.evaluate(points)
            if mutation=='construction':object.__setattr__(binding.region,'material_area',binding.region.material_area*2.)
            elif mutation=='source':model._units='mm'
            else:chart.inverse[0,0]*=1.2
            np.testing.assert_array_equal(chart.evaluate(points),expected)


def test_rebinding_refuses_and_rebound_close_cannot_mask_original_exception(monkeypatch):
    from anygeometry._detached_geometry_read import MaterialNumericalRead
    model,binding,options,chart,points=chart_fixture();error=RuntimeError('original attempt failure');initial=_MATERIAL_READ_SCOPE.get()
    with pytest.raises(RuntimeError) as caught:
        with material_read_scope(binding,options) as state:
            chart.evaluate(points);view=state['view']
            monkeypatch.setattr(MaterialNumericalRead,'close',lambda self:(_ for _ in ()).throw(AssertionError('untrusted close')))
            raise error
    assert caught.value is error and _MATERIAL_READ_SCOPE.get() is initial and not view._active


def test_nested_and_copied_thread_context_keep_ownership():
    model,binding,options,chart,points=chart_fixture();failures=[]
    with material_read_scope(binding,options) as outer:
        chart.evaluate(points);outer_view=outer['view']
        with material_read_scope(binding,options) as inner:
            chart.evaluate(points);assert inner['view'] is not outer_view
        assert not inner['view']._active and outer_view._active
        context=contextvars.copy_context()
        def read():
            try:context.run(chart.evaluate,points)
            except BaseException as error:failures.append(error)
        thread=threading.Thread(target=read);thread.start();thread.join()
        assert len(failures)==1 and isinstance(failures[0],MeshError)
        chart.evaluate(points)
    assert not outer_view._active


def test_changed_numeric_implementation_refuses_before_delegate(monkeypatch):
    from anygeometry._detached_geometry_read import MaterialNumericalRead
    model,binding,options,chart,points=chart_fixture();called=[]
    with pytest.raises(MeshError):
        with material_read_scope(binding,options):
            chart.evaluate(points)
            monkeypatch.setattr(MaterialNumericalRead,'evaluate',lambda *args,**kwargs:called.append(1))
            chart.evaluate(points)
    assert called==[]


@pytest.mark.parametrize('raise_original',(False,True))
def test_rebound_chart_check_refuses_without_callback_and_preserves_exception(raise_original):
    model,binding,options,chart,points=chart_fixture();called=[];error=RuntimeError('original error')
    with pytest.raises(RuntimeError if raise_original else MeshError) as caught:
        with material_read_scope(binding,options):
            chart.evaluate(points)
            chart.check=lambda phase:called.append(phase)
            if raise_original:raise error
            chart.evaluate(points)
    assert called==[]
    if raise_original:assert caught.value is error


def test_saved_chart_operands_are_genuinely_immutable():
    from anymesher._material_region_binding import _material_chart_snapshot
    model,binding,options,chart,points=chart_fixture();expected=chart.evaluate(points)
    with material_read_scope(binding,options):
        chart.evaluate(points);saved=_material_chart_snapshot(chart)
        with pytest.raises(TypeError):saved['inverse']=np.eye(2)*2.
        with pytest.raises(ValueError):saved['inverse'].setflags(write=True)
        with pytest.raises(ValueError):saved['transform'].setflags(write=True)
        np.testing.assert_array_equal(chart.evaluate(points),expected)


def test_unknown_native_driver_mutation_uses_strict_read_before_observer(monkeypatch):
    from anymesher import surface_mesh
    model,binding,options,chart,points=chart_fixture();received=[]
    def wrapper(*args,**kwargs):
        model._units='mm'
        received.append(chart.evaluate(points))
    monkeypatch.setattr(surface_mesh,'frontal_delaunay_refine',wrapper)
    with material_read_scope(binding,options) as state:
        assert state is None
        with pytest.raises((owner.GeometryError,MeshError)):
            surface_mesh.frontal_delaunay_refine()
        assert received==[]


@pytest.mark.parametrize('mutation',['wrapper','code','defaults','kwdefaults'])
def test_mid_scope_driver_change_refuses_before_driver_without_chart_read(monkeypatch,mutation):
    from anymesher import surface_mesh
    from test_physical_mixed_sizing import caller_operands
    from anymesher.surface_mesh import SurfaceMeshOptions
    model,binding,options,chart,points=chart_fixture();settings=SurfaceMeshOptions(target_size=.25)
    baseline,_=caller_operands(chart,settings);entered=[];original=surface_mesh.frontal_delaunay_refine
    def unknown(*args,**kwargs):entered.append(True);raise AssertionError('unknown driver entered')
    def changed_code(*args,**kwargs):raise AssertionError('changed driver code entered')
    with pytest.raises(MeshError,match='material snapshot identity'):
        with material_read_scope(binding,options):
            assert _material_read_view(binding) is not None
            if mutation=='wrapper':monkeypatch.setattr(surface_mesh,'frontal_delaunay_refine',unknown)
            elif mutation=='code':monkeypatch.setattr(original,'__code__',changed_code.__code__)
            elif mutation=='defaults':monkeypatch.setattr(original,'__defaults__',(None,))
            else:monkeypatch.setattr(original,'__kwdefaults__',dict(original.__kwdefaults__,qualified_seed=True))
            surface_mesh._run_frontal_quality_path(baseline,settings,None,model_uuid=None,geometry_revision=None,metric_to_physical=None,metric_jacobian=None,automatically_seeded_shared_segments=None,component_seed_registry=None,supplemental_metric_field=None)
    assert entered==[]


@pytest.mark.parametrize('active',[False,True])
def test_driver_pin_helper_replacement_never_enters_observer(monkeypatch,active):
    from anymesher import _material_region_binding as module
    model,binding,options,chart,points=chart_fixture();entered=[];original=module._material_frontal_driver_pin
    def observer():entered.append(True);return original()
    if not active:
        monkeypatch.setattr(module,'_material_frontal_driver_pin',observer)
        with material_read_scope(binding,options) as state:assert state is None
    else:
        with pytest.raises(MeshError,match='material snapshot identity'):
            with material_read_scope(binding,options):
                monkeypatch.setattr(module,'_material_frontal_driver_pin',observer)
                chart.evaluate(points)
    assert entered==[]


def test_dispatch_guard_replacement_is_not_invoked(monkeypatch):
    from anymesher import _material_region_binding as module,surface_mesh
    from test_physical_mixed_sizing import caller_operands
    from anymesher.surface_mesh import SurfaceMeshOptions
    model,binding,options,chart,points=chart_fixture();settings=SurfaceMeshOptions(target_size=.25);baseline,_=caller_operands(chart,settings);entered=[]
    def observer():entered.append(True)
    with pytest.raises(MeshError,match='material native dispatch guard changed'):
        with material_read_scope(binding,options):
            monkeypatch.setattr(module,'_material_native_dispatch_guard',observer)
            surface_mesh._run_frontal_quality_path(baseline,settings,None,model_uuid=None,geometry_revision=None,metric_to_physical=None,metric_jacobian=None,automatically_seeded_shared_segments=None,component_seed_registry=None,supplemental_metric_field=None)
    assert entered==[]
