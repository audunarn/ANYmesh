from dataclasses import replace


import numpy as np


import pytest


import anygeometry as owner


from anymesher.errors import MeshError


from anymesher._material_region_binding import (MaterialRegionBinding,material_read_scope,


    _material_read_view,_material_scope_backend)


from anymesher._analytic_metric_chart import AnalyticMetricChart


from anymesher.native_v2 import NativeMeshingOptions


from anymesher.refinement import SizeField


from test_nurbs_curvilinear_consumer import patch_model, options








def fixture():


    model,face,edges=patch_model(warp=.02)


    regions=owner.query_material_surface_regions(model,(face,))


    chords=owner.query_prepared_material_boundary_chords(model,regions,


        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=.01)


    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,.01)


    binding._curvilinear_state['chords']=chords


    opts=dict(native_options=NativeMeshingOptions(),native_backend='python',


        size_field=SizeField(model,.5),cancellation_check=None)


    return model,face,binding,opts








def test_real_analytic_chart_uses_detached_view_and_closes():


    model,face,binding,opts=fixture()


    chart=AnalyticMetricChart(model,face,region_binding=binding)


    points=np.array([[.2,.3],[.7,.6]]) @ chart.transform


    expected=chart.evaluate(points); derivatives=chart.jacobians(points)


    with material_read_scope(binding,opts) as state:


        np.testing.assert_array_equal(chart.evaluate(points),expected)


        np.testing.assert_array_equal(chart.jacobians(points),derivatives)


        view=state['view']


        assert view is not None and view.evaluations==2


    assert state['evaluations']==2 and _material_read_view(binding) is None


    with pytest.raises(owner.GeometryError,match='closed'):view.evaluate([[.5,.5]])








def test_failed_attempt_preserves_exception_and_closes():


    model,face,binding,opts=fixture()


    token=RuntimeError('original failure')


    with pytest.raises(RuntimeError) as caught:


        with material_read_scope(binding,opts) as state:


            view=_material_read_view(binding)


            raise token


    assert caught.value is token


    with pytest.raises(owner.GeometryError,match='closed'):view.evaluate([[.5,.5]])








def test_raw_source_mutation_cannot_pass_successful_scope_exit():


    model,face,binding,opts=fixture()


    with pytest.raises((owner.GeometryError,MeshError)):


        with material_read_scope(binding,opts):


            view=_material_read_view(binding)


            expected=view.evaluate([[.3,.4]])


            object.__setattr__(binding.region.support,'u_range',(.1,.9))


            np.testing.assert_array_equal(view.evaluate([[.3,.4]]),expected)








def test_callback_and_unknown_nested_control_keep_original_route():


    model,face,binding,opts=fixture()


    callback=lambda phase:False


    assert _material_scope_backend(dict(opts,cancellation_check=callback)) is None


    from anymesher.metric import MetricFieldSpec,IsotropicMetricControl


    class Custom(IsotropicMetricControl):pass


    metric=MetricFieldSpec(Custom(.5))


    assert _material_scope_backend(dict(opts,native_options=replace(options()['native_options'],metric_field=metric))) is None


    with material_read_scope(binding,dict(opts,cancellation_check=callback)) as state:


        assert state is None and _material_read_view(binding) is None








def test_patched_evaluator_keeps_original_route(monkeypatch):


    model,face,binding,opts=fixture()


    from anygeometry.nurbs import NurbsSurface


    original=NurbsSurface.evaluate_many


    monkeypatch.setattr(NurbsSurface,'evaluate_many',lambda self,uv:original(self,uv))


    assert _material_scope_backend(opts) is None








def test_actual_public_mesh_uses_scoped_material_evaluation(monkeypatch):


    from anygeometry._detached_geometry_read import MaterialNumericalRead


    from anymesher.hybrid import generate_hybrid_mesh_result


    model,face,_=patch_model(warp=.001)


    original=MaterialNumericalRead.evaluate; seen=[]


    def record(self,*args,**kwargs):


        result=original(self,*args,**kwargs);seen.append(self);return result


    monkeypatch.setattr(MaterialNumericalRead,'evaluate',record)


    result=generate_hybrid_mesh_result(model,**options())


    assert result.mesh.elements_of_face[face] and seen


    assert all(not view._active for view in seen)
