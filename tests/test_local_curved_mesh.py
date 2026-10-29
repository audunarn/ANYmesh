import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface,NurbsCurve,CurveOnSurface
from anymesher import generate_hybrid_mesh_result,GeometryApproximationPolicy
from test_nurbs_curvilinear_consumer import options,patch_model


def curved_loop_model():
    model=owner.GeometryModel()
    support=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))
    points=((.75,.5),(.5,.75),(.25,.5),(.5,.25))
    corners=((.75,.75),(.25,.75),(.25,.25),(.75,.25))
    curves=tuple(CurveOnSurface(support,NurbsCurve(2,(0.,0.,0.,1.,1.,1.),
        (points[i],corners[i],points[(i+1)%4]))) for i in range(4))
    vertices=tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges=tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve) for i,curve in enumerate(curves))
    face=model.add_face(edges,surface=support);model.add_sheet((face,))
    return model,face,edges,curves


def test_fixed_scalar_initial_curvature_controls_reach_real_metric_entry(monkeypatch):
    from anymesher.metric import MetricFieldSpec
    model,face,_=patch_model(warp=.25);before=owner.to_dict(model)
    original=MetricFieldSpec.from_size_field.__func__;seen=[]
    class MetricReached(Exception):pass
    def inspect(cls,size_field):
        result=original(cls,size_field)
        if any(control.name=='source curvature proposal' for control in result.feature_controls):
            seen.append(result);raise MetricReached()
        return result
    monkeypatch.setattr(MetricFieldSpec,'from_size_field',classmethod(inspect))
    request=options();requested=request['geometry_approximation_error']
    with pytest.raises(MetricReached):generate_hybrid_mesh_result(model,**request)
    assert seen and request['geometry_approximation_error']==requested
    assert any(control.target_size<request['target_size'] for control in seen[0].feature_controls)
    assert owner.to_dict(model)==before
