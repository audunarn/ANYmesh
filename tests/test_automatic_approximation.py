from dataclasses import replace
import pytest
import anygeometry as owner
from anymesher import (GeometryApproximationPolicy, resolve_geometry_approximation,
    validate_geometry_approximation_binding)
from anymesher.errors import MeshError
from test_nurbs_curvilinear_consumer import patch_model

@pytest.mark.parametrize('value',[True,0,-1,float('nan'),float('inf'),'0.1'])
@pytest.mark.parametrize('field',['relative_accuracy','maximum_error'])
def test_invalid_policy(field,value):
    with pytest.raises(MeshError): GeometryApproximationPolicy(**{field:value})

def test_policy_serialization_and_ambiguity():
    policy=GeometryApproximationPolicy(.002,.1)
    assert GeometryApproximationPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(MeshError): GeometryApproximationPolicy.from_dict({'unknown':1})
    model,face,_=patch_model()
    with pytest.raises(MeshError,match='mutually exclusive'):
        resolve_geometry_approximation(model,(face,),geometry_approximation_error=.01,policy=policy)

def test_real_owner_resolution_caps_floors_and_stale_binding():
    model,face,_=patch_model()
    result=resolve_geometry_approximation(model,(face,))
    assert result.mode == 'automatic'
    assert result.relative_accuracy == .001
    assert result.effective_error == max(.001*result.reference_scale,result.policy_floor)
    assert result.to_dict()['scale_definition'] == 'minimum_trim_enclosure_span'
    validate_geometry_approximation_binding(model,result)
    capped=resolve_geometry_approximation(model,(face,),policy=GeometryApproximationPolicy(.001,result.effective_error/2))
    assert capped.cap_limited and capped.effective_error == result.effective_error/2
    floor=resolve_geometry_approximation(model,(face,),policy=GeometryApproximationPolicy(1e-15,adaptation="scale"))
    assert floor.floor_limited and floor.effective_error == floor.policy_floor
    with pytest.raises(MeshError,match='floor'):
        resolve_geometry_approximation(model,(face,),policy=GeometryApproximationPolicy(maximum_error=result.policy_floor/2))
    with pytest.raises(MeshError,match='changed'):
        validate_geometry_approximation_binding(model,replace(result,effective_error=result.effective_error*2))
    explicit=resolve_geometry_approximation(model,(face,),geometry_approximation_error=result.policy_floor/2)
    assert explicit.mode == 'explicit' and explicit.effective_error == result.policy_floor/2
    model.add_point(20,20,20)
    with pytest.raises(owner.GeometryError): validate_geometry_approximation_binding(model,result)

def test_plain_faces_not_applicable_and_selected_scope():
    model,face,_=patch_model()
    plain=model.add_plate(model.add_points(((10,0,0),(20,0,0),(20,10,0),(10,10,0))))
    assert resolve_geometry_approximation(model,(plain,)).mode == 'not_applicable'
    result=resolve_geometry_approximation(model,(face,plain))
    assert result.face_ids == (face,)

def test_resolution_callback_cancellation():
    model,face,_=patch_model()
    with pytest.raises(owner.GeometryError):
        resolve_geometry_approximation(model,(face,),cancellation_check=lambda phase: True)

def scaled_patch(size, shift=0., crop=1.):
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
    model=owner.GeometryModel()
    support=NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((shift,0.,0.),(shift,size,0.)),((shift+2*size,0.,0.),(shift+2*size,size,0.))))
    corners=((0.,0.),(crop,0.),(crop,crop),(0.,crop))
    curves=tuple(CurveOnSurface(support,NurbsCurve(1,(0.,0.,1.,1.),
        (corners[i],corners[(i+1)%4]))) for i in range(4))
    points=tuple(model.add_point(*c.exact_point(0)) for c in curves)
    edges=tuple(model.add_curve(points[i],points[(i+1)%4],c) for i,c in enumerate(curves))
    face=model.add_face(edges,surface=support); model.add_sheet((face,))
    return model,face

@pytest.mark.parametrize('size,shift',[(1.,0.),(10.,0.),(.01,0.),(1.,1000.)])
def test_relative_branch_scales_with_selected_geometry(size,shift):
    import math
    model,face=scaled_patch(size,shift)
    result=resolve_geometry_approximation(model,(face,))
    assert result.reference_scale == pytest.approx(math.sqrt(5)*size)
    assert result.effective_error == pytest.approx(.001*math.sqrt(5)*size)
    assert not result.floor_limited and not result.cap_limited

@pytest.mark.parametrize('size',[1.,10.])
def test_fixed_cap_intentionally_breaks_proportionality(size):
    model,face=scaled_patch(size)
    result=resolve_geometry_approximation(model,(face,),policy={'maximum_error':1e-4})
    assert result.effective_error == 1e-4 and result.cap_limited

def test_public_policy_rejects_ambiguity_before_generation():
    from anymesher import generate_hybrid_mesh_result
    model,face=scaled_patch(1.)
    with pytest.raises(MeshError,match='mutually exclusive'):
        generate_hybrid_mesh_result(model,target_size=.5,face_ids=(face,),
            geometry_approximation_error=.01,geometry_approximation_policy=GeometryApproximationPolicy())

def test_tiny_trim_uses_boundary_enclosure_not_full_support_net():
    import math
    model,face=scaled_patch(100.,crop=.001)
    result=resolve_geometry_approximation(model,(face,))
    assert result.reference_scale == pytest.approx(math.sqrt(5)*.1)

def test_public_wrapper_revalidates_source_after_generation():
    from anymesher.approximation_policy import _resolve_public_approximation
    model,face=scaled_patch(1.)
    @_resolve_public_approximation
    def generation(geometry, **options):
        assert options['geometry_approximation_error'] > 0
        geometry.add_point(20,20,20)
        return object()
    with pytest.raises(owner.GeometryError): generation(model,face_ids=(face,))

def test_public_wrapper_preserves_callback_exception():
    from anymesher.approximation_policy import _resolve_public_approximation
    model,face=scaled_patch(1.)
    failure=RuntimeError('caller cancellation failure')
    def callback(phase): raise failure
    @_resolve_public_approximation
    def generation(geometry, **options): raise AssertionError('must not generate')
    with pytest.raises(RuntimeError) as caught:
        generation(model,face_ids=(face,),cancellation_check=callback)
    assert caught.value is failure



def test_connect_intent_is_explicit_source_context_and_fixed_route_unchanged():
    model,face,_=patch_model(warp=.001)
    ordinary=resolve_geometry_approximation(model,(face,))
    assert ordinary.connect_intent is False and ordinary._field.contact_intent is False
    planned=resolve_geometry_approximation(model,(face,),connect_intent=True,
        policy=GeometryApproximationPolicy(maximum_error=.0004))
    assert planned.connect_intent and planned._field.contact_intent
    assert planned.to_dict()['local_field']['contact_intent'] is True
    assert planned.effective_error<=.0004
    with pytest.raises(MeshError,match='intent'):
        validate_geometry_approximation_binding(model,replace(planned,connect_intent=False))
    fixed=resolve_geometry_approximation(model,(face,),geometry_approximation_error=.0025,connect_intent=True)
    assert fixed.mode=='explicit' and fixed.effective_error==.0025 and fixed._field is None
    assert fixed.connect_intent is False
    with pytest.raises(MeshError,match='boolean'):
        resolve_geometry_approximation(model,(face,),connect_intent='connect')
