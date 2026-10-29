import pytest
import anygeometry as owner
from anymesher import GeometryApproximationPolicy,resolve_geometry_approximation,generate_hybrid_mesh_result
from anygeometry._local_curvature import LocalApproximationNotCertified
from test_nurbs_curvilinear_consumer import patch_model,options

def test_positive_curved_public_mesh_uses_authenticated_local_receipts(monkeypatch):
    model,face,_=patch_model(warp=.001)
    before=owner.to_dict(model);receipts=[]
    actual=owner.validate_prepared_material_curvilinear_cells
    def observe(*args,**kwargs):
        result=actual(*args,**kwargs);receipts.append(result);return result
    monkeypatch.setattr(owner,'validate_prepared_material_curvilinear_cells',observe)
    request=options();request.pop('geometry_approximation_error')
    result=generate_hybrid_mesh_result(model,**request)
    assert result.mesh.elements_of_face[face]
    assert result.mesh.hybrid_diagnostics['geometry_approximation']['adaptation']=='local'
    assert receipts and all(r.local_field_fingerprint and r.surface_errors_by_cell and r.local_footprints_by_cell for r in receipts)
    assert all(all(value<=r.surface_error for value in r.surface_errors_by_cell) for r in receipts)
    assert owner.to_dict(model)==before

def test_live_original_resolution_is_forwarded_without_child_reresolution():
    model,face,_=patch_model(warp=.001)
    resolution=resolve_geometry_approximation(model,(face,))
    assert resolution._field is not None and resolution.clearance_face_ids==(face,)
    prepared=owner.clone_prepared_geometry(model,new_preparation_epoch=True)
    request=options();request['geometry_approximation_error']=resolution.effective_error
    result=generate_hybrid_mesh_result(prepared,**request,geometry_approximation_resolution=resolution,
        geometry_approximation_source=model)
    assert result.mesh.hybrid_diagnostics['geometry_approximation']['source']['model_id']==str(model.model_id)

def test_local_restriction_below_floor_refuses_instead_of_lifting():
    model,face,_=patch_model(warp=.1)
    with pytest.raises(LocalApproximationNotCertified,match='below'):
        resolve_geometry_approximation(model,(face,),policy=GeometryApproximationPolicy(1e-20))
    scale=resolve_geometry_approximation(model,(face,),policy=GeometryApproximationPolicy(1e-20,adaptation='scale'))
    assert scale.floor_limited and scale._field is None
