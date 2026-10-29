"""First-attempt owner route filters only optional generated rows."""
import numpy as np
import pytest
import anygeometry as owner
from anymesher import generate_hybrid_mesh_result
from test_local_curved_mesh import curved_loop_model
from test_nurbs_curvilinear_consumer import options
from importlib import import_module
surface=import_module('anymesher.surface_mesh')


def test_first_owner_attempt_uses_only_mandatory_rows_before_actual_native(monkeypatch):
    model,_,_,_=curved_loop_model();before=owner.to_dict(model);seen=[]
    original=surface._run_quality_path
    class Reached(BaseException):pass
    def no_lattice(*args,**kwargs):
        raise AssertionError('owner frontal route generated optional lattice')
    def no_midpoints(*args,**kwargs):
        raise AssertionError('owner frontal route recreated midpoint seeds')
    def entry(name,outer,holes,constraints,explicit,generated,settings,check,**kwargs):
        assert name=='staggered_chart' and kwargs['protect_generated_seed_disks'] is True
        assert kwargs['allow_refinement'] is False
        assert kwargs['protected_node_ids'] and len(generated)==0
        result=original(name,outer,holes,constraints,explicit,generated,settings,check,**kwargs)
        seen.append(result)
        return result
    def native_entry(triangulation,*args,**kwargs):
        path=seen[0]
        assert path['attempted_added_points']==path['attempted_rounds']==0
        assert path['point_budget']==0
        np.testing.assert_array_equal(triangulation.points,path['best'].points)
        assert triangulation.protected_node_rows
        raise Reached()
    monkeypatch.setattr(surface,'_target_points',no_lattice)
    monkeypatch.setattr(surface,'_refinement_midpoints',no_midpoints)
    monkeypatch.setattr(surface,'_run_quality_path',entry)
    monkeypatch.setattr(surface,'frontal_delaunay_refine',native_entry)
    request=options();request.pop('geometry_approximation_error')
    with pytest.raises(Reached):generate_hybrid_mesh_result(model,**request)
    assert owner.to_dict(model)==before


def test_plain_public_route_keeps_existing_seed_behavior(monkeypatch):
    from anymesher.native_v2 import NativeMeshingOptions
    class Reached(BaseException):pass
    def entry(name,outer,holes,constraints,explicit,generated,settings,check,**kwargs):
        assert not kwargs.get('protect_generated_seed_disks',False)
        assert len(generated)>0
        raise Reached()
    monkeypatch.setattr(surface,'_run_quality_path',entry)
    with pytest.raises(Reached):
        surface.mesh_planar_surface(((0.,0.),(1.,0.),(1.,1.),(0.,1.)),
            target_size=.25,backend='python',native_options=NativeMeshingOptions(
                point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))


def test_owner_interior_retry_changes_physical_metric_without_optional_lattice(monkeypatch):
    from test_nurbs_curvilinear_consumer import patch_model
    from anymesher._material_region_binding import _SourceCellResolutionRequired
    from importlib import import_module
    hybrid=import_module('anymesher.hybrid')
    model,face,edges=patch_model();before=owner.to_dict(model);seen=[]
    class Reached(BaseException):pass
    def entry(*args,**kwargs):
        assert kwargs['_mandatory_only_frontal_seed'] is True
        native=kwargs.get('native_options')
        if native is None:native=kwargs['options'].native_options
        seen.append(native.metric_field.global_control.target_size)
        if len(seen)==1:
            raise _SourceCellResolutionRequired('interior proposal requires refinement')
        raise Reached()
    monkeypatch.setattr(hybrid,'mesh_planar_surface',entry)
    request=options()
    with pytest.raises(Reached):generate_hybrid_mesh_result(model,**request)
    assert len(seen)==2 and seen[1] < seen[0]
    assert owner.to_dict(model)==before
    assert request['geometry_approximation_error']==.01
