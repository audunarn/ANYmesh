"""Consumer protocol: owner proposals precede, and never replace, real proofs."""
import builtins
from copy import copy
from fractions import Fraction
import pytest
import anygeometry as owner
from test_physical_mixed_repair_batch import chart_fixture
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.errors import MeshError

TARGET='anygeometry.nurbs_surface_intersection._mesh_proposals'


def pair_binding(chart_fixture,outer=False):
    model,chart=chart_fixture
    face=(next(face.id for face in model.faces.values() if face.holes
          and face.surface==model.faces[chart.face_id].surface) if outer else chart.face_id)
    regions=owner.query_material_surface_regions(model,(face,))
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,
        geometry_approximation_error=.0025)
    requested={int(path.source_edge):(0.,.25,.5,.75,1.)
        for loop in binding.region.boundaries for path in loop}
    return model,binding,requested


@pytest.mark.parametrize('numpy_rows',(False,True))
def test_real_owner_proposals_are_submitted_to_unchanged_sampler(chart_fixture,monkeypatch,numpy_rows):
    from importlib import import_module
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
    module=import_module(TARGET)
    original=module.propose_c2_boundary_parameters
    model,binding,requested=pair_binding(chart_fixture,outer=True)
    if numpy_rows:
        import numpy as np
        requested={edge:np.asarray(values,dtype=np.float64) for edge,values in requested.items()}
    before=owner.to_dict(model),repr(requested),dict(binding._curvilinear_state)
    issued={}; actual_edges={id(model.edges[edge].curve):edge for edge in requested}
    def measured(curve,parameters,*,chord_error,cancellation_check=None):
        edge=actual_edges[id(curve)]
        assert type(curve) is NurbsSurfaceIntersectionCurve
        assert tuple(parameters)==tuple(Fraction(float(value)) for value in requested[edge])
        assert all(type(value) is Fraction for value in parameters)
        assert chord_error==.0025/4 and cancellation_check is None
        result=original(curve,parameters,chord_error=chord_error,cancellation_check=cancellation_check)
        assert type(result) is tuple and set(parameters)<=set(result)
        issued[edge]=result
        return result
    monkeypatch.setattr(module,'propose_c2_boundary_parameters',measured)
    failure=RuntimeError('observed real sampler frontier; no proof issued')
    def observe(model_arg,collection,intervals,**kwargs):
        assert model_arg is model and collection is binding.collection
        assert kwargs['chord_error']==.0025/4
        submitted={handle.id:tuple(parameters) for handle,parameters in intervals}
        for edge,values in submitted.items():
            expected=issued.get(edge,requested[edge])
            assert values==tuple(Fraction(float(value)) for value in expected)
        assert issued and len(issued)<len(requested)  # literal COS stays unchanged
        raise failure
    monkeypatch.setattr(owner,'query_prepared_material_boundary_sampling',observe)
    with pytest.raises(RuntimeError) as caught:
        binding.prepare_curvilinear_seed(requested)
    assert caught.value is failure
    assert before==(owner.to_dict(model),repr(requested),binding._curvilinear_state)


@pytest.mark.parametrize('missing',(TARGET,'unrelated_owner_dependency'))
def test_only_exact_optional_module_absence_falls_back(chart_fixture,monkeypatch,missing):
    _model,binding,requested=pair_binding(chart_fixture)
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name==TARGET:
            raise ModuleNotFoundError('deliberate missing dependency',name=missing)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',importing)
    if missing==TARGET:
        assert binding._propose_pair_boundary_parameters(requested,.0025/4) is requested
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            binding._propose_pair_boundary_parameters(requested,.0025/4)
        assert caught.value.name==missing


def test_legacy_proposal_route_never_imports_new_owner_module(monkeypatch):
    from test_nurbs_curvilinear_consumer import patch_model
    model,face,edges=patch_model()
    regions=owner.query_material_surface_regions(model,(face,))
    binding=MaterialRegionBinding(model,regions,regions.regions[0],face,
        geometry_approximation_error=.01)
    requested={edge:(0.,.5,1.) for edge in edges}
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name==TARGET:
            raise AssertionError('legacy proposal imported new module')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',importing)
    assert binding._propose_pair_boundary_parameters(requested,.01) is requested


def test_unissued_binding_refuses_before_owner_proposal(chart_fixture):
    _model,binding,requested=pair_binding(chart_fixture)
    forged=copy(binding)
    with pytest.raises(MeshError,match='construction authority'):
        forged._propose_pair_boundary_parameters(requested,.0025/4)
