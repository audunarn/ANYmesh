"""Independent affine metric and public original-owner adapter contracts."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import anygeometry as owner

from anymesher._authored_metric_chart import AuthoredMetricChart
from anymesher.errors import MeshError


def prepared(cubic=False):
    model = owner.GeometryModel()
    if cubic:
        p = model.add_points(((0,0,0),(1,2,0),(2,-1,0),(3,1,0)))
        edge = model.add_spline(p[0],p[1:-1],p[-1])
        face = model.extrude((edge,),(.25,0,1.5))[0]
    else:
        origin = np.array([2.,-1.,3.]); a = np.array([2.,0.,1.]); b = np.array([1.,3.,2.])
        face = model.add_plate(model.add_points([origin,origin+a,origin+a+b,origin+b]))
        model.set_face_surface(face, owner.Plane(origin,a,b))
    plan = owner.plan_intersections(model,tuple(model.faces),policy='connect')
    owner.apply_intersections(model,plan,policy='connect')
    return model, owner.query_prepared_authored_boundary_correspondence(model,face)


@pytest.fixture(scope='module')
def plane(): return prepared()


def test_nonorthogonal_plane_metric_positions_and_jacobian(plane):
    model,binding = plane; before = owner.to_dict(model)
    chart = AuthoredMetricChart(model,binding)
    uv = np.array([[0.,0.],[.25,.5],[1.2,-.5]])
    # J columns (2,0,1),(1,3,2): exact Gram [[5,4],[4,14]].
    expected_l = np.array([[np.sqrt(5.),0.],[4/np.sqrt(5.),np.sqrt(54/5)]])
    np.testing.assert_allclose(chart.transform,expected_l,rtol=2e-15,atol=0)
    metric = chart.to_metric(uv)
    np.testing.assert_allclose(metric,uv@expected_l,rtol=0,atol=2e-15)
    np.testing.assert_allclose(chart.to_authored_uv(metric),uv,rtol=0,atol=2e-15)
    j = np.array([[2.,1.],[0.,3.],[1.,2.]])
    np.testing.assert_allclose(chart.evaluate(metric),[2,-1,3]+uv@j.T,rtol=0,atol=3e-15)
    expected_j = j@np.linalg.inv(expected_l).T
    np.testing.assert_allclose(chart.jacobians(metric),np.broadcast_to(expected_j,(3,3,2)),rtol=0,atol=2e-15)
    np.testing.assert_allclose(expected_j.T@expected_j,np.eye(2),rtol=0,atol=5e-16)
    assert chart.publication_qualified is False and owner.to_dict(model)==before


def test_cubic_physical_jacobians_vary_away_from_reference():
    model,binding = prepared(cubic=True); chart = AuthoredMetricChart(model,binding)
    uv = np.array([[0.,.25],[.5,.75],[1.25,-.5]])
    t,s = uv.T
    expected = np.stack((3*t+.25*s,6*t-15*t*t+10*t*t*t,1.5*s),axis=-1)
    j = np.zeros((3,3,2)); j[:,0,0]=3; j[:,1,0]=6-30*t+30*t*t
    j[:,0,1]=.25; j[:,2,1]=1.5
    metric = chart.to_metric(uv)
    np.testing.assert_allclose(chart.evaluate(metric),expected,rtol=0,atol=1e-14)
    # The public input here is metric coordinates: roundtrip UV is not
    # bitwise identical to the initial UV. Evaluate the independent derivative
    # at that represented coordinate; affine mapping is checked above.
    represented_t = chart.to_authored_uv(metric)[:,0]
    j[:,1,0] = 6-30*represented_t+30*represented_t*represented_t
    np.testing.assert_allclose(chart.jacobians(metric),j@chart.inverse.T,rtol=0,atol=5e-15)
    actual_j = chart.jacobians(metric)
    assert not np.allclose(actual_j[0].T@actual_j[0],np.eye(2))


@pytest.mark.parametrize('method',['evaluate','jacobians','to_metric','to_authored_uv'])
def test_empty_detached_input_and_output(plane,method):
    model,binding=plane; armed=False; values=np.array([[.25,.5]])
    def callback(_):
        if armed: values[:]=99
    chart=AuthoredMetricChart(model,binding,callback)
    expected=getattr(chart,method)(values.copy()); armed=True
    actual=getattr(chart,method)(values)
    np.testing.assert_array_equal(actual,expected)
    assert not np.shares_memory(actual,values)
    assert getattr(chart,method)(np.empty((0,2))).shape == ((0,3,2) if method=='jacobians' else (0,3) if method=='evaluate' else (0,2))


@pytest.mark.parametrize('rows',[[],[1,2],[[np.nan,0]],[[np.inf,0]],[[1+0j,0]],[[10**400,0]],[['1','2']]])
def test_invalid_rows_refuse_before_callback(plane,rows):
    model,binding=plane; armed=False
    def callback(_):
        if armed: raise AssertionError('invalid rows reached callback')
    chart=AuthoredMetricChart(model,binding,callback); armed=True
    for method in ('evaluate','jacobians','to_metric','to_authored_uv'):
        with pytest.raises(MeshError,match='finite real'): getattr(chart,method)(rows)


def test_transform_is_immutable_and_adapter_attributes_are_frozen(plane):
    chart=AuthoredMetricChart(*plane)
    for array in (chart.transform,chart.inverse):
        with pytest.raises(ValueError): array[0,0]=99
        with pytest.raises(ValueError): array.flags.writeable=True
    with pytest.raises((FrozenInstanceError,AttributeError,TypeError)): chart.transform=np.eye(2)


def test_stale_and_wrong_owner_errors_remain_geometry_errors(plane,monkeypatch):
    model,binding=plane; chart=AuthoredMetricChart(model,binding)
    with pytest.raises(owner.GeometryError): AuthoredMetricChart(model.clone(preserve_identity=True),binding)
    monkeypatch.setattr(model,'_revision',model.revision+1)
    with pytest.raises(owner.GeometryError): chart.evaluate([[0,0]])


@pytest.mark.parametrize('mutation',['context','receipt','source'])
def test_final_callback_cannot_switch_transform_binding_or_source(plane,mutation):
    model,binding=plane; remaining=None; chart=None; old= None
    def callback(_):
        nonlocal remaining
        if remaining is None: return False
        remaining-=1
        if remaining==0:
            if mutation=='context':
                context=chart._context
                object.__setattr__(chart,'_context',(*context[:4],np.eye(2).tobytes(),np.eye(2).tobytes()))
            elif mutation=='receipt': object.__setattr__(binding,'descendants',())
            else: model._revision+=1
        return False
    chart=AuthoredMetricChart(model,binding,callback)
    phases=[]
    # A separate chart records the identical public-call sequence.
    count_chart=AuthoredMetricChart(model,binding,lambda phase:phases.append(phase))
    phases.clear(); count_chart.evaluate([[.25,.5]])
    remaining=len(phases); old=(binding.descendants,model.revision)
    try:
        with pytest.raises((MeshError,owner.GeometryError)): chart.evaluate([[.25,.5]])
        assert remaining==0
    finally:
        object.__setattr__(binding,'descendants',old[0]); model._revision=old[1]


def test_late_cancellation_preserves_exception_identity(plane):
    model,binding=plane; remaining=None
    error=RuntimeError('metric cancellation')
    def callback(_):
        nonlocal remaining
        if remaining is not None:
            remaining-=1
            if remaining==0: raise error
    chart=AuthoredMetricChart(model,binding,callback)
    phases=[]; counter=AuthoredMetricChart(model,binding,lambda phase:phases.append(phase))
    phases.clear(); counter.jacobians([[.25,.5]])
    remaining=len(phases)
    with pytest.raises(RuntimeError) as caught: chart.jacobians([[.25,.5]])
    assert caught.value is error and remaining==0


@pytest.mark.parametrize('bad',['singular','nonfinite','shape','complex'])
def test_invalid_reference_differential_is_mesh_error(plane,monkeypatch,bad):
    import anymesher._authored_metric_chart as module
    du=np.array([[1.,0.,0.]]); dv=du.copy()
    if bad=='nonfinite': du[0,0]=np.nan
    elif bad=='shape': du=du.reshape(3)
    elif bad=='complex': du=du.astype(complex)+1j
    monkeypatch.setattr(module,'evaluate_prepared_authored_face',lambda *args,**kwargs:(du,dv))
    with pytest.raises(MeshError,match='reference differential'): AuthoredMetricChart(*plane)


def test_environment_uses_installed_owner():
    assert 'site-packages' in owner.__file__
    print('installed owner:',owner.__file__)


def test_input_conversion_cannot_replace_entry_chart_context(plane):
    chart = AuthoredMetricChart(*plane)
    class Rows:
        def __array__(self, dtype=None, copy=None):
            context = chart._context
            object.__setattr__(chart, '_context',
                (*context[:4], np.eye(2).tobytes(), np.eye(2).tobytes()))
            return np.array([[.25,.5]], dtype=dtype)
    with pytest.raises(MeshError, match='binding or transform changed'):
        chart.to_metric(Rows())


def test_fragmented_root_chart_keeps_original_support_and_scope():
    model = owner.GeometryModel()
    points = model.add_points(((0,0,0),(1,2,0),(2,-1,0),(3,1,0)))
    edge = model.add_spline(points[0],points[1:-1],points[-1])
    root = model.extrude((edge,),(.25,0,1.5))[0]
    model.add_plate(model.add_points(((-1,-2,.75),(4,-2,.75),(4,3,.75),(-1,3,.75))))
    plan = owner.plan_intersections(model,tuple(model.faces),policy='connect')
    owner.apply_intersections(model,plan,policy='connect')
    binding = owner.query_prepared_authored_boundary_correspondence(model,root)
    assert root not in model.faces and len(binding.descendants)==2
    before = owner.to_dict(model)
    chart = AuthoredMetricChart(model,binding)
    uv = np.array(((0.,.25),(.25,.75),(.75,.25),(1.,.75)))
    t,s = uv.T
    expected = np.column_stack((3*t+.25*s,6*t-15*t*t+10*t*t*t,1.5*s))
    np.testing.assert_allclose(chart.evaluate(chart.to_metric(uv)),expected,rtol=0,atol=1e-14)
    assert owner.to_dict(model)==before and chart.publication_qualified is False
