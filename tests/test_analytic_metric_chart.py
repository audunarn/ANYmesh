import numpy as np
import pytest
from anygeometry import GeometryModel, GeometryError, to_dict
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher.errors import MeshError


def test_polynomial_chart_evaluation_derivatives_and_source_binding():
    model=GeometryModel()
    p=model.add_points(((0,0,0),(.5,0,0),(1,1,0)))
    e=model.add_spline(p[0],p[1:-1],p[-1])
    model.extrude((e,),(0,0,2))
    face=next(iter(model.faces))
    from anygeometry import ExtrudedSurface, BezierDirectrix
    model.set_face_surface(face,ExtrudedSurface(BezierDirectrix(((0,0,0),(.5,0,0),(1,1,0))),(0,0,2)))
    before=to_dict(model)
    chart=AnalyticMetricChart(model,face)
    rows=np.array(((.2,.3),(.8,.7)))
    expected=np.column_stack((rows[:,0],rows[:,0]**2,2*rows[:,1]))
    np.testing.assert_allclose(chart.evaluate(rows @ chart.transform),expected,rtol=1e-14,atol=1e-14)
    jac=chart.jacobians(rows @ chart.transform) @ chart.transform.T
    np.testing.assert_allclose(jac[:,:,0],np.column_stack((np.ones(2),2*rows[:,0],np.zeros(2))))
    np.testing.assert_allclose(jac[:,:,1],((0,0,2),(0,0,2)))
    assert to_dict(model)==before
    with pytest.raises(MeshError,match='finite'):
        chart.evaluate([[np.nan,0]])
    model.add_point(2,2,2)
    with pytest.raises(GeometryError,match='stale'):
        chart.jacobians(rows)


def test_frontal_refinement_honors_selected_quality_angle():
    from anymesher.surface_mesh import mesh_planar_surface, SurfaceMeshOptions
    from anymesher.native_v2 import NativeMeshingOptions
    controls=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',max_insertions=8,max_topology_operations=32)
    def mesh(angle):
        return mesh_planar_surface(((0.,0.),(2.,0.),(0.,1.)),options=SurfaceMeshOptions(
            target_size=3.,recombine=False,backend='python',min_angle=angle,
            enforce_quality=True,native_options=controls))
    result=mesh(15.)
    assert result.num_triangles>0
    with pytest.raises(MeshError):
        mesh(30.)


def test_uniform_physical_metric_uses_nonisometric_chart(monkeypatch):
    import anymesher.native_v2 as native
    from anymesher.triangulation import triangulate_polygon
    from anymesher.metric import MetricFieldSpec
    seed=triangulate_polygon(((0.,0.),(.1,0.),(0.,.1)),backend='python')
    derivative=np.array(((2.,0.),(0.,3.),(0.,0.)))
    observed=[]
    original=native._pullback_spatial_metrics
    def capture(*args,**kwargs):
        value=original(*args,**kwargs);observed.append(value.copy());return value
    monkeypatch.setattr(native,'_pullback_spatial_metrics',capture)
    native.frontal_delaunay_refine(seed,native.NativeMeshingOptions(
        point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
        metric_field=MetricFieldSpec.uniform(1.),max_insertions=8,max_topology_operations=32),
        target_size=1.,metric_to_physical=lambda points:points@derivative.T,
        metric_jacobian=derivative,qualified_seed=True)
    assert observed
    for values in observed:
        np.testing.assert_allclose(values,np.broadcast_to(np.diag((4.,9.)),values.shape))


def test_physical_certification_uses_active_cells_and_mean_edge_growth():
    from anymesher.core import MeshCore
    from anymesher.surface_mesh import SurfaceMeshOptions
    # Equal unit edge lengths across a triangle/quad interface: growth is 1,
    # although total perimeters differ by 4/3. An inactive sliver is discarded.
    xyz=np.array(((0,0,0),(1,0,0),(.5,-np.sqrt(3)/2,0),(1,1,0),(0,1,0),(.001,0,0)))
    core=MeshCore(xyz,((1,0,2),(0,5,2)),((0,1,3,4),),triangle_active=(True,False))
    chart=object.__new__(AnalyticMetricChart)
    chart.face_id=1
    chart.check=None
    chart.evaluate=lambda rows:np.column_stack((rows,np.zeros(len(rows))))
    report=chart.certify_core(core,SurfaceMeshOptions(max_element_growth=1.1),xyz)
    assert report['accepted']
    assert report['maximum_element_growth']==pytest.approx(1.)


def test_physical_candidate_cannot_inherit_a_good_chart_score():
    from anymesher.surface_mesh import (SurfaceMeshOptions, _make_candidate,
                                        _physical_quality_candidate)
    points=np.array(((0.,0.),(1.,0.),(.5,np.sqrt(3)/2)))
    settings=SurfaceMeshOptions(min_angle=15.,prefer_quality_policy=True)
    chart=_make_candidate(points,np.array(((0,1,2),)),settings=settings)
    assert not chart.report['poor_element_ids']
    physical=_physical_quality_candidate(chart,settings,
        lambda rows:np.column_stack((10*rows[:,0],rows[:,1],np.zeros(len(rows)))))
    assert physical.report['poor_element_ids']==[1]
    assert physical.report['min_angle']<15.
    assert physical.score>chart.score
    assert np.array_equal(physical.points,chart.points)
    with pytest.raises(MeshError,match='finite'):
        _physical_quality_candidate(chart,settings,lambda rows:np.full((len(rows),3),np.nan))
