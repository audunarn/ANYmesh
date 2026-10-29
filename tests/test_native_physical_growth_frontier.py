import numpy as np
import pytest
from anymesher import native_v2 as native
from anymesher.metric import MetricFieldSpec
from anymesher.triangulation import PlanarTriangulation
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher._material_region_binding import material_read_scope
from test_material_numerical_read import fixture


def seed(chart):
    points=np.array(((.35,.4),(.65,.4),(.5,.85),(.5,.25)))@chart.transform
    cells=np.array(((0,1,2),(0,3,1)),dtype=np.int64)
    border=np.array(((0,2),(2,1),(1,3),(3,0)),dtype=np.int64)
    return PlanarTriangulation(points=points,triangles=cells,segments=border,
        boundary_segments=border,mandatory_segments=np.empty((0,2),dtype=np.int64),
        outer_loop=np.array((0,3,1,2)),hole_loops=())


@pytest.mark.parametrize('scale',(1e-3,1.,1e3))
@pytest.mark.parametrize('rotate',(False,True))
def test_frontier_uses_actual_xyz_mean_sizes_and_only_coarser_neighbor(scale,rotate):
    points=np.array(((0.,0.,0.),(1.,0.,0.),(.5,1.5,0.),(.5,-.5,0.)))
    if rotate:points=points[:,(2,0,1)]
    points=scale*points+13.*scale
    cells=np.array(((0,1,2),(0,3,1)),dtype=np.int64)
    frontier=native._physical_growth_frontier(cells,points,1.5)
    lengths=np.linalg.norm(np.roll(points[cells],-1,axis=1)-points[cells],axis=2).mean(axis=1)
    assert len(frontier)==1 and frontier[0][2]==(0,1,2)
    assert -frontier[0][0]==lengths[0]/lengths[1]/1.5
    assert native._physical_growth_frontier(cells,points,2.)==[]


def test_frontier_matches_full_published_growth_for_world_nonlinearity():
    from anymesher.surface_mesh import _published_growth
    from anymesher.core import MeshCore
    cells=np.array(((0,1,2),(0,3,1)),dtype=np.int64)
    points=np.array(((0.,0.,0.),(1.,0.,0.),(.5,1.,2.),(.5,-1.,0.)))
    maximum,poor=_published_growth(MeshCore(points,cells),1.5)
    frontier=native._physical_growth_frontier(cells,points,1.5)
    assert maximum>1.5 and poor=={1,2}
    assert len(frontier)==1 and frontier[0][2]==(0,1,2)


@pytest.mark.parametrize('generic',(False,True))
def test_actual_native_empty_metric_queue_routes_growth_only_in_owner_scope(generic):
    model,face,binding,scope_options=fixture();chart=AnalyticMetricChart(model,face,region_binding=binding)
    baseline=seed(chart)
    options=native.NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
        metric_field=MetricFieldSpec.uniform(5.),max_insertions=2,max_topology_operations=4)
    evaluator=(lambda rows:chart.evaluate(rows)) if generic else chart.evaluate
    with material_read_scope(binding,scope_options):
        result,report=native.frontal_delaunay_refine(baseline,options,target_size=5.,
            metric_to_physical=evaluator,metric_jacobian=chart.jacobians,
            qualified_seed=True,short_edge_offcentre=True,_physical_growth_limit=1.5)
    np.testing.assert_array_equal(baseline.points,result.points[:len(baseline.points)])
    assert report['topology_operations']<=4 and report['insertions']<=2
    if generic:assert 'physical_growth_frontier' not in report
    else:
        assert report['physical_growth_frontier']['rounds']>=1
        assert report['physical_growth_frontier']['proposals']>=1


@pytest.mark.parametrize('limit',(True,0.,float('inf'),float('nan')))
def test_invalid_private_limit_refuses_before_callbacks(limit):
    model,face,binding,_=fixture();chart=AnalyticMetricChart(model,face,region_binding=binding)
    with pytest.raises(Exception,match='physical growth limit'):
        native.frontal_delaunay_refine(seed(chart),native.NativeMeshingOptions(point_placement='frontal_delaunay'),
            target_size=5.,_physical_growth_limit=limit)


def test_frontier_cancellation_preserves_original_exception():
    error=RuntimeError('cancel')
    def cancel(phase):raise error
    with pytest.raises(RuntimeError) as caught:
        native._physical_growth_frontier(np.array(((0,1,2),)),np.eye(3),1.5,cancel)
    assert caught.value is error

