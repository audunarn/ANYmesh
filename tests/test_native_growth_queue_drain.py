"""A refused ordinary queue cannot hide a fresh owner growth frontier."""
import numpy as np
import pytest
from anymesher import native_v2 as native
from anymesher.metric import MetricFieldSpec
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher._material_region_binding import material_read_scope
from test_native_physical_growth_frontier import seed
from test_material_numerical_read import fixture


@pytest.mark.parametrize('limit',[None,1.5])
def test_real_owner_ordinary_queue_drains_before_growth_and_refused_growth_is_finite(monkeypatch,limit):
    model,face,binding,scope_options=fixture()
    chart=AnalyticMetricChart(model,face,region_binding=binding);baseline=seed(chart)
    options=native.NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
        metric_field=MetricFieldSpec.uniform(.05),max_insertions=2,max_topology_operations=64)
    # Only the geometric proposal is made impossible. Queue computation, actual
    # owner XYZ, growth proof, Work accounting and native refusal stay real.
    monkeypatch.setattr(native,'_offcentre',lambda *a,**kw:np.array((100.,100.)))
    monkeypatch.setattr(native,'_circumcentre',lambda *a,**kw:np.array((100.,100.)))
    with material_read_scope(binding,scope_options):
        result,report=native.frontal_delaunay_refine(baseline,options,target_size=.05,
            metric_to_physical=chart.evaluate,metric_jacobian=chart.jacobians,
            qualified_seed=True,short_edge_offcentre=True,_physical_growth_limit=limit)
    assert report['insertions']==0 and 0<report['topology_operations']<=64
    np.testing.assert_array_equal(result.points,baseline.points)
    assert report['selected_route']=='frontal_delaunay_geometry_limited'
    if limit is None:assert 'physical_growth_frontier' not in report
    else:
        growth=report['physical_growth_frontier']
        assert growth['rounds']==1 and growth['proposals']>=1 and growth['insertions']==0
