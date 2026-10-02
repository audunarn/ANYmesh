import numpy as np
import pytest

from anymesher.core import MeshCore
from anymesher.quality_v2 import evaluate_quality
from anymesher.surface_mesh import (SurfaceMeshOptions, _published_quality_report,
                                    _quality_threshold_report, mesh_planar_surface)
from anymesher import NativeMeshingOptions
from anymesher.errors import MeshError


def test_array_growth_matches_independent_mixed_active_cell_oracle():
    core = MeshCore(np.array([(0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.),
                             (3.,0.,0.),(3.,1.,0.),(.5,2.,0.)]),
                    np.array([(0,1,2),(0,2,3),(3,2,6)]),
                    np.array([(1,4,5,2)]), triangle_ids=[17,23,41],quad_ids=[71],
                    triangle_active=[True,False,True])
    settings = SurfaceMeshOptions(max_element_growth=1.5)
    lengths,incidence = {},{}
    for identifier in (*core.tris,*core.quads):
        nodes=core.corners_of(identifier)
        p=np.array([core.nodes[node] for node in nodes])
        lengths[identifier]=sum(np.linalg.norm(p[(i+1)%len(p)]-p[i]) for i in range(len(p)))/len(p)
        for a,b in zip(nodes,(*nodes[1:],nodes[0])):
            incidence.setdefault(tuple(sorted((a,b))),[]).append(identifier)
    expected_max=1.
    expected_poor=set()
    for owners in incidence.values():
        if len(owners)==2:
            a,b=(lengths[identifier] for identifier in owners)
            ratio=max(a/b,b/a)
            expected_max=max(expected_max,ratio)
            if ratio>settings.max_element_growth:
                expected_poor.update(owners)
    quality=evaluate_quality(core)
    thresholds=_quality_threshold_report(quality,settings)
    result=_published_quality_report(core,settings,quality,thresholds)
    assert result['max_element_growth'] == pytest.approx(expected_max,abs=1e-15)
    assert result['elements_above_maximum_growth'] == len(expected_poor)
    assert set(result['poor_element_ids']) == expected_poor | set(thresholds['poor_element_ids'])
    assert 23 not in result['poor_element_ids']


@pytest.mark.parametrize('recombine',[False,True])
def test_final_publication_uses_physical_quality_and_keeps_chart_coordinates(recombine):
    outer=np.array([(0.,0.),(10.,0.),(10.,1.),(0.,1.)])
    diagnostics={}
    def evaluate(rows):
        return np.column_stack((.1*rows[:,0],rows[:,1],np.zeros(len(rows))))
    options=SurfaceMeshOptions(target_size=10.,recombine=recombine,min_angle=15.,enforce_quality=True,
        prefer_quality_policy=True,native_options=NativeMeshingOptions(point_placement='frontal_delaunay'))
    core=mesh_planar_surface(outer,options=options,backend='python',diagnostics=diagnostics,
        _metric_to_physical=evaluate,_metric_jacobian=np.array([[.1,0.],[0.,1.],[0.,0.]]),
        _preserve_spatial_refinement=True,_boundary_is_seeded=True)
    np.testing.assert_array_equal(core.node_coordinates[:4,:2],outer)
    assert diagnostics['quality_policy']['accepted']
    assert diagnostics['quality_optimization']['final_quality']['poor_element_ids'] == []


def test_physical_publication_rejects_stretched_cells_even_if_chart_is_good():
    outer=np.array([(0.,0.),(1.,0.),(1.,1.),(0.,1.)])
    options=SurfaceMeshOptions(recombine=True,enforce_quality=True)
    def evaluate(rows):
        return np.column_stack((20.*rows[:,0],rows[:,1],np.zeros(len(rows))))
    with pytest.raises(MeshError,match='quality policy rejected'):
        mesh_planar_surface(outer,options=options,_metric_to_physical=evaluate,
                            _preserve_spatial_refinement=True,_boundary_is_seeded=True)
