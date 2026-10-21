"""Private improvement policy, isolation, and connected coverage regressions."""
import numpy as np
import pytest

from anymesher._smart_mapped_improve import improve
from anymesher._smart_mapped_prototype import _Work, build_smart_mapped
from anymesher.errors import MeshError
from anymesher.quad.quality_gate import quad_violation, corner_metrics


def run(points,triangles,**kwargs):
    options = dict(fixed_nodes=(),protected_edges=set(),enable_pairing=True,
                   enable_smoothing=True,budget=4096,work=_Work(100_000,None),
                   cancellation_check=None)
    options.update(kwargs)
    return improve(np.asarray(points,dtype=float),(),triangles,**options)


def test_pairing_exact_convex_footprint_and_protected_diagonal():
    points = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    triangles = ((0,1,2),(0,2,3))
    output,quads,tris,report = run(points,triangles)
    np.testing.assert_array_equal(output,points)
    assert quads == ((0,1,2,3),) and not tris
    assert quad_violation(output[list(quads[0])]) is None
    assert report['accepted_pairs'] == 1
    _,quads,tris,report = run(points,triangles,protected_edges={(0,2)})
    assert not quads and tris == triangles and report['accepted_pairs'] == 0


def test_pairing_does_not_admit_bad_aspect_quad():
    _,quads,tris,report = run(((0.,0.),(10.,0.),(10.,.1),(0.,.1)),
        ((0,1,2),(0,2,3)),enable_smoothing=False)
    assert not quads and len(tris) == 2 and report['accepted_pairs'] == 0


def test_movement_improves_triangle_deficits_without_fixed_or_remote_drift():
    points = np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.),(.05,.05),
                       (3.,0.),(4.,0.),(3.,1.)))
    original = points.copy()
    triangles = ((0,1,4),(1,2,4),(2,3,4),(3,0,4),(5,6,7))
    output,quads,tris,report = run(points,triangles,fixed_nodes={0,1,2,3},enable_pairing=False)
    assert report['accepted_moves'] > 0 and report['moved_node_ids'] == [4]
    assert report['after']['minimum_triangle_angle'] > report['before']['minimum_triangle_angle']
    np.testing.assert_array_equal(output[[0,1,2,3,5,6,7]],original[[0,1,2,3,5,6,7]])
    np.testing.assert_array_equal(points,original)
    for old,new in zip(triangles,tris):
        assert max(0.,15.-corner_metrics(output[list(new)])[0]) <= max(0.,15.-corner_metrics(original[list(old)])[0])
    repeated = run(points,triangles,fixed_nodes={0,1,2,3},enable_pairing=False)
    np.testing.assert_array_equal(output,repeated[0])
    assert report == repeated[3]


def test_zero_budget_and_disabled_stages_leave_mesh_identical():
    points = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    triangles = ((0,1,2),(0,2,3))
    for settings in (dict(budget=0),dict(enable_pairing=False,enable_smoothing=False)):
        output,quads,tris,report = run(points,triangles,**settings)
        np.testing.assert_array_equal(output,points)
        assert not quads and tris == triangles
        assert report['accepted_pairs'] == report['accepted_moves'] == 0


@pytest.mark.parametrize('stage',['smart-mapped:improvement','smart-mapped:node-movement'])
def test_cancellation_identity_and_input_immutability(stage):
    marker = MeshError('cancel improvement')
    def cancel(current):
        if current == stage:
            raise marker
    points = np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.),(.05,.05)))
    before = points.copy()
    with pytest.raises(MeshError) as caught:
        run(points,((0,1,4),(1,2,4),(2,3,4),(3,0,4)),enable_pairing=False,
            fixed_nodes={0,1,2,3},cancellation_check=cancel)
    assert caught.value is marker
    np.testing.assert_array_equal(points,before)


def test_connected_controls_keep_station_chains_and_exact_coverage():
    outer = np.array(((0.,0.),(8.,0.),(8.,4.),(8.,6.),(0.,6.),(0.,1.)))
    segments = (((0.,1.),(8.,4.)),)
    results = [build_smart_mapped(outer,constraints=segments,target_size=.5,
                enable_pairing=pair,enable_smoothing=move)
               for pair,move in ((False,False),(True,False),(True,True))]
    baseline = results[0]
    for result in results:
        np.testing.assert_array_equal(result.points[result.outer_loop],outer)
        np.testing.assert_array_equal(result.points[result.constraint_chains[0]],
                                      baseline.points[baseline.constraint_chains[0]])
        assert result.diagnostics['exact_coverage'] and result.diagnostics['prescribed_boundaries_exact']
        assert all(quad_violation(result.points[body]) is None for body in result.quads)
        assert result.diagnostics['improvement']['evaluations'] <= 8192
    assert results[-1].diagnostics['improvement']['after']['triangles_below_minimum'] <= \
           results[0].diagnostics['improvement']['before']['triangles_below_minimum']
