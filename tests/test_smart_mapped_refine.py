"""Bounded cavity refinement preserves physical interfaces and outside quads."""
import numpy as np
import pytest

from anymesher._smart_mapped_prototype import _Work,build_smart_mapped
from anymesher._smart_mapped_refine import refine
from anymesher.errors import MeshError


def sample(vertices,h=.5):
    result = []
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        count = int(np.ceil(np.linalg.norm(b-a)/h))
        result.extend(a+(b-a)*(i/count) for i in range(count))
    return np.asarray(result)


def test_tiny_hole_refinement_improves_and_preserves_outside_cells():
    outer = sample(np.array(((0.,0.),(8.,0.),(8.,5.),(0.,5.))))
    angle = np.arange(12)*2*np.pi/12
    hole = np.array((3.23,2.23))+.025*np.column_stack((np.cos(angle),np.sin(angle)))
    baseline = build_smart_mapped(outer,(hole,),target_size=.5,enable_refinement=False)
    result = build_smart_mapped(outer,(hole,),target_size=.5)
    evidence = result.diagnostics['refinement']
    assert evidence['accepted_cavities'] > 0
    assert evidence['after']['minimum_triangle_angle'] > evidence['before']['minimum_triangle_angle']
    assert evidence['after']['triangles_below_minimum'] <= evidence['before']['triangles_below_minimum']
    assert evidence['added_nodes'] <= 128 and evidence['cdt_calls'] <= 16
    np.testing.assert_array_equal(result.points[result.outer_loop],outer)
    np.testing.assert_array_equal(result.points[result.hole_loops[0]],hole)
    assert result.diagnostics['exact_coverage'] and result.diagnostics['prescribed_boundaries_exact']
    released = set(evidence['released_quad_indices'])
    footprints = {frozenset(map(tuple,result.points[q])) for q in result.quads}
    for i,q in enumerate(baseline.quads):
        if i not in released:
            assert frozenset(map(tuple,baseline.points[q])) in footprints
        polygon = baseline.points[q]
        if all((a[0] == b[0]) != (a[1] == b[1]) for a,b in zip(polygon,np.roll(polygon,-1,axis=0))):
            assert i not in released
            assert frozenset(map(tuple,polygon)) in footprints
    repeat = build_smart_mapped(outer,(hole,),target_size=.5)
    for field in ('points','quads','triangles'):
        np.testing.assert_array_equal(getattr(result,field),getattr(repeat,field))


def test_disabled_and_zero_budget_are_exact_prior_stage():
    outer = np.array(((0.,0.),(8.,0.),(8.,5.),(0.,5.)))
    a = build_smart_mapped(outer,target_size=.5,enable_refinement=False)
    b = build_smart_mapped(outer,target_size=.5,refinement_budget=0)
    for field in ('points','quads','triangles'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))


def test_passing_cells_are_bitwise_noop():
    points = np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    output,quads,triangles,report = refine(points,((0,1,2,3),),(),protected_edges=set(),
        enabled=True,budget=16,work=_Work(1000,None),cancellation_check=None)
    np.testing.assert_array_equal(output,points)
    assert quads == ((0,1,2,3),) and triangles == ()
    assert report['cdt_calls'] == report['added_nodes'] == 0


def test_passing_mesh_skips_adjacency_and_work_after_cancellation_start():
    stages = []
    class NoWork:
        def step(self,*args):
            pytest.fail('passing mesh built refinement adjacency')
    points = np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    refine(points,((0,1,2,3),),(),protected_edges=set(),enabled=True,budget=16,
           work=NoWork(),cancellation_check=stages.append)
    assert stages == ['smart-mapped:local-refinement']


def test_unexpected_cdt_station_declines_without_publication(monkeypatch):
    import anymesher._smart_mapped_refine as module
    from types import SimpleNamespace
    points = np.array(((0.,0.),(1.,0.),(.5,.01)))
    before = points.copy()
    def unexpected(*args,**kwargs):
        return SimpleNamespace(points=np.vstack((points,(.5,.005))),triangles=np.array(((0,1,2),)))
    monkeypatch.setattr(module,'triangulate_polygon',unexpected)
    output,quads,triangles,report = refine(points,(),((0,1,2),),protected_edges=set(),
        enabled=True,budget=16,work=_Work(1000,None),cancellation_check=None)
    np.testing.assert_array_equal(points,before)
    np.testing.assert_array_equal(output,before)
    assert not quads and triangles == ((0,1,2),)
    assert report['accepted_cavities'] == 0
    assert report['cavities'][0]['reason'] == 'changed_station_inventory'


def test_refinement_cancellation_preserves_identity_and_input():
    outer = np.array(((0.,0.),(8.,0.),(8.,5.),(0.,5.)))
    before = outer.copy()
    marker = MeshError('cancel local refinement')
    def cancel(stage):
        if stage == 'smart-mapped:local-refinement':
            raise marker
    with pytest.raises(MeshError) as caught:
        build_smart_mapped(outer,target_size=.5,cancellation_check=cancel)
    assert caught.value is marker
    np.testing.assert_array_equal(outer,before)


@pytest.mark.parametrize('budget',[-1,65,True,1.5])
def test_refinement_budget_is_explicitly_bounded(budget):
    with pytest.raises(MeshError,match='refinement'):
        build_smart_mapped(((0.,0.),(1.,0.),(1.,1.),(0.,1.)),target_size=.5,refinement_budget=budget)
