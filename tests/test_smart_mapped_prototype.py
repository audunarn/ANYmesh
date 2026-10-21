"""Hidden prototype coverage/conformity, independent of public route selection."""
from collections import Counter

import numpy as np
import pytest

from anymesher._smart_mapped_prototype import build_smart_mapped
from anymesher.errors import MeshError
from anymesher.triangulation import orient2d


def rectangle(x0=0.,y0=0.,x1=8.,y1=6.):
    return np.array(((x0,y0),(x1,y0),(x1,y1),(x0,y1)))


def area(loop):
    return abs(sum(float(a[0]*b[1]-a[1]*b[0])
                   for a,b in zip(loop,np.roll(loop,-1,axis=0)))/2.)


def assert_inventory(result,outer,holes=(),constraints=()):
    cells = [*result.quads,*result.triangles]
    incidence = Counter(tuple(sorted((int(a),int(b)))) for body in cells
                        for a,b in zip(body,np.roll(body,-1)))
    coord = {tuple(p):i for i,p in enumerate(result.points)}
    expected = {tuple(sorted((coord[tuple(a)],coord[tuple(b)])))
                for loop in (outer,*holes) for a,b in zip(loop,np.roll(loop,-1,axis=0))}
    assert {edge for edge,count in incidence.items() if count == 1} == expected
    assert all(count in (1,2) for count in incidence.values())
    assert len(result.constraint_chains) == len(constraints)
    derived = []
    for segment,chain in zip(constraints,result.constraint_chains):
        np.testing.assert_array_equal(result.points[chain[[0,-1]]],segment)
        for a,b in zip(chain,chain[1:]):
            assert incidence[tuple(sorted((int(a),int(b))))] == 2
            derived.append((int(a),int(b)))
    np.testing.assert_array_equal(result.constraints,np.asarray(derived,dtype=int).reshape((-1,2)))
    for body in cells:
        points = result.points[body]
        assert all(orient2d(points[i-1],points[i],points[(i+1)%len(points)]) > 0.
                   for i in range(len(points)))
    assert sum(area(result.points[body]) for body in cells) == pytest.approx(
        area(outer)-sum(area(hole) for hole in holes),rel=1e-13,abs=1e-13)
    assert result.diagnostics['exact_coverage']
    assert result.diagnostics['repair_scope'] == 'global_constrained_triangulation'


@pytest.mark.parametrize('holes',[
    (),
    (rectangle(3.125,2.125,4.625,3.625),),
    (rectangle(3.41,2.41,3.42,2.42),),
    (rectangle(2.1,2.1,2.9,3.1),rectangle(3.1,2.2,3.9,3.2)),
])
def test_holes_fixed_boundaries_regular_quads_and_no_input_mutation(holes):
    outer = rectangle()
    saved = [outer.copy(),*[hole.copy() for hole in holes]]
    result = build_smart_mapped(outer,holes,target_size=1.,enable_pairing=False,enable_smoothing=False,enable_refinement=False)
    assert_inventory(result,outer,holes)
    assert len(result.quads) > 0 and len(result.triangles) > 0
    for quad in result.quads:
        points = result.points[quad]
        np.testing.assert_array_equal(points[1]-points[0],(1.,0.))
        np.testing.assert_array_equal(points[3]-points[0],(0.,1.))
    assert all(np.array_equal(before,after) for before,after in zip(saved,(outer,*holes)))
    repeated = build_smart_mapped(outer,holes,target_size=1.,enable_pairing=False,enable_smoothing=False,enable_refinement=False)
    for name in ('points','quads','triangles','outer_loop','constraints'):
        np.testing.assert_array_equal(getattr(result,name),getattr(repeated,name))
        assert not getattr(result,name).flags.writeable


@pytest.mark.parametrize('outer',[
    np.array(((0.,0.),(8.,0.),(7.,6.),(1.,6.))),
    np.array(((0.,0.),(8.,0.),(8.,6.),(5.,6.),(5.,4.),(3.,4.),(3.,6.),(0.,6.))),
])
def test_oblique_and_concave_outer_boundaries(outer):
    result = build_smart_mapped(outer,target_size=1.)
    assert_inventory(result,outer)
    assert len(result.quads) > 0


def test_diagonal_internal_line_with_existing_boundary_endpoints():
    outer = np.array(((0.,0.),(8.,0.),(8.,4.),(8.,6.),(0.,6.),(0.,1.)))
    constraints = (np.array(((0.,1.),(8.,4.))),)
    result = build_smart_mapped(outer,constraints=constraints,target_size=.5)
    assert_inventory(result,outer,constraints=constraints)
    assert len(result.quads) > 0
    assert len(result.constraint_chains[0]) == 19
    assert np.max(np.linalg.norm(np.diff(result.points[result.constraint_chains[0]],axis=0),axis=1)) <= .5
    assert not result.constraint_chains[0].flags.writeable


def test_sampled_oblique_outer_does_not_acquire_unspecified_grid_stations():
    vertices = np.array(((0.,0.),(8.,0.),(7.,5.),(1.,5.)))
    outer = []
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        count = int(np.ceil(np.linalg.norm(b-a)/.5))
        outer.extend(a+(b-a)*(i/count) for i in range(count))
    outer = np.asarray(outer)
    angles = np.arange(24)*2*np.pi/24
    hole = np.array((3.5,2.5))+.7*np.column_stack((np.cos(angles),np.sin(angles)))
    result = build_smart_mapped(outer,(hole,),target_size=.5)
    assert_inventory(result,outer,(hole,))
    assert (.5,2.5) not in {tuple(point) for point in result.points}
    assert (7.5,2.5) not in {tuple(point) for point in result.points}
    assert len(result.quads) > 0


def test_unrequired_free_grid_points_are_not_seeded_next_to_constraints():
    import anymesher._smart_mapped_prototype as prototype
    outer = np.array(((0.,0.),(8.,0.),(8.,4.),(8.,6.),(0.,6.),(0.,1.)))
    segment = np.array(((0.,1.),(8.,4.)))
    result = prototype.build_smart_mapped(outer,constraints=(segment,),target_size=.5,enable_refinement=False)
    required = {tuple(result.points[n]) for quad in result.quads for n in quad}
    authored = {tuple(point) for point in outer} | {
        tuple(result.points[n]) for chain in result.constraint_chains for n in chain}
    assert {tuple(point) for point in result.points} <= required | authored


def test_refined_constraint_budget_refuses_before_grid_or_cdt(monkeypatch):
    import anymesher._smart_mapped_prototype as prototype
    monkeypatch.setattr(prototype.np,'linspace',lambda *a,**k:pytest.fail('grid allocated'))
    monkeypatch.setattr(prototype,'triangulate_polygon',lambda *a,**k:pytest.fail('CDT called'))
    with pytest.raises(MeshError,match='refined constraint station budget'):
        prototype.build_smart_mapped(rectangle(),constraints=(((1.,1.),(7.,5.)),),target_size=.0001)


def test_malformed_ring_width_refuses_before_array_copy(monkeypatch):
    import anymesher._smart_mapped_prototype as prototype
    malformed = np.zeros((4,3))
    monkeypatch.setattr(prototype.np,'array',lambda *a,**k:pytest.fail('malformed array copied'))
    with pytest.raises(MeshError,match='planar coordinates'):
        prototype.build_smart_mapped(malformed,target_size=1.)


@pytest.mark.parametrize('constraint',[False,True])
def test_malformed_higher_rank_array_refuses_before_array_copy(monkeypatch,constraint):
    import anymesher._smart_mapped_prototype as prototype
    malformed = np.zeros((2 if constraint else 4,2,7))
    outer = rectangle()
    original_array = prototype.np.array
    def recording_array(value,*args,**kwargs):
        assert value is not malformed, 'malformed higher-rank input copied'
        return original_array(value,*args,**kwargs)
    monkeypatch.setattr(prototype.np,'array',recording_array)
    with pytest.raises(MeshError,match='planar'):
        prototype.build_smart_mapped(outer if constraint else malformed,
            constraints=(malformed,) if constraint else (),target_size=1.)


def test_reversed_outer_and_hole_order_preserve_material():
    outer = rectangle()[::-1].copy()
    holes = (rectangle(3.1,2.1,4.1,3.1)[::-1].copy(),)
    assert_inventory(build_smart_mapped(outer,holes,target_size=1.),outer,holes)


@pytest.mark.parametrize('kind',['touching_hole','outside_hole','crossing_constraints','unsampled_endpoint'])
def test_unsupported_topology_refuses(kind):
    outer = rectangle()
    holes, constraints = (),()
    if kind == 'touching_hole':
        holes = (rectangle(0.,2.,1.,3.),)
    elif kind == 'outside_hole':
        holes = (rectangle(9.,2.,10.,3.),)
    elif kind == 'crossing_constraints':
        constraints = (np.array(((1.,1.),(7.,5.))),np.array(((1.,5.),(7.,1.))))
    else:
        constraints = (np.array(((0.,1.),(7.,5.))),)
    with pytest.raises(MeshError):
        build_smart_mapped(outer,holes,constraints,target_size=1.)


def test_budget_refuses_before_grid_allocation(monkeypatch):
    monkeypatch.setattr(np,'linspace',lambda *a,**k:pytest.fail('grid allocated before budget refusal'))
    with pytest.raises(MeshError,match='cell budget'):
        build_smart_mapped(rectangle(),target_size=.001,max_grid_cells=16)


@pytest.mark.parametrize('stage',['smart-mapped:start','smart-mapped:background-row',
                                 'smart-mapped:global-cdt','smart-mapped:complete'])
def test_cancellation_propagates_identity_and_preserves_inputs(stage):
    outer = rectangle()
    before = outer.copy()
    marker = MeshError('cancel prototype')
    def cancel(current):
        if current == stage:
            raise marker
    with pytest.raises(MeshError) as caught:
        build_smart_mapped(outer,target_size=1.,cancellation_check=cancel)
    assert caught.value is marker
    np.testing.assert_array_equal(outer,before)
