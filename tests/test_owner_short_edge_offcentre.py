"""Local metric short-edge proposals preserve the legacy route by default."""
from math import sqrt

import numpy as np
import pytest

import anymesher.native_v2 as native
from anymesher.errors import MeshError
from anymesher.triangulation import triangulate_polygon


def repeated(metric):
    return np.broadcast_to(np.asarray(metric, dtype=float), (3, 2, 2)).copy()


def legacy_reference(coordinates, tensors, size):
    lengths = [float(np.linalg.norm(coordinates[(i+1) % 3]-coordinates[i])) for i in range(3)]
    edge = min(range(3), key=lambda i: (lengths[i], i))
    a, b, c = coordinates[edge], coordinates[(edge+1) % 3], coordinates[(edge+2) % 3]
    midpoint = .5 * (a+b)
    delta = b-a
    normal = np.asarray((-delta[1], delta[0]), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1e-30)
    if float((c-midpoint) @ normal) < 0.:
        normal = -normal
    eigenvalue = float(np.max(np.linalg.eigvalsh(np.mean(tensors, axis=0))))
    target = min(size, 1. / sqrt(max(eigenvalue, 1e-30)))
    height = sqrt(max(target * target - (.5 * lengths[edge]) ** 2, (.35 * target) ** 2))
    return midpoint + normal * height


def independent_reference(points, tensors, size):
    """Quadratic-form normal, independent of Cholesky/whitening implementation."""
    metric = tensors.mean(axis=0)
    metric = .5 * metric + .5 * metric.T
    edges = np.roll(points, -1, axis=0)-points
    metric_lengths = np.sqrt(np.einsum('ni,ij,nj->n', edges, metric, edges))
    edge = min(range(3), key=lambda i: (metric_lengths[i], i))
    delta = edges[edge]
    covector = metric @ delta
    normal = np.asarray((-covector[1], covector[0]))
    normal /= sqrt(float(normal @ metric @ normal))
    midpoint = .5 * points[edge] + .5 * points[(edge+1) % 3]
    if float((points[(edge+2) % 3]-midpoint) @ metric @ normal) < 0.:
        normal = -normal
    target = min(size, 1. / sqrt(float(np.linalg.eigvalsh(metric)[-1])))
    chart_height = sqrt(max(target ** 2 - (.5 * np.linalg.norm(delta)) ** 2, (.35 * target) ** 2))
    metric_height = sqrt(max(1. - (.5 * metric_lengths[edge]) ** 2, .35 ** 2))
    height = min(sqrt(3.) * .5 * metric_lengths[edge], metric_height,
                 chart_height / np.linalg.norm(normal))
    return midpoint + normal * height


@pytest.mark.parametrize('points', (
    ((0., 0.), (.0013, 0.), (.3, .4)),
    ((0., 0.), (1.5, 0.), (.75, 3.)),
    ((1., 3.), (2., 3.), (1., 4.)),
))
@pytest.mark.parametrize('metric', (np.eye(2), np.asarray(((4., 1.2), (1.2, 2.)))))
def test_default_helper_matches_independent_legacy_bytes(points, metric):
    points, tensors = np.asarray(points), repeated(metric)
    expected = legacy_reference(points, tensors, .5)
    assert native._offcentre(points, tensors, .5).tobytes() == expected.tobytes()
    assert native._offcentre(points, tensors, .5, short_edge_metric=False).tobytes() == expected.tobytes()


def test_tiny_base_new_cap_discriminates_far_legacy_proposal():
    points = np.asarray(((0., 0.), (.0013, 0.), (.3, .4)))
    tensors = repeated(np.eye(2) / .5 ** 2)
    old = native._offcentre(points, tensors, .5)
    new = native._offcentre(points, tensors, .5, short_edge_metric=True)
    np.testing.assert_allclose(new, (.00065, sqrt(3.) * .0013 / 2), rtol=1e-14, atol=1e-18)
    assert old[1] > 400 * new[1]
    assert new[0] == old[0]


def test_active_large_base_cannot_enlarge_original_desired_height():
    points = np.asarray(((0., 0.), (1.5, 0.), (.75, 3.)))
    old = native._offcentre(points, repeated(np.eye(2)), 1.)
    new = native._offcentre(points, repeated(np.eye(2)), 1., short_edge_metric=True)
    assert new.tobytes() == old.tobytes()
    assert new[1] == sqrt(1. - .75 ** 2) < 1.


@pytest.mark.parametrize('size', (.00001, .01, .5, 2.))
def test_chart_desired_height_bound_is_preserved_in_skew_metric(size):
    points = np.asarray(((0., 0.), (.03, .01), (.2, .4)))
    tensors = repeated(((7., 2.5), (2.5, 2.)))
    expected = independent_reference(points, tensors, size)
    new = native._offcentre(points, tensors, size, short_edge_metric=True)
    np.testing.assert_allclose(new, expected, rtol=2e-14, atol=1e-16)
    assert np.linalg.norm(new - .5 * (points[0]+points[1])) <= size * (1+1e-14)


def test_shortest_edge_uses_metric_length_not_chart_length():
    points = np.asarray(((0., 0.), (1., 0.), (.5, .1)))
    tensors = repeated(np.diag((.0001, 1.)))
    result = native._offcentre(points, tensors, 10., short_edge_metric=True)
    np.testing.assert_allclose(result, (.5, sqrt(3.) * .01 / 2), rtol=1e-14, atol=1e-16)
    assert np.argmin(np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)) != 0


def test_nonuniform_local_mean_spd_and_inverse_transpose_match_reference():
    points = np.asarray(((.1, -.2), (.104, -.198), (.3, .5)))
    tensors = np.asarray((((7., 2.), (2., 2.)), ((4., -.7), (-.7, 3.)), ((9., 3.), (3., 4.))))
    before = points.tobytes(), tensors.tobytes()
    result = native._offcentre(points, tensors, .5, short_edge_metric=True)
    np.testing.assert_allclose(result, independent_reference(points,tensors,.5), rtol=1e-14, atol=1e-16)
    assert (points.tobytes(), tensors.tobytes()) == before


@pytest.mark.parametrize('affine', (
    np.asarray(((1., .7), (.2, 1.3))),
    np.asarray(((0., -2.), (1., 0.))),
    np.asarray(((2., .1), (.3, .7))),
))
def test_same_geometry_under_affine_chart_change_when_scalar_caps_inactive(affine):
    points = np.asarray(((0., 0.), (.0013, 0.), (.3, .4)))
    tensors = repeated(((4., 1.1), (1.1, 2.)))
    original = native._offcentre(points,tensors,100.,short_edge_metric=True)
    inverse = np.linalg.inv(affine)
    changed_metric = np.asarray([inverse @ m @ inverse.T for m in tensors])
    shift = np.asarray((.2,-.1))
    changed = native._offcentre(points @ affine + shift, changed_metric, 100., short_edge_metric=True)
    np.testing.assert_allclose(changed, original @ affine + shift, rtol=0., atol=3e-16)


@pytest.mark.parametrize('scale', (.001, .5, 2., 1000.))
def test_uniform_coordinate_and_target_rescaling(scale):
    points = np.asarray(((0.,0.),(.03,.01),(.2,.4)))
    tensors = repeated(((7.,2.5),(2.5,2.)))
    original = native._offcentre(points,tensors,.00001,short_edge_metric=True)
    changed = native._offcentre(points*scale,tensors/scale**2,.00001*scale,short_edge_metric=True)
    np.testing.assert_allclose(changed/scale,original,rtol=2e-14,atol=1e-16)


def test_centered_whitening_preserves_representable_translation():
    points = np.asarray(((0.,0.),(.125,0.),(.5,1.)))
    tensors = repeated(((4.,1.),(1.,2.)))
    original = native._offcentre(points,tensors,.5,short_edge_metric=True)
    shift = np.asarray((2.**35,-2.**34))
    changed = native._offcentre(points+shift,tensors,.5,short_edge_metric=True)
    assert changed.tobytes() == (original+shift).tobytes()


@pytest.mark.parametrize('points,tensors,size', (
    (np.zeros((3,2)), repeated(np.eye(2)), .5),
    (np.asarray(((0.,0.),(1.,0.),(2.,0.))), repeated(np.eye(2)), .5),
    (np.full((3,2),np.nan), repeated(np.eye(2)), .5),
    (np.asarray(((0.,0.),(.1,0.),(0.,1.))), repeated(((1.,2.),(2.,1.))), .5),
    (np.asarray(((0.,0.),(.1,0.),(0.,1.))), repeated(((1.,0.),(0.,0.))), .5),
    (np.asarray(((0.,0.),(.1,0.),(0.,1.))), repeated(((1.,.3),(0.,1.))), .5),
    (np.asarray(((0.,0.),(.1,0.),(0.,1.))), np.full((3,2,2),np.inf), .5),
    (np.asarray(((0.,0.),(.1,0.),(0.,1.))), repeated(np.eye(2)), 0.),
))
def test_invalid_owner_metric_or_degenerate_triangle_fails_typed(points,tensors,size):
    with pytest.raises(MeshError):
        native._offcentre(points,tensors,size,short_edge_metric=True)


def test_real_analytic_tiny_base_uses_existing_physical_metric_and_fixed_endpoints():
    from test_joint_triangle_owner_batches import analytic_chart
    from anygeometry import to_dict
    model, chart = analytic_chart()
    before = to_dict(model)
    raw = np.asarray(((.4,.1),(.4013,.1),(.6,.4)))
    points = raw @ chart.transform
    jacobians = chart.jacobians(points)
    tensors = np.einsum('nki,nkj->nij',jacobians,jacobians)/.5**2
    original_bytes = points.tobytes(),tensors.tobytes()
    old = native._offcentre(points,tensors,.5)
    new = native._offcentre(points,tensors,.5,short_edge_metric=True)
    np.testing.assert_allclose(new,independent_reference(points,tensors,.5),rtol=1e-14,atol=2e-16)
    xyz = chart.evaluate(np.vstack((points[:2],new)))
    lengths = np.linalg.norm(np.roll(xyz,-1,axis=0)-xyz,axis=1)
    assert np.max(lengths)/np.min(lengths) < 1.02
    midpoint = .5*(points[0]+points[1])
    assert np.linalg.norm(old-midpoint)>100*np.linalg.norm(new-midpoint)
    assert (points.tobytes(),tensors.tobytes()) == original_bytes and to_dict(model)==before


def test_real_constrained_owner_seed_inserts_atomically_and_conserves_exact_material():
    from fractions import Fraction
    from dataclasses import replace
    from test_joint_triangle_owner_batches import analytic_chart
    from anygeometry import to_dict
    model,chart=analytic_chart();before=to_dict(model)
    original=seed()
    original=replace(original,points=original.points @ chart.transform)
    boundary_bytes=original.points.tobytes()
    def exact_area(points,cells):
        total=Fraction(0)
        for row in cells:
            a,b,c=[[Fraction(float(v)) for v in points[node]] for node in row]
            twice=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            assert twice>0
            total+=twice/2
        return total
    area=exact_area(original.points,original.triangles)
    result,report=native.frontal_delaunay_refine(original,options(),target_size=.5,
        metric_to_physical=chart.evaluate,metric_jacobian=chart.jacobians,short_edge_offcentre=True,
        minimum_angle_target=15.)
    assert report['insertions']==1 and len(result.points)==len(original.points)+1
    assert result.points[:len(original.points)].tobytes()==boundary_bytes==original.points.tobytes()
    assert {tuple(sorted(e)) for e in result.segments}=={tuple(sorted(e)) for e in original.segments}
    assert exact_area(result.points,result.triangles)==area
    assert to_dict(model)==before and report['topology_operations']<=8


def test_nearer_circumcentre_guard_uses_same_metric_base_in_skew_tensor(monkeypatch):
    from anymesher.triangulation import PlanarTriangulation
    points=np.asarray(((0.,0.),(1.,0.),(.5,.1)))
    edges=np.asarray(((0,1),(1,2),(0,2)))
    original=PlanarTriangulation(points=points,triangles=np.asarray(((0,1,2),)),segments=edges,
        boundary_segments=edges,mandatory_segments=np.empty((0,2),dtype=np.int64),
        outer_loop=np.arange(3),hole_loops=())
    metric=np.asarray(((1.,20.),(20.,10000.)))
    derivative=np.vstack((np.linalg.cholesky(metric).T,np.zeros((1,2))))
    proposed=[];actual_offcentre=native._offcentre;inside=native._inside_ring
    def capture(coordinates,tensors,size,**kwargs):
        value=actual_offcentre(coordinates,tensors,size,**kwargs)
        proposed.append((value.copy(),coordinates.copy(),tensors.copy()))
        return value
    checked=[]
    def inside_capture(candidate,*args):
        checked.append(candidate.copy())
        return inside(candidate,*args)
    monkeypatch.setattr(native,'_offcentre',capture)
    monkeypatch.setattr(native,'_inside_ring',inside_capture)
    monkeypatch.setattr(native,'_circumcentre',lambda *args:np.asarray((.5,.03)))
    _,report=native.frontal_delaunay_refine(original,options(1),target_size=1.,minimum_angle_target=5.,
        metric_to_physical=lambda x:x @ derivative.T,metric_jacobian=derivative,
        short_edge_offcentre=True)
    assert proposed and checked
    point,coordinates,tensors=proposed[0]
    euclidean=min(range(3),key=lambda i:(np.linalg.norm(coordinates[(i+1)%3]-coordinates[i]),i))
    metric_edge=native._short_edge_metric_frame(coordinates,tensors)[-1]
    assert metric_edge!=euclidean
    old_mid=.5*(coordinates[euclidean]+coordinates[(euclidean+1)%3])
    new_mid=.5*(coordinates[metric_edge]+coordinates[(metric_edge+1)%3])
    circum=np.asarray((.5,.03))
    assert np.linalg.norm(point-old_mid)>np.linalg.norm(circum-old_mid)
    assert np.linalg.norm(point-new_mid)<np.linalg.norm(circum-new_mid)
    assert checked[0].tobytes()==point.tobytes()  # Kept for ordinary encroachment validation.
    assert report['insertions']==0  # Protected-base encroachment still refuses it.


def seed():
    return triangulate_polygon(((0.,0.),(.49935,0.),(.50065,0.),(1.,0.),(1.,1.),(0.,1.)),backend='python')


def options(operations=8):
    return native.NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
        max_insertions=1,max_topology_operations=operations,cancellation_interval=1)


def test_native_default_refine_receipt_and_callbacks_unchanged():
    outcomes, traces = [],[]
    original = seed()
    for supplied in ({},{'short_edge_offcentre':False}):
        phases=[]
        result,report=native.frontal_delaunay_refine(original,options(),target_size=.5,
            cancellation_check=phases.append,**supplied)
        outcomes.append((result.points.tobytes(),result.triangles.tobytes(),report))
        traces.append(phases)
    assert outcomes[0]==outcomes[1] and traces[0]==traces[1]
    assert 'short_edge_offcentre' not in outcomes[0][2]


def test_routing_uses_cached_tensors_without_extra_owner_queries_and_keeps_refusal_pool(monkeypatch):
    inputs=[]
    def proposal(points,tensors,size,**kwargs):
        inputs.append((points.copy(),tensors.copy(),kwargs.get('short_edge_metric',False)))
        return np.asarray((.5,.2))
    monkeypatch.setattr(native,'_offcentre',proposal)
    monkeypatch.setattr(native,'_circumcentre',lambda *args:None)
    error=native._GeometryLimited('controlled local refusal')
    monkeypatch.setattr(native.MutableT3Topology,'insert_point',lambda *args,**kwargs:(_ for _ in ()).throw(error))
    outcomes,owner_rows=[],[]
    original=seed();before=original.points.tobytes(),original.triangles.tobytes(),original.segments.tobytes()
    for guarded in (False,True):
        rows=[]
        def owner(x):
            rows.append(x.copy())
            return np.column_stack((x,np.zeros(len(x))))
        result,report=native.frontal_delaunay_refine(original,options(2),target_size=.5,
            metric_to_physical=owner,metric_jacobian=np.asarray(((1.,0.),(0.,1.),(0.,0.))),
            short_edge_offcentre=guarded)
        outcomes.append((result,report));owner_rows.append(rows)
    assert inputs and any(row[2] for row in inputs) and any(not row[2] for row in inputs)
    assert [r.tobytes() for r in owner_rows[0]]==[r.tobytes() for r in owner_rows[1]]
    for result,report in outcomes:
        assert report['topology_operations']<=2 and not report['insertions']
        assert report['geometry_limited_regions']>0
        assert result.points.tobytes()==original.points.tobytes()
        assert {tuple(sorted(e)) for e in result.segments}=={tuple(sorted(e)) for e in original.segments}
    assert outcomes[1][1]['short_edge_offcentre'] is True
    assert (original.points.tobytes(),original.triangles.tobytes(),original.segments.tobytes())==before


def test_owner_mode_cancellation_propagates_identity_and_seed_remains_exact():
    original=seed();before=original.points.tobytes(),original.triangles.tobytes(),original.segments.tobytes()
    error=RuntimeError('cancel owner offcentre queue')
    def cancel(phase):
        if phase=='native-v2 triangle queue processing':raise error
    with pytest.raises(RuntimeError) as caught:
        native.frontal_delaunay_refine(original,options(),target_size=.5,cancellation_check=cancel,
            metric_to_physical=lambda x:np.column_stack((x,np.zeros(len(x)))),
            metric_jacobian=np.asarray(((1.,0.),(0.,1.),(0.,0.))),short_edge_offcentre=True)
    assert caught.value is error
    assert (original.points.tobytes(),original.triangles.tobytes(),original.segments.tobytes())==before


@pytest.mark.parametrize('flag', (None,0,1,'yes'))
def test_frontal_flag_validation_occurs_before_owner_query(flag):
    calls=[]
    with pytest.raises(MeshError,match='option'):
        native.frontal_delaunay_refine(seed(),options(),target_size=.5,short_edge_offcentre=flag,
            metric_to_physical=lambda x:calls.append(x))
    assert not calls


def test_frontal_opt_in_requires_owner_coordinates():
    with pytest.raises(MeshError,match='owner coordinates'):
        native.frontal_delaunay_refine(seed(),options(),target_size=.5,short_edge_offcentre=True)


@pytest.mark.parametrize('analytic',(False,True))
def test_surface_opts_in_only_exact_bound_analytic_owner(monkeypatch,analytic):
    import anymesher.surface_mesh as surface
    import anymesher._frontal_transition_quality as transition
    import anymesher._physical_t3_refinement as physical
    from test_joint_triangle_owner_batches import analytic_chart
    from anygeometry import to_dict
    model,chart=analytic_chart();before=to_dict(model)
    seen=[];refine=surface.frontal_delaunay_refine
    def capture(*args,**kwargs):
        seen.append(kwargs['short_edge_offcentre'])
        return refine(*args,**kwargs)
    monkeypatch.setattr(surface,'frontal_delaunay_refine',capture)
    monkeypatch.setattr(transition,'repair_frontal_transition',lambda candidate,protected,settings,report,*args,**kwargs:
                        (candidate,dict(report,chart_transition_repair={})))
    monkeypatch.setattr(physical,'refine_physical_candidate',lambda candidate,triangulation,settings,report,*args,**kwargs:
                        (candidate,triangulation,report))
    surface.mesh_planar_surface(((0.,0.),(1.,0.),(1.,1.),(0.,1.)),
        options=surface.SurfaceMeshOptions(target_size=1.,recombine=False,backend='python',
            min_angle=45.,native_options=options()),
        _metric_to_physical=chart.evaluate if analytic else lambda x:chart.evaluate(x),
        _metric_jacobian=np.asarray(((1.,0.),(0.,1.),(0.,0.))),_preserve_spatial_refinement=True)
    assert seen==[analytic] and to_dict(model)==before
