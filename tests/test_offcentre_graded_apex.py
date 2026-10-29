"""Pure proposal controls: apex allowance is not final physical qualification."""
import numpy as np
import pytest
import anymesher.native_v2 as native
from anymesher.errors import MeshError


def operands(metric=np.eye(2), scale=1., rotation=0.):
    transform = np.linalg.cholesky(metric)
    whitened = np.array(((0.,0.),(.1,0.),(.05,2.)))
    points = np.linalg.solve(transform.T,whitened.T).T
    c,s = np.cos(rotation),np.sin(rotation)
    frame = scale*np.array(((c,-s),(s,c)))
    inverse = np.linalg.inv(frame)
    return points @ frame, np.broadcast_to(inverse @ metric @ inverse.T,(3,2,2)).copy()


def metric_angles(points,metric):
    values = []
    for index,point in enumerate(points):
        a,b = points[(index+1)%3]-point,points[(index+2)%3]-point
        cosine = float(a @ metric @ b / np.sqrt((a @ metric @ a)*(b @ metric @ b)))
        values.append(float(np.degrees(np.arccos(np.clip(cosine,-1.,1.)))))
    return values


@pytest.mark.parametrize('angle',(15.,30.,45.,60.))
@pytest.mark.parametrize('metric',(np.eye(2),np.array(((4.,1.2),(1.2,2.)))))
def test_apex_allowance_permits_actual_metric_coarsening(angle,metric):
    points,tensors = operands(metric)
    before = points.tobytes(),tensors.tobytes()
    old = native._offcentre(points,tensors,10.,short_edge_metric=True)
    new = native._offcentre(points,tensors,10.,short_edge_metric=True,_minimum_angle_target=angle)
    length = lambda a,b: float(np.sqrt((a-b) @ metric @ (a-b)))
    base = length(*points[:2])
    assert length(old,points[0]) == pytest.approx(base,rel=2e-14)
    if angle<60.:
        assert length(new,points[0]) > base
    assert min(metric_angles(np.array((points[0],points[1],new)),metric)) == pytest.approx(angle,abs=2e-12)
    assert (points.tobytes(),tensors.tobytes()) == before


@pytest.mark.parametrize('scale',(2.**-12,.5,2.))
@pytest.mark.parametrize('rotation',(.3,1.2))
def test_rotated_scaled_spd_proposal_is_covariant(scale,rotation):
    metric = np.array(((4.,1.2),(1.2,2.)))
    points,tensors = operands(metric)
    base = native._offcentre(points,tensors,10.,short_edge_metric=True,_minimum_angle_target=30.)
    other,other_tensors = operands(metric,scale,rotation)
    actual = native._offcentre(other,other_tensors,10.*scale,short_edge_metric=True,_minimum_angle_target=30.)
    c,s = np.cos(rotation),np.sin(rotation)
    np.testing.assert_allclose(actual,base @ (scale*np.array(((c,-s),(s,c)))),rtol=3e-14,atol=1e-17*scale)


def test_desired_metric_and_chart_fallback_still_cap_height():
    points = np.array(((0.,0.),(.1,0.),(.05,2.)))
    tensors = np.broadcast_to(np.eye(2),(3,2,2)).copy()
    limited = native._offcentre(points,tensors,.07,short_edge_metric=True,_minimum_angle_target=30.)
    assert np.linalg.norm(limited-points[0]) <= .07
    large = points*10.
    desired = native._offcentre(large,tensors,10.,short_edge_metric=True,_minimum_angle_target=30.)
    assert np.linalg.norm(desired-large[0]) <= 1.


@pytest.mark.parametrize('angle',(True,False,0.,-1.,90.,np.nan,np.inf,'30'))
def test_invalid_apex_angle_refuses_without_operand_mutation(angle):
    points,tensors = operands();before = points.tobytes(),tensors.tobytes()
    with pytest.raises(MeshError):
        native._offcentre(points,tensors,.5,short_edge_metric=True,_minimum_angle_target=angle)
    assert (points.tobytes(),tensors.tobytes()) == before


def test_explicit_none_preserves_exact_old_private_operator_and_legacy_path():
    points,tensors = operands(np.array(((4.,1.2),(1.2,2.))))
    old = native._offcentre(points,tensors,.5,short_edge_metric=True)
    assert native._offcentre(points,tensors,.5,short_edge_metric=True,_minimum_angle_target=None).tobytes() == old.tobytes()
    legacy = native._offcentre(points,tensors,.5)
    assert native._offcentre(points,tensors,.5,_minimum_angle_target=30.).tobytes() == legacy.tobytes()


@pytest.mark.parametrize('angle',(60.,75.,89.))
def test_sixty_and_higher_targets_retain_exact_old_equilateral_cap(angle):
    points,tensors = operands(np.array(((4.,1.2),(1.2,2.))))
    old = native._offcentre(points,tensors,.5,short_edge_metric=True)
    assert native._offcentre(points,tensors,.5,short_edge_metric=True,
                             _minimum_angle_target=angle).tobytes() == old.tobytes()


def test_subnormal_positive_angle_is_a_nonbinding_finite_proposal():
    points,tensors = operands()
    result = native._offcentre(points,tensors,10.,short_edge_metric=True,
                                _minimum_angle_target=np.nextafter(0.,1.))
    assert np.isfinite(result).all()


@pytest.mark.parametrize('guarded',(False,True))
@pytest.mark.parametrize('angle',(30.,60.,75.))
def test_frontal_forwards_angle_only_to_short_edge_proposal_and_preserves_failure(monkeypatch,guarded,angle):
    from test_owner_short_edge_offcentre import seed,options
    original = seed()
    before = original.points.tobytes(),original.triangles.tobytes(),original.segments.tobytes()
    seen = []
    error = RuntimeError('original offcentre proposal failure')
    def fail(points,tensors,size,**kwargs):
        seen.append(kwargs)
        raise error
    monkeypatch.setattr(native,'_offcentre',fail)
    with pytest.raises(RuntimeError) as caught:
        native.frontal_delaunay_refine(original,options(2),target_size=.5,
            metric_to_physical=lambda x:np.column_stack((x,np.zeros(len(x)))),
            metric_jacobian=np.array(((1.,0.),(0.,1.),(0.,0.))),
            short_edge_offcentre=guarded,minimum_angle_target=angle)
    assert caught.value is error and seen
    assert all(row == ({'short_edge_metric':True,'_minimum_angle_target':angle} if guarded else {})
               for row in seen)
    assert (original.points.tobytes(),original.triangles.tobytes(),original.segments.tobytes()) == before


@pytest.mark.parametrize('metric',(np.diag((0.,1.)),np.array(((1.,2.),(2.,1.)))))
def test_angle_proposal_still_refuses_invalid_spd_without_mutation(metric):
    points = np.array(((0.,0.),(.1,0.),(.05,2.)))
    tensors = np.broadcast_to(metric,(3,2,2)).copy()
    before = points.tobytes(),tensors.tobytes()
    with pytest.raises(MeshError):
        native._offcentre(points,tensors,.5,short_edge_metric=True,_minimum_angle_target=30.)
    assert (points.tobytes(),tensors.tobytes()) == before
