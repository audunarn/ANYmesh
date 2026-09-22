"""CH1: shared Q4/Q8/T3/T6 interpolation and strict mapping validity."""
from __future__ import annotations

import json

import numpy as np
import pytest

from anymesher.quad.high_order import (
    ElementFamily,
    HighOrderCertificationCancelled,
    ValidityStatus,
    certify_mapping_validity,
    evaluate_mapping,
    geometry_error,
    normal_error,
    physical_area,
    shape_gradients,
    shape_values,
)


Q8_PARAM_NODES = np.array([
    [-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.],
    [0.,-1.],[1.,0.],[0.,1.],[-1.,0.],
])
T6_PARAM_NODES = np.array([
    [0.,0.],[1.,0.],[0.,1.],[.5,0.],[.5,.5],[0.,.5],
])


def _plane_q8() -> np.ndarray:
    return np.column_stack(((Q8_PARAM_NODES[:,0]+1)/2, (Q8_PARAM_NODES[:,1]+1)/2, np.zeros(8)))


def _plane_t6() -> np.ndarray:
    return np.column_stack((T6_PARAM_NODES, np.zeros(6)))

@pytest.mark.parametrize("family,points", [
    ("Q4", [[-.7,-.2],[0.,0.],[.4,.8]]),
    ("Q8", [[-.7,-.2],[0.,0.],[.4,.8]]),
    ("T3", [[.1,.1],[.2,.3],[.7,.1]]),
    ("T6", [[.1,.1],[.2,.3],[.7,.1]]),
])
def test_partition_of_unity_and_gradient_sum_zero(family: str, points) -> None:
    p = np.asarray(points, dtype=float)
    values = shape_values(family, p)
    grads = shape_gradients(family, p)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=2e-14)
    np.testing.assert_allclose(grads.sum(axis=1), 0.0, atol=3e-14)


@pytest.mark.parametrize("family,nodes", [("Q8",Q8_PARAM_NODES),("T6",T6_PARAM_NODES)])
def test_quadratic_kronecker_node_ordering(family: str, nodes: np.ndarray) -> None:
    values = shape_values(family, nodes)
    np.testing.assert_allclose(values, np.eye(len(nodes)), atol=2e-14)


@pytest.mark.parametrize("family,point", [
    ("Q4",np.array([.17,-.31])), ("Q8",np.array([.17,-.31])),
    ("T3",np.array([.23,.31])), ("T6",np.array([.23,.31])),
])
def test_analytic_shape_gradients_match_centered_difference(family: str, point: np.ndarray) -> None:
    eps = 2e-7
    analytic = shape_gradients(family, point)
    numeric = np.empty_like(analytic)
    for axis in range(2):
        delta = np.zeros(2); delta[axis] = eps
        numeric[:,axis] = (shape_values(family,point+delta)-shape_values(family,point-delta))/(2*eps)
    np.testing.assert_allclose(analytic, numeric, rtol=2e-8, atol=2e-9)

Q4_PARAM_NODES = Q8_PARAM_NODES[:4]
T3_PARAM_NODES = T6_PARAM_NODES[:3]


def _affine_nodes(family: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    params = {"Q4":Q4_PARAM_NODES,"Q8":Q8_PARAM_NODES,"T3":T3_PARAM_NODES,"T6":T6_PARAM_NODES}[family]
    origin = np.array([2.0,-1.0,.4])
    a = np.array([1.2,.3,.1])
    b = np.array([-.2,.8,.4])
    xyz = origin + params[:,0,None]*a + params[:,1,None]*b
    return xyz,origin,a,b


@pytest.mark.parametrize("family,points", [
    ("Q4",np.array([[-.3,.1],[.4,-.2]])), ("Q8",np.array([[-.3,.1],[.4,-.2]])),
    ("T3",np.array([[.2,.2],[.6,.1]])), ("T6",np.array([[.2,.2],[.6,.1]])),
])
def test_affine_reproduction_area_and_positive_certificate(family: str, points: np.ndarray) -> None:
    nodes,origin,a,b = _affine_nodes(family)
    ev = evaluate_mapping(nodes,family,points)
    expected = origin + points[:,0,None]*a + points[:,1,None]*b
    np.testing.assert_allclose(ev.points, expected, atol=3e-14)
    jac = np.linalg.norm(np.cross(a,b))
    np.testing.assert_allclose(ev.jacobian_magnitude, jac, rtol=2e-14)
    expected_area = (4.0 if family.startswith("Q") else 0.5)*jac
    assert physical_area(nodes,family) == pytest.approx(expected_area, rel=2e-13)
    report = certify_mapping_validity(nodes,family)
    assert report.status is ValidityStatus.CERTIFIED_POSITIVE
    assert report.lower_bound > report.tolerance

def test_rigid_transform_and_uniform_scale_invariance() -> None:
    nodes = _plane_q8()
    params = np.array([[-.6,-.2],[.1,.3],[.7,-.4]])
    base = evaluate_mapping(nodes,"Q8",params)
    area0 = physical_area(nodes,"Q8")
    status0 = certify_mapping_validity(nodes,"Q8").status
    r = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    scale = 3.25
    moved = (nodes @ r.T)*scale + np.array([4.,-2.,7.])
    transformed = evaluate_mapping(moved,"Q8",params)
    assert certify_mapping_validity(moved,"Q8").status is status0
    assert physical_area(moved,"Q8") == pytest.approx(area0*scale*scale, rel=3e-13)
    np.testing.assert_allclose(transformed.normalized_quality, base.normalized_quality, atol=3e-14)


def test_orientation_permutation_discipline_with_fixed_expected_normal() -> None:
    nodes = _plane_q8()
    cyclic = nodes[[1,2,3,0,5,6,7,4]]
    reversed_nodes = nodes[[0,3,2,1,7,6,5,4]]
    expected = np.array([0.,0.,1.])
    assert certify_mapping_validity(nodes,"Q8",reference_normal=expected).status is ValidityStatus.CERTIFIED_POSITIVE
    assert certify_mapping_validity(cyclic,"Q8",reference_normal=expected).status is ValidityStatus.CERTIFIED_POSITIVE
    reversed_report = certify_mapping_validity(reversed_nodes,"Q8",reference_normal=expected)
    assert reversed_report.status is ValidityStatus.INVALID
    assert reversed_report.witness_value is not None and reversed_report.witness_value < 0.0

    tri = _plane_t6()
    reversed_tri = tri[[0,2,1,5,4,3]]
    assert certify_mapping_validity(tri,"T6",reference_normal=expected).status is ValidityStatus.CERTIFIED_POSITIVE
    assert certify_mapping_validity(reversed_tri,"T6",reference_normal=expected).status is ValidityStatus.INVALID

def test_hidden_q8_midside_inversion_is_detected_beyond_skeleton_quality() -> None:
    from anymesher.quality_v2 import quad_quality

    good = _plane_q8()
    bad = good.copy()
    bad[4] = np.array([.5,1.2,0.0])
    conn = np.array([[0,1,2,3,4,5,6,7]], dtype=np.int64)
    q_good = quad_quality(good,conn)
    q_bad = quad_quality(bad,conn)
    np.testing.assert_allclose(q_good.area,q_bad.area)
    np.testing.assert_allclose(q_good.warpage,q_bad.warpage)

    interior = evaluate_mapping(bad,"Q8",np.array([[0.0,-.9],[0.0,-.5],[0.0,0.0]])).signed_jacobian
    assert np.min(interior) < 0.0
    report = certify_mapping_validity(bad,"Q8")
    assert report.status is ValidityStatus.INVALID
    assert report.witness_value is not None and report.witness_value < 0.0
    assert report.witness_parameter is not None


def test_near_degenerate_positive_mapping_is_unresolved_not_false_positive() -> None:
    h = 1.0e-15
    nodes = np.array([[0,0,0],[1,0,0],[1,h,0],[0,h,0]], dtype=float)
    report = certify_mapping_validity(nodes,"Q4",max_depth=3,max_subdivisions=128)
    assert report.status is ValidityStatus.UNRESOLVED
    assert report.witness_value is not None and report.witness_value > 0.0
    assert report.lower_bound <= report.tolerance


def test_geometry_and_normal_error_interfaces() -> None:
    nodes = _plane_q8()
    params = np.array([[-.5,-.2],[0.,0.],[.6,.4]])
    ev = evaluate_mapping(nodes,"Q8",params)
    exact = geometry_error(nodes,"Q8",params,ev.points)
    assert exact.maximum == pytest.approx(0.0,abs=1e-15)
    shifted = geometry_error(nodes,"Q8",params,ev.points+np.array([.01,0.,0.]))
    assert shifted.maximum > 0.0 and shifted.rms > 0.0

    aligned = normal_error(nodes,"Q8",params,ev.jacobian_vector)
    assert aligned.maximum_angle_rad == pytest.approx(0.0,abs=1e-15)
    tilted_ref = ev.jacobian_vector + np.array([.1,0.,0.])
    tilted = normal_error(nodes,"Q8",params,tilted_ref)
    assert tilted.maximum_angle_rad > 0.0 and tilted.rms_angle_rad > 0.0

def test_validity_report_records_conservative_method_and_witness() -> None:
    report = certify_mapping_validity(_plane_t6(),"T6")
    assert report.status is ValidityStatus.CERTIFIED_POSITIVE
    assert report.method == "bernstein-triangle-degree2"
    assert report.lower_bound > report.tolerance
    assert report.upper_bound >= report.lower_bound
    assert report.subdivisions >= 0 and report.max_depth_reached >= 0
    assert report.witness_value is not None and report.witness_parameter is not None


def test_invalid_shapes_and_nonfinite_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        shape_values("Q8",[0.0])
    with pytest.raises(ValueError):
        shape_gradients("BOGUS",[0.0,0.0])
    with pytest.raises(ValueError):
        evaluate_mapping(np.zeros((7,3)),"Q8",[[0.,0.]])
    bad = _plane_q8(); bad[0,0] = np.nan
    with pytest.raises(ValueError):
        certify_mapping_validity(bad,"Q8")
    with pytest.raises(ValueError):
        certify_mapping_validity(_plane_q8(),"Q8",reference_normal=[0.,0.,0.])


def test_q8_shape_values_match_existing_coupling_convention() -> None:
    from anymesher.coupling import shape_functions_8node
    for point in ((-.73,.18),(0.,0.),(.41,-.62),(.9,.7)):
        np.testing.assert_allclose(shape_values("Q8",point), shape_functions_8node(*point), atol=2e-15)


def test_hidden_t6_midside_inversion_is_detected() -> None:
    from anymesher.quality_v2 import triangle_quality
    good = _plane_t6()
    bad = good.copy(); bad[3] = np.array([.5,.3,0.])
    conn = np.array([[0,1,2,3,4,5]], dtype=np.int64)
    qa = triangle_quality(good,conn)
    qb = triangle_quality(bad,conn)
    np.testing.assert_allclose(qa.area,qb.area)
    np.testing.assert_allclose(qa.aspect_ratio,qb.aspect_ratio)
    interior = evaluate_mapping(bad,"T6",np.array([[.98,.01],[.95,.02]])).signed_jacobian
    assert np.min(interior) < 0.0
    report = certify_mapping_validity(bad,"T6")
    assert report.status is ValidityStatus.INVALID
    assert report.witness_value is not None and report.witness_value < 0.0


def _curved_positive_q8() -> np.ndarray:
    nodes = _plane_q8().copy()
    nodes[4:, 2] = np.array([0.08, 0.05, 0.07, 0.04])
    return nodes


def _curved_positive_t6() -> np.ndarray:
    nodes = _plane_t6().copy()
    nodes[3:, 2] = np.array([0.05, 0.04, 0.03])
    return nodes


def test_validity_report_json_serialization_is_deterministic() -> None:
    a = certify_mapping_validity(_curved_positive_q8(), "Q8", reference_normal=[0., 0., 1.])
    b = certify_mapping_validity(_curved_positive_q8(), "Q8", reference_normal=[0., 0., 1.])
    assert a == b
    da, db = a.to_dict(), b.to_dict()
    assert da == db
    assert da["status"] == "CERTIFIED_POSITIVE"
    assert da["witness_parameter"] is None or isinstance(da["witness_parameter"], list)
    assert json.dumps(da, sort_keys=True) == json.dumps(db, sort_keys=True)


def test_curved_q8_certificate_bounds_dense_independent_samples() -> None:
    nodes = _curved_positive_q8()
    report = certify_mapping_validity(nodes, "Q8", reference_normal=[0., 0., 1.])
    assert report.status is ValidityStatus.CERTIFIED_POSITIVE
    grid = np.linspace(-1.0, 1.0, 17)
    params = np.array([(x, y) for x in grid for y in grid], dtype=float)
    signed = evaluate_mapping(nodes, "Q8", params, reference_normal=[0., 0., 1.]).signed_jacobian
    assert float(np.min(signed)) >= report.lower_bound - report.tolerance
    assert float(np.max(signed)) <= report.upper_bound + report.tolerance


def test_curved_t6_certificate_bounds_dense_independent_samples() -> None:
    nodes = _curved_positive_t6()
    report = certify_mapping_validity(nodes, "T6", reference_normal=[0., 0., 1.])
    assert report.status is ValidityStatus.CERTIFIED_POSITIVE
    params = np.array([(i / 16.0, j / 16.0) for i in range(17) for j in range(17 - i)], dtype=float)
    signed = evaluate_mapping(nodes, "T6", params, reference_normal=[0., 0., 1.]).signed_jacobian
    assert float(np.min(signed)) >= report.lower_bound - report.tolerance
    assert float(np.max(signed)) <= report.upper_bound + report.tolerance


def test_cancellation_and_budget_do_not_mutate_nodes() -> None:
    nodes = _curved_positive_q8()
    before = nodes.copy()
    calls: list[str] = []

    def cancel(stage: str) -> bool:
        calls.append(stage)
        return True

    with pytest.raises(HighOrderCertificationCancelled):
        certify_mapping_validity(nodes, "Q8", reference_normal=[0., 0., 1.], cancellation_check=cancel)
    assert calls
    np.testing.assert_array_equal(nodes, before)

    unresolved = certify_mapping_validity(
        nodes, "Q8", reference_normal=[0., 0., 1.], max_depth=0, max_subdivisions=0
    )
    assert unresolved.status in (ValidityStatus.CERTIFIED_POSITIVE, ValidityStatus.UNRESOLVED)
    np.testing.assert_array_equal(nodes, before)


def _refinement_sensitive_q8() -> np.ndarray:
    return np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.6839453660318275, -0.26137527417896605, 0.0],
        [0.9664520122785643, 0.6294242495768334, 0.0],
        [0.6581510148714178, 1.1371371615319996, 0.0],
        [0.21607649591507117, 0.7393370330089456, 0.0],
    ], dtype=float)


def test_subdivision_budget_can_refine_unresolved_to_certified() -> None:
    nodes = _refinement_sensitive_q8()
    before = nodes.copy()
    shallow = certify_mapping_validity(
        nodes, "Q8", reference_normal=[0., 0., 1.], max_depth=0, max_subdivisions=0
    )
    deep = certify_mapping_validity(
        nodes, "Q8", reference_normal=[0., 0., 1.], max_depth=6, max_subdivisions=4096
    )
    assert shallow.status is ValidityStatus.UNRESOLVED
    assert deep.status is ValidityStatus.CERTIFIED_POSITIVE
    assert deep.subdivisions > shallow.subdivisions
    np.testing.assert_array_equal(nodes, before)


def test_invalid_witness_is_never_overturned_by_more_budget() -> None:
    bad = _plane_q8()
    bad[4] = np.array([.5, 1.2, 0.0])
    shallow = certify_mapping_validity(
        bad, "Q8", max_depth=0, max_subdivisions=0
    )
    deep = certify_mapping_validity(
        bad, "Q8", max_depth=8, max_subdivisions=8192
    )
    assert shallow.status is ValidityStatus.INVALID
    assert deep.status is ValidityStatus.INVALID
    assert shallow.witness_value is not None and shallow.witness_value <= 0.0
    assert deep.witness_value is not None and deep.witness_value <= 0.0
