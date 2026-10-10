"""Submodel point correspondence on a cylinder (slice S3b): parameter-plane location."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.mesh import Mesh
from anymesher.submodel_correspondence import CorrespondenceError, CylinderFrame, locate_for_transfer

R = 1.0
TOL = 1e-9


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class Frame:
    """A cylinder of radius R, rotated and translated, with its own (theta, z) map."""

    def __init__(self, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), angle0=0.0):
        self.origin = np.asarray(origin, float)
        self.rotation = rotation(np.cross([0, 0, 1], axis) if np.linalg.norm(np.cross([0, 0, 1], axis)) > 0 else [1, 0, 0],
                                 np.arccos(np.clip(np.dot([0, 0, 1], np.asarray(axis) / np.linalg.norm(axis)), -1, 1)))
        self.angle0 = angle0
        self.cylinder = CylinderFrame(tuple(self.origin), tuple(self.rotation @ [0, 0, 1]), R, TOL)

    def point(self, theta, z, radius=R):
        local = np.array([radius * np.cos(theta + self.angle0), radius * np.sin(theta + self.angle0), z])
        return self.origin + self.rotation @ local


def strip_mesh(frame, n_theta=8, n_z=3, theta0=-np.pi / 4, theta1=np.pi / 4, z0=0.0, z1=1.0, triangles=False):
    nodes, quads, tris, ids = {}, {}, {}, {}
    k = 1
    for j in range(n_z + 1):
        for i in range(n_theta + 1):
            nodes[k] = frame.point(theta0 + (theta1 - theta0) * i / n_theta, z0 + (z1 - z0) * j / n_z)
            ids[i, j] = k
            k += 1
    e = 1
    for j in range(n_z):
        for i in range(n_theta):
            a, b, c, d = ids[i, j], ids[i + 1, j], ids[i + 1, j + 1], ids[i, j + 1]
            if triangles:
                for conn in ((a, b, c), (a, c, d)) if (i + j) % 2 else ((a, b, d), (b, c, d)):
                    tris[e] = conn
                    e += 1
            else:
                quads[e] = (a, b, c, d)
                e += 1
    mesh = Mesh(nodes=nodes, quads=quads, tris=tris)
    mesh.elements_of_face = {1: list(mesh.shells)}
    return mesh


def everything(mesh, keys):
    return {key: tuple(mesh.shells) for key in keys}


def test_two_triangles_sharing_an_edge_conform_on_the_parametric_edge():
    """Round-1 review counterexample: both triangles' projections contain P; parameter space does not."""

    h = np.pi / 12
    nodes = {1: np.array([np.cos(h), -np.sin(h), 0.0]), 2: np.array([np.cos(h), np.sin(h), 0.0]),
             3: np.array([1.0, 0.0, 1.0]), 4: np.array([1.0, 0.0, -1.0])}
    mesh = Mesh(nodes=nodes, tris={1: (1, 2, 3), 2: (1, 2, 4)})
    mesh.elements_of_face = {1: [1, 2]}
    cylinder = CylinderFrame((0, 0, 0), (0, 0, 1), 1.0, TOL)
    hits = locate_for_transfer(mesh, {10: (1.0, 0.0, 0.0)}, candidate_elements={10: (1, 2)}, cylinder=cylinder)
    hit = hits[10]
    weights = dict(zip(hit.source_node_ids, hit.weights))
    assert weights.get(3, 0.0) == pytest.approx(0.0, abs=1e-12) and weights.get(4, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert weights[1] == pytest.approx(0.5, abs=1e-12) and weights[2] == pytest.approx(0.5, abs=1e-12)
    assert hit.parametric and hit.distance <= hit.bound + hit.float_allowance
    # the planar route refuses the same conforming mesh
    with pytest.raises(CorrespondenceError, match="NONCONFORMING"):
        locate_for_transfer(mesh, {10: (1.0, 0.0, 0.0)}, candidate_elements={10: (1, 2)}, physical_tolerance=0.1)


def test_axial_point_outside_the_parametric_element_is_not_located():
    """Round-1 review counterexample 2: projects inside the triangle, outside its parameter domain."""

    h = np.pi / 12
    nodes = {1: np.array([np.cos(h), -np.sin(h), 0.0]), 2: np.array([np.cos(h), np.sin(h), 0.0]),
             3: np.array([1.0, 0.0, 0.001])}
    mesh = Mesh(nodes=nodes, tris={1: (1, 2, 3)})
    mesh.elements_of_face = {1: [1]}
    cylinder = CylinderFrame((0, 0, 0), (0, 0, 1), 1.0, TOL)
    with pytest.raises(CorrespondenceError, match="NOT_LOCATED"):
        locate_for_transfer(mesh, {10: (1.0, 0.0, -0.1)}, candidate_elements={10: (1,)}, cylinder=cylinder)


@pytest.mark.parametrize("triangles", [False, True])
@pytest.mark.parametrize("frame", [Frame(), Frame(origin=(2.0, -1.0, 0.5), axis=(1.0, 2.0, 0.5), angle0=0.7)],
                         ids=["aligned", "rotated"])
def test_random_points_lie_within_the_facet_bound_and_reproduce_linear_fields(frame, triangles):
    mesh = strip_mesh(frame, triangles=triangles)
    rng = np.random.default_rng(5)
    params = [(rng.uniform(-np.pi / 4, np.pi / 4), rng.uniform(0.0, 1.0)) for _ in range(40)]
    points = {100 + k: frame.point(t, z) for k, (t, z) in enumerate(params)}
    hits = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points), cylinder=frame.cylinder)
    span = (np.pi / 2) / 8
    for key, hit in hits.items():
        assert hit.parametric and hit.distance <= hit.bound + hit.float_allowance
        assert hit.bound <= R * span ** 2 / 8 + 1e-12  # exact nodes: the a-priori bound holds
        # x_p is the element's own map at the hit
        x_p = sum(w * mesh.nodes[n] for w, n in zip(hit.weights, hit.source_node_ids))
        assert np.allclose(x_p, hit.x_p, atol=1e-15)
        # the hit's (theta, z) reproduce the target's: parametric weights reproduce linear (theta, z) fields
        r, theta, z = frame.cylinder.coordinates(np.array([mesh.nodes[n] for n in hit.source_node_ids])).T
        r_f, theta_f, z_f = frame.cylinder.coordinates(np.asarray(points[key]))[0]
        theta = theta_f + np.angle(np.exp(1j * (theta - theta_f)))
        assert np.dot(hit.weights, z) == pytest.approx(z_f, abs=1e-12)
        assert np.dot(hit.weights, theta) == pytest.approx(theta_f, abs=1e-12)


def test_arc_midpoints_locate_without_a_physical_tolerance_and_across_the_wrap():
    frame = Frame()
    mesh = strip_mesh(frame, n_theta=6, theta0=np.pi - np.pi / 4, theta1=np.pi + np.pi / 4)  # straddles +-pi
    span = (np.pi / 2) / 6
    sag = R * (1 - np.cos(span / 2))
    assert sag > 1e5 * TOL
    mids = {200 + i: frame.point(np.pi - np.pi / 4 + span * (i + 0.5), 0.5) for i in range(6)}
    hits = locate_for_transfer(mesh, mids, candidate_elements=everything(mesh, mids), cylinder=frame.cylinder)
    assert len(hits) == 6
    for hit in hits.values():
        assert hit.distance == pytest.approx(sag, rel=1e-6) and hit.distance <= hit.bound + hit.float_allowance


def test_opposite_radial_deviations_stay_within_the_computed_bound():
    frame = Frame()
    span = 0.01
    deviation = 0.9 * TOL
    nodes = {1: frame.point(0.0, 0.0, R - deviation), 2: frame.point(span, 0.0, R - deviation),
             3: frame.point(span, 1.0, R - deviation), 4: frame.point(0.0, 1.0, R - deviation)}
    mesh = Mesh(nodes=nodes, quads={1: (1, 2, 3, 4)})
    mesh.elements_of_face = {1: [1]}
    target = frame.point(span / 2, 0.5, R + deviation)
    hit = locate_for_transfer(mesh, {10: target}, candidate_elements={10: (1,)}, cylinder=frame.cylinder)[10]
    assert hit.distance > hit.a_priori_bound + TOL  # beyond the exact-node bound plus one tolerance
    assert hit.distance <= hit.bound + hit.float_allowance


def test_points_just_inside_and_outside_an_edge_within_the_slack():
    frame = Frame()
    mesh = strip_mesh(frame, n_theta=2, n_z=1, theta0=0.0, theta1=0.2)
    natural = 1e-9
    inside = frame.point(0.2 - 0.05 * natural, 0.5)
    outside = frame.point(0.2 + 50.0 * natural, 0.5)
    hit = locate_for_transfer(mesh, {1: inside}, candidate_elements={1: tuple(mesh.shells)}, cylinder=frame.cylinder,
                              natural_tolerance=natural)[1]
    assert hit.distance <= hit.bound + hit.float_allowance
    with pytest.raises(CorrespondenceError, match="NOT_LOCATED"):
        locate_for_transfer(mesh, {1: outside}, candidate_elements={1: tuple(mesh.shells)}, cylinder=frame.cylinder,
                            natural_tolerance=natural)


def test_nonmatching_triangulated_source_under_quad_targets():
    frame = Frame()
    source = strip_mesh(frame, n_theta=5, n_z=2, triangles=True)
    target = strip_mesh(frame, n_theta=13, n_z=7)
    points = {100 + n: x for n, x in target.nodes.items()}
    hits = locate_for_transfer(source, points, candidate_elements=everything(source, points), cylinder=frame.cylinder)
    assert len(hits) == len(points)
    assert all(h.distance <= h.bound + h.float_allowance and h.weight_disagreement <= 1e-12 for h in hits.values())


@pytest.mark.parametrize("case, code", [
    ("point_off", "OFF_SURFACE"), ("node_off", "OFF_SURFACE"), ("too_coarse", "FACET_TOO_COARSE"),
    ("degenerate", "DEGENERATE"),
])
def test_refusals(case, code):
    frame = Frame()
    mesh = strip_mesh(frame, n_theta=4, n_z=1)
    point = frame.point(0.1, 0.5)
    if case == "point_off":
        point = frame.point(0.1, 0.5, R * 1.001)
    elif case == "node_off":
        first = next(iter(mesh.nodes))
        mesh.nodes[first] = mesh.nodes[first] * 1.001
    elif case == "too_coarse":
        mesh = strip_mesh(frame, n_theta=1, n_z=1, theta0=-0.3 * np.pi, theta1=0.3 * np.pi)
        point = frame.point(0.0, 0.5)
    elif case == "degenerate":
        nodes = {1: frame.point(0.0, 0.0), 2: frame.point(0.1, 0.0), 3: frame.point(0.2, 0.0)}
        mesh = Mesh(nodes=nodes, tris={1: (1, 2, 3)})
        mesh.elements_of_face = {1: [1]}
        point = frame.point(0.1, 0.0)
    with pytest.raises(CorrespondenceError, match=code):
        locate_for_transfer(mesh, {1: point}, candidate_elements={1: tuple(mesh.shells)}, cylinder=frame.cylinder)


def test_planar_route_is_unchanged_without_a_cylinder():
    frame = Frame()
    mesh = strip_mesh(frame, n_theta=8)
    point = frame.point(np.pi / 32, 0.5)  # an arc midpoint, far off the chord
    with pytest.raises(CorrespondenceError, match="NOT_LOCATED"):
        locate_for_transfer(mesh, {1: point}, candidate_elements={1: tuple(mesh.shells)})


def test_the_prefilter_keeps_every_point_the_inside_test_admits():
    """Thin T3 near a vertex with two negative weights, and Q4 corner extrapolation."""

    frame = Frame()
    t = 1e-6
    # A thin triangle in the parameter plane: p0 = (0, 0), p1 = (L, H), p2 = (L, -H), H << L.
    L, H = 0.2, 0.002
    nodes = {1: frame.point(0.0, 0.5), 2: frame.point(L, 0.5 + H), 3: frame.point(L, 0.5 - H)}
    mesh = Mesh(nodes=nodes, tris={1: (1, 2, 3)})
    mesh.elements_of_face = {1: [1]}
    xi = eta = -0.75 * t  # passes the inside test with slack t
    param = (1 - xi - eta) * np.array([0.0, 0.5]) + xi * np.array([L, 0.5 + H]) + eta * np.array([L, 0.5 - H])
    hit = locate_for_transfer(mesh, {1: frame.point(param[0], param[1])}, candidate_elements={1: (1,)},
                              cylinder=frame.cylinder, natural_tolerance=t)[1]
    assert min(hit.weights) < 0 and hit.distance <= hit.bound + hit.float_allowance
    quad = strip_mesh(frame, n_theta=1, n_z=1, theta0=0.0, theta1=0.1, z0=0.0, z1=0.1)
    corner = frame.point(0.1 + 0.5 * t * 0.05, 0.1 + 0.5 * t * 0.05)  # beyond the corner, within the slack
    hit = locate_for_transfer(quad, {1: corner}, candidate_elements={1: (1,)}, cylinder=frame.cylinder,
                              natural_tolerance=t)[1]
    assert max(abs(v) for v in hit.natural) > 1.0


@pytest.mark.parametrize("case", ["generator_edge", "tiny_span", "far_translated"])
def test_valid_hits_survive_roundoff(case):
    if case == "far_translated":
        frame = Frame(origin=(1.0e3, -2.0e3, 5.0e2), axis=(0.3, -1.0, 2.0), angle0=2.0)
    else:
        frame = Frame()
    span = 1e-7 if case == "tiny_span" else np.pi / 16
    mesh = strip_mesh(frame, n_theta=2, n_z=2, theta0=0.0, theta1=2 * span)
    points = {1: frame.point(span, 0.3), 2: frame.point(0.0, 0.7), 3: frame.point(0.5 * span, 0.5)}
    hits = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points), cylinder=frame.cylinder)
    assert len(hits) == 3


def test_an_injected_discrepancy_beyond_the_allowance_refuses(monkeypatch):
    import anymesher.submodel_correspondence as module

    frame = Frame()
    mesh = strip_mesh(frame, n_theta=4, n_z=1)
    real = module._element_map

    def shifted(weights, coordinates):
        return real(weights, coordinates) + np.array([1e-9, 0.0, 0.0])  # x_p no longer the parameter map's

    monkeypatch.setattr(module, "_element_map", shifted)
    with pytest.raises(CorrespondenceError, match="FACET_BOUND"):
        # at a node the computed bound is ~0, so a 1e-9 m inconsistency exceeds bound plus allowance
        locate_for_transfer(mesh, {1: mesh.nodes[next(iter(mesh.nodes))]}, candidate_elements={1: tuple(mesh.shells)},
                            cylinder=frame.cylinder)


def test_one_shot_candidate_iterables_give_the_same_hits():
    frame = Frame()
    mesh = strip_mesh(frame)
    points = {1: frame.point(0.05, 0.3), 2: frame.point(-0.2, 0.8)}
    as_tuples = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points), cylinder=frame.cylinder)
    as_generators = locate_for_transfer(mesh, points, candidate_elements={k: (e for e in mesh.shells) for k in points},
                                        cylinder=frame.cylinder)
    assert as_generators == as_tuples


@pytest.mark.parametrize("physical, padding", [(-1.0, None), (1.0, 0.5), (float("nan"), None)])
def test_planar_controls_do_not_affect_the_cylindrical_route(physical, padding):
    frame = Frame()
    mesh = strip_mesh(frame)
    points = {1: frame.point(0.05, 0.3)}
    with pytest.raises(CorrespondenceError, match="TOLERANCE"):
        locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points), physical_tolerance=physical,
                            bounds_padding=padding)
    hits = locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points), physical_tolerance=physical,
                               bounds_padding=padding, cylinder=frame.cylinder)
    assert hits == locate_for_transfer(mesh, points, candidate_elements=everything(mesh, points), cylinder=frame.cylinder)


def test_a_nonfinite_candidate_refuses_even_beside_a_valid_one():
    frame = Frame()
    mesh = strip_mesh(frame, n_theta=2, n_z=1, theta0=0.0, theta1=0.2)
    bad = max(mesh.nodes) + 1
    mesh.nodes[bad] = np.array([np.nan, 0.0, 0.0])
    extra = max(mesh.shells) + 1
    mesh.tris = {extra: (1, 2, bad)}
    mesh.elements_of_face = {1: list(mesh.shells)}
    with pytest.raises(CorrespondenceError, match="NONFINITE"):
        locate_for_transfer(mesh, {1: frame.point(0.05, 0.5)}, candidate_elements={1: tuple(mesh.shells)},
                            cylinder=frame.cylinder)


def test_position_uncertainty_bounds_the_physical_error_on_a_distorted_quad(monkeypatch):
    """A non-affine Q4 in the parameter plane, with a controlled residual in the natural coordinates."""

    import anymesher.submodel_correspondence as module

    frame = Frame()
    corners = [(0.0, 0.0), (0.24, 0.05), (0.30, 0.42), (-0.03, 0.30)]  # (theta, z), a distorted quad
    nodes = {k + 1: frame.point(theta, z) for k, (theta, z) in enumerate(corners)}
    mesh = Mesh(nodes=nodes, quads={1: (1, 2, 3, 4)})
    mesh.elements_of_face = {1: [1]}
    target = frame.point(0.13, 0.19)
    real = module._inverse_2d
    shift = np.array([3e-8, -2e-8])

    def perturbed(kind, local, point, size, element_id):
        natural, _weights, _residual = real(kind, local, point, size, element_id)
        natural = natural + shift
        return natural, module.shape_functions_q4(*natural), 0.0

    exact = locate_for_transfer(mesh, {1: target}, candidate_elements={1: (1,)}, cylinder=frame.cylinder,
                                natural_tolerance=1e-6)[1]
    monkeypatch.setattr(module, "_inverse_2d", perturbed)
    hit = locate_for_transfer(mesh, {1: target}, candidate_elements={1: (1,)}, cylinder=frame.cylinder,
                              natural_tolerance=1e-6)[1]
    error = float(np.linalg.norm(np.asarray(hit.x_p) - np.asarray(exact.x_p)))
    assert error > 0.0
    assert error <= hit.inverse_residual
    assert hit.inverse_residual <= 20.0 * error  # conservative, not vacuous
