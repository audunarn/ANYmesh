"""Support-frame face queries for an owner Cylinder with a distinct parameterization.

MESH-B option A2. On the owner-certified trimmed-cylinder route the quad-first
mesher works entirely in the support frame: chart coordinates are the support's
normalized ``(angle, height)``, points and derivatives come from the support,
and the face's own parameterization (for example a topology Coons map kept for
display) is never consulted. The owner's trims are support-frame curves and the
face map lies on that support, so the material region is the same.

Each method below reproduces, call for call, what ANYgeometry's GeometryModel
returns for the same face *without* its parameterization (the "twin"), using
the same owner helpers in the same order. A mesh of the parameterized face is
therefore bit-identical to the twin's mesh. The equivalence is tested in
tests/test_cylinder_support_frame_parameterization.py; a change to the owner's
built-in Cylinder face path must be mirrored here.
"""
from __future__ import annotations

import math

import numpy as np
from anygeometry import EllipticArc, GeometryError, GeometryModel
from anygeometry.curves import Straight
from anygeometry.surfaces import CoonsSurface, Cylinder, _evaluate_surface_many, _surface_derivatives_many

from .errors import MeshError

# GeometryModel.face_trim_loops_uv default, used by project_to_face(_many).
_TRIM_CURVE_SAMPLES = 17

# Ruled-strip orientation certificate (MESH-B fix round 2), see _certify_ruled_strip.
# Samples of the rail height difference h(w); between samples h is bounded below by
# the smaller sample minus max|h''| / (8 K^2).
_STRIP_SAMPLES = 64
# The certified lower bound of the face normal's component along the support
# normal must keep at least this fraction of its ideal (exact-family) minimum.
_NORMAL_MARGIN = 0.5
# Rounding allowance added to every float residual and sample, relative to the
# face's coordinate scale (|centre - origin| + |U| + |V| + radius).
_ROUNDING = 64.0 * float(np.finfo(float).eps)

_FAMILY_REFUSAL = ("support-frame cylinder chart admits only a ruled cylinder strip parameterization "
                   "(two opposite sides one EllipticArc edge each, the other two one Straight edge each); "
                   "the orientation of any other face parameterization relative to its support is undecided")
_CERTIFICATE_REFUSAL = ("support-frame cylinder chart cannot certify that the face parameterization keeps "
                        "the support orientation and lies on its support over the whole face")


def _signed_area(loop) -> float:
    points = np.asarray(loop, dtype=float)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _strip_rail(curve, forward: bool, fraction_reversed: bool, surface) -> dict:
    """One EllipticArc side of a ruled strip as a rail w -> point, w in [0, 1].

    The topology Coons map evaluates a one-edge side at chain fraction ``w``
    (sides 0, 1) or ``1 - w`` (sides 2, 3); the edge parameter is the
    fraction, or one minus it for a backward edge. On the arc the angle is
    ``phi = phi_start + w * delta``. Relative to the support (origin o, unit
    axis a, radial r0, c0 = a x r0, radius R) the arc is
    ``o + z(w) a + R e_r(theta(w)) + rho(w)``, with ``theta = gamma + sigma*phi``
    affine in w, ``z`` its exact axial coordinate and ``rho`` perpendicular to
    the axis, bounded by ``delta_pos`` and its w-derivative by ``delta_der``.
    """
    increasing = bool(forward) != bool(fraction_reversed)
    phi_start = curve.start_angle if increasing else curve.start_angle + curve.sweep_angle
    delta = curve.sweep_angle if increasing else -curve.sweep_angle
    o, a = np.asarray(surface.origin, dtype=float), np.asarray(surface.axis, dtype=float)
    r0 = np.asarray(surface.radial_direction, dtype=float)
    c0 = np.cross(a, r0)
    radius = float(surface.radius)
    centre = np.asarray(curve.center, dtype=float) - o
    u_vec, v_vec = np.asarray(curve.u_vector, dtype=float), np.asarray(curve.v_vector, dtype=float)
    c_a, u_a, v_a = float(centre @ a), float(u_vec @ a), float(v_vec @ a)
    c_perp, u_perp, v_perp = centre - c_a * a, u_vec - u_a * a, v_vec - v_a * a
    turn = float(a @ np.cross(u_perp, v_perp))
    if not math.isfinite(turn) or turn == 0.0 or not math.isfinite(delta) or delta == 0.0:
        raise MeshError(_CERTIFICATE_REFUSAL)
    sigma = 1.0 if turn > 0.0 else -1.0
    gamma = math.atan2(float(u_perp @ c0), float(u_perp @ r0))
    u_star = radius * (math.cos(gamma) * r0 + math.sin(gamma) * c0)
    v_star = sigma * radius * (-math.sin(gamma) * r0 + math.cos(gamma) * c0)
    d_u, d_v = float(np.linalg.norm(u_perp - u_star)), float(np.linalg.norm(v_perp - v_star))
    # Arc evaluation rounds at the absolute world coordinates, not only at the local offsets.
    world = float(np.linalg.norm(o) + np.linalg.norm(np.asarray(curve.center, dtype=float)))
    scale = float(np.linalg.norm(centre) + np.linalg.norm(u_vec) + np.linalg.norm(v_vec)) + radius + world
    tau = _ROUNDING * scale
    amplitude = math.hypot(u_a, v_a)
    return dict(
        phi_start=phi_start, delta=delta, c_a=c_a, u_a=u_a, v_a=v_a,
        theta_start=gamma + sigma * phi_start, beta=sigma * delta, scale=scale,
        delta_pos=float(np.linalg.norm(c_perp)) + d_u + d_v + tau,
        delta_der=abs(delta) * (d_u + d_v) + tau * (1.0 + abs(delta)),
        z_slope=abs(delta) * amplitude, z_curvature=delta * delta * amplitude,
    )


def _rail_heights(rail: dict, w: np.ndarray) -> np.ndarray:
    phi = rail["phi_start"] + w * rail["delta"]
    return rail["c_a"] + rail["u_a"] * np.cos(phi) + rail["v_a"] * np.sin(phi)


def _certify_ruled_strip(owner, face, surface) -> dict:
    """Certify a topology Coons map on a Cylinder support as an unfolded strip on that support.

    Admitted family: four corners, every side exactly one edge; one pair of
    opposite sides EllipticArc edges (the rails), the other pair Straight
    edges. Anything else raises MeshError (undecided, fail closed).

    Proof. The Coons map (ANYgeometry ``_topology_face_point_and_derivatives``)
    is P = (1-v)A(u) + vC(1-u) + (1-u)D(1-v) + uB(v) - bilinear(corners). A
    Straight side is linear between its vertex positions, so with Straight
    sides 1 and 3 the D/B terms cancel the bilinear blend up to the corner
    residual r(u, v) = bilinear(vertex - Coons corner), |r| <= dc and
    |r_u|, |r_v| <= 2 dc; with Straight sides 0 and 2 the A/C terms cancel it
    (r = 0; dc is still added). Hence P = (1-l)R0(w) + l R1(w) + r, a ruled
    map between the two rails, with (w, l) = (u, v) for arcs on sides 0/2 and
    (v, u) for arcs on sides 3/1 (then P_u x P_v = -(P_w x P_l)).
    Each rail is R_i = o + z_i(w) a + R e_r(theta_i(w)) + rho_i(w)
    (_strip_rail). The ideal map P* = o + ((1-l) z0 + l z1) a + R e_r(theta0(w))
    lies on the support and P*_w x P*_l = R beta0 h(w) e_r with h = z1 - z0,
    while the support normal is R sweep H e_r. So the ideal map keeps the
    support orientation iff s h(w) > 0 on [0, 1], with
    s = sign(sweep H beta0 orientation). The actual map differs from P* by
    eta = theta1 - theta0 (affine in w, so bounded by its end values) and by
    the residuals rho_i and r:
      |P - P*|     <= ep = max(e0, e1) + dc,
      |P_w - P*_w| <= ew = max(e0', e1') + 2 dc,
      |P_l - P*_l| <= el = e0 + e1 + 2 dc,
    with e0 = |rho0|, e0' = |rho0'|, e1 = |rho1| + R max|eta| and
    e1' = |rho1'| + R |beta1 - beta0| + R |beta0| max|eta|. The radial unit
    vector at P differs from e_r(theta0) by at most 2 ep / R. Then, everywhere
    on the unit square, with n the support's outward-oriented unit normal at
    the actual point P,
      s (P_u x P_v).n >= R|beta0| h_min (1 - 2 ep / R) - (ew h_max + T el + ew el),
    T = R|beta0| + max_i max|z_i'| >= |P*_w|. h_min and h_max are certified
    from K + 1 samples of s h and max|h''| <= sum_i delta_i^2 |(U_a, V_a)_i|.
    Admission requires that lower bound >= _NORMAL_MARGIN R|beta0| h_min with
    h_min > 0, and ep <= the owner's surface_residual (the map lies on its
    support). Rounding: every residual and sample carries _ROUNDING * scale.
    """
    if len(face.corners) != 4 or len(face.loop) != 4:
        raise MeshError(_FAMILY_REFUSAL)
    sides = face.sides()
    if any(len(side) != 1 for side in sides):
        raise MeshError(_FAMILY_REFUSAL)
    kinds = [type(owner.edges[side[0].edge].curve) for side in sides]
    if kinds == [EllipticArc, Straight, EllipticArc, Straight]:
        rails, orientation = ((sides[0][0], False), (sides[2][0], True)), 1.0
    elif kinds == [Straight, EllipticArc, Straight, EllipticArc]:
        rails, orientation = ((sides[3][0], True), (sides[1][0], False)), -1.0
    else:
        raise MeshError(_FAMILY_REFUSAL)
    radius = float(surface.radius)
    made = []
    corner_residual = 0.0
    for item, fraction_reversed in rails:
        edge = owner.edges[item.edge]
        made.append(_strip_rail(edge.curve, item.forward, fraction_reversed, surface))
        # Corner residual: the arc's own ends against its vertices (the Straight ends).
        ends = np.asarray(edge.curve.evaluate(np.asarray((0.0, 1.0))), dtype=float)
        for end, vertex in zip(ends, (edge.start, edge.end)):
            corner_residual = max(corner_residual, float(np.linalg.norm(
                end - np.asarray(owner.vertex_position(vertex), dtype=float))))
    rail0, rail1 = made
    tau = _ROUNDING * max(rail0["scale"], rail1["scale"])
    corner_residual += tau
    beta0 = rail0["beta"]
    eta0 = math.remainder(rail1["theta_start"] - rail0["theta_start"], 2.0 * math.pi)
    d_beta = rail1["beta"] - beta0
    eta_max = max(abs(eta0), abs(eta0 + d_beta)) + _ROUNDING * (
        1.0 + abs(rail0["theta_start"]) + abs(rail1["theta_start"]) + abs(beta0))
    e0, e0_der = rail0["delta_pos"], rail0["delta_der"]
    e1 = rail1["delta_pos"] + radius * eta_max
    e1_der = rail1["delta_der"] + radius * abs(d_beta) + radius * abs(beta0) * eta_max
    ep = max(e0, e1) + corner_residual
    ew = max(e0_der, e1_der) + 2.0 * corner_residual
    el = e0 + e1 + 2.0 * corner_residual
    sign = math.copysign(1.0, float(surface.sweep_angle) * float(surface.height) * beta0 * orientation)
    w = np.linspace(0.0, 1.0, _STRIP_SAMPLES + 1)
    heights = sign * (_rail_heights(rail1, w) - _rail_heights(rail0, w))
    slack = (rail0["z_curvature"] + rail1["z_curvature"]) / (8.0 * _STRIP_SAMPLES ** 2) + 2.0 * tau
    h_min = float(np.min(heights)) - slack
    h_max = float(np.max(np.abs(heights))) + slack
    ideal = radius * abs(beta0) * h_min
    tangent = radius * abs(beta0) + max(rail0["z_slope"], rail1["z_slope"])
    lower = ideal * (1.0 - 2.0 * ep / radius) - (ew * h_max + tangent * el + ew * el)
    certificate = dict(h_min=h_min, h_max=h_max, support_deviation=ep, normal_lower=lower,
                       normal_ideal=ideal)
    if not (all(math.isfinite(value) for value in certificate.values()) and h_min > 0.0
            and ep < 0.5 * radius and lower >= _NORMAL_MARGIN * ideal
            and ep <= float(owner.tolerance.surface_residual)):
        raise MeshError(_CERTIFICATE_REFUSAL)
    return certificate


class CylinderSupportFrame:
    """Face queries of one parameterized Cylinder face, in its support frame."""

    __slots__ = ("owner", "face_id", "face", "surface", "_trim_loops", "orientation_certificate")

    def __init__(self, owner, face_id: int) -> None:
        if type(owner) is not GeometryModel:
            # The reproduced owner helpers are those of GeometryModel itself.
            raise MeshError("support-frame cylinder chart requires an owner GeometryModel")
        face = owner.faces[int(face_id)]
        surface = face.surface
        if (not isinstance(surface, Cylinder) or face.parameterization is None
                or face.parameterization is surface):
            raise MeshError("support-frame chart requires a Cylinder with a distinct parameterization")
        self.owner = owner
        self.face_id = int(face_id)
        self.face = face
        self.surface = surface
        self._trim_loops = None
        self.orientation_certificate = None
        self._admit_orientation()

    def _admit_orientation(self) -> None:
        """Refuse unless the face's own orientation provably matches the support's.

        The support-frame mesh is oriented by the support normal (du x dv of
        the support), exactly like the twin's. The face's own normal comes from
        its parameterization, so a reversed parameterization would publish
        elements against the face normal. Admission is decided only where the
        orientation follows from the loop: a topology Coons map sends the unit
        square's counter-clockwise boundary onto the outer loop in loop order
        (ANYgeometry ``_topology_face_point_and_derivatives``), so its degree is
        the winding of that loop in the support (u, v) plane. Any other
        parameterization is undecided and refused. The loop must wind
        positively both as its vertex polygon and as its sampled trim polygon,
        and the parameterization normal at its centre must agree with the
        support normal there; otherwise the face is refused (fail closed).
        Boundary winding and a centre normal do not exclude folding inside the
        face (an L-shaped trim with four declared corners folds its Coons map).
        So the face must finally be a ruled cylinder strip whose orientation
        and support deviation are certified over the whole face
        (_certify_ruled_strip); every other face is refused as undecided.
        """
        face, owner = self.face, self.owner
        parameterization = face.parameterization
        if (type(parameterization) is not CoonsSurface or parameterization.has_boundaries
                or len(face.corners) != 4):
            raise MeshError("support-frame cylinder chart admits only a topology Coons face "
                            "parameterization (its orientation relative to the support is undecided)")
        vertices = np.asarray([
            self.face_local_uv(self.face_id, owner.vertex_position(owner.oriented_start_vertex(item)))
            for item in face.loop], dtype=float)
        loops = (vertices, self.face_trim_loops_uv(self.face_id)[0])
        if any(len(loop) < 3 or not _signed_area(loop) > 0.0 for loop in loops):
            raise MeshError("support-frame cylinder chart refuses a face parameterization whose "
                            "orientation is reversed relative to its support")
        centre = np.asarray(owner.face_point(self.face_id, .5, .5), dtype=float)
        support_normal = self.face_normal_many(
            self.face_id, np.asarray([self.face_local_uv(self.face_id, centre)], dtype=float))[0]
        own_normal = np.asarray(owner.face_normal(self.face_id, .5, .5), dtype=float)
        if not float(own_normal @ support_normal) > 0.0:
            raise MeshError("support-frame cylinder chart refuses a face parameterization whose "
                            "orientation is reversed relative to its support")
        self.orientation_certificate = _certify_ruled_strip(owner, face, self.surface)

    def _check(self, face_id) -> None:
        if int(face_id) != self.face_id:
            raise MeshError("support-frame chart is bound to another face")
        face = self.owner.faces.get(self.face_id)
        if face is not self.face or face.surface is not self.surface:
            raise MeshError("support-frame chart face changed after binding")

    # -- inverse --------------------------------------------------------------
    def face_local_uv(self, face_id, point):
        self._check(face_id)
        local = self.surface.local_uv(point)
        return (float(np.clip(local[0], 0.0, 1.0)), float(np.clip(local[1], 0.0, 1.0)))

    def face_local_uv_many(self, face_id, points):
        self._check(face_id)
        values = np.asarray(points, dtype=float)
        if values.size == 0:
            if values.ndim != 2 or values.shape[1] != 3:
                raise GeometryError("face local UV points must have shape (n, 3)")
            return np.empty((0, 2), dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise GeometryError("face local UV points must be finite (n, 3) coordinates")
        return np.clip(GeometryModel._builtin_local_uv_many(self.surface, values), 0.0, 1.0)

    # -- evaluation -----------------------------------------------------------
    def face_point(self, face_id, u, v):
        self._check(face_id)
        return np.asarray(self.surface.evaluate(float(u), float(v)), dtype=float)

    def evaluate_face_many(self, face_id, uv):
        self._check(face_id)
        values = self.owner._uv_array(uv)
        return _evaluate_surface_many(self.surface, values).reshape((-1, 3))

    def face_derivatives_many(self, face_id, uv):
        self._check(face_id)
        values = self.owner._uv_array(uv)
        return _surface_derivatives_many(self.surface, values)

    def face_normal_many(self, face_id, uv):
        du, dv = self.face_derivatives_many(face_id, uv)
        normals = np.cross(du, dv)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0):
            raise GeometryError(f"face {self.face_id} has a degenerate normal")
        return normals / lengths[:, None]

    def face_normal(self, face_id, u, v):
        # GeometryModel.face_normal: one-sided finite differences of face_point.
        step = 1.0e-6
        u0, v0 = float(u), float(v)
        point = self.face_point(face_id, u0, v0)
        du = self.face_point(face_id, min(1.0, u0 + step), v0) - point
        dv = self.face_point(face_id, u0, min(1.0, v0 + step)) - point
        if float(np.linalg.norm(du)) <= 0.0:
            du = point - self.face_point(face_id, max(0.0, u0 - step), v0)
        if float(np.linalg.norm(dv)) <= 0.0:
            dv = point - self.face_point(face_id, u0, max(0.0, v0 - step))
        normal = np.cross(du, dv)
        length = float(np.linalg.norm(normal))
        if length <= 0.0:
            raise GeometryError(f"face {self.face_id} has a degenerate normal")
        return normal / length

    # -- trim-aware projection ------------------------------------------------
    def face_trim_loops_uv(self, face_id):
        self._check(face_id)
        if self._trim_loops is None:
            owner = self.owner

            def polygon(loop):
                points = []
                for item in loop:
                    edge = owner.edges[item.edge]
                    samples = owner.sample_edge(item.edge, np.linspace(
                        0.0, 1.0, 2 if isinstance(edge.curve, Straight) else _TRIM_CURVE_SAMPLES))
                    if not item.forward:
                        samples = samples[::-1]
                    points.extend(samples[:-1])
                if not points:
                    return np.asarray([], dtype=float)
                return np.asarray([self.face_local_uv(self.face_id, point) for point in points],
                                  dtype=float).reshape((-1, 2))

            self._trim_loops = tuple(polygon(loop) for loop in (self.face.loop,) + tuple(self.face.holes))
        return tuple(loop.copy() for loop in self._trim_loops)

    def _project_builtin(self, targets, trim_loops):
        owner = self.owner
        uv, inside_domain = owner._builtin_projection_uv_many(self.surface, targets)
        projected = _evaluate_surface_many(self.surface, uv).reshape((-1, 3))
        inside_trim = owner._points_in_face_trim(uv, trim_loops)
        for index in np.flatnonzero(~(inside_domain & inside_trim)):
            # Frame-free: the closest point of the face's boundary edges.
            boundary = owner._closest_face_boundary_point(self.face_id, targets[index])
            projected[index] = boundary
            uv[index] = np.asarray(self.face_local_uv(self.face_id, boundary), dtype=float)
        distances = np.linalg.norm(projected - targets, axis=1)
        return projected, uv, distances

    def project_to_face(self, face_id, point):
        target = np.asarray(point, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise GeometryError("projection point must be a finite 3-vector")
        trim_loops = self.face_trim_loops_uv(face_id)
        projected, uv, distances = self._project_builtin(target.reshape(1, 3), trim_loops)
        return projected[0], (float(uv[0, 0]), float(uv[0, 1])), float(distances[0])

    def project_to_face_many(self, face_id, xyz, initial_uv=None):
        self._check(face_id)
        points = np.asarray(xyz, dtype=float)
        if points.ndim == 1 and points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim == 1 and points.shape == (3,):
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise GeometryError("projection points must be a finite (n, 3) array")
        if initial_uv is not None and len(self.owner._uv_array(initial_uv)) != len(points):
            raise GeometryError("initial_uv must match the number of projection points")
        if len(points) == 0:
            return np.empty((0, 3)), np.empty((0, 2)), np.empty(0)
        return self._project_builtin(points, self.face_trim_loops_uv(face_id))
