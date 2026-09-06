"""Bounded analytic disjointness proof for topology-owned planar Arc rings.

This is not a curved-region overlay or an area estimate. A complete outer/hole
identity, a simple star-shaped ring and strict convex containment prove disjoint
material interiors. Anything not established here remains an owner query.
"""

import math

import numpy as np

from anygeometry.curves import Arc, Straight, arc_frame
from anygeometry.surfaces import Plane

from ._complementary_trims import complementary_trim_domains


def _det(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def certified_complementary_arc_domains(
    geometry, first, second, *, cancellation_check=None,
) -> bool:
    """Prove a single Arc hole and its topology-identical filling are disjoint.

    Planarity is qualified within owner tolerance with a stricter roundoff
    allowance. Endpoint/API errors propagate; numerical ambiguity returns False.
    No geometry, coordinates or source ownership are changed.
    """
    a, b = geometry.faces[first], geometry.faces[second]
    if len(a.holes) == 1 and not b.holes:
        parent, inner = a, b
    elif len(b.holes) == 1 and not a.holes:
        parent, inner = b, a
    else:
        return False
    if type(parent.surface) is not Plane or type(inner.surface) is not Plane:
        return False
    if not (3 <= len(parent.loop) <= 256 and 3 <= len(inner.loop) <= 256):
        return False
    if not all(isinstance(geometry.edges[item.edge].curve, Straight) for item in parent.loop):
        return False
    if not all(isinstance(geometry.edges[item.edge].curve, Arc) for item in inner.loop):
        return False
    if not complementary_trim_domains(geometry, first, second):
        return False
    for name in ("origin", "u_vector", "v_vector"):
        left = np.asarray(getattr(parent.surface, name), dtype=np.float64)
        right = np.asarray(getattr(inner.surface, name), dtype=np.float64)
        if left.shape != (3,) or not np.all(np.isfinite(left)) or not np.array_equal(left, right):
            return False
    origin = np.asarray(parent.surface.origin, dtype=np.float64)
    u = np.asarray(parent.surface.u_vector, dtype=np.float64)
    v = np.asarray(parent.surface.v_vector, dtype=np.float64)
    ul, vl = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if not np.isfinite(ul + vl) or min(ul, vl) <= 0.:
        return False
    x = u / ul
    normal = np.cross(x, v / vl)
    nl = float(np.linalg.norm(normal))
    if nl < 1.e-8:
        return False
    normal /= nl
    y = np.cross(normal, x)
    basis = np.column_stack((x, y))
    outer = np.asarray([
        geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in parent.loop
    ], dtype=np.float64)
    ring = np.asarray([
        geometry.vertex_position(geometry.oriented_start_vertex(item)) for item in inner.loop
    ], dtype=np.float64)
    if outer.shape != (len(parent.loop), 3) or ring.shape != (len(inner.loop), 3):
        return False
    points = np.vstack((outer, ring))
    if not np.all(np.isfinite(points)):
        return False
    # Match the owner's physical pair-AABB extent. Absolute coordinates and
    # arbitrary chart-vector lengths must never enlarge geometric tolerance.
    bounds = []
    for identifier in (first, second):
        raw = geometry._entity_bounds(("face", identifier))
        if raw is None:
            return False
        bound = np.asarray(raw, dtype=np.float64)
        if bound.shape != (6,) or not np.all(np.isfinite(bound)) or np.any(bound[3:] < bound[:3]):
            return False
        bounds.append(bound)
    low = np.minimum(bounds[0][:3], bounds[1][:3])
    high = np.maximum(bounds[0][3:], bounds[1][3:])
    extent = float(np.linalg.norm(high - low))
    if not np.isfinite(extent) or extent <= 0.:
        return False
    scale = max(1., float(np.max(np.abs(points))), float(np.max(np.abs(origin))), extent)
    uncertainty = 512. * np.finfo(np.float64).eps * scale
    tolerance = float(geometry.tolerance.effective_length(extent))
    if not np.isfinite(tolerance) or tolerance <= 16. * uncertainty:
        return False
    plane_allowance = min(tolerance / 4., 8. * uncertainty)
    interior_margin = tolerance + 32. * uncertainty
    if np.max(np.abs((points - origin) @ normal)) > plane_allowance:
        return False
    polygon = (outer - origin) @ basis
    vertices = (ring - origin) @ basis
    anchor = np.mean(vertices, axis=0)
    first_turn = _det(polygon[1] - polygon[0], polygon[2] - polygon[1])
    if abs(first_turn) <= 16. * uncertainty * scale:
        return False
    orientation = 1. if first_turn > 0. else -1.
    halfspaces = []
    for index, start in enumerate(polygon):
        direction = polygon[(index + 1) % len(polygon)] - start
        length = float(np.linalg.norm(direction))
        if length <= 16. * uncertainty:
            return False
        inward = orientation * np.asarray((-direction[1], direction[0])) / length
        # All nonincident vertices, not just local turn signs: this excludes
        # self-intersecting and nonconvex enclosing loops.
        for other, point in enumerate(polygon):
            if other not in (index, (index + 1) % len(polygon)):
                if float((point - start) @ inward) <= interior_margin:
                    return False
        halfspaces.append((start, inward))

    turn_sign = None
    angles = []
    angle_errors = []
    for index, item in enumerate(inner.loop):
        if cancellation_check is not None and index % 16 == 0:
            cancellation_check("structural preparation overlap narrow phase")
        start = np.asarray(geometry.vertex_position(geometry.oriented_start_vertex(item)), dtype=float)
        end = np.asarray(geometry.vertex_position(geometry.oriented_end_vertex(item)), dtype=float)
        via = np.asarray(geometry.vertex_position(geometry.edges[item.edge].curve.via_vertex), dtype=float)
        if not np.all(np.isfinite(via)):
            return False
        # Oriented endpoints already account for OrientedEdge.forward.
        frame = arc_frame(start, via, end)
        radius, sweep = float(frame.radius), float(frame.sweep)
        if not np.isfinite(radius + sweep) or radius <= 16. * uncertainty or sweep == 0.:
            return False
        center = np.asarray(frame.center, dtype=float)
        e1, e2 = np.asarray(frame.e1, dtype=float), np.asarray(frame.e2, dtype=float)
        if not np.all(np.isfinite(np.concatenate((center, e1, e2)))):
            return False
        if max(float(np.linalg.norm(center + radius * e1 - start)),
               float(np.linalg.norm(center + radius * (math.cos(sweep) * e1 + math.sin(sweep) * e2) - end))) > plane_allowance:
            return False
        excursion = abs(float((center - origin) @ normal)) + radius * math.hypot(float(e1 @ normal), float(e2 @ normal))
        if excursion > plane_allowance:
            return False
        c = (center - origin) @ basis
        eu, ev = e1 @ basis, e2 @ basis
        for point, inward in halfspaces:
            support = radius * math.hypot(float(eu @ inward), float(ev @ inward))
            if float((c - point) @ inward) - support <= interior_margin:
                return False
        offset = (c - anchor) / radius
        determinant = _det(eu, ev)
        allowance = 64. * (uncertainty / radius + np.finfo(float).eps)
        # Bound the polar derivative over the entire analytic arc.
        if abs(determinant) - math.hypot(_det(offset, eu), _det(offset, ev)) <= allowance:
            return False
        direction = 1. if sweep * determinant > 0. else -1.
        if turn_sign is not None and turn_sign != direction:
            return False
        turn_sign = direction
        singular = np.linalg.svd(np.column_stack((eu, ev)), compute_uv=False)
        distance = float(np.linalg.norm(offset))
        radial_lower = float(singular[-1]) - distance
        if radial_lower <= allowance:
            return False
        travel_upper = abs(sweep) * singular[0] * (distance + singular[0]) / radial_lower ** 2
        if not np.isfinite(travel_upper) or travel_upper >= math.pi - allowance:
            return False
        first_ray = vertices[index] - anchor
        second_ray = vertices[(index + 1) % len(vertices)] - anchor
        ray_lower = min(float(np.linalg.norm(first_ray)), float(np.linalg.norm(second_ray)), radius * radial_lower)
        error = 64. * (uncertainty / ray_lower + np.finfo(float).eps)
        angle = direction * math.atan2(_det(first_ray, second_ray), float(first_ray @ second_ray))
        if angle <= error or angle >= math.pi - error:
            return False
        angles.append(angle)
        angle_errors.append(error)
    total, error = math.fsum(angles), math.fsum(angle_errors)
    # Continuity plus a fixed-sign polar derivative gives an integer winding.
    # The error interval must contain one turn and exclude every other winding.
    return (
        error < math.pi / 4.
        and total - error > math.pi
        and total + error < 3. * math.pi
        and abs(total - 2. * math.pi) <= error
    )
