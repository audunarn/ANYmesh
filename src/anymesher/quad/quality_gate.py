"""Element-shape gates for the experimental quad-first route.

Strict convexity alone admits quads with a corner arbitrarily close to 180
degrees and residual triangles arbitrarily close to zero area.  This module
supplies the shape policy used at every quad-first Q4 admission point
(:func:`anymesher.quad.front.make_quad`) and by the final resident validator.

Metrics are evaluated in the face chart, which is exact for planar faces and
isometric for the cylindrical and conical charts.  Definitions match
:mod:`anymesher.quality_v2` (edge-length aspect ratio, interior corner angles
in degrees, corner scaled Jacobian ``sin`` of the corner angle signed by
orientation) but are computed in plain Python: admission runs for every
candidate pair, where per-call NumPy overhead dominates.

The thresholds are internal to the experimental route and are recorded in
public diagnostics; they are not part of ``QuadMeshingOptions``.
"""
from __future__ import annotations

from math import acos, degrees, hypot
from typing import Sequence

from ..structured import MeshQualityPolicy

__all__ = [
    "QUAD_FIRST_Q4_POLICY",
    "QUAD_FIRST_T3_MIN_ANGLE",
    "RELATIVE_EPS",
    "corner_metrics",
    "policy_record",
    "quad_violation",
    "triangle_violation",
]

# Q4 admission: every corner in [20, 160] degrees (which implies a corner
# scaled Jacobian of at least sin(20 deg) ~ 0.34 > 0.20) and edge ratio <= 10.
QUAD_FIRST_Q4_POLICY = MeshQualityPolicy(
    minimum_scaled_jacobian=0.20,
    maximum_aspect_ratio=10.0,
    minimum_angle=20.0,
    maximum_angle=160.0,
)
# Residual triangles are closures of the seed triangulation; publication
# rejects anything thinner than this.
QUAD_FIRST_T3_MIN_ANGLE = 15.0
# Scale-free degeneracy tolerance (relative to squared edge lengths).
RELATIVE_EPS = 1.0e-10

Point = Sequence[float]


def corner_metrics(points: Sequence[Point]) -> tuple[float, float, float, float]:
    """Return ``(min_angle, max_angle, min_scaled_jacobian, aspect_ratio)``.

    ``points`` is a counter-clockwise 2D polygon (3 or 4 corners).  A reflex
    or clockwise corner has a negative scaled Jacobian and an angle reported
    as ``360 - acos`` so a folded element can never look acceptable.
    """
    count = len(points)
    lengths = []
    for index in range(count):
        a = points[index]
        b = points[(index + 1) % count]
        lengths.append(hypot(b[0] - a[0], b[1] - a[1]))
    shortest = min(lengths)
    aspect = float("inf") if shortest <= 0.0 else max(lengths) / shortest
    angles = []
    jacobians = []
    for index in range(count):
        here = points[index]
        after = points[(index + 1) % count]
        before = points[index - 1]
        ox, oy = after[0] - here[0], after[1] - here[1]
        ix, iy = before[0] - here[0], before[1] - here[1]
        denominator = hypot(ox, oy) * hypot(ix, iy)
        if denominator <= 0.0:
            angles.append(0.0)
            jacobians.append(0.0)
            continue
        sine = (ox * iy - oy * ix) / denominator
        cosine = max(-1.0, min(1.0, (ox * ix + oy * iy) / denominator))
        angle = degrees(acos(cosine))
        angles.append(angle if sine >= 0.0 else 360.0 - angle)
        jacobians.append(sine)
    return min(angles), max(angles), min(jacobians), aspect


def quad_violation(
    points: Sequence[Point], policy: MeshQualityPolicy = QUAD_FIRST_Q4_POLICY,
) -> str | None:
    """Describe the first policy violation of a CCW quad, or ``None``."""
    min_angle, max_angle, jacobian, aspect = corner_metrics(points)
    if jacobian < policy.minimum_scaled_jacobian:
        return f"scaled Jacobian {jacobian:.4f} < {policy.minimum_scaled_jacobian:g}"
    if max_angle > policy.maximum_angle:
        return f"corner angle {max_angle:.2f} > {policy.maximum_angle:g} deg"
    if min_angle < policy.minimum_angle:
        return f"corner angle {min_angle:.2f} < {policy.minimum_angle:g} deg"
    if aspect > policy.maximum_aspect_ratio:
        return f"aspect ratio {aspect:.3g} > {policy.maximum_aspect_ratio:g}"
    return None


def triangle_violation(
    points: Sequence[Point], minimum_angle: float = QUAD_FIRST_T3_MIN_ANGLE,
) -> str | None:
    """Describe a residual-T3 violation of a CCW triangle, or ``None``."""
    min_angle, _max_angle, jacobian, _aspect = corner_metrics(points)
    if jacobian <= 0.0:
        return "triangle is not positively oriented"
    if min_angle < minimum_angle:
        return f"corner angle {min_angle:.2f} < {minimum_angle:g} deg"
    return None


def policy_record() -> dict[str, float]:
    """JSON-safe description of the active gates for diagnostics."""
    return {
        "q4_minimum_scaled_jacobian": QUAD_FIRST_Q4_POLICY.minimum_scaled_jacobian,
        "q4_minimum_angle": QUAD_FIRST_Q4_POLICY.minimum_angle,
        "q4_maximum_angle": QUAD_FIRST_Q4_POLICY.maximum_angle,
        "q4_maximum_aspect_ratio": QUAD_FIRST_Q4_POLICY.maximum_aspect_ratio,
        "t3_minimum_angle": QUAD_FIRST_T3_MIN_ANGLE,
    }
