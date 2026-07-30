"""Curve shapes carried by edges.

An edge owns its topology (a start and an end vertex); the curve shape says how
the edge sweeps between them.  Curve shapes never store coordinates, so moving
a point moves every curve that references it.

Two shapes cover the modelling paradigm:

``Straight``
    A line between the two end points.

``Arc``
    A circular arc from the start point to the end point passing through a
    third *via* point.  Defining an arc by an existing point keeps modelling
    point-driven: the user places points, then connects them.

A full circle is modelled as two arcs.  That keeps every edge open, so the
topology never needs a special case for an edge whose start and end coincide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

__all__ = [
    "Arc",
    "ArcFrame",
    "CurveShape",
    "DegenerateArcError",
    "Straight",
    "arc_frame",
]

# Relative tolerance for deciding that three points are collinear, applied to
# the squared triangle normal against the squared edge lengths.
_COLLINEAR_RTOL = 1.0e-12


class DegenerateArcError(ValueError):
    """Raised when three points cannot define a circular arc."""


@dataclass(frozen=True)
class Straight:
    """A straight line between the edge end points."""

    __slots__ = ()


@dataclass(frozen=True)
class Arc:
    """A circular arc through ``via_vertex``.

    The arc runs from the edge start point to the edge end point along the
    circle through all three points, taking whichever of the two possible
    sweeps actually passes through the via point.  Sweeps greater than 180
    degrees are therefore supported.
    """

    via_vertex: int


CurveShape = Union[Straight, Arc]


@dataclass(frozen=True)
class ArcFrame:
    """Resolved circle geometry for an arc.

    The arc is ``center + radius * (cos(t*sweep) * e1 + sin(t*sweep) * e2)``
    for ``t`` in ``[0, 1]``, which is the start point at ``t = 0`` and the end
    point at ``t = 1``.  ``sweep`` is signed: negative means the arc runs
    clockwise about ``normal``.
    """

    center: np.ndarray
    radius: float
    e1: np.ndarray
    e2: np.ndarray
    normal: np.ndarray
    sweep: float

    @property
    def length(self) -> float:
        return float(self.radius * abs(self.sweep))


def arc_frame(
    start: np.ndarray, via: np.ndarray, end: np.ndarray
) -> ArcFrame:
    """Resolve the circle through three points into a sampling frame.

    Raises ``DegenerateArcError`` when the points are collinear or coincident,
    because no unique circle exists.
    """

    a = np.asarray(start, dtype=float)
    b = np.asarray(via, dtype=float)
    c = np.asarray(end, dtype=float)

    ab = b - a
    ac = c - a
    normal = np.cross(ab, ac)
    normal_sq = float(normal @ normal)
    scale = float((ab @ ab) * (ac @ ac))
    if scale <= 0.0 or normal_sq <= _COLLINEAR_RTOL * scale:
        raise DegenerateArcError(
            "arc points are collinear or coincident; no unique circle exists"
        )

    # Circumcenter of the triangle a, b, c.
    center = a + (
        float(ab @ ab) * np.cross(ac, normal)
        + float(ac @ ac) * np.cross(normal, ab)
    ) / (2.0 * normal_sq)

    radial = a - center
    radius = float(np.linalg.norm(radial))
    if radius <= 0.0:
        raise DegenerateArcError("degenerate arc: zero radius")

    unit_normal = normal / np.sqrt(normal_sq)
    e1 = radial / radius
    e2 = np.cross(unit_normal, e1)

    theta_via = _wrapped_angle(b - center, e1, e2)
    theta_end = _wrapped_angle(c - center, e1, e2)
    if theta_end <= 0.0:
        raise DegenerateArcError(
            "arc end point coincides with its start point; "
            "model a full circle as two arcs"
        )

    # The via point selects which of the two sweeps is meant.  If it lies
    # before the end point going counter-clockwise, the arc runs
    # counter-clockwise; otherwise it runs the other way round.
    sweep = theta_end if theta_via < theta_end else theta_end - 2.0 * np.pi

    return ArcFrame(
        center=center,
        radius=radius,
        e1=e1,
        e2=e2,
        normal=unit_normal,
        sweep=float(sweep),
    )


def _wrapped_angle(
    radial: np.ndarray, e1: np.ndarray, e2: np.ndarray
) -> float:
    """Angle of a radial vector in the circle plane, wrapped into [0, 2*pi)."""

    angle = float(np.arctan2(radial @ e2, radial @ e1))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return angle


def sample_straight(
    start: np.ndarray, end: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Sample a straight line at parameters ``t`` in ``[0, 1]``."""

    t_column = np.asarray(t, dtype=float).reshape(-1, 1)
    return start + t_column * (end - start)


def sample_arc(frame: ArcFrame, t: np.ndarray) -> np.ndarray:
    """Sample an arc at parameters ``t`` in ``[0, 1]``.

    Uniform ``t`` gives uniform angle and therefore uniform arc length.
    """

    angle = np.asarray(t, dtype=float).reshape(-1, 1) * frame.sweep
    return frame.center + frame.radius * (
        np.cos(angle) * frame.e1 + np.sin(angle) * frame.e2
    )


def straight_tangent(
    start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """Unit tangent of a straight line, constant along its length."""

    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("cannot take the tangent of a zero-length line")
    return direction / norm


def arc_tangent(frame: ArcFrame, t: float) -> np.ndarray:
    """Unit tangent of an arc at parameter ``t``."""

    angle = float(t) * frame.sweep
    tangent = frame.sweep * (
        -np.sin(angle) * frame.e1 + np.cos(angle) * frame.e2
    )
    norm = float(np.linalg.norm(tangent))
    if norm <= 0.0:
        raise ValueError("cannot take the tangent of a zero-sweep arc")
    return tangent / norm
