"""Compatibility exports for curve geometry now owned by ANYgeometry."""

from anygeometry.curves import (
    Arc,
    ArcFrame,
    CurveShape,
    DegenerateArcError,
    Spline,
    Straight,
    arc_frame,
    arc_tangent,
    sample_arc,
    sample_straight,
    sample_spline,
    spline_tangent,
    straight_tangent,
)

__all__ = [
    "Arc",
    "ArcFrame",
    "CurveShape",
    "DegenerateArcError",
    "Straight",
    "Spline",
    "arc_frame",
    "arc_tangent",
    "sample_arc",
    "sample_straight",
    "sample_spline",
    "spline_tangent",
    "straight_tangent",
]
