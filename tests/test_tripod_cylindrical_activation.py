"""The retained plane-chart trace family activates the trimmed-cylinder route (family recognition only)."""
import numpy as np
import pytest

pytest.importorskip("anygeometry.plane_ruled")

from anygeometry import BezierQuadricCurve, EllipticArc, CylinderIntersectionCurve
from anygeometry.analytic_supports import plane_cylinder_support
from anygeometry.surfaces import Cylinder, Plane

from anymesher import _cylindrical_public


def test_exact_trace_families_include_the_retained_bezier_quadric_family():
    families = _cylindrical_public.EXACT_TRACE_CURVES
    assert BezierQuadricCurve in families and EllipticArc in families and CylinderIntersectionCurve in families


def test_a_real_near_parallel_plane_trace_is_recognized_as_an_exact_trace():
    plane = Plane((45.0, -10.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cylinder = Cylinder((44.0, 0.0, 0.0), (2.45e-16, 0.0, 1.0), (1.0, 0.0, 0.0), 5.0, 4.0)
    curves = plane_cylinder_support(plane, cylinder, tolerance=1e-9).curves
    assert curves
    assert all(isinstance(curve, _cylindrical_public.EXACT_TRACE_CURVES) for curve in curves)
    for curve in curves:
        assert np.all(np.isfinite(curve.evaluate(np.linspace(0.0, 1.0, 5))))
