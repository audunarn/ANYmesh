"""A per-face quad-first refusal records the working face and the failing cell's 3D centroid.

Message-neutral (recovery stores str(error)); caller-owned cancellation exceptions are untouched.
"""

import numpy as np
import pytest

from anygeometry.model import GeometryModel

from anymesher.errors import MeshError
from anymesher.hybrid import (_QuadFaceRefusalContext, _annotate_quad_face_refusal,
                              generate_hybrid_mesh_result)
from anymesher._stage_cancellation import _StageCancelled
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.validate import QuadQualityRejected


def _rectangle_and_sliver():
    geometry = GeometryModel()
    geometry.add_plate(geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))))
    sliver = geometry.add_plate(geometry.add_points(
        ((5.0, 0.0, 0.0), (6.0, 0.0, 0.0), (6.0, 0.05, 0.0))))
    return geometry, sliver


def _mesh(geometry, **extra):
    return generate_hybrid_mesh_result(geometry, target_size=0.5, strategy="native",
                                       quad_options=QuadMeshingOptions(), **extra)


def test_t3_refusal_records_the_face_and_a_location_on_it():
    geometry, sliver = _rectangle_and_sliver()
    with pytest.raises(QuadQualityRejected) as caught:
        _mesh(geometry)
    error = caught.value
    assert str(error).startswith("T3 cell ") and "quad-first face" not in str(error)
    assert error.quad_first_face == sliver
    x, y, z = error.quad_first_location
    assert 5.0 <= x <= 6.0 and 0.0 <= y <= 0.05 and z == 0.0


def test_cancellation_callback_exception_is_not_annotated():
    geometry, _sliver = _rectangle_and_sliver()
    stop = MeshError("stop")

    def check(stage):
        if stage == "quad-first:driver-start":
            raise stop

    with pytest.raises(MeshError) as caught:
        _mesh(geometry, cancellation_check=check)
    assert caught.value is stop and str(stop) == "stop"
    assert not hasattr(stop, "quad_first_face") and not hasattr(stop, "quad_first_location")


class _Domain:
    def __init__(self, face_id=7):
        self.face_id = face_id

    @staticmethod
    def lift(point):
        return (float(point[0]), float(point[1]), 1.0)


def _context(face_id=7):
    context = _QuadFaceRefusalContext()
    context.domain = _Domain(face_id)
    return context


def test_annotation_without_a_cell_records_only_the_face():
    error = MeshError("triangulation lost mandatory segments: [(0, 1)]")
    _annotate_quad_face_refusal(error, _context())
    assert str(error) == "triangulation lost mandatory segments: [(0, 1)]"
    assert error.quad_first_face == 7 and error.quad_first_location is None


def test_a_reused_refusal_records_the_current_face():
    error = MeshError("refused")
    _annotate_quad_face_refusal(error, _context(1))
    _annotate_quad_face_refusal(error, _context(2))
    assert error.quad_first_face == 2


def test_array_chart_points_and_nonstring_payloads_are_safe():
    error = MeshError(42)
    error.cell_uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    _annotate_quad_face_refusal(error, _context())
    assert error.args == (42,)
    assert error.quad_first_location == pytest.approx((2 / 3, 1 / 3, 1.0))
    bad = MeshError("refused")
    bad.cell_uv = "not points"
    _annotate_quad_face_refusal(bad, _context())
    assert bad.quad_first_face == 7 and bad.quad_first_location is None


def test_stage_cancellation_and_refusals_outside_the_face_loop_are_unchanged():
    outside = MeshError("quad-first requires at least one selected face")
    _annotate_quad_face_refusal(outside, _QuadFaceRefusalContext())
    assert not hasattr(outside, "quad_first_face")
    cancelled = _StageCancelled("quad-first:face-seed cancelled")
    _annotate_quad_face_refusal(cancelled, _context())
    assert not hasattr(cancelled, "quad_first_face")


def test_lift_failure_keeps_the_face():
    class Broken(_Domain):
        @staticmethod
        def lift(point):
            raise RuntimeError("no chart")

    context = _QuadFaceRefusalContext()
    context.domain = Broken()
    error = QuadQualityRejected("T3 cell 1 (1, 2, 3): corner angle 1.00 < 15 deg")
    error.cell_uv = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.1))
    _annotate_quad_face_refusal(error, context)
    assert error.quad_first_face == 7 and error.quad_first_location is None


def test_lost_segment_refusal_is_located_at_the_segments():
    error = MeshError("triangulation lost mandatory segments: [(0, 1)]")
    error.segment_uv = ((0.0, 0.0), (2.0, 0.0))
    _annotate_quad_face_refusal(error, _context())
    assert error.quad_first_location == (1.0, 0.0, 1.0)
