"""A per-face quad-first refusal names the working face and the failing cell's 3D centroid."""

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


def test_t3_refusal_names_the_face_and_a_location_on_it():
    geometry, sliver = _rectangle_and_sliver()
    with pytest.raises(QuadQualityRejected) as caught:
        generate_hybrid_mesh_result(geometry, target_size=0.5, strategy="native",
                                    quad_options=QuadMeshingOptions())
    error = caught.value
    assert str(error).startswith("T3 cell ")
    assert f"[quad-first face {sliver} near (" in str(error)
    assert error.quad_first_face == sliver
    x, y, z = error.quad_first_location
    assert 5.0 <= x <= 6.0 and 0.0 <= y <= 0.05 and z == 0.0


class _Domain:
    face_id = 7

    @staticmethod
    def lift(point):
        return (float(point[0]), float(point[1]), 1.0)


def test_annotation_without_a_cell_names_only_the_face():
    context = _QuadFaceRefusalContext()
    context.domain = _Domain()
    error = MeshError("triangulation lost mandatory segments: [(0, 1)]")
    _annotate_quad_face_refusal(error, context)
    assert str(error) == "triangulation lost mandatory segments: [(0, 1)] [quad-first face 7]"
    assert error.quad_first_face == 7 and error.quad_first_location is None
    _annotate_quad_face_refusal(error, context)  # idempotent
    assert str(error).count("[quad-first face") == 1


def test_cancellation_and_refusals_outside_the_face_loop_are_unchanged():
    context = _QuadFaceRefusalContext()
    outside = MeshError("quad-first requires at least one selected face")
    _annotate_quad_face_refusal(outside, context)
    assert str(outside) == "quad-first requires at least one selected face"
    assert not hasattr(outside, "quad_first_face")
    context.domain = _Domain()
    cancelled = _StageCancelled("quad-first:face-seed cancelled")
    _annotate_quad_face_refusal(cancelled, context)
    assert str(cancelled) == "quad-first:face-seed cancelled"
    assert not hasattr(cancelled, "quad_first_face")


def test_lift_failure_keeps_the_face_name():
    class Broken(_Domain):
        @staticmethod
        def lift(point):
            raise RuntimeError("no chart")

    context = _QuadFaceRefusalContext()
    context.domain = Broken()
    error = QuadQualityRejected("T3 cell 1 (1, 2, 3): corner angle 1.00 < 15 deg")
    error.cell_uv = ((0.0, 0.0), (1.0, 0.0), (1.0, 0.1))
    _annotate_quad_face_refusal(error, context)
    assert str(error).endswith("[quad-first face 7]") and error.quad_first_location is None


def test_lost_segment_refusal_is_located_at_the_segments():
    context = _QuadFaceRefusalContext()
    context.domain = _Domain()
    error = MeshError("triangulation lost mandatory segments: [(0, 1)]")
    error.segment_uv = ((0.0, 0.0), (2.0, 0.0))
    _annotate_quad_face_refusal(error, context)
    assert error.quad_first_location == (1.0, 0.0, 1.0)
    assert str(error).endswith("[quad-first face 7 near (1, 0, 1)]")
