from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.native import (
    NativeTriangulation,
    clear_native_boundary,
    register_native_boundary,
)
from anymesher.native_cpp import (
    COMPILED_TRIANGULATION_AVAILABLE,
    compiled_native_boundary,
)
from anymesher.triangulation import constrained_planar_triangulation


SQUARE = np.array(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    dtype=np.float64,
)
OUTER = (0, 1, 2, 3)


@dataclass
class FixedBoundary:
    points: np.ndarray
    triangles: np.ndarray
    name: str = "fixed-test-boundary"
    calls: int = 0

    def triangulate(self, points, segments, outer_loop, hole_loops):
        self.calls += 1
        return NativeTriangulation(self.points, self.triangles)


def _python_square() -> NativeTriangulation:
    result = constrained_planar_triangulation(SQUARE, OUTER, backend="python")
    return NativeTriangulation(result.points, result.triangles)


def test_custom_boundary_matches_python_oracle_and_records_provenance() -> None:
    oracle = _python_square()
    boundary = FixedBoundary(oracle.points, oracle.triangles)

    result = constrained_planar_triangulation(SQUARE, OUTER, backend=boundary)

    np.testing.assert_array_equal(result.points, oracle.points)
    np.testing.assert_array_equal(result.triangles, oracle.triangles)
    assert result.requested_backend == boundary.name
    assert result.selected_backend == boundary.name
    assert result.actual_backend == boundary.name
    assert result.fallback_reason is None
    assert boundary.calls == 1


def test_auto_snapshots_registered_boundary_and_never_switches_mid_call() -> None:
    oracle = _python_square()
    replacement = FixedBoundary(oracle.points, oracle.triangles, name="replacement")

    class ReplacingBoundary(FixedBoundary):
        def triangulate(self, points, segments, outer_loop, hole_loops):
            register_native_boundary(replacement)
            return super().triangulate(points, segments, outer_loop, hole_loops)

    selected = ReplacingBoundary(oracle.points, oracle.triangles, name="selected")
    previous = register_native_boundary(selected)
    try:
        result = constrained_planar_triangulation(SQUARE, OUTER, backend="auto")
        assert result.selected_backend == "selected"
        assert result.actual_backend == "selected"
        assert selected.calls == 1
        assert replacement.calls == 0
    finally:
        register_native_boundary(previous)


def test_auto_never_falls_back_after_selected_boundary_failure() -> None:
    class FailingBoundary:
        name = "failing"

        def triangulate(self, points, segments, outer_loop, hole_loops):
            raise RuntimeError("selected boundary failed")

    with pytest.raises(RuntimeError, match="selected boundary failed"):
        constrained_planar_triangulation(SQUARE, OUTER, backend=FailingBoundary())


def test_native_result_rejects_duplicate_cells_without_cleanup() -> None:
    oracle = _python_square()
    duplicate = np.ascontiguousarray(
        np.vstack((oracle.triangles, oracle.triangles[0])), dtype=np.int64
    )

    with pytest.raises(MeshError, match="duplicate"):
        constrained_planar_triangulation(
            SQUARE,
            OUTER,
            backend=FixedBoundary(oracle.points, duplicate),
        )


def test_native_result_rejects_nonmanifold_incidence() -> None:
    points = np.array(
        (
            (0.0, 0.0),
            (3.0, 0.0),
            (3.0, 3.0),
            (0.0, 3.0),
            (1.0, 1.0),
            (2.0, 1.0),
        ),
        dtype=np.float64,
    )
    triangles = np.array(
        ((0, 1, 4), (0, 5, 4), (0, 4, 3)), dtype=np.int64
    )

    with pytest.raises(MeshError, match="nonmanifold"):
        constrained_planar_triangulation(
            points,
            OUTER,
            backend=FixedBoundary(points, triangles),
        )


@pytest.mark.parametrize(
    ("points", "triangles", "message"),
    (
        (SQUARE.astype(np.float32), np.array(((0, 1, 2),), dtype=np.int64), "float64"),
        (SQUARE, np.array(((0, 1, 2),), dtype=np.int32), "int64"),
        (SQUARE[:, ::-1], np.array(((0, 1, 2),), dtype=np.int64), "C-contiguous"),
    ),
)
def test_native_result_rejects_lossy_or_strided_arrays(
    points, triangles, message
) -> None:
    with pytest.raises(MeshError, match=message):
        NativeTriangulation(points, triangles)


def test_legacy_boundary_cancellation_is_checked_before_execution() -> None:
    oracle = _python_square()
    boundary = FixedBoundary(oracle.points, oracle.triangles)

    def cancel(phase: str) -> None:
        raise LookupError(phase)

    with pytest.raises(LookupError, match="native triangulation start"):
        constrained_planar_triangulation(
            SQUARE,
            OUTER,
            backend=boundary,
            cancellation_check=cancel,
        )
    assert boundary.calls == 0


def test_extension_absent_auto_falls_back_but_strict_native_fails() -> None:
    previous = clear_native_boundary()
    try:
        if COMPILED_TRIANGULATION_AVAILABLE:
            pytest.skip("compiled triangulation is available in this environment")
        result = constrained_planar_triangulation(SQUARE, OUTER, backend="auto")
        assert result.actual_backend == "python"
        assert result.fallback_reason == "native_capability_absent"
        with pytest.raises(MeshError, match="no native triangulation boundary"):
            constrained_planar_triangulation(SQUARE, OUTER, backend="native")
    finally:
        register_native_boundary(previous)


@pytest.mark.skipif(
    not COMPILED_TRIANGULATION_AVAILABLE,
    reason="compiled triangulation extension has not been rebuilt",
)
def test_compiled_square_matches_python_oracle_exactly() -> None:
    oracle = constrained_planar_triangulation(SQUARE, OUTER, backend="python")
    compiled = constrained_planar_triangulation(SQUARE, OUTER, backend="native")

    assert compiled.actual_backend == "anymesher-cpp17"
    assert compiled.points.tobytes() == oracle.points.tobytes()
    np.testing.assert_array_equal(compiled.triangles, oracle.triangles)


@pytest.mark.skipif(
    not COMPILED_TRIANGULATION_AVAILABLE,
    reason="compiled triangulation extension has not been rebuilt",
)
def test_compiled_boundary_rejects_noncanonical_input_before_coercion() -> None:
    boundary = compiled_native_boundary()
    assert boundary is not None
    with pytest.raises(TypeError, match="float64"):
        boundary.triangulate(
            SQUARE.astype(np.float32),
            np.array(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64),
            np.array(OUTER, dtype=np.int64),
            (),
        )


@pytest.mark.skipif(
    not COMPILED_TRIANGULATION_AVAILABLE,
    reason="compiled triangulation extension has not been rebuilt",
)
def test_compiled_cancellation_propagates_original_exception() -> None:
    class Cancelled(Exception):
        pass

    def cancel(phase: str) -> None:
        if phase == "native triangulation insertion":
            raise Cancelled("cancel compiled insertion")

    with pytest.raises(Cancelled, match="cancel compiled insertion"):
        constrained_planar_triangulation(
            SQUARE,
            OUTER,
            backend="native",
            cancellation_check=cancel,
        )
