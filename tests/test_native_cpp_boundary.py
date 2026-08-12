from __future__ import annotations

import numpy as np
import pytest

from anymesher import native_cpp
from anymesher.triangulation import incircle as adaptive_incircle
from anymesher.triangulation import orient2d as adaptive_orient2d


POINTS = np.array(
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
)
TRIANGLES = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
EXPECTED_INCIDENCE = np.array(
    ((0, 1, 0, -1), (0, 2, 0, 1), (0, 3, 1, -1),
     (1, 2, 0, -1), (2, 3, 1, -1)),
    dtype=np.int64,
)


def _near_degenerate_predicate_signs() -> tuple[int, int, int, int, int]:
    above = (2.0, np.nextafter(2.0, np.inf))
    below = (2.0, np.nextafter(2.0, -np.inf))
    triangle = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    inside = (0.0, np.nextafter(-1.0, 0.0))
    outside = (0.0, np.nextafter(-1.0, -np.inf))
    boundary = (0.0, -1.0)
    return tuple(
        int(np.sign(value))
        for value in (
            adaptive_orient2d((0.0, 0.0), (1.0, 1.0), above),
            adaptive_orient2d((0.0, 0.0), (1.0, 1.0), below),
            adaptive_incircle(*triangle, inside),
            adaptive_incircle(*triangle, outside),
            adaptive_incircle(*triangle, boundary),
        )
    )


def _native_predicate_signs() -> tuple[int, int, int, int, int]:
    above = (2.0, np.nextafter(2.0, np.inf))
    below = (2.0, np.nextafter(2.0, -np.inf))
    triangle = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0))
    inside = (0.0, np.nextafter(-1.0, 0.0))
    outside = (0.0, np.nextafter(-1.0, -np.inf))
    boundary = (0.0, -1.0)
    return tuple(
        int(np.sign(value))
        for value in (
            native_cpp.orient2d((0.0, 0.0), (1.0, 1.0), above),
            native_cpp.orient2d((0.0, 0.0), (1.0, 1.0), below),
            native_cpp.incircle(*triangle, inside),
            native_cpp.incircle(*triangle, outside),
            native_cpp.incircle(*triangle, boundary),
        )
    )


def test_python_fallback_orientation_and_incidence(monkeypatch) -> None:
    monkeypatch.setattr(native_cpp, "_compiled", None)

    np.testing.assert_allclose(
        native_cpp.orient2d_many(POINTS, TRIANGLES), (1.0, 1.0)
    )
    np.testing.assert_array_equal(
        native_cpp.triangle_edge_incidence(TRIANGLES), EXPECTED_INCIDENCE
    )
    assert native_cpp.orient2d(POINTS[0], POINTS[1], POINTS[2]) > 0.0
    assert native_cpp.incircle(POINTS[0], POINTS[1], POINTS[2], (0.5, 0.5)) > 0.0


def test_adaptive_predicates_have_deterministic_near_degenerate_truth_values() -> None:
    assert _near_degenerate_predicate_signs() == (1, -1, 1, -1, 0)
    assert _near_degenerate_predicate_signs() == (1, -1, 1, -1, 0)


def test_python_boundary_fallback_matches_adaptive_near_degenerate_predicates(
    monkeypatch,
) -> None:
    monkeypatch.setattr(native_cpp, "_compiled", None)

    assert _native_predicate_signs() == _near_degenerate_predicate_signs()


@pytest.mark.skipif(
    not native_cpp.NATIVE_CPP_AVAILABLE, reason="optional C++ extension not built"
)
def test_compiled_boundary_matches_adaptive_near_degenerate_predicates() -> None:
    assert _native_predicate_signs() == _near_degenerate_predicate_signs()


@pytest.mark.parametrize("dtype", (np.int32, np.int64, np.uint32, np.uint64))
def test_integer_connectivity_widths_are_equivalent(dtype) -> None:
    actual = native_cpp.triangle_edge_incidence(TRIANGLES.astype(dtype))
    np.testing.assert_array_equal(actual, EXPECTED_INCIDENCE)


def test_nonmanifold_connectivity_fails_closed() -> None:
    triangles = np.array(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int64)
    with pytest.raises(ValueError, match="nonmanifold"):
        native_cpp.triangle_edge_incidence(triangles)


@pytest.mark.skipif(
    not native_cpp.NATIVE_CPP_AVAILABLE, reason="optional C++ extension not built"
)
def test_compiled_batch_matches_scalar_predicate() -> None:
    expected = np.array(
        [native_cpp.orient2d(*(POINTS[index] for index in triangle)) for triangle in TRIANGLES]
    )
    np.testing.assert_allclose(
        native_cpp.orient2d_many(POINTS, TRIANGLES), expected, rtol=0.0, atol=0.0
    )
