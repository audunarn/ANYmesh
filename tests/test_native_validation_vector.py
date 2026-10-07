"""Focused parity and counterexample tests for the vectorized native-result
validation checks (M3) and the vectorized point-in-polygon test (M4)."""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import native_cpp
from anymesher.errors import MeshError
from anymesher.native_cpp import (
    _orientation_rows,
    native_constrained_smoothing,
    native_local_edge_flip,
)
from anymesher.optimization import constrained_smoothing, local_edge_flip
from anymesher.surface_mesh import _domain_contains, _inside, _inside_scalar
from anymesher.triangulation import orient2d


class _FakeNative:
    """Fake compiled module; only the symbol under test is ever called."""

    def __init__(self, flip=None, smoothing=None) -> None:
        self._flip = flip
        self._smoothing = smoothing

    def native_v2_local_edge_flip(self, *args):
        if self._flip is None:
            raise AssertionError("unexpected local-edge-flip kernel call")
        return self._flip(*args)

    def native_v2_constrained_smoothing(self, *args):
        if self._smoothing is None:
            raise AssertionError("unexpected constrained-smoothing kernel call")
        return self._smoothing(*args)


def _install(monkeypatch, kernel: _FakeNative) -> None:
    monkeypatch.setattr(native_cpp, "_compiled", kernel)
    monkeypatch.setattr(native_cpp, "_complete_native_v2_available", lambda: True)


def _identity_metrics(count: int) -> np.ndarray:
    return np.repeat(np.eye(2)[None, :, :], count, axis=0)


def _flip_diagnostics(flip_count: int, queue_visits: int) -> dict:
    return {"flip_count": flip_count, "queue_visits": queue_visits, "converged": True}


def _smoothing_diagnostics(
    iterations: int, accepted: int, rejected: int = 0
) -> dict:
    return {
        "iterations": iterations,
        "accepted_moves": accepted,
        "rejected_moves": rejected,
        "converged": True,
    }


def _recording_oracle(calls: list):
    def oracle(first, second, third):
        calls.append((first, second, third))
        return orient2d(first, second, third)

    return oracle


# ---------------------------------------------------------------------------
# M3: orientation-row helper parity with the exact orient2d oracle
# ---------------------------------------------------------------------------


def test_orientation_rows_match_the_oracle_bit_for_bit() -> None:
    points = np.asarray(
        (
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (1.0, 1.0), (2.0, 2.0),
            (0.0, 0.0), (1.0, 0.0), (2.0, 0.0),
            (0.0, 0.0), (1.0e8, 1.0e-8), (2.0e8, 2.0e-8 + 1.0e-23),
        )
    )
    rows = np.asarray(
        ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)), dtype=np.int64
    )
    calls: list = []
    values = _orientation_rows(points, rows, _recording_oracle(calls))
    expected = [orient2d(points[a], points[b], points[c]) for a, b, c in rows]
    assert [float(value) for value in values] == expected
    # The two exactly collinear rows and the near-degenerate row are
    # uncertain in float arithmetic and must be resolved by the exact
    # oracle; the plain CCW row must not be.
    assert len(calls) == 3


def test_orientation_rows_near_degenerate_positive_uses_exact_oracle() -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0e8, 1.0e-8), (2.0e8, 2.0e-8 + 1.0e-23))
    )
    rows = np.asarray(((0, 1, 2),), dtype=np.int64)
    exact = orient2d(points[0], points[1], points[2])
    assert exact > 0.0
    calls: list = []
    values = _orientation_rows(points, rows, _recording_oracle(calls))
    assert len(calls) == 1
    assert float(values[0]) == exact


def test_orientation_rows_sum_matches_original_python_accumulation() -> None:
    points = np.asarray(
        (
            (0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0),
            (0.0, 0.0), (1.0, 1.0), (2.0, 2.0),
            (0.0, 0.0), (1.0e8, 1.0e-8), (2.0e8, 2.0e-8 + 1.0e-23),
        )
    )
    triangles = np.asarray(
        ((0, 1, 2), (0, 2, 3), (4, 5, 6), (7, 8, 9)), dtype=np.int64
    )
    values = _orientation_rows(points, triangles, orient2d)
    original = sum(
        float(orient2d(*points[triangle])) for triangle in triangles
    )
    assert sum(float(value) for value in values) == original


def test_orientation_rows_extremes_match_oracle_under_strict_errstate() -> None:
    # Finite extreme coordinates: the screening arithmetic overflows
    # (row 0), cancels through an invalid inf-inf difference (row 2), and
    # underflows to zero (row 1).  The oracle uses Python floats and never
    # raises, so strict np.errstate settings must not raise before the
    # exact fallback resolves these rows.
    points = np.asarray(
        (
            (1.7e308, 0.0), (0.0, 1.0e-10), (-1.7e308, 0.0),
            (1.0e-200, 1.0e-200), (1.0e-200, 1.0e-200), (0.0, 0.0),
            (1.0e200, 1.0e200), (1.0e200, 1.0e200), (0.0, 0.0),
        )
    )
    rows = np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8)), dtype=np.int64)
    expected = [orient2d(points[a], points[b], points[c]) for a, b, c in rows]
    calls: list = []
    with np.errstate(over="raise", invalid="raise", under="raise"):
        values = _orientation_rows(points, rows, _recording_oracle(calls))
    assert [float(value) for value in values] == expected
    # Every extreme row must be resolved by the exact oracle, not decided
    # by overflowed or underflowed screening values.
    assert len(calls) == 3


# ---------------------------------------------------------------------------
# M3: local-edge-flip result-contract counterexamples (original messages)
# ---------------------------------------------------------------------------


SQUARE_POINTS = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
SQUARE_TRIANGLES = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
NO_PROTECTED = np.empty((0, 2), dtype=np.int64)


def _run_flip(monkeypatch, points, triangles, rows, diagnostics, protected=NO_PROTECTED):
    kernel = _FakeNative(flip=lambda *_args: ([list(row) for row in rows], diagnostics))
    _install(monkeypatch, kernel)
    return native_local_edge_flip(
        points, triangles, protected, _identity_metrics(len(points)), 4
    )


def test_edge_flip_rejects_repeated_node_with_row_number(monkeypatch) -> None:
    rows = ((0, 1, 1), (0, 2, 3))
    with pytest.raises(MeshError, match="triangle 0 repeats a node"):
        _run_flip(
            monkeypatch, SQUARE_POINTS, SQUARE_TRIANGLES, rows, _flip_diagnostics(1, 1)
        )


def test_edge_flip_rejects_collinear_triangle_through_exact_oracle(monkeypatch) -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.0, 1.0)))
    triangles = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int64)
    rows = ((0, 1, 2), (1, 2, 3))
    with pytest.raises(MeshError, match="triangle 0 is not strict CCW"):
        _run_flip(monkeypatch, points, triangles, rows, _flip_diagnostics(1, 1))


def test_edge_flip_rejects_duplicate_triangles(monkeypatch) -> None:
    rows = ((0, 1, 2), (0, 1, 2))
    with pytest.raises(MeshError, match="contains duplicate triangles"):
        _run_flip(
            monkeypatch, SQUARE_POINTS, SQUARE_TRIANGLES, rows, _flip_diagnostics(1, 1)
        )


def test_edge_flip_combined_invalid_rows_report_original_first_error(
    monkeypatch,
) -> None:
    # Row 0 is clockwise and row 1 repeats a node: the original loop
    # checks row 0's orientation before row 1's repeats, so the vector
    # screen must not report the later row's repeats first.
    rows = ((0, 2, 1), (0, 1, 1))
    with pytest.raises(MeshError, match="triangle 0 is not strict CCW"):
        _run_flip(
            monkeypatch, SQUARE_POINTS, SQUARE_TRIANGLES, rows, _flip_diagnostics(1, 1)
        )


def test_edge_flip_combined_repeats_outrank_later_orientation(monkeypatch) -> None:
    # Row 0 repeats and row 1 is clockwise: repeats win within row 0.
    rows = ((0, 1, 1), (0, 3, 2))
    with pytest.raises(MeshError, match="triangle 0 repeats a node"):
        _run_flip(
            monkeypatch, SQUARE_POINTS, SQUARE_TRIANGLES, rows, _flip_diagnostics(1, 1)
        )


def test_edge_flip_orientation_outranks_duplicate_triangles(monkeypatch) -> None:
    # Both rows are the same clockwise triangle: the original loop raises
    # row 0's orientation error before the duplicate-triangle check.
    rows = ((0, 2, 1), (0, 2, 1))
    with pytest.raises(MeshError, match="triangle 0 is not strict CCW"):
        _run_flip(
            monkeypatch, SQUARE_POINTS, SQUARE_TRIANGLES, rows, _flip_diagnostics(1, 1)
        )


def test_edge_flip_rejects_nonmanifold_result(monkeypatch) -> None:
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (2.0, 2.0), (2.0, 3.0))
    )
    triangles = np.asarray(((0, 1, 2), (0, 1, 3), (0, 1, 4)), dtype=np.int64)
    with pytest.raises(MeshError, match="result is non-manifold"):
        _run_flip(monkeypatch, points, triangles, triangles, _flip_diagnostics(0, 0))


def test_edge_flip_rejects_removed_protected_edge(monkeypatch) -> None:
    rows = ((0, 1, 3), (1, 2, 3))
    with pytest.raises(MeshError, match="removed a protected edge"):
        _run_flip(
            monkeypatch,
            SQUARE_POINTS,
            SQUARE_TRIANGLES,
            rows,
            _flip_diagnostics(1, 1),
            protected=np.asarray(((2, 0),), dtype=np.int64),
        )


def test_edge_flip_accepts_unchanged_result(monkeypatch) -> None:
    result = _run_flip(
        monkeypatch, SQUARE_POINTS, SQUARE_TRIANGLES, SQUARE_TRIANGLES,
        _flip_diagnostics(0, 0),
    )
    assert result is not None
    np.testing.assert_array_equal(result[0], SQUARE_TRIANGLES)
    assert result[1] == {"flip_count": 0, "queue_visits": 0, "converged": True}


def test_local_edge_flip_public_path_validates_kernel_results(monkeypatch) -> None:
    # Caller integration: the public optimizer routes through the same
    # vectorized validation and surfaces its original errors.
    kernel = _FakeNative(
        flip=lambda *_args: (
            [list(row) for row in ((0, 1, 1), (0, 2, 3))],
            _flip_diagnostics(1, 1),
        )
    )
    _install(monkeypatch, kernel)
    with pytest.raises(MeshError, match="repeats a node"):
        local_edge_flip(SQUARE_POINTS, SQUARE_TRIANGLES)

    kernel = _FakeNative(
        flip=lambda *_args: (
            [list(row) for row in SQUARE_TRIANGLES],
            _flip_diagnostics(0, 0),
        )
    )
    _install(monkeypatch, kernel)
    result = local_edge_flip(SQUARE_POINTS, SQUARE_TRIANGLES)
    np.testing.assert_array_equal(result.triangles, SQUARE_TRIANGLES)
    assert result.flip_count == 0


# ---------------------------------------------------------------------------
# M3: constrained-smoothing result-contract counterexamples
# ---------------------------------------------------------------------------


def _run_smoothing(
    monkeypatch,
    points,
    cells,
    result_points,
    moved,
    diagnostics,
    *,
    fixed=None,
    constraints=None,
    preserve_boundary=False,
):
    kernel = _FakeNative(
        smoothing=lambda *_args: (
            np.asarray(result_points, dtype=np.float64),
            np.asarray(moved, dtype=np.int64),
            diagnostics,
        )
    )
    _install(monkeypatch, kernel)
    return native_constrained_smoothing(
        points,
        cells,
        np.asarray(sorted(fixed or ()), dtype=np.int64).reshape((-1, 1)),
        np.asarray(sorted(constraints or ()), dtype=np.int64).reshape((-1, 2)),
        preserve_boundary,
        _identity_metrics(len(points)),
        8,
        0.5,
    )


def test_smoothing_rejects_inverted_cell_with_cell_number(monkeypatch) -> None:
    points = np.asarray(
        (
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
            (0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
        )
    )
    cells = np.asarray(((0, 1, 2, -1), (3, 4, 5, -1)), dtype=np.int64)
    moved_points = points.copy()
    moved_points[4] = (-1.0, 0.0)
    with pytest.raises(MeshError, match="inverted cell 1"):
        _run_smoothing(
            monkeypatch, points, cells, moved_points, (4,),
            _smoothing_diagnostics(1, 1),
        )


def test_smoothing_rejects_exactly_collinear_cell_through_exact_oracle(
    monkeypatch,
) -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (0.0, 1.0)))
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    with pytest.raises(MeshError, match="inverted cell 0"):
        _run_smoothing(
            monkeypatch, points, cells, points, (),
            _smoothing_diagnostics(0, 0),
        )


def test_smoothing_rejects_moved_constrained_edge_endpoint(monkeypatch) -> None:
    moved_points = SQUARE_POINTS.copy()
    moved_points[0] = (0.1, 0.1)
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    with pytest.raises(MeshError, match="moved a fixed node"):
        _run_smoothing(
            monkeypatch, SQUARE_POINTS, cells, moved_points, (0,),
            _smoothing_diagnostics(1, 1),
            constraints=((0, 1),),
        )


def test_smoothing_extreme_coordinates_keep_original_no_raise_semantics(
    monkeypatch,
) -> None:
    # Every corner orientation is ~1e200, so the original before*after
    # product overflows to inf as a Python-float multiplication; the
    # original rule treats that as "not inverted", and strict np.errstate
    # settings must not raise where the original did not.
    big = 1.0e100
    points = np.asarray(((0.0, 0.0), (big, 0.0), (big, big), (0.0, big)))
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    with np.errstate(over="raise", invalid="raise", under="raise"):
        result = _run_smoothing(
            monkeypatch, points, cells, points, (), _smoothing_diagnostics(0, 0)
        )
    assert result is not None
    np.testing.assert_array_equal(result[0], points)
    # Scalar-oracle agreement: the original per-corner rule finds no
    # inversion even though each product overflows to inf.
    for cell in cells:
        for index in range(len(cell)):
            previous = int(cell[(index - 1) % len(cell)])
            current = int(cell[index])
            following = int(cell[(index + 1) % len(cell)])
            before = float(
                orient2d(points[previous], points[current], points[following])
            )
            assert before > 0.0
            assert not (before == 0.0 or before * before <= 0.0)


def test_constrained_smoothing_public_path_validates_kernel_results(
    monkeypatch,
) -> None:
    # Caller integration: the public optimizer routes through the same
    # vectorized validation and surfaces its original errors.
    cells = ((0, 1, 2), (0, 2, 3))
    kernel = _FakeNative(
        smoothing=lambda *_args: (
            SQUARE_POINTS.copy(),
            np.asarray((1,), dtype=np.int64),
            _smoothing_diagnostics(1, 1),
        )
    )
    _install(monkeypatch, kernel)
    result = constrained_smoothing(
        SQUARE_POINTS, cells, preserve_boundary=False, iterations=2
    )
    np.testing.assert_array_equal(result.points, SQUARE_POINTS)
    assert result.moved_nodes.tolist() == [1]
    assert result.iterations == 1

    inverted = SQUARE_POINTS.copy()
    inverted[1] = (0.8, 0.9)
    kernel = _FakeNative(
        smoothing=lambda *_args: (
            inverted,
            np.asarray((1,), dtype=np.int64),
            _smoothing_diagnostics(1, 1),
        )
    )
    _install(monkeypatch, kernel)
    with pytest.raises(MeshError, match="inverted cell 0"):
        constrained_smoothing(
            SQUARE_POINTS, cells, preserve_boundary=False, iterations=2
        )


# ---------------------------------------------------------------------------
# M4: vectorized even-odd point-in-polygon parity with the scalar original
# ---------------------------------------------------------------------------


CONCAVE_RING = np.asarray(
    ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (2.0, 1.0), (0.0, 4.0))
)


def _grid_points() -> list[tuple[float, float]]:
    points = [
        (float(x) * 0.5, float(y) * 0.5)
        for x in range(-2, 11)
        for y in range(-2, 11)
    ]
    points.extend(
        (
            (2.0, 1.0),  # reflex vertex
            (1.0, 0.0),  # on a horizontal boundary edge
            (4.0, 2.0),  # on a vertical boundary edge
            (3.0, 2.5),  # on a crossing (slanted) edge
            (2.0, 2.5),  # inside the notch, outside the polygon
            (2.0, 0.5),  # inside the polygon
            (-0.1, 0.0),  # just outside
        )
    )
    return points


@pytest.mark.parametrize("point", _grid_points())
def test_inside_matches_scalar_for_grid_boundary_and_crossing_points(point) -> None:
    assert _inside(point, CONCAVE_RING) == _inside_scalar(point, CONCAVE_RING)


def test_inside_strict_boundary_semantics_are_preserved() -> None:
    # On-edge verdicts follow the original strict inequalities: the
    # point on the right edge stays outside because crossing > x is
    # strict, while the bottom-edge point is counted through the
    # crossing at vertex (4, 0).
    assert _inside((4.0, 2.0), CONCAVE_RING) is False
    assert _inside((1.0, 0.0), CONCAVE_RING) is True
    assert _inside((2.0, 0.5), CONCAVE_RING) is True
    assert _inside((2.0, 2.5), CONCAVE_RING) is False
    assert _inside((1.0, -0.1), CONCAVE_RING) is False


@pytest.mark.parametrize(
    "ring",
    (
        CONCAVE_RING,
        CONCAVE_RING.astype(np.float32),
        CONCAVE_RING.astype(np.int64),
        [tuple(row) for row in CONCAVE_RING],
        np.column_stack((CONCAVE_RING, np.zeros(len(CONCAVE_RING)))),
    ),
)
def test_inside_matches_scalar_across_ring_representations(ring) -> None:
    for point in _grid_points():
        assert _inside(point, ring) == _inside_scalar(point, ring)


def test_inside_float64_ring_parity_across_point_types() -> None:
    # The vector path is float64-only; for a float64 ring every point
    # scalar type (Python float/int, NumPy scalars, list) must agree with
    # the scalar authority, including NumPy2 promotion corners.
    locations = (
        (2.0, 0.5), (2.0, 2.5), (4.0, 2.0), (1.0, 0.0),
        (0.5, 0.5), (2.0, 1.0), (-0.1, 0.0), (3.0, 2.5),
    )
    for x, y in locations:
        variants = [
            (x, y),
            np.asarray((x, y)),
            np.asarray((x, y), dtype=np.float32),
            [x, y],
            (np.float64(x), np.float32(y)),
        ]
        if x == int(x) and y == int(y):
            variants.append((int(x), int(y)))
        for point in variants:
            assert _inside(point, CONCAVE_RING) == _inside_scalar(
                point, CONCAVE_RING
            ), (point,)


def test_inside_extreme_float64_ring_matches_scalar() -> None:
    # Crossing arithmetic overflows and underflows; both paths run the
    # identical float64 operations, so every verdict must match.
    huge = 1.0e308
    big_ring = np.asarray(
        ((0.0, -huge), (huge, huge), (0.0, huge), (-huge, huge))
    )
    tiny = 1.0e-200
    tiny_ring = np.asarray(
        ((0.0, 0.0), (tiny, 0.0), (tiny, tiny), (0.0, tiny))
    )
    probes = (
        (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (huge, 0.0),
        (0.0, huge / 2.0), (tiny / 2.0, tiny / 2.0), (2.0 * tiny, 2.0 * tiny),
    )
    for point in probes:
        assert _inside(point, big_ring) == _inside_scalar(point, big_ring)
        assert _inside(point, tiny_ring) == _inside_scalar(point, tiny_ring)


def test_non_float64_rings_route_to_the_scalar_authority(monkeypatch) -> None:
    import anymesher.surface_mesh as surface_mesh

    calls: list = []
    original = surface_mesh._inside_scalar

    def spy(point, ring):
        calls.append(ring)
        return original(point, ring)

    monkeypatch.setattr(surface_mesh, "_inside_scalar", spy)
    rings = (
        CONCAVE_RING.astype(np.float32),
        CONCAVE_RING.astype(np.float16),
        CONCAVE_RING.astype(np.int64),
        CONCAVE_RING.astype(np.int32),
        CONCAVE_RING.astype(np.uint8),
        CONCAVE_RING.astype(object),
    )
    for ring in rings:
        for point in ((2.0, 0.5), (2.0, 2.5), (4.0, 2.0)):
            assert _inside(point, ring) == original(point, ring)
            assert calls and calls[-1] is ring
    # A Boolean ring also routes to the scalar authority, where the
    # original NumPy scalar Boolean subtraction raises TypeError; the
    # vector path must raise the identical exception (same type and
    # message) rather than evaluate.
    bool_ring = CONCAVE_RING.astype(np.bool_)
    with pytest.raises(TypeError) as scalar_error:
        original((2.0, 0.5), bool_ring)
    with pytest.raises(TypeError) as vector_error:
        _inside((2.0, 0.5), bool_ring)
    assert str(vector_error.value) == str(scalar_error.value)
    assert calls and calls[-1] is bool_ring
    # A float64 ring stays on the vector path.
    before = len(calls)
    assert _inside((2.0, 0.5), CONCAVE_RING) == original(
        (2.0, 0.5), CONCAVE_RING
    )
    assert len(calls) == before


def test_inside_falls_back_to_scalar_for_ragged_ring() -> None:
    ring = [(0.0, 0.0, 5.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    # The original scalar path raises inside np.vstack for ragged
    # rings; the fallback must preserve that error.
    with pytest.raises(ValueError):
        _inside((1.0, 0.5), ring)
    with pytest.raises(ValueError):
        _inside_scalar((1.0, 0.5), ring)


def test_inside_falls_back_to_scalar_for_single_column_ring() -> None:
    ring = np.asarray(((0.0,), (1.0,)))
    with pytest.raises(IndexError):
        _inside((0.5, 0.5), ring)
    with pytest.raises(IndexError):
        _inside_scalar((0.5, 0.5), ring)


def test_inside_preserves_point_unpack_errors() -> None:
    with pytest.raises(ValueError):
        _inside((0.0, 0.0, 0.0), CONCAVE_RING)
    with pytest.raises(ValueError):
        _inside_scalar((0.0, 0.0, 0.0), CONCAVE_RING)


def test_domain_contains_integration_matches_scalar_parity() -> None:
    outer = np.asarray(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    hole = np.asarray(((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)))
    for point in ((0.5, 0.5), (1.5, 1.5), (3.5, 3.5), (2.5, 1.5), (0.0, 2.0)):
        expected = _inside_scalar(point, outer) and not _inside_scalar(point, hole)
        assert _domain_contains(point, outer, (hole,)) is expected
