"""Optional C++17 kernels with deterministic NumPy-friendly fallbacks.

The extension is intentionally small.  Packaging can build it when a compiler
is available, while source checkouts retain the exact Python predicates used by
the reference triangulator.  This module is the stable boundary consumed by
benchmarks and higher-level native algorithms.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.util import find_spec
from types import TracebackType
from typing import Any

import numpy as np

from .errors import MeshError
from .native import NativeTriangulation


_extension_spec = find_spec(f"{__package__}._native")
if _extension_spec is None:  # Optional for source/developer installations.
    _compiled = None
else:  # A present but unloadable artifact is a packaging failure, not absence.
    from . import _native as _compiled


NATIVE_CPP_AVAILABLE = _compiled is not None
COMPILED_TRIANGULATION_AVAILABLE = bool(
    _compiled is not None and callable(getattr(_compiled, "constrained_triangulate", None))
)
COMPILED_QUALITY_PIPELINE_AVAILABLE = bool(
    _compiled is not None
    and all(
        callable(getattr(_compiled, name, None))
        for name in (
            "pslg_segment_memberships",
            "pslg_domain_classification",
            "validate_triangulation",
            "recombine_decisions",
            "element_quality",
        )
    )
)


class _CancellationSentinel(BaseException):
    pass


def _strict_float64_matrix(value: Any, columns: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array before native dispatch")
    if (
        value.dtype != np.dtype(np.float64)
        or not value.dtype.isnative
        or value.ndim != 2
        or value.shape[1] != columns
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise TypeError(
            f"{name} must be a C-contiguous native float64 matrix with {columns} columns"
        )
    return value


def _strict_int64_matrix(value: Any, columns: int, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array before native dispatch")
    if (
        value.dtype != np.dtype(np.int64)
        or not value.dtype.isnative
        or value.ndim != 2
        or value.shape[1] != columns
        or not value.flags.c_contiguous
    ):
        raise TypeError(
            f"{name} must be a C-contiguous native int64 matrix with {columns} columns"
        )
    return value


def _strict_int64_vector(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array before native dispatch")
    if (
        value.dtype != np.dtype(np.int64)
        or not value.dtype.isnative
        or value.ndim != 1
        or not value.flags.c_contiguous
    ):
        raise TypeError(f"{name} must be a C-contiguous native int64 vector")
    return value


class CompiledNativeBoundary:
    """Strict adapter for the built-in C++17 constrained triangulator."""

    name = "anymesher-cpp17"

    def triangulate(
        self,
        points: np.ndarray,
        segments: np.ndarray,
        outer_loop: np.ndarray,
        hole_loops: tuple[np.ndarray, ...],
    ) -> NativeTriangulation:
        return self.triangulate_cancellable(
            points,
            segments,
            outer_loop,
            hole_loops,
            cancellation_check=None,
        )

    def triangulate_cancellable(
        self,
        points: np.ndarray,
        segments: np.ndarray,
        outer_loop: np.ndarray,
        hole_loops: tuple[np.ndarray, ...],
        *,
        cancellation_check: Any = None,
    ) -> NativeTriangulation:
        if not COMPILED_TRIANGULATION_AVAILABLE:
            raise MeshError("compiled triangulation capability is unavailable")
        made_points = _strict_float64_matrix(points, 2, "points")
        made_segments = _strict_int64_matrix(segments, 2, "segments")
        made_outer = _strict_int64_vector(outer_loop, "outer_loop")
        made_holes = tuple(
            _strict_int64_vector(hole, f"hole_loops[{number}]")
            for number, hole in enumerate(hole_loops)
        )
        offsets = np.zeros(len(made_holes) + 1, dtype=np.int64)
        if made_holes:
            offsets[1:] = np.cumsum([len(hole) for hole in made_holes], dtype=np.int64)
            hole_indices = np.ascontiguousarray(np.concatenate(made_holes), dtype=np.int64)
        else:
            hole_indices = np.empty(0, dtype=np.int64)

        captured: list[tuple[BaseException, TracebackType | None]] = []

        def checked(phase: str) -> None:
            if cancellation_check is None:
                return
            try:
                cancellation_check(phase)
            except BaseException as error:
                captured.append((error, error.__traceback__))
                raise _CancellationSentinel from None

        try:
            raw_triangles, diagnostics = _compiled.constrained_triangulate(
                made_points,
                made_segments,
                made_outer,
                hole_indices,
                offsets,
                checked if cancellation_check is not None else None,
            )
        except _CancellationSentinel:
            error, traceback = captured[0]
            raise error.with_traceback(traceback)
        except RuntimeError as error:
            raise MeshError(str(error)) from error
        triangles = np.ascontiguousarray(raw_triangles, dtype=np.int64).reshape((-1, 3))
        return NativeTriangulation(made_points, triangles, diagnostics)


_COMPILED_BOUNDARY = CompiledNativeBoundary()


def compiled_native_boundary() -> CompiledNativeBoundary | None:
    return _COMPILED_BOUNDARY if COMPILED_TRIANGULATION_AVAILABLE else None


def orient2d(
    first: Sequence[float], second: Sequence[float], third: Sequence[float]
) -> float:
    """Signed twice-area using the compiled long-double path when present."""

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    c = np.asarray(third, dtype=np.float64)
    if a.shape != (2,) or b.shape != (2,) or c.shape != (2,):
        raise ValueError("orient2d points must be 2-vectors")
    if _compiled is not None:
        candidate = float(_compiled.orient2d(*a, *b, *c))
    else:
        candidate = float(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )
    ax, ay = a - c
    bx, by = b - c
    error = 8.0 * np.finfo(float).eps * (abs(ax * by) + abs(ay * bx))
    if abs(candidate) <= error:
        from .triangulation import orient2d as adaptive_orient2d

        return adaptive_orient2d(a, b, c)
    return candidate


def incircle(
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
    point: Sequence[float],
) -> float:
    """Oriented in-circle determinant with a dependency-free fallback."""

    values = [np.asarray(item, dtype=np.float64) for item in (first, second, third, point)]
    if any(item.shape != (2,) for item in values):
        raise ValueError("incircle points must be 2-vectors")
    a, b, c, d = values
    offsets = np.vstack((a - d, b - d, c - d))
    if _compiled is not None:
        candidate = float(_compiled.incircle(*a, *b, *c, *d))
    else:
        lifted = np.column_stack((offsets, np.sum(offsets * offsets, axis=1)))
        candidate = float(np.linalg.det(lifted))
    first_offset, second_offset, third_offset = offsets
    first_lift, second_lift, third_lift = np.sum(offsets * offsets, axis=1)
    first_cross = (
        second_offset[0] * third_offset[1]
        - second_offset[1] * third_offset[0]
    )
    second_cross = (
        first_offset[0] * third_offset[1]
        - first_offset[1] * third_offset[0]
    )
    third_cross = (
        first_offset[0] * second_offset[1]
        - first_offset[1] * second_offset[0]
    )
    scale = (
        abs(first_lift * first_cross)
        + abs(second_lift * second_cross)
        + abs(third_lift * third_cross)
    )
    if abs(candidate) <= 32.0 * np.finfo(float).eps * scale:
        from .triangulation import incircle as adaptive_incircle
        from .triangulation import orient2d as adaptive_orient2d

        candidate = adaptive_incircle(a, b, c, d)
        if adaptive_orient2d(a, b, c) < 0.0:
            candidate = -candidate
    return candidate


def orient2d_many(points: Any, triangles: Any) -> np.ndarray:
    """Return signed twice-area for dense triangle connectivity."""

    made_points = np.ascontiguousarray(points, dtype=np.float64)
    made_triangles = np.ascontiguousarray(triangles, dtype=np.int64)
    if made_points.ndim != 2 or made_points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if made_triangles.ndim != 2 or made_triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")
    if _compiled is not None:
        return np.asarray(
            _compiled.orient2d_many(made_points, made_triangles), dtype=np.float64
        )
    coordinates = made_points[made_triangles]
    first = coordinates[:, 1] - coordinates[:, 0]
    second = coordinates[:, 2] - coordinates[:, 0]
    return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]


def triangle_edge_incidence(triangles: Any) -> np.ndarray:
    """Return ``(node_a, node_b, left_row, right_row)`` in edge order."""

    made = np.ascontiguousarray(triangles, dtype=np.int64)
    if made.ndim != 2 or made.shape[1] != 3:
        raise ValueError("triangles must have shape (m, 3)")
    if _compiled is not None:
        return np.asarray(_compiled.triangle_adjacency(made), dtype=np.int64).reshape((-1, 4))
    incidence: dict[tuple[int, int], list[int]] = {}
    for row, triangle in enumerate(made):
        for index in range(3):
            edge = tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
            incidence.setdefault(edge, []).append(row)
    if any(len(rows) > 2 for rows in incidence.values()):
        raise ValueError("triangle connectivity contains a nonmanifold edge")
    return np.asarray(
        [(*edge, rows[0], rows[1] if len(rows) == 2 else -1) for edge, rows in sorted(incidence.items())],
        dtype=np.int64,
    ).reshape((-1, 4))


def pslg_segment_memberships(
    points: Any, segments: Any, tolerance: float
) -> tuple[tuple[int, ...], ...] | None:
    if not COMPILED_QUALITY_PIPELINE_AVAILABLE:
        return None
    made_points = _strict_float64_matrix(points, 2, "points")
    made_segments = _strict_int64_matrix(segments, 2, "segments")
    try:
        rows = _compiled.pslg_segment_memberships(
            made_points, made_segments, float(tolerance)
        )
    except RuntimeError as error:
        raise MeshError(str(error)) from error
    return tuple(tuple(int(row) for row in item) for item in rows)


def pslg_domain_classification(
    points: Any,
    outer: Any,
    holes: Sequence[Any],
    tolerance: float,
) -> np.ndarray | None:
    if not COMPILED_QUALITY_PIPELINE_AVAILABLE:
        return None
    made_points = _strict_float64_matrix(points, 2, "points")
    made_outer = _strict_int64_vector(outer, "outer")
    made_holes = tuple(
        _strict_int64_vector(hole, f"holes[{number}]")
        for number, hole in enumerate(holes)
    )
    offsets = np.zeros(len(made_holes) + 1, dtype=np.int64)
    if made_holes:
        offsets[1:] = np.cumsum([len(hole) for hole in made_holes], dtype=np.int64)
        indices = np.ascontiguousarray(np.concatenate(made_holes), dtype=np.int64)
    else:
        indices = np.empty(0, dtype=np.int64)
    try:
        return np.asarray(
            _compiled.pslg_domain_classification(
                made_points, made_outer, indices, offsets, float(tolerance)
            ),
            dtype=bool,
        ).reshape((-1, 3))
    except RuntimeError as error:
        raise MeshError(str(error)) from error


def validate_native_triangulation(
    points: Any,
    triangles: Any,
    segments: Any,
    boundary_segments: Any,
    mandatory_segments: Any,
    outer: Any,
    holes: Sequence[Any],
    tolerance: float,
) -> np.ndarray | None:
    if not COMPILED_QUALITY_PIPELINE_AVAILABLE:
        return None
    made_points = _strict_float64_matrix(points, 2, "points")
    made_triangles = _strict_int64_matrix(triangles, 3, "triangles")
    made_segments = _strict_int64_matrix(segments, 2, "segments")
    made_boundary = _strict_int64_matrix(
        boundary_segments, 2, "boundary_segments"
    )
    made_mandatory = _strict_int64_matrix(
        mandatory_segments, 2, "mandatory_segments"
    )
    made_outer = _strict_int64_vector(outer, "outer")
    made_holes = tuple(
        _strict_int64_vector(hole, f"holes[{number}]")
        for number, hole in enumerate(holes)
    )
    offsets = np.zeros(len(made_holes) + 1, dtype=np.int64)
    if made_holes:
        offsets[1:] = np.cumsum([len(hole) for hole in made_holes], dtype=np.int64)
        indices = np.ascontiguousarray(np.concatenate(made_holes), dtype=np.int64)
    else:
        indices = np.empty(0, dtype=np.int64)
    try:
        rows = _compiled.validate_triangulation(
            made_points,
            made_triangles,
            made_segments,
            made_boundary,
            made_mandatory,
            made_outer,
            indices,
            offsets,
            float(tolerance),
        )
    except RuntimeError as error:
        raise MeshError(str(error)) from error
    return np.ascontiguousarray(rows, dtype=np.int64).reshape((-1, 3))


def native_recombination_decisions(
    points: Any,
    triangles: Any,
    triangle_ids: Any,
    node_ids: Any,
    active_rows: Any,
    protected_edges: Any,
    *,
    min_scaled_jacobian: float,
    max_aspect_ratio: float,
    min_angle: float,
    max_angle: float,
    max_warpage: float,
    max_exchange_work: int,
) -> dict[str, Any] | None:
    if not COMPILED_QUALITY_PIPELINE_AVAILABLE:
        return None
    empty = np.empty(0, dtype=np.int64)
    try:
        return dict(
            _compiled.recombine_decisions(
                _strict_float64_matrix(points, 3, "points"),
                _strict_int64_matrix(triangles, 3, "triangles"),
                _strict_int64_vector(triangle_ids, "triangle_ids"),
                _strict_int64_vector(node_ids, "node_ids"),
                _strict_int64_vector(active_rows, "active_rows"),
                _strict_int64_matrix(protected_edges, 2, "protected_edges"),
                empty,
                empty,
                float(min_scaled_jacobian),
                float(max_aspect_ratio),
                float(min_angle),
                float(max_angle),
                float(max_warpage),
                int(max_exchange_work),
            )
        )
    except RuntimeError as error:
        raise MeshError(str(error)) from error


def native_element_quality(
    points: Any, cells: Any, corners: int
) -> np.ndarray | None:
    if not COMPILED_QUALITY_PIPELINE_AVAILABLE:
        return None
    made_points = _strict_float64_matrix(points, 3, "points")
    made_cells = _strict_int64_matrix(cells, int(corners), "cells")
    try:
        return np.asarray(
            _compiled.element_quality(made_points, made_cells, int(corners)),
            dtype=np.float64,
        ).reshape((-1, 6))
    except RuntimeError as error:
        raise MeshError(str(error)) from error


__all__ = [
    "COMPILED_TRIANGULATION_AVAILABLE",
    "COMPILED_QUALITY_PIPELINE_AVAILABLE",
    "CompiledNativeBoundary",
    "NATIVE_CPP_AVAILABLE",
    "compiled_native_boundary",
    "incircle",
    "orient2d",
    "orient2d_many",
    "native_recombination_decisions",
    "native_element_quality",
    "pslg_domain_classification",
    "pslg_segment_memberships",
    "triangle_edge_incidence",
    "validate_native_triangulation",
]
