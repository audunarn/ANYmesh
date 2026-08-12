"""Optional C++17 kernels with deterministic NumPy-friendly fallbacks.

The extension is intentionally small.  Packaging can build it when a compiler
is available, while source checkouts retain the exact Python predicates used by
the reference triangulator.  This module is the stable boundary consumed by
benchmarks and higher-level native algorithms.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

try:  # Optional by design; ``setup.py`` marks the extension optional as well.
    from . import _native as _compiled
except ImportError:  # pragma: no cover - depends on local build capability
    _compiled = None


NATIVE_CPP_AVAILABLE = _compiled is not None


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


__all__ = [
    "NATIVE_CPP_AVAILABLE",
    "incircle",
    "orient2d",
    "orient2d_many",
    "triangle_edge_incidence",
]
