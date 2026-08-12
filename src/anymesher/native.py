"""Optional boundary between the Python reference mesher and native kernels.

No extension is required.  A native package may register one implementation at
runtime; callers can still explicitly request the deterministic Python path for
qualification and comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .errors import MeshError

__all__ = [
    "NativeBoundary",
    "NativeTriangulation",
    "clear_native_boundary",
    "get_native_boundary",
    "has_native_boundary",
    "register_native_boundary",
    "run_native_triangulation",
]


@dataclass(frozen=True)
class NativeTriangulation:
    points: np.ndarray
    triangles: np.ndarray

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.points, dtype=np.float64)
        triangles = np.ascontiguousarray(self.triangles, dtype=np.int64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise MeshError("a native triangulator returned invalid points")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise MeshError("a native triangulator must return T3 connectivity")
        if triangles.size and (np.min(triangles) < 0 or np.max(triangles) >= len(points)):
            raise MeshError("a native triangulator returned invalid connectivity")
        points.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "triangles", triangles)


@runtime_checkable
class NativeBoundary(Protocol):
    """Protocol intentionally small enough for a C, Rust, or C++ adapter."""

    name: str

    def triangulate(
        self,
        points: np.ndarray,
        segments: np.ndarray,
        outer_loop: np.ndarray,
        hole_loops: tuple[np.ndarray, ...],
    ) -> NativeTriangulation | tuple[np.ndarray, np.ndarray]: ...


_LOCK = RLock()
_BOUNDARY: NativeBoundary | None = None


def register_native_boundary(boundary: NativeBoundary | None) -> NativeBoundary | None:
    """Register a native implementation and return the previous one."""

    if boundary is not None and not callable(getattr(boundary, "triangulate", None)):
        raise TypeError("native boundary must provide triangulate()")
    global _BOUNDARY
    with _LOCK:
        previous = _BOUNDARY
        _BOUNDARY = boundary
    return previous


def clear_native_boundary() -> NativeBoundary | None:
    return register_native_boundary(None)


def get_native_boundary() -> NativeBoundary | None:
    with _LOCK:
        return _BOUNDARY


def has_native_boundary() -> bool:
    return get_native_boundary() is not None


def run_native_triangulation(
    points: np.ndarray,
    segments: np.ndarray,
    outer_loop: np.ndarray,
    hole_loops: tuple[np.ndarray, ...],
    *,
    boundary: NativeBoundary | None = None,
) -> NativeTriangulation:
    implementation = boundary if boundary is not None else get_native_boundary()
    if implementation is None:
        raise MeshError("no native triangulation boundary is registered")
    raw: Any = implementation.triangulate(points, segments, outer_loop, hole_loops)
    if isinstance(raw, NativeTriangulation):
        return raw
    if isinstance(raw, tuple) and len(raw) == 2:
        return NativeTriangulation(raw[0], raw[1])
    result_points = getattr(raw, "points", None)
    result_triangles = getattr(raw, "triangles", None)
    if result_points is None or result_triangles is None:
        raise MeshError("native triangulation result needs points and triangles")
    return NativeTriangulation(result_points, result_triangles)

