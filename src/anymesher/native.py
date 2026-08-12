"""Optional boundary between the Python reference mesher and native kernels.

No extension is required.  A native package may register one implementation at
runtime; callers can still explicitly request the deterministic Python path for
qualification and comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .errors import MeshError

__all__ = [
    "NativeBoundary",
    "NativeBoundarySelection",
    "NativeTriangulation",
    "clear_native_boundary",
    "get_native_boundary",
    "has_native_boundary",
    "register_native_boundary",
    "run_native_triangulation",
    "snapshot_native_boundary",
]


@dataclass(frozen=True)
class NativeTriangulation:
    points: np.ndarray
    triangles: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray):
            raise MeshError("a native triangulator must return a float64 point array")
        if not isinstance(self.triangles, np.ndarray):
            raise MeshError("a native triangulator must return an int64 triangle array")
        if (
            self.points.dtype != np.dtype(np.float64)
            or not self.points.dtype.isnative
            or self.points.ndim != 2
            or self.points.shape[1] != 2
            or not self.points.flags.c_contiguous
            or not np.all(np.isfinite(self.points))
        ):
            raise MeshError(
                "a native triangulator must return C-contiguous native float64 points"
            )
        if (
            self.triangles.dtype != np.dtype(np.int64)
            or not self.triangles.dtype.isnative
            or self.triangles.ndim != 2
            or self.triangles.shape[1] != 3
            or not self.triangles.flags.c_contiguous
        ):
            raise MeshError(
                "a native triangulator must return C-contiguous native int64 T3 connectivity"
            )
        points = np.array(self.points, dtype=np.float64, order="C", copy=True)
        triangles = np.array(self.triangles, dtype=np.int64, order="C", copy=True)
        if triangles.size and (np.min(triangles) < 0 or np.max(triangles) >= len(points)):
            raise MeshError("a native triangulator returned invalid connectivity")
        points.setflags(write=False)
        triangles.setflags(write=False)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "triangles", triangles)
        object.__setattr__(
            self, "diagnostics", MappingProxyType(dict(self.diagnostics))
        )


@dataclass(frozen=True)
class NativeBoundarySelection:
    boundary: NativeBoundary
    name: str
    source: str


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
    return snapshot_native_boundary() is not None


def snapshot_native_boundary(
    explicit: NativeBoundary | None = None,
) -> NativeBoundarySelection | None:
    """Resolve exactly one boundary for a call before preparation starts."""

    if explicit is not None:
        if not callable(getattr(explicit, "triangulate", None)):
            raise TypeError("native boundary must provide triangulate()")
        return NativeBoundarySelection(
            explicit, str(getattr(explicit, "name", "native")), "explicit"
        )
    with _LOCK:
        registered = _BOUNDARY
    if registered is not None:
        return NativeBoundarySelection(
            registered,
            str(getattr(registered, "name", "native")),
            "registered",
        )
    from .native_cpp import compiled_native_boundary

    built_in = compiled_native_boundary()
    if built_in is None:
        return None
    return NativeBoundarySelection(built_in, built_in.name, "built_in")


def run_native_triangulation(
    points: np.ndarray,
    segments: np.ndarray,
    outer_loop: np.ndarray,
    hole_loops: tuple[np.ndarray, ...],
    *,
    boundary: NativeBoundary | None = None,
    cancellation_check: Callable[[str], None] | None = None,
) -> NativeTriangulation:
    selection = snapshot_native_boundary(boundary)
    if selection is None:
        raise MeshError("no native triangulation boundary is registered")
    implementation = selection.boundary
    cancellable = getattr(implementation, "triangulate_cancellable", None)
    if callable(cancellable):
        raw: Any = cancellable(
            points,
            segments,
            outer_loop,
            hole_loops,
            cancellation_check=cancellation_check,
        )
    else:
        if cancellation_check is not None:
            cancellation_check("native triangulation start")
        raw = implementation.triangulate(points, segments, outer_loop, hole_loops)
        if cancellation_check is not None:
            cancellation_check("native triangulation complete")
    if isinstance(raw, NativeTriangulation):
        return raw
    if isinstance(raw, tuple) and len(raw) == 2:
        return NativeTriangulation(raw[0], raw[1])
    if isinstance(raw, tuple) and len(raw) == 3:
        return NativeTriangulation(raw[0], raw[1], raw[2])
    result_points = getattr(raw, "points", None)
    result_triangles = getattr(raw, "triangles", None)
    if result_points is None or result_triangles is None:
        raise MeshError("native triangulation result needs points and triangles")
    diagnostics = getattr(raw, "diagnostics", {})
    return NativeTriangulation(result_points, result_triangles, diagnostics)
