"""Optional boundary between the Python reference mesher and native kernels.

No extension is required.  A native package may register one implementation at
runtime; callers can still explicitly request the deterministic Python path for
qualification and comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
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


class _ReferenceSeedCallbackFailure(BaseException):
    """Keep callback exceptions distinct from native construction refusal."""


def _run_reference_native_seed(points, orientation, circle, cancellation_check):
    """Optional insertion only; the Python caller owns all PSLG finishing.

    The new function is the capability marker: an older installed extension
    must never silently substitute its different constrained triangulator.
    Registered third-party boundaries do not participate in this private
    Python-reference acceleration route.
    """
    from types import BuiltinFunctionType

    module_name = f"{__package__}._native"
    if find_spec(module_name) is None:
        return None, "reference_seed_capability_absent"
    # A present but unloadable extension is a packaging failure, just as in
    # native_cpp.py.  In particular, missing DLL imports must not be hidden.
    extension = import_module(module_name)
    kernel = getattr(extension, "reference_delaunay", None)
    if type(kernel) is not BuiltinFunctionType:
        return None, "reference_seed_capability_absent"
    if (type(points) is not np.ndarray or points.dtype != np.dtype(np.float64)
            or points.ndim != 2 or points.shape[1] != 2
            or not points.flags.c_contiguous):
        return None, "reference_seed_array_unsupported"
    magnitudes = np.abs(points)
    nonzero = magnitudes[magnitudes != 0.]
    # This range also bounds the common binary denominator, so exact
    # orientation expansions and degree-four predicates cannot under/overflow.
    if (not np.isfinite(magnitudes).all()
            or np.max(magnitudes, initial=0.) > 2.**64
            or (len(nonzero) and np.min(nonzero) < 2.**-64)):
        return None, "reference_seed_coordinate_range"
    captured = []

    def checked(function):
        def invoke(*args):
            try:
                return function(*args)
            except BaseException as error:
                captured.append((error, error.__traceback__))
                raise _ReferenceSeedCallbackFailure from None
        return invoke

    try:
        raw = kernel(
            points, checked(orientation), checked(circle),
            checked(cancellation_check) if cancellation_check is not None else None,
        )
    except _ReferenceSeedCallbackFailure:
        error, traceback = captured[0]
        raise error.with_traceback(traceback)
    except RuntimeError as error:
        # A refused cavity/point-location attempt has published nothing.
        # The unchanged full Python insertion remains authoritative.
        return None, "reference_seed_native_refusal: " + str(error)
    triangles = [tuple(map(int, row)) for row in raw]
    if (triangles != sorted(set(triangles))
            or any(len(row) != 3 or min(row) < 0 or max(row) >= len(points)
                   for row in triangles)):
        raise MeshError("reference insertion returned malformed connectivity")
    return triangles, None
