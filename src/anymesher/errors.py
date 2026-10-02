"""Meshing errors plus the historical geometry-error import path."""

from __future__ import annotations

from anygeometry.errors import GeometryError

__all__ = ["GeometryError", "MeshError", "StructuredQualityRejected", "NativeSurfaceUnsupported"]

class MeshError(ValueError):
    """Raised when a mesh cannot be generated from the given geometry."""


class StructuredQualityRejected(MeshError):
    """A completed structured candidate failed its unchanged quality gate."""


class NativeSurfaceUnsupported(MeshError):
    """The requested native placement has no qualified chart for this support."""
