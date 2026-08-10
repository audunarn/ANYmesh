"""Meshing errors plus the historical geometry-error import path."""

from __future__ import annotations

from anygeometry.errors import GeometryError

__all__ = ["GeometryError", "MeshError"]

class MeshError(ValueError):
    """Raised when a mesh cannot be generated from the given geometry."""
