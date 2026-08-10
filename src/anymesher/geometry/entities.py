"""Compatibility exports for topology entities now owned by ANYgeometry."""

from anygeometry.entities import (
    Edge,
    EntityKind,
    EntityRef,
    Face,
    OrientedEdge,
    Vertex,
)

__all__ = ["Edge", "EntityKind", "EntityRef", "Face", "OrientedEdge", "Vertex"]
