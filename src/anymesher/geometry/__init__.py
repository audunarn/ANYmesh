"""Temporary compatibility imports for geometry now owned by ANYgeometry.

The entity and model objects exported here are the owner objects themselves,
not adapted copies.  Existing ``anymesher.geometry`` imports therefore keep
working while new code can import :mod:`anygeometry` directly.
"""

from ..errors import GeometryError
from .chains import chain_breaks, chain_point, sample_chain
from .curves import Arc, ArcFrame, CurveShape, DegenerateArcError, Spline, Straight, arc_frame
from .entities import Edge, EntityKind, EntityRef, Face, OrientedEdge, Vertex
from .model import GeometryModel
from .operations import (
    MappabilityReport,
    check_mappable,
    punch_circular_hole,
    split_face_at,
    split_face_between,
    strip_face,
    surface_point,
    triangle_to_quads,
)

__all__ = [
    "Arc",
    "ArcFrame",
    "CurveShape",
    "DegenerateArcError",
    "Edge",
    "EntityKind",
    "EntityRef",
    "Face",
    "GeometryError",
    "GeometryModel",
    "MappabilityReport",
    "OrientedEdge",
    "Straight",
    "Spline",
    "Vertex",
    "arc_frame",
    "chain_breaks",
    "chain_point",
    "check_mappable",
    "sample_chain",
    "punch_circular_hole",
    "split_face_at",
    "split_face_between",
    "strip_face",
    "surface_point",
    "triangle_to_quads",
]
