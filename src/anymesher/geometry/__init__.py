"""Geometry kernel: points, lines, faces and the operations on them."""

from ..errors import GeometryError
from .chains import chain_breaks, chain_point, sample_chain
from .curves import Arc, ArcFrame, CurveShape, DegenerateArcError, Straight, arc_frame
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
