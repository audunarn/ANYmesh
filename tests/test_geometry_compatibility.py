"""The extraction keeps old imports while sharing the exact owner objects."""

from __future__ import annotations

import anymesher
from anygeometry.chains import chain_breaks as owner_chain_breaks
from anygeometry.chains import chain_point as owner_chain_point
from anygeometry.chains import sample_chain as owner_sample_chain
from anygeometry import operations as owner_operations
from anygeometry.curves import (
    Arc as OwnerArc,
    ArcFrame as OwnerArcFrame,
    CurveShape as OwnerCurveShape,
    Straight as OwnerStraight,
)
from anygeometry.entities import (
    Edge as OwnerEdge,
    EntityRef as OwnerEntityRef,
    Face as OwnerFace,
    OrientedEdge as OwnerOrientedEdge,
    Vertex as OwnerVertex,
)
from anygeometry.errors import GeometryError as OwnerGeometryError
from anygeometry.model import GeometryModel as OwnerGeometryModel

from anymesher import decomposition
from anymesher.geometry import (
    Arc,
    ArcFrame,
    CurveShape,
    Edge,
    EntityRef,
    Face,
    GeometryError,
    GeometryModel,
    OrientedEdge,
    Straight,
    Vertex,
    chain_breaks,
    chain_point,
    sample_chain,
)
from anymesher.geometry import operations as compatibility_operations


def test_geometry_compatibility_exports_are_owner_types() -> None:
    assert GeometryModel is OwnerGeometryModel
    assert EntityRef is OwnerEntityRef
    assert Arc is OwnerArc
    assert ArcFrame is OwnerArcFrame
    assert CurveShape is OwnerCurveShape
    assert Straight is OwnerStraight
    assert Vertex is OwnerVertex
    assert Edge is OwnerEdge
    assert Face is OwnerFace
    assert OrientedEdge is OwnerOrientedEdge
    assert GeometryError is OwnerGeometryError
    assert chain_breaks is owner_chain_breaks
    assert chain_point is owner_chain_point
    assert sample_chain is owner_sample_chain
    assert anymesher.GeometryModel is OwnerGeometryModel
    assert anymesher.EntityRef is OwnerEntityRef


def test_general_operations_are_owner_functions() -> None:
    assert compatibility_operations.surface_point is owner_operations.surface_point


def test_mapped_decomposition_remains_owned_by_anymesher() -> None:
    assert compatibility_operations.check_mappable is decomposition.check_mappable
    assert compatibility_operations.split_face_at is decomposition.split_face_at
    assert compatibility_operations.split_face_between is decomposition.split_face_between
    assert compatibility_operations.strip_face is decomposition.strip_face
    assert compatibility_operations.triangle_to_quads is decomposition.triangle_to_quads
    assert compatibility_operations.punch_circular_hole is decomposition.punch_circular_hole
