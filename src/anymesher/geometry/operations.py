"""Compatibility facade across neutral and mapped-only geometry operations.

New code should import general geometry operations from :mod:`anygeometry` and
mapped decomposition from :mod:`anymesher.decomposition`.  This module keeps
the historical ``anymesher.geometry.operations`` path working during the
ecosystem migration.
"""

from anygeometry.operations import surface_point

from ..decomposition import (
    MappabilityReport,
    check_mappable,
    punch_circular_hole,
    split_face_at,
    split_face_between,
    strip_face,
    triangle_to_quads,
)

__all__ = [
    "MappabilityReport",
    "check_mappable",
    "punch_circular_hole",
    "split_face_at",
    "split_face_between",
    "strip_face",
    "surface_point",
    "triangle_to_quads",
]
