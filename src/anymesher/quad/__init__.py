"""Quad-first meshing subpackage.

At Q0 this exposes only the frozen strict options contract.  The driver,
contracts, layout, counts, transitions, validation and optimization modules
land in later milestones (Q1-Q7) and are not importable at Q0.
"""

from .options import QUAD_MESHING_OPTIONS_SCHEMA, QuadMeshingOptions

__all__ = ["QUAD_MESHING_OPTIONS_SCHEMA", "QuadMeshingOptions"]
