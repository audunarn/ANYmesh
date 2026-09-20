"""Quad-first meshing subpackage.

Q0 exposes the frozen strict options contract.  Q1 adds the resident
mixed T3/Q4 state, the transactional journal, and the single-step
advancing-front driver that converts one front T3 pair into a Q4.
"""

from .options import QUAD_MESHING_OPTIONS_SCHEMA, QuadMeshingOptions
from .front import (
    FrontNoCandidate,
    FrontRejected,
    area2,
    body_edges,
    candidate_partners,
    classify,
    edge_key,
    find_source_cell,
    front_step,
    local_swap,
    make_quad,
)

__all__ = [
    "QUAD_MESHING_OPTIONS_SCHEMA",
    "QuadMeshingOptions",
    "FrontNoCandidate",
    "FrontRejected",
    "area2",
    "body_edges",
    "candidate_partners",
    "classify",
    "edge_key",
    "find_source_cell",
    "front_step",
    "local_swap",
    "make_quad",
]
