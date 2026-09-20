"""Quad-first meshing subpackage.

Q0 exposes the frozen strict options contract.  Q1 adds the resident
mixed T3/Q4 state, the transactional journal, and the single-step
advancing-front driver that converts one front T3 pair into a Q4.
Q2 adds bounded deterministic front-edge Steiner-split recovery that
enables a Q4 where pure Q1 pairing yields none.  Q3a adds an opt-in
fourfold cross-field that only *ranks* the already-admissible front
candidates (unguided :func:`front_step` stays byte-compatible).
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
from .recovery import (
    DEFAULT_RECOVERY_RATIOS,
    AdvanceReport,
    Attempt,
    RecoveryExhausted,
    RecoveryRejected,
    SplitReport,
    edge_split_recover,
    recover_then_front_step,
)
from .guidance import (
    CrossFieldReport,
    GuidanceRejected,
    LowConfidenceError,
    NEAR_ZERO,
    build_cross_field,
    field_from_directions,
    four_of,
    front_step_guided,
    rank_bodies,
    reflect_x,
    rot2,
    score_body,
)

from .transitions import (
    ParentReplacement,
    TransitionRejected,
    TransitionReport,
    closure,
    collision,
    spacing_change,
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
    "DEFAULT_RECOVERY_RATIOS",
    "AdvanceReport",
    "Attempt",
    "RecoveryExhausted",
    "RecoveryRejected",
    "SplitReport",
    "edge_split_recover",
    "recover_then_front_step",
    "CrossFieldReport",
    "GuidanceRejected",
    "LowConfidenceError",
    "NEAR_ZERO",
    "build_cross_field",
    "field_from_directions",
    "four_of",
    "front_step_guided",
    "rank_bodies",
    "reflect_x",
    "rot2",
    "score_body",
    "TransitionRejected",
    "ParentReplacement",
    "TransitionReport",
    "spacing_change",
    "collision",
    "closure",
]
