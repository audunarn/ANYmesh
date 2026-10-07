"""PRIVATE deferred-finalization carrier for one hybrid mesh candidate.

The carrier is internal to :mod:`anymesher.hybrid`; it is not part of the
public API and may change without notice.  It exists so candidate generation
(structured, native, refined, conservative) can be separated from
finalization (connectivity, source-association remapping, validation,
publication): a candidate keeps its own working geometry, view, pipeline,
ownership maps, and diagnostics until exactly one selected candidate is
finalized, so connectivity is applied once and only to the published mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping


def new_candidate_runtime() -> dict[str, Any]:
    """Return the execution-only runtime counters of one fresh candidate."""

    return {
        "candidate_count": 1,
        "connectivity_application_count": 0,
        "phase_seconds": {},
    }


@dataclass
class _HybridCandidateContext:
    """One complete, not-yet-finalized hybrid mesh candidate.

    ``mesh`` associations are keyed by the candidate's own working geometry.
    ``publication_guard`` is the candidate's own freshness check; it is invoked
    exactly once, by finalization.  ``phase_seconds`` accumulates this
    candidate's own phases, ``extra_diagnostics`` carries selection-stage
    diagnostics that finalization merges into the published record, and
    ``runtime`` carries additive execution-only counters covering the nested
    candidate chain.  Source/model binding data stays outside ``runtime``.
    """

    source_geometry: Any
    working_geometry: Any
    view: Any
    pipeline: Any
    publication_guard: Callable[[str], None]
    mesh: Any
    preflight: tuple
    preparation_report: Any
    structured_report: Any
    source_faces: tuple
    faces: tuple
    mapped_faces: tuple
    source_to_final_faces: Mapping
    source_to_final_edges: Mapping
    prepared_to_final_edges: Mapping
    source_strategies: Mapping
    triangulation_backend_by_face: dict
    boundary_registry: Any
    final_declared_junction_edges: Any
    final_cylindrical_repairs: Mapping
    target_size: float
    certification_mode: Any
    change_set: Any
    audit_policy: Any
    qualified_s3_record: dict | None
    defer_qualified_s3: bool
    reuse_prepared_working_copy: bool
    phase_seconds: dict = field(default_factory=dict)
    extra_diagnostics: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=new_candidate_runtime)

    def with_mesh(
        self,
        mesh: Any,
        *,
        qualified_s3_record: dict | None = None,
    ) -> "_HybridCandidateContext":
        """Return this context bound to a repaired or re-qualified mesh.

        A mesh replacement invalidates any prior qualified-S3 admission
        record: admission data must describe the final shell, so a
        coordinate or topology repair may never publish a stale record.
        Pass the fresh record explicitly when the replacement is itself
        the qualification result.  Mutable diagnostic containers are
        shared with the receiver, so selection-stage records written
        before or after the replacement are published once, by
        finalization.
        """

        return replace(
            self, mesh=mesh, qualified_s3_record=qualified_s3_record
        )
