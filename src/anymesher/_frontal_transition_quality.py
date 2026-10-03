"""Bounded chart-quality repair before a frontal candidate is guarded.

Only the opt-in spatial route calls this helper. It changes detached interior
coordinates, never protected coordinates, point order, or connectivity. Physical
owner qualification remains the caller's later, independent publication gate.
Explicit owner opt-in may adopt physical progress while quality remains unsatisfied.
"""
from __future__ import annotations

import math
import numpy as np

from ._joint_triangle_repair import repair_joint_triangle_quality, _validate_pinned_nodes
from .errors import MeshError


def repair_frontal_transition(candidate, protected, settings, report,
                               cancellation_check=None, *, evaluate_coordinates=None,
                               coordinate_batch_size=1, pinned_nodes=(), allow_partial_progress=False):
    from .surface_mesh import _candidate_selection_key, _make_candidate

    used = report.get("topology_operations")
    if type(used) is not int or used < 0:
        raise MeshError("invalid frontal topology-work receipt")
    if type(allow_partial_progress) is not bool:
        raise MeshError("invalid frontal partial-progress option")
    pinned = _validate_pinned_nodes(pinned_nodes, len(candidate.points))
    if allow_partial_progress and evaluate_coordinates is not None and 'violation_counts' not in candidate.report:
        from .surface_mesh import _physical_quality_candidate
        candidate = _physical_quality_candidate(candidate, settings, evaluate_coordinates)
    available = max(0, settings.native_options.max_topology_operations - used)
    limit = min(2048, available)
    receipt = {
        "trial_budget": limit, "trials": 0, "accepted": False,
        "candidate_adopted": False,
        "quality_satisfied": candidate.report["invalid_element_count"] == 0 and not candidate.report["poor_element_ids"],
        "candidate_moved_nodes": [], "budget_exhausted": limit == 0,
        "initial_quality": dict(candidate.report),
        "final_quality": dict(candidate.report),
        "proposed_quality": None, "proposed_moved_nodes": [],
        "initial_penalty": None, "final_penalty": None,
    }
    if (not candidate.report["poor_element_ids"] or not limit
            or not 0 < settings.min_angle < 60
            or not math.isfinite(settings.max_element_growth)
            or settings.max_element_growth <= 1):
        return candidate, dict(report, chart_transition_repair=receipt)

    def checkpoint():
        if cancellation_check is not None:
            cancellation_check("native-v2 chart transition repair")

    checkpoint()
    repaired = repair_joint_triangle_quality(
        candidate.points, candidate.triangles, protected,
        [int(value) - 1 for value in candidate.report["poor_element_ids"]],
        min_angle=settings.min_angle, max_growth=settings.max_element_growth,
        max_trials=limit, cancellation_check=checkpoint, neighbourhood_rings=1,
        evaluate_coordinates=evaluate_coordinates,
        coordinate_batch_size=coordinate_batch_size,
        pinned_nodes=pinned,
    )
    fixed = sorted({int(node) for edge in protected for node in edge} | pinned)
    if (repaired.points.shape != candidate.points.shape
            or not np.isfinite(repaired.points).all()
            or repaired.points[fixed].tobytes() != candidate.points[fixed].tobytes()
            or type(repaired.trials) is not int or not 0 <= repaired.trials <= limit):
        raise MeshError("invalid frontal chart-transition repair result")
    proposed = _make_candidate(
        repaired.points, candidate.triangles,
        flips=candidate.flips,
        moved_nodes=tuple(sorted(set(candidate.moved_nodes) | set(repaired.moved_nodes))),
        added_points=candidate.added_points, rounds=candidate.rounds,
        settings=settings,
    )
    if evaluate_coordinates is not None:
        from .surface_mesh import _physical_quality_candidate
        proposed = _physical_quality_candidate(proposed, settings, evaluate_coordinates)
    accepted = (
        proposed.report["invalid_element_count"] == 0
        and not proposed.report["poor_element_ids"]
        and _candidate_selection_key(proposed, prefer_growth=False)
            < _candidate_selection_key(candidate, prefer_growth=False)
    )
    adopted = accepted
    if allow_partial_progress and evaluate_coordinates is not None and proposed.report['poor_element_ids']:
        from ._physical_t3_refinement import _alternative_progress
        adopted = _alternative_progress(candidate.report, proposed.report)
    checkpoint()
    receipt.update(
        trials=repaired.trials, accepted=accepted,
        candidate_adopted=adopted, quality_satisfied=accepted,
        candidate_moved_nodes=list(repaired.moved_nodes) if adopted else [],
        budget_exhausted=repaired.budget_exhausted,
        final_quality=dict(proposed.report if adopted else candidate.report),
        proposed_quality=dict(proposed.report), proposed_moved_nodes=list(repaired.moved_nodes),
        initial_penalty=repaired.initial_penalty, final_penalty=repaired.final_penalty,
    )
    return (proposed if adopted else candidate), dict(
        report, topology_operations=used + repaired.trials,
        chart_transition_repair=receipt,
    )
