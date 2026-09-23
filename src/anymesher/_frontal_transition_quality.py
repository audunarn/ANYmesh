"""Bounded chart-quality repair before a cylindrical frontal candidate is guarded.

Only the opt-in spatial route calls this helper. It changes detached interior
coordinates, never protected coordinates, point order, or connectivity. Physical
owner qualification remains the caller's later, independent publication gate.
"""
from __future__ import annotations

import math
import numpy as np

from ._joint_triangle_repair import repair_joint_triangle_quality
from .errors import MeshError


def repair_frontal_transition(candidate, protected, settings, report,
                               cancellation_check=None):
    from .surface_mesh import _candidate_selection_key, _make_candidate

    used = report.get("topology_operations")
    if type(used) is not int or used < 0:
        raise MeshError("invalid frontal topology-work receipt")
    available = max(0, settings.native_options.max_topology_operations - used)
    limit = min(2048, available)
    receipt = {
        "trial_budget": limit, "trials": 0, "accepted": False,
        "candidate_moved_nodes": [], "budget_exhausted": limit == 0,
        "initial_quality": dict(candidate.report),
        "final_quality": dict(candidate.report),
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
    )
    fixed = sorted({int(node) for edge in protected for node in edge})
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
    accepted = (
        proposed.report["invalid_element_count"] == 0
        and not proposed.report["poor_element_ids"]
        and _candidate_selection_key(proposed, prefer_growth=False)
            < _candidate_selection_key(candidate, prefer_growth=False)
    )
    checkpoint()
    receipt.update(
        trials=repaired.trials, accepted=accepted,
        candidate_moved_nodes=list(repaired.moved_nodes) if accepted else [],
        budget_exhausted=repaired.budget_exhausted,
        final_quality=dict(proposed.report if accepted else candidate.report),
    )
    return (proposed if accepted else candidate), dict(
        report, topology_operations=used + repaired.trials,
        chart_transition_repair=receipt,
    )
