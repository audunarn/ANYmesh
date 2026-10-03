"""Bounded chart-quality repair before a frontal candidate is guarded.

Only the opt-in spatial route calls this helper. It changes detached interior
coordinates, never protected coordinates, point order, or connectivity. Physical
owner qualification remains the caller's later, independent publication gate.
Explicit owner opt-in may adopt physical progress while quality remains unsatisfied.
"""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np

from ._joint_triangle_repair import repair_joint_triangle_quality, _validate_pinned_nodes
from .errors import MeshError


def repair_frontal_transition(candidate, protected, settings, report,
                               cancellation_check=None, *, evaluate_coordinates=None,
                               coordinate_batch_size=1, pinned_nodes=(), allow_partial_progress=False,
                               line_search_admission=False):
    from .surface_mesh import _candidate_selection_key, _make_candidate

    used = report.get("topology_operations")
    if type(used) is not int or used < 0:
        raise MeshError("invalid frontal topology-work receipt")
    if type(allow_partial_progress) is not bool:
        raise MeshError("invalid frontal partial-progress option")
    if (type(line_search_admission) is not bool or line_search_admission
            and (not allow_partial_progress or evaluate_coordinates is None)):
        raise MeshError("invalid frontal line-search admission option")
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
        "initial_penalty": None, "final_penalty": None, "proposed_penalty": None,
        "selected_nodes": [], "root_nodes": [], "neighbour_nodes": [],
        "priority_mode": 'physical_severity' if allow_partial_progress and evaluate_coordinates is not None else 'node_id',
        "candidate_observations": 0, "candidate_scorings": 0,
        "candidate_refusals": 0, "candidate_retentions": 0,
        "candidate_observation_seconds": 0., "candidate_scoring_seconds": 0.,
        "selected_trial": None, "selected_penalty": None,
        "admissible_quality": None, "admissible_penalty": None,
        "admissible_trial": None,
    }
    if line_search_admission:
        receipt.update(line_search_admission=True, admission_checks=0, admission_refusals=0,
            line_search_trials=0, line_search_refusals=0, line_search_admissions=0,
            line_search_advancements=0, refused_zero_trials=0)
    if (not candidate.report["poor_element_ids"] or not limit
            or not 0 < settings.min_angle < 60
            or not math.isfinite(settings.max_element_growth)
            or settings.max_element_growth <= 1):
        return candidate, dict(report, chart_transition_repair=receipt)

    def checkpoint():
        if cancellation_check is not None:
            cancellation_check("native-v2 chart transition repair")

    fixed = sorted({int(node) for edge in protected for node in edge} | pinned)
    admissible = None
    admissible_penalty = float('inf')
    admissible_trial = None
    observed_best_penalty = float('inf')
    observed_best_trial = None

    def observe(points, xyz, penalty, ordinal):
        nonlocal admissible, admissible_penalty, admissible_trial
        nonlocal observed_best_penalty, observed_best_trial
        started = perf_counter()
        receipt['candidate_observations'] += 1
        try:
            if penalty < observed_best_penalty:
                observed_best_penalty, observed_best_trial = penalty, ordinal
            if not line_search_admission and penalty >= admissible_penalty:
                return
            checkpoint()
            if points[fixed].tobytes() != candidate.points[fixed].tobytes():
                raise MeshError("invalid frontal chart-transition trial")
            moved = tuple(node for node in range(len(points))
                          if points[node].tobytes() != candidate.points[node].tobytes())
            from .surface_mesh import _physical_quality_candidate_from_xyz
            from ._physical_t3_refinement import _alternative_progress
            scoring_started = perf_counter()
            receipt['candidate_scorings'] += 1
            try:
                trial = _make_candidate(points.copy(), candidate.triangles, flips=candidate.flips,
                    moved_nodes=tuple(sorted(set(candidate.moved_nodes) | set(moved))),
                    added_points=candidate.added_points, rounds=candidate.rounds, settings=settings)
                trial = _physical_quality_candidate_from_xyz(trial, settings, xyz)
            finally:
                receipt['candidate_scoring_seconds'] += perf_counter() - scoring_started
            checkpoint()
            if not _alternative_progress(candidate.report, trial.report):
                receipt['candidate_refusals'] += 1
                return False
            if penalty < admissible_penalty:
                admissible, admissible_penalty, admissible_trial = trial, penalty, ordinal
                receipt['candidate_retentions'] += 1
            return True
        finally:
            receipt['candidate_observation_seconds'] += perf_counter() - started

    checkpoint()
    repaired = repair_joint_triangle_quality(
        candidate.points, candidate.triangles, protected,
        [int(value) - 1 for value in candidate.report["poor_element_ids"]],
        min_angle=settings.min_angle, max_growth=settings.max_element_growth,
        max_trials=limit, cancellation_check=checkpoint, neighbourhood_rings=1,
        evaluate_coordinates=evaluate_coordinates,
        coordinate_batch_size=coordinate_batch_size,
        pinned_nodes=pinned,
        physical_priority=allow_partial_progress and evaluate_coordinates is not None,
        candidate_callback=observe if allow_partial_progress and evaluate_coordinates is not None else None,
        line_search_admission=line_search_admission,
    )
    if line_search_admission:
        for name in ('admission_checks', 'admission_refusals', 'line_search_trials',
                     'line_search_refusals', 'line_search_admissions',
                     'line_search_advancements', 'refused_zero_trials'):
            receipt[name] = getattr(repaired, name)
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
    selected = proposed
    selected_trial = (observed_best_trial if observed_best_penalty < repaired.initial_penalty else None)
    selected_penalty = repaired.final_penalty
    adopted = accepted
    if not accepted and admissible is not None:
        selected, selected_trial, selected_penalty = admissible, admissible_trial, admissible_penalty
        adopted = True
        accepted = (selected.report['invalid_element_count'] == 0
            and not selected.report['poor_element_ids']
            and _candidate_selection_key(selected, prefer_growth=False)
                < _candidate_selection_key(candidate, prefer_growth=False))
    selected_moved = [node for node in range(len(selected.points))
                      if selected.points[node].tobytes() != candidate.points[node].tobytes()]
    checkpoint()
    receipt.update(
        trials=repaired.trials, accepted=accepted,
        candidate_adopted=adopted, quality_satisfied=accepted,
        candidate_moved_nodes=selected_moved if adopted else [],
        budget_exhausted=repaired.budget_exhausted,
        final_quality=dict(selected.report if adopted else candidate.report),
        proposed_quality=dict(proposed.report), proposed_moved_nodes=list(repaired.moved_nodes),
        initial_penalty=repaired.initial_penalty,
        final_penalty=((selected_penalty if adopted else repaired.initial_penalty)
                       if allow_partial_progress and evaluate_coordinates is not None
                       else repaired.final_penalty),
        proposed_penalty=repaired.final_penalty,
        selected_nodes=list(repaired.selected_nodes), root_nodes=list(repaired.root_nodes),
        neighbour_nodes=list(repaired.neighbour_nodes), priority_mode=repaired.priority_mode,
        selected_trial=selected_trial if adopted else None,
        selected_penalty=selected_penalty if adopted else None,
        admissible_quality=dict(admissible.report) if admissible is not None else None,
        admissible_penalty=admissible_penalty if admissible is not None else None,
        admissible_trial=admissible_trial,
    )
    return (selected if adopted else candidate), dict(
        report, topology_operations=used + repaired.trials,
        chart_transition_repair=receipt,
    )
