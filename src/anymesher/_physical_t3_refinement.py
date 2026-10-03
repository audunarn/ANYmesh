"""Conforming physical-quality bisection using the remaining frontal budget."""
from dataclasses import replace
from fractions import Fraction
import numpy as np

from .errors import MeshError
from .native_v2 import MutableT3Topology, _GeometryLimited


def _alternative_progress(before, after):
    """Admit progress against the same committed physical-quality report."""
    previous_counts, proposed_counts = before['violation_counts'], after['violation_counts']
    increases = (after['invalid_element_count'] != 0
        or any(proposed_counts[name] > count for name, count in previous_counts.items())
        or after['quality_violation_count'] > before['quality_violation_count']
        or after['elements_above_maximum_growth'] > before['elements_above_maximum_growth'])
    progresses = (any(proposed_counts[name] < count for name, count in previous_counts.items())
        or after['max_element_growth'] < before['max_element_growth'])
    return not increases and progresses


def refine_physical_candidate(candidate, triangulation, settings, report,
                              evaluate_coordinates, original_segments,
                              seed_registry, cancellation_check=None):
    from .surface_mesh import _make_candidate, _physical_quality_candidate

    cancellation_failure = None
    if cancellation_check is not None:
        incoming_cancellation_check = cancellation_check

        def cancellation_check(phase):
            nonlocal cancellation_failure
            try:
                incoming_cancellation_check(phase)
            except _GeometryLimited as error:
                cancellation_failure = error
                raise

    for name in ('topology_operations', 'insertions', 'reserved_node_reuses'):
        value = report.get(name, 0)
        if type(value) is not int or value < 0:
            raise MeshError('invalid physical refinement work receipt')
    used = report['topology_operations']
    insertions = report['insertions']
    reused = report.get('reserved_node_reuses', 0)
    nodes = {}
    for ends, interval in (original_segments or {}).items():
        edge, lower, upper = interval if isinstance(interval, tuple) else (interval, 0, 1)
        values = nodes.setdefault(int(edge), {})
        for node, station in zip(ends, (lower, upper)):
            station = Fraction(station)
            if node in values and values[node] != station:
                raise MeshError('physical refinement has conflicting source stations')
            values[int(node)] = station
    shared = {}
    for record in report.get('shared_nodes', ()):
        node, edge = int(record['local_node_id']), int(record['edge_id'])
        station = Fraction(*record['station'])
        values = nodes.setdefault(edge, {})
        if node in values and values[node] != station:
            raise MeshError('physical refinement has conflicting shared stations')
        values[node] = station
        identity = (int(record['node_id']), edge, station)
        if node in shared and shared[node] != identity:
            raise MeshError('physical refinement has conflicting shared identities')
        shared[node] = identity
    intervals = {}
    protected = []
    for a, b in triangulation.segments:
        owners = [edge for edge, values in nodes.items() if int(a) in values and int(b) in values]
        if len(owners) > 1:
            raise MeshError('physical refinement segment has ambiguous ownership')
        if owners:
            edge = owners[0]
            intervals[(int(a), int(b))] = (edge, nodes[edge][int(a)], nodes[edge][int(b)])
        else:
            protected.append((int(a), int(b)))
    topology = MutableT3Topology(candidate.points, candidate.triangles, protected,
                                splittable_edges=intervals, seed_registry=seed_registry)
    topology._shared_node_ids = shared
    splits = 0
    boundary_splits = 0
    blocked = None
    attempts = alternative_attempts = alternative_splits = refused_attempts = 0
    selected_cell_exhausted = False
    candidate_cells_exhausted = False
    exhausted_cells = candidate_cells_visited = 0
    stop_reason = None
    def physical_state(points, cells):
        xyz = np.asarray(evaluate_coordinates(points), dtype=float)
        trial = _physical_quality_candidate(_make_candidate(points, cells, settings=settings),
                                            settings, lambda _: xyz)
        return trial, xyz

    # Reuse each detached state's validated values until its next atomic change.
    current, xyz = physical_state(*topology.canonical_export())
    staged = None
    require_progress = False
    owner_failure = None

    def validate_split(points, cells):
        nonlocal staged, owner_failure
        try:
            trial, coordinates = physical_state(points, cells)
        except _GeometryLimited as error:
            # An evaluator exception is not a local split refusal, even when a
            # callback happens to use the mesher's private exception type.
            owner_failure = error
            raise
        if trial.report['invalid_element_count']:
            raise _GeometryLimited('physical bisection would invalidate the mesh')
        if require_progress:
            if not _alternative_progress(current.report, trial.report):
                raise _GeometryLimited('alternative physical bisection makes no admissible progress')
        staged = trial, coordinates
    while (used < settings.native_options.max_topology_operations
           and insertions + reused < settings.native_options.max_insertions):
        if cancellation_check is not None:
            cancellation_check('native-v2 physical quality bisection')
        points, cells = current.points, current.triangles
        if current.report['invalid_element_count'] or not current.report['poor_element_ids']:
            stop_reason = 'invalid_candidate' if current.report['invalid_element_count'] else 'satisfied'
            break
        lengths = np.linalg.norm(np.roll(xyz[cells], -1, axis=1)-xyz[cells], axis=2).mean(axis=1)
        incidence = {}
        for row, cell in enumerate(cells):
            if cancellation_check is not None and row % settings.native_options.cancellation_interval == 0:
                cancellation_check('native-v2 physical growth incidence')
            for a, b in zip(cell, np.roll(cell, -1)):
                incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(row)
        growth = []
        for ordinal, owners in enumerate(incidence.values()):
            if cancellation_check is not None and ordinal % settings.native_options.cancellation_interval == 0:
                cancellation_check('native-v2 physical growth priority')
            if len(owners) == 2:
                large, small = sorted(owners, key=lambda row: (lengths[row], row), reverse=True)
                ratio = lengths[large]/lengths[small]
                if ratio > settings.max_element_growth:
                    growth.append((-ratio, large))
        rows = dict.fromkeys(row for _, row in sorted(growth))
        for ordinal, element in enumerate(current.report['poor_element_ids']):
            if cancellation_check is not None and ordinal % settings.native_options.cancellation_interval == 0:
                cancellation_check('native-v2 physical quality candidate priority')
            rows.setdefault(int(element)-1, None)
        # Refusals apply only to this unchanged, owner-scored baseline. A commit
        # starts a new pass, including a new edge cache and growth priority.
        refused_edges = set()
        committed = False
        for cell_ordinal, row in enumerate(rows):
            if cancellation_check is not None:
                cancellation_check('native-v2 physical quality candidate cell')
            candidate_cells_visited += 1
            cell = cells[row]
            edges = sorted((tuple(sorted((int(a), int(b)))) for a, b in zip(cell, np.roll(cell, -1))),
                key=lambda ends: (float(np.linalg.norm(xyz[ends[1]]-xyz[ends[0]])), ends), reverse=True)
            for ordinal, edge in enumerate(edges):
                if edge in refused_edges:
                    continue
                if used >= settings.native_options.max_topology_operations:
                    stop_reason = 'topology_budget'
                    break
                if insertions + reused >= settings.native_options.max_insertions:
                    stop_reason = 'insertion_budget'
                    break
                if cancellation_check is not None:
                    cancellation_check('native-v2 physical quality edge attempt')
                used += 1
                attempts += 1
                require_progress = cell_ordinal > 0 or ordinal > 0
                alternative_attempts += int(require_progress)
                if edge in topology.protected_edges:
                    refused_attempts += 1
                    refused_edges.add(edge)
                    blocked = edge
                    continue
                staged = None
                owner_failure = None
                cancellation_failure = None
                boundary = edge in topology.splittable_edges
                try:
                    if boundary:
                        topology.split_segment(edge, cancellation_check=cancellation_check,
                                               _validate_candidate=validate_split)
                    else:
                        topology.bisect_interior_edge(edge, cancellation_check=cancellation_check,
                                                     _validate_candidate=validate_split)
                except _GeometryLimited as error:
                    if owner_failure is error or cancellation_failure is error:
                        raise
                    refused_attempts += 1
                    refused_edges.add(edge)
                    blocked = edge
                    continue
                boundary_splits += int(boundary)
                alternative_splits += int(require_progress)
                insertions += 1
                splits += 1
                current, xyz = staged
                blocked = None
                committed = True
                break
            if committed or stop_reason is not None:
                break
            exhausted_cells += 1
            if cell_ordinal == 0:
                selected_cell_exhausted = True
        if not committed:
            if stop_reason is None:
                candidate_cells_exhausted = True
                stop_reason = 'no_progress_for_candidate_cells'
            break
    points, cells = current.points, current.triangles
    result = replace(candidate, points=points, triangles=cells, report=current.report,
                     score=current.score, aspect_ratios=current.aspect_ratios)
    updated = dict(report, topology_operations=used, insertions=insertions,
                   shared_segment_splits=int(report['shared_segment_splits'])+boundary_splits,
                   physical_quality_bisections=splits, physical_quality_blocked_edge=blocked,
                   physical_quality_attempts=attempts,
                   physical_quality_alternative_attempts=alternative_attempts,
                   physical_quality_alternative_bisections=alternative_splits,
                   physical_quality_refused_attempts=refused_attempts,
                   physical_quality_selected_cell_exhausted=selected_cell_exhausted,
                   physical_quality_candidate_cells_exhausted=candidate_cells_exhausted,
                   physical_quality_exhausted_cell_count=exhausted_cells,
                   physical_quality_candidate_cells_visited=candidate_cells_visited,
                   shared_nodes=[{'local_node_id': int(node), 'node_id': int(value[0]),
                                  'edge_id': int(value[1]),
                                  'station': (value[2].numerator, value[2].denominator)}
                                 for node, value in sorted(topology.shared_node_ids.items())])
    updated['selected_route'] = ('frontal_delaunay_physical_quality_satisfied'
        if not result.report['poor_element_ids'] and not result.report['invalid_element_count']
        else 'frontal_delaunay_geometry_limited' if candidate_cells_exhausted
        else 'frontal_delaunay_budget_limited')
    updated['physical_quality_stop_reason'] = (stop_reason or
        ('satisfied' if not result.report['poor_element_ids'] and not result.report['invalid_element_count']
         else 'topology_budget' if used >= settings.native_options.max_topology_operations
         else 'insertion_budget'))
    return result, replace(triangulation, points=points, triangles=cells,
                           segments=topology.constraint_edges), updated
