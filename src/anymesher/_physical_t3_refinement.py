"""Conforming physical-quality bisection using the remaining frontal budget."""
from dataclasses import replace
from fractions import Fraction
import numpy as np

from .errors import MeshError
from .native_v2 import MutableT3Topology, _GeometryLimited


def refine_physical_candidate(candidate, triangulation, settings, report,
                              evaluate_coordinates, original_segments,
                              seed_registry, cancellation_check=None):
    from .surface_mesh import _make_candidate, _physical_quality_candidate

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
    def physical_state(points, cells):
        xyz = np.asarray(evaluate_coordinates(points), dtype=float)
        trial = _physical_quality_candidate(_make_candidate(points, cells, settings=settings),
                                            settings, lambda _: xyz)
        return trial, xyz

    # Reuse each detached state's validated values until its next atomic change.
    current, xyz = physical_state(*topology.canonical_export())
    staged = None

    def validate_split(points, cells):
        nonlocal staged
        trial, coordinates = physical_state(points, cells)
        if trial.report['invalid_element_count']:
            raise _GeometryLimited('physical bisection would invalidate the mesh')
        staged = trial, coordinates
    while (used < settings.native_options.max_topology_operations
           and insertions + reused < settings.native_options.max_insertions):
        if cancellation_check is not None:
            cancellation_check('native-v2 physical quality bisection')
        points, cells = current.points, current.triangles
        if current.report['invalid_element_count'] or not current.report['poor_element_ids']:
            break
        lengths = np.linalg.norm(np.roll(xyz[cells], -1, axis=1)-xyz[cells], axis=2).mean(axis=1)
        incidence = {}
        for row, cell in enumerate(cells):
            if cancellation_check is not None and row % settings.native_options.cancellation_interval == 0:
                cancellation_check('native-v2 physical growth incidence')
            for a, b in zip(cell, np.roll(cell, -1)):
                incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(row)
        growth = []
        for owners in incidence.values():
            if len(owners) == 2:
                large, small = sorted(owners, key=lambda row: (lengths[row], row), reverse=True)
                ratio = lengths[large]/lengths[small]
                if ratio > settings.max_element_growth:
                    growth.append((-ratio, large))
        row = min(growth)[1] if growth else int(current.report['poor_element_ids'][0])-1
        cell = cells[row]
        edge = max((tuple(sorted((int(a), int(b)))) for a, b in zip(cell, np.roll(cell, -1))),
                   key=lambda ends: (float(np.linalg.norm(xyz[ends[1]]-xyz[ends[0]])), ends))
        used += 1
        if edge in topology.protected_edges:
            blocked = edge
            break
        if edge in topology.splittable_edges:
            try:
                topology.split_segment(edge, cancellation_check=cancellation_check,
                                       _validate_candidate=validate_split)
            except _GeometryLimited:
                blocked = edge
                break
            boundary_splits += 1
        else:
            try:
                topology.bisect_interior_edge(edge, cancellation_check=cancellation_check,
                                             _validate_candidate=validate_split)
            except _GeometryLimited:
                blocked = edge
                break
        insertions += 1
        splits += 1
        current, xyz = staged
    points, cells = current.points, current.triangles
    result = replace(candidate, points=points, triangles=cells, report=current.report,
                     score=current.score, aspect_ratios=current.aspect_ratios)
    updated = dict(report, topology_operations=used, insertions=insertions,
                   shared_segment_splits=int(report['shared_segment_splits'])+boundary_splits,
                   physical_quality_bisections=splits, physical_quality_blocked_edge=blocked,
                   shared_nodes=[{'local_node_id': int(node), 'node_id': int(value[0]),
                                  'edge_id': int(value[1]),
                                  'station': (value[2].numerator, value[2].denominator)}
                                 for node, value in sorted(topology.shared_node_ids.items())])
    updated['selected_route'] = ('frontal_delaunay_physical_quality_satisfied'
        if not result.report['poor_element_ids'] and not result.report['invalid_element_count']
        else 'frontal_delaunay_geometry_limited' if blocked is not None
        else 'frontal_delaunay_budget_limited')
    return result, replace(triangulation, points=points, triangles=cells,
                           segments=topology.constraint_edges), updated
