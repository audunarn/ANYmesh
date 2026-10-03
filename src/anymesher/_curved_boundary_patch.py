"""Plane-only detached staging from an owner-bound curved boundary receipt.

This grants neither split permission nor publication qualification. Curve-error
qualification and a transaction over all participating faces remain separate.
The established collinear topology API is deliberately not used or changed.
Coordinates are the bound owner's Plane support UV, not AnalyticMetricChart's
metric rows. This cannot be routed directly from a transformed analytic chart.
Whole-child coverage currently uses prototype exact owner-module operations;
publication requires the separate public owner coverage contract as well.
"""
from dataclasses import dataclass
from copy import deepcopy
from fractions import Fraction
from numbers import Integral

import numpy as np

from .errors import MeshError
from .native_v2 import MutableT3Topology, _GeometryLimited, _canonical_triangle, orient2d
from ._material_region_binding import BoundaryIntervalReceipt, MaterialRegionBinding


@dataclass(frozen=True)
class CurvedBoundaryPatch:
    topology: MutableT3Topology
    candidate: object
    baseline: object
    source_interval: BoundaryIntervalReceipt
    parameter: Fraction
    local_node: int
    xyz: np.ndarray
    child_intervals: tuple
    physical_progress: bool
    publication_qualified: bool = False


def _no_patch_overlap(points, children, retained, checkpoint):
    """Check the changed cells against each other and unchanged mesh cells."""
    def overlap(first, second):
        for a, b in zip(first, (*first[1:], first[0])):
            for c, d in zip(second, (*second[1:], second[0])):
                if {a, b} == {c, d}:
                    continue  # The same registered adjacency, not a weld.
                signs = (orient2d(points[a], points[b], points[c]),
                         orient2d(points[a], points[b], points[d]),
                         orient2d(points[c], points[d], points[a]),
                         orient2d(points[c], points[d], points[b]))
                opposite = lambda x, y: (x > 0. and y < 0.) or (x < 0. and y > 0.)
                if opposite(*signs[:2]) and opposite(*signs[2:]):
                    return True
                for node, edge, sign in ((c, (a, b), signs[0]), (d, (a, b), signs[1]),
                                         (a, (c, d), signs[2]), (b, (c, d), signs[3])):
                    if node not in edge and sign == 0. and (
                            np.all(points[node] >= np.minimum(points[edge[0]], points[edge[1]]))
                            and np.all(points[node] <= np.maximum(points[edge[0]], points[edge[1]]))):
                        return True  # Collinear overlap or an unregistered T-junction.
        for vertex, cell in ((node, second) for node in first):
            if all(orient2d(points[a], points[b], points[vertex]) > 0.
                   for a, b in zip(cell, (*cell[1:], cell[0]))):
                return True
        for vertex, cell in ((node, first) for node in second):
            if all(orient2d(points[a], points[b], points[vertex]) > 0.
                   for a, b in zip(cell, (*cell[1:], cell[0]))):
                return True
        return False

    comparisons = [(children[0], children[1])]
    comparisons.extend((child, tuple(map(int, cell))) for child in children for cell in retained)
    for first, second in comparisons:
        checkpoint('curved boundary unchanged-cell overlap')
        if overlap(tuple(first), tuple(second)):
            raise _GeometryLimited('curved boundary patch overlaps unchanged material cells')


def _plane_child_coverage(domain, points, child, tolerance, checkpoint):
    """Delegate every trim root and interval membership to exact owner math.

    For a Plane, linear chart sides lift to exact world lines. Owner-junction
    roots partition each side into constant-membership intervals. A hole wholly
    inside the triangle is excluded separately; no centroid/sample coverage
    assertion, sampled curve, or copied geometry classifier is involved.
    """
    from anygeometry.arrangement_geometry import LinePath, line_curve_junctions

    def root_check():
        checkpoint('curved boundary owner domain roots')
        return False

    rows = points[np.asarray(child)]
    world = np.asarray([domain.support.evaluate(*uv) for uv in rows])
    for point in world:
        if not domain.contains(LinePath(tuple(point), tuple(point)), 0., tolerance):
            raise _GeometryLimited('curved boundary child vertex is outside owner material')
    for first, last in zip(world, np.roll(world, -1, axis=0)):
        checkpoint('curved boundary owner side coverage')
        side = LinePath(tuple(first), tuple(last))
        parameters = {0., 1.}
        for loop in domain.boundaries:
            for path in loop:
                junctions = line_curve_junctions(side, path.curve,
                    tolerance=tolerance, cancellation_check=root_check)
                parameters.update(float(t) for t, _ in junctions)
        ordered = sorted(parameters)
        if any(not np.isfinite(t) or not 0. <= t <= 1. for t in ordered):
            raise MeshError('owner returned an invalid child-side junction')
        for lower, upper in zip(ordered, ordered[1:]):
            checkpoint('curved boundary owner side interval')
            if not domain.contains(side, .5*(lower+upper), tolerance):
                raise _GeometryLimited('curved boundary child side is outside owner material')
    # A hole can miss every triangle side while still occupying its interior.
    # Without side crossings it is wholly in or out. Conservatively refuse
    # a hole endpoint on the triangle boundary as well as in its interior.
    for hole in domain.boundaries[1:]:
        checkpoint('curved boundary owner enclosed hole')
        for path in hole:
            point = domain.uv(path.curve, 0.)
            if all(orient2d(a, b, point) >= 0. for a, b in zip(rows, np.roll(rows, -1, axis=0))):
                raise _GeometryLimited('curved boundary child encloses owner void')


def stage_curved_boundary_patch(topology, receipt, mesh, registry, parameter, *,
                                node_to_row, settings, work,
                                station_xyz=None, cancellation_check=None):
    """Stage one boundary occurrence; consume original work even on refusal.

    ``node_to_row`` carries known global identity correspondence, not a coordinate
    search. The exact source parameter and immutable bound receipt determine the
    new point. ``station_xyz`` is an optional assertion, never a placement rule.
    Input topology and receipt UV must share the owner's support coordinates;
    metric-chart transforms are not inferred or silently inverted here.
    Successful staging returns full physical reports and the unchanged progress
    decision; it still cannot authorize publication or a curved-error waiver.
    """
    from anygeometry import evaluate_material_surface_region
    from anygeometry.surfaces import Plane
    from .surface_mesh import SurfaceMeshOptions, _make_candidate, _physical_quality_candidate_from_xyz
    from ._physical_t3_refinement import _alternative_progress

    if not isinstance(settings, SurfaceMeshOptions) or not isinstance(work, dict):
        raise MeshError('curved boundary staging requires original settings and work receipt')
    for name in ('topology_operations', 'insertions', 'reserved_node_reuses'):
        if name not in work:
            raise MeshError('curved boundary staging requires every original work receipt field')
        value = work[name]
        if type(value) is not int or value < 0:
            raise MeshError('invalid curved boundary work receipt')
    if work['topology_operations'] >= settings.native_options.max_topology_operations:
        raise _GeometryLimited('curved boundary topology allowance exhausted')
    if work['insertions']+work['reserved_node_reuses'] >= settings.native_options.max_insertions:
        raise _GeometryLimited('curved boundary insertion allowance exhausted')
    # This is a charged staging attempt, not a committed-station count. No work
    # refund when a later owner, validity, cancellation or coverage check fails.
    work['topology_operations'] += 1
    work['insertions'] += 1
    asserted_xyz = None if station_xyz is None else np.array(station_xyz, dtype=float, copy=True)
    original = topology
    topology = MutableT3Topology(original._points, original._triangles, tuple(original.protected_edges),
        node_owners=original.node_owners, triangle_owners=original.triangle_owners,
        splittable_edges=original._splittable_intervals, seed_registry=original._seed_registry)
    topology._shared_node_ids = dict(original.shared_node_ids)
    topology.epoch = original.epoch
    topology.free_triangle_ids = list(original.free_triangle_ids)
    topology.quality_cache = deepcopy(original.quality_cache)
    node_to_row = dict(node_to_row) if isinstance(node_to_row, dict) else node_to_row

    def checkpoint(phase):
        if cancellation_check is not None:
            cancellation_check(phase)

    checkpoint('curved boundary staging start')
    if not isinstance(receipt, BoundaryIntervalReceipt) or type(receipt.binding) is not MaterialRegionBinding:
        raise MeshError('curved boundary staging requires an authoritative source receipt')
    binding = receipt.binding
    binding.validate(cancellation_check)
    if not any(region is binding.region for region in binding.collection.regions):
        raise MeshError('curved boundary domain is not selected by its owner binding')
    if type(binding.region.support) is not Plane:
        raise _GeometryLimited('curved boundary child coverage requires a qualified Plane support')
    if not isinstance(parameter, Fraction):
        raise MeshError('curved boundary station requires an exact source parameter')
    uv, new_xyz = binding.boundary_station(receipt, mesh, registry, parameter, cancellation_check)
    if asserted_xyz is not None and (asserted_xyz.shape != (3,) or
            not np.array_equal(asserted_xyz, new_xyz)):
        raise MeshError('curved boundary proposed point is not its exact owner station')
    if not isinstance(node_to_row, dict) or any(
            isinstance(node, (bool, np.bool_)) or not isinstance(node, Integral) or node < 1
            or isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral)
            or not 0 <= row < len(topology._points) for node, row in node_to_row.items()):
        raise MeshError('curved boundary requires known global node correspondence')
    if len(set(node_to_row.values())) != len(node_to_row):
        raise MeshError('curved boundary has ambiguous global node correspondence')
    try:
        local = tuple(int(node_to_row[node]) for node in receipt.nodes)
    except KeyError:
        raise MeshError('curved boundary source endpoints are missing from topology') from None
    points, cells = topology.points.copy(), topology.triangles.copy()
    if not np.array_equal(points[list(local)], receipt.uv):
        raise MeshError('curved boundary source endpoints use another chart receipt')
    edge = tuple(sorted(local))
    if edge not in topology.protected_edges or edge in topology.splittable_edges:
        raise MeshError('curved boundary staging requires its protected boundary occurrence')
    attached = topology._topology_index.attached(edge)
    if len(attached) != 1:
        raise MeshError('curved boundary staging requires one incident parent cell')
    parent_row = attached[0]
    parent = cells[parent_row]
    if orient2d(*points[parent]) <= 0.:
        raise _GeometryLimited('curved boundary parent is invalid or inverted')
    start = next(i for i in range(3) if tuple(sorted((int(parent[i]), int(parent[(i+1) % 3])))) == edge)
    a, b, opposite = (int(parent[(start+i) % 3]) for i in range(3))
    node = len(points)
    extended = np.vstack((points, uv))
    raw_children = ((a, node, opposite), (node, b, opposite))
    if any(orient2d(*extended[list(child)]) <= 0. for child in raw_children):
        raise _GeometryLimited('curved boundary children invalidate parent orientation')
    for child in raw_children:
        _plane_child_coverage(binding.region.domain, extended, child,
                              binding.region.world_tolerance, checkpoint)
    if any(np.array_equal(uv, row) for row in points):
        raise _GeometryLimited('curved boundary station coincides with another node identity')
    _no_patch_overlap(extended, raw_children, np.delete(cells, parent_row, axis=0), checkpoint)
    children = [_canonical_triangle(child, extended) for child in raw_children]
    retained = [(tuple(map(int, cell)), int(topology.triangle_owners[row]))
                for row, cell in enumerate(cells) if row != parent_row]
    retained.extend((child, int(topology.triangle_owners[parent_row])) for child in children)
    retained.sort(key=lambda item: item[0])
    triangles = np.asarray([cell for cell, _ in retained], dtype=np.int64)
    protected = set(topology.protected_edges)-{edge}
    protected.update((tuple(sorted((a, node))), tuple(sorted((node, b)))))
    staged = MutableT3Topology(extended, triangles, tuple(protected),
        node_owners=np.append(topology.node_owners, -1),
        triangle_owners=[owner for _, owner in retained],
        splittable_edges=topology._splittable_intervals, seed_registry=topology._seed_registry)
    staged._shared_node_ids = dict(topology.shared_node_ids)
    staged.epoch = topology.epoch+1
    staged.free_triangle_ids = list(topology.free_triangle_ids)
    xyz = evaluate_material_surface_region(binding.geometry, binding.collection,
        binding.representative, points, require_material=True, cancellation_check=cancellation_check)
    # Restore known registered rows, preserving the owner-authored input bytes.
    for global_node, row in node_to_row.items():
        value = np.asarray(mesh.nodes[global_node], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or not np.allclose(
                xyz[row], value, rtol=0., atol=binding.region.world_tolerance):
            raise MeshError('curved boundary original row lost its owner correspondence')
        xyz[row] = value
    xyz = np.vstack((xyz, new_xyz))
    baseline = _physical_quality_candidate_from_xyz(_make_candidate(points, cells, settings=settings), settings, xyz[:-1])
    candidate = _physical_quality_candidate_from_xyz(_make_candidate(extended, triangles, settings=settings), settings, xyz)
    if candidate.report['invalid_element_count']:
        raise _GeometryLimited('curved boundary candidate has invalid physical elements')
    checkpoint('curved boundary staging final binding')
    binding.validate(cancellation_check)
    # Geometry revision alone cannot detect registry or edge-chain changes.
    # Reconstruct the same source occurrence after every staging checkpoint.
    final_uv, final_xyz = binding.boundary_station(receipt, mesh, registry,
                                                 parameter, cancellation_check)
    if not np.array_equal(final_uv, uv) or not np.array_equal(final_xyz, new_xyz):
        raise MeshError('curved boundary station changed during detached staging')
    xyz.setflags(write=False)
    intervals = (((local[0], node), (receipt.source_edge, receipt.parameters[0], parameter)),
                 ((node, local[1]), (receipt.source_edge, parameter, receipt.parameters[1])))
    return CurvedBoundaryPatch(staged, candidate, baseline, receipt, parameter, node,
        xyz, intervals, _alternative_progress(baseline.report, candidate.report))
