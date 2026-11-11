"""Fixed-interface subdivision of a small disk in a source-backed surface mesh."""
from collections import defaultdict
from fractions import Fraction
from math import log

import numpy as np

from .errors import MeshError
from .refinement import Refinement, SizeField
from ._stage_cancellation import check_stage, preserve_stage_cancellation, stage_scope


def _sides(cell):
    return list(zip(cell, (*cell[1:], cell[0])))


def _key(a, b):
    return (min(a, b), max(a, b))


def _positive(points, point=None):
    vectors = np.roll(points, -1, axis=0) - points
    other = np.roll(vectors, -1, axis=0) if point is None else point - points
    return bool(np.all(vectors[:, 0] * other[:, 1] - vectors[:, 1] * other[:, 0] > 0))


def _cut(data, hit, uv):
    cells = data['cells']
    adjacency = defaultdict(list)
    for index, cell in enumerate(cells):
        for a, b in _sides(cell):
            adjacency[_key(a, b)].append((index, a, b))
    centre = cells[hit]
    neighbors = []
    for a, b in _sides(centre):
        owners = adjacency[_key(a, b)]
        if len(owners) != 2:
            return None
        opposite = [k for k, p, q in owners if (p, q) == (b, a)]
        if len(opposite) != 1:
            return None
        neighbors.append(opposite[0])
    def adjacent(k):
        return {index for a,b in _sides(cells[k]) for index,_,_ in adjacency[_key(a,b)] if index != k}
    diagonals = []
    for i in (0, 2):
        common = (adjacent(neighbors[i]) & adjacent(neighbors[i+1])) - {hit}
        if len(common) != 1:
            return None
        diagonals.append(common.pop())
    selected = [hit, *neighbors, *diagonals]
    if len(set(selected)) != 7 or any(len(cells[k]) != 4 or not _positive(uv[cells[k]]) for k in selected):
        return None
    rim = {n for _, stations in data['rim'] for _, n in stations}
    if any(rim.intersection(cells[k]) for k in selected):
        return None
    # This seven-quad disk has twelve oriented external sides, each incident
    # to one unmodified cell. Return them as a single simple loop.
    boundary = {}
    for k in selected:
        for a, b in _sides(cells[k]):
            rows = adjacency[_key(a, b)]
            if len(rows) != 2 or (rows[0][1], rows[0][2]) != (rows[1][2], rows[1][1]):
                return None
            if any(index not in selected for index, _, _ in rows):
                if a in boundary:
                    return None
                boundary[a] = b
    if len(boundary) != 12:
        return None
    loop = [min(boundary)]
    for _ in range(11):
        nxt = boundary.get(loop[-1])
        if nxt is None or nxt in loop:
            return None
        loop.append(nxt)
    return (selected, loop) if boundary.get(loop[-1]) == loop[0] else None


def _subdivide(data, selected, support):
    old_cells = data['cells']
    rows = list(data['node_rows'])
    def mean_row(indices):
        uv = np.mean([rows[i][0] for i in indices], axis=0)
        rows.append([uv.tolist(), np.asarray(support.evaluate(*uv)).tolist()])
        return len(rows) - 1
    centre = old_cells[selected[0]]
    middle = mean_row(centre)
    edge_midpoints = {}
    incidences = defaultdict(list)
    for k in selected:
        for a,b in _sides(old_cells[k]):
            incidences[_key(a,b)].append(k)
    for edge, owners in incidences.items():
        if len(owners) == 2:
            edge_midpoints[edge] = mean_row(edge)
    midpoints = [edge_midpoints[_key(a,b)] for a,b in _sides(centre)]
    children = [[v, midpoints[i], middle, midpoints[i-1]] for i, v in enumerate(centre)]
    new_cells = list(children)
    for neighbor in selected[1:]:
        cell = old_cells[neighbor]
        mids = [edge_midpoints.get(_key(a,b)) for a,b in _sides(cell)]
        starts = [i for i in range(4) if mids[i] is not None and mids[(i+1)%4] is not None]
        if len(starts) != 1 or sum(m is not None for m in mids) != 2:
            raise MeshError('source patch requires two adjacent splits on each transition cell')
        i = starts[0]
        a,b,c,d = [cell[(i+j)%4] for j in range(4)]
        ab,bc = mids[i], mids[(i+1)%4]
        inner = mean_row(cell)
        new_cells.extend([[ab,b,bc,inner], [bc,c,d,inner], [d,a,ab,inner]])
    kept = {index: cell for index, cell in enumerate(old_cells) if index not in selected}
    return dict(data, node_rows=rows, cells=[*kept.values(), *new_cells], grid=None), kept, children


def _support(source, face):
    from anygeometry.nurbs import NurbsSurface
    from ._material_region_binding import polynomial_tensor_chart
    surface = source.faces[face].surface
    if type(surface) is NurbsSurface:
        return surface
    chart = polynomial_tensor_chart(source, face)
    return chart.support if chart is not None else None


def _outside_support(support, uv, cells, center, radius, cancellation_check):
    net = support.exact_net()
    point = tuple(Fraction(float(x)) for x in center)
    distance = Fraction(float(radius))**2
    bounds = np.array([(uv[cell].min(axis=0), uv[cell].max(axis=0)) for cell in cells])
    centers = bounds.mean(axis=1)
    pending = [np.arange(len(bounds))]
    while pending:
        check_stage(cancellation_check, 'source patch exterior sizing bounds')
        indices = pending.pop()
        lower_uv = bounds[indices, 0].min(axis=0)
        upper_uv = bounds[indices, 1].max(axis=0)
        box = tuple((float(lo), float(hi)) for lo,hi in zip(lower_uv, upper_uv))
        lower, upper, _ = support.patch_bounds(box, net)
        if sum(max(lo-p, p-hi, 0)**2 for lo, hi, p in zip(lower, upper, point)) > distance:
            continue
        if len(indices) == 1:
            return False
        axis = int(np.argmax(upper_uv-lower_uv))
        middle = len(indices)//2
        order = indices[np.argpartition(centers[indices,axis], middle)]
        pending.extend((order[:middle], order[middle:]))
    return True


def _achieves_zone(field, zone, points, tolerance):
    centroids = points.mean(axis=1)
    longest = np.linalg.norm(points - np.roll(points, 1, axis=1), axis=2).max(axis=1)
    in_core = np.linalg.norm(centroids - zone.center, axis=1) <= zone.radius + tolerance
    sizes = field.size_at(centroids)
    influenced = sizes < field.target_size
    return (np.any(in_core) and np.all(longest[in_core] <= zone.size + tolerance)
            and np.all(longest[influenced] <= sizes[influenced] + tolerance))


@stage_scope
def plan_source_face_point_patch(source, record, zone, *, target_size, strategy='auto', recombine=True,
                                 cancellation_check=None):
    """Plan a seven-Q4 disk for one added point refinement on an unchanged face.

    The original source proposal is obtained via capture_reusable_source_faces.
    None means the operator cannot prove its domain, achieved size or fixed
    interface. Invalid proposal data raise MeshError. A successful plan is
    codec-safe and must be reissued with the current source and actual field.
    """
    from anygeometry.surfaces import closest_uv
    from .regional_source_faces import (_digest, _face_signature, _native_definitions,
                                        _ordinary_face, _vertex_faces)
    if (type(zone) is not Refinement or zone.mode != 'refine' or zone.center is None
            or not 0 < zone.size < target_size or strategy not in ('auto', 'native', None) or not recombine):
        return None
    try:
        data = record['data']
        if record['digest'] != _digest(data) or data['source'] != str(source.model_id):
            raise MeshError('invalid source patch proposal digest or namespace')
        face = data['face']
        if type(face) is not int or face not in source.faces or not _ordinary_face(source, face, _vertex_faces(source)):
            return None
        if data['face_checksum'] != _face_signature(source, face, _native_definitions(source)[face]):
            return None
        support = _support(source, face)
        if support is None:
            return None
        uv = np.asarray([row[0] for row in data['node_rows']], dtype=float)
        xyz = np.asarray([row[1] for row in data['node_rows']], dtype=float)
        cells = data['cells']
        if (not len(cells) or not np.isfinite(uv).all() or not np.isfinite(xyz).all()
                or any(len(c) not in (3,4) or len(set(c)) != len(c)
                       or any(type(n) is not int or n < 0 or n >= len(uv) for n in c) for c in cells)):
            raise MeshError('invalid source patch row table')
        center = np.asarray(zone.center)
        target_uv = np.array(closest_uv(support, center))
        scale = float(np.linalg.norm(np.ptp(xyz, axis=0)))
        tol = source.tolerance.effective_length(scale)
        if np.linalg.norm(support.evaluate(*target_uv) - center) > source.tolerance.effective_surface_residual(scale):
            return None
        hits = [i for i, c in enumerate(cells) if len(c)==4 and _positive(uv[c], target_uv)]
        if len(hits) != 1 or (disk := _cut(data, hits[0], uv)) is None:
            return None
        selected, interface = disk
        grown, kept, _ = _subdivide(data, selected, support)
        radius = zone.radius + zone.size * log(target_size/zone.size)/log(zone.growth)
        if not _outside_support(support, uv, kept.values(), center, radius + tol, cancellation_check):
            return None
        new_cells = grown['cells'][len(kept):]
        points = np.asarray([[grown['node_rows'][n][1] for n in cell] for cell in new_cells])
        if not _achieves_zone(SizeField(source, target_size, (zone,)), zone, points, tol):
            return None
        return {'face': face, 'proposal_digest': record['digest'], 'target_size': float(target_size),
                'strategy': strategy, 'zone': {'center': list(zone.center), 'size': zone.size,
                'radius': zone.radius, 'growth': zone.growth}, 'replaced_cell_rows': selected,
                'interface_rows': interface}
    except (KeyError, TypeError, IndexError, ValueError, OverflowError) as error:
        preserve_stage_cancellation(error)
        raise MeshError('invalid source patch coordinates or topology') from error
