"""Provider-independent admission for a single convex planar normal prism.

All predicates use translated, diagonal-normalized coordinates. Geometric
tolerance admits source-boundary approximation; roundoff tolerances for cell
intersection/topology are separate and are never enlarged to that tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from types import MappingProxyType
from itertools import combinations
import math
import re
import uuid
import numpy as np

from .volume import (InvalidVolumeMesh, UnsupportedVolumeGeometry,
                     VolumeMeshingCancelled, _array, _cancelled, _integers, _freeze, _positive)

_EPS = 256 * np.finfo(float).eps
_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_EDGES = tuple(combinations(range(4), 2))


def _check_cancel(cancel):
    if _cancelled(cancel):
        raise VolumeMeshingCancelled('cancelled during volume validation')


@dataclass(frozen=True)
class _Source:
    coordinates: np.ndarray
    loops: tuple
    refs: tuple
    groups: Mapping
    normals: np.ndarray
    areas: np.ndarray
    perimeter: np.ndarray
    normalized: np.ndarray
    origin: np.ndarray
    diagonal: float
    delta: float
    volume: float
    model_id: str
    revision: int
    region_id: str
    signature: str
    provenance: Mapping


def validate_source(snapshot, delta):
    """Capture and independently check the owner snapshot without mutation."""
    delta = _positive(delta, 'boundary_tolerance')
    try:
        if isinstance(snapshot.schema_version, bool) or snapshot.schema_version != 1 or snapshot.units != 'm':
            raise UnsupportedVolumeGeometry('only solid snapshot v1 in metres is supported')
        if snapshot.geometry_class != 'convex_normal_prism':
            raise UnsupportedVolumeGeometry('only single convex normal prisms are supported')
        x = _array(snapshot.coordinates, float)
        loops = tuple(tuple(loop) for loop in snapshot.face_loops)
        refs = tuple(snapshot.face_refs)
        groups = {name: tuple(handles) for name, handles in snapshot.face_groups.items()}
        model_id, revision, region_id = snapshot.model_id, snapshot.revision, snapshot.region_id
        signature, volume = snapshot.source_signature, float(snapshot.volume)
        if not isinstance(snapshot.provenance, Mapping):
            raise InvalidVolumeMesh('source provenance must be a JSON-compatible mapping')
        provenance = _freeze(snapshot.provenance)
        bounds = np.asarray(snapshot.bounds, dtype=float)
    except (AttributeError, TypeError, OverflowError, ValueError) as exc:
        if isinstance(exc, (InvalidVolumeMesh, UnsupportedVolumeGeometry)):
            raise
        raise InvalidVolumeMesh('malformed solid-region snapshot') from exc
    try:
        uuid.UUID(model_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise InvalidVolumeMesh('snapshot model_id must be a UUID string') from exc
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0 or not isinstance(region_id, str) or not region_id:
        raise InvalidVolumeMesh('invalid source revision or region identity')
    if not isinstance(signature, str) or re.fullmatch('[0-9a-f]{64}', signature) is None:
        raise InvalidVolumeMesh('source_signature must be a SHA-256 hex string')
    if x.ndim != 2 or x.shape[1:] != (3,) or len(x) < 6 or not np.isfinite(x).all():
        raise InvalidVolumeMesh('snapshot coordinates must be finite (N,3), N>=6')
    if len(np.unique(x, axis=0)) != len(x):
        raise InvalidVolumeMesh('duplicate source coordinate rows')
    if bounds.shape != (2, 3) or not np.array_equal(bounds, [x.min(axis=0), x.max(axis=0)]):
        raise InvalidVolumeMesh('source bounds disagree with coordinates')
    with np.errstate(over='ignore', invalid='ignore'):
        diagonal = float(np.linalg.norm(x.max(axis=0) - x.min(axis=0)))
    if not math.isfinite(diagonal) or diagonal <= 0:
        raise InvalidVolumeMesh('source bounds have unsupported numerical scale')
    y = (x - x[0]) / diagonal
    tolerance = _EPS * max(1., np.max(np.abs(x)) / diagonal)
    if tolerance > delta / diagonal / 16:
        raise InvalidVolumeMesh('boundary_tolerance is too small for source floating-point resolution')
    from anygeometry.identity import EntityHandle
    if len(refs) != len(loops) or any(
            not isinstance(h, EntityHandle) or h.kind != 'face' or str(h.model_id) != model_id for h in refs) or len(set(refs)) != len(refs):
        raise InvalidVolumeMesh('source faces need unique owner-bound face handles')
    if any(not isinstance(k, str) or not k or any(not isinstance(h, EntityHandle) or h not in refs for h in v) or len(set(v)) != len(v)
           for k, v in groups.items()):
        raise InvalidVolumeMesh('source face groups must select unique known face handles')
    normals, areas, perimeters = [], [], []
    edges = defaultdict(list)
    used = set()
    source_volume = 0.
    for fi, loop in enumerate(loops):
        if len(loop) < 3 or any(not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)) or k < 0 or k >= len(y) for k in loop) or len(set(loop)) != len(loop):
            raise InvalidVolumeMesh('source loops must contain unique known coordinate rows')
        used.update(loop)
        p = y[list(loop)]
        edge = np.roll(p, -1, axis=0) - p
        if np.any(np.linalg.norm(edge, axis=1) <= tolerance):
            raise InvalidVolumeMesh('source face has a degenerate edge')
        cross = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(cross)
        if norm <= tolerance:
            raise InvalidVolumeMesh('source face begins with collinear corners')
        n = cross / norm
        if np.max(np.abs((p - p[0]) @ n)) > tolerance:
            raise UnsupportedVolumeGeometry('nonplanar source face')
        if np.max((y - p[0]) @ n) > tolerance:
            raise UnsupportedVolumeGeometry('source is not convex with outward face loops')
        turns = np.einsum('ij,j->i', np.cross(edge, np.roll(edge, -1, axis=0)), n)
        if np.any(turns <= tolerance * np.linalg.norm(edge, axis=1)):
            raise UnsupportedVolumeGeometry('source polygon must be strictly convex without collinear corners')
        area_vec = np.sum(np.cross(p - p[0], np.roll(p, -1, axis=0) - p[0]), axis=0) / 2
        area = float(area_vec @ n)
        normals.append(n)
        areas.append(area)
        perimeters.append(float(np.linalg.norm(edge, axis=1).sum()))
        for k, j in zip(loop, (*loop[1:], loop[0])):
            edges[tuple(sorted((k, j)))].append((k, j, fi))
        for k in range(1, len(p) - 1):
            source_volume += float(np.dot(p[0], np.cross(p[k], p[k + 1]))) / 6
    if used != set(range(len(y))) or any(len(v) != 2 or v[0][:2] != v[1][:2][::-1] for v in edges.values()):
        raise UnsupportedVolumeGeometry('source must have one closed oriented manifold boundary')
    if len(y) - len(edges) + len(loops) != 2:
        raise UnsupportedVolumeGeometry('source must have spherical closed-prism topology')
    normals = np.asarray(normals)
    # Independently enforce the declared normal-prism envelope. An arbitrary
    # convex polyhedron with a forged class label cannot reach the provider.
    is_prism = False
    for i, j in combinations(range(len(loops)), 2):
        a, b = loops[i], loops[j]
        if len(a) != len(b) or len(y) != 2 * len(a) or len(loops) != len(a) + 2 or set(a) & set(b):
            continue
        if np.linalg.norm(normals[i] + normals[j]) > tolerance:
            continue
        height = float((y[b[0]] - y[a[0]]) @ normals[i])
        shift = normals[i] * height
        if all(np.min(np.linalg.norm(y[list(b)] - (y[k] + shift), axis=1)) <= tolerance for k in a):
            if all(len(loop) == 4 for k, loop in enumerate(loops) if k not in (i, j)):
                is_prism = True
                break
    if not is_prism:
        raise UnsupportedVolumeGeometry('source is not a single normal extrusion of a convex polygon')
    if not math.isfinite(volume) or volume <= 0 or source_volume <= 0 or abs(volume / diagonal / diagonal / diagonal - source_volume) > _EPS * max(source_volume, 1.):
        raise InvalidVolumeMesh('analytic source volume disagrees with closed loops')
    # Keep tolerance well below every distinct face separation and source edge.
    # This is an explicit refusal, never a silently reduced requested tolerance.
    feature = min(np.linalg.norm(y[a] - y[b]) for a, b in edges)
    for loop, n in zip(loops, normals):
        distances = -(y - y[loop[0]]) @ n
        positive = distances[distances > tolerance]
        if len(positive):
            feature = min(feature, float(positive.min()))
    if delta / diagonal >= feature / 8:
        raise InvalidVolumeMesh('boundary_tolerance must be less than one eighth of source feature separation')
    return _Source(x, loops, refs, MappingProxyType(groups), _array(normals),
                   _array(areas), _array(perimeters), _array(y), x[0], diagonal,
                   delta, volume, model_id, revision, region_id, signature, provenance)


def _normalized_mesh(x, t):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1:] != (3,) or len(x) < 4 or x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
        raise InvalidVolumeMesh('coordinates must be finite (N,3), N>=4')
    t = _integers(t, 'cells')
    if t.ndim != 2 or t.shape[1:] != (4,) or not len(t):
        raise InvalidVolumeMesh('only nonempty tet4 cells are supported')
    if t.min() < 0 or t.max() >= len(x):
        raise InvalidVolumeMesh('cells reference unknown rows')
    if np.any(np.diff(np.sort(t, axis=1), axis=1) == 0):
        raise InvalidVolumeMesh('cell repeats a vertex')
    if len(np.unique(np.sort(t, axis=1), axis=0)) != len(t):
        raise InvalidVolumeMesh('duplicate cells')
    if len(np.unique(t)) != len(x):
        raise InvalidVolumeMesh('all coordinate rows must belong to cells')
    if len(np.unique(x, axis=0)) != len(x):
        raise InvalidVolumeMesh('duplicate coordinate rows')
    with np.errstate(over='ignore', invalid='ignore'):
        diagonal = np.linalg.norm(x.max(axis=0) - x.min(axis=0))
        y = (x - x[0]) / diagonal
    if not np.isfinite(y).all() or not math.isfinite(diagonal) or diagonal <= 0:
        raise InvalidVolumeMesh('mesh has unsupported coordinate scale')
    return y, t


def _tet_planes(p):
    f = p[np.asarray(_FACES)]
    normals = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return normals, np.einsum('ij,ij->i', normals, f[:, 0])


def _intersection_points(p, q, planes):
    """Vertices of an intersection via clipping tetra edges to halfspaces.

    Includes coplanar contacts. Every bounded polytope intersection vertex lies
    on an edge of at least one tetrahedron, including the coincident-face case.
    """
    normals, offsets = planes
    points = []
    for i, j in _EDGES:
        a, direction = p[i], p[j] - p[i]
        distances = normals @ a - offsets
        slopes = normals @ direction
        lo, hi = 0., 1.
        feasible = True
        for d, slope in zip(distances, slopes):
            if abs(slope) <= _EPS:
                if d > _EPS:
                    feasible = False
                    break
            elif slope > 0:
                hi = min(hi, -d / slope)
            else:
                lo = max(lo, -d / slope)
        if feasible and lo <= hi and hi >= 0 and lo <= 1:
            points.extend((a + max(lo, 0) * direction, a + min(hi, 1) * direction))
    return points


def _orientation_sign(a, b, c, point):
    """Certified sign of orient3d on the supplied binary coordinates, else zero.

    The determinant is expanded after translation. 64 machine epsilons times
    the absolute-product permanent conservatively exceeds the orient3d
    forward-error bound for the subtractions, products and additions. The
    tiny additive floor rejects underflow-sensitive results. Zero means that
    the original SAT/clipping/exact route must decide; it never means coplanar.
    """
    ux, uy, uz = (float(b[k]) - float(a[k]) for k in range(3))
    vx, vy, vz = (float(c[k]) - float(a[k]) for k in range(3))
    wx, wy, wz = (float(point[k]) - float(a[k]) for k in range(3))
    yz, zy = vy*wz, vz*wy
    zx, xz = vz*wx, vx*wz
    xy, yx = vx*wy, vy*wx
    determinant = ux*(yz-zy) + uy*(zx-xz) + uz*(xy-yx)
    permanent = abs(ux)*(abs(yz)+abs(zy)) + abs(uy)*(abs(zx)+abs(xz)) + abs(uz)*(abs(xy)+abs(yx))
    error = 64*np.finfo(float).eps*permanent + 64*np.finfo(float).tiny
    return 1 if determinant > error else -1 if determinant < -error else 0


def _certified_face_contact(p, q, common, distances):
    """Prove the entire tet intersection equals its shared topological simplex.

    A face of p contains every common vertex. Its opposite vertex is strictly
    inside, while every noncommon vertex of q is strictly outside, all proved
    by filtered orientation signs. Thus p is in the inside closed halfspace;
    q meets that halfspace only in conv(common). That simplex belongs to both
    tets, so it is exactly their intersection. This covers vertex/edge contact
    without assuming that arbitrary topological neighbours cannot overlap.
    """
    in_p = np.any(np.all(p[:, None, :] == common[None, :, :], axis=2), axis=1)
    in_q = np.any(np.all(q[:, None, :] == common[None, :, :], axis=2), axis=1)
    if np.count_nonzero(in_p) != len(common) or np.count_nonzero(in_q) != len(common):
        return False
    # Normalized distances are only a cheap candidate screen, never the proof.
    candidates = np.flatnonzero(~in_p & np.all(distances[~in_q] > 16*_EPS, axis=0))
    for opposite in candidates:
        a, b, c = p[list(_FACES[opposite])]
        if _orientation_sign(a, b, c, p[opposite]) != -1:
            continue
        if all(_orientation_sign(a, b, c, vertex) == 1 for vertex in q[~in_q]):
            return True
    return False


def _check_pair(p, q, common, planes_p, planes_q):
    # Direct validate_cells callers may retain float32 normalization. Its
    # projection rounding does not satisfy the float64 fast-path margin, so
    # preserve the original routine for every non-float64 dtype.
    if p.dtype != np.float64 or q.dtype != np.float64:
        return _check_pair_fallback(p, q, common, planes_p, planes_q)
    # Face axes are a subset of the original SAT axes. A full extra _EPS
    # margin makes this sufficient early exit conservative with respect to
    # the original -_EPS threshold; ambiguous/near-threshold results fall back.
    axes = np.concatenate((planes_p[0], planes_q[0]))
    proj_p, proj_q = p @ axes.T, q @ axes.T
    overlap = np.minimum(proj_p.max(axis=0), proj_q.max(axis=0)) - np.maximum(proj_p.min(axis=0), proj_q.min(axis=0))
    if np.any(overlap < -2*_EPS):
        return
    if 0 < len(common) < 3:
        if _certified_face_contact(p, q, common, proj_q[:, :4] - planes_p[1]):
            return
        if _certified_face_contact(q, p, common, proj_p[:, 4:] - planes_q[1]):
            return
    _check_pair_fallback(p, q, common, planes_p, planes_q)


def _check_pair_fallback(p, q, common, planes_p, planes_q):
    # Strict SAT on all tetra face normals and edge cross axes detects positive
    # volume intersections. A numerical uncertainty is a refusal, not a repair.
    axes = [*planes_p[0], *planes_q[0]]
    for a, b in _EDGES:
        for c, d in _EDGES:
            cross = np.cross(p[b] - p[a], q[d] - q[c])
            norm = np.linalg.norm(cross)
            if norm > _EPS:
                axes.append(cross / norm)
    axes = np.asarray(axes)
    proj_p, proj_q = p @ axes.T, q @ axes.T
    overlap = np.minimum(proj_p.max(axis=0), proj_q.max(axis=0)) - np.maximum(proj_p.min(axis=0), proj_q.min(axis=0))
    if np.any(overlap < -_EPS):
        return
    if np.all(overlap > _EPS):
        raise InvalidVolumeMesh('tetrahedra overlap in positive volume')
    points = _intersection_points(p, q, planes_q) + _intersection_points(q, p, planes_p)
    if not points:
        return
    if len(common) == 0:
        raise InvalidVolumeMesh('nonconforming cell contact or boundary crossing')
    points = np.asarray(points)
    if len(common) == 1:
        distance = np.linalg.norm(points - common[0], axis=1)
    elif len(common) == 2:
        e = common[1] - common[0]
        fraction = (points - common[0]) @ e / (e @ e)
        distance = np.linalg.norm(points - (common[0] + np.clip(fraction, 0, 1)[:, None] * e), axis=1)
    else:
        n = np.cross(common[1] - common[0], common[2] - common[0])
        n /= np.linalg.norm(n)
        distance = np.abs((points - common[0]) @ n)
    # Numerical contact uncertainty is separate from boundary_tolerance.
    if np.any(distance > 32 * _EPS):
        # Near-parallel planes in thin prisms make floating edge clipping ill
        # conditioned. Resolve the predicate on the *same binary coordinates*
        # with exact rational arithmetic; never enlarge an acceptance tolerance.
        _exact_contact(p, q, common)


def _exact_contact(p, q, common):
    from fractions import Fraction
    def rows(a):
        return [tuple(Fraction(float(v)) for v in row) for row in a]
    def sub(a, b): return tuple(x - y for x, y in zip(a, b))
    def dot(a, b): return sum(x * y for x, y in zip(a, b))
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    p, q, common = rows(p), rows(q), rows(common)
    def clip(a, b):
        planes = [(cross(sub(b[j], b[i]), sub(b[k], b[i])), b[i]) for i, j, k in _FACES]
        out = []
        for i, j in _EDGES:
            direction = sub(a[j], a[i])
            lo, hi = Fraction(0), Fraction(1)
            feasible = True
            for n, origin in planes:
                d, slope = dot(n, sub(a[i], origin)), dot(n, direction)
                if not slope:
                    if d > 0:
                        feasible = False
                        break
                elif slope > 0: hi = min(hi, -d / slope)
                else: lo = max(lo, -d / slope)
            if feasible and lo <= hi and hi >= 0 and lo <= 1:
                for value in (max(lo, 0), min(hi, 1)):
                    out.append(tuple(x + value * d for x, d in zip(a[i], direction)))
        return out
    for point in clip(p, q) + clip(q, p):
        if not common:
            valid = False
        elif len(common) == 1:
            valid = point == common[0]
        elif len(common) == 2:
            e, v = sub(common[1], common[0]), sub(point, common[0])
            valid = not any(cross(e, v)) and 0 <= dot(e, v) <= dot(e, e)
        else:
            n = cross(sub(common[1], common[0]), sub(common[2], common[0]))
            valid = dot(n, sub(point, common[0])) == 0
        if not valid:
            raise InvalidVolumeMesh('cell intersection exceeds its shared topological simplex (exact predicate)')


def validate_cells(x, t, *, cancel=None):
    """Validate connected conforming positive tet4 cells; return outward facets."""
    y, t = _normalized_mesh(x, t)
    p = y[t]
    d = p[:, 1:] - p[:, :1]
    scales = np.max(np.linalg.norm(d, axis=2), axis=1)
    determinants = np.linalg.det(d)
    if np.any(determinants <= 64 * np.finfo(float).eps * scales**3):
        raise InvalidVolumeMesh('cells must have positive Jacobians and nondegenerate volume')
    incidence = defaultdict(list)
    for i, cell in enumerate(t):
        for local in _FACES:
            face = tuple(cell[list(local)])
            incidence[tuple(sorted(face))].append((i, face))
    adjacency = [[] for _ in t]
    exterior = []
    for key, uses in incidence.items():
        if len(uses) > 2:
            raise InvalidVolumeMesh('nonmanifold face incidence')
        if len(uses) == 1:
            exterior.append(uses[0][1])
        else:
            (i, a), (j, b) = uses
            if not any(tuple(np.roll(b[::-1], k)) == a for k in range(3)):
                raise InvalidVolumeMesh('shared cell facets have inconsistent orientation')
            adjacency[i].append(j)
            adjacency[j].append(i)
    seen, pending = {0}, [0]
    while pending:
        for other in adjacency[pending.pop()]:
            if other not in seen:
                seen.add(other)
                pending.append(other)
    if len(seen) != len(t):
        raise InvalidVolumeMesh('cells must form one face-connected region')
    edge_uses = defaultdict(list)
    for face in exterior:
        for a, b in zip(face, (*face[1:], face[0])):
            edge_uses[tuple(sorted((a, b)))].append((a, b))
    if any(len(v) != 2 or v[0] != v[1][::-1] for v in edge_uses.values()):
        raise InvalidVolumeMesh('exterior must be a closed oriented manifold')
    planes = [_tet_planes(cell) for cell in p]
    low, high = p.min(axis=1), p.max(axis=1)
    # Sweep broad phase: inspect only overlapping AABBs, no all-pairs tensor.
    order = np.argsort(low[:, 0], kind='stable')
    active = []
    for iteration, i in enumerate(order):
        if iteration % 16 == 0:
            _check_cancel(cancel)
        active = [j for j in active if high[j, 0] >= low[i, 0] - _EPS]
        for j in active:
            if np.any(high[j, 1:] < low[i, 1:] - _EPS) or np.any(high[i, 1:] < low[j, 1:] - _EPS):
                continue
            common_ids = sorted(set(t[i]) & set(t[j]))
            if len(common_ids) == 3:
                # Positive cells with oppositely oriented shared facets lie in
                # opposite closed halfspaces. Their only possible intersection
                # is that common triangle; clipping is unnecessary and ill
                # conditioned for a very thin shared face.
                continue
            _check_pair(p[i], p[j], y[common_ids], planes[i], planes[j])
        active.append(i)
    exterior.sort(key=lambda face: tuple(sorted(face)))
    return np.asarray(exterior, dtype=np.int64).reshape(-1, 3)


def _projected_triangle_overlap(a, b):
    # Strict positive-area intersection of convex 2D triangles via edge SAT.
    for triangle in (a, b):
        for e in np.roll(triangle, -1, axis=0) - triangle:
            axis = np.array([-e[1], e[0]])
            axis /= np.linalg.norm(axis)
            pa, pb = a @ axis, b @ axis
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= _EPS:
                return False
    return True


def validate_geometry(x, t, facets, source, *, cancel=None):
    """Check convex containment and exact ownership plus projected coverage.

    No nearest-face fallback. Each whole triangle must match one plane and its
    closed convex loop, with an outward normal. Coverage is geometric and global,
    not sampled: projected triangle non-overlap and per-face area conservation.
    """
    y = (np.asarray(x) - source.origin) / source.diagonal
    delta = source.delta / source.diagonal
    if not np.isfinite(y).all():
        raise InvalidVolumeMesh('output coordinate scale is invalid')
    for loop, n in zip(source.loops, source.normals):
        if np.max((y - source.normalized[loop[0]]) @ n) > delta + _EPS:
            raise InvalidVolumeMesh('mesh crosses a source boundary plane')
    owners, by_face = [], defaultdict(list)
    for index, facet in enumerate(facets):
        if index % 128 == 0:
            _check_cancel(cancel)
        p = y[facet]
        candidates = []
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        for fi, (loop, n) in enumerate(zip(source.loops, source.normals)):
            polygon = source.normalized[list(loop)]
            if np.max(np.abs((p - polygon[0]) @ n)) > delta + _EPS:
                continue
            if normal @ n <= 0:
                continue
            # Every projected point must lie inside the actual source loop,
            # permitting only the explicit geometric boundary tolerance.
            if any(np.min(np.cross(edge, p - a) @ n) < -(delta + _EPS) * np.linalg.norm(edge)
                   for a, edge in zip(polygon, np.roll(polygon, -1, axis=0) - polygon)):
                continue
            candidates.append(fi)
        if len(candidates) != 1:
            raise InvalidVolumeMesh('each boundary triangle must have exactly one geometric source face owner')
        fi = candidates[0]
        owners.append(source.refs[fi])
        by_face[fi].append(index)
    max_area_error = 0.
    for fi, (loop, n, area, perimeter) in enumerate(zip(source.loops, source.normals, source.areas, source.perimeter)):
        _check_cancel(cancel)
        indices = by_face[fi]
        if not indices:
            raise InvalidVolumeMesh('source face has no output boundary coverage')
        polygon = source.normalized[list(loop)]
        u = polygon[1] - polygon[0]
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        triangles = y[np.asarray(facets)[indices]]
        projected = np.stack(((triangles - polygon[0]) @ u, (triangles - polygon[0]) @ v), axis=2)
        a, b = projected[:, 1] - projected[:, 0], projected[:, 2] - projected[:, 0]
        actual_area = float(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2)
        # Area allowed by the Minkowski boundary strip of width delta. The
        # explicit feature/edge/corner tests below constrain boundary topology.
        error = abs(actual_area - area)
        if error > perimeter * delta + math.pi * delta**2 + _EPS:
            raise InvalidVolumeMesh('projected source face coverage area is incomplete or excessive')
        max_area_error = max(max_area_error, error)
        low, high = projected.min(axis=1), projected.max(axis=1)
        order = np.argsort(low[:, 0], kind='stable')
        active = []
        for k in order:
            active = [j for j in active if high[j, 0] >= low[k, 0] - _EPS]
            for j in active:
                if high[j, 1] < low[k, 1] - _EPS or high[k, 1] < low[j, 1] - _EPS:
                    continue
                if _projected_triangle_overlap(projected[k], projected[j]):
                    raise InvalidVolumeMesh('source-face projected boundary triangles overlap')
            active.append(k)
    # Corners and every protected edge require explicit boundary vertices/edges,
    # not a sampled distance-to-plane or a summed-area-only test.
    boundary_nodes = np.unique(facets)
    for corner in source.normalized:
        if np.min(np.linalg.norm(y[boundary_nodes] - corner, axis=1)) > delta + _EPS:
            raise InvalidVolumeMesh('source corner is not preserved within boundary_tolerance')
    mesh_edge_faces = defaultdict(set)
    for facet, owner in zip(facets, owners):
        for a, b in combinations(facet, 2):
            mesh_edge_faces[tuple(sorted((a, b)))].add(owner)
    source_edges = defaultdict(set)
    for loop, ref in zip(source.loops, source.refs):
        for a, b in zip(loop, (*loop[1:], loop[0])):
            source_edges[tuple(sorted((a, b)))].add(ref)
    allowed_pairs = {frozenset(refs) for refs in source_edges.values()}
    if any(len(adjacent) > 1 and frozenset(adjacent) not in allowed_pairs for adjacent in mesh_edge_faces.values()):
        raise InvalidVolumeMesh('mesh crease joins nonadjacent source faces')
    for (a, b), face_refs in source_edges.items():
        start, end = source.normalized[[a, b]]
        direction = end - start
        length = float(np.linalg.norm(direction))
        unit = direction / length
        intervals = []
        graph = defaultdict(set)
        for edge, adjacent in mesh_edge_faces.items():
            if adjacent != face_refs:
                continue
            p = y[list(edge)]
            positions = (p - start) @ unit
            projected = start + positions[:, None] * unit
            if np.max(np.linalg.norm(p - projected, axis=1)) > delta + _EPS:
                raise InvalidVolumeMesh('source protected edge was not preserved')
            if positions.min() < -delta - _EPS or positions.max() > length + delta + _EPS:
                raise InvalidVolumeMesh('protected edge extends beyond its source segment')
            intervals.append((float(positions.min()), float(positions.max())))
            graph[edge[0]].add(edge[1]); graph[edge[1]].add(edge[0])
        if not intervals:
            raise InvalidVolumeMesh('source protected edge is missing')
        if any(len(neighbours) > 2 for neighbours in graph.values()):
            raise InvalidVolumeMesh('source protected edge branches')
        endpoints = [node for node, neighbours in graph.items() if len(neighbours) == 1]
        if len(endpoints) != 2:
            raise InvalidVolumeMesh('source protected edge must form one open chain')
        reached, todo = {endpoints[0]}, [endpoints[0]]
        while todo:
            for node in graph[todo.pop()]:
                if node not in reached:
                    reached.add(node); todo.append(node)
        if len(reached) != len(graph):
            raise InvalidVolumeMesh('source protected edge chain is disconnected')
        endcoords = y[endpoints]
        if not ((np.linalg.norm(endcoords[0] - start) <= delta + _EPS and np.linalg.norm(endcoords[1] - end) <= delta + _EPS)
                or (np.linalg.norm(endcoords[1] - start) <= delta + _EPS and np.linalg.norm(endcoords[0] - end) <= delta + _EPS)):
            raise InvalidVolumeMesh('protected edge chain endpoints do not preserve source corners')
        coverage = sorted(intervals)
        cursor = 0.
        for lo, hi in coverage:
            if lo > cursor + 2 * delta + _EPS:
                raise InvalidVolumeMesh('source protected edge coverage has a gap')
            cursor = max(cursor, hi)
        if cursor < length - delta - _EPS:
            raise InvalidVolumeMesh('source protected edge coverage is incomplete')
    determinants = np.linalg.det(y[np.asarray(t)[:, 1:]] - y[np.asarray(t)[:, :1]])
    volume = float(np.sum(determinants) / 6)
    expected = source.volume / source.diagonal / source.diagonal / source.diagonal
    # Convex boundary displacements bounded by delta give a conservative shell
    # volume bound from the source surface area plus edge/corner terms.
    volume_bound = float(np.sum(source.areas)) * delta + float(np.sum(source.perimeter)) * delta**2 + 4 * math.pi * delta**3 / 3
    if abs(volume - expected) > volume_bound + _EPS:
        raise InvalidVolumeMesh('mesh volume disagrees with source within boundary tolerance')
    return tuple(owners), dict(cells=len(t), boundary_facets=len(facets),
                              source_faces=len(source.loops), volume=volume * source.diagonal * source.diagonal * source.diagonal,
                              source_volume=source.volume,
                              max_projected_face_area_error=max_area_error * source.diagonal**2,
                              cell_intersections='exhaustive AABB candidates with tetrahedron SAT and clipped contacts',
                              coverage='projected non-overlap, area and protected edge chains; no sampling')
