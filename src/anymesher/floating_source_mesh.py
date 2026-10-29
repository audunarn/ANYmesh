"""Bounded BEM source discretisation of the mean-wet hull, independent of the structural FE mesh.

Input: planar convex wet plate polygons (world axes, outward normals, already clipped at the waterplane by the geometry
owner).  Output: triangle panels from a regular in-plane grid per plate (``anygeometry.floating_overlap.panelize``),
with an explicit panel budget and provenance.  The size is either given (refused when it exceeds the budget) or chosen
by a deterministic geometric search; nothing is silently coarsened beyond that recorded size.  Conformity across plates is
verified by the consumer's manifold check (shared edges must be split identically on both plates), never assumed.
"""
from __future__ import annotations

import math

import numpy as np


class SourceMeshError(ValueError):
    """The requested wet-hull discretisation is impossible within its budget or inputs."""


CONFORMING_TOLERANCE_RELATIVE = 1e-9      # relative to the source radius about its centroid, at least 1 m


def conform_triangles(triangles, labels, *, tolerance_relative: float = CONFORMING_TOLERANCE_RELATIVE):
    """Conforming shared-edge refinement of independently panelised plates (no consumer repair, no jitter, no coarsening).

    1. vertices closer than the declared tolerance are CANONICALISED to one exact coordinate (a vertex within tolerance of z = 0 is
       placed exactly on the waterplane);
    2. every triangle edge is split at ALL canonical vertices lying on it (within the tolerance, strictly between its end points),
       so both plates sharing a physical edge use the same subdivision, whatever grids they were panelised on;
    3. a triangle whose boundary acquired split nodes is re-triangulated by a TRUE interior-centroid fan over its whole boundary node
       loop (collinear nodes become nondegenerate panels).  Orientation (outward normal), plate label and area are preserved exactly.
    Returns (triangles, labels, statistics).  Untouched triangles are returned unchanged (canonical coordinates)."""
    T = np.array([np.asarray(t, dtype=float) for t in triangles])
    if T.size == 0:
        return [], [], dict(edge_splits=0, refined_triangles=0, tolerance=0.0, removed_panels=0, removed_area=0.0)
    centre = T.reshape(-1, 3).mean(axis=0)
    scale = max(1.0, float(np.linalg.norm(T - centre, axis=-1).max()))
    tol = tolerance_relative * scale

    def canonicalise(points):
        """Cluster points within ``tol``; returns (ids (same shape), exact representative coordinates)."""
        cells, reps = {}, []

        def one(p):
            key = tuple(np.floor(p / tol).astype(np.int64))
            for d in np.ndindex(3, 3, 3):
                for index in cells.get((key[0] + d[0] - 1, key[1] + d[1] - 1, key[2] + d[2] - 1), ()):
                    if np.linalg.norm(reps[index] - p) <= tol:
                        return index
            reps.append(np.array(p, dtype=float))
            cells.setdefault(key, []).append(len(reps) - 1)
            return len(reps) - 1

        ids = np.array([one(p) for p in points.reshape(-1, 3)], dtype=np.int64).reshape(points.shape[:-1])
        rep = np.array(reps)
        rep[np.abs(rep[:, 2]) <= tol, 2] = 0.0            # waterplane vertices exactly on z = 0
        return ids, rep

    tri_ids, V = canonicalise(T)
    # snapping to the waterplane can bring two representatives within tolerance: canonicalise AGAIN on the snapped coordinates
    tri_ids, V = canonicalise(V[tri_ids])
    area_in = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
    # Demonstrably zero / roundoff-collapsed panels (a repeated canonical vertex, or collinear within the declared tolerance, i.e. area <=
    # tol x longest edge) are omitted WITH explicit accounting; anything larger is real load area and is never dropped.
    keep, removed_area = [], 0.0
    for k, ids in enumerate(tri_ids):
        p = V[ids]
        lmax = max(float(np.linalg.norm(p[(i + 1) % 3] - p[i])) for i in range(3))
        area_c = 0.0 if len(set(int(i) for i in ids)) < 3 else 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        if area_c <= tol * max(lmax, tol):
            lmax_in = max(float(np.linalg.norm(T[k][(i + 1) % 3] - T[k][i])) for i in range(3))
            if area_in[k] > 2.0 * tol * max(lmax_in, tol):
                raise SourceMeshError(f'source panel {k} collapses under the declared tolerance {tol:.3g} m but carries area '
                                      f'{area_in[k]:.3g} m^2: unresolvable geometry, refused (not dropped)')
            removed_area += float(area_in[k])
            continue
        keep.append(k)
    removed_panels = len(tri_ids) - len(keep)
    if not keep:
        raise SourceMeshError('every source panel collapsed under the declared tolerance')
    tri_ids = tri_ids[keep]
    labels = [labels[k] for k in keep]
    used = np.unique(tri_ids)                         # split candidates: vertices of KEPT panels only (no dangling nodes)
    edges = {}
    for ids in tri_ids:
        for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            edges.setdefault((min(int(a), int(b)), max(int(a), int(b))), None)
    splits, count = {}, 0
    for (a, b) in edges:
        e = V[b] - V[a]
        length2 = float(e @ e)
        d = V[used] - V[a]
        t = d @ e / length2
        dist = np.linalg.norm(d - np.outer(t, e), axis=1)
        eps = tol / np.sqrt(length2)
        hit = np.nonzero((dist <= tol) & (t > eps) & (t < 1.0 - eps))[0]
        if hit.size:
            splits[(a, b)] = [int(used[i]) for i in hit[np.argsort(t[hit])]]
            count += int(hit.size)
    out_tri, out_labels, refined = [], [], 0
    for ids, label in zip(tri_ids, labels):
        loop = []
        touched = False
        for a, b in ((int(ids[0]), int(ids[1])), (int(ids[1]), int(ids[2])), (int(ids[2]), int(ids[0]))):
            loop.append(a)
            inner = splits.get((min(a, b), max(a, b)))
            if inner:
                touched = True
                loop.extend(inner if a < b else inner[::-1])
        if not touched:
            out_tri.append(np.array([V[ids[0]], V[ids[1]], V[ids[2]]])); out_labels.append(label)
            continue
        refined += 1
        centre = V[ids].mean(axis=0)
        for i in range(len(loop)):
            out_tri.append(np.array([centre, V[loop[i]], V[loop[(i + 1) % len(loop)]]])); out_labels.append(label)
    area_out = sum(0.5 * float(np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0]))) for t in out_tri)
    if any(0.5 * np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) <= 0.0 for t in out_tri):
        raise SourceMeshError('conforming refinement produced a zero-area panel')
    total_in = float(area_in.sum())
    if abs(area_out - (total_in - removed_area)) > 1e-8 * max(total_in, 1e-300):
        raise SourceMeshError(f'area not conserved by the conforming refinement (in {total_in:.12g}, removed {removed_area:.3g}, '
                              f'out {area_out:.12g})')
    return out_tri, out_labels, dict(edge_splits=count, refined_triangles=refined, tolerance=tol, removed_panels=removed_panels,
                                     removed_area=removed_area, area_in=total_in, area_out=area_out)


def verify_closed_below_waterplane(triangles, tolerance: float) -> None:
    """Every edge is used exactly twice with opposite directions, except free edges lying on the waterplane (z = 0)."""
    key = lambda p: tuple(np.round(np.asarray(p) / tolerance).astype(np.int64))
    uses = {}
    for tri in triangles:
        k = [key(p) for p in tri]
        for a, b in ((k[0], k[1]), (k[1], k[2]), (k[2], k[0])):
            uses.setdefault(frozenset((a, b)), []).append((a, b))
    for pair, u in uses.items():
        a, b = tuple(pair)
        on_waterplane = a[2] == 0 and b[2] == 0
        if len(u) == 1 and on_waterplane:
            continue
        if len(u) != 2 or u[0] != u[1][::-1]:
            raise SourceMeshError('conformed source panels are not closed below the waterplane (an edge is open, non-manifold or '
                                  'inconsistently oriented); the plate junctions cannot be made conforming within the declared tolerance')


def _assemble(plates, size):
    from anygeometry import floating_overlap as ov
    triangles, labels = [], []
    for p in plates:
        for piece in ov.panelize(p['polygon'], size):
            for tri in ov.fan_triangles(piece):
                triangles.append(tri); labels.append(str(p['label']))
    before = len(triangles)
    triangles, labels, stats = conform_triangles(triangles, labels)
    stats['panels_before_conforming'] = before
    return triangles, labels, stats


def _count(plates, size):
    """Panel count AFTER conforming refinement (the budget applies to the refined source)."""
    return len(_assemble(plates, size)[0])


def choose_panel_size(plates, max_panels: int, *, growth: float = 1.15, limit: int = 200) -> float:
    """Smallest size on a deterministic geometric ladder whose panel count is within the budget."""
    from anygeometry import floating_overlap as ov
    area = sum(ov.polygon_area(p['polygon']) for p in plates)
    if not area > 0.0 or max_panels < 4:
        raise SourceMeshError('wet hull has no area or the panel budget is below 4')
    size = math.sqrt(area / max(1.0, max_panels / 4.0))
    for _ in range(limit):
        if _count(plates, size) <= max_panels:
            return float(size)
        size *= growth
    raise SourceMeshError('no panel size within the budget was found')


def build_wet_source_mesh(plates, *, size: float = 0.0, max_panels: int = 600, require_closed: bool = False) -> dict:
    """``plates``: list of dict(label=str, polygon=(m,3)).  Returns triangles, labels and provenance.

    ``require_closed=False`` (default, the original standalone-plate behaviour): open patches are allowed.  Consumers that mesh a COMPLETE
    hull (the floating source) pass ``True``: the refined panels must then be closed below the waterplane (every edge used exactly twice
    with opposite directions except free waterplane edges) or the build is refused - nothing is repaired."""
    from anygeometry import floating_overlap as ov
    if not plates:
        raise SourceMeshError('no wet plates')
    chosen = float(size) if size and size > 0.0 else choose_panel_size(plates, max_panels)
    triangles, labels, stats = _assemble(plates, chosen)
    if not triangles or len(triangles) > max_panels:
        raise SourceMeshError(f'{len(triangles)} source panels (after conforming refinement of {stats["panels_before_conforming"]}) at '
                              f'size {chosen:.6g} m are outside the budget 1..{max_panels}; the requested size is not changed')
    if require_closed:
        verify_closed_below_waterplane(triangles, stats['tolerance'])
    provenance = dict(generator='anymesher.floating_source_mesh.v2', panel_size_m=chosen, panels=len(triangles),
                      closed_hull_verified=bool(require_closed), conforming_removed_panels=stats['removed_panels'],
                      conforming_removed_area_m2=stats['removed_area'], conforming_area_in_m2=stats['area_in'],
                      conforming_area_out_m2=stats['area_out'],
                      panels_before_conforming=stats['panels_before_conforming'], conforming_edge_splits=stats['edge_splits'],
                      conforming_refined_triangles=stats['refined_triangles'], conforming_tolerance_m=stats['tolerance'],
                      conforming='shared-edge splits + interior-centroid fans; vertices canonicalised within the declared tolerance',
                      budget=int(max_panels), size_chosen_automatically=not (size and size > 0.0))
    return dict(triangles=triangles, labels=labels, provenance=provenance)
