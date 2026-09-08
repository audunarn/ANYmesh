"""Bounded reference repair for a protected corner with unequal edge spacing.

Only a two-triangle corner cavity may gain a point. Its opposite, unprotected
vertex may move inside its existing star. All edits are detached proposals.
"""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CornerCavityRepair:
    points: np.ndarray
    triangles: np.ndarray
    attempts: int
    added_points: int
    moved_nodes: tuple
    improved: bool
    corner_node: int | None = None


def _canonical(cells):
    rows = []
    for row in cells:
        row = tuple(int(v) for v in row)
        k = row.index(min(row))
        rows.append(row[k:] + row[:k])
    return np.asarray(sorted(rows), dtype=np.int64).reshape(-1, 3)


def _areas(points, cells):
    xy = points[cells]
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def repair_corner_cavity(points, triangles, protected_edges, poor_triangle_ids, *,
                         score, max_candidates=256, max_insertions=1,
                         cancellation_check=None, excluded_corners=(), relax=None,
                         consume_work=None, target_met=None):
    """Propose at most one interior insertion; ``poor_triangle_ids`` are zero based."""
    original = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    if (original.ndim != 2 or original.shape[1] != 2
            or not np.isfinite(original).all() or cells.ndim != 2 or cells.shape[1] != 3
            or np.any(cells < 0) or np.any(cells >= len(original))
            or np.any(_areas(original, cells) <= 0)):
        raise ValueError("invalid corner-cavity input")
    for value in (max_candidates, max_insertions):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid corner-cavity budget")
    edges = {}
    for index, row in enumerate(cells):
        for a, b in zip(row, np.roll(row, -1), strict=True):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    protected = {tuple(sorted(map(int, edge))) for edge in protected_edges}
    if any(len(rows) > 2 for rows in edges.values()) or not protected.issubset(edges):
        raise ValueError("invalid corner-cavity incidence")
    protected.update(edge for edge, rows in edges.items() if len(rows) == 1)
    fixed = {v for edge in protected for v in edge}
    bad = sorted(set(int(i) for i in poor_triangle_ids))
    if any(i < 0 or i >= len(cells) for i in bad):
        raise ValueError("invalid corner-cavity poor element")
    # Spend the bounded search on the worst angle first, not the first element
    # record. atan2 remains well conditioned for near-collinear triangles.
    severity = {}
    for i in bad:
        xy = original[cells[i]]
        forward = np.roll(xy, -1, axis=0) - xy
        backward = np.roll(xy, 1, axis=0) - xy
        cross = np.abs(forward[:, 0] * backward[:, 1] - forward[:, 1] * backward[:, 0])
        angles = np.arctan2(cross, np.sum(forward * backward, axis=1))
        severity[i] = (float(angles.min()), tuple(sorted(map(int, cells[i]))))
    bad.sort(key=severity.__getitem__)
    best_points, best_cells = original.copy(), cells.copy()
    best_score = score(best_points, best_cells)
    attempts = 0
    moved = ()
    best_corner = None
    last_corner = None
    excluded = set(map(int, excluded_corners))
    fixed_rows = sorted(fixed)

    def finish():
        inserted = len(best_points) > len(original)
        changed = tuple(i for i in range(len(original))
                        if best_points[i].tobytes() != original[i].tobytes())
        return CornerCavityRepair(best_points, best_cells, attempts, int(inserted), changed,
                                  inserted, best_corner if best_corner is not None else last_corner)

    def consider(trial, candidate_cells):
        nonlocal best_points, best_cells, best_score, moved, best_corner
        if relax is not None:
            trial = np.asarray(relax(trial.copy(), candidate_cells.copy()), dtype=np.float64).copy()
            if trial.shape != (len(original) + 1, 2) or not np.isfinite(trial).all():
                raise ValueError("invalid corner relaxation result")
            if trial[fixed_rows].tobytes() != original[fixed_rows].tobytes():
                raise ValueError("corner relaxation changed protected coordinates")
            if np.any(_areas(trial, candidate_cells) <= 0):
                raise ValueError("corner relaxation produced invalid topology")
        candidate_score = score(trial, candidate_cells)
        if candidate_score < best_score:
            best_points, best_cells, best_score = trial, candidate_cells.copy(), candidate_score
            moved, best_corner = (opposite,), corner
            return target_met is not None and target_met(best_points, best_cells)
        return False
    if not max_candidates or not max_insertions:
        return CornerCavityRepair(best_points, best_cells, 0, 0, (), False)

    for index in bad:
        row = cells[index]
        for corner in sorted(map(int, row)):
            if cancellation_check is not None:
                cancellation_check()
            if corner in excluded:
                continue
            other = [int(v) for v in row if int(v) != corner]
            if len(other) != 2:
                continue
            b, c = other
            diagonal = tuple(sorted((b, c)))
            if (tuple(sorted((corner, b))) in protected
                    and tuple(sorted((corner, c))) in protected
                    and diagonal not in protected and len(edges[diagonal]) == 2):
                neighbor = next(i for i in edges[diagonal] if i != index)
                opposite = next(int(v) for v in cells[neighbor] if int(v) not in (b, c))
            else:
                # The corner may already have a two-triangle fan. Its internal
                # corner-to-interior edge, rather than the boundary diagonal,
                # identifies the same four-sided cavity.
                boundary = [v for v in other if tuple(sorted((corner, v))) in protected]
                interior = [v for v in other if v not in fixed]
                if len(boundary) != 1 or len(interior) != 1:
                    continue
                b, opposite = boundary[0], interior[0]
                shared = tuple(sorted((corner, opposite)))
                if shared in protected or len(edges[shared]) != 2:
                    continue
                neighbor = next(i for i in edges[shared] if i != index)
                c = next(int(v) for v in cells[neighbor] if int(v) not in (corner, opposite))
                if tuple(sorted((corner, c))) not in protected:
                    continue
            if opposite in fixed:
                continue
            last_corner = corner
            # Keep the original oriented two-cell boundary as the replacement fan.
            local = cells[[index, neighbor]]
            incidence = Counter(tuple(sorted((int(a), int(z))))
                                for tri in local for a, z in zip(tri, np.roll(tri, -1)))
            perimeter = [(int(a), int(z)) for tri in local
                         for a, z in zip(tri, np.roll(tri, -1))
                         if incidence[tuple(sorted((int(a), int(z))))] == 1]
            vectors = original[[b, c]] - original[corner]
            lengths = np.linalg.norm(vectors, axis=1)
            order = np.argsort(-lengths, kind="stable")
            basis = vectors[order] / lengths[order, None]
            spacing = math.sqrt(float(lengths[0] * lengths[1]))
            star = np.flatnonzero(np.any(cells == opposite, axis=1))
            remaining = [i for i in star if i not in (index, neighbor)]
            new_id = len(original)
            fan = np.asarray([(a, z, new_id) for a, z in perimeter], dtype=np.int64)
            trial_cells = _canonical(np.concatenate((np.delete(cells, [index, neighbor], axis=0), fan)))
            # Three rays at a tight corner need an expanded cavity. Keeping the
            # old two-cell perimeter would put the new long-side ray outside it.
            long_node, short_node = (b, c) if lengths[0] >= lengths[1] else (c, b)
            outer_edge = tuple(sorted((long_node, opposite)))
            outer_rows = edges.get(outer_edge, ())
            third = [i for i in outer_rows if i not in (index, neighbor)]
            if outer_edge not in protected and len(outer_rows) == 2 and len(third) == 1:
                third = third[0]
                far_node = next(int(v) for v in cells[third] if int(v) not in outer_edge)
                cross_basis = float(basis[0, 0] * basis[1, 1] - basis[0, 1] * basis[1, 0])
                if far_node not in (corner, short_node) and abs(cross_basis) > 1e-12:
                    phi = math.acos(float(np.clip(np.dot(basis[0], basis[1]), -1., 1.)))
                    transverse = (basis[1] - math.cos(phi) * basis[0]) / math.sin(phi)
                    replacement = np.asarray([
                        (corner, long_node, new_id), (corner, new_id, opposite),
                        (corner, opposite, short_node), (long_node, far_node, new_id),
                        (far_node, opposite, new_id),
                    ], dtype=np.int64)
                    if cross_basis < 0:
                        replacement = replacement[:, [0, 2, 1]]
                    removed = (index, neighbor, third)
                    expanded_cells = _canonical(np.concatenate((np.delete(cells, removed, axis=0), replacement)))
                    unaffected_star = cells[[i for i in star if i not in removed]]
                    changed = np.concatenate((unaffected_star, replacement))
                    for long_scale in (.65, .7, .6, .75):
                        for short_scale in (1.7, 1.6, 1.5, 1.4):
                            for first_offset, second_offset in ((-.1, -.02), (-.05, -.01), (0., 0.), (-.2, -.04)):
                                if attempts >= max_candidates or (consume_work is not None and not consume_work()):
                                    return finish()
                                if cancellation_check is not None:
                                    cancellation_check()
                                attempts += 1
                                first = phi / 3 + math.radians(first_offset)
                                second = 2 * phi / 3 + math.radians(second_offset)
                                if not 0 < first < second < phi:
                                    continue
                                new_point = original[corner] + max(lengths) * long_scale * (
                                    math.cos(first) * basis[0] + math.sin(first) * transverse)
                                trial = np.vstack((original, new_point))
                                trial[opposite] = original[corner] + min(lengths) * short_scale * (
                                    math.cos(second) * basis[0] + math.sin(second) * transverse)
                                if np.any(_areas(trial, changed) <= 0):
                                    continue
                                if consider(trial, expanded_cells):
                                    return finish()
            for u in (.55, .65, .75, .85):
                for v in (.55, .65, .75, .85):
                    for w in (.75, 1., 1.25, 1.5):
                        for z in (.75, 1., 1.25, 1.5):
                            if attempts >= max_candidates or (consume_work is not None and not consume_work()):
                                return finish()
                            if cancellation_check is not None:
                                cancellation_check()
                            attempts += 1
                            trial = np.vstack((original, original[corner] + spacing * (u * basis[0] + v * basis[1])))
                            trial[opposite] = original[corner] + spacing * (w * basis[0] + z * basis[1])
                            changed_cells = np.concatenate((cells[remaining], fan))
                            if np.any(_areas(trial, changed_cells) <= 0):
                                continue
                            if consider(trial, trial_cells):
                                return finish()
    return finish()
