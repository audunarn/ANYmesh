"""Bounded deterministic one-ring repair for detached chart triangulations."""

from dataclasses import dataclass

import numpy as np

from .errors import MeshError
from .quality_v2 import triangle_quality


@dataclass(frozen=True)
class AngleRepair:
    points: np.ndarray
    moved_nodes: tuple
    trials: int
    accepted_moves: int
    budget_exhausted: bool


def repair_triangle_angles(points, triangles, protected_edges, settings, *,
                           max_trials=2048, cancellation_check=None):
    """Improve failed local policy without changing connectivity or fixed rows.

    Incidence is prepared once. Each trial evaluates only the moving vertex's
    star and growth links incident on it. Coordinates remain detached until the
    caller requalifies the chart and physical surface.
    """
    if isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 0:
        raise MeshError("angle repair requires a nonnegative trial budget")
    original = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    if original.ndim != 2 or original.shape[1] != 2 or not np.all(np.isfinite(original)):
        raise MeshError("angle repair needs finite chart pairs")
    if cells.ndim != 2 or cells.shape[1] != 3 or np.any(cells < 0) or np.any(cells >= len(original)):
        raise MeshError("angle repair needs valid T3 connectivity")
    working = original.copy()
    fixed = frozenset(int(n) for n in np.asarray(protected_edges).flat)
    stars = [set() for _ in working]
    edge_rows = {}
    for row, triangle in enumerate(cells):
        for node in triangle:
            stars[int(node)].add(row)
        for i, a in enumerate(triangle):
            pair = tuple(sorted((int(a), int(triangle[(i + 1) % 3]))))
            edge_rows.setdefault(pair, []).append(row)
    if any(len(rows) > 2 for rows in edge_rows.values()):
        raise MeshError("angle repair refuses non-manifold chart topology")
    boundary_nodes = {n for pair, rows in edge_rows.items() if len(rows) == 1 for n in pair}
    fixed = fixed | boundary_nodes
    neighbours = [set() for _ in cells]
    for rows in edge_rows.values():
        if len(rows) == 2:
            a, b = rows
            neighbours[a].add(b)
            neighbours[b].add(a)

    def qualities(coordinates):
        # Local cells use disjoint rows so the quality oracle needs no global
        # coordinate copy, native topology rebuild, or remapped node identity.
        edges = np.roll(coordinates, -1, axis=1) - coordinates
        delta, other = coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]
        signed = delta[:, 0] * other[:, 1] - delta[:, 1] * other[:, 0]
        if np.any(signed <= 0):
            return None
        xyz = np.column_stack((coordinates.reshape((-1, 2)), np.zeros(coordinates.size // 2)))
        quality = triangle_quality(xyz, np.arange(len(xyz), dtype=np.int64).reshape((-1, 3)))
        values = np.column_stack((quality.minimum_angle, quality.maximum_angle,
                                  quality.aspect_ratio, quality.scaled_jacobian))
        lengths = np.mean(np.linalg.norm(edges, axis=2), axis=1)
        if not np.all(np.isfinite(values)) or np.any(lengths <= 0) or np.any(quality.scaled_jacobian <= 0):
            return None
        return values, lengths

    initial = qualities(working[cells])
    if initial is None:
        raise MeshError("angle repair requires an initially valid triangulation")
    values, lengths = initial

    def local_key(affected, updates=None, length_updates=None):
        updates = {} if updates is None else updates
        length_updates = {} if length_updates is None else length_updates
        records = np.array([updates.get(row, values[row]) for row in affected])
        growth = []
        for row in affected:
            size = length_updates.get(row, lengths[row])
            ratios = [max(size / length_updates.get(other, lengths[other]),
                          length_updates.get(other, lengths[other]) / size)
                      for other in neighbours[row]]
            growth.append(max(ratios, default=1.0))
        penalties = np.column_stack((
            np.maximum(settings.min_angle - records[:, 0], 0.) / max(settings.min_angle, 1.),
            np.maximum(records[:, 1] - settings.max_angle, 0.) / max(settings.max_angle, 1.),
            np.maximum(records[:, 2] - settings.max_aspect_ratio, 0.) / settings.max_aspect_ratio,
            np.maximum(settings.min_scaled_jacobian - records[:, 3], 0.) / max(settings.min_scaled_jacobian, 1.e-15),
            np.maximum(np.asarray(growth) - settings.max_element_growth, 0.) / settings.max_element_growth,
        ))
        failed = np.any(penalties > 0., axis=1)
        return (float(np.sum(penalties)), int(np.count_nonzero(failed)),
                -float(np.min(records[:, 0])), float(np.max(records[:, 2])))

    def failing_rows():
        failed = ((values[:, 0] < settings.min_angle) |
                  (values[:, 1] > settings.max_angle) |
                  (values[:, 2] > settings.max_aspect_ratio) |
                  (values[:, 3] < settings.min_scaled_jacobian))
        for row, adjacent in enumerate(neighbours):
            if any(max(lengths[row] / lengths[other], lengths[other] / lengths[row]) >
                   settings.max_element_growth for other in adjacent):
                failed[row] = True
        return np.flatnonzero(failed)

    def publication_key():
        penalty, count, angle, aspect = local_key(range(len(cells)))
        return count, penalty, angle, aspect

    trials = 0
    accepted = 0
    best_points = working.copy()
    best_key = publication_key()
    best_accepted = 0
    for sweep in range(4):
        poor = failing_rows()
        movable = sorted({int(n) for row in poor for n in cells[row] if int(n) not in fixed})
        progress = False
        for node in movable:
            if trials >= max_trials:
                break
            if cancellation_check is not None:
                cancellation_check("quadratic local angle repair node")
            star = sorted(stars[node])
            affected = sorted(set(star).union(*(neighbours[row] for row in star)))
            adjacent = sorted({int(n) for row in star for n in cells[row] if int(n) != node})
            origin = working[node].copy()
            scale = float(np.mean(np.linalg.norm(working[adjacent] - origin, axis=1)))
            if not np.isfinite(scale) or scale <= 0:
                raise MeshError("angle repair has invalid local feature scale")
            targets = [np.mean(working[adjacent], axis=0)]
            for row in star:
                if values[row, 0] >= settings.min_angle:
                    continue
                a, b = [int(n) for n in cells[row] if int(n) != node]
                edge = working[b] - working[a]
                normal = np.array((-edge[1], edge[0]))
                middle = 0.5 * (working[a] + working[b])
                direction = 1. if float((origin - middle) @ normal) >= 0 else -1.
                targets.append(middle + direction * (np.sqrt(3.) / 2.) * normal)
            proposals = []
            for target in targets:
                delta = target - origin
                distance = float(np.linalg.norm(delta))
                if distance > 0.35 * scale:
                    delta *= 0.35 * scale / distance
                proposals.extend(origin + weight * delta for weight in (0.2, 0.4, 0.6, 1.0))
            axis = working[adjacent[0]] - origin
            axis /= np.linalg.norm(axis)
            normal = np.array((-axis[1], axis[0]))
            for distance in (0.05, 0.15, 0.3):
                for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    proposals.append(origin + distance * scale * (x * axis + y * normal) / np.sqrt(x*x + y*y))
            local_best_key = local_key(affected)
            best = None
            seen = set()
            coordinates = working[cells[star]].copy()
            mask = cells[star] == node
            for proposal in proposals:
                if trials >= max_trials:
                    break
                key = proposal.tobytes()
                if key in seen or key == origin.tobytes():
                    continue
                seen.add(key)
                trials += 1
                if cancellation_check is not None:
                    cancellation_check("quadratic local angle repair trial")
                trial_coordinates = coordinates.copy()
                trial_coordinates[mask] = proposal
                evaluated = qualities(trial_coordinates)
                if evaluated is None:
                    continue
                new_values, new_lengths = evaluated
                updates = dict(zip(star, new_values))
                length_updates = dict(zip(star, new_lengths))
                trial_key = local_key(affected, updates, length_updates)
                if trial_key < local_best_key:
                    local_best_key = trial_key
                    best = (proposal.copy(), new_values, new_lengths)
            if best is not None:
                working[node], values[star], lengths[star] = best
                accepted += 1
                progress = True
        # Exploration may trade small local violations while decreasing the
        # continuous policy penalty. Only a whole-face improved checkpoint is
        # returned; discarded exploration never inflates committed move counts.
        candidate_key = publication_key()
        if candidate_key < best_key:
            best_key = candidate_key
            best_points = working.copy()
            best_accepted = accepted
        if not progress or trials >= max_trials:
            break
    if cancellation_check is not None:
        cancellation_check("quadratic local angle repair complete")
    exhausted = trials >= max_trials and best_key[0] > 0
    moved = tuple(row for row in range(len(original))
                  if best_points[row].tobytes() != original[row].tobytes())
    return AngleRepair(best_points, moved, trials, best_accepted, exhausted)
