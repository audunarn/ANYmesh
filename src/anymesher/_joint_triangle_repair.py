"""Bounded coupled angle/perimeter repair, used only on detached chart points."""
from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class JointTriangleRepair:
    points: np.ndarray
    trials: int
    moved_nodes: tuple
    initial_penalty: float
    final_penalty: float
    budget_exhausted: bool
    selected_nodes: tuple = ()
    root_nodes: tuple = ()
    neighbour_nodes: tuple = ()
    priority_mode: str = 'node_id'


def _validate_pinned_nodes(nodes, count):
    try:
        values = tuple(nodes)
    except TypeError as error:
        raise ValueError("invalid pinned joint triangle repair nodes") from error
    if any(isinstance(node, (bool, np.bool_)) or not isinstance(node, Integral)
           or not 0 <= node < count for node in values):
        raise ValueError("invalid pinned joint triangle repair nodes")
    return frozenset(map(int, values))


def _physical_priority_key(points, cells, angles, lengths, pairs, min_angle, max_growth,
                           cancellation_check=None):
    """Reduce cached physical defects across incidence once, without owner calls."""
    if cancellation_check is not None:
        cancellation_check()
    if not np.isfinite(angles).all() or not np.isfinite(lengths).all():
        raise ValueError('nonfinite physical joint triangle priority')
    growth = np.ones(len(cells))
    if len(pairs):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = lengths[pairs[:, 0]] / lengths[pairs[:, 1]]
            ratios = np.maximum(ratios, 1. / ratios)
        if not np.isfinite(ratios).all():
            raise ValueError('nonfinite physical joint triangle priority')
        np.maximum.at(growth, pairs.ravel(), np.repeat(ratios, 2))
    minimum_angle = np.full(len(points), np.inf)
    maximum_growth = np.ones(len(points))
    np.minimum.at(minimum_angle, cells.ravel(), np.repeat(np.min(angles, axis=1), 3))
    np.maximum.at(maximum_growth, cells.ravel(), np.repeat(growth, 3))
    angle_deficit = np.maximum(0., (min_angle - minimum_angle) / min_angle)
    growth_deficit = np.maximum(0., maximum_growth / max_growth - 1.)
    severity = np.maximum(angle_deficit, growth_deficit)
    if not all(np.isfinite(values).all() for values in (growth, angle_deficit, growth_deficit, severity)):
        raise ValueError('nonfinite physical joint triangle priority')
    if cancellation_check is not None:
        cancellation_check()
    return lambda node: (-severity[node], -angle_deficit[node], -growth_deficit[node],
                         *points[node], node)


def repair_joint_triangle_quality(points, triangles, protected_edges, poor_triangle_ids, *,
                                  min_angle, max_growth, max_trials=2048,
                                  cancellation_check=None, evaluate_coordinates=None,
                                  neighbourhood_rings=0, coordinate_batch_size=1, pinned_nodes=(),
                                  physical_priority=False, candidate_callback=None):
    original = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    if (original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all()
            or cells.ndim != 2 or cells.shape[1] != 3 or np.any(cells < 0)
            or np.any(cells >= len(original)) or not 0 < min_angle < 60
            or not np.isfinite(max_growth) or max_growth <= 1
            or isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 0
            or type(neighbourhood_rings) is not int or neighbourhood_rings not in (0, 1)
            or type(coordinate_batch_size) is not int or not 1 <= coordinate_batch_size <= 8
            or type(physical_priority) is not bool
            or physical_priority and evaluate_coordinates is None
            or candidate_callback is not None and (not callable(candidate_callback)
                                                    or evaluate_coordinates is None)):
        raise ValueError("invalid joint triangle repair input")
    incidence = {}
    for i, row in enumerate(cells):
        for a, b in zip(row, np.roll(row, -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
    protected = {tuple(sorted(map(int, edge))) for edge in protected_edges}
    if not protected.issubset(incidence) or any(len(v) > 2 for v in incidence.values()):
        raise ValueError("invalid joint triangle repair incidence")
    fixed = {n for edge in protected for n in edge}
    fixed.update(n for edge, rows in incidence.items() if len(rows) == 1 for n in edge)
    fixed.update(_validate_pinned_nodes(pinned_nodes, len(original)))
    bad = sorted(set(map(int, poor_triangle_ids)))
    if any(i < 0 or i >= len(cells) for i in bad):
        raise ValueError("invalid joint triangle repair selection")
    pairs = np.asarray([v for v in incidence.values() if len(v) == 2], dtype=np.int64).reshape(-1, 2)
    # Search margins cannot demand more total angle than a vertex star owns.
    # In particular, three triangles at a right-angle corner cannot each be
    # pushed above 30 degrees in a planar chart. Use the physical star when an
    # owner evaluator is supplied, without changing the actual quality limit.
    budget_points = original
    if evaluate_coordinates is not None:
        budget_points = np.asarray(evaluate_coordinates(original), dtype=np.float64)
        if budget_points.shape != (len(original), 3) or not np.isfinite(budget_points).all():
            raise ValueError("invalid owner coordinates in joint triangle repair")
    budget_cells = budget_points[cells]
    forward = np.roll(budget_cells, -1, axis=1) - budget_cells
    backward = np.roll(budget_cells, 1, axis=1) - budget_cells
    if budget_points.shape[1] == 3:
        cross = np.linalg.norm(np.cross(forward, backward), axis=2)
    else:
        cross = np.abs(forward[:, :, 0] * backward[:, :, 1] - forward[:, :, 1] * backward[:, :, 0])
    angles = np.degrees(np.arctan2(cross, np.sum(forward * backward, axis=2)))
    priority = None
    if physical_priority:
        priority = _physical_priority_key(original, cells, angles,
            np.linalg.norm(forward, axis=2).mean(axis=1), pairs, min_angle, max_growth,
            cancellation_check)
    movable = sorted({int(n) for i in bad for n in cells[i]} - fixed, key=priority)[:12]
    root_nodes = tuple(movable)
    neighbour_nodes = ()
    if neighbourhood_rings and movable and len(movable) < 12:
        # Direct defect roots precede the ring; neighbours only fill unused slots.
        roots = set(movable)
        neighbouring = {int(n) for row in cells if any(int(v) in roots for v in row) for n in row}
        neighbour_nodes = tuple(sorted(neighbouring - fixed - roots, key=priority)[:12-len(movable)])
        movable += list(neighbour_nodes)
    selection = dict(selected_nodes=tuple(movable), root_nodes=root_nodes,
                     neighbour_nodes=neighbour_nodes,
                     priority_mode='physical_severity' if physical_priority else 'node_id')
    counts = np.bincount(cells.ravel(), minlength=len(original))
    totals = np.bincount(cells.ravel(), weights=angles.ravel(), minlength=len(original))
    average = np.divide(totals, counts, out=np.full(len(original), float(min_angle)), where=counts > 0)
    margins = np.minimum(min(.02, (60 - min_angle) * .5), np.maximum(0., average - min_angle) * .25)
    margins[margins < 64 * np.finfo(float).eps * max(1., min_angle)] = 0.
    target_angle = min_angle + margins[cells]
    target_growth = max_growth - min(.001, (max_growth - 1) * .01)

    def chart_is_valid(x):
        xy = x[cells]
        forward = np.roll(xy, -1, axis=1) - xy
        backward = np.roll(xy, 1, axis=1) - xy
        area = forward[:, 0, 0] * backward[:, 0, 1] - forward[:, 0, 1] * backward[:, 0, 0]
        return not np.any(area <= 0) and np.isfinite(x).all()

    def owner_coordinates(x):
        xyz = np.asarray(evaluate_coordinates(x), dtype=np.float64)
        if xyz.shape != (len(x), 3) or not np.isfinite(xyz).all():
            raise ValueError("invalid owner coordinates in joint triangle repair")
        return xyz

    def penalty(x, owner_xyz=None, trial_ordinal=None):
        xy = x[cells]
        forward = np.roll(xy, -1, axis=1) - xy
        backward = np.roll(xy, 1, axis=1) - xy
        area = forward[:, 0, 0] * backward[:, 0, 1] - forward[:, 0, 1] * backward[:, 0, 0]
        if np.any(area <= 0) or not np.isfinite(x).all():
            return float("inf")
        if evaluate_coordinates is not None:
            xyz = owner_coordinates(x) if owner_xyz is None else owner_xyz
            xy = xyz[cells]
            forward = np.roll(xy, -1, axis=1) - xy
            backward = np.roll(xy, 1, axis=1) - xy
            cross = np.linalg.norm(np.cross(forward, backward), axis=2)
        else:
            cross = np.abs(forward[:, :, 0] * backward[:, :, 1] - forward[:, :, 1] * backward[:, :, 0])
        angles = np.degrees(np.arctan2(cross, np.sum(forward * backward, axis=2)))
        angle_defect = np.maximum(0., (target_angle - angles) / target_angle)
        perimeter = np.linalg.norm(forward, axis=2).sum(axis=1)
        ratios = perimeter[pairs[:, 0]] / perimeter[pairs[:, 1]]
        growth_defect = np.maximum(0., np.maximum(ratios, 1 / ratios) / target_growth - 1)
        value = float(np.sum(angle_defect ** 2) + np.sum(growth_defect ** 2))
        if candidate_callback is not None and trial_ordinal is not None and np.isfinite(value):
            if cancellation_check is not None:
                cancellation_check()
            # Detached read-only receipts cannot modify the search or a later
            # row in the same owner batch, even if a callback changes flags.
            observed_points, observed_xyz = x.copy(), xyz.copy()
            observed_points.setflags(write=False)
            observed_xyz.setflags(write=False)
            candidate_callback(observed_points, observed_xyz, value, trial_ordinal)
            if cancellation_check is not None:
                cancellation_check()
        return value

    initial = penalty(original)
    if not np.isfinite(initial):
        raise ValueError("non-positive joint triangle repair input")
    if not movable or not max_trials or initial == 0:
        return JointTriangleRepair(original.copy(), 0, (), initial, initial, max_trials == 0, **selection)
    scale = float(np.median([np.linalg.norm(original[a] - original[b]) for a, b in incidence]))
    base = original[movable].copy()
    trials = 0
    best_points, best_penalty = original.copy(), initial

    def evaluate(values):
        nonlocal trials, best_points, best_penalty
        if trials >= max_trials:
            return None
        if cancellation_check is not None:
            cancellation_check()
        trials += 1
        x = original.copy()
        x[movable] = base + values.reshape(-1, 2) * scale
        value = penalty(x, trial_ordinal=trials)
        if value < best_penalty:
            best_points, best_penalty = x, value
        return value

    def gradient(values, value):
        nonlocal trials, best_points, best_penalty
        result = np.empty_like(values)
        step = min(1e-5, max(1e-8, np.sqrt(max(0., value)) * .01))
        probes = []
        # Only explicitly opted-in row-independent owner evaluators batch. Keep
        # at most eight detached candidates and preferably <=8192 flattened rows.
        chunk_size = (min(coordinate_batch_size, max(1, 8192 // len(original)))
                      if evaluate_coordinates is not None else 1)
        while len(probes) < 2 * len(values):
            count = min(chunk_size, 2 * len(values) - len(probes), max_trials - trials)
            if not count:
                # An unmatched plus probe has already updated the best point.
                return None
            if count == 1:
                delta = np.zeros_like(values)
                delta[len(probes) // 2] = step
                probe = values + delta if len(probes) % 2 == 0 else values - delta
                probes.append(evaluate(probe))
                continue
            candidates = []
            valid = []
            chunk_start = trials
            # Cancellation is checked before every charged trial. Owner work
            # follows admission of this bounded chunk, so exception interleaving
            # and cancellation latency can differ from the serial callback path.
            for index in range(count):
                if cancellation_check is not None:
                    cancellation_check()
                trials += 1
                delta = np.zeros_like(values)
                ordinal = len(probes) + index
                delta[ordinal // 2] = step
                probe = values + delta if ordinal % 2 == 0 else values - delta
                x = original.copy()
                x[movable] = base + probe.reshape(-1, 2) * scale
                candidates.append(x)
                if chart_is_valid(x):
                    valid.append(index)
            coordinates = {}
            if valid:
                rows = np.concatenate([candidates[index] for index in valid], axis=0)
                xyz = owner_coordinates(rows).reshape(len(valid), len(original), 3)
                coordinates = dict(zip(valid, xyz))
            for index, x in enumerate(candidates):
                candidate = (penalty(x, coordinates[index], chunk_start + index + 1) if index in coordinates
                             else float("inf"))
                if candidate < best_penalty:
                    best_points, best_penalty = x, candidate
                probes.append(candidate)
        for j in range(len(values)):
            plus, minus = probes[2 * j:2 * j + 2]
            if np.isfinite(plus) and np.isfinite(minus):
                result[j] = (plus - minus) / (2 * step)
            elif np.isfinite(plus):
                result[j] = (plus - value) / step
            elif np.isfinite(minus):
                result[j] = (value - minus) / step
            else:
                result[j] = 0.
        return result

    values = np.zeros(2 * len(movable))
    value = initial
    identity = np.eye(len(values))
    inverse = identity * 100.
    g = gradient(values, value)
    for _ in range(48):
        if g is None or best_penalty == 0. or trials >= max_trials:
            break
        direction = -inverse @ g
        if float(direction @ g) >= 0:
            inverse = identity * 100.
            direction = -inverse @ g
        length = float(np.max(np.linalg.norm(direction.reshape(-1, 2), axis=1)))
        if length == 0 or not np.isfinite(length):
            break
        if length > .2:
            direction *= .2 / length
        accepted = False
        for power in range(12):
            delta = direction * (0.5 ** power)
            candidate = evaluate(values + delta)
            if candidate is None:
                break
            if candidate <= value + 1e-4 * float(g @ delta):
                accepted = True
                break
        if not accepted:
            break
        next_values = values + delta
        next_gradient = gradient(next_values, candidate)
        if next_gradient is None:
            break
        y = next_gradient - g
        curvature = float(y @ delta)
        if curvature > 1e-12 * float(np.linalg.norm(y) * np.linalg.norm(delta)):
            transform = identity - np.outer(delta, y) / curvature
            inverse = transform @ inverse @ transform.T + np.outer(delta, delta) / curvature
        else:
            inverse = identity * 100.
        values, value, g = next_values, candidate, next_gradient
    moved = tuple(n for n in movable if best_points[n].tobytes() != original[n].tobytes())
    if best_points[sorted(fixed)].tobytes() != original[sorted(fixed)].tobytes():
        raise ValueError("joint triangle repair changed fixed nodes")
    return JointTriangleRepair(best_points, trials, moved, initial, best_penalty, trials >= max_trials, **selection)
