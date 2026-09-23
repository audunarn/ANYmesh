"""Bounded coupled angle/perimeter repair, used only on detached chart points."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class JointTriangleRepair:
    points: np.ndarray
    trials: int
    moved_nodes: tuple
    initial_penalty: float
    final_penalty: float
    budget_exhausted: bool


def repair_joint_triangle_quality(points, triangles, protected_edges, poor_triangle_ids, *,
                                  min_angle, max_growth, max_trials=2048,
                                  cancellation_check=None, evaluate_coordinates=None,
                                  neighbourhood_rings=0):
    original = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    if (original.ndim != 2 or original.shape[1] != 2 or not np.isfinite(original).all()
            or cells.ndim != 2 or cells.shape[1] != 3 or np.any(cells < 0)
            or np.any(cells >= len(original)) or not 0 < min_angle < 60
            or not np.isfinite(max_growth) or max_growth <= 1
            or isinstance(max_trials, bool) or not isinstance(max_trials, int) or max_trials < 0
            or type(neighbourhood_rings) is not int or neighbourhood_rings not in (0, 1)):
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
    bad = sorted(set(map(int, poor_triangle_ids)))
    if any(i < 0 or i >= len(cells) for i in bad):
        raise ValueError("invalid joint triangle repair selection")
    movable = sorted({int(n) for i in bad for n in cells[i]} - fixed)[:12]
    if neighbourhood_rings and movable:
        # An angle-only boundary defect can require motion in the adjacent
        # interior star, even though those adjacent cells currently pass. Keep
        # the original defect nodes first and cap the coupled search at 12.
        roots = set(movable)
        neighbouring = {int(n) for row in cells if any(int(v) in roots for v in row) for n in row}
        movable = (movable + sorted(neighbouring - fixed - roots))[:12]
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
    counts = np.bincount(cells.ravel(), minlength=len(original))
    totals = np.bincount(cells.ravel(), weights=angles.ravel(), minlength=len(original))
    average = np.divide(totals, counts, out=np.full(len(original), float(min_angle)), where=counts > 0)
    margins = np.minimum(min(.02, (60 - min_angle) * .5), np.maximum(0., average - min_angle) * .25)
    margins[margins < 64 * np.finfo(float).eps * max(1., min_angle)] = 0.
    target_angle = min_angle + margins[cells]
    target_growth = max_growth - min(.001, (max_growth - 1) * .01)

    def penalty(x):
        xy = x[cells]
        forward = np.roll(xy, -1, axis=1) - xy
        backward = np.roll(xy, 1, axis=1) - xy
        area = forward[:, 0, 0] * backward[:, 0, 1] - forward[:, 0, 1] * backward[:, 0, 0]
        if np.any(area <= 0) or not np.isfinite(x).all():
            return float("inf")
        if evaluate_coordinates is not None:
            xyz = np.asarray(evaluate_coordinates(x), dtype=np.float64)
            if xyz.shape != (len(x), 3) or not np.isfinite(xyz).all():
                raise ValueError("invalid owner coordinates in joint triangle repair")
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
        return float(np.sum(angle_defect ** 2) + np.sum(growth_defect ** 2))

    initial = penalty(original)
    if not np.isfinite(initial):
        raise ValueError("non-positive joint triangle repair input")
    if not movable or not max_trials or initial == 0:
        return JointTriangleRepair(original.copy(), 0, (), initial, initial, max_trials == 0)
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
        value = penalty(x)
        if value < best_penalty:
            best_points, best_penalty = x, value
        return value

    def gradient(values, value):
        result = np.empty_like(values)
        step = min(1e-5, max(1e-8, np.sqrt(max(0., value)) * .01))
        for j in range(len(values)):
            delta = np.zeros_like(values)
            delta[j] = step
            plus, minus = evaluate(values + delta), evaluate(values - delta)
            if plus is None or minus is None:
                return None
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
    return JointTriangleRepair(best_points, trials, moved, initial, best_penalty, trials >= max_trials)
