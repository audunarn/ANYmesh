"""Bounded convex projection for a triangle star's minimum-angle constraints.

This generates a candidate only. Callers must retain their physical-quality,
growth, topology, protected-node and publication checks.
"""
import math
import numpy as np


def angle_star_target(points, triangles, node, minimum_angle, *, max_sweeps=64,
                      consume_work=None, cancellation_check=None):
    x = np.asarray(points, dtype=np.float64)
    cells = np.asarray(triangles, dtype=np.int64)
    if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
            or cells.ndim != 2 or cells.shape[1] != 3
            or np.any(cells < 0) or np.any(cells >= len(x))
            or isinstance(node, bool) or int(node) != node or not 0 <= node < len(x)
            or not math.isfinite(minimum_angle) or not 0 < minimum_angle < 60
            or isinstance(max_sweeps, bool) or not isinstance(max_sweeps, int)
            or max_sweeps < 0):
        raise ValueError("invalid triangle-star projection input")
    node = int(node)
    rows = cells[np.any(cells == node, axis=1)]
    if not len(rows) or not max_sweeps:
        return None
    theta = math.radians(minimum_angle)
    tangent, sine = math.tan(theta), math.sin(theta)
    origin = x[node].copy()
    constraints = []
    scale = 0.0
    for row in rows:
        positions = np.flatnonzero(row == node)
        if len(positions) != 1:
            raise ValueError("repeated node in triangle star")
        i = int(positions[0])
        a = x[row[(i + 1) % 3]] - origin
        b = x[row[(i + 2) % 3]] - origin
        vector = b - a
        length = float(np.linalg.norm(vector))
        if length == 0:
            raise ValueError("collapsed opposite edge in triangle star")
        direction = vector / length
        normal = np.array([-direction[1], direction[0]])
        if float(np.dot(-a, normal)) <= 0:
            raise ValueError("non-positive input triangle star")
        scale = max(scale, length)
        # Endpoint angles impose two half-planes on the moving vertex.
        for n, offset_point in ((normal - tangent * direction, a),
                                (normal + tangent * direction, b)):
            n = n / np.linalg.norm(n)
            constraints.append((0, n, float(np.dot(n, offset_point))))
        # The angle at the moving vertex imposes the interior of this disk.
        center = (a + b) * .5 + normal * (length / (2 * tangent))
        constraints.append((1, center, length / (2 * sine)))
    value = np.zeros(2)
    corrections = np.zeros((len(constraints), 2))
    tolerance = 64 * np.finfo(np.float64).eps * scale

    def project(point, constraint):
        kind, vector, bound = constraint
        if kind == 0:
            return point + max(0.0, bound - float(np.dot(vector, point))) * vector
        delta = point - vector
        distance = float(np.linalg.norm(delta))
        return vector + delta * (bound / distance) if distance > bound else point

    for _ in range(max_sweeps):
        if cancellation_check is not None:
            cancellation_check()
        previous = value.copy()
        for i, constraint in enumerate(constraints):
            if consume_work is not None and not consume_work():
                return None
            proposed = value + corrections[i]
            updated = project(proposed, constraint)
            corrections[i] = proposed - updated
            value = updated
        if float(np.linalg.norm(value - previous)) <= tolerance:
            feasible = True
            for constraint in constraints:
                if consume_work is not None and not consume_work():
                    return None
                if float(np.linalg.norm(project(value, constraint) - value)) > tolerance:
                    feasible = False
                    break
            if feasible:
                result = origin + value
                return result if np.isfinite(result).all() else None
    return None
