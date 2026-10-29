"""Opt-in station-conforming tensor core for one untrimmed four-corner owner face.

The boundary rows already registered by the owner-bound route are the only
authority: they are never moved, resampled or removed. The interior is the
exact affine tensor of two opposite boundary chains in the face chart, so every
cell is a chart parallelogram whose corners are lifted later through the same
owner chart as any native interior row. Anything else is refused with
``ParametricTensorUnsupported`` and the caller keeps the native route.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import MeshCore
from .errors import MeshError
from ._row_provenance import OutputRowProvenance


class ParametricTensorUnsupported(MeshError):
    """The face is outside the narrow tensor contract; use the native route."""


_BOX_TOLERANCE = 1.0e-9
_CELL_BUDGET = 250_000
_SAMPLES = 129


@dataclass(frozen=True)
class TensorAdmission:
    """Geometry-level verdict; ``code`` is ``'eligible'`` or a stable refusal reason code."""
    eligible: bool
    code: str
    detail: str = ''


def _active_controls(pcurve):
    """Controls whose basis support meets the active domain of a NURBS pcurve (2-D), or None."""
    try:
        degree = int(pcurve.degree)
        knots = np.asarray(pcurve.knots, dtype=float)
        controls = np.asarray(pcurve.controls, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return None
    if controls.ndim != 2 or controls.shape[1] != 2 or len(knots) != len(controls) + degree + 1:
        return None
    domain = getattr(pcurve, 'domain', None)
    low, high = (knots[degree], knots[-degree - 1]) if domain is None else sorted(float(v) for v in domain)
    index = np.arange(len(controls))
    keep = (knots[index] < high) & (knots[index + degree + 1] > low)
    if not keep.any():
        return None
    weights = getattr(pcurve, 'weights', None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (len(controls),) or not np.isfinite(weights).all():
            return 'weighted'
        # Scale invariant: only the RELATIVE spread of the weights matters (a common factor is no weighting). A
        # sub-domain curve is refused when ANY weight differs, active or not: the restriction of a rational curve to
        # a sub-interval is not reparametrisation-safe for the station balancing.
        subdomain = domain is not None and (low > knots[degree] or high < knots[-degree - 1])
        used = weights if subdomain else weights[keep]
        if np.ptp(used) > 1.0e-12 * float(np.max(np.abs(used))):
            return 'weighted'
    return controls[keep]


def nurbs_tensor_admission(geometry, face_id) -> TensorAdmission:
    """Pure, read-only check that one face can ever feed the tensor core.

    Admits an untrimmed four-edge NURBS face whose sides are iso-parametric UV-box sides with no collapsed side.
    It proves nothing about stations, size field, junctions or quality; those are decided later by
    ``parametric_tensor_core`` and the final gates, each of which refuses with a typed error.
    """
    def no(code, detail=''):
        return TensorAdmission(False, code, detail)
    from anygeometry.nurbs import NurbsSurface
    face = geometry.faces.get(face_id) if hasattr(geometry.faces, 'get') else None
    if face is None:
        return no('no_such_face', f'face {face_id}')
    surface = face.surface if isinstance(face.surface, NurbsSurface) else face.parameterization
    if not isinstance(surface, NurbsSurface):
        return no('not_nurbs_surface')
    if len(face.holes):
        return no('holes')
    if len(face.loop) != 4:
        return no('not_four_edges', f'{len(face.loop)} edges')
    t = np.linspace(0., 1., _SAMPLES)
    sides = []
    for use in face.loop:
        try:
            uv = geometry.face_boundary_uv(face_id, use.edge, t)
        except Exception as error:
            return no('no_boundary_uv', str(error))
        if uv is None:
            return no('no_boundary_uv', f'edge {use.edge}')
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (_SAMPLES, 2) or not np.isfinite(uv).all():
            return no('no_boundary_uv', f'edge {use.edge}')
        # Exact: the local convex-hull property bounds the ACTIVE part of the pcurve by the controls whose basis
        # support meets the active parameter interval. A constant coordinate there proves the side iso-parametric
        # between samples; no usable proof (no NURBS pcurve) refuses.
        pcurve = getattr(getattr(geometry.edges[use.edge], 'curve', None), 'pcurve', None)
        active = _active_controls(pcurve)
        if isinstance(active, str):  # non-uniform rational weights re-parametrise the side along its iso line
            return no('weighted_pcurve', f'edge {use.edge} pcurve has non-uniform weights')
        if active is None:
            return no('unverified_pcurve', f'edge {use.edge} has no NURBS pcurve to bound')
        if not (np.ptp(active[:, 0]) < _BOX_TOLERANCE or np.ptp(active[:, 1]) < _BOX_TOLERANCE):
            return no('not_iso_parametric_rectangle', f'edge {use.edge} pcurve is not on one iso line')
        if np.ptp(uv[:, 0]) < _BOX_TOLERANCE:
            sides.append(('u', float(uv[0, 0])))
        elif np.ptp(uv[:, 1]) < _BOX_TOLERANCE:
            sides.append(('v', float(uv[0, 1])))
        else:
            return no('not_iso_parametric_rectangle', f'edge {use.edge} is not an iso-parameter side')
        ends = np.asarray([surface.evaluate(*uv[0]), surface.evaluate(*uv[-1])], dtype=float)
        if np.linalg.norm(ends[0] - ends[1]) == 0.0 and np.ptp(uv[:, 0] if sides[-1][0] == 'v' else uv[:, 1]) > 0.0:
            return no('degenerate_edge', f'edge {use.edge} collapses to a point')
    constant_u = sorted(value for kind, value in sides if kind == 'u')
    constant_v = sorted(value for kind, value in sides if kind == 'v')
    unit = np.array([0., 1.])
    if (len(constant_u) != 2 or len(constant_v) != 2
            or np.max(np.abs(np.asarray(constant_u) - unit)) > _BOX_TOLERANCE
            or np.max(np.abs(np.asarray(constant_v) - unit)) > _BOX_TOLERANCE):
        return no('not_iso_parametric_rectangle', 'sides are not the four sides of the unit UV box')
    return TensorAdmission(True, 'eligible')


def parametric_tensor_core(outer, holes, constraints, pinned, chart_inverse, protected_rows):
    """Return ``(core, provenance)`` or raise ``ParametricTensorUnsupported``."""
    outer = np.asarray(outer, dtype=float)
    if holes:
        raise ParametricTensorUnsupported("tensor route requires a hole-free face")
    if constraints and any(len(item) for item in constraints):
        raise ParametricTensorUnsupported("tensor route refuses interior constraints")
    if pinned is not None and len(pinned):
        raise ParametricTensorUnsupported("tensor route refuses pinned interior points")
    if outer.ndim != 2 or outer.shape[1] < 2 or len(outer) < 8 or not np.isfinite(outer).all():
        raise ParametricTensorUnsupported("tensor route needs a finite closed chart loop")
    count = len(outer)
    rows = {int(row): int(node) for row, node in protected_rows.items()}
    if set(rows) != set(range(count)):
        raise ParametricTensorUnsupported("tensor route needs one protected source per boundary row")
    inverse = np.asarray(chart_inverse, dtype=float)
    uv = outer[:, :2] @ inverse
    if not np.isfinite(uv).all():
        raise ParametricTensorUnsupported("chart inverse is not finite")
    on_box = ((np.abs(uv) < _BOX_TOLERANCE) | (np.abs(uv - 1.0) < _BOX_TOLERANCE)
              | ((uv > -_BOX_TOLERANCE) & (uv < 1.0 + _BOX_TOLERANCE)))
    near = lambda value, target: np.abs(value - target) < _BOX_TOLERANCE
    on_edge = near(uv[:, 0], 0.) | near(uv[:, 0], 1.) | near(uv[:, 1], 0.) | near(uv[:, 1], 1.)
    if not on_box.all() or not on_edge.all():
        raise ParametricTensorUnsupported("boundary is not the owner UV box (trimmed or reparametrized face)")
    corner = ((near(uv[:, 0], 0.) | near(uv[:, 0], 1.)) & (near(uv[:, 1], 0.) | near(uv[:, 1], 1.)))
    corner_rows = np.flatnonzero(corner)
    if len(corner_rows) != 4 or len({tuple(np.round(uv[row]).astype(int)) for row in corner_rows}) != 4:
        raise ParametricTensorUnsupported("boundary has no four distinct UV-box corners (pole or degenerate edge)")
    start = int(corner_rows[0])
    order = [(start + k) % count for k in range(count)]
    corners = [k for k, row in enumerate(order) if corner[row]]
    sides = [order[corners[s]:corners[s + 1] + 1] for s in range(3)]
    sides.append(order[corners[3]:] + [order[0]])
    if any(len(side) < 3 for side in sides):
        raise ParametricTensorUnsupported("a boundary side has no interior station")
    if len(sides[0]) != len(sides[2]) or len(sides[1]) != len(sides[3]):
        raise ParametricTensorUnsupported("opposite boundary sides have unequal station counts")
    # A side must stay on one UV-box edge, otherwise corners were mis-detected.
    for side in sides:
        chain = uv[side]
        if not (np.all(near(chain[:, 0], chain[0, 0])) or np.all(near(chain[:, 1], chain[0, 1]))):
            raise ParametricTensorUnsupported("a boundary side leaves its UV-box edge")
    nx, ny = len(sides[0]) - 1, len(sides[1]) - 1
    if (nx + 1) * (ny + 1) > _CELL_BUDGET:
        raise ParametricTensorUnsupported("tensor exceeds the fine candidate budget")
    bottom = outer[sides[0], :2]
    right = outer[sides[1], :2]
    top = outer[sides[2][::-1], :2]
    left = outer[sides[3][::-1], :2]
    origin = bottom[0]
    scale = max(float(np.max(np.abs(outer[:, :2]))), 1.0)
    tolerance = 64 * np.finfo(float).eps * scale
    offset_left = left - origin
    offset_bottom = bottom - origin
    if (np.max(np.abs(top - (bottom + (left[-1] - origin)))) > tolerance
            or np.max(np.abs(right - (bottom[-1] + offset_left))) > tolerance):
        raise ParametricTensorUnsupported("opposite sides are not affine translates (non-uniform stations)")
    steps_x = np.diff(bottom, axis=0)
    steps_y = np.diff(left, axis=0)
    cross = steps_x[:, None, 0] * steps_y[None, :, 1] - steps_x[:, None, 1] * steps_y[None, :, 0]
    if not np.all(np.isfinite(cross)) or not np.all(cross > 0.0):
        raise ParametricTensorUnsupported("tensor cells are not positively oriented")
    # Rows: boundary rows keep their input order; interior rows follow.
    grid = -np.ones((nx + 1, ny + 1), dtype=np.int64)
    grid[:, 0] = sides[0]
    grid[nx, :] = sides[1]
    grid[:, ny] = sides[2][::-1]
    grid[0, :] = sides[3][::-1]
    interior = grid < 0
    grid[interior] = count + np.arange(int(interior.sum()))
    points = np.zeros((int(grid.max()) + 1, 3), dtype=float)
    points[:count, :outer.shape[1]] = outer
    full = offset_bottom[:, None, :] + offset_left[None, :, :] + origin
    points[grid[interior], :2] = full[interior]
    cells = np.stack((grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]), axis=-1).reshape(-1, 4)
    core = MeshCore(points, np.empty((0, 3), dtype=np.int64), cells)
    pairs = tuple(sorted((rows[row], row) for row in range(count)))
    by_row = [None] * core.num_nodes
    for node, row in pairs:
        by_row[row] = node
    record = OutputRowProvenance(tuple(by_row), pairs)
    record.validate(core)
    core._output_row_provenance = record
    return core, record
