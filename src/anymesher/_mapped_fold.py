"""Fold detection for transfinite (Coons) mapped grids.

A mapped face is meshed by blending its four declared sides.  That blend is a
valid mesh only when it is injective: when a declared "side" bends back on
itself (for example an L-shaped plate whose re-entrant corner lies inside one
mapped side), the blended grid folds over and publishes elements outside the
face.  These helpers detect that from the grid itself, so the mapped route can
refuse the face and the automatic strategy can send it to native meshing.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from anygeometry.model import GeometryModel

__all__ = ["grid_folds", "mapped_face_folds"]

# Relative tolerance below which a corner Jacobian counts as degenerate-zero
# rather than inverted; strictly negative values beyond it are folds.
_FOLD_TOLERANCE = 1.0e-12
# Stations per mapped side used by the pre-check before seeding is known.
_CHECK_STATIONS = 24


def grid_folds(grid: np.ndarray) -> bool:
    """True when a ``(nu+1, nv+1, 3)`` station grid inverts or folds.

    Each cell's four corner cross products must agree in orientation with the
    cell's diagonal normal, and neighbouring cells must agree with each other.
    Both tests are orientation-only, so curved mapped faces are accepted.
    """
    points = np.asarray(grid, dtype=float)
    if points.ndim != 3 or points.shape[0] < 2 or points.shape[1] < 2:
        return False
    if points.shape[2] == 2:
        points = np.concatenate((points, np.zeros(points.shape[:2] + (1,))), axis=2)
    p00, p10 = points[:-1, :-1], points[1:, :-1]
    p11, p01 = points[1:, 1:], points[:-1, 1:]
    normal = np.cross(p11 - p00, p01 - p10)
    normal_length = np.linalg.norm(normal, axis=-1)
    corners = (
        np.cross(p10 - p00, p01 - p00),
        np.cross(p11 - p10, p00 - p10),
        np.cross(p01 - p11, p10 - p11),
        np.cross(p00 - p01, p11 - p01),
    )
    for corner in corners:
        dot = np.einsum("ijk,ijk->ij", corner, normal)
        scale = np.linalg.norm(corner, axis=-1) * normal_length
        if np.any(dot < -_FOLD_TOLERANCE * scale):
            return True
    for first, second in ((normal[1:, :], normal[:-1, :]), (normal[:, 1:], normal[:, :-1])):
        if first.size == 0:
            continue
        dot = np.einsum("ijk,ijk->ij", first, second)
        scale = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        if np.any(dot < -_FOLD_TOLERANCE * scale):
            return True
    return False


def _side_polyline(geometry: GeometryModel, side, samples_per_length: float) -> np.ndarray:
    pieces = []
    for use in side:
        length = float(geometry.edge_length(use.edge))
        count = max(4, int(ceil(length * samples_per_length)))
        parameters = np.linspace(0.0, 1.0, count + 1)
        points = np.asarray(geometry.sample_edge(use.edge, parameters), dtype=float)
        if not use.forward:
            points = points[::-1]
        pieces.append(points if not pieces else points[1:])
    return np.vstack(pieces)


def _resample(polyline: np.ndarray, stations: int) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, cumulative[-1], stations + 1)
    return np.column_stack([
        np.interp(targets, cumulative, polyline[:, axis]) for axis in range(polyline.shape[1])
    ])


def mapped_face_folds(geometry: GeometryModel, face_id: int, stations: int = _CHECK_STATIONS) -> bool:
    """Pre-check whether a four-sided face's transfinite map folds.

    Samples the declared sides by arc length (the mapped route's station
    distribution under a uniform size field) and tests the blended grid.  A
    face without exactly four declared corners, or with holes, is not a mapped
    candidate and reports ``False`` here.
    """
    from .mapped import coons_grid

    face = geometry.faces[int(face_id)]
    if len(face.corners) != 4 or face.holes:
        return False
    sides = face.sides()
    perimeter = sum(float(geometry.edge_length(use.edge)) for side in sides for use in side)
    if not perimeter > 0.0:
        return False
    density = 16.0 * 4.0 / perimeter
    polylines = [_side_polyline(geometry, side, density) for side in sides]
    side_a = _resample(polylines[0], stations)
    side_b = _resample(polylines[1], stations)
    # Sides C and D run against the parameter directions, as in _build_face.
    side_c = _resample(polylines[2], stations)[::-1]
    side_d = _resample(polylines[3], stations)[::-1]
    return grid_folds(coons_grid(side_a, side_b, side_c, side_d))
