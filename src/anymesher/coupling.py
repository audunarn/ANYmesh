"""Locating a beam node inside the plating under it.

When a stiffener crosses a panel mesh rather than following its edges, the beam
node lands somewhere inside a shell element.  This module finds that element and
evaluates its shape functions at the projected point, which is what lets the
coupling be exact without the mesh being aligned to the stiffeners.

The older alternative was to require every beam node to sit on a shell node row
or column.  That works until the division counts change, and then it silently
does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "StructuredShellGrid",
    "build_structured_shell_grid",
    "locate_shell_element_at_xy",
    "shape_functions_4node",
    "shape_functions_8node",
]

Coordinates = Mapping[int, Sequence[float]]


def shape_functions_4node(xi: float, eta: float) -> np.ndarray:
    """Bilinear Q4 shape functions at a natural coordinate."""

    return np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=float,
    )


def shape_functions_8node(xi: float, eta: float) -> np.ndarray:
    """Serendipity Q8 shape functions at a natural coordinate."""

    return np.array(
        [
            -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
            -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
            -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
            -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
            0.5 * (1.0 - xi**2) * (1.0 - eta),
            0.5 * (1.0 + xi) * (1.0 - eta**2),
            0.5 * (1.0 - xi**2) * (1.0 + eta),
            0.5 * (1.0 - xi) * (1.0 - eta**2),
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class StructuredShellGrid:
    """A reusable axis-aligned cell index for coupling-point lookup.

    Built once per mesh.  Without it, locating hundreds of beam nodes means
    hundreds of linear scans over every shell element.
    """

    x_edges: np.ndarray
    y_edges: np.ndarray
    cells: Dict[Tuple[int, int], Tuple[List[int], float]]


def build_structured_shell_grid(
    shell_nodes: Coordinates,
    shell_elements: Mapping[int, Tuple[Sequence[int], float]],
    tolerance: float,
) -> Optional[StructuredShellGrid]:
    """Index axis-aligned shell cells, or return ``None`` for an irregular mesh.

    Returning ``None`` rather than raising is deliberate: an irregular mesh is
    perfectly valid, it just cannot use the fast path, and the caller falls back
    to a sequential search.
    """

    tol = max(float(tolerance), 1.0e-10)
    records: List[Tuple[float, float, float, float, List[int], float]] = []
    x_edges: set[float] = set()
    y_edges: set[float] = set()
    try:
        for node_ids, thickness in shell_elements.values():
            corner_coords = np.asarray([shell_nodes[node_id] for node_id in node_ids[:4]], dtype=float)
            xmin = float(np.min(corner_coords[:, 0]))
            xmax = float(np.max(corner_coords[:, 0]))
            ymin = float(np.min(corner_coords[:, 1]))
            ymax = float(np.max(corner_coords[:, 1]))
            if xmax - xmin <= tol or ymax - ymin <= tol:
                return None
            # The fast index is intentionally limited to axis-aligned cells.
            for x_value, y_value in corner_coords[:, :2]:
                if min(abs(float(x_value) - xmin), abs(float(x_value) - xmax)) > tol:
                    return None
                if min(abs(float(y_value) - ymin), abs(float(y_value) - ymax)) > tol:
                    return None
            qxmin = round(xmin / tol) * tol
            qxmax = round(xmax / tol) * tol
            qymin = round(ymin / tol) * tol
            qymax = round(ymax / tol) * tol
            x_edges.update((qxmin, qxmax))
            y_edges.update((qymin, qymax))
            records.append((qxmin, qxmax, qymin, qymax, list(node_ids), float(thickness)))
    except (KeyError, TypeError, ValueError):
        return None

    xs = np.asarray(sorted(x_edges), dtype=float)
    ys = np.asarray(sorted(y_edges), dtype=float)
    nx = int(xs.size - 1)
    ny = int(ys.size - 1)
    if nx <= 0 or ny <= 0 or nx * ny != len(records):
        return None

    x_lookup = {float(value): index for index, value in enumerate(xs[:-1])}
    y_lookup = {float(value): index for index, value in enumerate(ys[:-1])}
    cells: Dict[Tuple[int, int], Tuple[List[int], float]] = {}
    for xmin, xmax, ymin, ymax, node_ids, thickness in records:
        i = x_lookup.get(float(xmin))
        j = y_lookup.get(float(ymin))
        if i is None or j is None:
            return None
        if abs(float(xs[i + 1]) - xmax) > tol or abs(float(ys[j + 1]) - ymax) > tol:
            return None
        if (i, j) in cells:
            return None
        cells[(i, j)] = (node_ids, thickness)
    if len(cells) != nx * ny:
        return None
    return StructuredShellGrid(xs, ys, cells)


def _interpolate_shell_point(
    x: float,
    y: float,
    node_ids: Sequence[int],
    shell_nodes: Coordinates,
    tolerance: float,
) -> Optional[Tuple[List[int], np.ndarray, np.ndarray]]:
    """Return interpolation weights and point for one axis-aligned shell cell."""

    tol = max(float(tolerance), 1.0e-10)
    corner_coords = np.asarray([shell_nodes[node_id] for node_id in node_ids[:4]], dtype=float)
    xmin, xmax = float(np.min(corner_coords[:, 0])), float(np.max(corner_coords[:, 0]))
    ymin, ymax = float(np.min(corner_coords[:, 1])), float(np.max(corner_coords[:, 1]))
    if x < xmin - tol or x > xmax + tol or y < ymin - tol or y > ymax + tol:
        return None
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= tol or dy <= tol:
        return None
    xi = float(np.clip(2.0 * (x - xmin) / dx - 1.0, -1.0, 1.0))
    eta = float(np.clip(2.0 * (y - ymin) / dy - 1.0, -1.0, 1.0))
    weights = shape_functions_8node(xi, eta) if len(node_ids) == 8 else shape_functions_4node(xi, eta)
    shell_coords = np.asarray([shell_nodes[node_id] for node_id in node_ids], dtype=float)
    return list(node_ids), weights, weights @ shell_coords


def locate_shell_element_at_xy(
    x: float,
    y: float,
    shell_nodes: Coordinates,
    shell_elements: Mapping[int, Tuple[Sequence[int], float]],
    tolerance: float,
    grid: Optional[StructuredShellGrid] = None,
) -> Optional[Tuple[List[int], np.ndarray, np.ndarray]]:
    """Find the shell element containing an x/y point.

    Returns its node IDs, the shape weights at the point, and the interpolated
    position on the shell, or ``None`` when the point lies outside every element.
    """

    tol = max(float(tolerance), 1.0e-10)

    # Built on demand for direct callers; mesh generation passes a shared index
    # so hundreds of beam nodes do not rebuild the same grid.
    index = grid or build_structured_shell_grid(shell_nodes, shell_elements, tol)
    if index is not None:
        i = int(np.searchsorted(index.x_edges, x) - 1)
        j = int(np.searchsorted(index.y_edges, y) - 1)
        i = max(0, min(i, len(index.x_edges) - 2))
        j = max(0, min(j, len(index.y_edges) - 2))
        candidate = index.cells.get((i, j))
        if candidate is not None:
            located = _interpolate_shell_point(x, y, candidate[0], shell_nodes, tol)
            if located is not None:
                return located

    for node_ids, _thickness in shell_elements.values():
        located = _interpolate_shell_point(x, y, list(node_ids), shell_nodes, tol)
        if located is not None:
            return located
    return None
