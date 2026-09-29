"""Scan-free quad-first/CDT paths must match the full scans they replace.

Each test keeps the replaced loop verbatim as an oracle:

* ``QuadMeshState.cells_at`` (and the ``View`` variant) derives incident
  cells from the node/edge maps instead of scanning every cell;
* ``seed._strict_inside_mask`` evaluates ``_strict_inside`` for many points;
* ``seed._uniform_lattice_points`` uses that mask;
* ``constrained_planar_triangulation`` skips segment recovery for segments that
  already are edges;
* ``_bowyer_watson`` locates cavity candidates through a uniform grid.  Larger
  point sets than ``test_triangulation_prefilter`` exercise many grid cells.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from anymesher.quad import seed
from anymesher.quad.state import QuadMeshState
from anymesher.triangulation import (
    _bowyer_watson,
    _finish_triangles,
    _normal_edge,
    _prepare_pslg,
    _recover_segment,
    constrained_planar_triangulation,
)

from test_triangulation_prefilter import _outcome, _reference_bowyer_watson


# --- QuadMeshState.cells_at ---------------------------------------------------

def _brute_cells_at(cells, node):
    return tuple(sorted(c for c, body in cells.items() if node in body))


def _grid_state(size: int) -> tuple[QuadMeshState, dict[tuple[int, int], tuple[int, int]]]:
    def nid(i: int, j: int) -> int:
        return j * (size + 1) + i

    nodes = {nid(i, j): (float(i), float(j)) for j in range(size + 1) for i in range(size + 1)}
    cells = {}
    squares = {}
    for j in range(size):
        for i in range(size):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            first, second = len(cells), len(cells) + 1
            cells[first] = (a, b, c)
            cells[second] = (a, c, d)
            squares[(i, j)] = (first, second)
    return QuadMeshState(nodes=nodes, cells=cells), squares


def _assert_cells_at(state) -> None:
    for node in state.nodes:
        assert state.cells_at(node) == _brute_cells_at(state.cells, node)


def test_cells_at_matches_full_scan_through_commits() -> None:
    rng = random.Random(20260929)
    state, squares = _grid_state(6)
    _assert_cells_at(state)
    merged: dict[tuple[int, int], int] = {}
    for step in range(60):
        key = rng.choice(sorted(squares))
        tx = state.transaction()
        if key in merged:
            # Split a Q4 back into two T3.
            quad = merged.pop(key)
            a, b, c, d = state.cell(quad)
            tx.remove_cell(quad)
            squares[key] = (tx.allocate_cell((a, b, c), "T3"), tx.allocate_cell((a, c, d), "T3"))
        else:
            first, second = squares[key]
            a, b, c = state.cell(first)
            d = state.cell(second)[2]
            tx.remove_cell(first)
            tx.remove_cell(second)
            merged[key] = tx.allocate_cell((a, b, c, d), "Q4")
        if step % 7 == 0:
            # A dangling node with a fan cell, removed again later.
            corner = rng.choice(sorted(state.nodes))
            x, y = state.position(corner)
            extra = tx.allocate_node((x + 0.25, y - 7.0))
            tx.allocate_cell((corner, extra, rng.choice(sorted(state.nodes))), "T3")
        view = tx.view
        for node in view.nodes:
            if node in state.nodes:
                assert view.cells_at(node) == _brute_cells_at(view.cells, node)
        tx.commit()
        _assert_cells_at(state)


# --- seed lattice containment ------------------------------------------------

def _polygons():
    rng = random.Random(7)
    yield [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]
    yield [(0.0, 0.0), (4.0, 0.0), (4.0, 0.8), (1.2, 0.8), (1.2, 2.6), (0.0, 2.6)]
    yield [(0.0, 0.0), (3.0, 0.4), (3.5, 2.6), (0.5, 2.2)]
    for _ in range(6):
        count = rng.randint(5, 40)
        radius = [1.0 + 0.4 * rng.random() for _ in range(count)]
        yield [(3.0 + radius[i] * np.cos(2 * np.pi * i / count),
                2.0 + radius[i] * np.sin(2 * np.pi * i / count)) for i in range(count)]


@pytest.mark.parametrize("tol", [0.0, 1.0e-10, 1.0e-3])
def test_strict_inside_mask_matches_scalar(tol: float) -> None:
    rng = random.Random(11)
    for polygon in _polygons():
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        queries = [(rng.uniform(min(xs) - 1, max(xs) + 1), rng.uniform(min(ys) - 1, max(ys) + 1))
                   for _ in range(300)]
        queries += list(polygon)
        queries += [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                    for a, b in zip(polygon, polygon[1:] + polygon[:1])]
        # Lattice rows through vertices (horizontal edges, vertex crossings).
        queries += [(x, y) for y in ys for x in np.linspace(min(xs), max(xs), 17)]
        qx = np.asarray([q[0] for q in queries], dtype=float)
        qy = np.asarray([q[1] for q in queries], dtype=float)
        expected = [seed._strict_inside((float(x), float(y)), polygon, tol) for x, y in zip(qx, qy)]
        assert seed._strict_inside_mask(qx, qy, polygon, tol).tolist() == expected


def _reference_uniform_lattice_points(outer, holes, h, min_x, max_x, min_y, max_y, tol):
    xs = np.arange(min_x+h, max_x-tol, h, dtype=float)
    ys = np.arange(min_y+h, max_y-tol, h, dtype=float)
    clearance = seed._LATTICE_BOUNDARY_CLEARANCE * h
    segments = seed._SegmentGrid((outer, *holes), clearance) if clearance > 0.0 else None
    points = []
    for y in ys:
        for x in xs:
            point = (float(x), float(y))
            if not seed._strict_inside(point, outer, tol):
                continue
            if any(seed._strict_inside(point, hole, tol) for hole in holes):
                continue
            if segments is not None and segments.closer_than(point, clearance):
                continue
            points.append(point)
    return points


@pytest.mark.parametrize("h", [0.5, 0.3, 0.1])
def test_uniform_lattice_points_match_scalar_loop(h: float) -> None:
    outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]
    hole = [(3.2 + 1.2 * np.cos(2 * np.pi * i / 24), 2.4 - 1.2 * np.sin(2 * np.pi * i / 24))
            for i in range(24)]
    concave = [(0.0, 0.0), (4.0, 0.0), (4.0, 0.8), (1.2, 0.8), (1.2, 2.6), (0.0, 2.6)]
    for ring, holes in ((outer, [hole]), (outer, []), (concave, [])):
        bounds = (min(p[0] for p in ring), max(p[0] for p in ring),
                  min(p[1] for p in ring), max(p[1] for p in ring))
        tol = 1.0e-10 * max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1.0)
        args = (ring, holes, h, *bounds, tol)
        assert seed._uniform_lattice_points(*args) == _reference_uniform_lattice_points(*args)


# --- segment recovery ----------------------------------------------------------

def _reference_constrained(points, outer, constraints):
    prepared = _prepare_pslg(points, outer, (), constraints, None)
    triangles = _bowyer_watson(prepared.points)
    protected: set[tuple[int, int]] = set()
    for raw_segment in prepared.segments:
        segment = tuple(map(int, raw_segment))
        triangles = _recover_segment(prepared.points, triangles, segment, protected)
        protected.add(_normal_edge(*segment))
    return _finish_triangles(prepared.points, triangles, prepared)


def test_segment_recovery_skip_matches_recovering_every_segment() -> None:
    rng = random.Random(3)
    recovered_any = False
    for case in range(12):
        m = 6 + case % 4
        lattice = [(float(i), float(j)) for j in range(m) for i in range(m)]
        interior = [(rng.uniform(0.2, m - 1.2), rng.uniform(0.2, m - 1.2)) for _ in range(3 * m)]
        points = np.asarray(lattice + interior, dtype=float)
        outer = [0, m - 1, m * m - 1, m * (m - 1)]
        corner_ids = [0, m - 1, m * m - 1, m * (m - 1)]
        boundary = sorted(
            {j * m + i for j in range(m) for i in range(m) if i in (0, m - 1) or j in (0, m - 1)},
            key=lambda row: np.arctan2(points[row][1] - (m - 1) / 2, points[row][0] - (m - 1) / 2),
        )
        outer = boundary if case % 2 else corner_ids
        # Long diagonal constraints cut many Delaunay edges and need recovery.
        a, b = m + 1, m * m - m - 2
        constraints = [(a, b)] + [(m * m + k, m * m + k + 1) for k in range(0, 3 * m - 1, 5)]
        prepared = _prepare_pslg(points, outer, (), constraints, None)
        initial_edges = {
            _normal_edge(t[i], t[(i + 1) % 3]) for t in _bowyer_watson(prepared.points) for i in range(3)
        }
        recovered_any |= any(tuple(map(int, s)) not in initial_edges for s in prepared.segments)
        expected = _outcome(lambda _: _reference_constrained(points, outer, constraints), None)
        actual = _outcome(
            lambda _: constrained_planar_triangulation(
                points, outer, constraints=constraints, backend="python"
            ).triangles,
            None,
        )
        if isinstance(expected, np.ndarray):
            assert isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
        else:
            assert actual == expected
    assert recovered_any


# --- Bowyer-Watson grid on larger sets -----------------------------------------

def _large_cases():
    rng = random.Random(29)
    lattice = [(i * 0.1, j * 0.1) for j in range(18) for i in range(22)]
    yield np.asarray(lattice, dtype=float)
    yield np.asarray([(x + rng.uniform(-1e-12, 1e-12), y) for x, y in lattice], dtype=float) * 1e5
    yield np.asarray([(rng.uniform(0, 10), rng.uniform(0, 1)) for _ in range(350)], dtype=float)
    # Strip: one very long axis, few rows.
    yield np.asarray([(i * 0.05, j * 0.05) for j in range(3) for i in range(130)], dtype=float)
    # Clustered with a few distant outliers (most cells empty).
    cluster = [(rng.gauss(0, 1e-3), rng.gauss(0, 1e-3)) for _ in range(300)]
    yield np.asarray(cluster + [(1.0, 1.0), (-1.0, 0.7), (0.9, -1.0)], dtype=float)
    # Concentric cocircular rings.
    rings = [(r * np.cos(2 * np.pi * k / 40), r * np.sin(2 * np.pi * k / 40))
             for r in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) for k in range(40)]
    yield np.unique(np.asarray(rings + [(0.0, 0.0)], dtype=float), axis=0)


@pytest.mark.parametrize("index", range(6))
def test_grid_bowyer_watson_matches_full_scan_on_large_sets(index: int) -> None:
    points = list(_large_cases())[index]
    assert _outcome(_bowyer_watson, points) == _outcome(_reference_bowyer_watson, points)
