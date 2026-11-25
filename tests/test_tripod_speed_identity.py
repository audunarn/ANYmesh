"""Speed slice (tripod 2026-11-25): each shortcut is decision-identical to the route it replaces.

1. ``_finite_hull_seed``: bounding-box prefilter + tuple coordinates == the plain scan (driven through an
   ndarray subclass, which keeps the original code path), callbacks and errors included, with far fewer
   orientation predicates.
"""
from __future__ import annotations

from math import radians, tan

import numpy as np
import pytest

from anymesher import MeshError
from anymesher import triangulation


class _PlainArray(np.ndarray):
    """Same values, but not exactly ``np.ndarray``: ``_finite_hull_seed`` keeps its original scan."""


def _plain(points: np.ndarray) -> np.ndarray:
    made = _PlainArray(points.shape, dtype=points.dtype)  # owns its buffer, like the exact copy
    made[...] = points
    return made


def _cases():
    rng = np.random.default_rng(20261125)
    made = []
    for count in (6, 14, 30):
        # Quad-seed-like inputs: a lattice (cocircular quadruples), collinear boundary stations,
        # interior jitter and points exactly on lattice edges.
        xs, ys = np.meshgrid(np.linspace(0.0, 2.0, count), np.linspace(0.0, 1.0, max(3, count // 2)))
        lattice = np.c_[xs.ravel(), ys.ravel()]
        on_edges = np.c_[rng.integers(0, count - 1, count) * (2.0 / (count - 1)) + 1.0 / (count - 1),
                         np.zeros(count)]
        jitter = rng.uniform(0.0, 1.0, (count, 2)) * [2.0, 1.0]
        made.append(np.unique(np.r_[lattice, on_edges, jitter], axis=0))
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 9))
    made.append(np.unique(np.c_[xs.ravel(), ys.ravel()] @ rotation.T + [1.0e3, -2.5e2], axis=0))
    made.append(np.unique(rng.uniform(-1.0, 1.0, (120, 2)), axis=0))
    side = np.linspace(0.0, 1.0, 9)
    made.append(np.unique(np.r_[np.c_[side, 0 * side], np.c_[1 + 0 * side, side], np.c_[side, 1 + 0 * side],
                                np.c_[0 * side, side], [[0.5, 0.5]]], axis=0))
    # Duplicate stations (zero-area cells whose turns are all zero outside their box; Codex r1).
    made.append(np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]))
    made.append(np.r_[made[0], made[0][::3]])
    # Extreme magnitudes, subnormal offsets and signed zeros.
    made.append(np.unique(np.r_[made[1] * 1.0e300, [[1.0e300, 5.0e299]]], axis=0))
    tiny = np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (5.0e-324, 0.5), (0.5, 5.0e-324),
                     (0.5, 0.5), (-0.0, 0.25), (0.25, -0.0)])
    made.append(tiny)
    return made


CASES = _cases()


def _outcome(points, mutate=None):
    stages = []

    def callback(stage):
        stages.append(stage)
        if mutate is not None:
            mutate(points, stage, sum(1 for seen in stages if seen == stage))

    try:
        result = ("ok", triangulation._finite_hull_seed(points, callback))
    except Exception as error:  # compared, never swallowed: the type and message are the outcome
        result = ("error", type(error).__name__, str(error))
    return result, stages


@pytest.mark.parametrize("index", range(len(CASES)))
def test_finite_hull_seed_prefilter_matches_the_plain_scan(index):
    points = CASES[index]
    assert type(points) is np.ndarray and points.dtype == np.float64
    assert _outcome(points.copy()) == _outcome(_plain(points))


@pytest.mark.parametrize("stage,when", [("insertion", 1), ("insertion", 3), ("edge", 1), ("edge", 7),
                                        ("legality", 1)])
def test_finite_hull_seed_follows_a_callback_that_moves_the_input(stage, when):
    # Codex r1: a callback that changes the input must see the live-array behaviour of the original.
    def move(points, seen, count):
        if seen.endswith(stage) and count == when:
            points[-1] = (8.0, 8.0)
            points[len(points) // 2] = points[len(points) // 2] * 0.5

    for points in (np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0)]), CASES[1]):
        assert _outcome(points.copy(), move) == _outcome(_plain(points), move)


def test_finite_hull_seed_non_float64_input_keeps_the_plain_scan():
    as_float32 = CASES[0].astype(np.float32)
    assert _outcome(as_float32.copy()) == _outcome(as_float32.view(_PlainArray))


def test_finite_hull_seed_prefilter_skips_cells_that_cannot_contain_the_point(monkeypatch):
    # Fails on the old code, whose location scan tests every cell for every inserted point.
    points = CASES[2]
    calls = [0]
    original = triangulation.orient2d

    def counted(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(triangulation, "orient2d", counted)
    fast = triangulation._finite_hull_seed(points)
    fast_calls, calls[0] = calls[0], 0
    plain = triangulation._finite_hull_seed(_plain(points))
    assert fast == plain
    assert fast_calls < 0.75 * calls[0]


def test_finite_hull_seed_follows_a_callback_that_reshapes_the_input():
    # Codex r2: a byte-identical in-place reshape must also drop to the live array.
    def reshape(points, seen, count):
        if seen.endswith("insertion") and count == 1:
            points.shape = (2, 4)

    points = np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0)])
    assert _outcome(points.copy(), reshape) == _outcome(_plain(points), reshape)


def test_finite_hull_seed_calls_no_extra_predicates_before_the_first_insertion(monkeypatch):
    # Codex r2: the zero-area flags come from a vectorised float filter, not from extra orient2d calls.
    counts = {}
    original = triangulation.orient2d
    calls = [0]

    def counted(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(triangulation, "orient2d", counted)
    for label, points in (("fast", CASES[2].copy()), ("plain", _plain(CASES[2]))):
        calls[0] = 0
        seen = []

        def callback(stage):
            if stage.endswith("insertion") and not seen:
                seen.append(calls[0])

        triangulation._finite_hull_seed(points, callback)
        counts[label] = seen[0]
    assert counts["fast"] == counts["plain"]


@pytest.mark.parametrize("stage,when", [("convex hull", k) for k in range(1, 9)] + [("hull validation", 1)])
def test_finite_hull_seed_follows_a_callback_that_shrinks_the_input_during_the_hull(stage, when):
    # Codex r5: a hull computed before a callback shrank the array references stale rows; the
    # shortcut must then keep the original ndarray indexing (same IndexError and message).
    def shrink(points, seen, count):
        if seen.endswith(stage) and count == when:
            points.resize((3, 2), refcheck=False)

    for points in (np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)]),
                   np.array([(0.0, 0.0), (0.0, 4.0), (4.0, 4.0), (4.0, 0.0)]), CASES[1]):
        assert _outcome(points.copy(), shrink) == _outcome(_plain(points), shrink)
