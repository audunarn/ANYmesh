"""Native finite-hull seed (tripod speed option 2): decision-identical to the Python routine.

The native engine (``anymesher._native.finite_hull_seed``) replaces fan seeding, exact insertion and
Lawson flips of ``triangulation._finite_hull_seed`` for exact float64 (n, 2) finite ndarrays.  These
tests compare it with the Python routine (``_NATIVE_FINITE_HULL = False``) on outcomes (triangles or
exception type and message) and on the full callback trace, including callbacks that raise, move,
reshape, shrink or retype the input.  ``tools/fhs_native_harness.py`` runs the same comparison at
scale (>= 1e6 predicate cases, >= 1e4 triangulations, the captured tripod calls).

When the extension is absent the Python routine is the only route, so these tests are skipped
unless ``ANYMESHER_REQUIRE_NATIVE=1`` (then a missing kernel fails).
"""
from __future__ import annotations

import os
import struct
import warnings
from fractions import Fraction

import numpy as np
import pytest

from anymesher import triangulation
from anymesher.native import _finite_hull_seed_kernel

KERNEL = _finite_hull_seed_kernel()
REQUIRED = os.environ.get("ANYMESHER_REQUIRE_NATIVE", "0") in {"1", "true", "TRUE"}
pytestmark = pytest.mark.skipif(KERNEL is None and not REQUIRED, reason="native extension not built")


def _native_module():
    from anymesher import _native

    return _native


def _python(function, *args, **kwargs):
    saved = triangulation._NATIVE_FINITE_HULL
    triangulation._NATIVE_FINITE_HULL = False
    try:
        return function(*args, **kwargs)
    finally:
        triangulation._NATIVE_FINITE_HULL = saved


def _outcome(points, mutate=None, with_callback=True, raise_at=None):
    stages = []

    def callback(stage):
        stages.append(stage)
        if raise_at is not None and len(stages) == raise_at:
            raise KeyboardInterrupt("cancelled at %d" % raise_at)
        if mutate is not None:
            mutate(points, stage, sum(1 for seen in stages if seen == stage))

    engine = []  # the triangles handed to the completeness check, compared even when it refuses them
    real = triangulation._complete_hull_seed

    def recorded(points, triangles, cancellation_check=None):
        engine.append(list(triangles))
        return real(points, triangles, cancellation_check)

    triangulation._complete_hull_seed = recorded
    try:
        result = ("ok", triangulation._finite_hull_seed(points, callback if with_callback else None))
    except BaseException as error:  # compared, never swallowed: the type and message are the outcome
        result = ("error", type(error).__name__, str(error))
    finally:
        triangulation._complete_hull_seed = real
    return result, stages, engine


def _both(points, **kwargs):
    native = _outcome(points.copy(), **kwargs)
    python = _python(_outcome, points.copy(), **kwargs)
    return native, python


def _sign(value):
    return (value > 0) - (value < 0)


# ----------------------------------------------------------------------------------------------
# Predicates: the native signs equal the signs of the Python predicates.

def _adversarial_rows(rng, count):
    """Rows (fx fy sx sy tx ty px py): near-collinear, cocircular, duplicate, lattice, extreme."""
    rows = []
    per = count // 8
    # Near-collinear: third point on the line through the first two, nudged by a few ulps.
    a = rng.uniform(-10, 10, (per, 2))
    b = rng.uniform(-10, 10, (per, 2))
    t = rng.choice([0.25, 0.5, 0.75, 2.0, -1.0, 1.0 / 3.0], per)
    c = a + t[:, None] * (b - a)
    c = np.nextafter(c, c + rng.choice([-1.0, 0.0, 1.0], (per, 2)))
    d = rng.uniform(-10, 10, (per, 2))
    rows.append(np.c_[a, b, c, d])
    # Lattice and cocircular quadruples (rectangle corners, rotated by exact Pythagorean angles).
    i = rng.integers(-20, 20, (per, 8)).astype(float)
    rows.append(i * rng.choice([1.0, 0.5, 0.125, 1.0 / 3.0, 0.1], (per, 1)))
    x0, y0 = rng.uniform(-5, 5, (2, per))
    w, h = rng.integers(1, 9, (2, per)) * 0.25
    rect = np.c_[x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
    rows.append(rect + rng.choice([0.0, 0.0, 1e-17, -1e-17, 1e-15], (per, 8)))
    p, q = rng.integers(1, 30, (2, per)).astype(float)
    r = p * p + q * q
    circle = np.c_[r * 0 + 1 * (p * p - q * q), 2 * p * q, -(p * p - q * q), -2 * p * q,
                   2 * p * q, p * p - q * q, r, 0 * r] * rng.choice([1.0, 2.0 ** -7, 0.1], (per, 1))
    rows.append(circle)
    # Duplicates: repeated points in every position.
    base = rng.uniform(-3, 3, (per, 8))
    k = rng.integers(0, 4, per)
    for row, j in zip(base, k):
        row[2 * ((j + 1) % 4):2 * ((j + 1) % 4) + 2] = row[2 * j:2 * j + 2]
    rows.append(base)
    # Extreme and mixed magnitudes, subnormals, signed zeros.
    scale = rng.choice([1e-300, 1e-160, 1e-16, 1.0, 1e150, 1e300, 5e-324, 2.0 ** 401], (per, 8))
    rows.append(rng.uniform(-1, 1, (per, 8)) * scale)
    mixed = rng.uniform(-1, 1, (per, 8))
    mixed[:, ::3] *= 1e-16
    rows.append(mixed)
    rows.append(rng.choice([0.0, -0.0, 5e-324, -5e-324, 1.0, 2.0 ** -1022], (count - 7 * per, 8)))
    return np.ascontiguousarray(np.vstack(rows), dtype=np.float64)


def test_native_predicate_signs_match_python():
    rng = np.random.default_rng(20261125)
    rows = np.vstack([_adversarial_rows(rng, 16000), rng.uniform(-1, 1, (4000, 8))])
    signs = np.frombuffer(_native_module().fhs_predicate_signs(rows.tobytes()), dtype=np.int8).reshape(-1, 2)
    for row, (orient, circle) in zip(rows.tolist(), signs.tolist()):
        f, s, t, p = row[0:2], row[2:4], row[4:6], row[6:8]
        assert orient == _sign(triangulation.orient2d(f, s, t)), row
        assert circle == _sign(triangulation.incircle(f, s, t, p)), row


def test_native_exact_signs_match_fractions():
    # Exact references for the integer fallback: extreme spans that no float filter certifies.
    rng = np.random.default_rng(7)
    module = _native_module()
    for _ in range(3000):
        values = rng.uniform(-1, 1, 8) * rng.choice([5e-324, 1e-300, 1e-20, 1.0, 1e200, 1e300], 8)
        f = [Fraction(float(v)) for v in values]
        orient = (f[0] - f[4]) * (f[3] - f[5]) - (f[1] - f[5]) * (f[2] - f[4])
        assert module.fhs_orient_sign(*values[:6]) == _sign(orient)
        d = [(f[2 * k] - f[6], f[2 * k + 1] - f[7]) for k in range(3)]
        lift = [x * x + y * y for x, y in d]
        circle = (lift[0] * (d[1][0] * d[2][1] - d[1][1] * d[2][0])
                  - lift[1] * (d[0][0] * d[2][1] - d[0][1] * d[2][0])
                  + lift[2] * (d[0][0] * d[1][1] - d[0][1] * d[1][0]))
        expected = _sign(circle) * (-1 if _sign(orient) < 0 else 1)
        assert module.fhs_incircle_sign(*values) == expected


# ----------------------------------------------------------------------------------------------
# Triangulations.

def _families(rng):
    made = []
    for count in (4, 9, 17, 33):
        xs, ys = np.meshgrid(np.linspace(0.0, 2.0, count), np.linspace(0.0, 1.0, max(3, count // 2)))
        lattice = np.c_[xs.ravel(), ys.ravel()]
        stations = np.c_[rng.integers(0, count - 1, count) * (2.0 / (count - 1)) + 1.0 / (count - 1),
                         np.zeros(count)]
        jitter = rng.uniform(0.0, 1.0, (count, 2)) * [2.0, 1.0]
        made.append(np.unique(np.r_[lattice, stations, jitter], axis=0))
        made.append(lattice)
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 9))
    made.append(np.unique(np.c_[xs.ravel(), ys.ravel()] @ rotation.T + [1.0e3, -2.5e2], axis=0))
    made.append(np.unique(rng.uniform(-1.0, 1.0, (150, 2)), axis=0))
    side = np.linspace(0.0, 1.0, 9)
    square = np.unique(np.r_[np.c_[side, 0 * side], np.c_[1 + 0 * side, side], np.c_[side, 1 + 0 * side],
                             np.c_[0 * side, side], [[0.5, 0.5]]], axis=0)
    made.append(square)
    made.append(np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]))
    made.append(np.r_[made[0], made[0][::3]])
    made.append(np.unique(np.r_[made[1] * 1.0e300, [[1.0e300, 5.0e299]]], axis=0))
    made.append(np.array([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (5.0e-324, 0.5), (0.5, 5.0e-324),
                          (0.5, 0.5), (-0.0, 0.25), (0.25, -0.0)]))
    circle = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    made.append(np.c_[np.cos(circle), np.sin(circle)])
    made.append(np.r_[np.c_[np.cos(circle), np.sin(circle)], [[0.0, 0.0], [0.1, 0.0]]])
    near = np.c_[np.linspace(0, 1, 40), np.zeros(40)]
    near[1::2, 1] = 1e-17
    made.append(np.r_[near, [[0.5, 1.0], [0.5, -1.0]]])
    made.append(np.array([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]))  # zero-area hull: refusal
    for _ in range(40):
        n = int(rng.integers(4, 90))
        kind = int(rng.integers(0, 4))
        if kind == 0:
            pts = rng.integers(0, 6, (n, 2)) * 0.5
        elif kind == 1:
            pts = rng.uniform(0, 1, (n, 2))
        elif kind == 2:
            pts = np.r_[rng.integers(0, 5, (n, 2)) * 0.25, rng.uniform(0, 1, (n // 3 + 1, 2))]
        else:
            t = rng.uniform(0, 1, n)
            pts = np.c_[t, 3 * t + 1] + rng.choice([0.0, 1e-16, -1e-16, 0.3], (n, 2))
        made.append(pts.astype(np.float64))
    return made


FAMILIES = _families(np.random.default_rng(20261125))


@pytest.mark.parametrize("index", range(len(FAMILIES)))
def test_native_matches_python_outcome_and_callback_trace(index):
    points = FAMILIES[index]
    native, python = _both(points)
    assert native == python
    assert _outcome(points.copy(), with_callback=False)[0] == native[0]


def test_native_route_is_taken_and_python_route_is_kept_when_disabled(monkeypatch):
    calls = []
    real = triangulation._finite_hull_seed_kernel

    def spy():
        kernel = real()
        calls.append(kernel)
        return kernel

    monkeypatch.setattr(triangulation, "_finite_hull_seed_kernel", spy)
    triangulation._finite_hull_seed(FAMILIES[1].copy())
    assert calls and calls[-1] is not None
    calls.clear()
    _python(triangulation._finite_hull_seed, FAMILIES[1].copy())
    assert calls == []
    # Absent kernel: the Python routine, same result.
    monkeypatch.setattr(triangulation, "_finite_hull_seed_kernel", lambda: None)
    assert triangulation._finite_hull_seed(FAMILIES[1].copy()) == _python(
        triangulation._finite_hull_seed, FAMILIES[1].copy())


@pytest.mark.parametrize("index", [0, 1, 2, 9, 14])
def test_native_matches_python_when_a_callback_raises(index):
    points = FAMILIES[index]
    total = len(_outcome(points.copy())[1])
    for raise_at in sorted({1, 2, 3, total // 3, total // 2, total - 1, total}):
        native, python = _both(points, raise_at=raise_at)
        assert native == python


def _mutations():
    def move(stage, when):
        def apply(points, seen, count):
            if seen.endswith(stage) and count == when:
                points[-1] = (8.0, 8.0)
                points[len(points) // 2] = points[len(points) // 2] * 0.5
        return apply

    def nan(stage, when):
        def apply(points, seen, count):
            if seen.endswith(stage) and count == when:
                points[1] = (np.nan, 0.0)
        return apply

    def reshape(points, seen, count):
        if seen.endswith("insertion") and count == 1:
            points.shape = (len(points), 2)[::-1] if len(points) == 2 else (2, len(points))

    def retype(points, seen, count):
        if seen.endswith("edge") and count == 3:
            with warnings.catch_warnings():  # deprecated in NumPy 2.5, still a supported mutation
                warnings.simplefilter("ignore", DeprecationWarning)
                points.dtype = np.int64

    def shrink(stage, when):
        def apply(points, seen, count):
            if seen.endswith(stage) and count == when:
                points.resize((3, 2), refcheck=False)
        return apply

    def same_bytes(points, seen, count):
        if seen.endswith("edge") and count == 2:
            points[0] = points[0].copy()  # unchanged contents: stays on the native route

    def signed_zero(points, seen, count):
        if seen.endswith("legality") and count == 1:
            if points[0, 0] == 0.0:
                points[0, 0] = -0.0 if np.copysign(1.0, points[0, 0]) > 0 else 0.0  # bytes change only

    made = []
    for stage, when in (("insertion", 1), ("insertion", 3), ("edge", 1), ("edge", 7), ("edge", 40),
                        ("legality", 1), ("legality", 5)):
        made.append(move(stage, when))
        made.append(nan(stage, when))
    made += [reshape, retype, same_bytes, signed_zero, shrink("insertion", 2), shrink("edge", 5),
             shrink("legality", 2)]
    return made


MUTATIONS = _mutations()


@pytest.mark.parametrize("which", range(len(MUTATIONS)))
@pytest.mark.parametrize("index", [0, 1, 9, 14, 20])
def test_native_follows_a_callback_that_changes_the_input(which, index):
    native, python = _both(FAMILIES[index], mutate=MUTATIONS[which])
    assert native == python


def test_native_snapshot_detects_a_dtype_view_change():
    points = np.array([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (1.0, 1.0), (2.0, 1.0)])

    def swap_order(array, seen, count):
        if seen.endswith("insertion") and count == 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                array.dtype = array.dtype.newbyteorder()

    native, python = _both(points, mutate=swap_order)
    assert native == python


def test_native_matches_python_for_other_layouts_and_types():
    # float32 and non-finite inputs keep the Python route; reversed and Fortran-order float64 views
    # take the native route through their strides.
    base = FAMILIES[0]
    for points in (base.astype(np.float32), base[::-1], np.asfortranarray(base),
                   np.r_[base, [[np.inf, 0.0]]]):
        native, python = _both(points)
        assert native == python


def test_probe_signs_are_exact_on_bit_patterns():
    # A few hand-built cases around the filter boundary of orient2d.
    module = _native_module()
    one_ulp = struct.unpack("<d", struct.pack("<q", struct.unpack("<q", struct.pack("<d", 0.5))[0] + 1))[0]
    assert module.fhs_orient_sign(0.0, 0.0, 1.0, 1.0, 0.5, 0.5) == 0
    assert module.fhs_orient_sign(0.0, 0.0, 1.0, 1.0, 0.5, one_ulp) == 1
    assert module.fhs_orient_sign(0.0, 0.0, 1.0, 1.0, one_ulp, 0.5) == -1
    assert module.fhs_incircle_sign(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0) == 0
    assert module.fhs_incircle_sign(0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5) == 1
    # incircle is orientation-normalised: an interior point is positive for a clockwise triangle too.
    assert module.fhs_incircle_sign(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.5) == 1
    assert module.fhs_incircle_sign(0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0) == -1
