"""Complete seeds where the finite super triangle loses hull cells.

Rotated straight boundary edges carry stations whose rounded coordinates make
the hull very slightly convex.  The circumcircle of such a flat hull triangle
contains a finite super vertex, so the super-triangle insertion drops it and
used to fall back to the slow Python finite-hull fan.  The ghost seed completes
it; it is published only when strictly Delaunay, so every result is identical
to the finite-hull route's, and exact cocircular ties keep that route.
"""
import numpy as np
import pytest

import anymesher.native as native
import anymesher.triangulation as triangulation
from anymesher._ghost_seed import GhostSeedRefusal, ghost_bowyer_watson


def _extension():
    extension = pytest.importorskip("anymesher._native")
    assert callable(getattr(extension, "ghost_delaunay", None)), (
        "rebuild the extension: this checkout requires ghost_delaunay"
    )


def _rotated_rectangle(angle, stations=15, interior=True):
    c, s = np.cos(angle), np.sin(angle)
    t = np.linspace(0., 1., stations)[:-1]
    ring = np.vstack([
        np.column_stack((4. * t, 0. * t)), np.column_stack((4. + 0. * t, 2. * t)),
        np.column_stack((4. - 4. * t, 2. + 0. * t)), np.column_stack((0. * t, 2. - 2. * t)),
    ])
    points = ring
    if interior:
        inner = np.array([(x, y) for x in np.linspace(.5, 3.5, 5) for y in (.5, 1., 1.5)])
        points = np.vstack((ring, inner))
    rotation = np.array([[c, -s], [s, c]])
    return np.ascontiguousarray(ring @ rotation.T + 3.7), np.ascontiguousarray(points @ rotation.T + 3.7)


ANGLES = [.01, .3, .7, 1.1, 1.5]


def _ghost_or_tie(points):
    try:
        return ghost_bowyer_watson(points, triangulation.orient2d, triangulation.incircle)
    except GhostSeedRefusal as refusal:
        assert refusal.reason == "cocircular"
        return None


@pytest.mark.parametrize("angle", ANGLES)
def test_super_triangle_seed_is_incomplete_on_rotated_stations(angle):
    # Documents the cause; if this ever passes, the reproducer is stale.
    _, points = _rotated_rectangle(angle)
    assert not triangulation._complete_hull_seed(points, triangulation._bowyer_watson(points))


def _without_ghost(monkeypatch):
    """The pre-change route: incomplete seeds always reach the finite-hull fan."""
    def refuse(*args, **kwargs):
        raise GhostSeedRefusal("disabled", reason="disabled")

    monkeypatch.setattr(triangulation, "ghost_bowyer_watson", refuse)
    monkeypatch.setattr(triangulation, "_run_ghost_native_seed",
                        lambda *args: (None, "ghost_seed_native_refusal: disabled"))


def _review_tie_fixture():
    # Review fixture: interior stations on one circle (cocircular ties) next to
    # an almost collinear hull edge; another Delaunay tie choice changes quality.
    outer = np.array([[-10., -10.], [0., np.nextafter(-10., np.inf)], [10., -10.],
                      [10., 10.], [-10., 10.]])
    interior = np.array([[5., 0.], [-4., 3.], [3., -4.], [4., -3.]])
    return outer, interior


def _scaled_review_tie_fixture():
    # Review r2 fixture: exactly cocircular interior stations (scaled integer
    # coordinates); under FP contraction a native certificate published the
    # other diagonal.  It must refuse and keep the finite-hull fan's choice.
    s, k = 2. ** -45, 75890737893414
    radius = 10. * k * s
    outer = np.array([[-radius, -radius], [0., np.nextafter(-radius, np.inf)], [radius, -radius],
                      [radius, radius], [-radius, radius]])
    interior = np.array([[s * (133755125355830 + k * u), s * (-19747421224915 + k * v)]
                         for u, v in ((5, 0), (-4, 3), (3, -4), (4, -3))])
    return outer, interior


def _inputs():
    for angle in ANGLES:
        outer, points = _rotated_rectangle(angle)
        yield f"rotated-{angle}", outer, points[len(outer):]
    yield ("review-tie", *_review_tie_fixture())
    yield ("review-tie-scaled", *_scaled_review_tie_fixture())


def test_scaled_review_fixture_is_an_exact_tie_refused_on_both_paths():
    _extension()
    outer, interior = _scaled_review_tie_fixture()
    points = np.ascontiguousarray(np.vstack((outer, interior)))
    with pytest.raises(GhostSeedRefusal) as caught:
        ghost_bowyer_watson(points, triangulation.orient2d, triangulation.incircle)
    assert caught.value.reason == "cocircular"
    actual, reason = native._run_ghost_native_seed(points, None)
    assert actual is None and reason.startswith("ghost_seed_native_refusal: cocircular:")


@pytest.mark.parametrize("reference_native_seed", [True, False])
@pytest.mark.parametrize("name,outer,interior", list(_inputs()))
def test_results_are_identical_to_the_finite_hull_route(monkeypatch, name, outer, interior,
                                                         reference_native_seed):
    if reference_native_seed:
        _extension()
    kwargs = dict(interior_points=interior, backend="python",
                  _reference_native_seed=reference_native_seed)
    actual = triangulation.triangulate_polygon(outer, **kwargs)
    with monkeypatch.context() as patch:
        _without_ghost(patch)
        expected = triangulation.triangulate_polygon(outer, **kwargs)
    assert actual.points.tobytes() == expected.points.tobytes()
    assert actual.triangles.tobytes() == expected.triangles.tobytes()
    assert actual.segments.tobytes() == expected.segments.tobytes()


def _forbidden(message):
    def call(*args, **kwargs):
        raise AssertionError(message)
    return call


@pytest.mark.parametrize("reference_native_seed", [True, False])
@pytest.mark.parametrize("angle", ANGLES)
def test_incomplete_seed_skips_the_finite_hull_fan_unless_cocircular(monkeypatch, angle,
                                                                      reference_native_seed):
    if reference_native_seed:
        _extension()
    outer, points = _rotated_rectangle(angle)
    tie = _ghost_or_tie(points) is None
    if not reference_native_seed:
        monkeypatch.setattr(triangulation, "_run_ghost_native_seed", _forbidden("native used"))
    used = []
    finite = triangulation._finite_hull_seed
    monkeypatch.setattr(triangulation, "_finite_hull_seed",
                        lambda *args: used.append(None) or finite(*args))
    result = triangulation.triangulate_polygon(
        outer, interior_points=points[len(outer):], backend="python",
        _reference_native_seed=reference_native_seed,
    )
    assert used == ([None] if tie else [])
    if reference_native_seed:
        assert result.actual_backend == ("python" if tie else "anymesher-cpp17-ghost-seed")
        assert result.fallback_reason == "reference_seed_incomplete_hull"
    else:
        assert result.actual_backend == "python"


def test_python_ghost_seed_runs_when_the_native_one_is_absent(monkeypatch):
    angle = next(a for a in ANGLES if _ghost_or_tie(_rotated_rectangle(a)[1]) is not None)
    outer, points = _rotated_rectangle(angle)
    expected = ghost_bowyer_watson(points, triangulation.orient2d, triangulation.incircle)
    monkeypatch.setattr(triangulation, "_run_ghost_native_seed",
                        lambda *args: (None, "ghost_seed_capability_absent"))
    monkeypatch.setattr(triangulation, "_finite_hull_seed", _forbidden("finite-hull fan used"))
    seen = []
    real = triangulation.ghost_bowyer_watson
    monkeypatch.setattr(triangulation, "ghost_bowyer_watson",
                        lambda *args: seen.append(real(*args)) or seen[-1])
    result = triangulation.triangulate_polygon(
        outer, interior_points=points[len(outer):], backend="python", _reference_native_seed=True,
    )
    assert seen == [expected]
    assert result.actual_backend == "python"
    assert result.native_diagnostics["ghost_seed"] == "python"


@pytest.mark.parametrize("angle", ANGLES)
def test_native_and_python_paths_give_identical_triangulations(angle):
    _extension()
    outer, points = _rotated_rectangle(angle)
    kwargs = dict(interior_points=points[len(outer):], backend="python")
    first = triangulation.triangulate_polygon(outer, _reference_native_seed=True, **kwargs)
    second = triangulation.triangulate_polygon(outer, **kwargs)
    assert first.points.tobytes() == second.points.tobytes()
    assert first.triangles.tobytes() == second.triangles.tobytes()


def _cases():
    rng = np.random.default_rng(20261125)
    for angle in np.linspace(.01, 1.5, 12):
        yield "rotated", _rotated_rectangle(angle)[1]
        yield "rotated-ring", _rotated_rectangle(angle, interior=False)[1]
    for _ in range(60):
        count = int(rng.integers(3, 150))
        yield "random", rng.random((count, 2)) * 10. ** rng.uniform(-3, 3) + rng.uniform(-100, 100)
    for scale in (2. ** -60, 2. ** 60):
        yield "extreme", (rng.random((40, 2)) + 1.) * scale
    for count in (3, 5, 9):
        values = np.linspace(0., 3., count)
        lattice = np.array([(x, y) for x in values for y in values])
        for angle in (0., .3, 1.1):
            c, s = np.cos(angle), np.sin(angle)
            yield "lattice", lattice @ np.array([[c, -s], [s, c]]).T + 1.25
    yield "collinear-run", np.vstack((np.column_stack((np.linspace(0., 1., 9), np.zeros(9))), [[.5, 1e-12]]))


@pytest.mark.parametrize("kind,points", list(_cases()))
def test_native_ghost_seed_matches_python_reference(kind, points):
    _extension()
    points = np.ascontiguousarray(points, dtype=float)
    expected = _ghost_or_tie(points)
    actual, reason = native._run_ghost_native_seed(points, None)
    if expected is None:
        assert actual is None and reason.startswith("ghost_seed_native_refusal: cocircular:")
        assert kind != "random"
        return
    assert reason is None
    assert actual == expected
    assert triangulation._complete_hull_seed(points, expected)
    # Strictly Delaunay: the triangulation is unique, so the ghost seed equals
    # the finite-hull fan it replaces cell for cell.
    assert expected == triangulation._finite_hull_seed(points)


@pytest.mark.parametrize("points,message", [
    (np.array([[0., 0.], [1., 1.], [2., 2.]]), "collinear"),
    (np.array([[0., 0.], [1., 0.]]), "too_few"),
    (np.array([[0., 0.], [1., 0.], [1., 0.], [0., 1.]]), "duplicate"),
    (np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]), "cocircular"),
])
def test_ghost_seed_refusals_are_typed_and_shared(points, message):
    _extension()
    code = message
    with pytest.raises(GhostSeedRefusal) as caught:
        ghost_bowyer_watson(points, triangulation.orient2d, triangulation.incircle)
    assert caught.value.reason == code
    actual, reason = native._run_ghost_native_seed(np.ascontiguousarray(points), None)
    assert actual is None and reason.startswith(f"ghost_seed_native_refusal: {code}: ")


class _Stop(RuntimeError):
    pass


def test_native_ghost_seed_cancellation_propagates_with_identity():
    _extension()
    _, points = _rotated_rectangle(.3, stations=60)

    def cancel(phase):
        raise _Stop(phase)

    with pytest.raises(_Stop) as caught:
        native._run_ghost_native_seed(points, cancel)
    assert type(caught.value) is _Stop


@pytest.mark.parametrize("phase", ["native ghost seed insertion", "native ghost seed certificate"])
def test_cancellation_inside_the_ghost_seed_is_not_a_fallback(monkeypatch, phase):
    _extension()
    angle = next(a for a in ANGLES if _ghost_or_tie(_rotated_rectangle(a, stations=80)[1]) is not None)
    outer, points = _rotated_rectangle(angle, stations=80)

    def cancel(where):
        if where == phase:
            raise _Stop(where)

    monkeypatch.setattr(triangulation, "_finite_hull_seed", _forbidden("fallback ran"))
    monkeypatch.setattr(triangulation, "ghost_bowyer_watson", _forbidden("python retry ran"))
    with pytest.raises(_Stop) as caught:
        triangulation.triangulate_polygon(
            outer, interior_points=points[len(outer):], backend="python",
            _reference_native_seed=True, cancellation_check=cancel,
        )
    assert str(caught.value) == phase


class _Hostile(RuntimeError):
    def with_traceback(self, traceback):  # a replacement must not reach the caller
        return RuntimeError("replaced")


@pytest.mark.parametrize("error_type", [_Stop, _Hostile, GhostSeedRefusal, KeyboardInterrupt])
@pytest.mark.parametrize("native_path", [True, False])
def test_callback_exceptions_keep_their_identity_and_never_fall_back(monkeypatch, error_type,
                                                                     native_path):
    if native_path:
        _extension()
    else:
        monkeypatch.setattr(triangulation, "_run_ghost_native_seed",
                            lambda *args: (None, "ghost_seed_capability_absent"))
    angle = next(a for a in ANGLES if _ghost_or_tie(_rotated_rectangle(a)[1]) is not None)
    outer, points = _rotated_rectangle(angle)
    if error_type is GhostSeedRefusal:
        error = GhostSeedRefusal("stop", reason="stop")
    else:
        error = error_type("stop")
    phase = ("native ghost seed certificate" if native_path
             else "python triangulation ghost seed certificate")

    def cancel(where):
        if where == phase:
            raise error

    monkeypatch.setattr(triangulation, "_finite_hull_seed", _forbidden("fallback ran"))
    with pytest.raises(error_type) as caught:
        triangulation.triangulate_polygon(
            outer, interior_points=points[len(outer):], backend="python",
            _reference_native_seed=True, cancellation_check=cancel,
        )
    assert caught.value is error



@pytest.mark.parametrize("phase", ["native triangulation insertion"])
def test_reference_seed_callback_identity_survives_hostile_with_traceback(phase):
    _extension()
    outer, points = _rotated_rectangle(.3)
    error = _Hostile("stop")

    def cancel(where):
        if where == phase:
            raise error

    with pytest.raises(_Hostile) as caught:
        triangulation.triangulate_polygon(
            outer, interior_points=points[len(outer):], backend="python",
            _reference_native_seed=True, cancellation_check=cancel,
        )
    assert caught.value is error


def test_native_ghost_codes_only_input_properties_as_refusals(monkeypatch):
    # A broken native invariant is an internal error: Python retries it.
    _extension()
    angle = next(a for a in ANGLES if _ghost_or_tie(_rotated_rectangle(a)[1]) is not None)
    outer, points = _rotated_rectangle(angle)
    complete = triangulation._complete_hull_seed
    monkeypatch.setattr(triangulation, "_run_ghost_native_seed",
                        lambda *args: (None, "ghost_seed_native_error: ghost seed lost an edge twin"))
    monkeypatch.setattr(triangulation, "_finite_hull_seed", _forbidden("fallback ran"))
    result = triangulation.triangulate_polygon(
        outer, interior_points=points[len(outer):], backend="python", _reference_native_seed=True,
    )
    assert result.native_diagnostics["ghost_seed_attempts"] == (
        "ghost_seed_native_error: ghost seed lost an edge twin", "python")
    assert complete is triangulation._complete_hull_seed


def _rounding_control():
    import ctypes
    import platform
    import sys
    if sys.platform == "win32":
        library, downward = ctypes.CDLL("ucrtbase"), 0x0100
    else:
        library = ctypes.CDLL(None)
        machine = platform.machine().lower()
        downward = 0x00800000 if machine in {"arm64", "aarch64"} else 0x0400
    get, put = library.fegetround, library.fesetround
    original = get()
    if put(downward) != 0 or get() != downward:
        put(original)
        pytest.skip("cannot set FE_DOWNWARD with the assumed constant")
    put(original)
    return get, put, downward


def _review_r3_points():
    # Review r3 fixture: exactly cocircular; under FE_DOWNWARD a certificate
    # evaluated in that mode accepted the tie.
    s, k = 2. ** -45, 28340226917889
    cx, cy = -6823211715099, 42014152124534
    radius = 10. * k * s
    outer = [[-radius, -radius], [0., np.nextafter(-radius, np.inf)], [radius, -radius],
             [radius, radius], [-radius, radius]]
    interior = [[s * (cx + k * u), s * (cy + k * v)] for u, v in ((5, 0), (-4, 3), (3, -4), (4, -3))]
    return np.array(outer), np.array(interior)


def test_ghost_seeds_refuse_a_non_nearest_rounding_mode_and_change_nothing(monkeypatch):
    _extension()
    try:
        get, put, downward = _rounding_control()
    except OSError:
        pytest.skip("no C runtime rounding control")
    outer, interior = _review_r3_points()
    points = np.ascontiguousarray(np.vstack((outer, interior)))
    original = get()
    put(downward)
    try:
        native_result = native._run_ghost_native_seed(points, None)
        try:
            ghost_bowyer_watson(points, triangulation.orient2d, triangulation.incircle)
            python_reason = None
        except GhostSeedRefusal as refusal:
            python_reason = refusal.reason
        actual = triangulation.triangulate_polygon(
            outer, interior_points=interior, backend="python", _reference_native_seed=True)
        with monkeypatch.context() as patch:
            _without_ghost(patch)
            expected = triangulation.triangulate_polygon(
                outer, interior_points=interior, backend="python", _reference_native_seed=True)
        after = get()
    finally:
        put(original)
    assert after == downward  # the seed never changes the environment
    assert native_result[0] is None
    assert native_result[1].startswith("ghost_seed_native_refusal: rounding_mode:")
    assert python_reason == "rounding_mode"
    assert actual.triangles.tobytes() == expected.triangles.tobytes()


@pytest.mark.parametrize("native_path", [True, False])
def test_a_callback_changing_the_rounding_mode_is_a_refusal(monkeypatch, native_path):
    _extension()
    try:
        get, put, downward = _rounding_control()
    except OSError:
        pytest.skip("no C runtime rounding control")
    angle = next(a for a in ANGLES if _ghost_or_tie(_rotated_rectangle(a)[1]) is not None)
    _, points = _rotated_rectangle(angle, stations=80)
    original = get()
    flipped = []

    def cancel(phase):
        if not flipped:
            flipped.append(phase)
            put(downward)

    try:
        if native_path:
            actual, reason = native._run_ghost_native_seed(points, cancel)
        else:
            with pytest.raises(GhostSeedRefusal) as caught:
                ghost_bowyer_watson(points, triangulation.orient2d, triangulation.incircle, cancel)
    finally:
        put(original)
    if native_path:
        assert actual is None
        assert reason.startswith("ghost_seed_native_refusal: rounding_mode:")
    else:
        assert caught.value.reason == "rounding_mode"


class _HostileTraceback(RuntimeError):
    @property
    def __traceback__(self):
        raise RuntimeError("traceback denied")

    @__traceback__.setter
    def __traceback__(self, value):
        pass


@pytest.mark.parametrize("phase", ["native triangulation insertion", "native ghost seed certificate"])
def test_hostile_traceback_property_keeps_identity_and_never_falls_back(monkeypatch, phase):
    _extension()
    angle = next(a for a in ANGLES if _ghost_or_tie(_rotated_rectangle(a)[1]) is not None)
    outer, points = _rotated_rectangle(angle)
    error = _HostileTraceback("stop")

    def cancel(where):
        if where == phase:
            raise error

    monkeypatch.setattr(triangulation, "_finite_hull_seed", _forbidden("fallback ran"))
    monkeypatch.setattr(triangulation, "ghost_bowyer_watson", _forbidden("python retry ran"))
    with pytest.raises(_HostileTraceback) as caught:
        triangulation.triangulate_polygon(
            outer, interior_points=points[len(outer):], backend="python",
            _reference_native_seed=True, cancellation_check=cancel,
        )
    assert caught.value is error
