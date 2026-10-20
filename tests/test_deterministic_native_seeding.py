"""The private native insertion path must reproduce Python reference seeds."""
from copy import deepcopy

import numpy as np
import pytest

import anymesher.native as native
import anymesher.triangulation as triangulation


def _extension():
    extension = pytest.importorskip("anymesher._native")
    assert callable(getattr(extension, "reference_delaunay", None)), (
        "rebuild the extension: this checkout requires reference_delaunay"
    )
    return extension


def _grid(count, angle=0., shift=0.):
    values = np.linspace(0., 3., count)
    points = np.array([(x, y) for x in values for y in values])
    matrix = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    return np.ascontiguousarray(points @ matrix + shift)


@pytest.mark.parametrize("count,angle,shift", [
    (3, 0., 0.), (7, 0., 0.), (13, 0., 0.),
    (7, .17, 0.), (7, .7, 120.25), (9, 0., -17.125),
])
def test_ordered_insertion_matches_reference(count, angle, shift):
    _extension()
    points = _grid(count, angle, shift)
    expected = triangulation._bowyer_watson(points)
    actual, reason = native._run_reference_native_seed(
        points, triangulation.orient2d, triangulation.incircle, None,
    )
    assert reason is None
    assert actual == expected


def _same_result(first, second):
    for name in ("points", "triangles", "segments", "boundary_segments",
                 "mandatory_segments", "outer_loop"):
        a, b = getattr(first, name), getattr(second, name)
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), name
    assert [x.tobytes() for x in first.hole_loops] == [x.tobytes() for x in second.hole_loops]
    assert first.protected_node_rows == second.protected_node_rows


@pytest.mark.parametrize("case", ["grid", "hole", "crossing", "concave", "near_collinear"])
def test_complete_constrained_result_uses_reference_finishing(case):
    _extension()
    outer = np.array([[0., 0.], [4., 0.], [4., 4.], [0., 4.]])
    kwargs = {}
    if case == "grid":
        kwargs["interior_points"] = np.array([(x, y) for x in (1., 2., 3.) for y in (1., 2., 3.)])
    elif case == "hole":
        kwargs["holes"] = [np.array([[1., 1.], [1., 2.], [2., 2.], [2., 1.]])]
        kwargs["interior_points"] = np.array([[3., 1.], [3., 2.], [3., 3.]])
    elif case == "crossing":
        kwargs["constraints"] = [
            np.array([[0., 2.], [4., 2.]]), np.array([[2., 0.], [2., 4.]]),
        ]
        kwargs["interior_points"] = np.array([[1., 1.], [3., 3.]])
    elif case == "concave":
        outer = np.array([[0., 0.], [4., 0.], [4., 1.], [1., 1.], [1., 4.], [0., 4.]])
    else:
        kwargs["interior_points"] = np.array([[1., 1.], [2., 2.], [3., 3.+1.e-14]])
    old_outer = outer.tobytes()
    expected = triangulation.triangulate_polygon(outer, backend="python", **kwargs)
    actual = triangulation.triangulate_polygon(
        outer, backend="python", _reference_native_seed=True, **kwargs,
    )
    _same_result(actual, expected)
    assert outer.tobytes() == old_outer
    assert actual.actual_backend == "anymesher-cpp17-reference-seed"
    assert actual.requested_backend == "python"
    assert actual.native_diagnostics == {"reference_native_seed": True}


def test_explicit_python_default_never_invokes_acceleration(monkeypatch):
    def unexpected(*args):
        raise AssertionError("explicit Python must remain Python")
    monkeypatch.setattr(triangulation, "_run_reference_native_seed", unexpected)
    result = triangulation.triangulate_polygon([[0., 0.], [1., 0.], [0., 1.]], backend="python")
    assert result.actual_backend == "python"


def test_stale_extension_capability_falls_back(monkeypatch):
    extension = _extension()
    monkeypatch.delattr(extension, "reference_delaunay")
    outer = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    actual = triangulation.triangulate_polygon(outer, backend="python", _reference_native_seed=True)
    expected = triangulation.triangulate_polygon(outer, backend="python")
    _same_result(actual, expected)
    assert actual.actual_backend == "python"
    assert actual.fallback_reason == "reference_seed_capability_absent"


def test_absent_extension_falls_back_without_import(monkeypatch):
    monkeypatch.setattr(native, "find_spec", lambda name: None)
    def unexpected(name):
        raise AssertionError("a missing extension must not be imported")
    monkeypatch.setattr(native, "import_module", unexpected)
    result, reason = native._run_reference_native_seed(
        _grid(3), triangulation.orient2d, triangulation.incircle, None,
    )
    assert result is None and reason == "reference_seed_capability_absent"


def test_present_unloadable_extension_error_is_not_absence(monkeypatch):
    error = ImportError("dependent native DLL failed to load")
    monkeypatch.setattr(native, "find_spec", lambda name: object())
    def broken(name):
        raise error
    monkeypatch.setattr(native, "import_module", broken)
    with pytest.raises(ImportError) as caught:
        native._run_reference_native_seed(
            _grid(3), triangulation.orient2d, triangulation.incircle, None,
        )
    assert caught.value is error


def test_discarded_native_hull_reports_actual_python(monkeypatch):
    _extension()
    complete = triangulation._complete_hull_seed
    checks = []
    reject_next = [True]
    def reject_first_seed(*args):
        checks.append(None)
        if reject_next:
            reject_next.pop()
            return False
        return complete(*args)
    monkeypatch.setattr(triangulation, "_complete_hull_seed", reject_first_seed)
    outer = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    actual = triangulation.triangulate_polygon(outer, backend="python", _reference_native_seed=True)
    reject_next.append(True)
    expected = triangulation.triangulate_polygon(outer, backend="python")
    _same_result(actual, expected)
    assert actual.actual_backend == actual.backend == "python"
    assert actual.selected_backend == "anymesher-cpp17-reference-seed"
    assert actual.fallback_reason == "reference_seed_incomplete_hull"
    assert actual.native_diagnostics == {
        "reference_native_seed": True, "insertion_discarded": True,
    }
    assert len(checks) == 4  # initial rejection and real finite-hull check, twice


def test_extreme_coordinates_are_reference_fallback():
    _extension()
    points = _grid(3) * 2.**-80
    result, reason = native._run_reference_native_seed(
        points, triangulation.orient2d, triangulation.incircle, None,
    )
    assert result is None and reason == "reference_seed_coordinate_range"


def test_registered_custom_boundary_is_not_used_by_private_reference_route():
    _extension()
    class Custom:
        name = "custom"
        def triangulate(self, *args):
            raise AssertionError("Python reference seeds never invoke registered custom boundaries")
    previous = native.register_native_boundary(Custom())
    try:
        result = triangulation.triangulate_polygon(
            [[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
            backend="python", _reference_native_seed=True,
        )
        assert result.actual_backend == "anymesher-cpp17-reference-seed"
    finally:
        native.register_native_boundary(previous)


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, KeyboardInterrupt])
def test_native_callback_exception_identity_is_not_fallback(error_type):
    _extension()
    error = error_type("cancel sentinel")
    def cancel(phase):
        raise error
    with pytest.raises(error_type) as caught:
        native._run_reference_native_seed(
            _grid(5), triangulation.orient2d, triangulation.incircle, cancel,
        )
    assert caught.value is error


def test_ambiguous_predicate_failure_is_not_native_refusal():
    _extension()
    error = RuntimeError("oracle sentinel")
    def circle(*args):
        raise error
    with pytest.raises(RuntimeError) as caught:
        native._run_reference_native_seed(_grid(5), triangulation.orient2d, circle, None)
    assert caught.value is error


def test_public_native_insertion_cancellation_is_atomic():
    _extension()
    from test_curved_native_qualification import _model_face
    from test_cylindrical_frontal_integration import _persistent_state
    from quad_first_curved.test_ch8_conical_public import _generate
    model, face, _ = _model_face("cone")
    before = _persistent_state(model)
    error = RuntimeError("native seed cancellation")
    def cancel(phase):
        if phase == "native triangulation insertion":
            raise error
    with pytest.raises(RuntimeError) as caught:
        _generate(model, (face,), .6, order="quadratic", cancellation_check=cancel)
    assert caught.value is error
    assert _persistent_state(model) == before


def test_seed_revision_mutation_is_still_refused():
    _extension()
    from anymesher.errors import MeshError
    from anymesher.quad.domain import ConicalQuadDomain
    from anymesher.quad.seed import build_planar_quad_seed
    from test_curved_native_qualification import _model_face
    model, face, _ = _model_face("cone")
    domain = ConicalQuadDomain.from_geometry(model, face)
    changed = []
    def mutate(phase):
        if phase == "native triangulation insertion" and not changed:
            changed.append(model.add_point(100., 100., 100.))
    with pytest.raises(MeshError, match="source geometry changed|geometry revision|stale"):
        build_planar_quad_seed(model, face, .6, domain=domain, cancellation_check=mutate)
    assert changed


def test_original_ch9_point_three_mesh_matches_complete_reference(monkeypatch, record_property):
    _extension()
    from anymesher.quad import seed
    from anymesher.serialize import mesh_to_dict
    from test_curved_native_qualification import _model_face
    from test_cylindrical_frontal_integration import _persistent_state
    from quad_first_curved.test_ch8_conical_public import _generate

    model, face, _ = _model_face("cone")
    persistent = _persistent_state(model)
    insertion = triangulation._run_reference_native_seed
    attempts = []
    def observe_insertion(points, *args):
        result, reason = insertion(points, *args)
        nonzero = np.abs(points[points != 0.])
        attempts.append({
            "points": len(points), "native_returned": result is not None,
            "reason": reason,
            "minimum_nonzero": float(np.min(nonzero, initial=np.inf)),
            "maximum_absolute": float(np.max(np.abs(points), initial=0.)),
        })
        if result is not None:
            # Prove insertion parity even if the unchanged hull validator later
            # discards this attempt and constructs a finite reference hull.
            assert result == triangulation._bowyer_watson(points)
        else:
            assert reason == "reference_seed_coordinate_range", attempts
            assert (not np.isfinite(points).all()
                    or np.max(np.abs(points), initial=0.) > 2.**64
                    or np.min(nonzero, initial=np.inf) < 2.**-64)
        return result, reason
    monkeypatch.setattr(triangulation, "_run_reference_native_seed", observe_insertion)
    original = seed.triangulate_polygon
    completed = []
    def observe_seed(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append({
            "actual": result.actual_backend, "selected": result.selected_backend,
            "reason": result.fallback_reason, "diagnostics": result.native_diagnostics,
        })
        return result
    monkeypatch.setattr(seed, "triangulate_polygon", observe_seed)
    actual = _generate(model, (face,), .3, order="quadratic").mesh
    record_property("ch9_native_attempts", repr(attempts))
    record_property("ch9_completed_seeds", repr(completed))
    assert attempts and len(attempts) == len(completed)
    for attempt, result in zip(attempts, completed):
        if attempt["native_returned"]:
            assert result["selected"] == "anymesher-cpp17-reference-seed"
            if result["actual"] == "python":
                assert result["reason"] == "reference_seed_incomplete_hull"
                assert result["diagnostics"]["insertion_discarded"] is True
        else:
            assert result["actual"] == "python"
            assert result["reason"] == "reference_seed_coordinate_range"
    def python_seed(*args, **kwargs):
        kwargs["_reference_native_seed"] = False
        return original(*args, **kwargs)
    monkeypatch.setattr(seed, "triangulate_polygon", python_seed)
    expected = _generate(model, (face,), .3, order="quadratic").mesh
    assert _persistent_state(model) == persistent
    a, b = deepcopy(mesh_to_dict(actual)), mesh_to_dict(expected)
    # Only the explicit implementation provenance differs; every physical
    # field, quality report, warning and counter must remain in the comparison.
    rows = a["hybrid_diagnostics"]["front"]["faces"]
    for row in rows.values():
        row["seed_triangulation_backend"] = "python"
    assert a == b


def test_constrained_annular_chart_has_successful_native_insertion():
    _extension()
    angles = np.linspace(.2, .9, 9)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    outer = np.concatenate((2. * directions, directions[::-1])) + 10.
    interior = 1.5 * directions[1:-1] + 10.
    expected = triangulation.triangulate_polygon(
        outer, interior_points=interior, backend="python",
    )
    actual = triangulation.triangulate_polygon(
        outer, interior_points=interior, backend="python", _reference_native_seed=True,
    )
    _same_result(actual, expected)
    assert actual.actual_backend == "anymesher-cpp17-reference-seed"
    assert actual.native_diagnostics == {"reference_native_seed": True}
