"""Cancellation, oracle parity and malformed-result checks for gradation."""
import numpy as np
import pytest

from anymesher import metric, native_cpp
from anymesher.errors import MeshError


def require_kernel():
    if not native_cpp.COMPILED_NATIVE_V2_AVAILABLE:
        pytest.skip("optional native-v2 extension is absent")
    assert callable(getattr(native_cpp._compiled,
                            "native_v2_gradation_limit_cancellable", None))


def inputs():
    x = np.arange(16385, dtype=float) * .003
    points = np.column_stack((x, .01 * np.sin(x)))
    edges = np.column_stack((np.arange(16384), np.arange(1, 16385))).astype(np.int64)
    values = np.ones(len(points))
    values[0] = .05
    return points, edges, values


def test_compiled_callback_path_matches_python_oracle(monkeypatch):
    require_kernel()
    points, edges, values = inputs()
    phases = []
    actual, count = metric.limit_metric_gradation(
        points, edges, values, 1.5, cancellation_check=phases.append,
    )
    assert "native-v2 compiled gradation work" in phases
    with monkeypatch.context() as patch:
        patch.setattr(metric, "_native_v2_available", lambda: False)
        expected, expected_count = metric.limit_metric_gradation(
            points, edges, values, 1.5, cancellation_check=lambda phase: None,
        )
    np.testing.assert_allclose(actual, expected, rtol=1.e-14, atol=1.e-14)
    assert count == expected_count
    assert actual[-1] == 1. and actual[0] == .05
    assert np.all(actual <= values)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("limit", [1, 4])
def test_order_and_iteration_limit_match_reference(monkeypatch, reverse, limit):
    require_kernel()
    points, edges, values = inputs()
    if reverse:
        edges = edges[::-1].copy()
        for use_native in (True, False):
            with monkeypatch.context() as patch:
                if not use_native:
                    patch.setattr(metric, "_native_v2_available", lambda: False)
                with pytest.raises(MeshError, match="did not converge"):
                    metric.limit_metric_gradation(
                        points, edges, values, 1.5, max_iterations=limit,
                        cancellation_check=lambda phase: None,
                    )
        return
    actual, count = metric.limit_metric_gradation(
        points, edges, values, 1.5, max_iterations=limit,
        cancellation_check=lambda phase: None,
    )
    with monkeypatch.context() as patch:
        patch.setattr(metric, "_native_v2_available", lambda: False)
        expected, expected_count = metric.limit_metric_gradation(
            points, edges, values, 1.5, max_iterations=limit,
            cancellation_check=lambda phase: None,
        )
    np.testing.assert_allclose(actual, expected, rtol=1.e-14, atol=1.e-14)
    assert count == expected_count


@pytest.mark.parametrize("phase", ["prepare", "input", "work", "output"])
def test_native_phase_cancellation_keeps_input_and_exception(phase):
    require_kernel()
    points, edges, values = inputs()
    original = values.copy()
    error = RuntimeError("cancel metric")

    def callback(stage):
        if stage == "native-v2 compiled gradation " + phase:
            raise error

    with pytest.raises(RuntimeError) as caught:
        metric.limit_metric_gradation(
            points, edges, values, 1.5, cancellation_check=callback,
        )
    assert caught.value is error
    np.testing.assert_array_equal(values, original)


def test_late_work_cancellation_is_not_postponed_to_export():
    require_kernel()
    points, edges, values = inputs()
    before = values.copy()
    visits = []

    def callback(phase):
        if phase == "native-v2 compiled gradation work":
            visits.append(phase)
            if len(visits) == 3:
                raise KeyboardInterrupt("cancel inside sweep")

    with pytest.raises(KeyboardInterrupt, match="inside sweep"):
        metric.limit_metric_gradation(
            points, edges, values, 1.5, cancellation_check=callback,
        )
    assert len(visits) == 3
    np.testing.assert_array_equal(values, before)


def test_absent_new_capability_keeps_cancellable_reference(monkeypatch):
    require_kernel()
    points, edges, values = inputs()
    monkeypatch.delattr(native_cpp._compiled,
                        "native_v2_gradation_limit_cancellable")
    phases = []
    actual, count = metric.limit_metric_gradation(
        points, edges, values, 1.5, cancellation_check=phases.append,
    )
    assert "native-v2 metric gradation scan" in phases
    assert "native-v2 compiled gradation work" not in phases
    assert np.all(actual <= values)


@pytest.mark.parametrize("defect", ["nan", "increase", "shape", "iterations", "fault", "capability", "unconverged"])
def test_present_bad_kernel_never_falls_back(monkeypatch, defect):
    require_kernel()
    points, edges, values = inputs()

    def kernel(*args):
        if defect == "fault":
            raise RuntimeError("kernel fault")
        rows, count = values.copy(), 2
        if defect == "nan":
            rows[0] = np.nan
        elif defect == "increase":
            rows[-1] = 2.
        elif defect == "shape":
            rows = rows[:-1]
        elif defect == "iterations":
            count = True
        return rows, count

    monkeypatch.setattr(native_cpp._compiled,
                        "native_v2_gradation_limit_cancellable",
                        None if defect == "capability" else kernel)
    before = values.copy()
    with pytest.raises((MeshError, RuntimeError)):
        metric.limit_metric_gradation(
            points, edges, values, 1.5, cancellation_check=lambda phase: None,
        )
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("with_callback", [False, True])
def test_both_native_paths_reject_unconverged_budget(with_callback):
    require_kernel()
    points, edges, values = inputs()
    before = values.copy()
    with pytest.raises(MeshError, match="did not converge"):
        metric.limit_metric_gradation(
            points, edges[::-1].copy(), values, 1.5, max_iterations=1,
            cancellation_check=(lambda phase: None) if with_callback else None,
        )
    np.testing.assert_array_equal(values, before)


def test_cancellation_during_post_kernel_convergence_keeps_input():
    require_kernel()
    points, edges, values = inputs()
    before = values.copy()
    error = RuntimeError("cancel final convergence")

    def cancel(phase):
        if phase == "native-v2 metric gradation validation":
            raise error

    with pytest.raises(RuntimeError) as caught:
        metric.limit_metric_gradation(
            points, edges, values, 1.5, cancellation_check=cancel,
        )
    assert caught.value is error
    np.testing.assert_array_equal(values, before)


def test_convergence_guard_uses_the_reference_tolerance():
    from anymesher._cancellable_gradation import require_gradation_convergence

    points = np.array([[0., 0.], [1., 0.]])
    edges = np.array([[0, 1]])
    tolerance = 64. * np.finfo(float).eps
    require_gradation_convergence(
        points, edges, np.array([.25, .75 + tolerance / 2]), 1.5,
    )
    with pytest.raises(MeshError, match="did not converge"):
        require_gradation_convergence(
            points, edges, np.array([.25, .75 + tolerance * 2]), 1.5,
        )
