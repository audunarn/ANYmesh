"""Raw-metric reuse must be exact, bounded, and failure-atomic."""
import importlib
from pathlib import Path

import numpy as np
import pytest

from anymesher._point_metric_cache import PointMetricCache
from anymesher.errors import MeshError


def tensor(points):
    scales = 1. + points[:, 0] ** 2 + points[:, 1] ** 2
    return np.eye(2)[None, :, :] * scales[:, None, None]


def test_append_only_evaluates_new_rows_and_returns_defensive_arrays():
    calls = []
    def evaluate(points):
        calls.append(points.copy())
        return tensor(points)
    cache = PointMetricCache(evaluate, max_rows=10, batch_rows=2)
    original = np.asarray(((0., .5), (1., .25), (2., .5)))
    first = cache.evaluate(original)
    np.testing.assert_array_equal(first, tensor(original))
    assert [len(row) for row in calls] == [2, 1]
    appended = np.vstack((original, (3., .75)))
    result = cache.evaluate(appended)
    assert [len(row) for row in calls] == [2, 1, 1]
    assert cache.evaluated_rows == 4
    np.testing.assert_array_equal(result, tensor(appended))
    result[:] = -1
    np.testing.assert_array_equal(cache.evaluate(appended), tensor(appended))
    assert cache.evaluated_rows == 4 and len(cache._valid) <= 10


def test_changed_reordered_and_signed_zero_rows_are_not_reused():
    cache = PointMetricCache(tensor, max_rows=6)
    points = np.asarray(((0., 1.), (2., 3.), (4., 5.)))
    cache.evaluate(points)
    moved = points.copy()
    moved[1, 0] += .25
    np.testing.assert_array_equal(cache.evaluate(moved), tensor(moved))
    assert cache.evaluated_rows == 4
    reversed_points = moved[::-1].copy()
    np.testing.assert_array_equal(cache.evaluate(reversed_points), tensor(reversed_points))
    assert cache.evaluated_rows == 6
    cache.evaluate(reversed_points[:1])
    cache.evaluate(reversed_points)
    assert cache.evaluated_rows == 6
    reversed_points[2, 0] = -0.
    cache.evaluate(reversed_points)
    assert cache.evaluated_rows == 7


@pytest.mark.parametrize("points", (
    np.asarray(((float("nan"), 0.),)), np.zeros((1, 3)),
    np.zeros((4, 2)), np.zeros(2),
))
def test_bad_coordinates_fail_before_evaluation(points):
    def forbidden(_):
        pytest.fail("invalid query reached evaluator")
    cache = PointMetricCache(forbidden, max_rows=3)
    with pytest.raises(MeshError):
        cache.evaluate(points)
    assert cache.evaluated_rows == 0


@pytest.mark.parametrize("failure", ("exception", "shape", "nan"))
def test_later_batch_failure_keeps_existing_slots_and_counters(failure):
    enabled = False
    count = 0
    error = RuntimeError("provider failure")
    def evaluate(points):
        nonlocal count
        count += 1
        if enabled and count == 3:
            if failure == "exception":
                raise error
            if failure == "shape":
                return np.zeros((1, 3, 3))
            return tensor(points) * float("nan")
        return tensor(points)
    cache = PointMetricCache(evaluate, max_rows=8, batch_rows=2)
    old = np.asarray(((0., 1.), (1., 1.)))
    expected = cache.evaluate(old)
    enabled = True
    with pytest.raises((RuntimeError, MeshError)) as caught:
        cache.evaluate(np.vstack((old, (2., 1.), (3., 1.), (4., 1.))))
    if failure == "exception":
        assert caught.value is error
    assert cache.evaluated_rows == 2
    np.testing.assert_array_equal(cache.evaluate(old), expected)


def test_cancellation_before_commit_does_not_cache_pending_rows():
    enabled = False
    error = RuntimeError("cancel metric cache")
    def cancel(phase):
        if enabled and phase == "native-v2 metric cache commit":
            raise error
    cache = PointMetricCache(tensor, max_rows=5, cancellation_check=cancel)
    points = np.asarray(((0., .5), (1., .5)))
    original = cache.evaluate(points)
    enabled = True
    with pytest.raises(RuntimeError) as caught:
        cache.evaluate(np.vstack((points, (2., .5))))
    assert caught.value is error and cache.evaluated_rows == 2
    enabled = False
    np.testing.assert_array_equal(cache.evaluate(points), original)


def test_real_position_dependent_pullback_matches_uncached_bytes():
    from anymesher import FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec
    from anymesher.metric import SpatialMetricField
    from anymesher.native_v2 import _pullback_spatial_metrics
    field = SpatialMetricField(MetricFieldSpec(
        IsotropicMetricControl(.4),
        feature_controls=(FeatureDistanceMetricControl(((1., 0., 0.),), .1, .3, 1.5, "zone"),),
    ))
    def derivative(points):
        result = np.zeros((len(points), 3, 2))
        result[:, 0, 0] = -np.sin(points[:, 0])
        result[:, 1, 0] = np.cos(points[:, 0])
        result[:, 2, 1] = 1.
        return result
    def evaluate(points):
        physical = np.column_stack((np.cos(points[:, 0]), np.sin(points[:, 0]), points[:, 1]))
        return _pullback_spatial_metrics(field.evaluate(physical), points, derivative)
    points = np.asarray(((0., 0.), (.1, .2), (.3, .1), (.5, .4), (.7, .5)))
    cache = PointMetricCache(evaluate, max_rows=8, batch_rows=2)
    cache.evaluate(points[:3])
    assert cache.evaluate(points).tobytes() == evaluate(points).tobytes()
    assert cache.evaluated_rows == len(points)


def test_cylindrical_public_mesh_is_byte_exact_without_cache(monkeypatch):
    from anymesher import _point_metric_cache
    from anymesher.hybrid import generate_hybrid_mesh_result, _neutral_shell_core
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    bench = importlib.import_module("native_v2_baseline")
    fixture = bench.cylinder_case("cylinder_patch")
    counts = []
    original = PointMetricCache.evaluate
    def observed(self, points):
        result = original(self, points)
        counts.append((len(points), self.evaluated_rows))
        return result
    monkeypatch.setattr(_point_metric_cache.PointMetricCache, "evaluate", observed)
    def generate():
        return generate_hybrid_mesh_result(
            fixture.model, face_ids=fixture.face_ids, target_size=.4, strategy="native",
            overrides=fixture.overrides(.4), refinements=fixture.refinements(.4),
            native_backend="python", native_options=fixture.options(.4, "frontal", 128),
            recombine=True,
        )
    cached = generate()
    assert counts and sum(row[0] for row in counts) > counts[-1][1]
    monkeypatch.setattr(
        _point_metric_cache.PointMetricCache, "evaluate",
        lambda self, points: self.evaluator(np.ascontiguousarray(points, dtype=np.float64)),
    )
    reference = generate()
    assert bench._mesh_digest(_neutral_shell_core(cached.mesh)) == (
        bench._mesh_digest(_neutral_shell_core(reference.mesh)))
    assert bench._digest_contract(fixture.mesh_contract(cached.mesh, .4)) == (
        bench._digest_contract(fixture.mesh_contract(reference.mesh, .4)))
    assert cached.triangulation_backend_by_face[1]["native_v2"] == (
        reference.triangulation_backend_by_face[1]["native_v2"])
