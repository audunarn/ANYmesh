"""Small memory-shape and exact-distance regressions, not benchmarks."""

import numpy as np
import pytest

import anymesher.metric as metric


@pytest.fixture
def small_tiles(monkeypatch):
    monkeypatch.setattr(metric, "_METRIC_QUERY_CHUNK_SIZE", 2)
    monkeypatch.setattr(metric, "_METRIC_SOURCE_CHUNK_SIZE", 3)


def _dense(points, sources, *, return_indices, **kwargs):
    squared = np.sum((points[:, None, :] - sources[None, :, :]) ** 2, axis=2)
    return np.argmin(squared, axis=1) if return_indices else np.min(squared, axis=1)


def _nearest(points, sources, indices=False, callback=None):
    return metric._bounded_metric_nearest(
        points, sources, return_indices=indices,
        cancellation_check=callback, stage="lookup",
    )


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("indices", [False, True])
def test_tiled_reduction_is_bit_exact(small_tiles, dimension, indices):
    random = np.random.default_rng(187)
    points = random.normal(size=(14, dimension))[::2]
    sources = random.normal(size=(16, dimension))[::2]
    points.setflags(write=False)
    sources.setflags(write=False)
    expected = _dense(points, sources, return_indices=indices)
    actual = _nearest(points, sources, indices)
    assert actual.tobytes() == expected.tobytes()


def test_equal_distances_keep_first_source_across_tiles(small_tiles):
    points = np.asarray(((0., 0.), (0.75, 2.)))
    sources = np.asarray(((-1., 0.), (1., 0.), (9., 0.), (-1., 0.), (1., 0.), (.75, 2.)))
    np.testing.assert_array_equal(_nearest(points, sources, True), (0, 5))


def test_overflow_retains_dense_argmin_semantics(small_tiles):
    points = np.zeros((1, 2))
    sources = np.full((5, 2), 1.e308)
    with np.errstate(over="ignore"):
        assert _nearest(points, sources, True)[0] == 0
        assert np.isinf(_nearest(points, sources)[0])
        sources[4] = (1., 0.)
        assert _nearest(points, sources, True)[0] == 4


def test_pairwise_allocation_is_bounded_on_both_axes(small_tiles, monkeypatch):
    original = np.sum
    shapes = []

    def record(values, *args, **kwargs):
        shapes.append(values.shape)
        return original(values, *args, **kwargs)

    monkeypatch.setattr(metric.np, "sum", record)
    _nearest(np.zeros((7, 3)), np.ones((8, 3)))
    assert len(shapes) == 12
    assert all(q <= 2 and s <= 3 and d == 3 for q, s, d in shapes)


@pytest.mark.parametrize("shape", [(1, 8), (7, 1)])
def test_cancellation_between_source_or_query_tiles(small_tiles, shape):
    failure = RuntimeError("cancel tiled lookup")

    def cancel(stage):
        assert stage == "lookup"
        raise failure

    with pytest.raises(RuntimeError) as caught:
        _nearest(np.zeros((shape[0], 2)), np.ones((shape[1], 2)), callback=cancel)
    assert caught.value is failure


def _field(dimension, mode):
    sources = tuple(tuple(float(i + j) / 4 for j in range(dimension)) for i in range(8))
    imported = metric.ImportedMetricSamples(
        "model", 3, sources,
        tuple(tuple(tuple(float(i + 2) if a == b else 0. for b in range(dimension))
                    for a in range(dimension)) for i in range(8)),
    )
    feature = metric.FeatureDistanceMetricControl(
        sources, target_size=.25, influence_distance=.1, growth=1.5,
    )
    spec = metric.MetricFieldSpec(
        metric.IsotropicMetricControl(1.),
        feature_controls=(feature,) if mode != "imported" else (),
        imported_samples=(imported,) if mode != "feature" else (),
    )
    return metric.SpatialMetricField(spec, model_uuid="model", geometry_revision=3)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("mode", ["feature", "imported", "combined"])
@pytest.mark.parametrize("interval", [1, 100000])
def test_public_tensors_match_dense_distance_reference(
    small_tiles, monkeypatch, dimension, mode, interval,
):
    field = _field(dimension, mode)
    points = np.random.default_rng(45).normal(size=(14, dimension))[::2]
    with monkeypatch.context() as reference:
        reference.setattr(metric, "_bounded_metric_nearest", _dense)
        expected = field.evaluate(points, cancellation_interval=interval)
    actual = field.evaluate(points, cancellation_interval=interval)
    assert actual.tobytes() == expected.tobytes()


def test_public_imported_tie_selects_first_tensor(small_tiles):
    samples = metric.ImportedMetricSamples(
        "model", 3, ((-1., 0.), (8., 0.), (9., 0.), (1., 0.)),
        tuple(((value, 0.), (0., value)) for value in (4., 6., 8., 9.)),
    )
    field = metric.SpatialMetricField(
        metric.MetricFieldSpec(metric.IsotropicMetricControl(1.), imported_samples=(samples,)),
        model_uuid="model", geometry_revision=3,
    )
    np.testing.assert_array_equal(field.evaluate(((0., 0.),))[0], np.eye(2) * 4.)


def test_public_empty_query_preserves_completion_checkpoint(small_tiles):
    phases = []
    value = _field(2, "combined").evaluate(np.empty((0, 2)), cancellation_check=phases.append)
    assert value.shape == (0, 2, 2)
    assert phases == ["native-v2 metric evaluation complete"]


@pytest.mark.parametrize("mode,stage", [
    ("feature", "native-v2 feature metric evaluation"),
    ("imported", "native-v2 imported metric lookup"),
])
def test_public_cancellation_propagates_before_completion(small_tiles, mode, stage):
    phases = []
    failure = RuntimeError("cancel metric")

    def cancel(phase):
        phases.append(phase)
        if phases.count(stage) == 2:
            raise failure

    with pytest.raises(RuntimeError) as caught:
        _field(2, mode).evaluate(((0., 0.),), cancellation_check=cancel)
    assert caught.value is failure
    assert "native-v2 metric evaluation complete" not in phases
