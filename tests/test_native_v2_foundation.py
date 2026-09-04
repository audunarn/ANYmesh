from __future__ import annotations

import json
import _thread
import threading

import numpy as np
import pytest
import anymesher.native_v2 as native_v2_module

from anymesher import (
    ComponentSeedRegistry,
    ExperimentalMetricProvider,
    FeatureDistanceMetricControl,
    ImportedMetricSamples,
    IsotropicMetricControl,
    MetricFieldSpec,
    MutableT3Topology,
    NativeMeshingOptions,
    SpatialMetricField,
    limit_metric_gradation,
    metric_length,
    pullback_metric,
)
from anymesher.errors import MeshError
from anymesher.native_cpp import COMPILED_NATIVE_V2_AVAILABLE


def test_native_options_are_strict_canonical_and_do_not_expose_future_modes() -> None:
    options = NativeMeshingOptions(
        point_placement="frontal_delaunay",
        metric_mode="isotropic_spatial",
        metric_field=MetricFieldSpec.uniform(0.2),
        max_insertions=32,
    )
    encoded = json.dumps(options.to_dict(), sort_keys=True, separators=(",", ":"))
    assert NativeMeshingOptions.from_dict(json.loads(encoded)) == options
    assert len(options.canonical_digest) == 64
    assert "q_morph" not in encoded
    with pytest.raises(MeshError, match="point_placement"):
        NativeMeshingOptions(point_placement="q_morph")
    with pytest.raises(MeshError, match="unknown field"):
        NativeMeshingOptions.from_dict({**options.to_dict(), "provider": "openai"})


def test_runtime_metric_provider_is_explicitly_not_serializable() -> None:
    provider = ExperimentalMetricProvider(
        lambda points: np.repeat(np.eye(2)[None, :, :], len(points), axis=0)
    )
    options = NativeMeshingOptions(
        point_placement="frontal_delaunay",
        metric_mode="isotropic_spatial",
        experimental_metric_provider=provider,
    )
    with pytest.raises(MeshError, match="not serializable"):
        options.to_dict()


def test_metric_composition_binding_pullback_and_gradation() -> None:
    spec = MetricFieldSpec(
        IsotropicMetricControl(1.0),
        feature_controls=(
            FeatureDistanceMetricControl(((0.0, 0.0),), 0.25, 0.1, 1.5, "hole"),
        ),
        imported_samples=(
            ImportedMetricSamples(
                "model", 3, ((1.0, 0.0),), (((25.0, 0.0), (0.0, 25.0)),)
            ),
        ),
    )
    field = SpatialMetricField(spec, model_uuid="model", geometry_revision=3)
    tensors = field.evaluate(np.asarray(((0.0, 0.0), (1.0, 0.0))))
    assert tensors.shape == (2, 2, 2)
    # The nearest imported sample is the strongest control and therefore
    # conservatively dominates the feature-distance value at this query.
    assert metric_length((0.0, 0.0), (0.25, 0.0), tensors[0]) == pytest.approx(1.25)
    np.testing.assert_allclose(
        pullback_metric(np.eye(3), np.asarray(((2.0, 0.0), (0.0, 3.0), (0.0, 0.0)))),
        np.diag((4.0, 9.0)),
    )
    with pytest.raises(MeshError, match="stale"):
        SpatialMetricField(spec, model_uuid="model", geometry_revision=4).evaluate(
            np.asarray(((0.0, 0.0),))
        )
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    limited, iterations = limit_metric_gradation(points, edges, (0.1, 1.0, 10.0), 1.5)
    assert iterations >= 1
    assert limited[1] < 1.0 and limited[2] < 10.0


def test_imported_metric_composition_is_order_independent() -> None:
    first = ImportedMetricSamples(
        "model", 3, ((0.0, 0.0),), (((25.0, 0.0), (0.0, 1.0)),), "x"
    )
    second = ImportedMetricSamples(
        "model", 3, ((0.0, 0.0),), (((1.0, 0.0), (0.0, 25.0)),), "y"
    )
    forward = SpatialMetricField(
        MetricFieldSpec(IsotropicMetricControl(1.0), imported_samples=(first, second)),
        model_uuid="model",
        geometry_revision=3,
    ).evaluate(((0.0, 0.0),))
    reverse = SpatialMetricField(
        MetricFieldSpec(IsotropicMetricControl(1.0), imported_samples=(second, first)),
        model_uuid="model",
        geometry_revision=3,
    ).evaluate(((0.0, 0.0),))
    np.testing.assert_allclose(forward, reverse, rtol=0.0, atol=0.0)


def _square_topology() -> MutableT3Topology:
    return MutableT3Topology(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        np.asarray(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64),
    )


def test_mutable_t3_insertion_is_atomic_canonical_and_protected() -> None:
    topology = _square_topology()
    before = topology.canonical_export()
    report = topology.insert_point((0.5, 0.5))
    points, triangles = topology.canonical_export()
    assert report["point_id"] == 4
    assert len(points) == 5
    assert len(triangles) == 4
    assert list(map(tuple, triangles)) == sorted(map(tuple, triangles))
    with pytest.raises(MeshError, match="protected edge"):
        topology.insert_point((0.5, 0.0))
    after = topology.canonical_export()
    np.testing.assert_array_equal(after[0], points)
    np.testing.assert_array_equal(after[1], triangles)
    assert before[0].flags.writeable is False


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_mutable_t3_python_cpp_parity() -> None:
    reference = _square_topology()
    expected, _ = reference._python_insert(np.asarray((0.5, 0.5)))
    compiled = _square_topology()
    compiled.insert_point((0.5, 0.5))
    np.testing.assert_array_equal(compiled.triangles, expected)


def test_component_seed_registry_reuses_reduced_rational_station() -> None:
    registry = ComponentSeedRegistry(100)
    assert registry.resolve(7, 1, 2) == 100
    assert registry.resolve(7, 2, 4) == 100
    assert registry.resolve(7, 3, 4) == 101


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_native_v2_rejects_unaligned_buffers_at_the_abi_boundary() -> None:
    from anymesher.native_cpp import native_metric_lengths

    raw = bytearray(1 + 4 * np.dtype(np.float64).itemsize)
    points = np.ndarray((2, 2), dtype=np.float64, buffer=raw, offset=1)
    points[:] = ((0.0, 0.0), (1.0, 0.0))
    edges = np.asarray(((0, 1),), dtype=np.int64)
    tensors = np.asarray(
        ((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)), dtype=np.float64
    )
    assert points.flags.c_contiguous and not points.flags.aligned
    with pytest.raises(TypeError, match="aligned C-contiguous"):
        native_metric_lengths(points, edges, tensors)


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_native_v2_uncertain_incircle_falls_back_to_adaptive_python() -> None:
    from anymesher.native_cpp import native_mutable_t3_insert

    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    protected = np.empty((0, 2), dtype=np.int64)
    assert native_mutable_t3_insert(
        points, triangles, protected, (1.0, 1.0)
    ) is None


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_native_v2_physical_gradation_matches_python_oracle() -> None:
    from anymesher.native_cpp import native_gradation_limit

    points = np.asarray(((0.0, 0.0), (0.1, 0.0), (10.0, 0.0)))
    edges = np.asarray(((0, 1), (1, 2)), dtype=np.int64)
    expected, expected_iterations = limit_metric_gradation(
        points, edges, (1.0, 100.0, 100.0), 1.2, max_iterations=8
    )
    actual = native_gradation_limit(points, edges, (1.0, 100.0, 100.0), 1.2, 8)
    assert actual is not None
    np.testing.assert_allclose(actual[0], expected, rtol=0.0, atol=1.0e-15)
    assert actual[1] == expected_iterations


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_native_v2_uncertain_orientation_and_near_cocircle_use_or_match_oracle() -> None:
    from anymesher.native_cpp import native_mutable_t3_insert

    protected = np.empty((0, 2), dtype=np.int64)
    almost_collinear = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 2.0**-55)))
    assert native_mutable_t3_insert(
        almost_collinear,
        np.asarray(((0, 1, 2),), dtype=np.int64),
        protected,
        (0.5, 0.25),
    ) is None

    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    for offset in (-2.0**-48, 0.0, 2.0**-48):
        candidate = (1.0, 1.0 + offset)
        actual = native_mutable_t3_insert(points, triangles, protected, candidate)
        if actual is not None:
            oracle = MutableT3Topology(points, triangles)._python_insert(
                np.asarray(candidate)
            )[0]
            np.testing.assert_array_equal(actual[0], oracle)


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_native_v2_material_loop_honors_python_signal() -> None:
    from anymesher.native_cpp import native_mutable_t3_insert

    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    triangles = np.tile(np.asarray(((0, 1, 2),), dtype=np.int64), (150_000, 1))
    timer = threading.Timer(0.001, _thread.interrupt_main)
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            native_mutable_t3_insert(
                points,
                triangles,
                np.empty((0, 2), dtype=np.int64),
                (0.2, 0.2),
            )
    finally:
        timer.cancel()
        timer.join()


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_native_v2_translation_covariant_insert_and_normalized_diagnostics() -> None:
    from anymesher.native_cpp import (
        NATIVE_V2_SIGNAL_CHECK_INTERVAL,
        native_mutable_t3_insert,
        native_v2_insert_provenance,
        normalized_native_v2_insert_diagnostics,
    )

    offset = np.asarray((1.0e9, -1.0e9))
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    protected = np.asarray(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64)
    origin = native_mutable_t3_insert(
        points, triangles, protected, (0.5, 0.5)
    )
    translated = native_mutable_t3_insert(
        np.ascontiguousarray(points + offset),
        triangles,
        protected,
        offset + (0.5, 0.5),
    )
    assert origin is not None and translated is not None
    np.testing.assert_array_equal(origin[0], translated[0])
    assert normalized_native_v2_insert_diagnostics(
        origin[1]
    ) == normalized_native_v2_insert_diagnostics(translated[1])
    assert native_v2_insert_provenance(origin[1]) == "anymesher-cpp17"
    assert NATIVE_V2_SIGNAL_CHECK_INTERVAL == 4096


def test_loaded_extension_with_zero_native_v2_symbols_fails_hard(monkeypatch) -> None:
    import anymesher.native_cpp as native_cpp

    class LegacyOnlyExtension:
        pass

    monkeypatch.setattr(native_cpp, "_compiled", LegacyOnlyExtension())
    monkeypatch.setattr(native_cpp, "COMPILED_NATIVE_V2_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="incomplete ABI"):
        native_cpp.native_metric_lengths(
            np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64),
            np.asarray(((0, 1),), dtype=np.int64),
            np.asarray(((1.0, 0.0, 1.0), (1.0, 0.0, 1.0)), dtype=np.float64),
        )


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE, reason="optional C++17 module absent")
def test_large_native_metric_and_gradation_kernels_match_python_oracle() -> None:
    from anymesher.native_cpp import native_gradation_limit, native_metric_lengths

    count = 4097
    x = np.linspace(-2.0, 3.0, count, dtype=np.float64)
    points = np.column_stack((x, np.sin(x) * 0.125))
    edges = np.column_stack(
        (np.arange(count - 1, dtype=np.int64), np.arange(1, count, dtype=np.int64))
    )
    tensors = np.repeat(
        np.asarray(((4.0, 0.5, 2.0),), dtype=np.float64), count, axis=0
    )
    delta = points[edges[:, 1]] - points[edges[:, 0]]
    expected_lengths = np.sqrt(
        4.0 * delta[:, 0] ** 2
        + delta[:, 0] * delta[:, 1]
        + 2.0 * delta[:, 1] ** 2
    )
    actual_lengths = native_metric_lengths(points, edges, tensors)
    assert actual_lengths is not None
    np.testing.assert_allclose(actual_lengths, expected_lengths, rtol=2e-15, atol=0.0)

    targets = np.linspace(0.04, 0.4, count, dtype=np.float64)
    expected_targets, expected_iterations = limit_metric_gradation(
        points, edges, targets, 1.35, max_iterations=8
    )
    native_result = native_gradation_limit(points, edges, targets, 1.35, 8)
    assert native_result is not None
    actual_targets, actual_iterations = native_result
    np.testing.assert_allclose(actual_targets, expected_targets, rtol=2e-15, atol=0.0)
    assert actual_iterations == expected_iterations


def test_native_v2_operational_failure_is_not_geometry_limited(monkeypatch) -> None:
    import anymesher.native_cpp as native_cpp

    class BrokenKernel:
        @staticmethod
        def native_v2_metric_lengths(*_args):
            return ()

        @staticmethod
        def native_v2_gradation_limit(*_args):
            return (), 0

        @staticmethod
        def native_v2_mutable_t3_insert(*_args):
            raise RuntimeError("allocator or ABI failure")

    monkeypatch.setattr(native_cpp, "COMPILED_NATIVE_V2_AVAILABLE", True)
    monkeypatch.setattr(native_cpp, "_compiled", BrokenKernel())
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    with pytest.raises(RuntimeError, match="allocator or ABI failure"):
        native_cpp.native_mutable_t3_insert(
            points,
            triangles,
            np.empty((0, 2), dtype=np.int64),
            (0.2, 0.2),
        )


def test_mutable_t3_preserves_retained_owners_and_fails_hard_on_native_error(monkeypatch) -> None:
    topology = MutableT3Topology(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (3.0, 0.0), (4.0, 0.0), (3.0, 1.0))),
        np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64),
        triangle_owners=np.asarray((10, 20), dtype=np.int64),
    )
    topology.insert_point((0.2, 0.2), owner=30)
    assert 20 in topology.triangle_owners
    assert set(topology.triangle_owners).issubset({20, 30})

    def fail_native(*_args):
        raise MeshError("compiled kernel failure")

    monkeypatch.setattr(native_v2_module, "native_mutable_t3_insert", fail_native)
    with pytest.raises(MeshError, match="compiled kernel failure"):
        _square_topology().insert_point((0.5, 0.5))


def test_shared_segment_split_uses_registry_and_protected_edges_remain_fixed() -> None:
    registry = ComponentSeedRegistry(200)
    made = MutableT3Topology(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        np.asarray(((0, 1),), dtype=np.int64),
        splittable_edges={(0, 2): 17},
        seed_registry=registry,
    )
    report = made.split_segment((0, 2))
    assert report["shared_node_id"] == 200
    assert report["station"] == (1, 2)
    assert len(made.triangles) == 4
    with pytest.raises(MeshError, match="unsplittable"):
        made.split_segment((0, 1))


def test_mutable_insertion_cancellation_is_atomic() -> None:
    topology = _square_topology()
    before = topology.canonical_export()

    def cancel(phase: str) -> None:
        if phase == "native-v2 mutable insertion commit":
            raise RuntimeError("cancel commit")

    with pytest.raises(RuntimeError, match="cancel commit"):
        topology.insert_point((0.5, 0.5), cancellation_check=cancel)
    after = topology.canonical_export()
    np.testing.assert_array_equal(after[0], before[0])
    np.testing.assert_array_equal(after[1], before[1])
