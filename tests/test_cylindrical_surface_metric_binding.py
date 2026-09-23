"""Surface-path metric plumbing, not owner atlas or cylinder activation proof.

These synthetic charts isolate the existing private metric callback boundary.
Actual owner chart/trim/seam qualification remains a separate acceptance gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from anymesher import (
    FeatureDistanceMetricControl,
    IsotropicMetricControl,
    MetricFieldSpec,
    NativeMeshingOptions,
)
from anymesher.surface_mesh import SurfaceMeshOptions, mesh_planar_surface


def _surface_case(radius: float, height: float):
    width = radius * np.pi / 2.0
    outer = np.array(((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)))
    physical_batches: list[np.ndarray] = []
    derivative_batches: list[np.ndarray] = []

    def evaluate(points):
        points = np.asarray(points, dtype=np.float64)
        physical_batches.append(points.copy())
        angle = points[:, 0] / radius
        return np.column_stack(
            (radius * np.cos(angle), radius * np.sin(angle), points[:, 1])
        )

    def jacobians(points):
        points = np.asarray(points, dtype=np.float64)
        derivative_batches.append(points.copy())
        angle = points[:, 0] / radius
        result = np.zeros((len(points), 3, 2), dtype=np.float64)
        result[:, 0, 0] = -np.sin(angle)
        result[:, 1, 0] = np.cos(angle)
        result[:, 2, 1] = 1.0
        return result

    # A 3D feature forces physical evaluation/pullback rather than the 2D path.
    center = (radius / np.sqrt(2.0), radius / np.sqrt(2.0), height / 2.0)
    metric = MetricFieldSpec(
        IsotropicMetricControl(0.20),
        feature_controls=(
            FeatureDistanceMetricControl((center,), 0.12, 0.40, 1.5, "physical"),
        ),
    )
    settings = SurfaceMeshOptions(
        target_size=0.45,
        backend="python",
        recombine=False,
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            metric_field=metric,
            max_insertions=8,
            max_topology_operations=2000,
            cancellation_interval=1,
        ),
    )
    return outer, settings, evaluate, jacobians, physical_batches, derivative_batches


@pytest.mark.parametrize("radius,height", ((0.5, 1.0), (1.0, 0.75)))
def test_surface_forwards_position_dependent_physical_metric(radius, height):
    outer, settings, evaluate, jacobians, physical, derivatives = _surface_case(
        radius, height
    )
    original = outer.tobytes()
    mesh = mesh_planar_surface(
        outer,
        options=settings,
        _metric_to_physical=evaluate,
        _metric_jacobian=jacobians,
    )
    assert len(mesh.triangle_connectivity) > 0
    assert physical, "The surface path must evaluate the physical 3D field"
    assert derivatives, "The surface path must evaluate varying chart Jacobians"
    assert all(batch.ndim == 2 and batch.shape[1] == 2 for batch in physical)
    assert all(0 < len(batch) <= 4096 for batch in derivatives)
    assert all(np.all(np.isfinite(batch)) for batch in physical + derivatives)
    assert outer.tobytes() == original


def test_surface_physical_derivative_failure_propagates_without_input_mutation():
    outer, settings, evaluate, _, _, _ = _surface_case(0.5, 1.0)
    original = outer.tobytes()
    failure = RuntimeError("synthetic physical derivative failure")

    def fail_derivatives(points):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        mesh_planar_surface(
            outer,
            options=settings,
            _metric_to_physical=evaluate,
            _metric_jacobian=fail_derivatives,
        )
    assert caught.value is failure
    assert outer.tobytes() == original
