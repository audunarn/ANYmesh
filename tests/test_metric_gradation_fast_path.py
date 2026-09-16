"""Exact fast paths for metric gradation."""

import numpy as np

from anymesher.metric import limit_metric_gradation


def test_uniform_metric_skips_graph_sweeps_and_preserves_cancellation_boundary():
    points = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=np.float64,
    )
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64)
    target = np.full(len(points), 0.25, dtype=np.float64)
    phases = []

    limited, iterations = limit_metric_gradation(
        points,
        edges,
        target,
        1.5,
        cancellation_check=phases.append,
    )

    assert iterations == 1
    assert limited.tobytes() == target.tobytes()
    assert phases == ["native-v2 uniform metric gradation"]


def test_nonuniform_metric_still_uses_the_gradation_algorithm():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64)
    edges = np.asarray(((0, 1),), dtype=np.int64)
    target = np.asarray((0.1, 1.0), dtype=np.float64)

    limited, iterations = limit_metric_gradation(points, edges, target, 1.5)

    assert iterations >= 1
    np.testing.assert_allclose(limited, (0.1, 0.6))
