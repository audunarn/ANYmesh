"""Parity for invocation-local native gradation edge-length reuse."""

import math

import numpy as np
import pytest


def _reference(points, edges, input_values, growth, max_iterations):
    values = input_values.copy()
    for iteration in range(1, max_iterations + 1):
        changed = False
        for first, second in edges:
            dx = float(points[second, 0] - points[first, 0])
            dy = float(points[second, 1] - points[first, 1])
            maximum_delta = (growth - 1.0) * math.hypot(dx, dy)
            if values[second] > values[first] + maximum_delta:
                values[second] = values[first] + maximum_delta
                changed = True
            if values[first] > values[second] + maximum_delta:
                values[first] = values[second] + maximum_delta
                changed = True
        if not changed:
            return values, iteration
    return values, max_iterations


@pytest.mark.parametrize("max_iterations", [1, 32])
def test_native_gradation_reuses_lengths_without_changing_decisions(max_iterations):
    native = pytest.importorskip("anymesher._native")
    kernel = getattr(native, "native_v2_gradation_limit", None)
    if kernel is None:
        pytest.skip("optional native-v2 gradation kernel unavailable")
    count = 16
    points = np.column_stack((np.arange(count, dtype=np.float64) * 0.25,
                              np.zeros(count, dtype=np.float64)))
    edges = np.asarray([(index, index + 1) for index in range(count - 2, -1, -1)],
                       dtype=np.int64)
    values = np.full(count, 2.0, dtype=np.float64)
    values[0] = 0.2
    original_points = points.copy()
    original_edges = edges.copy()
    original_values = values.copy()

    actual, actual_iterations = kernel(points, edges, values, 1.5, max_iterations)
    expected, expected_iterations = _reference(points, edges, values, 1.5,
                                               max_iterations)

    np.testing.assert_array_equal(np.asarray(actual), expected)
    assert actual_iterations == expected_iterations
    if max_iterations > 1:
        assert actual_iterations > 2
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(edges, original_edges)
    np.testing.assert_array_equal(values, original_values)
