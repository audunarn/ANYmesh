"""Adapter for optional cancellable scalar gradation; no semantic fallback."""
import numpy as np

from .errors import MeshError


def native_cancellable_gradation(points, edges, values, growth, iterations, callback):
    from . import native_cpp
    name = "native_v2_gradation_limit_cancellable"
    if not hasattr(native_cpp._compiled, name):
        return None  # A supported older extension keeps the Python oracle.
    kernel = getattr(native_cpp._compiled, name)
    if not callable(kernel):
        raise MeshError("compiled cancellable gradation capability is malformed")
    # The public metric boundary validates inputs. The kernel copies the value
    # array before changing it, and invokes the host without wrapping its error.
    raw, count = kernel(points, edges, values, growth, iterations, callback)
    limited = np.asarray(raw)
    if (limited.dtype.kind != "f" or limited.shape != values.shape
            or not np.all(np.isfinite(limited)) or np.any(limited <= 0.)
            or np.any(limited > values) or type(count) is not int
            or not 1 <= count <= iterations):
        raise MeshError("cancellable metric gradation returned an invalid result")
    require_gradation_convergence(points, edges, limited, growth, callback)
    callback("native-v2 metric gradation complete")
    return np.ascontiguousarray(limited, dtype=np.float64), count


def require_gradation_convergence(points, edges, values, growth, callback=None):
    """Retain the reference's bounded-sweep acceptance test in small batches."""
    for start in range(0, len(edges), 4096):
        if callback is not None:
            callback("native-v2 metric gradation validation")
        rows = edges[start:start + 4096]
        first, second = rows[:, 0], rows[:, 1]
        distance = np.linalg.norm(points[second] - points[first], axis=1)
        allowance = (growth - 1.0) * distance
        tolerance = 64.0 * np.finfo(float).eps * np.maximum.reduce((
            values[first], values[second], allowance, np.ones(len(rows)),
        ))
        if np.any(values[second] > values[first] + allowance + tolerance) or np.any(
            values[first] > values[second] + allowance + tolerance
        ):
            raise MeshError(
                "metric gradation did not converge within the bounded sweep budget"
            )
