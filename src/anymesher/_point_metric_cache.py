"""Bounded, invocation-local raw chart-metric cache.

The evaluator retains responsibility for its SPD/chart/owner validation. This
cache only avoids repeating that exact evaluation for unchanged coordinate rows.
It never stores topology-dependent gradation or exposes its internal arrays.
"""
from __future__ import annotations

import numpy as np

from .errors import MeshError


class PointMetricCache:
    def __init__(self, evaluator, *, max_rows, batch_rows=4096,
                 cancellation_check=None):
        if (not callable(evaluator) or type(max_rows) is not int or max_rows < 1
                or type(batch_rows) is not int or not 1 <= batch_rows <= 4096):
            raise MeshError("invalid point metric cache bounds")
        self.evaluator = evaluator
        self.max_rows = max_rows
        self.batch_rows = batch_rows
        self.cancellation_check = cancellation_check
        self._points = np.empty((0, 2), dtype=np.float64)
        self._metrics = np.empty((0, 2, 2), dtype=np.float64)
        self._valid = np.empty(0, dtype=bool)
        self.evaluated_rows = 0

    def _checkpoint(self, phase):
        if self.cancellation_check is not None:
            self.cancellation_check(phase)

    def _grow(self, count):
        if count <= len(self._valid):
            return
        capacity = min(self.max_rows, max(count, 16, 2 * len(self._valid)))
        points = np.empty((capacity, 2), dtype=np.float64)
        metrics = np.empty((capacity, 2, 2), dtype=np.float64)
        valid = np.zeros(capacity, dtype=bool)
        old = len(self._valid)
        points[:old], metrics[:old], valid[:old] = self._points, self._metrics, self._valid
        self._points, self._metrics, self._valid = points, metrics, valid

    def evaluate(self, points):
        values = np.ascontiguousarray(points, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != 2
                or len(values) > self.max_rows or not np.isfinite(values).all()):
            raise MeshError("invalid point metric cache coordinates")
        missing = np.ones(len(values), dtype=bool)
        overlap = min(len(values), len(self._valid))
        self._checkpoint("native-v2 metric cache lookup")
        for start in range(0, overlap, self.batch_rows):
            self._checkpoint("native-v2 metric cache lookup")
            stop = min(start + self.batch_rows, overlap)
            different = np.any(
                values[start:stop].view(np.uint64)
                != self._points[start:stop].view(np.uint64), axis=1,
            )
            missing[start:stop] = ~self._valid[start:stop] | different
        indices = np.flatnonzero(missing)
        pending = np.empty((len(indices), 2, 2), dtype=np.float64)
        for start in range(0, len(indices), self.batch_rows):
            self._checkpoint("native-v2 metric cache evaluation")
            stop = min(start + self.batch_rows, len(indices))
            result = np.asarray(self.evaluator(values[indices[start:stop]]), dtype=np.float64)
            if result.shape != (stop - start, 2, 2) or not np.isfinite(result).all():
                raise MeshError("invalid point metric cache evaluator result")
            pending[start:stop] = result
        self._checkpoint("native-v2 metric cache commit")
        # No valid slot is changed until every pending batch has succeeded.
        self._grow(len(values))
        self._points[indices] = values[indices]
        self._metrics[indices] = pending
        self._valid[indices] = True
        self.evaluated_rows += len(indices)
        return self._metrics[:len(values)].copy()
