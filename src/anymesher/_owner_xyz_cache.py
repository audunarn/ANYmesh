"""Bounded owner coordinates for one append-only frontal invocation.

Exact parameter keys locate receipts inside an explicitly registered global
point array. They never merge mesh nodes or confer owner validity: every lookup
and registration still performs the chart's public owner binding validation.
"""
from __future__ import annotations

from time import perf_counter
import sys
import numpy as np

from .errors import MeshError


class OwnerXYZCache:
    def __init__(self, evaluator, *, max_rows, batch_rows=4096, cancellation_check=None):
        from ._analytic_metric_chart import AnalyticMetricChart
        chart = getattr(evaluator, '__self__', None)
        if (not isinstance(chart, AnalyticMetricChart)
                or getattr(evaluator, '__func__', None) is not AnalyticMetricChart.evaluate
                or type(max_rows) is not int or max_rows < 1
                or type(batch_rows) is not int or not 1 <= batch_rows <= 4096):
            raise MeshError('invalid owner XYZ cache binding or bounds')
        self.evaluator, self.chart = evaluator, chart
        self.max_rows, self.batch_rows = max_rows, batch_rows
        self.cancellation_check = cancellation_check
        self._owner, self._binding, self._region = chart.owner, chart.binding, chart.region_binding
        self._face = chart.face_id
        self._transform = np.asarray(chart.transform, dtype=np.float64).copy()
        self._inverse = np.asarray(chart.inverse, dtype=np.float64).copy()
        if any(a.shape != (2, 2) or not np.isfinite(a).all() for a in (self._transform,self._inverse)):
            raise MeshError('invalid owner XYZ cache chart transform')
        self._points = np.empty((0, 2), dtype=np.float64)
        self._registered = set()
        self._values = {}
        self.registrations = self.requests = self.requested_rows = self.reused_rows = 0
        self.owner_calls = self.evaluated_rows = self.binding_checks = 0
        self.evaluation_seconds = self.total_seconds = 0.
        self._guard()

    def _checkpoint(self, phase):
        if self.cancellation_check is not None:
            self.cancellation_check(phase)

    def _stamp(self):
        chart = self.chart
        if (chart.owner is not self._owner or chart.binding is not self._binding
                or chart.region_binding is not self._region or chart.face_id != self._face
                or np.asarray(chart.transform).dtype != np.dtype(np.float64)
                or np.asarray(chart.inverse).dtype != np.dtype(np.float64)
                or np.asarray(chart.transform).shape != (2, 2)
                or np.asarray(chart.inverse).shape != (2, 2)
                or np.asarray(chart.transform).tobytes() != self._transform.tobytes()
                or np.asarray(chart.inverse).tobytes() != self._inverse.tobytes()):
            raise MeshError('owner XYZ cache chart binding changed')

    def _guard(self):
        self._stamp()
        self.binding_checks += 1
        # _current delegates to validate_trimmed_surface_charts_binding or the
        # region binding's public validate_material_surface_regions_binding.
        self.chart._current()
        self._stamp()

    @staticmethod
    def _keys(points):
        return [tuple(map(int, row)) for row in points.view(np.uint64)]

    def _rows(self, points):
        # Capture before the first checkpoint: caller-owned contiguous arrays
        # may otherwise change between key construction and owner evaluation.
        values = np.array(points, dtype=np.float64, order='C', copy=True)
        if (values.ndim != 2 or values.shape[1] != 2 or len(values) > self.max_rows
                or not np.isfinite(values).all()):
            raise MeshError('invalid owner XYZ cache coordinates')
        values.setflags(write=False)
        return values

    def register(self, points):
        """Atomically bind the global append-only slots, without evaluating XYZ."""
        started = perf_counter()
        try:
            values = self._rows(points)
            self._checkpoint('native-v2 owner XYZ registration')
            self._guard()
            count = len(self._points)
            if len(values) < count or not np.array_equal(
                    values[:count].view(np.uint64), self._points.view(np.uint64)):
                raise MeshError('owner XYZ cache requires append-only exact coordinates')
            staged = values
            registered = set(self._registered)
            for start in range(count, len(values), self.batch_rows):
                self._checkpoint('native-v2 owner XYZ registration')
                registered.update(self._keys(values[start:start+self.batch_rows]))
            self._checkpoint('native-v2 owner XYZ registration commit')
            self._guard()
            staged.setflags(write=False)
            self._points, self._registered = staged, registered
            self.registrations += 1
        finally:
            self.total_seconds += perf_counter()-started

    def evaluate(self, points):
        """Return ordered copies for registered full arrays or metric subsets."""
        started = perf_counter()
        try:
            values = self._rows(points)
            self._checkpoint('native-v2 owner XYZ lookup')
            self._guard()
            keys = self._keys(values)
            if len(values) > len(self._points) or any(key not in self._registered for key in keys):
                raise MeshError('owner XYZ cache subset is not globally registered')
            missing = [i for i,key in enumerate(keys) if key not in self._values]
            self.requests += 1
            self.requested_rows += len(values)
            self.reused_rows += len(values)-len(missing)
            pending = {}
            for start in range(0,len(missing),self.batch_rows):
                self._checkpoint('native-v2 owner XYZ evaluation')
                indices = missing[start:start+self.batch_rows]
                submitted = values[indices].copy()
                evaluating = perf_counter()
                self.owner_calls += 1
                self.evaluated_rows += len(indices)
                try:
                    result = np.asarray(self.evaluator(submitted), dtype=np.float64)
                finally:
                    self.evaluation_seconds += perf_counter()-evaluating
                if result.shape != (len(indices),3) or not np.isfinite(result).all():
                    raise MeshError('invalid owner XYZ cache evaluator result')
                for row,index in enumerate(indices):
                    key = keys[index]
                    receipt = result[row].copy()
                    previous = pending.setdefault(key,receipt)
                    if previous.tobytes() != receipt.tobytes():
                        raise MeshError('owner XYZ cache duplicate coordinate receipts differ')
                    receipt.setflags(write=False)
            staged = dict(self._values)
            staged.update(pending)
            output = np.asarray([staged[key] for key in keys],dtype=np.float64).reshape(-1,3).copy()
            self._checkpoint('native-v2 owner XYZ commit')
            self._guard()
            # Neither old values nor staged backing escape. Allocate before
            # the final check so a failure cannot publish a partial batch.
            self._values = staged
            return output
        finally:
            self.total_seconds += perf_counter()-started

    def receipt(self):
        # Account owned arrays and Python keys/containers once by object identity;
        # identity is used only for memory accounting, never validity acceptance.
        objects = [self._points,self._transform,self._inverse,self._registered,self._values]
        objects.extend(self._registered)
        objects.extend(self._values.keys())
        objects.extend(self._values.values())
        objects.extend(value for key in self._registered for value in key)
        objects.extend(value for key in self._values for value in key)
        seen = set()
        storage = 0
        for obj in objects:
            if id(obj) not in seen:
                seen.add(id(obj)); storage += sys.getsizeof(obj)
        return dict(registrations=self.registrations,registered_rows=len(self._points),
            capacity_rows=self.max_rows,cached_unique_rows=len(self._values),
            requests=self.requests,requested_rows=self.requested_rows,reused_rows=self.reused_rows,
            owner_calls=self.owner_calls,evaluated_rows=self.evaluated_rows,binding_checks=self.binding_checks,
            storage_bytes=storage,evaluation_seconds=self.evaluation_seconds,total_seconds=self.total_seconds)
