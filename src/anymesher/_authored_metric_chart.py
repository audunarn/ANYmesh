"""Read-only metric coordinates for a bound original owner face support."""
from dataclasses import dataclass, fields, is_dataclass
from numbers import Real
from uuid import UUID

import numpy as np
from anygeometry import (evaluate_prepared_authored_face,
                         validate_prepared_authored_boundary_correspondence_binding)

from .errors import MeshError


def _snapshot(value):
    """Detach public immutable receipt structure, without interpreting geometry."""
    if is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((field.name, _snapshot(getattr(value, field.name)))
                                   for field in fields(value)))
    if type(value) is tuple:
        return tuple(_snapshot(item) for item in value)
    if type(value) in (str, bytes, int, bool, float, type(None), UUID):
        return value
    raise MeshError('authored metric chart requires an immutable owner correspondence')


def _rows(value):
    message = 'authored metric chart requires finite real (n, 2) rows'
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'biufO' or (raw.dtype.kind == 'O' and
                any(not isinstance(item, Real) for item in raw.flat)):
            raise MeshError(message)
        rows = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise MeshError(message) from error
    if rows.ndim != 2 or rows.shape[1] != 2 or not np.isfinite(rows).all():
        raise MeshError(message)
    return rows


def _matrix(data):
    # A bytes buffer cannot have WRITEABLE re-enabled by a caller.
    return np.frombuffer(data, dtype=np.float64).reshape((2, 2))


@dataclass(frozen=True, slots=True, init=False)
class AuthoredMetricChart:
    """A constant reference metric, not an isometry or publication certificate.

    Metric rows equal original UV rows times the center-differential Cholesky
    factor. Evaluation and derivatives always come from the public owner. No
    world-coordinate inversion, registered-node replacement, containment or
    meshing permission is supplied.
    """
    _context: tuple

    def __init__(self, geometry, correspondence, cancellation_check=None):
        context = (geometry, correspondence, cancellation_check, _snapshot(correspondence), None, None)
        object.__setattr__(self, '_context', context)
        self._current(context)
        du, dv = evaluate_prepared_authored_face(geometry, correspondence, [[.5, .5]],
            derivatives=True, cancellation_check=self._callback(context))
        self._current(context)
        try:
            if np.iscomplexobj(du) or np.iscomplexobj(dv):
                raise ValueError('complex differential')
            du, dv = np.asarray(du, dtype=float), np.asarray(dv, dtype=float)
            if du.shape != (1, 3) or dv.shape != (1, 3):
                raise ValueError('invalid differential shape')
            derivative = np.column_stack((du[0], dv[0]))
            if not np.isfinite(derivative).all():
                raise ValueError('nonfinite differential')
            with np.errstate(over='ignore', invalid='ignore'):
                metric = derivative.T @ derivative
            if not np.isfinite(metric).all():
                raise ValueError('nonfinite metric')
            transform = np.linalg.cholesky(metric)
            inverse = np.linalg.inv(transform)
            if not np.isfinite(inverse).all():
                raise ValueError('nonfinite inverse')
        except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as error:
            raise MeshError('authored metric chart has an invalid reference differential') from error
        self._current(context)
        object.__setattr__(self, '_context', (*context[:4], transform.tobytes(), inverse.tobytes()))

    @property
    def publication_qualified(self):
        return False

    @property
    def transform(self):
        return _matrix(self._context[4])

    @property
    def inverse(self):
        return _matrix(self._context[5])

    def _same(self, context):
        if self._context is not context or _snapshot(context[1]) != context[3]:
            raise MeshError('authored metric chart binding or transform changed')

    def _callback(self, context):
        def check(phase):
            self._same(context)
            cancelled = bool(context[2](phase)) if context[2] is not None else False
            self._same(context)
            return cancelled
        return check

    def _current(self, context):
        self._same(context)
        validate_prepared_authored_boundary_correspondence_binding(
            context[0], context[1], cancellation_check=self._callback(context))
        self._same(context)

    def _apply(self, value, operation):
        context = self._context
        rows = _rows(value)  # owned input before any owner callback
        self._same(context)  # input conversion must not switch the entry chart
        self._current(context)
        transform, inverse = _matrix(context[4]), _matrix(context[5])
        with np.errstate(over='ignore', invalid='ignore'):
            uv = rows @ inverse if operation != 'to_metric' else rows
        if not np.isfinite(uv).all():
            raise MeshError('authored metric coordinates overflowed')
        if operation == 'evaluate':
            result = evaluate_prepared_authored_face(context[0], context[1], uv,
                cancellation_check=self._callback(context))
        elif operation == 'jacobians':
            du, dv = evaluate_prepared_authored_face(context[0], context[1], uv,
                derivatives=True, cancellation_check=self._callback(context))
            result = np.stack((du, dv), axis=-1) @ inverse.T
        elif operation == 'to_metric':
            with np.errstate(over='ignore', invalid='ignore'):
                result = rows @ transform
        else:
            result = uv
        result = np.array(result, dtype=float, copy=True)
        if not np.isfinite(result).all():
            raise MeshError('authored metric evaluation produced nonfinite results')
        self._current(context)
        return result

    def evaluate(self, metric_rows):
        return self._apply(metric_rows, 'evaluate')

    def jacobians(self, metric_rows):
        return self._apply(metric_rows, 'jacobians')

    def to_metric(self, original_uv):
        return self._apply(original_uv, 'to_metric')

    def to_authored_uv(self, metric_rows):
        return self._apply(metric_rows, 'to_authored_uv')
