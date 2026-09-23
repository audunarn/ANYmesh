"""Bounded, invocation-local reuse of exact insertion correspondence.

Snapshots are owned and read-only. Cache hits compare current arrays with those
snapshots, never trust native diagnostic counts or mutable array identities.
"""
from __future__ import annotations

from contextvars import ContextVar
from functools import wraps

import numpy as np

_ACTIVE = ContextVar("anymesher_t3_insertion_correspondence", default=None)


def insertion_correspondence_scope(function):
    @wraps(function)
    def scoped(*args, **kwargs):
        subject = (args[0] if args and hasattr(args[0], "_local_validation_source")
                   else None)
        source = None
        if subject is not None:
            from ._t3_local_validation import source_for_scope
            source = source_for_scope(subject)
        frame = [None, source, None, subject]
        token = _ACTIVE.set(frame)
        try:
            result = function(*args, **kwargs)
            if subject is not None:
                from ._t3_local_validation import finish_scope
                finish_scope(subject, frame)
            return result
        finally:
            _ACTIVE.reset(token)
    return scoped


def _same(actual, saved):
    return (isinstance(actual, np.ndarray)
            and actual.dtype == saved.dtype and actual.shape == saved.shape
            and np.array_equal(actual, saved))


def _owned(array):
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def current_correspondence(before, after):
    scope = _ACTIVE.get()
    if scope is None or scope[0] is None:
        return None
    old, new, source = scope[0]
    if _same(before, old) and _same(after, new):
        return old, new, source
    return None


def source_rows(before, after, compute):
    scope = _ACTIVE.get()
    if scope is None:
        return compute(before, after)
    current = current_correspondence(before, after)
    if current is not None:
        return current[2].copy()
    result = compute(before, after)
    # Do not change general helper behavior for non-array callers.
    if isinstance(before, np.ndarray) and isinstance(after, np.ndarray):
        staged = (_owned(before), _owned(after), _owned(result))
        scope[0] = staged
    return result


def row_storage(rows):
    cells = np.fromiter(rows, dtype=object, count=len(rows))
    matrix = np.asarray(tuple(rows), dtype=np.int64).reshape(-1, 3)
    cells.setflags(write=False)
    matrix.setflags(write=False)
    return cells, matrix


def reuse_row_storage(index, triangles, cancellation_check):
    """Reuse existing tuple identities only for an exact checked array pair."""
    from .errors import MeshError

    if not isinstance(triangles, np.ndarray) or triangles.dtype != np.dtype(np.int64):
        return None
    current = current_correspondence(index._row_matrix, triangles)
    if current is None:
        return None
    _, matrix, source = current
    if cancellation_check is not None:
        cancellation_check("native-v2 incidence scan")
    if (source.shape != (len(matrix),)
            or np.any(source < -1) or np.any(source >= len(index._row_cells))):
        raise MeshError("insertion correspondence has invalid source rows")
    retained = source >= 0
    cells = np.empty(len(matrix), dtype=object)
    cells[retained] = index._row_cells[source[retained]]
    for number, row in enumerate(np.flatnonzero(~retained)):
        if cancellation_check is not None and number % 4096 == 0:
            cancellation_check("native-v2 incidence scan")
        cell = tuple(map(int, matrix[row]))
        if len(cell) != 3 or len(set(cell)) != 3:
            raise MeshError("mutable T3 incidence requires distinct triangle rows")
        cells[row] = cell
    from ._t3_row_lookup import advance_row_lookup

    rows = advance_row_lookup(index, cells, matrix, source, cancellation_check)
    if rows is None:
        rows = dict(zip(cells, range(len(cells))))
    if len(rows) != len(matrix):
        raise MeshError("mutable T3 incidence requires distinct triangle rows")
    cells.setflags(write=False)
    if cancellation_check is not None:
        cancellation_check("native-v2 incidence scan")
    return rows, cells, matrix
