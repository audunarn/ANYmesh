"""Opt-in sidecar timings for development; never part of a published mesh."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter


_sink: ContextVar[dict[str, float] | None] = ContextVar("quad_stage_timing", default=None)


@contextmanager
def collect_quad_stage_timings():
    values: dict[str, float] = {}
    token = _sink.set(values)
    try:
        yield values
    finally:
        _sink.reset(token)


def timed_quad_call(stage, function, *args, **kwargs):
    values = _sink.get()
    if values is None:
        return function(*args, **kwargs)
    started = perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        values[stage] = values.get(stage, 0.0) + perf_counter() - started


def record_quad_stage(stage, elapsed):
    values = _sink.get()
    if values is not None:
        values[stage] = values.get(stage, 0.0) + float(elapsed)
