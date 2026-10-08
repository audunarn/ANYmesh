"""PRIVATE runtime-only operation counters for one mesh generation.

These counters are execution diagnostics only.  They are never inserted
into preparation hashes, quality records, or semantic model records, and
they never influence mesh results.  A generation installs a counts
mapping for the duration of one scope; code running inside the scope adds
integer counts lazily.  Without an installed scope every helper is a
no-op, so standalone use of the mesher is unchanged.

The sink is a :class:`contextvars.ContextVar`, so counts never leak
across threads or nested generations: each candidate chain accumulates
its own work and the hybrid finalization merges the chain exactly once.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

__all__ = [
    "MESHER_BVH_BUILDS",
    "MESHER_BVH_LOOKUPS",
    "MESHER_CHART_QUERY_BATCHES",
    "MESHER_CHART_QUERY_FACES",
    "add_operation_count",
    "geometry_runtime_diagnostics",
    "merge_operation_counts",
    "new_operation_counts",
    "operation_counts_scope",
]

MESHER_BVH_BUILDS = "mesher.bvh_builds"
MESHER_BVH_LOOKUPS = "mesher.bvh_lookups"
MESHER_CHART_QUERY_BATCHES = "mesher.chart_query_batches"
MESHER_CHART_QUERY_FACES = "mesher.chart_query_faces"

_current_counts: ContextVar[dict[str, int] | None] = ContextVar(
    "anymesher_runtime_operation_counts", default=None
)


def new_operation_counts() -> dict[str, int]:
    """Return a fresh runtime operation-count mapping."""

    return {}


def add_operation_count(key: str, amount: int = 1) -> None:
    """Add ``amount`` to ``key`` when a counts scope is installed."""

    counts = _current_counts.get()
    if counts is not None:
        counts[key] = counts.get(key, 0) + int(amount)


@contextmanager
def operation_counts_scope(sink: dict[str, int]) -> Iterator[dict[str, int]]:
    """Accumulate runtime operation counts into ``sink`` within the scope."""

    token = _current_counts.set(sink)
    try:
        yield sink
    finally:
        _current_counts.reset(token)


def merge_operation_counts(
    target: dict[str, int], source: Mapping[str, int], *, prefix: str = ""
) -> None:
    """Merge integer counts of ``source`` into ``target`` under ``prefix``.

    Non-integer values are ignored: the runtime contract publishes integer
    counts only.  Existing keys are summed, so chain merging never drops a
    candidate's own work and never double counts shared dictionaries.
    """

    for key, value in source.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        made = f"{prefix}{key}"
        target[made] = target.get(made, 0) + int(value)


@contextmanager
def _empty_runtime_diagnostics() -> Iterator[dict[str, int]]:
    """Fallback context for geometry without runtime diagnostics."""

    yield {}


def geometry_runtime_diagnostics():
    """Return ANYgeometry's runtime-diagnostics context manager.

    The geometry owner publishes
    ``anygeometry.runtime_diagnostics.collect_runtime_diagnostics()``,
    a context manager yielding a mapping of unprefixed integer counts.
    Older supported geometry revisions do not provide the optional module;
    only that absent module falls back to an empty mapping so the
    surrounding preparation is unchanged.  Any other import failure (a
    present but broken module or dependency) propagates.
    """

    try:
        from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
    except ModuleNotFoundError as error:
        if error.name != "anygeometry.runtime_diagnostics":
            raise
        return _empty_runtime_diagnostics()
    return collect_runtime_diagnostics()
