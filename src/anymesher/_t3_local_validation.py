"""Local result validation bound to an independently qualified source mesh."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .errors import MeshError


@dataclass(frozen=True, slots=True)
class QualifiedSource:
    subject_id: int
    point_owner_id: int
    row_owner_id: int
    points: np.ndarray
    rows: np.ndarray
    index: object
    protected: frozenset
    counts: object
    epoch: int


@dataclass(frozen=True, slots=True)
class PendingCommit:
    source: QualifiedSource
    points: np.ndarray
    rows: np.ndarray
    counts: object
    changed_counts: object


def _protected(values):
    return frozenset(tuple(sorted(map(int, edge))) for edge in values)


def _same_points(a, b):
    return (isinstance(a, np.ndarray) and a.dtype == b.dtype
            and a.shape == b.shape
            and np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def _same_rows(a, b):
    return (isinstance(a, np.ndarray) and a.dtype == b.dtype
            and a.shape == b.shape and np.array_equal(a, b))


def _owned(value):
    result = np.array(value, copy=True, order="C")
    result.setflags(write=False)
    return result


def _snapshot(subject, points, rows, counts, epoch):
    from ._t3_incidence_counts import IncidenceCounts

    return QualifiedSource(
        id(subject), id(subject._points), id(subject._triangles),
        _owned(points), _owned(rows), subject._topology_index,
        _protected(subject.protected_edges), IncidenceCounts.from_mapping(counts), epoch,
    )


def enable_local_validation(subject, *, cancellation_check=None):
    """Independently establish the source invariant once per frontal run."""
    from ._t3_insertion_result import _positive_cells, _keys
    from .triangulation import orient2d

    def check(phase):
        if cancellation_check is not None:
            cancellation_check("native-v2 qualified source " + phase)

    check("start")
    points = _owned(subject._points)
    rows = _owned(subject._triangles)
    index = subject._topology_index
    if (points.dtype != np.dtype(np.float64) or points.ndim != 2
            or points.shape[1] != 2 or not np.all(np.isfinite(points))
            or rows.dtype != np.dtype(np.int64) or rows.ndim != 2
            or rows.shape[1] != 3 or np.any(rows < 0)
            or np.any(rows >= len(points))):
        raise MeshError("qualified insertion source arrays are invalid")
    if not _same_rows(index._row_matrix, rows):
        raise MeshError("qualified insertion source index binding is stale")
    if len(rows):
        starts = np.argmin(rows, axis=1)
        canonical = np.take_along_axis(
            rows, (starts[:, None] + np.arange(3)) % 3, axis=1)
        if len(np.unique(_keys(canonical))) != len(rows):
            raise MeshError("qualified insertion source repeats a triangle")
    _positive_cells(points, rows, orient2d, lambda _: check("orientation"))
    members = {}
    for number, raw in enumerate(rows):
        if number % 4096 == 0:
            check("incidence")
        cell = tuple(map(int, raw))
        for a, b in zip(cell, (*cell[1:], cell[0])):
            edge = (min(a, b), max(a, b))
            members.setdefault(edge, []).append(cell)
    if members.keys() != index._edge_cells.keys():
        raise MeshError("qualified insertion source edge inventory is stale")
    counts = {}
    for number, (edge, attached) in enumerate(members.items()):
        if number % 4096 == 0:
            check("incidence")
        if len(attached) > 2 or tuple(sorted(attached)) != index._edge_cells[edge]:
            raise MeshError("qualified insertion source incidence is invalid")
        counts[edge] = len(attached)
    protected = _protected(subject.protected_edges)
    if any(edge not in counts for edge in protected):
        raise MeshError("qualified insertion source lost a protected edge")
    if (subject._topology_index is not index
            or not _same_points(subject._points, points)
            or not _same_rows(subject._triangles, rows)):
        raise MeshError("qualified insertion source changed during qualification")
    snapshot = _snapshot(subject, points, rows, counts, subject.epoch)
    check("commit")
    subject._local_validation_source = snapshot


def source_for_scope(subject):
    source = getattr(subject, "_local_validation_source", None)
    if (type(source) is not QualifiedSource or source.subject_id != id(subject)
            or subject._topology_index is not source.index
            or subject.epoch != source.epoch
            or _protected(subject.protected_edges) != source.protected
            or not _same_points(subject._points, source.points)
            or not _same_rows(subject._triangles, source.rows)):
        return None
    return source


def source_for_validation(points, rows, protected):
    from ._t3_correspondence import _ACTIVE

    frame = _ACTIVE.get()
    if frame is None or len(frame) < 4 or frame[1] is None:
        return None
    source = frame[1]
    if (source_for_scope(frame[3]) is not source
            or not _same_points(points, source.points)
            or not _same_rows(rows, source.rows)
            or _protected(protected) != source.protected):
        return None
    return source


def _changed_counts(source, changes):
    values = {edge: source.counts.get(edge, 0) + delta
              for edge, delta in changes.items()}
    if any(value > 2 for value in values.values()):
        raise MeshError("native mutable T3 insertion returned non-manifold topology")
    if any(value < 0 for value in values.values()):
        raise MeshError("native mutable T3 insertion source incidence is inconsistent")
    if any(value == 0 and edge in source.protected for edge, value in values.items()):
        raise MeshError("native mutable T3 insertion removed a protected edge")
    return values


def validate_changed_incidence(source, old_edges, old_counts, new_edges, new_counts):
    changes = {}
    for sign, edges, counts in ((-1, old_edges, old_counts), (1, new_edges, new_counts)):
        for edge, count in zip(edges, counts):
            key = tuple(map(int, edge))
            changes[key] = changes.get(key, 0) + sign * int(count)
    _changed_counts(source, changes)


def stage_qualified_commit(points, rows, old_points, old_rows, old_index, changes):
    """Called only after the independent transactional cavity guard succeeds."""
    from ._t3_correspondence import _ACTIVE

    frame = _ACTIVE.get()
    if frame is None or len(frame) < 4 or frame[1] is None:
        return
    source, subject = frame[1], frame[3]
    if (source.index is not old_index or source.subject_id != id(subject)
            or getattr(subject, "_local_validation_source", None) is not source
            or points is not subject._points or rows is not subject._triangles
            or not _same_points(old_points, source.points)
            or not _same_rows(old_rows, source.rows)
            or _protected(subject.protected_edges) != source.protected):
        return
    values = _changed_counts(source, changes)
    counts = source.counts.with_updates(values)
    frame[2] = PendingCommit(
        source, _owned(points), _owned(rows), counts,
        MappingProxyType(values),
    )


def bind_qualified_commit(subject):
    """Bind staged proof after index construction, inside insertion rollback."""
    from ._t3_correspondence import _ACTIVE

    frame = _ACTIVE.get()
    if frame is None or len(frame) < 4 or frame[2] is None:
        return
    pending = frame[2]
    if (type(pending) is not PendingCommit or frame[3] is not subject
            or pending.source is not frame[1]
            or pending.source.subject_id != id(subject)
            or subject._local_validation_source is not pending.source
            or subject.epoch != pending.source.epoch
            or _protected(subject.protected_edges) != pending.source.protected
            or not _same_points(subject._points, pending.points)
            or not _same_rows(subject._triangles, pending.rows)
            or not _same_rows(subject._topology_index._row_matrix, pending.rows)):
        raise MeshError("qualified insertion commit index binding is stale")
    for edge, count in pending.changed_counts.items():
        if len(subject._topology_index._edge_cells.get(edge, ())) != count:
            raise MeshError("qualified insertion commit incidence is stale")
    # All allocation and index-binding failures remain rollback-protected.
    frame[2] = _snapshot(
        subject, pending.points, pending.rows, pending.counts,
        pending.source.epoch + 1,
    )


def finish_scope(subject, frame):
    candidate = frame[2]
    if type(candidate) is not QualifiedSource:
        candidate = None
    elif (
        subject._topology_index is not candidate.index
        or subject.epoch != candidate.epoch
        or id(subject._points) != candidate.point_owner_id
        or id(subject._triangles) != candidate.row_owner_id
    ):
        candidate = None
    subject._local_validation_source = candidate
