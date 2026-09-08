"""Independent, batched qualification of a detached native insertion result.

No native certificate or diagnostic is treated as authority for connectivity.
The unchanged-row map is derived from exact connectivity. Only new-cell owners
need incidence lookups; the source snapshot is never modified.
"""
from __future__ import annotations

import numpy as np

from .errors import MeshError

_BATCH = 4096
_CELL_KEY = np.dtype([("a", np.int64), ("b", np.int64), ("c", np.int64)])
_EDGE_KEY = np.dtype([("a", np.int64), ("b", np.int64)])


def _poll(callback, phase):
    if callback is not None:
        callback(phase)


def _keys(rows, edge=False):
    return np.ascontiguousarray(rows, dtype=np.int64).view(
        _EDGE_KEY if edge else _CELL_KEY
    ).reshape(-1)


def _source_rows(before, after):
    from ._t3_correspondence import source_rows
    return source_rows(before, after, _compute_source_rows)


def _compute_source_rows(before, after):
    from ._t3_correspondence import source_rows
    return source_rows(before, after, _compute_source_rows)


def _compute_source_rows(before, after):
    """Return exact source row numbers, or -1 for genuinely new cells."""
    from ._t3_packed_correspondence import packed_source_rows
    result = packed_source_rows(before, after)
    if result is not None:
        return result
    return _reference_source_rows(before, after)


def _reference_source_rows(before, after):
    """General structured-key oracle, including reordered and wide inputs."""
    old_keys = _keys(before)
    keys = _keys(after)
    order = np.argsort(old_keys, kind="stable")
    ordered = old_keys[order]
    positions = np.searchsorted(ordered, keys)
    matched = positions < len(ordered)
    indices = np.flatnonzero(matched)
    matched[indices] = ordered[positions[indices]] == keys[indices]
    result = np.full(len(after), -1, dtype=np.int64)
    result[matched] = order[positions[matched]]
    return result


def _edges_and_counts(cells):
    if not len(cells):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    edges = np.concatenate((cells[:, [0, 1]], cells[:, [1, 2]],
                            cells[:, [2, 0]]))
    edges.sort(axis=1)
    keys, counts = np.unique(_keys(edges, edge=True), return_counts=True)
    return keys.view(np.int64).reshape((-1, 2)), counts


def _positive_cells(points, rows, oracle, callback):
    # Same binary64 determinant as the adaptive predicate's fast path.
    # Ambiguous/overflowed determinants always use the original owner oracle.
    for start in range(0, len(rows), _BATCH):
        _poll(callback, "native-v2 insertion result orientation validation")
        selected = rows[start:start + _BATCH]
        a, b, c = (points[selected[:, column]] for column in range(3))
        with np.errstate(over="ignore", invalid="ignore"):
            ac, bc = a - c, b - c
            first, second = ac[:, 0] * bc[:, 1], ac[:, 1] * bc[:, 0]
            determinants = first - second
            bound = 8.0 * np.finfo(np.float64).eps * (abs(first) + abs(second))
        uncertain = ~np.isfinite(determinants) | ~np.isfinite(bound) | (
            abs(determinants) <= bound
        )
        if np.any(determinants[~uncertain] <= 0.0):
            raise MeshError("native mutable T3 insertion returned a non-positive cell")
        for row in np.flatnonzero(uncertain):
            value = oracle(a[row], b[row], c[row])
            if not np.isfinite(value) or value <= 0.0:
                raise MeshError("native mutable T3 insertion returned a non-positive cell")


def validate_insertion(points, before, protected, candidate, raw_rows,
                       diagnostics, oracle, callback=None):
    rows = np.asarray(raw_rows)
    if rows.ndim != 2 or rows.shape[1:] != (3,) or rows.dtype.kind not in "iu":
        raise MeshError("native mutable T3 insertion connectivity is malformed")
    required = {"removed_triangles", "added_triangles", "native"}
    if not isinstance(diagnostics, dict) or set(diagnostics) != required:
        raise MeshError("native mutable T3 insertion diagnostics are malformed")
    report = dict(diagnostics)
    removed, added = report["removed_triangles"], report["added_triangles"]
    if (type(removed) is not int or type(added) is not int
            or report["native"] is not True
            or not 1 <= removed <= len(before) or added < 3
            or len(rows) != len(before) - removed + added
            or np.any(rows < 0) or np.any(rows > len(points))):
        raise MeshError("native mutable T3 insertion result is inconsistent")
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    _poll(callback, "native-v2 insertion result identity validation")
    keys = _keys(rows)
    if (np.any(rows[:, 0] >= rows[:, 1])
            or np.any(rows[:, 0] >= rows[:, 2])
            or np.any(rows[:, 1] == rows[:, 2])
            or not np.array_equal(np.argsort(keys, kind="stable"), np.arange(len(rows)))
            or np.any(keys[1:] == keys[:-1])):
        raise MeshError("native mutable T3 insertion is not canonical")
    source = _source_rows(before, rows)
    new = source < 0
    if (int(np.count_nonzero(new)) != added
            or int(np.count_nonzero(~new)) != len(before) - removed
            or not np.all(np.any(rows[new] == len(points), axis=1))
            or np.any(rows[~new] == len(points))):
        raise MeshError("native mutable T3 insertion cavity accounting is inconsistent")
    from ._t3_local_validation import source_for_validation, validate_changed_incidence
    qualified = source_for_validation(points, before, protected)
    extended = np.vstack((points, candidate))
    _positive_cells(extended, rows if qualified is None else rows[new], oracle, callback)

    _poll(callback, "native-v2 insertion result incidence validation")
    if qualified is None:
        edges, counts = _edges_and_counts(rows)
        if np.any(counts > 2):
            raise MeshError("native mutable T3 insertion returned non-manifold topology")
        if len(protected):
            ordered = np.sort(np.asarray(protected, dtype=np.int64), axis=1)
            if not np.all(np.isin(_keys(ordered, edge=True), _keys(edges, edge=True))):
                raise MeshError("native mutable T3 insertion removed a protected edge")
    # A valid-looking star cannot replace another region or lose domain coverage.
    retained = np.zeros(len(before), dtype=bool)
    retained[source[~new]] = True
    old_edges, old_counts = _edges_and_counts(before[~retained])
    new_edges, new_counts = _edges_and_counts(rows[new])
    if qualified is not None:
        validate_changed_incidence(qualified, old_edges, old_counts, new_edges, new_counts)
    if not np.array_equal(old_edges[old_counts == 1], new_edges[new_counts == 1]):
        raise MeshError("native mutable T3 insertion changed the cavity boundary")
    _poll(callback, "native-v2 insertion result validated")
    return rows, report


def insertion_owners(before, after, owners, index, *, inserted_node, owner,
                     cancellation_check=None):
    _poll(cancellation_check, "native-v2 insertion owner mapping")
    source = _source_rows(before, after)
    retained = source >= 0
    result = np.empty(len(after), dtype=np.int64)
    result[retained] = owners[source[retained]]
    new_rows = np.flatnonzero(~retained)
    for number, row in enumerate(new_rows):
        if number % _BATCH == 0:
            _poll(cancellation_check, "native-v2 insertion owner mapping")
        cell = after[row]
        if np.count_nonzero(cell == inserted_node) != 1:
            raise MeshError("compiled mutable T3 insertion returned an unknown cell")
        opposite = cell[cell != inserted_node]
        edge = (min(map(int, opposite)), max(map(int, opposite)))
        sources = index.attached(edge)
        if not sources:
            raise MeshError("compiled mutable T3 insertion returned an ownerless cavity cell")
        result[row] = int(owner) if int(owner) != -1 else owners[sources[0]]
    _poll(cancellation_check, "native-v2 insertion owner mapping complete")
    return result
