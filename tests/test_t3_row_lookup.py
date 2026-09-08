"""Persistent canonical ranks versus the exact dictionary oracle."""
from types import SimpleNamespace

import numpy as np
import pytest

from anymesher import _t3_correspondence as correspondence
from anymesher._t3_row_lookup import (
    CanonicalRowLookup, canonical_row_lookup, advance_row_lookup,
)
from anymesher._t3_insertion_result import _reference_source_rows
from anymesher.errors import MeshError


def lookup(rows):
    matrix = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    mapping = {tuple(map(int, row)): n for n, row in enumerate(matrix)}
    cells, owned = correspondence.row_storage(mapping)
    return canonical_row_lookup(cells, owned), cells, owned


def advance(before, old_cells, old_matrix, rows):
    matrix = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
    mapping = {tuple(map(int, row)): n for n, row in enumerate(matrix)}
    cells, owned = correspondence.row_storage(mapping)
    source = _reference_source_rows(old_matrix, owned)
    index = SimpleNamespace(_row_by_cell=before, _row_cells=old_cells)
    return advance_row_lookup(index, cells, owned, source, None), cells, owned


def audit(node):
    if node is None:
        return 0, 0
    lh, left = audit(node.left)
    rh, right = audit(node.right)
    assert node.height == 1 + max(lh, rh) and abs(lh - rh) <= 1
    assert node.size == 1 + left + right
    return node.height, node.size


@pytest.mark.parametrize("seed", range(5))
def test_random_local_ranks_and_branch_differences(seed):
    rng = np.random.default_rng(seed)
    rows = [(n, n + 1, n + 2) for n in range(100)]
    current, cells, matrix = lookup(rows)
    history = []
    for step in range(40):
        before, old_rows = current, rows
        history.append((before, list(rows)))
        removed = set(map(int, rng.choice(len(rows), 3, replace=False)))
        rows = sorted([row for i, row in enumerate(rows) if i not in removed]
                      + [(100 + step * 3 + n, 1000 + n, 2000 + n) for n in range(3)])
        current, cells, matrix = advance(current, cells, matrix, rows)
        assert dict(current) == {row: n for n, row in enumerate(rows)}
        assert before.keys() - current.keys() == set(old_rows) - set(rows)
        assert current.keys() - before.keys() == set(rows) - set(old_rows)
        assert audit(current._root)[1] == len(rows)
    for snapshot, old_rows in history:
        assert dict(snapshot) == {row: n for n, row in enumerate(old_rows)}
        assert snapshot.keys() - current.keys() == set(old_rows) - set(rows)


def test_empty_root_parent_sentinel_and_unrelated_maps():
    empty, cells, matrix = lookup([])
    first, _, _ = lookup([(1, 2, 3)])
    assert first.keys() - empty.keys() == {(1, 2, 3)}
    grown, new_cells, new_matrix = advance(empty, cells, matrix, [(1, 2, 3)])
    assert grown[(1, 2, 3)] == 0
    assert grown.keys() - empty.keys() == {(1, 2, 3)}
    erased, _, _ = advance(grown, new_cells, new_matrix, [])
    assert grown.keys() - erased.keys() == {(1, 2, 3)}
    assert erased.keys() - grown.keys() == set()
    assert erased._root is None
    assert empty.keys() - first.keys() == set()


@pytest.mark.parametrize("rows", [
    [(3, 4, 5), (0, 1, 2)],
    [(2, 1, 4), (2, 0, 5)],
])
def test_reordered_inputs_fall_back(rows):
    assert lookup(rows)[0] is None
    old, cells, matrix = lookup(sorted(rows))
    assert advance(old, cells, matrix, rows)[0] is None


def test_adjacent_key_differences_do_not_iterate_lookup(monkeypatch):
    old, cells, matrix = lookup([(n, n + 1, n + 2) for n in range(1000)])
    new, _, _ = advance(old, cells, matrix,
                        [(n, n + 1, n + 2) for n in range(1, 1001)])
    def forbidden(*args):
        raise AssertionError("full key scan")
    monkeypatch.setattr(CanonicalRowLookup, "__iter__", forbidden)
    assert old.keys() - new.keys() == {(0, 1, 2)}
    assert new.keys() - old.keys() == {(1000, 1001, 1002)}
    assert new.keys() - new.keys() == set()


def test_key_lookup_and_zero_rank():
    current, _, _ = lookup([(0, 1, 2), (0, 3, 4), (2, 3, 4)])
    assert current[(0, 1, 2)] == 0 and (0, 1, 2) in current
    assert current.get((99, 100, 101)) is None
    assert current.get((99, 100, 101), -7) == -7
    with pytest.raises(KeyError):
        current[(99, 100, 101)]
    with pytest.raises(TypeError):
        current[(0, 1, 2)] = 1


def test_cancelled_rank_update_preserves_parent():
    old, cells, matrix = lookup([(n, n + 1, n + 2) for n in range(10)])
    root = old._root
    next_matrix = np.asarray([(n, n + 1, n + 2) for n in range(1, 11)])
    next_cells, owned = correspondence.row_storage(dict.fromkeys(map(tuple, next_matrix)))
    source = _reference_source_rows(matrix, owned)
    def cancel(phase):
        raise RuntimeError("cancel rank update")
    with pytest.raises(RuntimeError, match="cancel rank update"):
        advance_row_lookup(SimpleNamespace(_row_by_cell=old, _row_cells=cells),
                           next_cells, owned, source, cancel)
    assert old._root is root and old[(0, 1, 2)] == 0


def test_repeated_source_rank_is_rejected():
    old, cells, matrix = lookup([(0, 1, 2), (3, 4, 5)])
    with pytest.raises(MeshError, match="canonical ranks"):
        advance_row_lookup(SimpleNamespace(_row_by_cell=old, _row_cells=cells),
                           cells, matrix, np.asarray([0, 0]), None)


def test_index_update_uses_rank_map_and_reference_export():
    from anymesher._t3_incidence import T3IncidenceIndex
    old = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    new = np.asarray([(0, 1, 4), (0, 2, 3), (0, 4, 2), (1, 2, 4)])
    index = T3IncidenceIndex(old)
    assert type(index._row_by_cell) is CanonicalRowLookup
    @correspondence.insertion_correspondence_scope
    def update():
        correspondence.source_rows(old, new, _reference_source_rows)
        return index.updated(new)
    changed = update()
    expected = T3IncidenceIndex(new)
    assert type(changed._row_by_cell) is CanonicalRowLookup
    assert changed._row_by_cell == expected._row_by_cell
    for edge in expected._edge_cells:
        assert changed.attached(edge) == expected.attached(edge)
    assert np.array_equal(changed.canonical_edges(), expected.canonical_edges())
    assert dict(index._row_by_cell) == {(0, 1, 2): 0, (0, 2, 3): 1}

