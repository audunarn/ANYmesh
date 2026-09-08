"""Persistent export-row ranks for canonical T3 connectivity."""
from __future__ import annotations

from collections.abc import Mapping, KeysView
from dataclasses import dataclass

import numpy as np

from ._t3_edge_membership import _build
from ._t3_geometry_cache import _erase, _set
from .errors import MeshError

_NO_PARENT = object()


def _ordered(matrix):
    if len(matrix) < 2:
        return True
    left, right = matrix[:-1], matrix[1:]
    return bool(np.all(
        (left[:, 0] < right[:, 0])
        | ((left[:, 0] == right[:, 0]) & (left[:, 1] < right[:, 1]))
        | ((left[:, 0] == right[:, 0]) & (left[:, 1] == right[:, 1])
           & (left[:, 2] < right[:, 2]))
    ))


class _RowKeys(KeysView):
    def __iter__(self):
        return iter(self._mapping)

    def __sub__(self, other):
        if isinstance(other, _RowKeys):
            before, after = self._mapping, other._mapping
            if before._root is after._root:
                return set()
            if after._parent_root is before._root:
                return set(after._removed)
            if before._parent_root is after._root:
                return set(before._added)
        return set(self).difference(other)


class CanonicalRowLookup(Mapping):
    """Immutable local rank updates; array movement remains outside this map."""
    __slots__ = ("_root", "_cells", "_parent_root", "_removed", "_added")

    def __init__(self, root, cells, parent_root=_NO_PARENT, removed=(), added=()):
        self._root = root
        self._cells = cells
        self._parent_root = parent_root
        self._removed = removed
        self._added = added

    def __len__(self):
        return len(self._cells)

    def __iter__(self):
        # Owned tuple identities are already in exact export order.
        return iter(self._cells)

    def get(self, cell, default=None):
        node, rank = self._root, 0
        while node is not None:
            left_size = 0 if node.left is None else node.left.size
            if cell == node.key:
                return rank + left_size
            if cell < node.key:
                node = node.left
            else:
                rank += left_size + 1
                node = node.right
        return default

    def __getitem__(self, cell):
        rank = self.get(cell)
        if rank is None:
            raise KeyError(cell)
        return rank

    def __contains__(self, cell):
        return self.get(cell) is not None

    def keys(self):
        return _RowKeys(self)


def canonical_row_lookup(cells, matrix):
    """Initial unordered inputs keep their original dictionary semantics."""
    if not _ordered(matrix):
        return None
    root = _build([(cell, None) for cell in cells], 0, len(cells))
    return CanonicalRowLookup(root, cells)


def advance_row_lookup(index, cells, matrix, source, cancellation_check):
    """Use invocation-bound correspondence, never native report counters."""
    before = index._row_by_cell
    if not isinstance(before, CanonicalRowLookup) or not _ordered(matrix):
        return None
    retained = source >= 0
    kept = source[retained]
    # Canonical unique rows must retain old cells in strictly increasing order.
    if len(kept) > 1 and np.any(kept[1:] <= kept[:-1]):
        raise MeshError("insertion correspondence has inconsistent canonical ranks")
    seen = np.zeros(len(before), dtype=bool)
    seen[kept] = True
    removed_set = set(index._row_cells[~seen])
    added_set = set(cells[~retained])
    removed = tuple(sorted(removed_set - added_set))
    added = tuple(sorted(added_set - removed_set))
    root = before._root
    for number, cell in enumerate(removed):
        if cancellation_check is not None and number % 4096 == 0:
            cancellation_check("native-v2 incidence update")
        root = _erase(root, cell)
    for number, cell in enumerate(added):
        if cancellation_check is not None and number % 4096 == 0:
            cancellation_check("native-v2 incidence update")
        root = _set(root, cell, None)
    size = 0 if root is None else root.size
    if size != len(cells):
        raise MeshError("insertion correspondence has inconsistent canonical ranks")
    return CanonicalRowLookup(root, cells, before._root, removed, added)

@dataclass(frozen=True, slots=True)
class _FrozenRows(Mapping):
    """Retain immutable nodes, not a mutable lookup object or owner history."""
    _root: object
    _parent_root: object
    _removed: tuple
    _added: tuple

    get = CanonicalRowLookup.get
    __getitem__ = CanonicalRowLookup.__getitem__
    __contains__ = CanonicalRowLookup.__contains__

    def __len__(self):
        return 0 if self._root is None else self._root.size

    def __iter__(self):
        stack, node = [], self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key
            node = node.right


def freeze_row_keys(rows):
    if type(rows) is not CanonicalRowLookup:
        return None
    return _RowKeys(_FrozenRows(rows._root, rows._parent_root,
                               rows._removed, rows._added))

