"""Structurally shared edge membership for detached T3 incidence snapshots."""
from __future__ import annotations

from collections.abc import Mapping

from ._t3_geometry_cache import _make, _set, _erase

_MISSING = object()


def _build(items, first, last):
    if first >= last:
        return None
    middle = (first + last) // 2
    key, value = items[middle]
    return _make(key, value, _build(items, first, middle),
                 _build(items, middle + 1, last))


class EdgeMembershipMap(Mapping):
    """Private mutable root over immutable nodes and immutable memberships.

    Iteration is canonical key order. Consumers must not derive element
    numbering from incidental dictionary insertion order.
    """
    __slots__ = ("_root", "_size")

    @classmethod
    def from_items(cls, source):
        items = sorted(source)
        if any(items[i - 1][0] == items[i][0] for i in range(1, len(items))):
            raise ValueError("duplicate edge membership key")
        result = cls()
        result._root = _build(items, 0, len(items))
        result._size = len(items)
        return result

    def __init__(self):
        self._root = None
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        stack = []
        node = self._root
        while stack or node is not None:
            while node is not None:
                stack.append(node)
                node = node.left
            node = stack.pop()
            yield node.key
            node = node.right

    def get(self, key, default=None):
        node = self._root
        while node is not None:
            if key == node.key:
                return node.value
            node = node.left if key < node.key else node.right
        return default

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self.get(key, _MISSING) is not _MISSING

    def __setitem__(self, key, value):
        added = self.get(key, _MISSING) is _MISSING
        root = _set(self._root, key, value)
        self._root = root
        self._size += added

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        root = _erase(self._root, key)
        self._root = root
        self._size -= 1

    def copy(self):
        result = type(self)()
        result._root = self._root
        result._size = self._size
        return result

