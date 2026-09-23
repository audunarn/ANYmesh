"""Persistent cell-geometry cache for detached T3 incidence snapshots.

Only root references are mutable. Immutable AVL nodes share surviving records;
a rejected descendant cannot modify its source's cached geometry.
"""
from __future__ import annotations

from typing import NamedTuple


class _Node(NamedTuple):
    key: tuple[int, ...]
    value: object
    left: _Node | None
    right: _Node | None
    height: int
    size: int


def _height(node):
    return 0 if node is None else node.height


def _make(key, value, left, right):
    return _Node(key, value, left, right, 1 + max(_height(left), _height(right)),
                 1 + (0 if left is None else left.size)
                 + (0 if right is None else right.size))


def _left(node):
    pivot = node.right
    return _make(pivot.key, pivot.value,
                 _make(node.key, node.value, node.left, pivot.left), pivot.right)


def _right(node):
    pivot = node.left
    return _make(pivot.key, pivot.value, pivot.left,
                 _make(node.key, node.value, pivot.right, node.right))


def _balance(node):
    difference = _height(node.left) - _height(node.right)
    if difference > 1:
        if _height(node.left.left) < _height(node.left.right):
            node = _make(node.key, node.value, _left(node.left), node.right)
        return _right(node)
    if difference < -1:
        if _height(node.right.right) < _height(node.right.left):
            node = _make(node.key, node.value, node.left, _right(node.right))
        return _left(node)
    return node


def _set(node, key, value):
    if node is None:
        return _make(key, value, None, None)
    if key == node.key:
        if value is node.value:
            return node
        return _make(key, value, node.left, node.right)
    if key < node.key:
        left = _set(node.left, key, value)
        if left is node.left:
            return node
        return _balance(_make(node.key, node.value, left, node.right))
    right = _set(node.right, key, value)
    if right is node.right:
        return node
    return _balance(_make(node.key, node.value, node.left, right))


def _erase(node, key):
    if node is None:
        return None
    if key < node.key:
        left = _erase(node.left, key)
        if left is node.left:
            return node
        return _balance(_make(node.key, node.value, left, node.right))
    if key > node.key:
        right = _erase(node.right, key)
        if right is node.right:
            return node
        return _balance(_make(node.key, node.value, node.left, right))
    if node.left is None:
        return node.right
    if node.right is None:
        return node.left
    successor = node.right
    while successor.left is not None:
        successor = successor.left
    return _balance(_make(successor.key, successor.value, node.left,
                          _erase(node.right, successor.key)))


class CellGeometryCache:
    """Private append-only-geometry cache; forks remove only deleted cells."""

    __slots__ = ("_root",)

    def __init__(self):
        self._root = None

    def __contains__(self, key):
        node = self._root
        while node is not None:
            if key == node.key:
                return True
            node = node.left if key < node.key else node.right
        return False

    def get(self, key, default=None):
        node = self._root
        while node is not None:
            if key == node.key:
                return node.value
            node = node.left if key < node.key else node.right
        return default

    def __setitem__(self, key, value):
        # Complete allocation before changing this snapshot's root.
        root = _set(self._root, key, value)
        self._root = root

    def without(self, removed):
        root = self._root
        for key in removed:
            root = _erase(root, key)
        result = type(self)()
        result._root = root
        return result
