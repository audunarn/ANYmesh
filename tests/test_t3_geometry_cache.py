"""Structural sharing, exact lookup and failure isolation for geometry caches."""
import random

import pytest

from anymesher._t3_geometry_cache import CellGeometryCache


def key(number):
    return (number, number + 1, number + 2)


def audit(node, lower=None, upper=None):
    if node is None:
        return 0, {}
    assert lower is None or lower < node.key
    assert upper is None or node.key < upper
    lh, left = audit(node.left, lower, node.key)
    rh, right = audit(node.right, node.key, upper)
    assert abs(lh - rh) <= 1
    assert node.height == 1 + max(lh, rh)
    return node.height, {**left, node.key: node.value, **right}


@pytest.mark.parametrize("seed", range(6))
def test_randomized_forks_preserve_exact_records(seed):
    rng = random.Random(seed)
    cache, expected = CellGeometryCache(), {}
    history = []
    for step in range(250):
        if step % 7 == 0:
            history.append((cache, expected.copy(), cache._root))
            removed = [key(rng.randrange(100)) for _ in range(5)]
            cache = cache.without(removed)
            expected = {k: v for k, v in expected.items() if k not in removed}
        else:
            cell, value = key(rng.randrange(100)), object()
            cache[cell] = value
            expected[cell] = value
        assert audit(cache._root)[1] == expected
        for cell, value in expected.items():
            assert cache.get(cell) is value
    for saved, values, root in history:
        assert saved._root is root
        assert audit(root)[1] == values


@pytest.mark.parametrize("order", [range(200), range(199, -1, -1)])
def test_rotations_deletion_and_shared_values(order):
    cache = CellGeometryCache()
    values = {key(n): object() for n in range(200)}
    for n in order:
        cache[key(n)] = values[key(n)]
    root = cache._root
    same = cache.without([key(999)])
    assert same is not cache and same._root is root
    descendant = cache.without([key(n) for n in range(0, 200, 2)])
    assert audit(descendant._root)[1] == {
        k: v for k, v in values.items() if k[0] % 2}
    assert audit(cache._root)[1] == values
    assert descendant.without(values)._root is None
    assert cache._root is root


def test_nodes_immutable_and_replacement_isolated():
    cache = CellGeometryCache()
    first, second = object(), object()
    cache[key(1)] = first
    child = cache.without([])
    root = cache._root
    child[key(1)] = second
    assert cache.get(key(1)) is first
    assert child.get(key(1)) is second
    with pytest.raises(AttributeError):
        root.value = second
    child[key(1)] = second
    saved = child._root
    child[key(1)] = second
    assert child._root is saved


def test_failed_removal_iterator_does_not_modify_parent():
    cache = CellGeometryCache()
    cache[key(1)] = object()
    root = cache._root
    def failing():
        yield key(1)
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        cache.without(failing())
    assert cache._root is root


def test_failed_allocation_does_not_publish(monkeypatch):
    import anymesher._t3_geometry_cache as module
    cache = CellGeometryCache()
    cache[key(1)] = object()
    root = cache._root
    def fail(*args):
        raise MemoryError("allocation")
    monkeypatch.setattr(module, "_make", fail)
    with pytest.raises(MemoryError):
        cache[key(2)] = object()
    assert cache._root is root


def test_empty_cache_and_explicit_default():
    cache = CellGeometryCache()
    sentinel = object()
    assert cache.get(key(0)) is None
    assert cache.get(key(0), sentinel) is sentinel
    assert cache.without([key(0)])._root is None


def test_membership_preserves_retained_and_removed_snapshot_records():
    cache = CellGeometryCache()
    assert key(0) not in cache
    cache[key(0)] = None
    cache[key(1)] = object()
    cache[key(2)] = False
    original_root = cache._root
    child = cache.without([key(1)])
    child_root = child._root
    assert key(0) in child
    assert key(1) not in child
    assert key(2) in child
    assert key(3) not in child
    assert key(1) in cache
    assert cache._root is original_root
    assert child._root is child_root
