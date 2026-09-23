"""Persistent edge membership matches dictionary values without full copies."""
import random

import pytest

from anymesher._t3_edge_membership import EdgeMembershipMap


def edge(n):
    return (n, n + 1)


def check_tree(node, lower=None, upper=None):
    if node is None:
        return 0, 0
    assert lower is None or lower < node.key
    assert upper is None or node.key < upper
    left, nl = check_tree(node.left, lower, node.key)
    right, nr = check_tree(node.right, node.key, upper)
    assert node.height == 1 + max(left, right)
    assert abs(left - right) <= 1
    return node.height, nl + nr + 1


@pytest.mark.parametrize("seed", range(5))
def test_branch_updates_match_dictionary(seed):
    rng = random.Random(seed)
    reference = {edge(n): ((n, n + 1, n + 2),) for n in range(100)}
    current = EdgeMembershipMap.from_items(reference.items())
    history = []
    for step in range(250):
        if step % 9 == 0:
            history.append((current, reference.copy(), current._root))
            current = current.copy()
        key = edge(rng.randrange(150))
        if rng.randrange(3) == 0 and key in reference:
            del current[key]
            del reference[key]
        else:
            value = ((step, step + 1, step + 2),)
            current[key] = reference[key] = value
        assert len(current) == len(reference)
        assert current.keys() == reference.keys()
        assert dict(current) == reference
        assert list(current) == sorted(reference)
        assert check_tree(current._root)[1] == len(current)
    for saved, values, root in history:
        assert saved._root is root
        assert dict(saved) == values


def test_copy_shares_root_and_unchanged_membership():
    values = {edge(n): ((n, n + 1, n + 2),) for n in range(100)}
    parent = EdgeMembershipMap.from_items(values.items())
    child = parent.copy()
    assert child._root is parent._root
    child[edge(0)] = ((100, 101, 102),)
    del child[edge(1)]
    assert child[edge(99)] is parent[edge(99)]
    assert parent[edge(0)] is values[edge(0)]
    assert edge(1) in parent
    assert child._root is not parent._root


def test_empty_duplicate_missing_and_none():
    cache = EdgeMembershipMap.from_items([])
    assert len(cache) == 0 and list(cache.items()) == []
    with pytest.raises(ValueError, match="duplicate"):
        EdgeMembershipMap.from_items([(edge(0), ()), (edge(0), ())])
    with pytest.raises(KeyError):
        del cache[edge(0)]
    with pytest.raises(KeyError):
        cache[edge(0)]
    cache[edge(0)] = None
    assert edge(0) in cache and cache[edge(0)] is None
    del cache[edge(0)]
    assert len(cache) == 0 and cache._root is None


def test_failed_update_and_delete_do_not_change_root_or_size(monkeypatch):
    import anymesher._t3_edge_membership as module
    cache = EdgeMembershipMap.from_items([(edge(0), ())])
    root = cache._root
    def fail(*args):
        raise MemoryError("allocation")
    monkeypatch.setattr(module, "_set", fail)
    monkeypatch.setattr(module, "_erase", fail)
    with pytest.raises(MemoryError):
        cache[edge(1)] = ()
    assert cache._root is root and len(cache) == 1
    with pytest.raises(MemoryError):
        del cache[edge(0)]
    assert cache._root is root and len(cache) == 1


def test_copy_never_iterates_memberships(monkeypatch):
    cache = EdgeMembershipMap.from_items((edge(n), ()) for n in range(1000))
    def forbidden(*args):
        raise AssertionError("full map scan")
    monkeypatch.setattr(EdgeMembershipMap, "__iter__", forbidden)
    child = cache.copy()
    assert child._root is cache._root
    assert len(child) == 1000
    child[edge(1001)] = ((1, 2, 3),)
    del child[edge(3)]
    assert child[edge(1001)] == ((1, 2, 3),)
    assert edge(3) not in child and edge(3) in cache


def test_bulk_build_is_balanced_and_preserves_values():
    values = [(edge(n), ((n, n + 1, n + 2),)) for n in range(8191)]
    cache = EdgeMembershipMap.from_items(reversed(values))
    assert check_tree(cache._root) == (13, 8191)
    assert all(cache[k] is value for k, value in values)

