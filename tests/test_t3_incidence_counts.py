"""Independent count proofs survive local edits and failed publication."""
from dataclasses import FrozenInstanceError
import random

import pytest

from anymesher._t3_incidence_counts import IncidenceCounts
from anymesher.errors import MeshError


def edge(n):
    return (n, n + 1)


def test_mapping_is_detached_immutable_and_reusable():
    values = {edge(0): 1, edge(1): 2}
    counts = IncidenceCounts.from_mapping(values)
    values[edge(0)] = 2
    assert counts[edge(0)] == 1
    assert IncidenceCounts.from_mapping(counts) is counts
    with pytest.raises(TypeError):
        counts[edge(0)] = 0
    with pytest.raises(FrozenInstanceError):
        counts._root = None
    assert counts.keys() == {edge(0), edge(1)}
    assert dict(counts) == {edge(0): 1, edge(1): 2}


@pytest.mark.parametrize("seed", range(5))
def test_randomized_persistent_proof_branches(seed):
    rng = random.Random(seed)
    values = {edge(n): 1 + n % 2 for n in range(100)}
    counts = IncidenceCounts.from_mapping(values)
    ancestors = []
    for _ in range(100):
        ancestors.append((counts, values.copy()))
        changes = {edge(rng.randrange(150)): rng.randrange(3) for _ in range(4)}
        counts = counts.with_updates(changes)
        for key, value in changes.items():
            if value:
                values[key] = value
            else:
                values.pop(key, None)
        assert dict(counts) == values
    for snapshot, expected in ancestors:
        assert dict(snapshot) == expected


@pytest.mark.parametrize("item", [
    ((0, 1), 3), ((0, 1), -1), ((0, 1), True), ((0, 1), 1.0),
    ((1, 0), 1), ((0, 0), 1), ((-1, 0), 1), ((0, 1, 2), 1),
])
def test_invalid_updates_cannot_change_source(item):
    counts = IncidenceCounts.from_mapping({edge(0): 1})
    root = counts._root
    with pytest.raises(MeshError, match="count snapshot"):
        counts.with_updates(dict([item]))
    assert counts._root is root and counts[edge(0)] == 1


def test_source_cannot_contain_zero_count():
    with pytest.raises(MeshError, match="count snapshot"):
        IncidenceCounts.from_mapping({edge(0): 0})
    counts = IncidenceCounts.from_mapping({edge(0): 1})
    assert len(counts.with_updates({edge(0): 0})) == 0
    assert counts.with_updates({edge(999): 0})._root is counts._root


def test_updates_and_rebinding_do_not_scan_prior_counts(monkeypatch):
    counts = IncidenceCounts.from_mapping({edge(n): 1 for n in range(1000)})
    def forbidden(*args):
        raise AssertionError("full proof scan")
    monkeypatch.setattr(IncidenceCounts, "__iter__", forbidden)
    monkeypatch.setattr(IncidenceCounts, "items", forbidden)
    assert IncidenceCounts.from_mapping(counts) is counts
    updated = counts.with_updates({edge(0): 2, edge(1): 0, edge(1001): 1})
    assert updated[edge(0)] == 2 and edge(1) not in updated
    assert counts[edge(0)] == 1 and edge(1) in counts


def test_mid_update_allocation_failure_keeps_prior_proof(monkeypatch):
    import anymesher._t3_edge_membership as module
    counts = IncidenceCounts.from_mapping({edge(0): 1})
    root = counts._root
    original = module._set
    calls = []
    def fail_second(*args):
        calls.append(True)
        if len(calls) == 2:
            raise MemoryError("second allocation")
        return original(*args)
    monkeypatch.setattr(module, "_set", fail_second)
    with pytest.raises(MemoryError):
        counts.with_updates({edge(0): 2, edge(1): 1})
    assert counts._root is root and dict(counts) == {edge(0): 1}


def test_insertion_reuses_proof_without_scanning_and_matches_oracle(monkeypatch):
    from test_t3_local_validation import mesh
    from anymesher import _t3_local_validation as local
    from anymesher import _t3_insertion_result as result
    subject = mesh()
    local.enable_local_validation(subject)
    initial = subject._local_validation_source
    initial_values = dict(initial.counts)
    assert initial.counts._root is not initial.index._edge_cells._root
    def forbidden(*args):
        raise AssertionError("full proof scan during insertion")
    with monkeypatch.context() as patch:
        patch.setattr(IncidenceCounts, "__iter__", forbidden)
        patch.setattr(IncidenceCounts, "items", forbidden)
        subject.insert_point((.21, .13), owner=42)
    current = subject._local_validation_source
    assert type(current.counts) is IncidenceCounts
    assert current.epoch == initial.epoch + 1
    edges, values = result._edges_and_counts(subject.triangles)
    assert dict(current.counts) == {
        tuple(map(int, e)): int(n) for e, n in zip(edges, values)}
    assert dict(initial.counts) == initial_values


def test_count_allocation_failure_rolls_back_whole_insertion(monkeypatch):
    from test_t3_local_validation import mesh, snapshot, unchanged
    from anymesher import _t3_local_validation as local
    subject = mesh()
    local.enable_local_validation(subject)
    before = snapshot(subject)
    def fail(*args):
        raise MemoryError("proof allocation")
    monkeypatch.setattr(IncidenceCounts, "with_updates", fail)
    with pytest.raises(MemoryError, match="proof allocation"):
        subject.insert_point((.21, .13), owner=42)
    unchanged(subject, before)

