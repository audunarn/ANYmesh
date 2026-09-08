"""Canonical export consumes qualified local changes without full key scans."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from anymesher import native_v2
from anymesher import _t3_canonical_export as cached
from anymesher import _t3_row_lookup as ranks
from anymesher._t3_edge_membership import EdgeMembershipMap
from test_t3_canonical_export import grid, oracle


def test_local_export_never_scans_all_keys_or_canonical_map(monkeypatch):
    points, rows = grid()
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    previous = topology._topology_index._canonical_export_snapshot
    saved = dict(previous.by_cell)
    topology.insert_point((.18, .13))
    assert type(topology._topology_index._row_by_cell) is ranks.CanonicalRowLookup
    expected = topology.canonical_export()
    def forbidden(*args):
        raise AssertionError("full export key/map scan")
    with monkeypatch.context() as patch:
        patch.setattr(ranks._RowKeys, "__iter__", forbidden)
        patch.setattr(EdgeMembershipMap, "__iter__", forbidden)
        actual = cached.canonical_frontal_export(topology)
    for value, reference in zip(actual, expected):
        assert value.tobytes() == reference.tobytes()
    assert dict(previous.by_cell) == saved


def test_frozen_key_binding_does_not_follow_lookup_attribute_changes():
    points, rows = grid(1)
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    snapshot = topology._topology_index._canonical_export_snapshot
    lookup = topology._topology_index._row_by_cell
    keys = snapshot.source_keys
    expected = set(keys)
    lookup._root = None
    lookup._cells = np.empty(0, dtype=object)
    assert set(keys) == expected
    with pytest.raises(FrozenInstanceError):
        keys._mapping._root = None


def test_unknown_mapping_has_no_unqualified_local_key_binding():
    points, rows = grid(1)
    cells = {tuple(map(int, row)): n for n, row in enumerate(rows)}
    result = cached.CanonicalTriangleExport.prepare(
        None, points, cells, native_v2._canonical_triangle, lambda _: None)
    assert result.source_keys is None
    assert result.rows.tobytes() == oracle(points, rows).tobytes()


def test_shared_export_map_forks_without_changing_ancestor():
    points, rows = grid()
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    previous = topology._topology_index._canonical_export_snapshot
    clone = previous.by_cell.copy()
    assert type(clone) is EdgeMembershipMap
    expected = dict(previous.by_cell)
    identity = next(iter(clone))
    del clone[identity]
    assert dict(previous.by_cell) == expected
    topology.insert_point((.18, .13))
    cached.canonical_frontal_export(topology)
    assert dict(previous.by_cell) == expected


def test_export_allocation_failure_keeps_previous_snapshot(monkeypatch):
    points, rows = grid()
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    previous = topology._topology_index._canonical_export_snapshot
    saved = dict(previous.by_cell)
    topology.insert_point((.18, .13))
    expected = topology.canonical_export()
    def fail(*args):
        raise MemoryError("export map allocation")
    monkeypatch.setattr(EdgeMembershipMap, "copy", fail)
    with pytest.raises(MemoryError, match="export map allocation"):
        cached.canonical_frontal_export(topology)
    assert topology._topology_index._canonical_export_snapshot is previous
    assert dict(previous.by_cell) == saved
    for actual, reference in zip(topology.canonical_export(), expected):
        assert actual.tobytes() == reference.tobytes()


def test_frozen_key_differences_match_arbitrary_set_operands():
    points, rows = grid(1)
    topology = native_v2.MutableT3Topology(points, rows, ())
    keys = ranks.freeze_row_keys(topology._topology_index._row_by_cell)
    expected = set(keys)
    assert keys - set() == expected
    assert keys - expected == set()
    assert expected - keys == set()
    assert keys - keys == set()

