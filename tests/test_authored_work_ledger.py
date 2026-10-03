"""Authored-root work consumes an existing budget, never a fresh one."""

from types import SimpleNamespace

import pytest

from anymesher._authored_work_ledger import AuthoredWorkLedger
from anymesher.errors import MeshError


OPTIONS = SimpleNamespace(max_insertions=10, max_topology_operations=20)


def report(**changes):
    data = dict(selected_route="frontal_delaunay", cancelled=False,
                insertion_budget=10, topology_budget=20, insertions=3,
                topology_operations=8, shared_segment_splits=1,
                shared_nodes=[{"node_id": 11}], reserved_node_reuses=2,
                staged_point_insertions=5)
    data.update(changes)
    return data


def test_existing_work_is_debited_and_charge_is_immutable():
    ledger = AuthoredWorkLedger.from_native_report(OPTIONS, report())
    assert (ledger.remaining_insertions, ledger.remaining_operations) == (5, 12)
    later = ledger.charge(insertions=2, operations=4)
    assert (later.remaining_insertions, later.remaining_operations) == (3, 8)
    assert (ledger.remaining_insertions, ledger.remaining_operations) == (5, 12)


def test_missing_changed_or_partial_receipts_refuse():
    for changed in (
        {"insertion_budget": 11},
        {"staged_point_insertions": 4},
        {"topology_operations": 21},
        {"reserved_node_reuses": -1},
        {"selected_route": "other_engine"},
        {"cancelled": True},
    ):
        with pytest.raises(MeshError):
            AuthoredWorkLedger.from_native_report(OPTIONS, report(**changed))
    incomplete = report()
    del incomplete["topology_budget"]
    with pytest.raises(MeshError, match="lacks original budget"):
        AuthoredWorkLedger.from_native_report(OPTIONS, incomplete)
    incomplete = report()
    del incomplete["reserved_node_reuses"]
    with pytest.raises(MeshError, match="incomplete"):
        AuthoredWorkLedger.from_native_report(OPTIONS, incomplete)


def test_exhaustion_and_invalid_new_counts_do_not_modify_ledger():
    ledger = AuthoredWorkLedger.from_native_report(OPTIONS, report())
    with pytest.raises(MeshError, match="exceed"):
        ledger.charge(insertions=6, operations=0)
    with pytest.raises(MeshError, match="exceed"):
        ledger.charge(insertions=0, operations=13)
    with pytest.raises(MeshError, match="nonnegative"):
        ledger.charge(insertions=True, operations=0)
    assert (ledger.remaining_insertions, ledger.remaining_operations) == (5, 12)
