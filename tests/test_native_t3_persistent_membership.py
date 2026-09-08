"""Behavioral snapshot persistence, branching and insertion parity."""
from __future__ import annotations

import gc
import numpy as np
import pytest

from anymesher.triangulation import orient2d

native = pytest.importorskip("anymesher._native")


def grid(n=6):
    points = np.asarray([(x / n, y / n)
                         for y in range(n + 1) for x in range(n + 1)])
    cells = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            b, d = a + 1, a + n + 1
            cells.extend(((a, b, d + 1), (a, d + 1, d)))
    return points, np.asarray(sorted(cells), dtype=np.int64)


def snapshot(rows, previous=None):
    return native.native_v2_t3_incidence(np.ascontiguousarray(rows), previous)


def check(state, rows):
    assert native.native_v2_t3_incidence_check(state, np.ascontiguousarray(rows)) is True


def compare_insertion(points, rows, state, candidate):
    protected = np.empty((0, 2), dtype=np.int64)
    outputs, failures = [], []
    for handle in (None, state):
        try:
            outputs.append(native.native_v2_mutable_t3_insert(
                points, rows, protected, orient2d,
                float(candidate[0]), float(candidate[1]), handle))
            failures.append(None)
        except RuntimeError as error:
            outputs.append(None)
            failures.append(str(error))
    assert failures[0] == failures[1]
    assert outputs[0] == outputs[1]
    check(state, rows)
    return outputs[0]


def test_long_branching_history_and_reordered_rows():
    rng = np.random.default_rng(47261)
    points, rows = grid()
    state = snapshot(rows)
    retained = [(points.copy(), rows.copy(), state)]
    successes = 0
    for step in range(48):
        if step % 7 == 6:
            points, rows, state = retained[step // 7 % len(retained)]
            points, rows = points.copy(), rows.copy()
        cell = rows[int(rng.integers(len(rows)))]
        weights = rng.uniform(.2, .8, 3)
        weights /= weights.sum()
        candidate = weights @ points[cell]
        value = compare_insertion(points, rows, state, candidate)
        if value is None:
            continue  # Same typed native rejection in both independent routes.
        cells, diagnostic = value
        assert diagnostic["added_triangles"] > 0
        points = np.vstack((points, candidate))
        new_rows = np.asarray(cells, dtype=np.int64)
        old_state, old_rows = state, rows
        state = snapshot(new_rows, state)
        rows = new_rows
        check(old_state, old_rows)
        check(state, rows)
        if step % 5 == 0:
            # Export-row changes cannot change membership identity.
            rows = rows[rng.permutation(len(rows))].copy()
            state = snapshot(rows, state)
        retained.append((points.copy(), rows.copy(), state))
        successes += 1
    assert successes >= 32
    # Every ancestor remains usable after all descendant edits and row changes.
    for points, rows, state in retained[::5]:
        candidate = np.asarray((.317293, .418971))
        compare_insertion(points, rows, state, candidate)


@pytest.mark.parametrize("order", ["forward", "reverse", "random"])
def test_deletion_to_empty_repopulation_and_source_lifetime(order):
    points, rows = grid(9)
    root = snapshot(rows)
    state = root
    permutation = np.arange(len(rows))
    if order == "reverse":
        permutation = permutation[::-1]
    elif order == "random":
        permutation = np.random.default_rng(882).permutation(permutation)
    kept = np.ones(len(rows), dtype=bool)
    ancestors = [(root, rows.copy())]
    for ordinal, row in enumerate(permutation):
        kept[row] = False
        active = rows[kept]
        state = snapshot(active, state)
        check(state, active)
        if ordinal % 23 == 0:
            ancestors.append((state, active.copy()))
    empty = np.empty((0, 3), dtype=np.int64)
    check(state, empty)
    restored = snapshot(rows, state)
    del state
    gc.collect()
    check(restored, rows)
    compare_insertion(points, rows, restored, np.asarray((.211713, .631729)))
    for ancestor, original in ancestors:
        check(ancestor, original)
    check(root, rows)


@pytest.mark.parametrize("kind", ["duplicate", "reversed_duplicate", "repeated_node", "negative"])
def test_failed_update_keeps_prior_snapshot_usable(kind):
    points, rows = grid()
    root = snapshot(rows)
    invalid = rows.copy()
    if kind == "duplicate":
        invalid[1] = invalid[0]
    elif kind == "reversed_duplicate":
        invalid[1] = invalid[0][::-1]
    elif kind == "repeated_node":
        invalid[1, 1] = invalid[1, 0]
    else:
        invalid[1, 0] = -1
    with pytest.raises(RuntimeError, match="distinct triangle rows"):
        snapshot(invalid, root)
    check(root, rows)
    compare_insertion(points, rows, root, np.asarray((.211713, .631729)))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16])
@pytest.mark.parametrize("order", ["forward", "reverse", "random"])
def test_bulk_and_incremental_builds_have_identical_membership(n, order):
    points, rows = grid(n)
    if order == "reverse":
        rows = rows[::-1].copy()
    elif order == "random":
        rows = rows[np.random.default_rng(641).permutation(len(rows))].copy()
    bulk = snapshot(rows)
    empty = np.empty((0, 3), dtype=np.int64)
    incremental = snapshot(empty)
    ancestors = [(incremental, empty)]
    step = max(1, len(rows) // 7)
    for count in range(step, len(rows), step):
        prefix = rows[:count].copy()
        incremental = snapshot(prefix, incremental)
        ancestors.append((incremental, prefix))
    incremental = snapshot(rows, incremental)
    for state in (bulk, incremental):
        check(state, rows)
    for candidate in ((.211713, .631729), (.791317, .417291), (.413219, .219713)):
        first = compare_insertion(points, rows, bulk, np.asarray(candidate))
        second = compare_insertion(points, rows, incremental, np.asarray(candidate))
        assert first == second
    # Bulk-built ancestors remain valid through subsequent removals and inserts.
    subset = rows[::2].copy()
    smaller = snapshot(subset, bulk)
    restored = snapshot(rows, smaller)
    check(bulk, rows)
    check(smaller, subset)
    check(restored, rows)
    for ancestor, prefix in ancestors:
        check(ancestor, prefix)


@pytest.mark.parametrize("slot", ["first", "middle", "last"])
def test_rank_updates_preserve_export_positions_and_ancestor_lookup(slot):
    points, rows = grid(14)
    initial = rows.copy()
    root = snapshot(rows)
    position = {"first": 0, "middle": len(rows) // 2, "last": len(rows) - 1}[slot]
    retained = [(root, rows.copy())]
    state = root
    for step in range(12):
        active = np.delete(rows, position, axis=0) if step % 2 == 0 else rows.copy()
        state = snapshot(active, state)
        check(state, active)
        cell = active[len(active) // 3]
        candidate = np.asarray((.213, .327, .460)) @ points[cell]
        compare_insertion(points, active, state, candidate)
        retained.append((state, active.copy()))
    for state, original in retained:
        check(state, original)
    check(root, initial)


@pytest.mark.parametrize("permutation", [(1, 2, 0), (2, 0, 1), (0, 2, 1)])
def test_replaced_cell_identity_allows_reorientation_not_duplicate(permutation):
    points, rows = grid(8)
    root = snapshot(rows)
    changed = rows.copy()
    changed[7] = changed[7, list(permutation)]
    changed = np.asarray(sorted(map(tuple, changed)), dtype=np.int64)
    updated = snapshot(changed, root)
    check(updated, changed)
    compare_insertion(points, changed, updated, np.asarray((.217319, .671321)))
    # Both inputs are strictly ordered, but share one unoriented identity.
    duplicate = np.asarray(sorted((*map(tuple, changed), tuple(rows[7]))), dtype=np.int64)
    with pytest.raises(RuntimeError, match="distinct triangle rows"):
        snapshot(duplicate, updated)
    check(updated, changed)
    check(root, rows)


def test_repeated_ordered_unordered_transitions_keep_independent_rank():
    points, rows = grid(10)
    rng = np.random.default_rng(881)
    state = snapshot(rows)
    ancestors = [(state, rows.copy())]
    for step in range(16):
        current = rows[rng.permutation(len(rows))].copy() if step % 2 else rows.copy()
        state = snapshot(current, state)
        check(state, current)
        compare_insertion(points, current, state, np.asarray((.311729, .713217)))
        ancestors.append((state, current.copy()))
    for ancestor, current in ancestors:
        check(ancestor, current)

def test_snapshots_own_input_storage_and_reject_wrong_binding():
    points, rows = grid()
    original = rows.copy()
    state = snapshot(rows)
    rows[:] = rows[::-1].copy()
    with pytest.raises(RuntimeError, match="connectivity mismatch"):
        check(state, rows)
    check(state, original)
    compare_insertion(points, original, state, np.asarray((.211713, .631729)))
