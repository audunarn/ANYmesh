"""Exact insertion correspondence is local, defensive and failure-safe."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from anymesher import native_cpp, native_v2
from anymesher import _t3_correspondence as reuse
from anymesher import _t3_insertion_result as result
from anymesher._t3_incidence import T3IncidenceIndex


def arrays():
    before = np.asarray(((0, 1, 2), (0, 2, 3), (2, 4, 3)), dtype=np.int64)
    after = np.asarray(((0, 1, 5), (0, 5, 3), (1, 2, 5),
                        (2, 3, 5), (2, 4, 3)), dtype=np.int64)
    return before, after


def observe(monkeypatch):
    calls = []
    original = result._compute_source_rows
    def compute(before, after):
        calls.append((before.copy(), after.copy()))
        return original(before, after)
    monkeypatch.setattr(result, "_compute_source_rows", compute)
    return calls


def test_one_owned_correspondence_per_insertion_scope(monkeypatch):
    before, after = arrays()
    calls = observe(monkeypatch)
    @reuse.insertion_correspondence_scope
    def operation():
        expected = result._source_rows(before, after)
        expected[-1] = 999
        for _ in range(3):
            actual = result._source_rows(before.copy(), after.copy())
            np.testing.assert_array_equal(actual, [-1, -1, -1, -1, 2])
        old, new, source = reuse.current_correspondence(before, after)
        assert not old.flags.writeable
        assert not new.flags.writeable
        assert not source.flags.writeable
        return actual
    operation()
    assert len(calls) == 1
    assert reuse._ACTIVE.get() is None
    operation()
    assert len(calls) == 2


@pytest.mark.parametrize("which", ["before", "after", "dtype", "reorder"])
def test_mutable_input_change_never_reuses_stale_mapping(monkeypatch, which):
    before, after = arrays()
    calls = observe(monkeypatch)
    @reuse.insertion_correspondence_scope
    def operation():
        nonlocal before, after
        result._source_rows(before, after)
        if which == "before":
            before[-1] = (3, 4, 2)
        elif which == "after":
            after[-1] = (3, 4, 2)
        elif which == "dtype":
            after = after.astype(np.int32)
        else:
            before = before[::-1].copy()
        expected = result._compute_source_rows(before, after)
        actual = result._source_rows(before, after)
        np.testing.assert_array_equal(actual, expected)
    operation()
    assert len(calls) == 3  # Initial, independent oracle, changed binding.


def test_nested_and_exceptional_scopes_restore_parent(monkeypatch):
    before, after = arrays()
    calls = observe(monkeypatch)
    error = KeyboardInterrupt("nested cancellation")
    @reuse.insertion_correspondence_scope
    def child():
        result._source_rows(before, after)
        raise error
    @reuse.insertion_correspondence_scope
    def parent():
        result._source_rows(before, after)
        saved = reuse._ACTIVE.get()
        with pytest.raises(KeyboardInterrupt) as caught:
            child()
        assert caught.value is error
        assert reuse._ACTIVE.get() is saved
        result._source_rows(before, after)
    parent()
    assert len(calls) == 2
    assert reuse._ACTIVE.get() is None


def test_threads_do_not_share_scopes():
    barrier = Barrier(2)
    @reuse.insertion_correspondence_scope
    def worker(offset):
        before, after = arrays()
        before += offset
        after += offset
        first = result._source_rows(before, after)
        scope = reuse._ACTIVE.get()
        barrier.wait(timeout=5)
        np.testing.assert_array_equal(first, result._source_rows(before, after))
        assert reuse._ACTIVE.get() is scope
        return scope
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(worker, 0)
        two = pool.submit(worker, 100)
        assert one.result(timeout=10) is not two.result(timeout=10)
    assert reuse._ACTIVE.get() is None


def test_outside_insertion_keeps_uncached_reference(monkeypatch):
    before, after = arrays()
    calls = observe(monkeypatch)
    result._source_rows(before, after)
    result._source_rows(before, after)
    assert len(calls) == 2
    assert reuse.current_correspondence(before, after) is None


def test_incidence_reuses_exact_retained_tuple_identities(monkeypatch):
    before, after = arrays()
    index = T3IncidenceIndex(before)
    expected = T3IncidenceIndex(after)
    previous_cells = index._row_cells
    previous_matrix = index._row_matrix.copy()
    @reuse.insertion_correspondence_scope
    def operation():
        source = result._source_rows(before, after)
        def forbidden(*args):
            raise AssertionError("full Python row reconstruction")
        monkeypatch.setattr(T3IncidenceIndex, "_read_rows", staticmethod(forbidden))
        updated = index.updated(after)
        assert updated._row_by_cell == expected._row_by_cell
        for row in np.flatnonzero(source >= 0):
            assert updated._row_cells[row] is index._row_cells[source[row]]
        for edge in expected._edge_cells:
            assert updated.attached(edge) == expected.attached(edge)
        np.testing.assert_array_equal(updated.canonical_edges(), expected.canonical_edges())
        assert not updated._row_cells.flags.writeable
        assert not updated._row_matrix.flags.writeable
        return updated
    updated = operation()
    assert index._row_cells is previous_cells
    np.testing.assert_array_equal(index._row_matrix, previous_matrix)
    assert updated._row_matrix is not index._row_matrix


@pytest.mark.parametrize("mismatch", ["source_order", "after_change", "int32"])
def test_unmatched_update_falls_back_without_trusting_cache(monkeypatch, mismatch):
    before, after = arrays()
    index = T3IncidenceIndex(before[::-1] if mismatch == "source_order" else before)
    original = T3IncidenceIndex._read_rows
    calls = []
    def read(rows, callback):
        calls.append(1)
        return original(rows, callback)
    monkeypatch.setattr(T3IncidenceIndex, "_read_rows", staticmethod(read))
    @reuse.insertion_correspondence_scope
    def operation():
        result._source_rows(before, after)
        selected = after
        if mismatch == "after_change":
            selected = after[::-1].copy()
        elif mismatch == "int32":
            selected = after.astype(np.int32)
        updated = index.updated(selected)
        assert updated._row_by_cell == {tuple(map(int, row)): i
                                       for i, row in enumerate(selected)}
    operation()
    assert len(calls) == 1


def large_topology():
    n = 49
    points = np.array([(x / n, y / n) for y in range(n + 1) for x in range(n + 1)])
    rows = []
    edges = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            b, d = a + 1, a + n + 1
            rows.extend(((a, b, d + 1), (a, d + 1, d)))
    for i in range(n):
        edges.extend(((i, i + 1), (n * (n + 1) + i, n * (n + 1) + i + 1),
                      (i * (n + 1), (i + 1) * (n + 1)),
                      (i * (n + 1) + n, (i + 1) * (n + 1) + n)))
    return native_v2.MutableT3Topology(
        points, np.asarray(rows), edges,
        triangle_owners=np.arange(len(rows)) % 7,
    )


def require_native():
    if not native_cpp.COMPILED_NATIVE_V2_AVAILABLE:
        pytest.skip("optional native-v2 extension is absent")
    assert callable(getattr(native_cpp._compiled,
                            "native_v2_mutable_t3_insert_cancellable", None))


def test_actual_large_insertion_computes_correspondence_once(monkeypatch):
    require_native()
    topology = large_topology()
    point = np.asarray((.371923, .229573))
    expected_rows, expected_owners, expected_report = topology._python_insert_with_owners(
        point, owner=42)
    previous = topology._topology_index
    old_keys = set(previous._row_by_cell)
    calls = observe(monkeypatch)
    def forbidden(*args):
        raise AssertionError("global row tuple reconstruction during insertion")
    monkeypatch.setattr(T3IncidenceIndex, "_read_rows", staticmethod(forbidden))
    report = topology.insert_point(point, owner=42, cancellation_check=lambda _: None)
    assert report["native"] is True
    assert len(calls) == 1
    assert report["added_triangles"] == expected_report["added_triangles"]
    np.testing.assert_array_equal(topology.triangles, expected_rows)
    np.testing.assert_array_equal(topology.triangle_owners, expected_owners)
    assert set(previous._row_by_cell) == old_keys
    assert reuse._ACTIVE.get() is None
    topology.validate()


def test_incidence_cancellation_rolls_back_scope_and_topology():
    require_native()
    topology = large_topology()
    before = (topology.points, topology.triangles, topology._topology_index,
              topology.node_owners.copy(), topology.triangle_owners.copy(), topology.epoch)
    error = RuntimeError("cancel reused row publication")
    def cancel(phase):
        if phase == "native-v2 incidence scan":
            raise error
    with pytest.raises(RuntimeError) as caught:
        topology.insert_point((.371923, .229573), owner=42, cancellation_check=cancel)
    assert caught.value is error
    assert topology._topology_index is before[2]
    assert topology.epoch == before[5]
    for current, expected in zip(
        (topology.points, topology.triangles, topology.node_owners, topology.triangle_owners),
        (before[0], before[1], before[3], before[4]),
    ):
        np.testing.assert_array_equal(current, expected)
    assert reuse._ACTIVE.get() is None

