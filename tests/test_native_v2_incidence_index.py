"""Small differential and rollback checks for persistent incidence."""

import numpy as np
import pytest

from anymesher._t3_incidence import T3IncidenceIndex
from anymesher.errors import MeshError
from anymesher import native_v2


def reference(rows):
    result = {}
    for row, cell in enumerate(rows):
        for a, b in zip(cell, (*cell[1:], cell[0])):
            result.setdefault(tuple(sorted((int(a), int(b)))), []).append(row)
    return {edge: tuple(attached) for edge, attached in result.items()}


def assert_index(index, rows):
    expected = reference(rows)
    assert set(index._edge_cells) == set(expected)
    assert {edge: index.attached(edge) for edge in expected} == expected
    missing = max((node for edge in expected for node in edge), default=-1) + 1
    assert index.attached((missing, missing + 1)) == ()


def test_local_update_preserves_untouched_memberships_and_old_snapshot():
    before = [(0, 1, 2), (0, 2, 3)]
    after = [(0, 1, 4), (0, 2, 3), (0, 4, 2), (1, 2, 4)]
    index = T3IncidenceIndex(before)
    updated = index.updated(after)
    assert_index(index, before)
    assert_index(updated, after)
    assert updated._edge_cells[(2, 3)] is index._edge_cells[(2, 3)]
    assert updated._edge_cells[(0, 2)] is not index._edge_cells[(0, 2)]


def test_row_reordering_keeps_memberships_but_resolves_current_rows():
    rows = [(0, 1, 2), (0, 2, 3), (2, 4, 3)]
    index = T3IncidenceIndex(rows)
    reordered = [rows[2], rows[0], rows[1]]
    updated = index.updated(reordered)
    assert_index(updated, reordered)
    assert all(updated._edge_cells[e] is cells for e, cells in index._edge_cells.items())
    assert_index(updated.updated(rows), rows)


@pytest.mark.parametrize("stage", ["native-v2 incidence scan", "native-v2 incidence update"])
def test_cancelled_index_update_never_mutates_source(stage):
    before = [(0, 1, 2), (0, 2, 3)]
    index = T3IncidenceIndex(before)

    def cancel(value):
        if value == stage:
            raise RuntimeError("cancel index")

    with pytest.raises(RuntimeError, match="cancel index"):
        index.updated([(0, 1, 3), (1, 2, 3)], cancellation_check=cancel)
    assert_index(index, before)


def make_topology(monkeypatch):
    monkeypatch.setattr(native_v2, "native_mutable_t3_insert", lambda *args: None)
    return native_v2.MutableT3Topology(
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
        [(0, 1), (1, 2), (2, 3), (0, 3)],
    )


def test_insertion_sequence_matches_reference_and_keeps_old_indices(monkeypatch):
    topology = make_topology(monkeypatch)
    for point in [(0.2, 0.15), (0.8, 0.2), (0.6, 0.75)]:
        before_rows = topology.triangles
        before_index = topology._topology_index
        topology.insert_point(point)
        assert_index(topology._topology_index, topology.triangles)
        assert_index(before_index, before_rows)
        assert topology._incidence() == {
            edge: list(rows) for edge, rows in reference(topology.triangles).items()
        }
    assert topology.epoch == 3


def test_insertion_index_failure_restores_all_local_state(monkeypatch):
    topology = make_topology(monkeypatch)
    before_points, before_rows = topology.points, topology.triangles
    before_index = topology._topology_index
    before_owners = topology.triangle_owners.copy()
    topology.quality_cache[(0, 1, 2)] = (1.0,)

    def fail(*args, **kwargs):
        raise MemoryError("index allocation")

    monkeypatch.setattr(T3IncidenceIndex, "updated", fail)
    with pytest.raises(MemoryError, match="index allocation"):
        topology.insert_point((0.2, 0.15))
    np.testing.assert_array_equal(topology.points, before_points)
    np.testing.assert_array_equal(topology.triangles, before_rows)
    np.testing.assert_array_equal(topology.triangle_owners, before_owners)
    assert topology._topology_index is before_index
    assert topology.epoch == 0
    assert topology.quality_cache == {(0, 1, 2): (1.0,)}


@pytest.mark.parametrize("error", [MeshError("invalid index"), MemoryError("index allocation")])
def test_flip_index_failure_restores_connectivity_and_propagates_operational_error(monkeypatch, error):
    topology = make_topology(monkeypatch)
    before = topology.triangles
    before_index = topology._topology_index

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(T3IncidenceIndex, "updated", fail)
    if isinstance(error, MeshError):
        assert not topology.flip_edge((0, 2))
    else:
        with pytest.raises(MemoryError, match="index allocation"):
            topology.flip_edge((0, 2))
    np.testing.assert_array_equal(topology.triangles, before)
    assert topology._topology_index is before_index
    assert topology.epoch == 0


def test_successful_flip_updates_index_and_preserves_boundary(monkeypatch):
    topology = make_topology(monkeypatch)
    assert topology.flip_edge((0, 2))
    assert_index(topology._topology_index, topology.triangles)
    assert (0, 2) not in topology._topology_index
    assert (1, 3) in topology._topology_index
    assert all(edge in topology._topology_index for edge in topology.protected_edges)


def make_shared_topology(node_id_allocator=None):
    registry = native_v2.ComponentSeedRegistry(100, node_id_allocator=node_id_allocator)
    topology = native_v2.MutableT3Topology(
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
        [(0, 1), (1, 2), (2, 3), (0, 3)],
        node_owners=[7, 8, 9, 10],
        triangle_owners=[11, 22],
        splittable_edges={(0, 2): (7, 0, 1)},
        seed_registry=registry,
    )
    return topology, registry


def test_shared_split_preserves_owners_intervals_and_index():
    topology, registry = make_shared_topology()
    before_index = topology._topology_index
    before_rows = topology.triangles
    report = topology.split_segment((0, 2), owner=42)
    assert report == {
        "epoch": 1, "point_id": 4, "shared_node_id": 100,
        "edge_id": 7, "station": (1, 2),
    }
    assert registry.assigned_node_ids == (100,)
    assert registry.resolve(7, 2, 4) == 100
    assert_index(topology._topology_index, topology.triangles)
    assert_index(before_index, before_rows)
    np.testing.assert_array_equal(topology.points[4], [0.5, 0.5])
    np.testing.assert_array_equal(topology.node_owners, [7, 8, 9, 10, 42])
    assert dict(zip(map(tuple, topology.triangles), topology.triangle_owners)) == {
        (0, 1, 4): 11, (1, 2, 4): 11, (0, 4, 3): 22, (2, 3, 4): 22,
    }
    assert topology.splittable_edges == ((0, 4), (2, 4))
    assert topology._splittable_intervals[(0, 4)] == (7, 0, report["station"][0] / report["station"][1])
    # Parameter endpoints follow the canonical node pair, not sorted parameter order.
    assert topology._splittable_intervals[(2, 4)] == (7, 1, report["station"][0] / report["station"][1])
    assert all(edge in topology._topology_index for edge in topology.protected_edges)


def assert_shared_local_state(topology, before):
    points, rows, node_owners, triangle_owners, index, intervals, shared, epoch, cache = before
    np.testing.assert_array_equal(topology.points, points)
    np.testing.assert_array_equal(topology.triangles, rows)
    np.testing.assert_array_equal(topology.node_owners, node_owners)
    np.testing.assert_array_equal(topology.triangle_owners, triangle_owners)
    assert topology._topology_index is index
    assert topology._splittable_intervals == intervals
    assert topology.shared_node_ids == shared
    assert topology.epoch == epoch
    assert topology.quality_cache == cache
    assert_index(index, rows)


def shared_state(topology):
    return (
        topology.points, topology.triangles, topology.node_owners.copy(),
        topology.triangle_owners.copy(), topology._topology_index,
        dict(topology._splittable_intervals), topology.shared_node_ids,
        topology.epoch, dict(topology.quality_cache),
    )


def test_shared_split_index_failure_restores_local_state(monkeypatch):
    topology, _ = make_shared_topology()
    topology.quality_cache[(0, 1, 2)] = (1.0,)
    before = shared_state(topology)

    def fail(*args, **kwargs):
        raise MemoryError("shared index allocation")

    monkeypatch.setattr(T3IncidenceIndex, "updated", fail)
    with pytest.raises(MemoryError, match="shared index allocation"):
        topology.split_segment((0, 2), owner=42)
    assert_shared_local_state(topology, before)


@pytest.mark.parametrize("cancel_phase", [
    "native-v2 shared segment reconstruction",
    "native-v2 incidence update",
    "native-v2 shared segment split commit",
])
def test_cancelled_shared_split_does_not_consume_component_node_identity(cancel_phase):
    allocations = []

    def allocate():
        allocations.append(100)
        return 100

    topology, registry = make_shared_topology(allocate)
    before = shared_state(topology)
    assigned_before = registry.assigned_node_ids
    next_before = registry._next

    def cancel(phase):
        assert_shared_local_state(topology, before)
        if phase == cancel_phase:
            raise RuntimeError("cancel detached shared index")

    with pytest.raises(RuntimeError, match="cancel detached shared index"):
        topology.split_segment((0, 2), cancellation_check=cancel)
    assert_shared_local_state(topology, before)
    assert registry.assigned_node_ids == assigned_before
    assert registry._next == next_before
    assert allocations == []


@pytest.mark.parametrize("stop_at", [2, 4, 6])
def test_late_update_cancellation_preserves_memberships(stop_at):
    # Small index-only fixture: crosses the fixed 4096-cell checkpoints without
    # triangulation, meshing, or a performance workload.
    rows = [(3 * n, 3 * n + 1, 3 * n + 2) for n in range(4100)]
    replacement = [(20000 + 3 * n, 20001 + 3 * n, 20002 + 3 * n) for n in range(4100)]
    index = T3IncidenceIndex(rows)
    memberships = index._edge_cells.copy()
    callbacks = 0

    def cancel(phase):
        nonlocal callbacks
        if phase == "native-v2 incidence update":
            callbacks += 1
            if callbacks == stop_at:
                raise RuntimeError("cancel partial update")

    with pytest.raises(RuntimeError, match="cancel partial update"):
        index.updated(replacement, cancellation_check=cancel)
    assert callbacks == stop_at
    assert_index(index, rows)
    assert all(index._edge_cells[edge] is cells for edge, cells in memberships.items())


def test_native_insertion_sequence_matches_reference_with_nondefault_owners(monkeypatch):
    from anymesher.native_cpp import COMPILED_NATIVE_V2_AVAILABLE

    if not COMPILED_NATIVE_V2_AVAILABLE:
        pytest.skip("optional native-v2 extension is absent")
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    protected = [(0, 1), (1, 2), (2, 3), (0, 3)]
    native = native_v2.MutableT3Topology(
        points, triangles, protected, node_owners=[7, 8, 9, 10],
        triangle_owners=[11, 22],
    )
    oracle = native_v2.MutableT3Topology(
        points, triangles, protected, node_owners=[7, 8, 9, 10],
        triangle_owners=[11, 22],
    )
    for point in [(0.2, 0.15), (0.8, 0.2), (0.6, 0.75)]:
        native_report = native.insert_point(point, owner=42)
        assert native_report["native"] is True
        with monkeypatch.context() as patch:
            patch.setattr(native_v2, "native_mutable_t3_insert", lambda *args: None)
            oracle_report = oracle.insert_point(point, owner=42)
        assert {k: v for k, v in native_report.items() if k != "native"} == {
            k: v for k, v in oracle_report.items() if k != "native"
        }
        np.testing.assert_array_equal(native.points, oracle.points)
        np.testing.assert_array_equal(native.triangles, oracle.triangles)
        np.testing.assert_array_equal(native.node_owners, oracle.node_owners)
        np.testing.assert_array_equal(native.triangle_owners, oracle.triangle_owners)
        assert_index(native._topology_index, native.triangles)


@pytest.mark.parametrize("existing", [False, True])
def test_registry_publication_failure_preserves_unrelated_and_existing_stations(existing):
    registry = native_v2.ComponentSeedRegistry(100)
    assert registry.resolve(9, 1, 2) == 100
    if existing:
        assert registry.resolve(7, 1, 2) == 101
    assigned = registry.assigned_node_ids
    next_id = registry._next

    def fail(node_id):
        assert node_id == 101
        raise MemoryError("seed publication")

    with pytest.raises(MemoryError, match="seed publication"):
        registry._resolve_and_publish(7, 2, 4, fail)
    assert registry.assigned_node_ids == assigned
    assert registry._next == next_id
    assert registry.resolve(9, 2, 4) == 100
    assert registry.resolve(7, 1, 2) == 101


def test_registry_reentrant_allocator_fails_without_registration():
    registry = native_v2.ComponentSeedRegistry(
        100, node_id_allocator=lambda: registry.resolve(9, 1, 2)
    )
    with pytest.raises(MeshError, match="already active"):
        registry.resolve(7, 1, 2)
    assert registry.assigned_node_ids == ()
    assert registry._next == 100


def test_failed_external_allocator_leaves_split_state_unchanged():
    calls = []

    def allocate():
        calls.append("attempted")
        raise RuntimeError("allocator failure")

    topology, registry = make_shared_topology(allocate)
    before = shared_state(topology)
    with pytest.raises(RuntimeError, match="allocator failure"):
        topology.split_segment((0, 2), owner=42)
    assert_shared_local_state(topology, before)
    assert registry.assigned_node_ids == ()
    assert registry._next == 100
    # The callback's own side effect is truthful, not claimed to be rolled back.
    assert calls == ["attempted"]


def test_shared_split_publication_failure_restores_registry_and_local_state(monkeypatch):
    topology, registry = make_shared_topology()
    before = shared_state(topology)
    original = registry._resolve_and_publish

    def fail_after_local_publish(edge_id, numerator, denominator, publish):
        def fail(node_id):
            publish(node_id)
            raise MemoryError("after local publication")
        return original(edge_id, numerator, denominator, fail)

    monkeypatch.setattr(registry, "_resolve_and_publish", fail_after_local_publish)
    with pytest.raises(MemoryError, match="after local publication"):
        topology.split_segment((0, 2), owner=42)
    assert_shared_local_state(topology, before)
    assert registry.assigned_node_ids == ()
    assert registry._next == 100
