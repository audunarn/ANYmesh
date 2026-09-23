from __future__ import annotations

import pytest

from anymesher.errors import MeshError
from anymesher.quad.front import FrontNoCandidate, find_source_cell, front_step
from anymesher.quad.journal import TransactionStateError
from anymesher.quad.state import CancellationRequested, QuadMeshState


def _square_pair(*, protected_nodes=(), protected_edges=(), is_cancelled=None):
    return QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
        cells={0: (0, 1, 2), 1: (1, 3, 2)},
        cell_kinds={0: "T3", 1: "T3"},
        initial_front=((0, 1), (0, 2), (1, 3), (2, 3)),
        protected_nodes=protected_nodes,
        protected_edges=protected_edges,
        is_cancelled=is_cancelled,
    )


def _two_square_strip():
    return QuadMeshState(
        nodes={
            0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0),
            3: (1.0, 1.0), 4: (2.0, 0.0), 5: (2.0, 1.0),
        },
        cells={0: (0, 1, 2), 1: (1, 3, 2), 2: (1, 4, 3), 3: (4, 5, 3)},
        cell_kinds={0: "T3", 1: "T3", 2: "T3", 3: "T3"},
        initial_front=((0, 1), (0, 2), (2, 3), (3, 5), (4, 5), (1, 4)),
    )


def _allocator_state(*, is_cancelled=None):
    return QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)},
        cells={0: (0, 1, 2)},
        cell_kinds={0: "T3"},
        initial_front=(),
        is_cancelled=is_cancelled,
    )


def test_q4_t3_interface_is_active_and_source_is_residual_t3():
    state = QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0), 4: (2.0, 0.0)},
        cells={0: (0, 1, 2, 3), 1: (1, 4, 2)},
        cell_kinds={0: "Q4", 1: "T3"},
        initial_front=((1, 2),),
    )
    assert state.is_front_edge((1, 2))
    assert find_source_cell(state, (1, 2)) == 1


def test_sequential_two_q4_growth_keeps_q4_t3_interface_active():
    state = _two_square_strip()
    first_id, _ = front_step(state, (0, 1))
    assert state.cell_kind(first_id) == "Q4"
    assert state.is_front_edge((1, 3))
    assert find_source_cell(state, (1, 3)) == 2
    second_id, _ = front_step(state, (1, 3))
    assert state.cell_kind(second_id) == "Q4"
    assert all(state.cell_kind(cid) == "Q4" for cid in state.cells)
    assert state.front == frozenset()


def test_protected_boundary_nodes_and_edge_participate_and_survive():
    state = _square_pair(protected_nodes=(0, 1), protected_edges=((0, 1),))
    new_id, body = front_step(state, (0, 1))
    assert state.cell_kind(new_id) == "Q4"
    assert {0, 1} <= set(body)
    assert state.is_protected_node(0)
    assert state.is_protected_node(1)
    assert state.is_protected_edge((0, 1))
    assert not state.is_front_edge((0, 1))


def test_protected_interior_diagonal_rejects_without_mutation():
    state = _square_pair(protected_edges=((1, 2),))
    before = state.digest()
    generation = state.generation
    with pytest.raises(FrontNoCandidate):
        front_step(state, (0, 1))
    assert state.digest() == before
    assert state.generation == generation


def test_transactional_move_node_commit_reject_and_rollback():
    state = _square_pair()
    generation = state.generation
    tx = state.transaction()
    tx.move_node(3, (1.2, 0.9))
    assert tx.view.position(3) == (1.2, 0.9)
    tx.commit()
    assert state.position(3) == (1.2, 0.9)
    assert state.generation == generation + 1

    protected = _square_pair(protected_nodes=(0,))
    before = protected.digest()
    tx = protected.transaction()
    with pytest.raises(MeshError):
        tx.move_node(0, (0.1, 0.1))
    assert protected.digest() == before

    tx = protected.transaction()
    with pytest.raises(MeshError):
        tx.move_node(1, (float("nan"), 0.0))
    assert protected.digest() == before

    rolled = _square_pair()
    before_pos = rolled.position(2)
    before_gen = rolled.generation
    tx = rolled.transaction()
    tx.move_node(2, (0.0, 2.0))
    tx.rollback()
    assert rolled.position(2) == before_pos
    assert rolled.generation == before_gen


def test_transaction_owned_allocators_commit_and_rollback_without_id_consumption():
    state = _allocator_state()
    node0 = state.next_node_id
    cell0 = state.next_cell_id

    tx = state.transaction()
    rolled_node = tx.allocate_node((1.0, 1.0))
    rolled_cell = tx.allocate_cell((1, rolled_node, 2), "T3")
    assert (rolled_node, rolled_cell) == (node0, cell0)
    tx.rollback()
    assert state.next_node_id == node0
    assert state.next_cell_id == cell0

    tx = state.transaction()
    node = tx.allocate_node((1.0, 1.0))
    cell = tx.allocate_cell((1, node, 2), "T3")
    assert (node, cell) == (node0, cell0)
    tx.commit()
    assert state.position(node) == (1.0, 1.0)
    assert state.cell(cell) == (1, node, 2)
    assert state.next_node_id == node0 + 1
    assert state.next_cell_id == cell0 + 1


def test_stale_transaction_cannot_publish_after_other_commit():
    state = _square_pair()
    tx_a = state.transaction()
    tx_b = state.transaction()
    tx_b.move_node(3, (1.1, 1.0))
    tx_b.commit()
    assert state.generation == 1
    tx_a.move_node(2, (-0.1, 1.0))
    with pytest.raises(TransactionStateError):
        tx_a.commit()
    assert state.generation == 1
    assert state.position(3) == (1.1, 1.0)
    assert state.position(2) == (0.0, 1.0)


def test_cancellation_preserves_digest_generation_and_allocator_ids():
    cancelled = [False]
    state = _allocator_state(is_cancelled=lambda: cancelled[0])
    before = state.digest()
    generation = state.generation
    node0 = state.next_node_id
    cell0 = state.next_cell_id
    tx = state.transaction()
    node = tx.allocate_node((1.0, 1.0))
    tx.allocate_cell((1, node, 2), "T3")
    cancelled[0] = True
    with pytest.raises(CancellationRequested):
        tx.commit()
    assert state.digest() == before
    assert state.generation == generation
    assert state.next_node_id == node0
    assert state.next_cell_id == cell0


def test_move_only_commit_does_not_bulk_iterate_front_or_protected_sets():
    class NoBulkSet(set):
        def __iter__(self):
            raise AssertionError("bulk set iteration is forbidden")
        def __sub__(self, other):
            raise AssertionError("bulk set-subtraction is forbidden")
        def __or__(self, other):
            raise AssertionError("bulk set-union is forbidden")
        def copy(self):
            raise AssertionError("bulk set copy is forbidden")

    state = _square_pair()
    state._front = NoBulkSet(state._front)
    state._front_bits = NoBulkSet(state._front_bits)
    state._prot_nodes = NoBulkSet(state._prot_nodes)
    state._prot_edges = NoBulkSet(state._prot_edges)
    generation = state.generation
    tx = state.transaction()
    tx.move_node(3, (1.05, 0.95))
    tx.commit()
    assert state.position(3) == (1.05, 0.95)
    assert state.generation == generation + 1


def test_coordinate_only_move_commit_does_not_iterate_whole_node_map():
    class NoIterDict(dict):
        def __iter__(self):
            raise AssertionError("whole node-map iteration is forbidden")
        def keys(self):
            raise AssertionError("whole node-map keys() is forbidden")
        def items(self):
            raise AssertionError("whole node-map items() is forbidden")
        def values(self):
            raise AssertionError("whole node-map values() is forbidden")

    state = _square_pair()
    state._pos = NoIterDict(state._pos)
    generation = state.generation
    tx = state.transaction()
    tx.move_node(3, (1.05, 0.95))
    tx.commit()
    assert state.position(3) == (1.05, 0.95)
    assert state.generation == generation + 1
