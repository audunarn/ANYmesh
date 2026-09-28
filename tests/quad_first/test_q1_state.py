"""Q1 state/journal invariant tests for the quad-first advancing front.

Covers the resident ``QuadMeshState`` + sparse ``Delta``/``View``/``Transaction``
contract: local node-edge adjacency, View-after-staging freshness, discard-only
rollback with exact digest identity, generation guards, cancellation gates,
protected-feature residency, duplicate-body rejection, and the local-patch
commit mechanism (no whole-mesh structure rebuild per commit).
"""

from __future__ import annotations

import pytest

from anymesher.errors import MeshError
from anymesher.quad.journal import Transaction, TransactionStateError
from anymesher.quad.state import (
    CancellationRequested,
    QuadMeshState,
    StaleHandleError,
)


def _triangle_state(cancelled: bool = False) -> QuadMeshState:
    return QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)},
        cells={0: (0, 1, 2)},
        initial_front=[(0, 1)],
        is_cancelled=lambda: cancelled,
    )


def _cancellable_state(flag: dict[str, bool]) -> QuadMeshState:
    """State whose cancellation flag is read *live* from ``flag['cancelled']``."""
    return QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0)},
        cells={0: (0, 1, 2)},
        initial_front=[(0, 1)],
        is_cancelled=lambda: flag["cancelled"],
    )


# --- node-edge adjacency ----------------------------------------------------

def test_node_edges_only_contain_incident_edges() -> None:
    st = _triangle_state()
    assert st.neighbors_at(0) == ((0, 1), (0, 2))
    assert st.neighbors_at(1) == ((0, 1), (1, 2))
    assert st.neighbors_at(2) == ((0, 2), (1, 2))


def test_node_edges_exact_after_add_and_remove_cell() -> None:
    st = _triangle_state()
    with st.transaction() as tx:
        tx.add_node(3, (1.0, 1.0))
        tx.add_cell(7, (0, 1, 2, 3), "Q4")
        tx.commit()
    # node 1 touches (0,1) and (1,2); node 3 touches (0,3) and (2,3) only.
    assert st.neighbors_at(1) == ((0, 1), (1, 2))
    assert st.neighbors_at(3) == ((0, 3), (2, 3))
    with st.transaction() as tx:
        tx.remove_cell(7)
        tx.commit()
    assert st.neighbors_at(3) == tuple()
    assert st.neighbors_at(2) == ((0, 2), (1, 2))


# --- View-after-staging freshness -------------------------------------------

def test_view_sees_staged_removal_before_commit() -> None:
    st = _triangle_state()
    tx = st.transaction()
    view = tx.view
    tx.remove_cell(0)
    # Edge (0,1) lost its only incident cell in the candidate.
    assert view.edge_cells((0, 1)) == ()
    assert tx.view.edge_cells((0, 1)) == ()
    # Base is untouched until commit.
    assert st.edge_cells((0, 1)) == (0,)
    tx.rollback()
    assert st.edge_cells((0, 1)) == (0,)


def test_view_sees_staged_addition_before_commit() -> None:
    st = _triangle_state()
    tx = st.transaction()
    tx.add_node(3, (1.0, 1.0))
    tx.add_node(4, (0.5, 1.0))
    tx.add_cell(9, (0, 2, 4, 3), "Q4")
    view = tx.view
    assert 9 in view.edge_cells((0, 2))
    assert view.position(4) == (0.5, 1.0)
    assert view.cell_kind(9) == "Q4"
    tx.rollback()
    assert st.edge_cells((0, 2)) == (0,)


def test_view_freshness_after_late_staging() -> None:
    st = _triangle_state()
    tx = st.transaction()
    early_view = tx.view
    tx.add_node(3, (1.0, 1.0))
    tx.add_cell(7, (0, 1, 2, 3), "Q4")
    fresh = tx.view
    # Views are live candidates: both reflect the CURRENT contents of the
    # shared delta, so the late-staged cell is visible through either handle
    # (commit prevalidates exactly what views show).
    assert 7 in fresh.cells
    assert 7 in early_view.cells
    assert 0 in fresh.edge_cells((0, 1))
    assert 7 in fresh.edge_cells((0, 1))
    tx.rollback()


# --- rollback / digest identity ---------------------------------------------

def test_rollback_leaves_base_digest_untouched() -> None:
    st = _triangle_state()
    before = st.digest()
    with st.transaction() as tx:
        tx.add_node(3, (1.0, 1.0))
        tx.add_cell(7, (0, 1, 2, 3), "Q4")
        tx.rollback()
    assert st.digest() == before
    assert 7 not in st.cells
    assert st.generation == 0


def test_failed_commit_preserves_digest_and_generation() -> None:
    st = _triangle_state()
    before = (st.digest(), st.generation)
    tx = st.transaction()
    tx.add_cell(7, (0, 1, 2, 3), "Q4")  # node 3 unknown
    with pytest.raises(MeshError):
        tx.commit()
    assert (st.digest(), st.generation) == before


# --- generation guard -------------------------------------------------------

def test_stale_generation_transaction_cannot_commit() -> None:
    st = _triangle_state()
    stale_tx = st.transaction()
    stale_tx.add_node(3, (1.0, 1.0))
    stale_tx.add_cell(7, (0, 1, 2, 3), "Q4")
    with st.transaction() as bump:
        bump.remove_cell(0)
        bump.add_node(3, (1.0, 1.0))
        bump.add_cell(5, (0, 1, 3), "T3")
        bump.commit()
    assert st.generation == 1
    with pytest.raises(TransactionStateError):
        stale_tx.commit()


def test_use_after_commit_or_rollback_raises() -> None:
    st = _triangle_state()
    tx = st.transaction()
    with pytest.raises(TransactionStateError):
        tx.rollback()
        tx.commit()
    st2 = _triangle_state()
    tx2 = st2.transaction()
    tx2.rollback()
    with pytest.raises(TransactionStateError):
        tx2.add_cell(1, (0, 1, 2))
    st3 = _triangle_state()
    tx3 = st3.transaction()
    tx3.add_node(3, (1.0, 1.0))
    tx3.add_cell(7, (0, 1, 2, 3), "Q4")
    tx3.commit()
    with pytest.raises(TransactionStateError):
        tx3.add_node(4, (0.5, 0.5))


# --- cancellation -----------------------------------------------------------

def test_cancellation_checkpoint_blocks_commit() -> None:
    flag = {"cancelled": False}
    st = _cancellable_state(flag)
    before = st.digest()
    tx = st.transaction()
    tx.add_node(3, (1.0, 1.0))
    tx.checkpoint()  # ok while not cancelled
    flag["cancelled"] = True
    with pytest.raises(CancellationRequested):
        tx.commit()
    assert st.digest() == before
    assert st.generation == 0


def test_cancellation_requested_at_staged_checkpoint() -> None:
    flag = {"cancelled": False}
    st = _cancellable_state(flag)
    tx = st.transaction()
    flag["cancelled"] = True
    with pytest.raises(CancellationRequested):
        tx.checkpoint()
    assert st.digest() is not None


# --- protected features -----------------------------------------------------

def test_protect_edge_not_resident_rejected() -> None:
    st = _triangle_state()
    before = st.digest()
    tx = st.transaction()
    tx.protect_edge(0, 9)  # edge (0,9) does not exist in the mesh
    with pytest.raises(MeshError):
        tx.commit()
    assert st.digest() == before
    assert st.protected_edges == frozenset()


def test_protect_edge_resident_ok() -> None:
    st = _triangle_state()
    tx = st.transaction()
    tx.protect_edge(0, 1)
    tx.protect_node(2)
    tx.commit()
    assert st.protected_edges == frozenset({(0, 1)})
    assert st.protected_nodes == frozenset({2})


def test_release_protected_edge_after_protect() -> None:
    st = _triangle_state()
    tx = st.transaction()
    tx.protect_edge(0, 1)
    tx.release_edge(0, 1)
    tx.commit()
    assert st.protected_edges == frozenset()


# --- duplicate body / stale cell ---------------------------------------------

def test_duplicate_body_rejected() -> None:
    st = _triangle_state()
    before = st.digest()
    tx = st.transaction()
    tx.add_node(3, (1.0, 1.0))
    tx.add_cell(7, (2, 1, 0), "T3")  # same sorted body as cell 0
    with pytest.raises(MeshError, match="duplicate cell body"):
        tx.commit()
    assert st.digest() == before


def test_added_body_may_replace_removed_body() -> None:
    st = _triangle_state()
    with st.transaction() as tx:
        # Removing cell 0 frees body (0,1,2); a re-add of the same body is
        # legal local topology replacement.
        tx.remove_cell(0)
        tx.add_node(3, (1.0, 1.0))
        tx.add_cell(5, (0, 1, 3), "T3")
        tx.commit()
    assert st.cells == {5: (0, 1, 3)}


def test_stale_cell_removal_rejected() -> None:
    st = _triangle_state()
    before = st.digest()
    tx = st.transaction()
    tx.remove_cell(123)
    with pytest.raises(StaleHandleError):
        tx.commit()
    assert st.digest() == before


def test_stale_cell_handle_lookup() -> None:
    st = _triangle_state()
    with pytest.raises(StaleHandleError):
        st.cell(99)
    with pytest.raises(StaleHandleError):
        st.position(99)


# --- commit locality (white-box) --------------------------------------------

def test_commit_is_local_patch_not_full_rebuild() -> None:
    st = _triangle_state()
    untouched_node_adj = st._node_edges[2]  # node 2 is not touched by the edit
    st._cells_ids_before = dict(st.cells)
    before_cells_obj = st._cells
    before_edges_obj = st._edge_to_cells
    with st.transaction() as tx:
        tx.remove_front_edge(0, 1)
        tx.add_node(3, (1.0, 1.0))
        tx.add_cell(7, (0, 1, 2, 3), "Q4")
        tx.commit()
    # In-place patch: container identity retained, entries updated locally,
    # and untouched node adjacency objects are not rebuilt.
    assert st._cells is before_cells_obj
    assert st._edge_to_cells is before_edges_obj
    assert st._node_edges[2] is untouched_node_adj
    assert st.generation == 1
    del st._cells_ids_before


def test_empty_commit_is_noop_but_concludes() -> None:
    st = _triangle_state()
    tx = st.transaction()
    result = tx.commit()
    assert result is st
    assert tx.committed
    assert st.generation == 0  # no delta applied -> no bump? (allowed no-op)


def test_edge_key_normalization_and_errors() -> None:
    st = _triangle_state()
    assert st.edge_cells((1, 0)) == st.edge_cells((0, 1)) == (0,)
    with pytest.raises(MeshError):
        st.edge_cells((0, 0))


def test_duplicate_node_in_cell_rejected_at_construction() -> None:
    with pytest.raises(MeshError, match="repeats a node"):
        QuadMeshState(nodes={0: (0.0, 0.0), 1: (1.0, 0.0)}, cells={0: (0, 1, 1)})


def test_kind_len_mismatch_rejected_at_construction() -> None:
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    # A 2-node body with T3 kind must be rejected.
    with pytest.raises(MeshError):
        QuadMeshState(nodes=nodes, cells={0: (0, 1)}, cell_kinds={0: "T3"})


def test_count_kind_tracks_commits_without_scanning():
    from anymesher.quad.state import QuadMeshState

    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
    state = QuadMeshState(nodes, {0: (0, 1, 3), 1: (1, 2, 3)}, {0: "T3", 1: "T3"},
                          initial_front=((0, 1), (1, 2), (2, 3), (0, 3)))
    assert (state.count_kind("T3"), state.count_kind("Q4")) == (2, 0)
    with state.transaction() as tx:
        tx.remove_cell(0)
        tx.remove_cell(1)
        tx.allocate_cell((0, 1, 2, 3), "Q4")
        tx.commit()
    assert (state.count_kind("T3"), state.count_kind("Q4")) == (0, 1)
    with state.transaction() as tx:  # discarded without commit
        tx.remove_cell(2)
    assert (state.count_kind("T3"), state.count_kind("Q4")) == (0, 1)
    with pytest.raises(MeshError):
        state.count_kind("Q8")
