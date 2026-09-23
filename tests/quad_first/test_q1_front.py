"""Q1 front-driver vertical-slice tests.

Validates the single-step front operation: a unique ``T3`` source below a front
edge plus one ``T3`` partner across a non-front (interior diagonal) edge are
consumed into one ``Q4`` atomically.  The front-edge set of a well-formed pair
is invariant under the operation (all exposed outer edges are re-backed by the
new Q4), which the driver's local bookkeeping must preserve.  Typed failures
(:class:`FrontRejected` for a specific rule, :class:`FrontNoCandidate` when
nothing admissible remains) are raised rather than silently no-op'ing.
"""

from __future__ import annotations

import pytest

from anymesher.errors import MeshError
from anymesher.quad.front import (
    FrontNoCandidate,
    FrontRejected,
    area2,
    body_edges,
    candidate_partners,
    classify,
    edge_key,
    find_source_cell,
    front_step,
    local_swap,
    make_quad,
)
from anymesher.quad.state import QuadMeshState


def _state_A():
    """Two ``T3`` split by the interior diagonal (1,2) of the unit square.

    Vertices: 0=BL(0,0), 1=BR(1,0), 2=TL(0,1), 3=TR(1,1).
    Cells: 0=(0,1,2), 1=(1,2,3).  All four outer edges are front; the
    diagonal (1,2) is interior.
    """
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    cells = {0: (0, 1, 2), 1: (1, 2, 3)}
    return QuadMeshState(
        nodes=nodes,
        cells=cells,
        initial_front=[(0, 1), (0, 2), (1, 3), (2, 3)],
    )


def _state_C():
    """Front edge (0,1) with a resident Q4 (not T3) below it.

    Quad (0,1,2,3); edge (0,1) is front, its sole backing cell is cell 0 (Q4).
    """
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 1.0), 3: (0.0, 1.0)}
    return QuadMeshState(
        nodes=nodes,
        cells={0: (0, 1, 2, 3)},
        initial_front=[(0, 1), (0, 3), (1, 2), (2, 3)],
    )


def _state_D():
    """Two ``T3`` sharing interior diagonal (0,2) whose union is a dart.

    Vertices: 0=(0,0), 1=(2,0), 2=(1,0.3) (concave), 3=(1,1).
    Union boundary (0,1),(1,2),(2,3),(3,0) is concave at node 2.
    """
    nodes = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 0.3), 3: (1.0, 1.0)}
    cells = {0: (0, 1, 2), 1: (0, 2, 3)}
    return QuadMeshState(
        nodes=nodes,
        cells=cells,
        initial_front=[(0, 1), (1, 2), (2, 3), (0, 3)],
    )


# --- pure geometry ------------------------------------------------------------

def test_edge_key_normalizes_and_rejects_equal() -> None:
    assert edge_key(3, 1) == (1, 3)
    assert edge_key(1, 3) == (1, 3)
    with pytest.raises(MeshError):
        edge_key(2, 2)


def test_body_edges_cyclic_keys() -> None:
    assert body_edges((0, 1, 2)) == [(0, 1), (1, 2), (0, 2)]
    assert body_edges((0, 1, 3, 2)) == [(0, 1), (1, 3), (2, 3), (0, 2)]


def test_area2_unit_square() -> None:
    st = _state_A()
    # Unit square shoelace area is 1, so signed double area is 2.
    assert area2(st, (0, 1, 3, 2)) == pytest.approx(2.0)
    assert area2(st, (0, 2, 3, 1)) == pytest.approx(-2.0)


def test_make_quad_convex_ccw() -> None:
    st = _state_A()
    # Already CCW convex: returned unchanged.
    assert make_quad(st, (0, 1, 3, 2)) == (0, 1, 3, 2)
    # Clockwise input is normalized to CCW.
    assert make_quad(st, (0, 2, 3, 1)) == (0, 1, 3, 2)


def test_make_quad_rejects_non_convex() -> None:
    st = _state_D()
    # Union boundary (0, 2, 1, 3) has a reflex turn at node 2.
    with pytest.raises(FrontRejected, match="strictly convex"):
        make_quad(st, (0, 2, 1, 3))


def test_make_quad_rejects_degenerate() -> None:
    # All four nodes collinear -> zero signed area -> "degenerate".
    nodes = {0: (0.0, 0.0), 1: (0.5, 0.5), 2: (1.0, 1.0), 3: (1.5, 1.5)}
    tmp = QuadMeshState(nodes=nodes, cells={0: (0, 1, 3)})
    with pytest.raises(FrontRejected, match="degenerate"):
        make_quad(tmp, (0, 1, 2, 3))


def test_make_quad_rejects_repeated_node() -> None:
    st = _state_A()
    with pytest.raises(FrontRejected):
        make_quad(st, (0, 1, 1, 2))


def test_local_swap_tiles_other_diagonal() -> None:
    assert local_swap((0, 1, 3, 2)) == ((0, 1, 2), (1, 3, 2))
    with pytest.raises(FrontRejected):
        local_swap((0, 1, 2))


# --- candidate discovery ------------------------------------------------------

def test_find_source_cell_unique_t3() -> None:
    st = _state_A()
    # Edge (2,3) is backed only by cell 1.
    assert find_source_cell(st, (2, 3)) == 1


def test_find_source_cell_rejects_multiple_incident() -> None:
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    # Two triangles sharing edge (0,1) -> (0,1) has two incident cells.
    st = QuadMeshState(
        nodes=nodes,
        cells={0: (0, 1, 2), 1: (0, 1, 3)},
        initial_front=[(0, 1)],
    )
    with pytest.raises(FrontRejected, match="incident"):
        find_source_cell(st, (0, 1))


def test_candidate_partners_deterministic() -> None:
    st = _state_A()
    assert candidate_partners(st, 0) == [1]


def test_candidate_partners_requires_t3_source() -> None:
    st = _state_C()
    with pytest.raises(FrontRejected, match="not a T3"):
        candidate_partners(st, 0)


def test_classify_rejects_front_equal_to_shared() -> None:
    # Two triangles sharing edge (0,1) with (0,1) itself as the candidate's
    # declared front edge must be rejected (the pair would not be a quad).
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    st = QuadMeshState(nodes=nodes, cells={0: (0, 1, 2), 1: (0, 1, 3)})
    with pytest.raises(FrontRejected, match="front edge is the shared"):
        classify(st, (0, 1), 0, 1)


# --- front_step vertical slice -------------------------------------------------

def test_front_step_A_stages_quad_and_updates_front() -> None:
    st = _state_A()
    before = st.digest()
    new_id, body = front_step(st, (1, 3))
    # Quad body is a CCW rotation of the unit square (1,3,2,0) or (0,1,3,2).
    assert body in ((0, 1, 3, 2), (1, 3, 2, 0), (3, 2, 0, 1), (2, 0, 1, 3))
    assert st.cell(new_id) == body
    assert st.cell_kind(new_id) == "Q4"
    # Source/partner cells consumed.
    assert 0 not in st.cells
    assert 1 not in st.cells
    # PQ1 semantics: the active front is the boundary of the residual T3
    # region; after both T3s become one Q4 there is no residual front.
    assert st.front == frozenset()
    assert st.generation == 1
    assert st.digest() != before


def test_front_step_B_rejects_when_no_partner_t3() -> None:
    # Source T3 (0,1,2) has its only non-front edge (0,1) backed by a resident
    # Q4 (not T3), so no admissible T3 partner exists.
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0), 4: (2.0, 0.0)}
    st = QuadMeshState(
        nodes=nodes,
        cells={0: (0, 1, 2), 1: (0, 1, 3, 4)},
        cell_kinds={0: "T3", 1: "Q4"},
        initial_front=[(1, 2), (0, 2), (1, 3), (3, 4), (0, 4)],
    )
    # Edge (1,2) is the source's front edge; its unique incident cell is 0 (T3).
    # source edges: (0,1) [non-front, partner Q4 -> rejected], (1,2) [front], (0,2) [front].
    # No T3 partner across a non-front edge -> FrontNoCandidate.
    before = st.digest()
    with pytest.raises(FrontNoCandidate):
        front_step(st, (1, 2))
    assert st.digest() == before


def test_front_step_C_rejects_non_t3_source() -> None:
    st = _state_C()
    with pytest.raises(FrontNoCandidate):
        front_step(st, (0, 1))


def test_front_step_D_rejects_non_convex_union() -> None:
    st = _state_D()
    # Edge (2,3): unique T3 source is cell 1; partner is cell 0 across the
    # shared interior edge (0,2). The union is a concave dart -> reject.
    with pytest.raises(FrontNoCandidate, match="candidate"):
        front_step(st, (2, 3))


def test_front_step_blocked_by_protected_edge() -> None:
    nodes = {0: (0.0, 0.0), 1: (2.0, 0.0), 2: (1.0, 0.3), 3: (1.0, 1.0)}
    st = QuadMeshState(
        nodes=nodes,
        cells={0: (0, 1, 2), 1: (0, 2, 3)},
        initial_front=[(0, 1), (1, 2), (2, 3), (0, 3)],
        protected_edges=[(0, 2)],  # the shared interior diagonal is protected.
    )
    before = st.digest()
    with pytest.raises(FrontNoCandidate):
        front_step(st, (2, 3))
    assert st.digest() == before


def test_front_step_allows_protected_endpoint_participation() -> None:
    nodes = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
    st = QuadMeshState(
        nodes=nodes,
        cells={0: (0, 1, 2), 1: (1, 3, 2)},
        initial_front=[(0, 1), (0, 2), (1, 3), (2, 3)],
        protected_nodes=[0],
    )
    new_id, body = front_step(st, (0, 1))
    assert 0 in body
    assert st.cell_kind(new_id) == "Q4"
    assert st.is_protected_node(0)
    assert st.front == frozenset()


def test_front_step_rejects_nonfront_edge() -> None:
    st = _state_A()
    # Diagonal (0,3) is a resident edge with two incident cells -> not front.
    with pytest.raises(FrontNoCandidate, match="not on the active front"):
        front_step(st, (0, 3))


def test_front_step_rejects_bad_options() -> None:
    st = _state_A()
    with pytest.raises(MeshError):
        front_step(st, (1, 3), options="not-valid")


def test_front_step_reverses_on_rejection() -> None:
    st = _state_D()
    before = st.digest()
    with pytest.raises(FrontNoCandidate):
        front_step(st, (2, 3))
    assert st.digest() == before
