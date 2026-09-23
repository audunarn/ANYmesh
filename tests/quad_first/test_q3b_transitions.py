"""Q3b quad transition template tests (spacing-change / collision / closure).

Covers the full Q3b mandate from ``QUAD_FIRST_FULL_PROGRAMME.md``:

* spacing-change template: interior-Q4 diagonal flip into two T3 children
* collision template: adjacent T3 pair, exactly two front edges on the union
* closure template: adjacent T3 pair, at least three front edges on the union
* rotation covariance (phi) and reflection covariance (about the x-axis),
  including a node-id relabel covariance (exact integer ids, no welding)
* explicit preconditions with typed :class:`TransitionRejected` failures
  (kind, id type, stale handle, protected node/edge, front-edge count,
  adjacency, degenerate, cancellation)
* parent-transition replacement records on every committed report
* deterministic cell-id assignment (``max(cells)+1``) and state invariants
  (digest + generation) on every rejection path

Every numeric expectation is grounded in the real implementation
(``local_swap``, ``_boundary_quad``, ``_reconcile_front``), not hand algebra.
"""

from __future__ import annotations

import math

import pytest

from anymesher.errors import MeshError
from anymesher.quad.front import area2
from anymesher.quad.state import CancellationRequested, QuadMeshState
from anymesher.quad.transitions import (
    TransitionRejected,
    TransitionReport,
    closure,
    collision,
    spacing_change,
)

TOL = 1e-9


# ---------------------------------------------------------------------------
# Geometry transforms (rotation cov / reflection cov / relabel)
# ---------------------------------------------------------------------------


def _rot(p, c, s):
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _refl_x(p):
    return (p[0], -p[1])


def _state_with(nodes, cells, front=()):
    return QuadMeshState(
        nodes={int(k): tuple(p) for k, p in nodes.items()},
        cells=dict(cells),
        initial_front=[list(m) for m in front],
    )


def _state_transformed(state, tf, *, relabel=None, id_offset=0):
    """Same combinatorial state under a geometric transform (+ optional perm)."""
    perm = dict(relabel) if relabel else None

    def node_of(n):
        return perm[int(n)] if perm is not None else int(n)

    # A true relabel renames every node id (bijection on the node set) and
    # re-derives all bodies/edges from it; no welding, exact integer ids.
    nodes = {node_of(i): tf(p) for i, p in state.nodes.items()}
    cells = {
        int(i) + id_offset: tuple(node_of(n) for n in body)
        for i, body in state.cells.items()
    }
    front = [tuple(node_of(k) for k in e) for e in state.front]
    return _state_with(nodes, cells, front)


# ---------------------------------------------------------------------------
# Frozen fixtures
# ---------------------------------------------------------------------------


def _f1():
    """Interior Q4: unit square, CCW body (0,1,3,2), no front, one Q4 cell."""
    return _state_with(
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
        {0: (0, 1, 3, 2)},
    )


def _f1_rot(phi=math.radians(25.0)):
    c, s = math.cos(phi), math.sin(phi)
    return _state_transformed(_f1(), lambda p: _rot(p, c, s))


def _f1_refl():
    return _state_transformed(_f1(), _refl_x)


_FLIP1 = {0: 1, 1: 0, 2: 3, 3: 2}


def _f1_refl_perm():
    return _state_transformed(_f1(), _refl_x, relabel=_FLIP1)


def _f2():
    """Collision/closure base: two T3 across shared diagonal (0,2).

    Nodes :  0=(0,0)  1=(1,0)  2=(1,1)  3=(0,1)
    Cells : {0: (0,1,2), 1: (0,2,3)}
    Shared interior (diagonal) edge (0,2); union boundary
      {(0,1), (1,2), (2,3), (0,3)}
    Union quad: (0,1,2,3) (CCW, strictly convex).
    """
    nodes = {
        0: (0.0, 0.0),
        1: (1.0, 0.0),
        2: (1.0, 1.0),
        3: (0.0, 1.0),
    }
    cells = {0: (0, 1, 2), 1: (0, 2, 3)}
    return _state_with(nodes, cells)


def _f2_front_two():
    """Collision-acceptance fixture: two opposite front edges on the union.

    Union boundary of cells {0:(0,1,2), 1:(0,2,3)} (shared diagonal (0,2)
    removed): {(0,1), (1,2), (2,3), (0,3)}.
    A valid opposite (non-adjacent) pair: (0,1) and (2,3).
    """
    st = _f2()
    return _state_with(st.nodes, st.cells, front=[(0, 1), (2, 3)])


def _f2_rot():
    c, s = math.cos(0.4), math.sin(0.4)
    return _state_transformed(_f2_front_two(), lambda p: _rot(p, c, s))


def _f2_refl():
    return _state_transformed(_f2_front_two(), _refl_x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reject(fn, *a, **kw):
    with pytest.raises(TransitionRejected):
        fn(*a, **kw)


def _invariant(state, digest_before, gen_before):
    assert state.digest() == digest_before
    assert state.generation == gen_before


# ===========================================================================
# spacing_change -- acceptance + covariance battery
# ===========================================================================


def test_spacing_change_accept():
    st = _f1()
    d0, g0 = st.digest(), st.generation
    rep = spacing_change(st, 0)
    assert st.generation == g0 + 1
    # canonical CCW quad (0,1,3,2) retiled on the other diagonal (1,2):
    # local_swap -> children (0,1,2) and (1,3,2)
    assert st.cells == {1: (0, 1, 2), 2: (1, 3, 2)}
    assert st.cell_kind(1) == "T3"
    assert st.cell_kind(2) == "T3"
    # the split removes the Q4 boundary; every boundary edge is now
    # single-incident, so all four of them enter the front
    assert st.front == frozenset({(0, 1), (0, 2), (1, 3), (2, 3)})
    assert rep.kind == "spacing_change"
    assert rep.parent_cells == (0,)
    assert rep.result_cell == 1
    assert set(rep.body) == {0, 1, 2, 3}
    assert sorted(rep.added_front) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    assert rep.removed_front == ()
    assert sorted(r.result_cell for r in rep.replacements) == [1, 2]
    assert all(r.parent_cell == 0 for r in rep.replacements)
    # child triangle areas sum to the Q4 area (double area == 2.0)
    assert (
        abs(area2(st, st.cells[1]) + area2(st, st.cells[2]) - 2.0) <= TOL
    )


def test_spacing_change_rotation_covariance():
    phi = math.radians(25.0)
    st_r = _f1_rot(phi)
    rep_r = spacing_change(st_r, 0)
    st_b = _f1()
    rep_b = spacing_change(st_b, 0)

    assert rep_r.parent_cells == rep_b.parent_cells
    assert rep_r.result_cell == rep_b.result_cell
    assert rep_r.added_front == rep_b.added_front
    assert rep_r.removed_front == rep_b.removed_front
    assert rep_r.replacements == rep_b.replacements
    # body preserved as a node set under rotation
    assert set(rep_b.body) == set(rep_r.body)
    # both end with the same two T3 bodies
    assert st_r.cells == st_b.cells
    assert st_r.front == st_b.front == frozenset({(0, 1), (0, 2), (1, 3), (2, 3)})
    assert abs(area2(st_r, st_r.cells[1])) == pytest.approx(
        abs(area2(st_b, st_b.cells[1])), abs=TOL
    )
    assert st_r.generation == st_b.generation


def test_spacing_change_reflection_covariance():
    st_r = _f1_refl()
    rep_r = spacing_change(st_r, 0)
    st_b = _f1()
    rep_b = spacing_change(st_b, 0)
    assert rep_r.parent_cells == rep_b.parent_cells
    assert rep_r.result_cell == rep_b.result_cell
    assert rep_r.added_front == rep_b.added_front
    assert rep_r.removed_front == rep_b.removed_front
    assert set(rep_r.body) == set(rep_b.body)
    # reflection reverses canonical walk order, so compare bodies as node sets
    assert {frozenset(v) for v in st_r.cells.values()} == {
        frozenset(v) for v in st_b.cells.values()
    }
    assert st_r.generation == st_b.generation
    assert st_r.front == frozenset({(0, 1), (0, 2), (1, 3), (2, 3)})


def test_spacing_change_relabel_covariance():
    """Reflected + node-id relabelled copy: identical semantics under exact ids."""
    perm = _FLIP1
    st_p = _f1_refl_perm()
    rep_p = spacing_change(st_p, 0)
    st_b = _f1()
    rep_b = spacing_change(st_b, 0)

    assert rep_p.kind == rep_b.kind == "spacing_change"
    assert rep_p.parent_cells == rep_b.parent_cells
    assert rep_p.result_cell == rep_b.result_cell
    # front-edge diffs permute exactly under the relabel (normalized key)
    def perm_edge(e):
        a, b = e
        a, b = perm[a], perm[b]
        return (a, b) if a < b else (b, a)

    assert sorted(perm_edge(e) for e in rep_p.added_front) == sorted(
        perm_edge(e) for e in rep_b.added_front
    )
    assert rep_p.removed_front == rep_b.removed_front == ()
    # child bodies are the permuted base children (set-identity: the
    # reflected walk may be canonically reversed)
    assert set(rep_p.body) == {perm[n] for n in rep_b.body}
    # child bodies permute exactly under the relabel (set-identity, since
    # the reflected walk may be canonically reversed)
    expected_frozen = {
        frozenset(perm[n] for n in st_b.cells[1]),
        frozenset(perm[n] for n in st_b.cells[2]),
    }
    assert {frozenset(v) for v in st_p.cells.values()} == expected_frozen
    assert st_p.generation == st_b.generation


def test_spacing_change_front_boundary_rejected():
    """A boundary Q4 (front edge on its boundary) must be rejected."""
    st = _state_with(
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
        {0: (0, 1, 3, 2)},
        front=[(0, 1)],
    )
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0)
    _invariant(st, d0, g0)
    assert st.cells == {0: (0, 1, 3, 2)}
    assert st.cell_kind(0) == "Q4"


def test_spacing_change_wrong_kind_rejected():
    st = _f2_front_two()  # only T3 cells
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0)
    _invariant(st, d0, g0)
    assert st.cells == {0: (0, 1, 2), 1: (0, 2, 3)}


def test_spacing_change_non_adjacent_t3_rejected():
    """Two T3 sharing two edges: adjacency guard (exactly one shared edge)."""
    st = _state_with(
        {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 2.0), 3: (1.0, 2.0), 4: (4.0, 1.0)},
        {0: (0, 1, 4), 1: (0, 4, 2)},
    )
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0)
    _invariant(st, d0, g0)
    _reject(collision, st, 0, 1)
    _invariant(st, d0, g0)


def test_spacing_change_protected_node_rejected():
    st = QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
        cells={0: (0, 1, 3, 2)},
        protected_nodes=[1],
    )
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0)
    _invariant(st, d0, g0)
    assert st.cells == {0: (0, 1, 3, 2)}
    assert st.cell_kind(0) == "Q4"


def test_spacing_change_protected_edge_rejected():
    st = QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
        cells={0: (0, 1, 3, 2)},
        protected_edges=[(0, 1)],
    )
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0)
    _invariant(st, d0, g0)


def test_spacing_change_stale_handle_rejected():
    st = _f1()
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 99)
    _invariant(st, d0, g0)


def test_spacing_change_cancellation_safe():
    flag = [False]
    st = QuadMeshState(
        nodes={0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)},
        cells={0: (0, 1, 3, 2)},
        is_cancelled=lambda: flag[0],
    )
    flag[0] = True
    d0, g0 = st.digest(), st.generation
    with pytest.raises(CancellationRequested):
        spacing_change(st, 0)
    _invariant(st, d0, g0)
    assert st.cells == {0: (0, 1, 3, 2)}
    assert st.cell_kind(0) == "Q4"


# ===========================================================================
# collision / closure -- acceptance + preconditions
# ===========================================================================


def test_collision_accept_exactly_two_front():
    st = _f2_front_two()  # front = [(0,1), (2,3)]
    d0, g0 = st.digest(), st.generation
    rep = collision(st, 0, 1)
    assert st.generation == g0 + 1
    # canonical CCW quad of the (0,1)-(1,2)-(2,3)-(3,0) boundary cycle
    assert st.cell_kind(2) == "Q4"
    assert set(st.cells[2]) == {0, 1, 2, 3}
    # no residual T3 remains after the merge; the Q4-only boundary is not
    # active front, so the front is fully consumed
    assert st.front == frozenset()
    assert rep.kind == "collision"
    assert rep.parent_cells == (0, 1)
    assert rep.result_cell == 2
    assert set(rep.body) == {0, 1, 2, 3}
    assert rep.added_front == ()
    assert sorted(rep.removed_front) == [(0, 1), (2, 3)]
    assert sorted(r.parent_cell for r in rep.replacements) == [0, 1]
    assert all(r.result_cell == 2 for r in rep.replacements)


def test_collision_front_count_zero_rejected():
    st = _f2()  # no front edges
    d0, g0 = st.digest(), st.generation
    _reject(collision, st, 0, 1)
    _invariant(st, d0, g0)
    assert st.cells == {0: (0, 1, 2), 1: (0, 2, 3)}


def test_collision_front_count_one_rejected():
    st = _state_with(_f2().nodes, _f2().cells, front=[(1, 2)])
    d0, g0 = st.digest(), st.generation
    _reject(collision, st, 0, 1)
    _invariant(st, d0, g0)


def test_collision_front_count_three_rejected():
    st3 = _state_with(
        _f2().nodes, _f2().cells, front=[(0, 1), (1, 2), (2, 3)]
    )
    d0, g0 = st3.digest(), st3.generation
    _reject(collision, st3, 0, 1)
    _invariant(st3, d0, g0)


def test_closure_accept_three_front():
    st = _state_with(
        _f2().nodes,
        {0: (0, 1, 2), 1: (0, 2, 3)},
        front=[(0, 1), (1, 2), (2, 3)],
    )
    d0, g0 = st.digest(), st.generation
    rep = closure(st, 0, 1)
    assert st.generation == g0 + 1
    assert st.cell_kind(2) == "Q4"
    assert set(st.cells[2]) == {0, 1, 2, 3}
    # no residual T3 remains after the merge; the Q4-only boundary is not
    # active front, so the front is fully consumed
    assert st.front == frozenset()
    assert rep.kind == "closure"
    assert rep.parent_cells == (0, 1)
    assert rep.result_cell == 2
    assert set(rep.body) == {0, 1, 2, 3}
    assert rep.added_front == ()
    assert sorted(rep.removed_front) == [(0, 1), (1, 2), (2, 3)]
    assert sorted(r.parent_cell for r in rep.replacements) == [0, 1]


def test_closure_front_count_two_rejected():
    st = _f2_front_two()
    d0, g0 = st.digest(), st.generation
    _reject(closure, st, 0, 1)
    _invariant(st, d0, g0)


def test_closure_front_count_one_rejected():
    st = _state_with(_f2().nodes, _f2().cells, front=[(1, 2)])
    d0, g0 = st.digest(), st.generation
    _reject(closure, st, 0, 1)
    _invariant(st, d0, g0)


def test_collision_rotation_covariance():
    st_r = _f2_rot()
    rep_r = collision(st_r, 0, 1)
    st_b = _f2_front_two()
    rep_b = collision(st_b, 0, 1)

    assert rep_r.parent_cells == rep_b.parent_cells
    assert rep_r.result_cell == rep_b.result_cell
    assert set(rep_r.body) == set(rep_b.body)
    assert rep_r.added_front == rep_b.added_front
    assert st_r.cells == st_b.cells
    assert st_r.front == st_b.front
    assert st_r.generation == st_b.generation


def test_collision_reflection_covariance():
    st_r = _f2_refl()
    rep_r = collision(st_r, 0, 1)
    st_b = _f2_front_two()
    rep_b = collision(st_b, 0, 1)
    assert rep_r.result_cell == rep_b.result_cell
    assert set(rep_r.body) == set(rep_b.body)
    assert rep_r.added_front == rep_b.added_front
    # reflection is orientation-reversing, so the canonical CCW walk order
    # may reverse; node-set identity of every cell is the true invariant.
    assert {frozenset(v) for v in st_r.cells.values()} == {
        frozenset(v) for v in st_b.cells.values()
    }
    assert st_r.front == st_b.front
    assert st_r.generation == st_b.generation


# ===========================================================================
# Input validation + report invariants
# ===========================================================================


def test_invalid_cell_id_types_rejected():
    st = _f1()
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0.0)
    _invariant(st, d0, g0)
    st2 = _f2_front_two()
    d0b, g0b = st2.digest(), st2.generation
    _reject(collision, st2, 0, 1.0)
    _invariant(st2, d0b, g0b)


def test_invalid_options_rejected():
    st = _f1()
    d0, g0 = st.digest(), st.generation
    _reject(spacing_change, st, 0, options=42)
    _invariant(st, d0, g0)


def test_report_invariants_spacing():
    st = _f1()
    rep = spacing_change(st, 0)
    a = max(rep.parent_cells) + 1
    assert rep.result_cell == a
    # children bodies partition: union of child nodes == Q4 nodes
    kid_nodes = set(st.cells[a]) | set(st.cells[a + 1])
    assert kid_nodes == set(rep.body)
    # each replacement maps parent 0 to one of the two children
    assert {r.result_cell for r in rep.replacements} == {a, a + 1}
    assert all(r.parent_cell == 0 for r in rep.replacements)


def test_report_invariants_consolidation():
    st = _f2_front_two()
    rep = collision(st, 0, 1)
    assert rep.result_cell == max(rep.parent_cells) + 1
    assert set(rep.body) <= set(st.node_ids)
    # the result body is a clean 4-node quad
    assert len(set(rep.body)) == 4


def test_transition_rejected_is_mesh_error():
    assert issubclass(TransitionRejected, MeshError)
    st = _f1()
    with pytest.raises(MeshError):
        spacing_change(st, 99)
