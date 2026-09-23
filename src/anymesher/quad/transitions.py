"""Q3b quad transitions: deterministic, transactional local re-tiles.

Fixed quad-transition operations built on the resident Q1/Q2 state and the
sparse :class:`~.journal.Transaction` writer:

* :func:`spacing_change` — flip the diagonal of one **interior** ``Q4`` into
  two ``T3`` children (a pure spacing/re-tile of an interior quad);
* :func:`collision` / :func:`closure` — merge two adjacent admissible ``T3``
  cells into one ``Q4`` on the other diagonal (their shared edge is the
  interior diagonal being removed).

Every operation is

* **deterministic** - result cell ids come from transaction-owned monotone
  allocators and are only consumed by successful commits; node identity is
  never coordinate-welded or renumbered;
* **sparse-local** — only the touched boundary edges, the cell set and the
  front state of those edges change;
* **transactional / cancellation-safe** — a failure raises a typed
  :class:`TransitionRejected` (a :class:`MeshError`) and leaves the state
  byte-identical (digest + generation invariant);
* **provenance-bearing** — a success returns a frozen :class:`TransitionReport`.

Geometry (strict convexity, signed area, CCW normalisation) is owned by
:mod:`anymesher.quad.front`; the front reconciliation mirrors
:func:`anymesher.quad.front.front_step` exactly (residual-T3 incidence on the
staged view). Node identity is the exact integer id; edges are keyed by the
exact node pair (``lo < hi``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..errors import MeshError
from .front import EPS, FrontRejected, _is_residual_front, area2, body_edges, local_swap, make_quad
from .options import QuadMeshingOptions
from .state import EdgeKey, QuadMeshState

__all__ = [
    "TransitionRejected",
    "ParentReplacement",
    "TransitionReport",
    "spacing_change",
    "collision",
    "closure",
]


# ---------------------------------------------------------------------------
# Typed failure + provenance
# ---------------------------------------------------------------------------

class TransitionRejected(MeshError):
    """A specific transition rule rejected the operation (typed, non-fatal)."""


@dataclass(frozen=True)
class ParentReplacement:
    """One parent cell consumed by the transition, mapped to its replacement."""

    parent_cell: int
    result_cell: int


@dataclass(frozen=True)
class TransitionReport:
    """Immutable provenance for a committed quad transition.

    ``kind`` is ``"spacing_change"`` | ``"collision"`` | ``"closure"``;
    ``parent_cells`` are the consumed cells; ``result_cell`` is a canonical
    result cell id; ``body`` is the canonical convex quad region;
    ``added_front`` / ``removed_front`` are the front-set diff;
    ``replacements`` map each parent to its replacing cell(s).
    """

    kind: str
    parent_cells: tuple[int, ...]
    result_cell: int
    body: tuple[int, ...]
    generation_before: int
    generation_after: int
    added_front: tuple[EdgeKey, ...] = field(default=())
    removed_front: tuple[EdgeKey, ...] = field(default=())
    replacements: tuple[ParentReplacement, ...] = field(default=())


# ---------------------------------------------------------------------------
# Shared helpers (guards, geometry, staging)
# ---------------------------------------------------------------------------

def _validate_options(options: Any) -> None:
    if options is not None and not isinstance(options, (QuadMeshingOptions, Mapping)):
        raise TransitionRejected("options must be None, QuadMeshingOptions or a mapping")


def _as_cell_id(value: Any, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TransitionRejected(f"{role} must be an integer cell id")
    return value


def _guard_resident(state: QuadMeshState, cid: int, expected_kind: str, role: str) -> tuple[int, ...]:
    """Validate ``cid`` is a resident cell of ``expected_kind``; return its body.

    Also rejects a cell touching a protected node or edge (the transition would
    move or remove a protected feature), mirroring ``front.classify`` semantics.
    """
    try:
        body = state.cell(cid)
    except MeshError as exc:  # StaleHandleError / MeshError
        raise TransitionRejected(f"{role} cell {cid} is not resident") from exc
    if state.cell_kind(cid) != expected_kind:
        raise TransitionRejected(
            f"{role} cell {cid} must be {expected_kind} (found {state.cell_kind(cid)})"
        )
    for n in body:
        if state.is_protected_node(n):
            raise TransitionRejected(f"{role} cell {cid} touches protected node {n}")
    for e in body_edges(body):
        if state.is_protected_edge(e):
            raise TransitionRejected(f"{role} cell {cid} touches protected edge {tuple(e)}")
    return body


def _strict_quad(state: QuadMeshState, body: Sequence[int], role: str) -> tuple[int, int, int, int]:
    """A canonical CCW strictly-convex quad for a 4-node body; typed rejection otherwise."""
    if abs(area2(state, body)) <= EPS:
        raise TransitionRejected(f"{role} quad is degenerate (zero area)")
    try:
        return make_quad(state, tuple(body))
    except FrontRejected as exc:
        raise TransitionRejected(str(exc)) from exc


def _shared_interior_edge(state: QuadMeshState, a: int, b: int) -> EdgeKey:
    """Exactly one shared edge, not on the front; :class:`TransitionRejected` otherwise."""
    shared = set(body_edges(state.cell(a))) & set(body_edges(state.cell(b)))
    if len(shared) != 1:
        raise TransitionRejected(
            f"cells {a} and {b} must share exactly one interior edge (share {len(shared)})"
        )
    diag = next(iter(shared))
    if state.is_front_edge(diag):
        raise TransitionRejected(
            f"cells {a} and {b} share a front edge {tuple(diag)}; the shared edge must be interior"
        )
    return diag


def _boundary_quad(state: QuadMeshState, boundary: set[EdgeKey]) -> tuple[int, int, int, int]:
    """A canonical CCW strictly-convex quad from a clean 4-edge boundary.

    Walks the 4-edge cycle (each node degree 2) and normalises orientation with
    :func:`anymesher.quad.front.make_quad` (strict-convexity + CcW).
    """
    if len(boundary) != 4:
        raise TransitionRejected(
            f"union has {len(boundary)} boundary edges (need exactly 4 for a clean quad)"
        )
    adj: dict[int, list[int]] = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    for nn, nb in adj.items():
        if len(nb) != 2:
            raise TransitionRejected("candidate boundary is not a clean 4-walk (node degree != 2)")

    start = min(nn for k in boundary for nn in k)
    first = adj[start][0]
    path = [start, first]
    prev, cur = start, first
    for _ in range(2):
        nxt = adj[cur][1 if adj[cur][0] == prev else 0]
        path.append(nxt)
        prev, cur = cur, nxt
    return _strict_quad(state, tuple(path), "candidate boundary")


def _reconcile_front(tx: Any, state: QuadMeshState, touched: set[EdgeKey]) -> tuple[tuple[EdgeKey, ...], tuple[EdgeKey, ...]]:
    """Reconcile front membership of ``touched`` edges on the staged edit.

    Mirrors :func:`anymesher.quad.front.front_step`: an edge is *front* iff it
    has exactly one incident cell.  Stages the add/remove front edges into
    ``tx`` and returns the (added, removed) diff for the report.
    """
    added, removed = [], []
    for k in sorted(touched):
        is_front_now = _is_residual_front(tx.view, k)
        was_front = state.is_front_edge(k)
        if is_front_now and not was_front:
            tx.add_front_edge(k[0], k[1])
            added.append(k)
        elif (not is_front_now) and was_front:
            tx.remove_front_edge(k[0], k[1])
            removed.append(k)
    return tuple(added), tuple(removed)


# ---------------------------------------------------------------------------
# Public transitions
# ---------------------------------------------------------------------------

def spacing_change(
    state: QuadMeshState,
    cell: int,
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> TransitionReport:
    """Flip the diagonal of an interior ``Q4`` into two ``T3`` children.

    ``cell`` must be a strictly-convex **interior** ``Q4``: none of its boundary
    edges on the active front, and no boundary edge or endpoint protected.  The
    quad is retiled on its other diagonal (via
    :func:`anymesher.quad.front.local_swap`) into two nondegenerate ``T3``
    triangles, committed atomically.

    Raises a typed :class:`TransitionRejected` on any violated rule; the state
    digest and generation are invariant on every rejection path.
    """
    _validate_options(options)
    cid = _as_cell_id(cell, "spacing_change source")
    body = _guard_resident(state, cid, "Q4", "spacing_change source")
    quad = _strict_quad(state, body, "spacing_change source")

    # Interior guard: no boundary edge may be on the active front.
    for e in body_edges(quad):
        if state.is_front_edge(e):
            raise TransitionRejected(
                f"boundary edge {tuple(e)} of Q4 {cid} is on the active front; "
                "spacing_change requires an interior quad"
            )

    t3a, t3b = local_swap(quad)
    for t, label in ((t3a, "first"), (t3b, "second")):
        if abs(area2(state, t)) <= EPS:
            raise TransitionRejected(f"{label} split triangle {t!r} is degenerate")

    touched = set(body_edges(quad))
    generation_before = state.generation

    new_id_a = new_id_b = -1
    with state.transaction() as tx:
        tx.remove_cell(cid)
        new_id_a = tx.allocate_cell(t3a, "T3")
        new_id_b = tx.allocate_cell(t3b, "T3")
        added, removed = _reconcile_front(tx, state, touched)
        tx.commit()  # atomic: prevalidates (incl. front residency) then applies

    return TransitionReport(
        kind="spacing_change",
        parent_cells=(cid,),
        result_cell=new_id_a,
        body=quad,
        generation_before=generation_before,
        generation_after=state.generation,
        added_front=added,
        removed_front=removed,
        replacements=(
            ParentReplacement(parent_cell=cid, result_cell=new_id_a),
            ParentReplacement(parent_cell=cid, result_cell=new_id_b),
        ),
    )


def _consolidate(
    state: QuadMeshState,
    cell_a: int,
    cell_b: int,
    kind: str,
    options: QuadMeshingOptions | Mapping[str, Any] | None,
) -> TransitionReport:
    """Merge two adjacent admissible ``T3`` cells into one ``Q4``.

    ``kind`` is ``"collision"`` or ``"closure"``; the only difference is the
    front-edge count guard on the union boundary and the reported provenance.
    """
    _validate_options(options)
    ca = _as_cell_id(cell_a, f"{kind} lower cell")
    cb = _as_cell_id(cell_b, f"{kind} upper cell")
    body_a = _guard_resident(state, ca, "T3", f"{kind} lower cell")
    body_b = _guard_resident(state, cb, "T3", f"{kind} upper cell")
    diag = _shared_interior_edge(state, ca, cb)

    ea = set(body_edges(body_a))
    eb = set(body_edges(body_b))
    boundary = (ea | eb) - {diag}
    quad = _boundary_quad(state, boundary)

    # Front-membership of the union boundary before consolidation.
    front_before = [k for k in boundary if state.is_front_edge(k)]
    if kind == "collision" and len(front_before) != 2:
        raise TransitionRejected(
            f"collision requires exactly two active front edges on the union "
            f"(found {len(front_before)}: {sorted(front_before)!r})"
        )
    if kind == "collision" and len(front_before) == 2:
        e1, e2 = front_before
        if set(e1) & set(e2):
            raise TransitionRejected(
                f"collision front edges {tuple(e1)} and {tuple(e2)} are adjacent "
                "(share a node); they must be opposite/non-adjacent"
            )
    if kind == "closure" and len(front_before) < 3:
        raise TransitionRejected(
            f"closure requires at least three active front edges on the union "
            f"(found {len(front_before)}: {sorted(front_before)!r})"
        )

    touched = ea | eb
    generation_before = state.generation

    new_id = -1
    with state.transaction() as tx:
        tx.remove_cell(ca)
        tx.remove_cell(cb)
        new_id = tx.allocate_cell(quad, "Q4")
        added, removed = _reconcile_front(tx, state, touched)
        tx.commit()  # atomic: prevalidates (incl. front residency) then applies

    return TransitionReport(
        kind=kind,
        parent_cells=(ca, cb),
        result_cell=new_id,
        body=quad,
        generation_before=generation_before,
        generation_after=state.generation,
        added_front=added,
        removed_front=removed,
        replacements=(
            ParentReplacement(parent_cell=ca, result_cell=new_id),
            ParentReplacement(parent_cell=cb, result_cell=new_id),
        ),
    )


def collision(
    state: QuadMeshState,
    cell_a: int,
    cell_b: int,
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> TransitionReport:
    """Merge two adjacent admissible ``T3`` cells with **exactly two** active
    front edges on their union boundary into one ``Q4`` (collision).

    The union must be strictly convex, the two cells must share exactly one
    interior (non-front) diagonal, and neither may touch a protected node or
    edge.  Committed atomically.
    """
    return _consolidate(state, cell_a, cell_b, "collision", options)


def closure(
    state: QuadMeshState,
    cell_a: int,
    cell_b: int,
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> TransitionReport:
    """Merge two adjacent admissible ``T3`` cells with **at least three** active
    front edges on their union boundary into one ``Q4`` (closure).

    Same mechanical merge as :func:`collision`; distinguished by the front-edge
    count guard and the reported ``kind`` provenance.
    """
    return _consolidate(state, cell_a, cell_b, "closure", options)
