"""Bounded, deterministic front-edge Steiner split recovery for the quad-first front.

When a front edge's unique ``T3`` source *cannot* be paired into a ``Q4`` by the
pure :func:`~anymesher.quad.front.front_step` driver (the source and its ``T3``
partner form a concave / degenerate union), the only geometric way to enable a
Quad from the same T3 seed is to insert a Steiner node on a front edge and split
the source triangle into two, so that a *new* ``T3`` child pairs with a
neighbour into a strictly-convex quad.

This module provides that recovery as a **bounded, deterministic** operation:

* :func:`edge_split_recover` — commit one front-edge split at an explicit
  ratio (1 ``T3`` -> 2 ``T3`` children) atomically.
* :func:`recover_then_front_step` — try a fixed, ordered ratio schedule; for
  each ratio dry-run the split through a :class:`~.journal.Transaction` view,
  and *only if* a child front edge is provably admissible for a
  :func:`~anymesher.quad.front.front_step` quad, commit the split and then drive
  the **real** :func:`~anymesher.quad.front.front_step` on that child edge.  No
  hidden fallback: if no ratio enables a quad, the state is left untouched and
  :class:`RecoveryExhausted` is raised.

Geometry is owned by ANYgeometry; this module only classifies via the existing
:mod:`~anymesher.quad.front` primitives and node positions.  Node identity is an
exact integer id (no coordinate welding); the Steiner node id and the two child
cell ids are allocated by the transaction-local cursors
(:meth:`Transaction.allocate_node` / :meth:`Transaction.allocate_cell`) and
remain deterministic for a deterministic delta order — the base allocator is
only advanced on successful commit, so cancelled dry-runs consume no ids.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import MeshError
from .front import (
    EPS,
    FrontRejected,
    _is_residual_front,
    area2,
    body_edges,
    candidate_partners,
    classify,
    edge_key,
    find_source_cell,
    front_step,
    make_quad,
)
from .options import QuadMeshingOptions
from .state import EdgeKey, QuadMeshState

__all__ = [
    "RecoveryRejected",
    "RecoveryExhausted",
    "Attempt",
    "SplitReport",
    "AdvanceReport",
    "DEFAULT_RECOVERY_RATIOS",
    "edge_split_recover",
    "recover_then_front_step",
]

# Default ordered ratio schedule: the canonical mid-split first, then a
# deterministic fallback set.  Order is part of the contract (determinism).
DEFAULT_RECOVERY_RATIOS: tuple[float, ...] = (0.5, 0.4, 0.6, 0.3, 0.7)


class RecoveryRejected(MeshError):
    """A specific recovery rule rejected the edge/ratio (typed, non-fatal)."""


class RecoveryExhausted(MeshError):
    """Every scheduled recovery ratio failed to enable a ``Q4``."""


@dataclass(frozen=True)
class Attempt:
    """Outcome of one recovery ratio within :func:`recover_then_front_step`."""

    ratio: float
    midpoint_id: int
    enabled_edge: tuple[int, int] | None
    ok: bool
    detail: str


@dataclass(frozen=True)
class SplitReport:
    """Result of a committed :func:`edge_split_recover` split."""

    front_edge: tuple[int, int]
    ratio: float
    midpoint_id: int
    source_cell: int
    child_cells: tuple[int, int, ...]


@dataclass(frozen=True)
class AdvanceReport:
    """Result of a successful :func:`recover_then_front_step` recovery."""

    front_edge: tuple[int, int]
    ratio: float
    midpoint_id: int
    child_cells: tuple[int, int, ...]
    quad_cell_id: int
    quad_body: tuple[int, int, int, int]
    enabling_edge: tuple[int, int]
    attempts: tuple[Attempt, ...]
    local_front_edges: tuple[EdgeKey, ...] = ()


# ---------------------------------------------------------------------------
# Validation helpers (typed failures, never silent)
# ---------------------------------------------------------------------------

def _validate_options(options: Any) -> None:
    if options is not None and not isinstance(options, (QuadMeshingOptions, Mapping)):
        raise MeshError("options must be None, QuadMeshingOptions or a mapping")


def _validate_ratio(ratio: Any) -> float:
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise RecoveryRejected("ratio must be a real number")
    r = float(ratio)
    if not math.isfinite(r) or r <= 0.0 or r >= 1.0:
        raise RecoveryRejected(f"ratio {r!r} must be strictly inside (0, 1)")
    return r


def _third_vertex(body: Sequence[int], fe: EdgeKey) -> int:
    """The source T3 vertex not on the front edge (the split's apex)."""
    ab = {fe[0], fe[1]}
    remaining = [n for n in body if n not in ab]
    if len(remaining) != 1:
        raise RecoveryRejected(
            f"front edge {tuple(fe)} does not cut the source triangle cleanly"
        )
    return remaining[0]


def _midpoint(state: QuadMeshState, a: int, b: int, r: float) -> tuple[float, float]:
    pa = state.position(a)
    pb = state.position(b)
    return (pa[0] + r * (pb[0] - pa[0]), pa[1] + r * (pb[1] - pa[1]))


def _resolve_source(state: QuadMeshState, fe: EdgeKey) -> int:
    """Validate ``fe`` is a recoverable front edge; return its unique T3 source."""
    if not state.is_front_edge(fe):
        raise RecoveryRejected(f"edge {tuple(fe)} is not on the active front")
    if state.is_protected_edge(fe):
        raise RecoveryRejected(f"edge {tuple(fe)} is a protected edge")
    if state.is_protected_node(fe[0]) or state.is_protected_node(fe[1]):
        raise RecoveryRejected(f"endpoint of edge {tuple(fe)} is a protected node")
    try:
        return find_source_cell(state, fe)
    except FrontRejected as exc:
        raise RecoveryRejected(str(exc)) from exc


def _stage_split(tx: Any, source: int, a: int, b: int, third: int,
                 pm: tuple[float, float]) -> tuple[int, int, int]:
    """Stage one Steiner split through the transaction; return ``(m, ca, cb)``.

    The midpoint node id and the two child cell ids come from the transaction
    cursors, so a deterministic delta order pins them exactly while cancelled
    dry-runs consume nothing on the base.
    """
    m_id = tx.allocate_node(pm)
    tx.remove_cell(source)
    ca_id = tx.allocate_cell((a, m_id, third), "T3")
    cb_id = tx.allocate_cell((m_id, b, third), "T3")
    tx.remove_front_edge(a, b)
    tx.add_front_edge(a, m_id)
    tx.add_front_edge(m_id, b)
    return m_id, ca_id, cb_id


def _ccw_tri(view: Any, raw: Sequence[int]) -> tuple[int, int, int]:
    body = tuple(int(x) for x in raw)
    if len(body) != 3 or len(set(body)) != 3:
        raise RecoveryRejected(f"triangle body {body!r} must have three distinct nodes")
    signed = area2(view, body)
    if abs(signed) <= EPS:
        raise RecoveryRejected(f"triangle body {body!r} is degenerate")
    return body if signed > 0.0 else (body[0], body[2], body[1])


def _accepted_q4_neighbor(state: QuadMeshState, fe: EdgeKey, source: int) -> int | None:
    q4 = [
        int(cid)
        for cid in state.edge_cells(fe)
        if int(cid) != int(source) and state.cell_kind(int(cid)) == "Q4"
    ]
    if len(q4) > 1:
        raise RecoveryRejected(
            f"front edge {tuple(fe)} has multiple accepted Q4 neighbours {tuple(sorted(q4))}"
        )
    return q4[0] if q4 else None


def _q4_edge_walk(state: QuadMeshState, q4_cell: int, fe: EdgeKey) -> tuple[int, int, int, int]:
    body = tuple(int(x) for x in state.cell(q4_cell))
    if len(body) != 4:
        raise RecoveryRejected(f"accepted cell {q4_cell} is not a four-node Q4")
    for i in range(4):
        u, v = body[i], body[(i + 1) % 4]
        if edge_key(u, v) == fe:
            return (u, v, body[(i + 2) % 4], body[(i + 3) % 4])
    raise RecoveryRejected(f"accepted Q4 {q4_cell} does not contain front edge {tuple(fe)}")


def _stage_conforming_split(
    tx: Any,
    state: QuadMeshState,
    source: int,
    fe: EdgeKey,
    third: int,
    pm: tuple[float, float],
    accepted_q4: int,
    variant: int,
) -> tuple[int, int, int, tuple[EdgeKey, ...]]:
    """Split a Q4/T3 front edge on both sides with one shared midpoint."""
    a, b = fe
    source_body = tuple(int(x) for x in state.cell(source))
    q4_body = tuple(int(x) for x in state.cell(accepted_q4))
    u, v, c, d = _q4_edge_walk(state, accepted_q4, fe)

    m_id = tx.allocate_node(pm)
    tx.remove_cell(source)
    ca_body = _ccw_tri(tx.view, (a, m_id, third))
    cb_body = _ccw_tri(tx.view, (m_id, b, third))
    ca_id = tx.allocate_cell(ca_body, "T3")
    cb_id = tx.allocate_cell(cb_body, "T3")

    tx.remove_cell(accepted_q4)
    if variant == 0:
        accepted_tri = _ccw_tri(tx.view, (u, m_id, d))
        accepted_quad = make_quad(tx.view, (m_id, v, c, d))
    elif variant == 1:
        accepted_tri = _ccw_tri(tx.view, (m_id, v, c))
        accepted_quad = make_quad(tx.view, (u, m_id, c, d))
    else:
        raise RecoveryRejected(f"unknown conforming recovery variant {variant}")
    tx.allocate_cell(accepted_quad, "Q4")
    tx.allocate_cell(accepted_tri, "T3")

    touched = (
        set(body_edges(source_body))
        | set(body_edges(q4_body))
        | set(body_edges(ca_body))
        | set(body_edges(cb_body))
        | set(body_edges(accepted_quad))
        | set(body_edges(accepted_tri))
    )
    for k in sorted(touched):
        is_front_now = _is_residual_front(tx.view, k)
        was_front = state.is_front_edge(k)
        if is_front_now and not was_front:
            tx.add_front_edge(k[0], k[1])
        elif (not is_front_now) and was_front:
            tx.remove_front_edge(k[0], k[1])
    return m_id, ca_id, cb_id, tuple(sorted(touched))


# ---------------------------------------------------------------------------
# Public operations
# ---------------------------------------------------------------------------

def edge_split_recover(
    state: QuadMeshState,
    edge: Sequence[int],
    ratio: float = 0.5,
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> SplitReport:
    """Split the front edge ``edge`` at ``ratio`` into two ``T3`` children.

    Atomically removes the unique ``T3`` source below ``edge`` and publishes two
    ``T3`` children sharing a new Steiner node placed at
    ``pos[a] + ratio * (pos[b] - pos[a])``; the front edge is re-exposed as its
    two children.  Typed :class:`RecoveryRejected` on invalid input; transaction
    atomicity otherwise leaves the state untouched.
    """
    _validate_options(options)
    fe = edge_key(int(edge[0]), int(edge[1]))
    state.checkpoint()
    source = _resolve_source(state, fe)
    r = _validate_ratio(ratio)
    third = _third_vertex(tuple(state.cell(source)), fe)
    a, b = fe
    pm = _midpoint(state, a, b, r)

    m_id, ca_id, cb_id = (-1, -1, -1)
    with state.transaction() as tx:
        m_id, ca_id, cb_id = _stage_split(tx, source, a, b, third, pm)
        tx.commit()  # atomic: prevalidates (incidence, residency) then applies

    return SplitReport(
        front_edge=fe,
        ratio=r,
        midpoint_id=m_id,
        source_cell=source,
        child_cells=(ca_id, cb_id),
    )


def _dry_run_enables(
    view: Any, a: int, b: int, m_id: int
) -> tuple[tuple[int, int] | None, str]:
    """On the *candidate* view, find a child front edge that classifies cleanly.

    Deterministic order: child edge ``(a, m)`` is tried before ``(m, b)``; its
    candidates are iterated in :func:`candidate_partners` order.  Returns the
    enabling child edge (or ``None``) plus a diagnostic detail string.
    """
    details: list[str] = []
    for child in ((a, m_id), (m_id, b)):
        ck = edge_key(*child)
        if not view.is_front_edge(ck):
            details.append(f"child edge {tuple(ck)} is not on the front")
            continue
        try:
            src = find_source_cell(view, ck)
        except FrontRejected as exc:
            details.append(str(exc))
            continue
        try:
            partners = candidate_partners(view, src)
        except FrontRejected as exc:
            details.append(str(exc))
            continue
        if not partners:
            details.append(f"child edge {tuple(ck)}: source {src} has no T3 partner")
            continue
        for p in partners:
            try:
                classify(view, ck, src, p)
                return child, f"enabled via source {src} / partner {p}"
            except FrontRejected:
                continue
    return None, " | ".join(details) or "no child edge admissible"


def recover_then_front_step(
    state: QuadMeshState,
    edge: Sequence[int],
    ratios: Sequence[float] = DEFAULT_RECOVERY_RATIOS,
    options: QuadMeshingOptions | Mapping[str, Any] | None = None,
) -> AdvanceReport:
    """Enable a Q4 through bounded recovery without creating a hanging interface.

    Pure-T3 recovery keeps the original one-sided split contract.  When the
    active front separates a residual T3 from an accepted Q4, the accepted Q4
    is re-tiled in the same transaction as one Q4 plus one T3 using the same
    midpoint, so the committed mesh remains conforming.
    """
    _validate_options(options)
    if ratios is None:
        ratios = DEFAULT_RECOVERY_RATIOS
    fe = edge_key(int(edge[0]), int(edge[1]))
    state.checkpoint()
    source = _resolve_source(state, fe)
    third = _third_vertex(tuple(state.cell(source)), fe)
    a, b = fe
    accepted_q4 = _accepted_q4_neighbor(state, fe, source)

    attempts: list[Attempt] = []
    for r in ratios:
        rr = _validate_ratio(r)
        state.checkpoint()
        pm = _midpoint(state, a, b, rr)
        ratio_details: list[str] = []
        variants = (0, 1) if accepted_q4 is not None else (None,)

        for variant in variants:
            m_id = state.next_node_id
            ca_id = state.next_cell_id
            cb_id = state.next_cell_id + 1
            enabled: tuple[int, int] | None = None
            detail = ""
            touched: tuple[EdgeKey, ...] = ()
            try:
                with state.transaction() as tx:
                    if accepted_q4 is None:
                        _stage_split(tx, source, a, b, third, pm)
                        touched = tuple(
                            sorted(
                                set(body_edges(state.cell(source)))
                                | {edge_key(a, m_id), edge_key(m_id, b)}
                            )
                        )
                    else:
                        m_id, ca_id, cb_id, touched = _stage_conforming_split(
                            tx, state, source, fe, third, pm, accepted_q4, int(variant)
                        )
                    view = tx.view
                    enabled, detail = _dry_run_enables(view, a, b, m_id)
                    if enabled is not None:
                        tx.commit()
            except (FrontRejected, RecoveryRejected) as exc:
                ratio_details.append(f"variant={variant}: {exc}")
                continue

            if enabled is None:
                ratio_details.append(
                    f"variant={variant}: {detail or 'no enabling child edge'}"
                )
                continue

            new_id, body = front_step(state, enabled, options)
            local = set(touched) | set(body_edges(body))
            local_front = tuple(sorted(k for k in local if state.is_front_edge(k)))
            success_detail = detail or "enabled"
            if accepted_q4 is not None:
                success_detail = f"conforming variant={variant}; {success_detail}"
            attempts.append(Attempt(rr, m_id, tuple(enabled), True, success_detail))
            return AdvanceReport(
                front_edge=fe,
                ratio=rr,
                midpoint_id=m_id,
                child_cells=(ca_id, cb_id),
                quad_cell_id=new_id,
                quad_body=tuple(body),
                enabling_edge=edge_key(*enabled),
                attempts=tuple(attempts),
                local_front_edges=local_front,
            )

        attempts.append(
            Attempt(
                rr,
                state.next_node_id,
                None,
                False,
                " | ".join(ratio_details) or "no enabling child edge",
            )
        )

    raise RecoveryExhausted(
        f"edge {tuple(fe)}: no recovery ratio enabled a Q4; "
        + "; ".join(f"r={at.ratio} ({at.detail})" for at in attempts)
    )
