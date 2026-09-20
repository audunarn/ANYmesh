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
exact integer id (no coordinate welding); the Steiner node id is ``max(node_ids)+1``
and the two child cell ids are ``max(cells)+1``, ``+2`` — fully deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..errors import MeshError
from .front import (
    FrontRejected,
    candidate_partners,
    classify,
    edge_key,
    find_source_cell,
    front_step,
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


def _next_node_id(state: QuadMeshState) -> int:
    return max(state.node_ids, default=-1) + 1


def _next_cell_ids(state: QuadMeshState) -> tuple[int, int]:
    ca = max(state.cells, default=-1) + 1
    return ca, ca + 1


def _stage_split(tx: Any, source: int, a: int, b: int, third: int, m_id: int,
                 ca_id: int, cb_id: int, pm: tuple[float, float]) -> None:
    tx.add_node(m_id, pm)
    tx.remove_cell(source)
    tx.add_cell(ca_id, (a, m_id, third), "T3")
    tx.add_cell(cb_id, (m_id, b, third), "T3")
    tx.remove_front_edge(a, b)
    tx.add_front_edge(a, m_id)
    tx.add_front_edge(m_id, b)


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

    m_id = _next_node_id(state)
    ca_id, cb_id = _next_cell_ids(state)
    pm = _midpoint(state, a, b, r)

    with state.transaction() as tx:
        _stage_split(tx, source, a, b, third, m_id, ca_id, cb_id, pm)
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
    """Enable a ``Q4`` at ``edge`` from a T3-only state via bounded Steiner split.

    Ratios are tried in the given order (default :data:`DEFAULT_RECOVERY_RATIOS`).
    For each ratio the split is *dry-run* through a transaction view; only when a
    child front edge is provably admissible for :func:`front_step` is the split
    committed, and the **real** :func:`front_step` then runs on that child edge to
    publish the ``Q4``.  A ratio that cannot enable a quad is discarded
    (state unchanged).  If no ratio succeeds, :class:`RecoveryExhausted` is raised
    and the state is left exactly as found.
    """
    _validate_options(options)
    if ratios is None:
        ratios = DEFAULT_RECOVERY_RATIOS
    fe = edge_key(int(edge[0]), int(edge[1]))
    state.checkpoint()
    source = _resolve_source(state, fe)
    third = _third_vertex(tuple(state.cell(source)), fe)
    a, b = fe

    attempts: list[Attempt] = []
    for r in ratios:
        rr = _validate_ratio(r)
        state.checkpoint()
        m_id = _next_node_id(state)
        ca_id, cb_id = _next_cell_ids(state)
        pm = _midpoint(state, a, b, rr)

        enabled: tuple[int, int] | None = None
        detail = ""
        with state.transaction() as tx:
            _stage_split(tx, source, a, b, third, m_id, ca_id, cb_id, pm)
            view = tx.view
            enabled, detail = _dry_run_enables(view, a, b, m_id)
            if enabled is not None:
                tx.commit()

        if enabled is None:
            attempts.append(Attempt(rr, m_id, None, False, detail or "no enabling child edge"))
            continue

        # Split committed.  Drive the real front driver on the enabling child edge.
        new_id, body = front_step(state, enabled, options)
        attempts.append(
            Attempt(rr, m_id, tuple(enabled), True, detail or "enabled")
        )
        return AdvanceReport(
            front_edge=fe,
            ratio=rr,
            midpoint_id=m_id,
            child_cells=(ca_id, cb_id),
            quad_cell_id=new_id,
            quad_body=tuple(body),
            enabling_edge=edge_key(*enabled),
            attempts=tuple(attempts),
        )

    raise RecoveryExhausted(
        f"edge {tuple(fe)}: no recovery ratio enabled a Q4; "
        + "; ".join(f"r={at.ratio} ({at.detail})" for at in attempts)
    )
