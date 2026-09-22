"""Deterministic planar quad-front drivers.

``advance_planar_front`` is the narrow Q1/Q2 compatibility driver.  PQ-M1
uses ``run_planar_quad_driver``: one frozen cross field, a deterministic local
front queue, bounded recovery and explicit counters.  Both mutate only the
fresh resident :class:`QuadMeshState` supplied by the caller.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from ..errors import MeshError
from .front import FrontNoCandidate, FrontRejected, body_edges, edge_key, front_step
from .guidance import GuidanceRejected, build_cross_field, front_step_guided
from .options import QuadMeshingOptions
from .recovery import RecoveryExhausted, RecoveryRejected, recover_then_front_step
from .seed import PlanarQuadSeed
from .state import EdgeKey, QuadMeshState

__all__ = [
    "QuadDriverReport",
    "PlanarQuadDriverReport",
    "PlanarQuadDriverResult",
    "advance_planar_front",
    "run_planar_quad_driver",
]


@dataclass(frozen=True)
class QuadDriverReport:
    """Compatibility report for the original deterministic plain-front loop."""

    quads_created: int
    steps_attempted: int
    skipped: tuple
    front_edges_remaining: tuple
    hit_iteration_limit: bool

    def to_dict(self) -> dict:
        return {
            "quads_created": self.quads_created,
            "steps_attempted": self.steps_attempted,
            "skipped": [tuple(edge) for edge, _ in self.skipped],
            "front_edges_remaining": [tuple(edge) for edge in self.front_edges_remaining],
            "hit_iteration_limit": self.hit_iteration_limit,
        }


def _state_cells(state: QuadMeshState) -> Mapping:
    cells = getattr(state, "cells", None)
    if cells is None:
        raise MeshError("QuadMeshState does not expose .cells")
    return cells


def _count_kind(state: QuadMeshState, kind: str) -> int:
    return sum(1 for cid in _state_cells(state) if state.cell_kind(int(cid)) == kind)


def _has_residual_t3(state: QuadMeshState) -> bool:
    return _count_kind(state, "T3") > 0


def _active_front_edges(state: QuadMeshState) -> tuple[EdgeKey, ...]:
    return tuple(sorted(state.front))


def advance_planar_front(
    state: QuadMeshState,
    options: Optional[Any] = None,
    *,
    max_front_iterations: Optional[int] = None,
    cancellation_check: Optional[Callable[[str], None]] = None,
) -> tuple[QuadMeshState, QuadDriverReport]:
    """Compatibility plain-front loop used by earlier focused tests."""
    if options is not None and not isinstance(options, (QuadMeshingOptions, Mapping)):
        raise MeshError("options must be None, QuadMeshingOptions or a mapping")
    initial_cell_count = len(_state_cells(state))
    limit = max(16, 8 + 4 * initial_cell_count) if max_front_iterations is None else int(max_front_iterations)
    if limit <= 0:
        raise MeshError("max_front_iterations must be positive")
    quads_created = 0
    steps_attempted = 0
    skipped: list = []
    blocked: set[EdgeKey] = set()
    while steps_attempted < limit and _has_residual_t3(state):
        if cancellation_check is not None:
            cancellation_check("quad-first:front-step")
        candidates = tuple(e for e in _active_front_edges(state) if e not in blocked)
        if not candidates:
            break
        edge = candidates[0]
        steps_attempted += 1
        try:
            front_step(state, edge, options)
        except (FrontNoCandidate, FrontRejected) as exc:
            blocked.add(edge)
            skipped.append((tuple(edge), f"{type(exc).__name__}: {exc}"))
            continue
        quads_created += 1
    return state, QuadDriverReport(
        quads_created=quads_created,
        steps_attempted=steps_attempted,
        skipped=tuple(skipped),
        front_edges_remaining=_active_front_edges(state),
        hit_iteration_limit=steps_attempted >= limit and _has_residual_t3(state),
    )


@dataclass(frozen=True)
class PlanarQuadDriverReport:
    initial_t3: int
    attempts: int
    stale_skips: int
    guided_accepts: int
    direct_accepts: int
    recovery_attempts: int
    recovery_accepts: int
    recovery_inserted_nodes: int
    no_candidate: int
    queue_pushes: int
    final_q4: int
    final_t3: int
    exhausted_budget: bool
    front_remaining: tuple[EdgeKey, ...]
    field_mode: str
    field_diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_t3": self.initial_t3,
            "attempts": self.attempts,
            "stale_skips": self.stale_skips,
            "guided_accepts": self.guided_accepts,
            "direct_accepts": self.direct_accepts,
            "recovery_attempts": self.recovery_attempts,
            "recovery_accepts": self.recovery_accepts,
            "recovery_inserted_nodes": self.recovery_inserted_nodes,
            "no_candidate": self.no_candidate,
            "queue_pushes": self.queue_pushes,
            "final_q4": self.final_q4,
            "final_t3": self.final_t3,
            "exhausted_budget": self.exhausted_budget,
            "front_remaining": [list(edge) for edge in self.front_remaining],
            "field_mode": self.field_mode,
            "field_diagnostics": list(self.field_diagnostics),
        }


@dataclass(frozen=True)
class PlanarQuadDriverResult:
    state: QuadMeshState
    report: PlanarQuadDriverReport


def _enqueue_local(
    queue: deque[EdgeKey],
    queued: set[EdgeKey],
    state: QuadMeshState,
    edges: Any,
) -> int:
    pushed = 0
    canonical = sorted({edge_key(int(e[0]), int(e[1])) for e in edges})
    for edge in canonical:
        if state.is_front_edge(edge) and edge not in queued:
            queue.append(edge)
            queued.add(edge)
            pushed += 1
    return pushed


def _prioritize_local(
    queue: deque[EdgeKey],
    queued: set[EdgeKey],
    state: QuadMeshState,
    edges: Any,
) -> int:
    """Move live local recovery-front edges to the head, deterministically."""
    live = [
        edge
        for edge in sorted({edge_key(int(e[0]), int(e[1])) for e in edges})
        if state.is_front_edge(edge)
    ]
    for edge in live:
        if edge in queued:
            try:
                queue.remove(edge)
            except ValueError:
                pass
        else:
            queued.add(edge)
    for edge in reversed(live):
        queue.appendleft(edge)
    return sum(1 for edge in live if edge in queued)


def run_planar_quad_driver(
    seed: PlanarQuadSeed,
    options: QuadMeshingOptions,
    *,
    allow_recovery: bool = True,
    cancellation_check: Callable[[str], None] | None = None,
) -> PlanarQuadDriverResult:
    """Run the PQ-M1 bounded guided/direct/recovery advancing-front contract.

    The seed must be fresh for the caller: this function mutates its resident
    state in place.  The cross field is built once.  A recovery-inserted node
    is therefore handled by the ordinary front step when it is absent from the
    frozen field instead of rebuilding the full field after each local edit.
    """
    if not isinstance(seed, PlanarQuadSeed):
        raise MeshError("run_planar_quad_driver requires a PlanarQuadSeed")
    if not isinstance(options, QuadMeshingOptions):
        raise MeshError("run_planar_quad_driver requires QuadMeshingOptions")
    if type(allow_recovery) is not bool:
        raise MeshError("allow_recovery must be bool")

    state = seed.state
    if cancellation_check is not None:
        cancellation_check("quad-first:driver-start")
    field = build_cross_field(state, mode=options.orientation)
    queue: deque[EdgeKey] = deque(sorted(state.front))
    queued: set[EdgeKey] = set(queue)
    initial_t3 = _count_kind(state, "T3")
    attempts = stale_skips = guided_accepts = direct_accepts = 0
    recovery_attempts = recovery_accepts = recovery_inserted_nodes = 0
    no_candidate = 0
    queue_pushes = len(queue)

    while queue and attempts < options.max_front_iterations and _has_residual_t3(state):
        edge = queue.popleft()
        queued.discard(edge)
        if not state.is_front_edge(edge):
            stale_skips += 1
            continue
        attempts += 1
        if cancellation_check is not None and (
            attempts == 1 or attempts % options.cancellation_interval == 0
        ):
            cancellation_check("quad-first:driver-iteration")

        accepted_body: tuple[int, int, int, int] | None = None
        geometric_failure = False
        field_has_edge = edge[0] in field.field and edge[1] in field.field
        if field_has_edge:
            try:
                _cid, accepted_body = front_step_guided(
                    state, edge, report=field, options=options
                )
                guided_accepts += 1
            except GuidanceRejected:
                # Guidance-only failure: the ordinary admissibility path remains valid.
                pass
            except (FrontNoCandidate, FrontRejected):
                geometric_failure = True
        if accepted_body is None and not geometric_failure:
            try:
                _cid, accepted_body = front_step(state, edge, options)
                direct_accepts += 1
            except (FrontNoCandidate, FrontRejected):
                geometric_failure = True

        if accepted_body is not None:
            queue_pushes += _enqueue_local(
                queue, queued, state, body_edges(accepted_body)
            )
            continue

        no_candidate += 1
        can_recover = (
            allow_recovery
            and not state.is_protected_edge(edge)
            and not state.is_protected_node(edge[0])
            and not state.is_protected_node(edge[1])
        )
        if not can_recover:
            continue
        recovery_attempts += 1
        node_before = state.next_node_id
        try:
            recovered = recover_then_front_step(state, edge, options=options)
        except (RecoveryExhausted, RecoveryRejected, FrontNoCandidate, FrontRejected):
            continue
        recovery_accepts += 1
        recovery_inserted_nodes += max(0, state.next_node_id - node_before)
        q4_edges = list(body_edges(recovered.quad_body))
        child_edges: list[EdgeKey] = []
        # Recovery leaves one split child resident.  Resolve that local front
        # before unrelated queued work so subsequent acceptance cannot strand
        # the child behind a nonconforming T-junction.
        for child_id in recovered.child_cells:
            try:
                if state.cell_kind(child_id) == "T3":
                    child_edges.extend(body_edges(state.cell(child_id)))
            except (MeshError, KeyError):
                pass
        child_edges.extend(
            (
                edge_key(edge[0], recovered.midpoint_id),
                edge_key(recovered.midpoint_id, edge[1]),
            )
        )
        child_edges.extend(getattr(recovered, "local_front_edges", ()))
        queue_pushes += _prioritize_local(queue, queued, state, child_edges)
        queue_pushes += _enqueue_local(queue, queued, state, q4_edges)

    final_t3 = _count_kind(state, "T3")
    final_q4 = _count_kind(state, "Q4")
    exhausted = attempts >= options.max_front_iterations and final_t3 > 0
    return PlanarQuadDriverResult(
        state=state,
        report=PlanarQuadDriverReport(
            initial_t3=initial_t3,
            attempts=attempts,
            stale_skips=stale_skips,
            guided_accepts=guided_accepts,
            direct_accepts=direct_accepts,
            recovery_attempts=recovery_attempts,
            recovery_accepts=recovery_accepts,
            recovery_inserted_nodes=recovery_inserted_nodes,
            no_candidate=no_candidate,
            queue_pushes=queue_pushes,
            final_q4=final_q4,
            final_t3=final_t3,
            exhausted_budget=exhausted,
            front_remaining=tuple(sorted(state.front)),
            field_mode=field.mode,
            field_diagnostics=tuple(field.diagnostics),
        ),
    )
