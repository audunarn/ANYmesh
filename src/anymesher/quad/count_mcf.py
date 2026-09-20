"""Quad-first Q4: pure integer min-cost-flow reduction + deterministic tie handling.

This module owns the *reduction* step (``CountInstance -> generic MCF request``)
and the *inverse* step (``MCF response -> validated integer solution``).  The
worker binary (``third_party/quad/worker/quad_mcf_worker.exe``) is a pure,
opaque MCF solver: it knows nothing about quads, stations or rails.  This
module is the only place in the package that knows both sides.

Deterministic tie handling
--------------------------

The solver is a general-purpose min-cost-flow solver; we do *not* trust it to
be deterministic across runs, platforms or solver-internal tie rules.  Instead,
this module encodes a *unique* optimum into the request with a base-``B``
lexicographic perturbation:

    S     = total supply            (strictly > 0)
    B     = S + 1
    m     = number of arcs in the request
    cost'(a) = cost(a) * B**m  +  B**(m-1-k(a))

where ``k(a)`` is the row-major index (``0 <= k < m``) of arc ``a`` in the
request's arc list.  The per-arc tie weights are therefore
``(B**(m-1), B**(m-2), ..., B, 1)``.

Uniqueness: every feasible flow ``F`` on this instance carries flow within
``[0, S]`` on each arc, so the tie contribution

    tie(F) = sum_k  f_k * B**(m-1-k)

is a base-``B`` numeral whose digits are all ``<= S = B - 1`` — the base-``B``
digit string of ``tie(F)`` *is* the flow tuple ``(f_0, ..., f_{m-1})``.
Minimizing the perturbed objective therefore (1) first minimizes the true
primary cost (the primary scale ``B**m`` dominates any tie difference, whose
magnitude is bounded by ``S * (B**m - 1) / (B - 1) < B**m``) and (2) among
equal-primary optima picks the *lexicographically smallest* flow tuple,
because an earlier, smaller digit lowers the base-``B`` value regardless of
the remaining digits.  The optimum is hence unique by construction;
:func:`tie_value` and the cross-check in :func:`validate_response` mirror the
*same* weights, so the Python side — not the solver — breaks ties.  The
solver is asked to minimize ``cost'``; the true primary cost is then read off
the returned flow and used for all public reporting (the perturbation is
internal bookkeeping, never a public objective).

Int64 overflow guard
--------------------

The worker keeps costs, flows and its running total in C ``long long``
(exact int64).  Before the request is allowed to leave this module,
:func:`build_request` verifies with arbitrary-precision Python integers that
neither ``max(perturbed arc cost)`` nor ``S * max(perturbed arc cost)`` (an
upper bound on the total cost of any feasible flow, since no arc carries more
than ``S`` units) exceeds ``INT64_MAX``.  A violating instance is rejected
with :class:`CountRejected` instead of being sent into the worker, where the
arithmetic would wrap around.

Typed failure surface
---------------------

* :class:`CountInfeasible` — solver reports INFEASIBLE for a *representable*
  instance (this should never happen on a TU, balanced instance, but is
  surfaced explicitly rather than as a generic error).
* :class:`InvalidSolution` — solver returned OPTIMAL but the independent
  validation failed (corruption, non-integer flow, conservation violation,
  arc out of range, or cost mismatch with the worker's reported ``total_cost``).
* :class:`NotSolvedUnexpected` — solver reported UNBOUNDED or ERROR.

These are *independent* of :class:`anymesher.quad.count_model.CountRejected`
(which is the "out of scope" rejection channel).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Mapping

from ..errors import MeshError
from .count_model import CountInstance, CountRejected

__all__ = [
    "INT64_MAX",
    "MCF_REQUEST_SCHEMA",
    "MCF_RESPONSE_SCHEMA",
    "MCFEncoding",
    "MCFRequest",
    "MCFResponse",
    "SolveReport",
    "CountInfeasible",
    "InvalidSolution",
    "NotSolvedUnexpected",
    "build_request",
    "tie_value",
    "decode_response",
    "validate_response",
]

MCF_REQUEST_SCHEMA = "anymesher.quad-mcf-request/1"
MCF_RESPONSE_SCHEMA = "anymesher.quad-mcf-response/1"

#: Largest signed 64-bit integer: the worker's arithmetic (costs, flows,
#: running total) is exact ``long long`` and must never wrap.
INT64_MAX: int = (1 << 63) - 1

_STATUS_OPTIMAL = "OPTIMAL"
_STATUS_INFEASIBLE = "INFEASIBLE"
_STATUS_UNBOUNDED = "UNBOUNDED"
_STATUS_ERROR = "ERROR"


class CountInfeasible(MeshError):
    """The solver returned INFEASIBLE for a representable instance.

    This should be unreached on a balanced, TU, isolated-instance, but is
    surfaced explicitly so callers can distinguish "no solution" from "solver
    bug" rather than as a generic ``MeshError``.
    """


class InvalidSolution(MeshError):
    """The solver returned OPTIMAL but the independent validation failed.

    The worker is only *trusted* on the path from "request in" to "status
    out"; every returned integer solution is re-validated on the Python side
    against the original hard constraints (supplies, demands, blocked arcs,
    arc bounds, and the recomputed primary cost).  This failure indicates a
    worker / transport / deserialization bug, not a modelling bug.
    """


class NotSolvedUnexpected(MeshError):
    """The solver reported UNBOUNDED or ERROR.

    Bounded, integer, balanced transportation instances cannot be unbounded;
    ERROR is a transport/lifecycle problem (worker crash, JSON parse error,
    or malformed response) — both are typed rather than a generic raise.
    """


# ---------------------------------------------------------------------------
# Request / response dataclasses (schema-tagged)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCFRequest:
    """Generic min-cost-flow request (worker-side schema 1).

    ``arcs`` is a tuple of ``(u, v, lower, upper, cost)`` ints; ``supply``
    is a per-node int (positive for sources, negative for sinks); the number
    of nodes is implicit from ``nodes``.
    """

    schema: str
    nodes: int
    arcs: tuple[tuple[int, int, int, int, int], ...]
    supply: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema != MCF_REQUEST_SCHEMA:
            raise ValueError("MCFRequest.schema mismatch")
        for a in self.arcs:
            if len(a) != 5:
                raise ValueError("MCFRequest.arcs entry is not a 5-tuple")
        if len(self.supply) != self.nodes:
            raise ValueError("len(supply) must equal nodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "nodes": self.nodes,
            "arcs": [list(a) for a in self.arcs],
            "supply": list(self.supply),
        }


@dataclass(frozen=True)
class MCFResponse:
    """Generic min-cost-flow response (worker-side schema 1).

    ``flows`` is present exactly when ``status == OPTIMAL``; otherwise the
    worker omits it and ``message`` carries the (optional) diagnosis.
    """

    schema: str
    status: str
    total_cost: int = 0
    flows: tuple[int, ...] | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.schema != MCF_RESPONSE_SCHEMA:
            raise ValueError("MCFResponse.schema mismatch")
        if self.status == _STATUS_OPTIMAL:
            if self.flows is None:
                raise ValueError("OPTIMAL response is missing flows")
        else:
            if self.flows is not None:
                raise ValueError("non-OPTIMAL response must not carry flows")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MCFResponse":
        expected = {"schema", "status", "total_cost", "flows", "message"}
        if set(raw) - expected:
            unknown = sorted(set(raw) - expected)
            raise InvalidSolution(f"response has unknown field(s) {unknown}")
        if raw.get("schema") != MCF_RESPONSE_SCHEMA:
            raise InvalidSolution(f"response schema {raw.get('schema')!r} != expected")
        status = raw.get("status")
        if status not in (_STATUS_OPTIMAL, _STATUS_INFEASIBLE, _STATUS_UNBOUNDED, _STATUS_ERROR):
            raise InvalidSolution(f"response status {status!r} is not a known status")
        tc = raw.get("total_cost", 0)
        if isinstance(tc, bool) or not isinstance(tc, Integral):
            raise InvalidSolution(f"total_cost must be an integer (got {tc!r})")
        fl = raw.get("flows")
        if fl is not None:
            if not isinstance(fl, (list, tuple)) or any(
                isinstance(x, bool) or not isinstance(x, Integral) for x in fl
            ):
                raise InvalidSolution("flows must be a list of integers")
        msg = raw.get("message", "")
        if not isinstance(msg, str):
            raise InvalidSolution("message must be a string")
        return cls(
            schema=MCF_RESPONSE_SCHEMA,
            status=str(status),
            total_cost=int(tc),
            flows=tuple(int(x) for x in fl) if fl is not None else None,
            message=str(msg),
        )


# ---------------------------------------------------------------------------
# Reduction + validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCFEncoding:
    """Deterministic tie-break parameters of a built request.

    The perturbed per-arc costs embedded in the request are
    ``primary * primary_scale + tie_weight[k]`` (arc index ``k`` in the
    request's arc order); :func:`tie_value` and the cross-check in
    :func:`validate_response` recompute the *same* quantities from this
    struct so the Python side — not the worker — owns the tie rule.
    """

    S: int
    B: int
    m: int
    primary_scale: int
    tie_weights: tuple[int, ...]


def tie_value(encoding: MCFEncoding, flows: tuple[int, ...]) -> int:
    """Tie contribution ``sum_k f_k * B**(m-1-k)`` of a flow tuple.

    Mirrors the per-arc weights used by :func:`build_request`; valid only for
    ``len(flows) == encoding.m``.
    """
    if len(flows) != encoding.m:
        raise ValueError(
            f"tie_value expects {encoding.m} flows, got {len(flows)}"
        )
    total = 0
    for f, w in zip(flows, encoding.tie_weights):
        total += f * w
    return total


def build_request(instance: CountInstance) -> tuple[MCFRequest, MCFEncoding]:
    """Encode ``instance`` as a unique-optimum MCF request.

    Returns ``(request, encoding)``.  The worker minimizes the perturbed
    objective ``primary * B**m + tie``; the solver is deterministic on the
    uniquely optimal perturbed instance by construction, and the true primary
    cost is recovered from the flow (the worker's self-reported
    ``total_cost`` is only cross-checked against ``primary * B**m + tie``).

    Raises :class:`CountRejected` if the instance is zero-supply (the worker
    would be invoked for nothing) or if the perturbed costs / cost bound would
    exceed the worker's int64 arithmetic (:data:`INT64_MAX`).
    """
    instance.validate()
    s = instance.supplies
    d = instance.demands
    c = instance.cost
    n_in = len(s)
    n_out = len(d)
    S = sum(s)
    if S == 0:
        raise CountRejected("total supply is zero; the solver must not be invoked")

    B = S + 1
    nodes = n_in + n_out
    active: list[tuple[int, int, int]] = []  # (i, j, primary cost)
    for i, si in enumerate(s):
        if si == 0:
            continue
        for j, dj in enumerate(d):
            if dj == 0:
                continue
            if instance.is_block(i, j):
                continue
            active.append((i, j, c[i][j]))
    m = len(active)
    primary_scale = B**m
    tie_weights = tuple(B**(m - 1 - k) for k in range(m))

    arcs: list[tuple[int, int, int, int, int]] = []
    max_perturbed = 0
    for (i, j, cc), w in zip(active, tie_weights):
        cost_pert = cc * primary_scale + w
        if cost_pert > max_perturbed:
            max_perturbed = cost_pert
        arcs.append((i, j + n_in, 0, S, cost_pert))

    if max_perturbed > INT64_MAX:
        raise CountRejected(
            f"perturbed arc cost {max_perturbed} exceeds int64 "
            f"({INT64_MAX}); the instance cannot be posed to the "
            "worker without overflow"
        )
    if S * max_perturbed > INT64_MAX:
        raise CountRejected(
            f"worst-case total cost {S * max_perturbed} exceeds int64 "
            f"({INT64_MAX}); the worker's running total would overflow "
            "before the solver could terminate"
        )

    supply = tuple(int(x) for x in s) + tuple(-int(x) for x in d)
    encoding = MCFEncoding(
        S=S,
        B=B,
        m=m,
        primary_scale=primary_scale,
        tie_weights=tie_weights,
    )
    return MCFRequest(
        schema=MCF_REQUEST_SCHEMA,
        nodes=nodes,
        arcs=tuple(arcs),
        supply=supply,
    ), encoding


def decode_response(raw: Mapping[str, Any]) -> MCFResponse:
    """Parse a raw (already ``json.loads``-ed) response dict into a typed shape.

    This is the *transport* boundary: any structural mismatch is typed as
    :class:`InvalidSolution` so the caller can distinguish "worker lied" from
    "worker is unavailable".
    """
    return MCFResponse.from_dict(raw)


def validate_response(instance: CountInstance, response: MCFResponse, *, encoding: MCFEncoding | None = None) -> SolveReport:
    """Independently validate the worker's OPTIMAL solution.

    Checks (all on the Python side; the worker is only trusted on the
    *transport* path):

    * status is OPTIMAL;
    * ``len(flows)`` equals the request arc count (``i, j`` pairs where
      ``supplies[i] > 0 and demands[j] > 0 and not blocked(i, j)``);
    * every flow is a nonnegative integer within ``[0, sum(supplies)]``;
    * conservation: for every ``i`` with ``s_i > 0``, ``sum_j f_ij == s_i``;
      for every ``j`` with ``d_j > 0``, ``sum_i f_ij == d_j``;
    * blocked arcs carry zero flow (defence in depth);
    * when ``encoding`` is passed (the parameters used by
      :func:`build_request`), the worker's reported ``total_cost`` must equal
      ``primary_scale * primary + tie`` exactly — a byte-level cross-check
      that the returned flows describe the solver's own reported optimum under
      the uniquely tied objective.

    Raises :class:`CountInfeasible` / :class:`InvalidSolution` / :class:`NotSolvedUnexpected`
    on any violation.
    """
    if response.status == _STATUS_INFEASIBLE:
        raise CountInfeasible(
            f"worker reports INFEASIBLE: {response.message or 'no detail'}"
        )
    if response.status == _STATUS_UNBOUNDED:
        raise NotSolvedUnexpected(
            f"worker reports UNBOUNDED: {response.message or 'no detail'}"
        )
    if response.status == _STATUS_ERROR:
        raise NotSolvedUnexpected(
            f"worker reports ERROR: {response.message or 'no detail'}"
        )
    if response.status != _STATUS_OPTIMAL:
        raise NotSolvedUnexpected(f"worker status {response.status!r} is not known")

    if response.flows is None:
        raise InvalidSolution("OPTIMAL response is missing flows")

    n_in = instance.n_in
    n_out = instance.n_out
    s = instance.supplies
    d = instance.demands

    # Build the expected arc list in *the same order* the worker was asked to
    # emit (row-major over (i, j), skipping empty stations and blocked arcs).
    expected: list[tuple[int, int]] = []
    for i, si in enumerate(s):
        if si == 0:
            continue
        for j, dj in enumerate(d):
            if dj == 0:
                continue
            if instance.is_block(i, j):
                continue
            expected.append((i, j))

    if len(response.flows) != len(expected):
        raise InvalidSolution(
            f"flow count {len(response.flows)} != expected arc count {len(expected)}"
        )

    S = sum(s)
    primary = 0
    for ((i, j), f) in zip(expected, response.flows):
        if isinstance(f, bool) or not isinstance(f, int):
            raise InvalidSolution(f"flow({i},{j}) not an integer: {f!r}")
        if f < 0 or f > S:
            raise InvalidSolution(f"flow({i},{j})={f} outside [0, {S}]")
        primary += instance.cost[i][j] * f

    # Conservation per supply / demand station.
    row_sum = [0] * n_in
    col_sum = [0] * n_out
    for (i, j), f in zip(expected, response.flows):
        row_sum[i] += f
        col_sum[j] += f
    for i, si in enumerate(s):
        if si > 0 and row_sum[i] != si:
            raise InvalidSolution(
                f"row conservation: supply[{i}]={si} but flow out={row_sum[i]}"
            )
    for j, dj in enumerate(d):
        if dj > 0 and col_sum[j] != dj:
            raise InvalidSolution(
                f"column conservation: demand[{j}]={dj} but flow in={col_sum[j]}"
            )

    # Cross-check the worker's self-reported total_cost against the perturbed
    # instance (optional, only when the encoding parameters are known).
    if encoding is not None:
        tie = tie_value(encoding, tuple(response.flows))
        expected_worker_total = encoding.primary_scale * primary + tie
        if int(response.total_cost) != expected_worker_total:
            raise InvalidSolution(
                "worker total_cost "
                f"({response.total_cost}) != primary_scale*primary+tie "
                f"({expected_worker_total}); "
                "the returned flows do not describe the solver's own "
                "reported optimum — possible worker / JSON desync"
            )

    return SolveReport(
        status="OPTIMAL",
        flows=tuple(response.flows),
        total_cost=primary,
        message=response.message,
    )


@dataclass(frozen=True)
class SolveReport:
    """Normalized, validated result of a successful solve.

    ``flows`` is aligned with :func:`build_request`'s arc order (the
    canonical row-major, skip-empty, skip-blocked (i, j) order).  The caller
    can re-index via ``instance.unblocked()`` which returns exactly that
    tuple of (i, j) pairs.
    """

    status: str
    flows: tuple[int, ...]
    total_cost: int
    message: str = ""

    def flows_by_index(self, instance: CountInstance) -> dict[tuple[int, int], int]:
        """Return ``{(i, j): f_ij}`` for every used arc, keyed by (i, j)."""
        return dict(zip(instance.unblocked(), self.flows))
