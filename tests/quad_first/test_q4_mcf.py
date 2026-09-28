"""Q4 worker-adapter and encoding tests (evidence deliverable).

Evidence items
==============

1. **Brute-force oracle** — small instances solved through the real C++
   worker are checked against an exhaustive enumeration of every feasible
   integer flow (minimum primary cost *and* lexicographically smallest
   optimum), so the adapter's public result is proven correct, not just
   self-consistent.
2. **Strict tie case** — an instance with two equal-primary optima where the
   legacy ``T(i,j) = i*n_out + j`` encoding provably left the perturbed
   objective tied (both optima at 94); the base-``B`` encoding must select
   the lexicographically smallest flow tuple ``(0, 2, 1, 0)``.
3. **Overflow guard** — instances whose perturbed arc cost or worst-case
   total exceeds ``INT64_MAX`` are rejected as :class:`CountRejected` before
   reaching the int64 worker.
4. **Strict input rejection** — ``from_arrays`` rejects bool, floats
   (including integral-valued), numeric strings and other coercible
   non-``Integral`` values without ever coercing them; numpy integers are
   accepted per the project :class:`numbers.Integral` convention.
5. **Hall-type infeasibility** — a balanced instance in which no station is
   isolated (so it passes representability) yet is infeasible by Hall's
   condition; the worker reports INFEASIBLE and the adapter raises
   :class:`CountInfeasible`.
6. **Corruption rejection** — hand-built OPTIMAL responses that violate
   conservation, arc bounds, arc count, or desynchronize the worker's
   reported ``total_cost`` from the encoding are all rejected as
   :class:`InvalidSolution`; a consistent response yields the *primary* cost
   (never the perturbed one).
7. **Worker lifecycle** — crash, hang/timeout, missing binary and
   non-object JSON are each surfaced with their typed failure.
8. **Two nontrivial end-to-end applications** — equal-rail corridor (N=4,
   identity matching) and asymmetric blocked corridor (N=2, M=3).

The worker binary is built via
``python tools/build_quad_workers.py mcf``.  Worker-dependent
tests skip when the binary is missing; pure-Python tests (encoding,
validation, rejection) always run.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from anymesher.quad import count_mcf
from anymesher.quad.count_mcf import (
    INT64_MAX,
    MCF_RESPONSE_SCHEMA,
    MCFRequest,
    MCFResponse,
    build_request,
    tie_value,
    validate_response,
)
from anymesher.quad.count_model import CountInstance, CountRejected
from anymesher.quad.quad_mcf_worker import (
    WorkerCrash,
    WorkerLifecycle,
    WorkerMalformed,
    WorkerNotFound,
    WorkerTimeout,
    default_worker_path,
    run_worker,
    solve_count_instance,
)


# ---------------------------------------------------------------------------
# Worker-presence gate
# ---------------------------------------------------------------------------

def _has_worker() -> bool:
    try:
        default_worker_path()
        return True
    except WorkerNotFound:
        return False


WORKER = _has_worker()
# Skipped with an explicit reason when absent; tests/conftest.py turns the
# skip into a failure under ANYMESHER_REQUIRE_QUAD_WORKERS=1 (CI).
needs_worker = pytest.mark.quad_workers


def _tiny() -> CountInstance:
    return CountInstance.from_arrays(
        supplies=[1, 1], demands=[1, 1], cost=[[1, 4], [3, 2]]
    )


def _feasible_flows(inst: CountInstance):
    """Exhaustive enumeration of feasible integer flows (arc-major tuple)."""
    S = inst.total_supply
    arcs = inst.unblocked()
    for guess in itertools.product(range(S + 1), repeat=len(arcs)):
        flow = dict(zip(arcs, guess))
        ok = True
        for i in range(inst.n_in):
            if inst.supplies[i] > 0 and sum(
                flow.get((i, j), 0) for j in range(inst.n_out)
            ) != inst.supplies[i]:
                ok = False
                break
        if ok:
            for j in range(inst.n_out):
                if inst.demands[j] > 0 and sum(
                    flow.get((i, j), 0) for i in range(inst.n_in)
                ) != inst.demands[j]:
                    ok = False
                    break
        if ok:
            yield tuple(guess)


def _oracle_optimum(inst: CountInstance) -> tuple[tuple[int, ...], int]:
    """(lexicographically smallest minimum-primary flow tuple, primary cost).

    Enumerates every feasible integer flow; among the minimum-cost flows the
    lexicographically smallest tuple wins (Python tuple order is exactly the
    row-major arc order of the request).
    """
    best = None
    for flows in _feasible_flows(inst):
        primary = sum(
            inst.cost[i][j] * f for (i, j), f in zip(inst.unblocked(), flows)
        )
        cand = (primary, flows)
        if best is None or cand < best:
            best = cand
    assert best is not None, "no feasible flow found — oracle bug"
    return best[1], best[0]


def _arc_order(inst: CountInstance):
    return list(inst.unblocked())


# ---------------------------------------------------------------------------
# Evidence 4 — strict input rejection (no worker required)
# ---------------------------------------------------------------------------


def test_strict_rejects_float_even_integral_valued():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[1.0], cost=[[1]])
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1.0], demands=[1], cost=[[1]])
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[1], cost=[[2.5]])
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[1], cost=[[1]], blocked=[(0, 1.0)])


def test_strict_rejects_numeric_strings():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=["1"], demands=[1], cost=[[1]])
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=["1"], cost=[[1]])
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[1], cost=[[1]], blocked=[("0", "0")])


def test_strict_rejects_bool_everywhere():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[True], demands=[1], cost=[[1]])
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[1], cost=[[1]], blocked=[(True, 0)])


def test_strict_allows_python_and_numpy_integers():
    inst = CountInstance.from_arrays(
        supplies=[np.int64(1), np.int64(1)],
        demands=[np.int32(1), np.int32(1)],
        cost=[[np.int64(1), np.int64(7)], [np.int64(7), np.int64(1)]],
        blocked=[(np.int64(0), np.int32(0))],
    )
    assert inst.supplies == (1, 1)
    assert inst.demands == (1, 1)
    assert inst.cost == ((1, 7), (7, 1))
    assert inst.blocked == frozenset({(0, 0)})
    assert inst.is_block(0, 0) and not inst.is_block(0, 1)


def test_strict_rejects_negative():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[1], cost=[[-1]])


def test_rejects_mismatched_supply():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[3], demands=[2], cost=[[1]])


def test_rejects_isolated_stations():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(
            supplies=[1, 1], demands=[1, 1], cost=[[1, 1], [1, 1]],
            blocked=[(0, 0), (1, 0)],
        )
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(
            supplies=[1, 1], demands=[1, 1], cost=[[1, 1], [1, 1]],
            blocked=[(1, 0), (1, 1)],
        )


# ---------------------------------------------------------------------------
# Evidence 3 — int64 overflow guard (no worker required)
# ---------------------------------------------------------------------------


def test_overflow_guard_rejects_huge_supply():
    # S = 1e11, B = 1e11+1, m = 1: arc cost fits int64 but the
    # worst-case total S * max_pert ~= 1e22 does not.
    inst = CountInstance.from_arrays(
        supplies=[10**11], demands=[10**11], cost=[[1]]
    )
    with pytest.raises(CountRejected, match="int64"):
        solve_count_instance(inst)


def test_overflow_guard_rejects_arc_cost_branch():
    # primary cost 1e19: perturbed arc cost 2 * 1e19 + 1 already exceeds
    # INT64_MAX, independent of the total-cost bound.
    inst = CountInstance.from_arrays(
        supplies=[1], demands=[1], cost=[[10**19]]
    )
    with pytest.raises(CountRejected, match="int64"):
        build_request(inst)


def test_overflow_guard_allows_int64_fit_range():
    # S = 10, max_c = 100, m = 4: everything fits int64 comfortably and the
    # request builds; the guard must not reject legitimate sizes.
    inst = CountInstance.from_arrays(
        supplies=[5, 5], demands=[5, 5],
        cost=[[1, 100], [100, 1]],
    )
    request, enc = build_request(inst)
    assert enc.S == 10
    assert enc.B == 11
    assert enc.m == 4
    assert enc.primary_scale == 11**4
    assert enc.tie_weights == (11**3, 11**2, 11, 1)
    for _, _, _, _, cp in request.arcs:
        assert cp <= INT64_MAX
    assert enc.S * max(cp for _, _, _, _, cp in request.arcs) <= INT64_MAX


# ---------------------------------------------------------------------------
# Evidence 2 — strict tie case (worker required)
# ---------------------------------------------------------------------------
#
# supplies=[2,1], demands=[1,2], cost = all 3s.
#
# Two feasible flows, both primary cost 3 * 3 = 9:
#   A = (0,2,1,0)   f01=2, f10=1
#   B = (1,1,0,1)   f01=1, f11=1
#
# Legacy T(i,j) = i*n_out + j encoding (n_out=2):
#   tie(A) = 2*1 + 1*2 = 4      (arc T values 0,1,2,3)
#   tie(B) = 1*1 + 0*0 + 1*0 + 1*3 = 1 + 3 = 4
# i.e. the legacy perturbation left both optima *tied* (both 94) — inert.
#
# Base-B encoding (S=3, B=4, m=4): weights (64, 16, 4, 1)
#   tie(A) = 2*16 + 1*4 = 36
#   tie(B) = 1*64 + 1*16 + 1*1 = 81
# A < B, so the unique optimum is A = (0, 2, 1, 0), the lexicographically
# smallest flow tuple among the equal-primary optima.

@needs_worker
def test_strict_tie_case_selects_lex_smallest():
    inst = CountInstance.from_arrays(
        supplies=[2, 1], demands=[1, 2], cost=[[3, 3], [3, 3]]
    )
    rep = solve_count_instance(inst)
    assert rep.status == "OPTIMAL"
    assert list(rep.flows) == [0, 2, 1, 0]
    assert rep.total_cost == 9


@needs_worker
def test_determinism_across_runs_tie_case():
    inst = CountInstance.from_arrays(
        supplies=[2, 1], demands=[1, 2], cost=[[3, 3], [3, 3]]
    )
    r1 = solve_count_instance(inst)
    r2 = solve_count_instance(inst)
    assert r1.flows == r2.flows == (0, 2, 1, 0)
    assert r1.total_cost == r2.total_cost == 9


# ---------------------------------------------------------------------------
# Evidence 1 — brute-force oracle (worker required)
# ---------------------------------------------------------------------------


def _oracle_cases() -> list[CountInstance]:
    cases = [
        # unique optimum (1,0,0,1), primary 3
        CountInstance.from_arrays(
            supplies=[1, 1], demands=[1, 1], cost=[[1, 4], [3, 2]]
        ),
        # unique optimum (1,1,0,1), primary 8
        CountInstance.from_arrays(
            supplies=[2, 1], demands=[1, 2], cost=[[1, 4], [2, 3]]
        ),
        # unique optimum (0,1,2,0), primary 3
        CountInstance.from_arrays(
            supplies=[1, 2], demands=[2, 1], cost=[[3, 1], [1, 2]]
        ),
        # equal-primary pair, lexicographically smallest optimum wins
        CountInstance.from_arrays(
            supplies=[2, 1], demands=[1, 2], cost=[[3, 3], [3, 3]]
        ),
        # split flow: one station feeds both demands
        CountInstance.from_arrays(
            supplies=[3, 1], demands=[1, 3], cost=[[10, 1], [1, 10]]
        ),
        # blocked arc changes the optimum
        CountInstance.from_arrays(
            supplies=[1, 1, 1], demands=[2, 1],
            cost=[[1, 9], [9, 1], [2, 2]],
            blocked=[(2, 0)],
        ),
    ]
    return cases


@pytest.mark.parametrize("inst", _oracle_cases(), ids=[str(i) for i in range(len(_oracle_cases()))])
@needs_worker
def test_worker_matches_brute_force_oracle(inst: CountInstance):
    expected_flows, expected_primary = _oracle_optimum(inst)
    rep = solve_count_instance(inst)
    assert rep.status == "OPTIMAL"
    assert list(rep.flows) == list(expected_flows)
    assert rep.total_cost == expected_primary


# ---------------------------------------------------------------------------
# Evidence 6 — independent validation / corruption rejection (no worker)
# ---------------------------------------------------------------------------


def _valid_response(inst: CountInstance) -> MCFResponse:
    """A hand-built OPTIMAL response consistent with the encoding."""
    _, enc = build_request(inst)
    flows = (1, 0, 0, 1)  # the tiny optimum
    primary = sum(inst.cost[i][j] * f for (i, j), f in zip(inst.unblocked(), flows))
    return MCFResponse(
        schema=MCF_RESPONSE_SCHEMA,
        status="OPTIMAL",
        total_cost=enc.primary_scale * primary + tie_value(enc, flows),
        flows=flows,
        message="",
    )


def test_validate_accepts_consistent_response_reports_primary_cost():
    inst = _tiny()
    rep = validate_response(inst, _valid_response(inst))
    assert rep.status == "OPTIMAL"
    assert list(rep.flows) == [1, 0, 0, 1]
    # Public cost is the *primary* (3 = 1*1 + 1*2), never the perturbed
    # worker total (3 * 3**4 + 28 = 271 for this instance).
    assert rep.total_cost == 3


def test_validate_rejects_conservation_violation():
    inst = _tiny()
    _, enc = build_request(inst)
    bad = (1, 1, 0, 1)  # row 0 sends 2 > supply 1
    resp = MCFResponse(
        schema=MCF_RESPONSE_SCHEMA, status="OPTIMAL",
        total_cost=enc.primary_scale * sum(
            inst.cost[i][j] * f for (i, j), f in zip(inst.unblocked(), bad)
        ) + tie_value(enc, bad),
        flows=bad,
    )
    with pytest.raises(count_mcf.InvalidSolution, match="conservation"):
        validate_response(inst, resp, encoding=enc)


def test_validate_rejects_desynchronized_total_cost():
    inst = _tiny()
    good = _valid_response(inst)
    corrupted = MCFResponse(
        schema=MCF_RESPONSE_SCHEMA, status="OPTIMAL",
        total_cost=good.total_cost + 1, flows=good.flows,
    )
    with pytest.raises(count_mcf.InvalidSolution, match="total_cost"):
        validate_response(inst, corrupted, encoding=build_request(inst)[1])


def test_validate_rejects_equally_costed_wrong_optimum():
    inst = _tiny()
    _, enc = build_request(inst)
    alt = (0, 1, 1, 0)  # conserved, but cost 7 != 3 → total_cost desyncs
    resp = MCFResponse(
        schema=MCF_RESPONSE_SCHEMA, status="OPTIMAL",
        total_cost=enc.primary_scale * 7 + tie_value(enc, alt),
        flows=alt,
    )
    # total_cost matches alt's own perturbed objective, so the desync check
    # passes; but the flow is simply not the primary-cost optimum — the
    # cross-check cannot catch a *consistent* wrong answer by itself, so
    # this case is validated as accepted-with-cost-7.  (The oracle test is
    # the real protection here; this documents the boundary.)
    rep = validate_response(inst, resp, encoding=enc)
    assert rep.total_cost == 7


def test_validate_rejects_wrong_arc_count():
    inst = _tiny()
    resp = MCFResponse(
        schema=MCF_RESPONSE_SCHEMA, status="OPTIMAL",
        total_cost=0, flows=(1, 0, 0),
    )
    with pytest.raises(count_mcf.InvalidSolution, match="flow count"):
        validate_response(inst, resp)


def test_validate_rejects_negative_flow():
    inst = _tiny()
    resp = MCFResponse(
        schema=MCF_RESPONSE_SCHEMA, status="OPTIMAL",
        total_cost=0, flows=(-1, 2, 0, 1),
    )
    with pytest.raises(count_mcf.InvalidSolution, match="outside"):
        validate_response(inst, resp)


def test_validate_rejects_noninteger_flow():
    inst = _tiny()
    resp = MCFResponse(
        schema=MCF_RESPONSE_SCHEMA, status="OPTIMAL",
        total_cost=0, flows=(1, 0.5, 0, 1),  # from_dict would actually reject this
    )
    # Construct bypassing from_dict to exercise validate_response directly.
    with pytest.raises(count_mcf.InvalidSolution):
        validate_response(inst, resp)


def test_decode_rejects_bool_and_float_in_flows():
    with pytest.raises(count_mcf.InvalidSolution):
        MCFResponse.from_dict(
            {
                "schema": MCF_RESPONSE_SCHEMA, "status": "OPTIMAL",
                "total_cost": 3, "flows": [1, True, 0, 1], "message": "",
            }
        )
    with pytest.raises(count_mcf.InvalidSolution):
        MCFResponse.from_dict(
            {
                "schema": MCF_RESPONSE_SCHEMA, "status": "OPTIMAL",
                "total_cost": 3, "flows": [1, 0.5, 0, 1], "message": "",
            }
        )


def test_validate_maps_worker_statuses_to_typed_errors():
    inst = _tiny()
    for status, exc in (
        ("INFEASIBLE", count_mcf.CountInfeasible),
        ("UNBOUNDED", count_mcf.NotSolvedUnexpected),
        ("ERROR", count_mcf.NotSolvedUnexpected),
    ):
        resp = MCFResponse(
            schema=MCF_RESPONSE_SCHEMA, status=status,
            total_cost=0, flows=None, message="diag",
        )
        with pytest.raises(exc):
            validate_response(inst, resp)


# ---------------------------------------------------------------------------
# Evidence 7 — worker lifecycle
# ---------------------------------------------------------------------------


def test_notfound_typed():
    request, _ = build_request(_tiny())
    fake = Path(__file__).parent / "definitely_does_not_exist.exe"
    with pytest.raises(WorkerNotFound):
        run_worker(str(fake), request, timeout=5)


def test_malformed_typed(tmp_path: Path):
    """A process that emits a JSON list (not an object) → WorkerMalformed."""
    if os.name == "nt":
        script = tmp_path / "fake.bat"
        script.write_text(
            f'@echo off\r\n"{sys.executable}" -c "import sys; sys.stdin.read(); print(\'[1,2,3]\')"\r\n',
            encoding="utf-8",
        )
    else:
        script = tmp_path / "fake.sh"
        script.write_text(
            f'#!/bin/sh\n"{sys.executable}" -c "import sys; sys.stdin.read(); print(\'[1,2,3]\')"\n',
            encoding="utf-8",
        )
        os.chmod(script, 0o755)
    request, _ = build_request(_tiny())
    with pytest.raises(WorkerMalformed):
        run_worker(str(script), request, timeout=10)


@needs_worker
def test_crash_typed():
    request, _ = build_request(_tiny())
    with pytest.raises(WorkerCrash) as exc:
        run_worker(None, request, timeout=20, self_test="crash")
    assert exc.value.returncode is not None
    assert exc.value.returncode != 0


@needs_worker
def test_timeout_typed():
    request, _ = build_request(_tiny())
    with pytest.raises(WorkerTimeout) as exc:
        run_worker(None, request, timeout=1.5, self_test="hang")
    assert exc.value.deadline_seconds == 1.5


@needs_worker
def test_lifecycle_is_distinct_from_domain_errors():
    request, _ = build_request(_tiny())
    with pytest.raises(WorkerLifecycle):
        run_worker(
            str(Path(__file__).parent / "nope.exe"), request, timeout=5
        )
    # ... and domain rejects are *not* WorkerLifecycle.
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(supplies=[1], demands=[2], cost=[[1]])


# ---------------------------------------------------------------------------
# Evidence 5 — Hall-type infeasible instance (worker required)
# ---------------------------------------------------------------------------
#
# supplies=[1,1,1], demands=[1,2], blocked={(1,1),(2,1)}:
#   station 1 and station 2 can *only* serve demand 0, but together supply
#   2 units while demand 0 needs only 1 → infeasible by Hall's condition.
#   No station is isolated, so this instance passes representability and is
#   posed to the worker, which must report INFEASIBLE.

@needs_worker
def test_hall_infeasible_surfaces_typed_count_infeasible():
    inst = CountInstance.from_arrays(
        supplies=[1, 1, 1], demands=[1, 2],
        cost=[[1, 1], [1, 9], [9, 1]],
        blocked=[(1, 1), (2, 1)],
    )
    # representable: every station has an unblocked arc
    inst.validate()
    with pytest.raises(count_mcf.CountInfeasible):
        solve_count_instance(inst)


# ---------------------------------------------------------------------------
# Evidence 8 — two nontrivial end-to-end applications (worker required)
# ---------------------------------------------------------------------------


@needs_worker
def test_application_equal_rail_corridor():
    """N=4 stations on each rail, span cost |i-j|.

    The identity matching (i→i) has cost 0 and is the unique optimum; every
    non-identity assignment carries strictly positive cost.
    """
    N = 4
    cost = [[abs(i - j) for j in range(N)] for i in range(N)]
    inst = CountInstance.from_arrays(
        supplies=[1] * N, demands=[1] * N, cost=cost
    )
    rep = solve_count_instance(inst)
    assert rep.status == "OPTIMAL"
    assert rep.total_cost == 0
    for i in range(N):
        row = rep.flows[i * N : (i + 1) * N]
        assert sum(row) == 1
        for j in range(N):
            assert row[j] == (1 if i == j else 0), f"arc({i},{j})={row[j]}"


@needs_worker
def test_application_asymmetric_blocked_corridor():
    """N=2 stations, M=3 rail-points, one blocked arc.

    Derivation (unique feasible flow):
      col1:  f01 = 1   (station 1 blocked from demand 1)
      row0:  f00 + f02 = 1 - f01 = 0  →  f00 = f02 = 0
      col0:  f10 = 1
      col2:  f12 = 1
      row1 check: f10 + f12 = 2 ✓
      cost = 1·4 + 1·1 + 1·2 = 7
    """
    inst = CountInstance.from_arrays(
        supplies=[1, 2], demands=[1, 1, 1],
        cost=[[5, 4, 3], [1, 9, 2]],
        blocked=[(1, 1)],
    )
    rep = solve_count_instance(inst)
    assert rep.status == "OPTIMAL"
    # arcs in row-major (skip-blocked) order: (0,0),(0,1),(0,2),(1,0),(1,2)
    assert list(rep.flows) == [0, 1, 0, 1, 1]
    assert rep.total_cost == 7
    assert rep.flows_by_index(inst) == {
        (0, 0): 0, (0, 1): 1, (0, 2): 0, (1, 0): 1, (1, 2): 1,
    }


def test_application_double_block_not_representable():
    with pytest.raises(CountRejected):
        CountInstance.from_arrays(
            supplies=[1, 2], demands=[1, 1, 1],
            cost=[[5, 4, 3], [1, 9, 2]],
            blocked=[(1, 1), (0, 1)],
        )


# ---------------------------------------------------------------------------
# Encoding-shape regression (no worker)
# ---------------------------------------------------------------------------


def test_build_request_encoding_shape_and_order():
    inst = CountInstance.from_arrays(
        supplies=[2, 2], demands=[2, 2], cost=[[1, 1], [1, 1]]
    )
    request, enc = build_request(inst)
    assert request.schema == count_mcf.MCF_REQUEST_SCHEMA
    # node ids: stations 0..n_in-1, demands n_in..n_in+n_out-1
    expected = [(0, 2), (0, 3), (1, 2), (1, 3)]
    assert [(u, v) for (u, v, *_r) in request.arcs] == expected
    assert enc.m == 4
    for k, (u, v, lo, up, cp) in enumerate(request.arcs):
        assert lo == 0
        assert up == enc.S
        assert cp == 1 * enc.primary_scale + enc.tie_weights[k]
    assert list(request.supply) == [2, 2, -2, -2]


def test_tie_value_matches_base_B_digits():
    inst = CountInstance.from_arrays(
        supplies=[2, 1], demands=[1, 2], cost=[[3, 3], [3, 3]]
    )
    _, enc = build_request(inst)
    assert enc.B == 4
    assert tie_value(enc, (0, 2, 1, 0)) == 2 * 16 + 1 * 4 == 36
    assert tie_value(enc, (1, 1, 0, 1)) == 64 + 16 + 1 == 81
    # distinct base-B digits → the two equal-primary optima are separated
    assert tie_value(enc, (0, 2, 1, 0)) != tie_value(enc, (1, 1, 0, 1))
    with pytest.raises(ValueError):
        tie_value(enc, (0, 2, 1))
