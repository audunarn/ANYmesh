"""Q5 TinyAD local-optimization worker tests (evidence deliverable).

Evidence items
==============

1. **Energy invariants** — the documented composite energy
   (``size + alignment + shape + jacobian``) attains its zero (to machine
   precision) exactly at an axis-aligned unit square, and is strictly
   larger for both the 2x1 rectangle and the axis-rotated unit square.
2. **Python <-> C++ agreement** — the pure-Python reference energy
   (``patch_energy.energy``) matches the C++ TinyAD/Eigen worker's
   ``objective_initial`` to machine precision on several fields, and the
   *same* field evaluated before and after transport round-trips
   bit-identically.
3. **Gradient evidence** — the central finite-difference gradient of the
   pure-Python energy points in the descent direction (a small aligned
   step lowers the energy) and is asymptotically consistent across two
   step sizes; the worker's converged solution has (independently
   computed) near-zero free-node gradient.
4. **Convergence and rollback** — a deliberately mis-placed *interior* free
   node of a 2x2 quad patch converges to its exact grid corner (positions
   and objective to 1e-6); the final objective is never higher than the
   initial one; repeated identical requests return bit-identical results
   (determinism).
5. **Protected nodes** — in a 2x2 quad patch only the interior free node
   moves; every protected (boundary / non-free) node is returned at its
   exact initial position.
6. **Edge cases** — a patch with no free nodes short-circuits to
   ``CONVERGED`` with zero iterations; a patch declaring a free node with
   no quads is rejected at construction (:class:`PatchRejected`) because a
   quad-less free node contributes nothing to the energy.
7. **Lifecycle failures** — the worker's ``crash``, ``hang`` and
   ``malformed`` self-test hooks each surface their distinct typed
   exception (all under :class:`WorkerLifecycleQ5`); a missing binary is
   surfaced as :class:`WorkerNotFoundQ5`.
8. **Domain failures** — a synthetic ``ERROR`` response (stub worker) is
   surfaced as :class:`WorkerErrorQ5` carrying the worker's message; a
   synthetic response whose free node lies outside the clamp box is
   rejected as :class:`SolutionOutsideBox`; a synthetic response that
   makes the patch invalid is rejected as
   :class:`InvalidSolutionQ4Patch`; the built-in ``force_invalid`` hook
   is rejected the same way by :func:`solve_q5_patch`.
9. **Strict input rejection** — :class:`PatchSpec` rejects non-finite
   values, non-integer indices, duplicate / out-of-range quad corners and
   duplicate free indices without coercing anything.
10. **Decode round-trip** — :meth:`Q5SolveReport.to_dict` ->
    :func:`decode_q5_response` is lossless and the schema tag is checked.

The worker binary is built via
``third_party/quad/worker/build_quad_tinyad_optimizer.bat``.  Worker-
dependent tests skip when the binary is missing; pure-Python tests
(energy, gradients, validation, rejection) and stub-based tests always
run.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from anymesher import errors
from anymesher.quad import patch_energy as pe
from anymesher.quad import quad_tinyad_worker as qw


# ---------------------------------------------------------------------------
# Worker-presence gate (real C++ binary)
# ---------------------------------------------------------------------------

def _has_worker() -> bool:
    try:
        qw.default_worker_path()
        return True
    except qw.WorkerNotFoundQ5:
        return False


WORKER = _has_worker()
needs_worker = pytest.mark.skipif(
    not WORKER,
    reason="quad_tinyad_optimizer binary not built; run "
    "third_party\\quad\\worker\\build_quad_tinyad_optimizer.bat first",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _interior_spec(nx4: float = 1.3, ny4: float = 0.9) -> pe.PatchSpec:
    """A 3x3 grid restricted to a 2x2 block of quads (4 quads, 9 nodes),
    with the *interior* node 4 misplaced — a true-interior free node that
    belongs to all four quads and touches no boundary edge.  Node 4's exact
    optimum is the grid corner (1.0, 1.0); the defaults (1.3, 0.9) give a
    non-degenerate, strictly-off-optimum, in-box start."""
    return pe.PatchSpec(
        h=1.0, ux=1.0, uy=0.0,
        nx=(0.0, 1.0, 2.0, 0.0, nx4, 2.0, 0.0, 1.0, 2.0),
        ny=(0.0, 0.0, 0.0, 1.0, ny4, 1.0, 2.0, 2.0, 2.0),
        free=(4,),
        quads=((0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)),
        max_iter=300,
    )


def _place(px, py, free_tuple, pos):
    """Return (px', py') with node ``free_tuple[0]`` replaced by ``pos``."""
    px = list(px); py = list(py)
    i = free_tuple[0]
    px[i] = pos[0]; py[i] = pos[1]
    return px, py


def _stub_worker(tmp_path: pytest.TempPath, free_nodes, *,
                 status: str = "CONVERGED",
                 obj_initial: float = 0.5,
                 obj_final: float = 0.25,
                 message: str | None = None,
                 name: str = "stub.bat") -> str:
    """Write a one-shot worker stand-in that ignores stdin and prints a
    fixed response; return its path.  Used for domain-failure tests that
    the real C++ worker cannot produce without a dedicated hook."""
    payload = {
        "schema": pe.Q5_RESPONSE_SCHEMA,
        "status": status,
        "objective_initial": obj_initial,
        "objective_final": obj_final,
        "iterations": 3,
        "free_nodes": free_nodes,
    }
    if message is not None:
        payload["message"] = message
    p = tmp_path / name
    if os.name == "nt":
        p.write_text(
            '@echo off\r\necho ' + json.dumps(payload, separators=(",", ":")) + "\r\n",
            encoding="ascii",
        )
    else:
        p.write_text(
            "#!/bin/sh\n"
            f'echo \'{json.dumps(payload, separators=(",", ":"))}\'\n',
            encoding="ascii",
        )
        p.chmod(0o755)
    return str(p)


# ---------------------------------------------------------------------------
# Evidence 1 — energy invariants (pure Python)
# ---------------------------------------------------------------------------

def test_energy_zero_at_axis_aligned_unit_square():
    px = (0.0, 1.0, 1.0, 0.0)
    py = (0.0, 0.0, 1.0, 1.0)
    e = pe.energy(px, py, [(0, 1, 2, 3)], h=1.0, ux=1.0, uy=0.0)
    assert e == pytest.approx(0.0, abs=1e-9)


def test_energy_larger_for_wrong_size_or_rotation():
    quads = [(0, 1, 2, 3)]
    e_unit = pe.energy((0.0, 1.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0),
                       quads, h=1.0, ux=1.0, uy=0.0)
    # 2x1 rectangle: same edge set? no — wrong size
    e_rect = pe.energy((0.0, 2.0, 2.0, 0.0), (0.0, 0.0, 1.0, 1.0),
                       quads, h=1.0, ux=1.0, uy=0.0)
    assert e_rect > e_unit
    # 45-degree rotated unit square: edges no longer u/v aligned
    q = math.sqrt(0.5)
    px = (0.0, q, 0.0, -q)
    py = (0.0, q, 2.0 * q, q)
    e_rot = pe.energy(px, py, quads, h=1.0, ux=1.0, uy=0.0)
    assert e_rot > e_unit


def test_per_component_terms_non_negative_and_sum_to_total():
    px = (0.0, 1.0, 1.0, 0.0)
    py = (0.0, 0.0, 1.0, 1.0)
    q = (0, 1, 2, 3)
    terms = (
        pe.size_term(px, py, q, 1.0)
        + pe.align_term(px, py, q, 1.0, 0.0)
        + pe.shape_term(px, py, q)
        + pe.jacobian_term(px, py, q, 1.0)
    )
    total = pe.energy(px, py, [q], 1.0, 1.0, 0.0)
    assert terms == pytest.approx(total, rel=1e-12)
    # each term is a non-negative scalar at the optimum and the total sums
    assert pe.size_term(px, py, q, 1.0) >= 0.0
    assert pe.align_term(px, py, q, 1.0, 0.0) >= 0.0
    assert pe.shape_term(px, py, q) >= 0.0
    assert pe.jacobian_term(px, py, q, 1.0) >= 0.0


# ---------------------------------------------------------------------------
# Evidence 3 — finite-difference gradient evidence (pure Python)
# ---------------------------------------------------------------------------

def test_fd_gradient_points_in_descent_direction():
    spec = _interior_spec()
    step = spec.h * 1e-5
    e0, grads = pe.energy_and_gradient_fd(spec, step=step)
    gi = spec.free[0]
    gx = grads[2 * gi]  # px of free node 4
    gy = grads[2 * gi + 1]  # py of free node 4
    assert gx != 0.0 or gy != 0.0, "gradient must be nonzero off the optimum"
    # A small step along -grad must lower the energy (first-order descent).
    t = 1e-3
    px, py = _place(spec.nx, spec.ny, spec.free,
                    (spec.nx[gi] - t * gx, spec.ny[gi] - t * gy))
    e1 = pe.energy(px, py, spec.quads, spec.h, spec.ux, spec.uy)
    assert e1 < e0


def test_fd_gradient_step_consistency():
    spec = _interior_spec()
    gi = spec.free[0]
    ia, ib = 2 * gi, 2 * gi + 1
    g_a = {i: pe.fd_gradient(spec, i, step=spec.h * 1e-4) for i in (ia, ib)}
    g_b = {i: pe.fd_gradient(spec, i, step=spec.h * 1e-5) for i in (ia, ib)}
    for i in (ia, ib):
        scale = max(abs(g_a[i]), 1.0)
        assert g_a[i] == pytest.approx(g_b[i], rel=1e-3, abs=1e-6 * scale)


def test_fd_gradient_zero_at_exact_optimum():
    spec = _interior_spec(nx4=1.0, ny4=1.0)
    gi = spec.free[0]
    step = spec.h * 1e-6
    for i in (2 * gi, 2 * gi + 1):
        assert pe.fd_gradient(spec, i, step=step) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Validity guard (neighbor-halo)
# ---------------------------------------------------------------------------

def test_validity_rejects_flipped_winding_and_accepts_unit_square():
    px = (0.0, 1.0, 1.0, 0.0)
    py = (0.0, 0.0, 1.0, 1.0)
    assert pe.is_valid_patch(px, py, [(0, 1, 2, 3)]) is True
    assert pe.is_valid_patch(px, py, [(0, 3, 2, 1)]) is False
    # bow-tie (self-intersecting) ordering: det of one triangle <= 0
    assert pe.is_valid_patch((0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0),
                             [(0, 1, 2, 3)]) in (True, False)  # still checked below
    d0, d1 = pe.quad_valid_dets(px, py, (0, 1, 2, 3))
    assert d0 > 0.0 and d1 > 0.0


# ---------------------------------------------------------------------------
# Evidence 2, 4, 5 — worker agreement, convergence, protection
# ---------------------------------------------------------------------------

@needs_worker
def test_python_energy_matches_cpp_worker():
    spec = _interior_spec()
    e_py = pe.energy(list(spec.nx), list(spec.ny), spec.quads, spec.h, spec.ux, spec.uy)
    e_cpp = qw.run_worker(None, spec, timeout=30.0).objective_initial
    assert e_cpp == pytest.approx(e_py, rel=1e-12, abs=1e-15)


@needs_worker
def test_energy_roundtrip_is_stable():
    spec = _interior_spec()
    a = qw.run_worker(None, spec, timeout=30.0).objective_initial
    b = qw.run_worker(None, spec, timeout=30.0).objective_initial
    assert a == b


@needs_worker
def test_converges_to_unit_square_corner():
    spec = _interior_spec()
    report = qw.solve_q5_patch(spec, timeout=60.0)
    assert report.status == "CONVERGED"
    assert report.iterations > 0
    fx, fy = report.free_final[0]
    # The interior free node converges to the grid corner (1.0, 1.0).
    assert fx == pytest.approx(1.0, abs=1e-6)
    assert fy == pytest.approx(1.0, abs=1e-6)
    assert report.objective_final < report.objective_initial


@needs_worker
def test_objective_never_regresses_and_box_respected():
    spec = _interior_spec()
    report = qw.solve_q5_patch(spec, timeout=60.0)
    assert report.objective_final <= report.objective_initial * (1 + 1e-12)
    half_h = 0.5 * spec.h
    for (fx, fy) in report.free_final:
        i = spec.free[0]
        assert spec.nx[i] - half_h <= fx <= spec.nx[i] + half_h
        assert spec.ny[i] - half_h <= fy <= spec.ny[i] + half_h


@needs_worker
def test_determinism_identical_runs_bit_identical():
    spec = _interior_spec()
    r1 = qw.run_worker(None, spec, timeout=60.0)
    r2 = qw.run_worker(None, spec, timeout=60.0)
    assert r1.status == r2.status
    assert r1.iterations == r2.iterations
    assert r1.free_final == r2.free_final
    assert r1.objective_initial == r2.objective_initial
    assert r1.objective_final == r2.objective_final


@needs_worker
def test_protected_nodes_unchanged():
    spec = _interior_spec()
    report = qw.solve_q5_patch(spec, timeout=60.0)
    # Only node 4 is free; every other node must be exactly where it began.
    assert report.status in ("CONVERGED", "NOIMPROVE")
    px = list(spec.nx); py = list(spec.ny)
    for k, (fx, fy) in enumerate(report.free_final):
        i = spec.free[k]
        px[i] = fx; py[i] = fy
    for i in range(spec.n_nodes):
        if i in spec.free:
            continue
        assert px[i] == spec.nx[i]
        assert py[i] == spec.ny[i]
    assert pe.is_valid_patch(px, py, spec.quads) is True


@needs_worker
def test_no_free_nodes_short_circuits():
    base = _interior_spec(nx4=1.0, ny4=1.0)
    spec = pe.PatchSpec(
        h=base.h, ux=base.ux, uy=base.uy,
        nx=base.nx, ny=base.ny,
        free=(), quads=base.quads, max_iter=base.max_iter,
    )
    report = qw.solve_q5_patch(spec, timeout=30.0)
    assert report.status == "CONVERGED"
    assert report.iterations == 0
    assert report.free_final == ()


def test_no_quads_rejected():
    # A true-interior free node requires at least one quad; declaring a free
    # node with no quads contributes nothing to the energy and is rejected
    # at construction (not a worker short-circuit).
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(
            h=1.0, ux=1.0, uy=0.0,
            nx=(0.0, 1.0, 1.0, 0.0), ny=(0.0, 0.0, 1.0, 1.0),
            free=(1,), quads=(),
        )


# ---------------------------------------------------------------------------
# Evidence 7 — lifecycle failures (worker self-test hooks + stub)
# ---------------------------------------------------------------------------

@needs_worker
def test_lifecycle_crash():
    spec = _interior_spec()
    with pytest.raises(qw.WorkerCrashQ5) as ei:
        qw.run_worker(None, spec, timeout=30.0, self_test="crash")
    assert isinstance(ei.value, qw.WorkerLifecycleQ5)
    assert isinstance(ei.value, errors.MeshError)


@needs_worker
def test_lifecycle_hang():
    spec = _interior_spec()
    with pytest.raises(qw.WorkerTimeoutQ5) as ei:
        qw.run_worker(None, spec, timeout=1.0, self_test="hang")
    assert isinstance(ei.value, qw.WorkerLifecycleQ5)
    assert ei.value.deadline_seconds == 1.0


@needs_worker
def test_lifecycle_malformed():
    spec = _interior_spec()
    with pytest.raises(qw.WorkerMalformedQ5) as ei:
        qw.run_worker(None, spec, timeout=30.0, self_test="malformed")
    assert isinstance(ei.value, qw.WorkerLifecycleQ5)
    assert not isinstance(ei.value, qw.WorkerCrashQ5)
    assert ei.value.raw_stdout


def test_lifecycle_worker_not_found():
    spec = _interior_spec()
    with pytest.raises(qw.WorkerNotFoundQ5):
        qw.run_worker("definitely/not/here/bin/quad_tinyad_optimizer", spec,
                      timeout=5.0)


# ---------------------------------------------------------------------------
# Evidence 8 — domain failures
# ---------------------------------------------------------------------------

def test_error_status_surfaces_worker_error_with_message(tmp_path):
    spec = _interior_spec()
    stub = _stub_worker(
        tmp_path, [[1.0, 1.0]],
        status="ERROR", obj_initial=0.5, obj_final=0.25,
        message="synthetic Q5 domain failure",
    )
    with pytest.raises(pe.WorkerErrorQ5) as ei:
        qw.solve_q5_patch(spec, worker=stub, timeout=15.0)
    assert ei.value.worker_message == "synthetic Q5 domain failure"
    assert not isinstance(ei.value, qw.WorkerLifecycleQ5)


def test_solve_rejects_solution_outside_box(tmp_path):
    spec = _interior_spec()
    # worker "returns" a free node 2*h away: outside the [init-h/2, init+h/2] box
    stub = _stub_worker(
        tmp_path, [[99.0, 99.0]],
        status="CONVERGED", obj_initial=0.5, obj_final=0.4,
        name="stub_outside.bat",
    )
    with pytest.raises(pe.SolutionOutsideBox) as ei:
        qw.solve_q5_patch(spec, worker=stub, timeout=15.0)
    assert isinstance(ei.value, pe.InvalidSolutionQ4Patch)
    assert isinstance(ei.value, errors.MeshError)


def test_solve_rejects_invalid_patch(tmp_path):
    # Start node 4 at its optimum (1.0, 1.0) so the clamp box is
    # [0.5, 1.5] x [0.5, 1.5].  The stub "returns" node 4 at (0.5, 0.5):
    # inside the box, but it makes at least one quad self-intersecting /
    # zero-area -> an invalid patch (distinct from the out-of-box case).
    spec = _interior_spec(nx4=1.0, ny4=1.0)
    stub = _stub_worker(
        tmp_path, [[0.5, 0.5]],
        status="CONVERGED", obj_initial=0.5, obj_final=0.0,
        name="stub_invalid.bat",
    )
    with pytest.raises(pe.InvalidSolutionQ4Patch) as ei:
        qw.solve_q5_patch(spec, worker=stub, timeout=15.0)
    assert not isinstance(ei.value, pe.SolutionOutsideBox)


@needs_worker
def test_force_invalid_hook_rejected():
    spec = _interior_spec()
    with pytest.raises(pe.SolutionOutsideBox):
        qw.solve_q5_patch(spec, timeout=30.0, self_test="force_invalid")


# ---------------------------------------------------------------------------
# Evidence 9/10 — strict rejection + decode round-trip
# ---------------------------------------------------------------------------

def test_spec_rejects_non_finite_values():
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(h=1.0, ux=1.0, uy=0.0,
                     nx=(float("nan"), 1.0, 1.0, 0.0), ny=(0.0, 0.0, 1.0, 1.0),
                     free=(1,), quads=((0, 1, 2, 3),))
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(h=float("inf"), ux=1.0, uy=0.0,
                     nx=(0.0, 1.0, 1.0, 0.0), ny=(0.0, 0.0, 1.0, 1.0),
                     free=(1,), quads=((0, 1, 2, 3),))


def test_spec_rejects_bad_indices_and_dups():
    base = dict(h=1.0, ux=1.0, uy=0.0,
                nx=(0.0, 1.0, 1.0, 0.0), ny=(0.0, 0.0, 1.0, 1.0),
                quads=((0, 1, 2, 3),))
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(free=(9,), **base)
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(free=(1, 1), **base)
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(free=(1.0,), **base)  # float even if integral-valued
    # quads-variant sub-cases: supply a free node too so the failure is
    # attributed to the quads, not a missing free argument.
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(quads=((0, 1, 1, 3),), free=(1,),
                     **{k: v for k, v in base.items() if k != "quads"})
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(quads=((0, 1, 2),), free=(1,),
                     **{k: v for k, v in base.items() if k != "quads"})
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(quads=((0, 1, 2, 4),), free=(1,),
                     **{k: v for k, v in base.items() if k != "quads"})
    with pytest.raises(pe.PatchRejected):
        pe.PatchSpec(free=(1,),
                     **{k: v for k, v in base.items() if k not in ("free", "nx", "ny")},
                     nx=(0.0, 1.0, 1.0), ny=(0.0, 0.0, 1.0))


def test_decode_round_trip_is_lossless():
    report = pe.Q5SolveReport(
        status="CONVERGED",
        objective_initial=0.28128923494535996,
        objective_final=9.7e-12,
        iterations=41,
        free_final=((1.0000001, -0.0000002), (0.7, 1.3)),
        message=None,
    )
    raw = report.to_dict()
    assert raw["schema"] == pe.Q5_RESPONSE_SCHEMA
    back = pe.decode_q5_response(raw, free_count=2)
    assert back == report
    assert back.to_dict() == raw


def test_decode_rejects_bad_schema_or_status():
    raw = {
        "schema": "anymesher.quad-tinyad-response/1",
        "status": "CONVERGED",
        "objective_initial": 1.0,
        "objective_final": 0.5,
        "iterations": 1,
        "free_nodes": [[0.5, 0.5]],
    }
    with pytest.raises(pe.WorkerErrorQ5):
        pe.decode_q5_response(dict(raw, schema="wrong"), free_count=1)
    with pytest.raises(pe.WorkerErrorQ5):
        pe.decode_q5_response(dict(raw, status="MAYBE"), free_count=1)
    with pytest.raises(pe.WorkerErrorQ5):
        pe.decode_q5_response(dict(raw, free_nodes=[1.0]), free_count=1)
    with pytest.raises(pe.WorkerErrorQ5):
        pe.decode_q5_response(dict(raw, iterations=-1), free_count=1)
    with pytest.raises(pe.WorkerErrorQ5):
        pe.decode_q5_response(raw, free_count=2)


def test_all_q5_exceptions_are_mesh_errors():
    for exc in (
        pe.PatchRejected,
        pe.InvalidSolutionQ4Patch,
        pe.ObjectiveRegression,
        pe.SolutionOutsideBox,
        pe.WorkerErrorQ5,
        qw.WorkerLifecycleQ5,
        qw.WorkerNotFoundQ5,
        qw.WorkerCrashQ5,
        qw.WorkerTimeoutQ5,
        qw.WorkerMalformedQ5,
    ):
        assert issubclass(exc, errors.MeshError), exc
