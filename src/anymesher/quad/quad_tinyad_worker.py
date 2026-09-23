"""Quad-first Q5: subprocess adapter for the TinyAD re-optimization worker.

The worker (built from ``third_party/quad/worker/quad_tinyad_optimizer.cc``)
is a pure, opaque *local* patch optimizer: it takes a :class:`PatchSpec`,
minimizes the documented composite energy over the free interior Q4-patch
nodes with bounded steepest-descent + Armijo backtracking line search, and
returns a typed :class:`Q5SolveReport`.  This module owns the *subprocess*
lifecycle, the transport boundary (stdin/stdout JSON), and the independent
re-validation of the returned solution against the patch's own energy and
validity guard.

Design notes
------------

* **No hidden fallback.**  Every failure mode is typed (see below) and no
  silent fallback to a pure-Python optimizer is performed.
* **Worker binary location.**  By default this module locates the worker in
  ``third_party/quad/worker/out/tinyad/quad_tinyad_optimizer.exe``.  Tests
  and other callers may pass ``worker_path`` to point at a custom build
  (e.g., a deliberately crashing / malformed / hanging stub) for lifecycle
  tests.
* **Cancellation.**  The Python side issues a cancel token by ``kill()``;
  there is *no* in-band cancel request field.
* **Timeout.**  ``subprocess.run(timeout=...)``; :class:`WorkerTimeoutQ5`
  is raised.
* **Determinism.**  The worker is deterministic for a fixed request: the
  steepest-descent line search is a pure function of the initial position,
  the analytic gradient (TinyAD) and documented energy; for a given input
  it always returns the same ``status``, ``iterations``, objective and
  free-node positions.

Failure surface
---------------

* :class:`WorkerNotFoundQ5`   — cannot locate the worker binary.
* :class:`WorkerCrashQ5`      — non-zero exit or empty stdout.
* :class:`WorkerTimeoutQ5`    — the worker did not finish within the deadline.
* :class:`WorkerMalformedQ5`  — the worker produced well-formed JSON that did
  not match the response schema.
* :class:`.patch_energy.WorkerErrorQ5`       — the worker reported ``ERROR``
  (or the response failed schema checks in a way that is not a lifecycle
  problem); carries the worker's ``message``.
* :class:`.patch_energy.ObjectiveRegression` — the returned final objective
  is higher than the initial (violates the documented "never regress"
  safeguard).
* :class:`.patch_energy.SolutionOutsideBox`  — a returned free-node
  coordinate is outside the finite clamp box ``[init - h/2, init + h/2]``.
* :class:`.patch_energy.InvalidSolutionQ4Patch` — the returned free-node
  positions do not yield a *valid* patch (a quad became self-intersecting or
  zero-area at some corner).

All are subclasses of :class:`anymesher.errors.MeshError` so callers can
catch the whole Q5 surface with a single ``except MeshError``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

from ..errors import MeshError
from .patch_energy import (
    Eps,
    Q5SolveReport,
    PatchSpec,
    PatchRejected,
    ObjectiveRegression,
    SolutionOutsideBox,
    InvalidSolutionQ4Patch,
    WorkerErrorQ5,
    energy as _energy,
    is_valid_patch as _is_valid_patch,
    decode_q5_response as _decode,
)

__all__ = [
    "SCHEMA_REQUEST",
    "SCHEMA_RESPONSE",
    "WorkerLifecycleQ5",
    "WorkerNotFoundQ5",
    "WorkerCrashQ5",
    "WorkerTimeoutQ5",
    "WorkerMalformedQ5",
    "WorkerErrorQ5",
    "ObjectiveRegression",
    "SolutionOutsideBox",
    "InvalidSolutionQ4Patch",
    "default_worker_path",
    "find_worker",
    "run_worker",
    "solve_q5_patch",
]

# Re-export for caller convenience; the actual constants live in patch_energy.
from .patch_energy import (
    Q5_REQUEST_SCHEMA as SCHEMA_REQUEST,
    Q5_RESPONSE_SCHEMA as SCHEMA_RESPONSE,
)


class WorkerLifecycleQ5(MeshError):
    """Base class for all Q5 worker transport/lifecycle failures.

    ``WorkerNotFoundQ5``, ``WorkerCrashQ5``, ``WorkerTimeoutQ5``,
    ``WorkerMalformedQ5`` are all subclasses; an ``except WorkerLifecycleQ5``
    captures the *entire* Q5 transport surface without accidentally catching
    a domain error like :class:`PatchRejected` or :class:`ObjectiveRegression`.

    (Named ``…Q5`` to avoid colliding with the Q4 ``WorkerCrash`` /
    ``WorkerTimeout`` … exported by :mod:`.quad_mcf_worker`.)
    """


class WorkerNotFoundQ5(WorkerLifecycleQ5):
    """The Q5 worker binary cannot be located on this machine."""


class WorkerCrashQ5(WorkerLifecycleQ5):
    """The Q5 worker exited non-zero without a valid response.

    ``stderr`` and ``returncode`` are carried on the exception so the caller
    can inspect (or assert on) the exact crash signature in lifecycle tests.
    """

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class WorkerTimeoutQ5(WorkerLifecycleQ5):
    """The Q5 worker did not finish within the deadline.

    The killed worker's accumulated stdout/stderr (typically empty) and the
    deadline (seconds) are carried on the exception.
    """

    def __init__(self, message: str, *, deadline_seconds: float, 
                 stdout_partial: str = "", stderr_partial: str = "") -> None:
        super().__init__(message)
        self.deadline_seconds = deadline_seconds
        self.stdout_partial = stdout_partial
        self.stderr_partial = stderr_partial


class WorkerMalformedQ5(WorkerLifecycleQ5):
    """The Q5 worker produced well-formed JSON that did not match the schema.

    This is distinct from :class:`WorkerCrashQ5`: the worker *finished* and
    produced *some* JSON on stdout, but the JSON is not a valid
    ``anymesher.quad-tinyad-response/1`` document (bad schema tag, missing
    fields, unknown fields, or wrong type in a typed slot).
    """

    def __init__(self, message: str, *, raw_stdout: str = "") -> None:
        super().__init__(message)
        self.raw_stdout = raw_stdout


# ---------------------------------------------------------------------------
# Worker path discovery
# ---------------------------------------------------------------------------

_WORKER_RELATIVE = Path("third_party") / "quad" / "worker" / "out" / "tinyad"


def _candidate_paths() -> list[Path]:
    """Candidate worker binary locations, in priority order on any platform."""
    candidates: list[Path] = []
    # 1) explicit override (envvar first so tests can pin the location)
    env = os.environ.get("ANYMESH_QUAD_TINYAD_WORKER")
    if env:
        candidates.append(Path(env))
        return candidates
    # 2) repo-relative (the canonical location per the build script)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "QUAD_FIRST_FULL_PROGRAMME.md").is_file():
            name = "quad_tinyad_optimizer.exe" if os.name == "nt" else "quad_tinyad_optimizer"
            candidates.append(parent / _WORKER_RELATIVE / name)
            break
    # 3) adjacent to this module (handy for ad-hoc builds)
    repo = here.parent.parent.parent.parent.parent
    candidates.extend([
        repo / _WORKER_RELATIVE / "quad_tinyad_optimizer.exe",
        repo / _WORKER_RELATIVE / "quad_tinyad_optimizer",
    ])
    return [c for c in candidates if c.is_file()]


def default_worker_path() -> Path:
    """Return the first existing worker binary, raising :class:`WorkerNotFoundQ5` otherwise."""
    paths = _candidate_paths()
    if not paths:
        raise WorkerNotFoundQ5(
            "quad_tinyad_optimizer binary not found; "
            "run `third_party\\quad\\worker\\build_quad_tinyad_optimizer.bat` "
            "(or set ANYMESH_QUAD_TINYAD_WORKER to a pre-built binary)"
        )
    return paths[0]


def find_worker(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a usable worker binary for ``path`` (``None`` = auto-discover).

    A caller-supplied path is returned as-is after an ``os.access`` check;
    the worker *must* exist at call time (or, on Windows, simply be a file —
    the OS enforces the PE magic at exec time and the adapter surfaces a
    :class:`WorkerCrashQ5` with ``returncode != 0`` and an empty stdout).
    """
    if path is None:
        return default_worker_path()
    p = Path(path)
    if not p.is_file():
        raise WorkerNotFoundQ5(f"worker binary not found at {p}")
    if os.name != "nt" and os.access(p, os.X_OK) is False:
        raise WorkerNotFoundQ5(f"worker binary {p} is not executable")
    return p


# ---------------------------------------------------------------------------
# Subprocess run (transport boundary)
# ---------------------------------------------------------------------------

def _load_request_json(request: PatchSpec, *, self_test: str | None = None) -> str:
    """Serialize the request to the worker's expected JSON (compact, UTF-8).

    ``self_test`` is a lifecycle-test hook only: when non-``None`` the worker
    ignores the solver path entirely and either crashes (``"crash"``), hangs
    (``"hang"``), or writes malformed JSON (``"malformed"``).  Production
    code never sets it.
    """
    return json.dumps(
        request.to_request_dict(self_test=self_test),
        separators=(",", ":"),
        allow_nan=False,
    )


def _run_worker_core(worker: Path, request_json: str, *, timeout: float) -> tuple[int, str, str]:
    """Spawn, write, read, and reap the worker in a single subprocess.call.

    Uses ``subprocess.run(timeout=...)`` so that, on timeout, the worker is
    killed (``TerminateProcess`` on Windows, ``SIGKILL`` on POSIX) and the
    typed :class:`WorkerTimeoutQ5` is raised by the caller.  Returns
    ``(returncode, stdout, stderr)`` on a clean exit (zero or non-zero); the
    worker's *typed domain* status is decoded separately by
    :func:`.patch_energy.decode_q5_response`.
    """
    try:
        proc = subprocess.run(
            [str(worker)],
            input=request_json,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise WorkerTimeoutQ5(
            f"worker {worker} did not finish within {timeout:.2f}s",
            deadline_seconds=timeout,
            stdout_partial=(e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace"),
            stderr_partial=(e.stderr or "") if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace"),
        ) from e
    except OSError as e:
        raise WorkerCrashQ5(
            f"failed to spawn worker {worker!r}: {e}",
            returncode=None,
            stderr=str(e),
        ) from e
    return proc.returncode, proc.stdout, proc.stderr


# ---------------------------------------------------------------------------
# Public: transport-only (used directly by tests, and internally by solve)
# ---------------------------------------------------------------------------

def run_worker(
    worker: str | os.PathLike[str] | None,
    request: PatchSpec,
    *,
    timeout: float = 30.0,
    self_test: str | None = None,
) -> Q5SolveReport:
    """Run ``request`` through ``worker`` and return the typed response.

    Lifecycle failures are typed (see module docstring).  Domain failures
    (``ERROR``) are *not* raised as :class:`WorkerLifecycleQ5` — they are
    decoded here and re-raised as :class:`WorkerErrorQ5` (which carries the
    worker's ``message`` so the caller can distinguish "worker failed to
    parse the request" from "transport problem").  Re-validation is a
    separate step and is *not* done here — see :func:`solve_q5_patch`.

    ``self_test`` is a lifecycle-test hook only (``"crash"``, ``"hang"``,
    ``"malformed"``, ``"force_invalid"``); production callers never set it.
    """
    path = find_worker(worker)
    payload = _load_request_json(request, self_test=self_test)
    returncode, stdout, stderr = _run_worker_core(path, payload, timeout=timeout)

    if not stdout:
        raise WorkerCrashQ5(
            f"worker {path} exited ({returncode}) with no stdout",
            returncode=returncode,
            stderr=stderr or "(no stderr)",
        )
    try:
        raw: Any = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise WorkerMalformedQ5(
            f"worker {path} produced malformed JSON: {e}",
            raw_stdout=stdout,
        ) from e

    # Non-zero exit with a typed response: the worker signalled a fatal
    # failure; surface a WorkerCrashQ5 that carries the typed fields so tests
    # can assert on the exact signature.
    if returncode != 0:
        raise WorkerCrashQ5(
            f"worker {path} exited non-zero ({returncode})",
            returncode=returncode,
            stderr=stderr or f"stdout={stdout[:256]}",
        )

    # Decode and surface any schema mismatch as WorkerMalformedQ5 (transport).
    try:
        return _decode(raw, free_count=len(request.free))
    except WorkerErrorQ5 as e:
        raise WorkerMalformedQ5(
            f"worker {path} response failed schema validation: {e}",
            raw_stdout=stdout,
        ) from e


# ---------------------------------------------------------------------------
# Public: end-to-end solve (transport + independent re-validation)
# ---------------------------------------------------------------------------

def _recompute_after_solve(spec: PatchSpec, report: Q5SolveReport) -> Q5SolveReport:
    """Independently re-compute the patch energy and validity at the
    returned free-node positions and confirm the documented safeguards:

    * the returned final objective must *not* be higher than the initial one;
    * every returned free-node coordinate must lie within
      ``[init - h/2, init + h/2]`` (the worker's per-coordinate clamp box);
    * the resulting patch must be *valid* (every quad has strictly positive
      triangle orientants in both triangles — the neighbor-halo guard).

    Raises :class:`ObjectiveRegression`, :class:`SolutionOutsideBox`, or
    :class:`InvalidSolutionQ4Patch` (all subclasses of
    :class:`MeshError`) on any violation.
    """
    if report.status == "ERROR":
        raise WorkerErrorQ5(
            f"worker reported ERROR: {report.message or '(no message)'}",
            worker_message=report.message,
        )

    f_initial = float(report.objective_initial)
    f_final = float(report.objective_final)
    if f_final > f_initial * (1.0 + 1e-12) + 1e-12:
        raise ObjectiveRegression(
            f"objective regressed: initial={f_initial:.17g}, final={f_final:.17g}",
            worker_message=report.message,
        )

    half_h = 0.5 * spec.h
    for free_i, (fx, fy) in enumerate(report.free_final):
        fi = spec.free[free_i]
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise InvalidSolutionQ4Patch(
                f"free node {free_i} has a non-finite position "
                f"({fx!r}, {fy!r})",
            )
        if fx < spec.nx[fi] - half_h - 1e-12 or fx > spec.nx[fi] + half_h + 1e-12:
            raise SolutionOutsideBox(
                "free node {} (x={!r}) is outside [{}, {}]".format(
                    free_i, fx, spec.nx[fi] - half_h, spec.nx[fi] + half_h,
                ),
            )
        if fy < spec.ny[fi] - half_h - 1e-12 or fy > spec.ny[fi] + half_h + 1e-12:
            raise SolutionOutsideBox(
                "free node {} (y={!r}) is outside [{}, {}]".format(
                    free_i, fy, spec.ny[fi] - half_h, spec.ny[fi] + half_h,
                ),
            )

    # Rebuild the full patch with the returned free-node positions and
    # re-validate against the neighbor-halo guard.
    px = list(spec.nx); py = list(spec.ny)
    for free_i, (fx, fy) in enumerate(report.free_final):
        fi = spec.free[free_i]
        px[fi] = fx; py[fi] = fy
    if not _is_valid_patch(px, py, spec.quads):
        raise InvalidSolutionQ4Patch(
            "returned free-node positions do not yield a valid patch "
            "(a quad lost strict positive orientation in both triangles)",
        )
    return report

def solve_q5_patch(
    spec: PatchSpec,
    worker: str | os.PathLike[str] | None = None,
    *,
    timeout: float = 30.0,
    self_test: str | None = None,
) -> Q5SolveReport:
    """Run the TinyAD re-optimization worker on ``spec`` and return a fully
    validated :class:`Q5SolveReport`.

    The full typed-failure surface applies (see module docstring).  After
    transport, the module independently re-computes the energy and
    re-validates the patch — a worker solution that violates either the
    clamp box or the neighbor-halo guard is rejected with a domain-typed
    error, *not* a lifecycle-typed one.

    For lifecycle tests pass ``self_test="crash"``, ``"hang"`` or
    ``"malformed"`` to force the corresponding failure mode (production
    callers never set this).
    """
    # Re-serialize the canonical spec (also re-validates the invariant).
    canonical = PatchSpec(
        h=spec.h, ux=spec.ux, uy=spec.uy,
        nx=tuple(spec.nx), ny=tuple(spec.ny),
        free=tuple(spec.free), quads=tuple(spec.quads),
        max_iter=spec.max_iter,
    )
    report = run_worker(worker, canonical, timeout=timeout, self_test=self_test)
    return _recompute_after_solve(canonical, report)
