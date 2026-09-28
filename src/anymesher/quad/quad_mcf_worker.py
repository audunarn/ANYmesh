"""Quad-first Q4: subprocess adapter for the LEMON NetworkSimplex worker.

The worker (built from
``third_party/quad/worker/quad_mcf_worker.cc``) speaks a strict JSON-over-stdio
protocol.  This module owns the *subprocess* lifecycle, the transport boundary
(stdin/stdout JSON), and the typed failure surface for *lifecycle* problems
(missing binary, crash, hang/timeout, malformed response).

Design notes
------------

* **No hidden fallback.**  Every failure mode is typed (see below) and no
  silent fallback to a Python MCF solver is performed.
* **Worker binary location.**  By default this module locates the worker in
  ``third_party/quad/worker/out/lemon/quad_mcf_worker[.exe]`` (built by
  ``tools/build_quad_workers.py``)  Tests and other
  callers may pass ``worker_path`` to point at a custom build (e.g., a
  deliberately crashing stub) for lifecycle tests.
* **Determinism contract.**  The worker is *deterministic across runs* for a
  fixed request (the solver is integer and the request is fully specified);
  the per-arc perturbation in :func:`anymesher.quad.count_mcf.build_request`
  makes that determinism *unique* rather than merely stable.
* **Cancellation.**  The Python side issues a cancel token by ``terminate()``
  and then ``kill()``; the worker is then reaped.  There is no in-band
  cancel — the worker has *no* request field for it (it does not model
  "cancel" — the caller chooses to abort the whole solve).
* **Timeout.**  Python ``subprocess.run(timeout=...)`` kills the worker after
  ``timeout`` seconds; the typed failure :class:`WorkerTimeout` is raised.

Failure surface
---------------

* :class:`WorkerNotFound` — cannot locate the worker binary.
* :class:`WorkerCrash`     — non-zero exit, empty or malformed stdout.
* :class:`WorkerTimeout`   — the worker did not finish within the deadline.
* :class:`WorkerMalformed` — the worker produced well-formed JSON that did
  not match the response schema (typed via :mod:`.count_mcf`).

All are subclasses of :class:`anymesher.errors.MeshError` so callers can use a
single ``except MeshError`` to handle the whole quad-first surface.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..errors import MeshError
from .count_mcf import (
    MCFRequest,
    MCFResponse,
    SolveReport,
    build_request,
    decode_response,
    validate_response,
)
from .count_model import CountInstance, CountRejected

__all__ = [
    "SCHEMA_REQUEST",
    "SCHEMA_RESPONSE",
    "WorkerNotFound",
    "WorkerCrash",
    "WorkerTimeout",
    "WorkerMalformed",
    "WorkerLifecycle",
    "default_worker_path",
    "find_worker",
    "run_worker",
    "solve_count_instance",
]

# Re-export for caller convenience; the actual constants live in count_mcf.
from .count_mcf import MCF_REQUEST_SCHEMA as SCHEMA_REQUEST
from .count_mcf import MCF_RESPONSE_SCHEMA as SCHEMA_RESPONSE


class WorkerLifecycle(MeshError):
    """Base class for all worker transport/lifecycle failures.

    ``WorkerNotFound``, ``WorkerCrash``, ``WorkerTimeout``,
    ``WorkerMalformed`` are all subclasses; an ``except WorkerLifecycle``
    captures the *entire* transport surface without accidentally catching a
    domain error like :class:`anymesher.quad.count_model.CountRejected`.
    """


class WorkerNotFound(WorkerLifecycle):
    """The worker binary cannot be located on this machine."""


class WorkerCrash(WorkerLifecycle):
    """The worker exited non-zero without a valid response.

    ``stderr`` and ``returncode`` are carried on the exception so the caller
    can inspect (or assert on) the exact crash signature in lifecycle tests.
    """

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class WorkerTimeout(WorkerLifecycle):
    """The worker did not finish within the deadline.

    The killed worker's accumulated stdout/stderr (typically empty) and the
    deadline (seconds) are carried on the exception.
    """

    def __init__(self, message: str, *, deadline_seconds: float, stdout_partial: str = "", stderr_partial: str = "") -> None:
        super().__init__(message)
        self.deadline_seconds = deadline_seconds
        self.stdout_partial = stdout_partial
        self.stderr_partial = stderr_partial


class WorkerMalformed(WorkerLifecycle):
    """The worker produced well-formed JSON that did not match the schema.

    This is distinct from :class:`WorkerCrash`: the worker *finished* and
    produced *some* JSON on stdout, but the JSON is not a valid
    ``anymesher.quad-mcf-response/1`` document (bad schema tag, missing
    fields, unknown fields, or non-integer in a typed slot).
    """

    def __init__(self, message: str, *, raw_stdout: str = "") -> None:
        super().__init__(message)
        self.raw_stdout = raw_stdout


# ---------------------------------------------------------------------------
# Worker path discovery
# ---------------------------------------------------------------------------

_WORKER_RELATIVE = Path("third_party") / "quad" / "worker" / "out"


def _candidate_paths() -> list[Path]:
    """Candidate worker binary locations, in priority order on any platform."""
    # 1) explicit override (envvar first so tests and installed wheels can pin it)
    env = os.environ.get("ANYMESH_QUAD_MCF_WORKER")
    if env:
        return [Path(env)] if Path(env).is_file() else []
    # 2) repo-relative output of ``tools/build_quad_workers.py``
    candidates: list[Path] = []
    for parent in Path(__file__).resolve().parents:
        if (parent / "QUAD_FIRST_FULL_PROGRAMME.md").is_file():
            out = parent / _WORKER_RELATIVE / "lemon"
            candidates.append(out / ("quad_mcf_worker.exe" if os.name == "nt" else "quad_mcf_worker"))
            if os.name != "nt":
                candidates.append(out / "quad_mcf_worker.out")  # legacy POSIX name
            break
    return [c for c in candidates if c.is_file()]


def default_worker_path() -> Path:
    """Return the first existing worker binary, raising :class:`WorkerNotFound` otherwise."""
    paths = _candidate_paths()
    if not paths:
        raise WorkerNotFound(
            "quad_mcf_worker binary not found; "
            "run `python tools/build_quad_workers.py mcf` "
            "(or set ANYMESH_QUAD_MCF_WORKER to a pre-built binary)"
        )
    return paths[0]


def find_worker(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a usable worker binary for ``path`` (``None`` = auto-discover).

    A caller-supplied path is returned as-is after an ``os.access`` check;
    the worker *must* be executable at call time (or, on Windows, simply a
    file — the OS enforces the PE magic at exec time and the adapter surfaces
    a :class:`WorkerCrash` with ``returncode != 0`` and an empty stdout).
    """
    if path is None:
        return default_worker_path()
    p = Path(path)
    if not p.is_file():
        raise WorkerNotFound(f"worker binary not found at {p}")
    if os.name != "nt" and os.access(p, os.X_OK) is False:
        raise WorkerNotFound(f"worker binary {p} is not executable")
    return p


# ---------------------------------------------------------------------------
# Subprocess run (transport boundary)
# ---------------------------------------------------------------------------

def _load_request_json(request: MCFRequest, *, self_test: str | None = None) -> str:
    """Serialize the request to the worker's expected JSON (compact, UTF-8).

    ``self_test`` is a lifecycle-test hook only: when non-``None`` the worker
    ignores the solver path entirely and either crashes (``"crash"``) or hangs
    (``"hang"``).  Production code never sets it.
    """
    payload = request.to_dict()
    if self_test is not None:
        payload["worker_self_test"] = self_test
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _run_worker_core(worker: Path, request_json: str, *, timeout: float) -> tuple[int, str, str]:
    """Spawn, write, read, and reap the worker in a single subprocess.call.

    Uses ``subprocess.run(timeout=...)`` so that, on timeout, the worker is
    killed (``TerminateProcess`` on Windows, ``SIGKILL`` on POSIX) and the
    typed :class:`WorkerTimeout` is raised by the caller.  Returns
    ``(returncode, stdout, stderr)`` on a clean exit (zero or non-zero);
    the worker's *typed domain* status is parsed separately by
    :func:`.count_mcf.decode_response`.
    """
    import subprocess as sp
    try:
        proc = sp.run(
            [str(worker)],
            input=request_json,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except sp.TimeoutExpired as e:
        # `e.stdout`/`e.stderr` were captured in bytes before the kill.
        raise WorkerTimeout(
            f"worker {worker} did not finish within {timeout:.2f}s",
            deadline_seconds=timeout,
            stdout_partial=(e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", "replace"),
            stderr_partial=(e.stderr or "") if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", "replace"),
        ) from e
    except OSError as e:
        raise WorkerCrash(
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
    request: MCFRequest,
    *,
    timeout: float = 30.0,
    self_test: str | None = None,
) -> MCFResponse:
    """Run ``request`` through ``worker`` and return the typed response.

    Lifecycle failures are typed (see module docstring).  Domain failures
    (INFEASIBLE/UNBOUNDED/ERROR) are *not* raised here — they are encoded in
    the ``MCFResponse.status`` field and surfaced by
    :func:`anymesher.quad.count_mcf.validate_response`.

    ``self_test`` is a lifecycle-test hook only (``"crash"`` or ``"hang"``);
    production callers never set it.
    """
    path = find_worker(worker)
    payload = _load_request_json(request, self_test=self_test)
    returncode, stdout, stderr = _run_worker_core(path, payload, timeout=timeout)

    if not stdout:
        raise WorkerCrash(
            f"worker {path} exited ({returncode}) with no stdout",
            returncode=returncode,
            stderr=stderr or "(no stderr)",
        )
    try:
        raw = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise WorkerMalformed(
            f"worker {path} produced malformed JSON: {e}",
            raw_stdout=stdout,
        ) from e
    if not isinstance(raw, dict):
        raise WorkerMalformed(
            f"worker {path} response is not a JSON object",
            raw_stdout=stdout,
        )

    # Non-zero exit *with* a typed response: the worker signalled a fatal
    # failure; surface a WorkerCrash that carries the typed fields so tests
    # can assert on the exact signature.
    if returncode != 0:
        raise WorkerCrash(
            f"worker {path} exited non-zero ({returncode})",
            returncode=returncode,
            stderr=stderr or "(no stderr)",
        )

    # Zero exit: decode is either a valid typed response (OPTIMAL/INFEASIBLE/
    # UNBOUNDED/ERROR) or a schema mismatch (typed as InvalidSolution by
    # ``count_mcf.MCFResponse.from_dict``); translate the latter into the
    # transport-typed WorkerMalformed.
    from .count_mcf import InvalidSolution  # local import avoids circularity at module load
    try:
        return decode_response(raw)
    except InvalidSolution as e:
        raise WorkerMalformed(
            f"worker {path} response failed schema validation: {e}",
            raw_stdout=stdout,
        ) from e


# ---------------------------------------------------------------------------
# Public: end-to-end solve (reduction + transport + validation)
# ---------------------------------------------------------------------------

def solve_count_instance(
    instance: CountInstance,
    worker: str | os.PathLike[str] | None = None,
    *,
    timeout: float = 30.0,
) -> SolveReport:
    """Reduce, send, validate, and return a fully validated ``SolveReport``.

    The full typed-failure surface applies:

    * :class:`anymesher.quad.count_model.CountRejected`  — the instance is
      out of scope for the Q4 worker (zero supply, conservation, blocked
      isolation, non-integer fields…).
    * :class:`WorkerNotFound` / :class:`WorkerCrash` / :class:`WorkerTimeout`
      / :class:`WorkerMalformed` — transport/lifecycle failures.
    * :class:`anymesher.quad.count_mcf.CountInfeasible` /
      :class:`anymesher.quad.count_mcf.InvalidSolution` /
      :class:`anymesher.quad.count_mcf.NotSolvedUnexpected` — the worker
      finished but the result does not meet its own contract.
    """
    request, encoding = build_request(instance)
    response = run_worker(worker, request, timeout=timeout)
    return validate_response(instance, response, encoding=encoding)
