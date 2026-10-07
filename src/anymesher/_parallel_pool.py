"""Owned spawn worker pool with a bounded, contained lifecycle.

Public contract (see ``docs/PARALLEL_POOL_LIFECYCLE.md``):

- ``create_parallel_pool(workers=None)`` starts a caller-owned warm pool of
  spawned worker processes and returns a :class:`ParallelPoolLease`.  A job
  borrows the pool exclusively through ``ParallelOptions.pool_lease``.
- The library creates cold, library-owned pools itself when no lease is given.
- Only this module's own child processes are ever created or terminated; a
  bare, shared or unknown ``Executor`` is never used or killed.
- Cancellation is cooperative and shared per job: the parent polls the caller's
  ``cancellation_check`` every 100 ms and sets one shared event that reaches
  the meshing engine inside each worker.  All termination and joining is
  bounded by a single total deadline (``TERMINATION_DEADLINE_SECONDS``).

Containment contracts actually implemented here:

- Windows: a Job Object is created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` armed through
  ``SetInformationJobObject(JobObjectExtendedLimitInformation)`` — the
  correct SDK information class is 9 — *before anything is spawned*; if the
  limit cannot be armed the pool fails closed
  (:class:`ParallelContainmentUnavailable`) with no process created.  Every
  worker is assigned to that job before the pool reports started; assignment
  failure fails the pool closed and disposes every spawned process.  The
  kernel-enforced kill-on-close limit is the whole parent-death containment
  story on Windows: when the last job handle closes (orderly shutdown or
  abrupt owner death) the kernel terminates every process in the job,
  workers and task descendants alike.  Bounded cleanup uses
  ``TerminateJobObject`` on the pool's job only.
- POSIX: each worker puts itself in its own process group (failure fails the
  pool closed through the handshake) and a watchdog thread inside the worker
  actively polls the parent-death pipe *for the whole life of the worker,
  including while a task runs*; on parent death it kills the worker's own
  process group.  Because an in-worker thread and a ``finally`` clause both
  disappear on SIGKILL/``os._exit``, an independent *guardian* process (in
  its own process group, spawned before the workers) holds one sentinel read
  end per worker plus the parent admission pipe.  Verified worker group ids
  (pgid == worker pid, from the startup handshake) are admitted to the
  guardian and it acknowledges before the pool reports started.  When a
  worker dies abruptly its sentinel closes and the guardian kills that
  verified group — sleeping task descendants die with it; when the parent
  dies the admission pipe closes and the guardian kills every verified
  group.  Graceful shutdown sweeps the recorded groups (descendants
  included, even when the worker already died) and only then releases the
  guardian.  Only recorded verified groups are ever signalled; the caller's
  group and unrelated processes are never targeted.  This branch is
  implemented but was NOT measurable on the Windows source runtime; no
  POSIX support is claimed.

Transport: every task and result message is pickled to bytes *before* it
enters a ``multiprocessing.Queue``, so serialization failures surface
synchronously (prompt typed failure and cleanup) instead of dying in the
queue's asynchronous feeder thread.

No private ``ProcessPoolExecutor`` internals are used: workers are plain
``multiprocessing`` spawn processes serving two queues, and results surface
through the public ``concurrent.futures.Future`` interface.
"""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import pickle
import queue
import select
import signal
import sys
import threading
import time
from concurrent.futures import Future
from itertools import count as _count
from typing import Any, Callable

from .errors import MeshError

__all__ = [
    "ParallelPoolLease",
    "ParallelContainmentUnavailable",
    "create_parallel_pool",
    "TERMINATION_DEADLINE_SECONDS",
]

#: Total budget for every termination and join, whatever the cause.
TERMINATION_DEADLINE_SECONDS = 5.0
#: Parent-side polling period for cancellation and worker liveness.
POLL_SECONDS = 0.1
#: How long a pool start waits for all startup handshakes.
START_TIMEOUT_SECONDS = 30.0
#: Windows spawn pool size limit (documented ProcessPoolExecutor behaviour).
MAX_WORKERS = 61
#: Cooperative grace before hard termination when a job is being torn down.
_COOPERATIVE_GRACE_SECONDS = 1.0
#: Cooperative grace on an orderly shutdown of an idle pool.
_ORDERLY_GRACE_SECONDS = 2.0
#: Worker environment variables defaulted to single-threaded numerical
#: libraries.  Existing (caller-provided) values are never overwritten.
_THREAD_ENV_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class ParallelContainmentUnavailable(MeshError):
    """Typed failure: this platform could not provide safe worker containment.

    A cold job may fall back to the serial route *before any work* when it
    sees this exception; a warm ``create_parallel_pool`` caller sees it as
    the pool's typed startup failure.
    """


class ParallelJobCancelled(Exception):
    """A parallel job was cancelled through the shared cooperative signal.

    The caller's own cancellation exception (raised by its
    ``cancellation_check``) always keeps its identity; this class only covers
    the truthy-return style of ``cancellation_check`` and the worker-side
    cooperative signal.
    """


# ---------------------------------------------------------------------------
# Windows Job Object containment (measured on this runtime)


if sys.platform == "win32":
    import ctypes.wintypes as _wt

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", ctypes.c_uint),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint),
            ("SchedulingClass", ctypes.c_uint),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        # Official SDK layout (winnt.h): exactly four SIZE_T fields after
        # IoInfo.  144 bytes on 64-bit Windows.
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint),
            ("PageFaultCount", ctypes.c_uint),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", ctypes.c_uint),
            ("dwHighDateTime", ctypes.c_uint),
        ]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    # JobObjectExtendedLimitInformation in JOBOBJECTINFOCLASS (winnt.h):
    # class 7 is the completion-port association class and is invalid here.
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_QUERY_INFORMATION = 0x0400
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _PROCESS_VM_READ = 0x0010
    _BELOW_NORMAL_PRIORITY_CLASS = 0x4000

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.restype = _wt.HANDLE
    _kernel32.CreateJobObjectW.argtypes = [_wt.LPVOID, _wt.LPCWSTR]
    _kernel32.SetInformationJobObject.restype = _wt.BOOL
    _kernel32.SetInformationJobObject.argtypes = [
        _wt.HANDLE, ctypes.c_uint, _wt.LPVOID, _wt.DWORD
    ]
    _kernel32.AssignProcessToJobObject.restype = _wt.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [_wt.HANDLE, _wt.HANDLE]
    _kernel32.OpenProcess.restype = _wt.HANDLE
    _kernel32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
    _kernel32.TerminateJobObject.restype = _wt.BOOL
    _kernel32.TerminateJobObject.argtypes = [_wt.HANDLE, _wt.UINT]
    _kernel32.TerminateProcess.restype = _wt.BOOL
    _kernel32.TerminateProcess.argtypes = [_wt.HANDLE, _wt.UINT]
    _kernel32.CloseHandle.restype = _wt.BOOL
    _kernel32.CloseHandle.argtypes = [_wt.HANDLE]
    _kernel32.GetExitCodeProcess.restype = _wt.BOOL
    _kernel32.GetExitCodeProcess.argtypes = [_wt.HANDLE, _wt.LPDWORD]
    _kernel32.GetProcessTimes.restype = _wt.BOOL
    _kernel32.GetProcessTimes.argtypes = [
        _wt.HANDLE,
        ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME),
        ctypes.POINTER(_FILETIME), ctypes.POINTER(_FILETIME),
    ]
    _kernel32.GetCurrentProcess.restype = _wt.HANDLE
    _kernel32.GetCurrentProcess.argtypes = []
    _kernel32.SetPriorityClass.restype = _wt.BOOL
    _kernel32.SetPriorityClass.argtypes = [_wt.HANDLE, _wt.DWORD]
    _kernel32.GetPriorityClass.restype = _wt.DWORD
    _kernel32.GetPriorityClass.argtypes = [_wt.HANDLE]

    try:
        _psapi = ctypes.WinDLL("psapi", use_last_error=True)
        _psapi.GetProcessMemoryInfo.restype = _wt.BOOL
        _psapi.GetProcessMemoryInfo.argtypes = [
            _wt.HANDLE, ctypes.POINTER(_PROCESS_MEMORY_COUNTERS), _wt.DWORD
        ]
    except OSError:  # pragma: no cover - psapi is always present on Windows
        _psapi = None

_STILL_ACTIVE = 259

def _priority_name(value: int | None) -> str | None:
    if value is None:
        return None
    names = {
        0x0020: "normal",
        0x4000: "below_normal",
        0x0040: "idle",
        0x0080: "above_normal",
        0x8000: "high",
        0x0100: "realtime",
    }
    return names.get(int(value), f"priority-class-{int(value)}")


def _pid_alive(pid: int) -> bool:
    """Portable liveness probe with full-width handles (no truncation).

    Windows: ``OpenProcess``/``GetExitCodeProcess`` with explicit 64-bit-safe
    signatures.  POSIX: ``os.kill(pid, 0)``.
    """

    pid = int(pid)
    if pid <= 0:
        return False
    if sys.platform == "win32":
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = _wt.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == _STILL_ACTIVE
        finally:
            _kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pid_signature(pid: int) -> int | None:
    """Creation-time signature guarding against PID reuse in assertions.

    Windows: the process creation FILETIME (0 when unavailable).  POSIX:
    ``None`` (no equivalent measured here; tests must not treat it as
    evidence).
    """

    if sys.platform != "win32":
        return None
    handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return None
    try:
        created = _FILETIME()
        exit_time = _FILETIME()
        kernel = _FILETIME()
        user = _FILETIME()
        if not _kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created), ctypes.byref(exit_time),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            return None
        return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
    finally:
        _kernel32.CloseHandle(handle)


def _win_create_job() -> tuple[int | None, bool]:
    """Job Object handle and whether kill-on-close was armed.

    The limit is set through ``SetInformationJobObject`` with the correct
    ``JobObjectExtendedLimitInformation`` class (9).  Returns
    ``(None, False)`` outside Windows, when job creation fails, or when the
    kill-on-close limit cannot be armed — callers must fail closed then,
    because without the kernel limit there is no parent-death containment.
    """

    if sys.platform != "win32":
        return None, False
    job = _kernel32.CreateJobObjectW(None, None)
    if not job:
        return None, False
    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not _kernel32.SetInformationJobObject(
        job,
        _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        _kernel32.CloseHandle(job)
        return None, False
    return int(job), True


def _win_assign(job: int | None, pid: int) -> bool:
    if job is None:
        return False
    handle = _kernel32.OpenProcess(
        _PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid
    )
    if not handle:
        return False
    try:
        return bool(_kernel32.AssignProcessToJobObject(job, handle))
    finally:
        _kernel32.CloseHandle(handle)


def _win_terminate_job(job: int | None) -> bool:
    if job is None:
        return False
    return bool(_kernel32.TerminateJobObject(job, 1))


def _win_close_handle(handle: int | None) -> None:
    if handle:
        _kernel32.CloseHandle(handle)


def _win_working_set_bytes(pid: int) -> int | None:
    if sys.platform != "win32" or _psapi is None:
        return None
    handle = _kernel32.OpenProcess(
        _PROCESS_QUERY_INFORMATION | _PROCESS_VM_READ, False, pid
    )
    if not handle:
        return None
    try:
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return None
    finally:
        _kernel32.CloseHandle(handle)


def _win_apply_below_normal_priority() -> tuple[bool, int | None]:
    """Best-effort below-normal priority; returns (applied, actual class)."""

    if sys.platform != "win32":
        return False, None
    handle = _kernel32.GetCurrentProcess()
    applied = bool(
        _kernel32.SetPriorityClass(handle, _BELOW_NORMAL_PRIORITY_CLASS)
    )
    actual = int(_kernel32.GetPriorityClass(handle))
    return applied, actual


# ---------------------------------------------------------------------------
# worker


class _WorkerCancelCheck:
    """Engine-visible cooperative cancellation, bound to the shared event."""

    __slots__ = ("_event",)

    def __init__(self, event: Any) -> None:
        self._event = event

    def __call__(self, phase: str) -> None:
        if self._event.is_set():
            raise ParallelJobCancelled(f"cancelled during {phase}")


def _parent_gone(watch: Any) -> bool:
    if watch is None:
        return False
    try:
        if not watch.poll(0):
            return False
        try:
            watch.recv()
        except EOFError:
            pass
        return True
    except Exception:
        return True


def _posix_watchdog(watch: Any) -> None:
    """Kill this worker's own process group when the parent disappears.

    Runs for the whole life of the worker, including inside long tasks, so an
    uncooperative task cannot outlive the pool owner.  Only the worker's own
    group (created by ``os.setpgid(0, 0)`` at startup) is ever targeted; the
    caller's group is never signalled.
    """

    while True:
        if _parent_gone(watch):
            try:
                os.killpg(0, signal.SIGKILL)
            except Exception:
                os._exit(3)
            return
        time.sleep(POLL_SECONDS)


def _posix_guardian(
    admission: Any,
    sentinels: list[Any],
    expected: int,
    result_queue: Any,
) -> None:
    """Independent POSIX guardian for abrupt worker hard-death.

    A separate process in its own process group, so it survives a worker's
    SIGKILL/``os._exit`` — unlike the worker's own watchdog thread and
    ``finally`` clause.  It receives one verified worker group id (pgid ==
    worker pid, from the startup handshake) per sentinel read end through
    the admission pipe, acknowledges ``guardian-ready`` through the result
    queue, then waits on:

    - each worker sentinel: EOF means that worker exited for ANY reason
      (abrupt included) — the guardian kills that verified group, which
      takes sleeping task descendants down with it;
    - the admission pipe: EOF means the pool parent closed it (orderly
      shutdown release or abrupt parent death) — the guardian kills every
      verified group and exits.

    Only group ids the parent verified at the startup handshake are ever
    signalled; the caller's group and unrelated processes are never
    targeted.
    """

    try:
        os.setpgid(0, 0)  # own group: survives caller-group kills, too
    except Exception:
        pass
    pgids: dict[int, int] = {}  # sentinel position -> verified pgid
    try:
        while len(pgids) < int(expected):
            pgid = int(admission.recv())
            position = len(pgids)
            if position < len(sentinels):
                pgids[position] = pgid
    except Exception:
        pass  # the parent died mid-admission: watch whatever was admitted
    try:
        result_queue.put(pickle.dumps(("guardian-ready", tuple(pgids.values()))))
    except Exception:
        pass  # the parent is gone; guarding continues regardless
    watch: list[tuple[Any, int]] = [(admission, -1)]
    watch.extend((sentinel, position) for position, sentinel in enumerate(sentinels))
    while True:
        try:
            readable, _, _ = select.select(
                [end for end, _ in watch], [], [], POLL_SECONDS
            )
        except Exception:
            return
        for end, position in watch:
            if end not in readable:
                continue
            try:
                if end.recv() is not None:
                    continue  # unexpected data; the holder is still alive
            except Exception:
                pass  # EOF: the holder of the write end is gone
            if position < 0:
                # Parent released the guardian (or died): sweep everything.
                for pgid in pgids.values():
                    _posix_kill_group(pgid)
                return
            _posix_kill_group(pgids.get(position, 0))


def _posix_kill_group(pgid: int) -> None:
    """Kill one verified owned process group; never the caller's group."""

    if not pgid or pgid <= 1:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        pass


def _worker_thread_env() -> dict[str, str | None]:
    """Default numerical-library thread limits; never overwrite explicit."""

    applied: dict[str, str | None] = {}
    for name in _THREAD_ENV_LIMITS:
        if name not in os.environ:
            os.environ[name] = "1"
        applied[name] = os.environ.get(name)
    return applied


def _worker_threadpool_info() -> Any:
    """Actual numerical-library thread counts, when introspection exists.

    Environment strings alone do not prove effective counts (a library may
    already be imported before the worker entry point runs), so an optional
    runtime introspection records the real per-library settings.  ``None``
    means the counts are unknown on this runtime, not that they are safe.
    """

    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except Exception:
        return None


def _pool_worker(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    cancel_event: Any,
    watch: Any,
    sentinel: Any = None,
) -> None:
    """Serve tasks until a shutdown sentinel, cancellation or parent death.

    ``sentinel`` (POSIX) is this worker's death-notification pipe write end:
    it is held open for the whole life of the process and never written, so
    the guardian sees EOF exactly when this worker exits — abruptly included.
    """

    containment = None
    if os.name == "posix":
        try:
            os.setpgid(0, 0)  # own process group: group-wide bounded cleanup
            containment = "process-group"
        except Exception as error:
            _put(result_queue, (
                "containment-failed", worker_id, os.getpid(),
                f"setpgid failed: {error!r}",
            ))
            os._exit(3)
        watchdog = threading.Thread(
            target=_posix_watchdog, args=(watch,), daemon=True
        )
        watchdog.start()
        try:
            _serve_tasks(
                worker_id, task_queue, result_queue, cancel_event, watch, containment
            )
        finally:
            # The worker owns this group; taking it down takes any task
            # descendant that stayed in the group along with it.
            try:
                os.killpg(0, signal.SIGKILL)
            except Exception:
                pass
    else:
        _serve_tasks(
            worker_id, task_queue, result_queue, cancel_event, watch, containment
        )


def _serve_tasks(
    worker_id: int,
    task_queue: Any,
    result_queue: Any,
    cancel_event: Any,
    watch: Any,
    containment: str | None,
) -> None:
    # Test-only crash injection hooks (never set outside the test suite).
    if os.environ.get("_ANYMESHER_PARALLEL_TEST_CRASH_AT_START") == "1":
        os._exit(3)
    priority_applied, priority_actual = _win_apply_below_normal_priority()
    startup = {
        "containment": containment,  # Windows: job object, verified by parent
        "priority_applied": priority_applied,
        "priority_class": priority_actual,
        "thread_env": _worker_thread_env(),
        "threadpool": _worker_threadpool_info(),
    }
    _put(result_queue, ("ready", worker_id, os.getpid(), startup))
    crash_hook = os.environ.get("_ANYMESHER_PARALLEL_TEST_CRASH") == "1"
    while True:
        try:
            message = task_queue.get(timeout=POLL_SECONDS)
        except queue.Empty:
            if cancel_event.is_set() or _parent_gone(watch):
                return
            continue
        if message is None:
            return
        _, task_id, fn, args, cooperative = pickle.loads(message)
        if crash_hook:
            os._exit(3)  # test-only injected worker crash
        if cancel_event.is_set():
            _put(result_queue, ("cancelled", task_id, worker_id))
            continue
        try:
            if cooperative:
                value = fn(*args, cancellation_check=_WorkerCancelCheck(cancel_event))
            else:
                value = fn(*args)
        except BaseException as error:  # reported to the parent, never swallowed
            _put(result_queue, ("error", task_id, worker_id, error))
        else:
            _put(result_queue, ("done", task_id, worker_id, value))


def _put(result_queue: Any, message: tuple) -> None:
    """Synchronous transport: pickle first, so failures surface here.

    ``multiprocessing.Queue.put`` pickles in a feeder thread, so an
    untransportable task, result or exception would otherwise hang the pool
    while the workers stay alive.  Pre-serializing turns every such failure
    into a prompt, typed, picklable error message instead.
    """

    try:
        data = pickle.dumps(message)
    except Exception as error:
        fallback = message[0]
        if fallback == "error" and len(message) >= 4:
            fallback_message = (
                "error", message[1], message[2],
                MeshError(
                    "a worker raised an exception that could not be "
                    f"transported: {error!r}"
                ),
            )
        else:
            fallback_message = (
                "error", message[1], message[2],
                MeshError(
                    "a worker message could not be transported: "
                    f"{type(message[0]).__name__}: {error!r}"
                ),
            )
        try:
            data = pickle.dumps(fallback_message)
        except Exception:
            os._exit(4)  # transport is broken; the parent sees a dead worker
    try:
        result_queue.put(data)
    except Exception:
        os._exit(4)  # transport is broken; the parent sees a dead worker


# ---------------------------------------------------------------------------
# pool


class _ParallelPool:
    """Spawned workers, two queues, one shared cancel event, bounded ends."""

    def __init__(
        self,
        workers: int,
        *,
        name: str = "cold",
        start_timeout: float = START_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("a parallel pool needs at least one worker")
        self._name = str(name)
        self._ctx = multiprocessing.get_context("spawn")
        self._cancel_event = self._ctx.Event()
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._watch_recv = self._watch_send = None
        self._admission_recv = self._admission_send = None
        self._guardian: multiprocessing.process.BaseProcess | None = None
        self._guardian_ready = False
        self._sentinel_sends: list[Any] = []
        self._worker_pgids: dict[int, int] = {}
        if os.name == "posix":
            # Worker parent-death fast path, and the guardian admission pipe.
            self._watch_recv, self._watch_send = self._ctx.Pipe(duplex=False)
            self._admission_recv, self._admission_send = self._ctx.Pipe(duplex=False)
        # Fail closed before anything is spawned: without verified containment
        # no pool may ever admit work.
        self._job = None
        self._kill_on_close = False
        if sys.platform == "win32":
            self._job, self._kill_on_close = _win_create_job()
            if self._job is None or not self._kill_on_close:
                raise ParallelContainmentUnavailable(
                    "safe worker containment failed: the Windows Job Object "
                    "kill-on-close limit could not be armed "
                    "(SetInformationJobObject, JobObjectExtendedLimitInformation)"
                )
        self._containment = (
            "job-object-kill-on-close"
            if sys.platform == "win32"
            else "process-group+guardian"
            if os.name == "posix"
            else None
        )
        self._t0 = time.monotonic()
        self._lifecycle: list[list[Any]] = []
        self._workers: list[multiprocessing.process.BaseProcess] = []
        self._startup_info: dict[int, dict[str, Any]] = {}
        self._pending: dict[int, Future] = {}
        self._counter = _count()
        self._closed = False
        self._last_evidence: dict[str, Any] = {}
        try:
            if os.name == "posix":
                # One death sentinel per worker: the worker holds the write
                # end for its whole life, the guardian holds the read end.
                sentinel_recvs: list[Any] = []
                for _ in range(workers):
                    recv_end, send_end = self._ctx.Pipe(duplex=False)
                    sentinel_recvs.append(recv_end)
                    self._sentinel_sends.append(send_end)
                self._guardian = self._ctx.Process(
                    target=_posix_guardian,
                    args=(
                        self._admission_recv,
                        sentinel_recvs,
                        workers,
                        self._result_queue,
                    ),
                )
                self._guardian.start()
                for recv_end in sentinel_recvs:
                    recv_end.close()  # only the guardian holds these now
            for index in range(workers):
                process = self._ctx.Process(
                    target=_pool_worker,
                    args=(
                        index,
                        self._task_queue,
                        self._result_queue,
                        self._cancel_event,
                        None if sys.platform == "win32" else self._watch_recv,
                        None if sys.platform == "win32" else self._sentinel_sends[index],
                    ),
                )
                process.start()
                # Owned tracking BEFORE any assignment can fail, so a refusal
                # or startup error disposes the started process.
                self._workers.append(process)
                if (
                    sys.platform == "win32"
                    and not _win_assign(self._job, process.pid)
                ):
                    raise ParallelContainmentUnavailable(
                        "safe worker containment failed: worker "
                        f"{index} (pid {process.pid}) could not be assigned "
                        "to the pool's Job Object"
                    )
                if os.name == "posix":
                    self._sentinel_sends[index].close()  # the worker holds it
            self._record("workers-spawned")
            ready: set[int] = set()
            deadline = time.monotonic() + float(start_timeout)
            while len(ready) < workers or (
                os.name == "posix" and not self._guardian_ready
            ):
                self._require_alive("startup")
                if self._guardian is not None and not self._guardian.is_alive():
                    raise ParallelContainmentUnavailable(
                        "safe worker containment failed: the worker-death "
                        "guardian died during startup"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MeshError(
                        "parallel pool startup handshake timed out "
                        f"({len(ready)} of {workers} workers ready)"
                    )
                try:
                    message = pickle.loads(self._result_queue.get(
                        timeout=min(remaining, POLL_SECONDS)
                    ))
                except queue.Empty:
                    continue
                except Exception as error:
                    raise MeshError(
                        "parallel pool startup transport failed: "
                        f"{error!r}"
                    ) from None
                if message[0] == "ready":
                    ready.add(message[1])
                    self._startup_info[int(message[1])] = dict(message[3])
                    if (
                        os.name == "posix"
                        and message[3].get("containment") == "process-group"
                    ):
                        # Verified own group (pgid == pid): admit it to the
                        # guardian so worker hard-death cannot orphan it.
                        self._worker_pgids[int(message[1])] = int(message[2])
                        self._admission_send.send(int(message[2]))
                elif message[0] == "guardian-ready":
                    self._guardian_ready = True
                elif message[0] == "containment-failed":
                    raise ParallelContainmentUnavailable(
                        "safe worker containment failed in worker "
                        f"{message[1]}: {message[3]}"
                    )
            self._record("pool-ready")
        except BaseException as error:
            evidence = self.shutdown(terminate_first=True)
            if evidence.get("survivors"):
                # Cleanup failure must never become a successful serial
                # fallback: raise untyped so the cold path propagates it.
                raise MeshError(
                    "parallel pool startup failed and its own cleanup left "
                    f"worker survivors {evidence['survivors']}"
                ) from error
            raise

    # -- introspection ------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def worker_count(self) -> int:
        return len(self._workers)

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(int(p.pid) for p in self._workers)

    def worker_startup_info(self) -> dict[int, dict[str, Any]]:
        """Per-worker startup acknowledgement (containment, priority, env)."""

        return {
            index: dict(info)
            for index, info in self._startup_info.items()
        }

    @property
    def containment_kind(self) -> str | None:
        """Containment mechanism this pool verified at startup."""

        return self._containment

    def dead_workers(self) -> tuple[int, ...]:
        return tuple(
            index for index, process in enumerate(self._workers)
            if not process.is_alive()
        )

    def memory_sample(self) -> list[int | None]:
        return [_win_working_set_bytes(pid) for pid in self.worker_pids]

    def lifecycle(self) -> list[list[Any]]:
        return [list(event) for event in self._lifecycle]

    def _record(self, event: str, *detail: Any) -> None:
        self._lifecycle.append([event, round(time.monotonic() - self._t0, 6), *detail])

    def _require_alive(self, where: str) -> None:
        dead = self.dead_workers()
        if dead:
            raise MeshError(
                f"a parallel pool worker died during {where} "
                f"(worker {dead[0]} of {self.worker_count})"
            )

    # -- job ---------------------------------------------------------------

    def begin_job(self) -> None:
        if self._closed:
            raise MeshError("parallel pool is shut down")
        self._cancel_event.clear()
        self._record("job-begin")

    def end_job(self) -> None:
        self._record("job-end")

    def set_cancel(self) -> None:
        self._cancel_event.set()
        self._record("cancel-signalled")

    def submit(self, fn: Callable[..., Any], *args: Any, cooperative: bool = False) -> Future:
        if self._closed:
            raise MeshError("cannot submit to a shut-down parallel pool")
        task_id = next(self._counter)
        future: Future = Future()
        # Synchronous transport: a task that cannot be pickled fails here,
        # before any worker can pick it up, so the caller can clean up.
        try:
            data = pickle.dumps(("task", task_id, fn, args, bool(cooperative)))
        except Exception as error:
            raise MeshError(
                "a parallel task could not be serialized for transport: "
                f"{error!r}"
            ) from None
        self._pending[task_id] = future
        self._task_queue.put(data)
        return future

    def cancel_pending(self) -> int:
        cancelled = 0
        for future in self._pending.values():
            if future.cancel():
                cancelled += 1
        self._record("queued-cancelled", cancelled)
        return cancelled

    def drain_results(self, timeout: float) -> None:
        """Move finished results into their futures for up to ``timeout`` s."""

        deadline = time.monotonic() + float(timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            try:
                data = self._result_queue.get(timeout=remaining)
            except queue.Empty:
                return
            try:
                message = pickle.loads(data)
            except Exception as error:
                raise MeshError(
                    "the parallel pool transport delivered an unreadable "
                    f"message: {error!r}"
                ) from None
            self._absorb(message)

    def _absorb(self, message: tuple) -> None:
        kind = message[0]
        if kind in ("ready", "guardian-ready", "containment-failed"):
            return
        task_id = message[1]
        future = self._pending.pop(task_id, None)
        if future is None or future.cancelled() or future.done():
            return
        if kind == "done":
            future.set_result(message[3])
        elif kind == "error":
            future.set_exception(message[3])
        elif kind == "cancelled":
            future.set_exception(
                ParallelJobCancelled("component cancelled before it started")
            )

    # -- teardown -----------------------------------------------------------

    def _kill_owned(self, index: int) -> bool:
        """Terminate one owned worker; on POSIX kill its verified own group.

        The group id comes from the startup handshake, never ``getpgid``:
        a worker that already died hard leaves sleeping task descendants in
        its group, and ``getpgid`` fails on a dead pid.  Only groups this
        pool created and verified (pgid == worker pid) are signalled.
        """

        process = self._workers[index]
        pgid = self._worker_pgids.get(index)
        if os.name == "posix" and pgid:
            _posix_kill_group(pgid)
            return True
        try:
            process.terminate()
        except Exception:
            return False
        return True

    def shutdown(
        self,
        *,
        terminate_first: bool = False,
        deadline: float = TERMINATION_DEADLINE_SECONDS,
    ) -> dict[str, Any]:
        """Stop every owned worker within one total deadline.

        Returns evidence: which workers left cooperatively, which had to be
        terminated, which joined, any survivors of the deadline, and whether
        the parent-death watchdog was disposed.
        """

        if self._closed:
            return dict(self._last_evidence)
        self._closed = True
        started = time.monotonic()
        evidence: dict[str, Any] = {
            "cooperative": [],
            "terminated": [],
            "joined": [],
            "survivors": [],
            "seconds": 0.0,
        }
        self._record("shutdown-begin", "terminate" if terminate_first else "orderly")
        if terminate_first:
            self._cancel_event.set()
            grace = min(_COOPERATIVE_GRACE_SECONDS, deadline)
        else:
            for _ in self._workers:
                try:
                    self._task_queue.put(None)
                except Exception:
                    break
            grace = min(_ORDERLY_GRACE_SECONDS, deadline)
        while time.monotonic() - started < grace:
            if not any(p.is_alive() for p in self._workers):
                break
            time.sleep(0.02)
        evidence["cooperative"] = [
            int(p.pid) for p in self._workers if not p.is_alive()
        ]
        survivors = [p for p in self._workers if p.is_alive()]
        if survivors:
            if self._job is not None:
                _win_terminate_job(self._job)  # kills the whole owned tree
            for index, process in enumerate(self._workers):
                if process in survivors and self._kill_owned(index):
                    evidence["terminated"].append(int(process.pid))
        for process in self._workers:
            remaining = deadline - (time.monotonic() - started)
            try:
                process.join(timeout=max(0.0, remaining))
            except Exception:
                pass
            if process.is_alive():
                evidence["survivors"].append(int(process.pid))
            else:
                evidence["joined"].append(int(process.pid))
        # Descendants that outlived their worker: on POSIX sweep every
        # verified owned group — the recorded pgid works even when the
        # worker itself already died.  On Windows the kill-on-close job
        # object owns this.
        if os.name == "posix":
            for pgid in self._worker_pgids.values():
                _posix_kill_group(pgid)
        try:
            self._task_queue.cancel_join_thread()
            self._task_queue.close()
            self._result_queue.cancel_join_thread()
            self._result_queue.close()
        except Exception:
            pass
        # Closing the watch pipes releases the workers' fast-path watchdogs
        # and the guardian: the guardian sweeps every verified group once
        # more (covering descendants of workers that already died) and exits.
        for end in self._sentinel_sends:  # parent copies; workers hold their own
            try:
                end.close()
            except Exception:
                pass
        for end in (self._watch_send, self._watch_recv,
                    self._admission_send, self._admission_recv):
            if end is not None:
                try:
                    end.close()
                except Exception:
                    pass
        if self._guardian is not None:
            try:
                self._guardian.join(
                    timeout=max(0.0, deadline - (time.monotonic() - started))
                )
            except Exception:
                pass
            evidence["guardian_pid"] = int(self._guardian.pid)
            evidence["guardian_joined"] = not self._guardian.is_alive()
        _win_close_handle(self._job)
        self._job = None
        evidence["seconds"] = round(time.monotonic() - started, 6)
        self._record("shutdown-end", len(evidence["survivors"]))
        self._last_evidence = evidence
        return evidence


# ---------------------------------------------------------------------------
# public lease


class ParallelPoolLease:
    """Exclusive borrow ticket for a caller-owned warm parallel pool.

    Pass it as ``ParallelOptions(pool_lease=...)``.  While a job runs, the
    lease is exclusively active.  After a successful job the pool stays warm
    and the lease is reusable; after a cancelled or failed job the lease is
    invalidated and the owner must create a new pool with
    :func:`create_parallel_pool` for the next job.

    All state transitions are guarded by one lock, so exclusive admission is
    thread-safe and a caller cannot close (and thereby terminate) a pool
    whose workers are borrowed by a running job.
    """

    def __init__(self, pool: _ParallelPool) -> None:
        self._pool = pool
        self._state = "idle"  # idle -> active -> idle | invalid | closed
        self._lock = threading.Lock()

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"ParallelPoolLease(state={self._state}, workers={self._pool.worker_count})"
        )

    @property
    def workers(self) -> int:
        """Number of worker processes in the leased pool."""

        return self._pool.worker_count

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return self._pool.worker_pids

    @property
    def busy(self) -> bool:
        return self._state == "active"

    @property
    def valid(self) -> bool:
        return self._state in ("idle", "active")

    @property
    def closed(self) -> bool:
        return self._state == "closed"

    def close(self, *, timeout: float = TERMINATION_DEADLINE_SECONDS) -> None:
        """Dispose the warm pool, terminating workers within ``timeout`` s.

        Rejected while a job is borrowing the pool: closing it would
        terminate workers a running job still depends on.
        """

        with self._lock:
            if self._state == "closed":
                return
            if self._state == "active":
                raise MeshError(
                    "cannot close a parallel pool lease while its job is "
                    "still running; the job borrows these workers"
                )
            self._state = "closed"
        evidence = self._pool.shutdown(deadline=timeout)
        if evidence.get("survivors"):
            raise MeshError(
                "parallel pool workers survived the termination deadline: "
                f"{evidence['survivors']}"
            )

    def __enter__(self) -> "ParallelPoolLease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- library-internal lease transitions ---------------------------------

    def _acquire(self) -> _ParallelPool:
        with self._lock:
            if self._state == "invalid":
                raise MeshError(
                    "this parallel pool lease was invalidated by a canceled or "
                    "failed job; create a new pool with create_parallel_pool()"
                )
            if self._state != "idle":
                raise MeshError(
                    "this parallel pool is already running a job; leases are "
                    "exclusive"
                )
            self._state = "active"
        self._pool.begin_job()
        return self._pool

    def _release(self) -> None:
        with self._lock:
            if self._state == "active":
                self._state = "idle"

    def _invalidate(self) -> None:
        with self._lock:
            if self._state != "closed":
                self._state = "invalid"


def create_parallel_pool(
    workers: int | None = None,
    *,
    name: str | None = None,
    start_timeout: float = START_TIMEOUT_SECONDS,
) -> ParallelPoolLease:
    """Start a caller-owned warm pool and return its exclusive lease.

    ``workers`` defaults to ``min(cpu_count, 61)``.  The pool is contained:
    on Windows its workers live in a kill-on-close Job Object of this
    process, so they cannot outlive it, and every worker completes a startup
    handshake before the pool is returned.  If safe containment cannot be
    provided the factory fails with the typed
    :class:`ParallelContainmentUnavailable`.
    """

    if multiprocessing.current_process().daemon:
        raise MeshError(
            "a daemonic process cannot own a parallel pool; run mesh jobs in "
            "a non-daemonic application worker (ANYfem outer containment)"
        )
    count = workers if workers is not None else min(multiprocessing.cpu_count(), MAX_WORKERS)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer in 1..{MAX_WORKERS}")
    pool = _ParallelPool(
        count, name=name or "warm", start_timeout=start_timeout
    )
    return ParallelPoolLease(pool)
