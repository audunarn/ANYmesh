"""Unit-tier lifecycle tests for the owned parallel pool (no meshing)."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import signal
import sys
import threading
import time

import pytest

import anymesher._parallel_pool as pool_module
from anymesher import create_parallel_pool
from anymesher._parallel_pool import (
    ParallelContainmentUnavailable,
    ParallelJobCancelled,
    _ParallelPool,
    _pid_alive,
    _pid_signature,
    TERMINATION_DEADLINE_SECONDS,
)
from anymesher.errors import MeshError


def _same_process(pid: int, signature: int | None) -> bool:
    """Alive AND (where measurable) the same process, not a reused PID."""

    if not _pid_alive(pid):
        return False
    current = _pid_signature(pid)
    return signature is None or current is None or current == signature


def _kill_hard(pid: int) -> None:
    """Abrupt termination with no cleanup, like SIGKILL on the owner."""

    if sys.platform == "win32":
        handle = pool_module._kernel32.OpenProcess(0x0001, False, int(pid))
        if handle:
            pool_module._kernel32.TerminateProcess(handle, 1)
            pool_module._kernel32.CloseHandle(handle)
    else:
        os.kill(pid, signal.SIGKILL)


def _add(a, b):
    return a + b


def _sleep(seconds):
    time.sleep(seconds)
    return seconds


def _crash(_payload=None):
    os._exit(3)


def _raise_boom():
    raise ValueError("boom")


def _raise_unpickleable():
    raise ValueError(lambda: "exceptions cannot hold lambdas")


def _return_unpickleable():
    return lambda: "functions are not picklable"


def _cooperative(seconds, *, cancellation_check=None):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cancellation_check is not None:
            cancellation_check("tick")
        time.sleep(0.02)
    return seconds


def _spawn_grandchild():
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    return process.pid


def test_pool_starts_with_handshake_and_stops_cleanly():
    lease = create_parallel_pool(2)
    try:
        assert lease.valid and not lease.busy and not lease.closed
        assert lease.workers == 2
        assert len(lease.worker_pids) == 2
        signatures = {pid: _pid_signature(pid) for pid in lease.worker_pids}
        assert all(_same_process(pid, signatures[pid]) for pid in lease.worker_pids)
        startup = lease._pool.worker_startup_info()
        assert set(startup) == {0, 1}
        # Containment is verified at startup: by the parent on Windows (Job
        # Object assignment), by the worker on POSIX (own process group).
        assert lease._pool.containment_kind is not None
    finally:
        lease.close()
    assert lease.closed and not lease.valid
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(
        _same_process(pid, signatures[pid]) for pid in lease.worker_pids
    ):
        time.sleep(0.05)
    assert not any(
        _same_process(pid, signatures[pid]) for pid in lease.worker_pids
    )


def test_tasks_return_results_through_public_futures():
    lease = create_parallel_pool(2)
    try:
        pool = lease._pool
        futures = [pool.submit(_add, i, i) for i in range(4)]
        deadline = time.monotonic() + 10.0
        while not all(f.done() for f in futures) and time.monotonic() < deadline:
            pool.drain_results(0.1)
        assert [f.result() for f in futures] == [0, 2, 4, 6]
    finally:
        lease.close()


def test_worker_exception_reaches_the_future():
    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        future = pool.submit(_raise_boom)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        assert isinstance(future.exception(), ValueError)
        assert str(future.exception()) == "boom"
    finally:
        lease.close()


def test_unpickleable_result_becomes_a_typed_transport_error():
    """The queue's feeder thread would hang; pre-serialization must not."""

    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        future = pool.submit(_return_unpickleable)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        error = future.exception()
        assert isinstance(error, MeshError)
        assert "transported" in str(error)
    finally:
        lease.close()


def test_unpickleable_exception_becomes_a_typed_transport_error():
    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        future = pool.submit(_raise_unpickleable)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        error = future.exception()
        assert isinstance(error, MeshError)
        assert "exception that could not be transported" in str(error)
    finally:
        lease.close()


def test_unpickleable_argument_fails_promptly_and_keeps_the_pool_usable():
    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        with pytest.raises(MeshError, match="could not be serialized"):
            pool.submit(_add, lambda: "not picklable", 1)
        # No task reached a worker; the pool still serves picklable tasks.
        future = pool.submit(_add, 2, 3)
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        assert future.result() == 5
    finally:
        lease.close()


def test_crashed_worker_is_detected():
    lease = create_parallel_pool(2)
    try:
        pool = lease._pool
        pool.submit(_crash)
        deadline = time.monotonic() + 10.0
        while not pool.dead_workers() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        assert pool.dead_workers()
    finally:
        lease.close()


def test_lease_is_exclusive_and_invalidation_is_typed():
    lease = create_parallel_pool(1)
    try:
        pool = lease._acquire()
        assert lease.busy
        with pytest.raises(MeshError, match="exclusive"):
            lease._acquire()
        lease._release()
        assert not lease.busy and lease.valid
        lease._invalidate()
        with pytest.raises(MeshError, match="invalidated"):
            lease._acquire()
        assert not lease.valid
    finally:
        lease.close()


def test_concurrent_acquire_admits_exactly_one_job():
    lease = create_parallel_pool(1)
    try:
        outcomes: list[str] = []
        lock = threading.Lock()

        def attempt():
            try:
                lease._acquire()
            except MeshError:
                with lock:
                    outcomes.append("rejected")
            else:
                with lock:
                    outcomes.append("admitted")

        threads = [threading.Thread(target=attempt) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        assert outcomes.count("admitted") == 1
        assert outcomes.count("rejected") == 3
        lease._release()
    finally:
        lease.close()


def test_close_during_active_borrowing_is_rejected():
    """Caller-close must not terminate workers a running job depends on."""

    lease = create_parallel_pool(1)
    lease._acquire()
    try:
        with pytest.raises(MeshError, match="still running"):
            lease.close()
        assert lease.valid and lease.busy
    finally:
        lease._release()
        lease.close()
    assert lease.closed


def test_cancel_terminates_uncooperative_tasks_within_the_deadline():
    lease = create_parallel_pool(2)
    try:
        pool = lease._pool
        lease._acquire()
        futures = [pool.submit(_sleep, 30.0) for _ in range(2)]
        time.sleep(0.3)
        signatures = {pid: _pid_signature(pid) for pid in lease.worker_pids}
        started = time.monotonic()
        pool.set_cancel()
        pool.cancel_pending()
        evidence = pool.shutdown(terminate_first=True)
        elapsed = time.monotonic() - started
        assert elapsed < TERMINATION_DEADLINE_SECONDS + 1.0
        assert not evidence["survivors"]
        assert evidence["terminated"] or evidence["cooperative"]
        assert not any(
            _same_process(pid, signatures[pid]) for pid in lease.worker_pids
        )
        lease._invalidate()
        del futures
    finally:
        lease.close()


def test_cooperative_signal_reaches_the_running_task():
    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        lease._acquire()
        future = pool.submit(_cooperative, 30.0, cooperative=True)
        # Let the task start, then signal cancellation; the running task must
        # surface the shared cooperative signal as its future's exception.
        time.sleep(0.5)
        pool.set_cancel()
        deadline = time.monotonic() + 10.0
        while not future.done() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        assert isinstance(future.exception(), ParallelJobCancelled)
        pool.shutdown(terminate_first=True)
        lease._invalidate()
    finally:
        lease.close()


def test_descendants_die_with_the_pool():
    """A task's own child cannot outlive the pool that spawned the worker."""

    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        future = pool.submit(_spawn_grandchild)
        deadline = time.monotonic() + 20.0
        while not future.done() and time.monotonic() < deadline:
            pool.drain_results(0.1)
        grandchild = future.result()
        signature = _pid_signature(grandchild)
        assert _same_process(grandchild, signature)
    finally:
        lease.close()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and _same_process(grandchild, signature):
        time.sleep(0.1)
    assert not _same_process(grandchild, signature)


def _parent_death_controller(queue):
    from anymesher._parallel_pool import _ParallelPool

    pool = _ParallelPool(2, name="controller")
    future = pool.submit(_spawn_grandchild)
    deadline = time.monotonic() + 30.0
    while not future.done() and time.monotonic() < deadline:
        pool.drain_results(0.1)
    grandchild = future.result()
    pool.submit(_sleep, 30.0)  # a task is running when the owner dies
    queue.put(("ready", pool.worker_pids, grandchild))
    time.sleep(60)


def test_abrupt_parent_death_kills_workers_and_task_descendants():
    """Real lifecycle: SIGKILL-style owner death during a running task.

    The owner process is terminated with no cleanup, while one task runs and
    one task descendant sleeps. Every owned process must die through
    containment alone. On Windows this measures the kill-on-close Job
    Object; on POSIX the in-worker watchdog thread and the independent
    guardian implement the same contract.
    """

    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=_parent_death_controller, args=(result,))
    process.start()
    try:
        kind, worker_pids, grandchild = result.get(timeout=90)
        assert kind == "ready" and len(worker_pids) == 2
        signatures = {
            pid: _pid_signature(pid) for pid in (*worker_pids, grandchild)
        }
        assert all(
            _same_process(pid, signatures[pid]) for pid in (*worker_pids, grandchild)
        )
    finally:
        # Abrupt owner death: no atexit, no finally blocks, no cleanup.
        _kill_hard(process.pid)
        process.join(10)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and any(
        _same_process(pid, signatures[pid]) for pid in (*worker_pids, grandchild)
    ):
        time.sleep(0.1)
    assert not any(
        _same_process(pid, signatures[pid]) for pid in (*worker_pids, grandchild)
    )


def _worker_death_controller(queue):
    from anymesher._parallel_pool import _ParallelPool

    pool = _ParallelPool(1, name="controller")
    future = pool.submit(_spawn_grandchild)
    deadline = time.monotonic() + 30.0
    while not future.done() and time.monotonic() < deadline:
        pool.drain_results(0.1)
    grandchild = future.result()
    pool.submit(_sleep, 30.0)  # a task is running when the worker dies
    queue.put((
        "ready", pool.worker_pids[0], grandchild, pool._guardian.pid,
    ))
    time.sleep(60)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group guardian")
def test_abrupt_worker_hard_death_kills_the_task_descendants():
    """Real lifecycle: SIGKILL-style WORKER death during a running task.

    The worker's own watchdog thread and finally clause disappear with it,
    so only the independent guardian — holding the worker's death sentinel
    and the group id verified at startup — can kill the sleeping task
    descendant.  The pool owner stays alive throughout, so this is the
    worker-hard-death gap, separate from parent-death containment.
    """

    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=_worker_death_controller, args=(result,))
    process.start()
    try:
        kind, worker_pid, grandchild, guardian_pid = result.get(timeout=90)
        assert kind == "ready"
        signatures = {
            pid: _pid_signature(pid)
            for pid in (worker_pid, grandchild, guardian_pid)
        }
        assert all(_same_process(pid, signatures[pid]) for pid in signatures)
        # Abrupt worker death: no finally clause, no in-worker watchdog.
        _kill_hard(worker_pid)
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and _same_process(
            grandchild, signatures[grandchild]
        ):
            time.sleep(0.1)
        assert not _same_process(grandchild, signatures[grandchild])
    finally:
        _kill_hard(process.pid)
        process.join(10)
    # Released by the owner's death, the guardian sweeps and must not linger.
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and _same_process(
        guardian_pid, signatures[guardian_pid]
    ):
        time.sleep(0.1)
    assert not _same_process(guardian_pid, signatures[guardian_pid])


def _daemonic_child(queue):
    try:
        create_parallel_pool(1)
        queue.put(("ok", None))
    except MeshError as error:
        queue.put(("mesh-error", str(error)))
    except BaseException as error:  # pragma: no cover - failure detail
        queue.put(("other", repr(error)))


def test_daemonic_process_cannot_own_a_pool():
    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=_daemonic_child, args=(result,), daemon=True)
    process.start()
    process.join(60)
    kind, detail = result.get(timeout=10)
    assert kind == "mesh-error" and "daemonic" in detail


def test_create_parallel_pool_validates_worker_count():
    for bad in (0, -1, 62, 1000):
        with pytest.raises(ValueError):
            create_parallel_pool(bad)


def test_pool_start_failure_cleans_up_started_workers(monkeypatch):
    # A worker that dies before its startup handshake must not leak.
    monkeypatch.setenv("_ANYMESHER_PARALLEL_TEST_CRASH_AT_START", "1")
    try:
        with pytest.raises(MeshError, match="died during startup"):
            _ParallelPool(2, name="test")
    finally:
        monkeypatch.delenv("_ANYMESHER_PARALLEL_TEST_CRASH_AT_START", raising=False)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object containment")
def test_extended_limit_information_matches_the_sdk_abi():
    """Official SDK layout: exactly four SIZE_T fields after IoInfo.

    ``JobProcessMemoryLimit`` is NOT part of
    ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` (winnt.h); a structure that
    carries it misaligns every following field and arms garbage limits.
    On 64-bit Windows the official structure is 144 bytes.
    """

    structure = pool_module._JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    fields = [name for name, _ in structure._fields_]
    assert fields == [
        "BasicLimitInformation",
        "IoInfo",
        "ProcessMemoryLimit",
        "JobMemoryLimit",
        "PeakProcessMemoryUsed",
        "PeakJobMemoryUsed",
    ]
    assert "JobProcessMemoryLimit" not in fields
    assert all(
        getattr(structure, name).size == ctypes.sizeof(ctypes.c_size_t)
        for name in fields[2:]
    )
    assert ctypes.sizeof(structure) == 144
    # The information class stays the SDK's JobObjectExtendedLimitInformation.
    assert pool_module._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS == 9


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object containment")
def test_job_object_kill_on_close_limit_is_armed():
    """The SDK enum correction is real: information class 9 (the actual
    JobObjectExtendedLimitInformation) arms the kill-on-close limit that
    class 7 (the completion-port class) never could."""

    job, armed = pool_module._win_create_job()
    try:
        assert job is not None
        assert armed is True
    finally:
        pool_module._win_close_handle(job)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object containment")
def test_unarmed_job_object_fails_closed_before_any_spawn(monkeypatch):
    """Without the kernel kill-on-close limit there is no parent-death
    containment: the pool must fail typed before any process is spawned."""

    assigned: list[int] = []

    def record(job, pid):
        assigned.append(int(pid))
        return True

    monkeypatch.setattr(pool_module, "_win_create_job", lambda: (None, False))
    monkeypatch.setattr(pool_module, "_win_assign", record)
    with pytest.raises(ParallelContainmentUnavailable, match="kill-on-close"):
        _ParallelPool(2, name="test")
    assert not assigned, "no worker may be spawned when containment is unavailable"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Object containment")
def test_assignment_refusal_fails_closed_and_disposes_every_worker(monkeypatch):
    spawned: list[int] = []

    def refuse(job, pid):
        spawned.append(int(pid))
        return False

    monkeypatch.setattr(pool_module, "_win_assign", refuse)
    with pytest.raises(ParallelContainmentUnavailable, match="Job Object"):
        _ParallelPool(2, name="test")
    assert spawned, "the worker was spawned before assignment was refused"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(_pid_alive(pid) for pid in spawned):
        time.sleep(0.05)
    assert not any(_pid_alive(pid) for pid in spawned)
