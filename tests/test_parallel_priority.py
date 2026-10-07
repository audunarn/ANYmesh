"""Focused worker priority tests: POSIX nice floor, dispatcher, warm reuse.

Pure-mock POSIX cases run on every platform by mocking the ``os`` priority
attributes directly (never ``os.name``/``sys.platform``).  Actual spawn and
warm-reuse cases assert only what the native platform can measure: Windows
priority-class semantics run here, the POSIX/WSL nice-floor cases are left
to Root's coordinated Linux venv and skip on Windows.  No native meshing
engine is involved, and the parent process priority is never lowered by the
tests themselves (mocks only, plus read-only introspection).
"""

import os
import json
import sys
import time

import pytest

from anymesher._parallel_pool import (
    _apply_worker_priority,
    _posix_apply_nice_floor,
    _priority_name,
    create_parallel_pool,
)

NICE_FLOOR = 5


class _FakePriority:
    """Scripted os.getpriority/os.setpriority pair for pure-mock cases."""

    def __init__(self, readings, set_error=None):
        self._readings = list(readings)
        self._set_error = set_error
        self.set_calls = []

    def getpriority(self, which, who):
        value = self._readings.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def setpriority(self, which, who, nice):
        if self._set_error is not None:
            raise self._set_error
        self.set_calls.append((which, who, nice))


def _install(monkeypatch, fake):
    monkeypatch.setattr(os, "getpriority", fake.getpriority, raising=False)
    monkeypatch.setattr(os, "setpriority", fake.setpriority, raising=False)
    monkeypatch.setattr(os, "PRIO_PROCESS", 0, raising=False)


def test_posix_floor_from_zero_measures_readback(monkeypatch):
    fake = _FakePriority([0, NICE_FLOOR])
    _install(monkeypatch, fake)
    applied, priority_class, controls = _posix_apply_nice_floor()
    assert applied is True
    assert priority_class is None
    assert controls["mechanism"] == "posix_nice_floor"
    assert controls["status"] == "applied"
    assert controls["error"] is None
    assert controls["nice_before"] == 0
    assert controls["nice_requested"] == NICE_FLOOR
    assert controls["nice_effective"] == NICE_FLOOR
    assert fake.set_calls == [(0, 0, NICE_FLOOR)]


@pytest.mark.parametrize("inherited", [NICE_FLOOR, 7, 10, 19])
def test_posix_already_lowered_never_raises_or_sets(monkeypatch, inherited):
    fake = _FakePriority([inherited])
    _install(monkeypatch, fake)
    applied, _, controls = _posix_apply_nice_floor()
    assert applied is True
    assert controls["status"] == "already_lowered"
    assert controls["nice_before"] == inherited
    assert controls["nice_requested"] == inherited
    assert controls["nice_effective"] == inherited
    assert fake.set_calls == []


def test_posix_repeated_calls_do_not_accumulate(monkeypatch):
    fake = _FakePriority([0, NICE_FLOOR, NICE_FLOOR, NICE_FLOOR])
    _install(monkeypatch, fake)
    first = _posix_apply_nice_floor()
    second = _posix_apply_nice_floor()
    third = _posix_apply_nice_floor()
    assert first[0] and second[0] and third[0]
    assert first[2]["status"] == "applied"
    assert second[2]["status"] == "already_lowered"
    assert third[2]["status"] == "already_lowered"
    assert all(c["nice_effective"] == NICE_FLOOR for _, _, c in (first, second, third))
    assert fake.set_calls == [(0, 0, NICE_FLOOR)]


def test_posix_missing_getpriority_api_is_explicit(monkeypatch):
    monkeypatch.setattr(os, "getpriority", None, raising=False)
    monkeypatch.setattr(os, "PRIO_PROCESS", 0, raising=False)
    applied, priority_class, controls = _posix_apply_nice_floor()
    assert applied is False
    assert priority_class is None
    assert controls["status"] == "unsupported"
    assert controls["error"]
    assert controls["nice_before"] is None
    assert controls["nice_requested"] is None
    assert controls["nice_effective"] is None


def test_posix_missing_setpriority_api_is_explicit(monkeypatch):
    fake = _FakePriority([0])
    monkeypatch.setattr(os, "getpriority", fake.getpriority, raising=False)
    monkeypatch.setattr(os, "setpriority", None, raising=False)
    monkeypatch.setattr(os, "PRIO_PROCESS", 0, raising=False)
    applied, _, controls = _posix_apply_nice_floor()
    assert applied is False
    assert controls["status"] == "unsupported"
    assert controls["error"]
    assert controls["nice_before"] == 0
    assert controls["nice_requested"] == NICE_FLOOR
    assert controls["nice_effective"] is None


def test_posix_permission_denied_is_explicit(monkeypatch):
    fake = _FakePriority([0], set_error=PermissionError(1, "not entitled"))
    _install(monkeypatch, fake)
    applied, _, controls = _posix_apply_nice_floor()
    assert applied is False
    assert controls["status"] == "permission_denied"
    assert controls["error"]
    assert controls["nice_effective"] is None


def test_posix_readback_failure_is_explicit(monkeypatch):
    fake = _FakePriority([0, OSError("readback gone")])
    _install(monkeypatch, fake)
    applied, _, controls = _posix_apply_nice_floor()
    assert applied is False
    assert controls["status"] == "readback_failed"
    assert controls["error"]
    assert controls["nice_effective"] is None
    assert fake.set_calls == [(0, 0, NICE_FLOOR)]


def test_posix_readback_below_floor_is_not_verified(monkeypatch):
    fake = _FakePriority([0, 3])
    _install(monkeypatch, fake)
    applied, _, controls = _posix_apply_nice_floor()
    assert applied is False
    assert controls["status"] == "not_verified"
    assert controls["error"]
    assert controls["nice_effective"] == 3


def test_posix_initial_read_error_is_explicit(monkeypatch):
    fake = _FakePriority([OSError("no such measurement")])
    _install(monkeypatch, fake)
    applied, _, controls = _posix_apply_nice_floor()
    assert applied is False
    assert controls["status"] == "error"
    assert controls["error"]
    assert controls["nice_before"] is None
    assert fake.set_calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows dispatcher branch")
def test_dispatcher_keeps_windows_legacy_fields(monkeypatch):
    import anymesher._parallel_pool as pool_module

    calls = []

    def fake_win():
        calls.append(1)
        return True, 0x4000

    monkeypatch.setattr(pool_module, "_win_apply_below_normal_priority", fake_win)
    applied, priority_class, controls = _apply_worker_priority()
    assert calls == [1]
    assert applied is True
    assert priority_class == 0x4000
    assert _priority_name(priority_class) == "below_normal"
    assert controls["mechanism"] == "windows_priority_class"
    assert controls["status"] == "applied"
    assert controls["error"] is None
    assert controls["nice_before"] is None
    assert controls["nice_requested"] is None
    assert controls["nice_effective"] is None

    monkeypatch.setattr(
        pool_module, "_win_apply_below_normal_priority", lambda: (False, None)
    )
    applied, priority_class, controls = _apply_worker_priority()
    assert applied is False
    assert priority_class is None
    assert controls["status"] == "failed"
    assert controls["error"]


def _run_two_warm_jobs(pool):
    """Two jobs on the same warm pool; returns per-job (pid, nice) probes."""

    results = []
    for _ in range(2):
        pool.begin_job()
        pid_future = pool.submit(os.getpid)
        nice_future = (
            pool.submit(os.getpriority, os.PRIO_PROCESS, 0)
            if os.name == "posix"
            else None
        )
        deadline = time.monotonic() + 15.0
        futures = [pid_future] + ([] if nice_future is None else [nice_future])
        while not all(future.done() for future in futures) and time.monotonic() < deadline:
            pool.drain_results(min(0.1, max(0.0, deadline - time.monotonic())))
        assert pid_future.done(), "the worker pid probe did not return"
        pid = pid_future.result()
        nice = None
        if nice_future is not None:
            assert nice_future.done(), "the worker nice probe did not return"
            nice = nice_future.result()
        pool.end_job()
        results.append((pid, nice))
    return results


@pytest.mark.skipif(sys.platform != "win32", reason="Windows native runtime")
def test_windows_worker_priority_class_and_warm_reuse():
    """Real 1-worker pool: legacy below-normal class, new control fields,
    unchanged parent priority class, same warm PID across two jobs."""

    import anymesher._parallel_pool as pool_module

    parent_before = int(pool_module._kernel32.GetPriorityClass(
        pool_module._kernel32.GetCurrentProcess()
    ))
    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        info = pool.worker_startup_info()
        assert set(info) == {0}
        worker = info[0]
        assert worker["priority_applied"] is True
        assert _priority_name(worker["priority_class"]) == "below_normal"
        controls = worker["priority_controls"]
        assert controls["mechanism"] == "windows_priority_class"
        assert controls["status"] == "applied"
        assert controls["error"] is None
        assert controls["nice_before"] is None
        assert controls["nice_requested"] is None
        assert controls["nice_effective"] is None
        parent_after = int(pool_module._kernel32.GetPriorityClass(
            pool_module._kernel32.GetCurrentProcess()
        ))
        assert parent_after == parent_before
        (pid1, _), (pid2, _) = _run_two_warm_jobs(pool)
        assert pid1 == pid2 == pool.worker_pids[0]
    finally:
        lease.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX/WSL native runtime")
def test_posix_worker_nice_floor_and_warm_reuse():
    """Real 1-worker pool on POSIX: measured nice floor 5 at startup, same
    warm PID and no nice accumulation across two jobs, parent untouched."""

    parent_before = os.getpriority(os.PRIO_PROCESS, 0)
    lease = create_parallel_pool(1)
    try:
        pool = lease._pool
        info = pool.worker_startup_info()
        assert set(info) == {0}
        worker = info[0]
        controls = worker["priority_controls"]
        assert worker["priority_applied"] is True
        assert worker["priority_class"] is None
        assert controls["mechanism"] == "posix_nice_floor"
        assert controls["status"] in ("applied", "already_lowered")
        assert controls["nice_effective"] >= NICE_FLOOR
        assert controls["nice_requested"] == max(
            controls["nice_before"], NICE_FLOOR
        )
        assert os.getpriority(os.PRIO_PROCESS, 0) == parent_before
        (pid1, nice1), (pid2, nice2) = _run_two_warm_jobs(pool)
        assert pid1 == pid2 == pool.worker_pids[0]
        assert nice1 >= NICE_FLOOR
        assert nice2 >= NICE_FLOOR
        assert nice2 == nice1
        assert nice1 == controls["nice_effective"]
        assert nice2 == controls["nice_effective"]
        print("priority_evidence=" + json.dumps({
            "parent_before": parent_before,
            "parent_after": os.getpriority(os.PRIO_PROCESS, 0),
            "startup": controls,
            "warm_jobs": [(pid1, nice1), (pid2, nice2)],
        }))
    finally:
        lease.close()
