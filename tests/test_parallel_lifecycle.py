"""Lifecycle contract tests for the parallel route (pool leases, fallbacks,
typed failures, cancellation, crash, diagnostics, progress).

The real-meshing tests here (anything that calls the mesh engine) must only
run through the check harness; the pure decision tests monkeypatch the serial
route so no meshing happens.
"""

from __future__ import annotations

import multiprocessing
import os
import sys

import pytest
from anygeometry import GeometryModel
from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    ParameterRange,
)

import anygeometry
from anymesher import (
    Mesh,
    ParallelOptions,
    create_parallel_pool,
    generate_hybrid_mesh_parallel,
)
from anymesher._parallel_pool import _ParallelPool, _pid_alive, _pid_signature
from anymesher.component_parallel import generate_hybrid_mesh_result_parallel
from anymesher.errors import MeshError

pytestmark = pytest.mark.skipif(
    not hasattr(anygeometry, "plan_independent_components"),
    reason="needs ANYgeometry with plan_independent_components",
)

TARGET = 0.25


def _same_process(pid, signature=None):
    if not _pid_alive(pid):
        return False
    current = _pid_signature(pid)
    return signature is None or current is None or current == signature


def _stiffened_plate(model: GeometryModel, x: float, y: float = 0.0) -> int:
    vertices = model.add_points(
        ((x, y, 0.0), (x + 2.0, y, 0.0), (x + 2.0, y + 1.0, 0.0), (x, y + 1.0, 0.0))
    )
    face = model.add_plate(vertices)
    model.add_sheet((face,))
    edge = model.faces[face].loop[0].edge
    member = model.add_member((OrientedEdge(edge, False),))
    model.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
    )
    return face


def _model(count: int, spacing: float) -> GeometryModel:
    model = GeometryModel()
    for index in range(count):
        _stiffened_plate(model, index * spacing)
    return model


@pytest.fixture(scope="module")
def lease():
    pool = create_parallel_pool(2)
    yield pool
    pool.close()


# ---------------------------------------------------------------------------
# decisions before dispatch (no meshing: the serial route is recorded)


def test_bare_executor_falls_back_to_serial_before_submission(monkeypatch):
    """A bare/shared/unknown executor is never submitted to."""

    import anymesher.component_parallel as module

    seen = []

    def record(geometry, options, why):
        seen.append(why)
        return "serial"

    monkeypatch.setattr(module, "_serial", record)
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0),
        target_size=TARGET,
        parallel=ParallelOptions(executor=object()),
    )
    assert result == "serial"
    assert seen == [module._EXECUTOR_FALLBACK_REASON]


def test_executor_and_lease_are_rejected_together():
    with pytest.raises(ValueError, match="not both"):
        generate_hybrid_mesh_result_parallel(
            _model(2, spacing=10.0),
            target_size=TARGET,
            parallel=ParallelOptions(executor=object(), pool_lease=object()),
        )


@pytest.mark.parametrize(
    "faces,members,edges,expected",
    [
        (None, None, None, (None, None, ())),
        ([], [], [], ((), (), ())),
        (
            (i for i in [2, 1]),
            (i for i in [7, 4]),
            (i for i in [8, 3]),
            ((1, 2), (4, 7), (3, 8)),
        ),
    ],
)
def test_public_refusal_preserves_one_shot_selection_semantics(
    monkeypatch, faces, members, edges, expected
):
    """Every refusal — executor fallback included — canonicalizes selections.

    Ported from the Root-owned public refusal selection probe: the serial
    fallback must see the same None vs explicit-empty faces/members and the
    canonical beamNone->() semantics as the parallel route, even when the
    caller passed one-shot iterables.
    """

    import anymesher.component_parallel as module

    observed = []

    def record(geometry, options, why):
        observed.append(
            (options["face_ids"], options["member_ids"], options["beam_edges"])
        )
        return "serial"

    monkeypatch.setattr(module, "_serial", record)
    result = generate_hybrid_mesh_result_parallel(
        GeometryModel(),
        target_size=TARGET,
        parallel=ParallelOptions(executor=object()),
        face_ids=faces,
        member_ids=members,
        beam_edges=edges,
    )
    assert result == "serial"
    assert observed == [expected]


def test_stale_binding_is_a_typed_failure_not_a_fallback(monkeypatch):
    """A partition that no longer binds the model raises, never falls back."""

    real = anygeometry.plan_independent_components

    def stale(model, **kwargs):
        partition = real(model, **kwargs)
        model.add_points(((100.0, 100.0, 0.0),))  # mutate after planning
        return partition

    monkeypatch.setattr(anygeometry, "plan_independent_components", stale)
    with pytest.raises(GeometryError):
        generate_hybrid_mesh_result_parallel(
            _model(2, spacing=10.0), target_size=TARGET
        )


def test_missing_binding_validator_falls_back_to_serial(monkeypatch):
    monkeypatch.delattr(anygeometry, "validate_component_partition_binding")
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0), target_size=TARGET
    )
    assert result.mesh.hybrid_diagnostics["parallel"] == {
        "used": False,
        "route": "serial",
        "reason": "ANYgeometry has no validate_component_partition_binding",
    }


def test_validate_binding_preserves_the_owner_error_object():
    """An owner binding failure surfaces as the owner's own error object."""

    from anymesher.component_parallel import _validate_binding

    owner_error = GeometryError("stale binding")
    phases = []

    def cancellation(phase):
        phases.append(phase)
        return False

    def validator(geometry, partition, expected_revision, cancellation_check):
        cancellation_check("component partition binding")
        raise owner_error

    with pytest.raises(GeometryError) as exc_info:
        _validate_binding(validator, _model(1, spacing=10.0), object(), cancellation)
    assert exc_info.value is owner_error
    # The callback ran once at the owner check and never again after the error.
    assert phases == ["component partition binding"]


def test_validate_binding_truthy_cancellation_raises_parallel_job_cancelled():
    """A truthy owner cancellation becomes ParallelJobCancelled at the check."""

    from anymesher._parallel_pool import ParallelJobCancelled
    from anymesher.component_parallel import _validate_binding

    def cancellation(phase):
        return True

    def validator(geometry, partition, expected_revision, cancellation_check):
        cancellation_check("component partition binding")

    with pytest.raises(ParallelJobCancelled) as exc_info:
        _validate_binding(validator, _model(1, spacing=10.0), object(), cancellation)
    assert "component partition binding" in str(exc_info.value)


def test_validate_binding_raised_callback_exception_propagates_unchanged():
    """A cancellation callback that raises is never wrapped or replaced."""

    from anymesher.component_parallel import _validate_binding

    class CallbackChannelFailed(RuntimeError):
        pass

    boom = CallbackChannelFailed("owner cancellation channel failed")

    def cancellation(phase):
        raise boom

    def validator(geometry, partition, expected_revision, cancellation_check):
        cancellation_check("component partition binding")

    with pytest.raises(CallbackChannelFailed) as exc_info:
        _validate_binding(validator, _model(1, spacing=10.0), object(), cancellation)
    assert exc_info.value is boom


def test_min_estimated_elements_runs_serial_below_the_threshold(monkeypatch):
    import anymesher.component_parallel as module

    seen = []

    def record(geometry, options, why):
        seen.append(why)
        return "serial"

    monkeypatch.setattr(module, "_serial", record)
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0),
        target_size=TARGET,
        parallel=ParallelOptions(min_estimated_elements=10 ** 6),
    )
    assert result == "serial"
    assert seen == ["estimated elements below min_estimated_elements"]


def test_unknown_estimate_is_explicit_serial_not_invented(monkeypatch):
    import anymesher.component_parallel as module

    seen = []

    def record(geometry, options, why):
        seen.append(why)
        return "serial"

    monkeypatch.setattr(module, "_serial", record)
    model = _model(2, spacing=10.0)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("no owner measurement")

    monkeypatch.setattr(model, "face_side_lengths", refuse)
    result = generate_hybrid_mesh_result_parallel(
        model,
        target_size=TARGET,
        parallel=ParallelOptions(min_estimated_elements=10),
    )
    assert result == "serial"
    assert seen == ["component element estimate unknown"]


def _daemonic_mesh_child(queue):
    from anymesher.component_parallel import (
        ParallelOptions,
        generate_hybrid_mesh_result_parallel,
    )

    model = GeometryModel()
    for index in range(2):
        _stiffened_plate(model, index * 10.0)
    outcome = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions()
    )
    queue.put(outcome.mesh.hybrid_diagnostics["parallel"])


def test_daemonic_caller_falls_back_to_serial_with_a_reason():
    """The GUI default (daemonic mesh worker) stays serial until ANYfem
    provides outer containment through a non-daemonic application worker."""

    ctx = multiprocessing.get_context("spawn")
    result = ctx.Queue()
    process = ctx.Process(target=_daemonic_mesh_child, args=(result,), daemon=True)
    process.start()
    process.join(120)
    info = result.get(timeout=30)
    assert info == {
        "used": False,
        "route": "serial",
        "reason": "worker processes unavailable (daemonic process)",
    }


# ---------------------------------------------------------------------------
# real-meshing lifecycle behaviour (run only through the check harness)


def test_progress_reports_component_k_of_n(lease):
    phases = []

    def check(phase):
        phases.append(phase)

    generate_hybrid_mesh_result_parallel(
        _model(3, spacing=10.0),
        target_size=TARGET,
        parallel=ParallelOptions(pool_lease=lease),
        cancellation_check=check,
    )
    assert "parallel dispatch" in phases
    assert "parallel component 3 of 3" in phases


def test_runtime_diagnostics_are_machine_dependent_and_hash_safe(lease):
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0),
        target_size=TARGET,
        parallel=ParallelOptions(pool_lease=lease),
    )
    info = result.mesh.hybrid_diagnostics["parallel"]
    assert set(info) == {
        "used", "route", "components", "separation", "merge_reasons", "runtime",
    }
    assert info["used"] is True and info["route"] == "parallel"
    runtime = info["runtime"]
    for key in (
        "pool", "workers", "worker_pids", "plan_seconds", "prepare_seconds",
        "mesh_seconds", "component_seconds", "component_estimated_elements",
        "worker_working_set_bytes", "join_seconds", "lifecycle",
    ):
        assert key in runtime, key
    assert runtime["pool"] == "leased"
    assert len(runtime["worker_pids"]) == runtime["workers"]
    assert all(isinstance(event, list) for event in runtime["lifecycle"])


def test_cold_pool_is_library_owned_and_disposed():
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0), target_size=TARGET
    )
    runtime = result.mesh.hybrid_diagnostics["parallel"]["runtime"]
    assert runtime["pool"] == "cold"
    assert not any(_pid_alive(pid) for pid in runtime["worker_pids"])


def test_mesh_wrapper_delegates_to_the_result(lease):
    mesh = generate_hybrid_mesh_parallel(
        _model(2, spacing=10.0),
        target_size=TARGET,
        parallel=ParallelOptions(pool_lease=lease),
    )
    assert isinstance(mesh, Mesh)
    assert mesh.hybrid_diagnostics["parallel"]["used"] is True


def test_cancellation_invalidates_the_lease_and_leaves_no_children():
    class Cancelled(RuntimeError):
        pass

    def stop(phase):
        # Cancel only once the job is actually running (first component done).
        if phase.startswith("parallel component") and phase != "parallel component 0 of 3":
            raise Cancelled

    pool = create_parallel_pool(2)
    try:
        with pytest.raises(Cancelled):
            generate_hybrid_mesh_result_parallel(
                _model(3, spacing=10.0),
                target_size=TARGET,
                parallel=ParallelOptions(pool_lease=pool),
                cancellation_check=stop,
            )
        assert not pool.valid
        assert not any(_pid_alive(pid) for pid in pool.worker_pids)
        with pytest.raises(MeshError, match="invalidated"):
            generate_hybrid_mesh_result_parallel(
                _model(3, spacing=10.0),
                target_size=TARGET,
                parallel=ParallelOptions(pool_lease=pool),
            )
    finally:
        pool.close()


def test_successful_lease_stays_reusable(lease):
    for _ in range(2):
        result = generate_hybrid_mesh_result_parallel(
            _model(2, spacing=10.0),
            target_size=TARGET,
            parallel=ParallelOptions(pool_lease=lease),
        )
        assert result.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert lease.valid and not lease.busy


# ---------------------------------------------------------------------------
# cleanup evidence, worker controls and estimates


def test_cold_cleanup_failure_publishes_nothing(monkeypatch):
    """Survivors of the cold disposal deadline fail the job; no mesh appears."""

    original = _ParallelPool.shutdown

    def incomplete(self, **kwargs):
        evidence = original(self, **kwargs)
        evidence["survivors"] = list(evidence.get("survivors", [])) + [424242]
        return evidence

    monkeypatch.setattr(_ParallelPool, "shutdown", incomplete)
    with pytest.raises(MeshError, match="cleanup incomplete"):
        generate_hybrid_mesh_result_parallel(
            _model(2, spacing=10.0), target_size=TARGET
        )


def test_cancellation_identity_survives_a_cleanup_failure(monkeypatch):
    """A failing cleanup attaches evidence but never masks the cancellation."""

    class Cancelled(RuntimeError):
        pass

    original = _ParallelPool.shutdown

    def failing(self, **kwargs):
        original(self, **kwargs)  # the workers really are cleaned up
        raise RuntimeError("cleanup boom")

    monkeypatch.setattr(_ParallelPool, "shutdown", failing)

    def stop(phase):
        if phase.startswith("parallel component"):
            raise Cancelled

    lease = create_parallel_pool(2)
    try:
        with pytest.raises(Cancelled) as caught:
            generate_hybrid_mesh_result_parallel(
                _model(2, spacing=10.0),
                target_size=TARGET,
                parallel=ParallelOptions(pool_lease=lease),
                cancellation_check=stop,
            )
        notes = getattr(caught.value, "__notes__", [])
        assert any("cleanup itself failed" in note for note in notes)
        # Invalidation stays visible even though cleanup failed.
        assert not lease.valid
        assert not any(_pid_alive(pid) for pid in lease.worker_pids)
    finally:
        monkeypatch.undo()
        lease.close()


def test_runtime_reports_actual_worker_controls(lease):
    """Diagnostics report measured priority/thread facts, not requests."""

    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0),
        target_size=TARGET,
        parallel=ParallelOptions(pool_lease=lease),
    )
    runtime = result.mesh.hybrid_diagnostics["parallel"]["runtime"]
    controls = runtime["worker_controls"]
    assert set(controls) == set(range(runtime["workers"]))
    for info in controls.values():
        assert info["priority_requested"] == "below_normal"
        if sys.platform == "win32":
            assert info["priority_applied"] is True
            assert info["priority_class"] == "below_normal"
        expected_env = {
            name: os.environ.get(name, "1")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        }
        assert info["thread_env"] == expected_env
        # Actual per-library thread counts from runtime introspection in the
        # worker; None means the counts are unknown, never that they are safe.
        threadpool = info["threadpool"]
        if threadpool is not None:
            assert isinstance(threadpool, list)
            for entry in threadpool:
                assert isinstance(entry["num_threads"], int)
                assert entry["num_threads"] >= 1
                envvar = entry.get("envvar")
                if envvar in expected_env and expected_env[envvar] == "1":
                    assert entry["num_threads"] == 1


def test_member_only_component_estimate_is_measured_not_invented():
    """A standalone member component is estimated, never reported as 0."""

    from anymesher.component_parallel import _component_estimates

    model = _model(1, spacing=10.0)
    member_id = next(iter(model.members))

    class _MemberOnly:
        face_ids = ()
        member_ids = (member_id,)
        edge_ids = ()

    class _Partition:
        components = (_MemberOnly(),)

    estimates, basis = _component_estimates(model, _Partition(), {}, TARGET)
    assert basis == ["heuristic"]
    assert estimates[0] is not None and estimates[0] > 0


def test_unmeasurable_component_estimate_stays_unknown():
    """No owner-measured geometry means unknown, never an invented 0."""

    from anymesher.component_parallel import _component_estimates

    model = _model(1, spacing=10.0)

    class _Empty:
        face_ids = ()
        member_ids = ()
        edge_ids = ()

    class _Partition:
        components = (_Empty(),)

    estimates, basis = _component_estimates(model, _Partition(), {}, TARGET)
    assert estimates == [None]
    assert basis == ["unknown"]

    member_id = next(iter(model.members))

    class _MemberOnly:
        face_ids = ()
        member_ids = (member_id,)
        edge_ids = ()

    class _Refusing:
        components = (_MemberOnly(),)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("no owner measurement")

    model.edge_length = refuse
    estimates, basis = _component_estimates(model, _Refusing(), {}, TARGET)
    assert estimates == [None]
    assert basis == ["unknown"]
