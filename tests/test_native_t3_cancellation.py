"""Behavioral checks for cancellable native insertion, without mesh publication."""
import numpy as np
import pytest

from anymesher import native_cpp, native_v2
from anymesher.errors import MeshError


def require_kernel():
    if not native_cpp.COMPILED_NATIVE_V2_AVAILABLE:
        pytest.skip("optional native-v2 extension is absent")
    assert callable(getattr(native_cpp._compiled,
                            "native_v2_mutable_t3_insert_cancellable", None)), (
        "rebuild the present native-v2 extension for cancellable insertion"
    )


def grid():
    n = 49
    points = np.array([(x / n, y / n)
                       for y in range(n + 1) for x in range(n + 1)])
    cells = []
    edges = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            b, d = a + 1, a + n + 1
            cells.extend(((a, b, d + 1), (a, d + 1, d)))
    for i in range(n):
        edges.extend(((i, i + 1),
                      (n * (n + 1) + i, n * (n + 1) + i + 1),
                      (i * (n + 1), (i + 1) * (n + 1)),
                      (i * (n + 1) + n, (i + 1) * (n + 1) + n)))
    return points, np.array(cells, dtype=np.int64), edges


def topology():
    points, cells, edges = grid()
    return native_v2.MutableT3Topology(
        points, cells, edges,
        triangle_owners=np.arange(len(cells), dtype=np.int64) % 7,
    )


POINT = (0.371923, 0.229573)


def state(mesh):
    return (mesh.points, mesh.triangles, mesh.node_owners.copy(),
            mesh.triangle_owners.copy(), mesh._topology_index, mesh.epoch)


def unchanged(mesh, before):
    for actual, expected in zip(
        (mesh.points, mesh.triangles, mesh.node_owners, mesh.triangle_owners),
        before[:4],
    ):
        np.testing.assert_array_equal(actual, expected)
    assert mesh._topology_index is before[4]
    assert mesh.epoch == before[5]


def test_large_callback_uses_native_and_matches_python_owners(monkeypatch):
    require_kernel()
    mesh = topology()
    oracle_rows, oracle_owners, oracle_report = mesh._python_insert_with_owners(
        np.array(POINT), owner=42,
    )
    phases = []

    def forbidden(*args, **kwargs):
        raise AssertionError("large cancellable insertion re-entered the oracle")

    monkeypatch.setattr(native_v2.MutableT3Topology,
                        "_python_insert_with_owners", forbidden)
    report = mesh.insert_point(POINT, owner=42, cancellation_check=phases.append)
    assert report["native"] is True
    np.testing.assert_array_equal(mesh.triangles, oracle_rows)
    np.testing.assert_array_equal(mesh.triangle_owners, oracle_owners)
    assert report["removed_triangles"] == oracle_report["removed_triangles"]
    assert report["added_triangles"] == oracle_report["added_triangles"]
    assert phases.count("native-v2 compiled insertion work") > 1
    assert mesh.epoch == 1


@pytest.mark.parametrize("stage", [
    "prepare", "binding", "points", "cells", "protected", "work", "output",
])
def test_cancel_at_native_phase_keeps_source_and_exception(stage):
    require_kernel()
    mesh = topology()
    before = state(mesh)
    error = RuntimeError("host cancellation")
    seen = []

    def cancel(phase):
        seen.append(phase)
        if phase == "native-v2 compiled insertion " + stage:
            raise error

    with pytest.raises(RuntimeError) as caught:
        mesh.insert_point(POINT, cancellation_check=cancel)
    assert caught.value is error
    assert "native-v2 compiled insertion " + stage in seen
    unchanged(mesh, before)


@pytest.mark.parametrize("stage", [
    "identity validation", "orientation validation", "incidence validation",
])
def test_cancel_after_native_export_keeps_source(stage):
    require_kernel()
    mesh = topology()
    before = state(mesh)

    def cancel(phase):
        if phase == "native-v2 insertion result " + stage:
            raise KeyboardInterrupt("cancel validation")

    with pytest.raises(KeyboardInterrupt, match="cancel validation"):
        mesh.insert_point(POINT, cancellation_check=cancel)
    unchanged(mesh, before)


def test_late_native_work_cancellation_preserves_retained_snapshot():
    require_kernel()
    mesh = topology()
    before = state(mesh)
    visits = 0

    def cancel(phase):
        nonlocal visits
        if phase == "native-v2 compiled insertion work":
            visits += 1
            if visits == 3:
                raise RuntimeError("cancel after native work")

    with pytest.raises(RuntimeError, match="cancel after native work"):
        mesh.insert_point(POINT, cancellation_check=cancel)
    assert visits == 3
    unchanged(mesh, before)


@pytest.mark.parametrize("prefix", [
    "ANYMESHER_NATIVE_V2_PREDICATE_UNCERTAIN:",
    "ANYMESHER_NATIVE_V2_GEOMETRY_LIMITED:",
])
def test_callback_error_cannot_be_reclassified_as_predicate_fallback(prefix):
    require_kernel()
    mesh = topology()
    before = state(mesh)
    error = RuntimeError(prefix + "callback error")

    def cancel(phase):
        if phase == "native-v2 compiled insertion work":
            raise error

    with pytest.raises(RuntimeError) as caught:
        mesh.insert_point(POINT, cancellation_check=cancel)
    assert caught.value is error
    unchanged(mesh, before)


def test_older_extension_keeps_cancellable_python_reference(monkeypatch):
    require_kernel()
    mesh = topology()
    before = state(mesh)
    monkeypatch.delattr(native_cpp._compiled,
                        "native_v2_mutable_t3_insert_cancellable")

    def cancel(phase):
        if phase != "native-v2 mutable insertion start":
            raise RuntimeError("old extension cancellation")

    with pytest.raises(RuntimeError, match="old extension cancellation"):
        mesh.insert_point(POINT, cancellation_check=cancel)
    unchanged(mesh, before)


def test_present_broken_kernel_never_becomes_absence(monkeypatch):
    require_kernel()
    mesh = topology()
    before = state(mesh)

    def broken(*args):
        raise RuntimeError("kernel fault")

    monkeypatch.setattr(native_cpp._compiled,
                        "native_v2_mutable_t3_insert_cancellable", broken)
    with pytest.raises(RuntimeError, match="kernel fault"):
        mesh.insert_point(POINT, cancellation_check=lambda phase: None)
    unchanged(mesh, before)


def test_malformed_callback_is_rejected():
    require_kernel()
    points, cells, edges = grid()
    with pytest.raises(TypeError, match="cancellation must be callable"):
        native_cpp.native_mutable_t3_insert(
            points, cells, edges, POINT, cancellation_check=False,
        )


def test_malformed_capability_is_fatal(monkeypatch):
    require_kernel()
    mesh = topology()
    before = state(mesh)
    monkeypatch.setattr(native_cpp._compiled,
                        "native_v2_mutable_t3_insert_cancellable", False)
    with pytest.raises(MeshError, match="capability is malformed"):
        mesh.insert_point(POINT, cancellation_check=lambda phase: None)
    unchanged(mesh, before)
