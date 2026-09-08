"""Canonical snapshot reuse must reproduce the full mutable-topology export."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anymesher import native_v2
from anymesher._t3_incidence import T3IncidenceIndex
from anymesher import _t3_canonical_export as cached
from anymesher.errors import MeshError


def grid(width=4):
    points = np.array([(x / width, y / width) for y in range(width + 1)
                       for x in range(width + 1)])
    rows = []
    for y in range(width):
        for x in range(width):
            a = y * (width + 1) + x
            b, d = a + 1, a + width + 1
            rows.extend(((a, b, d + 1), (a, d + 1, d)))
    return points, np.asarray(rows, dtype=np.int64)


def oracle(points, rows):
    return np.asarray(sorted(native_v2._canonical_triangle(row, points)
                             for row in rows), dtype=np.int64).reshape(-1, 3)


def test_initial_and_unchanged_snapshots_do_not_recanonicalize(monkeypatch):
    points, rows = grid()
    topology = native_v2.MutableT3Topology(points, rows, ())
    expected = topology.canonical_export()
    calls = []
    original = native_v2._canonical_triangle
    def traced(cell, coordinates):
        calls.append(tuple(cell))
        return original(cell, coordinates)
    monkeypatch.setattr(native_v2, "_canonical_triangle", traced)
    first = cached.canonical_frontal_export(topology)
    snapshot = topology._topology_index._canonical_export_snapshot
    assert len(calls) == len(rows)
    for _ in range(8):
        actual = cached.canonical_frontal_export(topology)
        assert topology._topology_index._canonical_export_snapshot is snapshot
        for result, reference in zip(actual, expected):
            assert result.tobytes() == reference.tobytes()
    assert len(calls) == len(rows)
    with pytest.raises(ValueError, match="read-only"):
        first[0][:] = 99
    first[1][:] = 0
    actual = cached.canonical_frontal_export(topology)
    for result, reference in zip(actual, expected):
        assert result.tobytes() == reference.tobytes()


def test_only_added_cells_are_canonicalized_after_insertion(monkeypatch):
    points, rows = grid()
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    previous = topology._topology_index._canonical_export_snapshot
    previous_bytes = previous.rows.tobytes()
    report = topology.insert_point((.18, .13))
    expected = topology.canonical_export()
    calls = []
    original = native_v2._canonical_triangle
    def traced(cell, coordinates):
        calls.append(tuple(cell))
        return original(cell, coordinates)
    monkeypatch.setattr(native_v2, "_canonical_triangle", traced)
    actual = cached.canonical_frontal_export(topology)
    assert len(calls) == report["added_triangles"]
    assert len(calls) < len(topology.triangles)
    assert previous.rows.tobytes() == previous_bytes
    assert not previous.rows.flags.writeable
    assert not previous.points.flags.writeable
    with pytest.raises(TypeError):
        previous.by_cell[(0, 1, 2)] = (0, 1, 2)
    for result, reference in zip(actual, expected):
        assert result.tobytes() == reference.tobytes()


def test_reorder_orientation_and_coordinate_changes_match_reference():
    points, rows = grid()
    index = T3IncidenceIndex(rows)
    topology = SimpleNamespace(points=points.copy(), _topology_index=index)
    for iteration in range(12):
        ordered = rows[::-1].copy() if iteration % 2 else rows.copy()
        ordered = np.roll(ordered, iteration % 3, axis=1)
        if iteration % 4 == 1:
            ordered = ordered[:, ::-1]
        topology._topology_index = topology._topology_index.updated(ordered)
        if iteration == 5:
            topology.points[:, 0] *= -1
        if iteration == 9:
            topology.points[0, 0] = -0.0
        exported_points, actual = cached.canonical_frontal_export(topology)
        assert actual.tobytes() == oracle(topology.points, ordered).tobytes()
        assert exported_points.tobytes() == topology.points.tobytes()


def test_append_delete_to_empty_then_restore_exactly():
    points, rows = grid(1)
    topology = SimpleNamespace(points=points.copy(),
                               _topology_index=T3IncidenceIndex(rows))
    for changed in (rows, rows[:1], np.empty((0, 3), dtype=np.int64), rows):
        topology._topology_index = topology._topology_index.updated(changed)
        _, actual = cached.canonical_frontal_export(topology)
        assert actual.shape == (len(changed), 3)
        assert actual.tobytes() == oracle(topology.points, changed).tobytes()
    topology.points = np.vstack((points, (.7, .2)))
    changed = np.asarray(((0, 1, 4), (1, 3, 4), (0, 4, 3), (0, 3, 2)))
    topology._topology_index = topology._topology_index.updated(changed)
    _, actual = cached.canonical_frontal_export(topology)
    assert actual.tobytes() == oracle(topology.points, changed).tobytes()


@pytest.mark.parametrize("phase", ["prepare", "cells", "commit"])
def test_cancelled_export_keeps_source_snapshot_and_geometry(phase):
    points, rows = grid()
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    previous = topology._topology_index._canonical_export_snapshot
    topology.insert_point((.18, .13))
    before = topology.canonical_export()
    error = RuntimeError("cancel canonical export")
    def cancel(value):
        if value == "native-v2 canonical export " + phase:
            raise error
    with pytest.raises(RuntimeError) as caught:
        cached.canonical_frontal_export(topology, cancellation_check=cancel)
    assert caught.value is error
    assert topology._topology_index._canonical_export_snapshot is previous
    for result, reference in zip(topology.canonical_export(), before):
        assert result.tobytes() == reference.tobytes()


@pytest.mark.parametrize("defect", ["negative", "range", "duplicate", "nonfinite"])
def test_bad_snapshot_does_not_commit(defect):
    points, rows = grid(1)
    index = T3IncidenceIndex(rows)
    topology = SimpleNamespace(points=points.copy(), _topology_index=index)
    cached.canonical_frontal_export(topology)
    previous = index._canonical_export_snapshot
    bad = dict(index._row_by_cell)
    if defect == "negative":
        bad[(-1, 1, 3)] = 9
    elif defect == "range":
        bad[(0, 1, 99)] = 9
    elif defect == "duplicate":
        bad[(1, 3, 0)] = 9
    else:
        topology.points[0, 0] = np.nan
    topology._topology_index = SimpleNamespace(
        _row_by_cell=bad, _canonical_export_snapshot=previous)
    with pytest.raises(MeshError):
        cached.canonical_frontal_export(topology)
    assert topology._topology_index._canonical_export_snapshot is previous


@pytest.mark.parametrize("backend", ["python", "native"])
def test_public_cylinder_full_export_parity(monkeypatch, backend):
    from anygeometry import to_dict
    from anymesher.hybrid import generate_hybrid_mesh_result, _neutral_shell_core
    from anymesher.quality_v2 import assert_valid_mesh

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    bench = importlib.import_module("native_v2_baseline")
    fixture = bench.cylinder_case("cylinder_patch")
    before = bench._digest_contract(to_dict(fixture.model))
    target = .07414706429645167
    def generate():
        return generate_hybrid_mesh_result(
            fixture.model, face_ids=fixture.face_ids, target_size=target,
            strategy="native", overrides=fixture.overrides(target),
            refinements=fixture.refinements(target), recombine=True,
            native_backend=backend,
            native_options=fixture.options(target, "frontal", 1000),
        )
    incremental = generate()
    def reference(topology, *, cancellation_check=None):
        return topology.canonical_export()
    monkeypatch.setattr(cached, "canonical_frontal_export", reference)
    full = generate()
    for result in (incremental, full):
        assert_valid_mesh(_neutral_shell_core(result.mesh))
        work = bench._native_work(
            {"faces": {str(k): dict(v) for k, v in
                       result.triangulation_backend_by_face.items()}},
            "frontal", fixture.name)
        assert work["published_insertions"] > 0
    assert bench._mesh_digest(_neutral_shell_core(incremental.mesh)) == (
        bench._mesh_digest(_neutral_shell_core(full.mesh)))
    assert bench._digest_contract(fixture.mesh_contract(incremental.mesh, target)) == (
        bench._digest_contract(fixture.mesh_contract(full.mesh, target)))
    assert incremental.triangulation_backend_by_face[1]["native_v2"] == (
        full.triangulation_backend_by_face[1]["native_v2"])
    assert bench._digest_contract(to_dict(fixture.model)) == before

def test_repeated_unchanged_export_has_constant_snapshot_ownership():
    points, rows = grid(1)
    topology = native_v2.MutableT3Topology(points, rows, ())
    cached.canonical_frontal_export(topology)
    snapshot = topology._topology_index._canonical_export_snapshot
    for _ in range(2048):
        cached.canonical_frontal_export(topology)
        assert topology._topology_index._canonical_export_snapshot is snapshot
    np.testing.assert_array_equal(snapshot.rows, oracle(points, rows))


def test_unused_appended_points_share_mapping_without_proxy_chains():
    points, rows = grid(1)
    topology = SimpleNamespace(points=points, _topology_index=T3IncidenceIndex(rows))
    cached.canonical_frontal_export(topology)
    first = topology._topology_index._canonical_export_snapshot
    topology.points = np.vstack((points, (2., 2.)))
    exported, cells = cached.canonical_frontal_export(topology)
    second = topology._topology_index._canonical_export_snapshot
    assert second is not first
    assert second.by_cell is first.by_cell
    assert second.rows is first.rows
    assert len(exported) == 5
    assert len(first.points) == 4
    assert cells.tobytes() == first.rows.tobytes()

