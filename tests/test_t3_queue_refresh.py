"""Incremental queue refresh compared with the full-refresh reference."""
import importlib
from pathlib import Path
import time

import numpy as np
import pytest

from anymesher._t3_incidence import T3IncidenceIndex
from anymesher._t3_queue_refresh import QueueRefreshIndex
from anymesher._t3_runtime import TriangleWorkQueue
from anymesher.errors import MeshError


def grid(flips=(), width=4):
    points = np.array([(x, y) for y in range(width + 1)
                       for x in range(width + 1)], dtype=float)
    rows = []
    for y in range(width):
        for x in range(width):
            a = y * (width + 1) + x
            b, d = a + 1, a + width + 1
            c = d + 1
            rows.extend(((a, b, d), (b, c, d)) if y * width + x in flips
                        else ((a, b, c), (a, c, d)))
    return points, np.asarray(rows, dtype=np.int64)


def metrics(points):
    return np.repeat((2. * np.eye(2))[None], len(points), axis=0)


def full_round(self, index, points, tensors, protected):
    self.begin_round()
    return np.asarray(sorted(index._row_by_cell), dtype=np.int64).reshape(-1, 3)


def refresh(queue, index, points, tensors, protected, incremental):
    rows = (queue.prepare_incremental_round(index, points, tensors, protected)
            if incremental else full_round(queue, index, points, tensors, protected))
    for row in rows:
        cell = tuple(map(int, row))
        geometry = index.geometry(points, cell)
        local = tensors[row]
        edges = {tuple(sorted((a, b))) for a, b in
                 zip(cell, (*cell[1:], cell[0]))}
        corner = bool(edges & protected)
        severity = float(np.trace(local, axis1=1, axis2=2).mean())
        queue.refresh(cell, geometry, local, corner,
                      lambda: (severity, corner and severity > 8.))
    queue.finish_round()
    return tuple(map(tuple, rows))


def assert_same(a, b):
    assert {k: v[:3] for k, v in a._active.items()} == {
        k: v[:3] for k, v in b._active.items()}
    assert a._records.keys() == b._records.keys()
    for cell, record in a._records.items():
        other = b._records[cell]
        assert (record.stamp, record.protected_corner, record.severity,
                record.geometry_limited) == (
                    other.stamp, other.protected_corner, other.severity,
                    other.geometry_limited)
    assert (a.evaluations, a.cache_hits) == (b.evaluations, b.cache_hits)


def test_random_metric_protected_flip_reorder_and_consumption_parity():
    points, cells = grid()
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    delta, full = TriangleWorkQueue(), TriangleWorkQueue()
    rng = np.random.default_rng(41093)
    flips, protected = set(), set()
    for iteration in range(40):
        flips.symmetric_difference_update(
            rng.choice(16, size=2, replace=False).tolist())
        _, cells = grid(flips)
        index = index.updated(cells[::-1] if iteration % 2 else cells)
        for node in rng.choice(len(points), size=2, replace=False):
            tensors[node] = np.eye(2) * float(rng.choice((.25, 1., 2., 5.)))
        if iteration % 3 == 0:
            edge = tuple(map(int, index.canonical_edges()[iteration % 20]))
            protected.symmetric_difference_update((edge,))
        refresh(delta, index, points, tensors, protected, True)
        refresh(full, index, points, tensors, protected, False)
        assert_same(delta, full)
        for _ in range(iteration % 5):
            assert bool(delta) == bool(full)
            if not delta:
                break
            assert delta.pop()[:3] == full.pop()[:3]
    while delta:
        assert delta.pop()[:3] == full.pop()[:3]
    assert not full


def test_unchanged_round_has_no_cell_geometry_refresh_and_reactivates_consumed():
    points, cells = grid()
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    delta, full = TriangleWorkQueue(), TriangleWorkQueue()
    refresh(delta, index, points, tensors, set(), True)
    refresh(full, index, points, tensors, set(), False)
    assert delta.pop()[:3] == full.pop()[:3]
    assert refresh(delta, index, points, tensors, set(), True) == ()
    refresh(full, index, points, tensors, set(), False)
    assert_same(delta, full)
    assert len(delta._active) == len(cells)


def test_one_changed_metric_row_refreshes_only_incident_cells():
    points, cells = grid()
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    delta = TriangleWorkQueue()
    refresh(delta, index, points, tensors, set(), True)
    tensors[12] *= 1.5
    rows = refresh(delta, index, points, tensors, set(), True)
    assert set(rows) == {tuple(row) for row in cells if 12 in row}
    assert len(rows) < len(cells)


def test_insertion_removal_and_empty_round_match_reference():
    points, cells = grid(width=1)
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    delta, full = TriangleWorkQueue(), TriangleWorkQueue()
    for step in range(4):
        refresh(delta, index, points, tensors, set(), True)
        refresh(full, index, points, tensors, set(), False)
        assert_same(delta, full)
        if delta:
            assert delta.pop()[:3] == full.pop()[:3]
        if step == 0:
            points = np.vstack((points, (0.7, 0.2)))
            tensors = metrics(points)
            cells = np.asarray(((0, 1, 4), (1, 3, 4), (0, 4, 3), (0, 3, 2)))
        elif step == 1:
            cells = np.empty((0, 3), dtype=np.int64)
        else:
            cells = np.asarray(((0, 1, 3), (0, 3, 2)))
        index = index.updated(cells)


@pytest.mark.parametrize("defect", ["move", "signed_zero", "shrink", "nan", "shape"])
def test_invalid_binding_does_not_commit_preparation(defect):
    points, cells = grid()
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    helper = QueueRefreshIndex()
    helper.prepare(index, points, tensors, (), lambda _: None)
    before = (helper._index, helper._points, helper._tensors, helper._incident)
    points = points.copy()
    if defect == "move":
        points[0, 0] = .001
    elif defect == "signed_zero":
        points[0, 0] = -0.0
    elif defect == "shrink":
        points, tensors = points[:-1], tensors[:-1]
    elif defect == "nan":
        tensors[0, 0, 0] = np.nan
    else:
        tensors = tensors[:, :1]
    with pytest.raises(MeshError):
        helper.prepare(index, points, tensors, (), lambda _: None)
    assert all(actual is expected for actual, expected in zip(
        (helper._index, helper._points, helper._tensors, helper._incident), before))


def test_cancelled_preparation_keeps_previous_index_and_arrays():
    points, cells = grid()
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    helper = QueueRefreshIndex()
    helper.prepare(index, points, tensors, (), lambda _: None)
    before = (helper._index, helper._points, helper._tensors, helper._incident)
    _, changed = grid({3})
    updated = index.updated(changed)
    error = RuntimeError("cancel queue preparation")
    calls = []
    def cancel(number):
        calls.append(number)
        if len(calls) == 2:
            raise error
    with pytest.raises(RuntimeError) as caught:
        helper.prepare(updated, points, tensors, (), cancel)
    assert caught.value is error
    assert all(actual is expected for actual, expected in zip(
        (helper._index, helper._points, helper._tensors, helper._incident), before))
    assert tuple(index._row_by_cell) == tuple(map(tuple, cells))


def test_preparation_owns_bit_exact_metric_snapshot():
    points, cells = grid()
    index = T3IncidenceIndex(cells)
    tensors = metrics(points)
    helper = QueueRefreshIndex()
    helper.prepare(index, points, tensors, (), lambda _: None)
    assert not helper._points.flags.writeable
    assert not helper._tensors.flags.writeable
    tensors[12, 0, 1] = -0.0
    rows, removed = helper.prepare(index, points, tensors, (), lambda _: None)
    assert removed == ()
    assert set(map(tuple, rows)) == {tuple(row) for row in cells if 12 in row}


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("target", [.4, .07414706429645167])
def test_public_cylinder_full_refresh_matches_incremental(monkeypatch, backend, target):
    from anygeometry import to_dict
    from anymesher.hybrid import generate_hybrid_mesh_result, _neutral_shell_core
    from anymesher.quality_v2 import assert_valid_mesh

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    bench = importlib.import_module("native_v2_baseline")
    fixture = bench.cylinder_case("cylinder_patch")
    before = bench._digest_contract(to_dict(fixture.model))
    deadline = time.monotonic() + 90.
    def cancel(phase):
        if time.monotonic() > deadline:
            raise TimeoutError("bounded queue parity at " + phase)
    def generate():
        return generate_hybrid_mesh_result(
            fixture.model, face_ids=fixture.face_ids, target_size=target,
            strategy="native", overrides=fixture.overrides(target),
            refinements=fixture.refinements(target), recombine=True,
            native_backend=backend,
            native_options=fixture.options(target, "frontal", 1000),
            cancellation_check=cancel,
        )
    incremental = generate()
    monkeypatch.setattr(TriangleWorkQueue, "prepare_incremental_round", full_round)
    reference = generate()
    for result in (incremental, reference):
        assert_valid_mesh(_neutral_shell_core(result.mesh))
        work = bench._native_work(
            {"faces": {str(k): dict(v) for k, v in
                       result.triangulation_backend_by_face.items()}},
            "frontal", fixture.name)
        assert work["published_insertions"] > 0
    assert bench._mesh_digest(_neutral_shell_core(incremental.mesh)) == (
        bench._mesh_digest(_neutral_shell_core(reference.mesh)))
    assert bench._digest_contract(fixture.mesh_contract(incremental.mesh, target)) == (
        bench._digest_contract(fixture.mesh_contract(reference.mesh, target)))
    assert incremental.triangulation_backend_by_face[1]["native_v2"] == (
        reference.triangulation_backend_by_face[1]["native_v2"])
    assert bench._digest_contract(to_dict(fixture.model)) == before

