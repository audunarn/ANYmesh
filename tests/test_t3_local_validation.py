"""Local native-result proof versus independent full-mesh validation."""
from dataclasses import replace
from itertools import combinations
import importlib
from pathlib import Path

import numpy as np
import pytest

from anymesher import native_v2
from anymesher import _t3_local_validation as local
from anymesher import _t3_insertion_result as result
from anymesher import _t3_correspondence as correspondence
from anymesher.errors import MeshError
from anymesher.triangulation import orient2d


def mesh():
    n = 3
    points = np.array([(x / n, y / n) for y in range(n + 1) for x in range(n + 1)])
    cells = []
    edges = []
    for y in range(n):
        for x in range(n):
            a = y * (n + 1) + x
            b, d = a + 1, a + n + 1
            cells.extend(((a, b, d + 1), (a, d + 1, d)))
    for i in range(n):
        edges.extend(((i, i + 1), (12 + i, 13 + i),
                      (i * 4, (i + 1) * 4), (i * 4 + 3, (i + 1) * 4 + 3)))
    return native_v2.MutableT3Topology(points, np.asarray(cells), edges)


def arguments(subject):
    point = np.asarray((.21, .13))
    rows, _, report = subject._python_insert_with_owners(point, owner=42)
    return [subject.points, subject.triangles,
            np.asarray(sorted(subject.protected_edges)), point, rows,
            {"removed_triangles": report["removed_triangles"],
             "added_triangles": report["added_triangles"], "native": True}]


def full(args):
    return result.validate_insertion(*args, orient2d)


@correspondence.insertion_correspondence_scope
def qualified(subject, args):
    assert local.source_for_validation(*args[:3]) is not None
    return result.validate_insertion(*args, orient2d)


def snapshot(subject):
    return (subject.points, subject.triangles, subject.node_owners.copy(),
            subject.triangle_owners.copy(), subject._topology_index, subject.epoch,
            subject._local_validation_source)


def unchanged(subject, before):
    for actual, expected in zip(
        (subject.points, subject.triangles, subject.node_owners, subject.triangle_owners),
        before[:4],
    ):
        assert actual.tobytes() == expected.tobytes()
    assert subject._topology_index is before[4]
    assert subject.epoch == before[5]
    assert subject._local_validation_source is before[6]


def test_source_establishment_and_transactional_advancement():
    subject = mesh()
    local.enable_local_validation(subject)
    initial = subject._local_validation_source
    for point in ((.21, .13), (.69, .81), (.51, .39)):
        before = subject._local_validation_source
        subject.insert_point(point, owner=42, cancellation_check=lambda _: None)
        current = subject._local_validation_source
        assert local.source_for_scope(subject) is current
        assert current is not before
        assert current.epoch == before.epoch + 1
        assert not current.points.flags.writeable and not current.rows.flags.writeable
        edges, counts = result._edges_and_counts(subject.triangles)
        assert dict(current.counts) == {tuple(map(int, e)): int(n)
                                        for e, n in zip(edges, counts)}
        subject.validate()
    assert initial.epoch == 0
    assert len(initial.points) == 16
    with pytest.raises(TypeError):
        initial.counts[(0, 1)] = 0
    assert correspondence._ACTIVE.get() is None


def test_pending_proof_is_bound_only_after_index_update(monkeypatch):
    subject = mesh()
    local.enable_local_validation(subject)
    prior = subject._local_validation_source
    original = local.bind_qualified_commit
    observed = []

    def bind(value):
        frame = correspondence._ACTIVE.get()
        pending = frame[2]
        assert type(pending) is local.PendingCommit
        assert pending.source is prior
        assert not pending.points.flags.writeable
        assert not pending.rows.flags.writeable
        assert local.source_for_scope(value) is None
        assert value._local_validation_source is prior
        assert value._topology_index is not prior.index
        assert np.array_equal(value._topology_index._row_matrix, pending.rows)
        original(value)
        assert type(frame[2]) is local.QualifiedSource
        assert value._local_validation_source is prior
        observed.append(True)

    monkeypatch.setattr(local, "bind_qualified_commit", bind)
    subject.insert_point((.21, .13), owner=42)
    assert observed == [True]
    assert local.source_for_scope(subject) is subject._local_validation_source
    assert subject._local_validation_source is not prior


@pytest.mark.parametrize("defect", ["rows", "incidence"])
def test_binding_mismatch_rolls_back_commit(monkeypatch, defect):
    subject = mesh()
    local.enable_local_validation(subject)
    before = snapshot(subject)
    original = local.bind_qualified_commit

    def bind(value):
        frame = correspondence._ACTIVE.get()
        pending = frame[2]
        if defect == "rows":
            frame[2] = replace(pending, rows=local._owned(pending.rows[::-1]))
        else:
            counts = dict(pending.changed_counts)
            edge = next(iter(counts))
            counts[edge] += 1
            frame[2] = replace(pending, changed_counts=counts)
        original(value)

    monkeypatch.setattr(local, "bind_qualified_commit", bind)
    with pytest.raises(MeshError, match="commit .* is stale"):
        subject.insert_point((.21, .13), owner=42)
    unchanged(subject, before)
    assert correspondence._ACTIVE.get() is None


def test_unbound_pending_data_is_not_published():
    subject = mesh()
    local.enable_local_validation(subject)
    source = subject._local_validation_source
    pending = local.PendingCommit(source, source.points, source.rows, source.counts, {})
    local.finish_scope(subject, [None, source, pending, subject])
    assert subject._local_validation_source is None


def test_local_result_checks_only_removed_and_added_cells(monkeypatch):
    subject = mesh()
    args = arguments(subject)
    expected = full(args)
    local.enable_local_validation(subject)
    incidence_sizes, geometry_sizes = [], []
    old_edges, old_positive = result._edges_and_counts, result._positive_cells
    def counted_edges(cells):
        incidence_sizes.append(len(cells))
        return old_edges(cells)
    def counted_geometry(points, cells, oracle, callback):
        geometry_sizes.append(len(cells))
        return old_positive(points, cells, oracle, callback)
    monkeypatch.setattr(result, "_edges_and_counts", counted_edges)
    monkeypatch.setattr(result, "_positive_cells", counted_geometry)
    actual = qualified(subject, args)
    assert actual[0].tobytes() == expected[0].tobytes()
    assert actual[1] == expected[1]
    assert geometry_sizes == [args[5]["added_triangles"]]
    assert incidence_sizes == [args[5]["removed_triangles"], args[5]["added_triangles"]]
    assert max(incidence_sizes) < len(args[4])


@pytest.mark.parametrize("defect", [
    "shape", "float", "range", "duplicate", "order", "negative_cell",
    "report_field", "report_count", "extra_star_cell",
])
def test_malformed_local_result_matches_full_failure(defect):
    subject = mesh()
    args = arguments(subject)
    if defect == "shape":
        args[4] = args[4].ravel()
    elif defect == "float":
        args[4] = args[4].astype(float)
    elif defect == "range":
        args[4][0, 0] = -1
    elif defect == "duplicate":
        args[4][1] = args[4][0]
    elif defect == "order":
        args[4] = args[4][::-1]
    elif defect == "negative_cell":
        row = np.flatnonzero(np.any(args[4] == len(args[0]), axis=1))[0]
        args[4][row, [1, 2]] = args[4][row, [2, 1]]
        args[4] = np.asarray(sorted(map(tuple, args[4])))
    elif defect == "report_field":
        args[5]["trusted"] = True
    elif defect == "report_count":
        args[5]["added_triangles"] += 1
    else:
        inserted = len(args[0])
        old_nodes = sorted(set(args[4][np.any(args[4] == inserted, axis=1)].ravel()) - {inserted})
        points = np.vstack((args[0], args[3]))
        existing = set(map(tuple, args[4]))
        extra = None
        for a, b in combinations(old_nodes, 2):
            if orient2d(points[a], points[b], points[inserted]) == 0:
                continue
            candidate = native_v2._canonical_triangle((a, b, inserted), points)
            if candidate not in existing:
                extra = candidate
                break
        assert extra is not None
        args[4] = np.asarray(sorted((*existing, extra)))
        args[5]["added_triangles"] += 1
    with pytest.raises(MeshError) as reference:
        full(args)
    local.enable_local_validation(subject)
    before = snapshot(subject)
    with pytest.raises(MeshError) as actual:
        qualified(subject, args)
    assert str(actual.value) == str(reference.value)
    unchanged(subject, before)


def test_local_guard_cannot_remove_a_previously_protected_internal_edge():
    subject = mesh()
    args = arguments(subject)
    old_edges, _ = result._edges_and_counts(args[1])
    new_edges, _ = result._edges_and_counts(args[4])
    lost = sorted(set(map(tuple, old_edges)) - set(map(tuple, new_edges)))
    assert lost
    subject._protected_edges = set(subject.protected_edges) | {lost[0]}
    args[2] = np.asarray(sorted(subject.protected_edges))
    local.enable_local_validation(subject)
    with pytest.raises(MeshError, match="removed a protected edge") as expected:
        full(args)
    with pytest.raises(MeshError) as actual:
        qualified(subject, args)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("defect", ["nan", "rows", "members", "missing_edge", "protected"])
def test_invalid_source_is_not_registered(defect):
    subject = mesh()
    local.enable_local_validation(subject)
    prior = subject._local_validation_source
    if defect == "nan":
        subject._points = subject._points.copy()
        subject._points[0, 0] = np.nan
    elif defect == "rows":
        subject._triangles = subject._triangles[::-1].copy()
    elif defect == "members":
        edge = next(iter(subject._topology_index._edge_cells))
        subject._topology_index._edge_cells[edge] = ()
    elif defect == "missing_edge":
        edge = next(iter(subject._topology_index._edge_cells))
        del subject._topology_index._edge_cells[edge]
    else:
        subject._protected_edges = set(subject.protected_edges) | {(0, len(subject.points) - 1)}
    with pytest.raises(MeshError):
        local.enable_local_validation(subject)
    assert subject._local_validation_source is prior


@pytest.mark.parametrize("phase", ["start", "orientation", "incidence", "commit"])
def test_source_registration_cancellation_keeps_previous_state(phase):
    subject = mesh()
    local.enable_local_validation(subject)
    before = snapshot(subject)
    error = RuntimeError("cancel source establishment")
    def cancel(value):
        if value == "native-v2 qualified source " + phase:
            raise error
    with pytest.raises(RuntimeError) as caught:
        local.enable_local_validation(subject, cancellation_check=cancel)
    assert caught.value is error
    unchanged(subject, before)


def test_snapshot_allocation_failure_rolls_back_commit(monkeypatch):
    subject = mesh()
    local.enable_local_validation(subject)
    before = snapshot(subject)
    error = MemoryError("candidate qualification allocation")
    def fail(*args):
        raise error
    monkeypatch.setattr(local, "_snapshot", fail)
    with pytest.raises(MemoryError) as caught:
        subject.insert_point((.21, .13), owner=42, cancellation_check=lambda _: None)
    assert caught.value is error
    unchanged(subject, before)
    assert correspondence._ACTIVE.get() is None


def test_stale_source_uses_full_result_validation(monkeypatch):
    subject = mesh()
    local.enable_local_validation(subject)
    subject._local_validation_source = replace(
        subject._local_validation_source, epoch=subject.epoch + 1)
    sizes = []
    original = result._positive_cells
    def track(points, rows, oracle, callback):
        sizes.append(len(rows))
        return original(points, rows, oracle, callback)
    monkeypatch.setattr(result, "_positive_cells", track)
    subject.insert_point((.21, .13), owner=42, cancellation_check=lambda _: None)
    # If the optional native kernel is absent, no result-validator call occurs.
    if sizes:
        assert len(subject.triangles) in sizes
    assert subject._local_validation_source is None
    subject.validate()


@pytest.mark.parametrize("backend", ["python", "native"])
def test_public_cylinder_local_full_guard_parity(monkeypatch, backend):
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
            native_options=fixture.options(target, "frontal", 1000))
    incremental = generate()
    monkeypatch.setattr(local, "enable_local_validation", lambda *args, **kwargs: None)
    reference = generate()
    for value in (incremental, reference):
        assert_valid_mesh(_neutral_shell_core(value.mesh))
        work = bench._native_work(
            {"faces": {str(k): dict(v) for k, v in value.triangulation_backend_by_face.items()}},
            "frontal", fixture.name)
        assert work["published_insertions"] > 0
    assert bench._mesh_digest(_neutral_shell_core(incremental.mesh)) == (
        bench._mesh_digest(_neutral_shell_core(reference.mesh)))
    assert bench._digest_contract(fixture.mesh_contract(incremental.mesh, target)) == (
        bench._digest_contract(fixture.mesh_contract(reference.mesh, target)))
    assert incremental.triangulation_backend_by_face[1]["native_v2"] == (
        reference.triangulation_backend_by_face[1]["native_v2"])
    assert bench._digest_contract(to_dict(fixture.model)) == before
