"""Changed-cavity validation must remain equivalent to full topology checks."""
import numpy as np
import pytest

from anymesher import native_v2
from anymesher import _t3_insertion_transaction as transaction
from anymesher.errors import MeshError
from anymesher.triangulation import orient2d


def topology():
    n = 3
    points = np.array([(x / n, y / n) for y in range(n + 1)
                       for x in range(n + 1)])
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
    return native_v2.MutableT3Topology(
        points, np.array(cells), edges,
        node_owners=np.arange(len(points)), triangle_owners=np.arange(len(cells)),
    )


def snapshot(mesh):
    return (mesh.points, mesh.triangles, mesh.node_owners.copy(),
            mesh.triangle_owners.copy(), mesh._topology_index, mesh.epoch,
            dict(mesh.quality_cache))


def assert_unchanged(mesh, before):
    for value, expected in zip(
        (mesh.points, mesh.triangles, mesh.node_owners, mesh.triangle_owners),
        before[:4],
    ):
        np.testing.assert_array_equal(value, expected)
    assert mesh._topology_index is before[4]
    assert mesh.epoch == before[5]
    assert mesh.quality_cache == before[6]


def prepared():
    mesh = topology()
    candidate = np.array([.21, .13])
    rows, owners, report = mesh._python_insert_with_owners(candidate, owner=42)
    args = [
        np.vstack((mesh.points, candidate)), rows.copy(),
        np.append(mesh.node_owners, 42), owners.copy(), mesh.points,
        mesh.triangles, mesh.node_owners.copy(), mesh.triangle_owners.copy(),
        mesh._topology_index, mesh.protected_edges, {}, dict(report),
    ]
    return mesh, args


def qualify(args, callback=None):
    transaction.validate_insertion_transaction(
        *args, owner=42, orientation_oracle=orient2d,
        cancellation_check=callback,
    )


def test_actual_insertion_no_longer_calls_full_validate(monkeypatch):
    mesh = topology()
    full_validate = mesh.validate

    def forbidden():
        raise AssertionError("full topology scan during insertion")

    monkeypatch.setattr(mesh, "validate", forbidden)
    report = mesh.insert_point((.21, .13), owner=42)
    assert report["added_triangles"] >= 3
    full_validate()


def test_local_geometry_work_is_exactly_the_added_cells(monkeypatch):
    mesh = topology()
    counts = []
    original = transaction._positive_cells

    def tracked(points, cells, oracle, callback):
        counts.append(len(cells))
        return original(points, cells, oracle, callback)

    monkeypatch.setattr(transaction, "_positive_cells", tracked)
    report = mesh.insert_point((.21, .13), owner=42)
    assert counts == [report["added_triangles"]]
    assert counts[0] < len(mesh.triangles)
    mesh.validate()


@pytest.mark.parametrize("point", [
    (.21, .13), (.69, .81), (.51, .39), (.88, .17),
])
def test_local_success_passes_independent_full_guard(point):
    mesh = topology()
    report = mesh.insert_point(point, owner=42)
    assert mesh.epoch == 1
    mesh.validate()
    assert len(mesh.points) == 17


def test_repeated_insertions_preserve_validity_and_prior_snapshots():
    mesh = topology()
    for point in [(0.21, .13), (.69, .81), (.51, .39), (.88, .17)]:
        before = snapshot(mesh)
        report = mesh.insert_point(point, owner=42)
        mesh.validate()
        assert report["epoch"] == before[5] + 1
        assert np.array_equal(mesh.points[:-1], before[0])
        assert set(before[4]._row_by_cell) == set(map(tuple, before[1]))


@pytest.mark.parametrize("phase", [
    "validation", "geometry", "incidence", "constraints", "qualified",
])
def test_local_cancellation_restores_entire_transaction(phase):
    mesh = topology()
    mesh.quality_cache[(0, 1, 5)] = (1.,)
    before = snapshot(mesh)
    error = RuntimeError("cancel changed cavity")

    def cancel(value):
        if value == "native-v2 insertion cavity " + phase:
            raise error

    with pytest.raises(RuntimeError) as caught:
        mesh.insert_point((.21, .13), owner=42, cancellation_check=cancel)
    assert caught.value is error
    assert_unchanged(mesh, before)


@pytest.mark.parametrize("defect", [
    "retained_point", "signed_zero", "nonfinite_new_point",
    "retained_node_owner", "retained_cell_owner", "explicit_owner",
    "new_cell_owner_shape", "new_cell_range", "duplicate_cell",
    "removed_count", "added_count", "missing_inserted_node",
])
def test_bad_transaction_is_rejected_without_touching_inputs(defect):
    mesh, args = prepared()
    before = snapshot(mesh)
    retained = [i for i, row in enumerate(args[1])
                if tuple(map(int, row)) in mesh._topology_index._row_by_cell]
    added = [i for i in range(len(args[1])) if i not in retained]
    if defect == "retained_point":
        args[0][0, 0] = .001
    elif defect == "signed_zero":
        args[0][0, 0] = -0.0
    elif defect == "nonfinite_new_point":
        args[0][-1, 0] = np.nan
    elif defect == "retained_node_owner":
        args[2][0] = 100
    elif defect == "retained_cell_owner":
        args[3][retained[0]] = 999
    elif defect == "explicit_owner":
        args[3][added[0]] = 99
    elif defect == "new_cell_owner_shape":
        args[3] = args[3][:-1]
    elif defect == "new_cell_range":
        args[1][added[0], 2] = 999
    elif defect == "duplicate_cell":
        args[1][retained[0]] = args[1][retained[1]]
    elif defect == "removed_count":
        args[11]["removed_triangles"] += 1
    elif defect == "added_count":
        args[11]["added_triangles"] = True
    elif defect == "missing_inserted_node":
        row = args[1][added[0]]
        row[row == len(mesh.points)] = 0
    with pytest.raises(MeshError):
        qualify(args)
    assert_unchanged(mesh, before)


def test_protected_internal_edge_cannot_disappear():
    mesh, args = prepared()
    old_edges = set(mesh._topology_index._edge_cells)
    new_edges = {tuple(sorted((int(row[i]), int(row[(i + 1) % 3]))))
                 for row in args[1] for i in range(3)}
    lost = old_edges - new_edges
    assert lost
    args[9] = set(args[9]) | lost
    with pytest.raises(MeshError, match="constrained edge"):
        qualify(args)


def test_geometry_limited_rejection_keeps_original_state():
    mesh = topology()
    before = snapshot(mesh)
    with pytest.raises(MeshError):
        mesh.insert_point((0., 0.))
    assert_unchanged(mesh, before)
