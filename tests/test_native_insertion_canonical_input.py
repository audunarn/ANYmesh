"""Native retained-cell canonicalization must not look like cavity replacement."""
import numpy as np
import pytest
from anymesher import native_v2, native_cpp
from anymesher.errors import MeshError
from test_t3_insertion_transaction import topology, snapshot, assert_unchanged


@pytest.mark.parametrize("reverse", [False, True])
def test_owned_rows_preserve_cells_row_owners_and_caller_arrays(reverse):
    original = topology()
    rows = np.roll(original.triangles, 1, axis=1).copy()
    if reverse:
        rows[:, [1, 2]] = rows[:, [2, 1]]
    saved = rows.copy()
    made = native_v2.MutableT3Topology(original.points, rows,
        sorted(sorted(original.protected_edges)), triangle_owners=original.triangle_owners)
    np.testing.assert_array_equal(rows, saved)
    np.testing.assert_array_equal(made.triangles, original.triangles)
    np.testing.assert_array_equal(made.triangle_owners, original.triangle_owners)
    assert made.protected_edges == original.protected_edges
    made.validate()


@pytest.mark.parametrize("python_only", [False, True])
def test_rotated_seed_insertion_matches_canonical_oracle(monkeypatch, python_only):
    if python_only:
        monkeypatch.setattr(native_v2, "native_mutable_t3_insert", lambda *a, **k: None)
    elif native_cpp._compiled is None:
        pytest.skip("compiled insertion required")
    reference = topology()
    rows = np.roll(reference.triangles, 1, axis=1).copy()
    made = native_v2.MutableT3Topology(reference.points, rows,
        sorted(reference.protected_edges), node_owners=reference.node_owners,
        triangle_owners=reference.triangle_owners)
    expected = reference.insert_point((.21, .13), owner=42)
    actual = made.insert_point((.21, .13), owner=42)
    assert actual == expected
    np.testing.assert_array_equal(made.triangles, reference.triangles)
    np.testing.assert_array_equal(made.triangle_owners, reference.triangle_owners)
    np.testing.assert_array_equal(made.points, reference.points)
    made.validate()


def test_cancellation_before_commit_preserves_canonical_owned_snapshot():
    original = topology()
    made = native_v2.MutableT3Topology(original.points,
        np.roll(original.triangles, 1, axis=1), sorted(sorted(original.protected_edges)),
        triangle_owners=original.triangle_owners)
    before = snapshot(made)
    error = RuntimeError("cancel canonical insertion")
    def cancel(phase):
        if phase == "native-v2 mutable insertion commit":
            raise error
    with pytest.raises(RuntimeError) as caught:
        made.insert_point((.21, .13), cancellation_check=cancel)
    assert caught.value is error
    assert_unchanged(made, before)


@pytest.mark.parametrize("defect", ["duplicate", "degenerate", "range"])
def test_normalization_does_not_admit_bad_source(defect):
    original = topology()
    rows = np.roll(original.triangles, 1, axis=1).copy()
    if defect == "duplicate": rows[1] = rows[0]
    elif defect == "degenerate": rows[0] = rows[0, 0]
    else: rows[0, 0] = len(original.points)
    with pytest.raises(MeshError):
        native_v2.MutableT3Topology(original.points, rows, sorted(original.protected_edges))


def test_native_and_python_cavities_agree_for_rotated_seed(monkeypatch):
    if native_cpp._compiled is None:
        pytest.skip("compiled insertion required")
    original=topology()
    rows=np.roll(original.triangles,1,axis=1)
    def make():
        return native_v2.MutableT3Topology(original.points,rows,
            sorted(sorted(original.protected_edges)),triangle_owners=original.triangle_owners)
    compiled=make(); python=make()
    left=compiled.insert_point((.21,.13),owner=42)
    monkeypatch.setattr(native_v2,"native_mutable_t3_insert",lambda *a,**k:None)
    right=python.insert_point((.21,.13),owner=42)
    for key in ("removed_triangles","added_triangles","epoch","point_id"):
        assert left[key]==right[key]
    np.testing.assert_array_equal(compiled.triangles,python.triangles)
    np.testing.assert_array_equal(compiled.triangle_owners,python.triangle_owners)
    np.testing.assert_array_equal(compiled.points,python.points)
