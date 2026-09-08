"""Independent rejection and owner-parity checks for batched insertion guards."""
import numpy as np
import pytest

from anymesher import native_v2
from anymesher._t3_incidence import T3IncidenceIndex
from anymesher._t3_insertion_result import (
    _positive_cells, _source_rows, insertion_owners, validate_insertion,
)
from anymesher.errors import MeshError
from anymesher.triangulation import orient2d


def fixture():
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    before = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    protected = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    candidate = np.array([.3, .2])
    after = np.array([[0, 1, 4], [0, 4, 3], [1, 2, 4], [2, 3, 4]])
    report = {"removed_triangles": 2, "added_triangles": 4, "native": True}
    return points, before, protected, candidate, after, report


def validate(args, callback=None):
    return validate_insertion(*args, orient2d, callback)


@pytest.mark.parametrize("scale", [.001, 1., 1000.])
@pytest.mark.parametrize("offset", [0., 10000.])
def test_valid_star_covariance_and_no_input_mutation(scale, offset):
    args = list(fixture())
    args[0] = args[0] * scale + offset
    args[3] = args[3] * scale + offset
    copies = [value.copy() for value in args]
    actual, report = validate(args)
    np.testing.assert_array_equal(actual, args[4])
    assert report == args[5] and report is not args[5]
    for actual, original in zip(args, copies):
        if isinstance(actual, dict):
            assert actual == original
        else:
            np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("defect", [
    "shape", "float", "range", "unsigned_overflow", "duplicate",
    "ordering", "rotation", "negative_orientation", "unknown_field",
    "boolean_count", "bad_count", "non_native", "missing_inserted",
    "protected_loss",
])
def test_malformed_results_never_pass(defect):
    args = list(fixture())
    if defect == "shape":
        args[4] = np.array([0, 1, 4])
    elif defect == "float":
        args[4] = args[4].astype(float)
    elif defect == "range":
        args[4][0, 0] = -1
    elif defect == "unsigned_overflow":
        args[4] = args[4].astype(np.uint64)
        args[4][0, 0] = np.iinfo(np.uint64).max
    elif defect == "duplicate":
        args[4][1] = args[4][0]
    elif defect == "ordering":
        args[4] = args[4][::-1]
    elif defect == "rotation":
        args[4][0] = [1, 4, 0]
    elif defect == "negative_orientation":
        args[4][0] = [0, 4, 1]
        args[4] = np.array(sorted(map(tuple, args[4])))
    elif defect == "unknown_field":
        args[5]["trusted"] = True
    elif defect == "boolean_count":
        args[5]["removed_triangles"] = True
    elif defect == "bad_count":
        args[5]["added_triangles"] = 5
    elif defect == "non_native":
        args[5]["native"] = False
    elif defect == "missing_inserted":
        args[4][0] = [0, 1, 2]
    elif defect == "protected_loss":
        args[2] = np.array([[0, 2]])
    with pytest.raises(MeshError):
        validate(args)


def test_positive_star_for_wrong_region_is_rejected():
    points, before, protected, candidate, after, report = fixture()
    before = before[:1]
    after = np.array([[0, 1, 4], [0, 4, 3], [1, 3, 4]])
    report = {"removed_triangles": 1, "added_triangles": 3, "native": True}
    assert all(orient2d(np.vstack((points, candidate))[a],
                        np.vstack((points, candidate))[b],
                        np.vstack((points, candidate))[c]) > 0
               for a, b, c in after)
    with pytest.raises(MeshError, match="cavity boundary"):
        validate_insertion(points, before, protected[:0], candidate,
                           after, report, orient2d)


def test_retained_and_new_row_mapping_handles_reordered_input():
    before = np.array([[2, 3, 4], [0, 1, 2], [1, 3, 2]], dtype=np.int64)
    after = np.array([[0, 1, 2], [0, 2, 5], [2, 3, 4]], dtype=np.int64)
    np.testing.assert_array_equal(_source_rows(before, after), [1, -1, 0])


@pytest.mark.parametrize("owner", [-1, 42])
@pytest.mark.parametrize("reverse", [False, True])
def test_owner_lookup_matches_original_full_scan(owner, reverse):
    points, before, protected, candidate, after, report = fixture()
    if reverse:
        before = before[::-1].copy()
    mesh = native_v2.MutableT3Topology(
        points, before, protected, triangle_owners=[11, 22],
    )
    expected = mesh._native_insert_owners(after, inserted_node=4, owner=owner)
    actual = insertion_owners(
        mesh.triangles, after, mesh.triangle_owners, mesh._topology_index,
        inserted_node=4, owner=owner,
    )
    np.testing.assert_array_equal(actual, expected)


def test_uncertain_sign_uses_original_adaptive_oracle():
    points = np.array([[0., 0.], [1., 1.], [2., 2. + 2.**-49]])
    rows = np.array([[0, 1, 2]])
    calls = []

    def oracle(a, b, c):
        calls.append((a.copy(), b.copy(), c.copy()))
        return orient2d(a, b, c)

    _positive_cells(points, rows, oracle, None)
    assert len(calls) == 1
    with pytest.raises(MeshError, match="non-positive"):
        _positive_cells(points, rows[:, [0, 2, 1]], oracle, None)


@pytest.mark.parametrize("phase", [
    "identity validation", "orientation validation", "incidence validation",
    "validated",
])
def test_validation_cancellation_preserves_inputs(phase):
    args = fixture()
    points, rows = args[0].copy(), args[4].copy()
    error = RuntimeError("cancel detached checks")
    name = "native-v2 insertion result " + phase

    def cancel(value):
        if value == name:
            raise error

    with pytest.raises(RuntimeError) as caught:
        validate(args, cancel)
    assert caught.value is error
    np.testing.assert_array_equal(args[0], points)
    np.testing.assert_array_equal(args[4], rows)


def test_owner_cancellation_preserves_source_index():
    points, before, protected, candidate, after, report = fixture()
    index = T3IncidenceIndex(before)
    membership = index._edge_cells.copy()
    calls = 0

    def cancel(phase):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("cancel owners")

    with pytest.raises(RuntimeError, match="cancel owners"):
        insertion_owners(before, after, np.array([11, 22]), index,
                         inserted_node=4, owner=-1, cancellation_check=cancel)
    assert all(index._edge_cells[edge] is value
               for edge, value in membership.items())
