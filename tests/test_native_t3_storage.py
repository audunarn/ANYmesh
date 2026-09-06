"""Bounded stateful/stateless parity and publication checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from anymesher import native_cpp, native_v2


def data():
    return (
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64),
    )


def require_storage():
    if not native_cpp._native_t3_incidence_available():
        pytest.skip("optional compiled T3 storage capability is absent")


@pytest.mark.parametrize("reorder", [False, True])
def test_stateful_insertion_matches_stateless_and_retains_old_snapshot(reorder):
    require_storage()
    points, triangles, protected = data()
    if reorder:
        triangles = np.ascontiguousarray(triangles[::-1, [1, 2, 0]])
    state = native_cpp.native_t3_incidence(triangles)
    candidate = (0.2, 0.15)
    expected = native_cpp.native_mutable_t3_insert(points, triangles, protected, candidate)
    actual = native_cpp.native_mutable_t3_insert(points, triangles, protected, candidate, state)
    assert actual is not None and expected is not None
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    replacement = native_cpp.native_t3_incidence(actual[0], state)
    assert native_cpp._compiled.native_v2_t3_incidence_check(replacement, actual[0]) is True
    assert native_cpp._compiled.native_v2_t3_incidence_check(state, triangles) is True
    repeated = native_cpp.native_mutable_t3_insert(points, triangles, protected, candidate, state)
    np.testing.assert_array_equal(repeated[0], expected[0])


@pytest.mark.parametrize("mismatch", ["rows", "order", "cell"])
def test_snapshot_binding_is_exact_not_shape_or_pointer(mismatch):
    require_storage()
    points, triangles, protected = data()
    state = native_cpp.native_t3_incidence(triangles)
    changed = triangles.copy()
    if mismatch == "rows":
        changed = changed[:1].copy()
    elif mismatch == "order":
        changed = changed[::-1].copy()
    else:
        changed[0] = [0, 1, 3]
    with pytest.raises(RuntimeError, match="snapshot connectivity mismatch"):
        native_cpp.native_mutable_t3_insert(points, changed, protected, (0.2, 0.15), state)
    assert native_cpp._compiled.native_v2_t3_incidence_check(state, triangles) is True


def test_foreign_snapshot_is_fatal_not_backend_absence():
    require_storage()
    points, triangles, protected = data()
    with pytest.raises(ValueError, match="PyCapsule"):
        native_cpp.native_mutable_t3_insert(points, triangles, protected, (0.2, 0.15), object())


def test_snapshot_owns_input_connectivity():
    require_storage()
    _, triangles, _ = data()
    original = triangles.copy()
    state = native_cpp.native_t3_incidence(triangles)
    triangles[0] = [0, 1, 3]
    assert native_cpp._compiled.native_v2_t3_incidence_check(state, original) is True
    with pytest.raises(RuntimeError, match="snapshot connectivity mismatch"):
        native_cpp._compiled.native_v2_t3_incidence_check(state, triangles)


def test_present_malformed_snapshot_cannot_publish(monkeypatch):
    require_storage()
    points, triangles, protected = data()
    topology = native_v2.MutableT3Topology(points, triangles, protected)
    before_index = topology._topology_index
    extension = native_cpp._compiled

    class MalformedFactory:
        native_v2_t3_incidence = staticmethod(lambda *args: object())

        def __getattr__(self, name):
            return getattr(extension, name)

    monkeypatch.setattr(native_cpp, "_compiled", MalformedFactory())
    with pytest.raises(ValueError, match="PyCapsule"):
        topology.insert_point((0.2, 0.15))
    np.testing.assert_array_equal(topology.points, points)
    np.testing.assert_array_equal(topology.triangles, triangles)
    assert topology._topology_index is before_index
    assert topology.epoch == 0
    assert extension.native_v2_t3_incidence_check(before_index._native_state, triangles) is True


def test_old_compiled_abi_keeps_stateless_dispatch(monkeypatch):
    symbols = {name: lambda *args: None for name in native_cpp._NATIVE_V2_REQUIRED_SYMBOLS}
    old = SimpleNamespace(**symbols)
    monkeypatch.setattr(native_cpp, "_compiled", old)
    assert native_cpp.native_t3_incidence(object()) is None
    seen = []

    def insert(*args):
        seen.append(len(args))
        raise RuntimeError("old ABI reached")

    old.native_v2_mutable_t3_insert = insert
    points, triangles, protected = data()
    with pytest.raises(RuntimeError, match="old ABI reached"):
        native_cpp.native_mutable_t3_insert(points, triangles, protected, (0.2, 0.15))
    assert seen == [6]


def test_partial_capability_is_not_silently_ignored(monkeypatch):
    symbols = {name: lambda *args: None for name in native_cpp._NATIVE_V2_REQUIRED_SYMBOLS}
    symbols["native_v2_t3_incidence"] = lambda *args: None
    monkeypatch.setattr(native_cpp, "_compiled", SimpleNamespace(**symbols))
    with pytest.raises(RuntimeError, match="incidence ABI is incomplete"):
        native_cpp.native_t3_incidence([])


def test_later_binding_mismatch_precedes_early_predicate_uncertainty():
    require_storage()
    from anymesher.triangulation import orient2d

    points = np.array([[0., 0.], [1., 1.], [2., np.nextafter(2., np.inf)], [0., 2.]])
    original = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    changed = original.copy()
    changed[1] = [0, 1, 3]
    protected = np.empty((0, 2), dtype=np.int64)
    state = native_cpp.native_t3_incidence(original)
    with pytest.raises(RuntimeError, match="PREDICATE_UNCERTAIN"):
        native_cpp._compiled.native_v2_mutable_t3_insert(
            points, original, protected, orient2d, 0.2, 0.3
        )
    with pytest.raises(RuntimeError, match="snapshot connectivity mismatch"):
        native_cpp._compiled.native_v2_mutable_t3_insert(
            points, changed, protected, orient2d, 0.2, 0.3, state
        )
    with pytest.raises(RuntimeError, match="snapshot connectivity mismatch"):
        native_cpp.native_mutable_t3_insert(points, changed, protected, (0.2, 0.3), state)
