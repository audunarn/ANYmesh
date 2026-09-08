"""Private native snapshot ABI rejects foreign capsules before pointer use."""
from __future__ import annotations

import ctypes

import numpy as np
import pytest

from anymesher.triangulation import orient2d

native = pytest.importorskip("anymesher._native")

CURRENT_NAME = b"anymesher.native_v2.t3_incidence/2"
FOREIGN_NAMES = (
    b"anymesher.native_v2.t3_incidence/1",
    b"anymesher.native_v2.t3_incidence/999",
    b"unrelated.provider/snapshot",
)


def arrays():
    points = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
    cells = np.array(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    return points, cells, np.empty((0, 2), dtype=np.int64)


def capsule_name(capsule):
    getter = ctypes.pythonapi.PyCapsule_GetName
    getter.argtypes = (ctypes.py_object,)
    getter.restype = ctypes.c_char_p
    return getter(capsule)


def require_current_abi():
    _, cells, _ = arrays()
    actual = capsule_name(native.native_v2_t3_incidence(cells))
    if actual != CURRENT_NAME:
        raise AssertionError(
            f"stale native snapshot ABI: expected {CURRENT_NAME!r}, got {actual!r}; "
            "load a rebuilt extension before capsule rejection tests"
        )


@pytest.fixture(autouse=True)
def current_native_abi():
    require_current_abi()


def foreign_capsule(name):
    require_current_abi()
    maker = ctypes.pythonapi.PyCapsule_New
    maker.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    maker.restype = ctypes.py_object
    # No native object lives at this address. The name gate MUST reject it
    # before a cast/dereference. The capsule owns no destructor or allocation.
    # Name bytes remain alive in FOREIGN_NAMES throughout capsule lifetime.
    return maker(ctypes.c_void_p(1), name, None)


def invoke(consumer, state, points, cells, protected, oracle=orient2d):
    if consumer == "check":
        return native.native_v2_t3_incidence_check(state, cells)
    if consumer == "update":
        return native.native_v2_t3_incidence(cells, state)
    return native.native_v2_mutable_t3_insert(
        points, cells, protected, oracle, .37123, .21739, state)


@pytest.mark.parametrize("consumer", ["check", "update", "insert"])
@pytest.mark.parametrize("name", FOREIGN_NAMES)
def test_foreign_abi_is_rejected_before_pointer_or_predicate_use(consumer, name):
    points, cells, protected = arrays()
    original_points, original_cells = points.copy(), cells.copy()
    valid = native.native_v2_t3_incidence(cells)
    foreign = foreign_capsule(name)
    assert capsule_name(foreign) == name
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("foreign ABI reached a geometry predicate")
    with pytest.raises(ValueError, match="incorrect name"):
        invoke(consumer, foreign, points, cells, protected, forbidden)
    assert calls == []
    assert points.tobytes() == original_points.tobytes()
    assert cells.tobytes() == original_cells.tobytes()
    assert native.native_v2_t3_incidence_check(valid, cells) is True


@pytest.mark.parametrize("consumer", ["check", "update", "insert"])
def test_non_capsule_is_rejected_without_changing_valid_snapshot(consumer):
    points, cells, protected = arrays()
    valid = native.native_v2_t3_incidence(cells)
    with pytest.raises(ValueError, match="invalid PyCapsule object"):
        invoke(consumer, object(), points, cells, protected)
    assert native.native_v2_t3_incidence_check(valid, cells) is True


@pytest.mark.parametrize("name", FOREIGN_NAMES)
def test_preflight_precedes_invalid_capsule_allocation(monkeypatch, name):
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("invalid capsule allocation reached")
    monkeypatch.setattr(ctypes.pythonapi, "PyCapsule_New", forbidden)
    monkeypatch.setattr(__import__(__name__, fromlist=["capsule_name"]),
                        "capsule_name", lambda capsule: name)
    with pytest.raises(AssertionError, match="stale native snapshot ABI"):
        foreign_capsule(FOREIGN_NAMES[0])
    assert calls == []


def test_current_abi_owns_updates_and_remains_usable_after_rejection():
    points, cells, protected = arrays()
    state = native.native_v2_t3_incidence(cells)
    assert capsule_name(state) == CURRENT_NAME
    changed = native.native_v2_t3_incidence(cells[::-1].copy(), state)
    assert capsule_name(changed) == CURRENT_NAME
    assert native.native_v2_t3_incidence_check(changed, cells[::-1].copy()) is True
    assert native.native_v2_t3_incidence_check(state, cells) is True
    result = invoke("insert", state, points, cells, protected)
    reference = invoke("insert", None, points, cells, protected)
    assert result == reference
