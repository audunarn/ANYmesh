"""Coordinates-only aspect scans preserve the complete repair outcome."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from anymesher import hybrid
from anymesher.mesh import Mesh
from anymesher.quad.domain import PlanarQuadDomain


def _domain(cls=PlanarQuadDomain):
    return cls(
        model_id="test", revision=0, face_id=1, vertex_ids=(0, 1, 2, 3),
        edge_uses=(), origin=(0., 0., 0.), x_hat=(1., 0., 0.),
        y_hat=(0., 1., 0.), normal=(0., 0., 1.),
        outer_chart=((0., 0.), (40., 0.), (40., 4.), (0., 4.)),
    )


def _mesh(points, triangle=False):
    mesh = Mesh(nodes={i: np.array(p, dtype=float) for i, p in enumerate(points)})
    body = tuple(mesh.nodes)
    (mesh.tris if triangle else mesh.quads)[7] = body
    mesh.elements_of_face[1] = [7]
    return mesh


def _state(mesh):
    return (
        tuple((k, v.dtype.str, v.shape, v.tobytes()) for k, v in mesh.nodes.items()),
        mesh.quads, mesh.tris, mesh.elements_of_face, mesh.hybrid_diagnostics,
    )


def _repair(mesh, domain=None, callback=None):
    hybrid._repair_quad_first_quality(
        mesh, None, {1: domain or _domain()}, cancellation_check=callback,
        protected_interior_nodes=tuple(mesh.nodes),
    )


@pytest.mark.parametrize("points,triangle", [
    ([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)], False),
    ([(0, 0, 0), (40, 0, 0), (40, 1, 0), (0, 1, 0)], False),
    ([(0, 0), (2, 0), (2, 1), (0, 1)], False),
    ([(0, 0, 0), (2, 0, 0), (.8, .1, 0), (0, 1, 0)], False),
    ([(0, 0, 0), (2, 0, 0), (0, 1, 0)], True),
    ([(0, 0, 0), (40, 0, 0), (0, 1, 0)], True),
])
def test_repair_matches_uncached_topology_coordinates_diagnostics(
    monkeypatch, points, triangle,
):
    actual = _mesh(points, triangle)
    expected = deepcopy(actual)
    events = []
    old_events = []
    _repair(actual, callback=events.append)
    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", lambda *args: False)
    _repair(expected, callback=old_events.append)
    assert _state(actual) == _state(expected)
    assert events == old_events


def test_connected_aspect_pass_omits_only_unused_mapping(monkeypatch):
    calls = []
    original = hybrid.evaluate_mapping

    def count(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    # Instrument the native implementation, including its admission identity.
    monkeypatch.setattr(hybrid, "evaluate_mapping", count)
    monkeypatch.setattr(hybrid, "_REPAIR_NATIVE_MAPPING", count)
    mesh = _mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    _repair(mesh)
    assert calls == ["Q8"]
    calls.clear()
    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", lambda *args: False)
    _repair(mesh)
    assert calls == ["Q8", "Q8"]


def test_aspect_scan_rebuilds_after_corner_change_from_cancellation_callback(monkeypatch):
    # Two cells share coordinates; the second callback changes an aliased
    # corner array after the first trial has checked the earlier coordinates.
    def run():
        mesh = _mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
        mesh.quads[8] = mesh.quads[7]
        mesh.elements_of_face[1].append(8)
        events = []

        def callback(stage):
            events.append(stage)
            if len(events) == 2:
                mesh.nodes[2][0] = 2.25

        _repair(mesh, callback=callback)
        return _state(mesh), events

    actual = run()
    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", lambda *args: False)
    assert actual == run()


def test_cancellation_exception_and_partial_state_match(monkeypatch):
    class Cancelled(Exception):
        pass

    def run():
        mesh = _mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
        error = Cancelled("stop")

        def callback(stage):
            raise error

        with pytest.raises(Cancelled) as caught:
            _repair(mesh, callback=callback)
        assert caught.value is error
        return _state(mesh)

    actual = run()
    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", lambda *args: False)
    assert actual == run()


@pytest.mark.parametrize("normal", [(0., 0., 0.), (np.nan, 0., 1.)])
def test_invalid_normal_refusal_matches(monkeypatch, normal):
    def run():
        mesh = _mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
        with pytest.raises(ValueError) as caught:
            _repair(mesh, domain=replace(_domain(), normal=normal))
        return type(caught.value), str(caught.value), _state(mesh)

    actual = run()
    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", lambda *args: False)
    assert actual == run()


def _inputs():
    return np.arange(24, dtype=float).reshape(8, 3), np.array([0., 0., 1.])


def test_guard_accepts_independent_layouts_without_mutating_inputs():
    nodes, normal = _inputs()
    nodes[0, 0] = -0.
    node_bytes, normal_bytes = nodes.tobytes(), normal.tobytes()
    assert hybrid._repair_coordinates_only_safe(nodes.copy(order="F"), normal.copy(), "Q8")
    nodes.flags.writeable = False
    assert hybrid._repair_coordinates_only_safe(nodes, normal, "Q8")
    storage = np.empty((8, 6))
    alias = storage[:, ::2]
    alias[:] = nodes
    assert not alias.flags.c_contiguous
    assert hybrid._repair_coordinates_only_safe(alias, normal, "Q8")
    assert nodes.tobytes() == node_bytes and normal.tobytes() == normal_bytes


@pytest.mark.parametrize("kind", [
    "subclass", "normal_subclass", "2d", "float32", "empty", "large", "nan",
    "normal_zero", "normal_nan", "normal_tiny", "underflow_raise", "underflow_warn",
])
def test_unsupported_inputs_and_error_modes_keep_full_mapping(kind):
    nodes, normal = _inputs()
    if kind in ("subclass", "normal_subclass"):
        class Array(np.ndarray):
            pass
        if kind == "subclass":
            nodes = nodes.view(Array)
        else:
            normal = normal.view(Array)
    elif kind == "2d":
        nodes = nodes[:, :2]
    elif kind == "float32":
        nodes = nodes.astype(np.float32)
    elif kind == "empty":
        nodes = nodes[:0]
    elif kind == "large":
        nodes[0, 0] = 2.**201
    elif kind == "nan":
        nodes[0, 0] = np.nan
    elif kind == "normal_zero":
        normal[:] = 0.
    elif kind == "normal_nan":
        normal[0] = np.nan
    elif kind == "normal_tiny":
        normal[:] = 1.e-300
    under = {"underflow_raise": "raise", "underflow_warn": "warn"}.get(kind, "ignore")
    with np.errstate(under=under):
        assert not hybrid._repair_coordinates_only_safe(nodes, normal, "Q8")


@pytest.mark.parametrize("kind", ["2d", "large", "underflow", "replacement"])
def test_connected_unsupported_path_still_evaluates_aspect_mapping(monkeypatch, kind):
    original = hybrid.evaluate_mapping
    calls = []

    def count(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(hybrid, "evaluate_mapping", count)
    if kind != "replacement":
        monkeypatch.setattr(hybrid, "_REPAIR_NATIVE_MAPPING", count)
    mesh = _mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    if kind == "2d":
        mesh.nodes = {k: v[:2] for k, v in mesh.nodes.items()}
    elif kind == "large":
        mesh.nodes = {k: v * 2.**201 for k, v in mesh.nodes.items()}
    with np.errstate(under="raise" if kind == "underflow" else "ignore",
                     over="raise", invalid="raise", divide="raise"):
        _repair(mesh)
    assert calls == ["Q8", "Q8"]


def test_extreme_mapping_failure_matches_forced_original_path(monkeypatch):
    def run():
        mesh = _mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
        mesh.nodes = {k: v * 2.**300 for k, v in mesh.nodes.items()}
        with np.errstate(over="raise", under="ignore"):
            with pytest.raises(FloatingPointError) as caught:
                _repair(mesh)
        return type(caught.value), str(caught.value), _state(mesh)

    actual = run()
    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", lambda *args: False)
    assert actual == run()


def test_owner_subclasses_keep_original_trial_route(monkeypatch):
    class Owner(PlanarQuadDomain):
        pass

    def unexpected(*args):
        raise AssertionError("custom owner must not enter the coordinates-only path")

    monkeypatch.setattr(hybrid, "_repair_coordinates_only_safe", unexpected)
    _repair(_mesh([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)]), _domain(Owner))
