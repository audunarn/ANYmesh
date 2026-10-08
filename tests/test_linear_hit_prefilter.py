"""Exactness of the linear-face parameter hit prefilter and its face-local coordinate cache.

``_linear_face_parameter_hit`` skips an element whose corner box lies farther than
``_LINEAR_HIT_MARGIN`` from the target, and it caches face-local coordinates for one
connectivity application.  Both must leave the result unchanged, so the tests compare the
routine against a copy of its previous body on randomized linear meshes.
"""

from __future__ import annotations

import numpy as np
import pytest

from anymesher.mesh import Mesh
from anymesher.mesh_bvh import inverse_interpolate
from anymesher.structural_pipeline import (
    _LINEAR_HIT_MARGIN,
    StructuralMeshingPipeline,
    _ConnectivityIndex,
)


class _PlanarView:
    """Face-local coordinates are the (x, y) of a node, and every call is counted."""

    def __init__(self) -> None:
        self.calls = 0

    def face_local_uv(self, face_id, xyz):
        self.calls += 1
        return (float(xyz[0]), float(xyz[1]))


class _Stub:
    """Only the ``view`` attribute is read by the routine under test."""

    def __init__(self, view: _PlanarView) -> None:
        self.view = view


def _previous_routine(mesh, face_id, u, v, point, element_ids, view):
    """The routine before the prefilter and the cache, unchanged."""

    candidates = []
    target = np.array((float(u), float(v), 0.0))
    for element_id in sorted(element_ids):
        body = mesh.quads.get(int(element_id))
        family = "Q4"
        if body is None:
            body = mesh.tris.get(int(element_id))
            family = "T3"
        if body is None or len(body) not in (3, 4):
            continue
        nodes = tuple(int(node) for node in body)
        coords = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
        uv = np.asarray([view.face_local_uv(face_id, xyz) for xyz in coords])
        parametric = np.column_stack((uv, np.zeros(len(nodes))))
        hit = inverse_interpolate(family, parametric, target, tolerance=1e-8)
        if hit is None:
            continue
        parameter_projected = np.asarray(hit.weights @ coords, dtype=float)
        chord_gap = float(np.linalg.norm(point - parameter_projected))
        physical_hit = inverse_interpolate(
            family, coords, point, tolerance=max(1e-8, 2.0 * chord_gap)
        )
        if physical_hit is None:
            continue
        projected = np.asarray(physical_hit.point, dtype=float)
        candidates.append((
            float(np.linalg.norm(point - projected)), int(element_id), nodes,
            tuple(float(w) for w in physical_hit.weights), projected,
        ))
    if not candidates:
        return None
    _, _, nodes, weights, projected = min(candidates, key=lambda item: (item[0], item[1]))
    return nodes, weights, projected


def _random_linear_mesh(rng, cells: int) -> Mesh:
    """A jittered grid on z = 0, each cell a quad or split into two triangles."""

    mesh = Mesh()
    jitter = rng.uniform(-0.15, 0.15, size=(cells + 1, cells + 1, 2))
    node_of = {}
    for i in range(cells + 1):
        for j in range(cells + 1):
            node_of[i, j] = len(node_of) + 1
            x, y = i + jitter[i, j, 0], j + jitter[i, j, 1]
            mesh.nodes[node_of[i, j]] = np.array((x, y, 0.0))
    element = 100
    for i in range(cells):
        for j in range(cells):
            corners = (node_of[i, j], node_of[i + 1, j], node_of[i + 1, j + 1], node_of[i, j + 1])
            if rng.random() < 0.5:
                mesh.quads[element] = corners
                element += 1
            else:
                mesh.tris[element] = (corners[0], corners[1], corners[2])
                mesh.tris[element + 1] = (corners[0], corners[2], corners[3])
                element += 2
    return mesh


def _same(first, second) -> bool:
    if first is None or second is None:
        return first is second
    nodes_a, weights_a, projected_a = first
    nodes_b, weights_b, projected_b = second
    return (
        nodes_a == nodes_b
        and np.array_equal(np.asarray(weights_a), np.asarray(weights_b))
        and np.array_equal(np.asarray(projected_a), np.asarray(projected_b))
    )


def test_margin_is_far_above_the_inversion_residual_tolerance():
    assert _LINEAR_HIT_MARGIN >= 100 * 1.0e-8


@pytest.mark.parametrize("seed", range(12))
def test_prefilter_and_cache_leave_every_result_unchanged(seed):
    rng = np.random.default_rng(seed)
    mesh = _random_linear_mesh(rng, cells=5)
    view = _PlanarView()
    stub = _Stub(view)
    index = _ConnectivityIndex(mesh, stub)
    element_ids = sorted(set(mesh.quads) | set(mesh.tris))
    hits = 0
    for _ in range(25):
        u, v = rng.uniform(-0.5, 5.5, size=2)
        point = np.array((u + rng.normal(0.0, 0.02), v + rng.normal(0.0, 0.02), rng.normal(0.0, 0.05)))
        reference = _previous_routine(mesh, 1, u, v, point, element_ids, _PlanarView())
        current = StructuralMeshingPipeline._linear_face_parameter_hit(
            stub, mesh, 1, u, v, point, element_ids, index,
        )
        assert _same(reference, current)
        hits += current is not None
    assert hits > 0, "the randomized cases must exercise accepted hits too"


def test_inside_station_still_returns_its_element():
    mesh = Mesh()
    mesh.nodes.update({1: np.array((0.0, 0.0, 0.0)), 2: np.array((1.0, 0.0, 0.0)),
                       3: np.array((1.0, 1.0, 0.0)), 4: np.array((0.0, 1.0, 0.0))})
    mesh.quads[7] = (1, 2, 3, 4)
    view = _PlanarView()
    stub = _Stub(view)
    point = np.array((0.25, 0.5, 0.0))
    result = StructuralMeshingPipeline._linear_face_parameter_hit(
        stub, mesh, 1, 0.25, 0.5, point, [7], _ConnectivityIndex(mesh, stub),
    )
    assert result is not None
    nodes, weights, projected = result
    assert nodes == (1, 2, 3, 4)
    assert np.allclose(projected, point)
    assert np.isclose(sum(weights), 1.0)


def test_outside_target_is_rejected_before_any_inversion(monkeypatch):
    mesh = Mesh()
    mesh.nodes.update({1: np.array((0.0, 0.0, 0.0)), 2: np.array((1.0, 0.0, 0.0)),
                       3: np.array((1.0, 1.0, 0.0)), 4: np.array((0.0, 1.0, 0.0))})
    mesh.quads[7] = (1, 2, 3, 4)
    import anymesher.structural_pipeline as module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an element outside the target box must not be inverted")

    monkeypatch.setattr(module, "inverse_interpolate", forbidden)
    stub = _Stub(_PlanarView())
    point = np.array((2.0, 0.5, 0.0))
    assert StructuralMeshingPipeline._linear_face_parameter_hit(
        stub, mesh, 1, 2.0, 0.5, point, [7], _ConnectivityIndex(mesh, stub),
    ) is None


def test_face_local_coordinates_are_cached_for_one_application_only():
    rng = np.random.default_rng(3)
    mesh = _random_linear_mesh(rng, cells=4)
    view = _PlanarView()
    stub = _Stub(view)
    element_ids = sorted(set(mesh.quads) | set(mesh.tris))
    distinct_nodes = {
        (float(mesh.nodes[node][0]), float(mesh.nodes[node][1]), float(mesh.nodes[node][2]))
        for element in element_ids
        for node in (mesh.quads.get(element) or mesh.tris.get(element))
    }
    index = _ConnectivityIndex(mesh, stub)
    point = np.array((1.3, 1.1, 0.0))
    StructuralMeshingPipeline._linear_face_parameter_hit(stub, mesh, 1, 1.3, 1.1, point, element_ids, index)
    first_application = view.calls
    assert first_application == len(distinct_nodes)
    StructuralMeshingPipeline._linear_face_parameter_hit(stub, mesh, 1, 2.1, 0.7, point, element_ids, index)
    assert view.calls == first_application
    fresh = _ConnectivityIndex(mesh, stub)
    StructuralMeshingPipeline._linear_face_parameter_hit(stub, mesh, 1, 2.1, 0.7, point, element_ids, fresh)
    assert view.calls == 2 * first_application
