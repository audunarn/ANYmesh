"""Speed slice (tripod 2026-11-25): each shortcut is decision-identical to the route it replaces.

1. ``_finite_hull_seed``: bounding-box prefilter + tuple coordinates == the plain scan (driven through an
   ndarray subclass, which keeps the original code path), with far fewer orientation predicates.
4. ``_shell_owner_authority``: a memo of earlier evaluations on the same inputs returns the same normals.
5. ``_candidate_bisection``: the caller's incidence map gives the same candidate as a rebuilt one, and the
   repair loop no longer rebuilds the whole-mesh map per candidate.
"""
from __future__ import annotations

from math import radians, tan

import numpy as np
import pytest

from anygeometry import GeometryModel

from anymesher import Mesh, generate_hybrid_mesh_result, repair_s3_admission
from anymesher import s3_production, s3_repair, triangulation


class _PlainArray(np.ndarray):
    """Same values, but not exactly ``np.ndarray``: ``_finite_hull_seed`` keeps its original scan."""


def _plain(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).view(_PlainArray)


def _cases():
    rng = np.random.default_rng(20261125)
    made = []
    for count in (6, 14, 30):
        # Quad-seed-like inputs: a lattice (cocircular quadruples), collinear boundary stations,
        # interior jitter and points exactly on lattice edges.
        xs, ys = np.meshgrid(np.linspace(0.0, 2.0, count), np.linspace(0.0, 1.0, max(3, count // 2)))
        lattice = np.c_[xs.ravel(), ys.ravel()]
        on_edges = np.c_[rng.integers(0, count - 1, count) * (2.0 / (count - 1)) + 1.0 / (count - 1),
                         np.zeros(count)]
        jitter = rng.uniform(0.0, 1.0, (count, 2)) * [2.0, 1.0]
        made.append(np.unique(np.r_[lattice, on_edges, jitter], axis=0))
    # A rotated, offset lattice (inexact coordinates, exact-fallback orientations).
    angle = 0.3
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xs, ys = np.meshgrid(np.linspace(0.0, 1.0, 12), np.linspace(0.0, 1.0, 9))
    made.append(np.unique(np.c_[xs.ravel(), ys.ravel()] @ rotation.T + [1.0e3, -2.5e2], axis=0))
    # Random points, and a convex polygon with collinear sides only.
    made.append(np.unique(rng.uniform(-1.0, 1.0, (120, 2)), axis=0))
    side = np.linspace(0.0, 1.0, 9)
    made.append(np.unique(np.r_[np.c_[side, 0 * side], np.c_[1 + 0 * side, side], np.c_[side, 1 + 0 * side],
                                np.c_[0 * side, side], [[0.5, 0.5]]], axis=0))
    return made


@pytest.mark.parametrize("index", range(6))
def test_finite_hull_seed_prefilter_matches_the_plain_scan(index):
    points = _cases()[index]
    assert type(points) is np.ndarray and points.dtype == np.float64
    assert triangulation._finite_hull_seed(points) == triangulation._finite_hull_seed(_plain(points))


def test_finite_hull_seed_non_float64_input_keeps_the_plain_scan():
    points = _cases()[0]
    as_float32 = points.astype(np.float32)
    assert triangulation._finite_hull_seed(as_float32) == triangulation._finite_hull_seed(
        as_float32.view(_PlainArray))


def test_finite_hull_seed_prefilter_skips_cells_that_cannot_contain_the_point(monkeypatch):
    # Fails on the old code, whose location scan tests every cell for every inserted point.
    points = _cases()[2]
    calls = [0]
    original = triangulation.orient2d

    def counted(*args):
        calls[0] += 1
        return original(*args)

    monkeypatch.setattr(triangulation, "orient2d", counted)
    fast = triangulation._finite_hull_seed(points)
    fast_calls, calls[0] = calls[0], 0
    plain = triangulation._finite_hull_seed(_plain(points))
    assert fast == plain
    assert fast_calls < 0.75 * calls[0]


def _square_mesh():
    geometry = GeometryModel()
    face_id = geometry.add_plate(
        geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))))
    geometry.add_sheet((face_id,))
    result = generate_hybrid_mesh_result(geometry, target_size=0.25, strategy="native",
                                         native_backend="python", recombine=False,
                                         structural_preparation=False)
    return geometry, result.mesh


def test_owner_normal_memo_returns_the_same_normals_and_skips_unchanged_shells(monkeypatch):
    geometry, mesh = _square_mesh()
    fresh, fresh_sources = s3_production._shell_owner_authority(geometry, mesh)
    memo = {"geometry": geometry, "normals": {}}
    calls = [0]
    original = s3_production._physical_owner_normal

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(s3_production, "_physical_owner_normal", counted)
    first, first_sources = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert calls[0] == len(mesh.shells)
    second, second_sources = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert calls[0] == len(mesh.shells)            # every shell reused
    assert first == second == fresh and first_sources == second_sources == fresh_sources
    # A moved corner is a different input: only the shells using that node are evaluated again.
    node = next(iter(sorted(mesh.nodes)))
    mesh.nodes[node] = mesh.nodes[node] + np.array([1.0e-3, 0.0, 0.0])
    users = [element for element in mesh.shells if node in mesh.corners_of(element)]
    calls[0] = 0
    moved, _ = s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    assert calls[0] == len(users)
    monkeypatch.setattr(s3_production, "_physical_owner_normal", original)
    assert moved == s3_production._shell_owner_authority(geometry, mesh)[0]


def test_owner_normal_memo_of_another_geometry_object_is_ignored(monkeypatch):
    geometry, mesh = _square_mesh()
    memo = {"geometry": geometry, "normals": {}}
    s3_production._shell_owner_authority(geometry, mesh, _normal_memo=memo)
    other, _ = _square_mesh()
    calls = [0]
    original = s3_production._physical_owner_normal

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(s3_production, "_physical_owner_normal", counted)
    s3_production._shell_owner_authority(other, mesh, _normal_memo=memo)
    assert calls[0] == len(mesh.shells)


OWNER = np.asarray((0.0, 0.0, 1.0))


def _sliver_patch(angle_deg: float) -> Mesh:
    return Mesh(
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, tan(radians(angle_deg)), 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
            5: np.asarray((2.0, 0.5, 0.0)),
        },
        tris={10: (1, 2, 3), 20: (1, 3, 4), 30: (2, 5, 3)},
        elements_of_face={7: [10, 20, 30]},
        elements_of_sheet={8: [10, 20, 30]},
    )


def _signature(made):
    mesh = made[0]
    meshed = None if mesh is None else (
        tuple((k, tuple(float(v) for v in mesh.nodes[k])) for k in sorted(mesh.nodes)),
        tuple((k, tuple(mesh.tris[k])) for k in sorted(mesh.tris)),
        tuple((k, tuple(v)) for k, v in sorted(mesh.elements_of_face.items())))
    owners = None if made[1] is None else tuple((k, tuple(v)) for k, v in sorted(made[1].items()))
    return meshed, owners, made[2], made[3], made[4]


def test_candidate_bisection_with_the_callers_incidence_matches_a_rebuild():
    mesh = _sliver_patch(10.0)
    owners = {element: OWNER for element in mesh.tris}
    selected = frozenset(mesh.tris)
    incidence = s3_repair._shell_incidence(mesh)
    for edge in sorted(incidence):
        attached = tuple(sorted(owner for kind, owner in incidence[edge] if kind == "tri"))
        rebuilt = s3_repair._candidate_bisection(mesh, edge, attached, owners, selected,
                                                 next_node_id=6, next_element_id=31)
        reused = s3_repair._candidate_bisection(mesh, edge, attached, owners, selected,
                                                next_node_id=6, next_element_id=31, incidence=incidence)
        assert _signature(reused) == _signature(rebuilt)


def test_repair_bisection_does_not_rebuild_the_incidence_map(monkeypatch):
    # Fails on the old code, where every bisection candidate rebuilt the whole-mesh map.
    inside = [False]
    rebuilt_inside = [0]
    called = [0]
    original_incidence = s3_repair._shell_incidence
    original_bisection = s3_repair._candidate_bisection

    def incidence(mesh):
        if inside[0]:
            rebuilt_inside[0] += 1
        return original_incidence(mesh)

    def bisection(*args, **kwargs):
        called[0] += 1
        inside[0] = True
        try:
            return original_bisection(*args, **kwargs)
        finally:
            inside[0] = False

    monkeypatch.setattr(s3_repair, "_shell_incidence", incidence)
    monkeypatch.setattr(s3_repair, "_candidate_bisection", bisection)
    mesh = _sliver_patch(10.0)
    try:
        repair_s3_admission(mesh, element_owner_normals={element: tuple(OWNER) for element in mesh.tris})
    except Exception:
        pass  # the outcome is covered elsewhere; this test only counts rebuilds
    assert called[0] > 0 and rebuilt_inside[0] == 0
