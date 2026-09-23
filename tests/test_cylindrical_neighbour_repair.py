"""Detached shared-boundary repair must keep coordinates and topology exact."""

from types import SimpleNamespace

import numpy as np
import pytest

from anymesher._shared_triangle_split import repair_triangle_face
from anymesher.errors import MeshError


def _case():
    points = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., .25)))
    mesh = SimpleNamespace(
        nodes={i + 1: tuple((*point, 0.)) for i, point in enumerate(points)},
        tris={1: (1, 2, 3), 2: (1, 3, 4)},
        elements_of_face={1: [1, 2]},
    )
    return mesh, points


def test_neighbour_repair_improves_quality_without_moving_any_node():
    outputs = []
    for _ in range(2):
        mesh, points = _case()
        original_nodes = dict(mesh.nodes)
        cache = {}
        report = repair_triangle_face(mesh, 1, points, cache=cache)
        assert report["selected"]
        assert report["initial_score"][2] > 5.0
        assert report["final_score"][2] < 5.0
        assert mesh.nodes == original_nodes
        assert mesh.elements_of_face == {1: [1, 2]}
        assert (1, 3) not in cache[1]
        assert cache[1][(2, 4)] == {1, 2}
        outputs.append((mesh.tris, report))
    assert outputs[0] == outputs[1]


def test_neighbour_repair_cannot_flip_a_protected_internal_edge():
    mesh, points = _case()
    before = dict(mesh.tris)
    report = repair_triangle_face(mesh, 1, points, protected_edges=((1, 3),), cache={})
    assert not report["selected"]
    assert report["flips"] == 0
    assert mesh.tris == before


def test_neighbour_repair_rejects_malformed_candidate_without_publication(monkeypatch):
    import anymesher.optimization as optimization

    mesh, points = _case()
    before = dict(mesh.tris)
    monkeypatch.setattr(optimization, "local_edge_flip", lambda *args, **kwargs: SimpleNamespace(
        triangles=np.asarray(((0, 0, 2), (0, 2, 3))), flip_count=1, queue_visits=1))
    with pytest.raises(MeshError, match="protected topology or chart coverage"):
        repair_triangle_face(mesh, 1, points, cache={})
    assert mesh.tris == before


def _interior_case():
    points = np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.03, .03)))
    mesh = SimpleNamespace(
        nodes={i + 1: tuple((*point, 0.)) for i, point in enumerate(points)},
        tris={i + 1: (i + 1, (i + 1) % 4 + 1, 5) for i in range(4)},
        elements_of_face={1: [1, 2, 3, 4]},
    )
    return mesh, points


def _lift(points):
    return np.column_stack((points, np.zeros(len(points))))


def test_smoothing_repairs_only_interior_and_reports_committed_work():
    mesh, points = _interior_case()
    protected = {node: mesh.nodes[node] for node in range(1, 5)}
    report = repair_triangle_face(mesh, 1, points, cache={}, physical_evaluator=_lift)
    assert report["selected"]
    assert report["initial_score"][2] > 5.0
    assert report["final_score"][2] <= 5.0
    assert report["moved_nodes"] == 1
    assert 0 < report["smoothing_iterations"] <= 4
    assert not report["protected_coordinates_changed"]
    assert {node: mesh.nodes[node] for node in range(1, 5)} == protected


def test_smoothing_cancellation_before_commit_preserves_mesh_and_cache():
    mesh, points = _interior_case()
    nodes, triangles = dict(mesh.nodes), dict(mesh.tris)
    cache = {}

    def cancel(stage):
        if stage.endswith("before publish"):
            raise RuntimeError("cancelled before neighbour publication")

    with pytest.raises(RuntimeError, match="cancelled before neighbour publication"):
        repair_triangle_face(mesh, 1, points, cache=cache, physical_evaluator=_lift, cancellation_check=cancel)
    assert mesh.nodes == nodes
    assert mesh.tris == triangles
    assert cache == {}


def test_owner_evaluation_error_is_not_relabelled_or_published():
    mesh, points = _interior_case()
    nodes, triangles = dict(mesh.nodes), dict(mesh.tris)

    def fail(points):
        raise RuntimeError("owner evaluation failed")

    with pytest.raises(RuntimeError, match="owner evaluation failed"):
        repair_triangle_face(mesh, 1, points, cache={}, physical_evaluator=fail)
    assert mesh.nodes == nodes
    assert mesh.tris == triangles


def test_smoothing_cannot_move_a_protected_boundary(monkeypatch):
    import anymesher.optimization as optimization

    mesh, points = _interior_case()
    nodes, triangles = dict(mesh.nodes), dict(mesh.tris)
    changed = points.copy()
    changed[0, 0] += .1
    monkeypatch.setattr(optimization, "constrained_smoothing", lambda *args, **kwargs: SimpleNamespace(
        points=changed, iterations=1))
    with pytest.raises(MeshError, match="changed protected coordinates"):
        repair_triangle_face(mesh, 1, points, cache={}, physical_evaluator=_lift)
    assert mesh.nodes == nodes
    assert mesh.tris == triangles
