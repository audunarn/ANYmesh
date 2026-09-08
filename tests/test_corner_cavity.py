import numpy as np
import pytest

from anymesher._corner_cavity import repair_corner_cavity, _areas


def fixture():
    points = np.array([[0., 0.], [.3927, 0.], [0., .1111], [.2, .223],
                       [.3927, .5], [0., .5]])
    triangles = np.array([[0, 1, 2], [2, 1, 3], [1, 4, 3], [4, 5, 3], [5, 2, 3]])
    edges = np.array([[0, 1], [0, 2], [1, 4], [4, 5], [5, 2]])
    return points, triangles, edges


def score(points, triangles):
    cells = points[triangles]
    e = np.roll(cells, -1, axis=1) - cells
    lengths = np.linalg.norm(e, axis=2)
    angles = np.degrees(np.arccos(np.clip(np.sum(e * -np.roll(e, 1, axis=1), axis=2)
                                        / (lengths * np.roll(lengths, 1, axis=1)), -1, 1)))
    return (int(np.count_nonzero(_areas(points, triangles) <= 0)),
            int(np.count_nonzero(angles.min(axis=1) < 30)), -float(angles.min()))


def test_corner_insertion_preserves_boundary_and_coverage():
    p, t, edges = fixture()
    before = p.tobytes(), t.tobytes(), edges.tobytes()
    result = repair_corner_cavity(p, t, edges, [0], score=score)
    assert result.improved and result.added_points == 1
    assert len(result.triangles) == len(t) + 2
    assert result.attempts <= 256
    assert result.points[np.unique(edges)].tobytes() == p[np.unique(edges)].tobytes()
    assert np.all(_areas(result.points, result.triangles) > 0)
    assert np.isclose(_areas(result.points, result.triangles).sum(), _areas(p, t).sum(), atol=1e-14, rtol=0)
    assert score(result.points, result.triangles) < score(p, t)
    assert before == (p.tobytes(), t.tobytes(), edges.tobytes())
    repeated = repair_corner_cavity(p, t, edges, [0], score=score)
    assert result.points.tobytes() == repeated.points.tobytes()
    assert result.triangles.tobytes() == repeated.triangles.tobytes()


@pytest.mark.parametrize("budget", [{"max_candidates": 0}, {"max_insertions": 0}])
def test_zero_budget_preserves_exact_input(budget):
    p, t, edges = fixture()
    result = repair_corner_cavity(p, t, edges, [0], score=score, **budget)
    assert not result.improved and result.attempts == 0
    assert result.points.tobytes() == p.tobytes() and result.triangles.tobytes() == t.tobytes()


def test_rejected_candidate_and_cancel_do_not_mutate_inputs():
    p, t, edges = fixture()
    before = p.tobytes(), t.tobytes()
    result = repair_corner_cavity(p, t, edges, [0], score=lambda *args: (0,))
    assert not result.improved and result.points.tobytes() == p.tobytes()
    def cancel():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        repair_corner_cavity(p, t, edges, [0], score=score, cancellation_check=cancel)
    assert before == (p.tobytes(), t.tobytes())


def test_worst_corner_precedes_element_order_under_shared_budget():
    p, t, edges = fixture()
    easier = p.copy()
    easier[:, 0] *= .64
    points = np.vstack((easier, p + [1., 0.]))
    triangles = np.vstack((t, t + len(p)))
    protected = np.vstack((edges, edges + len(p)))
    result = repair_corner_cavity(points, triangles, protected, [0, len(t)], score=score)
    assert result.improved and result.moved_nodes == (len(p) + 3,)
    assert result.points[:len(p)].tobytes() == easier.tobytes()
    reverse = repair_corner_cavity(
        points, triangles[::-1], protected[::-1], [len(triangles) - 1, len(t) - 1], score=score,
    )
    assert reverse.points.tobytes() == result.points.tobytes()
    assert reverse.triangles.tobytes() == result.triangles.tobytes()


def test_existing_corner_fan_preserves_cavity_and_boundary():
    p, t, edges = fixture()
    t[:2] = [[0, 1, 3], [0, 3, 2]]
    # This callback isolates cavity construction; real-quality admission is
    # exercised separately above and by the physical component tests.
    result = repair_corner_cavity(p, t, edges, [0, 1], score=lambda x, cells: (-len(x),))
    assert result.improved and result.corner_node == 0 and result.added_points == 1
    assert np.all(_areas(result.points, result.triangles) > 0)
    assert result.points[np.unique(edges)].tobytes() == p[np.unique(edges)].tobytes()
    assert np.isclose(_areas(result.points, result.triangles).sum(), _areas(p, t).sum(), atol=1e-14, rtol=0)


def test_excluded_corner_is_not_revisited():
    p, t, edges = fixture()
    result = repair_corner_cavity(p, t, edges, [0], score=score, excluded_corners=[0])
    assert not result.improved and result.attempts == 0 and result.corner_node is None
    assert result.points.tobytes() == p.tobytes() and result.triangles.tobytes() == t.tobytes()


def test_expanded_corner_cavity_can_construct_three_corner_triangles():
    p, t, edges = fixture()
    # Isolate construction of the three-ray transition; physical policy is
    # independently applied by the cylindrical caller to every proposal.
    result = repair_corner_cavity(
        p, t, edges, [0], score=lambda x, cells: (-int(np.any(cells == 0, axis=1).sum()), -len(x)),
    )
    assert result.improved and result.added_points == 1
    assert int(np.any(result.triangles == 0, axis=1).sum()) == 3
    assert np.all(_areas(result.points, result.triangles) > 0)
    assert result.points[np.unique(edges)].tobytes() == p[np.unique(edges)].tobytes()
    assert np.isclose(_areas(result.points, result.triangles).sum(), _areas(p, t).sum(), atol=1e-14, rtol=0)


def test_candidate_is_relaxed_before_quality_comparison():
    p, t, edges = fixture()
    events = []
    def relax(x, cells):
        events.append("relax")
        return x
    def measure(x, cells):
        if len(x) > len(p):
            assert events[-1] == "relax"
            events.append("measure")
        return score(x, cells)
    repair_corner_cavity(p, t, edges, [0], score=measure, relax=relax, max_candidates=16)
    assert "measure" in events


def test_relaxation_cannot_modify_protected_coordinates():
    p, t, edges = fixture()
    raw = p.tobytes()
    def invalid(x, cells):
        x[0, 0] += .01
        return x
    with pytest.raises(ValueError, match="protected coordinates"):
        repair_corner_cavity(p, t, edges, [0], score=score, relax=invalid)
    assert p.tobytes() == raw


def test_external_work_budget_includes_candidate_construction():
    p, t, edges = fixture()
    remaining = [3]
    def consume():
        if remaining[0] == 0:
            return False
        remaining[0] -= 1
        return True
    result = repair_corner_cavity(p, t, edges, [0], score=score, consume_work=consume)
    assert result.attempts == 3 and remaining == [0]
