import numpy as np
import pytest

from anymesher._joint_triangle_repair import repair_joint_triangle_quality
from anymesher.surface_mesh import _make_candidate


def fixture():
    return (np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.15, .5]]),
            np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]),
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))


def test_joint_angle_and_perimeter_growth_repair():
    p, t, edges = fixture()
    before = p.tobytes(), t.tobytes()
    r = repair_joint_triangle_quality(p, t, edges, range(4), min_angle=30, max_growth=1.5)
    quality = _make_candidate(r.points, t).report
    assert quality["min_angle"] >= 30 and quality["max_element_growth"] <= 1.5
    assert r.final_penalty < r.initial_penalty and r.trials <= 2048
    assert r.points[:4].tobytes() == p[:4].tobytes()
    assert before == (p.tobytes(), t.tobytes())
    repeat = repair_joint_triangle_quality(p, t, edges, range(4), min_angle=30, max_growth=1.5)
    assert repeat.points.tobytes() == r.points.tobytes()


@pytest.mark.parametrize("budget", [0, 1, 3])
def test_exact_work_budget(budget):
    p, t, edges = fixture()
    r = repair_joint_triangle_quality(p, t, edges, range(4), min_angle=30, max_growth=1.5, max_trials=budget)
    assert r.trials <= budget
    assert r.points[:4].tobytes() == p[:4].tobytes()


def test_cancel_does_not_mutate():
    p, t, edges = fixture()
    raw = p.tobytes()
    def cancel():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        repair_joint_triangle_quality(p, t, edges, range(4), min_angle=30, max_growth=1.5,
                                      cancellation_check=cancel)
    assert p.tobytes() == raw


def test_joint_objective_uses_owner_coordinates_when_supplied():
    p = np.array([[0., 0.], [.4, 0.], [.2, .1152]])
    t = np.array([[0, 1, 2]])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    owner = lambda x: np.column_stack((np.cos(x[:, 0]), np.sin(x[:, 0]), x[:, 1]))
    result = repair_joint_triangle_quality(p, t, edges, [0], min_angle=30, max_growth=1.5,
                                          evaluate_coordinates=owner)
    assert result.initial_penalty == 0 and result.final_penalty == 0
    assert result.points.tobytes() == p.tobytes()


def test_search_margin_respects_a_tight_three_triangle_corner():
    angle = np.radians([0., 30., 60., 90.])
    p = np.vstack(([0., 0.], np.column_stack((np.cos(angle), np.sin(angle)))))
    t = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
    result = repair_joint_triangle_quality(p, t, edges, range(3), min_angle=30, max_growth=1.5)
    assert result.initial_penalty < 1e-24
    assert result.trials == 0 and result.points.tobytes() == p.tobytes()


def test_near_threshold_repair_meets_the_unrounded_policy():
    p, t, edges = fixture()
    p[4] = [.5 / np.sqrt(3) - 1e-6, .5]
    assert _make_candidate(p, t).report["min_angle"] < 30
    result = repair_joint_triangle_quality(p, t, edges, range(4), min_angle=30, max_growth=1.5)
    quality = _make_candidate(result.points, t).report
    assert quality["min_angle"] >= 30
    assert quality["max_element_growth"] <= 1.5
    assert result.trials <= 2048
