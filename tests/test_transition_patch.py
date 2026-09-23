import numpy as np
import pytest

from anymesher._transition_patch import repair_transition_patch


def angle_score(points, triangles):
    q = points[triangles]
    e = np.roll(q, -1, axis=1) - q
    lengths = np.linalg.norm(e, axis=2)
    cosine = np.sum(e * -np.roll(e, 1, axis=1), axis=2) / (lengths * np.roll(lengths, 1, axis=1))
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return int(np.count_nonzero(angles.min(axis=1) < 30)), -float(angles.min())


def case():
    return (np.array([[0., 0.], [2., 0.], [2., 1.], [0., 2.]]),
            np.array([[0, 1, 3], [1, 2, 3]]),
            np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))


def test_improving_patch_preserves_protected_points_and_source():
    points, triangles, edges = case()
    before = points.tobytes(), triangles.tobytes(), edges.tobytes()
    result = repair_transition_patch(points, triangles, edges, [0, 1], score=angle_score)
    assert result.improved and result.committed_flips == 1
    assert angle_score(result.points, result.triangles) < angle_score(points, triangles)
    assert result.points.tobytes() == points.tobytes()
    assert before == (points.tobytes(), triangles.tobytes(), edges.tobytes())
    repeated = repair_transition_patch(points, triangles, edges, [0, 1], score=angle_score)
    assert repeated.triangles.tobytes() == result.triangles.tobytes()


def test_protected_diagonal_and_zero_budget_do_not_change_input():
    points, triangles, edges = case()
    for protected, budget in [(np.vstack((edges, [1, 3])), 64), (edges, 0)]:
        result = repair_transition_patch(points, triangles, protected, [0, 1],
                                         score=angle_score, max_attempts=budget)
        assert not result.improved
        assert result.triangles.tobytes() == triangles.tobytes()


def test_cancellation_does_not_publish_an_intermediate():
    points, triangles, edges = case()
    before = points.tobytes(), triangles.tobytes()
    def cancel():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        repair_transition_patch(points, triangles, edges, [0], score=angle_score,
                                cancellation_check=cancel)
    assert before == (points.tobytes(), triangles.tobytes())


def test_bad_relaxation_cannot_move_protected_nodes():
    points, triangles, edges = case()
    def move_boundary(p, cells):
        p[0, 0] += 0.1
        return p
    with pytest.raises(ValueError, match="protected coordinates"):
        repair_transition_patch(points, triangles, edges, [0], score=angle_score,
                                relax=move_boundary)


def test_equal_score_never_changes_connectivity():
    points, triangles, edges = case()
    result = repair_transition_patch(points, triangles, edges, [0], score=lambda p, t: (0,))
    assert not result.improved and result.committed_flips == 0
    assert result.triangles.tobytes() == triangles.tobytes()


def test_target_gate_stops_after_qualified_improvement():
    points, triangles, edges = case()
    result = repair_transition_patch(points, triangles, edges, [0, 1],
                                     score=angle_score, target_met=lambda p, t: angle_score(p, t)[0] == 0)
    assert result.improved and result.attempts == 1
    accepted = repair_transition_patch(result.points, result.triangles, edges, [],
                                       score=angle_score, target_met=lambda p, t: angle_score(p, t)[0] == 0)
    assert accepted.attempts == 0 and not accepted.improved


def test_external_work_budget_stops_before_evaluating_candidate():
    points, triangles, edges = case()
    calls = []
    def relax(p, t):
        calls.append(1)
        return p
    result = repair_transition_patch(points, triangles, edges, [0, 1],
                                     score=angle_score, relax=relax,
                                     work_available=lambda: False)
    assert result.attempts == 0 and not calls and not result.improved
