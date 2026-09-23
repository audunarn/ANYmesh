import numpy as np
import pytest

from anymesher._angle_star_projection import angle_star_target


def fixture():
    return (np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [.15, .5]]),
            np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]))


def minimum_angle(points, triangles):
    cells = points[triangles]
    edges = np.roll(cells, -1, axis=1) - cells
    lengths = np.linalg.norm(edges, axis=2)
    cosine = np.sum(edges * -np.roll(edges, 1, axis=1), axis=2) / (lengths * np.roll(lengths, 1, axis=1))
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).min())


def test_feasible_star_target_improves_angles_without_input_mutation():
    points, triangles = fixture()
    original = points.tobytes(), triangles.tobytes()
    target = angle_star_target(points, triangles, 4, 30.25)
    assert target is not None
    candidate = points.copy()
    candidate[4] = target
    assert minimum_angle(candidate, triangles) >= 30.25 - 1e-10
    assert original == (points.tobytes(), triangles.tobytes())
    assert np.array_equal(target, angle_star_target(points, triangles, 4, 30.25))


def test_projection_is_scale_translation_and_rotation_covariant():
    points, triangles = fixture()
    target = angle_star_target(points, triangles, 4, 30.25)
    rotation = np.array([[.6, -.8], [.8, .6]])
    transformed = points @ rotation.T * 7 + np.array([3., -5.])
    actual = angle_star_target(transformed, triangles, 4, 30.25)
    assert np.allclose(actual, target @ rotation.T * 7 + [3., -5.], atol=1e-11, rtol=0)


def test_zero_projection_budget_returns_no_candidate():
    points, triangles = fixture()
    assert angle_star_target(points, triangles, 4, 30.25, max_sweeps=0) is None


def test_nonpositive_triangle_is_rejected():
    points, triangles = fixture()
    with pytest.raises(ValueError, match="non-positive"):
        angle_star_target(points, triangles[:, [0, 2, 1]], 4, 30.25)


def test_external_work_budget_stops_projection_without_mutation():
    points, triangles = fixture()
    original = points.tobytes()
    remaining = [3]
    def consume():
        if not remaining[0]:
            return False
        remaining[0] -= 1
        return True
    assert angle_star_target(points, triangles, 4, 30.25, consume_work=consume) is None
    assert remaining == [0] and points.tobytes() == original


def test_cancellation_propagates_without_mutation():
    points, triangles = fixture()
    original = points.tobytes()
    def cancel():
        raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        angle_star_target(points, triangles, 4, 30.25, cancellation_check=cancel)
    assert points.tobytes() == original
