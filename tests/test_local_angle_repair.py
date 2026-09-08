import numpy as np
import pytest

from anymesher._local_angle_repair import repair_triangle_angles
from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate


POINTS = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.), (.15, .5)))
TRIANGLES = np.array(((0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)))
EDGES = np.array(((0, 1), (1, 2), (2, 3), (0, 3)))
SETTINGS = SurfaceMeshOptions(recombine=False, prefer_quality_policy=True)


def test_failed_interior_star_is_repaired_without_moving_boundary():
    before = POINTS.tobytes()
    result = repair_triangle_angles(POINTS, TRIANGLES, EDGES, SETTINGS)
    assert not _make_candidate(result.points, TRIANGLES, settings=SETTINGS).report['poor_element_ids']
    assert result.points[:4].tobytes() == POINTS[:4].tobytes()
    assert POINTS.tobytes() == before
    assert result.moved_nodes == (4,) and result.accepted_moves > 0


def test_repeated_repair_and_rotated_translated_star_are_deterministic():
    a = repair_triangle_angles(POINTS, TRIANGLES, EDGES, SETTINGS)
    b = repair_triangle_angles(POINTS, TRIANGLES, EDGES, SETTINGS)
    assert a.points.tobytes() == b.points.tobytes()
    rotation = np.array(((0., -1.), (1., 0.)))
    transformed = POINTS @ rotation.T + (3., -2.)
    c = repair_triangle_angles(transformed, TRIANGLES, EDGES, SETTINGS)
    assert not _make_candidate(c.points, TRIANGLES, settings=SETTINGS).report['poor_element_ids']
    assert c.points[:4].tobytes() == transformed[:4].tobytes()


def test_no_work_on_accepted_or_fully_protected_stars():
    regular = POINTS.copy()
    regular[4] = (.5, .5)
    accepted = repair_triangle_angles(regular, TRIANGLES, EDGES, SETTINGS)
    assert accepted.trials == 0 and accepted.points.tobytes() == regular.tobytes()
    fixed = np.vstack((EDGES, (0, 4)))
    refused = repair_triangle_angles(POINTS, TRIANGLES, fixed, SETTINGS)
    assert refused.trials == 0 and refused.points.tobytes() == POINTS.tobytes()


def test_trial_budget_is_enforced():
    result = repair_triangle_angles(POINTS, TRIANGLES, EDGES, SETTINGS, max_trials=1)
    assert result.trials == 1
    assert result.points[:4].tobytes() == POINTS[:4].tobytes()


def test_cancellation_during_trial_preserves_input():
    before = POINTS.tobytes()
    error = RuntimeError('cancelled')
    def cancel(phase):
        if phase.endswith('trial'):
            raise error
    with pytest.raises(RuntimeError) as caught:
        repair_triangle_angles(POINTS, TRIANGLES, EDGES, SETTINGS, cancellation_check=cancel)
    assert caught.value is error and POINTS.tobytes() == before


def test_growth_only_failure_is_repaired_without_relaxing_its_limit():
    points = POINTS.copy()
    points[4] = (.35, .5)
    settings = SurfaceMeshOptions(recombine=False, prefer_quality_policy=True,
                                  max_element_growth=1.05)
    before = _make_candidate(points, TRIANGLES, settings=settings).report
    assert before['elements_below_minimum_angle'] == 0
    assert before['elements_above_maximum_growth'] > 0
    result = repair_triangle_angles(points, TRIANGLES, EDGES, settings)
    after = _make_candidate(result.points, TRIANGLES, settings=settings).report
    assert not after['poor_element_ids']
    assert after['max_element_growth'] <= 1.05
    assert result.points[:4].tobytes() == points[:4].tobytes()


@pytest.mark.parametrize('point', ((.05, .5), (.15, .1), (.8, .85)))
def test_only_nonregressing_checkpoints_are_returned(point):
    points = POINTS.copy()
    points[4] = point
    before = _make_candidate(points, TRIANGLES, settings=SETTINGS).report
    result = repair_triangle_angles(points, TRIANGLES, EDGES, SETTINGS, max_trials=17)
    after = _make_candidate(result.points, TRIANGLES, settings=SETTINGS).report
    assert after['invalid_element_count'] == 0
    assert after['quality_violation_count'] <= before['quality_violation_count']
    assert result.points[:4].tobytes() == points[:4].tobytes()
