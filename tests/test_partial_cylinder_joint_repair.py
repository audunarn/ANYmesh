"""Actual failed partial-cylinder boundary cells, not a synthetic seed mesh."""
import json
from pathlib import Path

import numpy as np
import pytest
from anymesher._joint_triangle_repair import repair_joint_triangle_quality


def _fixture():
    data = json.loads(Path(__file__).with_name('cylindrical_partial_transition.json').read_text())
    return (np.asarray(data['points']), np.asarray(data['triangles']),
            np.asarray(data['protected']), [i - 1 for i in data['poor_elements']])


def _physical(points):
    return np.column_stack((np.cos(points[:, 0]), np.sin(points[:, 0]), points[:, 1]))


def _limits(points, cells):
    xyz = _physical(points)[cells]
    forward = np.roll(xyz, -1, axis=1) - xyz
    backward = np.roll(xyz, 1, axis=1) - xyz
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(forward, backward), axis=2),
                                   np.sum(forward * backward, axis=2)))
    perimeter = np.linalg.norm(forward, axis=2).sum(axis=1)
    incidence = {}
    for index, row in enumerate(cells):
        for a, b in zip(row, np.roll(row, -1)):
            incidence.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    growth = max(max(perimeter[a] / perimeter[b], perimeter[b] / perimeter[a])
                 for rows in incidence.values() if len(rows) == 2 for a, b in [rows])
    return float(angles.min()), float(growth)


def test_one_ring_repairs_actual_boundary_angle_defects_without_topology_changes():
    points, cells, protected, poor = _fixture()
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    assert _limits(points, cells)[0] < 30.
    result = repair_joint_triangle_quality(
        points, cells, protected, poor, min_angle=30., max_growth=1.5,
        max_trials=2048, evaluate_coordinates=_physical, neighbourhood_rings=1)
    minimum, growth = _limits(result.points, cells)
    assert minimum >= 30., (minimum, growth, result.final_penalty)
    assert growth <= 1.5
    assert result.trials <= 2048 and len(result.moved_nodes) <= 12
    fixed = sorted({int(n) for edge in protected for n in edge})
    assert result.points[fixed].tobytes() == points[fixed].tobytes()
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before
    repeated = repair_joint_triangle_quality(
        points, cells, protected, poor, min_angle=30., max_growth=1.5,
        max_trials=2048, evaluate_coordinates=_physical, neighbourhood_rings=1)
    assert repeated.points.tobytes() == result.points.tobytes()
    assert repeated.trials == result.trials


@pytest.mark.parametrize('rings', (True, -1, 2, 1.5))
def test_one_ring_budget_is_explicitly_bounded(rings):
    points, cells, protected, poor = _fixture()
    with pytest.raises(ValueError, match='invalid joint'):
        repair_joint_triangle_quality(points, cells, protected, poor,
            min_angle=30., max_growth=1.5, neighbourhood_rings=rings)


def test_one_ring_cancellation_and_zero_budget_do_not_mutate():
    points, cells, protected, poor = _fixture()
    before = points.tobytes(), cells.tobytes(), protected.tobytes()
    result = repair_joint_triangle_quality(points, cells, protected, poor,
        min_angle=30., max_growth=1.5, max_trials=0, neighbourhood_rings=1)
    assert result.trials == 0 and result.points.tobytes() == points.tobytes()
    error = RuntimeError('cancel one-ring repair')
    def cancel():
        raise error
    with pytest.raises(RuntimeError) as caught:
        repair_joint_triangle_quality(points, cells, protected, poor,
            min_angle=30., max_growth=1.5, neighbourhood_rings=1, cancellation_check=cancel)
    assert caught.value is error
    assert (points.tobytes(), cells.tobytes(), protected.tobytes()) == before
