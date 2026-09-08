"""Transactional topology contracts for the bounded boundary-fan repair."""
from collections import Counter

import numpy as np
import pytest

from anymesher._boundary_fan_cavity import repair_boundary_fan


def fixture():
    points = np.array([[.785, 2.], [.785, 1.667], [.785, 1.778], [.785, 1.889],
                       [.393, 2.], [.689, 1.833], [.497, 1.660], [.652, 1.596], [.547, 1.864]])
    triangles = np.array([[0, 4, 8], [0, 5, 3], [0, 8, 5], [1, 2, 5],
                          [1, 5, 7], [2, 3, 5], [5, 6, 7], [5, 8, 6]])
    protected = np.array([[0, 4], [0, 3], [3, 2], [2, 1]])
    return points, triangles, protected


def edges(cells):
    return Counter(tuple(sorted((int(a), int(b)))) for row in cells
                   for a, b in zip(row, np.roll(row, -1)))


def test_interior_insertion_preserves_boundary_and_input():
    p, t, e = fixture()
    before = p.tobytes(), t.tobytes(), e.tobytes()
    result = repair_boundary_fan(p, t, e, [1], score=lambda x, c: (-len(x),),
                                 target_met=lambda x, c: True)
    assert result.improved and result.added_points == 1 and result.attempts <= 6
    assert len(result.triangles) == len(t) + 2
    assert result.points[:len(p)].tobytes() == p.tobytes()
    old_edges, new_edges = edges(t), edges(result.triangles)
    assert {edge for edge, count in old_edges.items() if count == 1} == {
        edge for edge, count in new_edges.items() if count == 1}
    assert all(new_edges[tuple(sorted(edge))] == old_edges[tuple(sorted(edge))] for edge in e)
    assert max(new_edges.values()) == 2
    assert (p.tobytes(), t.tobytes(), e.tobytes()) == before


def test_row_reordering_is_deterministic():
    p, t, e = fixture()
    a = repair_boundary_fan(p, t, e, [1], score=lambda x, c: (-len(x),))
    b = repair_boundary_fan(p, t[::-1], e[::-1], [len(t) - 2], score=lambda x, c: (-len(x),))
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.triangles, b.triangles)


@pytest.mark.parametrize('budget', [0, 1, 3])
def test_rejected_candidates_and_budget(budget):
    p, t, e = fixture()
    result = repair_boundary_fan(p, t, e, [1], score=lambda x, c: (0,), max_candidates=budget)
    assert not result.improved and result.attempts <= budget
    assert result.points.tobytes() == p.tobytes()
    assert result.triangles.tobytes() == t.tobytes()


def test_external_budget_and_cancellation():
    p, t, e = fixture()
    result = repair_boundary_fan(p, t, e, [1], score=lambda x, c: (-len(x),), consume_work=lambda: False)
    assert not result.improved and result.attempts == 0
    def cancel():
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        repair_boundary_fan(p, t, e, [1], score=lambda x, c: (-len(x),), cancellation_check=cancel)


def test_relaxation_cannot_change_protected_nodes():
    p, t, e = fixture()
    with pytest.raises(ValueError, match='invalid relaxed'):
        repair_boundary_fan(p, t, e, [1], score=lambda x, c: (-len(x),), relax=lambda x, c: x + .01)
    np.testing.assert_array_equal(p, fixture()[0])
