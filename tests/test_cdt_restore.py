import numpy as np
import pytest

from anymesher._cdt_restore import restore_constrained_delaunay


POINTS = np.array(((0., 0.), (2., 0.), (2., 1.), (0., 2.)))
TRIANGLES = np.array(((0, 1, 3), (1, 2, 3)), dtype=np.int64)
EDGES = np.array(((0, 1), (1, 2), (2, 3), (0, 3)), dtype=np.int64)


def test_illegal_diagonal_is_restored_without_changing_points():
    before = POINTS.tobytes(), TRIANGLES.tobytes()
    result = restore_constrained_delaunay(POINTS, TRIANGLES, EDGES, max_flips=8)
    assert result.flips == 1 and result.converged
    np.testing.assert_array_equal(result.triangles, ((0, 1, 2), (0, 2, 3)))
    assert before == (POINTS.tobytes(), TRIANGLES.tobytes())
    repeat = restore_constrained_delaunay(POINTS, result.triangles, EDGES, max_flips=8)
    assert repeat.flips == 0 and repeat.converged


def test_protected_diagonal_and_zero_budget_are_distinct():
    protected = restore_constrained_delaunay(POINTS, TRIANGLES, np.vstack((EDGES, (1, 3))), max_flips=8)
    assert protected.flips == 0 and protected.converged
    limited = restore_constrained_delaunay(POINTS, TRIANGLES, EDGES, max_flips=0)
    assert limited.flips == 0 and not limited.converged
    np.testing.assert_array_equal(limited.triangles, TRIANGLES)


def test_cancelled_restoration_has_no_partial_input_mutation():
    before = POINTS.tobytes(), TRIANGLES.tobytes()
    def cancel(phase):
        if phase.endswith('complete'):
            raise RuntimeError('cancel')
    with pytest.raises(RuntimeError, match='cancel'):
        restore_constrained_delaunay(POINTS, TRIANGLES, EDGES, max_flips=8, cancellation_check=cancel)
    assert before == (POINTS.tobytes(), TRIANGLES.tobytes())
