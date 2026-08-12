from __future__ import annotations

import numpy as np
import pytest

from anymesher.optimization import constrained_smoothing, local_edge_flip


def _edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    return {
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for triangle in triangles
        for index in range(3)
    }


def test_constrained_smoothing_moves_only_the_interior_without_inversion() -> None:
    points = np.asarray(
        (
            (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
            (0.0, 0.5), (0.7, 0.3), (1.0, 0.5),
            (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
        )
    )
    cells = np.asarray(
        ((0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7))
    )

    result = constrained_smoothing(
        points,
        cells,
        metric=np.diag((4.0, 1.0)),
        relaxation=1.0,
        iterations=2,
    )

    assert result.points[4] == pytest.approx((0.5, 0.5))
    assert result.points[[0, 1, 2, 3, 5, 6, 7, 8]] == pytest.approx(
        points[[0, 1, 2, 3, 5, 6, 7, 8]]
    )
    assert result.moved_nodes.tolist() == [4]
    assert result.converged
    assert not result.points.flags.writeable

    fixed = constrained_smoothing(
        points,
        cells,
        constrained_edges=((1, 4),),
        relaxation=1.0,
        iterations=2,
    )
    assert fixed.points[4] == pytest.approx(points[4])


def test_local_edge_flip_queue_improves_and_preserves_constraints() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.2)))
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)))
    metric = np.tile(np.diag((4.0, 1.0)), (len(points), 1, 1))

    first = local_edge_flip(points, triangles, metric=metric)
    second = local_edge_flip(points, triangles, metric=metric)

    assert first.flip_count == 1
    assert (1, 3) in _edges(first.triangles)
    assert (0, 2) not in _edges(first.triangles)
    assert np.array_equal(first.triangles, second.triangles)
    assert first.queue_visits == second.queue_visits
    assert first.converged

    protected = local_edge_flip(
        points,
        triangles,
        protected_edges=((0, 2),),
        metric=metric,
    )
    assert protected.flip_count == 0
    assert (0, 2) in _edges(protected.triangles)
    assert not protected.triangles.flags.writeable
