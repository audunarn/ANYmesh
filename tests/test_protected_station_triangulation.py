"""Source-only identity checks for staged authored component triangulation."""

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.native import NativeTriangulation
from anymesher.triangulation import constrained_planar_triangulation


SQUARE = np.array(((0., 0.), (1., 0.), (1., 1.), (0., 1.)))
STATIONS = {0: 100, 1: 101, 2: 102, 3: 103}


def test_protected_boundary_ids_survive_python_triangulation() -> None:
    result = constrained_planar_triangulation(
        SQUARE, (0, 1, 2, 3), backend="python", protected_node_ids=STATIONS
    )
    assert result.protected_node_rows == ((100, 0), (101, 1), (102, 2), (103, 3))
    np.testing.assert_array_equal(result.points[:4], SQUARE)
    assert result.points.flags.writeable is False


def test_repeated_constraint_endpoint_retains_one_registry_id() -> None:
    points = np.vstack((SQUARE, ((0.5, 0.5), (0.5, 0.5))))
    result = constrained_planar_triangulation(
        points, (0, 1, 2, 3), constraints=((0, 4), (5, 2)), backend="python",
        protected_node_ids={**STATIONS, 4: 104, 5: 104},
    )
    assert result.protected_node_rows[-1] == (104, 4)
    assert {tuple(edge) for edge in result.mandatory_segments} == {(0, 4), (2, 4)}


@pytest.mark.parametrize(
    ("points", "constraints", "stations", "message"),
    (
        (SQUARE, (), {0: 100, 1: 101, 2: 102}, "missing"),
        (np.vstack((SQUARE, ((0.5, 0.5), (0.5, 0.5)))), ((0, 4), (5, 2)),
         {**STATIONS, 4: 104}, "missing"),
        (np.vstack((SQUARE, ((0.5, 0.5), (0.5, 0.5)))), ((0, 4), (5, 2)),
         {**STATIONS, 4: 104, 5: 105}, "collapsed"),
        (np.vstack((SQUARE, ((0.5, 0.5), (0.6, 0.5)))), ((0, 4), (5, 2)),
         {**STATIONS, 4: 104, 5: 104}, "multiple"),
        (np.vstack((SQUARE, ((1.0e-13, 0.0),))), (),
         {**STATIONS, 4: 104}, "coordinate changed"),
        (SQUARE, (), {**STATIONS, 4: 104}, "outside"),
        (SQUARE, (), {**STATIONS, 0: True}, "nonnegative integer"),
    ),
)
def test_invalid_protected_bindings_fail_closed(points, constraints, stations, message) -> None:
    with pytest.raises(MeshError, match=message):
        constrained_planar_triangulation(
            points, (0, 1, 2, 3), constraints=constraints, backend="python",
            protected_node_ids=stations,
        )


def test_native_reordered_rows_cannot_carry_protected_ids() -> None:
    class ReorderedBoundary:
        name = "reordered-test-boundary"

        def triangulate(self, points, segments, outer_loop, hole_loops):
            reordered = np.ascontiguousarray(points[[1, 0, 2, 3]])
            return NativeTriangulation(
                reordered, np.ascontiguousarray(((1, 0, 2), (1, 2, 3)), dtype=np.int64)
            )

    with pytest.raises(MeshError, match="changed prepared PSLG point rows"):
        constrained_planar_triangulation(
            SQUARE, (0, 1, 2, 3), backend=ReorderedBoundary(),
            protected_node_ids=STATIONS,
        )


def test_cancellation_does_not_return_protected_binding() -> None:
    def cancel(stage: str) -> None:
        raise LookupError(stage)

    with pytest.raises(LookupError, match="python triangulation insertion start"):
        constrained_planar_triangulation(
            SQUARE, (0, 1, 2, 3), backend="python",
            protected_node_ids=STATIONS, cancellation_check=cancel,
        )
