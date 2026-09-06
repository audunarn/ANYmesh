"""Bounded behavioral qualification of incremental triangle runtime state."""
import heapq

import numpy as np
import pytest

from anymesher._t3_incidence import T3IncidenceIndex
from anymesher._t3_runtime import CellGeometry, TriangleWorkQueue
from anymesher.native_v2 import MutableT3Topology
from anymesher.triangulation import orient2d


def test_geometry_records_survive_only_retained_cells():
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    index = T3IncidenceIndex(np.array([[0, 1, 2], [1, 3, 2]]))
    first = index.geometry(points, (0, 1, 2))
    second = index.geometry(points, (1, 3, 2))
    second.angles = (45., 90., 45.)
    updated = index.updated(np.array([[1, 3, 2]]))
    assert updated.geometry(points, (1, 3, 2)) is second
    assert (0, 1, 2) not in updated._geometry
    assert index.geometry(points, (0, 1, 2)) is first
    assert not second.coordinates.flags.writeable
    recreated = updated.updated(np.array([[0, 1, 2], [1, 3, 2]]))
    assert recreated.geometry(points, (0, 1, 2)) is not first


def test_queue_refresh_invalidation_reactivation_and_tokens():
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    cell = (0, 1, 2)
    geometry = CellGeometry.prepare(points, cell)
    tensors = np.repeat(np.eye(2)[None], 3, axis=0)
    queue = TriangleWorkQueue()
    evaluations = []

    def evaluate():
        evaluations.append(1)
        return 2., False

    for _ in range(2):
        queue.begin_round()
        queue.refresh(cell, geometry, tensors, False, evaluate)
        queue.finish_round()
    assert len(evaluations) == 1
    first = queue.pop()
    queue.begin_round()
    queue.refresh(cell, geometry, tensors, False, evaluate)
    queue.finish_round()
    second = queue.pop()
    assert second[:3] == first[:3]
    assert second[3] > first[3]
    assert len(evaluations) == 1
    queue.begin_round()
    queue.refresh(cell, geometry, tensors * 2, False, evaluate)
    queue.finish_round()
    assert len(evaluations) == 2
    queue.begin_round()
    queue.refresh(cell, geometry, tensors * 2, True, evaluate)
    queue.finish_round()
    assert len(evaluations) == 3
    queue.begin_round()
    queue.finish_round()
    assert not queue
    queue.begin_round()
    queue.refresh(cell, geometry, tensors * 2, True, evaluate)
    queue.finish_round()
    assert queue.pop()[3] > second[3]
    assert len(evaluations) == 4


def test_incremental_queue_matches_rebuild_under_local_and_global_changes():
    rng = np.random.default_rng(791)
    points = rng.random((27, 2))
    cells = [tuple(range(index, index + 3)) for index in range(0, 24, 3)]
    records = {cell: CellGeometry.prepare(points, cell) for cell in cells}
    queue = TriangleWorkQueue()
    for round_number in range(40):
        live = [cell for number, cell in enumerate(cells)
                if (number + round_number) % 7 != 0]
        queue.begin_round()
        expected = []
        for number, cell in enumerate(live):
            severity = float(1 + ((number + round_number // 3) % 5))
            limited = (number + round_number) % 9 == 0
            stamp = np.full((3, 2, 2), severity + 10 * limited)
            queue.refresh(cell, records[cell], stamp, limited,
                          lambda s=severity, l=limited: (s, l))
            if not limited and severity > 1 + 1.e-12:
                heapq.heappush(expected, (-severity, 1, cell))
        queue.finish_round()
        take = min(len(expected), round_number % 4 + 1)
        for _ in range(take):
            assert queue.pop()[:3] == heapq.heappop(expected)
        assert bool(queue) == bool(expected)


@pytest.mark.parametrize("scale", [1.e-8, 1., 1.e5])
def test_cached_location_preserves_ordered_tolerance_predicate(scale):
    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]) * scale
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    topology = MutableT3Topology(points, triangles, [])
    queries = [np.array([.5, .5]) * scale,
               np.array([.8, .1]) * scale,
               np.array([-.000001, .5]) * scale,
               np.array([1.1, 1.1]) * scale]
    for point in queries:
        expected = None
        for row, triangle in enumerate(topology.triangles):
            coordinates = topology.points[triangle]
            tolerance = 64 * np.finfo(float).eps * max(
                float(np.ptp(coordinates, axis=0).max()), 1.) ** 2
            signs = [orient2d(coordinates[i], coordinates[(i+1) % 3], point)
                     for i in range(3)]
            if min(signs) >= -tolerance:
                expected = row
                break
        assert topology.locate(point) == expected
        assert topology.locate(point) == expected


def test_insertion_bearing_frontal_matches_uncached_rebuild(monkeypatch):
    import anymesher.native_v2 as native_v2
    from anymesher import NativeMeshingOptions
    from anymesher.triangulation import PlanarTriangulation

    points = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    segments = np.array([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=np.int64)
    seed = PlanarTriangulation(
        points=points, triangles=np.array([[0, 1, 2], [0, 2, 3]]),
        segments=segments, boundary_segments=segments,
        mandatory_segments=np.empty((0, 2), dtype=np.int64),
        outer_loop=np.arange(4, dtype=np.int64), hole_loops=(),
    )
    options = NativeMeshingOptions(
        point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
        max_insertions=8, max_topology_operations=128,
    )
    actual, actual_report = native_v2.frontal_delaunay_refine(seed, options, target_size=.25)

    class RebuiltQueue(TriangleWorkQueue):
        def begin_round(self):
            self._records.clear()
            self._active.clear()
            self._heap.clear()
            super().begin_round()

    monkeypatch.setattr(native_v2, "TriangleWorkQueue", RebuiltQueue)
    monkeypatch.setattr(T3IncidenceIndex, "geometry",
                        lambda self, points, cell: CellGeometry.prepare(points, tuple(cell)))
    expected, expected_report = native_v2.frontal_delaunay_refine(seed, options, target_size=.25)
    assert actual_report["insertions"] > 0
    assert actual_report == expected_report
    np.testing.assert_array_equal(actual.points, expected.points)
    np.testing.assert_array_equal(actual.triangles, expected.triangles)
    np.testing.assert_array_equal(actual.segments, expected.segments)
    np.testing.assert_array_equal(seed.points, points)


def test_nan_priority_matches_reference_positive_admission():
    cell = (0, 1, 2)
    geometry = CellGeometry.prepare(np.array([[0., 0.], [1., 0.], [0., 1.]]), cell)
    queue = TriangleWorkQueue()
    queue.begin_round()
    queue.refresh(cell, geometry, np.eye(2), False, lambda: (float("nan"), False))
    queue.finish_round()
    assert not queue


def test_queue_maintenance_cancellation_is_not_swallowed():
    def cancel(phase):
        raise RuntimeError(phase)
    queue = TriangleWorkQueue(cancel, 1)
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    cell = (0, 1, 2)
    queue.begin_round()
    queue.refresh(cell, CellGeometry.prepare(points, cell), np.eye(2),
                  False, lambda: (2., False))
    with pytest.raises(RuntimeError, match="queue maintenance"):
        queue.finish_round()
