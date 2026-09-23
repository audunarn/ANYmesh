from fractions import Fraction

import numpy as np
import pytest

from anymesher.errors import MeshError
from anymesher.native_v2 import ComponentSeedRegistry, MutableT3Topology


def make_topology(reverse=False):
    registry = ComponentSeedRegistry(100)
    topology = MutableT3Topology(
        ((0., 0.), (1., 0.), (0., 1.)), ((0, 1, 2),),
        splittable_edges={(0, 1): (7, int(reverse), int(not reverse))},
        seed_registry=registry,
    )
    return topology, registry


@pytest.mark.parametrize("reverse", (False, True))
def test_reserved_native_station_and_bytes_are_not_midpoint_recomputed(reverse):
    topology, registry = make_topology(reverse)
    station = Fraction.from_float(0.3)
    point = np.array((0.7 if reverse else 0.3, -0.0))
    seen = []

    def proposal(edge, lower, upper):
        seen.append((edge, lower, upper))
        return station, point, 23

    report = topology.split_segment((1, 0), _split_proposal=proposal)
    assert seen == [(7, Fraction(int(reverse)), Fraction(int(not reverse)))]
    assert report["station"] == (station.numerator, station.denominator)
    assert report["shared_node_id"] == 23
    assert topology.points[report["point_id"]].tobytes() == point.tobytes()
    assert registry.assigned_node_ids == (23,)
    assert topology._splittable_intervals[(1, 3)] == (7, Fraction(int(not reverse)), station)


@pytest.mark.parametrize("proposal", [
    None,
    (0.3, (0.3, 0.), 23),
    (Fraction(0), (0.3, 0.), 23),
    (Fraction(1, 3), (0.3, 0.), True),
    (Fraction(1, 3), (0.3, float("nan")), 23),
    (Fraction(1, 3), (0.3, 0.1), 23),
    (Fraction(1, 3), (0., 0.), 23),
    (Fraction(1, 3), (0.3,), 23),
])
def test_invalid_or_refused_proposal_preserves_topology_and_registry(proposal):
    topology, registry = make_topology()
    before = topology.canonical_export()
    with pytest.raises(MeshError):
        topology.split_segment((0, 1), _split_proposal=lambda *args: proposal)
    after = topology.canonical_export()
    assert all(a.tobytes() == b.tobytes() for a, b in zip(before, after))
    assert topology.epoch == 0
    assert registry.assigned_node_ids == ()


def test_commit_cancellation_does_not_reserve_midside_or_publish():
    topology, registry = make_topology()
    error = RuntimeError("cancelled")

    def cancel(phase):
        if phase == "native-v2 shared segment split commit":
            raise error

    with pytest.raises(RuntimeError) as raised:
        topology.split_segment(
            (0, 1), cancellation_check=cancel,
            _split_proposal=lambda *args: (Fraction.from_float(0.3), (0.3, 0.), 23),
        )
    assert raised.value is error
    assert topology.epoch == 0 and len(topology.points) == 3
    assert registry.assigned_node_ids == ()


def test_operational_proposal_failure_propagates_without_fallback():
    topology, registry = make_topology()
    error = RuntimeError("owner query failed")

    def proposal(*args):
        raise error

    with pytest.raises(RuntimeError) as raised:
        topology.split_segment((0, 1), _split_proposal=proposal)
    assert raised.value is error
    assert topology.epoch == 0 and registry.assigned_node_ids == ()


def test_default_split_is_still_midpoint():
    topology, registry = make_topology()
    report = topology.split_segment((0, 1))
    assert report["station"] == (1, 2)
    assert report["shared_node_id"] == 100
    np.testing.assert_array_equal(topology.points[3], (0.5, 0.))
