from fractions import Fraction

import numpy as np

from anymesher._shared_triangle_split import propagate_triangle_split
from anymesher.native_v2 import ComponentSeedRegistry, MutableT3Topology
from test_cylindrical_quadratic_staging import stage, promote


def test_new_shared_split_allocates_only_after_valid_proposal():
    registry = ComponentSeedRegistry(100)
    topology = MutableT3Topology(((0., 0.), (1., 0.), (0., 1.)), ((0, 1, 2),),
                                splittable_edges={(0, 1): 7}, seed_registry=registry)
    result = topology.split_segment((0, 1), _split_proposal=lambda *args: (Fraction(1, 4), (.25, 0.), None))
    assert result['shared_node_id'] == 100
    assert result['reused_node_identity'] is False
    assert registry.assigned_node_ids == (100,)


def test_child_native_station_is_evaluated_once_and_promoted_conformally():
    source, state = stage()
    calls = []
    def evaluate(edge, t):
        calls.append((edge, t))
        return (t, 0., 0.)
    proposal = state.split_proposal(lambda edge, node, p: p[:2], evaluate_edge=evaluate)
    mid = Fraction.from_float(.3)
    assert proposal(7, Fraction(0), Fraction(1))[2] == 4
    state.record_split(7, 1, 2, 4, mid)
    propagate_triangle_split(state.mesh, (9,), (1, 2), 4, cache={})
    state.mesh.nodes_of_edge[7] = [1, 4, 2]
    child, point, reserved = proposal(7, mid, Fraction(0))
    assert reserved is None and child == Fraction.from_float(float(mid / 2))
    repeated = proposal(7, Fraction(0), mid)
    assert repeated[0] == child and repeated[1].tobytes() == point.tobytes()
    assert len(calls) == 1
    state.record_split(7, 1, 4, 7, child)
    state.mesh.nodes[7] = state.proposed_point(7, child)
    propagate_triangle_split(state.mesh, (9,), (1, 4), 7, cache={})
    state.mesh.nodes_of_edge[7] = [1, 7, 4, 2]
    result = promote(state)
    assert result.mesh.nodes_of_edge[7][::2] == [1, 7, 4, 2]
    assert any(row[:3] == (7, float(child), 7) for row in result.boundary_stations)
    assert all(len(row) == 6 for row in result.mesh.tris.values())
    assert result.mesh.nodes[4].tobytes() == source.nodes[4].tobytes()
