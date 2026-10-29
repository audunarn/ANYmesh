"""MutableT3Topology must not silently repair the orientation of caller input.

An inverted input row stays inverted so every consumer that refuses inverted
parents (curved boundary staging, validity gates) still sees and refuses it.
Rotation to the canonical start vertex keeps the winding and is allowed.
"""
import numpy as np
import pytest

from anymesher.native_v2 import MutableT3Topology, orient2d

POINTS = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])


def rows_orientation(topology):
    return [float(orient2d(*topology.points[row])) for row in topology.triangles]


def test_inverted_input_row_is_not_repaired_by_the_constructor():
    topology = MutableT3Topology(POINTS, [(0, 1, 2), (3, 2, 0)], [])
    orientations = rows_orientation(topology)
    assert orientations[0] > 0.
    assert orientations[1] < 0.


def test_inverted_row_is_not_repaired_when_every_row_is_inverted():
    topology = MutableT3Topology(POINTS, [(2, 1, 0), (0, 2, 3)[::-1]], [])
    assert all(value < 0. for value in rows_orientation(topology))


def test_positive_rows_are_rotated_to_the_canonical_start_keeping_order_and_owners():
    topology = MutableT3Topology(POINTS, [(2, 0, 1), (3, 0, 2)], [],
                                 triangle_owners=[11, 22])
    assert topology.triangles.tolist() == [[0, 1, 2], [0, 2, 3]]
    assert topology.triangle_owners.tolist() == [11, 22]
    assert all(value > 0. for value in rows_orientation(topology))


def test_winding_repair_is_an_explicit_opt_in():
    topology = MutableT3Topology(POINTS, [(0, 1, 2), (3, 2, 0)], [], normalize_winding=True)
    assert all(value > 0. for value in rows_orientation(topology))
    assert topology.triangles.tolist() == [[0, 1, 2], [0, 3, 2]]


def test_normalize_winding_must_be_a_bool():
    from anymesher.errors import MeshError
    with pytest.raises(MeshError):
        MutableT3Topology(POINTS, [(0, 1, 2)], [], normalize_winding=1)
