import numpy as np
import pytest

from anymesher.mesh import Mesh
from anymesher.errors import MeshError
from anymesher._shared_triangle_split import propagate_triangle_split
from anymesher.native_v2 import ComponentSeedRegistry, MutableT3Topology


def _mesh():
    return Mesh(
        nodes={1: np.array((0., 0., 0.)), 2: np.array((1., 0., 0.)),
               3: np.array((0., 1., 0.)), 4: np.array((0., -1., 0.)),
               5: np.array((.5, 0., 0.))},
        tris={1: (1, 2, 3), 2: (2, 1, 4)},
        elements_of_face={1: [1], 2: [2]},
    )


def test_both_orientations_share_exact_station_and_local_cache_updates():
    mesh = _mesh()
    cache = {}
    before = {n: p.tobytes() for n, p in mesh.nodes.items()}
    assert propagate_triangle_split(mesh, (2, 1), (1, 2), 5, cache=cache) == 2
    assert len(mesh.tris) == 4
    assert (1, 2) not in cache[1] and (1, 2) not in cache[2]
    for face_id in (1, 2):
        assert len(cache[face_id][(1, 5)]) == len(cache[face_id][(2, 5)]) == 1
    mesh.nodes[6] = np.array((.25, 0., 0.))
    assert propagate_triangle_split(mesh, (1, 2), (1, 5), 6, cache=cache) == 2
    assert len(mesh.tris) == 6
    assert all(mesh.nodes[n].tobytes() == raw for n, raw in before.items())


def test_invalid_later_neighbour_does_not_partially_commit_elements():
    mesh = _mesh()
    mesh.tris[2] = (2, 3, 4)
    triangles = dict(mesh.tris)
    faces = {key: list(value) for key, value in mesh.elements_of_face.items()}
    with pytest.raises(MeshError, match="exactly one"):
        propagate_triangle_split(mesh, (1, 2), (1, 2), 5, cache={})
    assert mesh.tris == triangles and mesh.elements_of_face == faces


@pytest.mark.parametrize("reverse", (False, True))
def test_repeated_split_station_follows_canonical_endpoint_identity(reverse):
    points = np.array(((0., 0.), (1., 0.), (0., 1.), (1., 1.)))
    topology = MutableT3Topology(
        points, ((0, 1, 2), (1, 3, 2)),
        splittable_edges={(0, 1): (7, 1 if reverse else 0, 0 if reverse else 1)},
        seed_registry=ComponentSeedRegistry(100),
    )
    midpoint = topology.split_segment((0, 1))
    quarter = topology.split_segment((1, midpoint["point_id"]))
    assert midpoint["station"] == (1, 2)
    assert quarter["station"] == ((1, 4) if reverse else (3, 4))
    exported, _ = topology.canonical_export()
    np.testing.assert_array_equal(exported[quarter["point_id"]], (.75, 0.))
