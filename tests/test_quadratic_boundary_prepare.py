from fractions import Fraction
from types import SimpleNamespace
import pickle

import numpy as np
import pytest

from anymesher import Mesh
from anymesher._cylindrical_quadratic import QuadraticComponentStage
from anymesher._quadratic_boundary_prepare import synchronize_existing_midpoints
from anymesher.native_v2 import ComponentSeedRegistry
from anymesher.errors import MeshError
from anymesher._element_identity import next_unoccupied_shell_id


def fixture():
    points = [(0, 0), (1, 0), (0, 1), (.5, 0), (.5, .5), (0, .5),
              (0, -1), (0, -.5), (.5, -.5)]
    mesh = Mesh(nodes={i+1: np.array([*p, 0.]) for i,p in enumerate(points)},
                tris={20: (1, 2, 3, 4, 5, 6), 21: (2, 1, 7, 4, 8, 9)},
                elements_of_face={11: [20], 12: [21]}, order="quadratic",
                nodes_of_edge={1: [1, 4, 2], 2: [2, 5, 3], 3: [3, 6, 1],
                               4: [1, 8, 7], 5: [7, 9, 2]})
    entries = [SimpleNamespace(key=SimpleNamespace(edge_id=edge, parameter=t),
                               node_id=node, point=mesh.nodes[node])
               for edge, sequence in mesh.nodes_of_edge.items()
               for t,node in zip((0., .5, 1.), sequence)]
    stage = QuadraticComponentStage(mesh, (11, 12), entries, eligible_edges=(1,))
    return mesh, stage


def test_shared_midpoint_is_prepared_once_with_original_identity():
    source, stage = fixture()
    original = pickle.dumps(source)
    registry = ComponentSeedRegistry(10)
    report = synchronize_existing_midpoints(
        source, stage.mesh, stage, registry, (11, 12), (1,),
        lambda edge: (11, 12), lambda face, xyz: np.asarray(xyz)[:, :2], {}, 10,
        lambda phase: None,
    )
    assert report["new_node_count"] == 0 and report["reused_node_count"] == 1
    assert report["face_operations"] == {"11": 1, "12": 1}
    assert stage.mesh.nodes_of_edge[1] == [1, 4, 2]
    assert registry.resolve(1, 1, 2) == 4
    assert stage.station(1, 4) == Fraction(1, 2)
    assert all(len(stage.mesh.elements_of_face[face]) == 2 for face in (11, 12))
    assert pickle.dumps(source) == original


def test_preparation_budget_failure_does_not_change_mesh():
    source, stage = fixture()
    original = pickle.dumps(stage.mesh)
    with pytest.raises(MeshError, match="budget"):
        synchronize_existing_midpoints(
            source, stage.mesh, stage, ComponentSeedRegistry(10), (11, 12), (1,),
            lambda edge: (11, 12), lambda face, xyz: np.asarray(xyz)[:, :2], {}, 0,
            lambda phase: None,
        )
    assert pickle.dumps(stage.mesh) == original


def test_face_replacement_ids_do_not_collide_with_propagated_neighbors():
    source, stage = fixture()
    old_next = max(source.shells) + 1
    synchronize_existing_midpoints(
        source, stage.mesh, stage, ComponentSeedRegistry(10), (11, 12), (1,),
        lambda edge: (11, 12), lambda face, xyz: np.asarray(xyz)[:, :2], {}, 10,
        lambda phase: None,
    )
    neighbor = {element: stage.mesh.tris[element] for element in stage.mesh.elements_of_face[12]}
    rows = [stage.mesh.tris[element] for element in stage.mesh.elements_of_face[11]]
    # The old allocator demonstrably overlaps an ID allocated by propagation.
    assert set(range(old_next, old_next + len(rows))) & set(neighbor)
    next_id = next_unoccupied_shell_id(stage.mesh, old_next)
    for element in stage.mesh.elements_of_face[11]:
        del stage.mesh.tris[element]
    replacement = list(range(next_id, next_id + len(rows)))
    stage.mesh.elements_of_face[11] = replacement
    for element, row in zip(replacement, rows, strict=True):
        stage.mesh.tris[element] = row
    assert not set(replacement) & set(neighbor)
    assert neighbor == {element: stage.mesh.tris[element] for element in neighbor}
    assert next_unoccupied_shell_id(stage.mesh, 1000) == 1000
