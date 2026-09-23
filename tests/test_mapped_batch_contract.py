"""Scalar numbering/coordinate oracle for the bounded mapped fast path."""
from types import SimpleNamespace

import numpy as np
import pytest
from anygeometry import OrientedEdge

from anymesher import mapped
from anymesher.mesh import Mesh
from anymesher.serialize import mesh_to_dict


@pytest.mark.parametrize("nu,nv", ((1, 1), (1, 7), (9, 1), (3, 5), (47, 53)))
def test_mapped_linear_batches_preserve_scalar_grid_and_ids(nu, nv):
    mesh = Mesh()
    nodes, elements = mapped._Counter(), mapped._Counter()
    # Nonzero starting counters catch accidental per-face numbering resets.
    nodes.take(17)
    elements.take(23)
    corners = np.array(((0., 0., 0.), (2., .3, 0.),
                        (2.2, 1.5, .2), (-.1, 1.1, -.05)))
    vertex_ids = [nodes.next() for _ in corners]
    mesh.nodes.update(zip(vertex_ids, corners.copy()))
    sequences = []
    for edge, count in enumerate((nu, nv, nu, nv)):
        a, b = corners[edge], corners[(edge + 1) % 4]
        ids = [vertex_ids[edge]]
        for index in range(1, count):
            identifier = nodes.next()
            mesh.nodes[identifier] = a + (b - a) * (index / count)
            ids.append(identifier)
        ids.append(vertex_ids[(edge + 1) % 4])
        mesh.nodes_of_edge[edge + 1] = ids
        sequences.append(ids)
    protected = {n: p.tobytes() for n, p in mesh.nodes.items()}
    side_a, side_b = sequences[:2]
    side_c, side_d = sequences[2][::-1], sequences[3][::-1]
    expected_grid = np.zeros((nu + 1, nv + 1), dtype=int)
    expected_grid[:, 0], expected_grid[:, -1] = side_a, side_c
    expected_grid[0, :], expected_grid[-1, :] = side_d, side_b
    blended = mapped.coons_grid(*[
        np.array([mesh.nodes[n] for n in side])
        for side in (side_a, side_b, side_c, side_d)
    ])
    next_node, next_element = nodes._value, elements._value
    expected_nodes = dict(mesh.nodes)
    for i in range(1, nu):
        for j in range(1, nv):
            next_node += 1
            expected_grid[i, j] = next_node
            expected_nodes[next_node] = blended[i, j]
    expected_quads = {}
    for i in range(nu):
        for j in range(nv):
            next_element += 1
            expected_quads[next_element] = tuple(map(int, (
                expected_grid[i, j], expected_grid[i + 1, j],
                expected_grid[i + 1, j + 1], expected_grid[i, j + 1],
            )))
    face = SimpleNamespace(sides=lambda: tuple(
        (OrientedEdge(edge, True),) for edge in range(1, 5)))
    mapped._build_face(SimpleNamespace(faces={7: face}), mesh, 7, nodes, elements)
    assert nodes._value == next_node and elements._value == next_element
    np.testing.assert_array_equal(mesh.grid_of_face[7], expected_grid)
    assert mesh.quads == expected_quads
    assert mesh.elements_of_face[7] == list(expected_quads)
    assert list(mesh.nodes) == list(expected_nodes)
    for identifier, point in mesh.nodes.items():
        assert point.tobytes() == expected_nodes[identifier].tobytes()
    assert all(mesh.nodes[n].tobytes() == value for n, value in protected.items())


@pytest.mark.parametrize("position", (
    np.array((0., -0., 1. / 3.), dtype=np.float64),
    np.array((0., -0., 1. / 3.), dtype=np.float32),
    np.array((1, 2, 3), dtype=np.int64),
    (0., -0., 1. / 3.), [1, 2, 3],
))
def test_serialized_node_fast_path_preserves_float_conversion(position):
    mesh = Mesh()
    mesh.nodes[7] = position
    expected = [float(value) for value in position]
    result = mesh_to_dict(mesh)["nodes"]["7"]
    assert [v.hex() for v in result] == [v.hex() for v in expected]
    assert all(type(v) is float for v in result)


def test_serialized_array_subclass_retains_reference_iteration():
    class CustomArray(np.ndarray):
        def tolist(self):
            raise AssertionError("subclass conversion must not be substituted")
    mesh = Mesh()
    mesh.nodes[1] = np.array((1., 2., 3.)).view(CustomArray)
    assert mesh_to_dict(mesh)["nodes"]["1"] == [1., 2., 3.]
