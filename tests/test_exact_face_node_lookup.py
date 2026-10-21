"""An exact shell-node shortcut must keep face ownership unambiguous."""

import numpy as np

from anymesher.mesh import Mesh
from anymesher.structural_pipeline import _FaceTargetIndex


def test_exact_face_node_lookup_keeps_full_element_and_refuses_ambiguity() -> None:
    mesh = Mesh()
    mesh.nodes.update({
        1: np.array((0., 0., 0.)), 2: np.array((1., 0., 0.)),
        3: np.array((1., 1., 0.)), 4: np.array((0., 1., 0.)),
        5: np.array((1., 0., 0.)), 6: np.array((2., 0., 0.)),
        7: np.array((2., 1., 0.)), 8: np.array((1., 1., 0.)),
    })
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.quads[20] = (5, 6, 7, 8)
    index = _FaceTargetIndex((10, 20), frozenset(range(1, 9)), 1e-9)
    assert index.exact_node(mesh, (0., 0., 0.)) == (1, (1, 2, 3, 4))
    assert index.exact_node(mesh, (2., 0., 0.)) == (6, (5, 6, 7, 8))
    assert index.exact_node(mesh, (1., 0., 0.)) is None
    assert index.exact_node(mesh, (1. + 1e-14, 0., 0.)) is None
