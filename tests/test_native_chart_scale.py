"""Native planar meshing depends on the plate, not on the other faces in the model.

A planar face is meshed in its physical chart: the target size is applied in metres.
Previously a face was meshed in raw parameter units when it was the only face of its
model, so a plate made from points (edge vectors equal to its sides) was sized wrongly
unless its sides were one metre.  These tests pin that the same plate gives the same
mesh alone and inside a larger model, and that unit plates are unaffected.
"""

from __future__ import annotations

import numpy as np
import pytest
from anygeometry import GeometryModel

from anymesher import generate_hybrid_mesh_result


def _plate(model: GeometryModel, x0: float, width: float, height: float) -> int:
    vertices = model.add_points(
        ((x0, 0.0, 0.0), (x0 + width, 0.0, 0.0), (x0 + width, height, 0.0), (x0, height, 0.0))
    )
    face = model.add_plate(vertices)
    model.add_sheet((face,))
    return face


def _face_signature(mesh, face_id: int):
    """Element count and sorted element-corner coordinates of one face."""

    elements = mesh.elements_of_face[face_id]
    corners = []
    for element in elements:
        nodes = mesh.quads.get(element) or mesh.tris.get(element)
        corners.append(tuple(sorted(
            tuple(np.round(mesh.nodes[node], 9)) for node in nodes
        )))
    return len(elements), sorted(corners)


@pytest.mark.parametrize("width, height", [(2.0, 1.0), (3.0, 0.5)])
def test_plate_alone_meshes_as_the_same_plate_inside_a_larger_model(width, height):
    alone = GeometryModel()
    face_alone = _plate(alone, 0.0, width, height)
    single = generate_hybrid_mesh_result(alone, target_size=0.25, strategy="native").mesh

    inside = GeometryModel()
    face_inside = _plate(inside, 0.0, width, height)
    _plate(inside, 30.0, width, height)
    whole = generate_hybrid_mesh_result(inside, target_size=0.25, strategy="native").mesh

    assert _face_signature(single, face_alone) == _face_signature(whole, face_inside)


def test_unit_plate_is_unchanged_by_the_model_it_sits_in():
    alone = GeometryModel()
    face_alone = _plate(alone, 0.0, 1.0, 1.0)
    single = generate_hybrid_mesh_result(alone, target_size=0.25, strategy="native").mesh

    inside = GeometryModel()
    face_inside = _plate(inside, 0.0, 1.0, 1.0)
    _plate(inside, 30.0, 1.0, 1.0)
    whole = generate_hybrid_mesh_result(inside, target_size=0.25, strategy="native").mesh

    assert _face_signature(single, face_alone) == _face_signature(whole, face_inside)


def test_the_face_count_flag_is_gone():
    import anymesher.hybrid as hybrid

    assert not hasattr(hybrid, "_WHOLE_MODEL_HAS_SEVERAL_FACES")
