"""A scalar beam offset stands off on the face-normal side on every meshing route.

The nodal normal used to average raw element normals, whose sign follows the face
loop on the mapped route, while nodes without quads fell back to the face normal.
A clockwise loop on an explicit +z plane then put a stiffener on +z when the face was
meshed with triangles near the line and on -z when it was meshed as mapped quads.
"""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.surfaces import Plane
from anymesher.hybrid import generate_hybrid_mesh

RECTANGLE_CW = [(0, 0, 0), (0, 1, 0), (0.4, 1, 0), (0.4, 0, 0)]
RECTANGLE_CCW = list(reversed(RECTANGLE_CW))
L_CW = [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0.6, 0), (0.6, 0.6, 0), (0.6, 0, 0)]
L_CCW = [(0, 0, 0), (0.6, 0, 0), (0.6, 0.6, 0), (1, 0.6, 0), (1, 1, 0), (0, 1, 0)]


@pytest.mark.parametrize("points", [RECTANGLE_CW, RECTANGLE_CCW, L_CW, L_CCW],
                         ids=["rectangle-cw", "rectangle-ccw", "L-cw", "L-ccw"])
@pytest.mark.parametrize("size", [0.2, 0.1])
@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_a_scalar_offset_follows_the_face_normal(points, size, order):
    model = GeometryModel()
    vertices = model.add_points(points)
    edges = [model.add_line(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
    face = model.add_face(edges, surface=Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    assert np.allclose(model.face_normal(face, 0.5, 0.5), (0, 0, 1))
    (beam,) = [e for e in edges if all(abs(model.vertex_position(v)[0]) < 1e-12
                                       for v in (model.edges[e].start, model.edges[e].end))]
    model.add_member([beam])
    mesh = generate_hybrid_mesh(model, strategy="auto", target_size=size, order=order, beam_edges=[beam],
                                beam_offsets={beam: 0.05})
    stations = list(zip(mesh.nodes_of_edge[beam], mesh.offset_nodes_of_edge[beam]))
    assert stations
    for base, stood in stations:
        assert np.allclose(np.asarray(mesh.nodes[stood]) - mesh.nodes[base], (0, 0, 0.05), rtol=0, atol=1e-12)
