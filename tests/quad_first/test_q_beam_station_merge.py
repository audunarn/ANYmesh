"""A shared source edge owns shell and beam stations by ID, not coordinates alone."""

import numpy as np
import pytest

from anygeometry.model import GeometryModel

from anymesher.errors import MeshError
from anymesher.hybrid import (CertificationMode, HybridMeshResult,
                              _merge_quad_first_and_legacy)
from anymesher.mesh import Mesh


def _inputs(midpoint=(0.5, 0.0, 0.0)):
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
         (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    )
    face = geometry.add_plate(vertices)
    edge = geometry.faces[face].loop[0].edge
    shell = Mesh(
        nodes={0: np.array((0.0, 0.0, 0.0)),
               1: np.array((1.0, 0.0, 0.0)),
               2: np.array((1.0, 1.0, 0.0)),
               3: np.array((0.0, 1.0, 0.0)),
               4: np.array((0.5, 0.0, 0.0))},
        quads={0: (0, 1, 2, 3)},
        node_of_vertex=dict(zip(vertices, (0, 1, 2, 3))),
        nodes_of_edge={edge: [0, 4, 1]},
        elements_of_face={face: [0]},
    )
    beam = Mesh(
        nodes={10: np.array((0.0, 0.0, 0.0)),
               11: np.array(midpoint),
               12: np.array((1.0, 0.0, 0.0))},
        beams={0: (10, 11), 1: (11, 12)},
        node_of_vertex={vertices[0]: 10, vertices[1]: 12},
        nodes_of_edge={edge: [10, 11, 12]},
        elements_of_edge={edge: [0, 1]},
    )
    def result(mesh):
        return HybridMeshResult(
            mesh=mesh, strategy_by_face={}, triangulation_backend_by_face={},
            preflight=(), connectivity=None, audit_report=None,
            certification_mode=CertificationMode.NONE, certifiable=True,
        )
    return geometry, face, edge, result(shell), result(beam)


def test_shared_edge_station_maps_beam_to_shell_owner():
    geometry, face, edge, shell, beam = _inputs()
    merged = _merge_quad_first_and_legacy(
        geometry, quad_faces=frozenset((face,)), legacy_faces=frozenset(),
        quad_result=shell, legacy_result=beam,
    ).mesh
    assert merged.nodes_of_edge[edge] == [0, 4, 1]
    assert any(4 in body for body in merged.beams.values())
    assert len(merged.nodes) == len(shell.mesh.nodes)


def test_shared_edge_station_refuses_different_positions():
    geometry, face, _edge, shell, beam = _inputs((0.6, 0.0, 0.0))
    with pytest.raises(MeshError, match="mismatched shell/beam station"):
        _merge_quad_first_and_legacy(
            geometry, quad_faces=frozenset((face,)), legacy_faces=frozenset(),
            quad_result=shell, legacy_result=beam,
        )
