"""Planar native qualification corpus for non-mapped face topology."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import EntityRef, GeometryModel, Plane, punch_hole
from anymesher.core import MeshCore
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quality_v2 import evaluate_quality


def _polygon(points) -> tuple[GeometryModel, int]:
    coordinates = np.asarray(points, dtype=float)
    minimum = np.min(coordinates, axis=0)
    span = np.ptp(coordinates, axis=0)
    geometry = GeometryModel()
    vertices = geometry.add_points(points)
    face = geometry.add_face(
        geometry.add_polyline(vertices, close=True),
        surface=Plane(
            minimum,
            np.asarray((span[0], 0.0, 0.0)),
            np.asarray((0.0, span[1], 0.0)),
        ),
    )
    return geometry, face


def _core(mesh) -> MeshCore:
    node_ids = np.asarray(sorted(mesh.nodes), dtype=np.int64)
    coordinates = np.asarray([mesh.nodes[int(item)] for item in node_ids])
    triangle_ids = np.asarray(sorted(mesh.tris), dtype=np.int64)
    triangles = np.asarray(
        [mesh.tris[int(item)] for item in triangle_ids], dtype=np.int64
    ).reshape((-1, 3))
    quad_ids = np.asarray(sorted(mesh.quads), dtype=np.int64)
    quads = np.asarray(
        [mesh.quads[int(item)] for item in quad_ids], dtype=np.int64
    ).reshape((-1, 4))
    return MeshCore.from_id_connectivity(
        coordinates,
        node_ids=node_ids,
        triangles=triangles,
        triangle_ids=triangle_ids,
        quadrilaterals=quads,
        quad_ids=quad_ids,
    )


@pytest.mark.parametrize(
    "points,target_size",
    (
        (((0, 0, 0), (2, 0, 0), (0.3, 1.2, 0)), 0.2),
        (((0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 0.35, 0), (0, 1, 0)), 0.2),
        (((0, 0, 0), (4, 0, 0), (4, 0.08, 0), (0, 0.08, 0)), 0.04),
    ),
    ids=("triangle", "concave", "narrow"),
)
def test_native_planar_corpus_is_valid_and_keeps_every_boundary(points, target_size) -> None:
    geometry, face = _polygon(points)

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=target_size,
        strategy="native",
        face_ids=[face],
    )
    mesh = result.mesh
    quality = evaluate_quality(_core(mesh))

    assert result.strategy_by_face == {face: "native"}
    assert mesh.shells
    assert not quality.validity.errors
    assert np.all(quality.triangles.area > 0.0)
    assert np.all(quality.quadrilaterals.area > 0.0)
    assert np.all(quality.triangles.scaled_jacobian > 0.0)
    assert np.all(quality.quadrilaterals.scaled_jacobian > 0.0)
    for oriented in geometry.faces[face].loop:
        nodes = mesh.nodes_on(EntityRef("edge", oriented.edge))
        assert len(nodes) >= 2
        assert all(node in mesh.nodes for node in nodes)


def test_native_hole_is_empty_and_its_exact_curve_is_a_mesh_constraint() -> None:
    geometry, face = _polygon(
        ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))
    )
    face, hole_edges = punch_hole(geometry, face, (1.0, 1.0, 0.0), 0.35)

    mesh = generate_hybrid_mesh_result(
        geometry,
        target_size=0.2,
        strategy="native",
        face_ids=[face],
    ).mesh
    quality = evaluate_quality(_core(mesh))

    assert not quality.validity.errors
    assert mesh.tris
    assert mesh.quads
    centroids = np.asarray(
        [
            np.mean([mesh.nodes[node] for node in mesh.corners_of(element_id)], axis=0)
            for element_id in mesh.shells
        ]
    )
    assert np.all(
        np.linalg.norm(centroids[:, :2] - np.asarray((1.0, 1.0)), axis=1) > 0.35
    )
    for edge_id in hole_edges:
        positions = np.asarray(
            [mesh.nodes[node] for node in mesh.nodes_on(EntityRef("edge", edge_id))]
        )
        assert len(positions) >= 2
        assert np.linalg.norm(
            positions[:, :2] - np.asarray((1.0, 1.0)), axis=1
        ) == pytest.approx(0.35, abs=1.0e-10)
