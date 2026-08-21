"""Curved native meshing stays in surface charts and converges physically."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import CoonsSurface, Cone, Cylinder, GeometryModel, RuledSurface
from anymesher.hybrid import generate_hybrid_mesh_result


def _surface(name: str):
    if name == "cylinder":
        return Cylinder(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            3.0,
            0.0,
            np.pi / 2.0,
        )
    if name == "cone":
        return Cone(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            1.0,
            3.0,
            0.0,
            np.pi / 2.0,
        )
    first = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    second = np.asarray(((0.0, 1.0, 0.0), (2.0, 1.0, 0.5)))
    if name == "ruled":
        return RuledSurface(first, second)
    if name == "coons":
        p00, p10 = first
        p01, p11 = second
        return CoonsSurface(
            bottom=np.asarray((p00, p10)),
            right=np.asarray((p10, p11)),
            top=np.asarray((p01, p11)),
            left=np.asarray((p00, p01)),
        )
    raise AssertionError(name)


def _model_face(name: str) -> tuple[GeometryModel, int, object]:
    surface = _surface(name)
    geometry = GeometryModel()
    uv = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    points = [geometry.add_point(*surface.evaluate(u, v)) for u, v in uv]
    if name in {"cylinder", "cone"}:
        low_mid = geometry.add_point(*surface.evaluate(0.5, 0.0))
        high_mid = geometry.add_point(*surface.evaluate(0.5, 1.0))
        edges = [
            geometry.add_arc(points[0], low_mid, points[1]),
            geometry.add_line(points[1], points[2]),
            geometry.add_arc(points[2], high_mid, points[3]),
            geometry.add_line(points[3], points[0]),
        ]
    else:
        edges = [
            geometry.add_line(points[index], points[(index + 1) % 4])
            for index in range(4)
        ]
    face = geometry.add_face(edges, corners=(0, 1, 2, 3), surface=surface)
    return geometry, face, surface


def _support_residual(mesh, surface) -> float:
    return max(
        float(np.linalg.norm(surface.evaluate(*surface.local_uv(point)) - point))
        for point in mesh.nodes.values()
    )


def _maximum_corner_edge(mesh) -> float:
    maximum = 0.0
    for element_id in mesh.shells:
        corners = mesh.corners_of(element_id)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            maximum = max(
                maximum,
                float(np.linalg.norm(mesh.nodes[second] - mesh.nodes[first])),
            )
    return maximum


@pytest.mark.parametrize("name", ("cylinder", "cone", "ruled", "coons"))
def test_supported_curved_faces_mesh_natively_without_world_flattening(name: str) -> None:
    geometry, face, surface = _model_face(name)

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.4,
        strategy="native",
        face_ids=[face],
    )
    mesh = result.mesh

    assert result.strategy_by_face == {face: "native"}
    assert mesh.geometry_model_id == geometry.model_id
    assert mesh.geometry_revision == geometry.revision
    assert mesh.shells
    assert mesh.elements_of_face[face] == sorted(mesh.shells)
    assert all(node in mesh.nodes for element in mesh.shells.values() for node in element)
    assert _support_residual(mesh, surface) < 1.0e-10


def test_cylinder_refinement_increases_topology_and_reduces_physical_edges() -> None:
    counts = []
    maximum_edges = []
    for target_size in (0.8, 0.4, 0.2):
        geometry, face, surface = _model_face("cylinder")
        mesh = generate_hybrid_mesh_result(
            geometry,
            target_size=target_size,
            strategy="native",
            face_ids=[face],
        ).mesh
        counts.append(mesh.num_elements)
        maximum_edges.append(_maximum_corner_edge(mesh))
        assert _support_residual(mesh, surface) < 1.0e-10

    assert counts[0] < counts[1] < counts[2]
    assert maximum_edges[0] > maximum_edges[1] > maximum_edges[2]


def test_quadratic_cylinder_shares_q8_t6_midsides_on_the_exact_support() -> None:
    geometry, face, surface = _model_face("cylinder")
    mesh = generate_hybrid_mesh_result(
        geometry,
        target_size=0.6,
        strategy="native",
        face_ids=[face],
        order="quadratic",
    ).mesh

    assert mesh.quads
    assert mesh.tris
    assert all(len(connectivity) == 8 for connectivity in mesh.quads.values())
    assert all(len(connectivity) == 6 for connectivity in mesh.tris.values())

    edge_midpoints: dict[tuple[int, int], int] = {}
    for connectivity in (*mesh.quads.values(), *mesh.tris.values()):
        corner_count = 4 if len(connectivity) == 8 else 3
        for index in range(corner_count):
            edge = tuple(
                sorted(
                    (
                        connectivity[index],
                        connectivity[(index + 1) % corner_count],
                    )
                )
            )
            midpoint = connectivity[corner_count + index]
            assert edge_midpoints.setdefault(edge, midpoint) == midpoint

    assert _support_residual(mesh, surface) < 1.0e-10
