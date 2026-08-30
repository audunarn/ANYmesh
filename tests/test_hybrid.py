from __future__ import annotations

import numpy as np

from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Plane

from anymesher import available_backends, generate_hybrid_mesh
from anymesher.hybrid import _stable_diagnostic_record
from anymesher.mapped import generate_mesh as generate_mapped_mesh


def _pentagon() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.5, 0.8, 0.0),
         (0.7, 1.4, 0.0), (-0.2, 0.8, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


def _mixed_faces() -> tuple[GeometryModel, int, int, int]:
    geometry = GeometryModel()
    v0, v1, v2, v3, v4 = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
         (0.0, 1.0, 0.0), (1.8, 0.5, 0.0))
    )
    e0 = geometry.add_line(v0, v1)
    shared = geometry.add_line(v1, v2)
    e2 = geometry.add_line(v2, v3)
    e3 = geometry.add_line(v3, v0)
    e4 = geometry.add_line(v1, v4)
    e5 = geometry.add_line(v4, v2)
    mapped = geometry.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in (e0, shared, e2, e3)),
        corners=(0, 1, 2, 3),
    )
    native = geometry.add_face_from_loop(
        (OrientedEdge(shared, False), OrientedEdge(e4, True), OrientedEdge(e5, True)),
        surface=Plane(
            np.array((1.0, 0.0, 0.0)),
            np.array((0.8, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
        ),
    )
    return geometry, mapped, native, shared


def _face_nodes(mesh, face_id: int) -> set[int]:
    result: set[int] = set()
    for element_id in mesh.elements_of_face[face_id]:
        connectivity = mesh.quads.get(element_id, mesh.tris.get(element_id))
        assert connectivity is not None
        result.update(connectivity)
    return result


def test_native_polygon_is_model_bound_and_deterministic() -> None:
    geometry, face_id = _pentagon()
    first = generate_hybrid_mesh(
        geometry, target_size=0.35, strategy="native", native_backend="python"
    )
    second = generate_hybrid_mesh(
        geometry, target_size=0.35, strategy="native", native_backend="python"
    )

    assert first.geometry_model_id == geometry.model_id
    assert first.geometry_revision == geometry.revision
    assert first.elements_of_face[face_id]
    assert first.quads == second.quads
    assert first.tris == second.tris
    assert tuple(first.nodes) == tuple(second.nodes)
    for node_id in first.nodes:
        np.testing.assert_allclose(first.nodes[node_id], second.nodes[node_id])


def test_persisted_hybrid_diagnostics_exclude_all_runtime_samples() -> None:
    diagnostics = {
        "phase_seconds": {"planning": 0.25},
        "native_diagnostics": {
            "insertion_seconds": 0.1,
            "segment_recovery_seconds": 0.2,
            "domain_filter_seconds": 0.3,
            "inserted_points": 12,
        },
        "selected_backend": "native",
    }

    assert _stable_diagnostic_record(diagnostics) == {
        "native_diagnostics": {"inserted_points": 12},
        "selected_backend": "native",
    }


def test_auto_mixed_faces_share_registry_identity() -> None:
    geometry, mapped_face, native_face, shared_edge = _mixed_faces()
    mesh = generate_hybrid_mesh(
        geometry, target_size=0.28, strategy="auto", native_backend="python"
    )

    assert mesh.hybrid_diagnostics["strategy_by_face"] == {
        mapped_face: "mapped",
        native_face: "native",
    }
    shared = set(mesh.nodes_of_edge[shared_edge])
    assert shared <= _face_nodes(mesh, mapped_face)
    assert shared <= _face_nodes(mesh, native_face)


def test_quadratic_native_boundary_uses_six_or_eight_node_shells() -> None:
    geometry, face_id = _pentagon()
    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.42,
        strategy="native",
        order="quadratic",
        native_backend="python",
    )

    connectivity = [
        mesh.quads.get(element_id, mesh.tris.get(element_id))
        for element_id in mesh.elements_of_face[face_id]
    ]
    assert connectivity
    assert all(len(item) in (6, 8) for item in connectivity if item is not None)
    assert all(len(mesh.nodes_of_edge[edge.edge]) % 2 == 1 for edge in geometry.faces[face_id].loop)


def test_explicit_mapped_strategy_preserves_mapped_connectivity() -> None:
    geometry = GeometryModel()
    face_id = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
             (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    expected = generate_mapped_mesh(geometry, target_size=0.4)
    actual = generate_hybrid_mesh(
        geometry, target_size=0.4, strategy="mapped", native_backend="python"
    )

    assert actual.quads == expected.quads
    assert actual.tris == expected.tris
    assert actual.elements_of_face[face_id] == expected.elements_of_face[face_id]
    assert tuple(actual.nodes) == tuple(expected.nodes)
    for node_id in expected.nodes:
        np.testing.assert_allclose(actual.nodes[node_id], expected.nodes[node_id])


def test_backend_registry_exposes_production_and_compatibility_paths() -> None:
    assert {"auto", "native", "mapped"} <= set(available_backends())
