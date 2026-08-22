from __future__ import annotations

from anygeometry import GeometryModel, OrientedEdge
from anygeometry.serialization import to_dict
from anygeometry.surfaces import Plane
import numpy as np
import pytest

from anymesher import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.seeding import solve_seeding


def _plate(points):
    geometry = GeometryModel()
    face = geometry.add_plate(geometry.add_points(points))
    return geometry, face


def test_auto_structured_triangle_is_detached_mapped_and_source_bound() -> None:
    geometry, face = _plate(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 3.0**0.5, 0.0),
        )
    )
    before = to_dict(geometry)

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        strategy="auto",
        structured_options={},
        native_backend="python",
    )

    assert to_dict(geometry) == before
    assert result.strategy_by_face == {face: "mapped"}
    assert result.structured_layout is not None
    assert result.structured_layout.status == "applied"
    assert len(result.structured_layout.source_to_working_faces[face]) == 3
    assert result.mesh.quads and not result.mesh.tris
    assert face not in result.mesh.grid_of_face
    assert len(result.mesh.block_grids_of_face[face]) == 3
    assert result.mesh.elements_of_face[face] == sorted(result.mesh.quads)
    assert result.mesh.geometry_model_id == geometry.model_id
    assert result.mesh.geometry_revision == geometry.revision
    assert result.mesh.structural_preparation["plan"]["plan_hash"].startswith(
        "sha256:"
    )
    assert result.mesh.structural_preparation["quality"]["accepted"] is True
    assert result.mesh.hybrid_diagnostics["structured_quality"]["accepted"] is True
    assert result.triangulation_backend_by_face[face]["actual_backend"] == "mapped"
    assert result.triangulation_backend_by_face[face]["working_face_ids"]


def test_explicit_mapped_fails_closed_when_partition_is_not_qualified() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 0.4, 0.0),
         (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    edges = tuple(
        geometry.add_line(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    )
    geometry.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in edges),
        corners=None,
        surface=Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )

    with pytest.raises(MeshError, match="explicit mapped strategy"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.25,
            strategy="mapped",
            structured_options={},
            native_backend="python",
        )


def test_native_strategy_rejects_contradictory_structured_options() -> None:
    geometry, _face = _plate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )

    with pytest.raises(MeshError, match="cannot be combined"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.25,
            strategy="native",
            structured_options={},
        )


def test_caller_seeding_is_preserved_when_structured_preview_needs_no_partition() -> None:
    geometry, face = _plate(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    bottom = geometry.faces[face].loop[0].edge
    supplied = solve_seeding(
        geometry, target_size=0.25, overrides={bottom: 7}
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        strategy="auto",
        structured_options={},
        seeding=supplied,
        native_backend="python",
    )

    assert result.structured_layout is not None
    assert result.structured_layout.plan.requires_working_clone is False
    assert result.mesh.seeding is not None
    assert result.mesh.seeding.divisions == supplied.divisions
    assert result.mesh.grid_of_face[face].size > 0


def test_auto_uses_quality_gated_native_fallback() -> None:
    geometry, face = _plate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 1.0, 0.0))
    )

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.25,
        strategy="auto",
        structured_options={"max_element_growth": 1.27},
        native_backend="python",
    )

    assert result.strategy_by_face == {face: "native"}
    assert result.structured_layout is not None
    assert result.structured_layout.status == "rejected_fallback"
    quality = result.mesh.structural_preparation["quality"]
    assert quality["accepted"] is True
    assert quality["selected_mesh"] == "native_fallback"
    assert quality["rejected_candidate"]["growth_violation_count"] > 0
    assert quality["accepted_fallback"]["growth_violation_count"] == 0


def test_explicit_mapped_rejects_structured_quality_failure() -> None:
    geometry, _face = _plate(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.2, 1.0, 0.0))
    )

    with pytest.raises(MeshError, match="explicit mapped strategy rejected"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.25,
            strategy="mapped",
            structured_options={"max_element_growth": 1.27},
            native_backend="python",
        )
