from __future__ import annotations

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.structural import Orientation

from anymesher import (
    Mesh,
    MeshError,
    S3OwnerAuthorityError,
    S3RepairError,
    generate_hybrid_mesh_result,
    prepare_qualified_s3_mesh,
)
from anymesher.automation import MeshAutomationSession, MeshCommand, MeshCommandBatch
from anymesher.serialize import mesh_from_dict, mesh_to_dict


def _square(
    *,
    owner: bool = True,
    orientation: Orientation = Orientation.FORWARD,
) -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    face_id = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    if owner:
        geometry.add_sheet((face_id,), orientations=(orientation,))
    return geometry, face_id


def _native(
    geometry: GeometryModel,
    **options: object,
):
    return generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        strategy="native",
        native_backend="python",
        recombine=False,
        structural_preparation=False,
        **options,
    )


def test_production_preparation_binds_deterministic_owner_and_nodal_normals() -> None:
    geometry, _face_id = _square()

    first = _native(geometry, qualified_s3=True)
    second = _native(geometry, qualified_s3=True)
    first_record = first.mesh.structural_preparation["qualified_s3"]
    second_record = second.mesh.structural_preparation["qualified_s3"]

    assert first_record == second_record
    assert first_record["status"] == "ADMITTED"
    assert first_record["formulation_id"] == "E4_PL_QUALIFIED_S3_COMPANION_V1"
    assert first_record["legacy_fallback"] == "FORBIDDEN"
    assert first_record["element_ids"] == sorted(first.mesh.tris)
    assert set(map(int, first_record["element_owner_normals"])) == set(
        first.mesh.shells
    )
    assert set(map(int, first_record["element_owner_sources"])) == set(
        first.mesh.shells
    )
    shell_nodes = {
        int(node_id)
        for connectivity in first.mesh.shells.values()
        for node_id in connectivity
    }
    assert set(map(int, first_record["nodal_normals"])) == shell_nodes
    for value in first_record["element_owner_normals"].values():
        np.testing.assert_allclose(value, (0.0, 0.0, 1.0))
    for value in first_record["nodal_normals"].values():
        np.testing.assert_allclose(value, (0.0, 0.0, 1.0))
    restored = mesh_from_dict(mesh_to_dict(first.mesh))
    assert restored.structural_preparation["qualified_s3"] == first_record


def test_crossing_sheet_junction_is_declared_before_qualified_s3_admission() -> None:
    geometry = GeometryModel()
    horizontal = geometry.add_plate(
        geometry.add_points(
            (
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
            )
        )
    )
    vertical = geometry.add_plate(
        geometry.add_points(
            (
                (-1.0, 0.0, -1.0),
                (1.0, 0.0, -1.0),
                (1.0, 0.0, 1.0),
                (-1.0, 0.0, 1.0),
            )
        )
    )
    geometry.add_sheet((horizontal,), name="horizontal")
    geometry.add_sheet((vertical,), name="vertical")

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        strategy="native",
        native_backend="python",
        recombine=False,
        qualified_s3=True,
    )
    mesh = result.mesh
    record = mesh.structural_preparation["qualified_s3"]
    incidence: dict[tuple[int, int], list[int]] = {}
    for element_id in sorted(mesh.shells):
        corners = mesh.corners_of(element_id)
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge = (min(first, second), max(first, second))
            incidence.setdefault(edge, []).append(element_id)

    assert mesh.declared_plate_junction_edges
    assert all(
        len(incidence[edge]) == 4
        for edge in mesh.declared_plate_junction_edges
    )
    assert record["status"] == "ADMITTED"
    assert record["admission"]["topology_violations"] == []
    assert record["repair"]["attempts"] == [
        {
            "action": "adjudication",
            "detail": (
                "all selected T3 elements satisfy the qualified-S3 "
                "admission contract"
            ),
            "edge": [],
            "element_ids": sorted(mesh.tris),
            "sequence": 1,
            "status": "accepted",
        }
    ]


def test_reversed_sheet_orientation_repairs_winding_to_physical_director() -> None:
    geometry, _face_id = _square(orientation=Orientation.REVERSED)

    result = _native(geometry, qualified_s3=True)
    record = result.mesh.structural_preparation["qualified_s3"]

    assert record["repair"]["winding_repairs"] == len(result.mesh.tris)
    for element_id, connectivity in result.mesh.tris.items():
        corners = np.asarray(
            [result.mesh.nodes[node_id] for node_id in connectivity[:3]], dtype=float
        )
        cross = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        owner = np.asarray(record["element_owner_normals"][str(element_id)])
        assert float(np.dot(cross, owner)) > 0.0
        np.testing.assert_allclose(owner, (0.0, 0.0, -1.0))


def test_mixed_q4_s3_normal_construction_uses_both_physical_owners() -> None:
    geometry = GeometryModel()
    point_ids = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.5, 0.0),
        )
    )
    quad_face = geometry.add_plate(point_ids[:4])
    tri_face = geometry.add_plate((point_ids[1], point_ids[4], point_ids[2]))
    geometry.add_sheet((quad_face,))
    geometry.add_sheet((tri_face,))
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={
            node_id: np.asarray(geometry.vertex_position(node_id), dtype=float)
            for node_id in point_ids
        },
        quads={10: tuple(point_ids[:4])},
        tris={20: (point_ids[1], point_ids[4], point_ids[2])},
        elements_of_face={quad_face: [10], tri_face: [20]},
    )

    prepared, record = prepare_qualified_s3_mesh(mesh, geometry)

    assert prepared.quads == mesh.quads
    assert prepared.tris == mesh.tris
    assert set(record["element_owner_normals"]) == {"10", "20"}
    assert set(map(int, record["nodal_normals"])) == set(point_ids)
    for value in record["nodal_normals"].values():
        np.testing.assert_allclose(value, (0.0, 0.0, 1.0))


def test_opt_out_preserves_legacy_generation_and_opt_in_fails_without_owner() -> None:
    geometry, _face_id = _square(owner=False)

    legacy = _native(geometry, qualified_s3=False)

    assert legacy.mesh.tris
    assert "qualified_s3" not in legacy.mesh.structural_preparation
    with pytest.raises(S3OwnerAuthorityError, match="lacks authoritative"):
        _native(geometry, qualified_s3=True)


def test_qualified_control_is_noop_authority_for_an_all_q4_mesh() -> None:
    geometry, _face_id = _square()

    result = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        strategy="mapped",
        structural_preparation=False,
        qualified_s3=True,
    )
    record = result.mesh.structural_preparation["qualified_s3"]

    assert result.mesh.quads
    assert not result.mesh.tris
    assert record["status"] == "NOT_APPLICABLE_NO_TRIANGLES"
    assert record["element_ids"] == []
    assert record["legacy_fallback"] == "FORBIDDEN"


def test_t6_is_a_typed_rejection_and_never_a_legacy_fallback() -> None:
    geometry, _face_id = _square()

    with pytest.raises(S3RepairError, match="requires exact T3") as caught:
        _native(geometry, qualified_s3=True, order="quadratic")

    assert caught.value.attempts[0].action == "validate"
    assert caught.value.attempts[0].status == "rejected"


def test_conflicting_sheet_owners_are_a_typed_rejection() -> None:
    geometry, face_id = _square()
    geometry.add_sheet((face_id,), orientations=(Orientation.REVERSED,))
    point_ids = tuple(sorted(geometry.vertices))
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={
            node_id: np.asarray(geometry.vertex_position(node_id), dtype=float)
            for node_id in point_ids
        },
        tris={1: (point_ids[0], point_ids[1], point_ids[2])},
        elements_of_face={face_id: [1]},
    )

    with pytest.raises(S3OwnerAuthorityError, match="conflicting"):
        prepare_qualified_s3_mesh(mesh, geometry)


def test_automation_propagates_explicit_qualified_s3_control() -> None:
    geometry, _face_id = _square()
    session = MeshAutomationSession(geometry)
    batch = MeshCommandBatch(
        1,
        "qualified-s3",
        session.session_id,
        geometry.model_id,
        geometry.revision,
        session.state_revision,
        (
            MeshCommand(
                "settings",
                "configure",
                {
                    "target_size": {"value": 0.5, "unit": "m"},
                    "strategy": "native",
                    "native_backend": "python",
                    "recombine": False,
                    "structural_preparation": False,
                    "qualified_s3": True,
                },
            ),
            MeshCommand("generate", "generate", {}),
        ),
    )

    plan = session.plan(batch)
    session.apply(plan)
    snapshot = session.mesh_snapshot()

    assert snapshot is not None
    assert snapshot.structural_preparation["qualified_s3"]["status"] == "ADMITTED"


def test_qualified_s3_controls_reject_non_boolean_values() -> None:
    geometry, _face_id = _square()

    with pytest.raises(MeshError, match="qualified_s3 must be Boolean"):
        _native(geometry, qualified_s3="yes")
