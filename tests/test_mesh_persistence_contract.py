"""Model-bound mesh persistence and deterministic legacy migration."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from anygeometry import EntityHandle
from anymesher import Mesh, MeshError
from anymesher.serialize import mesh_from_dict, mesh_to_dict


def _associated_mesh() -> Mesh:
    return Mesh(
        geometry_model_id=uuid4(),
        geometry_revision=7,
        nodes={
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, 1.0, 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
        },
        quads={20: (1, 2, 3, 4)},
        elements_of_sheet={11: [20]},
        elements_of_member={13: [20]},
        nodes_of_member={13: [1, 2]},
        activity={20: 0.25},
        block_grids_of_face={
            17: (
                np.asarray(((1, 2), (4, 3))),
                np.asarray(((2, 3), (1, 4))),
            )
        },
        structural_preparation={
            "schema": "anymesher.structural-preparation/1",
            "plan_hash": "sha256:abc",
            "source_to_working_faces": {"17": [31, 32]},
        },
        hybrid_diagnostics={"phase_seconds": {"planning": 0.25}},
    )


def test_v3_round_trip_preserves_model_identity_structure_and_activity() -> None:
    source = _associated_mesh()
    payload = mesh_to_dict(source)
    restored = mesh_from_dict(payload)

    assert payload["version"] == 3
    assert str(restored.geometry_model_id) == str(source.geometry_model_id)
    assert restored.geometry_revision == 7
    assert restored.elements_of_sheet == {11: [20]}
    assert restored.elements_of_member == {13: [20]}
    assert restored.nodes_of_member == {13: [1, 2]}
    assert restored.activity == {20: pytest.approx(0.25)}
    assert len(restored.block_grids_of_face[17]) == 2
    assert restored.nodes_on(
        EntityHandle(source.geometry_model_id, "face", 17)
    ) == [1, 2, 3, 4]
    assert restored.structural_preparation == source.structural_preparation
    assert restored.hybrid_diagnostics == source.hybrid_diagnostics

    sheet = EntityHandle(source.geometry_model_id, "sheet", 11)
    member = EntityHandle(source.geometry_model_id, "member", 13)
    assert restored.nodes_on(sheet) == [1, 2, 3, 4]
    assert restored.nodes_on(member) == [1, 2]


def test_mesh_associations_reject_a_handle_from_another_model() -> None:
    mesh = _associated_mesh()
    foreign = EntityHandle(uuid4(), "sheet", 11)

    with pytest.raises(MeshError, match="handle belongs to geometry model"):
        mesh.nodes_on(foreign)


@pytest.mark.parametrize("version", (1, 2))
def test_legacy_document_migrates_with_safe_empty_v3_fields(version: int) -> None:
    payload = mesh_to_dict(_associated_mesh())
    payload["version"] = version
    for name in (
        "geometry_model_id",
        "geometry_revision",
        "elements_of_sheet",
        "elements_of_member",
        "nodes_of_member",
        "activity",
        "block_grids_of_face",
        "structural_preparation",
        "hybrid_diagnostics",
    ):
        payload.pop(name)

    restored = mesh_from_dict(payload)

    assert restored.geometry_model_id is None
    assert restored.geometry_revision is None
    assert restored.elements_of_sheet == {}
    assert restored.elements_of_member == {}
    assert restored.nodes_of_member == {}
    assert restored.activity == {}
    assert restored.block_grids_of_face == {}
    assert restored.structural_preparation == {}
    assert restored.hybrid_diagnostics == {}


def test_future_mesh_format_fails_closed() -> None:
    payload = mesh_to_dict(_associated_mesh())
    payload["version"] = 4

    with pytest.raises(MeshError, match=r"reads 1-3"):
        mesh_from_dict(payload)


def test_mesh_audit_data_rejects_non_json_objects_and_nonfinite_values() -> None:
    source = _associated_mesh()
    source.structural_preparation = {"bad": object()}
    with pytest.raises(MeshError, match="must be JSON-safe"):
        mesh_to_dict(source)

    source.structural_preparation = {"bad": float("nan")}
    with pytest.raises(MeshError, match="non-finite"):
        mesh_to_dict(source)
