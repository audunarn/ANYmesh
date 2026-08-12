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
    )


def test_v2_round_trip_preserves_model_identity_structure_and_activity() -> None:
    source = _associated_mesh()
    payload = mesh_to_dict(source)
    restored = mesh_from_dict(payload)

    assert payload["version"] == 2
    assert str(restored.geometry_model_id) == str(source.geometry_model_id)
    assert restored.geometry_revision == 7
    assert restored.elements_of_sheet == {11: [20]}
    assert restored.elements_of_member == {13: [20]}
    assert restored.nodes_of_member == {13: [1, 2]}
    assert restored.activity == {20: pytest.approx(0.25)}

    sheet = EntityHandle(source.geometry_model_id, "sheet", 11)
    member = EntityHandle(source.geometry_model_id, "member", 13)
    assert restored.nodes_on(sheet) == [1, 2, 3, 4]
    assert restored.nodes_on(member) == [1, 2]


def test_mesh_associations_reject_a_handle_from_another_model() -> None:
    mesh = _associated_mesh()
    foreign = EntityHandle(uuid4(), "sheet", 11)

    with pytest.raises(MeshError, match="handle belongs to geometry model"):
        mesh.nodes_on(foreign)


def test_v1_document_migrates_with_safe_empty_v2_fields() -> None:
    payload = mesh_to_dict(_associated_mesh())
    payload["version"] = 1
    for name in (
        "geometry_model_id",
        "geometry_revision",
        "elements_of_sheet",
        "elements_of_member",
        "nodes_of_member",
        "activity",
    ):
        payload.pop(name)

    restored = mesh_from_dict(payload)

    assert restored.geometry_model_id is None
    assert restored.geometry_revision is None
    assert restored.elements_of_sheet == {}
    assert restored.elements_of_member == {}
    assert restored.nodes_of_member == {}
    assert restored.activity == {}


def test_future_mesh_format_fails_closed() -> None:
    payload = mesh_to_dict(_associated_mesh())
    payload["version"] = 3

    with pytest.raises(MeshError, match=r"reads 1-2"):
        mesh_from_dict(payload)
