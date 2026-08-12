"""Public ANYgeometry intersection mutation boundary and legacy quarantine."""

from __future__ import annotations

import pytest

from anygeometry import ConnectionIntent, EntityHandle, GeometryModel, IntersectionDimension
from anymesher import intersections
from anymesher.intersections import (
    apply_intersection_mutation,
    plan_intersection_mutation,
)


def _plate(geometry: GeometryModel, points) -> int:
    return geometry.add_plate([geometry.add_point(*point) for point in points])


def test_crossing_faces_plan_and_apply_through_the_public_kernel_contract() -> None:
    geometry = GeometryModel()
    first = _plate(
        geometry,
        ((-1, 0, 0), (1, 0, 0), (1, 2, 0), (-1, 2, 0)),
    )
    second = _plate(
        geometry,
        ((0, 0, -1), (0, 2, -1), (0, 2, 1), (0, 0, 1)),
    )
    first_handle = EntityHandle(geometry.model_id, "face", first)
    second_handle = EntityHandle(geometry.model_id, "face", second)
    revision = geometry.revision

    plan = plan_intersection_mutation(
        geometry,
        first_handle,
        second_handle,
        intent=ConnectionIntent.IMPRINT,
    )

    assert plan.model_id == geometry.model_id
    assert plan.revision == revision
    assert plan.operation.value == "face_imprint"
    assert plan.result.dimension is IntersectionDimension.CURVE
    assert plan.expected_changes

    application = apply_intersection_mutation(
        geometry,
        first_handle,
        second_handle,
        intent=ConnectionIntent.IMPRINT,
    )

    assert application.plan.model_id == geometry.model_id
    assert application.plan.revision == revision
    assert application.result.dimension is IntersectionDimension.CURVE
    assert application.change_set is not None
    assert geometry.revision > revision


def test_deprecated_legacy_entry_warns_and_forwards_explicitly(monkeypatch) -> None:
    geometry = GeometryModel()
    sentinel = object()
    received = {}

    def fake_legacy(model, **options):
        received.update(model=model, options=options)
        return sentinel

    monkeypatch.setattr(intersections, "generate_mesh_with_legacy_imprint", fake_legacy)

    with pytest.warns(DeprecationWarning, match="quarantined legacy"):
        result = intersections.generate_mesh_with_intersections(
            geometry, target_size=0.25
        )

    assert result is sentinel
    assert received == {"model": geometry, "options": {"target_size": 0.25}}
