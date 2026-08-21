"""Public ANYgeometry intersection mutation boundary and legacy quarantine."""

from __future__ import annotations

import pytest

from anygeometry import (
    ConnectionIntent,
    EntityHandle,
    GeometryModel,
    ImprintOperation,
    IntersectionDimension,
    IntersectionKind,
)
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


def test_face_connect_persists_sheet_face_use_and_coedge_ownership() -> None:
    geometry = GeometryModel()
    supporting = _plate(
        geometry,
        ((-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)),
    )
    terminating = _plate(
        geometry,
        ((-1, 0, 0), (1, 0, 0), (1, 0, 1), (-1, 0, 1)),
    )
    supporting_sheet = geometry.add_sheet((supporting,))
    terminating_sheet = geometry.add_sheet((terminating,))

    application = apply_intersection_mutation(
        geometry,
        geometry.handle("face", terminating),
        geometry.handle("face", supporting),
        intent=ConnectionIntent.CONNECT,
    )

    assert application.plan.operation is ImprintOperation.FACE_IMPRINT
    assert application.result.kind is IntersectionKind.CROSS
    assert application.result.dimension is IntersectionDimension.CURVE
    assert application.face_intersection is not None
    shared_edge = application.face_intersection.edge.id
    face_use_ids = geometry.face_uses_using_edge(shared_edge)
    assert {
        geometry.face_uses[face_use_id].sheet_id
        for face_use_id in face_use_ids
    } == {supporting_sheet, terminating_sheet}
    assert {
        geometry.face_uses[geometry.coedges[coedge_id].face_use_id].sheet_id
        for coedge_id in geometry.coedges_using_edge(shared_edge)
    } == {supporting_sheet, terminating_sheet}
    assert geometry.validate_topology() == ()


def test_nonplanar_face_connect_reuses_exact_boundary_and_sheet_topology() -> None:
    geometry = GeometryModel()
    support = _plate(
        geometry,
        ((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)),
    )
    start, control, end = geometry.add_points(
        ((0.5, 0.5, 0.0), (1.5, 1.5, 0.0), (2.5, 0.5, 0.0))
    )
    spline = geometry.add_spline(start, (control,), end)
    wall = geometry.extrude((spline,), (0.0, 0.0, 1.0))[0]
    support_sheet = geometry.add_sheet((support,))
    wall_sheet = geometry.add_sheet((wall,))
    revision = geometry.revision

    plan = plan_intersection_mutation(
        geometry,
        geometry.handle("face", support),
        geometry.handle("face", wall),
        intent=ConnectionIntent.CONNECT,
    )

    assert plan.result.kind is IntersectionKind.CONTAINED
    assert plan.result.classified
    assert plan.result.dimension is IntersectionDimension.CURVE
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    assert geometry.revision == revision

    application = apply_intersection_mutation(
        geometry,
        geometry.handle("face", support),
        geometry.handle("face", wall),
        intent=ConnectionIntent.CONNECT,
    )

    assert application.face_intersection is not None
    assert application.face_intersection.edge.id == spline
    assert {
        geometry.face_uses[face_use_id].sheet_id
        for face_use_id in geometry.face_uses_using_edge(spline)
    } == {support_sheet, wall_sheet}
    assert geometry.validate_topology() == ()


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
