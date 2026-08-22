"""Bounded, headless contracts for global structured planning."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from anygeometry import (
    AttachmentKind,
    AttachmentTargetKind,
    ConnectivityPolicy,
    GeometryModel,
    OrientedEdge,
    ParameterRange,
    Plane,
    SheetTopologyPolicy,
)

from anymesher import MeshError
from anymesher.structured import (
    MeshQualityPolicy,
    StructuredMeshingOptions,
    apply_structured_layout,
    plan_structured_layout,
)


PLANE = Plane(
    np.asarray((0.0, 0.0, 0.0)),
    np.asarray((1.0, 0.0, 0.0)),
    np.asarray((0.0, 1.0, 0.0)),
)


def _polygon(
    geometry: GeometryModel,
    points: tuple[tuple[float, float, float], ...],
) -> int:
    vertices = geometry.add_points(points)
    edges = geometry.add_polyline(vertices, close=True)
    return geometry.add_face_from_loop(geometry.order_loop(edges), surface=PLANE)


def _pentagon(geometry: GeometryModel, offset: float = 0.0) -> int:
    return _polygon(
        geometry,
        (
            (offset + 0.0, 0.0, 0.0),
            (offset + 2.0, 0.0, 0.0),
            (offset + 2.5, 1.0, 0.0),
            (offset + 1.0, 2.0, 0.0),
            (offset - 0.5, 1.0, 0.0),
        ),
    )


def _adjacent_pentagons() -> tuple[GeometryModel, int, int, int]:
    geometry = GeometryModel()
    a, b, c, d, e, u, v, w = geometry.add_points(
        (
            (0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
            (2.5, 1.0, 0.0), (1.0, 2.0, 0.0), (-0.5, 1.0, 0.0),
            (-0.5, -1.0, 0.0), (1.0, -2.0, 0.0), (2.5, -1.0, 0.0),
        )
    )
    shared = geometry.add_line(a, b)
    top = (
        OrientedEdge(shared, True),
        *(OrientedEdge(edge, True) for edge in (
            geometry.add_line(b, c), geometry.add_line(c, d),
            geometry.add_line(d, e), geometry.add_line(e, a),
        )),
    )
    bottom = (
        OrientedEdge(shared, False),
        *(OrientedEdge(edge, True) for edge in (
            geometry.add_line(a, u), geometry.add_line(u, v),
            geometry.add_line(v, w), geometry.add_line(w, b),
        )),
    )
    first = geometry.add_face_from_loop(top, surface=PLANE)
    second = geometry.add_face_from_loop(bottom, surface=PLANE)
    return geometry, first, second, shared


def test_quality_defaults_match_qualified_023_surface_policy() -> None:
    assert MeshQualityPolicy() == MeshQualityPolicy(
        minimum_scaled_jacobian=0.20,
        maximum_aspect_ratio=4.0,
        minimum_angle=30.0,
        maximum_angle=150.0,
        maximum_warpage=0.10,
    )


def test_sheet_and_shared_edges_form_true_global_components() -> None:
    geometry, first, second, shared = _adjacent_pentagons()
    plan = plan_structured_layout(geometry, target_size=0.5)

    assert {item.component_id for item in plan.faces} == {1}
    assert [(item.first_face_id, item.second_face_id) for item in plan.interfaces] == [
        (first, second)
    ]
    assert plan.interfaces[0].edge_id == shared
    assert plan.interfaces[0].conforming


def test_disconnected_faces_in_one_declared_sheet_share_candidate_budget() -> None:
    geometry = GeometryModel()
    first = _pentagon(geometry, 0.0)
    second = _pentagon(geometry, 5.0)
    geometry.add_sheet(
        (first, second),
        policy=SheetTopologyPolicy(
            connectivity=ConnectivityPolicy.ALLOW_DISCONNECTED
        ),
    )
    plan = plan_structured_layout(
        geometry,
        target_size=0.5,
        options=StructuredMeshingOptions(maximum_candidates_per_component=1),
    )

    assert {item.component_id for item in plan.faces} == {1}
    assert sum(item.action == "radial_partition" for item in plan.faces) == 1
    assert sum(item.action == "native" for item in plan.faces) == 1
    assert len(plan.blocks) == 5


def test_adjacent_partition_plan_and_application_are_order_independent() -> None:
    geometry, first, second, shared = _adjacent_pentagons()
    forward = plan_structured_layout(
        geometry, target_size=0.5, face_ids=(first, second)
    )
    reverse = plan_structured_layout(
        geometry, target_size=0.5, face_ids=(second, first)
    )

    assert reverse == forward
    first_working, first_report = apply_structured_layout(geometry, forward)
    second_working, second_report = apply_structured_layout(geometry, reverse)
    assert first_report.to_dict() == second_report.to_dict()
    assert len(first_report.blocks) == 10
    assert len(first_report.source_to_working_edges[shared]) == 2
    assert first_working.validate_topology() == second_working.validate_topology() == ()
    cross_face = [
        item for item in first_report.interfaces if item.source_edge_id == shared
    ]
    assert len(cross_face) == 2


def test_concave_face_is_native_and_not_a_structured_block() -> None:
    geometry = GeometryModel()
    face = _polygon(
        geometry,
        (
            (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.8, 0.6, 0.0),
            (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
        ),
    )
    plan = plan_structured_layout(geometry, target_size=0.4)

    assert plan.faces[0].source_face_id == face
    assert plan.faces[0].action == "native"
    assert plan.faces[0].block_ids == ()
    assert plan.blocks == ()
    assert "convex planar" in plan.blockers[0]


def test_attachment_and_member_intent_block_partitioning() -> None:
    geometry = GeometryModel()
    face = _pentagon(geometry)
    part = geometry.add_part()
    geometry.add_sheet((face,), part_id=part)
    ends = geometry.add_points(((1.0, 1.0, -1.0), (1.0, 1.0, 1.0)))
    member = geometry.add_member((geometry.add_line(*ends),), part_id=part)
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )

    plan = plan_structured_layout(geometry, target_size=0.5)
    assert plan.faces[0].action == "native"
    assert "attachments" in plan.faces[0].reason
    assert plan.blocks == ()


def test_nonmanifold_shared_edge_is_a_public_blocker() -> None:
    geometry = GeometryModel()
    a, b = geometry.add_points(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    shared = geometry.add_line(a, b)
    faces = []
    for point in ((1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 0.0, 1.0)):
        third = geometry.add_point(*point)
        first = geometry.add_line(b, third)
        second = geometry.add_line(third, a)
        faces.append(geometry.add_face_from_loop(
            (OrientedEdge(shared, True), OrientedEdge(first, True), OrientedEdge(second, True))
        ))

    plan = plan_structured_layout(geometry, target_size=0.5)
    assert all(item.action == "native" for item in plan.faces)
    assert all("non-manifold" in item.reason for item in plan.faces)
    assert plan.blocks == ()


def test_tampered_stale_and_resource_exhausted_plans_fail_closed() -> None:
    geometry = GeometryModel()
    _pentagon(geometry)
    plan = plan_structured_layout(geometry, target_size=0.5)

    with pytest.raises(MeshError, match="tampered"):
        apply_structured_layout(geometry, replace(plan, target_size=0.6))
    geometry.add_point(20.0, 20.0, 0.0)
    with pytest.raises(MeshError, match="stale"):
        apply_structured_layout(geometry, plan)

    bounded = GeometryModel()
    _pentagon(bounded)
    with pytest.raises(MeshError, match="maximum_blocks"):
        plan_structured_layout(
            bounded,
            target_size=0.5,
            options=StructuredMeshingOptions(maximum_blocks=4),
        )


def test_explicit_seeding_blocks_partitions_and_overrides_survive_apply() -> None:
    polygon = GeometryModel()
    _pentagon(polygon)
    preserved = plan_structured_layout(
        polygon, target_size=0.5, explicit_seeding=True
    )
    assert preserved.preserves_explicit_seeding
    assert preserved.faces[0].action == "native"
    assert preserved.blocks == ()

    mapped = GeometryModel()
    face = mapped.add_plate(mapped.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
         (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    ))
    sides = mapped.faces[face].sides()
    pinned = sides[0][0].edge
    opposite = sides[2][0].edge
    plan = plan_structured_layout(
        mapped, target_size=0.5, overrides={pinned: 7}
    )
    _working, report = apply_structured_layout(mapped, plan)
    assert plan.seed_overrides == ((pinned, 7),)
    assert report.seed_solution[pinned] == 7
    assert report.seed_solution[opposite] == 7


def test_cancellation_is_bounded_and_propagates_without_source_edits() -> None:
    geometry = GeometryModel()
    _pentagon(geometry)
    revision = geometry.revision

    def cancel(phase: str) -> None:
        if phase == "structured face qualification":
            raise RuntimeError("cancelled structured planner")

    with pytest.raises(RuntimeError, match="cancelled"):
        plan_structured_layout(
            geometry, target_size=0.5, cancellation_check=cancel
        )
    assert geometry.revision == revision
    assert len(geometry.faces) == 1
