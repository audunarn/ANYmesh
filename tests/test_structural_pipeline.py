"""Focused contract tests for the structural/dynamic meshing slice."""

from __future__ import annotations

import numpy as np
import pytest
from anygeometry import GeometryModel
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    ParameterRange,
)
from anygeometry.transactions import ChangeSet

from anymesher.boundary import GlobalEdgeBoundaryRegistry, MemberRegistry
from anymesher.damage import ElementActivity
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.mesh_bvh import ElementType, MeshElementBVH, inverse_interpolate
from anymesher.meshing_view import (
    ChangeSetQueue,
    GeometryMeshingView,
    StaleMeshingViewError,
)
from anymesher.structural_pipeline import (
    ComponentGenerationCache,
    GeometryMutationPolicy,
    OverlapPolicy,
    PreflightStatus,
    StructuralMeshingPipeline,
)


def _plate(
    geometry: GeometryModel, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[int, tuple[int, ...]]:
    x, y, z = origin
    vertices = tuple(
        geometry.add_points(
            ((x, y, z), (x + 1.0, y, z), (x + 1.0, y + 1.0, z), (x, y + 1.0, z))
        )
    )
    return geometry.add_plate(vertices), vertices


def test_view_owns_sheet_member_indexes_and_rejects_stale_geometry() -> None:
    geometry = GeometryModel()
    face, _vertices = _plate(geometry)
    part = geometry.add_part()
    sheet = geometry.add_sheet((face,), part_id=part)
    edge = geometry.faces[face].loop[0].edge
    member = geometry.add_member((edge,), part_id=part)

    view = GeometryMeshingView(geometry)
    assert view.faces_for_sheet(sheet) == (face,)
    assert view.sheets_using_edge(edge) == (sheet,)
    assert view.members_using_edge(edge) == (member,)
    assert view.edge_uses_for_member(member)[0].edge_id == edge

    geometry.add_point(10.0, 0.0, 0.0)
    assert not view.is_current()
    with pytest.raises(StaleMeshingViewError):
        view.edge_point(edge, 0.5)


def test_change_queue_coalesces_and_component_cache_guards_stale_jobs() -> None:
    queue = ChangeSetQueue(revision=2)
    queue.push(ChangeSet(2, 3, added=(("face", 7),)))
    queue.push(ChangeSet(3, 4, modified=(("member", 2),)))
    change = queue.drain_coalesced()
    assert change is not None
    assert (change.revision_before, change.revision_after) == (2, 4)
    assert change.added == (("face", 7),)
    assert change.modified == (("member", 2),)

    cache: ComponentGenerationCache[str] = ComponentGenerationCache("model", 4)
    first = cache.begin("first")
    second = cache.begin("second")
    cache.invalidate(("first",))
    cache.advance_revision(5)
    assert not cache.publish(first, "stale")
    assert cache.publish(second, "still valid")
    assert cache.get("second") == "still valid"


def test_boundary_parameters_are_exact_and_member_registry_spans_edges() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(((0, 0, 0), (1, 0, 0), (3, 0, 0)))
    first = geometry.add_line(points[0], points[1])
    second = geometry.add_line(points[1], points[2])
    member = geometry.add_member((first, second))
    view = GeometryMeshingView(geometry)
    boundaries = GlobalEdgeBoundaryRegistry(view)
    left = 0.5
    right = float(np.nextafter(left, 1.0))
    boundaries.register(first, left)
    boundaries.register(first, right)

    assert boundaries.parameters(first) == (left, right)
    with pytest.raises(MeshError, match="inconsistent points"):
        boundaries.register(first, left, (0.75, 0.0, 0.0))

    members = MemberRegistry(view)
    location = members.locate(member, 0.5)
    assert location.span.edge_id == second
    assert location.edge_parameter == pytest.approx(0.25)
    assert len(members.spans(member)) == 2


def _interpolation_mesh() -> Mesh:
    mesh = Mesh()
    coordinates = {
        1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
        5: (2, 0, 0), 6: (3, 0, 0), 7: (3, 1, 0), 8: (2, 1, 0),
        9: (2.5, 0, 0), 10: (3, 0.5, 0), 11: (2.5, 1, 0), 12: (2, 0.5, 0),
        13: (4, 0, 0), 14: (5, 0, 0), 15: (4, 1, 0),
        16: (6, 0, 0), 17: (7, 0, 0), 18: (6, 1, 0),
        19: (6.5, 0, 0), 20: (6.5, 0.5, 0), 21: (6, 0.5, 0),
    }
    mesh.nodes.update({key: np.asarray(value, dtype=float) for key, value in coordinates.items()})
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.quads[20] = (5, 6, 7, 8, 9, 10, 11, 12)
    mesh.tris[30] = (13, 14, 15)
    mesh.tris[40] = (16, 17, 18, 19, 20, 21)
    return mesh


@pytest.mark.parametrize(
    ("point", "element_id", "kind"),
    (
        ((0.5, 0.5, 0.0), 10, ElementType.Q4),
        ((2.5, 0.5, 0.0), 20, ElementType.Q8),
        ((4.25, 0.25, 0.0), 30, ElementType.T3),
        ((6.25, 0.25, 0.0), 40, ElementType.T6),
    ),
)
def test_mesh_bvh_inverse_interpolates_all_shell_families(
    point: tuple[float, float, float], element_id: int, kind: ElementType
) -> None:
    mesh = _interpolation_mesh()
    bvh = MeshElementBVH(mesh, tolerance=1.0e-10)
    hit = bvh.locate(point)
    assert hit is not None
    assert (hit.element_id, hit.element_type) == (element_id, kind)
    assert sum(hit.weights) == pytest.approx(1.0)
    coordinates = np.asarray([mesh.nodes[node] for node in hit.node_ids])
    inverse = inverse_interpolate(kind, coordinates, point, tolerance=1.0e-10)
    assert inverse is not None and inverse.inside
    assert np.asarray(inverse.weights) @ coordinates == pytest.approx(point)


def test_pipeline_preflight_is_local_and_connectivity_is_declared_only() -> None:
    geometry = GeometryModel()
    face, plate_vertices = _plate(geometry)
    second_face, _ = _plate(geometry, (3.0, 0.0, 0.0))
    part = geometry.add_part()
    first_sheet = geometry.add_sheet((face,), part_id=part)
    second_sheet = geometry.add_sheet((second_face,), part_id=part)
    member_vertices = geometry.add_points(((0.5, 0.5, -1.0), (0.5, 0.5, 1.0)))
    member_edge = geometry.add_line(*member_vertices)
    member = geometry.add_member((member_edge,), part_id=part)
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )
    view = GeometryMeshingView(geometry)
    with pytest.raises(TypeError):
        StructuralMeshingPipeline(view)  # type: ignore[call-arg]

    mesh = Mesh()
    for node_id, vertex_id in enumerate(plate_vertices, start=1):
        mesh.nodes[node_id] = np.array(geometry.vertices[vertex_id].position, copy=True)
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.elements_of_face[face] = [10]
    mesh.nodes.update(
        {
            5: np.asarray((0.5, 0.5, -1.0)),
            6: np.asarray((0.5, 0.5, 0.0)),
            7: np.asarray((0.5, 0.5, 1.0)),
        }
    )
    mesh.nodes_of_edge[member_edge] = [5, 6, 7]
    mesh.beams[20] = (5, 6, 7)
    mesh.elements_of_edge[member_edge] = [20]

    pipeline = StructuralMeshingPipeline(
        view,
        overlap_policy=OverlapPolicy.CONNECT_DECLARED,
        mutation_policy=GeometryMutationPolicy.READ_ONLY,
    )
    states = pipeline.preflight(mesh)
    assert {state.status for state in states} == {
        PreflightStatus.READY,
        PreflightStatus.BLOCKED,
    }
    assert any(state.component.sheet_ids == (second_sheet,) for state in states)
    assert any(
        state.ready and state.component.sheet_ids == (first_sheet,)
        for state in states
    )

    report = pipeline.apply_connectivity(mesh)
    assert report.connected == 1
    coupling = next(iter(mesh.couplings.values()))
    assert coupling.beam_node == 6
    assert coupling.plate_nodes == (1, 2, 3, 4)
    assert coupling.weights == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert 5 not in {item.beam_node for item in mesh.couplings.values()}
    assert 7 not in {item.beam_node for item in mesh.couplings.values()}


def test_pipeline_scope_keeps_whole_active_component_only() -> None:
    geometry = GeometryModel()
    active_face, _ = _plate(geometry)
    unrelated_face, _ = _plate(geometry, (3.0, 0.0, 0.0))
    active_sheet = geometry.add_sheet((active_face,))
    unrelated_sheet = geometry.add_sheet((unrelated_face,))
    member_edge = geometry.add_line(
        *geometry.add_points(((0.5, 0.5, -1.0), (0.5, 0.5, 1.0)))
    )
    connected_member = geometry.add_member((member_edge,))
    geometry.add_attachment(
        connected_member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        active_face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )

    pipeline = StructuralMeshingPipeline(
        GeometryMeshingView(geometry),
        overlap_policy=OverlapPolicy.CONNECT_DECLARED,
        mutation_policy=GeometryMutationPolicy.READ_ONLY,
        active_sheet_ids=(active_sheet,),
        active_member_ids=(),
    )

    assert len(pipeline.components) == 1
    assert pipeline.components[0].sheet_ids == (active_sheet,)
    assert pipeline.components[0].member_ids == (connected_member,)
    assert unrelated_sheet not in pipeline.components[0].sheet_ids

    states = pipeline.preflight(Mesh())
    assert len(states) == 1
    assert {issue.code for issue in states[0].issues} == {
        "unmeshed-face",
        "unmeshed-member",
    }


def test_pipeline_empty_explicit_scope_selects_no_components() -> None:
    geometry = GeometryModel()
    face, _ = _plate(geometry)
    geometry.add_sheet((face,))

    pipeline = StructuralMeshingPipeline(
        GeometryMeshingView(geometry),
        overlap_policy=OverlapPolicy.CONNECT_DECLARED,
        mutation_policy=GeometryMutationPolicy.READ_ONLY,
        active_sheet_ids=(),
        active_member_ids=(),
    )

    assert pipeline.components == ()
    assert pipeline.preflight() == ()


def test_damage_updates_only_local_sides_and_reuses_bvh() -> None:
    mesh = Mesh()
    mesh.nodes.update(
        {
            1: np.asarray((0, 0, 0)), 2: np.asarray((1, 0, 0)),
            3: np.asarray((2, 0, 0)), 4: np.asarray((0, 1, 0)),
            5: np.asarray((1, 1, 0)), 6: np.asarray((2, 1, 0)),
        }
    )
    mesh.quads[1] = (1, 2, 5, 4)
    mesh.quads[2] = (2, 3, 6, 5)
    activity = ElementActivity(mesh)
    assert len(activity.exposed_boundaries) == 6
    assert (2, 5) not in activity.exposed_boundaries

    update = activity.deactivate((2,))
    assert update.deactivated == (2,)
    assert (2, 5) in {item.key for item in update.newly_exposed}
    assert (1, 2) not in update.affected_sides
    assert set(activity.exposed_boundaries) == {(1, 2), (2, 5), (4, 5), (1, 4)}

    bvh = MeshElementBVH(mesh, tolerance=1.0e-10)
    activity.sync_bvh(bvh)
    assert bvh.locate((1.5, 0.5, 0.0)) is None
    activity.activate((2,))
    activity.sync_bvh(bvh)
    assert bvh.locate((1.5, 0.5, 0.0)).element_id == 2  # type: ignore[union-attr]

