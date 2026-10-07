"""Focused regressions for candidate finalization, admission freshness, and
the invocation-local connectivity index.

The route tests use fault injection only for route selection; every
numerical expectation (weights, eccentricities, record freshness, counters)
is computed independently of the implementation under test.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
from anygeometry import GeometryModel
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
)
from anygeometry.serialization import to_dict

from anymesher.errors import MeshError, StructuredQualityRejected
from anymesher.mesh import Mesh
from anymesher.mesh_bvh import MeshElementBVH, normalized_element_filter
from anymesher.meshing_view import GeometryMeshingView
from anymesher.structural_pipeline import StructuralMeshingPipeline
from anymesher.serialize import mesh_from_dict, mesh_to_dict


hybrid = importlib.import_module("anymesher.hybrid")

POLICY = {
    "minimum_scaled_jacobian": 0.1,
    "maximum_aspect_ratio": 5.0,
    "minimum_angle": 20.0,
    "maximum_angle": 160.0,
    "maximum_warpage": 0.1,
}


def _plate_model() -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    geometry.add_sheet((face,))
    return geometry, face


def _auto_call(**extra):
    return dict(
        target_size=10.0,
        strategy="auto",
        native_backend="python",
        beam_edges=(),
        member_ids=(),
        structural_preparation=False,
        recombine=False,
        structured_options={"quality_policy": POLICY},
        **extra,
    )


def _count_connectivity(monkeypatch):
    calls = []
    real = StructuralMeshingPipeline.apply_connectivity

    def counted(self, mesh, *args, **kwargs):
        calls.append(mesh)
        return real(self, mesh, *args, **kwargs)

    monkeypatch.setattr(StructuralMeshingPipeline, "apply_connectivity", counted)
    return calls


# ---------------------------------------------------------------------------
# Public BVH filter coercion and the private trusted fast path
# ---------------------------------------------------------------------------


def _quad_mesh() -> Mesh:
    mesh = Mesh()
    mesh.nodes.update(
        {
            1: np.asarray((0.0, 0.0, 0.0)),
            2: np.asarray((1.0, 0.0, 0.0)),
            3: np.asarray((1.0, 1.0, 0.0)),
            4: np.asarray((0.0, 1.0, 0.0)),
            5: np.asarray((2.0, 0.0, 0.0)),
            6: np.asarray((2.0, 1.0, 0.0)),
        }
    )
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.quads[20] = (2, 5, 6, 3)
    return mesh


@pytest.mark.parametrize(
    "element_ids",
    (
        frozenset({"10"}),
        frozenset({10.0}),
        frozenset({10}),
        ["10"],
        (10.0,),
        {True: 10}.values(),
    ),
)
def test_locate_all_coerces_every_public_iterable_filter(element_ids):
    bvh = MeshElementBVH(_quad_mesh(), tolerance=1.0e-9)
    hits = bvh.locate_all((0.5, 0.5, 0.0), element_ids=element_ids)
    assert [hit.element_id for hit in hits] == [10]


def test_locate_all_rejects_nonmatching_public_frozenset():
    bvh = MeshElementBVH(_quad_mesh(), tolerance=1.0e-9)
    # A plain frozenset is coerced member by member, so a wrong string id
    # filters the hit out instead of being mistaken for a trusted filter.
    assert bvh.locate_all((0.5, 0.5, 0.0), element_ids=frozenset({"20"})) == ()


def test_private_normalized_filter_supports_repeated_queries():
    bvh = MeshElementBVH(_quad_mesh(), tolerance=1.0e-9)
    trusted = normalized_element_filter(("10",))
    for _ in range(3):
        hits = bvh.locate_all((0.5, 0.5, 0.0), element_ids=trusted)
        assert [hit.element_id for hit in hits] == [10]
        assert bvh.locate((1.5, 0.5, 0.0), element_ids=trusted) is None


# ---------------------------------------------------------------------------
# Carrier contract: mesh replacement invalidates admission data
# ---------------------------------------------------------------------------


def _deferred_candidate(geometry, **extra):
    result = hybrid.generate_hybrid_mesh_result(
        geometry,
        **_auto_call(qualified_s3=False, **extra),
        _defer_finalization=True,
    )
    assert isinstance(result, hybrid._HybridCandidateContext)
    return result


def test_with_mesh_invalidates_prior_admission_record_by_default():
    geometry, _face = _plate_model()
    carrier = _deferred_candidate(geometry)
    record = {
        "element_ids": sorted(carrier.mesh.tris),
        "authority_model": {"source_revision": int(geometry.revision)},
    }
    carrier.qualified_s3_record = record
    moved = mesh_from_dict(mesh_to_dict(carrier.mesh))
    first_node = min(moved.nodes)
    moved.nodes[first_node] = moved.nodes[first_node] + np.asarray(
        (0.0, 0.0, 0.25)
    )
    # Coordinate-only movement keeps every triangle id; only the structural
    # invalidation can prevent a stale admission from being published.
    assert sorted(moved.tris) == sorted(carrier.mesh.tris)

    replaced = carrier.with_mesh(moved)

    assert replaced.mesh is moved
    assert replaced.qualified_s3_record is None
    explicit = carrier.with_mesh(moved, qualified_s3_record=record)
    assert explicit.qualified_s3_record is record


def test_with_mesh_shares_selection_diagnostics_containers():
    geometry, _face = _plate_model()
    carrier = _deferred_candidate(geometry)
    replaced = carrier.with_mesh(carrier.mesh)
    replaced.extra_diagnostics["late_selection_note"] = {"kept": True}
    assert carrier.extra_diagnostics["late_selection_note"] == {"kept": True}


# ---------------------------------------------------------------------------
# Route behavior: connectivity once, counters, winning working ids
# ---------------------------------------------------------------------------


def test_single_candidate_route_applies_connectivity_once(monkeypatch):
    geometry, _face = _plate_model()
    geometry_before = to_dict(geometry)
    calls = _count_connectivity(monkeypatch)

    result = hybrid.generate_hybrid_mesh_result(
        geometry, **_auto_call(qualified_s3=False)
    )

    assert len(calls) == 1
    assert calls[0] is result.mesh
    runtime = result.mesh.hybrid_diagnostics["runtime"]
    assert runtime["candidate_count"] == 1
    assert runtime["connectivity_application_count"] == 1
    assert to_dict(geometry) == geometry_before


def test_rejected_fallback_route_applies_connectivity_once(monkeypatch):
    geometry, _face = _plate_model()
    geometry_before = to_dict(geometry)
    calls = _count_connectivity(monkeypatch)
    real_quality = hybrid._structured_quality_report
    carriers = []
    quality_calls = 0

    def reject_structured(mesh, options, *args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        measured = real_quality(mesh, options, *args, **kwargs)
        if quality_calls == 1:
            # Route control only: the structured candidate is refused so the
            # established native fallback chain is exercised.  Every other
            # report is the unmodified real measurement.
            return {**measured, "accepted": False, "test_route": "fallback"}
        return measured

    real_generate = hybrid.generate_hybrid_mesh_result

    def capture_candidates(*args, **kwargs):
        result = real_generate(*args, **kwargs)
        if isinstance(result, hybrid._HybridCandidateContext):
            carriers.append(result)
        return result

    monkeypatch.setattr(hybrid, "_structured_quality_report", reject_structured)
    monkeypatch.setattr(hybrid, "generate_hybrid_mesh_result", capture_candidates)

    result = hybrid.generate_hybrid_mesh_result(
        geometry, **_auto_call(qualified_s3=False)
    )

    assert len(calls) == 1
    assert calls[0] is result.mesh
    runtime = result.mesh.hybrid_diagnostics["runtime"]
    assert runtime["connectivity_application_count"] == 1
    # The structured candidate plus exactly one adopted native fallback.
    assert runtime["candidate_count"] == 2
    assert len(carriers) == 1
    winner = carriers[0]
    # The published mesh is the winning candidate's own working mesh, and its
    # associations use the winner's working face ids.
    assert result.mesh is winner.mesh
    assert set(result.mesh.elements_of_face) == set(winner.mesh.elements_of_face)
    assert result.mesh.hybrid_diagnostics["complex_geometry"][
        "final_face_count"
    ] == len(winner.faces)
    assert to_dict(geometry) == geometry_before


def test_failed_candidate_chain_applies_connectivity_zero_times(monkeypatch):
    geometry, _face = _plate_model()
    geometry_before = to_dict(geometry)
    calls = _count_connectivity(monkeypatch)
    real_quality = hybrid._structured_quality_report

    def reject_all(mesh, options, *args, **kwargs):
        measured = real_quality(mesh, options, *args, **kwargs)
        return {**measured, "accepted": False, "test_route": "reject all"}

    monkeypatch.setattr(hybrid, "_structured_quality_report", reject_all)
    with pytest.raises(StructuredQualityRejected):
        hybrid.generate_hybrid_mesh_result(
            geometry, **_auto_call(qualified_s3=False)
        )
    assert calls == []
    assert to_dict(geometry) == geometry_before


# ---------------------------------------------------------------------------
# Repair ordering and admission freshness at finalization
# ---------------------------------------------------------------------------


def test_shell_repairs_and_s3_qualification_precede_connectivity(monkeypatch):
    geometry, _face = _plate_model()
    geometry_before = to_dict(geometry)
    order = []
    real_prepare = hybrid.prepare_qualified_s3_mesh
    real_quality = hybrid._structured_quality_report
    real_connectivity = StructuralMeshingPipeline.apply_connectivity
    quality_calls = 0

    def reject_structured(mesh, options, *args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        measured = real_quality(mesh, options, *args, **kwargs)
        if quality_calls == 1:
            return {**measured, "accepted": False, "test_route": "fallback"}
        return measured

    def prepare(mesh, owner, *args, **kwargs):
        order.append("prepare")
        return real_prepare(mesh, owner, *args, **kwargs)

    def counted(self, mesh, *args, **kwargs):
        order.append("connectivity")
        return real_connectivity(self, mesh, *args, **kwargs)

    monkeypatch.setattr(hybrid, "_structured_quality_report", reject_structured)
    monkeypatch.setattr(hybrid, "prepare_qualified_s3_mesh", prepare)
    monkeypatch.setattr(StructuralMeshingPipeline, "apply_connectivity", counted)

    result = hybrid.generate_hybrid_mesh_result(
        geometry, **_auto_call(qualified_s3=True)
    )

    assert order.count("connectivity") == 1
    assert order[-1] == "connectivity"
    assert set(order[:-1]) == {"prepare"}
    record = result.mesh.structural_preparation["qualified_s3"]
    assert record is not None
    assert record["element_ids"] == sorted(result.mesh.tris)
    assert record["authority_model"]["source_revision"] == geometry.revision
    published_nodes = set(result.mesh.nodes)
    for coupling in result.mesh.couplings.values():
        assert coupling.beam_node in published_nodes
        assert set(coupling.plate_nodes) <= published_nodes
    assert to_dict(geometry) == geometry_before


def test_finalizer_refuses_stale_admission_record(monkeypatch):
    geometry, _face = _plate_model()
    calls = _count_connectivity(monkeypatch)
    carrier = _deferred_candidate(geometry)
    assert carrier.qualified_s3_record is None
    # Fabricate the qualified record a selected candidate would carry, then
    # corrupt it so it no longer describes the candidate's shell.
    carrier.qualified_s3_record = {
        "element_ids": [-999999],
        "authority_model": {"source_revision": int(geometry.revision)},
    }
    guard_calls = []
    carrier.publication_guard = lambda stage: guard_calls.append(stage)

    with pytest.raises(MeshError, match="does not describe the published shell"):
        hybrid._finalize_hybrid_candidate(carrier)

    assert guard_calls == []
    # Connectivity ran, but nothing was published after the impossibility
    # was detected.
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Cancellation and final-callback mutation never publish
# ---------------------------------------------------------------------------


def test_cancellation_before_connectivity_publishes_nothing(monkeypatch):
    geometry, _face = _plate_model()
    geometry_before = to_dict(geometry)
    calls = _count_connectivity(monkeypatch)
    cancelled = KeyboardInterrupt("regression: cancel at connectivity start")

    def cancel(stage):
        if stage == "hybrid connectivity start":
            raise cancelled

    with pytest.raises(KeyboardInterrupt) as caught:
        hybrid.generate_hybrid_mesh_result(
            geometry,
            **_auto_call(qualified_s3=False),
            cancellation_check=cancel,
        )
    assert caught.value is cancelled
    assert calls == []
    assert to_dict(geometry) == geometry_before


def test_final_callback_geometry_mutation_refuses_publication(monkeypatch):
    geometry, _face = _plate_model()
    geometry_before = to_dict(geometry)
    _count_connectivity(monkeypatch)

    def mutate_at_final_callback(stage):
        if stage == "hybrid generation complete":
            geometry.add_point(10.0, 0.0, 0.0)

    with pytest.raises(MeshError, match="stale"):
        hybrid.generate_hybrid_mesh_result(
            geometry,
            **_auto_call(qualified_s3=False),
            cancellation_check=mutate_at_final_callback,
        )
    # The mutation happened, but no result escaped the call.
    assert to_dict(geometry) != geometry_before


# ---------------------------------------------------------------------------
# Invocation-local connectivity index behavior
# ---------------------------------------------------------------------------


def _crossing_geometry(target=AttachmentTargetKind.FACE, target_id=None):
    """One plate with a member crossing it through the plate center."""
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    sheet = geometry.add_sheet((face,))
    member_vertices = geometry.add_points(
        ((0.5, 0.5, -1.0), (0.5, 0.5, 1.0))
    )
    member_edge = geometry.add_line(*member_vertices)
    member = geometry.add_member((member_edge,))
    if target is not None:
        geometry.add_attachment(
            member,
            AttachmentKind.MEMBER_THROUGH_FACE,
            target,
            face if target_id is None else target_id,
            ParameterRange.point(0.5),
            (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        )
    return geometry, face, sheet, member, member_edge, vertices


def _crossing_mesh(geometry, face, member_edge, vertices, order="linear"):
    mesh = Mesh(order=order)
    for node_id, vertex_id in enumerate(vertices, start=1):
        mesh.nodes[node_id] = np.asarray(
            geometry.vertices[vertex_id].position, dtype=float
        )
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.elements_of_face[face] = [10]
    mesh.nodes.update(
        {
            5: np.asarray((0.5, 0.5, -1.0)),
            6: np.asarray((0.5, 0.5, 0.0)),
            7: np.asarray((0.5, 0.5, 1.0)),
        }
    )
    mesh.beams[20] = (5, 6, 7)
    mesh.nodes_of_edge[member_edge] = [5, 6, 7]
    mesh.elements_of_edge[member_edge] = [20]
    return mesh


def _pipeline(geometry):
    return StructuralMeshingPipeline(
        GeometryMeshingView(geometry),
        overlap_policy="connect_declared",
        mutation_policy="working_copy",
    )


def test_indexed_duplicate_attachment_creates_one_coupling():
    geometry, face, _sheet, member, _edge, vertices = _crossing_geometry()
    # A second, identical attachment for the same member station.
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )
    mesh = _crossing_mesh(geometry, face, _edge, vertices)
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert not report.issues
    couplings = list(mesh.couplings.values())
    assert len(couplings) == 1
    assert couplings[0].beam_node == 6
    # Analytic Q4 weights at the plate center.
    assert couplings[0].plate_nodes == (1, 2, 3, 4)
    assert couplings[0].weights == pytest.approx((0.25, 0.25, 0.25, 0.25))
    assert couplings[0].eccentricity == (0.0, 0.0, 0.0)
    assert [action.kind for action in report.actions] == ["attachment-coupling"]


def test_indexed_conflicting_attachment_reports_conflict():
    geometry, face, _sheet, member, member_edge, vertices = _crossing_geometry()
    second_vertices = geometry.add_points(
        ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 1.0, 0.0), (3.0, 1.0, 0.0))
    )
    second_face = geometry.add_plate(second_vertices)
    geometry.add_sheet((second_face,))
    # Same member station, different target face: the second coupling target
    # differs from the first, so the index must report a conflict.
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        second_face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )
    mesh = _crossing_mesh(geometry, face, member_edge, vertices)
    for node_id, vertex_id in enumerate(second_vertices, start=8):
        mesh.nodes[node_id] = np.asarray(
            geometry.vertices[vertex_id].position, dtype=float
        )
    mesh.quads[30] = (8, 9, 10, 11)
    mesh.elements_of_face[second_face] = [30]
    report = _pipeline(geometry).apply_connectivity(mesh)
    couplings = list(mesh.couplings.values())
    assert len(couplings) == 1
    assert couplings[0].plate_nodes == (1, 2, 3, 4)
    assert [issue.code for issue in report.issues] == ["coupling-conflict"]


def test_indexed_shared_node_station_creates_no_redundant_coupling():
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    geometry.add_sheet((face,))
    # The member's meshed end station IS shell node 1 at the plate corner.
    below = geometry.add_points(((0.0, 0.0, -1.0),))[0]
    member_edge = geometry.add_line(below, vertices[0])
    member = geometry.add_member((member_edge,))
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(1.0),
        (ParameterRange.point(0.0), ParameterRange.point(0.0)),
    )
    mesh = Mesh()
    for node_id, vertex_id in enumerate(vertices, start=1):
        mesh.nodes[node_id] = np.asarray(
            geometry.vertices[vertex_id].position, dtype=float
        )
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.elements_of_face[face] = [10]
    mesh.nodes[5] = np.asarray((0.0, 0.0, -1.0))
    mesh.beams[20] = (5, 1)
    mesh.nodes_of_edge[member_edge] = [5, 1]
    mesh.elements_of_edge[member_edge] = [20]
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert not report.issues
    assert mesh.couplings == {}
    assert [
        (action.kind, action.source, action.target)
        for action in report.actions
    ] == [("shared-node", ("member", int(member)), ("face", face))]


def test_junction_rewrites_indexed_coupling_beam_node():
    geometry = GeometryModel()
    a0, a1, b1 = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    first = geometry.add_member((geometry.add_line(a0, a1),))
    second = geometry.add_member((geometry.add_line(a1, b1),))
    geometry.add_junction(
        JunctionKind.ENDPOINT,
        (
            JunctionMemberUse(first, ParameterRange.point(1.0)),
            JunctionMemberUse(second, ParameterRange.point(0.0)),
        ),
    )
    # A plate beside the members; the second member's start station is
    # attached to it, so the station receives a coupling before the
    # junction merges its node.
    plate_vertices = geometry.add_points(
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    face = geometry.add_plate(plate_vertices)
    geometry.add_sheet((face,))
    geometry.add_attachment(
        second,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.5), ParameterRange.point(0.0)),
    )
    view = GeometryMeshingView(geometry)
    mesh = Mesh()
    mesh.nodes.update(
        {
            1: np.asarray((0.0, 1.0, 0.0)),
            2: np.asarray((1.0, 1.0, 0.0)),
            3: np.asarray((1.0, 2.0, 0.0)),
            4: np.asarray((0.0, 2.0, 0.0)),
            5: np.asarray((0.0, 0.0, 0.0)),
            6: np.asarray((1.0, 0.0, 0.0)),
            7: np.asarray((1.0, 0.0, 0.0)),
            8: np.asarray((2.0, 0.0, 0.0)),
        }
    )
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.elements_of_face[face] = [10]
    edges = [
        use.edge_id
        for member_id in (first, second)
        for use in view.edge_uses_for_member(member_id)
    ]
    mesh.beams[11] = (5, 6)
    mesh.beams[12] = (7, 8)
    mesh.nodes_of_edge[edges[0]] = [5, 6]
    mesh.nodes_of_edge[edges[1]] = [7, 8]
    mesh.elements_of_edge[edges[0]] = [11]
    mesh.elements_of_edge[edges[1]] = [12]
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert not report.issues
    couplings = list(mesh.couplings.values())
    assert len(couplings) == 1
    # The junction merged node 7 into node 6; the coupling recorded for the
    # replaced beam node must reference the surviving node.
    assert couplings[0].beam_node == 6
    assert couplings[0].plate_nodes == (1, 2, 3, 4)
    assert mesh.beams == {11: (5, 6), 12: (6, 8)}
    assert sorted(action.kind for action in report.actions) == [
        "attachment-coupling",
        "junction-shared-node",
    ]


def test_edge_target_attachment_reuses_indexed_edge_filter():
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    geometry.add_sheet((face,))
    edge = geometry.faces[face].loop[0].edge
    # A member parallel to and below the shared edge, attached to the edge.
    # No mesh node exists at the inner edge stations, so every station must
    # resolve through the indexed edge filter into the containing element.
    member_vertices = geometry.add_points(
        ((0.0, 0.0, -0.3), (1.0, 0.0, -0.3))
    )
    member_edge = geometry.add_line(*member_vertices)
    member = geometry.add_member((member_edge,))
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
    )
    mesh = Mesh()
    for node_id, vertex_id in enumerate(vertices, start=1):
        mesh.nodes[node_id] = np.asarray(
            geometry.vertices[vertex_id].position, dtype=float
        )
    mesh.quads[10] = (1, 2, 3, 4)
    mesh.elements_of_face[face] = [10]
    mesh.nodes.update(
        {
            5: np.asarray((0.25, 0.0, -0.3)),
            6: np.asarray((0.5, 0.0, -0.3)),
            7: np.asarray((0.75, 0.0, -0.3)),
        }
    )
    mesh.beams[20] = (5, 6, 7)
    mesh.nodes_of_edge[member_edge] = [5, 6, 7]
    mesh.elements_of_edge[member_edge] = [20]
    mesh.nodes_of_edge[edge] = [1, 2]
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert not report.issues
    couplings = sorted(mesh.couplings.values(), key=lambda item: item.beam_node)
    assert [item.beam_node for item in couplings] == [5, 6, 7]
    # All three stations resolve to the same containing element through the
    # cached edge filter; analytic Q4 weights on the edge (eta = -1).
    for coupling, parameter in zip(couplings, (0.25, 0.5, 0.75)):
        assert coupling.plate_nodes == (1, 2, 3, 4)
        assert coupling.weights == pytest.approx(
            (1.0 - parameter, parameter, 0.0, 0.0)
        )
        assert coupling.eccentricity == pytest.approx((0.0, 0.0, -0.3))
    assert [action.kind for action in report.actions] == [
        "attachment-coupling",
        "attachment-coupling",
        "attachment-coupling",
    ]


def test_sheet_attachment_resolves_one_declared_face():
    geometry, face, sheet, member, member_edge, vertices = _crossing_geometry(
        target=None
    )
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_CROSS_SHEET,
        AttachmentTargetKind.SHEET,
        sheet,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        metadata={"face_sequence": [face]},
    )
    mesh = _crossing_mesh(geometry, face, member_edge, vertices)
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert not report.issues
    couplings = list(mesh.couplings.values())
    assert len(couplings) == 1
    assert couplings[0].beam_node == 6
    assert couplings[0].plate_nodes == (1, 2, 3, 4)
    assert couplings[0].weights == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_ambiguous_sheet_attachment_is_refused():
    geometry, face, sheet, member, member_edge, vertices = _crossing_geometry(
        target=None
    )
    second_vertices = geometry.add_points(
        ((3.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 1.0, 0.0), (3.0, 1.0, 0.0))
    )
    second_face = geometry.add_plate(second_vertices)
    geometry.add_sheet((second_face,))
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_CROSS_SHEET,
        AttachmentTargetKind.SHEET,
        sheet,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        metadata={"face_sequence": [face, second_face]},
    )
    mesh = _crossing_mesh(geometry, face, member_edge, vertices)
    for node_id, vertex_id in enumerate(second_vertices, start=8):
        mesh.nodes[node_id] = np.asarray(geometry.vertices[vertex_id].position)
    mesh.quads[30] = (8, 9, 10, 11)
    mesh.elements_of_face[second_face] = [30]
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert mesh.couplings == {}
    assert [issue.code for issue in report.issues] == [
        "ambiguous-sheet-attachment-face"
    ]


def test_quadratic_attachment_couples_midside_nodes():
    geometry, face, _sheet, _member, member_edge, vertices = _crossing_geometry()
    mesh = _crossing_mesh(
        geometry, face, member_edge, vertices, order="quadratic"
    )
    # Upgrade the shell element to Q8 with midside nodes on the unit plate.
    mesh.nodes.update(
        {
            8: np.asarray((0.5, 0.0, 0.0)),
            9: np.asarray((1.0, 0.5, 0.0)),
            10: np.asarray((0.5, 1.0, 0.0)),
            11: np.asarray((0.0, 0.5, 0.0)),
        }
    )
    mesh.quads[10] = (1, 2, 3, 4, 8, 9, 10, 11)
    assert mesh.is_quadratic
    report = _pipeline(geometry).apply_connectivity(mesh)
    assert not report.issues
    couplings = list(mesh.couplings.values())
    assert len(couplings) == 1
    assert couplings[0].beam_node == 6
    assert couplings[0].plate_nodes == (1, 2, 3, 4, 8, 9, 10, 11)
    # Analytic Q8 weights at the element center: corners -1/4, midsides 1/2.
    assert couplings[0].weights == pytest.approx(
        (-0.25, -0.25, -0.25, -0.25, 0.5, 0.5, 0.5, 0.5)
    )
    assert couplings[0].eccentricity == (0.0, 0.0, 0.0)


def test_deferred_s3_finalizer_uses_working_geometry_before_connectivity(monkeypatch):
    geometry, _ = _plate_model()
    carrier = _deferred_candidate(geometry)
    assert carrier.working_geometry is not geometry
    carrier.defer_qualified_s3 = True
    stages = []
    real_prepare = hybrid.prepare_qualified_s3_mesh
    real_connectivity = StructuralMeshingPipeline.apply_connectivity

    def prepare(mesh, owner, **kwargs):
        assert owner is carrier.working_geometry
        stages.append('qualification')
        return real_prepare(mesh, owner, **kwargs)

    def connect(pipeline, mesh, **kwargs):
        assert pipeline is carrier.pipeline
        stages.append('connectivity')
        return real_connectivity(pipeline, mesh, **kwargs)

    monkeypatch.setattr(hybrid, 'prepare_qualified_s3_mesh', prepare)
    monkeypatch.setattr(StructuralMeshingPipeline, 'apply_connectivity', connect)
    result = hybrid._finalize_hybrid_candidate(carrier)
    assert stages == ['qualification', 'connectivity']
    assert result.mesh.structural_preparation['qualified_s3']['authority_model'][
        'source_model_id'] == str(geometry.model_id)


@pytest.mark.parametrize('selected_count', [3, 4], ids=['refined', 'conservative'])
def test_later_fallback_routes_finalize_only_the_winner(monkeypatch, selected_count):
    geometry, _ = _plate_model()
    real_generate = hybrid.generate_hybrid_mesh_result
    real_quality = hybrid._structured_quality_report
    real_connectivity = StructuralMeshingPipeline.apply_connectivity
    carriers, finalized = [], []
    quality_calls = 0

    def generate(*args, **kwargs):
        candidate = real_generate(*args, **kwargs)
        if isinstance(candidate, hybrid._HybridCandidateContext):
            # Route injection only: enter the declared-junction fallback
            # ladder without relying on a numerically poor test geometry.
            candidate.mesh.declared_plate_junction_edges = ((1, 2),)
            carriers.append(candidate)
        return candidate

    def quality(mesh, options, *args, **kwargs):
        nonlocal quality_calls
        quality_calls += 1
        report = real_quality(mesh, options, *args, **kwargs)
        if quality_calls < selected_count:
            return {**report, 'accepted': False}
        return report

    def connect(pipeline, mesh, **kwargs):
        assert pipeline is carriers[-1].pipeline
        assert mesh is carriers[-1].mesh
        finalized.append(mesh)
        return real_connectivity(pipeline, mesh, **kwargs)

    monkeypatch.setattr(hybrid, 'generate_hybrid_mesh_result', generate)
    monkeypatch.setattr(hybrid, '_structured_quality_report', quality)
    monkeypatch.setattr(hybrid, '_junction_growth_repair',
                        lambda owner, mesh, report, options: (None, report, {}))
    monkeypatch.setattr(StructuralMeshingPipeline, 'apply_connectivity', connect)
    result = generate(geometry, **_auto_call(qualified_s3=False))
    assert len(carriers) == selected_count - 1
    assert finalized == [result.mesh]
    assert result.mesh.hybrid_diagnostics['runtime']['candidate_count'] == selected_count
    assert result.mesh.hybrid_diagnostics['runtime']['connectivity_application_count'] == 1


@pytest.mark.parametrize("missing", ["beam", "shell"])
def test_incomplete_selected_connectivity_cannot_publish(monkeypatch, missing):
    from anymesher.mesh import Coupling

    geometry, _ = _plate_model()
    carrier = _deferred_candidate(geometry)
    publications = []
    carrier.publication_guard = publications.append
    carrier.mesh.nodes[100] = np.asarray((0., 0., -1.))
    carrier.mesh.nodes[101] = np.asarray((1., 0., -1.))
    carrier.mesh.beams[1000] = (100, 101)
    carrier.mesh.couplings[2000] = Coupling(
        beam_node=999 if missing == "beam" else 100,
        plate_nodes=(999 if missing == "shell" else min(carrier.mesh.nodes),),
        weights=(1.,), eccentricity=(0., 0., 0.))
    with pytest.raises(MeshError, match="does not reference the final beam and shell mesh"):
        hybrid._finalize_hybrid_candidate(carrier)
    assert publications == []
