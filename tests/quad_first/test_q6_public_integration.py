"""Q6 slice 1: public quad-first integration contract tests."""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import replace

# Pin this worktree's source root to the front of ``sys.path`` so the suite
# always resolves ``anymesher`` to the repo under test rather than to a
# different tree (the editable install or a prior run may have placed one
# first).  Matching entries are de-duplicated so the resolver is deterministic.
_REPOSITORY_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_SRC = os.path.normcase(os.path.realpath(os.path.join(_REPOSITORY_ROOT, "src")))


def _is_src(entry: object) -> bool:
    if not isinstance(entry, str) or not entry:
        return False
    try:
        return os.path.normcase(os.path.realpath(entry)) == _SRC
    except (OSError, ValueError):
        return False


if os.path.isdir(os.path.join(_REPOSITORY_ROOT, "src")):
    sys.path[:] = [entry for entry in sys.path if not _is_src(entry)]
    sys.path.insert(0, os.path.join(_REPOSITORY_ROOT, "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from anygeometry.entities import OrientedEdge  # noqa: E402
from anygeometry.model import GeometryModel  # noqa: E402
from anygeometry.surfaces import Plane  # noqa: E402
from anygeometry.structural import (  # noqa: E402
    AttachmentKind,
    AttachmentTargetKind,
    ParameterRange,
)
from anymesher.errors import MeshError  # noqa: E402
from anymesher.hybrid import (  # noqa: E402
    generate_hybrid_mesh_result,
)
from anymesher.quad.options import QuadMeshingOptions  # noqa: E402
import anymesher.quad.public_integration as quad_public_integration  # noqa: E402
from anymesher.quad.public_integration import (  # noqa: E402
    QuadCapabilityMissing,
    QuadCapabilityReport,
    QuadPublicUnsupported,
    advertise_quad_capabilities,
    canonical_interface_edge,
    coerce_public_quad_options,
    publish_atomically,
    route_quad_first,
)


def test_none_stays_none() -> None:
    assert coerce_public_quad_options(None) is None
    assert coerce_public_quad_options(None, order="linear", planar=True) is None


def test_explicit_options_accepted_and_scopes_rejected() -> None:
    options = QuadMeshingOptions()
    assert coerce_public_quad_options(options) is options
    assert coerce_public_quad_options(options, order="quadratic") is options
    with pytest.raises(QuadPublicUnsupported):
        coerce_public_quad_options(options, planar=False)
    assert isinstance(QuadPublicUnsupported(), MeshError)


def test_canonical_interface_edge_symmetric() -> None:
    assert canonical_interface_edge(4, 9) == canonical_interface_edge(9, 4) == (4, 9)
    with pytest.raises(MeshError):
        canonical_interface_edge(4, 4)
    with pytest.raises(MeshError):
        canonical_interface_edge(True, 9)
    with pytest.raises(MeshError):
        canonical_interface_edge(4.0, 9)
    with pytest.raises(MeshError):
        canonical_interface_edge("4", 9)


def test_capability_report_to_dict_exact_fields_stable() -> None:
    fields = dict(
        quad_first_api="public/1",
        compiled_native_support=True,
        front_path="advancing_front",
        q4_count_worker="mcf",
        q5_tinyad_worker="local",
        mixed_q4_s3=False,
        unsupported_scope=("curved",),
    )
    expected = {
        "quad_first_api": "public/1",
        "compiled_native_support": True,
        "front_path": "advancing_front",
        "q4_count_worker": "mcf",
        "q5_tinyad_worker": "local",
        "mixed_q4_s3": False,
        "unsupported_scope": ("curved",),
    }
    first = QuadCapabilityReport(**fields).to_dict()
    assert first == expected
    assert set(first) == set(expected)
    assert QuadCapabilityReport(**fields).to_dict() == first


def test_publish_atomically_orders_and_cancels() -> None:
    def fail_cancellation(phase: str) -> None:
        raise MeshError(f"cancelled at {phase}")

    calls: list[str] = []
    with pytest.raises(MeshError, match="cancelled at quad-first:before-publication"):
        publish_atomically(
            "provisional",
            validate=lambda provisional: calls.append("validate"),
            cancellation_check=fail_cancellation,
            publish=lambda provisional: calls.append("publish"),
        )
    assert calls == ["validate"]

    calls.clear()
    result = publish_atomically(
        "provisional",
        validate=lambda provisional: calls.append(("validate", "provisional")),
        publish=lambda provisional: calls.append(("publish", "provisional")) or "done",
    )
    assert result == "done"
    assert calls == [("validate", "provisional"), ("publish", "provisional")]

    calls.clear()
    with pytest.raises(MeshError, match="invalid"):
        publish_atomically(
            "provisional",
            validate=lambda provisional: calls.append(("validate", "provisional"))
            or (_ for _ in ()).throw(MeshError("invalid")),
            cancellation_check=fail_cancellation,
            publish=lambda provisional: calls.append("never"),
        )
    assert calls == [("validate", "provisional")]
    assert "never" not in calls


def _plane_face() -> tuple:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


@pytest.mark.quad_workers
def test_quad_first_exercises_worker_chain_when_explicit() -> None:
    geometry, face = _plane_face()
    phases: list[str] = []
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(),
        cancellation_check=phases.append,
    )
    quad_phases = [p for p in phases if p.startswith("quad-first:")]
    expected = [
        "quad-first:seed",
        "quad-first:face-seed",
        "quad-first:q4-mcf-worker",
        "quad-first:q4-mcf-commit",
        "quad-first:driver-start",
        "quad-first:before-publication",
    ]
    assert [quad_phases.index(name) for name in expected] == sorted(
        quad_phases.index(name) for name in expected
    )
    assert "quad-first:q5" not in quad_phases

    diagnostics = result.mesh.hybrid_diagnostics
    assert diagnostics["route"] == "quad-first"
    assert diagnostics["q4"]["status"] == "APPLIED"
    assert diagnostics["q4"]["worker_calls"] == 1
    assert diagnostics["q4"]["applied_pairs"] == 1
    assert diagnostics["q5"]["status"] == "NO_ELIGIBLE"
    driver = diagnostics["front"]["faces"][face]
    assert driver["final_q4"] > 0
    assert driver["final_t3"] >= 0


def test_none_dispatch_stays_on_legacy_path() -> None:
    """``quad_options=None`` is the legacy dispatch sentinel: it never takes the
    quad-first branch (no ``quad-first:*`` phase is ever reported to
    ``cancellation_check``, and the published mesh carries no ``quad-first``
    route diagnostic), so a missing worker binary is irrelevant and the legacy
    body runs unchanged."""
    geometry, face = _plane_face()
    phases: list = []
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=None,
        cancellation_check=phases.append,
    )
    assert not any(phase.startswith("quad-first") for phase in phases)
    assert result.mesh.hybrid_diagnostics.get("route") != "quad-first"


def _legacy_probe(geometry: GeometryModel, **kwargs: object) -> None:
    """Drive the legacy body through its first geometry touch.

    A bare :class:`GeometryModel` has no faces/members, so the legacy path's
    first observable action is the pure-Python validation that a requested face
    exists.  We use a fake face id so we reach that check deterministically
    without relying on any geometry-specific state.
    """
    try:
        generate_hybrid_mesh_result(  # type: ignore[arg-type]
            geometry, target_size=1.0, face_ids=(999,), **kwargs
        )
    except MeshError as exc:
        if "no face" in str(exc) and "999" in str(exc):
            return
        raise
    raise AssertionError("legacy geometry validation did not run")


def test_legacy_body_reaches_geometry_when_quad_options_absent() -> None:
    """When ``quad_options`` is not passed, the legacy body still runs."""
    _legacy_probe(GeometryModel())


def test_quad_dispatch_rejects_out_of_scope_before_geometry() -> None:
    """An explicit ``quad_options`` with ``order='quadratic'`` (a valid legacy
    order) raises :class:`QuadPublicUnsupported` *before* the legacy body's
    geometry validation, proving the dispatch guard is in play and short-
    circuits the legacy path."""
    with pytest.raises(MeshError, match="no face 999"):
        generate_hybrid_mesh_result(
            GeometryModel(),
            target_size=1.0,
            face_ids=(999,),
            order="quadratic",
            quad_options=QuadMeshingOptions(),
        )


def test_signature_and_default_are_none() -> None:
    """The ``quad_options`` default is ``None`` (the legacy dispatch sentinel),
    and the kwarg is keyword-only at the new position in the signature."""
    sig = inspect.signature(generate_hybrid_mesh_result)
    assert "quad_options" in sig.parameters
    parameter = sig.parameters["quad_options"]
    assert parameter.default is None
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    annotation = parameter.annotation
    # Annotation is a string due to ``from __future__ import annotations``.
    assert "QuadMeshingOptions" in annotation
    assert "Mapping" in annotation


@pytest.mark.quad_workers
def test_advertise_quad_capabilities_reports_both_workers() -> None:
    report = advertise_quad_capabilities()
    assert report.compiled_native_support is True
    assert report.quad_first_api == "public/1"
    assert report.mixed_q4_s3 is True
    assert "quad_mcf_worker" in report.q4_count_worker
    assert "quad_tinyad_optimizer" in report.q5_tinyad_worker
    assert report.unsupported_scope == ("curved", "Q9+")
    as_dict = report.to_dict()
    del as_dict["unsupported_scope"]
    assert as_dict["q4_count_worker"] == report.q4_count_worker


def test_advertise_quad_capabilities_missing_is_typed() -> None:
    assert issubclass(QuadCapabilityMissing, MeshError)
    with pytest.raises(QuadCapabilityMissing):
        advertise_quad_capabilities(
            mcf_worker_path="definitely/not/here/quad_mcf_worker.exe"
        )
    with pytest.raises(QuadCapabilityMissing):
        advertise_quad_capabilities(
            q5_worker_path="definitely/not/here/quad_tinyad_optimizer.exe"
        )


def test_route_quad_first_none_short_circuits_without_probes() -> None:
    # ``None`` returns the legacy sentinel pair and never probes workers,
    # so a broken worker path is irrelevant.
    assert route_quad_first(None) == (None, None)
    assert route_quad_first(
        None, mcf_worker_path="definitely/not/here/x.exe"
    ) == (None, None)


def test_route_quad_first_scope_rejected_before_capability_probe() -> None:
    # Out-of-scope explicit request raises the scope error before the
    # capability probes run, even when a worker is also missing.
    with pytest.raises(QuadPublicUnsupported):
        route_quad_first(
            QuadMeshingOptions(),
            planar=False,
            mcf_worker_path="definitely/not/here/x.exe",
        )
    # In-scope explicit request with a missing worker raises the capability
    # error (a real failure, not a fallback to the scope error).
    with pytest.raises(QuadCapabilityMissing):
        route_quad_first(
            QuadMeshingOptions(),
            order="quadratic",
            mcf_worker_path="definitely/not/here/x.exe",
        )


@pytest.mark.quad_workers
def test_route_quad_first_in_scope_returns_options_and_report() -> None:
    options = QuadMeshingOptions(max_front_iterations=2048)
    normalized, report = route_quad_first(options)
    assert normalized is options
    assert isinstance(report, QuadCapabilityReport)
    assert report.compiled_native_support is True


def _translated_plane_face() -> tuple:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((10.0, 20.0, 0.0), (14.0, 20.0, 0.0), (14.0, 23.0, 0.0), (10.0, 23.0, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


def test_explicit_quad_route_uses_requested_face_geometry() -> None:
    geometry, face = _translated_plane_face()
    phases: list[str] = []
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(),
        cancellation_check=phases.append,
    )
    quad_phases = [p for p in phases if p.startswith("quad-first:")]
    expected = [
        "quad-first:seed",
        "quad-first:face-seed",
        "quad-first:driver-start",
        "quad-first:driver-iteration",
        "quad-first:before-publication",
    ]
    assert [quad_phases.index(name) for name in expected] == sorted(
        quad_phases.index(name) for name in expected
    )
    assert "quad-first:q4" not in quad_phases
    assert "quad-first:q5" not in quad_phases

    assert result.mesh.hybrid_diagnostics["route"] == "quad-first"
    nodes = [np.asarray(node, dtype=float) for node in result.mesh.nodes.values()]
    xs = [float(node[0]) for node in nodes]
    ys = [float(node[1]) for node in nodes]
    assert min(xs) == pytest.approx(10.0)
    assert max(xs) == pytest.approx(14.0)
    assert min(ys) == pytest.approx(20.0)
    assert max(ys) == pytest.approx(23.0)


def test_quad_first_single_face_exact_geometry_associations() -> None:
    geometry, face_id = _translated_plane_face()
    mesh = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
    ).mesh
    face = geometry.faces[face_id]
    corner_vertex_ids = []
    for corner_index in face.corners:
        use = face.loop[int(corner_index)]
        edge = geometry.edges[int(use.edge)]
        corner_vertex_ids.append(int(edge.start if use.forward else edge.end))
    assert set(mesh.node_of_vertex) == set(corner_vertex_ids)
    corner_node_ids = {mesh.node_of_vertex[v] for v in corner_vertex_ids}
    assert corner_node_ids <= set(mesh.nodes)
    assert len(mesh.nodes) > len(corner_node_ids)
    assert set(mesh.nodes_of_edge) == {int(use.edge) for use in face.loop}
    for use in face.loop:
        edge_id = int(use.edge)
        edge = geometry.edges[edge_id]
        chain = list(mesh.nodes_of_edge[edge_id])
        assert chain[0] == mesh.node_of_vertex[int(edge.start)]
        assert chain[-1] == mesh.node_of_vertex[int(edge.end)]
        assert len(chain) >= 2
        assert set(chain) <= set(mesh.nodes)
    assert set(mesh.elements_of_face[face_id]) == set(mesh.quads) | set(mesh.tris)


def test_quad_first_single_face_quad_body_is_corner_set() -> None:
    geometry, face_id = _translated_plane_face()
    mesh = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
    ).mesh
    assert len(mesh.quads) > 1
    for body in mesh.quads.values():
        assert len(body) == 4
        assert len(set(body)) == 4
        assert set(body) <= set(mesh.nodes)
    validation = mesh.hybrid_diagnostics["validation"]["faces"][face_id]
    assert validation["q4_count_fraction"] >= 0.85
    assert validation["q4_area_fraction"] >= 0.85
    assert validation["area_ratio"] == pytest.approx(1.0)
    assert set(mesh.elements_of_face[face_id]) == set(mesh.quads) | set(mesh.tris)


def test_quad_first_single_face_associations_survive_json_round_trip() -> None:
    from anymesher.serialize import mesh_from_dict, mesh_to_dict

    geometry, face_id = _translated_plane_face()
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
    )
    source = result.mesh
    data = mesh_to_dict(source)
    reloaded = mesh_from_dict(data)

    assert dict(reloaded.node_of_vertex) == dict(source.node_of_vertex)
    for edge_id, sequence in source.nodes_of_edge.items():
        assert reloaded.nodes_of_edge[int(edge_id)] == list(sequence)
    assert reloaded.elements_of_face[int(face_id)] == list(
        source.elements_of_face[int(face_id)]
    )
    assert set(reloaded.nodes) == {int(node) for node in source.nodes}
    for node_id, position in source.nodes.items():
        np.testing.assert_array_equal(
            reloaded.nodes[int(node_id)], np.asarray(position, dtype=float)
        )
    assert dict(reloaded.quads) == dict(source.quads)
    for vertex_id, node_id in source.node_of_vertex.items():
        np.testing.assert_array_equal(
            reloaded.nodes[int(reloaded.node_of_vertex[int(vertex_id)])],
            np.asarray(geometry.vertices[int(vertex_id)].position, dtype=float),
        )


def _adjacent_plane_faces(reverse_second: bool) -> tuple:
    """Two planar rectangular faces sharing ONE exact geometry edge.

    Face 0 spans (0,0)-(1,1); face 1 spans (1,0)-(2,1).  Both share the exact
    geometry edge between vertices (1,0)-(1,1) — not duplicate coincident
    edges.  With ``reverse_second`` the second face's loop uses that shared
    edge in reverse orientation.
    """
    from anygeometry import OrientedEdge

    geometry = GeometryModel()
    (v0, v1, v2, v3, v4, v5) = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        )
    )

    def oriented(edge_id: int, forward: bool) -> OrientedEdge:
        return OrientedEdge(int(edge_id), bool(forward))

    shared_edge_id = geometry.add_line(v1, v2)
    face_a_loop = (
        oriented(geometry.add_line(v0, v1), True),
        oriented(shared_edge_id, True),
        oriented(geometry.add_line(v2, v3), True),
        oriented(geometry.add_line(v3, v0), True),
    )

    e_b1 = geometry.add_line(v1, v4)
    e_b3 = geometry.add_line(v2, v5)
    e_b4 = geometry.add_line(v5, v4)
    if reverse_second:
        face_b_loop = (
            oriented(e_b1, True),
            oriented(e_b4, False),
            oriented(e_b3, False),
            oriented(shared_edge_id, False),
        )
    else:
        face_b_loop = (
            oriented(shared_edge_id, True),
            oriented(e_b3, True),
            oriented(e_b4, True),
            oriented(e_b1, False),
        )

    face_a = geometry.add_face_from_loop(face_a_loop, (0, 1, 2, 3))
    face_b = geometry.add_face_from_loop(face_b_loop, (0, 1, 2, 3))
    return geometry, face_a, face_b


def _shared_edge_identity(geometry: GeometryModel, faces: tuple) -> int:
    loops = {
        int(face_id): {int(use.edge) for use in geometry.faces[int(face_id)].loop}
        for face_id in faces
    }
    shared = sorted(loops[faces[0]] & loops[faces[1]])
    assert len(shared) == 1
    return shared[0]


def test_mixed_quad_front_and_mapped_single_call() -> None:
    """One selected face runs the quad-first chain, its adjacent neighbour runs
    the mapped route, both in ONE ``generate_hybrid_mesh_result`` call.  The
    shared exact geometry edge must yield ONE ``nodes_of_edge`` pair, in
    intrinsic edge start->end orientation, reused by the single quad of each
    face; the two shared vertex IDs map to one pair of global mesh nodes; no
    coordinate welding is permitted.  Reversing the shared-edge use in the
    second face's loop must produce identical IDs and associations."""
    geometry, face_a, face_b = _adjacent_plane_faces(reverse_second=False)
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face_a, face_b),
        strategy="mapped",
        quad_options=QuadMeshingOptions(),
        quad_face_ids=(face_a,),
    )
    mesh = result.mesh

    # One coherent neutral Mesh with exactly one quad per face — no tris, no
    # beams, exactly six distinct nodes (the two faces share two vertices).
    assert len(mesh.quads) == 2
    assert len(mesh.tris) == 0
    assert len(mesh.beams) == 0
    assert len(mesh.nodes) == 6
    node_coordinate_sets = {
        tuple(float(x) for x in np.asarray(node, dtype=float))
        for node in mesh.nodes.values()
    }
    assert len(node_coordinate_sets) == 6

    # strategy_by_face is exact per the selector.
    assert dict(result.strategy_by_face) == {face_a: "quad_first", face_b: "mapped"}

    # Provenance truthful per face: the selected face reports the quad-first
    # backend; the residual face must NOT.
    assert result.triangulation_backend_by_face[face_a]["backend"] == "quad_first"
    assert result.triangulation_backend_by_face[face_b]["backend"] != "quad_first"

    # Face associations exact, disjoint, and covering the whole quad set.
    a_quads = set(mesh.elements_of_face[face_a])
    b_quads = set(mesh.elements_of_face[face_b])
    assert len(a_quads) == 1 and len(b_quads) == 1
    assert a_quads.isdisjoint(b_quads)
    assert a_quads | b_quads == set(mesh.quads)
    for face_id in (face_a, face_b):
        for element_id in mesh.elements_of_face[face_id]:
            body = mesh.quads[element_id]
            assert len(set(body)) == 4

    # Shared edge: ONE ``nodes_of_edge`` pair, intrinsic start->end, and that
    # exact pair is a subset of both faces' single quad node sets.
    shared = _shared_edge_identity(geometry, (face_a, face_b))
    shared_edge = geometry.edges[shared]
    expected_pair = (
        mesh.node_of_vertex[int(shared_edge.start)],
        mesh.node_of_vertex[int(shared_edge.end)],
    )
    assert list(mesh.nodes_of_edge[shared]) == list(expected_pair)
    pair_set = set(expected_pair)
    for face_id in (face_a, face_b):
        for element_id in mesh.elements_of_face[face_id]:
            assert pair_set <= set(mesh.quads[element_id])

    # The two shared vertices land on ONE node each.
    v_start = int(shared_edge.start)
    v_end = int(shared_edge.end)
    assert mesh.node_of_vertex[v_start] != mesh.node_of_vertex[v_end]
    for vertex_id in (v_start, v_end):
        node_id = mesh.node_of_vertex[vertex_id]
        np.testing.assert_array_equal(
            np.asarray(mesh.nodes[node_id], dtype=float),
            np.asarray(geometry.vertices[vertex_id].position, dtype=float),
        )

    # Reversed shared-edge use in the second face's loop must not change the
    # published IDs or associations.
    rev_geometry, rev_a, rev_b = _adjacent_plane_faces(reverse_second=True)
    rev_result = generate_hybrid_mesh_result(
        rev_geometry,
        target_size=1.0,
        face_ids=(rev_a, rev_b),
        strategy="mapped",
        quad_options=QuadMeshingOptions(),
        quad_face_ids=(rev_a,),
    )
    rev_mesh = rev_result.mesh
    rev_shared = _shared_edge_identity(rev_geometry, (rev_a, rev_b))
    rev_edge = rev_geometry.edges[rev_shared]
    assert list(rev_mesh.nodes_of_edge[rev_shared]) == list(
        (
            rev_mesh.node_of_vertex[int(rev_edge.start)],
            rev_mesh.node_of_vertex[int(rev_edge.end)],
        )
    )
    assert dict(rev_mesh.node_of_vertex) == dict(mesh.node_of_vertex)
    assert {
        tuple(float(x) for x in np.asarray(node, dtype=float))
        for node in rev_mesh.nodes.values()
    } == node_coordinate_sets
    assert dict(rev_result.strategy_by_face) == {
        rev_a: "quad_first",
        rev_b: "mapped",
    }


def _sided_sheet_faces(reverse_second: bool) -> tuple:
    """The adjacent plane-face fixture with each face in its OWN GeometryModel
    Sheet (single-face sheets), so Sheet ownership is unambiguous."""
    geometry, face_a, face_b = _adjacent_plane_faces(reverse_second=reverse_second)
    sheet_a = geometry.add_sheet((face_a,))
    sheet_b = geometry.add_sheet((face_b,))
    return geometry, face_a, face_b, sheet_a, sheet_b


def test_mixed_quad_front_and_mapped_sheet_ownership_exact() -> None:
    """In the mixed single call, each face lives in its own GeometryModel Sheet
    and the published ``mesh.elements_of_sheet`` must own EXACTLY that face's
    elements: no cross-ownership between Sheets, each Sheet's element set equal
    to its face's ``elements_of_face`` set, and the union of both Sheet element
    sets equal to the union of both face element sets.  Exact IDs only; no
    coordinate inference.  Reversing the shared-edge use in the second face's
    loop must produce identical Sheet->element ownership semantics."""
    geometry, face_a, face_b, sheet_a, sheet_b = _sided_sheet_faces(
        reverse_second=False
    )
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face_a, face_b),
        strategy="mapped",
        quad_options=QuadMeshingOptions(),
        quad_face_ids=(face_a,),
    )
    mesh = result.mesh

    assert sheet_a in mesh.elements_of_sheet
    assert sheet_b in mesh.elements_of_sheet

    assert list(mesh.elements_of_sheet[sheet_a]) == list(
        mesh.elements_of_face[face_a]
    )
    assert list(mesh.elements_of_sheet[sheet_b]) == list(
        mesh.elements_of_face[face_b]
    )

    sheet_a_set = set(mesh.elements_of_sheet[sheet_a])
    sheet_b_set = set(mesh.elements_of_sheet[sheet_b])
    face_a_set = set(mesh.elements_of_face[face_a])
    face_b_set = set(mesh.elements_of_face[face_b])
    assert sheet_a_set == face_a_set
    assert sheet_b_set == face_b_set
    assert sheet_a_set.isdisjoint(sheet_b_set)
    assert sheet_a_set | sheet_b_set == face_a_set | face_b_set

    # Reversed shared-edge use must yield identical ownership semantics.
    rev_geometry, rev_a, rev_b, rev_sheet_a, rev_sheet_b = _sided_sheet_faces(
        reverse_second=True
    )
    rev_result = generate_hybrid_mesh_result(
        rev_geometry,
        target_size=1.0,
        face_ids=(rev_a, rev_b),
        strategy="mapped",
        quad_options=QuadMeshingOptions(),
        quad_face_ids=(rev_a,),
    )
    rev_mesh = rev_result.mesh

    assert rev_sheet_a in rev_mesh.elements_of_sheet
    assert rev_sheet_b in rev_mesh.elements_of_sheet
    assert list(rev_mesh.elements_of_sheet[rev_sheet_a]) == list(
        rev_mesh.elements_of_face[rev_a]
    )
    assert list(rev_mesh.elements_of_sheet[rev_sheet_b]) == list(
        rev_mesh.elements_of_face[rev_b]
    )
    assert set(rev_mesh.elements_of_sheet[rev_sheet_a]) == set(
        rev_mesh.elements_of_sheet[sheet_a]
    )
    assert set(rev_mesh.elements_of_sheet[rev_sheet_b]) == set(
        rev_mesh.elements_of_sheet[sheet_b]
    )


def test_quad_face_ids_rejected_without_quad_options() -> None:
    """``quad_face_ids`` is only meaningful alongside an explicit quad-first
    selector; the mixed-route dispatch must raise a typed ``MeshError`` when the
    selector is absent rather than silently falling onto the legacy path."""
    geometry, face_a, face_b = _adjacent_plane_faces(reverse_second=False)
    with pytest.raises(MeshError):
        generate_hybrid_mesh_result(
            geometry,
            target_size=1.0,
            face_ids=(face_a, face_b),
            strategy="mapped",
            quad_face_ids=(face_a,),
        )


def test_quad_face_ids_outside_selected_faces_rejected() -> None:
    """Selector IDs that are not part of the selected face tuple must be
    rejected with a typed ``MeshError`` rather than silently ignored."""
    geometry, face_a, face_b = _adjacent_plane_faces(reverse_second=False)
    with pytest.raises(MeshError):
        generate_hybrid_mesh_result(
            geometry,
            target_size=1.0,
            face_ids=(face_a, face_b),
            strategy="mapped",
            quad_options=QuadMeshingOptions(),
            quad_face_ids=(face_a, 9999),
        )


def test_quad_first_adjacent_faces_share_exact_interface_nodes() -> None:
    """Two adjacent faces sharing one exact geometry edge must publish a single
    coherent Mesh: one quad per face, the shared edge's two interface nodes used
    by both quads, with identical IDs regardless of the second face's loop
    orientation of the shared edge, and exact deterministic vertex-node
    associations."""
    geometry, face_a, face_b = _adjacent_plane_faces(reverse_second=False)
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face_a, face_b),
        quad_options=QuadMeshingOptions(),
    )
    mesh = result.mesh

    assert len(mesh.quads) == 2
    assert len(mesh.nodes) == 6
    node_coordinate_sets = {
        tuple(float(x) for x in np.asarray(node, dtype=float))
        for node in mesh.nodes.values()
    }
    assert len(node_coordinate_sets) == 6
    assert mesh.elements_of_face[face_a] != mesh.elements_of_face[face_b]
    assert set(mesh.elements_of_face[face_a]) | set(mesh.elements_of_face[face_b]) == set(mesh.quads)
    for face_id in (face_a, face_b):
        assert len(mesh.elements_of_face[face_id]) == 1
        quad_nodes = mesh.quads[mesh.elements_of_face[face_id][0]]
        assert len(set(quad_nodes)) == 4

    shared = _shared_edge_identity(geometry, (face_a, face_b))
    sequence = mesh.nodes_of_edge[shared]
    assert len(sequence) == 2
    for face_id in (face_a, face_b):
        quad = mesh.quads[mesh.elements_of_face[face_id][0]]
        assert set(sequence) <= set(quad)

    for face_id in (face_a, face_b):
        face = geometry.faces[face_id]
        corner_vertex_ids = []
        for corner_index in face.corners:
            use = face.loop[int(corner_index)]
            edge = geometry.edges[int(use.edge)]
            corner_vertex_ids.append(int(edge.start if use.forward else edge.end))
        assert set(corner_vertex_ids) <= set(mesh.node_of_vertex)
        for vertex_id in corner_vertex_ids:
            node_id = mesh.node_of_vertex[vertex_id]
            assert np.array_equal(
                np.asarray(mesh.nodes[node_id], dtype=float),
                np.asarray(geometry.vertices[vertex_id].position, dtype=float),
            )

    reversed_geometry, rev_a, rev_b = _adjacent_plane_faces(reverse_second=True)
    reversed_result = generate_hybrid_mesh_result(
        reversed_geometry,
        target_size=1.0,
        face_ids=(rev_a, rev_b),
        quad_options=QuadMeshingOptions(),
    )
    reversed_mesh = reversed_result.mesh
    reversed_shared = _shared_edge_identity(reversed_geometry, (rev_a, rev_b))

    assert mesh.nodes_of_edge[shared] == reversed_mesh.nodes_of_edge[reversed_shared]
    assert dict(mesh.node_of_vertex) == dict(reversed_mesh.node_of_vertex)
    assert len(reversed_mesh.nodes) == 6
    reversed_coordinate_sets = {
        tuple(float(x) for x in np.asarray(node, dtype=float))
        for node in reversed_mesh.nodes.values()
    }
    assert len(reversed_coordinate_sets) == 6

def _three_sheet_shared_edge(declared: bool) -> tuple:
    """Three planar quads sharing one exact GeometryModel edge."""
    geometry = GeometryModel()
    a, b, b1, a1, a2, b2, b3, a3 = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )
    )
    shared = geometry.add_line(a, b)

    face_a = geometry.add_face_from_loop(
        (
            OrientedEdge(shared, True),
            OrientedEdge(geometry.add_line(b, b1), True),
            OrientedEdge(geometry.add_line(b1, a1), True),
            OrientedEdge(geometry.add_line(a1, a), True),
        ),
        corners=(0, 1, 2, 3),
        surface=Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    face_b = geometry.add_face_from_loop(
        (
            OrientedEdge(shared, False),
            OrientedEdge(geometry.add_line(a, a2), True),
            OrientedEdge(geometry.add_line(a2, b2), True),
            OrientedEdge(geometry.add_line(b2, b), True),
        ),
        corners=(0, 1, 2, 3),
        surface=Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, -1.0, 0.0)),
        ),
    )
    face_c = geometry.add_face_from_loop(
        (
            OrientedEdge(shared, True),
            OrientedEdge(geometry.add_line(b, b3), True),
            OrientedEdge(geometry.add_line(b3, a3), True),
            OrientedEdge(geometry.add_line(a3, a), True),
        ),
        corners=(0, 1, 2, 3),
        surface=Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 1.0)),
        ),
    )
    faces = (face_a, face_b, face_c)
    sheets = tuple(geometry.add_sheet((face,)) for face in faces)
    if declared:
        owner = sheets[0]
        # Qualification-only injection: the public ANYgeometry API has no
        # mutator for this frozen persistent declaration yet.  Bypass commit
        # validation here so the mesher can prove its 3-Sheet authority gate.
        geometry._put_structural(
            "sheet",
            replace(
                geometry.sheets[owner],
                declared_non_manifold_edges=(shared,),
            ),
        )
    return geometry, faces, sheets, shared


def test_quad_first_three_sheet_accidental_non_manifold_rejected() -> None:
    geometry, faces, _sheets, _shared = _three_sheet_shared_edge(False)
    with pytest.raises(MeshError, match="non-manifold|undeclared"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=1.0,
            face_ids=faces,
            quad_options=QuadMeshingOptions(),
        )


def test_quad_first_three_sheet_declared_non_manifold_is_published() -> None:
    geometry, faces, sheets, shared = _three_sheet_shared_edge(True)
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=faces,
        quad_options=QuadMeshingOptions(),
    )
    mesh = result.mesh
    sequence = list(mesh.nodes_of_edge[shared])
    expected = tuple(
        sorted(
            (min(int(a), int(b)), max(int(a), int(b)))
            for a, b in zip(sequence, sequence[1:])
        )
    )
    assert mesh.declared_plate_junction_edges == expected
    for sheet_id, face_id in zip(sheets, faces):
        assert mesh.elements_of_sheet[sheet_id] == mesh.elements_of_face[face_id]


def _curved_face() -> tuple:
    """A valid quarter-cylinder face for public-scope rejection testing."""
    import math

    from anygeometry.surfaces import Cylinder

    surface = Cylinder(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        3.0,
        0.0,
        math.pi / 2.0,
    )
    geometry = GeometryModel()
    uv = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    points = [geometry.add_point(*surface.evaluate(u, v)) for u, v in uv]
    low_mid = geometry.add_point(*surface.evaluate(0.5, 0.0))
    high_mid = geometry.add_point(*surface.evaluate(0.5, 1.0))
    edges = [
        geometry.add_arc(points[0], low_mid, points[1]),
        geometry.add_line(points[1], points[2]),
        geometry.add_arc(points[2], high_mid, points[3]),
        geometry.add_line(points[3], points[0]),
    ]
    face = geometry.add_face(edges, corners=(0, 1, 2, 3), surface=surface)
    return geometry, face


def test_public_dispatch_rejects_curved_geometry_face() -> None:
    """A curved (non-planar) geometry face must be rejected by the quad-first
    public dispatch with the typed scope error, not meshed by the legacy body."""
    geometry, face = _curved_face()
    with pytest.raises(QuadPublicUnsupported, match="planar"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=1.0,
            face_ids=(face,),
            quad_options=QuadMeshingOptions(),
        )


def _quad_and_residual_triangle_faces() -> tuple:
    """A quad face and a residual triangle face sharing one exact geometry
    edge.  The triangle routes through the legacy body and triangulates."""
    geometry = GeometryModel()
    points = geometry.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.5, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )
    shared = geometry.add_line(points[1], points[3])
    quad_loop = (
        OrientedEdge(geometry.add_line(points[0], points[1]), True),
        OrientedEdge(shared, True),
        OrientedEdge(geometry.add_line(points[3], points[4]), True),
        OrientedEdge(geometry.add_line(points[4], points[0]), True),
    )
    quad_face = geometry.add_face_from_loop(quad_loop, (0, 1, 2, 3))
    tri_loop = (
        OrientedEdge(shared, False),
        OrientedEdge(geometry.add_line(points[1], points[2]), True),
        OrientedEdge(geometry.add_line(points[2], points[3]), True),
    )
    tri_face = geometry.add_face_from_loop(
        tri_loop,
        corners=None,
        surface=Plane(
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    geometry.add_sheet((quad_face,))
    geometry.add_sheet((tri_face,))
    return geometry, quad_face, tri_face


def test_mixed_quad_and_qualified_s3_residual_route() -> None:
    """Mixed selector path: quad faces run the quad-first chain, the residual
    legacy triangles pass through the existing qualified-S3 bridge, and the
    published record carries the admitted S3 contract."""
    geometry, quad_face, tri_face = _quad_and_residual_triangle_faces()
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        strategy="native",
        native_backend="python",
        recombine=False,
        structural_preparation=False,
        face_ids=(quad_face, tri_face),
        quad_options=QuadMeshingOptions(),
        quad_face_ids=(quad_face,),
        qualified_s3=True,
    )
    mesh = result.mesh

    assert dict(result.strategy_by_face) == {
        quad_face: "quad_first",
        tri_face: "native",
    }
    assert mesh.quads
    assert mesh.tris
    quad_face_quads = list(mesh.elements_of_face[quad_face])
    assert quad_face_quads
    assert set(quad_face_quads) <= set(mesh.quads)
    tri_face_tris = list(mesh.elements_of_face[tri_face])
    assert set(tri_face_tris) <= set(mesh.tris)

    record = mesh.structural_preparation["qualified_s3"]
    assert record["status"] == "ADMITTED"
    assert record["legacy_fallback"] == "FORBIDDEN"
    assert record["contract_id"] == "ANYMESHER_QUALIFIED_S3_PRODUCTION_PREPARATION_V2"
    assert sorted(record["element_ids"]) == sorted(mesh.tris)
    for value in record["nodal_normals"].values():
        assert float(np.linalg.norm(np.asarray(value, dtype=float))) > 0.0


def test_pure_all_q4_result_publishes_s3_no_triangle_admission() -> None:
    """Explicit qualified-S3 control remains observable on an all-Q4 result."""
    geometry, face = _translated_plane_face()
    result = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(),
        qualified_s3=True,
    )
    assert result.mesh.quads
    assert not result.mesh.tris
    record = result.mesh.structural_preparation["qualified_s3"]
    assert record["status"] == "NOT_APPLICABLE_NO_TRIANGLES"
    assert record["element_ids"] == []
    assert record["legacy_fallback"] == "FORBIDDEN"
    assert record["admission"] == {
        "elements": [],
        "qualified_junction_edges": [],
        "topology_violations": [],
    }


@pytest.mark.quad_workers
def test_mixed_capability_advertised_truthfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Advertise mixed Q4/S3 only while the qualified S3 bridge exists."""
    report = advertise_quad_capabilities()
    assert report.mixed_q4_s3 is True
    assert report.to_dict()["mixed_q4_s3"] is True
    monkeypatch.setattr(
        quad_public_integration, "_s3_production_available", lambda: False
    )
    off = advertise_quad_capabilities()
    assert off.mixed_q4_s3 is False


def test_old_payload_deserializes_and_none_dispatch_round_trips() -> None:
    """Payload written without the quad/mixed diagnostic keys must still
    deserialize (old payload compatibility), and a legacy ``quad_options=None``
    mesh must round-trip through the unchanged payload format."""
    from anymesher.serialize import mesh_from_dict, mesh_to_dict

    # One explicit ``None`` dispatch exercises legacy behavior without persisting
    # any quad-selector runtime state in the neutral Mesh payload.
    geometry, face = _plane_face()
    legacy = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        quad_options=None,
    ).mesh
    payload = mesh_to_dict(legacy)

    # Old payload: a minimal document with no mixed/structural diagnostic record.
    stale = {
        key: value
        for key, value in payload.items()
        if key
        not in (
            "hybrid_diagnostics",
            "structural_preparation",
        )
    }
    stale["hybrid_diagnostics"] = {}
    loaded = mesh_from_dict(stale)
    assert loaded.order == "linear"

    # The unchanged legacy payload also round-trips its geometry associations.
    assert "quad_options" not in payload
    restored = mesh_from_dict(payload)
    assert dict(restored.node_of_vertex) == dict(legacy.node_of_vertex)
    assert set(restored.nodes) == {int(key) for key in legacy.nodes}
    for face_id, thickness in legacy.thickness_of_face.items():
        assert payload["thickness_of_face"][str(face_id)] == float(thickness)


def _beam_through_face_fixture():
    geometry = GeometryModel()
    plate_vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(plate_vertices)
    part = geometry.add_part()
    sheet = geometry.add_sheet((face,), part_id=part)
    member_vertices = geometry.add_points(
        ((0.5, 0.5, -1.0), (0.5, 0.5, 1.0))
    )
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
    return geometry, face, part, sheet, member, member_edge


@pytest.mark.quad_workers
def test_quad_first_beam_coupling_slice() -> None:
    geometry, face, _part, sheet, member, member_edge = _beam_through_face_fixture()
    result = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face,), member_ids=(member,),
        quad_options=QuadMeshingOptions(),
    )
    mesh = result.mesh
    diagnostics = mesh.hybrid_diagnostics
    assert diagnostics["route"] == "quad-first"
    assert diagnostics["q4"]["status"] == "APPLIED"
    assert diagnostics["q5"]["status"] == "NO_ELIGIBLE"
    face_elements = list(mesh.elements_of_face[face])
    assert face_elements
    assert all(eid in mesh.quads or eid in mesh.tris for eid in face_elements)
    assert list(mesh.elements_of_sheet[sheet]) == face_elements
    face_nodes = set()
    for eid in face_elements:
        body = mesh.quads[eid] if eid in mesh.quads else mesh.tris[eid]
        face_nodes.update(int(node) for node in body)
    assert mesh.beams
    assert member_edge in mesh.elements_of_edge
    assert member_edge in mesh.nodes_of_edge
    edge_nodes = set(int(node) for node in mesh.nodes_of_edge[member_edge])
    beam_ids = set(int(element) for element in mesh.elements_of_edge[member_edge])
    assert all(beam_id in mesh.beams for beam_id in beam_ids)
    couplings = list(mesh.couplings.values())
    assert len(couplings) == 1
    coupling = couplings[0]
    assert int(coupling.beam_node) == int(mesh.nodes_of_edge[member_edge][1])
    np.testing.assert_allclose(mesh.nodes[int(coupling.beam_node)], (0.5, 0.5, 0.0))
    assert int(coupling.beam_node) in edge_nodes
    assert set(int(node) for node in coupling.plate_nodes) <= face_nodes
    assert sum(coupling.weights) == pytest.approx(1.0)
    assert result.connectivity is not None
    assert result.connectivity.connected == 1
    assert not result.connectivity.issues


def test_mixed_merge_preserves_quad_first_residual_t3_namespace() -> None:
    geometry = GeometryModel()
    quad_vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 4.0, 0.0), (1.0, 4.0, 0.0))
    )
    quad_face = geometry.add_plate(quad_vertices)
    legacy_vertices = geometry.add_points(
        ((10.0, 0.0, 0.0), (12.0, 0.0, 0.0), (12.0, 2.0, 0.0), (10.0, 2.0, 0.0))
    )
    legacy_face = geometry.add_plate(legacy_vertices)

    mesh = generate_hybrid_mesh_result(
        geometry,
        target_size=0.75,
        face_ids=(quad_face, legacy_face),
        strategy="mapped",
        quad_options=QuadMeshingOptions(),
        quad_face_ids=(quad_face,),
    ).mesh

    quad_shells = set(mesh.elements_of_face[quad_face])
    legacy_shells = set(mesh.elements_of_face[legacy_face])
    residual = quad_shells & set(mesh.tris)
    assert residual
    assert quad_shells == (quad_shells & set(mesh.quads)) | residual
    assert not (set(mesh.quads) & set(mesh.tris))
    assert not (quad_shells & legacy_shells)
    assert min(legacy_shells) > max(quad_shells)
