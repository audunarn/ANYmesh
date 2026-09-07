"""Test-only review of committed junction authority and selected S3 fallbacks.

The fault-injection tests assert publication contracts, not source spellings.
They deliberately do not import the pending ANYgeometry trim-domain API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib

import numpy as np
import pytest

from anygeometry import EntityRef, GeometryModel, OrientedEdge, Plane
from anygeometry.serialization import to_dict
from anymesher import Mesh, MeshError, evaluate_s3_admission
from anymesher.preparation import prepare_structural_closure
from anymesher.prepared import remap_prepared_mesh_associations
from anymesher.serialize import mesh_from_dict, mesh_to_dict
from test_cylinder_plate_quality_regression import _cylinder_through_plate


hybrid = importlib.import_module("anymesher.hybrid")
production = importlib.import_module("anymesher.s3_production")

POLICY = {
    "minimum_scaled_jacobian": 0.1,
    "maximum_aspect_ratio": 5.0,
    "minimum_angle": 20.0,
    "maximum_angle": 160.0,
    "maximum_warpage": 0.1,
}


def _connectivity(mesh: Mesh) -> tuple:
    return (
        tuple((key, np.asarray(value, dtype=np.float64).tobytes())
              for key, value in sorted(mesh.nodes.items())),
        tuple((key, tuple(value)) for key, value in sorted(mesh.tris.items())),
        tuple((key, tuple(value)) for key, value in sorted(mesh.quads.items())),
    )


def _square(geometry: GeometryModel, x: float = 0.0) -> int:
    return geometry.add_plate(geometry.add_points(
        ((x, 0., 0.), (x + 1., 0., 0.),
         (x + 1., 1., 0.), (x, 1., 0.))
    ))


def _balanced_junction() -> Mesh:
    return Mesh(
        nodes={key: np.asarray(value, dtype=float) for key, value in {
            1: (0., 0., 0.), 2: (1., 0., 0.),
            3: (0., 1., 0.), 4: (1., -1., 0.),
            5: (0., 0., 1.), 6: (1., 0., -1.),
            7: (1., 0., 1.), 8: (0., 0., -1.),
        }.items()},
        tris={10: (1, 2, 3), 20: (2, 1, 4)},
        quads={30: (1, 2, 7, 5), 40: (2, 1, 8, 6)},
        elements_of_sheet={101: [10, 20], 202: [30, 40]},
        declared_plate_junction_edges=((1, 2),),
    )


def _cylinder_closure():
    source = _cylinder_through_plate()
    source.add_sheet((max(source.faces),), name="deck")
    before = to_dict(source)
    working, report = prepare_structural_closure(
        source, face_ids=tuple(source.faces), beam_edges=(), options=True,
    )
    assert report is not None
    assert to_dict(source) == before
    return source, working, report, before


def _generate(working, target_size=0.20, **extra):
    return hybrid.generate_hybrid_mesh_result(
        working,
        target_size=target_size,
        strategy="auto",
        beam_edges=(),
        member_ids=(),
        face_ids=tuple(working.faces),
        structured_options={"quality_policy": POLICY},
        structural_preparation={
            "automatic_face_connections": False,
            "automatic_member_connections": False,
            "automatic_member_sheet_connections": False,
            "declare_missing_owners": True,
        },
        mutation_policy="working_copy",
        native_backend="python",
        qualified_s3=True,
        **extra,
    )


def test_unrelated_sheet_incidence_does_not_authorize_other_edges():
    geometry = GeometryModel()
    first = _square(geometry)
    geometry.add_sheet((first,))
    first_edges = set(geometry.edges)
    with geometry.transaction():
        a, b, c, d, e, f = geometry.add_points(
            ((3., 0., 0.), (4., 0., 0.), (4., 1., 0.),
             (3., 1., 0.), (3., 0., 1.), (4., 0., 1.))
        )
        shared = geometry.add_line(a, b)
        bc = geometry.add_line(b, c)
        cd = geometry.add_line(c, d)
        da = geometry.add_line(d, a)
        ae = geometry.add_line(a, e)
        ef = geometry.add_line(e, f)
        fb = geometry.add_line(f, b)
        other = geometry.add_face_from_loop(
            tuple(OrientedEdge(edge, True) for edge in (shared, bc, cd, da)),
            surface=Plane((3., 0., 0.), (1., 0., 0.), (0., 1., 0.)),
        )
        wall = geometry.add_face_from_loop(
            (OrientedEdge(shared, False), OrientedEdge(ae, True),
             OrientedEdge(ef, True), OrientedEdge(fb, True)),
            surface=Plane((4., 0., 0.), (-1., 0., 0.), (0., 0., 1.)),
        )
        other_sheet = geometry.add_sheet((other,))
        wall_sheet = geometry.add_sheet((wall,))
    assert other != wall and other_sheet != wall_sheet
    assert set(geometry.faces_using_edge(shared)) == {other, wall}
    assert set(geometry.sheets_using_edge(shared)) == {other_sheet, wall_sheet}
    assert len(geometry.face_uses_using_edge(shared)) == 2
    assert len(geometry.coedges_using_edge(shared)) == 2
    assert geometry.validate_topology() == ()
    before = to_dict(geometry)

    recovered = set(hybrid._geometry_plate_junction_edge_ids(geometry))

    assert recovered == {shared}
    assert recovered.isdisjoint(first_edges)
    assert all(len(geometry.sheets_using_edge(edge)) == 2 for edge in recovered)
    assert to_dict(geometry) == before
    mesh = _balanced_junction()
    mesh.declared_plate_junction_edges = ((5, 7),)
    rejected = evaluate_s3_admission(
        mesh, element_owner_normals={10: (0., 0., 1.), 20: (0., 0., 1.)},
    )
    assert not rejected.admitted
    assert (1, 2) not in rejected.qualified_junction_edges


def test_same_orientation_duplicate_owner_does_not_qualify_junction():
    mesh = _balanced_junction()
    normals = {10: (0., 0., 1.), 20: (0., 0., 1.)}
    assert evaluate_s3_admission(mesh, element_owner_normals=normals).admitted
    mesh.elements_of_sheet[303] = [10, 20]
    before = deepcopy(mesh_to_dict(mesh))

    report = evaluate_s3_admission(mesh, element_owner_normals=normals)

    assert not report.admitted
    assert report.topology_violations
    assert report.qualified_junction_edges == ()
    assert mesh_to_dict(mesh) == before


def test_cancellation_after_local_s3_repair_preserves_input(monkeypatch):
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0., 0., 0.), (1., 0., 0.), (1., 2., 0.), (0., 1., 0.))
    )
    face = geometry.add_plate(vertices)
    sheet = geometry.add_sheet((face,))
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={key: np.asarray(geometry.vertex_position(key)) for key in vertices},
        tris={10: (vertices[0], vertices[1], vertices[2]),
              20: (vertices[0], vertices[2], vertices[3])},
        elements_of_face={face: [10, 20]},
        elements_of_sheet={sheet: [10, 20]},
    )
    before = deepcopy(mesh_to_dict(mesh))
    geometry_before = to_dict(geometry)
    real_repair = production.repair_s3_admission
    cancelled = KeyboardInterrupt("junction review: cancel after detached repair")
    attempts = []

    def interrupted(*args, **kwargs):
        result = real_repair(*args, **kwargs)
        attempts.append(result)
        assert result.edge_flips == 1
        assert result.mesh is not args[0]
        raise cancelled

    monkeypatch.setattr(production, "repair_s3_admission", interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        production.prepare_qualified_s3_mesh(mesh, geometry)

    assert caught.value is cancelled
    assert len(attempts) == 1
    assert mesh_to_dict(mesh) == before
    assert to_dict(geometry) == geometry_before


def test_source_working_remap_preserves_exact_junction_identity():
    source, working, preparation, source_before = _cylinder_closure()
    working_before = to_dict(working)
    result = _generate(working, target_size=0.25)
    mesh = result.mesh
    before = _connectivity(mesh)
    junctions = tuple(mesh.declared_plate_junction_edges)
    assert junctions
    edge_ids = tuple(preparation.declared_face_connection_edges)
    assert edge_ids
    exact_edge_nodes = {
        edge: tuple(mesh.nodes_of_edge[edge]) for edge in edge_ids
    }
    for edge, nodes in exact_edge_nodes.items():
        assert edge in source.edges and edge in working.edges
        assert source.edges[edge] == working.edges[edge]
        assert len(nodes) >= 2
        assert all(tuple(sorted(pair)) in junctions for pair in zip(nodes, nodes[1:]))
    face_map = {
        face: tuple(ref.id for ref in working.resolve_ref(EntityRef("face", face)))
        for face in source.faces
    }
    edge_map = {
        edge: tuple(ref.id for ref in working.resolve_ref(EntityRef("edge", edge)))
        for edge in source.edges
    }

    remapped = remap_prepared_mesh_associations(
        mesh, source, working,
        source_to_working_faces=face_map,
        source_to_working_edges=edge_map,
    )

    assert _connectivity(remapped) == before
    assert tuple(remapped.declared_plate_junction_edges) == junctions
    assert remapped.geometry_model_id == source.model_id
    assert remapped.geometry_revision == source.revision
    for edge, nodes in exact_edge_nodes.items():
        assert tuple(remapped.nodes_of_edge[edge]) == nodes
    assert to_dict(source) == source_before
    assert to_dict(working) == working_before


def test_selected_fallback_does_not_publish_stale_nonnull_admission(monkeypatch):
    source, working, _report, source_before = _cylinder_closure()
    working_before = to_dict(working)
    real_generate = hybrid.generate_hybrid_mesh_result
    injected = []

    def stale_recursive_candidate(*args, **kwargs):
        result = real_generate(*args, **kwargs)
        if result.mesh.tris:
            record = deepcopy(result.mesh.structural_preparation.get("qualified_s3"))
            if record is not None:
                record["element_ids"] = [-999999]
                record["authority_model"]["source_revision"] = -1
                record["junction_review_stale"] = True
                result.mesh.structural_preparation["qualified_s3"] = record
                injected.append(_connectivity(result.mesh))
        return result

    # Capture the outer callable before replacing only recursive candidate calls.
    monkeypatch.setattr(hybrid, "generate_hybrid_mesh_result", stale_recursive_candidate)
    try:
        with monkeypatch.context() as outer:
            # _generate resolves the module attribute; invoke the saved callable
            # for the outer call while preserving patched recursive dispatch.
            outer.setattr(hybrid, "generate_hybrid_mesh_result", real_generate)
            def dispatch(*args, **kwargs):
                outer.setattr(hybrid, "generate_hybrid_mesh_result", stale_recursive_candidate)
                return real_generate(*args, **kwargs)
            outer.setattr(hybrid, "generate_hybrid_mesh_result", dispatch)
            result = _generate(working)
    except MeshError:
        assert injected, "stale-candidate branch was not exercised"
    else:
        assert injected, "stale-candidate branch was not exercised"
        record = result.mesh.structural_preparation["qualified_s3"]
        assert not record.get("junction_review_stale", False)
        assert record["element_ids"] == sorted(result.mesh.tris)
        assert record["authority_model"]["source_revision"] == working.revision
    finally:
        assert to_dict(source) == source_before
        assert to_dict(working) == working_before


def test_selected_repaired_topology_receives_whole_mesh_policy_check(monkeypatch):
    source, working, _report, source_before = _cylinder_closure()
    working_before = to_dict(working)
    real_prepare = hybrid.prepare_qualified_s3_mesh
    real_quality = hybrid._structured_quality_report
    state = {"selected": False, "attempted": False}
    repaired = []
    checked = []

    def phase(label):
        if label == "selected fallback qualified S3 preparation start":
            state["selected"] = True

    def prepare(mesh, geometry, **kwargs):
        if not state["selected"] or state["attempted"]:
            return real_prepare(mesh, geometry, **kwargs)
        state["attempted"] = True
        trial = mesh_from_dict(mesh_to_dict(mesh))
        quad_id = next(key for key, row in sorted(trial.quads.items()) if len(row) == 4)
        a, b, c, d = trial.quads.pop(quad_id)
        new_id = max(trial.shells, default=quad_id) + 1
        new_id = max(new_id, quad_id + 1)
        trial.tris[quad_id] = (a, b, c)
        trial.tris[new_id] = (a, c, d)
        for associations in (trial.elements_of_face, trial.elements_of_sheet):
            for members in associations.values():
                if quad_id in members:
                    members.append(new_id)
                    members.sort()
        if quad_id in trial.activity:
            trial.activity[new_id] = trial.activity[quad_id]
        result, record = real_prepare(trial, geometry, **kwargs)
        assert _connectivity(result) != _connectivity(mesh)
        repaired.append(_connectivity(result))
        return result, record

    def quality(*args, **kwargs):
        for value in (*args, *kwargs.values()):
            if isinstance(value, Mesh):
                checked.append(_connectivity(value))
        return real_quality(*args, **kwargs)

    monkeypatch.setattr(hybrid, "prepare_qualified_s3_mesh", prepare)
    monkeypatch.setattr(hybrid, "_structured_quality_report", quality)
    try:
        result = _generate(working, cancellation_check=phase)
        assert repaired, "no admitted topology-changing selected repair was exercised"
        assert repaired[-1] in checked, "repaired mesh escaped whole-mesh policy evaluation"
        record = result.mesh.structural_preparation["qualified_s3"]
        assert record["element_ids"] == sorted(result.mesh.tris)
        assert record["admission"]["topology_violations"] == []
        options = result.structured_layout.plan.options
        final_quality = real_quality(result.mesh, options)
        assert final_quality["accepted"]
        assert result.structured_layout.to_dict()["quality"]["accepted_fallback"] == final_quality
        assert result.mesh.hybrid_diagnostics["structured_quality"][
            "accepted_fallback"
        ] == final_quality
        assert result.structured_layout.to_dict()["metrics"]["accepted_fallback"] == (
            hybrid.regularity_metrics(
                result.mesh,
                target_size=0.20,
                minimum_size_ratio=options.minimum_size_ratio,
                maximum_size_ratio=options.maximum_size_ratio,
                mapped_element_ids=(),
            )
        )
        assert set(map(int, record["element_owner_sources"])) == set(result.mesh.shells)
        assert set(map(int, record["nodal_normals"])) == {
            node for row in result.mesh.shells.values() for node in row
        }
    finally:
        assert to_dict(source) == source_before
        assert to_dict(working) == working_before


@pytest.mark.parametrize("growth_limit,accepted", ((1.2, False), (2.0, True)))
def test_real_s3_flip_obeys_requested_growth_at_public_fallback(
    monkeypatch, growth_limit, accepted,
):
    """Force only route selection; all numerical quality/repair checks are real."""
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0., 0., 0.), (1., 0., 0.), (1., 1.8, 0.), (0., 1., 0.))
    )
    face = geometry.add_plate(vertices)
    sheet = geometry.add_sheet((face,))
    source_before = to_dict(geometry)
    patch = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={key: np.asarray(geometry.vertex_position(key)) for key in vertices},
        tris={10: (vertices[0], vertices[1], vertices[2]),
              20: (vertices[0], vertices[2], vertices[3])},
        elements_of_face={face: [10, 20]},
        elements_of_sheet={sheet: [10, 20]},
    )
    patch_before = deepcopy(mesh_to_dict(patch))
    options = hybrid.StructuredMeshingOptions(
        quality_policy=hybrid.MeshQualityPolicy(**POLICY),
        max_element_growth=growth_limit,
    )
    real_generate = hybrid.generate_hybrid_mesh_result
    real_quality = hybrid._structured_quality_report
    real_prepare = hybrid.prepare_qualified_s3_mesh
    before_quality = real_quality(patch, options)
    repaired_patch, repaired_record = real_prepare(patch, geometry)
    after_quality = real_quality(repaired_patch, options)
    assert before_quality["accepted"]
    assert repaired_record["status"] == "ADMITTED"
    assert repaired_record["repair"]["edge_flips"] == 1
    assert _connectivity(repaired_patch) != _connectivity(patch)
    assert after_quality["accepted"] is accepted
    assert not any(after_quality["violation_counts"].values())
    if not accepted:
        assert after_quality["growth_violation_count"] == 1
        assert after_quality["maximum_adjacent_element_growth"] > (
            after_quality["effective_growth_limit"] + 1.e-14
        )
        assert {item for pair in after_quality["growth_violation_pairs"]
                for item in pair[:2]} == {10, 20}

    template = real_generate(
        geometry, target_size=10., strategy="native", native_backend="python",
        structural_preparation=False, recombine=False, qualified_s3=False,
    )
    state = {"routed": False, "late": False}
    late_records = []
    checks = []

    def candidate(*args, **kwargs):
        return replace(template, mesh=mesh_from_dict(deepcopy(patch_before)))

    def route_and_measure(mesh, selected_options):
        measured = real_quality(mesh, selected_options)
        if mesh.quads and not mesh.tris:
            state["routed"] = True
            # This route-control fault is not the numerical rejection oracle.
            # Both selected-patch and post-repair reports below are unmodified.
            return {**measured, "accepted": False,
                    "test_route": "exercise selected native fallback"}
        checks.append((_connectivity(mesh), measured))
        return measured

    def phase(label):
        if label == "selected fallback qualified S3 preparation start":
            state["late"] = True

    def prepare(mesh, owner, **kwargs):
        made, record = real_prepare(mesh, owner, **kwargs)
        if state["late"]:
            late_records.append(record)
        return made, record

    monkeypatch.setattr(hybrid, "generate_hybrid_mesh_result", candidate)
    monkeypatch.setattr(hybrid, "_structured_quality_report", route_and_measure)
    monkeypatch.setattr(hybrid, "prepare_qualified_s3_mesh", prepare)
    call_options = dict(
        target_size=10., strategy="auto", native_backend="python",
        beam_edges=(), member_ids=(), structural_preparation=False,
        qualified_s3=True, cancellation_check=phase,
        structured_options={"quality_policy": POLICY,
                            "max_element_growth": growth_limit},
    )
    caught_error = None
    result = None
    try:
        result = real_generate(geometry, **call_options)
    except MeshError as error:
        caught_error = error
    assert state["late"], "the intended late-S3 fallback branch was not entered"
    if accepted:
        assert caught_error is None, str(caught_error)
        assert result is not None
        assert result.structured_layout.to_dict()["quality"]["accepted_fallback"] == after_quality
        assert result.structured_layout.to_dict()["quality"]["pre_s3_fallback"] == before_quality
        assert _connectivity(result.mesh) == _connectivity(repaired_patch)
        record = result.mesh.structural_preparation["qualified_s3"]
        assert record["element_ids"] == sorted(result.mesh.tris)
        assert {item["element_id"] for item in record["admission"]["elements"]} == (
            set(result.mesh.tris)
        )
    else:
        assert caught_error is not None, "numerically rejected repaired mesh was published"
        assert "after qualified S3 preparation" in str(caught_error)
    assert len(late_records) == 1
    assert late_records[0]["repair"]["edge_flips"] == 1
    assert (_connectivity(repaired_patch), after_quality) in checks
    assert mesh_to_dict(patch) == patch_before
    assert to_dict(geometry) == source_before
