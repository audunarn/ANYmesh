"""Nonperiodic owner certificates and public partial-cylinder refinement."""

from contextlib import ExitStack
from dataclasses import replace
import json

import numpy as np
import pytest

from anygeometry import CylinderAtlasError, CylinderPatchError, CylinderPatchErrorCode
from anymesher import FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec
from anymesher import NativeMeshingOptions, hybrid
from anymesher._cylindrical_atlas import prepare_cylindrical_atlas
from anymesher._cylindrical_patch import prepare_cylindrical_patch
from anymesher._cylindrical_public import component_local_split_edges, prepare_bindings
from anymesher.quality_v2 import assert_valid_mesh
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state


@pytest.mark.parametrize("hole", (False, True))
def test_single_patch_keeps_owner_identity_and_physical_chart(hole):
    model, selected = _sector_model(hole=hole)
    before = _persistent_state(model)
    binding = prepare_cylindrical_patch(model, selected[:1])
    assert binding.face_uses == selected[:1]
    assert binding.face_records == (binding.patch,)
    assert binding.patch.certificate.complete
    assert binding.certification_kind == "nonperiodic_owner_patch"
    assert not hasattr(binding, "atlas")
    chart = binding.chart_for(selected[0])
    assert chart.circumferential_length == pytest.approx(np.pi / 4)
    assert chart.axial_length == 2.
    assert any(row.external_coedges for row in binding.patch.occurrences)
    assert _persistent_state(model) == before
    with pytest.raises(CylinderPatchError):
        replace(binding, face_uses=selected[1:2]).validate()


@pytest.mark.parametrize("open_transaction", (False, True))
def test_patch_final_callback_cannot_publish_stale_or_busy_binding(open_transaction):
    model, selected = _sector_model()
    with ExitStack() as transactions:
        def callback(phase):
            if phase == "cylindrical patch preparation complete":
                if open_transaction:
                    transactions.enter_context(model.transaction())
                else:
                    with model.transaction():
                        model.add_point(10., 10., 10.)

        with pytest.raises(CylinderPatchError) as caught:
            prepare_cylindrical_patch(model, selected[:1], cancellation_check=callback)
        assert caught.value.code is (CylinderPatchErrorCode.BUSY_MODEL if open_transaction
                                     else CylinderPatchErrorCode.STALE_REVISION)


def test_patch_staleness_and_cancellation_propagate():
    model, selected = _sector_model()
    binding = prepare_cylindrical_patch(model, selected[:1])
    error = RuntimeError("cancel patch")
    def cancel(phase):
        raise error
    with pytest.raises(RuntimeError) as caught:
        binding.validate(cancellation_check=cancel)
    assert caught.value is error
    with model.transaction():
        model.add_point(10., 10., 10.)
    with pytest.raises(CylinderPatchError):
        binding.chart_for(selected[0])


def test_atlas_failure_is_not_relabelled_as_partial_patch(monkeypatch):
    model, selected = _sector_model()
    faces = tuple(model.face_uses[item.id].face_id for item in selected[:4])
    def forbidden(*args, **kwargs):
        pytest.fail("incomplete atlas cannot silently fall back to per-face patches")
    monkeypatch.setattr("anymesher._cylindrical_public.prepare_cylindrical_patch", forbidden)
    with pytest.raises(CylinderAtlasError):
        prepare_bindings(model, faces, NativeMeshingOptions(point_placement="frontal_delaunay"))


def test_external_shared_edges_are_frozen_for_every_incident_face():
    model, selected = _sector_model()
    face = model.face_uses[selected[0].id].face_id
    patch = prepare_cylindrical_patch(model, selected[:1])
    atlas = prepare_cylindrical_atlas(model, selected, reference_face_use=selected[0])
    shared = frozenset(e for e in model.edges if len(model.faces_using_edge(e)) == 2)
    filtered = component_local_split_edges(model, shared, {face: patch})
    assert filtered == frozenset(e for e in shared if face not in model.faces_using_edge(e))
    assert component_local_split_edges(
        model, shared, {row.face.id: atlas for row in atlas.face_records}) == shared
    assert component_local_split_edges(model, shared, {}) is shared


def test_public_patch_owner_operational_failure_precedes_generation(monkeypatch):
    model, selected = _sector_model()
    before = _persistent_state(model)
    error = RuntimeError("owner patch operational failure")
    def failed(*args, **kwargs):
        raise error
    def forbidden(*args, **kwargs):
        pytest.fail("failed owner proof must not invoke native generation")
    monkeypatch.setattr("anymesher._cylindrical_patch.query_cylinder_patch", failed)
    monkeypatch.setattr(hybrid, "_mesh_native_face", forbidden)
    with pytest.raises(RuntimeError) as caught:
        hybrid.generate_hybrid_mesh_result(
            model, face_ids=(model.face_uses[selected[0].id].face_id,),
            target_size=.4, strategy="native", native_backend="python", recombine=False,
            native_options=NativeMeshingOptions(point_placement="frontal_delaunay"))
    assert caught.value is error
    assert _persistent_state(model) == before


@pytest.mark.parametrize("order", ("linear", "quadratic"))
@pytest.mark.parametrize("recombine", (False, True))
@pytest.mark.parametrize("hole", (False, True), ids=("patch", "notched-patch"))
def test_public_partial_patch_preserves_seed_topology(order, recombine, hole, monkeypatch, tmp_path):
    model, selected = _sector_model(hole=hole)
    before = _persistent_state(model)
    face = model.face_uses[selected[0].id].face_id
    center = (float(np.cos(np.pi / 8)), float(np.sin(np.pi / 8)), 1.)
    options = NativeMeshingOptions(
        point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
        metric_field=MetricFieldSpec(IsotropicMetricControl(.4), feature_controls=(
            FeatureDistanceMetricControl((center,), .12, .35, 1.5, "patch-interior"),)),
        max_insertions=128, max_topology_operations=20000,
    )
    protected = {}
    sequences = {}
    original = hybrid._mesh_native_face
    def observe(geometry, mesh, face_id, **kwargs):
        assert face_id == face
        for loop in (geometry.faces[face_id].loop, *geometry.faces[face_id].holes):
            for use in loop:
                sequences[use.edge] = tuple(mesh.nodes_of_edge[use.edge])
                for node in sequences[use.edge]:
                    protected[node] = np.asarray(mesh.nodes[node]).tobytes()
        return original(geometry, mesh, face_id, **kwargs)
    monkeypatch.setattr(hybrid, "_mesh_native_face", observe)
    result = hybrid.generate_hybrid_mesh_result(
        model, face_ids=(face,), target_size=.4, strategy="native", native_backend="python",
        order=order, recombine=recombine, native_options=options,
    )
    mesh = result.mesh
    assert _persistent_state(model) == before
    assert set(mesh.elements_of_face) == {face}
    assert result.strategy_by_face == {face: "native"}
    assert mesh.tris and bool(mesh.quads) is recombine
    assert all(len(nodes) == (6 if order == "quadratic" else 3) for nodes in mesh.tris.values())
    assert all(len(nodes) == (8 if order == "quadratic" else 4) for nodes in mesh.quads.values())
    assert all(np.asarray(mesh.nodes[node]).tobytes() == raw for node, raw in protected.items())
    assert all(tuple(mesh.nodes_of_edge[edge]) == nodes for edge, nodes in sequences.items())
    assert_valid_mesh(hybrid._neutral_shell_core(mesh))
    used = sorted({node for nodes in mesh.shells.values() for node in nodes})
    _, _, residual = model.project_to_face_many(face, [mesh.nodes[node] for node in used])
    assert np.max(residual) <= model.tolerance.effective_length(2.)
    diagnostics = result.triangulation_backend_by_face[face]
    if order == "quadratic":
        receipt = diagnostics["cylindrical_component"]
        assert receipt["owner_contract"] == "nonperiodic_owner_patch"
        assert receipt["accepted"] and receipt["quality"]["accepted"]
        assert receipt["published_insertions"] > 0
        midsides = {}
        for element, nodes in mesh.shells.items():
            count = 3 if element in mesh.tris else 4
            for index in range(count):
                pair = tuple(sorted((nodes[index], nodes[(index + 1) % count])))
                assert pair not in midsides or midsides[pair] == nodes[count + index]
                midsides[pair] = nodes[count + index]
        for sequence in sequences.values():
            for a, middle, b in zip(sequence[::2], sequence[1::2], sequence[2::2]):
                assert midsides[tuple(sorted((a, b)))] == middle
    else:
        receipt = diagnostics["cylindrical_frontal_delaunay"]
        assert receipt["chart"] == "owner_certified_patch_physical_lengths"
        assert receipt["periodic_topology"] == "nonperiodic_owner_patch"
    (tmp_path / "diagnostics.json").write_text(
        json.dumps(diagnostics, sort_keys=True, default=str), encoding="utf-8")
