"""Real hybrid seed and owner chart trial, not a public activation claim."""

import json
import pickle

import numpy as np
import pytest

from anymesher import hybrid, NativeMeshingOptions, MetricFieldSpec, IsotropicMetricControl, FeatureDistanceMetricControl
from anymesher._cylindrical_atlas import prepare_cylindrical_atlas
from anymesher._cylindrical_quadratic_refine import refine_quadratic_component
from anymesher.surface_mesh import SurfaceMeshOptions
from anymesher import native_v2, native_cpp
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state


@pytest.mark.parametrize("recombine", (False, True), ids=("T6", "T6-Q8"))
@pytest.mark.parametrize("hole", (False, True), ids=("closed", "seam-trim"))
def test_work_bearing_quadratic_trial_from_real_hybrid_seed(monkeypatch, tmp_path, recombine, hole):
    model, _ = _sector_model(hole=hole)
    before = _persistent_state(model)
    original = hybrid._mesh_native_face
    captured = []
    centers = tuple((float(np.cos((i + .5) * np.pi / 4)),
                     float(np.sin((i + .5) * np.pi / 4)), 1.) for i in range(8))
    options = NativeMeshingOptions(
        point_placement="frontal_delaunay", metric_mode="isotropic_spatial",
        metric_field=MetricFieldSpec(IsotropicMetricControl(.4), feature_controls=tuple(
            FeatureDistanceMetricControl((p,), .12, .35, 1.5, f"interior-{i}")
            for i, p in enumerate(centers))), max_insertions=128, max_topology_operations=20000)

    def seeded(geometry, mesh, face, **kwargs):
        result = original(geometry, mesh, face, **{**kwargs, "native_options": NativeMeshingOptions()})
        if set(mesh.elements_of_face) == set(geometry.faces):
            selected = tuple(geometry.handle("face_use", i) for i in sorted(geometry.face_uses))
            binding = prepare_cylindrical_atlas(geometry, selected, reference_face_use=selected[0])
            from anygeometry.curves import Straight
            eligible = tuple(e for e, edge in geometry.edges.items()
                             if isinstance(edge.curve, Straight) and len(geometry.faces_using_edge(e)) == 2)
            seed_bytes = pickle.dumps(mesh.__dict__, protocol=5)
            protected = {n: mesh.nodes[n].tobytes() for sequence in mesh.nodes_of_edge.values() for n in sequence}
            trial = refine_quadratic_component(
                geometry, mesh, binding, kwargs["boundary_registry"].entries(),
                SurfaceMeshOptions(order="quadratic", recombine=recombine, target_size=.4,
                                   native_options=options, prefer_quality_policy=True, enforce_quality=True),
                eligible_edges=eligible, metric_model_uuid=kwargs["metric_model_uuid"],
                metric_geometry_revision=kwargs["metric_geometry_revision"])
            (tmp_path / "trial.json").write_text(json.dumps(trial.diagnostics, sort_keys=True, default=str), encoding="utf-8")
            assert pickle.dumps(mesh.__dict__, protocol=5) == seed_bytes
            if native_cpp._complete_native_v2_available():
                with monkeypatch.context() as reference:
                    reference.setattr(native_v2, "native_mutable_t3_insert", lambda *args, **kw: None)
                    oracle = refine_quadratic_component(
                        geometry, mesh, binding, kwargs["boundary_registry"].entries(),
                        SurfaceMeshOptions(order="quadratic", recombine=recombine, target_size=.4,
                                           native_options=options, prefer_quality_policy=True, enforce_quality=True),
                        eligible_edges=eligible, metric_model_uuid=kwargs["metric_model_uuid"],
                        metric_geometry_revision=kwargs["metric_geometry_revision"])
                (tmp_path / "oracle.json").write_text(json.dumps(oracle.diagnostics, sort_keys=True, default=str), encoding="utf-8")
                assert pickle.dumps(mesh.__dict__, protocol=5) == seed_bytes
                assert oracle.accepted is trial.accepted
                assert oracle.mesh.nodes.keys() == trial.mesh.nodes.keys()
                assert all(oracle.mesh.nodes[n].tobytes() == p.tobytes() for n, p in trial.mesh.nodes.items())
                for name in ("tris", "quads", "nodes_of_edge", "elements_of_face"):
                    assert getattr(oracle.mesh, name) == getattr(trial.mesh, name)
                assert oracle.diagnostics == trial.diagnostics
                (tmp_path / "insertion_parity.json").write_text(json.dumps({
                    "compiled_insertion_vs_python_oracle": "exact",
                    "coordinates": "byte-exact", "connectivity": "exact", "diagnostics": "exact",
                    "other_kernels": "unchanged; not an all-kernel Python-only claim",
                }, sort_keys=True), encoding="utf-8")
            if trial.accepted:
                assert all(trial.mesh.nodes[n].tobytes() == raw for n, raw in protected.items())
                for edge, sequence in mesh.nodes_of_edge.items():
                    final = trial.mesh.nodes_of_edge[edge]
                    positions = [final.index(node) for node in sequence]
                    assert positions == sorted(positions)
                    for face in geometry.faces_using_edge(edge):
                        edge_midsides = {}
                        for element in trial.mesh.elements_of_face[face]:
                            nodes = trial.mesh.shells[element]
                            count = 3 if element in trial.mesh.tris else 4
                            for i in range(count):
                                pair = tuple(sorted((nodes[i], nodes[(i + 1) % count])))
                                midpoint = nodes[count + i]
                                assert pair not in edge_midsides or edge_midsides[pair] == midpoint
                                edge_midsides[pair] = midpoint
                        for i in range(0, len(final) - 2, 2):
                            a, m, b = final[i:i + 3]
                            assert edge_midsides[tuple(sorted((a, b)))] == m
                for face, elements in trial.mesh.elements_of_face.items():
                    nodes = sorted({n for e in elements for n in trial.mesh.shells[e]})
                    _, _, residual = geometry.project_to_face_many(face, [trial.mesh.nodes[n] for n in nodes])
                    assert np.all(np.isfinite(residual)) and np.max(residual) <= geometry.tolerance.effective_length(2.)
                (tmp_path / "counts.json").write_text(json.dumps({
                    "nodes": len(trial.mesh.nodes), "T6": len(trial.mesh.tris), "Q8": len(trial.mesh.quads),
                    "protected_nodes": len(protected), "protected_bytes": "exact",
                    "shared_midsides": "exact", "source_seed": "unchanged",
                }, sort_keys=True), encoding="utf-8")
            captured.append(trial)
        return result

    monkeypatch.setattr(hybrid, "_mesh_native_face", seeded)
    hybrid.generate_hybrid_mesh_result(model, target_size=.4, order="quadratic", recombine=False,
                                      strategy="native", native_backend="python", native_options=NativeMeshingOptions())
    assert _persistent_state(model) == before
    assert len(captured) == 1
    trial = captured[0]
    assert trial.accepted, trial.diagnostics
    assert trial.diagnostics["published_insertions"] > 0
    assert all(len(nodes) == 6 for nodes in trial.mesh.tris.values())
    assert all(len(nodes) == 8 for nodes in trial.mesh.quads.values())
    assert bool(trial.mesh.quads) is recombine
