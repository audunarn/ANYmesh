"""Actual cylindrical component refusal for a caller's explicit strict policy."""
import json
import pickle

import numpy as np
import pytest

from anymesher import (
    FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec,
    NativeMeshingOptions, hybrid,
)
from anymesher._cylindrical_atlas import prepare_cylindrical_atlas
from anymesher._cylindrical_quadratic_refine import refine_quadratic_component
from anymesher.surface_mesh import SurfaceMeshOptions
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state


def test_actual_cylindrical_component_refuses_explicit_maximum_angle(monkeypatch, tmp_path):
    model, _ = _sector_model(hole=True)
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

    class ObservedComponent(BaseException):
        pass

    observed = ObservedComponent()

    def seeded(geometry, mesh, face, **kwargs):
        result = original(geometry, mesh, face,
                          **{**kwargs, "native_options": NativeMeshingOptions()})
        if set(mesh.elements_of_face) == set(geometry.faces):
            selected = tuple(geometry.handle("face_use", i) for i in sorted(geometry.face_uses))
            binding = prepare_cylindrical_atlas(geometry, selected, reference_face_use=selected[0])
            from anygeometry.curves import Straight
            eligible = tuple(e for e, edge in geometry.edges.items()
                             if isinstance(edge.curve, Straight) and len(geometry.faces_using_edge(e)) == 2)
            seed_bytes = pickle.dumps(mesh.__dict__, protocol=5)
            # A physical nondegenerate T3 necessarily has a maximum angle>=60.
            # This explicit caller request is therefore a deterministic miss;
            # it changes no default, geometry, work budget or final report.
            trial = refine_quadratic_component(
                geometry, mesh, binding, kwargs["boundary_registry"].entries(),
                SurfaceMeshOptions(order="quadratic", recombine=False, target_size=.4,
                    native_options=options, prefer_quality_policy=True,
                    enforce_quality=True, max_angle=50.),
                eligible_edges=eligible, metric_model_uuid=kwargs["metric_model_uuid"],
                metric_geometry_revision=kwargs["metric_geometry_revision"])
            (tmp_path / "strict-trial.json").write_text(
                json.dumps(trial.diagnostics, sort_keys=True, default=str), encoding="utf-8")
            assert pickle.dumps(mesh.__dict__, protocol=5) == seed_bytes
            assert trial.mesh.nodes.keys() == mesh.nodes.keys()
            assert all(trial.mesh.nodes[n].tobytes() == p.tobytes() for n, p in mesh.nodes.items())
            for name in ("tris", "quads", "nodes_of_edge", "elements_of_face"):
                assert getattr(trial.mesh, name) == getattr(mesh, name)
            assert trial.accepted is False
            assert trial.diagnostics["reason"] == "component_quality_policy"
            assert trial.diagnostics["quality_enforced"] is True
            assert trial.diagnostics["component_policy_accepted"] is False
            assert trial.diagnostics["target_met"] is False
            quality = trial.diagnostics["physical_quality"]
            assert quality["accepted"] is False
            assert quality["thresholds"]["maximum_angle"] == 50.
            assert quality["violation_counts"]["maximum_angle"] > 0
            assert quality["worst"]["maximum_angle"] >= 60.
            assert "promoted_quality" not in trial.diagnostics
            captured.append(trial)
            # End this observer after the genuine private refusal. There is no
            # second component refinement or publication/qualification claim.
            raise observed
        return result

    monkeypatch.setattr(hybrid, "_mesh_native_face", seeded)
    with pytest.raises(ObservedComponent) as caught:
        hybrid.generate_hybrid_mesh_result(
            model, target_size=.4, order="quadratic", recombine=False,
            strategy="native", native_backend="python", native_options=options)
    assert caught.value is observed
    assert len(captured) == 1
    assert _persistent_state(model) == before
