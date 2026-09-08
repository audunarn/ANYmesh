"""Public certified-sector integration, refinement, lifting and shared identity."""

import json

import numpy as np
import pytest
from anygeometry import to_dict

from anymesher import (
    FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec,
    NativeMeshingOptions,
)
from anymesher import hybrid, surface_mesh
from anymesher.errors import MeshError
from anymesher.quality_v2 import triangle_quality
from test_cylindrical_atlas_binding import _sector_model


def _persistent_state(model):
    # Owner serialization includes persistent topology/geometry, not disposable
    # query caches populated by valid geometry evaluation during generation.
    return model.revision, json.dumps(to_dict(model), sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("metric_mode", ("legacy", "isotropic_spatial"))
def test_certified_full_sector_cylinder_uses_real_native_path(metric_mode, monkeypatch):
    model, _ = _sector_model()
    before = _persistent_state(model)
    bindings = {}
    face_results = []
    insertions = []
    native_face = hybrid._mesh_native_face
    refine = surface_mesh.frontal_delaunay_refine

    def observed_refine(*args, **kwargs):
        count = len(args[0].points)
        result = refine(*args, **kwargs)
        insertions.append(len(result[0].points) - count)
        return result

    def bound_native(geometry, mesh, face_id, **kwargs):
        binding = kwargs["_cylindrical_binding"]
        assert binding is not None
        bindings.setdefault((geometry.model_id, geometry.revision), binding)
        assert binding is bindings[(geometry.model_id, geometry.revision)]
        result = native_face(geometry, mesh, face_id, **kwargs)
        print("CYLINDER_FACE_DIAGNOSTICS " + json.dumps(
            {"face_id": face_id, "diagnostics": result}, sort_keys=True, default=str
        ))
        face_results.append(result)
        return result

    monkeypatch.setattr(hybrid, "_mesh_native_face", bound_native)
    monkeypatch.setattr(surface_mesh, "frontal_delaunay_refine", observed_refine)
    # Physical feature resolution is independent of the coarse boundary target:
    # eight interior measurement regions need 0.12 m detail, with a 0.35 m
    # influence radius that does not reach their sector's protected edges.
    centers = tuple(
        (float(np.cos((i + .5) * np.pi / 4)),
         float(np.sin((i + .5) * np.pi / 4)), 1.)
        for i in range(8)
    )
    field = None if metric_mode == "legacy" else MetricFieldSpec(
        IsotropicMetricControl(.4),
        feature_controls=tuple(
            FeatureDistanceMetricControl((center,), .12, .35, 1.5, f"interior-{i}")
            for i, center in enumerate(centers)
        ),
    )
    result = hybrid.generate_hybrid_mesh_result(
        model, target_size=.4, strategy="native", native_backend="python",
        recombine=False,
        native_options=NativeMeshingOptions(
            point_placement="frontal_delaunay", metric_mode=metric_mode,
            metric_field=field,
            max_insertions=128, max_topology_operations=20000,
        ),
    )
    assert len(face_results) == 8
    assert set(result.strategy_by_face.values()) == {"native"}
    assert len(insertions) >= 8
    if metric_mode == "isotropic_spatial":
        assert sum(insertions) > 0, "Seed-only meshes do not qualify refinement"
    mesh = result.mesh
    assert mesh.tris and not mesh.quads
    points = np.asarray(tuple(mesh.nodes.values()))
    np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 1., atol=1.e-12)
    rows = {identifier: row for row, identifier in enumerate(mesh.nodes)}
    connectivity = np.asarray([
        [rows[node] for node in nodes] for nodes in mesh.tris.values()
    ])
    quality = triangle_quality(points, connectivity)
    assert np.all(quality.area > 0.)
    assert np.all(quality.scaled_jacobian > 0.)
    assert np.max(quality.aspect_ratio) <= 5.
    incidence = {}
    for triangle in mesh.tris.values():
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            key = tuple(sorted((a, b)))
            incidence[key] = incidence.get(key, 0) + 1
    assert max(incidence.values()) == 2
    if metric_mode == "isotropic_spatial":
        cell_points = points[connectivity]
        centroids = np.mean(cell_points, axis=1)
        lengths = np.linalg.norm(cell_points - np.roll(cell_points, -1, axis=1), axis=2)
        for center in centers:
            near = np.linalg.norm(centroids - center, axis=1) < .12
            assert np.any(near), "Every requested interior region must be resolved"
            assert np.max(lengths[near]) < .4
    for edge_id, edge in model.edges.items():
        node_ids = mesh.nodes_of_edge[edge_id]
        assert len(node_ids) == len(set(node_ids))
        if len(model.faces_using_edge(edge_id)) == 2:
            assert all(
                incidence[tuple(sorted((a, b)))] == 2
                for a, b in zip(node_ids, node_ids[1:])
            )
        for face_id in model.faces_using_edge(edge_id):
            owned_elements = mesh.elements_of_face[face_id]
            used = {
                node for element_id in owned_elements
                for node in mesh.tris[element_id]
            }
            assert set(node_ids).issubset(used)
    assert _persistent_state(model) == before
    print(json.dumps({
        "metric_mode": metric_mode, "insertions_by_face": insertions,
        "nodes": len(mesh.nodes), "triangles": len(mesh.tris),
        "minimum_scaled_jacobian": float(np.min(quality.scaled_jacobian)),
        "maximum_aspect_ratio": float(np.max(quality.aspect_ratio)),
        "shared_edges_and_incidence": "passed", "persistent_geometry": "unchanged",
    }, sort_keys=True))


def test_public_cylindrical_owner_failure_prevents_publication(monkeypatch):
    from anymesher import _cylindrical_public
    def unavailable(*args, **kwargs):
        raise MeshError("owner atlas binding unavailable")
    monkeypatch.setattr(_cylindrical_public, "prepare_cylindrical_atlas", unavailable)
    model, _ = _sector_model()
    before = _persistent_state(model)
    with pytest.raises(MeshError, match="owner atlas binding"):
        hybrid.generate_hybrid_mesh_result(
            model, target_size=.4, strategy="native", native_backend="python",
            recombine=False,
            native_options=NativeMeshingOptions(point_placement="frontal_delaunay"),
        )
    assert _persistent_state(model) == before
