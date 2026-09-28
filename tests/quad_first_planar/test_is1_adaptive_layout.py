from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

import numpy as np
import pytest

from anygeometry import GeometryModel, from_dict, punch_hole, to_dict
from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from benchmarks.is1.fixtures import build
from benchmarks.is1.measure import audit_irregular
from benchmarks.sg1.measure import promotion_checks


def _irregular_model(with_hole: bool) -> tuple[GeometryModel, int]:
    model = GeometryModel()
    vertices = model.add_points(((0, 0, 0), (4, 0, 0), (4, 1.2, 0),
                                 (2.2, 1.2, 0), (2.2, 2.6, 0), (0, 2.6, 0)))
    face = model.add_face(model.add_polyline(vertices, close=True), surface=None)
    if with_hole:
        face, _ = punch_hole(model, face, (1.0, 1.0, 0.0), 0.25)
    return model, face


def _signature(model: GeometryModel) -> tuple:
    return (str(model.model_id), int(model.revision),
            tuple((i, tuple(model.vertex_position(i))) for i in sorted(model.vertices)),
            tuple((i, model.edges[i].start, model.edges[i].end)
                  for i in sorted(model.edges)),
            tuple(sorted(model.faces)))


@pytest.mark.parametrize("with_hole", (False, True))
def test_adaptive_layout_keeps_source_and_quadratic_corners(with_hole: bool) -> None:
    model, face = _irregular_model(with_hole)
    signature = _signature(model)
    arguments = dict(target_size=0.5, face_ids=(face,),
                     quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
                     layout_policy="adaptive")
    linear = generate_hybrid_mesh_result(model, order="linear", **arguments).mesh
    quadratic = generate_hybrid_mesh_result(model, order="quadratic", **arguments).mesh
    repeated = generate_hybrid_mesh_result(model, order="linear", **arguments).mesh
    assert _signature(model) == signature
    assert linear.nodes.keys() == repeated.nodes.keys()
    assert linear.quads == repeated.quads and linear.tris == repeated.tris
    for node in linear.nodes:
        np.testing.assert_array_equal(linear.nodes[node], repeated.nodes[node])
        np.testing.assert_array_equal(linear.nodes[node], quadratic.nodes[node])
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert {eid: tuple(body[:3]) for eid, body in quadratic.tris.items()} == linear.tris
    assert quadratic.hybrid_diagnostics["layout_policy"] == "adaptive"
    assert quadratic.hybrid_diagnostics["high_order_geometry"]["status"] == "CERTIFIED_POSITIVE"


def test_adaptive_layout_requires_explicit_quad_route() -> None:
    model, face = _irregular_model(False)
    with pytest.raises(MeshError, match="explicit quad_options"):
        generate_hybrid_mesh_result(model, target_size=0.5, face_ids=(face,),
                                    layout_policy="adaptive")


def test_adaptive_seed_cancellation_keeps_source_unchanged() -> None:
    model, face = _irregular_model(False)
    before = _signature(model)

    def cancel(stage: str) -> None:
        if stage == "quad-first:adaptive-fine-seed":
            raise RuntimeError("cancelled adaptive seed")

    with pytest.raises(RuntimeError, match="cancelled adaptive seed"):
        generate_hybrid_mesh_result(
            model, target_size=.5, face_ids=(face,),
            quad_options=QuadMeshingOptions(), layout_policy="adaptive",
            cancellation_check=cancel,
        )
    assert _signature(model) == before


@pytest.mark.parametrize("case,h", (
    ("planar_opening", .8), ("planar_opening", .4),
    ("planar_concave", .4),
))
def test_adaptive_planar_boundary_quality_precedes_promotion(case: str, h: float) -> None:
    source, faces, center, overrides = build(case, h)
    meshes = []
    for order in ("linear", "quadratic"):
        model = from_dict(to_dict(source))
        meshes.append(generate_hybrid_mesh_result(
            model, face_ids=faces, target_size=h, order=order,
            strategy="native", native_backend="python",
            quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
            layout_policy="adaptive", overrides=overrides,
        ).mesh)
    assert not [item for item in promotion_checks(*meshes)
                if item["status"] != "passed"]
    metrics = audit_irregular(model, faces, meshes[1], h, (), center)
    assert not [item for item in metrics["checks"] if item["status"] != "passed"]
    repair = meshes[1].hybrid_diagnostics["quad_quality_repair"]
    if case == "planar_opening" and h == .8:
        assert repair["precert_moved"]
    else:
        # Since the quad-first shape gates (PQ7) no admitted element reaches
        # the post-publication repair here; before them these cases needed
        # quad relocation.
        assert not repair["relocated_quads"]
        assert not repair["split_quads"]
    assert not repair["unresolved_quads"]


def test_irregular_independent_audit_detects_duplicated_internal_node() -> None:
    model, faces, center, overrides = build("planar_concave", .4)
    mesh = generate_hybrid_mesh_result(
        model, face_ids=faces, target_size=.4, order="linear",
        strategy="native", native_backend="python",
        quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
        layout_policy="adaptive", overrides=overrides,
    ).mesh
    edges = defaultdict(list)
    for element, body in (*mesh.quads.items(), *mesh.tris.items()):
        for a, b in zip(body, body[1:]+body[:1]):
            edges[tuple(sorted((a, b)))].append(element)
    boundary_nodes = {node for chain in mesh.nodes_of_edge.values() for node in chain}
    (old, _), (victim, _) = next(
        (edge, elements) for edge, elements in edges.items()
        if len(elements) == 2 and not any(node in boundary_nodes for node in edge)
    )
    damaged = deepcopy(mesh)
    duplicate = max(damaged.nodes) + 1
    damaged.nodes[duplicate] = np.asarray(damaged.nodes[old]).copy()
    cells = damaged.quads if victim in damaged.quads else damaged.tris
    cells[victim] = tuple(duplicate if node == old else node for node in cells[victim])
    metrics = audit_irregular(model, faces, damaged, .4, (), center)
    failed = {item["name"] for item in metrics["checks"] if item["status"] == "failed"}
    assert f"face/{faces[0]}/boundary-complete" in failed
    assert "global/boundary-complete" in failed


def test_adaptive_route_rejects_promotion_that_moves_linear_corners(monkeypatch) -> None:
    import anymesher.hybrid as hybrid

    model, face = _irregular_model(False)
    before = _signature(model)
    promote = hybrid._promote_quad_first_quadratic

    def moving_promotion(mesh, *args, **kwargs):
        promote(mesh, *args, **kwargs)
        node = min(mesh.nodes)
        mesh.nodes[node] = np.asarray(mesh.nodes[node]) + (1e-5, 0., 0.)

    monkeypatch.setattr(hybrid, "_promote_quad_first_quadratic", moving_promotion)
    with pytest.raises(MeshError, match="changed linear corner coordinates"):
        generate_hybrid_mesh_result(
            model, face_ids=(face,), target_size=.5, order="quadratic",
            quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
            layout_policy="adaptive",
        )
    assert _signature(model) == before
