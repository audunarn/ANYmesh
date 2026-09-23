from __future__ import annotations

import numpy as np
import pytest

from test_cylindrical_frontal_integration import _persistent_state
from test_curved_native_qualification import _model_face

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.domain import ParametricQuadDomain
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher.refinement import Refinement
from quad_first_curved.test_ch10_parametric_curved_linear import _generate, _support_residual
from quad_first_curved.test_ch9_conical_quadratic import (
    _assert_strict_valid,
    _edge_mid_map,
    _linear_boundary_segments,
)


@pytest.mark.parametrize(
    ("name", "h", "linear_nodes", "q8_count", "quadratic_nodes", "midsides"),
    (("ruled", 0.6, 15, 8, 37, 22), ("ruled", 0.3, 40, 28, 107, 67),
     ("coons", 0.6, 15, 8, 37, 22), ("coons", 0.3, 40, 28, 107, 67)),
)
def test_ch11_preserves_ch10_topology_and_strict_validity(
    name, h, linear_nodes, q8_count, quadratic_nodes, midsides,
):
    linear_model, linear_face, _ = _model_face(name)
    linear = _generate(linear_model, linear_face, h, order="linear")
    model, face, surface = _model_face(name)
    before = _persistent_state(model)
    quadratic = _generate(model, face, h, order="quadratic")

    assert (len(linear.nodes), len(linear.quads), len(linear.tris)) == (
        linear_nodes, q8_count, 0,
    )
    assert (len(quadratic.nodes), len(quadratic.quads), len(quadratic.tris)) == (
        quadratic_nodes, q8_count, 0,
    )
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert quadratic.tris == {}
    for node, point in linear.nodes.items():
        np.testing.assert_allclose(quadratic.nodes[node], point, rtol=0.0, atol=0.0)
    edge_mid = _edge_mid_map(quadratic)
    assert len(edge_mid) == midsides
    assert set(edge_mid.values()) == set(quadratic.nodes) - set(linear.nodes)
    for edge_id, chain in linear.nodes_of_edge.items():
        assert tuple(quadratic.nodes_of_edge[edge_id][::2]) == tuple(chain)
    assert _support_residual(quadratic, surface) <= 1.0e-10
    _assert_strict_valid(quadratic)
    assert _persistent_state(model) == before


def test_ch11_interior_midsides_use_parametric_owner_chart_and_certificate():
    model, face, _ = _model_face("coons")
    linear = _generate(model, face, 0.3, order="linear")
    quadratic_model, quadratic_face, surface = _model_face("coons")
    quadratic = _generate(quadratic_model, quadratic_face, 0.3, order="quadratic")
    domain = ParametricQuadDomain.from_geometry(quadratic_model, quadratic_face)
    midsides = _edge_mid_map(quadratic)
    boundary = _linear_boundary_segments(linear)
    checked = 0
    for edge, midpoint_id in midsides.items():
        if edge in boundary:
            continue
        a, b = edge
        ca = domain.project(quadratic.nodes[a])
        cb = domain.project(quadratic.nodes[b])
        expected = domain.lift((0.5 * (ca[0] + cb[0]), 0.5 * (ca[1] + cb[1])))
        np.testing.assert_allclose(
            quadratic.nodes[midpoint_id], expected, rtol=0.0, atol=1.0e-12
        )
        checked += 1
    assert checked > 0
    assert _support_residual(quadratic, surface) <= 1.0e-10

    payload = quadratic.hybrid_diagnostics["high_order_geometry"]
    assert payload["status"] == "CERTIFIED_POSITIVE"
    assert (payload["q8_count"], payload["t6_count"], payload["unique_midside_count"]) == (
        28, 0, 67,
    )
    report = payload["reports"][0]
    assert report["geometry_family"] == "coons"
    assert report["chart_kind"] == "ParametricQuadDomain"
    assert report["interior_projection"] == "owner-chart-midpoint"
    assert payload["max_geometry_residual"] <= 1.0e-10


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch11_repeat_and_cancellation_are_atomic(name):
    model, face, _ = _model_face(name)
    first = _generate(model, face, 0.6, order="quadratic")
    repeat_model, repeat_face, _ = _model_face(name)
    repeat = _generate(repeat_model, repeat_face, 0.6, order="quadratic")
    assert repeat.quads == first.quads
    assert repeat.tris == first.tris
    assert repeat.nodes_of_edge == first.nodes_of_edge

    cancelled_model, cancelled_face, _ = _model_face(name)
    before = _persistent_state(cancelled_model)

    class Cancelled(RuntimeError):
        pass

    def cancel(phase: str) -> None:
        if phase == "quad-first:quadratic-promotion-ready":
            raise Cancelled(phase)

    with pytest.raises(Cancelled):
        _generate(
            cancelled_model, cancelled_face, 0.6, order="quadratic",
            cancellation_check=cancel,
        )
    assert _persistent_state(cancelled_model) == before


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch11_parametric_curved_quadratic_beam_remains_typed_unsupported(name):
    model, face, _ = _model_face(name)
    straight_edge = next(
        int(use.edge) for use in model.faces[face].loop
        if type(model.edges[int(use.edge)].curve).__name__ == "Straight"
    )
    before = _persistent_state(model)
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(
            model,
            face_ids=(face,),
            beam_edges=(straight_edge,),
            target_size=0.6,
            strategy="native",
            native_backend="python",
            recombine=True,
            order="quadratic",
            quad_options=QuadMeshingOptions(),
        )
    assert _persistent_state(model) == before


@pytest.mark.parametrize("name", ("ruled", "coons"))
def test_ch11_graded_residual_promotes_mixed_q8_t6(name):
    linear_model, linear_face, linear_surface = _model_face(name)
    linear_center = tuple(map(float, linear_surface.evaluate(0.5, 0.5)))
    linear_refinement = Refinement(
        size=0.2, radius=0.35, center=linear_center, growth=1.5, name="ch11-mixed"
    )
    linear = generate_hybrid_mesh_result(
        linear_model, face_ids=(linear_face,), target_size=0.6,
        strategy="native", native_backend="python", recombine=True,
        order="linear", quad_options=QuadMeshingOptions(),
        refinements=(linear_refinement,),
    ).mesh

    model, face, surface = _model_face(name)
    center = tuple(map(float, surface.evaluate(0.5, 0.5)))
    refinement = Refinement(
        size=0.2, radius=0.35, center=center, growth=1.5, name="ch11-mixed"
    )
    quadratic = generate_hybrid_mesh_result(
        model, face_ids=(face,), target_size=0.6,
        strategy="native", native_backend="python", recombine=True,
        order="quadratic", quad_options=QuadMeshingOptions(),
        refinements=(refinement,),
    ).mesh

    assert (len(linear.nodes), len(linear.quads), len(linear.tris)) == (35, 25, 4)
    assert (len(quadratic.nodes), len(quadratic.quads), len(quadratic.tris)) == (98, 25, 4)
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert {eid: tuple(body[:3]) for eid, body in quadratic.tris.items()} == linear.tris
    assert len(_edge_mid_map(quadratic)) == 63
    for edge_id, chain in linear.nodes_of_edge.items():
        assert tuple(quadratic.nodes_of_edge[edge_id][::2]) == tuple(chain)
    assert _support_residual(quadratic, surface) <= 1.0e-10
    _assert_strict_valid(quadratic)

    payload = quadratic.hybrid_diagnostics["high_order_geometry"]
    assert payload["status"] == "CERTIFIED_POSITIVE"
    assert (payload["q8_count"], payload["t6_count"], payload["unique_midside_count"]) == (
        25, 4, 63,
    )
    report = payload["reports"][0]
    assert report["geometry_family"] == name
    assert report["chart_kind"] == "ParametricQuadDomain"
    assert report["interior_projection"] == "owner-chart-midpoint"
    assert payload["max_geometry_residual"] <= 1.0e-10
