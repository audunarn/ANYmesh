from __future__ import annotations

import numpy as np
import pytest

from test_cylindrical_frontal_integration import _persistent_state
from test_curved_native_qualification import _model_face

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.domain import ConicalQuadDomain
from anymesher.quad.high_order import ValidityStatus, certify_mapping_validity
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from quad_first_curved.test_ch8_conical_public import _generate, _support_residual


def _corners(body: tuple[int, ...]) -> tuple[int, ...]:
    if len(body) in (4, 8):
        return tuple(body[:4])
    if len(body) in (3, 6):
        return tuple(body[:3])
    raise AssertionError(f"unexpected shell width {len(body)}")


def _edge_mid_map(mesh) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for body in list(mesh.quads.values()) + list(mesh.tris.values()):
        corners = _corners(tuple(body))
        midsides = tuple(body[len(corners):])
        assert len(midsides) == len(corners)
        for index, midpoint in enumerate(midsides):
            a, b = corners[index], corners[(index + 1) % len(corners)]
            key = (min(a, b), max(a, b))
            if key in result:
                assert result[key] == midpoint
            result[key] = midpoint
    return result


def _assert_strict_valid(mesh) -> None:
    for body in mesh.quads.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "Q8").status is ValidityStatus.CERTIFIED_POSITIVE
    for body in mesh.tris.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "T6").status is ValidityStatus.CERTIFIED_POSITIVE


def _linear_boundary_segments(mesh) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for chain in mesh.nodes_of_edge.values():
        for a, b in zip(chain, chain[1:]):
            result.add((min(int(a), int(b)), max(int(a), int(b))))
    return result


def test_ch9_coarse_cone_promotes_exact_ch8_topology() -> None:
    linear_model, linear_face, _ = _model_face("cone")
    linear = _generate(linear_model, (linear_face,), 0.6, order="linear").mesh
    model, face, surface = _model_face("cone")
    before = _persistent_state(model)
    quadratic = _generate(model, (face,), 0.6, order="quadratic").mesh

    assert (len(linear.nodes), len(linear.quads), len(linear.tris)) == (42, 28, 6)
    assert (len(quadratic.nodes), len(quadratic.quads), len(quadratic.tris)) == (117, 28, 6)
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert {eid: tuple(body[:3]) for eid, body in quadratic.tris.items()} == linear.tris
    for node, point in linear.nodes.items():
        np.testing.assert_allclose(quadratic.nodes[node], point, rtol=0.0, atol=0.0)

    midsides = _edge_mid_map(quadratic)
    assert len(midsides) == 75
    assert set(midsides.values()) == set(quadratic.nodes) - set(linear.nodes)
    for edge_id, chain in linear.nodes_of_edge.items():
        assert tuple(quadratic.nodes_of_edge[edge_id][::2]) == tuple(chain)
    assert _support_residual(quadratic, surface) <= 1.0e-10
    _assert_strict_valid(quadratic)
    assert _persistent_state(model) == before


def test_ch9_fine_cone_preserves_ch8_corner_topology() -> None:
    linear_model, linear_face, _ = _model_face("cone")
    linear = _generate(linear_model, (linear_face,), 0.3, order="linear").mesh
    model, face, surface = _model_face("cone")
    quadratic = _generate(model, (face,), 0.3, order="quadratic").mesh

    assert (len(linear.nodes), len(linear.quads), len(linear.tris)) == (118, 89, 14)
    assert (len(quadratic.nodes), len(quadratic.quads), len(quadratic.tris)) == (338, 89, 14)
    assert {eid: tuple(body[:4]) for eid, body in quadratic.quads.items()} == linear.quads
    assert {eid: tuple(body[:3]) for eid, body in quadratic.tris.items()} == linear.tris
    assert len(_edge_mid_map(quadratic)) == 220
    assert _support_residual(quadratic, surface) <= 1.0e-10
    _assert_strict_valid(quadratic)


def test_ch9_interior_midsides_use_developed_chart_midpoints_and_certificate() -> None:
    linear_model, linear_face, _ = _model_face("cone")
    linear = _generate(linear_model, (linear_face,), 0.6, order="linear").mesh
    model, face, _ = _model_face("cone")
    quadratic = _generate(model, (face,), 0.6, order="quadratic").mesh
    domain = ConicalQuadDomain.from_geometry(model, face)
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
        np.testing.assert_allclose(quadratic.nodes[midpoint_id], expected, rtol=0.0, atol=1.0e-12)
        checked += 1
    assert checked > 0

    payload = quadratic.hybrid_diagnostics["high_order_geometry"]
    assert payload["status"] == "CERTIFIED_POSITIVE"
    assert (payload["q8_count"], payload["t6_count"], payload["unique_midside_count"]) == (28, 6, 75)
    report = payload["reports"][0]
    assert report["geometry_family"] == "conical"
    assert report["chart_kind"] == "ConicalQuadDomain"
    assert report["interior_projection"] == "owner-chart-midpoint"
    assert payload["max_geometry_residual"] <= 1.0e-10
    assert quadratic.hybrid_diagnostics["route"] == "quad-first-conical"


def test_ch9_repeat_and_cancellation_are_atomic() -> None:
    model, face, _ = _model_face("cone")
    first = _generate(model, (face,), 0.6, order="quadratic").mesh
    repeat_model, repeat_face, _ = _model_face("cone")
    repeat = _generate(repeat_model, (repeat_face,), 0.6, order="quadratic").mesh
    assert (len(repeat.nodes), len(repeat.quads), len(repeat.tris)) == (
        len(first.nodes), len(first.quads), len(first.tris)
    )
    assert repeat.quads == first.quads
    assert repeat.tris == first.tris
    assert repeat.nodes_of_edge == first.nodes_of_edge

    cancelled_model, cancelled_face, _ = _model_face("cone")
    before = _persistent_state(cancelled_model)

    class Cancelled(RuntimeError):
        pass

    def cancel(phase: str) -> None:
        if phase == "quad-first:quadratic-promotion-ready":
            raise Cancelled(phase)

    with pytest.raises(Cancelled):
        _generate(cancelled_model, (cancelled_face,), 0.6, order="quadratic", cancellation_check=cancel)
    assert _persistent_state(cancelled_model) == before


def test_ch9_conical_quadratic_source_boundary_beam_remains_typed_unsupported() -> None:
    model, face, _ = _model_face("cone")
    before = _persistent_state(model)
    straight_edge = next(
        int(use.edge) for use in model.faces[face].loop
        if type(model.edges[int(use.edge)].curve).__name__ == "Straight"
    )
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
