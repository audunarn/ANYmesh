"""CH12 — coherent straight B3 ownership on conical quadratic quad-first."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.high_order import ValidityStatus, certify_mapping_validity
from anymesher.quad.options import QuadMeshingOptions
from anymesher.serialize import mesh_from_dict, mesh_to_dict
from test_cylindrical_frontal_integration import _persistent_state
from test_curved_native_qualification import _model_face


def _cone_member_fixture():
    model, face, surface = _model_face("cone")
    part = next(iter(model.parts)) if model.parts else model.add_part(name="CH12 conical B3 owner")
    model.add_sheet((face,), part_id=part)
    point = surface.evaluate(0.5, 0.5)
    point = np.asarray(point, dtype=float)
    p0 = model.add_point(*(0.5 * point))
    p1 = model.add_point(*(1.5 * point))
    edge = model.add_line(p0, p1)
    member = model.add_member((edge,), part_id=part)
    uv = surface.local_uv(tuple(point))
    model.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (
            ParameterRange.point(float(uv[0])),
            ParameterRange.point(float(uv[1])),
        ),
    )
    return model, face, surface, member, edge, point


def _generate_member(model, face, member, order, *, cancellation_check=None):
    return generate_hybrid_mesh_result(
        model,
        face_ids=(face,),
        member_ids=(member,),
        target_size=0.6,
        strategy="native",
        native_backend="python",
        recombine=True,
        order=order,
        quad_options=QuadMeshingOptions(),
        cancellation_check=cancellation_check,
    ).mesh


def _corners(body):
    if len(body) in (4, 8):
        return tuple(body[:4])
    if len(body) in (3, 6):
        return tuple(body[:3])
    raise AssertionError(f"unexpected shell width {len(body)}")


def _assert_strict_valid(mesh) -> None:
    for body in mesh.quads.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "Q8").status is ValidityStatus.CERTIFIED_POSITIVE
    for body in mesh.tris.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "T6").status is ValidityStatus.CERTIFIED_POSITIVE


def test_ch12_conical_quadratic_quad_first_owns_straight_member_as_b3():
    model, face, surface, member, edge, point = _cone_member_fixture()
    before = _persistent_state(model)
    mesh = _generate_member(model, face, member, "quadratic")

    assert mesh.order == "quadratic"
    assert set(mesh.elements_of_face) == {face}
    assert all(len(body) == 8 for body in mesh.quads.values())
    assert all(len(body) == 6 for body in mesh.tris.values())
    assert len(mesh.quads) == 23
    assert len(mesh.tris) == 6
    assert mesh.beams, "straight member must be owned as beams"
    # No hidden linear B2 publication inside a quadratic result.
    assert all(len(body) == 3 for body in mesh.beams.values())
    assert edge in mesh.elements_of_edge and edge in mesh.nodes_of_edge

    beam_ids = tuple(int(eid) for eid in mesh.elements_of_edge[edge])
    assert beam_ids and all(eid in mesh.beams for eid in beam_ids)
    for eid in beam_ids:
        start, mid, end = mesh.beams[eid]
        np.testing.assert_allclose(
            mesh.nodes[mid], 0.5 * (mesh.nodes[start] + mesh.nodes[end]), atol=1.0e-12
        )

    station_nodes = tuple(int(node) for node in mesh.nodes_of_edge[edge])
    center = min(
        station_nodes,
        key=lambda node: float(np.linalg.norm(np.asarray(mesh.nodes[node]) - point)),
    )
    np.testing.assert_allclose(np.asarray(mesh.nodes[center]), point, atol=1.0e-12)

    assert len(mesh.couplings) == 1
    coupling = next(iter(mesh.couplings.values()))
    assert int(coupling.beam_node) == center
    assert sum(coupling.weights) == pytest.approx(1.0)
    # The owner point itself is exact; only the polynomial shell approximation
    # may contribute a small, recorded projection gap.
    assert np.isfinite(coupling.eccentricity).all()
    assert float(np.linalg.norm(coupling.eccentricity)) < 1.0e-2

    _assert_strict_valid(mesh)
    shell_nodes = {
        int(node)
        for eid in mesh.elements_of_face[face]
        for node in (mesh.quads[eid] if eid in mesh.quads else mesh.tris[eid])
    }
    assert max(_support_residual_point(surface, mesh.nodes[node]) for node in shell_nodes) <= 1.0e-10
    assert _persistent_state(model) == before

    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.order == "quadratic"
    assert restored.beams == mesh.beams
    assert restored.nodes_of_edge[edge] == mesh.nodes_of_edge[edge]
    assert sorted(map(int, restored.elements_of_edge[edge])) == sorted(
        map(int, mesh.elements_of_edge[edge])
    )
    assert len(restored.couplings) == len(mesh.couplings)


def _support_residual_point(surface, point) -> float:
    point = np.asarray(point, dtype=float)
    uv = surface.local_uv(tuple(point))
    return float(np.linalg.norm(np.asarray(surface.evaluate(*uv), dtype=float) - point))


def test_ch12_conical_b3_route_is_deterministic_and_cancellation_is_atomic():
    model, face, _surface, member, _edge, _point = _cone_member_fixture()
    before = _persistent_state(model)
    first = _generate_member(model, face, member, "quadratic")
    second = _generate_member(model, face, member, "quadratic")
    assert mesh_to_dict(first) == mesh_to_dict(second)

    class Cancelled(RuntimeError):
        pass

    def cancel(stage):
        if stage == "quad-first:quadratic-promotion-ready":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        _generate_member(model, face, member, "quadratic", cancellation_check=cancel)
    assert _persistent_state(model) == before


def test_ch12_curved_quadratic_beam_edge_remains_rejected():
    from anymesher.errors import MeshError

    model, face, _surface = _model_face("cone")
    curved_edge = next(
        int(use.edge)
        for use in model.faces[face].loop
        if type(model.edges[int(use.edge)].curve).__name__ != "Straight"
    )
    with pytest.raises(MeshError):
        generate_hybrid_mesh_result(
            model, face_ids=(face,), beam_edges=(curved_edge,), target_size=0.6,
            strategy="native", native_backend="python", recombine=True,
            order="quadratic", quad_options=QuadMeshingOptions(),
        )


@pytest.mark.parametrize("family", ("ruled", "coons"))
def test_ch12_does_not_activate_parametric_curved_b3(family):
    from anymesher.quad.public_integration import QuadPublicUnsupported

    model, face, _surface = _model_face(family)
    straight_edge = int(model.faces[face].loop[0].edge)
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(
            model, face_ids=(face,), beam_edges=(straight_edge,), target_size=0.6,
            strategy="native", native_backend="python", recombine=True,
            order="quadratic", quad_options=QuadMeshingOptions(),
        )
