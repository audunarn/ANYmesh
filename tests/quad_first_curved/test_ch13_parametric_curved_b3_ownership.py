"""CH13 — straight B3 ownership on metric-curved quadratic quad-first."""

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


def _member_fixture(family: str):
    model, face, surface = _model_face(family)
    part = model.add_part(name=f"CH13 {family} B3 owner")
    model.add_sheet((face,), part_id=part)
    point = np.asarray(surface.evaluate(0.5, 0.5), dtype=float)
    du = np.asarray(surface.evaluate(0.5001, 0.5)) - np.asarray(surface.evaluate(0.4999, 0.5))
    dv = np.asarray(surface.evaluate(0.5, 0.5001)) - np.asarray(surface.evaluate(0.5, 0.4999))
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal)
    p0 = model.add_point(*(point - 0.5 * normal))
    p1 = model.add_point(*(point + 0.5 * normal))
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


def _generate(model, face, member, *, cancellation_check=None):
    return generate_hybrid_mesh_result(
        model, face_ids=(face,), member_ids=(member,), target_size=0.6,
        strategy="native", native_backend="python", recombine=True,
        order="quadratic", quad_options=QuadMeshingOptions(),
        cancellation_check=cancellation_check,
    ).mesh


def _support_residual(surface, point) -> float:
    point = np.asarray(point, dtype=float)
    uv = surface.local_uv(tuple(point))
    return float(np.linalg.norm(np.asarray(surface.evaluate(*uv), dtype=float) - point))


def _assert_strict_valid(mesh) -> None:
    for body in mesh.quads.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "Q8").status is ValidityStatus.CERTIFIED_POSITIVE
    for body in mesh.tris.values():
        coords = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        assert certify_mapping_validity(coords, "T6").status is ValidityStatus.CERTIFIED_POSITIVE


@pytest.mark.parametrize("family", ("ruled", "coons"))
def test_ch13_metric_curved_quadratic_owns_straight_member_as_b3(family):
    model, face, surface, member, edge, point = _member_fixture(family)
    before = _persistent_state(model)
    mesh = _generate(model, face, member)

    assert mesh.order == "quadratic"
    assert (len(mesh.quads), len(mesh.tris)) == (8, 0)
    assert all(len(body) == 8 for body in mesh.quads.values())
    assert mesh.beams and all(len(body) == 3 for body in mesh.beams.values())
    assert edge in mesh.elements_of_edge and edge in mesh.nodes_of_edge
    for eid in mesh.elements_of_edge[edge]:
        start, mid, end = mesh.beams[int(eid)]
        np.testing.assert_allclose(mesh.nodes[mid], 0.5 * (mesh.nodes[start] + mesh.nodes[end]), atol=1e-12)
    station_nodes = tuple(int(node) for node in mesh.nodes_of_edge[edge])
    center = min(station_nodes, key=lambda node: np.linalg.norm(mesh.nodes[node] - point))
    np.testing.assert_allclose(mesh.nodes[center], point, atol=1e-12)
    assert len(mesh.couplings) == 1
    coupling = next(iter(mesh.couplings.values()))
    assert int(coupling.beam_node) == center
    assert sum(coupling.weights) == pytest.approx(1.0)
    assert np.isfinite(coupling.eccentricity).all()
    assert float(np.linalg.norm(coupling.eccentricity)) < 1e-2
    _assert_strict_valid(mesh)
    shell_nodes = {
        int(node) for eid in mesh.elements_of_face[face]
        for node in mesh.quads[int(eid)]
    }
    assert max(_support_residual(surface, mesh.nodes[node]) for node in shell_nodes) <= 1e-10
    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.beams == mesh.beams
    assert restored.nodes_of_edge[edge] == mesh.nodes_of_edge[edge]
    assert len(restored.couplings) == 1
    assert _persistent_state(model) == before


@pytest.mark.parametrize("family", ("ruled", "coons"))
def test_ch13_repeat_and_cancellation_are_atomic(family):
    model, face, _surface, member, _edge, _point = _member_fixture(family)
    before = _persistent_state(model)
    first = _generate(model, face, member)
    second = _generate(model, face, member)
    assert mesh_to_dict(first) == mesh_to_dict(second)
    class Cancelled(RuntimeError):
        pass

    def cancel(stage):
        if stage == "quad-first:quadratic-promotion-ready":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        _generate(model, face, member, cancellation_check=cancel)
    assert _persistent_state(model) == before


@pytest.mark.parametrize("family", ("ruled", "coons"))
def test_ch13_source_boundary_beam_remains_typed_unsupported(family):
    from anymesher.quad.public_integration import QuadPublicUnsupported

    model, face, _surface = _model_face(family)
    boundary_edge = int(model.faces[face].loop[0].edge)
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(
            model, face_ids=(face,), beam_edges=(boundary_edge,), target_size=0.6,
            strategy="native", native_backend="python", recombine=True,
            order="quadratic", quad_options=QuadMeshingOptions(),
        )
