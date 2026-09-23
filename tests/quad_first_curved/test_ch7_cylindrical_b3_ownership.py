"""CH7 — coherent straight B3 ownership on cylindrical quadratic quad-first."""

import math

import numpy as np
import pytest

from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.serialize import mesh_from_dict, mesh_to_dict
from quad_first_curved.test_ch3_cylindrical_public import _face_id
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state


def _cylinder_member_fixture():
    model, selected = _sector_model(False)
    faces = tuple(_face_id(model, use) for use in selected)
    face = faces[0]
    part = next(iter(model.parts))
    theta = math.pi / 8.0
    radial = np.array((math.cos(theta), math.sin(theta), 0.0))
    p0 = model.add_point(*(0.5 * radial + np.array((0.0, 0.0, 1.0))))
    p1 = model.add_point(*(1.5 * radial + np.array((0.0, 0.0, 1.0))))
    edge = model.add_line(p0, p1)
    member = model.add_member((edge,), part_id=part)
    model.add_attachment(
        member, AttachmentKind.MEMBER_THROUGH_FACE, AttachmentTargetKind.FACE,
        face, ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )
    return model, faces, face, member, edge, radial


def _generate(model, faces, member, *, cancellation_check=None):
    return generate_hybrid_mesh_result(
        model,
        target_size=0.5,
        face_ids=tuple(faces),
        member_ids=(member,),
        strategy="native",
        native_backend="python",
        quad_options=QuadMeshingOptions(),
        order="quadratic",
        cancellation_check=cancellation_check,
    ).mesh


def test_cylindrical_quadratic_quad_first_owns_straight_member_as_b3():
    model, faces, face, member, edge, radial = _cylinder_member_fixture()
    before = _persistent_state(model)
    mesh = _generate(model, faces, member)

    assert mesh.order == "quadratic"
    assert (len(mesh.nodes), len(mesh.quads), len(mesh.tris), len(mesh.beams), len(mesh.couplings)) == (261, 64, 16, 2, 1)
    assert all(len(body) == 8 for body in mesh.quads.values())
    assert all(len(body) == 6 for body in mesh.tris.values())
    assert mesh.beams and all(len(body) == 3 for body in mesh.beams.values())
    assert edge in mesh.elements_of_edge and edge in mesh.nodes_of_edge

    beam_ids = tuple(int(eid) for eid in mesh.elements_of_edge[edge])
    assert beam_ids and all(eid in mesh.beams for eid in beam_ids)
    for eid in beam_ids:
        start, mid, end = mesh.beams[eid]
        np.testing.assert_allclose(mesh.nodes[mid], 0.5 * (mesh.nodes[start] + mesh.nodes[end]), atol=1e-12)
    station_nodes = tuple(int(node) for node in mesh.nodes_of_edge[edge])
    center = min(station_nodes, key=lambda node: abs(float(np.linalg.norm(mesh.nodes[node][:2])) - 1.0))
    expected = np.array((radial[0], radial[1], 1.0))
    np.testing.assert_allclose(mesh.nodes[center], expected, atol=1e-12)
    assert len(mesh.couplings) == 1
    coupling = next(iter(mesh.couplings.values()))
    assert int(coupling.beam_node) == center
    assert sum(coupling.weights) == pytest.approx(1.0)
    # The beam is on the exact owner cylinder while Q8 is a polynomial
    # approximation; the coupling records the small physical projection gap.
    assert 0.0 < float(np.linalg.norm(coupling.eccentricity)) < 1.0e-3

    shell_nodes = {
        int(node)
        for eid in mesh.elements_of_face[face]
        for node in (mesh.quads[eid] if eid in mesh.quads else mesh.tris[eid])
    }
    shell_points = np.asarray([mesh.nodes[node] for node in shell_nodes], dtype=float)
    np.testing.assert_allclose(np.linalg.norm(shell_points[:, :2], axis=1), 1.0, atol=1e-10)
    assert _persistent_state(model) == before

    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.beams == mesh.beams
    assert restored.nodes_of_edge[edge] == mesh.nodes_of_edge[edge]
    assert len(restored.couplings) == 1


def test_cylindrical_b3_route_is_deterministic_and_cancellation_is_atomic():
    model, faces, _face, member, _edge, _radial = _cylinder_member_fixture()
    before = _persistent_state(model)
    first = _generate(model, faces, member)
    second = _generate(model, faces, member)
    assert mesh_to_dict(first) == mesh_to_dict(second)

    class Cancelled(RuntimeError):
        pass

    def cancel(stage):
        if stage == "quad-first:quadratic-promotion-ready":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        _generate(model, faces, member, cancellation_check=cancel)
    assert _persistent_state(model) == before
