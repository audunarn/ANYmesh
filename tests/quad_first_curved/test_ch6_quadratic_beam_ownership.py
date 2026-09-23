"""CH6 — coherent B3 ownership for quadratic quad-first mixed shell/beam output."""

from copy import deepcopy

import numpy as np
import pytest

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.serialize import mesh_from_dict, mesh_to_dict
from quad_first.test_q6_public_integration import _beam_through_face_fixture


def _source_signature(geometry):
    return (
        str(geometry.model_id),
        int(geometry.revision),
        tuple((i, tuple(map(float, geometry.vertex_position(i)))) for i in sorted(geometry.vertices)),
        tuple((i, geometry.edges[i].start, geometry.edges[i].end) for i in sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
    )


def _shell_corners(mesh):
    return {
        int(eid): tuple(int(node) for node in mesh.corners_of(eid))
        for eid in (*mesh.quads, *mesh.tris)
    }


def test_public_quadratic_quad_first_owns_straight_member_as_b3():
    geometry, face, _part, sheet, member, member_edge = _beam_through_face_fixture()
    before = _source_signature(geometry)
    linear = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        member_ids=(member,),
        quad_options=QuadMeshingOptions(),
        order="linear",
    ).mesh
    quadratic = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        member_ids=(member,),
        quad_options=QuadMeshingOptions(),
        order="quadratic",
    ).mesh

    assert quadratic.order == "quadratic"
    assert len(quadratic.quads) == len(linear.quads)
    assert len(quadratic.tris) == len(linear.tris)
    assert _shell_corners(quadratic) == _shell_corners(linear)
    assert all(len(body) == 8 for body in quadratic.quads.values())
    assert all(len(body) == 6 for body in quadratic.tris.values())
    beam_ids = tuple(int(eid) for eid in quadratic.elements_of_edge[member_edge])
    assert beam_ids
    assert all(eid in quadratic.beams for eid in beam_ids)
    assert all(len(quadratic.beams[eid]) == 3 for eid in beam_ids)
    for eid in beam_ids:
        start, mid, end = quadratic.beams[eid]
        np.testing.assert_allclose(
            quadratic.nodes[mid],
            0.5 * (quadratic.nodes[start] + quadratic.nodes[end]),
            atol=1.0e-12,
        )

    station_nodes = tuple(int(node) for node in quadratic.nodes_of_edge[member_edge])
    assert len(station_nodes) == 2 * len(beam_ids) + 1
    linear_station_nodes = tuple(int(node) for node in linear.nodes_of_edge[member_edge])
    assert len(station_nodes[::2]) == len(linear_station_nodes)
    for qnode, lnode in zip(station_nodes[::2], linear_station_nodes):
        np.testing.assert_allclose(quadratic.nodes[qnode], linear.nodes[lnode], atol=1.0e-12)
    center = min(station_nodes, key=lambda node: float(np.linalg.norm(quadratic.nodes[node] - np.array((0.5, 0.5, 0.0)))))
    np.testing.assert_allclose(quadratic.nodes[center], (0.5, 0.5, 0.0), atol=1.0e-12)

    assert len(quadratic.couplings) == 1
    coupling = next(iter(quadratic.couplings.values()))
    assert int(coupling.beam_node) == center
    assert sum(coupling.weights) == pytest.approx(1.0)
    assert list(quadratic.elements_of_sheet[sheet]) == list(quadratic.elements_of_face[face])
    assert _source_signature(geometry) == before
    restored = mesh_from_dict(mesh_to_dict(quadratic))
    assert restored.order == "quadratic"
    assert restored.beams == quadratic.beams
    assert restored.nodes_of_edge[member_edge] == quadratic.nodes_of_edge[member_edge]
    assert len(restored.couplings) == 1


def test_quadratic_b3_public_route_is_deterministic_and_cancellation_is_atomic():
    geometry, face, _part, _sheet, member, _member_edge = _beam_through_face_fixture()
    before = _source_signature(geometry)
    first = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face,), member_ids=(member,),
        quad_options=QuadMeshingOptions(), order="quadratic",
    ).mesh
    second = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face,), member_ids=(member,),
        quad_options=QuadMeshingOptions(), order="quadratic",
    ).mesh
    assert mesh_to_dict(first) == mesh_to_dict(second)

    class Cancelled(RuntimeError):
        pass

    def cancel(stage):
        if stage == "quad-first:quadratic-promotion-ready":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        generate_hybrid_mesh_result(
            geometry, target_size=1.0, face_ids=(face,), member_ids=(member,),
            quad_options=QuadMeshingOptions(), order="quadratic",
            cancellation_check=cancel,
        )
    assert _source_signature(geometry) == before


def test_quadratic_quad_first_still_rejects_curved_b3_lines():
    from anymesher.errors import MeshError

    geometry, face, _part, _sheet, _member, _member_edge = _beam_through_face_fixture()
    start = geometry.add_point(2.0, 0.0, 0.0)
    via = geometry.add_point(2.5, 0.5, 0.0)
    end = geometry.add_point(3.0, 0.0, 0.0)
    arc = geometry.add_arc(start, via, end)
    before = _source_signature(geometry)

    with pytest.raises(MeshError, match="curved|straight-sided|B3"):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.5,
            face_ids=(face,),
            beam_edges=(arc,),
            quad_options=QuadMeshingOptions(),
            order="quadratic",
        )
    assert _source_signature(geometry) == before
