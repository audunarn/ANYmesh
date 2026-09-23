"""CH2 — planar explicit Q8/T6 promotion over the qualified linear quad-first result.

Runs against the qualified planar linear quad-first topology; the FINAL active
Q4/T3 cells are promoted in place (corner IDs + element IDs stable) to Q8/T6 by
adding exactly one canonical midside node per final-cell edge.
"""

import numpy as np
import pytest

from anygeometry.model import GeometryModel
from anymesher import Mesh
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.high_order import ValidityStatus, certify_mapping_validity
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher.refinement import Refinement
from anymesher.serialize import mesh_from_dict, mesh_to_dict
from quad_first_planar.fixtures import p01_geometry
from quad_first.test_q6_public_integration import _beam_through_face_fixture
from quad_first_planar.test_pq4_general_domains import _p03_geometry
from quad_first_planar.test_pq4b_staged_domains import _p07_geometry, _p09_geometry


def _source_signature(geometry):
    return (
        str(geometry.model_id),
        int(geometry.revision),
        tuple(
            (i, tuple(map(float, geometry.vertex_position(i))))
            for i in sorted(geometry.vertices)
        ),
        tuple(
            (i, geometry.edges[i].start, geometry.edges[i].end)
            for i in sorted(geometry.edges)
        ),
        tuple(sorted(geometry.faces)),
    )


def test_p01_planar_quadratic_promotion_contract():
    geometry, face_id = p01_geometry()
    snapshot = _source_signature(geometry)

    linear = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
        order="linear",
    ).mesh
    assert len(linear.nodes) == 273
    assert len(linear.quads) == 240
    assert len(linear.tris) == 0
    assert all(len(body) == 4 for body in linear.quads.values())

    quadratic = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
        order="quadratic",
    ).mesh
    assert _source_signature(geometry) == snapshot
    for node_id, position in linear.nodes.items():
        np.testing.assert_array_equal(quadratic.nodes[node_id], position)
    assert quadratic.elements_of_face == linear.elements_of_face

    assert isinstance(quadratic, Mesh)
    assert len(quadratic.nodes) == 785
    assert len(quadratic.quads) == 240
    assert len(quadratic.tris) == 0
    assert all(len(body) == 8 for body in quadratic.quads.values())
    assert quadratic.order == "quadratic"
    promotion = quadratic.hybrid_diagnostics["quadratic_promotion"]
    assert promotion["status"] == "APPLIED"
    assert promotion["repair_count"] == 0
    assert promotion["unique_midsides"] == 512
    assert promotion["max_corner_displacement"] == pytest.approx(0.0)

    assert set(quadratic.shells) == set(linear.shells)
    linear_node_ids = frozenset(linear.nodes)
    midside_ids: dict[frozenset, int] = {}
    for element_id, body in quadratic.quads.items():
        corners = tuple(body[:4])
        assert corners == tuple(linear.quads[element_id]), (
            f"element {element_id}: corners {corners} != "
            f"linear {linear.quads[element_id]}"
        )
        edge_pairs = (
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        )
        for (a, b), mid in zip(edge_pairs, body[4:8]):
            key = frozenset((a, b))
            assert mid not in linear_node_ids, (
                f"midside {mid} collides with a linear node"
            )
            previous = midside_ids.get(key)
            if previous is None:
                midside_ids[key] = mid
            else:
                assert previous == mid, (
                    f"shared edge {sorted(key)} maps to "
                    f"{previous} and {mid}"
                )
    assert len(midside_ids) == 512
    assert set(midside_ids.values()) <= set(quadratic.nodes)
    assert len(set(midside_ids.values())) == 512

    assert quadratic.node_of_vertex == linear.node_of_vertex
    for edge_id in sorted(set(linear.nodes_of_edge) & set(quadratic.nodes_of_edge)):
        lchain = linear.nodes_of_edge[edge_id]
        qchain = quadratic.nodes_of_edge[edge_id]
        assert len(qchain) == 2 * len(lchain) - 1
        assert qchain[::2] == lchain


def _quadratic_result(geometry, face_ids, h=0.5, **kwargs):
    return generate_hybrid_mesh_result(
        geometry,
        target_size=float(h),
        face_ids=tuple(face_ids),
        quad_options=QuadMeshingOptions(),
        order="quadratic",
        **kwargs,
    ).mesh


def _assert_strict_quadratic_validity(mesh):
    for body in mesh.quads.values():
        nodes = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        report = certify_mapping_validity(nodes, "Q8")
        assert report.status is ValidityStatus.CERTIFIED_POSITIVE
    for body in mesh.tris.values():
        nodes = np.asarray([mesh.nodes[node] for node in body], dtype=float)
        report = certify_mapping_validity(nodes, "T6")
        assert report.status is ValidityStatus.CERTIFIED_POSITIVE


def test_mixed_trapezoid_promotes_to_q8_t6_with_strict_validity():
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0),
         (3.0, 4.0, 0.0), (1.0, 4.0, 0.0))
    )
    face_id = geometry.add_plate(vertices)
    before = _source_signature(geometry)
    linear = generate_hybrid_mesh_result(
        geometry, target_size=0.75, face_ids=(face_id,),
        quad_options=QuadMeshingOptions(), order="linear",
    ).mesh
    mesh = _quadratic_result(geometry, (face_id,), h=0.75)
    assert _source_signature(geometry) == before
    assert (len(mesh.quads), len(mesh.tris)) == (len(linear.quads), len(linear.tris))
    for element_id, body in linear.quads.items():
        assert mesh.quads[element_id][:4] == body
    for element_id, body in linear.tris.items():
        assert mesh.tris[element_id][:3] == body
    assert mesh.quads and mesh.tris
    assert all(len(body) == 8 for body in mesh.quads.values())
    assert all(len(body) == 6 for body in mesh.tris.values())
    _assert_strict_quadratic_validity(mesh)


def test_adjacent_faces_reuse_exact_quadratic_shared_edge_chain():
    geometry, first, second, shared = _p09_geometry()
    d1 = PlanarQuadDomain.from_geometry(geometry, first)
    d2 = PlanarQuadDomain.from_geometry(geometry, second)
    linear = generate_hybrid_mesh_result(
        geometry, target_size=0.5, face_ids=(first, second),
        quad_options=QuadMeshingOptions(), order="linear",
    ).mesh
    quadratic = _quadratic_result(geometry, (first, second), h=0.5)
    lchain = linear.nodes_of_edge[shared]
    qchain = quadratic.nodes_of_edge[shared]
    assert len(qchain) == 2 * len(lchain) - 1
    assert qchain[::2] == lchain
    use1 = dict(d1.edge_uses)[shared]
    use2 = dict(d2.edge_uses)[shared]
    assert use1 != use2
    oriented1 = qchain if use1 else list(reversed(qchain))
    oriented2 = qchain if use2 else list(reversed(qchain))
    assert oriented1 == list(reversed(oriented2))


def _concave_geometry():
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0, 0, 0), (6, 0, 0), (6, 2, 0),
         (3, 2, 0), (3, 5, 0), (0, 5, 0))
    )
    face_id = geometry.add_face(
        geometry.add_polyline(vertices, close=True), surface=None
    )
    return geometry, face_id


@pytest.mark.parametrize("fixture", ["hole", "concave"])
def test_general_planar_domains_promote_without_topology_duplication(fixture):
    if fixture == "hole":
        geometry, face_id, _ = _p03_geometry()
    else:
        geometry, face_id = _concave_geometry()
    linear = generate_hybrid_mesh_result(
        geometry, target_size=0.5, face_ids=(face_id,),
        quad_options=QuadMeshingOptions(), order="linear",
    ).mesh
    quadratic = _quadratic_result(geometry, (face_id,), h=0.5)
    assert set(quadratic.shells) == set(linear.shells)
    assert set(quadratic.nodes).issuperset(linear.nodes)
    for element_id, body in linear.quads.items():
        assert quadratic.quads[element_id][:4] == body
    for element_id, body in linear.tris.items():
        assert quadratic.tris[element_id][:3] == body
    assert all(len(body) == 8 for body in quadratic.quads.values())
    assert all(len(body) == 6 for body in quadratic.tris.values())
    _assert_strict_quadratic_validity(quadratic)


def test_graded_refinement_quadratic_preserves_final_linear_corner_topology():
    refinement = Refinement(
        size=0.25, radius=0.75, center=(2.0, 2.0, 0.0),
        growth=1.5, name="ch2-local",
    )
    geometry, face_id = _p07_geometry()
    before = _source_signature(geometry)
    linear = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face_id,),
        quad_options=QuadMeshingOptions(), order="linear",
        refinements=(refinement,),
    ).mesh
    quadratic = _quadratic_result(
        geometry, (face_id,), h=1.0, refinements=(refinement,)
    )
    assert set(quadratic.shells) == set(linear.shells)
    assert len(quadratic.nodes) > len(linear.nodes)
    for element_id, body in linear.quads.items():
        assert quadratic.quads[element_id][:4] == body
    for element_id, body in linear.tris.items():
        assert quadratic.tris[element_id][:3] == body
    for edge_id, chain in linear.nodes_of_edge.items():
        assert quadratic.nodes_of_edge[edge_id][::2] == chain
    assert _source_signature(geometry) == before
    _assert_strict_quadratic_validity(quadratic)


def test_quadratic_quad_first_serialization_round_trip_preserves_order_and_ownership():
    geometry, face_id = p01_geometry()
    mesh = _quadratic_result(geometry, (face_id,), h=0.5)
    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.order == "quadratic"
    assert restored.quads == mesh.quads
    assert restored.tris == mesh.tris
    assert restored.node_of_vertex == mesh.node_of_vertex
    assert restored.nodes_of_edge == mesh.nodes_of_edge
    assert restored.elements_of_face == mesh.elements_of_face
    assert all(len(body) == 8 for body in restored.quads.values())


class _PromotionCancelled(RuntimeError):
    pass


def test_quadratic_promotion_cancellation_is_atomic_and_source_immutable():
    geometry, face_id = p01_geometry()
    before = _source_signature(geometry)

    def cancel(stage):
        if stage == "quad-first:quadratic-promotion-ready":
            raise _PromotionCancelled(stage)

    with pytest.raises(_PromotionCancelled):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.5,
            face_ids=(face_id,),
            quad_options=QuadMeshingOptions(),
            order="quadratic",
            cancellation_check=cancel,
        )
    assert _source_signature(geometry) == before


def test_hole_boundary_quadratic_midsides_lie_on_exact_source_edges():
    geometry, face_id, hole_edges = _p03_geometry()
    source_before = _source_signature(geometry)
    mesh = _quadratic_result(geometry, (face_id,), h=0.5)
    assert _source_signature(geometry) == source_before
    promotion = mesh.hybrid_diagnostics["quadratic_promotion"]
    assert promotion["status"] == "APPLIED"
    assert promotion["repair_count"] >= 1
    assert promotion["max_corner_displacement"] <= 0.25 + 1.0e-12
    protected = {node for chain in mesh.nodes_of_edge.values() for node in chain}
    protected.update(mesh.node_of_vertex.values())
    assert set(promotion["repaired_corner_nodes"]).isdisjoint(protected)
    _assert_strict_quadratic_validity(mesh)
    center = np.asarray((3.2, 2.4), dtype=float)
    for edge_id in hole_edges:
        for node_id in mesh.nodes_of_edge[edge_id][1::2]:
            radius = float(np.linalg.norm(np.asarray(mesh.nodes[node_id], dtype=float)[:2] - center))
            assert radius == pytest.approx(0.9, abs=1.0e-10)
    for edge_id, chain in mesh.nodes_of_edge.items():
        length = geometry.edge_length(edge_id)
        tolerance = max(
            geometry.tolerance.effective_length(length),
            128.0 * np.finfo(float).eps * max(length, 1.0),
        )
        for node_id in chain[1::2]:
            _point, _parameter, distance = geometry.closest_edge_point(
                edge_id, mesh.nodes[node_id]
            )
            assert distance <= tolerance


def test_mixed_mesh_uses_one_midside_for_q8_q8_and_q8_t6_interfaces():
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0),
         (3.0, 4.0, 0.0), (1.0, 4.0, 0.0))
    )
    face_id = geometry.add_plate(vertices)
    mesh = _quadratic_result(geometry, (face_id,), h=0.75)
    incidence = {}
    for body in mesh.quads.values():
        pairs = ((0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7))
        for a, b, m in pairs:
            key = tuple(sorted((body[a], body[b])))
            incidence.setdefault(key, []).append(("Q8", body[m]))
    for body in mesh.tris.values():
        for a, b, m in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            key = tuple(sorted((body[a], body[b])))
            incidence.setdefault(key, []).append(("T6", body[m]))
    categories = set()
    for attached in incidence.values():
        assert len({mid for _kind, mid in attached}) == 1
        if len(attached) == 2:
            kinds = tuple(sorted(kind for kind, _mid in attached))
            categories.add(kinds)
    assert ("Q8", "Q8") in categories
    assert ("Q8", "T6") in categories


def test_quadratic_promotion_is_idempotent_on_already_quadratic_mesh():
    from copy import deepcopy
    from anymesher.hybrid import _promote_quad_first_quadratic

    geometry, face_id = p01_geometry()
    mesh = _quadratic_result(geometry, (face_id,), h=0.5)
    before = deepcopy(mesh_to_dict(mesh))
    _promote_quad_first_quadratic(mesh, geometry, target_size=0.5)
    assert mesh_to_dict(mesh) == before


def test_direct_quadratic_promotion_cancellation_leaves_supplied_mesh_unchanged():
    from copy import deepcopy
    from anymesher.hybrid import _promote_quad_first_quadratic

    geometry, face_id = p01_geometry()
    mesh = generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
        order="linear",
    ).mesh
    before = deepcopy(mesh_to_dict(mesh))

    def cancel(stage):
        if stage == "quad-first:quadratic-promotion-ready":
            raise _PromotionCancelled(stage)

    with pytest.raises(_PromotionCancelled):
        _promote_quad_first_quadratic(
            mesh, geometry, target_size=0.5, cancellation_check=cancel
        )
    assert mesh_to_dict(mesh) == before

def test_quadratic_promotion_fails_closed_with_unpromoted_beam_content():
    from anymesher.hybrid import _promote_quad_first_quadratic
    from anymesher.quad.public_integration import QuadPublicUnsupported

    geometry, _face_id = p01_geometry()
    mesh = Mesh(
        geometry_model_id=geometry.model_id,
        geometry_revision=geometry.revision,
        nodes={0: np.zeros(3), 1: np.ones(3)},
        beams={0: (0, 1)},
        order="linear",
    )
    before_nodes = {node: value.copy() for node, value in mesh.nodes.items()}
    with pytest.raises(QuadPublicUnsupported):
        _promote_quad_first_quadratic(mesh, geometry, target_size=0.5)
    assert mesh.order == "linear"
    assert mesh.beams == {0: (0, 1)}
    assert all(np.array_equal(mesh.nodes[node], value) for node, value in before_nodes.items())


def test_public_explicit_quadratic_curved_surface_remains_typed_unsupported():
    from anymesher.quad.public_integration import QuadPublicUnsupported
    from test_curved_native_qualification import _model_face

    geometry, face_id, _surface = _model_face("cylinder")
    before = _source_signature(geometry)
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(
            geometry,
            target_size=0.5,
            face_ids=(face_id,),
            quad_options=QuadMeshingOptions(),
            order="quadratic",
        )
    assert _source_signature(geometry) == before


def test_quadratic_quad_first_with_beam_coupling_uses_qualified_b3_route():
    geometry, face, _part, _sheet, member, _member_edge = _beam_through_face_fixture()
    mesh = generate_hybrid_mesh_result(
        geometry,
        target_size=1.0,
        face_ids=(face,),
        member_ids=(member,),
        quad_options=QuadMeshingOptions(),
        order="quadratic",
    ).mesh
    assert mesh.order == "quadratic"
    assert mesh.beams and all(len(body) == 3 for body in mesh.beams.values())
