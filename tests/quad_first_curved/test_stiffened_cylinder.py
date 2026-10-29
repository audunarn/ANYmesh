"""Linear cylindrical source-boundary ownership through the public hybrid route."""
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from anygeometry.curves import Straight
from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange
from anymesher.errors import MeshError
from anymesher.hybrid import (generate_hybrid_mesh_result,
    _merge_quad_first_and_legacy, _validate_linear_boundary_beam_ownership)
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher.serialize import mesh_to_dict, mesh_from_dict
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state


def fixture(reverse=False):
    model, selected = _sector_model(False)
    faces = tuple(model.face_uses[use.id].face_id for use in selected)
    edges = tuple(sorted({use.edge for face in model.faces.values() for use in face.loop}))
    part = next(iter(model.parts))
    for edge in edges:
        member = model.add_member((edge,), part_id=part)
        if reverse:
            model.reverse_member(member)
    return model, faces, edges


def generate(model, faces, edges, **kwargs):
    return generate_hybrid_mesh_result(model, target_size=.5,
        face_ids=faces, beam_edges=edges, strategy='native', native_backend='python',
        quad_options=QuadMeshingOptions(), **kwargs)


def assert_ownership(model, mesh, faces, edges):
    counts = Counter(e for edge in edges for e in mesh.elements_of_edge[edge])
    assert set(counts) == set(mesh.beams)
    assert set(counts.values()) == {1}
    for edge in edges:
        axis = mesh.nodes_of_edge[edge]
        assert [mesh.beams[e] for e in mesh.elements_of_edge[edge]] == list(zip(axis, axis[1:]))
        for face in model.faces_using_edge(edge):
            shell_nodes = {n for e in mesh.elements_of_face[face]
                           for n in (mesh.quads.get(e) or mesh.tris[e])}
            assert set(axis) <= shell_nodes
    _validate_linear_boundary_beam_ownership(model, mesh, frozenset(faces))


@pytest.mark.parametrize('reverse', [False, True])
def test_cylinder_rings_seam_and_member_reversal_share_exact_station_ids(reverse):
    model, faces, edges = fixture(reverse)
    before = _persistent_state(model)
    mesh = generate(model, faces, edges).mesh
    assert any(not isinstance(model.edges[e].curve, Straight) for e in edges)
    assert any(len(model.faces_using_edge(e)) == 2 for e in edges)
    assert_ownership(model, mesh, faces, edges)
    assert _persistent_state(model) == before
    restored = mesh_from_dict(mesh_to_dict(mesh))
    assert restored.beams == mesh.beams
    assert restored.nodes_of_edge == mesh.nodes_of_edge
    # Member traversal changes group order, never source-edge direction.
    for member, elements in mesh.elements_of_member.items():
        use = model.member_edge_uses[model.members[member].edge_use_ids[0]]
        expected = list(mesh.elements_of_edge[use.edge_id])
        assert elements == (list(reversed(expected)) if reverse else expected)


def test_unrelated_coincident_source_edge_is_not_welded():
    model, faces, _edges = fixture()
    edge = next(e for e in model.edges.values() if isinstance(e.curve, Straight))
    a, b = (model.vertices[v].position for v in (edge.start, edge.end))
    separate = model.add_line(model.add_point(*a), model.add_point(*b))
    shell = generate(model, faces, (), member_ids=())
    beam = generate_hybrid_mesh_result(model, target_size=.5, face_ids=(),
        beam_edges=(separate,), member_ids=(), strategy='native', native_backend='python')
    merged = _merge_quad_first_and_legacy(model, quad_faces=frozenset(faces),
        legacy_faces=frozenset(), quad_result=shell, legacy_result=beam).mesh
    assert set(merged.nodes_of_edge[separate]).isdisjoint(merged.nodes_of_edge[edge.id])
    np.testing.assert_allclose([merged.nodes[n] for n in merged.nodes_of_edge[separate]],
                              [merged.nodes[n] for n in merged.nodes_of_edge[edge.id]], atol=1e-12)


def test_incomplete_boundary_coverage_refuses():
    model, faces, edges = fixture()
    mesh = generate(model, faces, edges).mesh
    edge = edges[0]
    damaged = deepcopy(mesh)
    damaged.elements_of_edge[edge] = damaged.elements_of_edge[edge][1:]
    with pytest.raises(MeshError, match='incomplete linear boundary beam coverage'):
        _validate_linear_boundary_beam_ownership(model, damaged, frozenset(faces))
    damaged = deepcopy(mesh)
    node = damaged.nodes_of_edge[edge][1]
    damaged.nodes_of_edge[edge][1] = next(n for n in damaged.nodes if n not in damaged.nodes_of_edge[edge])
    with pytest.raises(MeshError, match='station chain|incident shell boundary'):
        _validate_linear_boundary_beam_ownership(model, damaged, frozenset(faces))
    assert node in mesh.nodes_of_edge[edge]
    damaged = deepcopy(mesh)
    damaged.elements_of_face[model.faces_using_edge(edge)[0]] = []
    with pytest.raises(MeshError, match='incident shell boundary'):
        _validate_linear_boundary_beam_ownership(model, damaged, frozenset(faces))


def test_quadratic_cylinder_source_beams_stay_refused_and_linear_cancel_is_atomic():
    model, faces, edges = fixture()
    before = _persistent_state(model)
    with pytest.raises(QuadPublicUnsupported, match='quadratic cylindrical'):
        generate(model, faces, edges, order='quadratic')
    marker = RuntimeError('stop before beams')
    def cancel(stage):
        if stage == 'quad-first:boundary-beam-validation':
            raise marker
    with pytest.raises(RuntimeError) as caught:
        generate(model, faces, edges, cancellation_check=cancel)
    assert caught.value is marker
    assert _persistent_state(model) == before


def test_scalar_offset_uses_exact_cylindrical_normal_without_shell_beam_pass():
    model, faces, edges = fixture()
    for member in model.members.values():
        use = model.member_edge_uses[member.edge_use_ids[0]]
        model.add_attachment(member.id, AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
            AttachmentTargetKind.EDGE, use.edge_id, ParameterRange(0., 1.),
            (ParameterRange(0., 1.),))
    before = _persistent_state(model)
    mesh = generate(model, faces, edges, beam_offsets={edge: .1 for edge in edges}).mesh
    couplings = {coupling.beam_node: coupling for coupling in mesh.couplings.values()}
    for edge in edges:
        axis, offset = mesh.nodes_of_edge[edge], mesh.offset_nodes_of_edge[edge]
        assert len(axis) == len(offset)
        for shell, beam in zip(axis, offset):
            point = np.asarray(mesh.nodes[shell])
            # The geometry owner's projection/normal is authoritative. At a
            # chart endpoint its numerical inverse can differ slightly from
            # the ideal radial direction; do not impose a new geometry tolerance.
            normals = []
            for face in model.faces_using_edge(edge):
                _projection, uv, _distance = model.project_to_face(face, point)
                normals.append(model.face_normal(face, *uv))
            normal = np.sum(normals, axis=0)
            normal /= np.linalg.norm(normal)
            np.testing.assert_allclose(mesh.nodes[beam] - point, .1 * normal, atol=1e-10, rtol=0)
            coupling = couplings[beam]
            reconstructed = sum((w * mesh.nodes[n] for n, w in zip(coupling.plate_nodes, coupling.weights)), np.zeros(3))
            np.testing.assert_allclose(reconstructed, point, atol=1e-10, rtol=0)
            np.testing.assert_allclose(coupling.eccentricity, .1 * normal, atol=1e-10, rtol=0)
    assert _persistent_state(model) == before
