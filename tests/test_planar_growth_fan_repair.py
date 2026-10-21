"""Local linear native transitions preserve stations and accept only full gates."""
from copy import deepcopy
import numpy as np
import pytest

from anygeometry import GeometryModel, to_dict
from anymesher.hybrid import _planar_growth_fan_repair, _structured_quality_report
from anymesher.mesh import Mesh, Coupling
from anymesher.serialize import mesh_to_dict
from anymesher.structured import StructuredMeshingOptions


def fixture(width=1.):
    geometry = GeometryModel()
    vertices = geometry.add_points(((0., 0., 0.), (width, 0., 0.), (width, 1., 0.),
                                    (0., 1., 0.), (width+.3, 0., 0.), (width+.3, 1., 0.)))
    a, b, c, d, e, f = vertices
    large = geometry.add_plate((a, b, c, d))
    small = geometry.add_plate((b, e, f, c))
    mesh = Mesh(geometry_model_id=geometry.model_id, geometry_revision=geometry.revision,
                nodes={v: np.array(geometry.vertex_position(v), copy=True) for v in vertices},
                quads={10: (a, b, c, d), 20: (b, e, f, c)},
                node_of_vertex={v: v for v in vertices},
                nodes_of_edge={edge: [body.start, body.end] for edge, body in geometry.edges.items()},
                elements_of_face={large: [10], small: [20]}, elements_of_sheet={7: [10, 20]},
                activity={10: .5}, order='linear')
    return geometry, mesh, large


def repair(geometry, mesh, face, **kwargs):
    options = StructuredMeshingOptions()
    quality = _structured_quality_report(mesh, options)
    assert not quality['accepted'] and quality['growth_violation_count']
    return _planar_growth_fan_repair(geometry, mesh, quality, options,
                                   native_face_ids=(face,), **kwargs)


def test_boundary_protected_fan_preserves_identity_area_ownership_and_source():
    geometry, mesh, face = fixture()
    # Existing node-to-node offsets do not bind the old cell's interpolation.
    mesh.couplings[100] = Coupling.node_to_node(beam_node=1, plate_node=1,
                                             eccentricity=(0., 0., 0.))
    before, source = mesh_to_dict(mesh), to_dict(geometry)
    result, quality, diagnostic = repair(geometry, mesh, face)
    assert result is not None and quality['accepted'] and diagnostic['committed']
    assert mesh_to_dict(mesh) == before and to_dict(geometry) == source
    assert result.quads == {20: mesh.quads[20]}
    ids = tuple(diagnostic['replaced_quads'][10])
    assert ids == (10, 101, 102, 103)
    assert result.elements_of_face[face] == list(ids)
    assert result.elements_of_sheet[7] == [*ids, 20]
    assert all(result.activity[e] == .5 for e in ids)
    assert result.couplings == mesh.couplings
    assert result.nodes_of_edge == mesh.nodes_of_edge
    assert result.node_of_vertex == mesh.node_of_vertex
    for node, xyz in mesh.nodes.items():
        np.testing.assert_array_equal(result.nodes[node], xyz)
    triangles = [np.asarray([result.nodes[n] for n in result.tris[e]]) for e in ids]
    area = sum(np.cross(p[1]-p[0], p[2]-p[0])[2]/2 for p in triangles)
    assert area == pytest.approx(1.)
    edges = {(a, b) for e in ids for a, b in zip(result.tris[e], result.tris[e][1:]+result.tris[e][:1])}
    assert set(zip(mesh.quads[10], mesh.quads[10][1:]+mesh.quads[10][:1])) <= edges
    repeated, _, repeated_diagnostic = repair(geometry, mesh, face)
    assert mesh_to_dict(repeated) == mesh_to_dict(result)
    assert repeated_diagnostic == diagnostic


@pytest.mark.parametrize('unsupported', ['quadratic', 'mapped_grid', 'member_owner', 'interpolated', 'connected'])
def test_unsupported_or_already_bound_candidate_is_not_changed(unsupported):
    geometry, mesh, face = fixture()
    options = StructuredMeshingOptions()
    quality = _structured_quality_report(mesh, options)
    if unsupported == 'quadratic':
        mesh.order = 'quadratic'
    elif unsupported == 'mapped_grid':
        mesh.grid_of_face[face] = np.asarray(mesh.quads[10]).reshape(2, 2)
    elif unsupported == 'member_owner':
        mesh.elements_of_member[9] = [10]
    elif unsupported == 'interpolated':
        mesh.couplings[99] = Coupling(beam_node=1, plate_nodes=(1, 2), weights=(.5, .5),
                                      eccentricity=(0., 0., 0.))
    else:
        mesh.automatic_shell_connections = 1
    before = mesh_to_dict(mesh)
    result, _, diagnostic = _planar_growth_fan_repair(
        geometry, mesh, quality, options, native_face_ids=(face,))
    assert result is None and not diagnostic['committed']
    assert mesh_to_dict(mesh) == before


def test_fan_quality_worsening_is_refused_without_partial_publication():
    geometry, mesh, face = fixture(width=4.)
    before = mesh_to_dict(mesh)
    result, _, diagnostic = repair(geometry, mesh, face)
    assert result is None and diagnostic['attempted'] and not diagnostic['committed']
    assert mesh_to_dict(mesh) == before


def test_new_growth_violation_cannot_bypass_final_gate(monkeypatch):
    import anymesher.hybrid as hybrid
    geometry, mesh, face = fixture()
    options = StructuredMeshingOptions()
    original = _structured_quality_report(mesh, options)
    actual = hybrid._structured_quality_report
    def reject_fan(candidate, policy):
        report = actual(candidate, policy)
        # An additional neighbour would appear only after subdivision. It is
        # deliberately unrepairable (triangles only), so keep truthful refusal.
        report.update(accepted=False, growth_violation_count=1,
                      growth_violation_pairs=[[10, 21, 2.]], maximum_adjacent_element_growth=2.)
        return report
    monkeypatch.setattr(hybrid, '_structured_quality_report', reject_fan)
    result, _, diagnostic = _planar_growth_fan_repair(
        geometry, mesh, original, options, native_face_ids=(face,))
    assert result is None and not diagnostic['committed']


def test_cancellation_keeps_original_mesh_and_exception_identity():
    geometry, mesh, face = fixture()
    before = mesh_to_dict(mesh)
    marker = RuntimeError('fan cancelled')
    def cancel(phase):
        if phase == 'planar growth fan validity':
            raise marker
    with pytest.raises(RuntimeError) as caught:
        repair(geometry, mesh, face, cancellation_check=cancel)
    assert caught.value is marker and mesh_to_dict(mesh) == before
