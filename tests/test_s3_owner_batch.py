"""Ordinary owner authority and curved projection arithmetic are separate gates."""
import numpy as np
import pytest

from anygeometry import Arc, Cylinder, GeometryModel
from anygeometry.nurbs import NurbsCurve
from anygeometry.errors import GeometryError
from anygeometry.structural import Orientation
from anymesher.mesh import Mesh
from anymesher.errors import MeshError
from anymesher import s3_production as production


def fixture(*, owner=True, reversed=False):
    geometry = GeometryModel()
    curve = NurbsCurve(2, (0.,0.,0.,1.,1.,1.), ((0.,0.,0.),(0.,1.,2.),(0.,3.,0.)))
    ends = [geometry.add_point(*point) for point in curve.evaluate([0.,1.])]
    face, = geometry.extrude((geometry.add_curve(*ends, curve),), (3.,0.,0.))
    if owner:
        geometry.add_sheet((face,), orientations=(Orientation.REVERSED if reversed else Orientation.FORWARD,))
    nodes = {i+1: geometry.face_point(face,*uv) for i,uv in enumerate(
        ((0.,0.),(.5,0.),(1.,0.),(0.,1.),(.5,1.),(1.,1.)))}
    mesh = Mesh(nodes=nodes, quads={1:(1,2,5,4)}, tris={2:(2,3,6),3:(2,6,5)}, elements_of_face={face:[1,2,3]})
    return geometry, mesh, face


def ordinary_fixture(*, owner=True, reversed=False):
    """Actual analytic quarter-cylinder; no curved-cell authority is asserted."""
    geometry = GeometryModel()
    start, via, end = geometry.add_points(((1.,0.,0.),(np.sqrt(.5),np.sqrt(.5),0.),(0.,1.,0.)))
    edge = geometry.add_arc(start, via, end)
    assert type(geometry.edges[edge].curve) is Arc
    face, = geometry.extrude((edge,), (0.,0.,3.))
    assert type(geometry.faces[face].surface) is Cylinder
    if owner:
        geometry.add_sheet((face,), orientations=(Orientation.REVERSED if reversed else Orientation.FORWARD,))
    nodes = {i+1: geometry.face_point(face,*uv) for i,uv in enumerate(
        ((0.,0.),(.5,0.),(1.,0.),(0.,1.),(.5,1.),(1.,1.)))}
    mesh = Mesh(nodes=nodes, quads={1:(1,2,5,4)}, tris={2:(2,3,6),3:(2,6,5)}, elements_of_face={face:[1,2,3]})
    return geometry, mesh, face


@pytest.mark.parametrize('reversed', [False, True])
def test_owner_ids_and_scalar_physical_normals_are_identical(monkeypatch, reversed):
    geometry, mesh, face = ordinary_fixture(reversed=reversed)
    revision = geometry.revision
    batch = production._shell_owner_authority(geometry, mesh)
    monkeypatch.setattr(production, '_shell_owner_projection_uv', lambda *args: {})
    scalar = production._shell_owner_authority(geometry, mesh)
    assert batch == scalar
    assert geometry.revision == revision
    assert all(source['face_id'] == face for source in batch[1].values())
    assert set(batch[0]) == set(batch[1]) == set(mesh.shells) == {1,2,3}
    uses = production._face_use_ids(geometry)[face]
    assert all(source['face_use_ids'] == list(uses) for source in batch[1].values())
    assert all(source['sheet_ids'] == [geometry.face_uses[use].sheet_id for use in uses] for source in batch[1].values())
    for element, normal in batch[0].items():
        centroid = np.mean([mesh.nodes[node] for node in mesh.corners_of(element)], axis=0)
        _, uv, _ = geometry.project_to_face(face, centroid)
        assert np.dot(normal, geometry.face_normal(face,*uv)) * (-1 if reversed else 1) > 0


@pytest.mark.parametrize('fault', ['missing', 'conflict', 'coordinates'])
def test_authority_refusals_match_original_scalar_route(monkeypatch, fault):
    geometry, mesh, face = ordinary_fixture(owner=fault!='missing')
    if fault == 'conflict':
        geometry.add_sheet((face,), orientations=(Orientation.REVERSED,))
    elif fault == 'coordinates':
        mesh.nodes[1] = np.array([np.nan,0.,0.])
    with pytest.raises(production.S3OwnerAuthorityError) as batch:
        production._shell_owner_authority(geometry, mesh)
    monkeypatch.setattr(production, '_shell_owner_projection_uv', lambda *args: {})
    with pytest.raises(production.S3OwnerAuthorityError) as scalar:
        production._shell_owner_authority(geometry, mesh)
    assert str(batch.value) == str(scalar.value)


def _component_normals(geometry, mesh, face_by_element, uv):
    """Component arithmetic ONLY; this is not public curved-cell authority."""
    uses = production._face_use_ids(geometry)
    return {element: production._physical_owner_normal(geometry, mesh,
        element_id=element, face_id=face_by_element[element], face_use_ids=uses,
        projected_uv=uv.get(element)) for element in sorted(mesh.shells)}


def test_original_extruded_component_really_batches_and_matches_scalar_uv_and_normals(monkeypatch):
    geometry, mesh, face = fixture()
    revision = geometry.revision
    face_by_element = production._element_face_map(mesh)
    original = geometry.project_to_face_many
    calls = []
    def project(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(geometry, 'project_to_face_many', project)
    uv = production._shell_owner_projection_uv(geometry, mesh, face_by_element)
    assert len(calls) == 1 and calls[0][0] == face
    assert calls[0][1].shape == (3,3)
    assert set(uv) == set(mesh.shells) == {1,2,3}
    for element in sorted(mesh.shells):
        centroid = np.mean([mesh.nodes[node] for node in mesh.corners_of(element)], axis=0)
        _, scalar_uv, _ = geometry.project_to_face(face, centroid)
        np.testing.assert_array_equal(uv[element], scalar_uv)
    assert _component_normals(geometry, mesh, face_by_element, uv) == _component_normals(geometry, mesh, face_by_element, {})
    assert geometry.revision == revision


def test_optional_batch_failure_replays_original_scalar_component(monkeypatch):
    geometry, mesh, face = fixture()
    face_by_element = production._element_face_map(mesh)
    expected = _component_normals(geometry, mesh, face_by_element, {})
    bulk_calls = []
    def fail(*args):
        bulk_calls.append(args)
        raise GeometryError('optional batch unavailable')
    monkeypatch.setattr(geometry, 'project_to_face_many', fail)
    uv = production._shell_owner_projection_uv(geometry, mesh, face_by_element)
    assert len(bulk_calls) == 1 and bulk_calls[0][0] == face
    assert uv == {}
    original = geometry.project_to_face
    scalar_calls = []
    def project(*args):
        scalar_calls.append(args)
        return original(*args)
    monkeypatch.setattr(geometry, 'project_to_face', project)
    assert _component_normals(geometry, mesh, face_by_element, uv) == expected
    assert len(scalar_calls) == len(mesh.shells)


def test_custom_scalar_projection_protocol_is_not_bypassed(monkeypatch):
    geometry, mesh, face = ordinary_fixture()
    original = geometry.project_to_face
    calls = []
    def project(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(geometry, 'project_to_face', project)
    production._shell_owner_authority(geometry, mesh)
    assert len(calls) == len(mesh.shells)


@pytest.mark.parametrize('forged_metadata', [False, True])
def test_public_manual_extruded_normals_require_actual_owner_cells(monkeypatch, forged_metadata):
    geometry, mesh, face = fixture()
    from anymesher._material_region_binding import curvilinear_reference_candidates, curvilinear_reference_registry
    assert curvilinear_reference_candidates(mesh) == () and curvilinear_reference_registry(mesh) is None
    if forged_metadata:
        # Metadata-only forgery is a NEGATIVE input, never an authorization.
        mesh.geometry_model_id = geometry.model_id
        mesh.geometry_revision = geometry.revision
        reason = 'curvilinear physical normals require actual owner cell authority'
    else:
        assert mesh.geometry_model_id is None and mesh.geometry_revision is None
        reason = 'curvilinear physical normals require the current geometry namespace'
    calls = []
    def no_projection(*args):
        calls.append(args)
        pytest.fail('public curved authority must refuse before optional batch projection')
    monkeypatch.setattr(production, '_shell_owner_projection_uv', no_projection)
    with pytest.raises(MeshError) as caught:
        production._shell_owner_authority(geometry, mesh)
    assert str(caught.value) == reason and calls == []
