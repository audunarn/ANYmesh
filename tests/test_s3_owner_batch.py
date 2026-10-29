"""Batching owner projections cannot change S3 physical normal authority."""
import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.nurbs import NurbsCurve
from anygeometry.errors import GeometryError
from anygeometry.structural import Orientation
from anymesher.mesh import Mesh
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


@pytest.mark.parametrize('reversed', [False, True])
def test_owner_ids_and_scalar_physical_normals_are_identical(monkeypatch, reversed):
    geometry, mesh, face = fixture(reversed=reversed)
    revision = geometry.revision
    batch = production._shell_owner_authority(geometry, mesh)
    monkeypatch.setattr(production, '_shell_owner_projection_uv', lambda *args: {})
    scalar = production._shell_owner_authority(geometry, mesh)
    assert batch == scalar
    assert geometry.revision == revision
    assert all(source['face_id'] == face for source in batch[1].values())


@pytest.mark.parametrize('fault', ['missing', 'conflict', 'coordinates'])
def test_authority_refusals_match_original_scalar_route(monkeypatch, fault):
    geometry, mesh, face = fixture(owner=fault!='missing')
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


def test_optional_batch_failure_replays_original_authority(monkeypatch):
    geometry, mesh, face = fixture()
    expected = production._shell_owner_authority(geometry, mesh)
    def fail(*args):
        raise GeometryError('optional batch unavailable')
    monkeypatch.setattr(geometry, 'project_to_face_many', fail)
    assert production._shell_owner_authority(geometry, mesh) == expected


def test_custom_scalar_projection_protocol_is_not_bypassed(monkeypatch):
    geometry, mesh, face = fixture()
    original = geometry.project_to_face
    calls = []
    def project(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(geometry, 'project_to_face', project)
    production._shell_owner_authority(geometry, mesh)
    assert len(calls) == len(mesh.shells)
