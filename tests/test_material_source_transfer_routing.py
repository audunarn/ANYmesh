"""The routing query grants no source-publication authority."""
from types import SimpleNamespace
import pytest
from anygeometry import GeometryModel, ExtrudedSurface, GeometryError
from anymesher import requires_material_source_transfer

def test_empty_geometry_needs_no_transfer():
    assert not requires_material_source_transfer(GeometryModel(), object())

@pytest.mark.parametrize("curved", [False, True])
def test_owner_face_classification_is_respected(monkeypatch, curved):
    import anymesher.material_source_transfer as routing
    geometry = SimpleNamespace(faces={7: SimpleNamespace(id=7, surface=None, loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: curved)
    assert requires_material_source_transfer(geometry, object()) is curved

def test_failed_owner_region_query_cannot_allow_generic_remap(monkeypatch):
    import anygeometry
    import anymesher.material_source_transfer as routing
    # Routing only inspects the support type; region validation is owner work.
    surface = object.__new__(ExtrudedSurface)
    geometry = SimpleNamespace(faces={7: SimpleNamespace(id=7, surface=surface, loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: False)
    def reject(*args):
        raise GeometryError("unqualified material chart")
    monkeypatch.setattr(anygeometry, "query_material_surface_regions", reject)
    assert requires_material_source_transfer(geometry, object()) is True


def test_unmeshed_curved_face_labels_do_not_exempt_work_closure(monkeypatch):
    import anymesher.material_source_transfer as routing
    geometry = SimpleNamespace(faces={
        7: SimpleNamespace(id=7, surface=None, loop=(), holes=()),
        8: SimpleNamespace(id=8, surface=object(), loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: f == 8)
    # An empty mutable label is not immutable evidence of a selected subset.
    assert requires_material_source_transfer(geometry, SimpleNamespace(elements_of_face={7: [1], 8: []}))

def test_unknown_support_requires_owner_transfer(monkeypatch):
    import anymesher.material_source_transfer as routing
    geometry = SimpleNamespace(faces={7: SimpleNamespace(id=7, surface=object(), loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: False)
    assert requires_material_source_transfer(geometry, object())


def test_planar_cell_labels_cannot_hide_curved_work_closure(monkeypatch):
    import anymesher.material_source_transfer as routing
    geometry = SimpleNamespace(faces={
        7: SimpleNamespace(id=7, surface=None, loop=(), holes=()),
        8: SimpleNamespace(id=8, surface=object(), loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: f == 8)
    assert requires_material_source_transfer(geometry, SimpleNamespace(elements_of_face={7: [1]}))


def test_actual_selected_planar_closure_stays_generic_with_curved_face_elsewhere():
    from anygeometry import extract_model_closure, to_dict
    from anymesher.mesh import Mesh
    from test_nurbs_curvilinear_consumer import patch_model
    source, curved, _edges = patch_model(warp=.1)
    planar = source.add_plate(source.add_points(((2.,0.,0.),(3.,0.,0.),(3.,1.,0.),(2.,1.,0.))))
    before = to_dict(source)
    closure = extract_model_closure(source, (source.handle('face', planar),),
                                    include_structural_closure=True, include_features=False)
    assert len(closure.working_model.faces) == 1
    mesh = Mesh(geometry_model_id=closure.working_model.model_id,
                geometry_revision=closure.working_model.revision)
    assert not requires_material_source_transfer(closure.working_model, mesh)
    assert to_dict(source) == before


@pytest.mark.parametrize('explicit', [False, True])
def test_only_implicit_straight_coons_marker_uses_generic_definition(monkeypatch, explicit):
    import anygeometry
    from anygeometry import CoonsSurface
    source = GeometryModel()
    corners = ((0.,0.,0.),(1.,0.,0.),(1.,1.,.1),(0.,1.,0.))
    points = source.add_points(corners)
    edges = tuple(source.add_line(points[i],points[(i+1)%4]) for i in range(4))
    surface = (CoonsSurface(bottom=(corners[0],corners[1]), right=(corners[1],corners[2]),
                            top=(corners[3],corners[2]), left=(corners[0],corners[3]))
               if explicit else CoonsSurface())
    source.add_face(edges, surface=surface)
    def reject(*args, **kwargs):
        raise GeometryError('explicit material chart was not qualified')
    monkeypatch.setattr(anygeometry, 'query_material_surface_regions', reject)
    assert requires_material_source_transfer(source, object()) is explicit
