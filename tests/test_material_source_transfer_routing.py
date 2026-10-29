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


def test_unmeshed_curved_face_does_not_block_planar_mesh(monkeypatch):
    import anymesher.material_source_transfer as routing
    geometry = SimpleNamespace(faces={
        7: SimpleNamespace(id=7, surface=None, loop=(), holes=()),
        8: SimpleNamespace(id=8, surface=object(), loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: f == 8)
    assert not requires_material_source_transfer(geometry, SimpleNamespace(elements_of_face={7: [1], 8: []}))

def test_unknown_support_requires_owner_transfer(monkeypatch):
    import anymesher.material_source_transfer as routing
    geometry = SimpleNamespace(faces={7: SimpleNamespace(id=7, surface=object(), loop=(), holes=())})
    monkeypatch.setattr(routing, "face_requires_curvilinear", lambda g, f: False)
    assert requires_material_source_transfer(geometry, object())
