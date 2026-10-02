import pytest

from anygeometry import (GeometryModel, clone_prepared_geometry,
    has_current_intersection_preparation, to_dict)
from anygeometry.generators import cylinder
from anymesher import preparation


def mixed_wall_pipe():
    model = GeometryModel()
    model.add_plate(model.add_points(((-3,-2,0),(7,-2,0),(7,4,0),(-3,4,0))))
    points = model.add_points(((0,0,0),(1,2,0),(2,-1,0),(3,1,0)))
    edge = model.add_spline(points[0], points[1:-1], points[-1])
    model.extrude((edge,), (.25,0,1.5))
    pipe = cylinder(.7,8,origin=(-2,.4,.6),axis=(1,.2,.1),
                    radial_direction=(0,1,0),circumferential_segments=8)
    for member in tuple(pipe.members):
        pipe.remove_member(member)
    model.insert_model(pipe)
    return model


def test_final_corner_edits_keep_owner_proof_through_attempt_clone(monkeypatch):
    source = mixed_wall_pipe()
    authored = to_dict(source)
    prepared, _ = preparation.prepare_structural_closure(source)
    assert to_dict(source) == authored
    assert has_current_intersection_preparation(prepared)
    assert any(len(face.corners) == 4 for face in prepared.faces.values())
    attempt = clone_prepared_geometry(prepared)
    assert has_current_intersection_preparation(attempt)
    def forbidden(*args, **kwargs):
        raise AssertionError('material reclassified after certified corner finalization')
    monkeypatch.setattr(preparation, 'plan_intersections', forbidden)
    monkeypatch.setattr(preparation, 'find_coplanar_overlaps', forbidden)
    preparation.prepare_structural_closure(attempt, options=False)
    attempt.add_point(100,100,100)
    assert not has_current_intersection_preparation(attempt)
    with pytest.raises(AssertionError, match='reclassified'):
        preparation.prepare_structural_closure(attempt, options=False)
