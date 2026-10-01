from anygeometry import (
    GeometryModel, ConnectionIntent, plan_intersections, apply_intersections,
    has_current_intersection_preparation, to_dict,
)
from anymesher import preparation


def test_detached_prepared_material_uses_current_owner_binding(monkeypatch):
    model = GeometryModel()
    for points in (
        ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)),
        ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)),
    ):
        face = model.add_plate(model.add_points(points))
        model.add_sheet((face,))
    plan = plan_intersections(model, tuple(model.faces), policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert has_current_intersection_preparation(model)
    before = to_dict(model)
    def unnecessary_classification(*args, **kwargs):
        raise AssertionError("current prepared material was re-classified")
    monkeypatch.setattr(preparation, "find_coplanar_overlaps", unnecessary_classification)
    monkeypatch.setattr(preparation, "plan_intersections", unnecessary_classification)
    working, report = preparation.prepare_structural_closure(
        model, face_ids=tuple(model.faces), options=False)
    assert working.validate_topology() == ()
    assert to_dict(model) == before
