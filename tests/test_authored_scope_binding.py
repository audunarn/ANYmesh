"""Complete owner visibility is a prerequisite, never mesh permission."""

import pytest

import anygeometry as owner
from anymesher._authored_scope_binding import bind_authored_root_inputs
from anymesher.errors import MeshError


def prepared(*, member=False, crossing=False, isolated=False):
    model = owner.GeometryModel()
    root = model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0),
                                             (2, 2, 0), (0, 2, 0))))
    if crossing:
        model.add_plate(model.add_points(((1, -1, -1), (1, 3, -1),
                                          (1, 3, 1), (1, -1, 1))))
    if member:
        first, second = model.add_points(((4, 0, 0), (4, 0, 1)))
        model.add_member((model.add_line(first, second),))
    if isolated:
        model.add_point(20, 20, 20)
    plan = owner.plan_intersections(model, tuple(model.faces), policy="connect")
    owner.apply_intersections(model, plan, policy="connect")
    binding = owner.query_prepared_authored_boundary_correspondence(model, root)
    return model, binding


def test_simple_complete_owner_scope_is_bound_but_not_publishable():
    model, correspondence = prepared()
    before = owner.to_dict(model)
    bound = bind_authored_root_inputs(model, correspondence, correspondence.descendants)
    assert bound.authored_face == correspondence.authored_definition.face_id
    assert bound.descendants == correspondence.descendants
    assert bound.publication_qualified is False
    assert bound.scope.authored_document["faces"]
    assert bound.constraint_receipt.selected_root_ids == (bound.authored_face,)
    assert bound.constraint_receipt.current_face_ids == bound.descendants
    assert bound.constraint_receipt.typed_inventory_complete
    assert not bound.constraint_receipt.semantic_mapping_qualified
    assert owner.to_dict(model) == before


def test_partial_descendant_selection_refuses_before_reference_interpretation():
    model, correspondence = prepared(crossing=True)
    assert len(correspondence.descendants) > 1
    with pytest.raises(MeshError, match="every current descendant"):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants[:1])
    with pytest.raises(MeshError, match="every current descendant"):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants * 2)


def test_member_and_physical_interface_need_separate_consumers():
    model, correspondence = prepared(member=True)
    with pytest.raises(MeshError, match="members need a qualified consumer"):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants)
    model, correspondence = prepared(crossing=True)
    with pytest.raises(MeshError, match="attachments need a qualified consumer|junctions need a qualified consumer|atomic shared publication"):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants)
    model, correspondence = prepared(isolated=True)
    with pytest.raises(MeshError, match="isolated vertices need a qualified consumer"):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants)


def test_face_corner_offsets_do_not_hide_an_isolated_vertex():
    model = owner.GeometryModel()
    isolated = model.add_point(20, 20, 20)
    root = model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0),
                                             (2, 2, 0), (0, 2, 0))))
    owner.apply_intersections(
        model, owner.plan_intersections(model, (root,), policy="connect"),
        policy="connect")
    correspondence = owner.query_prepared_authored_boundary_correspondence(model, root)
    receipt = owner.query_prepared_authored_constraint_scope(model, (root,))
    assert isolated in receipt.inventory["original"]["isolated_vertex_ids"]
    with pytest.raises(MeshError, match="isolated vertices need a qualified consumer"):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants)


def test_missing_scope_capability_and_stale_binding_refuse(monkeypatch):
    model, correspondence = prepared()
    with monkeypatch.context() as patch:
        patch.delattr(owner, "query_prepared_authored_constraint_scope")
        with pytest.raises(MeshError, match="scope capability is unavailable"):
            bind_authored_root_inputs(model, correspondence, correspondence.descendants)
    model.add_point(10, 10, 10)
    with pytest.raises(owner.GeometryError):
        bind_authored_root_inputs(model, correspondence, correspondence.descendants)
