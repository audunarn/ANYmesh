"""Original owner associations must not be inferred from descendant cells."""

import numpy as np
import pytest

import anygeometry as owner
from anymesher._authored_associations import plan_authored_root_associations
from anymesher._authored_scope_binding import bind_authored_root_inputs
from anymesher.errors import MeshError
from anymesher.mesh import Mesh


def prepared(*, sheet=True, reversed_use=False):
    model = owner.GeometryModel()
    root = model.add_plate(model.add_points(
        ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))))
    if sheet:
        orientation = owner.Orientation.REVERSED if reversed_use else owner.Orientation.FORWARD
        model.add_sheet((root,), orientations=(orientation,))
    plan = owner.plan_intersections(model, tuple(model.faces), policy="connect")
    owner.apply_intersections(model, plan, policy="connect")
    correspondence = owner.query_prepared_authored_boundary_correspondence(model, root)
    bound = bind_authored_root_inputs(model, correspondence, correspondence.descendants)
    return model, bound


def root_mesh():
    return Mesh(nodes={1: np.array((0., 0., 0.)),
                       2: np.array((2., 0., 0.)),
                       3: np.array((0., 2., 0.))}, tris={7: (1, 2, 3)})


@pytest.mark.parametrize("sheet,reversed_use,expected_orientation", [
    (False, False, ()), (True, False, ("forward",)),
    (True, True, ("reversed",)),
])
def test_plan_retains_only_original_face_and_face_use_meaning(
    sheet, reversed_use, expected_orientation,
):
    model, bound = prepared(sheet=sheet, reversed_use=reversed_use)
    mesh = root_mesh()
    before = owner.to_dict(model)
    association = plan_authored_root_associations(model, bound, mesh, [7])
    assert association.source_face == bound.authored_face
    assert association.element_ids == (7,)
    assert tuple(use[2] for use in association.face_uses) == expected_orientation
    assert association.sheet_ids == ((1,) if sheet else ())
    assert association.publication_qualified is False
    staged = association.stage_mesh(mesh)
    assert staged.elements_of_face == {bound.authored_face: [7]}
    assert staged.elements_of_sheet == ({1: [7]} if sheet else {})
    assert owner.to_dict(model) == before
    assert mesh.elements_of_face == {}


def test_partial_or_preassociated_candidate_refuses_without_mutation():
    model, bound = prepared()
    mesh = root_mesh()
    for ids in ((), (7, 7), (8,)):
        with pytest.raises(MeshError, match="every root-local shell cell"):
            plan_authored_root_associations(model, bound, mesh, ids)
    mesh.elements_of_face[bound.descendants[0]] = [7]
    with pytest.raises(MeshError, match="unassociated root-local"):
        plan_authored_root_associations(model, bound, mesh, [7])
    model, bound = prepared()
    association = plan_authored_root_associations(model, bound, root_mesh(), [7])
    with pytest.raises(MeshError, match="candidate changed"):
        association.stage_mesh(mesh)
    assert mesh.elements_of_face == {bound.descendants[0]: [7]}


def test_stale_owner_scope_refuses():
    model, bound = prepared()
    model.add_point(10, 10, 10)
    with pytest.raises(owner.GeometryError):
        plan_authored_root_associations(model, bound, root_mesh(), [7])
