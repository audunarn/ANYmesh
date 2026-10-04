"""Current-only owner joint records bind to manual linear cells, never routing."""

from dataclasses import replace
import json

import pytest

from test_authored_component_cells import candidate, created_candidate
from anymesher._authored_current_only_joint_cells import (
    _records, query_current_only_sheet_joint_cells,
    validate_current_only_sheet_joint_cells,
)
from anymesher.errors import MeshError
from anygeometry import GeometryError


@pytest.fixture(scope="module")
def owner_record_component():
    return candidate()[1]


def test_current_only_joint_records_bind_each_segment_sheet_use_and_cell():
    geometry, component, mesh, registry, faces = candidate()
    receipt = component.bind_current_only_joint_cells(
        geometry, mesh, registry, faces)
    assert receipt.schema == "anymesher.current-only-sheet-joint-cells-v1"
    assert receipt.authored_roots == component.authored_face_ids
    assert receipt.current_faces == component.current_face_ids
    assert receipt.attachment_ids == component.owner_receipt.attachment_ids
    assert receipt.junction_id == component.owner_receipt.junction_ids[0]
    assert receipt.joint_chain == tuple(mesh.nodes_of_edge[component.joint_edge_id])
    assert receipt.occurrence_correspondence == component.occurrence_correspondence
    assert len(receipt.sheet_use_segment_cells) == len(receipt.joint_chain) - 1
    assert all({sheet for sheet, _use, _cell in sides} == set(component.sheet_ids)
               for _segment, sides in receipt.sheet_use_segment_cells)
    assert all(len([1 for side_sheet, _use, _cell in sides if side_sheet == sheet]) == 2
               for _segment, sides in receipt.sheet_use_segment_cells
               for sheet in component.sheet_ids)
    assert all(cell in dict(receipt.current_associations.elements_of_face_use)[use]
               for _segment, sides in receipt.sheet_use_segment_cells
               for _sheet, use, cell in sides)
    assert receipt.current_only_joint_cell_binding_qualified
    assert not receipt.source_reference_transfer_qualified
    assert not receipt.solver_admitted and not receipt.publication_qualified
    validate_current_only_sheet_joint_cells(
        geometry, receipt, component, mesh, registry, faces)


@pytest.mark.parametrize("change", (
    "missing", "duplicate", "unmatched", "source_member", "foreign_tag",
))
def test_incomplete_or_unsupported_owner_records_refuse(change, owner_record_component):
    component = owner_record_component
    inventory = component.constraint_receipt.inventory
    original = inventory["original"]["records"]
    current = inventory["current"]["records"]
    if change == "missing":
        current["attachments"].pop()
    elif change == "duplicate":
        current["attachments"].append(dict(current["attachments"][0]))
    elif change == "unmatched":
        current["attachments"][0]["target_id"] += 1
    elif change == "source_member":
        original["members"].append({"id": 999})
    else:
        inventory["current"]["opaque_unqualified"]["tags"][0]["values"] = ["foreign"]
    forged = replace(component, constraint_receipt=replace(
        component.constraint_receipt,
        inventory_json=json.dumps(inventory, sort_keys=True)))
    with pytest.raises(MeshError, match="unsupported source|missing or unmatched"):
        _records(forged)


def test_known_seam_tags_are_optional_for_current_only_record_shape(
    owner_record_component,
):
    component = owner_record_component
    inventory = component.constraint_receipt.inventory
    inventory["current"]["opaque_unqualified"]["tags"] = []
    no_tags = replace(component, constraint_receipt=replace(
        component.constraint_receipt,
        inventory_json=json.dumps(inventory, sort_keys=True)))
    assert no_tags.unqualified_reference_categories == ("attachments", "junctions")
    assert _records(no_tags)[0] == component.owner_receipt.attachment_ids


def test_forged_receipt_changed_chain_and_stale_geometry_refuse():
    geometry, component, mesh, registry, faces = candidate()
    receipt = query_current_only_sheet_joint_cells(
        geometry, component, mesh, registry, faces)
    with pytest.raises(MeshError, match="receipt contents changed"):
        validate_current_only_sheet_joint_cells(
            geometry, replace(receipt, junction_id=999), component,
            mesh, registry, faces)
    mesh.nodes_of_edge[component.joint_edge_id].reverse()
    with pytest.raises(MeshError, match="station order"):
        validate_current_only_sheet_joint_cells(
            geometry, receipt, component, mesh, registry, faces)
    geometry, component, mesh, registry, faces = candidate()
    geometry.add_point(20, 20, 20)
    with pytest.raises(GeometryError, match="complete current preparation"):
        query_current_only_sheet_joint_cells(
            geometry, component, mesh, registry, faces)


def test_created_uv_current_association_round_trips_through_joint_receipt():
    import anymesher._authored_current_only_joint_cells as route

    geometry, component, mesh, registry, faces, created, _node = created_candidate()
    current = route.query_prepared_current_component_associations(
        geometry, component, mesh, registry, faces,
        created_material_uv_by_root=created,
    )
    assert current.schema == "anymesher.prepared-current-associations-v2"
    receipt = query_current_only_sheet_joint_cells(
        geometry, component, mesh, registry, faces,
        current_associations=current,
    )
    validate_current_only_sheet_joint_cells(
        geometry, receipt, component, mesh, registry, faces)


def test_one_sided_boundary_shape_refuses_interior_crossing_contract():
    import anymesher._authored_current_only_joint_cells as route

    geometry, component, mesh, registry, faces = candidate()
    current = route.query_prepared_current_component_associations(
        geometry, component, mesh, registry, faces)
    sheet, cells = current.elements_of_sheet[0]
    segment = tuple(sorted(current.joint_chains[0][1]))
    incident = [cell for cell in cells
                if segment in route._edge_pairs(mesh.tris[cell])]
    assert len(incident) == 2
    one_sided = replace(current, elements_of_sheet=(
        (sheet, tuple(cell for cell in cells if cell != incident[0])),
        current.elements_of_sheet[1],
    ))
    with pytest.raises(MeshError, match="contents or owner binding changed"):
        query_current_only_sheet_joint_cells(
            geometry, component, mesh, registry, faces,
            current_associations=one_sided)


@pytest.mark.parametrize("mutation", ("repair_forgery", "corrupt_valid"))
def test_callback_cannot_change_receipt_during_validation(mutation):
    geometry, component, mesh, registry, faces = candidate()
    valid = query_current_only_sheet_joint_cells(
        geometry, component, mesh, registry, faces)
    receipt = (replace(valid, junction_id=999)
               if mutation == "repair_forgery" else valid)
    changed = False

    def mutate(_stage):
        nonlocal changed
        if not changed:
            changed = True
            object.__setattr__(receipt, "junction_id",
                               valid.junction_id if mutation == "repair_forgery"
                               else 999)
        return False

    with pytest.raises(MeshError, match="receipt contents changed"):
        validate_current_only_sheet_joint_cells(
            geometry, receipt, component, mesh, registry, faces,
            cancellation_check=mutate,
        )
    assert changed


@pytest.mark.parametrize("mutation", ("repair_forgery", "corrupt_valid"))
def test_supplied_association_is_pinned_before_owner_callbacks(mutation):
    import anymesher._authored_current_only_joint_cells as route

    geometry, component, mesh, registry, faces = candidate()
    valid = route.query_prepared_current_component_associations(
        geometry, component, mesh, registry, faces)
    supplied = (replace(valid, mesh_digest="forged")
                if mutation == "repair_forgery" else valid)
    changed = False

    def mutate(_stage):
        nonlocal changed
        if not changed:
            changed = True
            object.__setattr__(supplied, "mesh_digest",
                               valid.mesh_digest if mutation == "repair_forgery"
                               else "changed")
        return False

    with pytest.raises(MeshError, match=(
        "contents or owner binding changed" if mutation == "repair_forgery"
        else "supplied current associations changed during joint proof"
    )):
        query_current_only_sheet_joint_cells(
            geometry, component, mesh, registry, faces,
            current_associations=supplied,
            cancellation_check=mutate,
        )
    assert changed


@pytest.mark.parametrize("mutation", (
    "repair_component", "corrupt_component", "repair_mesh", "corrupt_mesh",
))
def test_component_and_mesh_are_pinned_before_owner_callbacks(mutation):
    geometry, valid_component, mesh, registry, faces = candidate()
    component = (replace(valid_component, current_face_ids=())
                 if mutation == "repair_component" else valid_component)
    edge = valid_component.joint_edge_id
    if mutation == "repair_mesh":
        mesh.nodes_of_edge[edge].reverse()
    changed = False

    def mutate(_stage):
        nonlocal changed
        if not changed:
            changed = True
            if mutation in ("repair_component", "corrupt_component"):
                object.__setattr__(
                    component, "current_face_ids",
                    valid_component.current_face_ids
                    if mutation == "repair_component" else (),
                )
            else:
                mesh.nodes_of_edge[edge].reverse()
        return False

    with pytest.raises(MeshError):
        query_current_only_sheet_joint_cells(
            geometry, component, mesh, registry, faces,
            cancellation_check=mutate,
        )
    assert changed


def test_late_callback_mesh_mutation_refuses(monkeypatch):
    import anymesher._authored_current_only_joint_cells as route

    geometry, component, mesh, registry, faces = candidate()
    current = route.query_prepared_current_component_associations(
        geometry, component, mesh, registry, faces)
    original = route.validate_prepared_current_component_associations
    changed = False

    def late_validate(*args, cancellation_check=None, **kwargs):
        nonlocal changed
        result = original(*args, cancellation_check=cancellation_check, **kwargs)
        if cancellation_check is not None and not changed:
            changed = True
            node = next(iter(mesh.nodes))
            mesh.nodes[node] = mesh.nodes[node] + (0., 0., .01)
        return result

    monkeypatch.setattr(route, "validate_prepared_current_component_associations",
                        late_validate)
    with pytest.raises(MeshError):
        route.query_current_only_sheet_joint_cells(
            geometry, component, mesh, registry, faces,
            current_associations=current,
            cancellation_check=lambda stage: False)
    assert changed
