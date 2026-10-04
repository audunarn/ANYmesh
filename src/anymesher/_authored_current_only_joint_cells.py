"""Private proof of one CURRENT-ONLY Sheet joint on staged linear cells.

This is a narrow owner-to-cell composition. It does not remap source records,
consume external project references, admit a solver, or publish a mesh.
"""

from copy import deepcopy
from dataclasses import dataclass
import json

from ._authored_component_binding import (
    BoundAuthoredSheetJointComponent, bind_authored_sheet_joint_component,
)
from ._authored_component_stage import _mesh_digest
from .errors import MeshError
from .prepared_current_associations import (
    PreparedCurrentAssociationReceipt,
    query_prepared_current_component_associations,
    validate_prepared_current_component_associations,
)


SCHEMA = "anymesher.current-only-sheet-joint-cells-v1"
_KINDS = ("members", "member_edge_uses", "attachments", "junctions")


@dataclass(frozen=True)
class CurrentOnlySheetJointCells:
    schema: str
    model_id: object
    revision: int
    mesh_digest: str
    authored_roots: tuple[int, int]
    current_faces: tuple[int, ...]
    sheet_ids: tuple[int, int]
    joint_edge_id: int
    joint_chain: tuple[int, ...]
    attachment_ids: tuple[int, int]
    junction_id: int
    occurrence_correspondence: tuple
    sheet_use_segment_cells: tuple
    owner_records_json: str
    owner_component: object
    owner_constraint_scope: object
    current_associations: PreparedCurrentAssociationReceipt
    current_only_joint_cell_binding_qualified: bool = True
    source_reference_transfer_qualified: bool = False
    solver_admitted: bool = False
    publication_qualified: bool = False


def _plain_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _records(component):
    inventory = component.constraint_receipt.inventory
    original_document = component.constraint_receipt.scope.authored_document
    current_document = component.constraint_receipt.scope.current_document
    if ({face["id"] for face in original_document["faces"]}
            != set(component.authored_face_ids)
            or {face["id"] for face in current_document["faces"]}
            != set(component.current_face_ids)
            or any(face["surface"]["type"] != "plane"
                   for face in original_document["faces"])
            or any(edge["curve"]["type"] != "straight"
                   for edge in (*original_document["edges"],
                                *current_document["edges"]))):
        raise MeshError("current-only joint proof is limited to complete Plane/Straight pair")
    if (component.constraint_receipt.outside_root_ids
            or len(component.authored_face_ids) != 2
            or len(component.sheet_ids) != 2
            or len(component.owner_receipt.joint_edge_ids) != 1
            or component.owner_receipt.joint_edge_ids != (component.joint_edge_id,)
            or component.owner_receipt.preserved_joint_attachment_ids
            or component.owner_receipt.preserved_joint_junction_ids):
        raise MeshError("current-only joint needs one complete two-Sheet component")
    original = inventory["original"]
    current = inventory["current"]
    current_tags = current["opaque_unqualified"]["tags"]
    current_edges = {row["id"] for row in current_document["edges"]}
    if (any(original["records"][kind] for kind in _KINDS)
            or any(current["records"][kind] for kind in ("members", "member_edge_uses"))
            or any(inventory[side]["isolated_vertex_ids"] for side in ("original", "current"))
            or original["opaque_unqualified"]["tags"]
            or any(row["entity"][0] != "edge"
                   or row["entity"][1] not in current_edges
                   or row["entity"][1] == component.joint_edge_id
                   or row["values"] != ["intersection_decomposition_seam"]
                   for row in current_tags)
            or component.unqualified_reference_categories not in (
                ("attachments", "junctions"),
                ("attachments", "junctions", "tags"),
            )):
        raise MeshError("current-only joint has unsupported source or opaque references")
    attachments = current["records"]["attachments"]
    junctions = current["records"]["junctions"]
    if (len(attachments) != 2 or len(junctions) != 1
            or tuple(sorted(row["id"] for row in attachments))
            != tuple(component.owner_receipt.attachment_ids)
            or (junctions[0]["id"],) != tuple(component.owner_receipt.junction_ids)
            or {_plain_json(row) for row in attachments}
            != {_plain_json(row) for row in
                component.owner_receipt.current_records["attachments"]}
            or _plain_json(junctions)
            != _plain_json(component.owner_receipt.current_records["junctions"])):
        raise MeshError("current-only joint owner records are missing or unmatched")
    by_sheet = {}
    edge = component.joint_edge_id
    for row in attachments:
        sheet = row["source_id"]
        if (sheet in by_sheet or sheet not in component.sheet_ids
                or row["kind"] != "sheet_on_joint"
                or row["source_kind"] != "sheet"
                or row["target_kind"] != "edge" or row["target_id"] != edge
                or row["target_parameters"] != [[0.0, 1.0]]
                or row["member_id"] is not None
                or row["connection_intent"] != "connect"
                or row["evidence"] != "exact"):
            raise MeshError("current-only joint has unsupported attachment meaning")
        by_sheet[sheet] = row["id"]
    junction = junctions[0]
    if (set(by_sheet) != set(component.sheet_ids)
            or junction["kind"] != "sheet_joint"
            or set(junction["attachment_ids"]) != set(by_sheet.values())
            or set(junction["sheet_ids"]) != set(component.sheet_ids)
            or junction["member_uses"] or junction["connection_intent"] != "connect"):
        raise MeshError("current-only joint has unmatched Junction membership")
    dispositions = [row for row in inventory["record_dispositions"]
                    if row["kind"] in ("attachments", "junctions")]
    if (len(dispositions) != 3
            or any(row["literal_disposition"] != "current_only"
                   or row["semantic_mapping_qualified"]
                   or row["parameter_remapping_qualified"] for row in dispositions)):
        raise MeshError("current-only joint must retain current-only disposition")
    return tuple(sorted(by_sheet.values())), int(junction["id"]), _plain_json({
        "attachments": attachments, "junctions": junctions,
    })


def _edge_pairs(connection):
    return {(min(a, b), max(a, b))
            for a, b in zip(connection, (*connection[1:], connection[0]))}


def query_current_only_sheet_joint_cells(
    geometry, component: BoundAuthoredSheetJointComponent, mesh, registry,
    cell_current_faces, *, current_associations=None, cancellation_check=None,
) -> CurrentOnlySheetJointCells:
    """Bind owner-recorded current-only joint records to both active Sheet cell sides."""
    if not isinstance(component, BoundAuthoredSheetJointComponent):
        raise MeshError("current-only joint needs a bound Sheet component")
    supplied_component = component
    component = deepcopy(supplied_component)
    initial_mesh_digest = _mesh_digest(mesh)
    supplied_associations = current_associations
    if supplied_associations is not None:
        # Owner binding calls cancellation hooks. Capture caller evidence before
        # any such hook can repair a forged value or alter a valid one.
        current_associations = deepcopy(supplied_associations)
    rebound = bind_authored_sheet_joint_component(
        geometry, component.joint_edge_id, component.boundary_correspondences,
        component.authored_face_ids, cancellation_check=cancellation_check,
    )
    if rebound != component:
        raise MeshError("current-only joint component binding changed")
    attachment_ids, junction_id, records_json = _records(rebound)
    if current_associations is None:
        current_associations = query_prepared_current_component_associations(
            geometry, rebound, mesh, registry, cell_current_faces,
            cancellation_check=cancellation_check,
        )
    else:
        validate_prepared_current_component_associations(
            geometry, current_associations, rebound, mesh, registry,
            cell_current_faces, cancellation_check=cancellation_check,
        )
    current = current_associations
    if (current.association_namespace != "prepared_current"
            or current.geometry_model_id != geometry.model_id
            or current.geometry_revision != geometry.revision
            or current.authored_face_ids != rebound.authored_face_ids
            or current.current_face_ids != rebound.current_face_ids
            or current.sheet_ids != rebound.sheet_ids
            or current.owner_component != rebound.owner_receipt
            or current.occurrence_correspondence != rebound.occurrence_correspondence
            or current.source_reference_transfer_qualified
            or current.publication_qualified or current.solver_admitted):
        raise MeshError("current-only joint has wrong current-association context")
    edge = rebound.joint_edge_id
    joints = dict(current.joint_chains)
    chains = dict(current.nodes_of_edge)
    if (set(joints) != {edge} or chains.get(edge) != joints[edge]
            or tuple(mesh.nodes_of_edge[edge]) != joints[edge]
            or tuple(item.node_id for item in registry.entries(edge)) != joints[edge]
            or len(joints[edge]) < 2):
        raise MeshError("current-only joint lacks one exact active station chain")
    chain = joints[edge]
    sheets = dict(current.elements_of_sheet)
    faces = dict(current.cell_current_faces)
    use_cells = dict(current.elements_of_face_use)
    face_roots = {int(face): root for root, correspondence in
                  zip(rebound.authored_face_ids, rebound.boundary_correspondences)
                  for face in correspondence.descendants}
    sheet_roots = {sheet: root for sheet, root, _use, _current in
                   rebound.occurrence_correspondence}
    current_uses = {sheet: tuple(int(use) for use in uses)
                    for sheet, _root, _source_use, uses in
                    rebound.occurrence_correspondence}
    if (set(sheets) != set(rebound.sheet_ids)
            or set(sheet_roots) != set(rebound.sheet_ids)
            or set(use_cells) != {use for uses in current_uses.values()
                                  for use in uses}
            or set(faces) != (set(mesh.tris) | set(mesh.quads))):
        raise MeshError("current-only joint lacks complete cell ownership")
    segment_cells = []
    shells = {**mesh.tris, **mesh.quads}
    for a, b in zip(chain, chain[1:]):
        segment = (min(a, b), max(a, b))
        sides = []
        for sheet in rebound.sheet_ids:
            incident = tuple(cell for cell in sheets[sheet]
                             if segment in _edge_pairs(shells[cell])
                             and face_roots[faces[cell]] == sheet_roots[sheet])
            if len(incident) != 2:
                raise MeshError("current-only joint segment lacks two cells on each Sheet")
            sheet_uses = []
            for cell in incident:
                use = tuple(use for use in current_uses[sheet]
                            if cell in use_cells[use])
                if len(use) != 1:
                    raise MeshError("current-only joint cell lacks one current FaceUse")
                sheet_uses.append(use[0])
                sides.append((sheet, use[0], cell))
            if len(set(sheet_uses)) != 2:
                raise MeshError("current-only joint needs two FaceUse sides per Sheet")
        segment_cells.append((segment, tuple(sides)))
    # Callback-free final reproof pins owner records, mesh digest, cells and
    # stations after all callback-bearing owner/current work has completed.
    final_component = bind_authored_sheet_joint_component(
        geometry, component.joint_edge_id, component.boundary_correspondences,
        component.authored_face_ids,
    )
    if final_component != component or _records(final_component) != (
            attachment_ids, junction_id, records_json):
        raise MeshError("current-only joint owner records changed during proof")
    validate_prepared_current_component_associations(
        geometry, current, final_component, mesh, registry, cell_current_faces,
    )
    if supplied_component != component or _mesh_digest(mesh) != initial_mesh_digest:
        raise MeshError("supplied component or mesh changed during joint proof")
    if (supplied_associations is not None
            and supplied_associations != current_associations):
        raise MeshError("supplied current associations changed during joint proof")
    return CurrentOnlySheetJointCells(
        SCHEMA, geometry.model_id, geometry.revision, current.mesh_digest,
        tuple(rebound.authored_face_ids), tuple(rebound.current_face_ids),
        tuple(rebound.sheet_ids), edge, chain, attachment_ids, junction_id,
        tuple(rebound.occurrence_correspondence), tuple(segment_cells),
        records_json, rebound.owner_receipt,
        rebound.constraint_receipt, current,
    )


def validate_current_only_sheet_joint_cells(
    geometry, receipt: CurrentOnlySheetJointCells, component, mesh,
    registry, cell_current_faces, *, cancellation_check=None,
) -> None:
    """Recompute, never accept a digest or literal record equality alone."""
    if type(receipt) is not CurrentOnlySheetJointCells or receipt.schema != SCHEMA:
        raise MeshError("current-only joint needs an exact versioned receipt")
    # The caller's frozen dataclass can still contain mutable nested records,
    # and object.__setattr__ can replace fields. Pin a detached value before
    # any callback-bearing owner or cell proof runs.
    pinned = deepcopy(receipt)
    expected = query_current_only_sheet_joint_cells(
        geometry, component, mesh, registry, cell_current_faces,
        current_associations=pinned.current_associations,
        cancellation_check=cancellation_check,
    )
    if receipt != pinned or pinned != expected:
        raise MeshError("current-only joint receipt contents changed")
