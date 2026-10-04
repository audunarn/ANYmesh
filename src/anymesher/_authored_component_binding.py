"""Private complete-component preflight for prepared authored Sheet joints.

This binds owner coverage and occurrence meaning, not cell association remap,
meshing permission, or publication. No root-only fallback is permitted.
"""

from dataclasses import dataclass
from numbers import Integral

from .errors import MeshError


@dataclass(frozen=True)
class BoundAuthoredSheetJointComponent:
    joint_edge_id: int
    authored_face_ids: tuple[int, ...]
    current_face_ids: tuple[int, ...]
    sheet_ids: tuple[int, ...]
    occurrence_correspondence: tuple
    owner_receipt: object
    boundary_correspondences: tuple
    constraint_receipt: object

    @property
    def unqualified_reference_categories(self) -> tuple[str, ...]:
        inventory = self.constraint_receipt.inventory
        categories = set()
        for side in ("original", "current"):
            document = inventory[side]
            categories.update(kind for kind, records in document["records"].items()
                              if records)
            categories.update(kind for kind in (
                "groups", "tags", "construction_vertices", "extensions")
                              if document["opaque_unqualified"][kind])
            if document["opaque_unqualified"]["features"].get("records"):
                categories.add("features")
            if any(row["metadata"] for rows in
                   document["opaque_unqualified"]["metadata"].values()
                   for row in rows):
                categories.add("metadata")
            if document["isolated_vertex_ids"]:
                categories.add("isolated_vertices")
        if self.constraint_receipt.outside_root_ids:
            categories.add("outside_shared_trim_roots")
        return tuple(sorted(categories))

    @property
    def publication_qualified(self) -> bool:
        return False


def bind_authored_sheet_joint_component(
    geometry, current_joint_edge_id, boundary_correspondences,
    selected_authored_face_ids, *, cancellation_check=None,
) -> BoundAuthoredSheetJointComponent:
    """Require the exact whole component and every root's owner-bound boundary.

    The qualified occurrence correspondence retains each original FaceUse and
    all of its current descendants. It does not license rewriting arbitrary
    source material, loads, structural records or mesh associations.
    """
    try:
        from anygeometry import (
            query_prepared_sheet_joint_component,
            validate_prepared_sheet_joint_component_binding,
            validate_prepared_sheet_joint_component_selection,
            validate_prepared_authored_boundary_correspondence_binding,
            query_prepared_authored_constraint_scope,
            validate_prepared_authored_constraint_scope_binding,
        )
    except ImportError as error:
        raise MeshError("authored Sheet-joint component capability is unavailable") from error
    if isinstance(current_joint_edge_id, bool) or not isinstance(current_joint_edge_id, Integral):
        raise MeshError("authored Sheet-joint component needs a current edge ID")
    try:
        raw_selected = tuple(selected_authored_face_ids)
    except TypeError as error:
        raise MeshError("authored Sheet-joint component needs authored face IDs") from error
    if any(isinstance(face, bool) or not isinstance(face, Integral)
           for face in raw_selected):
        raise MeshError("authored Sheet-joint component needs authored face IDs")
    selected = tuple(int(face) for face in raw_selected)
    if len(selected) != len(set(selected)):
        raise MeshError("authored Sheet-joint component repeats a root")
    receipt = query_prepared_sheet_joint_component(
        geometry, int(current_joint_edge_id),
        expected_revision=geometry.revision,
        cancellation_check=cancellation_check,
    )
    validate_prepared_sheet_joint_component_selection(
        geometry, receipt, selected, cancellation_check=cancellation_check,
    )
    if set(selected) != set(receipt.authored_face_ids):
        raise MeshError("authored Sheet-joint selection includes unqualified roots")
    correspondences = tuple(boundary_correspondences)
    if len(correspondences) != len(receipt.authored_face_ids):
        raise MeshError("authored Sheet-joint component lacks root boundaries")
    by_root = {}
    for correspondence in correspondences:
        validate_prepared_authored_boundary_correspondence_binding(
            geometry, correspondence, cancellation_check=cancellation_check,
        )
        root = int(correspondence.authored_definition.face_id)
        if root in by_root or correspondence.face_preimages != receipt.scope.face_preimages:
            raise MeshError("authored Sheet-joint component boundaries disagree")
        by_root[root] = correspondence
    if set(by_root) != set(receipt.authored_face_ids):
        raise MeshError("authored Sheet-joint component has missing or extra boundaries")
    current = [int(face) for correspondence in by_root.values()
               for face in correspondence.descendants]
    if len(current) != len(set(current)) or set(current) != set(receipt.current_face_ids):
        raise MeshError("authored Sheet-joint boundaries omit current component faces")
    occurrences = tuple(receipt.occurrence_correspondence)
    if (not occurrences or {int(row[1]) for row in occurrences} != set(receipt.authored_face_ids)
            or {int(row[0]) for row in occurrences} != set(receipt.sheet_ids)
            or any(not row[3] for row in occurrences)):
        raise MeshError("authored Sheet-joint component lacks occurrence coverage")
    validate_prepared_sheet_joint_component_binding(
        geometry, receipt, cancellation_check=cancellation_check,
    )
    constraint = query_prepared_authored_constraint_scope(
        geometry, receipt.authored_face_ids,
        expected_revision=geometry.revision,
        cancellation_check=cancellation_check,
    )
    if (constraint.selected_root_ids != tuple(receipt.authored_face_ids)
            or constraint.current_face_ids != tuple(receipt.current_face_ids)
            or constraint.scope.face_preimages != receipt.scope.face_preimages
            or tuple(constraint.boundary_correspondences) != tuple(
                by_root[root] for root in receipt.authored_face_ids)
            or not constraint.typed_inventory_complete
            or constraint.semantic_mapping_qualified
            or constraint.parameter_remapping_qualified
            or constraint.publication_qualified):
        raise MeshError("authored Sheet-joint constraint inventory binds a different component")
    validate_prepared_authored_constraint_scope_binding(
        geometry, constraint, cancellation_check=cancellation_check,
    )
    # The caller's boundary objects can be changed by the final owner callback.
    # Pin their definitions and both owner receipts with callback-free checks
    # after every callback-bearing query/validation has completed.
    validate_prepared_sheet_joint_component_binding(geometry, receipt)
    validate_prepared_authored_constraint_scope_binding(geometry, constraint)
    final_boundaries = tuple(by_root[root] for root in receipt.authored_face_ids)
    for correspondence in final_boundaries:
        validate_prepared_authored_boundary_correspondence_binding(
            geometry, correspondence)
    final_current = tuple(int(face) for correspondence in final_boundaries
                          for face in correspondence.descendants)
    if (final_boundaries != constraint.boundary_correspondences
            or len(final_current) != len(set(final_current))
            or set(final_current) != set(receipt.current_face_ids)
            or set(final_current) != set(constraint.current_face_ids)):
        raise MeshError("authored Sheet-joint boundary definitions changed after owner validation")
    return BoundAuthoredSheetJointComponent(
        int(receipt.joint_edge_id), tuple(receipt.authored_face_ids),
        tuple(receipt.current_face_ids), tuple(receipt.sheet_ids),
        occurrences, receipt,
        final_boundaries, constraint,
    )
