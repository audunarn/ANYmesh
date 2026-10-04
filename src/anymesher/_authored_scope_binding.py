"""Conservative original/current owner-input preflight for authored-root work.

The complete owner snapshot exposes references; it does not interpret or remap
them. This first preflight refuses unsupported records and never grants mesh
publication permission.
"""

from dataclasses import dataclass
from numbers import Integral

from .errors import MeshError


@dataclass(frozen=True)
class BoundAuthoredRootInputs:
    authored_face: int
    descendants: tuple[int, ...]
    scope: object
    constraint_receipt: object | None = None

    @property
    def publication_qualified(self) -> bool:
        return False


def bind_authored_root_inputs(
    geometry, correspondence, selected_faces, cancellation_check=None,
) -> BoundAuthoredRootInputs:
    """Require complete selected scope and refuse unimplemented references.

    Both snapshots are validated by ANYgeometry before and after inspection.
    Absence here concerns persisted owner inputs only; application-side loads,
    materials and supports need their own explicit consumption contract.
    """
    try:
        from anygeometry import (
            query_prepared_authored_constraint_scope,
            validate_prepared_authored_constraint_scope_binding,
            validate_prepared_authored_boundary_correspondence_binding,
        )
    except ImportError as error:
        raise MeshError("complete authored-root constraint scope capability is unavailable") from error

    validate_prepared_authored_boundary_correspondence_binding(
        geometry, correspondence, cancellation_check=cancellation_check,
    )
    root = int(correspondence.authored_definition.face_id)
    receipt = query_prepared_authored_constraint_scope(
        geometry, (root,), expected_revision=geometry.revision,
        cancellation_check=cancellation_check,
    )
    scope = receipt.scope
    if scope.face_preimages != correspondence.face_preimages:
        raise MeshError("authored-root scope and boundary bind different preparations")
    descendants = tuple(int(face) for face in correspondence.descendants)
    selected = tuple(selected_faces)
    if any(isinstance(face, bool) or not isinstance(face, Integral) for face in selected):
        raise MeshError("authored-root request needs current face IDs")
    requested = tuple(sorted(set(int(face) for face in selected)))
    if requested != descendants or len(requested) != len(selected):
        raise MeshError("authored-root request must select every current descendant exactly once")

    if (receipt.selected_root_ids != (root,) or
            receipt.current_face_ids != descendants or
            len(receipt.boundary_correspondences) != 1 or
            receipt.boundary_correspondences[0] != correspondence or
            not receipt.typed_inventory_complete or
            receipt.semantic_mapping_qualified or
            receipt.parameter_remapping_qualified or
            receipt.publication_qualified):
        raise MeshError("authored-root constraint inventory binds a different root or capability")
    inventory = receipt.inventory
    original, current = inventory["original"], inventory["current"]
    original_metadata = {row["id"]: row["metadata"]
                         for row in original["opaque_unqualified"]["metadata"]["faces"]}
    current_metadata = {row["id"]: row["metadata"]
                        for row in current["opaque_unqualified"]["metadata"]["faces"]}
    if root not in original_metadata or any(face not in current_metadata
                                           for face in descendants):
        raise MeshError("authored-root inventory lacks source or descendant face metadata")
    for face in descendants:
        if current_metadata[face] != original_metadata[root]:
            raise MeshError("authored-root child-local face metadata needs an explicit remap")

    for label, document in (("original", original), ("current", current)):
        for kind in ("members", "member_edge_uses", "attachments", "junctions"):
            if document["records"][kind]:
                raise MeshError(f"authored-root {label} {kind} need a qualified consumer")
        opaque = document["opaque_unqualified"]
        for kind in ("groups", "tags", "construction_vertices", "extensions"):
            if opaque[kind]:
                raise MeshError(f"authored-root {label} {kind} need a qualified consumer")
        if opaque["features"].get("records"):
            raise MeshError(f"authored-root {label} features need a qualified consumer")
        if document["isolated_vertex_ids"]:
            raise MeshError(f"authored-root {label} isolated vertices need a qualified consumer")

    expected_edges = {edge for loop in correspondence.exterior_loops
                      for _source, _forward, current_edges in loop
                      for edge in current_edges}
    expected_edges.update(edge for edge, _uses in correspondence.interior_incidence)
    traces = inventory["traces"]
    if ({trace["current_edge_id"] for trace in traces} != expected_edges or
            any(trace["authored_root_id"] != root or
                not trace["root_child_incidence"] or
                not set(face for face, _forward in trace["root_child_incidence"])
                <= set(descendants) for trace in traces)):
        raise MeshError("authored-root constraint traces disagree with boundary correspondence")
    if receipt.outside_root_ids or any(trace["outside_authored_root_ids"]
                                       for trace in traces):
        raise MeshError("authored-root adjacent physical interface needs atomic shared publication")

    validate_prepared_authored_constraint_scope_binding(
        geometry, receipt, cancellation_check=cancellation_check,
    )
    return BoundAuthoredRootInputs(root, descendants, scope, receipt)
