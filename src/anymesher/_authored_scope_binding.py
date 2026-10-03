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
            query_prepared_model_scope,
            validate_prepared_model_scope_binding,
            validate_prepared_authored_boundary_correspondence_binding,
        )
    except ImportError as error:
        raise MeshError("complete authored-root owner scope capability is unavailable") from error

    validate_prepared_authored_boundary_correspondence_binding(
        geometry, correspondence, cancellation_check=cancellation_check,
    )
    scope = query_prepared_model_scope(
        geometry, expected_revision=geometry.revision,
        cancellation_check=cancellation_check,
    )
    if scope.face_preimages != correspondence.face_preimages:
        raise MeshError("authored-root scope and boundary bind different preparations")
    descendants = tuple(int(face) for face in correspondence.descendants)
    selected = tuple(selected_faces)
    if any(isinstance(face, bool) or not isinstance(face, Integral) for face in selected):
        raise MeshError("authored-root request needs current face IDs")
    requested = tuple(sorted(set(int(face) for face in selected)))
    if requested != descendants or len(requested) != len(selected):
        raise MeshError("authored-root request must select every current descendant exactly once")

    original, current = scope.authored_document, scope.current_document
    root = int(correspondence.authored_definition.face_id)
    original_face = next((face for face in original["faces"] if face["id"] == root), None)
    current_faces = {face["id"]: face for face in current["faces"]}
    if original_face is None or any(face not in current_faces for face in descendants):
        raise MeshError("authored-root snapshot has missing source or descendant faces")
    for face in descendants:
        if current_faces[face]["metadata"] != original_face["metadata"]:
            raise MeshError("authored-root child-local face metadata needs an explicit remap")

    for label, document in (("original", original), ("current", current)):
        structural = document["structural"]
        for kind in ("members", "member_edge_uses", "attachments", "junctions"):
            if structural[kind]:
                raise MeshError(f"authored-root {label} {kind} need a qualified consumer")
        for kind in ("groups", "tags", "construction_vertices", "extensions"):
            if document[kind]:
                raise MeshError(f"authored-root {label} {kind} need a qualified consumer")
        if document.get("features", {}).get("records"):
            raise MeshError(f"authored-root {label} features need a qualified consumer")
        referenced_vertices = set()
        for edge in document["edges"]:
            referenced_vertices.update((edge["start"], edge["end"]))
            referenced_vertices.update(edge["curve"].get("control_vertices", ()))
        for face in document["faces"]:
            referenced_vertices.update(face["corners"])
        if {vertex["id"] for vertex in document["vertices"]} - referenced_vertices:
            raise MeshError(f"authored-root {label} isolated vertices need a qualified consumer")

    selected = set(descendants)
    selected_edges = {
        edge for face in descendants
        for loop in (current_faces[face]["loop"], *current_faces[face]["holes"])
        for edge, _direction in loop
    }
    for face in current["faces"]:
        if face["id"] in selected:
            continue
        outside_edges = {
            edge for loop in (face["loop"], *face["holes"])
            for edge, _direction in loop
        }
        if selected_edges & outside_edges:
            raise MeshError("authored-root adjacent physical interface needs atomic shared publication")

    validate_prepared_model_scope_binding(
        geometry, scope, cancellation_check=cancellation_check,
    )
    return BoundAuthoredRootInputs(root, descendants, scope)
