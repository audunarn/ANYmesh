"""Private, source-only association plan for a complete authored-root shell.

The plan does not interpret external application references or permit mesh
publication. In particular it never assigns a root-born cell to a current
descendant merely to satisfy a child-local load or section reference.
"""

from dataclasses import dataclass
from copy import deepcopy
import json
from numbers import Integral

from ._authored_scope_binding import BoundAuthoredRootInputs
from .errors import MeshError


@dataclass(frozen=True)
class AuthoredRootAssociations:
    source_face: int
    element_ids: tuple[int, ...]
    # (original FaceUse ID, Sheet ID, orientation), retaining source meaning.
    face_uses: tuple[tuple[int, int, str], ...]
    sheet_ids: tuple[int, ...]

    @property
    def publication_qualified(self) -> bool:
        return False

    def stage_mesh(self, mesh):
        """Return detached source associations, never current-child aliases."""
        if (set(mesh.quads) | set(mesh.tris) != set(self.element_ids)
                or mesh.beams or mesh.couplings
                or mesh.elements_of_face or mesh.elements_of_sheet):
            raise MeshError("authored source association candidate changed")
        candidate = deepcopy(mesh)
        candidate.elements_of_face[self.source_face] = list(self.element_ids)
        for sheet_id in self.sheet_ids:
            candidate.elements_of_sheet[sheet_id] = list(self.element_ids)
        return candidate


def plan_authored_root_associations(
    geometry, bound: BoundAuthoredRootInputs, mesh, element_ids,
    cancellation_check=None,
) -> AuthoredRootAssociations:
    """Bind every root-local shell cell to its original face and Sheet uses.

    This is deliberately limited to a standalone root-local mesh. A composed
    document needs explicit source provenance per cell before it can use this
    plan, and external child-local references need a separate consumer.
    """
    try:
        from anygeometry import validate_prepared_model_scope_binding
    except ImportError as error:
        raise MeshError("authored source association needs owner scope validation") from error
    if not isinstance(bound, BoundAuthoredRootInputs):
        raise MeshError("authored source association needs a bound root scope")
    validate_prepared_model_scope_binding(
        geometry, bound.scope, cancellation_check=cancellation_check,
    )
    if any(isinstance(item, bool) or not isinstance(item, Integral) for item in element_ids):
        raise MeshError("authored source association needs shell element IDs")
    ids = tuple(int(item) for item in element_ids)
    shells = set(mesh.quads) | set(mesh.tris)
    if not ids or len(ids) != len(set(ids)) or set(ids) != shells:
        raise MeshError("authored source association needs every root-local shell cell once")
    if mesh.beams or mesh.couplings or mesh.elements_of_face or mesh.elements_of_sheet:
        raise MeshError("authored source association needs an unassociated root-local shell")

    original = bound.scope.authored_document
    current = bound.scope.current_document
    original_struct = original["structural"]
    current_struct = current["structural"]
    root = bound.authored_face
    source_uses = tuple(
        use for use in original_struct["face_uses"] if use["face_id"] == root
    )
    source_signature = sorted(
        (int(use["sheet_id"]), str(use["orientation"]),
         json.dumps(use["metadata"], sort_keys=True))
        for use in source_uses
    )
    for child in bound.descendants:
        child_uses = (
            use for use in current_struct["face_uses"] if use["face_id"] == child
        )
        if sorted(
            (int(use["sheet_id"]), str(use["orientation"]),
             json.dumps(use["metadata"], sort_keys=True))
            for use in child_uses
        ) != source_signature:
            raise MeshError("authored source FaceUse meaning changed on a descendant")
    source_sheets = {int(sheet["id"]): sheet for sheet in original_struct["sheets"]}
    current_sheets = {int(sheet["id"]): sheet for sheet in current_struct["sheets"]}
    for sheet_id, _orientation, _metadata in source_signature:
        if sheet_id not in source_sheets or sheet_id not in current_sheets:
            raise MeshError("authored source Sheet identity is missing")
        source_sheet, current_sheet = source_sheets[sheet_id], current_sheets[sheet_id]
        for key in ("part_id", "name", "metadata", "policy", "declared_non_manifold_edges"):
            if source_sheet[key] != current_sheet[key]:
                raise MeshError("authored source Sheet meaning changed")
    source_use_records = tuple(sorted(
        (int(use["id"]), int(use["sheet_id"]), str(use["orientation"]))
        for use in source_uses
    ))
    validate_prepared_model_scope_binding(
        geometry, bound.scope, cancellation_check=cancellation_check,
    )
    return AuthoredRootAssociations(
        root, tuple(sorted(ids)), source_use_records,
        tuple(sorted({sheet_id for _use, sheet_id, _orientation in source_use_records})),
    )
