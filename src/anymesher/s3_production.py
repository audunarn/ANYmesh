"""Production preparation for qualified three-node shell meshes.

The low-level quality and repair modules deliberately accept explicit physical
normal authority.  This module is the production bridge: it derives that
authority from oriented ANYgeometry ``Sheet``/``FaceUse`` records, applies the
bounded repair exactly once, and emits deterministic, JSON-safe provenance.

Nothing in this module can select a legacy triangle.  A missing or ambiguous
owner, an unsupported T6, or exhausted repair raises a typed S3 rejection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel

from .errors import MeshError
from .mapped import nodal_normals
from .mesh import Mesh
from .s3_quality import (
    S3_QUALITY_CONTRACT_ID,
    S3QualityError,
    assert_s3_admissible,
)
from .s3_repair import (
    DEFAULT_S3_REPAIR_POLICY,
    S3_REPAIR_CONTRACT_ID,
    S3RepairPolicy,
    repair_s3_admission,
)

__all__ = [
    "QUALIFIED_S3_FORMULATION_ID",
    "QUALIFIED_S3_PRODUCTION_CONTRACT_ID",
    "S3OwnerAuthorityError",
    "prepare_qualified_s3_mesh",
]


QUALIFIED_S3_PRODUCTION_CONTRACT_ID = (
    "ANYMESHER_QUALIFIED_S3_PRODUCTION_PREPARATION_V1"
)
# Exact formulation identity qualified by ANYsolver's accepted V6W record.
# The mesher binds this identity but does not import solver mechanics.
QUALIFIED_S3_FORMULATION_ID = "CANDIDATE_E4_PL_S3_V2D_NATIVE_PARITY_V1"


class S3OwnerAuthorityError(S3QualityError):
    """Physical Sheet/FaceUse normal authority is absent or ambiguous."""


def _vector_tuple(
    value: Sequence[float], *, label: str
) -> tuple[float, float, float]:
    made = np.asarray(value, dtype=float)
    if made.shape != (3,) or not np.all(np.isfinite(made)):
        raise S3OwnerAuthorityError(f"{label} must be a finite three-vector")
    length = float(np.linalg.norm(made))
    if length <= 0.0:
        raise S3OwnerAuthorityError(f"{label} must be nonzero")
    unit = made / length
    return tuple(
        0.0 if component == 0.0 else float(component) for component in unit
    )  # type: ignore[return-value]


def _element_face_map(mesh: Mesh) -> dict[int, int]:
    shell_ids = set(mesh.shells)
    faces_by_element: dict[int, list[int]] = {
        element_id: [] for element_id in shell_ids
    }
    for face_id, element_ids in sorted(mesh.elements_of_face.items()):
        for element_id in element_ids:
            made_id = int(element_id)
            if made_id in faces_by_element:
                faces_by_element[made_id].append(int(face_id))
    result: dict[int, int] = {}
    for element_id in sorted(shell_ids):
        face_ids = tuple(sorted(set(faces_by_element[element_id])))
        if len(face_ids) != 1:
            raise S3OwnerAuthorityError(
                f"shell element {element_id} needs exactly one geometry-face "
                f"association; found {face_ids}"
            )
        result[element_id] = face_ids[0]
    return result


def _face_use_ids(geometry: GeometryModel) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for use_id, use in sorted(geometry.face_uses.items()):
        grouped.setdefault(int(use.face_id), []).append(int(use_id))
    return {face_id: tuple(values) for face_id, values in sorted(grouped.items())}


def _physical_owner_normal(
    geometry: GeometryModel,
    mesh: Mesh,
    *,
    element_id: int,
    face_id: int,
    face_use_ids: Mapping[int, Sequence[int]],
) -> tuple[float, float, float]:
    uses = tuple(int(value) for value in face_use_ids.get(face_id, ()))
    if not uses:
        raise S3OwnerAuthorityError(
            f"shell element {element_id} face {face_id} lacks authoritative "
            "Sheet/FaceUse normal"
        )
    try:
        corners = np.asarray(
            [mesh.nodes[node_id] for node_id in mesh.corners_of(element_id)],
            dtype=float,
        )
    except (KeyError, MeshError) as error:
        raise S3OwnerAuthorityError(
            f"shell element {element_id} has incomplete owner-normal geometry"
        ) from error
    if (
        corners.ndim != 2
        or corners.shape[1:] != (3,)
        or not np.all(np.isfinite(corners))
    ):
        raise S3OwnerAuthorityError(
            f"shell element {element_id} has invalid coordinates for "
            "owner-normal projection"
        )
    centroid = np.mean(corners, axis=0)
    try:
        _projected, uv, _distance = geometry.project_to_face(face_id, centroid)
        geometric = np.asarray(geometry.face_normal(face_id, *uv), dtype=float)
    except (GeometryError, KeyError, TypeError, ValueError) as error:
        raise S3OwnerAuthorityError(
            f"shell element {element_id} face {face_id} owner-normal evaluation failed"
        ) from error
    geometric_tuple = _vector_tuple(
        geometric,
        label=f"shell element {element_id} face {face_id} geometric normal",
    )
    geometric = np.asarray(geometric_tuple, dtype=float)
    candidates = []
    for use_id in uses:
        use = geometry.face_uses[use_id]
        candidates.append(float(int(use.orientation)) * geometric)
    reference = candidates[0]
    angular_floor = max(
        float(geometry.tolerance.angular),
        64.0 * float(np.finfo(float).eps),
    )
    for candidate in candidates[1:]:
        if float(np.dot(reference, candidate)) <= angular_floor:
            raise S3OwnerAuthorityError(
                f"shell element {element_id} face {face_id} has conflicting "
                "authoritative Sheet/FaceUse normals"
            )
    return _vector_tuple(
        np.sum(np.asarray(candidates, dtype=float), axis=0),
        label=f"shell element {element_id} face {face_id} physical owner normal",
    )


def _shell_owner_authority(
    geometry: GeometryModel,
    mesh: Mesh,
) -> tuple[
    dict[int, tuple[float, float, float]],
    dict[int, dict[str, Any]],
]:
    face_by_element = _element_face_map(mesh)
    face_use_ids = _face_use_ids(geometry)
    normals = {
        element_id: _physical_owner_normal(
            geometry,
            mesh,
            element_id=element_id,
            face_id=face_by_element[element_id],
            face_use_ids=face_use_ids,
        )
        for element_id in sorted(mesh.shells)
    }
    sources = {
        element_id: {
            "face_id": int(face_by_element[element_id]),
            "face_use_ids": [
                int(value)
                for value in face_use_ids[face_by_element[element_id]]
            ],
            "sheet_ids": [
                int(geometry.face_uses[use_id].sheet_id)
                for use_id in face_use_ids[face_by_element[element_id]]
            ],
        }
        for element_id in sorted(mesh.shells)
    }
    return normals, sources


def _quality_record(item: Any) -> dict[str, Any]:
    return {
        "edge_ratio": float(item.edge_ratio),
        "element_id": int(item.element_id),
        "maximum_angle_deg": float(item.maximum_angle_deg),
        "minimum_angle_deg": float(item.minimum_angle_deg),
        "minimum_scaled_jacobian": float(item.minimum_scaled_jacobian),
        "normalized_area": float(item.normalized_area),
        "owner_normal_alignment": float(item.owner_normal_alignment),
        "signed_area_ratio": float(item.signed_area_ratio),
        "violations": list(item.violations),
    }


def _attempt_record(item: Any) -> dict[str, Any]:
    return {
        "action": str(item.action),
        "detail": str(item.detail),
        "edge": [int(value) for value in item.edge],
        "element_ids": [int(value) for value in item.element_ids],
        "sequence": int(item.sequence),
        "status": str(item.status),
    }


def prepare_qualified_s3_mesh(
    mesh: Mesh,
    geometry: GeometryModel,
    *,
    repair_policy: S3RepairPolicy = DEFAULT_S3_REPAIR_POLICY,
) -> tuple[Mesh, dict[str, Any]]:
    """Return a fully admitted copied mesh plus deterministic authority data.

    The function is called only when the production qualified-S3 control is
    selected.  It performs one bounded repair request.  It never retries and
    never returns the caller's mesh after a failed qualification.
    """

    if not isinstance(mesh, Mesh):
        raise TypeError("prepare_qualified_s3_mesh expects an anymesher.Mesh")
    if not isinstance(geometry, GeometryModel):
        raise TypeError("prepare_qualified_s3_mesh expects an ANYgeometry model")
    triangle_ids = tuple(sorted(int(value) for value in mesh.tris))
    if not triangle_ids:
        return mesh, {
            "authority_model": {
                "prepared_revision": int(geometry.revision),
                "scope": "PREPARED_GEOMETRY_ORIENTED_SHEET_FACE_USE",
            },
            "contract_id": QUALIFIED_S3_PRODUCTION_CONTRACT_ID,
            "element_ids": [],
            "formulation_id": QUALIFIED_S3_FORMULATION_ID,
            "legacy_fallback": "FORBIDDEN",
            "quality_contract_id": S3_QUALITY_CONTRACT_ID,
            "repair_contract_id": S3_REPAIR_CONTRACT_ID,
            "schema": "anymesher.qualified-s3-production-preparation-v1",
            "status": "NOT_APPLICABLE_NO_TRIANGLES",
        }

    shell_owners, _ = _shell_owner_authority(geometry, mesh)
    repair = repair_s3_admission(
        mesh,
        element_ids=triangle_ids,
        element_owner_normals={
            element_id: shell_owners[element_id] for element_id in triangle_ids
        },
        repair_policy=repair_policy,
    )
    made = repair.mesh
    shell_owners, owner_sources = _shell_owner_authority(geometry, made)
    final_admission = assert_s3_admissible(
        made,
        element_ids=repair.element_ids,
        element_owner_normals={
            element_id: shell_owners[element_id]
            for element_id in repair.element_ids
        },
    )
    try:
        mixed_normals = nodal_normals(
            made,
            element_owner_normals=shell_owners,
            include_triangles=True,
        )
    except MeshError as error:
        raise S3OwnerAuthorityError(
            "qualified mixed-shell nodal normal construction failed"
        ) from error
    shell_nodes = {
        int(node_id)
        for connectivity in made.shells.values()
        for node_id in connectivity
    }
    missing = tuple(sorted(shell_nodes.difference(mixed_normals)))
    if missing:
        raise S3OwnerAuthorityError(
            f"qualified mixed-shell nodal normals are ambiguous for nodes {missing}"
        )
    record = {
        "admission": {
            "elements": [_quality_record(item) for item in final_admission.elements],
            "qualified_junction_edges": [
                [int(first), int(second)]
                for first, second in final_admission.qualified_junction_edges
            ],
            "topology_violations": list(final_admission.topology_violations),
        },
        "authority_model": {
            "prepared_revision": int(geometry.revision),
            "scope": "PREPARED_GEOMETRY_ORIENTED_SHEET_FACE_USE",
        },
        "contract_id": QUALIFIED_S3_PRODUCTION_CONTRACT_ID,
        "element_ids": [int(value) for value in repair.element_ids],
        "element_owner_normals": {
            str(element_id): [float(value) for value in shell_owners[element_id]]
            for element_id in sorted(shell_owners)
        },
        "element_owner_sources": {
            str(element_id): owner_sources[element_id]
            for element_id in sorted(owner_sources)
        },
        "formulation_id": QUALIFIED_S3_FORMULATION_ID,
        "legacy_fallback": "FORBIDDEN",
        "nodal_normals": {
            str(node_id): [
                0.0 if value == 0.0 else float(value)
                for value in mixed_normals[node_id]
            ]
            for node_id in sorted(mixed_normals)
        },
        "quality_contract_id": S3_QUALITY_CONTRACT_ID,
        "repair": {
            "added_elements": int(repair.added_elements),
            "added_nodes": int(repair.added_nodes),
            "attempts": [_attempt_record(item) for item in repair.attempts],
            "edge_flip_attempts": int(repair.edge_flip_attempts),
            "edge_flips": int(repair.edge_flips),
            "refinement_attempts": int(repair.refinement_attempts),
            "refinement_splits": int(repair.refinement_splits),
            "winding_repairs": int(repair.winding_repairs),
        },
        "repair_contract_id": S3_REPAIR_CONTRACT_ID,
        "schema": "anymesher.qualified-s3-production-preparation-v1",
        "status": "ADMITTED",
    }
    return made, record
