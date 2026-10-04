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
    DEFAULT_S3_QUALITY_POLICY,
    S3_QUALITY_CONTRACT_ID,
    S3_TARGET_QUALITY_POLICY,
    S3QualityError,
    S3QualityPolicy,
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
    "prepare_source_bound_qualified_s3_mesh",
]


QUALIFIED_S3_PRODUCTION_CONTRACT_ID = (
    # V2: repair works towards the preferred target and admission enforces the
    # solver floor; the record carries both policies and target attainment.
    "ANYMESHER_QUALIFIED_S3_PRODUCTION_PREPARATION_V2"
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


def _policy_record(policy: S3QualityPolicy) -> dict[str, Any]:
    return {
        "maximum_angle_deg": float(policy.maximum_angle_deg),
        "maximum_edge_ratio": float(policy.maximum_edge_ratio),
        "minimum_angle_deg": float(policy.minimum_angle_deg),
        "minimum_normalized_area": float(policy.minimum_normalized_area),
        "minimum_scaled_jacobian": float(policy.minimum_scaled_jacobian),
    }


def prepare_qualified_s3_mesh(
    mesh: Mesh,
    geometry: GeometryModel,
    *,
    repair_policy: S3RepairPolicy = DEFAULT_S3_REPAIR_POLICY,
    quality_policy: S3QualityPolicy = DEFAULT_S3_QUALITY_POLICY,
    target_policy: S3QualityPolicy | None = S3_TARGET_QUALITY_POLICY,
) -> tuple[Mesh, dict[str, Any]]:
    """Return a fully admitted copied mesh plus deterministic authority data.

    The function is called only when the production qualified-S3 control is
    selected.  It performs one bounded repair request.  It never retries and
    never returns the caller's mesh after a failed qualification.

    Repair works towards ``target_policy`` (the preferred 30 degree shape);
    the mesh is rejected only if it violates ``quality_policy`` (the solver
    floor, 15 degrees by default).  The record reports target attainment.
    """

    if not isinstance(mesh, Mesh):
        raise TypeError("prepare_qualified_s3_mesh expects an anymesher.Mesh")
    if not isinstance(geometry, GeometryModel):
        raise TypeError("prepare_qualified_s3_mesh expects an ANYgeometry model")
    triangle_ids = tuple(sorted(int(value) for value in mesh.tris))
    if not triangle_ids:
        return mesh, {
            "admission": {
                "elements": [],
                "qualified_junction_edges": [],
                "topology_violations": [],
            },
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
            "schema": "anymesher.qualified-s3-production-preparation-v2",
            "status": "NOT_APPLICABLE_NO_TRIANGLES",
        }

    shell_owners, _ = _shell_owner_authority(geometry, mesh)
    repair = repair_s3_admission(
        mesh,
        element_ids=triangle_ids,
        element_owner_normals={
            element_id: shell_owners[element_id] for element_id in triangle_ids
        },
        quality_policy=quality_policy,
        repair_policy=repair_policy,
        target_policy=target_policy,
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
        policy=quality_policy,
    )
    target_shortfall = (
        []
        if repair.target_admission is None
        else [
            int(item.element_id)
            for item in repair.target_admission.elements
            if not item.admitted
        ]
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
        "quality_policy": _policy_record(quality_policy),
        "quality_target": {
            "met": not target_shortfall,
            "policy": (
                None if target_policy is None else _policy_record(target_policy)
            ),
            "shortfall_element_ids": target_shortfall,
        },
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
        "schema": "anymesher.qualified-s3-production-preparation-v2",
        "status": "ADMITTED",
    }
    return made, record


def prepare_source_bound_qualified_s3_mesh(
    source_mesh: Mesh,
    source_geometry: GeometryModel,
    prepared_geometry: GeometryModel,
    current_mesh: Mesh,
    association_receipt,
    association_component,
    association_registry,
    cell_current_faces,
    *,
    cancellation_check=None,
) -> tuple[Mesh, dict[str, Any]]:
    """Qualify a detached, source-bound T3 mesh without changing its cells.

    The CURRENT receipt is revalidated against the live prepared owner first.
    Physical cell IDs, connectivity, nodes, and oriented owner normals must
    agree with the SOURCE-bound mesh.  Source FaceUse/Sheet authority and S3
    admission are then derived afresh from the live source model.  This does
    not prove application reference transfer or grant publication permission.
    """
    from .prepared_current_associations import (
        validate_prepared_current_component_associations,
    )
    from ._authored_component_stage import _mesh_digest
    from anygeometry.serialization import to_dict as owner_to_dict

    if type(source_mesh) is not Mesh or type(current_mesh) is not Mesh:
        raise TypeError("source-bound S3 preparation requires canonical Mesh inputs")
    if not isinstance(source_geometry, GeometryModel) or not isinstance(
        prepared_geometry, GeometryModel
    ):
        raise TypeError("source-bound S3 preparation requires live owner models")
    if not source_mesh.tris:
        raise S3OwnerAuthorityError("source-bound S3 preparation requires T3 cells")
    source_identity = (source_geometry.model_id, source_geometry.revision)
    prepared_identity = (prepared_geometry.model_id, prepared_geometry.revision)
    source_definition = owner_to_dict(source_geometry)
    prepared_definition = owner_to_dict(prepared_geometry)
    if (source_mesh.geometry_model_id != source_geometry.model_id
            or source_mesh.geometry_revision != source_geometry.revision):
        raise S3OwnerAuthorityError("source-bound S3 mesh has stale source geometry identity")
    if source_mesh.structural_preparation.get("qualified_s3") is not None:
        raise S3OwnerAuthorityError("source-bound S3 mesh already has S3 authority")
    source_digest = _mesh_digest(source_mesh)
    current_digest = _mesh_digest(current_mesh)

    def assert_inputs_current():
        try:
            source_now = owner_to_dict(source_geometry)
            prepared_now = owner_to_dict(prepared_geometry)
        except (GeometryError, TypeError, ValueError) as error:
            raise S3OwnerAuthorityError(
                "source-bound S3 owner changed during qualification"
            ) from error
        if ((source_geometry.model_id, source_geometry.revision) != source_identity
                or (prepared_geometry.model_id, prepared_geometry.revision) != prepared_identity
                or source_now != source_definition
                or prepared_now != prepared_definition
                or (source_mesh.geometry_model_id, source_mesh.geometry_revision)
                != source_identity or _mesh_digest(source_mesh) != source_digest
                or _mesh_digest(current_mesh) != current_digest):
            raise S3OwnerAuthorityError(
                "source-bound S3 owner or input mesh changed during qualification"
            )

    validate_prepared_current_component_associations(
        prepared_geometry, association_receipt, association_component,
        current_mesh, association_registry, cell_current_faces,
        cancellation_check=cancellation_check,
    )
    assert_inputs_current()
    if (source_mesh.order != current_mesh.order or source_mesh.order != "linear"
            or source_mesh.tris != current_mesh.tris
            or source_mesh.quads != current_mesh.quads
            or source_mesh.beams != current_mesh.beams
            or source_mesh.declared_plate_junction_edges
            != current_mesh.declared_plate_junction_edges
            or set(source_mesh.nodes) != set(current_mesh.nodes)
            or any(not np.array_equal(source_mesh.nodes[node], current_mesh.nodes[node])
                   for node in source_mesh.nodes)):
        raise S3OwnerAuthorityError(
            "source-bound S3 physical cells differ from the validated CURRENT mesh"
        )
    current_normals, _ = _shell_owner_authority(prepared_geometry, current_mesh)
    source_normals, _ = _shell_owner_authority(source_geometry, source_mesh)
    source_face_cells = {
        int(face): tuple(int(cell) for cell in cells)
        for face, cells in source_mesh.elements_of_face.items()
    }
    if (any(face not in source_geometry.faces or len(cells) != len(set(cells))
            for face, cells in source_face_cells.items())
            or any(len(cells) != len(set(cells))
                   for cells in source_mesh.elements_of_sheet.values())):
        raise S3OwnerAuthorityError("source-bound S3 ownership has invalid face or sheet cells")
    expected_sheets: dict[int, set[int]] = {}
    for use in source_geometry.face_uses.values():
        if int(use.face_id) in source_face_cells:
            expected_sheets.setdefault(int(use.sheet_id), set()).update(
                source_face_cells[int(use.face_id)]
            )
    if (set(source_mesh.elements_of_sheet) != set(expected_sheets)
            or any(set(source_mesh.elements_of_sheet[sheet]) != cells
                   for sheet, cells in expected_sheets.items())):
        raise S3OwnerAuthorityError(
            "source-bound S3 sheet ownership disagrees with source FaceUse authority"
        )
    # Matching normals alone cannot distinguish a translated parallel face.
    # This authored-component bridge is planar/linear. Check source support
    # distance, then ask the owner to prove each *whole* closed triangle lies
    # in the SOURCE material, including concave trims and holes. Samples alone
    # would miss a side crossing excluded material.
    from anygeometry import (
        query_material_surface_regions,
        validate_material_surface_region_triangles,
    )

    positions = np.asarray(list(source_mesh.nodes.values()), dtype=float)
    diameter = float(np.linalg.norm(np.ptp(positions, axis=0)))
    support_limit = 1.0e-10 * max(1.0, diameter)
    regions_by_face = {}
    for cell, face in _element_face_map(source_mesh).items():
        corners = np.asarray([source_mesh.nodes[node]
                              for node in source_mesh.corners_of(cell)], dtype=float)
        samples = list(corners)
        samples.extend((first + second) / 2.0
                       for first, second in zip(corners, np.roll(corners, -1, axis=0)))
        samples.append(np.mean(corners, axis=0))
        try:
            distances = [source_geometry.project_to_face(face, point)[2]
                         for point in samples]
        except (GeometryError, KeyError, TypeError, ValueError) as error:
            raise S3OwnerAuthorityError(
                f"source-bound S3 cell {cell} lacks owner support"
            ) from error
        if any(not np.isfinite(distance) or distance > support_limit
               for distance in distances):
            raise S3OwnerAuthorityError(
                f"source-bound S3 cell {cell} lies outside its SOURCE face support"
            )
        if face not in regions_by_face:
            try:
                regions_by_face[face] = query_material_surface_regions(
                    source_geometry, operands=(source_geometry.handle("face", face),),
                    expected_revision=source_identity[1],
                    cancellation_check=cancellation_check,
                )
            except (GeometryError, KeyError, TypeError, ValueError) as error:
                raise S3OwnerAuthorityError(
                    f"source-bound S3 face {face} lacks exact material authority"
                ) from error
        regions = regions_by_face[face]
        matching = [region for region in regions.regions
                    if any(handle.id == face for handle in region.faces)]
        if len(matching) != 1 or len(matching[0].faces) != 1:
            raise S3OwnerAuthorityError(
                f"source-bound S3 cell {cell} lacks one exact SOURCE material region"
            )
        try:
            uv = [source_geometry.face_support_local_uv(face, point)
                  for point in corners]
            pieces = (uv,) if len(uv) == 3 else ((uv[0], uv[1], uv[2]),
                                                    (uv[0], uv[2], uv[3]))
            validate_material_surface_region_triangles(
                source_geometry, regions, face, pieces,
                cancellation_check=cancellation_check,
            )
        except (GeometryError, KeyError, TypeError, ValueError) as error:
            raise S3OwnerAuthorityError(
                f"source-bound S3 cell {cell} leaves exact SOURCE material"
            ) from error
    assert_inputs_current()
    if set(source_normals) != set(current_normals) or any(
        float(np.dot(source_normals[cell], current_normals[cell])) < 1.0 - 1.0e-10
        for cell in source_normals
    ):
        raise S3OwnerAuthorityError(
            "source-bound S3 physical normals disagree with prepared Sheet/FaceUse authority"
        )
    made, record = prepare_qualified_s3_mesh(
        source_mesh, source_geometry, target_policy=None,
    )
    assert_inputs_current()
    if (made.tris != source_mesh.tris or made.quads != source_mesh.quads
            or set(made.nodes) != set(source_mesh.nodes)
            or any(not np.array_equal(made.nodes[node], source_mesh.nodes[node])
                   for node in source_mesh.nodes)):
        raise S3OwnerAuthorityError(
            "source-bound S3 qualification requires a mesh repair before reference transfer"
        )
    record["authority_model"].update({
        "source_model_id": str(source_geometry.model_id),
        "source_revision": int(source_geometry.revision),
    })
    made.structural_preparation = dict(made.structural_preparation)
    made.structural_preparation["qualified_s3"] = record
    assert_inputs_current()
    return made, record
