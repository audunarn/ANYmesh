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
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np

from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel

from .errors import MeshError
from .mapped import oriented_shell_normal_contributions
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


def _exact_identifier(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise S3OwnerAuthorityError(f'{label} must be an integer identifier')
    return int(value)


def _element_face_map(mesh: Mesh) -> dict[int, int]:
    shell_ids = {_exact_identifier(value, 'shell element ID') for value in mesh.shells}
    faces_by_element: dict[int, list[int]] = {
        element_id: [] for element_id in shell_ids
    }
    for face_id, element_ids in mesh.elements_of_face.items():
        face_id = _exact_identifier(face_id, 'geometry face ID')
        for element_id in element_ids:
            made_id = _exact_identifier(element_id, 'face element ID')
            if made_id in faces_by_element:
                faces_by_element[made_id].append(face_id)
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
    projected_uv=None,
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
        if projected_uv is None:
            _projected, uv, _distance = geometry.project_to_face(face_id, centroid)
        else:
            uv = projected_uv
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


def _shell_owner_projection_uv(geometry, mesh, face_by_element):
    """Batch only the owner-qualified topology sweep projection capability.

    Missing capabilities, custom scalar protocols and unsupported faces retain
    scalar evaluation. Expected batch errors replay the original scalar route
    so its element-specific authority rejection remains unchanged.
    """
    if (type(geometry) is not GeometryModel
            or getattr(geometry.project_to_face, '__func__', None) is not GeometryModel.project_to_face
            or getattr(geometry.face_normal, '__func__', None) is not GeometryModel.face_normal):
        return {}
    project = getattr(geometry, 'project_to_face_many', None)
    boundary = getattr(geometry, 'face_boundary_uv', None)
    if not callable(project) or not callable(boundary):
        return {}
    groups = {}
    supported = {}
    try:
        for element in sorted(face_by_element):
            face_id = face_by_element[element]
            face = geometry.faces[face_id]
            if face_id not in supported:
                supported[face_id] = boundary(face_id, face.loop[0].edge, 0.) is not None
            if not supported[face_id]:
                continue
            corners = np.asarray([mesh.nodes[node] for node in mesh.corners_of(element)], dtype=float)
            if corners.ndim != 2 or corners.shape[1:] != (3,) or not np.all(np.isfinite(corners)):
                return {}
            groups.setdefault(face_id, []).append((element, np.mean(corners, axis=0)))
        made = {}
        for face_id, rows in groups.items():
            _, uv, _ = project(face_id, np.asarray([point for _, point in rows]))
            uv = np.asarray(uv, dtype=float)
            if uv.shape != (len(rows), 2) or not np.all(np.isfinite(uv)):
                return {}
            made.update((element, coordinates) for (element, _), coordinates in zip(rows, uv))
        return made
    except (GeometryError, KeyError, TypeError, ValueError):
        return {}


_DEFAULT_NUMPY_ERRORS = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def _owner_normal_memo_admitted(geometry, mesh) -> bool:
    """Whether owner normals may be reused (an allow-list; anything else is evaluated as before).

    Exact library types with their own methods, plain node storage, the library's
    tolerance policy and NumPy's default error handling with no handler, so that
    an evaluation has no observable effect beyond its result.
    """
    from anygeometry.tolerance import TolerancePolicy
    from .s3_repair import _plain_shells
    return (type(geometry) is GeometryModel and type(mesh) is Mesh and _plain_shells(mesh)
            and type(geometry.tolerance) is TolerancePolicy
            and np.geterr() == _DEFAULT_NUMPY_ERRORS and np.geterrcall() is None
            and all(getattr(getattr(geometry, name), "__func__", None) is getattr(GeometryModel, name)
                    for name in ("project_to_face", "face_normal", "face_point")))


_LIBRARY_SURFACE_IDS: frozenset[int] | None = None
_MAGNITUDE_CAP, _MAGNITUDE_FLOOR = 2.0 ** 64, 2.0 ** -64


def _library_surface_ids() -> frozenset[int]:
    global _LIBRARY_SURFACE_IDS
    if _LIBRARY_SURFACE_IDS is None:
        from anygeometry.nurbs import NurbsSurface
        from anygeometry.surfaces import Cone, CoonsSurface, Cylinder, ExtrudedSurface, Plane, RuledSurface
        _LIBRARY_SURFACE_IDS = frozenset(
            id(kind) for kind in (Plane, Cylinder, Cone, ExtrudedSurface, RuledSurface, CoonsSurface, NurbsSurface))
    return _LIBRARY_SURFACE_IDS


def _moderate(value, *, divisor: bool, depth: int = 0) -> bool:
    """Plain numbers whose evaluation cannot overflow (an allow-list: other types decline).

    Every float is finite and at most 2^64 in magnitude; a ``divisor`` value is also
    0 or at least 2^-64.  Products and quotients of such values stay far from the
    float64 range, so the projections and normals evaluated from them emit no
    NumPy warning and a reused result has no skipped side effect.
    """
    kind = type(value)
    if kind is bool or kind is int:
        return abs(value) <= _MAGNITUDE_CAP
    if kind is float or kind is np.float64:
        magnitude = abs(float(value))
        return (magnitude <= _MAGNITUDE_CAP
                and (not divisor or magnitude == 0.0 or magnitude >= _MAGNITUDE_FLOOR))
    if kind is np.ndarray:
        if value.dtype != np.float64 or not bool(np.all(np.isfinite(value))):
            return False
        magnitude = np.abs(value)
        return bool(np.all(magnitude <= _MAGNITUDE_CAP)
                    and (not divisor or np.all((magnitude == 0.0) | (magnitude >= _MAGNITUDE_FLOOR))))
    if (kind is tuple or kind is list) and depth < 4:
        return all(_moderate(item, divisor=divisor, depth=depth + 1) for item in value)
    return False


def _library_surface(geometry, face_id) -> bool:
    """The face's surface is an exact anygeometry surface with moderate parameters."""
    try:
        surface = geometry.faces[face_id].surface
    except (KeyError, AttributeError):
        return False
    if id(type(surface)) not in _library_surface_ids():
        return False
    fields = vars(surface)
    return all(type(name) is str and _moderate(value, divisor=True) for name, value in fields.items())


def _owner_normal_key(geometry, mesh, element_id, face_id, uses, uv):
    """Exact inputs of one ``_physical_owner_normal`` evaluation, or None.

    For an exact ``GeometryModel`` (its own projection and normal methods) the
    normal is a deterministic function of the model revision, the face, its
    FaceUses and their orientations, the angular tolerance, the NumPy error
    state, the element's corner coordinates and the supplied uv (or, without
    one, the projection of the corner centroid).  Only plain float64 corners and
    uv made of float64 values are keyed, so equal keys mean equal inputs.  Every
    other case (no FaceUse, unreadable corners, other value types) returns None
    and is evaluated by the original route, which raises its original error.
    """
    if not uses or not _library_surface(geometry, face_id):
        return None
    try:
        corners = [mesh.nodes[node_id] for node_id in mesh.corners_of(element_id)]
        orientations = tuple(int(geometry.face_uses[use].orientation) for use in uses)
        angular = float(geometry.tolerance.angular)
    except (KeyError, MeshError, AttributeError, TypeError, ValueError):
        return None
    made = []
    for corner in corners:
        if (type(corner) is not np.ndarray or corner.dtype != np.float64 or corner.shape != (3,)
                or not _moderate(corner, divisor=False)):
            return None
        made.append(corner.tobytes())
    if uv is None:
        made_uv = None
    elif type(uv) is np.ndarray:
        if uv.dtype != np.float64 or uv.shape != (2,) or not _moderate(uv, divisor=False):
            return None
        made_uv = uv.tobytes()
    elif (type(uv) is tuple and len(uv) == 2
          and all(type(value) is float or type(value) is np.float64 for value in uv)
          and _moderate(uv, divisor=False)):
        made_uv = np.asarray(uv, dtype=np.float64).tobytes() + b"t"
    else:
        return None
    return (geometry.model_id, geometry.revision, int(face_id), uses, orientations, angular,
            tuple(sorted(np.geterr().items())), tuple(made), made_uv)


def _shell_owner_authority(
    geometry: GeometryModel,
    mesh: Mesh,
    *,
    _certified_uv=None,
    _normal_memo=None,
) -> tuple[
    dict[int, tuple[float, float, float]],
    dict[int, dict[str, Any]],
]:
    face_by_element = _element_face_map(mesh)
    face_use_ids = _face_use_ids(geometry)
    from .material_source_transfer import _validated_curvilinear_reference_uv
    certified_uv = (_validated_curvilinear_reference_uv(geometry, mesh)
                    if _certified_uv is None else _certified_uv)
    # Curvilinear cells use the owner's proven source reference map, never
    # a nearest-point projection of their flat physical centroid.
    ordinary = {element: face for element, face in face_by_element.items()
                if element not in certified_uv}
    projected_uv = (_shell_owner_projection_uv(geometry, mesh, ordinary)
                    if ordinary else {})
    projected_uv.update(certified_uv)
    # ``_normal_memo`` (call-local, one geometry object) carries normals already
    # evaluated on exactly the same inputs, e.g. before an S3 repair that left most
    # shells unchanged; every other shell is evaluated as before.
    memo = (_normal_memo if isinstance(_normal_memo, dict)
            and _normal_memo.get("geometry") is geometry
            and _owner_normal_memo_admitted(geometry, mesh) else None)
    normals = {}
    for element_id in sorted(mesh.shells):
        face_id = face_by_element[element_id]
        uv = projected_uv.get(element_id)
        key = None
        if memo is not None:
            uses = tuple(int(value) for value in face_use_ids.get(face_id, ()))
            key = _owner_normal_key(geometry, mesh, element_id, face_id, uses, uv)
            if key is not None and key in memo["normals"]:
                normals[element_id] = memo["normals"][key]
                continue
        normals[element_id] = _physical_owner_normal(
            geometry,
            mesh,
            element_id=element_id,
            face_id=face_id,
            face_use_ids=face_use_ids,
            projected_uv=uv,
        )
        if key is not None:
            memo["normals"][key] = normals[element_id]
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


# A node whose owner-side element normals sum to less than this fraction of
# their count has no signed-average direction: the sum is cancellation noise.
_NODAL_CANCELLATION_RELATIVE = 1.0e-9


def _qualified_nodal_normals(
    mesh: Mesh,
    owners: Mapping[int, Sequence[float]],
) -> dict[int, np.ndarray]:
    """The qualified mixed-shell nodal normal at every shell node.

    Each node takes the normalized sum of its owner-side element normals
    (bit-identical to :func:`anymesher.mapped.nodal_normals`).  Where that sum
    cancels -- coplanar plates whose sheets are oriented oppositely, such as
    mirror-image webs meeting at a node no other plate uses -- the signed
    average has no direction although the plane normal is well defined: the
    node then takes that plane normal, on the side of its lowest-numbered
    element.  Cancelling normals that share no plane (a symmetric junction of
    three or more plates) define no director; the node takes the
    lowest-numbered element's owner-side normal as a recorded convention.  A
    node without any non-degenerate element term remains a typed refusal.
    """

    totals: dict[int, np.ndarray] = {}
    terms: dict[int, list[tuple[int, np.ndarray]]] = {}
    try:
        for element_id, nodes, normal in oriented_shell_normal_contributions(
            mesh,
            element_owner_normals=owners,
            include_triangles=True,
        ):
            for node in nodes:
                node = int(node)
                totals[node] = totals.get(node, np.zeros(3)) + normal
                terms.setdefault(node, []).append((int(element_id), normal))
    except MeshError as error:
        raise S3OwnerAuthorityError(
            "qualified mixed-shell nodal normal construction failed"
        ) from error
    shell_nodes = {
        int(node_id)
        for connectivity in mesh.shells.values()
        for node_id in connectivity
    }
    normals: dict[int, np.ndarray] = {}
    ambiguous: list[int] = []
    for node in sorted(shell_nodes):
        total = totals.get(node)
        if total is None:
            ambiguous.append(node)
            continue
        length = float(np.linalg.norm(total))
        count = len(terms[node])
        if not np.isfinite(length):
            # Non-finite element terms (overflowing coordinates) have no
            # direction.  The old construction dropped a NaN sum (refusal)
            # but recorded a zero or non-finite vector for an infinite one;
            # both are refused here.
            ambiguous.append(node)
            continue
        if length > _NODAL_CANCELLATION_RELATIVE * count:
            normals[node] = total / length
            continue
        # Cancelled: the lowest-numbered element's owner-side normal.  On a
        # shared plane that is the plane normal; at a junction whose normals
        # share no plane (three webs at 120 degrees) no director exists and
        # the value is a recorded convention (owner decision D1, 2026-11-21).
        reference = np.asarray(
            min(terms[node], key=lambda item: item[0])[1], dtype=np.float64
        )
        reference_length = float(np.linalg.norm(reference))
        if not np.isfinite(reference_length) or reference_length <= 0.0:
            ambiguous.append(node)
            continue
        normals[node] = reference / reference_length
    if ambiguous:
        raise S3OwnerAuthorityError(
            "qualified mixed-shell nodal normals are ambiguous for nodes "
            f"{tuple(ambiguous)}"
        )
    return normals


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
    triangle_ids = tuple(sorted(_exact_identifier(value, 'triangle element ID') for value in mesh.tris))
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

    from .material_source_transfer import _validated_curvilinear_reference_uv, _publication_pin
    # Reject invalid labels before a routing hint indexes geometry by them.
    _element_face_map(mesh)
    certified_uv = _validated_curvilinear_reference_uv(geometry, mesh)
    before_cells = _publication_pin(mesh) if certified_uv else None
    owner_memo = {"geometry": geometry, "normals": {}}
    shell_owners, _ = _shell_owner_authority(geometry, mesh, _certified_uv=certified_uv,
                                             _normal_memo=owner_memo)
    repair_input = mesh
    if certified_uv:
        # The live boundary registry owns immutable geometry views and proof
        # identity. Repair receives only copied canonical mesh data; copying
        # transient authority is neither necessary nor a grant of authority.
        from copy import deepcopy
        from dataclasses import fields
        repair_input = Mesh(**deepcopy({
            field.name: getattr(mesh, field.name) for field in fields(Mesh)
        }))
    repair = repair_s3_admission(
        repair_input,
        element_ids=triangle_ids,
        element_owner_normals={
            element_id: shell_owners[element_id] for element_id in triangle_ids
        },
        quality_policy=quality_policy,
        repair_policy=repair_policy,
        target_policy=target_policy,
    )
    made = repair.mesh
    if certified_uv and (_publication_pin(mesh) != before_cells or _publication_pin(made) != before_cells):
        raise S3OwnerAuthorityError('curvilinear S3 repair requires fresh owner cell qualification')
    shell_owners, owner_sources = _shell_owner_authority(geometry, made, _certified_uv=certified_uv,
                                                         _normal_memo=owner_memo)
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
    mixed_normals = _qualified_nodal_normals(made, shell_owners)
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
    # Owner queries can invoke caller callbacks.  All physical cells used by
    # CURRENT validation, SOURCE proof, and S3 preparation must come from the
    # same entry snapshot, even if a callback changes and restores a live mesh.
    frozen_source = deepcopy(source_mesh)
    frozen_current = deepcopy(current_mesh)

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

    assert_inputs_current()
    if (_mesh_digest(frozen_source) != source_digest
            or _mesh_digest(frozen_current) != current_digest):
        raise S3OwnerAuthorityError(
            "source-bound S3 input snapshot differs from entry mesh"
        )
    validate_prepared_current_component_associations(
        prepared_geometry, association_receipt, association_component,
        frozen_current, association_registry, cell_current_faces,
        cancellation_check=cancellation_check,
    )
    assert_inputs_current()
    if (frozen_source.order != frozen_current.order or frozen_source.order != "linear"
            or frozen_source.tris != frozen_current.tris
            or frozen_source.quads != frozen_current.quads
            or frozen_source.beams != frozen_current.beams
            or frozen_source.declared_plate_junction_edges
            != frozen_current.declared_plate_junction_edges
            or set(frozen_source.nodes) != set(frozen_current.nodes)
            or any(not np.array_equal(frozen_source.nodes[node], frozen_current.nodes[node])
                   for node in frozen_source.nodes)):
        raise S3OwnerAuthorityError(
            "source-bound S3 physical cells differ from the validated CURRENT mesh"
        )
    current_normals, _ = _shell_owner_authority(prepared_geometry, frozen_current)
    source_normals, _ = _shell_owner_authority(source_geometry, frozen_source)
    source_face_cells = {
        int(face): tuple(int(cell) for cell in cells)
        for face, cells in frozen_source.elements_of_face.items()
    }
    if (any(face not in source_geometry.faces or len(cells) != len(set(cells))
            for face, cells in source_face_cells.items())
            or any(len(cells) != len(set(cells))
                   for cells in frozen_source.elements_of_sheet.values())):
        raise S3OwnerAuthorityError("source-bound S3 ownership has invalid face or sheet cells")
    expected_sheets: dict[int, set[int]] = {}
    for use in source_geometry.face_uses.values():
        if int(use.face_id) in source_face_cells:
            expected_sheets.setdefault(int(use.sheet_id), set()).update(
                source_face_cells[int(use.face_id)]
            )
    if (set(frozen_source.elements_of_sheet) != set(expected_sheets)
            or any(set(frozen_source.elements_of_sheet[sheet]) != cells
                   for sheet, cells in expected_sheets.items())):
        raise S3OwnerAuthorityError(
            "source-bound S3 sheet ownership disagrees with source FaceUse authority"
        )
    # Matching normals cannot distinguish a parallel but unrelated SOURCE
    # face. The owner proves the actual XYZ corners lift exactly from its
    # Plane, then proves each whole closed triangle lies in its material.
    from anygeometry import (
        query_material_surface_regions,
        validate_material_surface_region_triangles_xyz,
    )

    regions_by_face = {}
    triangles_by_face: dict[int, list[np.ndarray]] = {}
    for cell, face in _element_face_map(frozen_source).items():
        corners = np.asarray([frozen_source.nodes[node]
                              for node in frozen_source.corners_of(cell)], dtype=float)
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
        pieces = (corners,) if len(corners) == 3 else (
            corners[[0, 1, 2]], corners[[0, 2, 3]],
        )
        triangles_by_face.setdefault(face, []).extend(pieces)
    for face, triangles in sorted(triangles_by_face.items()):
        try:
            validate_material_surface_region_triangles_xyz(
                source_geometry, regions_by_face[face], face, triangles,
                cancellation_check=cancellation_check,
            )
        except (GeometryError, KeyError, TypeError, ValueError) as error:
            raise S3OwnerAuthorityError(
                f"source-bound S3 face {face} has cells outside exact SOURCE material"
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
        frozen_source, source_geometry, target_policy=None,
    )
    assert_inputs_current()
    if (made.tris != frozen_source.tris or made.quads != frozen_source.quads
            or set(made.nodes) != set(frozen_source.nodes)
            or any(not np.array_equal(made.nodes[node], frozen_source.nodes[node])
                   for node in frozen_source.nodes)):
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
