"""Bounded immutable structural preparation for one mesh job.

The editable design model is never changed.  Missing structural owners are
declared only on a detached clone, and every physical relationship is created
through ANYgeometry's public query -> plan -> atomic apply workflow.  Exact
replacement lineage is retained for later association publication.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any

import numpy as np

from anygeometry.curves import Straight
from anygeometry.entities import EntityRef
from anygeometry.errors import GeometryError
from anygeometry.intersections import (
    ImprintOperation,
    apply_imprint,
    plan_imprint,
    query_intersection,
)
from anygeometry.model import GeometryModel
from anygeometry import IntersectionBatchPolicy, plan_intersections, apply_intersections
from anygeometry.overlaps import find_coplanar_overlaps, OverlapQualificationError
from anygeometry.policies import ConnectionIntent
from anygeometry.predicates import IntersectionKind
from anygeometry.surfaces import Cylinder, Plane, CoonsSurface, Cone, RuledSurface

from .errors import MeshError

__all__ = [
    "StructuralPreparationOptions",
    "StructuralPreparationReport",
    "prepare_structural_closure",
]

CancellationCheck = Callable[[str], None]


def _cancel(callback: CancellationCheck | None, phase: str) -> None:
    if callback is not None:
        callback(phase)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True, slots=True)
class StructuralPreparationOptions:
    """Resource and relationship policy for detached preparation."""

    automatic_face_connections: bool = True
    automatic_member_connections: bool = True
    automatic_member_sheet_connections: bool = True
    declare_missing_owners: bool = True
    maximum_candidate_pairs: int | None = None
    maximum_applications: int | None = None
    maximum_face_records: int | None = None
    maximum_edge_records: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "automatic_face_connections",
            "automatic_member_connections",
            "automatic_member_sheet_connections",
            "declare_missing_owners",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        for name in (
            "maximum_candidate_pairs",
            "maximum_applications",
            "maximum_face_records",
            "maximum_edge_records",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or int(value) < 1:
                raise MeshError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @classmethod
    def create(
        cls,
        value: "StructuralPreparationOptions | Mapping[str, Any] | bool | None" = None,
    ) -> "StructuralPreparationOptions | None":
        if value is False:
            return cls(
                automatic_face_connections=False,
                automatic_member_connections=False,
                automatic_member_sheet_connections=False,
                declare_missing_owners=False,
            )
        if value is None or value is True:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise MeshError(
            "structural_preparation must be a boolean, mapping, or "
            "StructuralPreparationOptions"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "automatic_face_connections": self.automatic_face_connections,
            "automatic_member_connections": self.automatic_member_connections,
            "automatic_member_sheet_connections": (
                self.automatic_member_sheet_connections
            ),
            "declare_missing_owners": self.declare_missing_owners,
            "maximum_candidate_pairs": self.maximum_candidate_pairs,
            "maximum_applications": self.maximum_applications,
            "maximum_face_records": self.maximum_face_records,
            "maximum_edge_records": self.maximum_edge_records,
        }


@dataclass(frozen=True, slots=True)
class StructuralPreparationReport:
    """Exact source-to-working evidence for one detached closure."""

    model_id: str
    source_revision: int
    working_revision: int
    options: StructuralPreparationOptions
    source_to_working_faces: Mapping[int, tuple[int, ...]]
    source_to_working_edges: Mapping[int, tuple[int, ...]]
    temporary_sheet_ids: tuple[int, ...] = ()
    temporary_member_ids: tuple[int, ...] = ()
    declared_face_connection_edges: tuple[int, ...] = ()
    candidate_queries: int = 0
    applications: int = 0
    face_connections: int = 0
    member_connections: int = 0
    member_sheet_connections: int = 0
    diagnostics: tuple[str, ...] = ()
    status: str = "applied"
    preparation_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "source_revision", int(self.source_revision))
        object.__setattr__(self, "working_revision", int(self.working_revision))
        object.__setattr__(
            self,
            "source_to_working_faces",
            _freeze(
                {
                    int(key): tuple(int(item) for item in values)
                    for key, values in self.source_to_working_faces.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "source_to_working_edges",
            _freeze(
                {
                    int(key): tuple(int(item) for item in values)
                    for key, values in self.source_to_working_edges.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "temporary_sheet_ids",
            tuple(sorted(set(map(int, self.temporary_sheet_ids)))),
        )
        object.__setattr__(
            self,
            "temporary_member_ids",
            tuple(sorted(set(map(int, self.temporary_member_ids)))),
        )
        object.__setattr__(
            self,
            "declared_face_connection_edges",
            tuple(sorted(set(map(int, self.declared_face_connection_edges)))),
        )
        for name in (
            "candidate_queries",
            "applications",
            "face_connections",
            "member_connections",
            "member_sheet_connections",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "diagnostics", tuple(map(str, self.diagnostics)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "source_revision": self.source_revision,
            "working_revision": self.working_revision,
            "options": self.options.to_dict(),
            "source_to_working_faces": {
                str(key): list(values)
                for key, values in sorted(self.source_to_working_faces.items())
            },
            "source_to_working_edges": {
                str(key): list(values)
                for key, values in sorted(self.source_to_working_edges.items())
            },
            "temporary_sheet_ids": list(self.temporary_sheet_ids),
            "temporary_member_ids": list(self.temporary_member_ids),
            "declared_face_connection_edges": list(
                self.declared_face_connection_edges
            ),
            "candidate_queries": self.candidate_queries,
            "applications": self.applications,
            "face_connections": self.face_connections,
            "member_connections": self.member_connections,
            "member_sheet_connections": self.member_sheet_connections,
            "diagnostics": list(self.diagnostics),
            "status": self.status,
            "preparation_hash": self.preparation_hash,
        }


def _resolved(
    geometry: GeometryModel,
    kind: str,
    identifier: int,
) -> tuple[int, ...]:
    values = tuple(
        item.id
        for item in geometry.resolve_ref(EntityRef(kind, int(identifier)))
        if item.kind == kind
    )
    if not values:
        raise MeshError(f"source {kind} {identifier} has no exact working descendant")
    return tuple(sorted(set(values)))


def _face_sheet_membership(
    geometry: GeometryModel,
) -> dict[int, tuple[int, ...]]:
    membership: dict[int, list[int]] = {
        int(face_id): [] for face_id in geometry.faces
    }
    for use in geometry.face_uses.values():
        if int(use.face_id) in membership:
            membership[int(use.face_id)].append(int(use.sheet_id))
    return {
        face_id: tuple(sorted(set(sheet_ids)))
        for face_id, sheet_ids in membership.items()
    }


def _selected_descendant_faces(
    working: GeometryModel,
    source_faces: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                descendant
                for face_id in source_faces
                for descendant in _resolved(working, "face", face_id)
            }
        )
    )


def _face_boundary_vertices(geometry: GeometryModel, face_id: int) -> set[int]:
    face = geometry.faces[face_id]
    return {
        vertex
        for loop in (face.loop,) + face.holes
        for item in loop
        for vertex in (
            geometry.oriented_start_vertex(item),
            geometry.oriented_end_vertex(item),
        )
    }


def _shared_boundary_vertices(
    geometry: GeometryModel, first: int, second: int
) -> set[int]:
    return _face_boundary_vertices(geometry, first).intersection(
        _face_boundary_vertices(geometry, second)
    )


def _share_boundary(geometry: GeometryModel, first: int, second: int) -> bool:
    """Whether two faces already share an authoritative boundary edge.

    A common edge is a complete structural connection.  A common vertex alone
    is not sufficient: two faces may meet at that vertex *and* intersect
    elsewhere.  Vertex-only pairs therefore still pass through the qualified
    geometric predicate and are suppressed only when its complete result is
    the already-shared point.
    """

    return bool(_shared_boundary_edges(geometry, first, second))


def _shared_boundary_edges(
    geometry: GeometryModel,
    first: int,
    second: int,
) -> tuple[int, ...]:
    first_loops = (geometry.faces[first].loop,) + geometry.faces[first].holes
    second_loops = (geometry.faces[second].loop,) + geometry.faces[second].holes
    first_edges = {item.edge for loop in first_loops for item in loop}
    return tuple(
        sorted(
            {
                item.edge
                for loop in second_loops
                for item in loop
                if item.edge in first_edges
            }
        )
    )


def _shared_transverse_plate_edges(
    geometry: GeometryModel,
    first: int,
    second: int,
    shared_edges: tuple[int, ...],
) -> tuple[int, ...]:
    first_surface = geometry.faces[first].surface
    second_surface = geometry.faces[second].surface
    if isinstance(first_surface, Plane) and isinstance(second_surface, Plane):
        first_normal = np.asarray(first_surface.normal, dtype=float)
        second_normal = np.asarray(second_surface.normal, dtype=float)
        normal_scale = float(
            np.linalg.norm(first_normal) * np.linalg.norm(second_normal)
        )
        if normal_scale <= 0.0:
            return ()
        transverse = float(np.linalg.norm(np.cross(first_normal, second_normal)))
        return shared_edges if transverse > 1.0e-12 * normal_scale else ()

    plane = (
        first_surface
        if isinstance(first_surface, Plane)
        else second_surface if isinstance(second_surface, Plane) else None
    )
    cylinder = (
        first_surface
        if isinstance(first_surface, Cylinder)
        else second_surface if isinstance(second_surface, Cylinder) else None
    )
    if plane is None or cylinder is None:
        return ()
    # A plane normal parallel to the cylinder axis meets the cylindrical
    # shell transversely around a ring.  Such a ring may already exist as a
    # generator boundary and can legitimately carry four shell elements
    # (two axial cylinder bands plus the two sides of the partitioned plate).
    alignment = abs(float(np.asarray(plane.normal) @ np.asarray(cylinder.axis)))
    return shared_edges if abs(alignment - 1.0) <= 1.0e-10 else ()


def _is_resolved_shared_vertex_touch(
    geometry: GeometryModel,
    first: int,
    second: int,
    result: Any,
) -> bool:
    """Whether the complete predicate result is one existing shared vertex."""

    shared = _shared_boundary_vertices(geometry, first, second)
    if (
        not shared
        or result.kind is not IntersectionKind.TOUCH_POINT
        or len(result.components) != 1
        or len(result.components[0].witnesses) != 1
    ):
        return False
    witness = np.asarray(result.components[0].witnesses[0], dtype=float)
    tolerance = max(
        float(result.tolerance_used or 0.0),
        64.0 * np.finfo(float).eps * max(1.0, float(np.linalg.norm(witness))),
    )
    return any(
        float(np.linalg.norm(witness - geometry.vertices[vertex].position))
        <= tolerance
        for vertex in shared
    )


def _face_pairs(
    geometry: GeometryModel,
    faces: Sequence[int],
    *,
    maximum_candidates: int | None,
    cancellation_check: CancellationCheck | None,
) -> tuple[tuple[int, int], ...]:
    selected = set(map(int, faces))
    pairs: set[tuple[int, int]] = set()
    for face_id in sorted(selected):
        bounds = geometry.conservative_face_bounds(face_id)
        if bounds is None:
            raise MeshError(f"face {face_id} has no conservative public bounds")
        for kind, candidate in geometry.spatial_candidates(
            bounds[:3],
            bounds[3:],
            kinds=("face",),
        ):
            if kind != "face" or candidate not in selected or candidate <= face_id:
                continue
            pairs.add((face_id, int(candidate)))
            if maximum_candidates is not None and len(pairs) > maximum_candidates:
                raise MeshError(
                    "structural preparation broad phase exceeded "
                    f"maximum_candidate_pairs={maximum_candidates}"
                )
            if len(pairs) % 64 == 0:
                _cancel(cancellation_check, "structural face broad phase")
    return tuple(sorted(pairs))


def _straight_edge_bounds(
    geometry: GeometryModel,
    edge_id: int,
) -> tuple[float, ...] | None:
    edge = geometry.edges[int(edge_id)]
    if not isinstance(edge.curve, Straight):
        return None
    points = np.asarray(
        (geometry.vertex_position(edge.start), geometry.vertex_position(edge.end)),
        dtype=float,
    )
    lower, upper = points.min(axis=0), points.max(axis=0)
    return (*lower, *upper)


def _member_edge_ids(geometry: GeometryModel, member_id: int) -> tuple[int, ...]:
    return tuple(
        int(geometry.member_edge_uses[use_id].edge_id)
        for use_id in geometry.members[int(member_id)].edge_use_ids
    )


def _member_pairs(
    geometry: GeometryModel,
    member_ids: Sequence[int],
    *,
    maximum_candidates: int | None,
    cancellation_check: CancellationCheck | None,
) -> tuple[tuple[int, int], ...]:
    selected = set(map(int, member_ids))
    pairs: set[tuple[int, int]] = set()
    unbounded: set[int] = set()
    for member_id in sorted(selected):
        for edge_id in _member_edge_ids(geometry, member_id):
            bounds = _straight_edge_bounds(geometry, edge_id)
            if bounds is None:
                unbounded.add(member_id)
                continue
            for kind, candidate_edge in geometry.spatial_candidates(
                bounds[:3], bounds[3:], kinds=("edge",)
            ):
                if kind != "edge":
                    continue
                for candidate_member in geometry.members_using_edge(candidate_edge):
                    if candidate_member in selected and candidate_member != member_id:
                        pairs.add(tuple(sorted((member_id, candidate_member))))
                        if maximum_candidates is not None and len(pairs) > maximum_candidates:
                            raise MeshError(
                                "structural preparation broad phase exceeded "
                                f"maximum_candidate_pairs={maximum_candidates}"
                            )
                        if len(pairs) % 64 == 0:
                            _cancel(cancellation_check, "structural member broad phase")
    for member_id in sorted(unbounded):
        for other in selected:
            if other == member_id:
                continue
            pairs.add(tuple(sorted((member_id, other))))
            if maximum_candidates is not None and len(pairs) > maximum_candidates:
                raise MeshError(
                    "structural preparation broad phase exceeded "
                    f"maximum_candidate_pairs={maximum_candidates}"
                )
            if len(pairs) % 64 == 0:
                _cancel(cancellation_check, "structural member broad phase")
    return tuple(sorted(pairs))


def _member_sheet_pairs(
    geometry: GeometryModel,
    member_ids: Sequence[int],
    sheet_ids: Sequence[int],
    face_sheet_membership: Mapping[int, Sequence[int]],
    *,
    maximum_candidates: int | None,
    cancellation_check: CancellationCheck | None,
) -> tuple[tuple[int, int], ...]:
    selected_sheets = set(map(int, sheet_ids))
    pairs: set[tuple[int, int]] = set()
    for member_id in sorted(set(map(int, member_ids))):
        unbounded = False
        for edge_id in _member_edge_ids(geometry, member_id):
            bounds = _straight_edge_bounds(geometry, edge_id)
            if bounds is None:
                unbounded = True
                continue
            for kind, face_id in geometry.spatial_candidates(
                bounds[:3], bounds[3:], kinds=("face",)
            ):
                if kind != "face":
                    continue
                for sheet_id in face_sheet_membership.get(int(face_id), ()):
                    if sheet_id not in selected_sheets:
                        continue
                    pairs.add((member_id, sheet_id))
                    if maximum_candidates is not None and len(pairs) > maximum_candidates:
                        raise MeshError(
                            "structural preparation broad phase exceeded "
                            f"maximum_candidate_pairs={maximum_candidates}"
                        )
                    if len(pairs) % 64 == 0:
                        _cancel(
                            cancellation_check,
                            "structural member/sheet broad phase",
                        )
        if unbounded:
            for sheet_id in selected_sheets:
                pairs.add((member_id, sheet_id))
                if maximum_candidates is not None and len(pairs) > maximum_candidates:
                    raise MeshError(
                        "structural preparation broad phase exceeded "
                        f"maximum_candidate_pairs={maximum_candidates}"
                    )
                if len(pairs) % 64 == 0:
                    _cancel(cancellation_check, "structural member/sheet broad phase")
    return tuple(sorted(pairs))


def _member_is_sheet_boundary(
    geometry: GeometryModel,
    member_id: int,
    sheet_id: int,
) -> bool:
    member_edges = set(_member_edge_ids(geometry, member_id))
    boundary_edges = {
        item.edge
        for face_use_id in geometry.sheets[sheet_id].face_use_ids
        for loop in (
            geometry.faces[geometry.face_uses[face_use_id].face_id].loop,
        )
        + geometry.faces[geometry.face_uses[face_use_id].face_id].holes
        for item in loop
    }
    return bool(member_edges.intersection(boundary_edges))


def _apply_connection(
    geometry: GeometryModel,
    first_kind: str,
    first_id: int,
    second_kind: str,
    second_id: int,
) -> tuple[bool, str | None, tuple[int, ...]]:
    first = geometry.handle(first_kind, first_id)
    second = geometry.handle(second_kind, second_id)
    try:
        result = query_intersection(geometry, first, second)
        if result.kind is IntersectionKind.DISJOINT:
            return False, None, ()
        if (
            first_kind == "face"
            and second_kind == "face"
            and _is_resolved_shared_vertex_touch(
                geometry, first_id, second_id, result
            )
        ):
            return False, "exact shared vertex topology", ()
        plan = plan_imprint(
            geometry,
            result,
            policy=ConnectionIntent.CONNECT,
        )
        if plan.operation is ImprintOperation.NO_TOPOLOGY:
            diagnostics = "; ".join(result.diagnostics)
            if result.kind is IntersectionKind.UNSUPPORTED:
                return False, diagnostics, ()
            raise MeshError(
                f"unqualified {first_kind}/{second_kind} relationship "
                f"{first_id}/{second_id}: {diagnostics or result.kind.value}"
            )
        application = apply_imprint(
            geometry,
            plan,
            policy=ConnectionIntent.CONNECT,
        )
    except GeometryError as error:
        raise MeshError(
            f"automatic {first_kind}/{second_kind} preparation failed for "
            f"{first_id}/{second_id}: {error}"
        ) from error
    changed = not application.change_set.is_empty
    declared_edges = (
        tuple(int(item.id) for item in application.face_intersection.edges)
        if first_kind == "face"
        and second_kind == "face"
        and application.face_intersection is not None
        else ()
    )
    return changed, None, declared_edges


def _report_hash(report: StructuralPreparationReport) -> str:
    payload = report.to_dict()
    payload["preparation_hash"] = ""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + sha256(encoded).hexdigest()


def _restore_collinear_mapped_corners(
    source: GeometryModel,
    working: GeometryModel,
    face_mapping: Mapping[int, tuple[int, ...]],
) -> None:
    """Restore mapped corners lost only to collinear imprint side splits."""

    angular_tolerance = 1.0e-7
    for source_face_id, descendants in face_mapping.items():
        if len(source.faces[source_face_id].corners) != 4:
            continue
        for face_id in descendants:
            face = working.faces[face_id]
            if len(face.corners) == 4 or face.holes or len(face.loop) < 4:
                continue
            deviations: list[float] = []
            for index in range(len(face.loop)):
                incoming = working.oriented_end_tangent(face.loop[index - 1])
                outgoing = working.oriented_start_tangent(face.loop[index])
                cosine = float(np.clip(incoming @ outgoing, -1.0, 1.0))
                deviations.append(float(np.arccos(cosine)))
            ranked = sorted(
                range(len(face.loop)),
                key=lambda index: (-deviations[index], index),
            )
            corners = tuple(sorted(ranked[:4]))
            if min(deviations[index] for index in corners) <= angular_tolerance:
                continue
            if any(
                deviations[index] > angular_tolerance
                for index in ranked[4:]
            ):
                continue
            working.set_face_corners(face_id, corners)


def prepare_structural_closure(
    geometry: GeometryModel,
    *,
    face_ids: Iterable[int] | None = None,
    beam_edges: Iterable[int] = (),
    options: StructuralPreparationOptions | Mapping[str, Any] | bool | None = None,
    cancellation_check: CancellationCheck | None = None,
    reuse_working_copy: bool = False,
) -> tuple[GeometryModel, StructuralPreparationReport | None]:
    """Return an exact, source-bound structural working closure.

    ``False`` disables automatic relationship creation. By default every path
    returns a detached clone. ``reuse_working_copy`` is reserved for callers
    that already own an isolated mesh-job closure and explicitly permit owner
    finalization on that closure.
    """

    policy = StructuralPreparationOptions.create(options)
    assert policy is not None
    source_faces = tuple(
        sorted(
            geometry.faces
            if face_ids is None
            else {int(item) for item in face_ids}
        )
    )
    source_edges = tuple(sorted({int(item) for item in beam_edges}))
    missing_faces = [item for item in source_faces if item not in geometry.faces]
    missing_edges = [item for item in source_edges if item not in geometry.edges]
    if missing_faces or missing_edges:
        raise MeshError(
            f"structural preparation references missing "
            f"face/edge {missing_faces[:1] or missing_edges[:1]}"
        )
    for name, count in (("maximum_face_records",len(geometry.faces)),
                        ("maximum_edge_records",len(geometry.edges))):
        limit=getattr(policy,name)
        if limit is not None and count>limit:
            raise MeshError(f"structural preparation exceeds explicit {name}={limit}")
    # Classification belongs to the atomic owner batch. Legacy discretization
    # and callers disabling that batch retain the independent overlap audit.
    def audit_overlaps():
        _cancel(cancellation_check, "structural preparation overlap broad phase")
        for pair in _face_pairs(geometry, source_faces,
                            maximum_candidates=policy.maximum_candidate_pairs,
                            cancellation_check=cancellation_check):
            from ._owner_trim_domains import validated_complementary_trim_domains
            _cancel(cancellation_check, "structural preparation overlap narrow phase")
            if validated_complementary_trim_domains(geometry, *pair,
                                                cancellation_check=cancellation_check):
                continue
            try:
                overlaps = find_coplanar_overlaps(geometry, candidate_pairs=(pair,))
            except OverlapQualificationError:
                # Historical curved overlap qualification can leave an exact
                # bilinear boundary contact unresolved. The new owner batch
                # may certify that pair; its positive-area/interior/unknown
                # refusals still propagate. Planning is read-only.
                plan_intersections(geometry,
                    tuple(geometry.handle("face", face) for face in pair),
                    policy=IntersectionBatchPolicy(
                        intent=ConnectionIntent.IMPRINT,
                        max_candidate_pairs=policy.maximum_candidate_pairs,
                        cancellation_check=lambda: (_cancel(cancellation_check,
                            "structural preparation overlap qualification") or False)))
                overlaps = ()
            if overlaps:
                item = overlaps[0]
                raise MeshError("positive-area coplanar overlap is not assigned implicitly "
                            f"(faces {item.first}/{item.second}, area={item.area:.7g}); "
                            "run the previewable Fragment Overlaps geometry command")
    if not policy.automatic_face_connections:
        audit_overlaps()
    no_automatic=not (policy.automatic_face_connections or policy.automatic_member_connections
                      or policy.automatic_member_sheet_connections)
    owned=(not policy.declare_missing_owners or (
        all(geometry._face_structural_uses.get(face) for face in source_faces)
        and all(geometry.members_using_edge(edge) for edge in source_edges)))
    working = (geometry if reuse_working_copy and no_automatic and owned else
               geometry.clone(include_features=False, preserve_identity=reuse_working_copy))
    temporary_sheets: list[int] = []
    temporary_members: list[int] = []
    diagnostics: list[str] = []
    queries = applications = 0
    face_connections = member_connections = member_sheet_connections = 0
    declared_face_connection_edges: set[int] = set()

    face_sheet_membership = _face_sheet_membership(working)
    if policy.declare_missing_owners:
        for face_id in source_faces:
            descendants = _resolved(working, "face", face_id)
            if not any(face_sheet_membership.get(item, ()) for item in descendants):
                temporary_sheets.append(
                    working.add_sheet(
                        descendants,
                        name=f"mesh closure for source face {face_id}",
                    )
                )
        face_sheet_membership = _face_sheet_membership(working)
        for edge_id in source_edges:
            descendants = _resolved(working, "edge", edge_id)
            if not any(working.members_using_edge(item) for item in descendants):
                temporary_members.append(
                    working.add_member(
                        descendants,
                        name=f"mesh closure for source edge {edge_id}",
                    )
                )
    elif policy.automatic_face_connections and source_faces and any(
        not face_sheet_membership.get(face_id, ()) for face_id in source_faces
    ):
        raise MeshError("automatic face preparation requires declared Sheet owners")
    elif (
        policy.automatic_member_connections
        or policy.automatic_member_sheet_connections
    ) and source_edges and any(
        not working.members_using_edge(edge_id) for edge_id in source_edges
    ):
        raise MeshError("automatic member preparation requires declared Member owners")

    if (policy.automatic_face_connections or policy.automatic_member_connections
            or policy.automatic_member_sheet_connections):
        members = sorted({member for source in source_edges
                          for edge in _resolved(working,"edge",source)
                          for member in working.members_using_edge(edge)})
        faces = _selected_descendant_faces(working,source_faces)
        operands = (*[working.handle("face",face) for face in faces],
                    *[working.handle("member",member) for member in members])
        legacy_curved_only = False
        if not members and faces and all(isinstance(working.faces[face].surface,
                (CoonsSurface, Cone, RuledSurface)) for face in faces):
            from anygeometry import query_trimmed_surface_charts
            unavailable = set()
            for face in faces:
                try:
                    query_trimmed_surface_charts(working, (working.handle("face", face),))
                except GeometryError as error:
                    if str(error) not in (f"face {face} has curved Coons boundaries",
                            f"face {face} is not planar", f"face {face} has an unsupported intersection surface"):
                        raise
                    unavailable.add(face)
            legacy_curved_only = len(unavailable) == len(faces)
        if legacy_curved_only:
            if policy.automatic_face_connections:
                audit_overlaps()
            # Existing Coons/Cone/Ruled authoring and trim operations keep their
            # explicit boundary topology and established discretization route.
            # They are outside the general Plane/Cylinder interior-joint contract.
            operands = ()
            diagnostics.append("legacy curved topology discretization; no automatic interior joints")
        elif len(faces)==1 and not members:
            # One isolated surface has no operand pair to prepare. Preserve
            # native Cone/Ruled meshing without claiming their intersections
            # are part of the Plane/Cylinder batch contract.
            operands=()
        elif not members:
            # Existing curved-boundary welds remain a supported compatibility
            # path. General interior Coons intersections are not inferred.
            from anygeometry import query_trimmed_surface_charts,query_intersection
            retained=[]
            for face_id in faces:
                try:
                    query_trimmed_surface_charts(working,(working.handle('face',face_id),))
                except GeometryError as error:
                    if str(error) not in (f'face {face_id} has curved Coons boundaries',
                                          f'face {face_id} is not planar'):
                        raise
                    compatible=True
                    for other in faces:
                        if other==face_id:continue
                        result=query_intersection(working,face_id,other)
                        shared=_shared_boundary_edges(working,face_id,other)
                        if (not result.classified or (result.components and
                                ('existing_shared_boundary_curve' not in result.diagnostics or not shared))):
                            compatible=False
                            break
                    if compatible:
                        diagnostics.append(f'face:{face_id} retains existing qualified curved-boundary joints')
                        continue
                retained.append(working.handle('face',face_id))
            operands=tuple(retained)
        def cancelled():
            _cancel(cancellation_check,"structural intersection batch")
            return False
        batch_policy = IntersectionBatchPolicy(ConnectionIntent.CONNECT,
            max_candidate_pairs=policy.maximum_candidate_pairs, cancellation_check=cancelled,
            face_connections=policy.automatic_face_connections,
            member_connections=policy.automatic_member_connections,
            member_face_connections=policy.automatic_member_sheet_connections)
        try:
            if policy.automatic_face_connections and len(faces)>1:
                _cancel(cancellation_check,'structural preparation overlap narrow phase')
            plan=plan_intersections(working,operands,policy=batch_policy)
            face_pairs=set()
            member_face_pairs=set()
            member_pairs=set()
            for arrangement in plan.arrangements:
                for path in arrangement.paths:
                    face_pairs.update(tuple(sorted((first,second))) for index,first in enumerate(path.owners)
                                      for second in path.owners[index+1:])
                    member_face_pairs.update((member,arrangement.face_id) for member in path.member_ids)
            for contact in plan.contacts:
                selected=sorted({member for member,_parameter in contact.member_parameters})
                member_pairs.update((first,second) for index,first in enumerate(selected)
                                    for second in selected[index+1:])
                if contact.face_id is not None:
                    member_face_pairs.update((member,contact.face_id) for member in selected)
            application=apply_intersections(working,plan,policy=batch_policy)
            declared_face_connection_edges.update(handle.id for handle in application.joint_edges
                                                  if len(working.faces_using_edge(handle.id))>1)
            applications=int(not application.reused)
            face_connections=len(face_pairs)*applications
            member_connections=len(member_pairs)*applications
            member_sheet_connections=len(member_face_pairs)*applications
            queries=len(face_pairs)+len(member_pairs)+len(member_face_pairs)
        except GeometryError as error:
            if 'positive-area overlap' in str(error):
                raise MeshError('positive-area overlap requires Fragment Overlaps to select ownership before meshing: '
                                +str(error)) from error
            raise MeshError(f"automatic intersection batch preparation failed: {error}") from error
        for name,count in (("maximum_face_records",len(working.faces)),
                           ("maximum_edge_records",len(working.edges))):
            limit=getattr(policy,name)
            if limit is not None and count>limit:
                raise MeshError(f"structural batch exceeds explicit {name}={limit}")

    _cancel(cancellation_check, "structural preparation exact lineage")
    face_mapping = {
        face_id: _resolved(working, "face", face_id)
        for face_id in geometry.faces
    }
    _restore_collinear_mapped_corners(geometry, working, face_mapping)
    edge_mapping: dict[int, tuple[int, ...]] = {}
    for position, edge_id in enumerate(geometry.edges):
        if position % 512 == 0:
            _cancel(cancellation_check, "structural preparation edge lineage")
        edge_mapping[edge_id] = _resolved(working, "edge", edge_id)
    report = StructuralPreparationReport(
        model_id=str(geometry.model_id),
        source_revision=geometry.revision,
        working_revision=working.revision,
        options=policy,
        source_to_working_faces=face_mapping,
        source_to_working_edges=edge_mapping,
        temporary_sheet_ids=tuple(temporary_sheets),
        temporary_member_ids=tuple(temporary_members),
        declared_face_connection_edges=tuple(
            sorted(
                {
                    descendant
                    for edge_id in declared_face_connection_edges
                    for descendant in _resolved(working, "edge", edge_id)
                }
            )
        ),
        candidate_queries=queries,
        applications=applications,
        face_connections=face_connections,
        member_connections=member_connections,
        member_sheet_connections=member_sheet_connections,
        diagnostics=tuple(diagnostics),
    )
    report = replace(report, preparation_hash=_report_hash(report))
    _cancel(cancellation_check, "structural preparation complete")
    if reuse_working_copy:
        geometry.restore_topology(working.topology_snapshot())
        working=geometry
        report=replace(report,working_revision=working.revision)
        report=replace(report,preparation_hash=_report_hash(report))
    return working, report
