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
from anygeometry.overlaps import find_coplanar_overlaps
from anygeometry.policies import ConnectionIntent
from anygeometry.predicates import IntersectionKind

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
    maximum_candidate_pairs: int = 100_000
    maximum_applications: int = 100_000
    maximum_face_records: int = 100_000
    maximum_edge_records: int = 200_000

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


def _share_boundary(geometry: GeometryModel, first: int, second: int) -> bool:
    """Whether two faces already share authoritative boundary topology.

    A common edge is the usual structural connection.  A common vertex is
    also already-resolved topology and needs no geometric imprint: grids with
    four mapped patches meeting at one corner necessarily contain diagonal
    face pairs that touch at that vertex only.  Sending those pairs through
    the geometric intersection planner misclassifies the legitimate grid
    corner as an unqualified free point touch.
    """

    first_loops = (geometry.faces[first].loop,) + geometry.faces[first].holes
    second_loops = (geometry.faces[second].loop,) + geometry.faces[second].holes
    first_edges = {
        item.edge
        for loop in first_loops
        for item in loop
    }
    if any(item.edge in first_edges for loop in second_loops for item in loop):
        return True
    first_vertices = {
        vertex
        for loop in first_loops
        for item in loop
        for vertex in (
            geometry.oriented_start_vertex(item),
            geometry.oriented_end_vertex(item),
        )
    }
    return any(
        vertex in first_vertices
        for loop in second_loops
        for item in loop
        for vertex in (
            geometry.oriented_start_vertex(item),
            geometry.oriented_end_vertex(item),
        )
    )


def _face_pairs(
    geometry: GeometryModel,
    faces: Sequence[int],
    *,
    maximum_candidates: int,
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
            if len(pairs) > maximum_candidates:
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
    maximum_candidates: int,
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
                        if len(pairs) > maximum_candidates:
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
            if len(pairs) > maximum_candidates:
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
    maximum_candidates: int,
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
                    if len(pairs) > maximum_candidates:
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
                if len(pairs) > maximum_candidates:
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
) -> tuple[bool, str | None]:
    first = geometry.handle(first_kind, first_id)
    second = geometry.handle(second_kind, second_id)
    try:
        result = query_intersection(geometry, first, second)
        if result.kind is IntersectionKind.DISJOINT:
            return False, None
        plan = plan_imprint(
            geometry,
            result,
            policy=ConnectionIntent.CONNECT,
        )
        if plan.operation is ImprintOperation.NO_TOPOLOGY:
            diagnostics = "; ".join(result.diagnostics)
            if result.kind is IntersectionKind.UNSUPPORTED:
                return False, diagnostics
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
    return changed, None


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


def prepare_structural_closure(
    geometry: GeometryModel,
    *,
    face_ids: Iterable[int] | None = None,
    beam_edges: Iterable[int] = (),
    options: StructuralPreparationOptions | Mapping[str, Any] | bool | None = None,
    cancellation_check: CancellationCheck | None = None,
) -> tuple[GeometryModel, StructuralPreparationReport | None]:
    """Return an exact, source-bound structural working closure.

    ``False`` disables automatic relationship creation but still returns a
    detached clone.  Every path therefore gives the mesh job an immutable
    working document and is resource bounded.
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
    if len(geometry.faces) > policy.maximum_face_records:
        raise MeshError(
            f"model has {len(geometry.faces)} face records; structural preparation "
            f"is bounded to {policy.maximum_face_records}"
        )
    if len(geometry.edges) > policy.maximum_edge_records:
        raise MeshError(
            f"model has {len(geometry.edges)} edge records; structural preparation "
            f"is bounded to {policy.maximum_edge_records}"
        )
    _cancel(cancellation_check, "structural preparation overlap broad phase")
    source_face_candidates = _face_pairs(
        geometry,
        source_faces,
        maximum_candidates=policy.maximum_candidate_pairs,
        cancellation_check=cancellation_check,
    )
    overlaps = []
    for position, pair in enumerate(source_face_candidates):
        if position % 16 == 0:
            _cancel(cancellation_check, "structural preparation overlap narrow phase")
        overlaps.extend(
            find_coplanar_overlaps(geometry, candidate_pairs=(pair,))
        )
        if len(overlaps) >= 8:
            break
    if overlaps:
        detail = ", ".join(
            f"faces {item.first}/{item.second}: {item.area:.7g} m^2"
            for item in overlaps[:8]
        )
        raise MeshError(
            "positive-area coplanar overlap is not assigned implicitly "
            f"({detail}); run the previewable Fragment Overlaps geometry command"
        )

    working = geometry.clone(include_features=False)
    temporary_sheets: list[int] = []
    temporary_members: list[int] = []
    diagnostics: list[str] = []
    queries = applications = 0
    face_connections = member_connections = member_sheet_connections = 0

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

    if policy.automatic_face_connections:
        settled: set[tuple[int, int]] = set()
        while True:
            changed = False
            descendants = _selected_descendant_faces(
                working,
                source_faces,
            )
            origin = {
                child: source
                for source in source_faces
                for child in _resolved(working, "face", source)
            }
            candidates = _face_pairs(
                working,
                descendants,
                maximum_candidates=policy.maximum_candidate_pairs,
                cancellation_check=cancellation_check,
            )
            for pair in candidates:
                if pair in settled or origin[pair[0]] == origin[pair[1]]:
                    continue
                if _share_boundary(working, *pair):
                    settled.add(pair)
                    continue
                queries += 1
                if queries > policy.maximum_candidate_pairs:
                    raise MeshError(
                        "structural preparation exceeded "
                        f"maximum_candidate_pairs={policy.maximum_candidate_pairs}"
                    )
                if queries % 64 == 0:
                    _cancel(cancellation_check, "structural face candidate queries")
                made, note = _apply_connection(
                    working, "face", pair[0], "face", pair[1]
                )
                if note:
                    diagnostics.append(f"faces {pair[0]}/{pair[1]}: {note}")
                if made:
                    applications += 1
                    face_connections += 1
                    if applications > policy.maximum_applications:
                        raise MeshError(
                            "structural preparation exceeded "
                            f"maximum_applications={policy.maximum_applications}"
                        )
                    if len(working.edges) > policy.maximum_edge_records:
                        raise MeshError(
                            "structural preparation topology exceeded "
                            f"maximum_edge_records={policy.maximum_edge_records}"
                        )
                    changed = True
                    break
                settled.add(pair)
            if not changed:
                break

    def selected_members() -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    member_id
                    for source_edge in source_edges
                    for edge_id in _resolved(working, "edge", source_edge)
                    for member_id in working.members_using_edge(edge_id)
                }
            )
        )

    def selected_sheets() -> tuple[int, ...]:
        membership = _face_sheet_membership(working)
        return tuple(
            sorted(
                {
                    sheet_id
                    for source_face in source_faces
                    for face_id in _resolved(working, "face", source_face)
                    for sheet_id in membership.get(face_id, ())
                }
            )
        )

    for group in ("member", "member_sheet"):
        if group == "member" and not policy.automatic_member_connections:
            continue
        if group == "member_sheet" and not policy.automatic_member_sheet_connections:
            continue
        settled: set[tuple[int, int]] = set()
        while True:
            pairs = (
                _member_pairs(
                    working,
                    selected_members(),
                    maximum_candidates=policy.maximum_candidate_pairs,
                    cancellation_check=cancellation_check,
                )
                if group == "member"
                else _member_sheet_pairs(
                    working,
                    selected_members(),
                    selected_sheets(),
                    _face_sheet_membership(working),
                    maximum_candidates=policy.maximum_candidate_pairs,
                    cancellation_check=cancellation_check,
                )
            )
            changed = False
            for first, second in pairs:
                if (first, second) in settled:
                    continue
                queries += 1
                if queries > policy.maximum_candidate_pairs:
                    raise MeshError(
                        "structural preparation exceeded "
                        f"maximum_candidate_pairs={policy.maximum_candidate_pairs}"
                    )
                if queries % 64 == 0:
                    _cancel(
                        cancellation_check,
                        f"structural {group} candidate queries",
                    )
                if group == "member":
                    made, note = _apply_connection(
                        working, "member", first, "member", second
                    )
                else:
                    if _member_is_sheet_boundary(working, first, second):
                        diagnostics.append(
                            f"member_sheet {first}/{second}: exact shared boundary topology"
                        )
                        settled.add((first, second))
                        continue
                    made, note = _apply_connection(
                        working, "member", first, "sheet", second
                    )
                if note:
                    diagnostics.append(f"{group} {first}/{second}: {note}")
                if not made:
                    settled.add((first, second))
                    continue
                applications += 1
                if group == "member":
                    member_connections += 1
                else:
                    member_sheet_connections += 1
                if applications > policy.maximum_applications:
                    raise MeshError(
                        "structural preparation exceeded "
                        f"maximum_applications={policy.maximum_applications}"
                    )
                if len(working.edges) > policy.maximum_edge_records:
                    raise MeshError(
                        "structural preparation topology exceeded "
                        f"maximum_edge_records={policy.maximum_edge_records}"
                    )
                changed = True
                break
            if not changed:
                break

    _cancel(cancellation_check, "structural preparation exact lineage")
    face_mapping = {
        face_id: _resolved(working, "face", face_id)
        for face_id in geometry.faces
    }
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
        candidate_queries=queries,
        applications=applications,
        face_connections=face_connections,
        member_connections=member_connections,
        member_sheet_connections=member_sheet_connections,
        diagnostics=tuple(diagnostics),
    )
    report = replace(report, preparation_hash=_report_hash(report))
    _cancel(cancellation_check, "structural preparation complete")
    return working, report
