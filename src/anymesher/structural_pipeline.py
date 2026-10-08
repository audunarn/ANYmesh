"""Component-local structural meshing orchestration.

This module deliberately does not infer welds from proximity.  Components and
mesh connections are formed only by persistent Sheet/Member ownership plus
Attachment/Junction intent from ANYgeometry.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import Lock
from types import MappingProxyType
from typing import Generic, TypeVar

import numpy as np
from anygeometry.structural import AttachmentTargetKind, JunctionKind
from anygeometry.transactions import ChangeSet

from .boundary import GlobalEdgeBoundaryRegistry, MemberRegistry
from .errors import MeshError
from .mesh import Coupling, Mesh
from .mesh_bvh import (
    MeshElementBVH,
    inverse_interpolate,
    normalized_element_filter,
)
from .meshing_view import GeometryMeshingView, StaleMeshingViewError

__all__ = [
    "ComponentGenerationCache",
    "ComponentKey",
    "ConnectivityAction",
    "ConnectivityReport",
    "GeometryMutationPolicy",
    "JobToken",
    "MutationPolicy",
    "OverlapPolicy",
    "PreflightIssue",
    "PreflightState",
    "PreflightStatus",
    "StructuralComponent",
    "StructuralMeshingPipeline",
    "build_structural_components",
]

ComponentKey = tuple[tuple[int, ...], tuple[int, ...]]
T = TypeVar("T")


class OverlapPolicy(StrEnum):
    REJECT = "reject"
    KEEP_SEPARATE = "keep_separate"
    CONNECT_DECLARED = "connect_declared"


class GeometryMutationPolicy(StrEnum):
    """Ownership contract for geometry supplied to one mesh generation.

    ``READ_ONLY`` preserves an editable source by preparing on a clone.
    ``WORKING_COPY`` declares that the caller already owns an isolated mesh-job
    copy. That copy can be consumed and structurally finalized directly without
    another full topology clone.
    """

    READ_ONLY = "read_only"
    WORKING_COPY = "working_copy"


MutationPolicy = GeometryMutationPolicy


class PreflightStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class StructuralComponent:
    sheet_ids: tuple[int, ...]
    member_ids: tuple[int, ...]
    attachment_ids: tuple[int, ...] = ()
    junction_ids: tuple[int, ...] = ()

    @property
    def key(self) -> ComponentKey:
        return (self.sheet_ids, self.member_ids)


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    code: str
    message: str
    entities: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class PreflightState:
    component: StructuralComponent
    status: PreflightStatus
    issues: tuple[PreflightIssue, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is PreflightStatus.READY


@dataclass(frozen=True, slots=True)
class JobToken:
    model_id: object
    revision: int
    component: Hashable
    generation: int


class ComponentGenerationCache(Generic[T]):
    """Per-component cache with generation-checked background publication."""

    def __init__(self, model_id: object, revision: int = 0) -> None:
        self.model_id = model_id
        self.revision = int(revision)
        self._generations: dict[Hashable, int] = {}
        self._values: dict[Hashable, tuple[int, T]] = {}
        self._lock = Lock()

    def generation(self, component: Hashable) -> int:
        with self._lock:
            return self._generations.get(component, 0)

    def begin(self, component: Hashable, *, revision: int | None = None) -> JobToken:
        with self._lock:
            return JobToken(
                self.model_id,
                self.revision if revision is None else int(revision),
                component,
                self._generations.get(component, 0),
            )

    def invalidate(self, components: Iterable[Hashable]) -> None:
        with self._lock:
            for component in set(components):
                self._generations[component] = self._generations.get(component, 0) + 1
                self._values.pop(component, None)

    def invalidate_all(self) -> None:
        with self._lock:
            components = set(self._generations) | set(self._values)
            for component in components:
                self._generations[component] = self._generations.get(component, 0) + 1
            self._values.clear()

    def advance_revision(self, revision: int) -> None:
        made = int(revision)
        with self._lock:
            if made < self.revision:
                raise ValueError("cache revision cannot move backwards")
            self.revision = made

    def publish(self, token: JobToken, value: T) -> bool:
        """Store a result only if its component generation is still current."""

        with self._lock:
            if (
                token.model_id != self.model_id
                or token.revision > self.revision
                or token.generation != self._generations.get(token.component, 0)
            ):
                return False
            self._values[token.component] = (token.generation, value)
            return True

    def get(self, component: Hashable, default: T | None = None) -> T | None:
        with self._lock:
            value = self._values.get(component)
            if value is None or value[0] != self._generations.get(component, 0):
                return default
            return value[1]


class _UnionFind:
    def __init__(self, values: Iterable[tuple[str, int]]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: tuple[str, int]) -> tuple[str, int]:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: tuple[str, int], second: tuple[str, int]) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def build_structural_components(
    view: GeometryMeshingView,
) -> tuple[StructuralComponent, ...]:
    """Build declared connectivity components without geometric proximity."""

    nodes = [*(('sheet', key) for key in view.sheets), *(('member', key) for key in view.members)]
    union = _UnionFind(nodes)
    for attachment in view.attachments.values():
        if attachment.member_id is None:
            continue
        member_node = ("member", int(attachment.member_id))
        if member_node not in union.parent:
            continue
        owners = (
            view.sheets_for_face(attachment.target_id)
            if attachment.target_kind is AttachmentTargetKind.FACE
            else view.sheets_using_edge(attachment.target_id)
        )
        for sheet_id in owners:
            sheet_node = ("sheet", int(sheet_id))
            if sheet_node in union.parent:
                union.union(member_node, sheet_node)
    for junction in view.junctions.values():
        participants = [
            *(("member", int(item.member_id)) for item in junction.member_uses),
            *(("sheet", int(item)) for item in junction.sheet_ids),
        ]
        participants = [item for item in participants if item in union.parent]
        for item in participants[1:]:
            union.union(participants[0], item)

    groups: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for node in nodes:
        groups.setdefault(union.find(node), set()).add(node)
    made: list[StructuralComponent] = []
    for values in groups.values():
        sheets = tuple(sorted(item[1] for item in values if item[0] == "sheet"))
        members = tuple(sorted(item[1] for item in values if item[0] == "member"))
        attachment_ids = tuple(
            sorted(
                item.id
                for item in view.attachments.values()
                if item.member_id in members
            )
        )
        junction_ids = tuple(
            sorted(
                item.id
                for item in view.junctions.values()
                if any(use.member_id in members for use in item.member_uses)
                or any(sheet in sheets for sheet in item.sheet_ids)
            )
        )
        made.append(StructuralComponent(sheets, members, attachment_ids, junction_ids))
    made.sort(key=lambda item: item.key)
    return tuple(made)


@dataclass(frozen=True, slots=True)
class ConnectivityAction:
    kind: str
    source: tuple[str, int]
    target: tuple[str, int]
    record_id: int | None = None


@dataclass(frozen=True, slots=True)
class ConnectivityReport:
    actions: tuple[ConnectivityAction, ...]
    issues: tuple[PreflightIssue, ...]
    states: tuple[PreflightState, ...]

    @property
    def connected(self) -> int:
        return len(self.actions)


class _FaceTargetIndex:
    """Invocation-local index of one attachment target face.

    Shell connectivity and node coordinates are read-only while connectivity
    is applied, so the cached element filter, node membership, and effective
    tolerance hold for the whole application.
    """

    __slots__ = ("elements", "filter", "member_nodes", "tolerance")

    def __init__(
        self,
        elements: tuple[int, ...],
        member_nodes: frozenset[int],
        tolerance: float,
    ) -> None:
        self.elements = elements
        self.filter = normalized_element_filter(elements)
        self.member_nodes = member_nodes
        self.tolerance = float(tolerance)


class _ConnectivityIndex:
    """Coupling indexes whose lifetime is one ``apply_connectivity`` call.

    The target-face and target-edge indexes are shell-scoped: shell
    connectivity is not mutated during connectivity application, so they
    cannot go stale.  The junction phase rewrites beam node identifiers, so
    the beam-node coupling index is invalidated there and rebuilt lazily if
    it is needed again; a shell or junction mutation therefore never leaks a
    stale coupling index into a later application.
    """

    def __init__(self, mesh: Mesh, pipeline: "StructuralMeshingPipeline") -> None:
        self._mesh = mesh
        self._pipeline = pipeline
        self._faces: dict[int, _FaceTargetIndex] = {}
        self._edge_filters: dict[int, frozenset[int]] = {}
        self._beam_couplings: dict[int, Coupling] | None = None

    def face(self, face_id: int) -> _FaceTargetIndex:
        key = int(face_id)
        index = self._faces.get(key)
        if index is None:
            mesh = self._mesh
            elements = tuple(mesh.elements_of_face.get(key, ()))
            member_nodes = frozenset(
                int(node)
                for element in elements
                for node in (
                    *mesh.quads.get(element, ()),
                    *mesh.tris.get(element, ()),
                )
            )
            if elements:
                points = np.asarray(
                    [
                        mesh.nodes[node]
                        for element in elements
                        for node in mesh.corners_of(element)
                    ],
                    dtype=float,
                )
                extent = float(np.max(np.ptp(points, axis=0)))
            else:
                extent = 0.0
            index = _FaceTargetIndex(
                elements,
                member_nodes,
                self._pipeline.view.effective_length(extent),
            )
            self._faces[key] = index
        return index

    def edge_filter(self, edge_id: int) -> frozenset[int]:
        key = int(edge_id)
        allowed = self._edge_filters.get(key)
        if allowed is None:
            allowed = normalized_element_filter(
                self._pipeline._target_elements_for_edge(self._mesh, key)
            )
            self._edge_filters[key] = allowed
        return allowed

    def coupling_for_beam_node(self, beam_node: int) -> Coupling | None:
        """Return the first coupling record of ``beam_node``, if any.

        The index preserves insertion order, so the first record wins
        exactly as the previous linear scan did.
        """

        index = self._beam_couplings
        if index is None:
            index = {}
            for record in self._mesh.couplings.values():
                index.setdefault(int(record.beam_node), record)
            self._beam_couplings = index
        return index.get(int(beam_node))

    def record_coupling(self, beam_node: int, record: Coupling) -> None:
        if self._beam_couplings is not None:
            self._beam_couplings.setdefault(int(beam_node), record)

    def invalidate_beam_couplings(self) -> None:
        self._beam_couplings = None


class StructuralMeshingPipeline:
    """Declared-only connectivity and local invalidation for one view."""

    def __init__(
        self,
        view: GeometryMeshingView,
        *,
        overlap_policy: OverlapPolicy | str,
        mutation_policy: GeometryMutationPolicy | str,
        active_sheet_ids: Iterable[int] | None = None,
        active_member_ids: Iterable[int] | None = None,
    ) -> None:
        if not isinstance(view, GeometryMeshingView):
            raise TypeError("pipeline requires a GeometryMeshingView")
        try:
            self.overlap_policy = OverlapPolicy(overlap_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("an explicit valid overlap policy is required") from error
        try:
            self.mutation_policy = GeometryMutationPolicy(mutation_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("an explicit valid mutation policy is required") from error
        self.view = view
        components = build_structural_components(view)
        if active_sheet_ids is not None or active_member_ids is not None:
            active_sheets = frozenset(
                int(item) for item in (() if active_sheet_ids is None else active_sheet_ids)
            )
            active_members = frozenset(
                int(item) for item in (() if active_member_ids is None else active_member_ids)
            )
            missing_sheets = sorted(active_sheets.difference(view.sheets))
            if missing_sheets:
                raise MeshError(f"no sheet {missing_sheets[0]}")
            missing_members = sorted(active_members.difference(view.members))
            if missing_members:
                raise MeshError(f"no structural member {missing_members[0]}")
            # Selection seeds components; it does not trim them.  Retaining the
            # whole declared connectivity component ensures an omitted but
            # connected owner still produces a blocking unmeshed diagnostic.
            components = tuple(
                component
                for component in components
                if active_sheets.intersection(component.sheet_ids)
                or active_members.intersection(component.member_ids)
            )
        self.components = components
        self.component_map: Mapping[ComponentKey, StructuralComponent] = MappingProxyType(
            {item.key: item for item in self.components}
        )
        self.boundaries = GlobalEdgeBoundaryRegistry(view)
        self.members = MemberRegistry(view)
        # Invocation-local cache of edge-node coordinates; one apply_connectivity
        # invocation owns it and merges invalidate it (see _connect_junction).
        self._edge_coordinate_cache: dict[tuple[int, tuple[int, ...]], np.ndarray] | None = None
        self.cache: ComponentGenerationCache[object] = ComponentGenerationCache(
            view.model_id, view.revision
        )

    def _preflight_component(
        self, component: StructuralComponent, mesh: Mesh | None
    ) -> PreflightState:
        issues: list[PreflightIssue] = []
        try:
            self.view.assert_current()
        except StaleMeshingViewError as error:
            return PreflightState(
                component,
                PreflightStatus.STALE,
                (PreflightIssue("stale-view", str(error)),),
            )

        for sheet_id in component.sheet_ids:
            sheet = self.view.sheets.get(sheet_id)
            if sheet is None:
                issues.append(
                    PreflightIssue("missing-sheet", f"no sheet {sheet_id}", (("sheet", sheet_id),))
                )
                continue
            for face_id in self.view.faces_for_sheet(sheet_id):
                if face_id not in self.view.faces:
                    issues.append(
                        PreflightIssue(
                            "missing-face",
                            f"sheet {sheet_id} references missing face {face_id}",
                            (("sheet", sheet_id), ("face", face_id)),
                        )
                    )
                elif mesh is not None and not mesh.elements_of_face.get(face_id):
                    issues.append(
                        PreflightIssue(
                            "unmeshed-face",
                            f"sheet {sheet_id} face {face_id} has no shell elements",
                            (("sheet", sheet_id), ("face", face_id)),
                        )
                    )

        for member_id in component.member_ids:
            for span in self.members.spans(member_id):
                if span.edge_id not in self.view.edges:
                    issues.append(
                        PreflightIssue(
                            "missing-member-edge",
                            f"member {member_id} references missing edge {span.edge_id}",
                            (("member", member_id), ("edge", span.edge_id)),
                        )
                    )
                elif mesh is not None and (
                    not mesh.nodes_of_edge.get(span.edge_id)
                    or not mesh.elements_of_edge.get(span.edge_id)
                ):
                    issues.append(
                        PreflightIssue(
                            "unmeshed-member",
                            f"member {member_id} edge {span.edge_id} has no beam mesh",
                            (("member", member_id), ("edge", span.edge_id)),
                        )
                    )

        for attachment_id in component.attachment_ids:
            attachment = self.view.attachments.get(attachment_id)
            if attachment is None:
                issues.append(
                    PreflightIssue(
                        "missing-attachment",
                        f"no attachment {attachment_id}",
                        (("attachment", attachment_id),),
                    )
                )
                continue
            target_store = {
                AttachmentTargetKind.FACE: self.view.faces,
                AttachmentTargetKind.EDGE: self.view.edges,
                AttachmentTargetKind.SHEET: self.view.sheets,
                AttachmentTargetKind.MEMBER: self.view.members,
            }.get(attachment.target_kind, {})
            if attachment.target_id not in target_store:
                issues.append(
                    PreflightIssue(
                        "missing-attachment-target",
                        f"attachment {attachment_id} target is missing",
                        (("attachment", attachment_id), attachment.target_key),
                    )
                )

        for junction_id in component.junction_ids:
            junction = self.view.junctions[junction_id]
            if (
                junction.kind is JunctionKind.OVERLAP
                and self.overlap_policy is OverlapPolicy.REJECT
            ):
                issues.append(
                    PreflightIssue(
                        "overlap-rejected",
                        f"declared overlap junction {junction_id} is rejected by policy",
                        (("junction", junction_id),),
                    )
                )
        return PreflightState(
            component,
            PreflightStatus.READY if not issues else PreflightStatus.BLOCKED,
            tuple(issues),
        )

    def preflight(self, mesh: Mesh | None = None) -> tuple[PreflightState, ...]:
        return tuple(self._preflight_component(item, mesh) for item in self.components)

    def preflight_states(
        self, mesh: Mesh | None = None
    ) -> Mapping[ComponentKey, PreflightState]:
        return MappingProxyType({item.component.key: item for item in self.preflight(mesh)})

    def affected_components(self, change: ChangeSet) -> tuple[ComponentKey, ...]:
        if change.document_settings_changed:
            return tuple(item.key for item in self.components)
        affected: set[ComponentKey] = set()
        unknown = False
        all_keys = {
            *change.changed,
            *change.ownership_changes,
            *change.member_changes,
            *change.attachment_changes,
            *change.invalidated_caches,
        }
        for kind, identifier in all_keys:
            matched = False
            for component in self.components:
                if kind == "sheet" and identifier in component.sheet_ids:
                    matched = True
                elif kind == "member" and identifier in component.member_ids:
                    matched = True
                elif kind == "attachment" and identifier in component.attachment_ids:
                    matched = True
                elif kind == "junction" and identifier in component.junction_ids:
                    matched = True
                elif kind == "face" and any(
                    identifier in self.view.faces_for_sheet(sheet)
                    for sheet in component.sheet_ids
                ):
                    matched = True
                elif kind == "edge" and (
                    any(
                        identifier == span.edge_id
                        for member in component.member_ids
                        for span in self.members.spans(member)
                    )
                    or any(
                        sheet in component.sheet_ids
                        for sheet in self.view.sheets_using_edge(identifier)
                    )
                ):
                    matched = True
                if matched:
                    affected.add(component.key)
                    matched = False
            if kind in {"part", "face_use", "coedge", "member_edge_use"}:
                unknown = True
        if unknown:
            return tuple(item.key for item in self.components)
        return tuple(sorted(affected))

    def consume_change(self, change: ChangeSet) -> tuple[ComponentKey, ...]:
        affected = self.affected_components(change)
        self.cache.advance_revision(change.revision_after)
        self.cache.invalidate(affected)
        return affected

    # Screening slack for the vectorized nearest-node batch, in units of
    # relative machine epsilon.  Batch and scalar norms of the same 3-vector
    # differ by only a few ulps, so every candidate that could win the exact
    # scalar comparison lies within this margin of the batch minimum.
    _EDGE_SCREEN_MARGIN_ULPS = 64.0

    def _edge_node_coordinates(
        self, mesh: Mesh, edge_id: int, candidates: tuple[int, ...]
    ) -> np.ndarray:
        """Return candidate coordinates, cached for this invocation."""

        cache = self._edge_coordinate_cache
        if cache is None:
            return np.asarray([mesh.nodes[node] for node in candidates], dtype=float)
        key = (int(edge_id), candidates)
        coordinates = cache.get(key)
        if coordinates is None:
            coordinates = np.asarray(
                [mesh.nodes[node] for node in candidates], dtype=float
            )
            cache[key] = coordinates
        return coordinates

    @staticmethod
    def _scalar_edge_distances(
        mesh: Mesh, candidates: tuple[int, ...], expected: object
    ) -> list[float]:
        return [
            float(np.linalg.norm(mesh.nodes[node] - expected)) for node in candidates
        ]

    def _nearest_edge_node(
        self,
        mesh: Mesh,
        edge_id: int,
        expected: object,
        candidates: tuple[int, ...],
    ) -> tuple[int, float]:
        """Return the nearest candidate index and its exact scalar distance.

        A vectorized batch norm only screens the candidates.  The original
        scalar ``np.linalg.norm`` distances of the screened nodes decide the
        winner and the tolerance, preserving the original node-order tie
        behavior; batch and scalar norms differ by ulps, so the batch never
        decides the winner directly.  Nonfinite or otherwise uncertain input
        falls back to the full original scalar scan.
        """

        coordinates = self._edge_node_coordinates(mesh, edge_id, candidates)
        made_expected = np.asarray(expected, dtype=float)
        if (
            not np.all(np.isfinite(coordinates))
            or made_expected.shape != (3,)
            or not np.all(np.isfinite(made_expected))
        ):
            distances = self._scalar_edge_distances(mesh, candidates, expected)
            index = int(np.argmin(distances))
            return index, distances[index]
        differences = coordinates - made_expected
        batch = np.sqrt(np.einsum("ij,ij->i", differences, differences))
        if not np.all(np.isfinite(batch)):
            distances = self._scalar_edge_distances(mesh, candidates, expected)
            index = int(np.argmin(distances))
            return index, distances[index]
        best = float(np.min(batch))
        margin = (
            self._EDGE_SCREEN_MARGIN_ULPS
            * float(np.finfo(float).eps)
            * max(1.0, abs(best))
        )
        winner = -1
        winner_distance = float("inf")
        for position in np.flatnonzero(batch <= best + margin):
            distance = float(
                np.linalg.norm(mesh.nodes[candidates[position]] - expected)
            )
            if distance < winner_distance:
                winner = int(position)
                winner_distance = distance
        if winner < 0:
            distances = self._scalar_edge_distances(mesh, candidates, expected)
            winner = int(np.argmin(distances))
            winner_distance = distances[winner]
        return winner, winner_distance

    def _edge_node(
        self, mesh: Mesh, edge_id: int, parameter: float, owner: Hashable
    ) -> int | None:
        expected = self.view.edge_point(edge_id, parameter)
        candidates = tuple(mesh.nodes_of_edge.get(edge_id, ()))
        if not candidates:
            return None
        index, distance = self._nearest_edge_node(mesh, edge_id, expected, candidates)
        tolerance = self.view.effective_length(self.view.edge_length(edge_id))
        if distance > tolerance:
            return None
        node = int(candidates[index])
        self.boundaries.register(
            edge_id,
            parameter,
            expected,
            node_id=node,
            owner=owner,
        )
        return node

    def _member_node(self, mesh: Mesh, member_id: int, parameter: float) -> int | None:
        location = self.members.locate(member_id, parameter)
        axis = tuple(mesh.nodes_of_edge.get(location.span.edge_id, ()))
        node = self._edge_node(
            mesh,
            location.span.edge_id,
            location.edge_parameter,
            ("member", member_id),
        )
        if node is None:
            return None
        offsets = tuple(mesh.offset_nodes_of_edge.get(location.span.edge_id, ()))
        if offsets and len(offsets) == len(axis):
            return int(offsets[axis.index(node)])
        return node

    def _member_stations(
        self, mesh: Mesh, member_id: int, lower: float, upper: float
    ) -> tuple[tuple[float, int], ...]:
        stations: dict[int, float] = {}
        parameter_tolerance = float(self.view.tolerance.parameter)
        for span in self.members.spans(member_id):
            axis = tuple(mesh.nodes_of_edge.get(span.edge_id, ()))
            offsets = tuple(mesh.offset_nodes_of_edge.get(span.edge_id, ()))
            for index, node in enumerate(axis):
                _point, edge_parameter, residual = self.view.closest_edge_point(
                    span.edge_id, mesh.nodes[node]
                )
                if residual > self.view.effective_length(self.view.edge_length(span.edge_id)):
                    continue
                member_parameter = span.member_parameter(edge_parameter)
                if lower - parameter_tolerance <= member_parameter <= upper + parameter_tolerance:
                    actual = int(offsets[index]) if len(offsets) == len(axis) else int(node)
                    stations[actual] = float(np.clip(member_parameter, lower, upper))
        return tuple(sorted(((value, node) for node, value in stations.items())))

    @staticmethod
    def _mapped_parameter(parameter: float, source: object, target: object) -> float:
        source_length = float(source.end - source.start)
        fraction = 0.0 if source_length == 0.0 else (parameter - source.start) / source_length
        return float(target.start + np.clip(fraction, 0.0, 1.0) * (target.end - target.start))

    def _target_elements_for_edge(self, mesh: Mesh, edge_id: int) -> tuple[int, ...]:
        faces = {
            face_id
            for sheet_id in self.view.sheets_using_edge(edge_id)
            for face_id in self.view.faces_for_sheet(sheet_id)
            if any(item.edge == edge_id for item in self.view.faces[face_id].loop)
        }
        return tuple(
            sorted(
                {
                    element
                    for face in faces
                    for element in mesh.elements_of_face.get(face, ())
                }
            )
        )

    @staticmethod
    def _quadratic_face_projection_tolerance(
        mesh: Mesh, element_ids: Iterable[int], base: float
    ) -> float:
        """Bound owner-surface to quadratic-shell projection by edge curvature.

        Exact curved owner points do not generally lie exactly on a polynomial
        Q8/T6 interior map.  The canonical midside-to-chord deviation supplies
        a local, mesh-derived geometric scale without relaxing linear meshes.
        """

        tolerance = float(base)
        for element_id in element_ids:
            body = mesh.quads.get(int(element_id))
            if body is None:
                body = mesh.tris.get(int(element_id))
            if body is None or len(body) not in (6, 8):
                continue
            corner_count = 4 if len(body) == 8 else 3
            corners = tuple(int(node) for node in body[:corner_count])
            midsides = tuple(int(node) for node in body[corner_count:])
            for index, midside in enumerate(midsides):
                first = corners[index]
                second = corners[(index + 1) % corner_count]
                chord_midpoint = 0.5 * (mesh.nodes[first] + mesh.nodes[second])
                deviation = float(np.linalg.norm(mesh.nodes[midside] - chord_midpoint))
                tolerance = max(tolerance, float(base) + deviation)
        return tolerance

    def _linear_face_parameter_hit(
        self, mesh: Mesh, face_id: int, u: float, v: float,
        point: np.ndarray, element_ids: Iterable[int],
    ) -> tuple[tuple[int, ...], tuple[float, ...], np.ndarray] | None:
        """Locate a curved source station on its declared linear host face.

        A linear chord shell generally does not contain the exact owner point.
        Its source UV, however, must belong to an element on the declared
        target face.  The resulting physical gap remains in the coupling
        eccentricity; it is never erased by a proximity weld.
        """
        candidates = []
        target = np.array((float(u), float(v), 0.0))
        for element_id in sorted(element_ids):
            body = mesh.quads.get(int(element_id))
            family = "Q4"
            if body is None:
                body = mesh.tris.get(int(element_id))
                family = "T3"
            if body is None or len(body) not in (3, 4):
                continue
            nodes = tuple(int(node) for node in body)
            coords = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
            uv = np.asarray([self.view.face_local_uv(face_id, xyz) for xyz in coords])
            parametric = np.column_stack((uv, np.zeros(len(nodes))))
            hit = inverse_interpolate(family, parametric, target, tolerance=1e-8)
            if hit is None:
                continue
            parameter_projected = np.asarray(hit.weights @ coords, dtype=float)
            chord_gap = float(np.linalg.norm(point - parameter_projected))
            physical_hit = inverse_interpolate(
                family, coords, point,
                tolerance=max(1e-8, 2.0 * chord_gap),
            )
            if physical_hit is None:
                continue
            projected = np.asarray(physical_hit.point, dtype=float)
            candidates.append((float(np.linalg.norm(point - projected)), int(element_id),
                               nodes, tuple(float(w) for w in physical_hit.weights), projected))
        if not candidates:
            return None
        _, _, nodes, weights, projected = min(candidates, key=lambda item: (item[0], item[1]))
        return nodes, weights, projected

    def _add_attachment_coupling(
        self,
        mesh: Mesh,
        bvh: MeshElementBVH,
        attachment: object,
        member_parameter: float,
        beam_node: int,
        next_record: list[int],
        index: _ConnectivityIndex,
    ) -> tuple[ConnectivityAction | None, PreflightIssue | None]:
        actual = np.asarray(mesh.nodes[beam_node], dtype=float)
        if attachment.target_kind is AttachmentTargetKind.EDGE:
            target_parameter = self._mapped_parameter(
                member_parameter, attachment.member_range, attachment.target_parameters[0]
            )
            # A canonical shell edge can also be this Member's axis span.
            # Its owner orientation, rather than interval ordering, maps the
            # parent station to that exact edge (including shared endpoints).
            spans = [span for span in self.members.spans(attachment.member_id)
                     if span.edge_id == attachment.target_id
                     and span.contains(member_parameter, tolerance=self.view.tolerance.parameter)]
            if len(spans) == 1:
                target_parameter = spans[0].edge_parameter(member_parameter)
            point = self.view.edge_point(attachment.target_id, target_parameter)
            master = self._edge_node(
                mesh,
                attachment.target_id,
                target_parameter,
                ("attachment", int(attachment.id)),
            )
            if master is not None:
                if master == beam_node:
                    return (
                        ConnectivityAction(
                            "shared-node",
                            ("member", int(attachment.member_id)),
                            ("edge", int(attachment.target_id)),
                        ),
                        None,
                    )
                plate_nodes = (master,)
                weights = (1.0,)
                projected = np.asarray(mesh.nodes[master], dtype=float)
            else:
                hit = bvh.locate(
                    point,
                    element_ids=index.edge_filter(attachment.target_id),
                    tolerance=self.view.effective_length(
                        self.view.edge_length(attachment.target_id)
                    ),
                )
                if hit is None:
                    return None, PreflightIssue(
                        "unresolved-attachment",
                        f"attachment {attachment.id} target edge station is not meshed",
                        (("attachment", int(attachment.id)),),
                    )
                plate_nodes, weights = hit.node_ids, hit.weights
                projected = np.asarray(hit.point, dtype=float)
        else:
            target_face_id = int(attachment.target_id)
            if attachment.target_kind is AttachmentTargetKind.SHEET:
                metadata = attachment.metadata.to_dict()
                face_sequence = tuple(
                    int(value) for value in metadata.get("face_sequence", ())
                )
                if len(face_sequence) != 1:
                    return None, PreflightIssue(
                        "ambiguous-sheet-attachment-face",
                        f"attachment {attachment.id} must identify one exact "
                        "face in its target sheet",
                        (("attachment", int(attachment.id)),),
                    )
                target_face_id = face_sequence[0]
                if target_face_id not in self.view.faces_for_sheet(
                    int(attachment.target_id)
                ):
                    return None, PreflightIssue(
                        "invalid-sheet-attachment-face",
                        f"attachment {attachment.id} face {target_face_id} is not "
                        f"in sheet {attachment.target_id}",
                        (("attachment", int(attachment.id)),),
                    )
            elif attachment.target_kind is not AttachmentTargetKind.FACE:
                return None, PreflightIssue(
                    "unsupported-attachment-target",
                    f"attachment {attachment.id} target kind "
                    f"{attachment.target_kind.value!r} is not a meshed face",
                    (("attachment", int(attachment.id)),),
                )
            u = self._mapped_parameter(
                member_parameter, attachment.member_range, attachment.target_parameters[0]
            )
            v = self._mapped_parameter(
                member_parameter, attachment.member_range, attachment.target_parameters[1]
            )
            point = self.view.face_point(target_face_id, u, v)
            face_index = index.face(target_face_id)
            allowed = face_index.elements
            base_tolerance = face_index.tolerance
            if (beam_node in face_index.member_nodes
                    and float(np.linalg.norm(actual-point)) <= base_tolerance):
                # Exact prepared topology already supplies this shell station.
                # Shape-function roundoff at a corner must not create a
                # redundant MPC that constrains the same node to itself.
                return (ConnectivityAction('shared-node',
                    ('member',int(attachment.member_id)),('face',target_face_id)),None)
            hit = bvh.locate(
                point,
                element_ids=face_index.filter,
                tolerance=base_tolerance,
            )
            if hit is None and mesh.is_quadratic:
                curved_tolerance = self._quadratic_face_projection_tolerance(
                    mesh, allowed, base_tolerance
                )
                if curved_tolerance > base_tolerance:
                    hit = bvh.locate(
                        point,
                        element_ids=allowed,
                        tolerance=curved_tolerance,
                    )
            linear_hit = None
            if hit is None and not mesh.is_quadratic:
                linear_hit = self._linear_face_parameter_hit(
                    mesh, target_face_id, u, v, np.asarray(point, dtype=float), allowed
                )
            if hit is None and linear_hit is None:
                return None, PreflightIssue(
                    "unresolved-attachment",
                    f"attachment {attachment.id} target face point is not meshed",
                    (("attachment", int(attachment.id)),),
                )
            if hit is None:
                plate_nodes, weights, projected = linear_hit
            else:
                plate_nodes, weights = hit.node_ids, hit.weights
                projected = np.asarray(hit.point, dtype=float)

        record = index.coupling_for_beam_node(beam_node)
        if record is not None:
            if tuple(record.plate_nodes) == tuple(plate_nodes):
                return None, None
            return None, PreflightIssue(
                "coupling-conflict",
                f"mesh node {beam_node} already has a different coupling",
                (("attachment", int(attachment.id)),),
            )
        next_record[0] += 1
        made_coupling = Coupling(
            beam_node=beam_node,
            plate_nodes=tuple(int(value) for value in plate_nodes),
            weights=tuple(float(value) for value in weights),
            eccentricity=tuple(float(value) for value in actual - projected),
        )
        mesh.couplings[next_record[0]] = made_coupling
        index.record_coupling(beam_node, made_coupling)
        return (
            ConnectivityAction(
                "attachment-coupling",
                ("member", int(attachment.member_id)),
                attachment.target_key,
                next_record[0],
            ),
            None,
        )

    @staticmethod
    def _shell_node_set(mesh: Mesh) -> set[int]:
        return {
            int(node)
            for table in (mesh.quads, mesh.tris)
            for connectivity in table.values()
            for node in connectivity
        }

    @staticmethod
    def _replace_beam_nodes(
        mesh: Mesh,
        replacements: Mapping[int, int],
        shell_nodes: set[int] | None = None,
    ) -> None:
        """Rewrite beam nodes; ``shell_nodes`` spares a rescan of every shell.

        Callers that already hold the shell-node set (shell connectivity is
        not changed here) pass it so the cost does not grow with the shell
        count for every junction.
        """

        if not replacements:
            return
        for element, nodes in tuple(mesh.beams.items()):
            mesh.beams[element] = tuple(replacements.get(node, node) for node in nodes)
        for edge, nodes in tuple(mesh.nodes_of_edge.items()):
            mesh.nodes_of_edge[edge] = [replacements.get(node, node) for node in nodes]
        for edge, nodes in tuple(mesh.offset_nodes_of_edge.items()):
            mesh.offset_nodes_of_edge[edge] = [replacements.get(node, node) for node in nodes]
        for identifier, coupling in tuple(mesh.couplings.items()):
            if coupling.beam_node in replacements:
                mesh.couplings[identifier] = replace(
                    coupling, beam_node=replacements[coupling.beam_node]
                )
        # Only the replaced nodes can become unreferenced, so test just those
        # instead of collecting every node of the mesh.
        candidates = {int(node) for node in replacements}
        referenced = set()
        if shell_nodes is None:
            shell_nodes = StructuralMeshingPipeline._shell_node_set(mesh)
        referenced.update(candidates & shell_nodes)
        for connectivity in mesh.beams.values():
            referenced.update(candidates.intersection(map(int, connectivity)))
        for coupling in mesh.couplings.values():
            referenced.update(
                candidates.intersection(
                    map(int, (coupling.beam_node, *coupling.plate_nodes))
                )
            )
        for node in replacements:
            if node not in referenced:
                mesh.nodes.pop(node, None)

    def _connect_junction(
        self,
        mesh: Mesh,
        junction: object,
        shell_nodes: Callable[[], set[int]] | None = None,
        index: _ConnectivityIndex | None = None,
    ) -> tuple[list[ConnectivityAction], list[PreflightIssue]]:
        """Connect one declared junction.

        ``shell_nodes`` optionally supplies the shell-node set lazily; it is
        only requested if a station row is actually connected.
        """

        actions: list[ConnectivityAction] = []
        issues: list[PreflightIssue] = []
        if junction.kind is JunctionKind.OVERLAP:
            if self.overlap_policy is not OverlapPolicy.CONNECT_DECLARED:
                return actions, issues
            station_groups = [
                self._member_stations(
                    mesh,
                    use.member_id,
                    use.member_range.start,
                    use.member_range.end,
                )
                for use in junction.member_uses
            ]
            if not station_groups or len({len(item) for item in station_groups}) != 1:
                issues.append(
                    PreflightIssue(
                        "unaligned-overlap",
                        f"overlap junction {junction.id} has unaligned member stations",
                        (("junction", int(junction.id)),),
                    )
                )
                return actions, issues
            rows = zip(*station_groups)
        else:
            point_nodes: list[tuple[tuple[float, int], ...]] = []
            for use in junction.member_uses:
                if not use.member_range.is_point:
                    continue
                node = self._member_node(mesh, use.member_id, use.member_range.start)
                if node is None:
                    issues.append(
                        PreflightIssue(
                            "missing-junction-station",
                            f"junction {junction.id} member {use.member_id} station is not meshed",
                            (("junction", int(junction.id)), ("member", int(use.member_id))),
                        )
                    )
                else:
                    point_nodes.append(((use.member_range.start, node),))
            rows = zip(*point_nodes) if point_nodes else ()

        shell_set: set[int] | None = None
        for row in rows:
            nodes = [int(item[1]) for item in row]
            if len(set(nodes)) < 2:
                continue
            coordinates = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
            extent = float(np.max(np.ptp(coordinates, axis=0)))
            tolerance = self.view.effective_length(extent)
            if any(
                float(np.linalg.norm(coordinates[index] - coordinates[0])) > tolerance
                for index in range(1, len(coordinates))
            ):
                issues.append(
                    PreflightIssue(
                        "junction-residual",
                        f"junction {junction.id} mesh stations do not coincide",
                        (("junction", int(junction.id)),),
                    )
                )
                continue
            if shell_set is None:
                shell_set = (
                    self._shell_node_set(mesh) if shell_nodes is None else shell_nodes()
                )
            master = min(nodes, key=lambda node: (node not in shell_set, node))
            replacements = {node: master for node in nodes if node != master}
            self._replace_beam_nodes(mesh, replacements, shell_set)
            # Merged stations rewrote beam node identifiers and may have
            # removed nodes; cached edge coordinates must not survive it.
            self._edge_coordinate_cache = {}
            if index is not None:
                # Beam node identifiers were rewritten; the beam-node
                # coupling index must not survive this mutation.
                index.invalidate_beam_couplings()
            for old in sorted(replacements):
                actions.append(
                    ConnectivityAction(
                        "junction-shared-node",
                        ("node", old),
                        ("node", master),
                        int(junction.id),
                    )
                )
        return actions, issues

    def apply_connectivity(
        self,
        mesh: Mesh,
        *,
        components: Iterable[ComponentKey] | None = None,
    ) -> ConnectivityReport:
        """Apply only declared attachment and junction connectivity to ``mesh``."""

        states = self.preflight(mesh)
        selected = (
            {item.key for item in self.components}
            if components is None
            else set(components)
        )
        ready = {
            state.component.key
            for state in states
            if state.ready and state.component.key in selected
        }
        issues = [
            issue
            for state in states
            if state.component.key in selected
            for issue in state.issues
        ]
        tolerance = self.view.effective_length(
            0.0
            if not mesh.nodes
            else float(np.max(np.ptp(np.asarray(list(mesh.nodes.values())), axis=0)))
        )
        bvh = MeshElementBVH._connectivity_only(mesh, tolerance=tolerance)
        previous_edge_cache = self._edge_coordinate_cache
        self._edge_coordinate_cache = {}
        try:
            index = _ConnectivityIndex(mesh, self)
            next_record = [max((0, *mesh.quads, *mesh.tris, *mesh.beams, *mesh.couplings))]
            actions: list[ConnectivityAction] = []

            attachment_ids = {
                identifier
                for component in self.components
                if component.key in ready
                for identifier in component.attachment_ids
            }
            for identifier in sorted(attachment_ids):
                attachment = self.view.attachments[identifier]
                if attachment.member_id is None:
                    continue
                if attachment.member_range.is_point:
                    node = self._member_node(
                        mesh, attachment.member_id, attachment.member_range.start
                    )
                    stations = () if node is None else ((attachment.member_range.start, node),)
                else:
                    stations = self._member_stations(
                        mesh,
                        attachment.member_id,
                        attachment.member_range.start,
                        attachment.member_range.end,
                    )
                if not stations:
                    issues.append(
                        PreflightIssue(
                            "missing-attachment-station",
                            f"attachment {identifier} has no member mesh station",
                            (("attachment", identifier),),
                        )
                    )
                    continue
                for member_parameter, beam_node in stations:
                    action, issue = self._add_attachment_coupling(
                        mesh,
                        bvh,
                        attachment,
                        member_parameter,
                        beam_node,
                        next_record,
                        index,
                    )
                    if action is not None:
                        actions.append(action)
                    if issue is not None:
                        issues.append(issue)

            junction_ids = {
                identifier
                for component in self.components
                if component.key in ready
                for identifier in component.junction_ids
            }
            shell_cache: list[set[int]] = []

            def shell_nodes() -> set[int]:
                # Shell connectivity is read-only during connectivity application.
                if not shell_cache:
                    shell_cache.append(self._shell_node_set(mesh))
                return shell_cache[0]

            for identifier in sorted(junction_ids):
                made_actions, made_issues = self._connect_junction(
                    mesh, self.view.junctions[identifier], shell_nodes, index
                )
                actions.extend(made_actions)
                issues.extend(made_issues)
            return ConnectivityReport(tuple(actions), tuple(issues), states)
        finally:
            self._edge_coordinate_cache = previous_edge_cache

    connect = apply_connectivity

