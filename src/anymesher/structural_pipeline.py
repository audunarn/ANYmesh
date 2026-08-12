"""Component-local structural meshing orchestration.

This module deliberately does not infer welds from proximity.  Components and
mesh connections are formed only by persistent Sheet/Member ownership plus
Attachment/Junction intent from ANYgeometry.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
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
from .mesh_bvh import MeshElementBVH
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
    """Permission for topology preparation; source geometry is never mutated."""

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


class StructuralMeshingPipeline:
    """Declared-only connectivity and local invalidation for one view."""

    def __init__(
        self,
        view: GeometryMeshingView,
        *,
        overlap_policy: OverlapPolicy | str,
        mutation_policy: GeometryMutationPolicy | str,
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
        self.components = build_structural_components(view)
        self.component_map: Mapping[ComponentKey, StructuralComponent] = MappingProxyType(
            {item.key: item for item in self.components}
        )
        self.boundaries = GlobalEdgeBoundaryRegistry(view)
        self.members = MemberRegistry(view)
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
            target_store = (
                self.view.faces
                if attachment.target_kind is AttachmentTargetKind.FACE
                else self.view.edges
            )
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

    def _edge_node(
        self, mesh: Mesh, edge_id: int, parameter: float, owner: Hashable
    ) -> int | None:
        expected = self.view.edge_point(edge_id, parameter)
        candidates = tuple(mesh.nodes_of_edge.get(edge_id, ()))
        if not candidates:
            return None
        distances = [float(np.linalg.norm(mesh.nodes[node] - expected)) for node in candidates]
        index = int(np.argmin(distances))
        tolerance = self.view.effective_length(self.view.edge_length(edge_id))
        if distances[index] > tolerance:
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

    def _add_attachment_coupling(
        self,
        mesh: Mesh,
        bvh: MeshElementBVH,
        attachment: object,
        member_parameter: float,
        beam_node: int,
        next_record: list[int],
    ) -> tuple[ConnectivityAction | None, PreflightIssue | None]:
        actual = np.asarray(mesh.nodes[beam_node], dtype=float)
        if attachment.target_kind is AttachmentTargetKind.EDGE:
            target_parameter = self._mapped_parameter(
                member_parameter, attachment.member_range, attachment.target_parameters[0]
            )
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
                allowed = self._target_elements_for_edge(mesh, attachment.target_id)
                hit = bvh.locate(
                    point,
                    element_ids=allowed,
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
            u = self._mapped_parameter(
                member_parameter, attachment.member_range, attachment.target_parameters[0]
            )
            v = self._mapped_parameter(
                member_parameter, attachment.member_range, attachment.target_parameters[1]
            )
            point = self.view.face_point(attachment.target_id, u, v)
            allowed = tuple(mesh.elements_of_face.get(attachment.target_id, ()))
            face_points = np.asarray(
                [mesh.nodes[node] for element in allowed for node in mesh.corners_of(element)],
                dtype=float,
            )
            extent = 0.0 if not len(face_points) else float(np.max(np.ptp(face_points, axis=0)))
            hit = bvh.locate(
                point,
                element_ids=allowed,
                tolerance=self.view.effective_length(extent),
            )
            if hit is None:
                return None, PreflightIssue(
                    "unresolved-attachment",
                    f"attachment {attachment.id} target face point is not meshed",
                    (("attachment", int(attachment.id)),),
                )
            plate_nodes, weights = hit.node_ids, hit.weights
            projected = np.asarray(hit.point, dtype=float)

        for record in mesh.couplings.values():
            if int(record.beam_node) == beam_node:
                if tuple(record.plate_nodes) == tuple(plate_nodes):
                    return None, None
                return None, PreflightIssue(
                    "coupling-conflict",
                    f"mesh node {beam_node} already has a different coupling",
                    (("attachment", int(attachment.id)),),
                )
        next_record[0] += 1
        mesh.couplings[next_record[0]] = Coupling(
            beam_node=beam_node,
            plate_nodes=tuple(int(value) for value in plate_nodes),
            weights=tuple(float(value) for value in weights),
            eccentricity=tuple(float(value) for value in actual - projected),
        )
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
    def _replace_beam_nodes(mesh: Mesh, replacements: Mapping[int, int]) -> None:
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
        referenced = {
            int(node)
            for connectivity in (*mesh.shells.values(), *mesh.beams.values())
            for node in connectivity
        }
        referenced.update(
            int(node)
            for coupling in mesh.couplings.values()
            for node in (coupling.beam_node, *coupling.plate_nodes)
        )
        for node in replacements:
            if node not in referenced:
                mesh.nodes.pop(node, None)

    def _connect_junction(
        self, mesh: Mesh, junction: object
    ) -> tuple[list[ConnectivityAction], list[PreflightIssue]]:
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

        shell_nodes = {
            int(node) for connectivity in mesh.shells.values() for node in connectivity
        }
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
            master = min(nodes, key=lambda node: (node not in shell_nodes, node))
            replacements = {node: master for node in nodes if node != master}
            self._replace_beam_nodes(mesh, replacements)
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
        bvh = MeshElementBVH(mesh, tolerance=tolerance)
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
        for identifier in sorted(junction_ids):
            made_actions, made_issues = self._connect_junction(
                mesh, self.view.junctions[identifier]
            )
            actions.extend(made_actions)
            issues.extend(made_issues)
        return ConnectivityReport(tuple(actions), tuple(issues), states)

    connect = apply_connectivity

