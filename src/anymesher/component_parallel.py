"""Mesh geometrically independent components in separate processes, then join.

Status: research prototype (see ``docs/PARALLEL_MESHING_STUDY.md``).

The planner here is a stand-in for an ANYgeometry independence planner.  Two
selections are *independent* when they share no vertex, no declared structural
relation (sheet, attachment, junction) and their padded conservative bounding
boxes do not overlap.  Independent components share no mesh node, so a join is
an id offset plus a handle remap, with no interface reconciliation.

Anything the planner cannot prove falls back to the serial pipeline, and the
reason is recorded on the result (``mesh.hybrid_diagnostics["parallel"]``).
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from anygeometry import GeometryModel, from_dict, to_dict

from .boundary import GlobalEdgeBoundaryRegistry
from .errors import MeshError
from .hybrid import (
    CertificationMode,
    HybridMeshResult,
    generate_hybrid_mesh_result,
)
from .mesh import Coupling, Mesh
from .meshing_view import GeometryMeshingView
from .structural_pipeline import (
    ConnectivityAction,
    ConnectivityReport,
)

__all__ = [
    "ComponentPlan",
    "ParallelOptions",
    "ParallelPlan",
    "generate_hybrid_mesh_result_parallel",
    "merge_component_meshes",
    "plan_independent_components",
]


@dataclass(frozen=True)
class ParallelOptions:
    """Controls for component-parallel meshing.

    ``workers`` of ``None`` uses ``min(components, cpu_count)``.  ``executor``
    lets a caller reuse a warm spawn pool (worker start-up is 0.5-0.9 s).
    ``pad_factor`` scales the separation, in multiples of ``target_size``, that
    two components must keep to be treated as independent.
    """

    workers: int | None = None
    min_components: int = 2
    pad_factor: float = 1.0
    executor: Executor | None = None


@dataclass(frozen=True)
class ComponentPlan:
    faces: tuple[int, ...]
    members: tuple[int, ...]
    beam_edges: tuple[int, ...]
    sheets: tuple[int, ...]
    attachments: tuple[int, ...]
    junctions: tuple[int, ...]
    bounds: tuple[float, ...] | None

    @property
    def handles(self) -> tuple[tuple[str, int], ...]:
        return (
            *(("face", i) for i in self.faces),
            *(("edge", i) for i in self.beam_edges),
            *(("sheet", i) for i in self.sheets),
            *(("member", i) for i in self.members),
            *(("attachment", i) for i in self.attachments),
            *(("junction", i) for i in self.junctions),
        )


@dataclass(frozen=True)
class ParallelPlan:
    components: tuple[ComponentPlan, ...]
    pad: float
    fallback_reason: str | None = None

    @property
    def parallel(self) -> bool:
        return self.fallback_reason is None and len(self.components) >= 2


class _Union:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _sweep_overlaps(
    boxes: Sequence[tuple[float, ...] | None], pad: float
) -> Iterable[tuple[int, int]]:
    """Pairs of boxes whose ``pad``-grown extents overlap (sort and sweep on x)."""

    order = sorted(
        (i for i, box in enumerate(boxes) if box is not None),
        key=lambda i: boxes[i][0],
    )
    active: list[int] = []
    for i in order:
        box = boxes[i]
        active = [j for j in active if boxes[j][3] + pad >= box[0] - 0.0]
        for j in active:
            other = boxes[j]
            if (
                other[1] - pad <= box[4]
                and box[1] - pad <= other[4]
                and other[2] - pad <= box[5]
                and box[2] - pad <= other[5]
            ):
                yield j, i
        active.append(i)


def plan_independent_components(
    geometry: GeometryModel,
    *,
    pad: float,
    face_ids: Iterable[int] | None = None,
    member_ids: Iterable[int] | None = None,
    beam_edges: Iterable[int] = (),
) -> ParallelPlan:
    """Partition the meshed selection into provably independent components."""

    if face_ids is not None or member_ids is not None:
        return ParallelPlan((), pad, "partial face/member selection")
    view = GeometryMeshingView(geometry)
    faces = tuple(sorted(int(i) for i in geometry.faces))
    members = tuple(sorted(int(i) for i in view.members))
    explicit_edges = {int(i) for i in beam_edges}

    units: list[tuple[str, int]] = [("face", f) for f in faces]
    units += [("member", m) for m in members]
    member_edge_ids = {
        m: tuple(int(use.edge_id) for use in view.edge_uses_for_member(m))
        for m in members
    }
    owned = {e for edges in member_edge_ids.values() for e in edges}
    units += [("edge", e) for e in sorted(explicit_edges - owned)]
    index = {unit: i for i, unit in enumerate(units)}
    union = _Union(len(units))

    unit_edges: list[tuple[int, ...]] = []
    for kind, ident in units:
        if kind == "face":
            face = geometry.faces[ident]
            unit_edges.append(
                tuple(int(u.edge) for loop in (face.loop, *face.holes) for u in loop)
            )
        elif kind == "member":
            unit_edges.append(member_edge_ids[ident])
        else:
            unit_edges.append((ident,))
    edge_units: dict[int, list[int]] = {}
    vertex_unit: dict[int, int] = {}
    for i, edges in enumerate(unit_edges):
        for edge_id in edges:
            edge_units.setdefault(edge_id, []).append(i)
            edge = geometry.edges[edge_id]
            for vertex in (edge.start, edge.end):
                first = vertex_unit.setdefault(int(vertex), i)
                union.union(first, i)
    for owners in edge_units.values():
        for other in owners[1:]:
            union.union(owners[0], other)

    for sheet_id in view.sheets:
        sheet_faces = [index[("face", f)] for f in view.faces_for_sheet(sheet_id)
                       if ("face", f) in index]
        for other in sheet_faces[1:]:
            union.union(sheet_faces[0], other)

    sheet_units = {
        sheet_id: [index[("face", f)] for f in view.faces_for_sheet(sheet_id)
                   if ("face", f) in index]
        for sheet_id in view.sheets
    }
    for attachment in view.attachments.values():
        if attachment.member_id is None:
            continue
        member_unit = index.get(("member", int(attachment.member_id)))
        if member_unit is None:
            continue
        # Declared relation: unite regardless of distance.
        if attachment.target_kind.name == "FACE":
            target = [index.get(("face", int(attachment.target_id)))]
        else:
            target = list(edge_units.get(int(attachment.target_id), ()))
        for other in target:
            if other is not None:
                union.union(member_unit, other)
    for junction in view.junctions.values():
        participants = [
            index[("member", int(use.member_id))]
            for use in junction.member_uses
            if ("member", int(use.member_id)) in index
        ]
        for sheet_id in junction.sheet_ids:
            participants.extend(sheet_units.get(int(sheet_id), ()))
        for other in participants[1:]:
            union.union(participants[0], other)

    boxes: list[tuple[float, ...] | None] = []
    for i, (kind, ident) in enumerate(units):
        entity_keys = (
            [("face", ident)]
            if kind == "face"
            else [("edge", e) for e in unit_edges[i]]
        )
        boxes.append(geometry.bounds(entity_keys))
        if boxes[-1] is None:
            return ParallelPlan((), pad, f"no bounds for {kind} {ident}")
    for a, b in _sweep_overlaps(boxes, pad):
        union.union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(len(units)):
        groups.setdefault(union.find(i), []).append(i)

    components: list[ComponentPlan] = []
    for root in sorted(groups):
        part = groups[root]
        faces_g = tuple(sorted(units[i][1] for i in part if units[i][0] == "face"))
        members_g = tuple(sorted(units[i][1] for i in part if units[i][0] == "member"))
        edges_g = tuple(sorted(units[i][1] for i in part if units[i][0] == "edge"))
        face_set, member_set = set(faces_g), set(members_g)
        sheets_g = tuple(
            sorted(
                s for s in view.sheets
                if face_set.intersection(view.faces_for_sheet(s))
            )
        )
        attachments_g = tuple(
            sorted(
                a.id for a in view.attachments.values()
                if a.member_id is not None and int(a.member_id) in member_set
            )
        )
        junctions_g = tuple(
            sorted(
                j.id for j in view.junctions.values()
                if any(int(u.member_id) in member_set for u in j.member_uses)
                or face_set.intersection(
                    f for s in j.sheet_ids for f in view.faces_for_sheet(int(s))
                )
            )
        )
        merged = [boxes[i] for i in part]
        arr = np.asarray(merged, dtype=float)
        bounds = (*arr[:, :3].min(axis=0), *arr[:, 3:].max(axis=0))
        components.append(
            ComponentPlan(faces_g, members_g, edges_g, sheets_g, attachments_g,
                          junctions_g, bounds)
        )
    return ParallelPlan(tuple(components), pad)


# --------------------------------------------------------------------------
# worker


@dataclass
class _ComponentOutput:
    mesh: Mesh
    registry_entries: list[tuple[int, float, Any, int | None]]
    strategy_by_face: dict[int, str]
    backend_by_face: dict[int, Any]
    preflight: tuple[Any, ...]
    connectivity: Any | None
    seconds: float


def _mesh_component(payload: Mapping[str, Any]) -> _ComponentOutput:
    started = time.perf_counter()
    model = from_dict(payload["model"])
    result = generate_hybrid_mesh_result(model, **payload["options"])
    mesh = result.mesh
    registry = getattr(mesh, "boundary_registry", None)
    entries: list[tuple[int, float, Any, int | None]] = []
    if registry is not None:
        entries = [
            (int(e.key.edge_id), float(e.key.parameter), np.asarray(e.point), e.node_id)
            for e in registry.entries()
        ]
    mesh.boundary_registry = None  # bound to a live model; rebuilt on join
    return _ComponentOutput(
        mesh,
        entries,
        dict(result.strategy_by_face),
        {k: dict(v) for k, v in result.triangulation_backend_by_face.items()},
        tuple(result.preflight),
        result.connectivity,
        time.perf_counter() - started,
    )


# --------------------------------------------------------------------------
# join


@dataclass(frozen=True)
class _IdMaps:
    """Work-model entity ids -> source-model ids, per kind, for one component."""

    by_kind: Mapping[str, Mapping[int, int]]
    node_offset: int
    element_offset: int

    def __call__(self, kind: str, ident: int) -> int:
        return int(self.by_kind[kind][int(ident)])


def _id_maps(closure: Any) -> dict[str, dict[int, int]]:
    made: dict[str, dict[int, int]] = {}
    for work, source in closure.work_to_source.items():
        made.setdefault(work.kind, {})[int(work.id)] = int(source.id)
    return made


def _max_element_id(mesh: Mesh) -> int:
    values = [0]
    for table in (mesh.quads, mesh.tris, mesh.beams, mesh.couplings, mesh.activity):
        values.extend(int(k) for k in table)
    return max(values)


def _shift_grid(grid: np.ndarray, offset: int) -> np.ndarray:
    array = np.asarray(grid)
    return np.where(array > 0, array + offset, array)


def merge_component_meshes(
    source: GeometryModel,
    outputs: Sequence[_ComponentOutput],
    id_maps: Sequence[Mapping[str, Mapping[int, int]]],
) -> tuple[Mesh, list[_IdMaps]]:
    """Join disjoint component meshes into one mesh keyed by source ids."""

    merged = Mesh(
        geometry_model_id=source.model_id,
        geometry_revision=source.revision,
        order=outputs[0].mesh.order,
    )
    node_offset = 0
    element_offset = 0
    maps: list[_IdMaps] = []
    divisions: dict[int, int] = {}
    classes: dict[int, int] = {}
    seeding_field = None
    for output, by_kind in zip(outputs, id_maps):
        part = output.mesh
        if part.order != merged.order:
            raise MeshError("component meshes disagree on element order")
        m = _IdMaps(by_kind, node_offset, element_offset)
        maps.append(m)
        nid = lambda n, o=node_offset: int(n) + o  # noqa: E731
        eid = lambda e, o=element_offset: int(e) + o  # noqa: E731
        for key, value in part.nodes.items():
            merged.nodes[nid(key)] = value
        for key, nodes in part.quads.items():
            merged.quads[eid(key)] = tuple(nid(n) for n in nodes)
        for key, nodes in part.tris.items():
            merged.tris[eid(key)] = tuple(nid(n) for n in nodes)
        for key, nodes in part.beams.items():
            merged.beams[eid(key)] = tuple(nid(n) for n in nodes)
        for key, c in part.couplings.items():
            merged.couplings[eid(key)] = Coupling(
                nid(c.beam_node),
                tuple(nid(n) for n in c.plate_nodes),
                c.weights,
                c.eccentricity,
            )
        for key, value in part.activity.items():
            merged.activity[eid(key)] = value
        for key, node in part.node_of_vertex.items():
            merged.node_of_vertex[m("vertex", key)] = nid(node)
        for name in ("nodes_of_edge", "offset_nodes_of_edge"):
            target = getattr(merged, name)
            for key, nodes in getattr(part, name).items():
                target[m("edge", key)] = [nid(n) for n in nodes]
        for key, grid in part.grid_of_face.items():
            merged.grid_of_face[m("face", key)] = _shift_grid(grid, node_offset)
        for key, grids in part.block_grids_of_face.items():
            merged.block_grids_of_face[m("face", key)] = tuple(
                _shift_grid(g, node_offset) for g in grids
            )
        for name, kind in (
            ("elements_of_face", "face"),
            ("elements_of_edge", "edge"),
            ("elements_of_sheet", "sheet"),
            ("elements_of_member", "member"),
        ):
            target = getattr(merged, name)
            for key, elements in getattr(part, name).items():
                target[m(kind, key)] = [eid(e) for e in elements]
        for key, nodes in part.nodes_of_member.items():
            merged.nodes_of_member[m("member", key)] = [nid(n) for n in nodes]
        for key, value in part.thickness_of_face.items():
            merged.thickness_of_face[m("face", key)] = value
        merged.declared_plate_junction_edges += tuple(
            (nid(a), nid(b)) for a, b in part.declared_plate_junction_edges
        )
        merged.automatic_intersections += part.automatic_intersections
        merged.automatic_beam_connections += part.automatic_beam_connections
        merged.automatic_shell_connections += part.automatic_shell_connections
        if part.seeding is not None:
            for key, value in part.seeding.divisions.items():
                divisions[m("edge", key)] = value
            for key, value in part.seeding.classes.items():
                classes[m("edge", key)] = value
            seeding_field = seeding_field or part.seeding.size_field
        node_offset += max((int(k) for k in part.nodes), default=0)
        element_offset += _max_element_id(part)
    if divisions or classes:
        from .seeding import Seeding

        merged.seeding = Seeding(divisions, 0, classes, seeding_field)
    merged.declared_plate_junction_edges = tuple(
        sorted(set(merged.declared_plate_junction_edges))
    )
    return merged, maps


def _offset_connectivity(report: Any, m: _IdMaps) -> tuple[Any, ...] | None:
    """Offset mesh-space ids only.

    Preflight and connectivity records name entities of the *prepared* working
    geometry (e.g. sheets and junctions created by structural preparation),
    which the serial route also does not map back to the source model.  They
    are therefore kept per component and unchanged; only node ids and coupling
    record ids, which live in mesh id space, are offset.
    """

    if report is None:
        return None

    def key(pair: tuple[str, int]) -> tuple[str, int]:
        kind, ident = pair
        return (kind, int(ident) + m.node_offset) if kind == "node" else (kind, ident)

    actions = tuple(
        ConnectivityAction(
            a.kind, key(a.source), key(a.target),
            int(a.record_id) + m.element_offset
            if a.record_id is not None and a.kind == "attachment-coupling"
            else a.record_id,
        )
        for a in report.actions
    )
    return actions, tuple(report.issues), tuple(report.states)


# --------------------------------------------------------------------------
# driver

_UNSUPPORTED = (
    ("seeding", None),
    ("quad_options", None),
    ("quad_face_ids", None),
    ("change_set", None),
    ("audit_policy", None),
    ("member_ids", None),
    ("face_ids", None),
)


def _serial(geometry: GeometryModel, options: dict[str, Any], reason: str) -> HybridMeshResult:
    result = generate_hybrid_mesh_result(geometry, **options)
    result.mesh.hybrid_diagnostics["parallel"] = {"used": False, "reason": reason}
    return result


def generate_hybrid_mesh_result_parallel(
    geometry: GeometryModel,
    *,
    target_size: float,
    parallel: ParallelOptions | None = None,
    **options: Any,
) -> HybridMeshResult:
    """``generate_hybrid_mesh_result`` with independent components in parallel.

    Result ids are source ids, but node and element ids differ from the serial
    run (components are numbered consecutively); geometry-keyed associations are
    equal up to that renumbering.  The audit report and structural-preparation
    report are not produced on the parallel route (``None``).  ``preflight`` and
    ``connectivity`` are concatenated per-component records in prepared-model
    ids (as in the serial route), with mesh-space ids offset.
    """

    parallel = parallel or ParallelOptions()
    options = dict(options, target_size=target_size)
    cancellation = options.get("cancellation_check")
    for name, empty in _UNSUPPORTED:
        if options.get(name, empty) is not empty:
            return _serial(geometry, options, f"unsupported option {name}")
    if str(getattr(options.get("certification_mode"), "value",
                   options.get("certification_mode", "none"))) != "none":
        return _serial(geometry, options, "certification requested")
    if str(getattr(options.get("mutation_policy"), "value",
                   options.get("mutation_policy", "read_only"))) != "read_only":
        return _serial(geometry, options, "mutation policy is not read_only")

    started = time.perf_counter()
    pad = parallel.pad_factor * float(target_size)
    offsets = options.get("beam_offsets") or {}
    if offsets:
        pad += max(
            float(np.max(np.abs(np.asarray(v, dtype=float))))
            for v in offsets.values()
        )
    plan = plan_independent_components(
        geometry, pad=pad, beam_edges=options.get("beam_edges", ())
    )
    plan_seconds = time.perf_counter() - started
    if plan.fallback_reason or len(plan.components) < max(2, parallel.min_components):
        return _serial(
            geometry, options,
            plan.fallback_reason or f"{len(plan.components)} component(s)",
        )

    closures = [geometry.extract_model_closure(c.handles) for c in plan.components]
    maps = [_id_maps(c) for c in closures]
    payloads = []
    for component, closure, by_kind in zip(plan.components, closures, maps):
        inverse = {kind: {s: w for w, s in table.items()} for kind, table in by_kind.items()}
        local = dict(options)
        local["face_ids"] = tuple(sorted(inverse["face"][f] for f in component.faces))
        local["member_ids"] = tuple(sorted(inverse["member"][m] for m in component.members)
                                    ) if component.members else ()
        local["beam_edges"] = tuple(inverse["edge"][e] for e in component.beam_edges)
        for name in ("overrides", "beam_offsets"):
            table = options.get(name)
            if table:
                local[name] = {
                    inverse["edge"][int(e)]: v for e, v in table.items()
                    if int(e) in inverse["edge"]
                }
        local.pop("cancellation_check", None)
        payloads.append({"model": to_dict(closure.working_model), "options": local})
    covered = {name: sum(len(p["options"].get(name) or ()) for p in payloads)
               for name in ("overrides", "beam_offsets")}
    for name, total in covered.items():
        if options.get(name) and total != len(options[name]):
            return _serial(geometry, options, f"{name} not owned by a component")
    prepare_seconds = time.perf_counter() - started - plan_seconds

    workers = parallel.workers or min(len(payloads), multiprocessing.cpu_count())
    owned = parallel.executor is None
    pool = parallel.executor or ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    )
    outputs: list[_ComponentOutput | None] = [None] * len(payloads)
    run_started = time.perf_counter()
    try:
        futures = {pool.submit(_mesh_component, p): i for i, p in enumerate(payloads)}
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
            if cancellation is not None:
                cancellation("parallel component complete")
    finally:
        if owned:
            pool.shutdown(wait=True, cancel_futures=True)
    run_seconds = time.perf_counter() - run_started

    join_started = time.perf_counter()
    mesh, id_maps = merge_component_meshes(geometry, outputs, maps)
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(geometry))
    for output, m in zip(outputs, id_maps):
        for edge_id, parameter, point, node_id in output.registry_entries:
            registry.register(
                m("edge", edge_id), parameter, point,
                node_id=None if node_id is None else int(node_id) + m.node_offset,
                owner=geometry.handle("edge", m("edge", edge_id)),
            )
    mesh.boundary_registry = registry

    strategy: dict[int, str] = {}
    backend: dict[int, Any] = {}
    preflight: list[Any] = []
    actions: list[Any] = []
    issues: list[Any] = []
    states: list[Any] = []
    for output, m in zip(outputs, id_maps):
        strategy.update({m("face", f): s for f, s in output.strategy_by_face.items()})
        backend.update({m("face", f): v for f, v in output.backend_by_face.items()})
        preflight.extend(output.preflight)
        remapped = _offset_connectivity(output.connectivity, m)
        if remapped is not None:
            actions.extend(remapped[0])
            issues.extend(remapped[1])
            states.extend(remapped[2])
    phase_totals: dict[str, float] = {}
    for output in outputs:
        for name, value in output.mesh.hybrid_diagnostics.get("phase_seconds", {}).items():
            phase_totals[name] = phase_totals.get(name, 0.0) + float(value)
    mesh.hybrid_diagnostics = {
        "phase_seconds": phase_totals,
        "parallel": {
            "used": True,
            "components": len(outputs),
            "workers": workers,
            "pad": pad,
            "plan_seconds": plan_seconds,
            "prepare_seconds": prepare_seconds,
            "mesh_seconds": run_seconds,
            "component_seconds": [o.seconds for o in outputs],
            "join_seconds": time.perf_counter() - join_started,
        },
    }
    connectivity = (
        ConnectivityReport(tuple(actions), tuple(issues), tuple(states))
        if any(o.connectivity is not None for o in outputs) else None
    )
    return HybridMeshResult(
        mesh=mesh,
        strategy_by_face=strategy,
        triangulation_backend_by_face=backend,
        preflight=tuple(preflight),
        connectivity=connectivity,
        audit_report=None,
        certification_mode=CertificationMode.NONE,
        certifiable=False,
    )
