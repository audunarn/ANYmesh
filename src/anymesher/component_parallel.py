"""Mesh geometrically independent components in separate processes, then join.

Status: research prototype (see ``docs/PARALLEL_MESHING_STUDY.md``).

Independence is decided by ANYgeometry's ``plan_independent_components``
(conservative, read-only, dependency-complete; see ANYgeometry
``docs/PARALLEL_COMPONENT_HANDOFF.md``), not here.  A certified partition's
components share no vertex, declared relation or grown support box, so they
share no mesh node; the join is an id offset plus a handle remap with no
interface reconciliation.  Components travel to workers as
``ModelClosure.to_transport()`` messages.

Anything not certified (older ANYgeometry without the planner or the binding
validator, a refused partition, one component, unsupported options) runs on
the serial pipeline and the reason is recorded on the result
(``mesh.hybrid_diagnostics["parallel"]``).

Pool lifecycle (see ``docs/PARALLEL_POOL_LIFECYCLE.md``): without a
``pool_lease`` the library starts a cold, library-owned spawn pool per job
and disposes it within a bounded deadline; ``create_parallel_pool()`` lends
a caller-owned warm pool through ``ParallelOptions.pool_lease``, borrowed
exclusively per job, reusable after success, invalidated by cancel/failure.
A bare, shared or unknown ``executor`` is never submitted to (its children
could not be identified, contained or safely terminated); it falls back to
the serial route before submission.
"""

from __future__ import annotations

import math
import multiprocessing
import time
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

import anygeometry
from anygeometry import GeometryModel

from .boundary import GlobalEdgeBoundaryRegistry
from anygeometry.errors import GeometryError

from ._parallel_pool import (
    POLL_SECONDS,
    TERMINATION_DEADLINE_SECONDS,
    ParallelContainmentUnavailable,
    ParallelJobCancelled,
    ParallelPoolLease,
    _ParallelPool,
    _priority_name,
    create_parallel_pool,
)
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
    "ParallelOptions",
    "ParallelPoolLease",
    "create_parallel_pool",
    "generate_hybrid_mesh_parallel",
    "generate_hybrid_mesh_result_parallel",
    "merge_component_meshes",
]


@dataclass(frozen=True)
class ParallelOptions:
    """Controls for component-parallel meshing.

    ``workers`` of ``None`` uses ``min(components, cpu_count)``.
    ``pool_lease`` borrows a warm pool from :func:`create_parallel_pool`
    exclusively for the job; without one the library starts a cold,
    library-owned pool.  ``executor`` is retained for source compatibility
    but a bare, shared or unknown executor is never submitted to: the route
    falls back to the serial pipeline before submission, and passing both an
    executor and a lease is rejected.  ``pad_factor`` is the gap, in
    multiples of ``target_size``, below which two components are merged:
    ANYgeometry grows each support box by half of it (plus any
    caller-declared beam offset reach), so a gap up to
    ``pad_factor * target_size`` merges.  ``min_estimated_elements`` is an
    explicit opt-in threshold: when set, a job whose cheap per-component
    element estimate totals less than it runs serially instead (no
    automatic threshold is ever applied).
    """

    workers: int | None = None
    min_components: int = 2
    pad_factor: float = 1.0
    executor: Executor | None = None
    pool_lease: ParallelPoolLease | None = None
    min_estimated_elements: int | None = None


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


def _mesh_component(
    payload: Mapping[str, Any],
    *,
    cancellation_check: Callable[[str], None] | None = None,
) -> _ComponentOutput:
    started = time.perf_counter()
    from anygeometry.closure import ModelClosure

    from . import hybrid

    if cancellation_check is not None:
        # The pool injects the shared per-job cooperative signal here so the
        # engine's own cancellation checkpoints honour it.
        payload["options"]["cancellation_check"] = cancellation_check
    model = ModelClosure.from_transport(payload["closure"]).working_model
    # Planar faces mesh with a chart metric only in multi-face models; a
    # one-face component must behave as it does inside the whole model.
    hybrid._WHOLE_MODEL_HAS_SEVERAL_FACES = bool(payload["whole_has_several_faces"])
    try:
        result = generate_hybrid_mesh_result(model, **payload["options"])
    finally:
        hybrid._WHOLE_MODEL_HAS_SEVERAL_FACES = False
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
    ("quad_options", None),
    ("quad_face_ids", None),
    ("change_set", None),
    ("audit_policy", None),
)

_EXECUTOR_FALLBACK_REASON = (
    "bare or shared executor unsupported; "
    "pass pool_lease from create_parallel_pool"
)


def _serial(geometry: GeometryModel, options: dict[str, Any], reason: str) -> HybridMeshResult:
    result = generate_hybrid_mesh_result(geometry, **options)
    result.mesh.hybrid_diagnostics["parallel"] = {
        "used": False,
        "route": "serial",
        "reason": reason,
    }
    return result


def _forwarded_check(
    cancellation: Callable[[str], Any] | None,
) -> Callable[[str], None]:
    """Owner cancellation as an exception, never a truthy return value."""

    def check(phase: str) -> None:
        if cancellation is not None and cancellation(phase):
            raise ParallelJobCancelled(f"cancelled during {phase}")

    return check


def _normalize_selections(options: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize face/member/beam selections once, preserving None/empty."""

    options = dict(options)
    for name in ("face_ids", "member_ids"):
        value = options.get(name)
        options[name] = (
            None if value is None else tuple(sorted(int(i) for i in value))
        )
    edges = options.get("beam_edges")
    options["beam_edges"] = (
        () if edges is None else tuple(sorted(int(e) for e in edges))
    )
    return options


def _validate_binding(
    validator: Callable[..., Any],
    geometry: GeometryModel,
    partition: Any,
    cancellation: Callable[[str], Any] | None,
) -> None:
    """Fresh owner binding validation; stale/wrong-owner failures stay typed."""

    validator(
        geometry,
        partition,
        expected_revision=geometry.revision,
        cancellation_check=_forwarded_check(cancellation),
    )


def _component_estimates(
    geometry: GeometryModel,
    partition: Any,
    options: dict[str, Any],
    target_size: float,
) -> tuple[list[int | None], list[str]]:
    """Cheap per-component element estimates; ``None`` where unknown.

    Returns the estimates and, in parallel, their basis per component:
    ``"seeding"`` (existing seeding divisions), ``"heuristic"`` (OWNER
    geometry measurements scaled by ``target_size`` — a mapped side-product
    heuristic, never an exact physical area) or ``"unknown"``.  Standalone
    member geometry is measured through the owner's member edge uses, so a
    member-only component no longer reports a invented ``0``; whatever the
    owner API cannot measure stays explicitly unknown.
    """

    seeding = options.get("seeding")
    estimates: list[int | None] = []
    basis: list[str] = []
    for component in partition.components:
        try:
            measured = False
            if seeding is not None:
                total = 0
                for face_id in component.face_ids:
                    sides = geometry.faces[face_id].sides()
                    total += (
                        seeding.side_divisions(sides[0])
                        * seeding.side_divisions(sides[1])
                    )
                    measured = True
                for member_id in component.member_ids:
                    for use_id in geometry.members[member_id].edge_use_ids:
                        edge_id = geometry.member_edge_uses[use_id].edge_id
                        total += int(seeding.divisions[edge_id])
                        measured = True
                kind = "seeding"
            else:
                total = 0.0
                for face_id in component.face_ids:
                    lengths = geometry.face_side_lengths(face_id)
                    total += (
                        float(lengths[0]) * float(lengths[1])
                        / float(target_size) ** 2
                    )
                    measured = True
                for member_id in component.member_ids:
                    for use_id in geometry.members[member_id].edge_use_ids:
                        edge_id = geometry.member_edge_uses[use_id].edge_id
                        total += geometry.edge_length(edge_id) / float(target_size)
                        measured = True
                kind = "heuristic"
            for edge_id in component.edge_ids:
                total += geometry.edge_length(edge_id) / float(target_size)
                measured = True
        except Exception:
            estimates.append(None)
            basis.append("unknown")
            continue
        if not measured:
            # No owner-measurable geometry in this component: unknown, never 0.
            estimates.append(None)
            basis.append("unknown")
            continue
        estimates.append(int(math.ceil(total)))
        basis.append(kind)
    return estimates, basis


def _teardown(
    pool: _ParallelPool,
    lease: ParallelPoolLease | None,
    error: BaseException,
) -> None:
    """Bounded cleanup after cancel or failure; the original error survives.

    The lease is invalidated *first* so invalidation stays visible even when
    cleanup itself fails, cleanup is guarded so its exceptions can never mask
    the caller's original cancellation/owner/deadline exception, and the
    full cleanup evidence is attached to that original exception as a note.
    """

    if lease is not None:
        lease._invalidate()
    evidence: Any = None
    cleanup_error: BaseException | None = None
    try:
        pool.set_cancel()
        pool.cancel_pending()
        evidence = pool.shutdown(
            terminate_first=True, deadline=TERMINATION_DEADLINE_SECONDS
        )
    except BaseException as failure:  # never masks the original error
        cleanup_error = failure
    notes = list(getattr(error, "__notes__", []))
    if evidence is not None:
        notes.append(f"parallel pool cleanup evidence: {evidence}")
    if cleanup_error is not None:
        notes.append(
            f"parallel pool cleanup itself failed: {cleanup_error!r}"
        )
    try:
        error.__notes__ = notes
    except Exception:
        pass


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

    Machine- and run-dependent evidence (worker count and PIDs, timings,
    memory, estimates, pool lifecycle) lives under
    ``mesh.hybrid_diagnostics["parallel"]["runtime"]`` so hashers can drop it;
    the deterministic semantics (``used``, ``route``, ``components``,
    ``separation``, ``merge_reasons``) stay outside that block.
    """

    parallel = parallel or ParallelOptions()
    # Canonicalize selections before ANY refusal: every serial fallback —
    # executor refusal included — must see the same None vs explicit-empty
    # faces/members and the canonical beamNone->() semantics.
    options = _normalize_selections(dict(options, target_size=target_size))
    if parallel.executor is not None and parallel.pool_lease is not None:
        raise ValueError(
            "pass either executor or pool_lease to ParallelOptions, not both"
        )
    if parallel.executor is not None:
        # A bare, shared or unknown executor's children cannot be identified,
        # contained or safely terminated, so nothing is ever submitted to it.
        return _serial(geometry, options, _EXECUTOR_FALLBACK_REASON)
    cancellation = options.get("cancellation_check")
    check = _forwarded_check(cancellation)
    for name, empty in _UNSUPPORTED:
        if options.get(name, empty) is not empty:
            return _serial(geometry, options, f"unsupported option {name}")
    # ``interactive`` only audits a supplied change set (refused above), so
    # without one it changes nothing here and workers receive the same mode.
    # ``strict`` audits the whole model and certifies published results.
    certification = str(getattr(options.get("certification_mode"), "value",
                                options.get("certification_mode", "none")))
    if certification not in ("none", "interactive"):
        return _serial(geometry, options, "certification requested")
    # ``working_copy`` declares the caller's model is already an isolated job
    # closure that may be finalised in place; this route never mutates it, and
    # workers mesh their own extracted copies, so both policies are equivalent.
    if str(getattr(options.get("mutation_policy"), "value",
                   options.get("mutation_policy", "read_only"))) not in (
        "read_only", "working_copy"
    ):
        return _serial(geometry, options, "unsupported mutation policy")
    # Selections that name everything are the default selection.
    selected_faces = options.get("face_ids")
    if selected_faces is not None and set(selected_faces) != {
        int(i) for i in geometry.faces
    }:
        return _serial(geometry, options, "partial face selection")
    selected_members = options.get("member_ids")
    if selected_members is not None and set(selected_members) != {
        int(i) for i in geometry.members
    }:
        return _serial(geometry, options, "partial member selection")

    started = time.perf_counter()
    planner = getattr(anygeometry, "plan_independent_components", None)
    if planner is None:
        return _serial(geometry, options, "ANYgeometry has no plan_independent_components")
    # Each support box is grown by this amount, so a gap up to twice it merges.
    # Beam offset reach is caller-declared mesh input the planner cannot see;
    # both neighbours may reach out by it.
    reach = 0.0
    offsets = options.get("beam_offsets") or {}
    if offsets:
        reach = max(
            float(np.max(np.abs(np.asarray(v, dtype=float)))) for v in offsets.values()
        )
    separation = 0.5 * parallel.pad_factor * float(target_size) + reach
    try:
        partition = planner(
            geometry,
            edge_ids=options["beam_edges"],
            separation=separation,
            expected_revision=geometry.revision,
            cancellation_check=check,
        )
    except GeometryError as error:
        return _serial(geometry, options, f"component planning failed: {error}")
    plan_seconds = time.perf_counter() - started
    if not partition.certified:
        reasons = "; ".join(sorted({r.reason for r in partition.refusals}))
        return _serial(geometry, options, f"partition refused: {reasons}")
    if len(partition.components) < max(2, parallel.min_components):
        return _serial(geometry, options, f"{len(partition.components)} component(s)")

    validator = getattr(anygeometry, "validate_component_partition_binding", None)
    if validator is None:
        return _serial(
            geometry, options, "ANYgeometry has no validate_component_partition_binding"
        )
    # Fresh owner binding check at dispatch: a stale, foreign or tampered
    # certificate is a typed GeometryError, never a silent serial fallback.
    _validate_binding(validator, geometry, partition, cancellation)

    estimates, estimate_basis = _component_estimates(
        geometry, partition, options, target_size
    )
    if parallel.min_estimated_elements is not None:
        if any(estimate is None for estimate in estimates):
            return _serial(geometry, options, "component element estimate unknown")
        if sum(estimates) < int(parallel.min_estimated_elements):
            return _serial(
                geometry, options, "estimated elements below min_estimated_elements"
            )

    closures = [
        geometry.extract_model_closure(component.handles)
        for component in partition.components
    ]
    maps = [_id_maps(c) for c in closures]
    payloads = []
    for component, closure, by_kind in zip(partition.components, closures, maps):
        inverse = {kind: {s: w for w, s in table.items()} for kind, table in by_kind.items()}
        local = dict(options)
        local["face_ids"] = tuple(sorted(inverse["face"][f] for f in component.face_ids))
        local["member_ids"] = tuple(
            sorted(inverse["member"][m] for m in component.member_ids)
        )
        local["beam_edges"] = tuple(inverse["edge"][e] for e in component.edge_ids)
        for name in ("overrides", "beam_offsets"):
            table = options.get(name)
            if table:
                local[name] = {
                    inverse["edge"][int(e)]: v for e, v in table.items()
                    if int(e) in inverse["edge"]
                }
        seeding = options.get("seeding")
        if seeding is not None:
            # Independent components share no edge, so a global seeding
            # restricts exactly to each component's own edges.
            from .seeding import Seeding

            local["seeding"] = Seeding(
                divisions={
                    inverse["edge"][int(e)]: v
                    for e, v in seeding.divisions.items()
                    if int(e) in inverse["edge"]
                },
                sweeps=int(seeding.sweeps),
                classes={
                    inverse["edge"][int(e)]: v
                    for e, v in seeding.classes.items()
                    if int(e) in inverse["edge"]
                },
                size_field=seeding.size_field,
            )
        local.pop("cancellation_check", None)
        payloads.append(
            {
                "closure": closure.to_transport(),
                "options": local,
                "whole_has_several_faces": len(geometry.faces) > 1,
            }
        )
    covered = {name: sum(len(p["options"].get(name) or ()) for p in payloads)
               for name in ("overrides", "beam_offsets")}
    for name, total in covered.items():
        if options.get(name) and total != len(options[name]):
            return _serial(geometry, options, f"{name} not owned by a component")
    prepare_seconds = time.perf_counter() - started - plan_seconds

    # A spawn pool on Windows cannot exceed 61 workers.
    workers = max(
        1,
        min(
            parallel.workers or multiprocessing.cpu_count(),
            len(payloads),
            61,
        ),
    )
    lease = parallel.pool_lease
    if lease is not None:
        # Borrow the caller's warm pool exclusively for this job.
        pool = lease._acquire()
        pool_kind = "leased"
    else:
        # Cold pool, owned and disposed by the library.  A daemonic process
        # (ANYfem's GUI worker before outer containment) cannot have children.
        if multiprocessing.current_process().daemon:
            return _serial(
                geometry, options, "worker processes unavailable (daemonic process)"
            )
        try:
            pool = _ParallelPool(workers, name="cold")
        except ParallelContainmentUnavailable as error:
            # Explicit cold-path serial fallback, only for the typed
            # safe-containment failure, before any work has been submitted.
            return _serial(
                geometry, options, f"safe worker containment unavailable: {error}"
            )
        pool_kind = "cold"

    # Guarded transaction: acquisition, submission, the run, the join, the
    # release and the cold disposal all sit inside one guarded region, so no
    # path can leak the lease or the workers.
    outputs: list[_ComponentOutput | None] = [None] * len(payloads)
    run_started = time.perf_counter()
    total = len(payloads)
    completed = 0
    futures: list[Any] = []
    try:
        check("parallel dispatch")
        futures = [pool.submit(_mesh_component, p, cooperative=True) for p in payloads]
        while completed < total:
            pool.drain_results(POLL_SECONDS)
            dead = pool.dead_workers()
            if dead:
                raise MeshError(
                    f"a component worker process died (worker {dead[0]} of "
                    f"{pool.worker_count}); no mesh was produced"
                )
            for index, future in enumerate(futures):
                if outputs[index] is None and future.done() and not future.cancelled():
                    outputs[index] = future.result()
                    completed += 1
            check(f"parallel component {completed} of {total}")
    except BaseException as error:
        _teardown(pool, lease, error)
        raise
    del futures
    run_seconds = time.perf_counter() - run_started
    memory_bytes = pool.memory_sample()
    startup_info = pool.worker_startup_info()

    join_started = time.perf_counter()
    try:
        # Fresh owner binding check at join: the certificate must still bind
        # the unchanged source model before anything is merged or published.
        _validate_binding(validator, geometry, partition, cancellation)
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
        component_diagnostics = [o.mesh.hybrid_diagnostics for o in outputs]
        complex_parts = [d.get("complex_geometry", {}) for d in component_diagnostics]
        base_diagnostics = {
            "requested_target_size": float(target_size),
            "complex_geometry": {
                "strategy_ladder": next(
                    (c["strategy_ladder"] for c in complex_parts if "strategy_ladder" in c), []
                ),
                **{
                    key: sum(int(c.get(key, 0)) for c in complex_parts)
                    for key in (
                        "source_face_count",
                        "final_face_count",
                        "declared_junction_edge_count",
                    )
                },
            },
            "strategy_by_face": dict(strategy),
            "triangulation_backend_by_face": backend,
            "geometry_model_id": str(geometry.model_id),
            "geometry_revision": int(geometry.revision),
            "certification_mode": certification,
            "certifiable": False,
            "preflight_count": len(preflight),
            "reused_prepared_working_copy": False,
            "phase_seconds": phase_totals,
            "completed_phases": sorted(phase_totals),
        }
    except BaseException as error:
        _teardown(pool, lease, error)
        raise
    # Committed: the job succeeded, so the lease goes back to the owner warm
    # and reusable; a cold pool is disposed within the same bounded deadline.
    # Incomplete cold cleanup fails the job: no result is published while any
    # owned worker survived the deadline.
    pool.end_job()
    if lease is not None:
        lease._release()
        shutdown_evidence = None
    else:
        shutdown_evidence = pool.shutdown()
    join_seconds = time.perf_counter() - join_started
    if shutdown_evidence is not None and shutdown_evidence.get("survivors"):
        raise MeshError(
            "parallel pool cleanup incomplete: workers "
            f"{shutdown_evidence['survivors']} survived the "
            f"{TERMINATION_DEADLINE_SECONDS}s termination deadline; "
            "no mesh was published"
        )
    worker_controls = {
        index: {
            "containment": pool.containment_kind,
            "priority_requested": "below_normal",
            "priority_applied": info.get("priority_applied"),
            "priority_class": _priority_name(info.get("priority_class")),
            "thread_env": info.get("thread_env"),
            # Actual per-library thread counts measured in the worker, or
            # None when no runtime introspection was available there.
            "threadpool": info.get("threadpool"),
        }
        for index, info in startup_info.items()
    }
    runtime: dict[str, Any] = {
        "pool": pool_kind,
        "workers": pool.worker_count,
        "worker_pids": list(pool.worker_pids),
        "worker_controls": worker_controls,
        "plan_seconds": plan_seconds,
        "prepare_seconds": prepare_seconds,
        "mesh_seconds": run_seconds,
        "component_seconds": [o.seconds for o in outputs],
        "component_estimated_elements": estimates,
        "component_estimate_basis": estimate_basis,
        "worker_working_set_bytes": memory_bytes,
        "join_seconds": join_seconds,
        "lifecycle": pool.lifecycle(),
    }
    base_diagnostics["parallel"] = {
        "used": True,
        "route": "parallel",
        "components": len(outputs),
        "separation": separation,
        "merge_reasons": len(partition.merge_reasons),
        "runtime": runtime,
    }
    mesh.hybrid_diagnostics = base_diagnostics
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
        certification_mode=CertificationMode(certification),
        certifiable=False,
    )


def generate_hybrid_mesh_parallel(
    geometry: GeometryModel,
    *,
    target_size: float,
    parallel: ParallelOptions | None = None,
    **options: Any,
) -> Mesh:
    """``generate_hybrid_mesh``-style wrapper for the parallel route."""

    return generate_hybrid_mesh_result_parallel(
        geometry, target_size=target_size, parallel=parallel, **options
    ).mesh
