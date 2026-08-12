"""Production hybrid meshing over authoritative ANYgeometry topology.

The orchestrator deliberately keeps geometry and discretization ownership
separate.  Geometry supplies persistent entity identity, surfaces, structural
records, tolerances and preflight truth.  ANYmesher supplies edge stations,
surface triangulation/recombination and neutral mesh associations.

Mapped faces are delegated to :mod:`anymesher.mapped` unchanged.  Native faces
consume the same model-edge node sequences, so a mapped/native interface shares
node IDs by construction rather than by coordinate welding.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel

from .boundary import GlobalEdgeBoundaryRegistry, MemberRegistry
from .errors import MeshError
from .mapped import ELEMENT_ORDERS, generate_mesh as generate_mapped_mesh
from .mesh import Mesh
from .meshing_view import GeometryMeshingView
from .quality_v2 import assert_valid_mesh
from .refinement import Refinement, SizeField
from .seeding import Seeding, edge_distribution, solve_seeding
from .structural_pipeline import (
    GeometryMutationPolicy,
    OverlapPolicy,
    PreflightStatus,
    StructuralMeshingPipeline,
)
from .surface_mesh import mesh_planar_surface

__all__ = [
    "CertificationMode",
    "HybridMeshResult",
    "MeshingStrategy",
    "generate_hybrid_mesh",
    "generate_hybrid_mesh_result",
]


class MeshingStrategy(str, Enum):
    """Face-discretization policy for the production orchestrator."""

    AUTO = "auto"
    MAPPED = "mapped"
    NATIVE = "native"


class CertificationMode(str, Enum):
    """Geometry audit scope requested for this generation."""

    NONE = "none"
    INTERACTIVE = "interactive"
    STRICT = "strict"


@dataclass(frozen=True)
class HybridMeshResult:
    """Mesh plus the provenance needed by UI/runtime publication."""

    mesh: Mesh
    strategy_by_face: Mapping[int, str]
    preflight: tuple[Any, ...]
    connectivity: Any | None
    audit_report: Any | None
    certification_mode: CertificationMode
    certifiable: bool


@dataclass(frozen=True)
class _LoopBoundary:
    node_ids: tuple[int, ...]
    midside_ids: tuple[int, ...]
    uv: np.ndarray


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value).strip().lower())
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise MeshError(f"unknown {name} {value!r}; expected one of {choices}") from error


def _face_ids(geometry: GeometryModel, face_ids: Iterable[int] | None) -> tuple[int, ...]:
    values = (
        tuple(sorted(int(item) for item in geometry.faces))
        if face_ids is None
        else tuple(sorted(dict.fromkeys(int(item) for item in face_ids)))
    )
    missing = [identifier for identifier in values if identifier not in geometry.faces]
    if missing:
        raise MeshError(f"no face {missing[0]}")
    return values


def _member_edges(
    view: GeometryMeshingView,
    beam_edges: Iterable[int],
    member_ids: Iterable[int] | None,
) -> tuple[int, ...]:
    result = {int(edge_id) for edge_id in beam_edges}
    available = getattr(view, "members", {})
    selected = (
        tuple(sorted(int(item) for item in available))
        if member_ids is None
        else tuple(sorted(dict.fromkeys(int(item) for item in member_ids)))
    )
    registry = MemberRegistry(view)
    for member_id in selected:
        try:
            spans = registry.spans(member_id)
        except (KeyError, ValueError) as error:
            raise MeshError(f"no structural member {member_id}") from error
        result.update(int(span.edge_id) for span in spans)
    missing = [identifier for identifier in sorted(result) if identifier not in view.edges]
    if missing:
        raise MeshError(f"no edge {missing[0]}")
    return tuple(sorted(result))


def _active_edges(
    geometry: GeometryModel,
    face_ids: Sequence[int],
    beam_edges: Sequence[int],
) -> tuple[int, ...]:
    result = set(int(item) for item in beam_edges)
    for face_id in face_ids:
        face = geometry.faces[face_id]
        for loop in (face.loop, *face.holes):
            result.update(int(item.edge) for item in loop)
    return tuple(sorted(result))


def _mappable(face: Any) -> bool:
    return len(face.corners) == 4 and not face.holes


def _blocked_preflight(states: Sequence[Any]) -> tuple[Any, ...]:
    blocked = []
    for state in states:
        status = getattr(state, "status", None)
        value = getattr(status, "value", status)
        if value in {PreflightStatus.BLOCKED.value, PreflightStatus.STALE.value}:
            blocked.append(state)
    return tuple(blocked)


def _next_identifier(values: Iterable[int]) -> int:
    return max((int(item) for item in values), default=0) + 1


def _station_parameters(
    geometry: GeometryModel,
    edge_id: int,
    stations: int,
    size_field: SizeField,
) -> np.ndarray:
    if stations < 1:
        raise MeshError(f"edge {edge_id} has an invalid zero-division seed")
    interior = (
        edge_distribution(geometry, edge_id, stations, size_field)
        if stations > 1
        else np.empty(0, dtype=float)
    )
    return np.concatenate(([0.0], np.asarray(interior, dtype=float), [1.0]))


def _ensure_edge_registry(
    geometry: GeometryModel,
    view: GeometryMeshingView,
    mesh: Mesh,
    registry: GlobalEdgeBoundaryRegistry,
    edge_ids: Sequence[int],
    seeding: Seeding,
    size_field: SizeField,
    order: str,
) -> None:
    next_node = _next_identifier(mesh.nodes)
    steps_per_division = 2 if order == "quadratic" else 1
    for edge_id in edge_ids:
        if edge_id not in seeding.divisions:
            raise MeshError(f"seeding has no division count for edge {edge_id}")
        sequence = mesh.nodes_of_edge.get(edge_id)
        if sequence is None:
            edge = geometry.edges[edge_id]
            for vertex_id in (edge.start, edge.end):
                if vertex_id not in mesh.node_of_vertex:
                    mesh.node_of_vertex[vertex_id] = next_node
                    mesh.nodes[next_node] = np.asarray(
                        geometry.vertex_position(vertex_id), dtype=float
                    )
                    next_node += 1
            stations = int(seeding[edge_id]) * steps_per_division
            parameters = _station_parameters(
                geometry, edge_id, stations, size_field
            )
            sequence = [mesh.node_of_vertex[edge.start]]
            if len(parameters) > 2:
                points = geometry.sample_edge(edge_id, parameters[1:-1])
                for point in points:
                    mesh.nodes[next_node] = np.asarray(point, dtype=float)
                    sequence.append(next_node)
                    next_node += 1
            sequence.append(mesh.node_of_vertex[edge.end])
            mesh.nodes_of_edge[edge_id] = sequence
        parameters = _station_parameters(
            geometry, edge_id, len(sequence) - 1, size_field
        )
        if len(parameters) != len(sequence):
            raise MeshError(
                f"edge {edge_id} boundary registry length disagrees with its seeding"
            )
        registry.register_many(
            edge_id,
            parameters,
            points=[mesh.nodes[node_id] for node_id in sequence],
            node_ids=sequence,
            owner=geometry.handle("edge", edge_id),
        )


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1))
        - np.sum(points[:, 1] * np.roll(points[:, 0], -1))
    )


def _reverse_loop(boundary: _LoopBoundary) -> _LoopBoundary:
    count = len(boundary.node_ids)
    nodes = tuple(reversed(boundary.node_ids))
    uv = boundary.uv[::-1].copy()
    if not boundary.midside_ids:
        return _LoopBoundary(nodes, (), uv)
    mids = tuple(
        boundary.midside_ids[(count - 2 - index) % count]
        for index in range(count)
    )
    return _LoopBoundary(nodes, mids, uv)


def _loop_boundary(
    geometry: GeometryModel,
    mesh: Mesh,
    face_id: int,
    loop: Sequence[OrientedEdge],
    *,
    quadratic: bool,
    counter_clockwise: bool,
) -> _LoopBoundary:
    corner_nodes: list[int] = []
    midside_nodes: list[int] = []
    for oriented in loop:
        sequence = list(mesh.nodes_of_edge[oriented.edge])
        if not oriented.forward:
            sequence.reverse()
        if quadratic:
            if (len(sequence) - 1) % 2:
                raise MeshError(
                    f"quadratic edge {oriented.edge} does not have paired corner/midside stations"
                )
            edge_corners = sequence[::2]
            edge_midsides = sequence[1::2]
            midside_nodes.extend(edge_midsides)
        else:
            edge_corners = sequence
        corner_nodes.extend(edge_corners[:-1])
    if len(corner_nodes) < 3:
        raise MeshError(f"face {face_id} has fewer than three boundary stations")
    if quadratic and len(midside_nodes) != len(corner_nodes):
        raise MeshError(f"face {face_id} has an inconsistent quadratic boundary")
    try:
        uv = np.asarray(
            [
                geometry.face_local_uv(face_id, mesh.nodes[node_id])
                for node_id in corner_nodes
            ],
            dtype=float,
        )
    except GeometryError as error:
        raise MeshError(
            f"native face {face_id} has no qualified surface chart: {error}. "
            "Attach an authoritative Plane/Cylinder surface or partition it "
            "into mapped patches; ANYmesher will not invent geometry truth."
        ) from error
    if uv.shape != (len(corner_nodes), 2) or not np.all(np.isfinite(uv)):
        raise MeshError(f"face {face_id} surface chart returned invalid UV coordinates")
    boundary = _LoopBoundary(tuple(corner_nodes), tuple(midside_nodes), uv)
    area = _signed_area(uv)
    scale = max(float(np.max(np.abs(uv))), 1.0)
    if abs(area) <= 128.0 * np.finfo(float).eps * scale * scale:
        raise MeshError(
            f"face {face_id} has a degenerate surface chart; its boundary cannot be triangulated"
        )
    if (area > 0.0) != bool(counter_clockwise):
        boundary = _reverse_loop(boundary)
    return boundary


def _core_edge_midsides(core: Any) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    triangles = np.asarray(core.triangle_connectivity, dtype=np.int64)
    quadrilaterals = np.asarray(core.quad_connectivity, dtype=np.int64)
    for row in triangles:
        if len(row) >= 6:
            for first, second, middle in (
                (row[0], row[1], row[3]),
                (row[1], row[2], row[4]),
                (row[2], row[0], row[5]),
            ):
                result[tuple(sorted((int(first), int(second))))] = int(middle)
    for row in quadrilaterals:
        if len(row) >= 8:
            for first, second, middle in (
                (row[0], row[1], row[4]),
                (row[1], row[2], row[5]),
                (row[2], row[3], row[6]),
                (row[3], row[0], row[7]),
            ):
                result[tuple(sorted((int(first), int(second))))] = int(middle)
    return result


def _active_rows(connectivity: Any, activity: Any) -> Iterable[np.ndarray]:
    rows = np.asarray(connectivity, dtype=np.int64)
    flags = np.asarray(activity, dtype=bool)
    if len(flags) != len(rows):
        raise MeshError("native mesh connectivity/activity arrays disagree")
    return rows[flags]


def _check_cancellation(
    cancellation_check: Callable[[str], None] | None, stage: str
) -> None:
    if cancellation_check is not None:
        cancellation_check(stage)


def _mesh_native_face(
    geometry: GeometryModel,
    mesh: Mesh,
    face_id: int,
    *,
    order: str,
    recombine: bool,
    native_backend: Any,
    cancellation_check: Callable[[str], None] | None,
) -> None:
    _check_cancellation(cancellation_check, f"native face {face_id} boundary start")
    face = geometry.faces[face_id]
    quadratic = order == "quadratic"
    outer = _loop_boundary(
        geometry,
        mesh,
        face_id,
        face.loop,
        quadratic=quadratic,
        counter_clockwise=True,
    )
    holes = tuple(
        _loop_boundary(
            geometry,
            mesh,
            face_id,
            loop,
            quadratic=quadratic,
            counter_clockwise=False,
        )
        for loop in face.holes
    )
    loops = (outer, *holes)
    segments = [
        float(np.linalg.norm(loop.uv[(index + 1) % len(loop.uv)] - loop.uv[index]))
        for loop in loops
        for index in range(len(loop.uv))
    ]
    # Edge seeding is authoritative.  A size just above the longest registered
    # chart segment lets the surface filler add interior points without adding
    # unregistered boundary stations.
    chart_size = max(segments) * (1.0 + 64.0 * np.finfo(float).eps)
    core = mesh_planar_surface(
        outer.uv,
        tuple(loop.uv for loop in holes),
        target_size=chart_size,
        recombine=recombine,
        order=order,
        backend=native_backend,
        owner=geometry.handle("face", face_id),
        cancellation_check=cancellation_check,
    )
    _check_cancellation(cancellation_check, f"native face {face_id} lifting start")
    assert_valid_mesh(core)

    input_uv = np.vstack([loop.uv for loop in loops])
    coordinates = np.asarray(core.node_coordinates, dtype=float)
    if len(coordinates) < len(input_uv) or not np.allclose(
        coordinates[: len(input_uv), :2], input_uv, rtol=0.0, atol=2.0e-14
    ):
        raise MeshError(
            f"native face {face_id} changed its registered boundary ordering; "
            "conformal identity cannot be guaranteed"
        )

    core_to_global: dict[int, int] = {}
    offset = 0
    for loop in loops:
        for local, node_id in enumerate(loop.node_ids):
            core_to_global[offset + local] = int(node_id)
        offset += len(loop.node_ids)

    if quadratic:
        edge_midsides = _core_edge_midsides(core)
        offset = 0
        for loop in loops:
            count = len(loop.node_ids)
            for local, node_id in enumerate(loop.midside_ids):
                key = tuple(sorted((offset + local, offset + (local + 1) % count)))
                core_mid = edge_midsides.get(key)
                if core_mid is None:
                    raise MeshError(
                        f"native face {face_id} lost a quadratic boundary segment"
                    )
                previous = core_to_global.setdefault(core_mid, int(node_id))
                if previous != int(node_id):
                    raise MeshError(
                        f"native face {face_id} produced conflicting midside identity"
                    )
            offset += count

    next_node = _next_identifier(mesh.nodes)
    for core_node in range(len(coordinates)):
        if core_node in core_to_global:
            continue
        u, v = (float(value) for value in coordinates[core_node, :2])
        point = np.asarray(geometry.face_point(face_id, u, v), dtype=float)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise MeshError(f"face {face_id} surface evaluation returned an invalid point")
        core_to_global[core_node] = next_node
        mesh.nodes[next_node] = point
        next_node += 1

    next_element = _next_identifier((*mesh.quads, *mesh.tris, *mesh.beams))
    elements: list[int] = []
    for row in _active_rows(core.triangle_connectivity, core.triangle_active):
        connectivity = tuple(core_to_global[int(item)] for item in row)
        mesh.tris[next_element] = connectivity
        elements.append(next_element)
        next_element += 1
    for row in _active_rows(core.quad_connectivity, core.quad_active):
        connectivity = tuple(core_to_global[int(item)] for item in row)
        mesh.quads[next_element] = connectivity
        elements.append(next_element)
        next_element += 1
    if not elements:
        raise MeshError(f"native meshing produced no active elements for face {face_id}")
    mesh.elements_of_face[face_id] = elements


def _audit_geometry(
    geometry: GeometryModel,
    mode: CertificationMode,
    *,
    change_set: Any | None,
    policy: Any | None,
) -> tuple[Any | None, bool]:
    if mode is CertificationMode.NONE:
        return None, False
    if mode is CertificationMode.INTERACTIVE:
        if change_set is None:
            return None, False
        try:
            from anygeometry.audit import audit_changed_region
        except ImportError as error:
            raise MeshError(
                "interactive changed-region audit requires ANYgeometry 0.2 gap-closure APIs"
            ) from error
        return audit_changed_region(geometry, change_set, policy=policy), False

    from anygeometry.audit import strict_audit

    report = strict_audit(geometry, policy=policy)
    certifiable = bool(getattr(report, "certifiable", True))
    for name in ("ok", "passed", "valid"):
        value = getattr(report, name, None)
        if value is not None and not bool(value):
            raise MeshError(f"strict geometry certification failed: {report}")
    issues = getattr(report, "issues", ())
    if issues:
        raise MeshError(f"strict geometry certification found {len(issues)} issue(s)")
    if not certifiable:
        raise MeshError("strict geometry audit returned a non-certifiable report")
    return report, True


def generate_hybrid_mesh_result(
    geometry: GeometryModel,
    *,
    target_size: float,
    strategy: MeshingStrategy | str = MeshingStrategy.AUTO,
    overrides: Mapping[int, int] | None = None,
    beam_edges: Iterable[int] = (),
    beam_offsets: Mapping[int, float] | None = None,
    member_ids: Iterable[int] | None = None,
    face_ids: Iterable[int] | None = None,
    seeding: Seeding | None = None,
    refinements: Iterable[Refinement] = (),
    order: str = "linear",
    recombine: bool = True,
    native_backend: Any = "python",
    overlap_policy: OverlapPolicy | str = OverlapPolicy.REJECT,
    mutation_policy: GeometryMutationPolicy | str = GeometryMutationPolicy.READ_ONLY,
    certification_mode: CertificationMode | str = CertificationMode.NONE,
    change_set: Any | None = None,
    audit_policy: Any | None = None,
    cancellation_check: Callable[[str], None] | None = None,
) -> HybridMeshResult:
    """Generate a model-bound mapped/native mesh without rewriting geometry.

    ``cancellation_check`` receives diagnostic safe-phase names. Cancellation is
    cooperative, so its latency is bounded by the current uninterrupted phase.
    """

    _check_cancellation(cancellation_check, "hybrid generation start")
    target_size = float(target_size)
    if not np.isfinite(target_size) or target_size <= 0.0:
        raise MeshError("target_size must be finite and positive")
    if order not in ELEMENT_ORDERS:
        raise MeshError(
            f"unknown element order {order!r}; expected one of {', '.join(ELEMENT_ORDERS)}"
        )
    strategy = _enum_value(strategy, MeshingStrategy, "meshing strategy")
    certification_mode = _enum_value(
        certification_mode, CertificationMode, "certification mode"
    )
    overlap_policy = _enum_value(overlap_policy, OverlapPolicy, "overlap policy")
    mutation_policy = _enum_value(
        mutation_policy, GeometryMutationPolicy, "geometry mutation policy"
    )

    view = GeometryMeshingView(geometry)
    faces = _face_ids(geometry, face_ids)
    beams = _member_edges(view, beam_edges, member_ids)
    if not faces and not beams:
        raise MeshError("nothing to mesh: no faces and no beam edges")

    pipeline = StructuralMeshingPipeline(
        view,
        overlap_policy=overlap_policy,
        mutation_policy=mutation_policy,
    )
    preflight = tuple(pipeline.preflight())
    blocked = _blocked_preflight(preflight)
    if blocked:
        detail = "; ".join(str(item) for item in blocked[:5])
        raise MeshError(f"structural meshing preflight blocked generation: {detail}")

    if strategy is MeshingStrategy.MAPPED:
        mapped_faces, native_faces = faces, ()
    elif strategy is MeshingStrategy.NATIVE:
        mapped_faces, native_faces = (), faces
    else:
        mapped_faces = tuple(face_id for face_id in faces if _mappable(geometry.faces[face_id]))
        native_faces = tuple(face_id for face_id in faces if face_id not in set(mapped_faces))

    refinements = tuple(refinements)
    size_field = (
        seeding.size_field
        if seeding is not None and seeding.size_field is not None
        else SizeField(geometry, target_size, refinements)
    )
    edges = _active_edges(geometry, faces, beams)
    if seeding is None:
        seeding = solve_seeding(
            geometry,
            size_field=size_field,
            overrides=overrides,
            edge_ids=edges,
        )
    _check_cancellation(cancellation_check, "hybrid seeding complete")

    if mapped_faces or beams:
        _check_cancellation(cancellation_check, "mapped generation start")
        mesh = generate_mapped_mesh(
            geometry,
            target_size=target_size,
            overrides=overrides,
            beam_edges=beams,
            beam_offsets=beam_offsets,
            face_ids=mapped_faces,
            seeding=seeding,
            refinements=refinements,
            order=order,
        )
        _check_cancellation(cancellation_check, "mapped generation complete")
    else:
        mesh = Mesh(
            geometry_model_id=geometry.model_id,
            geometry_revision=geometry.revision,
            seeding=seeding,
            order=order,
        )

    boundary_registry = GlobalEdgeBoundaryRegistry(view)
    _ensure_edge_registry(
        geometry,
        view,
        mesh,
        boundary_registry,
        edges,
        seeding,
        size_field,
        order,
    )
    for face_id in native_faces:
        _mesh_native_face(
            geometry,
            mesh,
            face_id,
            order=order,
            recombine=bool(recombine),
            native_backend=native_backend,
            cancellation_check=cancellation_check,
        )
        _check_cancellation(cancellation_check, f"native face {face_id} complete")

    _check_cancellation(cancellation_check, "hybrid connectivity start")
    for element_id in (*mesh.quads, *mesh.tris, *mesh.beams):
        mesh.activity.setdefault(int(element_id), 1.0)
    connectivity = pipeline.apply_connectivity(mesh)
    view.assert_current(geometry)
    audit_report, certifiable = _audit_geometry(
        geometry,
        certification_mode,
        change_set=change_set,
        policy=audit_policy,
    )
    strategies = {
        face_id: ("mapped" if face_id in set(mapped_faces) else "native")
        for face_id in faces
    }
    result = HybridMeshResult(
        mesh=mesh,
        strategy_by_face=strategies,
        preflight=preflight,
        connectivity=connectivity,
        audit_report=audit_report,
        certification_mode=certification_mode,
        certifiable=certifiable,
    )
    mesh.hybrid_diagnostics = {
        "strategy_by_face": dict(strategies),
        "geometry_model_id": str(geometry.model_id),
        "geometry_revision": int(geometry.revision),
        "certification_mode": certification_mode.value,
        "certifiable": bool(certifiable),
        "preflight_count": len(preflight),
    }
    mesh.boundary_registry = boundary_registry
    _check_cancellation(cancellation_check, "hybrid generation complete")
    return result


def generate_hybrid_mesh(geometry: GeometryModel, **options: Any) -> Mesh:
    """Return only the neutral mesh from :func:`generate_hybrid_mesh_result`."""

    return generate_hybrid_mesh_result(geometry, **options).mesh
