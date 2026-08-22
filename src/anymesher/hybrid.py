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
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel

from .boundary import GlobalEdgeBoundaryRegistry, MemberRegistry
from .core import MeshCore
from .errors import MeshError
from .mapped import ELEMENT_ORDERS, generate_mesh as generate_mapped_mesh
from .mesh import Mesh
from .meshing_view import GeometryMeshingView
from .prepared import remap_prepared_mesh_associations
from .quality_v2 import assert_valid_mesh, evaluate_quality
from .refinement import Refinement, SizeField
from .seeding import Seeding, edge_distribution, solve_seeding
from .structural_pipeline import (
    GeometryMutationPolicy,
    OverlapPolicy,
    PreflightStatus,
    StructuralMeshingPipeline,
)
from .surface_mesh import mesh_planar_surface
from .structured import (
    StructuredLayoutReport,
    StructuredMeshingOptions,
    apply_structured_layout,
    plan_structured_layout,
    regularity_metrics,
)

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
    triangulation_backend_by_face: Mapping[int, Mapping[str, Any]]
    preflight: tuple[Any, ...]
    connectivity: Any | None
    audit_report: Any | None
    certification_mode: CertificationMode
    certifiable: bool
    structured_layout: StructuredLayoutReport | None = None


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


def _published_boundary_registry(
    geometry: GeometryModel,
    mesh: Mesh,
) -> GlobalEdgeBoundaryRegistry:
    """Rebuild exact source-edge stations after detached preparation."""

    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(geometry))
    for edge_id, sequence in sorted(mesh.nodes_of_edge.items()):
        if edge_id not in geometry.edges or not sequence:
            continue
        parameters: list[float] = []
        for node_id in sequence:
            _point, parameter, distance = geometry.closest_edge_point(
                edge_id, mesh.nodes[node_id]
            )
            length = geometry.edge_length(edge_id)
            tolerance = max(
                geometry.tolerance.effective_length(length),
                128.0 * np.finfo(float).eps * max(length, 1.0),
            )
            if distance > tolerance:
                raise MeshError(
                    f"published node {node_id} is not on exact source edge "
                    f"{edge_id} (residual {distance:.6g} m)"
                )
            parameters.append(float(parameter))
        registry.register_many(
            edge_id,
            parameters,
            points=[mesh.nodes[node_id] for node_id in sequence],
            node_ids=sequence,
            owner=geometry.handle("edge", edge_id),
        )
    return registry


def _source_backend_diagnostics(
    report: StructuredLayoutReport,
    working_diagnostics: Mapping[int, Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    decisions = {item.source_face_id: item for item in report.plan.faces}
    result: dict[int, Mapping[str, Any]] = {}
    for source_face, descendants in sorted(
        report.source_to_working_faces.items()
    ):
        records = [
            dict(working_diagnostics[face_id])
            for face_id in descendants
            if face_id in working_diagnostics
        ]
        selected = {str(item.get("actual_backend", "unknown")) for item in records}
        decision = decisions[source_face]
        result[source_face] = {
            "requested_backend": (
                records[0].get("requested_backend") if len(records) == 1 else "mapped"
            ),
            "selected_backend": (
                records[0].get("selected_backend") if len(records) == 1 else "mapped"
            ),
            "actual_backend": (
                records[0].get("actual_backend")
                if len(selected) == 1 and records
                else ("mapped" if not selected or selected == {"mapped"} else "mixed")
            ),
            "fallback_reason": (
                records[0].get("fallback_reason") if len(records) == 1 else None
            ),
            "phase_seconds": {
                "working_face_total": sum(
                    float(value)
                    for record in records
                    for value in record.get("phase_seconds", {}).values()
                )
            },
            "structured_action": decision.action,
            "working_face_ids": list(descendants),
            "working_face_diagnostics": records,
        }
    return result


def _neutral_shell_core(mesh: Mesh) -> MeshCore:
    node_ids = np.asarray(sorted(mesh.nodes), dtype=np.int64)
    coordinates = np.asarray([mesh.nodes[int(item)] for item in node_ids], dtype=float)
    triangle_ids = np.asarray(sorted(mesh.tris), dtype=np.int64)
    quad_ids = np.asarray(sorted(mesh.quads), dtype=np.int64)
    triangle_width = 6 if mesh.is_quadratic else 3
    quad_width = 8 if mesh.is_quadratic else 4
    triangles = np.asarray(
        [mesh.tris[int(item)] for item in triangle_ids], dtype=np.int64
    ).reshape((-1, triangle_width))
    quadrilaterals = np.asarray(
        [mesh.quads[int(item)] for item in quad_ids], dtype=np.int64
    ).reshape((-1, quad_width))
    return MeshCore.from_id_connectivity(
        coordinates,
        node_ids=node_ids,
        triangles=triangles,
        quadrilaterals=quadrilaterals,
        triangle_ids=triangle_ids,
        quad_ids=quad_ids,
    )


def _element_growth(
    mesh: Mesh,
    *,
    limit: float,
) -> tuple[float, tuple[tuple[int, int, float], ...]]:
    characteristic: dict[int, float] = {}
    incidence: dict[tuple[int, int], list[int]] = {}
    for element_id in sorted(mesh.shells):
        corners = mesh.corners_of(element_id)
        lengths = []
        for first, second in zip(corners, corners[1:] + corners[:1]):
            edge = tuple(sorted((int(first), int(second))))
            lengths.append(
                float(np.linalg.norm(mesh.nodes[second] - mesh.nodes[first]))
            )
            incidence.setdefault(edge, []).append(int(element_id))
        characteristic[int(element_id)] = float(np.mean(lengths))
    maximum = 1.0
    violations: list[tuple[int, int, float]] = []
    for attached in incidence.values():
        if len(attached) != 2:
            continue
        first, second = attached
        small = min(characteristic[first], characteristic[second])
        ratio = (
            float("inf")
            if small <= 0.0
            else max(characteristic[first], characteristic[second]) / small
        )
        maximum = max(maximum, ratio)
        if ratio > limit + 1.0e-14:
            violations.append((min(first, second), max(first, second), ratio))
    return maximum, tuple(sorted(violations))


def _structured_quality_report(
    mesh: Mesh,
    options: StructuredMeshingOptions,
) -> dict[str, Any]:
    quality = evaluate_quality(_neutral_shell_core(mesh))
    policy = options.quality_policy
    violations: dict[str, int] = {
        "minimum_scaled_jacobian": 0,
        "maximum_aspect_ratio": 0,
        "minimum_angle": 0,
        "maximum_angle": 0,
        "maximum_warpage": 0,
    }
    poor: set[int] = set()
    for group in (quality.triangles, quality.quadrilaterals):
        masks = {
            "minimum_scaled_jacobian": (
                group.scaled_jacobian < policy.minimum_scaled_jacobian
            ),
            "maximum_aspect_ratio": (
                group.aspect_ratio > policy.maximum_aspect_ratio
            ),
            "minimum_angle": group.minimum_angle < policy.minimum_angle,
            "maximum_angle": group.maximum_angle > policy.maximum_angle,
            "maximum_warpage": group.warpage > policy.maximum_warpage,
        }
        for name, mask in masks.items():
            violations[name] += int(np.count_nonzero(mask))
            poor.update(int(item) for item in group.element_ids[mask])
    growth, growth_pairs = _element_growth(
        mesh,
        limit=options.max_element_growth,
    )
    growth_violations = len(growth_pairs)
    accepted = not any(violations.values()) and growth_violations == 0
    return {
        "accepted": accepted,
        "policy": policy.to_dict(),
        "violation_counts": violations,
        "poor_element_ids": sorted(poor),
        "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
        "maximum_aspect_ratio": quality.maximum_aspect_ratio,
        "minimum_angle": quality.minimum_angle,
        "maximum_angle": max(
            (
                float(np.max(group.maximum_angle))
                for group in (quality.triangles, quality.quadrilaterals)
                if len(group)
            ),
            default=90.0,
        ),
        "maximum_warpage": quality.maximum_warpage,
        "maximum_adjacent_element_growth": growth,
        "growth_limit": options.max_element_growth,
        "growth_violation_count": growth_violations,
        "growth_violation_pairs": [
            [first, second, ratio]
            for first, second, ratio in growth_pairs[:16]
        ],
    }


def _quality_rejection_message(quality: Mapping[str, Any]) -> str:
    counts = {
        str(name): int(value)
        for name, value in quality.get("violation_counts", {}).items()
        if int(value)
    }
    details = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    growth = int(quality.get("growth_violation_count", 0))
    if growth:
        details = f"{details}, " if details else ""
        details += f"element_growth={growth}"
    poor = tuple(int(item) for item in quality.get("poor_element_ids", ())[:16])
    suffix = f"; poor element IDs {poor}" if poor else ""
    return f"structured mesh violates quality_v2 ({details or 'unspecified'}){suffix}"


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
) -> dict[str, Any]:
    boundary_started = perf_counter()
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
    boundary_seconds = perf_counter() - boundary_started
    # Edge seeding is authoritative.  A size just above the longest registered
    # chart segment lets the surface filler add interior points without adding
    # unregistered boundary stations.
    chart_size = max(segments) * (1.0 + 64.0 * np.finfo(float).eps)
    surface_diagnostics: dict[str, Any] = {}
    core = mesh_planar_surface(
        outer.uv,
        tuple(loop.uv for loop in holes),
        target_size=chart_size,
        recombine=recombine,
        order=order,
        backend=native_backend,
        owner=geometry.handle("face", face_id),
        cancellation_check=cancellation_check,
        diagnostics=surface_diagnostics,
    )
    surface_diagnostics.setdefault("phase_seconds", {})[
        "boundary_registration"
    ] = boundary_seconds
    _check_cancellation(cancellation_check, f"native face {face_id} lifting start")
    lifting_started = perf_counter()
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
    surface_diagnostics.setdefault("phase_seconds", {})[
        "surface_lifting_and_publication"
    ] = perf_counter() - lifting_started
    return surface_diagnostics


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
    native_backend: Any = "auto",
    structured_options: StructuredMeshingOptions | Mapping[str, Any] | None = None,
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

    generation_started = perf_counter()
    phase_seconds: dict[str, float] = {}
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

    source_geometry = geometry
    requested_beam_edges = tuple(int(item) for item in beam_edges)
    requested_member_ids = (
        None if member_ids is None else tuple(int(item) for item in member_ids)
    )
    requested_face_ids = (
        None if face_ids is None else tuple(int(item) for item in face_ids)
    )
    requested_seeding = seeding
    refinements = tuple(refinements)

    preflight_started = perf_counter()
    source_view = GeometryMeshingView(source_geometry)
    source_faces = _face_ids(source_geometry, requested_face_ids)
    source_beams = _member_edges(
        source_view, requested_beam_edges, requested_member_ids
    )
    if not source_faces and not source_beams:
        raise MeshError("nothing to mesh: no faces and no beam edges")

    source_pipeline = StructuralMeshingPipeline(
        source_view,
        overlap_policy=overlap_policy,
        mutation_policy=mutation_policy,
    )
    source_preflight = tuple(source_pipeline.preflight())
    blocked = _blocked_preflight(source_preflight)
    if blocked:
        detail = "; ".join(str(item) for item in blocked[:5])
        raise MeshError(f"structural meshing preflight blocked generation: {detail}")

    structured_report: StructuredLayoutReport | None = None
    if structured_options is not None and strategy is MeshingStrategy.NATIVE:
        raise MeshError(
            "structured_options cannot be combined with strategy='native'; "
            "choose 'auto' for quality-gated structured fallback or 'mapped' "
            "to require mapped blocks"
        )
    if structured_options is not None and source_faces:
        structured_started = perf_counter()
        plan = plan_structured_layout(
            source_geometry,
            target_size=target_size,
            face_ids=source_faces,
            options=structured_options,
            explicit_seeding=seeding is not None,
            overrides=overrides,
            protected_edge_ids=source_beams,
            cancellation_check=cancellation_check,
        )
        if strategy is MeshingStrategy.MAPPED:
            residual = [item for item in plan.faces if not item.structured]
            if residual:
                detail = "; ".join(
                    f"face {item.source_face_id}: {item.reason}"
                    for item in residual[:8]
                )
                raise MeshError(
                    "explicit mapped strategy cannot create mapped blocks for "
                    f"every selected face: {detail}"
                )
        geometry, structured_report = apply_structured_layout(
            source_geometry,
            plan,
            cancellation_check=cancellation_check,
        )
        phase_seconds["structured_planning_and_application"] = (
            perf_counter() - structured_started
        )

    if structured_report is None:
        faces = source_faces
        beams = source_beams
        source_strategies = {
            face_id: (
                "mapped"
                if strategy is MeshingStrategy.MAPPED
                or (
                    strategy is MeshingStrategy.AUTO
                    and _mappable(source_geometry.faces[face_id])
                )
                else "native"
            )
            for face_id in source_faces
        }
    else:
        decisions = {
            item.source_face_id: item for item in structured_report.plan.faces
        }
        faces = tuple(
            working_face
            for source_face in source_faces
            for working_face in structured_report.source_to_working_faces[source_face]
        )
        beams = tuple(
            dict.fromkeys(
                working_edge
                for source_edge in source_beams
                for working_edge in structured_report.source_to_working_edges[
                    source_edge
                ]
            )
        )
        source_strategies = {
            source_face: (
                "mapped" if decisions[source_face].structured else "native"
            )
            for source_face in source_faces
        }

    mapped_source_faces = {
        face_id for face_id, value in source_strategies.items() if value == "mapped"
    }
    if structured_report is None:
        mapped_faces = tuple(
            face_id for face_id in faces if face_id in mapped_source_faces
        )
        native_faces = tuple(
            face_id for face_id in faces if face_id not in mapped_source_faces
        )
    else:
        mapped_faces = tuple(
            working_face
            for source_face in source_faces
            if source_face in mapped_source_faces
            for working_face in structured_report.source_to_working_faces[source_face]
        )
        native_faces = tuple(
            working_face
            for source_face in source_faces
            if source_face not in mapped_source_faces
            for working_face in structured_report.source_to_working_faces[source_face]
        )

    view = GeometryMeshingView(geometry)
    pipeline = StructuralMeshingPipeline(
        view,
        overlap_policy=overlap_policy,
        mutation_policy=mutation_policy,
    )
    preflight = tuple(pipeline.preflight())
    blocked = _blocked_preflight(preflight)
    if blocked:
        detail = "; ".join(str(item) for item in blocked[:5])
        raise MeshError(
            f"prepared structural meshing preflight blocked generation: {detail}"
        )
    phase_seconds["geometry_and_preflight"] = perf_counter() - preflight_started

    seeding_started = perf_counter()
    size_field = (
        seeding.size_field
        if seeding is not None and seeding.size_field is not None
        else SizeField(geometry, target_size, refinements)
    )
    edges = _active_edges(geometry, faces, beams)
    if seeding is None:
        effective_overrides = dict(overrides or {})
        if structured_report is not None:
            for edge_id, divisions in structured_report.seed_solution.items():
                previous = effective_overrides.setdefault(edge_id, divisions)
                if previous != divisions:
                    raise MeshError(
                        f"structured seed solution for edge {edge_id} ({divisions}) "
                        f"conflicts with explicit override {previous}"
                    )
        seeding = solve_seeding(
            geometry,
            size_field=size_field,
            overrides=effective_overrides,
            edge_ids=edges,
        )
    phase_seconds["seeding"] = perf_counter() - seeding_started
    _check_cancellation(cancellation_check, "hybrid seeding complete")

    if mapped_faces or beams:
        mapped_started = perf_counter()
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
        phase_seconds["mapped_generation"] = perf_counter() - mapped_started
    else:
        mesh = Mesh(
            geometry_model_id=geometry.model_id,
            geometry_revision=geometry.revision,
            seeding=seeding,
            order=order,
        )

    boundary_registry = GlobalEdgeBoundaryRegistry(view)
    triangulation_backend_by_face: dict[int, Mapping[str, Any]] = {
        int(face_id): {
            "requested_backend": "mapped",
            "selected_backend": "mapped",
            "actual_backend": "mapped",
            "fallback_reason": None,
            "phase_seconds": {},
        }
        for face_id in mapped_faces
    }
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
        face_diagnostics = _mesh_native_face(
            geometry,
            mesh,
            face_id,
            order=order,
            recombine=bool(recombine),
            native_backend=native_backend,
            cancellation_check=cancellation_check,
        )
        triangulation_backend_by_face[int(face_id)] = face_diagnostics
        _check_cancellation(cancellation_check, f"native face {face_id} complete")

    for sheet_id, sheet in geometry.sheets.items():
        element_ids = {
            int(element_id)
            for face_use_id in sheet.face_use_ids
            for element_id in mesh.elements_of_face.get(
                geometry.face_uses[face_use_id].face_id, ()
            )
        }
        mesh.elements_of_sheet[int(sheet_id)] = sorted(element_ids)

    connectivity_started = perf_counter()
    _check_cancellation(cancellation_check, "hybrid connectivity start")
    for element_id in (*mesh.quads, *mesh.tris, *mesh.beams):
        mesh.activity.setdefault(int(element_id), 1.0)
    connectivity = pipeline.apply_connectivity(mesh)
    phase_seconds["structural_connectivity"] = perf_counter() - connectivity_started
    view.assert_current(geometry)
    audit_report, certifiable = _audit_geometry(
        geometry,
        certification_mode,
        change_set=change_set,
        policy=audit_policy,
    )
    if structured_report is not None:
        working_backend_diagnostics = triangulation_backend_by_face
        remap_prepared_mesh_associations(
            mesh,
            source_geometry,
            geometry,
            source_to_working_faces=structured_report.source_to_working_faces,
            source_to_working_edges=structured_report.source_to_working_edges,
        )
        triangulation_backend_by_face = _source_backend_diagnostics(
            structured_report, working_backend_diagnostics
        )
        boundary_registry = _published_boundary_registry(source_geometry, mesh)
        quality = _structured_quality_report(
            mesh,
            structured_report.plan.options,
        )
        mapped_elements = {
            int(element_id)
            for face_id in mapped_source_faces
            for element_id in mesh.elements_of_face.get(face_id, ())
        }
        metrics = regularity_metrics(
            mesh,
            target_size=target_size,
            minimum_size_ratio=(
                structured_report.plan.options.minimum_size_ratio
            ),
            maximum_size_ratio=(
                structured_report.plan.options.maximum_size_ratio
            ),
            mapped_element_ids=mapped_elements,
        )
        structured_report = replace(
            structured_report,
            quality=quality,
            metrics=metrics,
        )
        if not quality["accepted"]:
            message = _quality_rejection_message(quality)
            if strategy is MeshingStrategy.MAPPED:
                raise MeshError(
                    f"explicit mapped strategy rejected: {message}. "
                    "Relax the documented quality policy only after reviewing "
                    "the reported stable element IDs."
                )
            _check_cancellation(
                cancellation_check,
                "structured quality rejected; native fallback start",
            )
            fallback = generate_hybrid_mesh_result(
                source_geometry,
                target_size=target_size,
                strategy=MeshingStrategy.AUTO,
                overrides=overrides,
                beam_edges=requested_beam_edges,
                beam_offsets=beam_offsets,
                member_ids=requested_member_ids,
                face_ids=requested_face_ids,
                seeding=requested_seeding,
                refinements=refinements,
                order=order,
                recombine=recombine,
                native_backend=native_backend,
                structured_options=None,
                overlap_policy=overlap_policy,
                mutation_policy=mutation_policy,
                certification_mode=certification_mode,
                change_set=change_set,
                audit_policy=audit_policy,
                cancellation_check=cancellation_check,
            )
            fallback_quality = _structured_quality_report(
                fallback.mesh,
                structured_report.plan.options,
            )
            if not fallback_quality["accepted"]:
                fallback_message = _quality_rejection_message(fallback_quality)
                raise MeshError(
                    f"{message}; automatic native fallback also rejected: "
                    f"{fallback_message}"
                )
            fallback_metrics = regularity_metrics(
                fallback.mesh,
                target_size=target_size,
                minimum_size_ratio=(
                    structured_report.plan.options.minimum_size_ratio
                ),
                maximum_size_ratio=(
                    structured_report.plan.options.maximum_size_ratio
                ),
                mapped_element_ids=(),
            )
            structured_report = replace(
                structured_report,
                diagnostics=(*structured_report.diagnostics, message),
                metrics={
                    "rejected_candidate": dict(metrics),
                    "accepted_fallback": fallback_metrics,
                },
                quality={
                    "accepted": True,
                    "selected_mesh": "native_fallback",
                    "rejected_candidate": quality,
                    "accepted_fallback": fallback_quality,
                },
                status="rejected_fallback",
            )
            fallback.mesh.structural_preparation = structured_report.to_dict()
            fallback.mesh.hybrid_diagnostics.update(
                {
                    "structured_layout_status": structured_report.status,
                    "structured_plan_hash": structured_report.plan.plan_hash,
                    "structured_quality": structured_report.to_dict()["quality"],
                }
            )
            _check_cancellation(
                cancellation_check,
                "structured quality fallback accepted",
            )
            return replace(fallback, structured_layout=structured_report)
        mesh.structural_preparation = structured_report.to_dict()
    strategies = dict(source_strategies)
    result = HybridMeshResult(
        mesh=mesh,
        strategy_by_face=strategies,
        triangulation_backend_by_face=triangulation_backend_by_face,
        preflight=preflight,
        connectivity=connectivity,
        audit_report=audit_report,
        certification_mode=certification_mode,
        certifiable=certifiable,
        structured_layout=structured_report,
    )
    mesh.hybrid_diagnostics = {
        "strategy_by_face": dict(strategies),
        "triangulation_backend_by_face": {
            int(face_id): dict(values)
            for face_id, values in triangulation_backend_by_face.items()
        },
        "geometry_model_id": str(source_geometry.model_id),
        "geometry_revision": int(source_geometry.revision),
        "certification_mode": certification_mode.value,
        "certifiable": bool(certifiable),
        "preflight_count": len(preflight),
        "structured_layout_status": (
            None if structured_report is None else structured_report.status
        ),
        "structured_plan_hash": (
            None if structured_report is None else structured_report.plan.plan_hash
        ),
        "structured_quality": (
            None if structured_report is None else structured_report.to_dict()["quality"]
        ),
        "phase_seconds": {
            **phase_seconds,
            "total_generation": perf_counter() - generation_started,
        },
    }
    mesh.boundary_registry = boundary_registry
    _check_cancellation(cancellation_check, "hybrid generation complete")
    return result


def generate_hybrid_mesh(geometry: GeometryModel, **options: Any) -> Mesh:
    """Return only the neutral mesh from :func:`generate_hybrid_mesh_result`."""

    return generate_hybrid_mesh_result(geometry, **options).mesh
