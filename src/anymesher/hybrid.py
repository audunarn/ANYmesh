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
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from anygeometry.curves import Straight
from anygeometry.entities import EntityRef, OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.surfaces import CoonsSurface, Cone, Cylinder, Plane, RuledSurface
try:
    from anygeometry.surfaces import ExtrudedSurface
except ImportError:  # Older supported ANYgeometry builds predate schema 6.
    ExtrudedSurface = ()

from .boundary import GlobalEdgeBoundaryRegistry, MemberRegistry
from .core import MeshCore
from .errors import MeshError, StructuredQualityRejected
from ._mapped_fold import mapped_face_folds
from .mapped import (
    ELEMENT_ORDERS,
    _refuse_curved_beams,
    generate_mesh as generate_mapped_mesh,
)
from .mesh import Mesh
from .meshing_view import GeometryMeshingView
from .prepared import remap_prepared_mesh_associations
from .preparation import (
    StructuralPreparationOptions,
    StructuralPreparationReport,
    prepare_structural_closure,
)
from .quality_v2 import assert_valid_mesh, evaluate_quality
from .refinement import Refinement, SizeField
from .metric import (
    ExperimentalMetricProvider,
    IsotropicMetricControl,
    MetricFieldSpec,
)
from .native_v2 import ComponentSeedRegistry, NativeMeshingOptions
from ._component_reservations import ComponentNodeReservationPool
from .quad.boundary import BoundaryStationKey, BoundaryStationRegistry
from .quad.timing import record_quad_stage, timed_quad_call
from .quad.domain import ConicalQuadDomain, CylindricalQuadDomain, ParametricQuadDomain, PlanarQuadDomain
from .quad.driver import run_planar_quad_driver
from .quad.high_order import (
    HighOrderBoundaryMidside,
    HighOrderGeometryReport,
    HighOrderMeshCertificate,
    ValidityStatus,
    certify_mapping_validity,
    evaluate_mapping,
)
from .quad.options import QuadMeshingOptions
from .quad.mcf_seed import optimize_q4_seed_mcf
from .quad.optimize import optimize_quad_state
from .quad.residual import improve_residual_triangles
from .quad.seed import build_planar_quad_seed, source_chart_axis_lengths
from .quad.public_integration import (
    QuadPublicUnsupported,
    publish_atomically,
    route_quad_first,
)
from .quad.state import QuadMeshState
from .quad.validate import QuadQualityRejected, validate_planar_quad_result
from .s3_production import prepare_qualified_s3_mesh
from .s3_repair import S3RepairError
from .seeding import Seeding, edge_distribution, solve_seeding
from .serialize import mesh_from_dict, mesh_to_dict
from .structural_pipeline import (
    GeometryMutationPolicy,
    OverlapPolicy,
    PreflightStatus,
    StructuralMeshingPipeline,
)
from .surface_mesh import SurfaceMeshOptions, mesh_planar_surface
from .structured import (
    MeshQualityPolicy,
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
    structural_preparation: StructuralPreparationReport | None = None


@dataclass(frozen=True)
class _LoopBoundary:
    node_ids: tuple[int, ...]
    midside_ids: tuple[int, ...]
    uv: np.ndarray
    segment_specs: tuple[tuple[int, float, float] | None, ...]


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


def _active_structural_owners(
    view: GeometryMeshingView,
    face_ids: Iterable[int],
    edge_ids: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return exact structural owners seeded by the requested topology."""

    sheets = {
        int(sheet_id)
        for face_id in face_ids
        for sheet_id in view.sheets_for_face(int(face_id))
    }
    members = {
        int(member_id)
        for edge_id in edge_ids
        for member_id in view.members_using_edge(int(edge_id))
    }
    return tuple(sorted(sheets)), tuple(sorted(members))


def _mappable(geometry: GeometryModel, face_id: int) -> bool:
    """Automatic strategy: a four-sided face whose transfinite map does not fold.

    An L-shaped plate can declare four corners with its re-entrant corner
    inside one mapped side; blending those sides folds the grid over and would
    publish elements outside the face, so such faces go to native meshing.
    """
    face = geometry.faces[face_id]
    return (
        len(face.corners) == 4
        and not face.holes
        and not mapped_face_folds(geometry, face_id)
    )


def _refuse_folded_topology_surface(geometry: GeometryModel, face_id: int) -> None:
    """Fail clearly when a face's only surface is a folded topology blend.

    A face created without a surface gets a topology-derived Coons surface from
    its four declared sides.  When that blend folds (an L-shaped outline, for
    example) the owner geometry itself overlaps, so neither the mapped nor the
    native chart can represent the face; say so instead of reporting a
    self-intersecting chart loop.
    """
    face = geometry.faces[face_id]
    surface = face.surface
    if (
        getattr(face, "parameterization", None) is None
        and isinstance(surface, CoonsSurface)
        and not surface.has_boundaries
        and mapped_face_folds(geometry, face_id)
    ):
        raise MeshError(
            f"face {face_id} has no surface of its own, and the Coons surface "
            "derived from its four declared sides folds over (a re-entrant "
            "corner inside one side, as on an L-shaped outline). Attach a Plane "
            "surface (for example GeometryModel.add_plate) or split the face "
            "into four-sided patches."
        )


def _native_chart_uv(geometry: GeometryModel, face: Any, face_id: int, point: Any) -> tuple[float, float]:
    """Chart coordinates of a boundary point for native planar meshing.

    ``face_local_uv`` clips to the unit patch, which for a Plane support
    spanned by four declared corners collapses every boundary point outside
    that quadrilateral (e.g. the second arm of an L-shaped plate).  A Plane's
    own affine coordinates are exact and unbounded, and ``face_point`` lifts
    them back without clipping; inside the unit patch they are identical.
    """
    if getattr(face, "parameterization", None) is None and isinstance(face.surface, Plane):
        u, v = face.surface.local_uv(point)
        return float(u), float(v)
    return geometry.face_local_uv(face_id, point)


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
        if isinstance(geometry.edges[edge_id].curve, Straight):
            # Existing straight-edge nodes can include canonical B3 chord
            # midpoints. Register their actual owner parameters rather than
            # regrading the doubled station count independently.
            edge = geometry.edges[edge_id]
            start = np.asarray(geometry.vertex_position(edge.start), dtype=float)
            chord = np.asarray(geometry.vertex_position(edge.end), dtype=float) - start
            denominator = float(chord @ chord)
            if denominator > 0.0:
                parameters = np.asarray([
                    float((mesh.nodes[node] - start) @ chord) / denominator
                    for node in sequence
                ])
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
    source_to_working_faces: Mapping[int, Sequence[int]],
    source_strategies: Mapping[int, str],
    working_diagnostics: Mapping[int, Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for source_face, descendants in sorted(
        source_to_working_faces.items()
    ):
        records = [
            dict(working_diagnostics[face_id])
            for face_id in descendants
            if face_id in working_diagnostics
        ]
        # A material region executes once while retaining every consumed face
        # diagnostic. Do not multiply its timing when publishing the authored face.
        region_records = set()
        unique_records = []
        for record in records:
            region = record.get('material_region')
            if region is not None:
                key = tuple(region['source_faces'])
                if key in region_records:
                    continue
                region_records.add(key)
            unique_records.append(record)
        records = unique_records
        if len(records) == 1:
            result[source_face] = {
                **records[0],
                "structured_action": source_strategies[source_face],
                "working_face_ids": list(descendants),
            }
            continue
        selected = {str(item.get("actual_backend", "unknown")) for item in records}
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
            "structured_action": source_strategies[source_face],
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


def _declared_junction_core_edges(
    mesh: Mesh, core: MeshCore
) -> tuple[tuple[int, int], ...]:
    node_rows = {
        int(node_id): row for row, node_id in enumerate(core.node_ids)
    }
    result: set[tuple[int, int]] = set()
    for first, second in mesh.declared_plate_junction_edges:
        try:
            edge = (node_rows[int(first)], node_rows[int(second)])
        except KeyError as error:
            raise MeshError(
                "declared plate-junction edge references a missing mesh node"
            ) from error
        result.add((min(edge), max(edge)))
    return tuple(sorted(result))


def _prepared_plate_junction_edges(
    mesh: Mesh,
    report: StructuralPreparationReport,
    edge_descendants: Mapping[int, Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    step = 2 if mesh.is_quadratic else 1
    result: set[tuple[int, int]] = set()
    for prepared_edge_id in report.declared_face_connection_edges:
        descendants = edge_descendants.get(int(prepared_edge_id))
        if not descendants:
            raise MeshError(
                "declared plate-junction edge "
                f"{prepared_edge_id} has no final structured descendant"
            )
        for edge_id in descendants:
            sequence = mesh.nodes_of_edge.get(int(edge_id))
            if sequence is None or len(sequence) < step + 1:
                raise MeshError(
                    f"declared plate-junction edge {edge_id} has no seeded mesh boundary"
                )
            if (len(sequence) - 1) % step:
                raise MeshError(
                    f"declared plate-junction edge {edge_id} has inconsistent order"
                )
            for index in range(0, len(sequence) - 1, step):
                first = int(sequence[index])
                second = int(sequence[index + step])
                result.add((min(first, second), max(first, second)))
    return tuple(sorted(result))


def _plate_junction_owner_state(
    geometry: GeometryModel, edge_id: int
) -> tuple[tuple[int, ...], bool]:
    """Return exact Sheet owners and explicit non-manifold authority.

    Two distinct Sheets sharing one geometry edge form an ordinary structural
    junction.  Three or more Sheet owners are genuinely non-manifold and must
    be explicitly declared on at least one owning Sheet; otherwise generation
    fails closed instead of silently converting accidental topology into an
    accepted junction.
    """

    owners = tuple(int(item) for item in geometry.sheets_using_edge(int(edge_id)))
    declared = any(
        int(edge_id) in geometry.sheets[sheet_id].declared_non_manifold_edges
        for sheet_id in owners
    )
    from anygeometry import query_joint_edge
    declared=declared or query_joint_edge(getattr(geometry,"source",geometry),edge_id).declared
    if len(owners) >= 3 and not declared:
        raise MeshError(
            f"edge {int(edge_id)} is an undeclared non-manifold junction shared "
            f"by {len(owners)} structural Sheets"
        )
    return owners, declared


def _topology_plate_junction_edges(
    mesh: Mesh,
    geometry: GeometryModel,
) -> tuple[tuple[int, int], ...]:
    """Return seeded mesh edges shared by distinct structural Sheets.

    Ordinary internal face boundaries in one Sheet (for example the facets of
    a cylinder) are manifold shell edges, not structural junctions.  Treating
    them as junctions makes the published connection set both misleading and
    far larger than the actual plate-to-plate interface.
    """

    step = 2 if mesh.is_quadratic else 1
    result: set[tuple[int, int]] = set()
    for edge_id in sorted(geometry.edges):
        sheet_ids, declared_non_manifold = _plate_junction_owner_state(
            geometry, int(edge_id)
        )
        if len(sheet_ids) < 2 and not declared_non_manifold:
            continue
        sequence = mesh.nodes_of_edge.get(int(edge_id))
        if sequence is None or len(sequence) < step + 1:
            continue
        if (len(sequence) - 1) % step:
            raise MeshError(
                f"topology-owned plate-junction edge {edge_id} has inconsistent order"
            )
        for index in range(0, len(sequence) - 1, step):
            first = int(sequence[index])
            second = int(sequence[index + step])
            result.add((min(first, second), max(first, second)))
    return tuple(sorted(result))


def _geometry_plate_junction_edge_ids(
    geometry: GeometryModel,
) -> tuple[int, ...]:
    """Return exact geometry edges carrying structural plate junctions.

    A caller may provide a closure that has already been imprinted by the
    public structural-preparation workflow.  Such a call intentionally has no
    new ``StructuralPreparationReport``; the committed Sheet ownership is then
    the junction authority.  Recovering these exact edge IDs keeps planning
    and native boundary alignment identical to a closure prepared inside this
    function, without proximity matching or hidden welding.
    """

    result: list[int] = []
    for edge_id in sorted(geometry.edges):
        sheet_ids, declared_non_manifold = _plate_junction_owner_state(
            geometry, int(edge_id)
        )
        if len(sheet_ids) >= 2 or declared_non_manifold:
            result.append(int(edge_id))
    return tuple(result)


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
    core = _neutral_shell_core(mesh)
    quality = evaluate_quality(
        core,
        declared_plate_junction_edges=_declared_junction_core_edges(mesh, core),
    )
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
    # Independently optimized conforming charts can differ by a sub-percent
    # amount at their shared boundary after their local quality passes.  Keep
    # the configured limit authoritative while allowing only that numerical
    # chart-merge tolerance; materially graded transitions remain rejected.
    growth_relative_tolerance = 0.01
    effective_growth_limit = options.max_element_growth * (
        1.0 + growth_relative_tolerance
    )
    growth, growth_pairs = _element_growth(
        mesh,
        limit=effective_growth_limit,
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
        "growth_relative_tolerance": growth_relative_tolerance,
        "effective_growth_limit": effective_growth_limit,
        "growth_violation_count": growth_violations,
        "growth_violation_pairs": [
            [first, second, ratio]
            for first, second, ratio in growth_pairs[:16]
        ],
    }


def _junction_growth_repair(
    geometry: GeometryModel,
    mesh: Mesh,
    quality: Mapping[str, Any],
    options: StructuredMeshingOptions,
) -> tuple[Mesh | None, Mapping[str, Any], Mapping[str, Any]]:
    """Repair marginal growth across declared plate junctions without topology edits."""

    declared = {
        tuple(sorted((int(first), int(second))))
        for first, second in mesh.declared_plate_junction_edges
    }
    protected_nodes = {
        int(node_id)
        for sequence in mesh.nodes_of_edge.values()
        for node_id in sequence
    }
    node_faces: dict[int, set[int]] = {}
    for face_id, element_ids in mesh.elements_of_face.items():
        for element_id in element_ids:
            for node_id in mesh.corners_of(int(element_id)):
                node_faces.setdefault(int(node_id), set()).add(int(face_id))

    targets: dict[int, list[np.ndarray]] = {}
    target_faces: dict[int, int] = {}
    junctions: set[tuple[int, int]] = set()
    for violation in quality.get("growth_violation_pairs", ()):
        first_element, second_element = map(int, violation[:2])
        first_corners = mesh.corners_of(first_element)
        second_corners = mesh.corners_of(second_element)
        shared = set(first_corners) & set(second_corners)
        if len(shared) != 2:
            continue
        junction = tuple(sorted(int(value) for value in shared))
        if junction not in declared:
            continue
        characteristic: dict[int, float] = {}
        for element_id, corners in (
            (first_element, first_corners),
            (second_element, second_corners),
        ):
            characteristic[element_id] = float(
                np.mean(
                    [
                        np.linalg.norm(mesh.nodes[second] - mesh.nodes[first])
                        for first, second in zip(corners, corners[1:] + corners[:1])
                    ]
                )
            )
        larger = max(
            (first_element, second_element),
            key=lambda element_id: (characteristic[element_id], element_id),
        )
        midpoint = 0.5 * (mesh.nodes[junction[0]] + mesh.nodes[junction[1]])
        movable: list[int] = []
        for node_id in mesh.corners_of(larger):
            owner_faces = node_faces.get(int(node_id), set())
            if (
                node_id in shared
                or int(node_id) in protected_nodes
                or len(owner_faces) != 1
            ):
                continue
            owner_face = next(iter(owner_faces))
            if owner_face not in geometry.faces or not isinstance(
                geometry.faces[owner_face].surface, Plane
            ):
                continue
            movable.append(int(node_id))
            target_faces[int(node_id)] = owner_face
        if not movable:
            continue
        junctions.add(junction)
        for node_id in movable:
            targets.setdefault(node_id, []).append(np.asarray(midpoint, dtype=float))

    initial = {
        "growth_violation_count": int(quality.get("growth_violation_count", 0)),
        "maximum_adjacent_element_growth": float(
            quality.get("maximum_adjacent_element_growth", 1.0)
        ),
        "maximum_aspect_ratio": float(quality.get("maximum_aspect_ratio", 1.0)),
    }
    if not targets:
        return None, quality, {
            "attempted": False,
            "committed": False,
            "junction_node_pairs": [],
            "face_ids": [],
            "moved_node_ids": [],
            "relaxation": 0.0,
            "initial_quality": initial,
            "final_quality": initial,
        }

    for relaxation in (0.005, 0.01, 0.02, 0.04, 0.08):
        candidate = mesh_from_dict(mesh_to_dict(mesh))
        candidate.hybrid_diagnostics = dict(mesh.hybrid_diagnostics)
        for node_id in sorted(targets):
            target = np.mean(np.vstack(targets[node_id]), axis=0)
            candidate.nodes[node_id] = (
                mesh.nodes[node_id]
                + relaxation * (target - mesh.nodes[node_id])
            )
        try:
            candidate_core = _neutral_shell_core(candidate)
            assert_valid_mesh(
                candidate_core,
                declared_plate_junction_edges=_declared_junction_core_edges(
                    candidate, candidate_core
                ),
            )
        except MeshError:
            continue
        candidate_quality = _structured_quality_report(candidate, options)
        if not candidate_quality["accepted"]:
            continue
        final = {
            "growth_violation_count": int(
                candidate_quality.get("growth_violation_count", 0)
            ),
            "maximum_adjacent_element_growth": float(
                candidate_quality.get("maximum_adjacent_element_growth", 1.0)
            ),
            "maximum_aspect_ratio": float(
                candidate_quality.get("maximum_aspect_ratio", 1.0)
            ),
        }
        return candidate, candidate_quality, {
            "attempted": True,
            "committed": True,
            "junction_node_pairs": [list(pair) for pair in sorted(junctions)],
            "face_ids": sorted(set(target_faces.values())),
            "moved_node_ids": sorted(targets),
            "relaxation": relaxation,
            "initial_quality": initial,
            "final_quality": final,
        }
    return None, quality, {
        "attempted": True,
        "committed": False,
        "junction_node_pairs": [list(pair) for pair in sorted(junctions)],
        "face_ids": sorted(set(target_faces.values())),
        "moved_node_ids": [],
        "relaxation": 0.0,
        "initial_quality": initial,
        "final_quality": initial,
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


def _quality_rejection_details(
    aligned: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    aligned_counts = {
        str(name): int(value)
        for name, value in aligned.get("violation_counts", {}).items()
    }
    baseline_counts = {
        str(name): int(value)
        for name, value in baseline.get("violation_counts", {}).items()
    }
    rejection_metrics = sorted(
        name for name, count in aligned_counts.items() if count > 0
    )
    aligned_growth = int(aligned.get("growth_violation_count", 0))
    if aligned_growth:
        rejection_metrics.append("element_growth")
    return {
        "rejection_metrics": rejection_metrics,
        "aligned_violation_counts": aligned_counts,
        "baseline_violation_counts": baseline_counts,
        "aligned_growth_violation_count": aligned_growth,
        "baseline_growth_violation_count": int(
            baseline.get("growth_violation_count", 0)
        ),
        "aligned_maximum_adjacent_element_growth": float(
            aligned.get("maximum_adjacent_element_growth", 1.0)
        ),
        "baseline_maximum_adjacent_element_growth": float(
            baseline.get("maximum_adjacent_element_growth", 1.0)
        ),
    }


def _compose_descendants(
    first: Mapping[int, Sequence[int]],
    second: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    return {
        int(source): tuple(
            dict.fromkeys(
                final
                for intermediate in intermediates
                for final in second.get(int(intermediate), (int(intermediate),))
            )
        )
        for source, intermediates in first.items()
    }


def _remap_edge_divisions(
    source: GeometryModel,
    working: GeometryModel,
    source_to_working_edges: Mapping[int, Sequence[int]],
    overrides: Mapping[int, int] | None,
) -> dict[int, int] | None:
    if overrides is None:
        return None
    remapped: dict[int, int] = {}
    for source_edge, raw_requested in sorted(overrides.items()):
        source_edge = int(source_edge)
        if source_edge not in source.edges:
            raise MeshError(f"edge override references missing source edge {source_edge}")
        requested = int(raw_requested)
        if isinstance(raw_requested, bool) or requested < 1:
            raise MeshError(f"edge {source_edge} override must be a positive integer")
        descendants = tuple(
            int(item)
            for item in source_to_working_edges.get(source_edge, (source_edge,))
        )
        if len(descendants) == 1:
            remapped[descendants[0]] = requested
            continue
        if requested < len(descendants):
            raise MeshError(
                f"edge {source_edge} override requests {requested} total divisions "
                f"but immutable preparation created {len(descendants)} non-empty "
                "descendants; increase the explicit total"
            )
        lengths = np.asarray(
            [working.edge_length(item) for item in descendants],
            dtype=float,
        )
        total_length = float(lengths.sum())
        if total_length <= 0.0:
            raise MeshError(f"source edge {source_edge} has zero-length descendants")
        remaining = requested - len(descendants)
        raw = lengths / total_length * remaining
        additions = np.floor(raw).astype(int)
        counts = np.ones(len(descendants), dtype=int) + additions
        remainder = requested - int(counts.sum())
        if remainder > 0:
            order = sorted(
                range(len(descendants)),
                key=lambda index: (
                    -(raw[index] - additions[index]),
                    descendants[index],
                ),
            )
            for index in order[:remainder]:
                counts[index] += 1
        for descendant, count in zip(descendants, counts):
            previous = remapped.setdefault(descendant, int(count))
            if previous != int(count):
                raise MeshError(
                    f"working edge {descendant} receives conflicting exact overrides"
                )
    return remapped


def _remap_edge_values(
    source: GeometryModel,
    source_to_working_edges: Mapping[int, Sequence[int]],
    values: Mapping[int, float | Sequence[float]] | None,
    *,
    label: str,
) -> dict[int, float | tuple[float, float, float]] | None:
    if values is None:
        return None
    remapped: dict[int, float | tuple[float, float, float]] = {}
    for raw_source_edge, raw_value in sorted(values.items()):
        source_edge = int(raw_source_edge)
        if source_edge not in source.edges:
            raise MeshError(f"{label} references missing source edge {source_edge}")
        array = np.asarray(raw_value, dtype=float).reshape(-1)
        if len(array) not in (1, 3) or not np.all(np.isfinite(array)):
            raise MeshError(f"{label} for edge {source_edge} must be finite")
        value: float | tuple[float, float, float] = (
            float(array[0])
            if len(array) == 1
            else tuple(float(item) for item in array)
        )
        for descendant in source_to_working_edges.get(source_edge, (source_edge,)):
            descendant = int(descendant)
            previous = remapped.setdefault(descendant, value)
            if previous != value:
                raise MeshError(
                    f"working edge {descendant} receives conflicting exact {label}"
                )
    return remapped


def _remap_refinements(
    working: GeometryModel,
    source_to_working_faces: Mapping[int, Sequence[int]],
    source_to_working_edges: Mapping[int, Sequence[int]],
    refinements: Sequence[Refinement],
) -> tuple[Refinement, ...]:
    made: list[Refinement] = []
    for refinement in refinements:
        reference = refinement.ref
        if reference is None:
            made.append(refinement)
            continue
        mapping = (
            source_to_working_faces
            if reference.kind == "face"
            else source_to_working_edges
            if reference.kind == "edge"
            else None
        )
        descendants = (
            tuple(mapping.get(reference.id, (reference.id,)))
            if mapping is not None
            else (reference.id,)
        )
        container = {
            "vertex": working.vertices,
            "edge": working.edges,
            "face": working.faces,
        }.get(reference.kind)
        if container is None or any(item not in container for item in descendants):
            raise MeshError(
                f"refinement {refinement.name!r} has unresolved {reference.kind} "
                f"target {reference.id}"
            )
        made.extend(
            replace(refinement, ref=EntityRef(reference.kind, int(item)))
            for item in descendants
        )
    return tuple(made)


def _preparation_payload(
    preparation: StructuralPreparationReport | None,
    structured: StructuredLayoutReport | None,
    source_to_working_faces: Mapping[int, Sequence[int]],
    source_to_working_edges: Mapping[int, Sequence[int]],
    *,
    source_model_id: str,
) -> dict[str, Any]:
    structured_payload = None if structured is None else structured.to_dict()
    if structured_payload is not None:
        structured_payload["plan"]["model_id"] = str(source_model_id)
    return {
        "status": (
            structured.status
            if structured is not None
            else (preparation.status if preparation is not None else "not_required")
        ),
        "structural_closure": (
            None if preparation is None else preparation.to_dict()
        ),
        "structured_layout": structured_payload,
        "source_to_working_faces": {
            str(key): list(values)
            for key, values in sorted(source_to_working_faces.items())
        },
        "source_to_working_edges": {
            str(key): list(values)
            for key, values in sorted(source_to_working_edges.items())
        },
    }


def _stable_diagnostic_record(value: Any) -> Any:
    """Remove wall-clock samples from persisted reproducibility records."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_diagnostic_record(item)
            for key, item in value.items()
            if str(key) != "phase_seconds"
        }
    if isinstance(value, (list, tuple)):
        return [_stable_diagnostic_record(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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
        specs = tuple(
            boundary.segment_specs[(count - 2 - index) % count]
            for index in range(count)
        )
        return _LoopBoundary(nodes, (), uv, specs)
    mids = tuple(
        boundary.midside_ids[(count - 2 - index) % count]
        for index in range(count)
    )
    specs = tuple(
        boundary.segment_specs[(count - 2 - index) % count]
        for index in range(count)
    )
    return _LoopBoundary(nodes, mids, uv, specs)


def _loop_boundary(
    geometry: GeometryModel,
    mesh: Mesh,
    face_id: int,
    loop: Sequence[OrientedEdge],
    *,
    quadratic: bool,
    counter_clockwise: bool,
    boundary_registry: GlobalEdgeBoundaryRegistry,
    automatically_seeded_shared_edges: frozenset[int],
) -> _LoopBoundary:
    corner_nodes: list[int] = []
    midside_nodes: list[int] = []
    segment_specs: list[tuple[int, float, float] | None] = []
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
        if int(oriented.edge) in automatically_seeded_shared_edges:
            parameter_by_node = {
                int(entry.node_id): float(entry.key.parameter)
                for entry in boundary_registry.entries(int(oriented.edge))
                if entry.node_id is not None
            }
            for first_node, second_node in zip(edge_corners[:-1], edge_corners[1:]):
                first_parameter = parameter_by_node[int(first_node)]
                second_parameter = parameter_by_node[int(second_node)]
                segment_specs.append(
                    (
                        int(oriented.edge),
                        min(first_parameter, second_parameter),
                        max(first_parameter, second_parameter),
                    )
                )
        else:
            segment_specs.extend(None for _ in range(len(edge_corners) - 1))
    if len(corner_nodes) < 3:
        raise MeshError(f"face {face_id} has fewer than three boundary stations")
    _refuse_folded_topology_surface(geometry, face_id)
    if quadratic and len(midside_nodes) != len(corner_nodes):
        raise MeshError(f"face {face_id} has an inconsistent quadratic boundary")
    try:
        uv = np.asarray(
            [
                _native_chart_uv(geometry, geometry.faces[face_id], face_id, mesh.nodes[node_id])
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
    boundary = _LoopBoundary(
        tuple(corner_nodes), tuple(midside_nodes), uv, tuple(segment_specs)
    )
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


def _publish_native_face_elements(mesh, face_id, core, core_to_global):
    """Publish qualified local connectivity through the existing source-ID map."""
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


def _mesh_native_face(
    geometry: GeometryModel,
    mesh: Mesh,
    face_id: int,
    *,
    order: str,
    recombine: bool,
    native_backend: Any,
    native_options: NativeMeshingOptions,
    size_field: SizeField,
    metric_model_uuid: str,
    metric_geometry_revision: int,
    boundary_registry: GlobalEdgeBoundaryRegistry,
    automatically_seeded_shared_edges: frozenset[int],
    component_seed_registry: ComponentSeedRegistry,
    quality_options: StructuredMeshingOptions | None,
    declared_junction_edges: Iterable[int],
    evaluate_declared_junction_alignment: bool,
    refine_declared_junction_transition: bool,
    cancellation_check: Callable[[str], None] | None,
    _cylindrical_binding: Any = None,
    _material_region_binding: Any = None,
    _material_input_eligibility: Any = None,
) -> dict[str, Any]:
    if getattr(component_seed_registry, "_deferred_cylindrical_components", None):
        from ._cylindrical_recombine import assert_face_open

        assert_face_open(component_seed_registry, face_id)
    boundary_started = perf_counter()
    _check_cancellation(cancellation_check, f"native face {face_id} boundary start")
    face = geometry.faces[face_id]
    quadratic = order == "quadratic"
    if _material_region_binding is not None:
        _material_region_binding.validate(cancellation_check)
        region_loops = tuple(
            _LoopBoundary(nodes, (), uv, specifications)
            for nodes, uv, specifications in _material_region_binding.loops(
                mesh, boundary_registry, automatically_seeded_shared_edges))
        if not region_loops:
            raise MeshError('material region has no outer boundary')
        normalized = []
        for index, loop in enumerate(region_loops):
            area = _signed_area(loop.uv)
            if not np.isfinite(area) or abs(area) <= 128 * np.finfo(float).eps * max(float(np.max(np.abs(loop.uv))), 1.)**2:
                raise MeshError('material region has a degenerate surface chart')
            normalized.append(_reverse_loop(loop) if (area > 0.) != (index == 0) else loop)
        outer, holes = normalized[0], tuple(normalized[1:])
    else:
        outer = _loop_boundary(
            geometry,
            mesh,
            face_id,
            face.loop,
            quadratic=quadratic,
            counter_clockwise=True,
            boundary_registry=boundary_registry,
            automatically_seeded_shared_edges=automatically_seeded_shared_edges,
        )
        holes = tuple(
            _loop_boundary(
                geometry,
                mesh,
                face_id,
                loop,
                quadratic=quadratic,
                counter_clockwise=False,
                boundary_registry=boundary_registry,
                automatically_seeded_shared_edges=automatically_seeded_shared_edges,
            )
            for loop in face.holes
        )
    loops = (outer, *holes)
    cylindrical_chart = None
    if isinstance(face.surface, Cylinder) and (
            native_options.point_placement == "frontal_delaunay" or _cylindrical_binding is not None):
        # The public orchestrator supplies revision-bound geometry-owner evidence.
        from ._cylindrical_atlas import CylindricalAtlasBinding
        from ._cylindrical_patch import CylindricalPatchBinding
        from ._trimmed_cylinder_binding import TrimmedCylinderBinding

        if not isinstance(_cylindrical_binding, (CylindricalAtlasBinding, CylindricalPatchBinding, TrimmedCylinderBinding)):
            raise MeshError("cylindrical frontal_delaunay requires an accepted owner binding")
        if (
            _cylindrical_binding.model_id != geometry.model_id
            or _cylindrical_binding.revision != geometry.revision
        ):
            raise MeshError("cylindrical atlas does not bind the working geometry")
        uses = tuple(
            sector.face_use for sector in _cylindrical_binding.face_records
            if sector.face.id == face_id
        )
        if len(uses) != 1:
            raise MeshError("cylindrical native face requires one qualified FaceUse")
        cylindrical_chart = _cylindrical_binding.chart_for(
            uses[0], cancellation_check=cancellation_check
        )
    analytic_chart = None
    if (native_options.point_placement == "frontal_delaunay" and not quadratic
            and (isinstance(face.surface, Cone) or isinstance(face.surface, ExtrudedSurface))):
        from ._analytic_metric_chart import AnalyticMetricChart
        analytic_chart = AnalyticMetricChart(geometry, face_id, cancellation_check,
                                           region_binding=_material_region_binding)
    chart_transform = (np.eye(2, dtype=float) if analytic_chart is None
                       else analytic_chart.transform)
    if cylindrical_chart is not None:
        chart_transform = np.diag((
            cylindrical_chart.circumferential_length,
            cylindrical_chart.axial_length,
        ))
    if isinstance(face.surface, Plane) and len(geometry.faces) > 1:
        u_vector = np.asarray(face.surface.u_vector, dtype=float)
        v_vector = np.asarray(face.surface.v_vector, dtype=float)
        metric = np.asarray(
            (
                (float(np.dot(u_vector, u_vector)), float(np.dot(u_vector, v_vector))),
                (float(np.dot(v_vector, u_vector)), float(np.dot(v_vector, v_vector))),
            ),
            dtype=float,
        )
        chart_transform = np.linalg.cholesky(metric)
    chart_inverse = np.linalg.inv(chart_transform)
    physical_jacobian = np.column_stack((
        np.asarray(face.surface.u_vector, dtype=np.float64),
        np.asarray(face.surface.v_vector, dtype=np.float64),
    )) @ chart_inverse.T if isinstance(face.surface, Plane) else None
    metric_to_physical = None
    if analytic_chart is not None:
        physical_jacobian = analytic_chart.jacobians
        metric_to_physical = analytic_chart.evaluate
    if cylindrical_chart is not None:
        physical_jacobian = cylindrical_chart.jacobians
        metric_to_physical = cylindrical_chart.evaluate
    if isinstance(face.surface, Plane):
        metric_origin = np.asarray(face.surface.origin, dtype=np.float64)
        metric_u = np.asarray(face.surface.u_vector, dtype=np.float64)
        metric_v = np.asarray(face.surface.v_vector, dtype=np.float64)

        def metric_to_physical(chart_points: np.ndarray) -> np.ndarray:
            parameter_points = np.asarray(chart_points, dtype=np.float64) @ chart_inverse
            return (
                metric_origin[None, :]
                + parameter_points[:, 0, None] * metric_u[None, :]
                + parameter_points[:, 1, None] * metric_v[None, :]
            )
    chart_outer = np.asarray(outer.uv, dtype=float) @ chart_transform
    chart_holes = tuple(
        np.asarray(loop.uv, dtype=float) @ chart_transform for loop in holes
    )
    chart_loops = (chart_outer, *chart_holes)
    region_constraints, region_pinned_ids, region_pinned_uv = (), (), np.empty((0, 2))
    if _material_region_binding is not None:
        region_constraints, region_pinned_ids, region_pinned_uv = _material_region_binding.interior(
            mesh, boundary_registry,
            boundary_rows={node: row for loop in loops for node, row in zip(loop.node_ids, loop.uv)})
    chart_constraints = tuple(segment @ chart_transform for segment in region_constraints)
    chart_pinned = region_pinned_uv @ chart_transform
    automatically_seeded_shared_segments: dict[
        tuple[int, int], tuple[int, float, float]
    ] = {}
    boundary_parameters = {}
    boundary_offset = 0
    for loop in loops:
        count = len(loop.node_ids)
        for local, specification in enumerate(loop.segment_specs):
            if specification is not None:
                if cylindrical_chart is not None or analytic_chart is not None:
                    edge_id = specification[0]
                    if edge_id not in boundary_parameters:
                        _check_cancellation(cancellation_check, "native edge boundary lookup")
                        boundary_parameters[edge_id] = {
                            int(entry.node_id): float(entry.key.parameter)
                            for entry in boundary_registry.entries(edge_id)
                            if entry.node_id is not None
                        }
                    parameters = boundary_parameters[edge_id]
                    following = (local + 1) % count
                    ends = (local, following) if local < following else (following, local)
                    specification = (
                        edge_id,
                        parameters[loop.node_ids[ends[0]]],
                        parameters[loop.node_ids[ends[1]]],
                    )
                automatically_seeded_shared_segments[
                    tuple(sorted((boundary_offset + local, boundary_offset + (local + 1) % count)))
                ] = specification
        boundary_offset += count
    segments = [
        float(np.linalg.norm(loop[(index + 1) % len(loop)] - loop[index]))
        for loop in chart_loops
        for index in range(len(loop))
    ]
    boundary_seconds = perf_counter() - boundary_started
    # Edge seeding is authoritative.  A size just above the longest registered
    # chart segment lets the surface filler add interior points without adding
    # unregistered boundary stations.
    chart_size = max(segments) * (1.0 + 64.0 * np.finfo(float).eps)
    seeded_general_chart = analytic_chart is not None
    if cylindrical_chart is not None:
        from ._trimmed_cylinder_binding import TrimmedCylinderBinding
        from anygeometry import EllipticArc,CylinderIntersectionCurve
        seeded_general_chart = (isinstance(_cylindrical_binding,TrimmedCylinderBinding)
            and any(isinstance(path.curve,(EllipticArc,CylinderIntersectionCurve))
                    for record in _cylindrical_binding.face_records
                    for loop in record.boundaries for path in loop))
        if seeded_general_chart:
            # Exact protected boundary stations may be much finer than the
            # requested interior field. Preserve them while the selected
            # native path fills the interior and enforces its quality gates.
            chart_size=size_field.target_size
    face_native_options = native_options
    if native_options.point_placement == "frontal_delaunay" and not isinstance(
        face.surface, Plane
    ) and cylindrical_chart is None and analytic_chart is None:
        from .errors import NativeSurfaceUnsupported
        raise NativeSurfaceUnsupported(
            "frontal_delaunay activation is currently limited to planar faces"
        )
    size_metric_spec = MetricFieldSpec.from_size_field(size_field)
    supplemental_metric_field = None
    if (
        native_options.metric_mode == "isotropic_spatial"
        and (isinstance(face.surface, Plane) or cylindrical_chart is not None or analytic_chart is not None)
    ):
        if native_options.metric_field is not None:
            explicit = native_options.metric_field
            face_native_options = replace(
                native_options,
                metric_field=MetricFieldSpec(
                    IsotropicMetricControl(
                        min(
                            explicit.global_control.target_size,
                            size_metric_spec.global_control.target_size,
                        )
                    ),
                    feature_controls=(
                        *explicit.feature_controls,
                        *size_metric_spec.feature_controls,
                    ),
                    imported_samples=explicit.imported_samples,
                    maximum_anisotropy=min(
                        explicit.maximum_anisotropy,
                        size_metric_spec.maximum_anisotropy,
                    ),
                    maximum_gradation=min(
                        explicit.maximum_gradation,
                        size_metric_spec.maximum_gradation,
                    ),
                ),
            )
        elif native_options.experimental_metric_provider is None:
            face_native_options = replace(
                native_options,
                metric_field=size_metric_spec,
            )
        else:
            supplemental_metric_field = size_metric_spec
    deferred_recombine = (cylindrical_chart is not None and not quadratic
        and bool(recombine) and native_options.point_placement == "frontal_delaunay")
    if deferred_recombine:
        from ._cylindrical_recombine import register_face

        deferred_settings = SurfaceMeshOptions(
            target_size=chart_size, recombine=True, order="linear",
            backend=native_backend, native_options=face_native_options,
            prefer_quality_policy=True,
            **({} if quality_options is None else {
                "min_scaled_jacobian": quality_options.quality_policy.minimum_scaled_jacobian,
                "max_aspect_ratio": quality_options.quality_policy.maximum_aspect_ratio,
                "min_angle": quality_options.quality_policy.minimum_angle,
                "max_angle": quality_options.quality_policy.maximum_angle,
                "max_warpage": quality_options.quality_policy.maximum_warpage,
                "max_element_growth": quality_options.max_element_growth,
            }),
        )
        register_face(
            geometry, mesh, face_id, _cylindrical_binding,
            component_seed_registry, deferred_settings,
            effective_metric_field=(
                MetricFieldSpec.uniform(chart_size)
                if face_native_options.metric_mode == "legacy"
                else face_native_options.metric_field
            ),
            supplemental_metric_field=supplemental_metric_field,
        )
        recombine = False
    native_outer, native_holes = chart_outer, chart_holes
    native_constraints, native_pinned = chart_constraints, chart_pinned
    protected_input_rows = None
    if _material_region_binding is not None:
        if _material_input_eligibility is None:
            raise MeshError('material input provenance requires route eligibility context')
        # The receipt certifies the same detached arrays passed into native
        # meshing. Owner callbacks cannot substitute the earlier chart arrays.
        native_outer = np.array(chart_outer, copy=True)
        native_holes = tuple(np.array(item, copy=True) for item in chart_holes)
        native_constraints = tuple(np.array(item, copy=True) for item in chart_constraints)
        native_pinned = np.array(chart_pinned, copy=True)
        emitted = {
            'transform': np.array(chart_transform, copy=True),
            'chart_loops': (native_outer, *native_holes),
            'chart_constraints': native_constraints,
            'chart_pinned': native_pinned,
        }
        receipt = _material_region_binding.input_provenance(
            mesh, boundary_registry, loops, region_pinned_ids,
            _material_input_eligibility, emitted, cancellation_check)
        from ._row_provenance import material_input_rows
        protected_input_rows = material_input_rows(receipt)
        records = getattr(component_seed_registry, '_material_input_provenance', None)
        if records is None:
            records = {}
            component_seed_registry._material_input_provenance = records
        records[face_id] = receipt
    surface_diagnostics: dict[str, Any] = {}
    primary_output_origin = None
    primary_output_entry = None
    if quality_options is None:
        core = mesh_planar_surface(
            native_outer,
            native_holes,
            constraints=native_constraints,
            interior_points=native_pinned,
            target_size=chart_size,
            recombine=recombine,
            order=order,
            backend=native_backend,
            owner=geometry.handle("face", face_id),
            cancellation_check=cancellation_check,
            diagnostics=surface_diagnostics,
            native_options=(
                None
                if cylindrical_chart is not None
                and native_options.metric_mode == "isotropic_spatial"
                else face_native_options
            ),
            _metric_model_uuid=metric_model_uuid,
            _metric_geometry_revision=metric_geometry_revision,
            _metric_to_physical=metric_to_physical,
            _coordinate_batch_size=8 if analytic_chart is not None else 1,
            _metric_jacobian=physical_jacobian,
            _automatically_seeded_shared_segments=automatically_seeded_shared_segments,
            _component_seed_registry=component_seed_registry,
            _supplemental_metric_field=supplemental_metric_field,
            _preserve_spatial_refinement=(cylindrical_chart is not None or analytic_chart is not None),
            _polish_quality_candidates=isinstance(face.surface, Plane),
            _boundary_is_seeded=seeded_general_chart,
            _protected_node_ids=protected_input_rows,
            options=(
                SurfaceMeshOptions(
                    target_size=chart_size, recombine=recombine, order=order,
                    backend=native_backend, native_options=face_native_options,
                    prefer_quality_policy=True,
                )
                if cylindrical_chart is not None
                and native_options.metric_mode == "isotropic_spatial"
                else None
            ),
        )
        if _material_region_binding is not None:
            primary_output_origin = getattr(core, '_output_row_provenance', None)
            if primary_output_origin is not None:
                primary_output_entry = (primary_output_origin.source_by_row,
                                        primary_output_origin.source_to_row)
    else:
        quality_policy = quality_options.quality_policy
        face_edge_ids = {
            int(item.edge)
            for boundary in (face.loop, *getattr(face, "holes", ()))
            for item in boundary
        }
        outer_edge_ids = {int(item.edge) for item in face.loop}
        if _material_region_binding is not None:
            region = _material_region_binding.region
            face_edge_ids = {int(path.source_edge) for paths in region.boundaries for path in paths}
            face_edge_ids.update(int(path.source_edge) for path in region.interior_constraints)
            outer_edge_ids = {int(path.source_edge) for path in region.boundaries[0]}
        declared_edge_ids = {int(edge_id) for edge_id in declared_junction_edges}
        has_declared_junction = bool(
            face_edge_ids.intersection(declared_edge_ids)
        )
        has_unconstrained_outer_boundary = bool(
            outer_edge_ids.difference(declared_edge_ids)
        )
        evaluate_outer_alignment = (
            not has_declared_junction
            or (
                evaluate_declared_junction_alignment
                and has_unconstrained_outer_boundary
            )
        )
        evaluate_hole_alignment = not has_declared_junction
        surface_options = SurfaceMeshOptions(
            recombine=recombine,
            order=order,
            target_size=chart_size,
            backend=native_backend,
            min_scaled_jacobian=quality_policy.minimum_scaled_jacobian,
            max_aspect_ratio=quality_policy.maximum_aspect_ratio,
            min_angle=quality_policy.minimum_angle,
            max_angle=quality_policy.maximum_angle,
            max_warpage=quality_policy.maximum_warpage,
            max_element_growth=quality_options.max_element_growth,
            prefer_quality_policy=True,
            declared_junction=has_declared_junction,
            native_options=face_native_options,
        )
        core = mesh_planar_surface(
            native_outer,
            native_holes,
            constraints=native_constraints,
            interior_points=native_pinned,
            options=surface_options,
            owner=geometry.handle("face", face_id),
            cancellation_check=cancellation_check,
            diagnostics=surface_diagnostics,
            _evaluate_boundary_alignment=evaluate_outer_alignment,
            _evaluate_hole_alignment=evaluate_hole_alignment,
            _refine_boundary_transition=refine_declared_junction_transition,
            _metric_model_uuid=metric_model_uuid,
            _metric_geometry_revision=metric_geometry_revision,
            _metric_to_physical=metric_to_physical,
            _coordinate_batch_size=8 if analytic_chart is not None else 1,
            _metric_jacobian=physical_jacobian,
            _automatically_seeded_shared_segments=automatically_seeded_shared_segments,
            _component_seed_registry=component_seed_registry,
            _supplemental_metric_field=supplemental_metric_field,
            _preserve_spatial_refinement=(cylindrical_chart is not None or analytic_chart is not None),
            _polish_quality_candidates=isinstance(face.surface, Plane),
            _boundary_is_seeded=seeded_general_chart,
            _protected_node_ids=protected_input_rows,
        )
        if _material_region_binding is not None:
            primary_output_origin = getattr(core, '_output_row_provenance', None)
            if primary_output_origin is not None:
                primary_output_entry = (primary_output_origin.source_by_row,
                                        primary_output_origin.source_to_row)
        if (
            isinstance(face.surface, Plane)
            and not np.allclose(chart_transform, np.eye(2), rtol=1.0e-12, atol=1.0e-14)
            and not surface_diagnostics.get("quality_policy", {}).get(
                "accepted", True
            )
        ):
            if _material_region_binding is not None:
                raise MeshError('material Plane parameter fallback has no output provenance')
            parameter_diagnostics: dict[str, Any] = {}
            parameter_core = mesh_planar_surface(
                np.asarray(outer.uv, dtype=float),
                tuple(np.asarray(loop.uv, dtype=float) for loop in holes),
                options=surface_options,
                owner=geometry.handle("face", face_id),
                cancellation_check=cancellation_check,
                diagnostics=parameter_diagnostics,
                _evaluate_boundary_alignment=evaluate_outer_alignment,
                _evaluate_hole_alignment=evaluate_hole_alignment,
                _refine_boundary_transition=refine_declared_junction_transition,
                _metric_model_uuid=metric_model_uuid,
                _metric_geometry_revision=metric_geometry_revision,
                _metric_to_physical=lambda parameter_points: (
                    metric_origin[None, :]
                    + np.asarray(parameter_points, dtype=np.float64)[:, 0, None] * metric_u[None, :]
                    + np.asarray(parameter_points, dtype=np.float64)[:, 1, None] * metric_v[None, :]
                ),
                _metric_jacobian=np.column_stack((metric_u, metric_v)),
                _automatically_seeded_shared_segments=automatically_seeded_shared_segments,
                _component_seed_registry=component_seed_registry,
                _supplemental_metric_field=supplemental_metric_field,
            )
            # The parameter chart may scale/shear lengths and angles. Its
            # apparent quality cannot replace a physically better candidate.
            from .surface_mesh import _quality_threshold_report, _quality_not_worse
            uv = parameter_core.node_coordinates[:, :2]
            physical_core = MeshCore(
                metric_origin[None, :] + uv[:, 0, None]*metric_u[None, :]
                + uv[:, 1, None]*metric_v[None, :],
                parameter_core.triangle_connectivity,
                parameter_core.quad_connectivity,
                node_active=parameter_core.node_active,
                triangle_active=parameter_core.triangle_active,
                quad_active=parameter_core.quad_active,
            )
            physical_policy = _quality_threshold_report(evaluate_quality(physical_core), surface_options)
            parameter_diagnostics["physical_quality_policy"] = physical_policy
            if (parameter_diagnostics.get("quality_policy", {}).get("accepted", False)
                    and _quality_not_worse(physical_policy, surface_diagnostics["quality_policy"])):
                parameter_diagnostics["chart_fallback"] = {
                    "from": "physical_metric",
                    "to": "parameter",
                    "reason": "physical metric candidate missed explicit policy",
                }
                core = parameter_core
                surface_diagnostics = parameter_diagnostics
                chart_inverse = np.eye(2, dtype=float)
                chart_loops = tuple(
                    np.asarray(loop.uv, dtype=float) for loop in loops
                )
        if has_declared_junction:
            alignment_scope = {
                "outer_boundary": (
                    "evaluated"
                    if evaluate_outer_alignment
                    else "skipped_declared_junction_only_boundary"
                ),
                "hole_boundary": "skipped_declared_junction_interface",
            }
            surface_diagnostics["declared_junction_alignment_scope"] = (
                alignment_scope
            )
            complex_geometry = surface_diagnostics.setdefault(
                "complex_geometry", {}
            )
            complex_geometry["alignment_scope"] = alignment_scope
            if not evaluate_outer_alignment:
                complex_geometry["alignment_evaluation"] = (
                    "skipped_declared_junction_only_boundary"
                )
                surface_diagnostics["boundary_collar_skip_reason"] = (
                    "declared_junction_only_boundary"
                )
    surface_diagnostics.setdefault("phase_seconds", {})[
        "boundary_registration"
    ] = boundary_seconds
    if cylindrical_chart is not None:
        diagnostic_key = ("cylindrical_frontal_delaunay"
            if native_options.point_placement == "frontal_delaunay" else "cylindrical_material_chart")
        surface_diagnostics[diagnostic_key] = {
            "chart": ("owner_certified_sector_physical_lengths"
                      if _cylindrical_binding.certification_kind == "full_period_sector_atlas"
                      else "owner_certified_patch_physical_lengths"),
            "circumferential_length": cylindrical_chart.circumferential_length,
            "axial_length": cylindrical_chart.axial_length,
            "protected_seeds": "existing_global_edge_registry",
            "periodic_topology": ("connected_owner_sector_faces"
                                  if _cylindrical_binding.certification_kind == "full_period_sector_atlas"
                                  else "nonperiodic_owner_patch"),
            "owner_contract": _cylindrical_binding.certification_kind,
        }
    _check_cancellation(cancellation_check, f"native face {face_id} lifting start")
    if cylindrical_chart is not None:
        _cylindrical_binding.validate()
    region_core_to_global = None
    if _material_region_binding is not None:
        receipt.validate(cancellation_check)
        output_origin = getattr(core, '_output_row_provenance', None)
        if (output_origin is None or output_origin is not primary_output_origin
                or (output_origin.source_by_row, output_origin.source_to_row)
                != primary_output_entry):
            raise MeshError('material primary output has no protected row provenance')
        region_core_to_global = output_origin.validate(core)
    if analytic_chart is not None:
        from .surface_mesh import SurfaceMeshOptions as AnalyticQualityOptions
        settings = (AnalyticQualityOptions() if quality_options is None else surface_options)
        registered = np.asarray([mesh.nodes[node] for loop in loops for node in loop.node_ids])
        registered_rows = None
        if _material_region_binding is not None:
            registered_rows = {row: mesh.nodes[node] for row, node in region_core_to_global.items()}
            _material_region_binding.validate_constraints(core, mesh, boundary_registry, region_core_to_global)
        surface_diagnostics['analytic_physical_quality'] = analytic_chart.certify_core(
            core, settings,
            registered if _material_region_binding is None else np.empty((0, 3)),
            registered_rows=registered_rows)
        if _material_region_binding is not None:
            region_settings = getattr(component_seed_registry, '_material_region_quality_settings', {})
            region_settings[face_id] = settings
            component_seed_registry._material_region_quality_settings = region_settings
    lifting_started = perf_counter()
    assert_valid_mesh(core)

    input_uv = np.vstack(chart_loops)
    coordinates = np.asarray(core.node_coordinates, dtype=float)
    region_coordinates = (None if _material_region_binding is None else
                          analytic_chart.evaluate(coordinates[:, :2]))
    if _material_region_binding is None and (len(coordinates) < len(input_uv) or not np.allclose(
        coordinates[: len(input_uv), :2], input_uv, rtol=0.0, atol=2.0e-14
    )):
        raise MeshError(
            f"native face {face_id} changed its registered boundary ordering; "
            "conformal identity cannot be guaranteed"
        )

    core_to_global: dict[int, int] = {}
    offset = 0
    if _material_region_binding is None:
        for loop in loops:
            for local, node_id in enumerate(loop.node_ids):
                core_to_global[offset + local] = int(node_id)
            offset += len(loop.node_ids)
    else:
        core_to_global.update(region_core_to_global)

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

    for record in surface_diagnostics.get("native_v2", {}).get("shared_nodes", ()):
        core_node = int(record["local_node_id"])
        node_id = int(record["node_id"])
        edge_id = int(record["edge_id"])
        if core_node in core_to_global or not 0 <= core_node < len(coordinates):
            raise MeshError(
                "native-v2 shared-node receipt conflicts with boundary topology: "
                f"face={face_id}, local_node={core_node}, node_count={len(coordinates)}, "
                f"protected_row={core_node in core_to_global}, edge={edge_id}, "
                f"route={surface_diagnostics.get('native_v2', {}).get('selected_route')}"
            )
        u, v = (
            float(value)
            for value in (coordinates[core_node, :2] @ chart_inverse)
        )
        candidate = (region_coordinates[core_node]
                     if _material_region_binding is not None else
                     np.asarray(geometry.face_point(face_id, u, v), dtype=float))
        tolerance = geometry.tolerance.effective_length(geometry.edge_length(edge_id))
        if cylindrical_chart is not None or _material_region_binding is not None:
            from fractions import Fraction

            numerator, denominator = record["station"]
            if not isinstance(numerator, int) or not isinstance(denominator, int) or denominator <= 0:
                raise MeshError("cylindrical split requires an exact native edge station")
            parameter = float(Fraction(numerator, denominator))
            if not 0. < parameter < 1.:
                raise MeshError("cylindrical split station must be strictly inside its source edge")
            edge_point = geometry.sample_edge(edge_id, np.asarray([parameter]))[0]
            distance = float(np.linalg.norm(candidate - edge_point))
        else:
            edge_point, parameter, distance = geometry.closest_edge_point(edge_id, candidate)
        if distance > tolerance:
            raise MeshError("native-v2 shared split left its topology-owned edge")
        exact_point = np.asarray(edge_point, dtype=float)
        previous_point = mesh.nodes.get(node_id)
        if previous_point is not None and not np.allclose(
            previous_point, exact_point, rtol=0.0, atol=tolerance
        ):
            raise MeshError("component seed registry reused a node at another point")
        mesh.nodes[node_id] = exact_point
        sequence = mesh.nodes_of_edge[edge_id]
        if node_id not in sequence:
            material_neighbour = any(
                other in getattr(component_seed_registry, '_material_region_representatives', {})
                for other in geometry.faces_using_edge(edge_id))
            if cylindrical_chart is not None or analytic_chart is not None or material_neighbour:
                from ._shared_triangle_split import propagate_triangle_split

                stations = {
                    int(entry.node_id): float(entry.key.parameter)
                    for entry in boundary_registry.entries(edge_id)
                    if entry.node_id is not None
                }
                ordered = sorted(sequence, key=stations.__getitem__)
                brackets = tuple(
                    (a, b) for a, b in zip(ordered, ordered[1:])
                    if stations[a] < parameter < stations[b]
                )
                if len(brackets) != 1:
                    raise MeshError("shared station has no unique registered source-edge interval")
                cache = getattr(component_seed_registry, "_published_triangle_incidence", None)
                if cache is None:
                    cache = {}
                    component_seed_registry._published_triangle_incidence = cache
                propagate_triangle_split(
                    mesh,
                    {getattr(component_seed_registry, '_material_region_representatives', {}).get(other, other)
                     for other in geometry.faces_using_edge(edge_id)
                     if other not in (_material_region_binding.face_ids
                                      if _material_region_binding is not None else (face_id,))},
                    brackets[0], node_id, cache=cache,
                )
                stations[node_id] = parameter
                sequence.append(node_id)
                sequence.sort(key=stations.__getitem__)
            else:
                sequence.append(node_id)
                sequence.sort(
                    key=lambda value: geometry.closest_edge_point(
                        edge_id, mesh.nodes[value]
                    )[1]
                )
        boundary_registry.register(
            edge_id,
            float(parameter),
            exact_point,
            node_id=node_id,
            owner=geometry.handle("edge", edge_id),
        )
        core_to_global[core_node] = node_id

    next_node = _next_identifier(mesh.nodes)
    reserved_node_ids = set(component_seed_registry.assigned_node_ids)
    for core_node in range(len(coordinates)):
        if core_node in core_to_global:
            continue
        u, v = (
            float(value)
            for value in (coordinates[core_node, :2] @ chart_inverse)
        )
        point = (region_coordinates[core_node]
                 if _material_region_binding is not None else
                 np.asarray(geometry.face_point(face_id, u, v), dtype=float))
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise MeshError(f"face {face_id} surface evaluation returned an invalid point")
        while next_node in reserved_node_ids or next_node in mesh.nodes:
            next_node += 1
        core_to_global[core_node] = next_node
        mesh.nodes[next_node] = point
        next_node += 1

    _publish_native_face_elements(mesh, face_id, core, core_to_global)
    if _material_region_binding is not None:
        _material_region_binding.validate(cancellation_check)
        region = _material_region_binding.region
        surface_diagnostics['material_region'] = {
            'source_faces': sorted(_material_region_binding.face_ids),
            'authored_face': _material_region_binding.authored_face,
            'cancelled_seams': [handle.id for handle in region.cancelled_seams],
            'interior_constraint_edges': [path.source_edge for path in region.interior_constraints],
            'retained_vertices': [handle.id for handle in region.retained_vertices],
            'pinned_stations': len(region_pinned_ids),
        }
    if deferred_recombine:
        from ._cylindrical_recombine import mark_staged

        mark_staged(component_seed_registry, face_id, mesh, surface_diagnostics)
    if cylindrical_chart is not None and not deferred_recombine:
        from ._shared_triangle_split import repair_completed_cylindrical_neighbours

        repair_reports = repair_completed_cylindrical_neighbours(
            mesh,
            geometry,
            _cylindrical_binding,
            component_seed_registry,
            cancellation_check=cancellation_check,
        )
        if repair_reports:
            surface_diagnostics["cylindrical_shared_boundary_repair"] = repair_reports
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


# ---------------------------------------------------------------------------
# Quad-first execution: front -> count -> TinyAD -> atomic publish.
#
# An explicit, in-scope ``quad_options`` must NOT silently fall back to the
# legacy body.  The worker chain below exercises the Q3 front step, the Q4
# integer min-cost-flow count worker, and the Q5 TinyAD patch optimizer, then
# atomically publishes a ``HybridMeshResult``.  The ``None`` dispatch sentinel
# still runs the legacy body byte-identically, so this helper is only reached
# for a validated explicit request.
# ---------------------------------------------------------------------------

def _quad_first_seed(
    geometry: GeometryModel,
    face_id: int,
) -> tuple[
    QuadMeshState, tuple[int, int], dict[int, np.ndarray], dict[int, int]
]:
    """Build the Q3 seed plus exact geometry-to-mesh corner associations."""
    if face_id not in geometry.faces:
        raise MeshError(f"no face {face_id}")
    face = geometry.faces[face_id]
    if len(face.corners) != 4 or len(face.loop) != 4 or getattr(face, "holes", ()):
        raise QuadPublicUnsupported(
            "quad-first public route currently requires one untrimmed four-corner face"
        )

    corners: list[np.ndarray] = []
    corner_vertex_ids: list[int] = []
    for corner_index in face.corners:
        use = face.loop[int(corner_index)]
        edge = geometry.edges[int(use.edge)]
        vertex_id = int(edge.start if use.forward else edge.end)
        corner_vertex_ids.append(vertex_id)
        corners.append(np.asarray(geometry.vertices[vertex_id].position, dtype=float))
    p0, p1, p2, p3 = corners

    x_axis = p1 - p0
    x_len = float(np.linalg.norm(x_axis))
    normal = np.cross(p1 - p0, p3 - p0)
    normal_len = float(np.linalg.norm(normal))
    if x_len <= 1.0e-14 or normal_len <= 1.0e-14:
        raise QuadPublicUnsupported("quad-first face has a degenerate planar frame")
    x_hat = x_axis / x_len
    normal_hat = normal / normal_len
    y_hat = np.cross(normal_hat, x_hat)
    scale = max(float(np.linalg.norm(point - p0)) for point in corners) or 1.0
    if max(abs(float((point - p0) @ normal_hat)) for point in corners) > max(1.0e-10, 1.0e-9 * scale):
        raise QuadPublicUnsupported("quad-first public route currently requires a planar face")

    # Q3's canonical node numbering is (BL, BR, TL, TR), while the face loop
    # is cyclic (BL, BR, TR, TL).  Preserve exact topology with this mapping.
    published = (p0, p1, p3, p2)
    nodes = {
        node: (
            float((point - p0) @ x_hat),
            float((point - p0) @ y_hat),
        )
        for node, point in enumerate(published)
    }
    state = QuadMeshState(
        nodes=nodes,
        cells={0: (0, 1, 2), 1: (1, 2, 3)},
        initial_front=((0, 1), (0, 2), (1, 3), (2, 3)),
    )
    published_vertex_ids = (
        corner_vertex_ids[0],
        corner_vertex_ids[1],
        corner_vertex_ids[3],
        corner_vertex_ids[2],
    )
    return (
        state,
        (0, 1),
        {
            node: np.array(point, dtype=float, copy=True)
            for node, point in enumerate(published)
        },
        {vertex_id: node for node, vertex_id in enumerate(published_vertex_ids)},
    )


def _quad_first_normalize_face_ids(
    value: Any,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        value = (value,)
    if not isinstance(value, Iterable) or isinstance(value, Mapping):
        value = (value,)
    values = tuple(dict.fromkeys(int(item) for item in value))
    if not values:
        return None
    return tuple(sorted(values))


def _merge_quad_first_and_legacy(
    geometry: Any,
    *,
    quad_faces: frozenset[int],
    legacy_faces: frozenset[int],
    quad_result: "HybridMeshResult",
    legacy_result: "HybridMeshResult",
) -> "HybridMeshResult":
    """Combine the quad-first and legacy neutral meshes on exact geometry IDs.

    The quad result owns the canonical node and element IDs; legacy nodes and
    shells are reindexed only when a legacy node ID collides with a quad one.
    Every association is matched on geometry identity: ``node_of_vertex``
    vertex IDs, ``nodes_of_edge`` edge IDs, ``elements_of_face`` face IDs.
    No coordinate welding is performed; disagreements raise.
    """

    if quad_faces & legacy_faces:
        raise MeshError("same face claimed by both quad-first and legacy routes")
    if not quad_faces:
        raise MeshError("quad-first/legacy merge requires quad-first faces")
    if not legacy_faces and not legacy_result.mesh.beams:
        raise MeshError("quad-first/legacy merge requires legacy faces or beams")

    quad_mesh = quad_result.mesh
    legacy_mesh = legacy_result.mesh

    vertex_to_node: dict[int, int] = {
        int(vertex): int(node)
        for vertex, node in quad_mesh.node_of_vertex.items()
    }

    legacy_node_to_vertex: dict[int, int] = {
        int(node): int(vertex)
        for vertex, node in legacy_mesh.node_of_vertex.items()
    }

    quad_node_ids = {int(node) for node in quad_mesh.nodes}
    # Linear beam stations must keep their IDs when the same shell corners
    # and member are promoted.  Reserve the linear shell/beam corner namespace
    # before adding either family's quadratic midsides.
    stable_beam_promotion = bool(legacy_mesh.beams and not legacy_faces and quad_mesh.order == "quadratic")
    if stable_beam_promotion:
        quad_corner_ids = {
            int(node) for body in quad_mesh.quads.values() for node in body[:4]
        } | {
            int(node) for body in quad_mesh.tris.values() for node in body[:3]
        } | {int(node) for node in quad_mesh.node_of_vertex.values()}
        if not quad_corner_ids:
            raise MeshError("quadratic quad-first beam merge has no shell corners")
        next_id = max(quad_corner_ids) + 1
    else:
        quad_corner_ids = quad_node_ids
        next_id = (max(quad_node_ids) + 1) if quad_node_ids else 0
    new_vertex_to_node: dict[int, int] = {}
    for legacy_vertex in sorted(set(legacy_node_to_vertex.values())):
        if legacy_vertex in vertex_to_node:
            continue
        new_vertex_to_node[legacy_vertex] = next_id
        next_id += 1
    node_map: dict[int, int] = {}
    for legacy_node_id, legacy_vertex in legacy_node_to_vertex.items():
        if legacy_vertex in vertex_to_node:
            node_map[legacy_node_id] = vertex_to_node[legacy_vertex]
        else:
            node_map[legacy_node_id] = new_vertex_to_node[legacy_vertex]
    vertex_to_node.update(new_vertex_to_node)
    legacy_corner_ids = (
        {int(node) for body in legacy_mesh.beams.values() for node in (body[0], body[-1])}
        if stable_beam_promotion else set()
    )
    legacy_order = (
        sorted(legacy_corner_ids) + sorted(set(map(int, legacy_mesh.nodes)) - legacy_corner_ids)
        if stable_beam_promotion else sorted(int(node) for node in legacy_mesh.nodes)
    )
    for legacy_node_id in legacy_order:
        if legacy_node_id in node_map:
            continue
        node_map[legacy_node_id] = next_id
        next_id += 1

    quad_node_map = {node: node for node in quad_corner_ids}
    if stable_beam_promotion:
        for quad_node_id in sorted(quad_node_ids - quad_corner_ids):
            quad_node_map[quad_node_id] = next_id
            next_id += 1

    merged_nodes: dict[int, np.ndarray] = {
        quad_node_map[int(node)]: np.asarray(pos, dtype=float)
        for node, pos in quad_mesh.nodes.items()
    }
    for legacy_node, legacy_pos in legacy_mesh.nodes.items():
        merged = node_map.get(int(legacy_node))
        if merged is None:
            raise MeshError(
                f"legacy node {legacy_node} is not associated with a vertex"
            )
        if merged not in merged_nodes:
            merged_nodes[merged] = np.asarray(legacy_pos, dtype=float)

    merged_quads: dict[int, tuple[int, ...]] = {
        int(element_id): tuple(quad_node_map[int(node)] for node in quad_body)
        for element_id, quad_body in quad_mesh.quads.items()
    }
    merged_tris: dict[int, tuple[int, ...]] = {
        int(element_id): tuple(quad_node_map[int(node)] for node in tri_body)
        for element_id, tri_body in quad_mesh.tris.items()
    }
    occupied_quad_shell_ids = set(merged_quads) | set(merged_tris)
    next_element_id = (
        max(occupied_quad_shell_ids) + 1 if occupied_quad_shell_ids else 0
    )

    def _remap_body(body: Any) -> tuple[int, ...]:
        return tuple(node_map[int(node)] for node in body)

    # Legacy element IDs are remapped in their own namespace; quad-first
    # shell IDs remain authoritative even when numeric IDs collide.
    element_id_map: dict[int, int] = {}
    for legacy_element_id in sorted(int(e) for e in legacy_mesh.quads):
        new_id = next_element_id
        next_element_id += 1
        element_id_map[legacy_element_id] = new_id
        merged_quads[new_id] = _remap_body(
            legacy_mesh.quads[legacy_element_id]
        )
    for legacy_element_id in sorted(int(e) for e in legacy_mesh.tris):
        new_id = next_element_id
        next_element_id += 1
        element_id_map[legacy_element_id] = new_id
        merged_tris[new_id] = _remap_body(
            legacy_mesh.tris[legacy_element_id]
        )
    if legacy_mesh.beams:
        legacy_beams: dict[int, tuple[int, ...]] = {}
        for legacy_element_id in sorted(int(e) for e in legacy_mesh.beams):
            new_id = next_element_id
            next_element_id += 1
            element_id_map[legacy_element_id] = new_id
            legacy_beams[new_id] = _remap_body(
                legacy_mesh.beams[legacy_element_id]
            )
    else:
        legacy_beams = {}

    nodes_of_edge: dict[int, list[int]] = {}
    for edge_id, sequence in quad_mesh.nodes_of_edge.items():
        nodes_of_edge[int(edge_id)] = [quad_node_map[int(node)] for node in sequence]
    for edge_id, sequence in legacy_mesh.nodes_of_edge.items():
        edge_key = int(edge_id)
        canonical = [
            node_map.get(int(node), int(node)) for node in sequence
        ]
        existing = nodes_of_edge.get(edge_key)
        if existing is not None:
            if set(existing) != set(canonical):
                raise MeshError(
                    f"legacy nodes_of_edge[{edge_key}] disagrees with "
                    f"quad-first nodes_of_edge: {existing!r} != "
                    f"{canonical!r}"
                )
            geometry_edge = getattr(
                geometry, "edges", {}
            ).get(edge_key, None)
            if geometry_edge is not None:
                expected = [
                    vertex_to_node.get(int(geometry_edge.start)),
                    vertex_to_node.get(int(geometry_edge.end)),
                ]
                if expected[0] is not None and expected[1] is not None:
                    canonical = [expected[0], expected[1]]
            nodes_of_edge[edge_key] = canonical
        else:
            nodes_of_edge[edge_key] = canonical

    elements_of_face: dict[int, list[int]] = {
        int(face_id): [int(eid) for eid in element_ids]
        for face_id, element_ids in quad_mesh.elements_of_face.items()
    }
    for face_id, legacy_element_ids in legacy_mesh.elements_of_face.items():
        face_id = int(face_id)
        if face_id in elements_of_face:
            raise MeshError(
                f"both the quad-first and legacy routes emitted elements "
                f"for face {face_id}"
            )
        elements_of_face[face_id] = [
            element_id_map[int(element_id)]
            for element_id in legacy_element_ids
        ]

    for face_id, quad_element_ids in quad_mesh.elements_of_face.items():
        if int(face_id) not in quad_faces:
            raise MeshError(
                f"quad-first emitted elements for face {face_id} outside the "
                f"selected quad faces {sorted(quad_faces)}"
            )
    for face_id, legacy_element_ids in legacy_mesh.elements_of_face.items():
        if int(face_id) not in legacy_faces:
            raise MeshError(
                f"legacy emitted elements for face {face_id} outside the "
                f"selected legacy faces {sorted(legacy_faces)}"
            )

    offset_nodes_of_edge: dict[int, list[int]] = {
        int(edge_id): [node_map[int(node)] for node in sequence]
        for edge_id, sequence in legacy_mesh.offset_nodes_of_edge.items()
    } | {
        int(edge_id): [quad_node_map[int(node)] for node in sequence]
        for edge_id, sequence in quad_mesh.offset_nodes_of_edge.items()
    }

    merged_mesh = Mesh(
        geometry_model_id=quad_mesh.geometry_model_id,
        geometry_revision=quad_mesh.geometry_revision,
        nodes=merged_nodes,
        quads=merged_quads,
        tris=merged_tris,
        beams=legacy_beams,
        node_of_vertex=vertex_to_node,
        nodes_of_edge=nodes_of_edge,
        offset_nodes_of_edge=offset_nodes_of_edge,
        elements_of_face=elements_of_face,
        elements_of_edge={
            int(edge_id): [
                element_id_map.get(int(element_id), int(element_id))
                for element_id in element_ids
            ]
            for edge_id, element_ids in legacy_mesh.elements_of_edge.items()
        },
        order=quad_mesh.order,
    )
    if legacy_mesh.seeding is not None:
        merged_mesh.seeding = legacy_mesh.seeding

    # The merged beam IDs/nodes have a new namespace. Rebuild member groups
    # from authoritative source edge uses, not the pre-merge numeric IDs.
    for member_id, member in geometry.members.items():
        member_elements: list[int] = []
        member_nodes: list[int] = []
        for use_id in member.edge_use_ids:
            use = geometry.member_edge_uses[use_id]
            edge_id = int(use.edge_id)
            elements = list(merged_mesh.elements_of_edge.get(edge_id, ()))
            nodes = list(merged_mesh.offset_nodes_of_edge.get(edge_id)
                         or merged_mesh.nodes_of_edge.get(edge_id, ()))
            if not elements:
                continue
            if int(use.orientation) < 0:
                elements.reverse()
                nodes.reverse()
            member_elements.extend(elements)
            member_nodes.extend(nodes)
        if member_elements:
            merged_mesh.elements_of_member[int(member_id)] = list(dict.fromkeys(member_elements))
            merged_mesh.nodes_of_member[int(member_id)] = list(dict.fromkeys(member_nodes))

    merged_mesh.thickness_of_face = {
        int(face_id): float(value)
        for face_id, value in quad_mesh.thickness_of_face.items()
    }
    for face_id, value in legacy_mesh.thickness_of_face.items():
        key = int(face_id)
        normalized = float(value)
        if key in merged_mesh.thickness_of_face:
            existing = merged_mesh.thickness_of_face[key]
            if existing != normalized:
                raise MeshError(
                    f"quad-first/legacy merge disagrees on face {key} "
                    f"thickness ({existing} != {normalized})"
                )
        else:
            merged_mesh.thickness_of_face[key] = normalized
    merged_mesh.structural_preparation = dict(quad_mesh.structural_preparation)
    for key, value in legacy_mesh.structural_preparation.items():
        if key in merged_mesh.structural_preparation:
            existing = merged_mesh.structural_preparation[key]
            if existing != value:
                raise MeshError(
                    f"quad-first/legacy merge disagrees on structural "
                    f"preparation record {key!r}"
                )
        else:
            merged_mesh.structural_preparation[key] = value

    legacy_s3 = legacy_mesh.structural_preparation.get("qualified_s3")
    if isinstance(legacy_s3, Mapping):
        remapped_s3 = deepcopy(dict(legacy_s3))

        def _mapped_element_id(value: Any) -> int:
            identifier = int(value)
            return int(element_id_map.get(identifier, identifier))

        def _mapped_node_id(value: Any) -> int:
            identifier = int(value)
            return int(node_map.get(identifier, identifier))

        if "element_ids" in remapped_s3:
            remapped_s3["element_ids"] = [
                _mapped_element_id(value)
                for value in remapped_s3["element_ids"]
            ]
        for record_key in ("element_owner_normals", "element_owner_sources"):
            values = remapped_s3.get(record_key)
            if isinstance(values, Mapping):
                remapped_s3[record_key] = {
                    str(_mapped_element_id(element_id)): deepcopy(value)
                    for element_id, value in values.items()
                }
        nodal_normals = remapped_s3.get("nodal_normals")
        if isinstance(nodal_normals, Mapping):
            remapped_s3["nodal_normals"] = {
                str(_mapped_node_id(node_id)): deepcopy(value)
                for node_id, value in nodal_normals.items()
            }
        admission = remapped_s3.get("admission")
        if isinstance(admission, Mapping):
            admission = deepcopy(dict(admission))
            elements = admission.get("elements")
            if isinstance(elements, list):
                for element in elements:
                    if isinstance(element, dict) and "element_id" in element:
                        element["element_id"] = _mapped_element_id(
                            element["element_id"]
                        )
            remapped_s3["admission"] = admission
        repair = remapped_s3.get("repair")
        if isinstance(repair, Mapping):
            repair = deepcopy(dict(repair))
            attempts = repair.get("attempts")
            if isinstance(attempts, list):
                for attempt in attempts:
                    if isinstance(attempt, dict) and "element_ids" in attempt:
                        attempt["element_ids"] = [
                            _mapped_element_id(value)
                            for value in attempt["element_ids"]
                        ]
            remapped_s3["repair"] = repair
        merged_mesh.structural_preparation["qualified_s3"] = remapped_s3

    for sheet_id, sheet in geometry.sheets.items():
        merged_mesh.elements_of_sheet[int(sheet_id)] = sorted(
            {
                int(element_id)
                for face_use_id in sheet.face_use_ids
                for element_id in merged_mesh.elements_of_face.get(
                    int(geometry.face_uses[face_use_id].face_id), ()
                )
            }
        )
    merged_mesh.declared_plate_junction_edges = _topology_plate_junction_edges(
        merged_mesh, geometry
    )

    if not legacy_faces:
        merged_mesh.hybrid_diagnostics.update(dict(quad_mesh.hybrid_diagnostics))

    for legacy_report in (legacy_result,):
        for legacy_face_id, legacy_diagnostic in (
            legacy_report.triangulation_backend_by_face.items()
        ):
            merged_mesh.hybrid_diagnostics[str(legacy_face_id)] = (
                legacy_diagnostic
            )
    merged_mesh.hybrid_diagnostics["quad_first_face_ids"] = sorted(
        quad_faces
    )
    merged_mesh.hybrid_diagnostics["legacy_face_ids"] = sorted(legacy_faces)
    merged_mesh.hybrid_diagnostics["quad_first"] = quad_mesh.hybrid_diagnostics

    legacy_strategies: dict[int, str] = {
        int(face_id): str(value)
        for face_id, value in legacy_result.strategy_by_face.items()
    }
    merged_strategies = {face_id: "quad_first" for face_id in sorted(quad_faces)}
    merged_strategies.update(
        {
            face_id: legacy_strategies[face_id]
            for face_id in sorted(legacy_faces)
            if face_id in legacy_strategies
        }
    )
    for face_id in sorted(legacy_faces):
        if face_id not in merged_strategies:
            raise MeshError(
                f"legacy result has no strategy for face {face_id}"
            )

    legacy_backends: dict[int, Mapping[str, Any]] = {
        int(face_id): dict(value)
        for face_id, value in legacy_result.triangulation_backend_by_face.items()
    }
    merged_backends: dict[int, Mapping[str, Any]] = {
        int(face_id): {"backend": "quad_first"}
        for face_id in sorted(quad_faces)
    }
    merged_backends.update(
        {
            face_id: {
                "backend": legacy_backends[face_id].get(
                    "actual_backend", "legacy"
                )
            }
            for face_id in sorted(legacy_faces)
            if face_id in legacy_backends
        }
    )
    for face_id in sorted(legacy_faces):
        if face_id not in merged_backends:
            raise MeshError(
                f"legacy result has no backend provenance for face {face_id}"
            )

    merged_preflight = tuple(quad_result.preflight)
    if legacy_result.preflight is not None:
        merged_preflight = merged_preflight + tuple(legacy_result.preflight)

    return HybridMeshResult(
        mesh=merged_mesh,
        strategy_by_face=merged_strategies,
        triangulation_backend_by_face=merged_backends,
        preflight=merged_preflight,
        connectivity=quad_result.connectivity if legacy_result.connectivity is None else legacy_result.connectivity,
        audit_report=(
            quad_result.audit_report
            if legacy_result.audit_report is None
            else legacy_result.audit_report
        ),
        certification_mode=quad_result.certification_mode,
        certifiable=bool(
            quad_result.certifiable and legacy_result.certifiable
        ),
        structured_layout=(
            legacy_result.structured_layout
            if quad_result.structured_layout is None
            else quad_result.structured_layout
        ),
        structural_preparation=(
            legacy_result.structural_preparation
            if quad_result.structural_preparation is None
            else quad_result.structural_preparation
        ),
    )


def _quad_first_apply_qualified_s3(
    result: HybridMeshResult,
    source_geometry: GeometryModel,
    cancellation_check: Callable[[str], None] | None,
) -> HybridMeshResult:
    """Attach the established qualified-S3 admission to a quad-first result."""
    _check_cancellation(cancellation_check, "quad-first qualified S3 preparation start")
    try:
        prepared_mesh, record = prepare_qualified_s3_mesh(result.mesh, source_geometry)
    except S3RepairError as error:
        # The geometry-valid candidate has already passed the quad-first
        # publication audit. Keep it available to the opt-in recovery caller;
        # strict callers still receive the same typed failure.
        error.inspectable_result = result
        raise
    record["authority_model"].update(
        {
            "source_model_id": str(source_geometry.model_id),
            "source_revision": int(source_geometry.revision),
        }
    )
    payload = dict(prepared_mesh.structural_preparation)
    payload["qualified_s3"] = record
    prepared_mesh.structural_preparation = payload
    diagnostics = dict(prepared_mesh.hybrid_diagnostics)
    diagnostics["qualified_s3_preparation"] = {
        "contract_id": record["contract_id"],
        "element_count": len(record["element_ids"]),
        "formulation_id": record["formulation_id"],
        "legacy_fallback": record["legacy_fallback"],
        "status": record["status"],
    }
    prepared_mesh.hybrid_diagnostics = diagnostics
    _check_cancellation(cancellation_check, "quad-first qualified S3 preparation complete")
    return replace(result, mesh=prepared_mesh)


def _promote_quad_first_quadratic(
    mesh: Mesh,
    geometry: GeometryModel,
    *,
    target_size: float,
    domains_by_face: Mapping[int, Any] | None = None,
    cancellation_check: Callable[[str], None] | None = None,
    protected_interior_nodes: Iterable[int] = (),
) -> None:
    """Promote final quad-first Q4/T3 cells to strict-valid canonical Q8/T6.

    Work is staged on a detached copy.  Exact source-edge midsides are used on
    every geometry-owned boundary segment.  If that exact curvature would fold
    a boundary-adjacent T6, only an unprotected opposite corner may be moved,
    deterministically along the source-curve bulge direction.  Publication is
    atomic after a final strict CH1 validity pass.
    """
    if mesh.is_quadratic:
        return
    if mesh.beams or mesh.couplings:
        raise QuadPublicUnsupported(
            "quadratic quad-first does not yet qualify beam/coupling promotion"
        )
    h = float(target_size)
    if not np.isfinite(h) or h <= 0.0:
        raise MeshError("quadratic promotion target_size must be finite and positive")
    _check_cancellation(cancellation_check, "quad-first:quadratic-promotion")

    working = deepcopy(mesh)

    edge_keys: set[tuple[int, int]] = set()
    for body in working.quads.values():
        corners = tuple(int(node) for node in body[:4])
        edge_keys.update(
            (min(first, second), max(first, second))
            for first, second in zip(corners, corners[1:] + corners[:1])
        )
    for body in working.tris.values():
        corners = tuple(int(node) for node in body[:3])
        edge_keys.update(
            (min(first, second), max(first, second))
            for first, second in zip(corners, corners[1:] + corners[:1])
        )

    face_domains = dict(domains_by_face or {})
    for face_id in working.elements_of_face:
        if int(face_id) not in face_domains:
            surface = geometry.faces[int(face_id)].surface
            if isinstance(surface, Cone):
                face_domains[int(face_id)] = ConicalQuadDomain.from_geometry(geometry, int(face_id))
            elif isinstance(surface, ExtrudedSurface):
                face_domains[int(face_id)] = ParametricQuadDomain.from_geometry(geometry, int(face_id))
            else:
                face_domains[int(face_id)] = PlanarQuadDomain.from_geometry(geometry, int(face_id))
    owner_faces_by_edge: dict[tuple[int, int], set[int]] = {}
    for face_id, element_ids in working.elements_of_face.items():
        for element_id in element_ids:
            body = working.quads.get(element_id)
            corner_count = 4
            if body is None:
                body = working.tris.get(element_id)
                corner_count = 3
            if body is None:
                continue
            corners = tuple(int(node) for node in body[:corner_count])
            for first, second in zip(corners, corners[1:] + corners[:1]):
                key = (min(first, second), max(first, second))
                owner_faces_by_edge.setdefault(key, set()).add(int(face_id))

    # Exact owner midsides for every current source-edge interval.
    boundary_data: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for edge_id, sequence in working.nodes_of_edge.items():
        chain = [int(node) for node in sequence]
        for first, second in zip(chain, chain[1:]):
            key = (min(first, second), max(first, second))
            if key not in edge_keys:
                raise MeshError(
                    f"quadratic promotion boundary segment {key} is not a final-cell edge"
                )
            _point0, parameter0, _distance0 = geometry.closest_edge_point(
                int(edge_id), working.nodes[first]
            )
            _point1, parameter1, _distance1 = geometry.closest_edge_point(
                int(edge_id), working.nodes[second]
            )
            parameter = 0.5 * (float(parameter0) + float(parameter1))
            midpoint = np.asarray(
                geometry.sample_edge(
                    int(edge_id), np.asarray([parameter], dtype=float)
                )[0],
                dtype=float,
            )
            if midpoint.shape != (3,) or not np.all(np.isfinite(midpoint)):
                raise MeshError(
                    f"quadratic promotion produced an invalid source-edge midside on edge {edge_id}"
                )
            previous = boundary_data.get(key)
            if previous is not None and not np.allclose(
                previous[1], midpoint, rtol=0.0, atol=1.0e-12
            ):
                raise MeshError(
                    f"quadratic promotion has conflicting source-edge midsides for {key}"
                )
            boundary_data[key] = (int(edge_id), midpoint)

    protected_nodes = {
        int(node)
        for sequence in working.nodes_of_edge.values()
        for node in sequence
    }
    protected_nodes.update(int(node) for node in working.node_of_vertex.values())
    interior_anchors = tuple(sorted({int(node) for node in protected_interior_nodes}))
    if any(node not in working.nodes for node in interior_anchors):
        raise MeshError("quadratic promotion has an unknown protected interior node")
    protected_nodes.update(interior_anchors)

    def _candidate_midpoint(first: int, second: int) -> np.ndarray:
        key = (min(int(first), int(second)), max(int(first), int(second)))
        owned = boundary_data.get(key)
        if owned is not None:
            return np.asarray(owned[1], dtype=float)
        owner_faces = tuple(sorted(owner_faces_by_edge.get(key, ())))
        chart_owned = [
            face_domains[face]
            for face in owner_faces
            if isinstance(face_domains.get(face), (CylindricalQuadDomain, ConicalQuadDomain, ParametricQuadDomain))
        ]
        if chart_owned:
            if len(owner_faces) != 1 or len(chart_owned) != 1:
                raise MeshError(
                    f"non-boundary curved shell edge {key} has conflicting face ownership"
                )
            domain = chart_owned[0]
            first_chart = domain.project(working.nodes[first])
            second_chart = domain.project(working.nodes[second])
            midpoint_chart = (
                0.5 * (float(first_chart[0]) + float(second_chart[0])),
                0.5 * (float(first_chart[1]) + float(second_chart[1])),
            )
            return np.asarray(domain.lift(midpoint_chart), dtype=float)
        return 0.5 * (
            np.asarray(working.nodes[first], dtype=float)
            + np.asarray(working.nodes[second], dtype=float)
        )

    def _tentative_report(body: tuple[int, ...], family: str):
        corner_count = 4 if family == "Q8" else 3
        corners = tuple(int(node) for node in body[:corner_count])
        midsides = [
            _candidate_midpoint(corners[index], corners[(index + 1) % corner_count])
            for index in range(corner_count)
        ]
        coordinates = np.asarray(
            [working.nodes[node] for node in corners] + midsides, dtype=float
        )
        return certify_mapping_validity(coordinates, family)

    element_specs: dict[tuple[int, str], tuple[int, ...]] = {}
    incident_by_node: dict[int, set[tuple[int, str]]] = {}
    curved_boundary_keys: set[tuple[int, int]] = set()
    scale_tol = max(1.0e-13, 128.0 * np.finfo(np.float64).eps * max(1.0, h))

    for key, (_edge_id, exact_midpoint) in boundary_data.items():
        first, second = key
        chord = 0.5 * (
            np.asarray(working.nodes[first], dtype=float)
            + np.asarray(working.nodes[second], dtype=float)
        )
        if float(np.linalg.norm(np.asarray(exact_midpoint) - chord)) > scale_tol:
            curved_boundary_keys.add(key)

    for element_id, body in sorted(working.quads.items()):
        key = (int(element_id), "Q8")
        corners = tuple(int(node) for node in body[:4])
        element_specs[key] = corners
        for node in corners:
            incident_by_node.setdefault(node, set()).add(key)
    for element_id, body in sorted(working.tris.items()):
        key = (int(element_id), "T6")
        corners = tuple(int(node) for node in body[:3])
        element_specs[key] = corners
        for node in corners:
            incident_by_node.setdefault(node, set()).add(key)

    repair_scope = tuple(
        key
        for key, corners in element_specs.items()
        if any(
            (min(first, second), max(first, second)) in curved_boundary_keys
            for first, second in zip(corners, corners[1:] + corners[:1])
        )
    )

    def _noncertified_scope():
        result: list[tuple[int, str, tuple[int, ...], Any]] = []
        for element_id, family in repair_scope:
            body = element_specs[(element_id, family)]
            report = _tentative_report(body, family)
            if report.status is not ValidityStatus.CERTIFIED_POSITIVE:
                result.append((element_id, family, body, report))
        result.sort(key=lambda row: (row[0], row[1]))
        return result

    repair_count = 0
    repaired_nodes: set[int] = set()
    max_corner_displacement = 0.0
    repair_cap = min(256, max(32, len(curved_boundary_keys)))
    alpha_values = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)

    while True:
        before_rows = _noncertified_scope()
        if not before_rows:
            break
        if repair_count >= repair_cap:
            raise MeshError(
                f"quadratic promotion exhausted {repair_cap} bounded corner repairs"
            )
        _check_cancellation(
            cancellation_check, "quad-first:quadratic-promotion-repair"
        )

        element_id, family, corners, report = before_rows[0]
        if family != "T6":
            raise MeshError(
                "quadratic promotion requires an unqualified Q8 boundary retile "
                f"outside CH2 scope (element {element_id}, status={report.status.value})"
            )

        candidates: list[
            tuple[float, tuple[int, int], int, int, np.ndarray]
        ] = []
        for index in range(3):
            first = corners[index]
            second = corners[(index + 1) % 3]
            opposite = corners[(index + 2) % 3]
            key = (min(first, second), max(first, second))
            owned = boundary_data.get(key)
            if key not in curved_boundary_keys or owned is None:
                continue
            chord = 0.5 * (
                np.asarray(working.nodes[first], dtype=float)
                + np.asarray(working.nodes[second], dtype=float)
            )
            direction = np.asarray(owned[1], dtype=float) - chord
            magnitude = float(np.linalg.norm(direction))
            if magnitude <= scale_tol or opposite in protected_nodes:
                continue
            candidates.append(
                (magnitude, key, int(owned[0]), int(opposite), direction)
            )
        if not candidates:
            raise MeshError(
                "quadratic promotion cannot repair non-certified T6 element "
                f"{element_id}: no movable curved-boundary opposite corner"
            )
        candidates.sort(key=lambda item: (-item[0], item[1], item[3]))

        accepted = False
        for _magnitude, edge_key, source_edge, corner_id, direction in candidates:
            old_position = np.asarray(working.nodes[corner_id], dtype=float).copy()
            incident = tuple(sorted(incident_by_node.get(corner_id, ())))
            for alpha in alpha_values:
                displacement = float(alpha) * direction
                displacement_norm = float(np.linalg.norm(displacement))
                if displacement_norm > 0.5 * h + scale_tol:
                    continue
                candidate = old_position + displacement
                if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
                    continue
                working.nodes[corner_id] = candidate
                if all(
                    _tentative_report(
                        element_specs[incident_key], incident_key[1]
                    ).status
                    is ValidityStatus.CERTIFIED_POSITIVE
                    for incident_key in incident
                ):
                    accepted = True
                    repair_count += 1
                    repaired_nodes.add(corner_id)
                    total_displacement = float(
                        np.linalg.norm(
                            np.asarray(working.nodes[corner_id], dtype=float)
                            - np.asarray(mesh.nodes[corner_id], dtype=float)
                        )
                    )
                    max_corner_displacement = max(
                        max_corner_displacement, total_displacement
                    )
                    break
            if accepted:
                break
            working.nodes[corner_id] = old_position

        if not accepted:
            raise MeshError(
                "quadratic promotion could not repair T6 element "
                f"{element_id} beside an exact curved source edge"
            )
    if any(not np.array_equal(working.nodes[node], mesh.nodes[node])
           for node in interior_anchors):
        raise MeshError("quadratic promotion moved a protected attachment anchor")
    # Allocate exactly one midside per final shell edge after all corner repairs.
    new_nodes = dict(working.nodes)
    next_node = max(new_nodes, default=-1) + 1
    midside_of_edge: dict[tuple[int, int], int] = {}
    for first, second in sorted(edge_keys):
        midpoint = _candidate_midpoint(first, second)
        if midpoint.shape != (3,) or not np.all(np.isfinite(midpoint)):
            raise MeshError(
                f"quadratic promotion produced an invalid midside on edge {(first, second)}"
            )
        midside_of_edge[(first, second)] = next_node
        new_nodes[next_node] = midpoint.copy()
        next_node += 1

    def _mid(first: int, second: int) -> int:
        key = (min(int(first), int(second)), max(int(first), int(second)))
        try:
            return midside_of_edge[key]
        except KeyError as error:
            raise MeshError(
                f"quadratic promotion shell segment {key} is not a final-cell edge"
            ) from error

    new_quads: dict[int, tuple[int, ...]] = {}
    for element_id, body in working.quads.items():
        c0, c1, c2, c3 = (int(node) for node in body[:4])
        new_quads[int(element_id)] = (
            c0, c1, c2, c3,
            _mid(c0, c1), _mid(c1, c2), _mid(c2, c3), _mid(c3, c0),
        )
    new_tris: dict[int, tuple[int, ...]] = {}
    for element_id, body in working.tris.items():
        c0, c1, c2 = (int(node) for node in body[:3])
        new_tris[int(element_id)] = (
            c0, c1, c2, _mid(c0, c1), _mid(c1, c2), _mid(c2, c0),
        )

    new_nodes_of_edge: dict[int, list[int]] = {}
    for edge_id, sequence in working.nodes_of_edge.items():
        chain = [int(node) for node in sequence]
        if not chain:
            new_nodes_of_edge[int(edge_id)] = []
            continue
        expanded = [chain[0]]
        for first, second in zip(chain, chain[1:]):
            expanded.extend((_mid(first, second), second))
        new_nodes_of_edge[int(edge_id)] = expanded

    # Final strict CH1 validity gate on the exact connectivity to be published.
    strict_started = perf_counter()
    for element_id, body in sorted(new_quads.items()):
        coordinates = np.asarray([new_nodes[node] for node in body], dtype=float)
        report = certify_mapping_validity(coordinates, "Q8")
        if report.status is not ValidityStatus.CERTIFIED_POSITIVE:
            raise MeshError(
                f"quadratic promotion final Q8 element {element_id} is "
                f"{report.status.value}"
            )
    for element_id, body in sorted(new_tris.items()):
        coordinates = np.asarray([new_nodes[node] for node in body], dtype=float)
        report = certify_mapping_validity(coordinates, "T6")
        if report.status is not ValidityStatus.CERTIFIED_POSITIVE:
            raise MeshError(
                f"quadratic promotion final T6 element {element_id} is "
                f"{report.status.value}"
            )
    record_quad_stage("strict_validation", perf_counter() - strict_started)

    boundary_records_by_face: dict[int, list[HighOrderBoundaryMidside]] = {int(fid): [] for fid in working.elements_of_face}
    for shell_edge, (source_edge_id, _exact_midpoint) in sorted(boundary_data.items()):
        first, second = shell_edge
        _p0, parameter0, _d0 = geometry.closest_edge_point(int(source_edge_id), working.nodes[first])
        _p1, parameter1, _d1 = geometry.closest_edge_point(int(source_edge_id), working.nodes[second])
        lower, upper = sorted((float(parameter0), float(parameter1)))
        midside_id = int(midside_of_edge[shell_edge])
        _near, parameter_mid, residual = geometry.closest_edge_point(int(source_edge_id), new_nodes[midside_id])
        curve_name = type(geometry.edges[int(source_edge_id)].curve).__name__
        curvature_class = (
            "straight" if curve_name == "Straight"
            else "analytic_curved" if curve_name == "Arc"
            else "sampled"
        )
        record = HighOrderBoundaryMidside(
            canonical_edge=(int(source_edge_id), round(lower, 15), round(upper, 15)),
            source_edge_id=int(source_edge_id), station_interval=(float(lower), float(upper)),
            parameter=float(parameter_mid), node_id=midside_id, residual=float(residual),
            curvature_class=curvature_class,
        )
        for owner_face in sorted(owner_faces_by_edge.get(shell_edge, ())):
            boundary_records_by_face.setdefault(int(owner_face), []).append(record)

    face_reports: list[HighOrderGeometryReport] = []
    all_residuals: list[float] = []
    for face_id in sorted(working.elements_of_face):
        domain = face_domains.get(int(face_id))
        if domain is None:
            raise MeshError(f"quadratic promotion lacks a chart domain for face {face_id}")
        face_elements = tuple(int(item) for item in working.elements_of_face[face_id])
        q8_count = sum(element_id in new_quads for element_id in face_elements)
        t6_count = sum(element_id in new_tris for element_id in face_elements)
        face_edges: set[tuple[int, int]] = set()
        for element_id in face_elements:
            body = new_quads.get(element_id)
            corner_count = 4
            if body is None:
                body = new_tris.get(element_id)
                corner_count = 3
            if body is None:
                raise MeshError(f"quadratic face {face_id} lost element {element_id}")
            corners = tuple(int(node) for node in body[:corner_count])
            face_edges.update((min(a, b), max(a, b)) for a, b in zip(corners, corners[1:] + corners[:1]))
        boundary_records = tuple(sorted(boundary_records_by_face.get(int(face_id), ()), key=lambda item: repr(item.canonical_edge)))
        residuals = [float(item.residual) for item in boundary_records]
        interior_edges = sorted(face_edges.difference(boundary_data))
        for shell_edge in interior_edges:
            midpoint = np.asarray(new_nodes[midside_of_edge[shell_edge]], dtype=float)
            projected = np.asarray(domain.lift(domain.project(midpoint)), dtype=float)
            residuals.append(float(np.linalg.norm(projected - midpoint)))
        max_residual = max(residuals, default=0.0)
        all_residuals.extend(residuals)
        cylindrical = isinstance(domain, CylindricalQuadDomain)
        conical = isinstance(domain, ConicalQuadDomain)
        parametric = isinstance(domain, ParametricQuadDomain)
        chart_owned = cylindrical or conical or parametric
        if cylindrical:
            chart_origin = tuple(float(value) for value in domain.lift((0.0, 0.0)))
        elif conical or parametric:
            chart_origin = tuple(float(value) for value in domain.lift(domain.outer_chart[0]))
        else:
            chart_origin = tuple(float(value) for value in domain.origin)
        if parametric:
            surface = geometry.faces[int(face_id)].surface
            geometry_family = (
                "extruded" if isinstance(surface, ExtrudedSurface)
                else "ruled" if isinstance(surface, RuledSurface) else "coons"
            )
        else:
            geometry_family = "conical" if conical else ("cylindrical" if cylindrical else "planar")
        curvature_classes = tuple(sorted({item.curvature_class for item in boundary_records}))
        face_reports.append(HighOrderGeometryReport(
            model_id=str(geometry.model_id), revision=int(geometry.revision), face_id=int(face_id),
            geometry_family=geometry_family, chart_kind=type(domain).__name__,
            chart_origin=chart_origin, boundary_projection="source-edge-parameter-midpoint",
            interior_projection="owner-chart-midpoint" if chart_owned else "chord-midpoint",
            q8_count=int(q8_count), t6_count=int(t6_count),
            certified_elements=int(q8_count + t6_count), total_elements=int(q8_count + t6_count),
            boundary_midsides=boundary_records, edge_curvature_classes=curvature_classes,
            interior_midside_count=len(interior_edges), max_geometry_residual=float(max_residual),
        ))

    unique_boundary = {item.canonical_edge for report in face_reports for item in report.boundary_midsides}
    certificate = HighOrderMeshCertificate(
        status=ValidityStatus.CERTIFIED_POSITIVE, model_id=str(geometry.model_id), revision=int(geometry.revision),
        reports=tuple(face_reports), q8_count=len(new_quads), t6_count=len(new_tris),
        unique_midside_count=len(midside_of_edge), unique_boundary_midside_count=len(unique_boundary),
        max_geometry_residual=max(all_residuals, default=0.0), order="quadratic", target_size=float(h),
        route=str(working.hybrid_diagnostics.get("route", "quad-first")),
    )

    diagnostics = deepcopy(working.hybrid_diagnostics)
    diagnostics["quadratic_promotion"] = {
        "status": "APPLIED",
        "unique_midsides": len(midside_of_edge),
        "exact_boundary_midsides": len(boundary_data),
        "repair_count": repair_count,
        "repaired_corner_nodes": sorted(repaired_nodes),
        "max_corner_displacement": float(max_corner_displacement),
    }

    diagnostics["high_order_geometry"] = certificate.to_dict()

    _check_cancellation(cancellation_check, "quad-first:quadratic-promotion-ready")
    mesh.nodes = new_nodes
    mesh.quads = new_quads
    mesh.tris = new_tris
    mesh.nodes_of_edge = new_nodes_of_edge
    mesh.hybrid_diagnostics = diagnostics
    mesh.order = "quadratic"

_QUAD_FIRST_MIN_NORMALIZED_JACOBIAN = 0.05
_QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES = 5.0


def _repair_quad_first_quality(
    mesh, geometry, domains_by_face, cancellation_check=None,
    protected_interior_nodes=(),
    allow_corner_relocation=False,
):
    """Replace a low-quality Q4 by its best certified T3 diagonal.

    Evaluate the would-be Q8/T6 maps on the owner surface before promotion.
    Both linear and quadratic requests therefore make the same deterministic
    corner-topology decision; residual triangles remain first-class cells.
    """
    quad_samples = np.asarray([(a, b) for a in np.linspace(-1., 1., 9)
                               for b in np.linspace(-1., 1., 9)], dtype=float)
    tri_samples = np.asarray([(i/8, j/8) for i in range(9)
                              for j in range(9-i)], dtype=float)
    boundary_midpoints = {}
    for edge_id, chain in mesh.nodes_of_edge.items():
        for first, second in zip(chain, chain[1:]):
            key = (min(first, second), max(first, second))
            _, t0, _ = geometry.closest_edge_point(edge_id, mesh.nodes[first])
            _, t1, _ = geometry.closest_edge_point(edge_id, mesh.nodes[second])
            point = np.asarray(geometry.sample_edge(edge_id, np.asarray([(t0+t1)/2]))[0], dtype=float)
            old = boundary_midpoints.get(key)
            if old is not None and not np.allclose(old, point, atol=1e-12, rtol=0):
                raise MeshError("quad-first quality repair has conflicting source midsides")
            boundary_midpoints[key] = point

    # Quality repair revisits the same corner through many incident cells and
    # candidate moves.  A chart projection rebuilds the face trim, so cache it
    # by node and position; a tentative move automatically invalidates its row.
    chart_positions: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    def node_chart(face_id: int, node_id: int) -> np.ndarray:
        key = (int(face_id), int(node_id))
        position = np.asarray(mesh.nodes[node_id], dtype=float)
        cached = chart_positions.get(key)
        if cached is not None and np.array_equal(cached[0], position):
            return cached[1]
        chart = np.asarray(domains_by_face[face_id].project(position), dtype=float)
        chart_positions[key] = (position.copy(), chart)
        return chart

    def midpoint(face_id, first, second):
        key = (min(first, second), max(first, second))
        owned = boundary_midpoints.get(key)
        if owned is not None:
            return owned
        domain = domains_by_face[face_id]
        a, b = mesh.nodes[first], mesh.nodes[second]
        if isinstance(domain, (CylindricalQuadDomain, ConicalQuadDomain, ParametricQuadDomain)):
            chart_a, chart_b = node_chart(face_id, first), node_chart(face_id, second)
            return np.asarray(domain.lift((.5*(chart_a[0]+chart_b[0]),
                                           .5*(chart_a[1]+chart_b[1]))), dtype=float)
        return .5 * (np.asarray(a) + np.asarray(b))

    sampled_by_nodes: dict[int, tuple[np.ndarray, Any]] = {}

    def trial(face_id, corners, family, samples):
        nodes = np.asarray([mesh.nodes[n] for n in corners] + [
            midpoint(face_id, corners[i], corners[(i+1) % len(corners)])
            for i in range(len(corners))], dtype=float)
        domain = domains_by_face[face_id]
        if isinstance(domain, PlanarQuadDomain):
            normal = np.asarray(domain.normal, dtype=float)
        else:
            centre = np.mean(nodes[:len(corners)], axis=0)
            u, v = geometry.face_local_uv(face_id, centre)
            normal = geometry.face_normal(face_id, u, v)
        sampled = evaluate_mapping(nodes, family, samples, reference_normal=normal)
        quality = float(np.min(sampled.normalized_quality))
        # Keep the exact immutable trial array alive until its normal check.
        # A moved corner creates a new array, so no old mapping is reused.
        sampled_by_nodes[id(nodes)] = (nodes, sampled)
        if len(sampled_by_nodes) > 512:
            sampled_by_nodes.clear()
            sampled_by_nodes[id(nodes)] = (nodes, sampled)
        return quality, nodes, normal

    def normal_error(face_id, nodes, family, samples, reference_normal):
        entry = sampled_by_nodes.get(id(nodes))
        sampled = (entry[1] if entry is not None and entry[0] is nodes else
                   evaluate_mapping(nodes, family, samples,
                                    reference_normal=reference_normal))
        jacobians = np.asarray(sampled.jacobian_vector, dtype=float)
        lengths = np.linalg.norm(jacobians, axis=1)
        if np.any(lengths <= 0.0):
            return float("inf")
        domain = domains_by_face[face_id]
        if isinstance(domain, PlanarQuadDomain):
            owner_normals = np.broadcast_to(np.asarray(domain.normal, dtype=float),
                                            jacobians.shape)
        else:
            batch_uv = getattr(geometry, "face_local_uv_many", None)
            if batch_uv is None:
                # ANYgeometry 0.4.3 did not yet expose this optional batch
                # API. Preserve the scalar owner result on that supported
                # dependency line.
                uv = np.asarray(
                    [geometry.face_local_uv(face_id, point) for point in sampled.points],
                    dtype=float,
                ).reshape((-1, 2))
            else:
                uv = batch_uv(face_id, sampled.points)
            owner_normals = geometry.face_normal_many(face_id, uv)
        cosines = np.einsum("ij,ij->i", jacobians / lengths[:, None], owner_normals)
        return float(np.max(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))))

    repaired = []
    normal_splits = []
    unresolved = []
    relocated_for_split = []
    relocated_quads = []
    protected = {int(node) for chain in mesh.nodes_of_edge.values() for node in chain}
    protected.update(int(node) for node in protected_interior_nodes)
    face_of_cell = {int(element): int(face) for face, elements in mesh.elements_of_face.items()
                    for element in elements}
    next_element = max((*mesh.quads, *mesh.tris), default=-1) + 1
    for face_id, elements in list(mesh.elements_of_face.items()):
        replacement = []
        for element_id in elements:
            body = mesh.quads.get(element_id)
            if body is None:
                replacement.append(element_id)
                continue
            if cancellation_check is not None:
                cancellation_check("quad-first:quality-repair")
            current, current_nodes, current_normal = trial(face_id, body, "Q8", quad_samples)
            current_normal_error = normal_error(
                face_id, current_nodes, "Q8", quad_samples, current_normal)
            needs_normal_repair = current_normal_error > _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES
            if current >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and not needs_normal_repair:
                replacement.append(element_id)
                continue
            if allow_corner_relocation and isinstance(domains_by_face[face_id], PlanarQuadDomain):
                # A boundary-adjacent seed can leave one free quad corner too
                # close to a source arc or re-entrant corner.  Move only that
                # corner, and accept the move only when every incident mapping
                # already meets the full owner-normal and strict certificates.
                domain = domains_by_face[face_id]
                for node in body:
                    if node in protected:
                        continue
                    old = np.asarray(mesh.nodes[node], dtype=float).copy()
                    own_uv = node_chart(face_id, node)
                    local_length = max(float(np.linalg.norm(old-mesh.nodes[other]))
                                       for other in body if other != node)
                    neighborhood = tuple((owner, cell) for owner, cells in
                                         mesh.elements_of_face.items() for cell in cells
                                         if node in (mesh.quads.get(cell) or mesh.tris[cell]))
                    accepted = False
                    for fraction in (.1, .2, .3, .4, .5, .6):
                        if accepted:
                            break
                        for angle in range(16):
                            direction = np.asarray((np.cos(angle*np.pi/8),
                                                    np.sin(angle*np.pi/8)))
                            try:
                                candidate = np.asarray(domain.lift(tuple(
                                    own_uv + fraction*local_length*direction)), dtype=float)
                            except MeshError:
                                continue
                            mesh.nodes[node] = candidate
                            evidence = [(owner, cell, *trial(
                                owner, mesh.quads.get(cell) or mesh.tris[cell],
                                "Q8" if cell in mesh.quads else "T6",
                                quad_samples if cell in mesh.quads else tri_samples))
                                for owner, cell in neighborhood]
                            accepted = all(q >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and
                                normal_error(owner, nodes,
                                             "Q8" if cell in mesh.quads else "T6",
                                             quad_samples if cell in mesh.quads else tri_samples,
                                             normal) <= _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES and
                                certify_mapping_validity(
                                    nodes, "Q8" if cell in mesh.quads else "T6",
                                    reference_normal=normal).status
                                is ValidityStatus.CERTIFIED_POSITIVE
                                for owner, cell, q, nodes, normal in evidence)
                            if accepted:
                                relocated_quads.append((face_id, element_id, node,
                                    float(np.linalg.norm(candidate-old)), current))
                                break
                            mesh.nodes[node] = old
                    if accepted:
                        replacement.append(element_id)
                        break
                if accepted:
                    continue
            choices = []
            for pair in (((body[0], body[1], body[2]), (body[0], body[2], body[3])),
                         ((body[0], body[1], body[3]), (body[1], body[2], body[3]))):
                evidence = [trial(face_id, tri, "T6", tri_samples) for tri in pair]
                score = min(item[0] for item in evidence)
                if score < _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN:
                    continue
                if needs_normal_repair and any(
                    normal_error(face_id, nodes, "T6", tri_samples, normal)
                    > _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES
                    for _, nodes, normal in evidence
                ):
                    continue
                if all(certify_mapping_validity(nodes, "T6", reference_normal=normal).status
                       is ValidityStatus.CERTIFIED_POSITIVE for _, nodes, normal in evidence):
                    choices.append((score, pair))
            if not choices and not needs_normal_repair:
                # A Q4 can put three consecutive corners nearly on a source
                # boundary. Its Q8 Jacobian then approaches zero at the middle
                # corner regardless of where the fourth corner is moved. A
                # diagonal split is viable only after the free corner moves
                # far enough into the owner chart. Try that move transactionally
                # against both proposed T6 cells and every incident neighbor.
                domain = domains_by_face[face_id]
                pairs = (((body[0], body[1], body[2]), (body[0], body[2], body[3])),
                         ((body[0], body[1], body[3]), (body[1], body[2], body[3])))
                for pair in pairs:
                    if choices:
                        break
                    common = set(pair[0]) & set(pair[1])
                    for node in sorted(common - protected):
                        old = np.asarray(mesh.nodes[node], dtype=float).copy()
                        own_uv = node_chart(face_id, node)
                        other_uv = np.mean([node_chart(face_id, other)
                                            for other in body if other != node], axis=0)
                        direction = own_uv - other_uv
                        if float(np.linalg.norm(direction)) <= 1e-14:
                            continue
                        local_length = max(float(np.linalg.norm(old-mesh.nodes[other]))
                                           for other in body if other != node)
                        for alpha in (1.0, 2.0, 4.0, 8.0):
                            try:
                                candidate = np.asarray(
                                    domain.lift(tuple(own_uv + alpha*direction)), dtype=float)
                            except MeshError:
                                continue
                            movement = float(np.linalg.norm(candidate-old))
                            if movement > .25*local_length:
                                continue
                            mesh.nodes[node] = candidate
                            evidence = [trial(face_id, tri, "T6", tri_samples)
                                        for tri in pair]
                            score = min(item[0] for item in evidence)
                            acceptable = score >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and all(
                                certify_mapping_validity(nodes, "T6", reference_normal=normal).status
                                is ValidityStatus.CERTIFIED_POSITIVE and
                                normal_error(face_id, nodes, "T6", tri_samples, normal)
                                <= _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES
                                for _, nodes, normal in evidence)
                            if acceptable:
                                for neighbor, neighbor_body in (*mesh.quads.items(), *mesh.tris.items()):
                                    if neighbor == element_id or node not in neighbor_body:
                                        continue
                                    neighbor_face = face_of_cell[neighbor]
                                    is_quad = neighbor in mesh.quads
                                    q, nodes, normal = trial(
                                        neighbor_face, neighbor_body,
                                        "Q8" if is_quad else "T6",
                                        quad_samples if is_quad else tri_samples)
                                    if (q < _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN or
                                        normal_error(neighbor_face, nodes,
                                                     "Q8" if is_quad else "T6",
                                                     quad_samples if is_quad else tri_samples,
                                                     normal) > _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES or
                                        certify_mapping_validity(
                                            nodes, "Q8" if is_quad else "T6",
                                            reference_normal=normal).status
                                            is not ValidityStatus.CERTIFIED_POSITIVE):
                                        acceptable = False
                                        break
                            if acceptable:
                                choices.append((score, pair))
                                relocated_for_split.append(
                                    (face_id, element_id, node, movement, score))
                                break
                            mesh.nodes[node] = old
                        if choices:
                            break
            if not choices:
                unresolved.append((face_id, element_id, current))
                replacement.append(element_id)
                continue
            _, pair = max(choices, key=lambda item: item[0])
            del mesh.quads[element_id]
            mesh.tris[element_id] = pair[0]
            mesh.tris[next_element] = pair[1]
            face_of_cell[next_element] = face_id
            replacement.extend((element_id, next_element))
            repaired.append((face_id, element_id, next_element, current))
            if needs_normal_repair:
                normal_splits.append((face_id, element_id, current_normal_error))
            next_element += 1
        mesh.elements_of_face[face_id] = replacement
    incident = {}
    for face_id, elements in mesh.elements_of_face.items():
        for element_id in elements:
            body = mesh.quads.get(element_id)
            if body is None:
                body = mesh.tris[element_id]
            for node in body:
                incident.setdefault(int(node), set()).add((face_id, element_id))

    def quality(face_id, element_id):
        body = mesh.quads.get(element_id)
        if body is not None:
            return trial(face_id, body, "Q8", quad_samples)
        return trial(face_id, mesh.tris[element_id], "T6", tri_samples)

    gauss, gauss_weights = np.polynomial.legendre.leggauss(8)

    def lengths(nodes, corner_count):
        values = []
        for index in range(corner_count):
            a, b, middle = nodes[index], nodes[(index+1) % corner_count], nodes[corner_count+index]
            tangent = (gauss-.5)[:, None]*a - 2*gauss[:, None]*middle + (gauss+.5)[:, None]*b
            values.append(float(gauss_weights @ np.linalg.norm(tangent, axis=1)))
        return values

    moved = []
    precert_moved = []
    for face_id, elements in mesh.elements_of_face.items():
        domain = domains_by_face[face_id]
        for element_id in elements:
            if element_id not in mesh.tris:
                continue
            before, initial_nodes, initial_normal = quality(face_id, element_id)
            strict_ok = (certify_mapping_validity(
                initial_nodes, "T6", reference_normal=initial_normal).status
                is ValidityStatus.CERTIFIED_POSITIVE)
            if before >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and strict_ok:
                continue
            body = mesh.tris[element_id]
            if allow_corner_relocation and isinstance(domain, PlanarQuadDomain) and not strict_ok:
                # Exact source-arc midsides can invalidate a T6 even when its
                # sampled quality is acceptable. Resolve that before publishing
                # linear corners, so quadratic promotion need not move them.
                precert_done = False
                for index in range(3):
                    first, second = body[index], body[(index+1) % 3]
                    node = body[(index+2) % 3]
                    key = (min(first, second), max(first, second))
                    exact = boundary_midpoints.get(key)
                    if exact is None or node in protected:
                        continue
                    chord = .5*(np.asarray(mesh.nodes[first]) + np.asarray(mesh.nodes[second]))
                    direction = exact-chord
                    if float(np.linalg.norm(direction)) <= 1e-12:
                        continue
                    old = np.asarray(mesh.nodes[node], dtype=float).copy()
                    neighborhood = tuple(sorted(incident[node]))
                    local_length = max(float(np.linalg.norm(old-mesh.nodes[other]))
                                       for other in body if other != node)
                    for alpha in (.5, 1., 2., 3., 4., 6., 8., 12.):
                        candidate = old + alpha*direction
                        movement = float(np.linalg.norm(candidate-old))
                        if movement > .5*local_length:
                            continue
                        mesh.nodes[node] = candidate
                        evidence = [quality(owner, cell) for owner, cell in neighborhood]
                        precert_done = all(q >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and
                            normal_error(owner, nodes, "Q8" if cell in mesh.quads else "T6",
                                quad_samples if cell in mesh.quads else tri_samples,
                                normal) <= _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES and
                            certify_mapping_validity(nodes, "Q8" if cell in mesh.quads else "T6",
                                reference_normal=normal).status is ValidityStatus.CERTIFIED_POSITIVE
                            for (owner, cell), (q, nodes, normal) in zip(neighborhood, evidence))
                        if precert_done:
                            precert_moved.append((face_id, element_id, node, movement))
                            break
                        mesh.nodes[node] = old
                    if precert_done:
                        break
                if precert_done:
                    continue
            for index, node in enumerate(body):
                if node in protected:
                    continue
                others = (body[(index+1) % 3], body[(index+2) % 3])
                old = np.asarray(mesh.nodes[node], dtype=float).copy()
                own_uv = node_chart(face_id, node)
                edge_uv = .5 * (node_chart(face_id, others[0])
                               + node_chart(face_id, others[1]))
                direction = own_uv - edge_uv
                local_length = max(float(np.linalg.norm(old-mesh.nodes[other])) for other in others)
                neighborhood = tuple(sorted(incident[node]))
                neighbours = sorted({other for owner, cell in neighborhood
                                     for other in (mesh.quads.get(cell) or mesh.tris[cell]) if other != node})
                centroid = np.mean([node_chart(face_id, other) for other in neighbours], axis=0)
                accepted = False
                for adjustment in (direction, centroid-own_uv):
                    for alpha in (.1, .25, .5, 1.0):
                        try:
                            candidate = np.asarray(domain.lift(tuple(own_uv + alpha*adjustment)), dtype=float)
                        except MeshError:
                            continue
                        if float(np.linalg.norm(candidate-old)) > .25*local_length:
                            continue
                        mesh.nodes[node] = candidate
                        evidence = [quality(owner, cell) for owner, cell in neighborhood]
                        accepted = all(q >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and
                            normal_error(owner, nodes, "Q8" if cell in mesh.quads else "T6",
                                quad_samples if cell in mesh.quads else tri_samples,
                                normal) <= _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES and
                            certify_mapping_validity(nodes, "Q8" if cell in mesh.quads else "T6",
                                reference_normal=normal).status is ValidityStatus.CERTIFIED_POSITIVE
                            for (owner, cell), (q, nodes, normal) in zip(neighborhood, evidence))
                        if accepted:
                            moved.append((face_id, element_id, node, float(np.linalg.norm(candidate-old))))
                            break
                        mesh.nodes[node] = old
                    if accepted:
                        break
                else:
                    continue
                break
    aspect_moves = []
    for _pass in range(2):
        changed = False
        for face_id, elements in mesh.elements_of_face.items():
            domain = domains_by_face[face_id]
            for element_id in elements:
                body = mesh.quads.get(element_id)
                if body is None:
                    body = mesh.tris[element_id]
                _, nodes, _ = quality(face_id, element_id)
                edge_lengths = lengths(nodes, len(body))
                ratio = max(edge_lengths) / min(edge_lengths)
                if ratio <= 20.0:
                    continue
                shortest = int(np.argmin(edge_lengths))
                ends = (body[shortest], body[(shortest+1) % len(body)])
                for node, other in ((ends[0], ends[1]), (ends[1], ends[0])):
                    if node in protected:
                        continue
                    old = np.asarray(mesh.nodes[node], dtype=float).copy()
                    own_uv = node_chart(face_id, node)
                    other_uv = node_chart(face_id, other)
                    direction = own_uv-other_uv
                    norm = float(np.linalg.norm(direction))
                    if norm <= 1e-14:
                        continue
                    delta = max(edge_lengths)/20.0-min(edge_lengths)
                    neighborhood = tuple(sorted(incident[node]))
                    accepted = False
                    for alpha in (1.0, 1.5, 2.0):
                        try:
                            candidate = np.asarray(domain.lift(tuple(own_uv+alpha*delta*direction/norm)), dtype=float)
                        except MeshError:
                            continue
                        if float(np.linalg.norm(candidate-old)) > .25*max(edge_lengths):
                            continue
                        mesh.nodes[node] = candidate
                        evidence = [quality(owner, cell) for owner, cell in neighborhood]
                        accepted = all(q >= _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN and
                            normal_error(owner, coords, "Q8" if cell in mesh.quads else "T6",
                                quad_samples if cell in mesh.quads else tri_samples,
                                normal) <= _QUAD_FIRST_MAX_NORMAL_ERROR_DEGREES and
                            max(lengths(coords, len(mesh.quads.get(cell) or mesh.tris[cell]))) /
                            min(lengths(coords, len(mesh.quads.get(cell) or mesh.tris[cell]))) <= 20.0 and
                            certify_mapping_validity(coords, "Q8" if cell in mesh.quads else "T6",
                                reference_normal=normal).status is ValidityStatus.CERTIFIED_POSITIVE
                            for (owner, cell), (q, coords, normal) in zip(neighborhood, evidence))
                        if accepted:
                            aspect_moves.append((face_id, element_id, node, float(np.linalg.norm(candidate-old))))
                            changed = True
                            break
                        mesh.nodes[node] = old
                    if accepted:
                        break
        if not changed:
            break
    mesh.hybrid_diagnostics["quad_quality_repair"] = {
        "minimum_normalized_jacobian": _QUAD_FIRST_MIN_NORMALIZED_JACOBIAN,
        "split_quads": repaired,
        "normal_splits": normal_splits,
        "relocated_for_split": relocated_for_split,
        "relocated_quads": relocated_quads,
        "unresolved_quads": unresolved,
        "moved_nodes": moved,
        "precert_moved": precert_moved,
        "aspect_moves": aspect_moves,
        "protected_interior_nodes": tuple(sorted(int(node) for node in protected_interior_nodes)),
    }
    if unresolved:
        # Never publish an element the repair itself classified as failing.
        face_id, element_id, value = unresolved[0]
        raise QuadQualityRejected(
            f"quad-first repair left {len(unresolved)} element(s) below the "
            f"normalized-Jacobian floor {_QUAD_FIRST_MIN_NORMALIZED_JACOBIAN}; "
            f"first: face {face_id} element {element_id} ({value!r})"
        )


def _quad_first_execute(
    geometry: GeometryModel,
    *,
    face_ids: tuple[int, ...],
    target_size: float,
    certification_mode: "CertificationMode",
    options: "QuadMeshingOptions",
    capabilities: "Any",
    order: str = "linear",
    refinements: tuple[Refinement, ...] = (),
    overrides: Mapping[int, int] | None = None,
    layout_policy: str = "existing",
    cancellation_check: "Callable[[str], None] | None" = None,
) -> "HybridMeshResult":
    """Execute the genuine PQ-M1 target-size planar quad route.

    A shared exact boundary-station registry is captured first, then every face
    receives a fresh PQ2 seed consumed by the bounded PQ-M1 front driver.
    Q4 MCF and TinyAD remain available as direct adapters, but are not part of
    this public dataflow and are reported truthfully as NOT_INTEGRATED.
    """
    h = float(target_size)
    if not np.isfinite(h) or h <= 0.0:
        raise MeshError("quad-first target_size must be finite and positive")
    _check_cancellation(cancellation_check, "quad-first:seed")
    quad_face_ids = tuple(sorted(set(int(item) for item in face_ids)))
    if not quad_face_ids:
        raise MeshError("quad-first requires at least one selected face")
    quad_size_field = SizeField(geometry, h, tuple(refinements))

    for face_id in quad_face_ids:
        face = geometry.faces.get(face_id)
        if face is None:
            raise QuadPublicUnsupported(f"unknown quad-first face {face_id}")
    cylinder_face_ids = tuple(
        face_id for face_id in quad_face_ids
        if isinstance(geometry.faces[face_id].surface, Cylinder)
    )
    cylindrical_bindings: dict[int, Any] = {}
    if cylinder_face_ids:
        from ._cylindrical_public import prepare_bindings

        try:
            cylindrical_bindings = timed_quad_call("qualification", prepare_bindings,
                geometry,
                cylinder_face_ids,
                NativeMeshingOptions(point_placement="frontal_delaunay"),
                cancellation_check=cancellation_check,
            )
        except MeshError as exc:
            raise QuadPublicUnsupported(
                "quad-first public route requires planar or owner-qualified "
                f"cylindrical faces: {exc}"
            ) from exc
    domains_list: list[Any] = []
    geometry_family_by_face: dict[int, str] = {}
    try:
        for face_id in quad_face_ids:
            surface = geometry.faces[face_id].surface
            if isinstance(surface, Cylinder):
                binding = cylindrical_bindings.get(face_id)
                if binding is None:
                    raise QuadPublicUnsupported(
                        f"owner cylindrical binding unavailable for face {face_id}"
                    )
                domain = CylindricalQuadDomain.from_binding(
                    geometry, face_id, binding
                )
                family = "cylindrical"
            elif isinstance(surface, Cone):
                domain = ConicalQuadDomain.from_geometry(geometry, face_id)
                family = "conical"
            elif isinstance(surface, ExtrudedSurface):
                # A small curved child may satisfy a corner-only planar check
                # even though its exact owner support remains curved.
                domain = ParametricQuadDomain.from_geometry(geometry, face_id)
                family = "extruded"
            elif isinstance(surface, (RuledSurface, CoonsSurface)):
                try:
                    domain = PlanarQuadDomain.from_geometry(geometry, face_id)
                    family = "planar"
                except MeshError:
                    domain = ParametricQuadDomain.from_geometry(geometry, face_id)
                    family = "ruled" if isinstance(surface, RuledSurface) else "coons"
            else:
                domain = PlanarQuadDomain.from_geometry(geometry, face_id)
                family = "planar"
            domains_list.append(domain)
            geometry_family_by_face[face_id] = family
        domains = tuple(domains_list)
    except MeshError as exc:
        raise QuadPublicUnsupported(str(exc)) from exc
    registry = timed_quad_call("boundary_stations", BoundaryStationRegistry.for_domains,
        geometry, domains, h, size_field=quad_size_field,
        overrides=overrides,
        _independent_refined_counts=not quad_size_field.is_uniform,
        _adaptive_independent_counts=layout_policy == "adaptive",
    )

    point_seed_cuts: dict[int, tuple[float, float]] = {}
    point_attachment_face_id: int | None = None
    if options.quality_model == "shape_jacobian":
        point_attachments = [
            item for item in geometry.attachments.values()
            if str(item.target_kind) == "face"
            and int(item.target_id) in quad_face_ids
            and str(item.kind) == "member_through_face"
            and len(item.target_parameters) == 2
            and all(parameter.is_point and 0.0 < parameter.start < 1.0
                    for parameter in item.target_parameters)
        ]
        if len(point_attachments) == 1:
            item = point_attachments[0]
            owner = next(domain for domain in domains
                         if domain.face_id == int(item.target_id))
            if len(owner.edge_uses) != 4 or owner.hole_edge_uses:
                raise QuadPublicUnsupported(
                    "interior member attachment point seed requires a four-sided owner face without holes"
                )
            owner_edges = {edge for edge, _ in owner.edge_uses}
            # On sufficiently resolved directly connected patches, the same
            # source-chart grid keeps both sides of the point load comparable
            # to the structured Q8 reference. At coarse sizes its mandatory
            # four-cell star would oversample the short chart direction.
            for domain in domains:
                if domain.face_id == owner.face_id or len(domain.edge_uses) != 4 or domain.hole_edge_uses:
                    continue
                if not owner_edges.intersection(edge for edge, _ in domain.edge_uses):
                    continue
                if min(source_chart_axis_lengths(geometry, domain, shortest=True)) >= 4.0 * h:
                    point_seed_cuts[domain.face_id] = (.5, .5)
            point_seed_cuts[int(item.target_id)] = tuple(
                parameter.start for parameter in item.target_parameters
            )
            point_attachment_face_id = int(item.target_id)


    global_nodes: dict[int, np.ndarray] = {}
    node_of_vertex: dict[int, int] = {}
    global_node_of_station: dict[BoundaryStationKey, int] = {}
    station_objects: dict[BoundaryStationKey, Any] = {}
    for edge_id in sorted(registry.chains):
        for station in registry.chain(edge_id, True):
            station_objects.setdefault(station.key, station)
    station_keys = sorted(
        station_objects,
        key=lambda key: (0 if key.kind == "vertex" else 1, int(key.entity_id), int(key.divisions), int(key.ordinal)),
    )
    for global_id, key in enumerate(station_keys):
        station = station_objects[key]
        global_node_of_station[key] = global_id
        global_nodes[global_id] = np.asarray(station.position, dtype=float)
        if key.kind == "vertex":
            node_of_vertex[int(key.entity_id)] = global_id

    quads: dict[int, tuple[int, ...]] = {}
    tris: dict[int, tuple[int, ...]] = {}
    elements_of_face: dict[int, list[int]] = {}
    face_driver: dict[int, dict[str, Any]] = {}
    face_validation: dict[int, dict[str, Any]] = {}
    face_q4: dict[int, dict[str, Any]] = {}
    face_q5: dict[int, dict[str, Any]] = {}
    q5_budget_total = min(int(options.max_local_optimizations), 8)
    q5_budget_remaining = q5_budget_total
    next_node_id = len(global_nodes)
    next_element_id = 0

    for domain in domains:
        _check_cancellation(cancellation_check, "quad-first:face-seed")
        seed = timed_quad_call("seeding", build_planar_quad_seed,
            geometry,
            domain.face_id,
            h,
            domain=domain,
            registry=registry,
            size_field=quad_size_field,
            source_aligned_cut=point_seed_cuts.get(domain.face_id),
            layout_policy=layout_policy,
            cancellation_check=cancellation_check,
        )
        q4_report = timed_quad_call("optimization", optimize_q4_seed_mcf,
            seed.state,
            target_size=h,
            cancellation_check=cancellation_check,
            size_field=quad_size_field,
            domain=domain,
        )
        face_q4[domain.face_id] = q4_report.to_dict()
        driven = timed_quad_call("front", run_planar_quad_driver,
            seed,
            options,
            allow_recovery=True,
            cancellation_check=cancellation_check,
        )
        state = driven.state
        flip_report = timed_quad_call("front", improve_residual_triangles,
            state, cancellation_check=cancellation_check,
        )
        q5_report = timed_quad_call("optimization", optimize_quad_state,
            state,
            target_size=h,
            max_local_optimizations=q5_budget_remaining,
            cancellation_check=cancellation_check,
            size_field=quad_size_field,
            domain=domain,
        )
        q5_budget_remaining = max(0, q5_budget_remaining - q5_report.worker_calls)
        validation = timed_quad_call("front_validation", validate_planar_quad_result,
            state, face=domain.face_id, reference_area=seed.discrete_area, seed=seed
        )
        face_driver[domain.face_id] = driven.report.to_dict()
        face_driver[domain.face_id]["residual_flips"] = flip_report.to_dict()
        face_driver[domain.face_id]["seed_triangulation_backend"] = (
            seed.triangulation.actual_backend
        )
        if layout_policy == "adaptive":
            face_driver[domain.face_id]["layout_seed"] = {
                "boundary_stations": len(seed.station_to_node),
                "interior_points": len(seed.state.nodes) - len(seed.station_to_node),
                "targeted_reseed_points": seed.targeted_reseed_count,
            }
        q5_face_report = q5_report.to_dict()
        q5_face_report["resident_moved_node_ids"] = list(q5_face_report["moved_node_ids"])
        face_q5[domain.face_id] = q5_face_report
        face_validation[domain.face_id] = validation.to_dict()

        local_to_global: dict[int, int] = {}
        station_for_local = {
            int(local): key for key, local in seed.station_to_node.items()
        }
        for local_node in sorted(int(node) for node in state.nodes):
            key = station_for_local.get(local_node)
            lifted = np.asarray(domain.lift(state.position(local_node)), dtype=float)
            if key is not None:
                global_id = global_node_of_station.get(key)
                if global_id is None:
                    raise MeshError(f"boundary station {key!r} was not preallocated")
                existing = global_nodes[global_id]
                scale = max(
                    1.0,
                    float(np.max(np.abs(existing))),
                    float(np.max(np.abs(lifted))),
                )
                if not np.allclose(
                    existing, lifted, rtol=0.0, atol=1.0e-10 * scale
                ):
                    raise MeshError(
                        f"shared boundary station {key!r} has inconsistent lifted coordinates"
                    )
                local_to_global[local_node] = global_id
            else:
                global_id = next_node_id
                next_node_id += 1
                global_nodes[global_id] = lifted
                local_to_global[local_node] = global_id

        face_q5[domain.face_id]["moved_node_ids"] = [
            local_to_global[int(node)]
            for node in face_q5[domain.face_id]["resident_moved_node_ids"]
        ]

        face_elements: list[int] = []
        for local_cell in sorted(int(cid) for cid in state.cells):
            body = tuple(
                local_to_global[int(node)] for node in state.cell(local_cell)
            )
            element_id = next_element_id
            next_element_id += 1
            kind = state.cell_kind(local_cell)
            if kind == "Q4":
                quads[element_id] = body
            elif kind == "T3":
                tris[element_id] = body
            else:
                raise MeshError(
                    f"quad-first final cell {local_cell} has kind {kind!r}"
                )
            face_elements.append(element_id)
        elements_of_face[domain.face_id] = face_elements
        domain.assert_current(geometry)

    nodes_of_edge: dict[int, list[int]] = {}
    for edge_id in sorted(registry.chains):
        chain_nodes: list[int] = []
        for station in registry.chain(edge_id, True):
            global_id = global_node_of_station.get(station.key)
            if global_id is None:
                raise MeshError(
                    f"canonical boundary station {station.key!r} was not published"
                )
            chain_nodes.append(global_id)
        nodes_of_edge[int(edge_id)] = chain_nodes

    mesh = Mesh(
        geometry_model_id=getattr(geometry, "model_id", None),
        geometry_revision=getattr(geometry, "revision", None),
        nodes=global_nodes,
        quads=quads,
        tris=tris,
        node_of_vertex=node_of_vertex,
        nodes_of_edge=nodes_of_edge,
        elements_of_face=elements_of_face,
        order="linear",
    )
    protected_interior_nodes = ()
    if options.quality_model == "shape_jacobian":
        if point_attachment_face_id is not None:
            anchor_xyz = np.asarray(geometry.face_point(
                point_attachment_face_id,
                *point_seed_cuts[point_attachment_face_id],
            ), dtype=float)
            owner_elements = mesh.elements_of_face[point_attachment_face_id]
            owner_corners = {
                int(node) for element_id in owner_elements
                for node in mesh.corners_of(element_id)
            }
            anchors = tuple(node for node in owner_corners
                            if float(np.linalg.norm(mesh.nodes[node] - anchor_xyz)) <= 1e-10)
            if len(anchors) != 1:
                raise MeshError("quad-first point seed did not publish one exact attachment anchor")
            protected_interior_nodes = anchors
        timed_quad_call("repair", _repair_quad_first_quality,
            mesh, geometry, {domain.face_id: domain for domain in domains},
            cancellation_check=cancellation_check,
            protected_interior_nodes=protected_interior_nodes,
            allow_corner_relocation=layout_policy == "adaptive",
        )
    for sheet_id, sheet in geometry.sheets.items():
        mesh.elements_of_sheet[int(sheet_id)] = sorted(
            {
                int(element_id)
                for face_use_id in sheet.face_use_ids
                for element_id in mesh.elements_of_face.get(
                    int(geometry.face_uses[face_use_id].face_id), ()
                )
            }
        )
    mesh.declared_plate_junction_edges = _topology_plate_junction_edges(
        mesh, geometry
    )
    mesh.hybrid_diagnostics["high_order_geometry"] = {"status": "NOT_APPLICABLE", "reports": []}
    if order == "quadratic":
        adaptive_corners = ({node: np.asarray(position, dtype=float).copy()
                             for node, position in mesh.nodes.items()}
                            if layout_policy == "adaptive" else None)
        timed_quad_call("promotion_including_validation", _promote_quad_first_quadratic,
            mesh,
            geometry,
            target_size=h,
            domains_by_face={domain.face_id: domain for domain in domains},
            cancellation_check=cancellation_check,
            protected_interior_nodes=protected_interior_nodes,
        )
        if adaptive_corners is not None and any(
            not np.array_equal(mesh.nodes[node], position)
            for node, position in adaptive_corners.items()
        ):
            raise MeshError(
                "adaptive quad-first quadratic promotion changed linear corner coordinates"
            )
    q4_reports = tuple(face_q4.values())
    q4_statuses = tuple(str(item["status"]) for item in q4_reports)
    if any(status == "APPLIED" for status in q4_statuses):
        q4_status = "APPLIED"
    elif any(status == "UNAVAILABLE_SKIPPED" for status in q4_statuses):
        q4_status = "UNAVAILABLE_SKIPPED"
    elif any(status == "NO_MATCH" for status in q4_statuses):
        q4_status = "NO_MATCH"
    else:
        q4_status = "NO_ELIGIBLE"
    q4_diagnostics = {
        "status": q4_status,
        "candidate_components": sum(int(item["candidate_components"]) for item in q4_reports),
        "eligible_components": sum(int(item["eligible_components"]) for item in q4_reports),
        "solved_components": sum(int(item["solved_components"]) for item in q4_reports),
        "worker_calls": sum(int(item["worker_calls"]) for item in q4_reports),
        "applied_pairs": sum(int(item["applied_pairs"]) for item in q4_reports),
        "initial_t3": sum(int(item["initial_t3"]) for item in q4_reports),
        "final_t3": sum(int(item["final_t3"]) for item in q4_reports),
        "initial_q4": sum(int(item["initial_q4"]) for item in q4_reports),
        "final_q4": sum(int(item["final_q4"]) for item in q4_reports),
        "skipped_large": sum(int(item["skipped_large"]) for item in q4_reports),
        "skipped_unbalanced": sum(int(item["skipped_unbalanced"]) for item in q4_reports),
        "skipped_nonbipartite": sum(int(item["skipped_nonbipartite"]) for item in q4_reports),
        "skipped_infeasible": sum(int(item["skipped_infeasible"]) for item in q4_reports),
        "skipped_component_cap": sum(int(item["skipped_component_cap"]) for item in q4_reports),
        "component_sizes": [
            int(value)
            for item in q4_reports
            for value in item["component_sizes"]
        ],
        "arc_counts": [
            int(value)
            for item in q4_reports
            for value in item["arc_counts"]
        ],
        "added_q4_ids_by_face": {
            int(face_id): list(item["added_q4_ids"])
            for face_id, item in face_q4.items()
        },
        "generation_by_face": {
            int(face_id): {
                "before": int(item["generation_before"]),
                "after": int(item["generation_after"]),
            }
            for face_id, item in face_q4.items()
        },
        "selected_pairs_by_face": {
            int(face_id): list(item["selected_pairs"])
            for face_id, item in face_q4.items()
        },
        "total_cost": sum(int(item["total_cost"]) for item in q4_reports),
        "component_bounds": (
            list(q4_reports[0]["component_bounds"]) if q4_reports else []
        ),
        "faces": face_q4,
    }

    q5_reports = tuple(face_q5.values())
    q5_statuses = tuple(str(item["status"]) for item in q5_reports)
    if any(status == "APPLIED" for status in q5_statuses):
        q5_status = "APPLIED"
    elif q5_statuses and all(status == "DISABLED" for status in q5_statuses):
        q5_status = "DISABLED"
    elif any(status == "UNAVAILABLE_SKIPPED" for status in q5_statuses):
        q5_status = "UNAVAILABLE_SKIPPED"
    elif any(status == "NOIMPROVE" for status in q5_statuses):
        q5_status = "NOIMPROVE"
    else:
        q5_status = "NO_ELIGIBLE"
    q5_diagnostics = {
        "status": q5_status,
        "eligible_nodes": sum(int(item["eligible_nodes"]) for item in q5_reports),
        "attempts": sum(int(item["attempts"]) for item in q5_reports),
        "applied": sum(int(item["applied"]) for item in q5_reports),
        "worker_calls": sum(int(item["worker_calls"]) for item in q5_reports),
        "objective_initial_sum": sum(float(item["objective_initial_sum"]) for item in q5_reports),
        "objective_final_sum": sum(float(item["objective_final_sum"]) for item in q5_reports),
        "moved_node_ids": [node for item in q5_reports for node in item["moved_node_ids"]],
        "max_displacement": max((float(item["max_displacement"]) for item in q5_reports), default=0.0),
        "worker_statuses": [status for item in q5_reports for status in item["worker_statuses"]],
        "budget": q5_budget_total,
        "budget_cap": 8,
        "faces": face_q5,
    }

    mesh.hybrid_diagnostics.update(
        {
            "route": (
                "quad-first-parametric-curved"
                if any(value in {"ruled", "coons"} for value in geometry_family_by_face.values())
                else (
                    "quad-first-conical"
                    if "conical" in geometry_family_by_face.values()
                    else (
                        "quad-first-cylindrical"
                        if "cylindrical" in geometry_family_by_face.values()
                        else "quad-first"
                    )
                )
            ),
            "geometry_family_by_face": dict(geometry_family_by_face),
            "quad_first_api": "public/1",
            "seed_mode": options.seed_mode,
            "layout_policy": layout_policy,
            "orientation": options.orientation,
            "quality_model": options.quality_model,
            "line_search": options.line_search,
            "front": {"faces": face_driver},
            "validation": {"faces": face_validation},
            "q4": q4_diagnostics,
            "q5": q5_diagnostics,
        }
    )

    strategy_by_face = {
        face_id: "quad_first" for face_id in quad_face_ids
    }
    triangulation_backend_by_face = {
        face_id: {"backend": "quad_first", "q4": face_q4[face_id]["status"]}
        for face_id in quad_face_ids
    }
    if capabilities is None:
        capability_dict: dict[str, Any] = {}
    elif getattr(capabilities, "to_dict", None) is not None:
        capability_dict = capabilities.to_dict()
    else:
        capability_dict = dict(capabilities)
    if any(value in {"cylindrical", "conical"} for value in geometry_family_by_face.values()):
        capability_dict["unsupported_scope"] = (
            "unqualified_curved",
            "Q9+",
        )
    provisional = HybridMeshResult(
        mesh=mesh,
        strategy_by_face=strategy_by_face,
        triangulation_backend_by_face=triangulation_backend_by_face,
        preflight=({"capability": capability_dict},),
        connectivity=None,
        audit_report=None,
        certification_mode=certification_mode,
        certifiable=True,
    )
    def _validate(candidate: Any) -> None:
        if not isinstance(candidate, HybridMeshResult):
            raise MeshError(
                "quad-first provisional result is not a HybridMeshResult"
            )
        if not candidate.mesh.quads:
            raise MeshError("quad-first result exposed no quad elements")
        coords = np.asarray(list(candidate.mesh.nodes.values()), dtype=float)
        if coords.size == 0 or not np.all(np.isfinite(coords)):
            raise MeshError(
                "quad-first result exposed a non-finite or empty mesh"
            )
        for domain in domains:
            domain.assert_current(geometry)
        registry.assert_current(geometry)

    return publish_atomically(
        provisional,
        validate=_validate,
        cancellation_check=cancellation_check,
        publish=lambda published: published,
    )

def generate_hybrid_mesh_result(
    geometry: GeometryModel,
    *,
    target_size: float,
    strategy: MeshingStrategy | str = MeshingStrategy.AUTO,
    overrides: Mapping[int, int] | None = None,
    beam_edges: Iterable[int] = (),
    beam_offsets: Mapping[int, float | Sequence[float]] | None = None,
    member_ids: Iterable[int] | None = None,
    face_ids: Iterable[int] | None = None,
    seeding: Seeding | None = None,
    refinements: Iterable[Refinement] = (),
    order: str = "linear",
    recombine: bool = True,
    native_backend: Any = "auto",
    native_options: NativeMeshingOptions | Mapping[str, Any] | None = None,
    structured_options: StructuredMeshingOptions | Mapping[str, Any] | None = None,
    structural_preparation: (
        StructuralPreparationOptions | Mapping[str, Any] | bool | None
    ) = None,
    qualified_s3: bool = False,
    quad_options: QuadMeshingOptions | Mapping[str, Any] | None = None,
    layout_policy: str = "existing",
    quad_face_ids: Iterable[int] | None = None,
    overlap_policy: OverlapPolicy | str = OverlapPolicy.CONNECT_DECLARED,
    mutation_policy: GeometryMutationPolicy | str = GeometryMutationPolicy.READ_ONLY,
    certification_mode: CertificationMode | str = CertificationMode.NONE,
    change_set: Any | None = None,
    audit_policy: Any | None = None,
    cancellation_check: Callable[[str], None] | None = None,
    _native_surface_options: StructuredMeshingOptions | None = None,
    _evaluate_declared_junction_alignment: bool = True,
    _refine_declared_junction_transition: bool = False,
) -> HybridMeshResult:
    """Generate a model-bound mapped/native mesh from an explicit ownership policy.

    ``cancellation_check`` receives diagnostic safe-phase names. Cancellation is
    cooperative, so its latency is bounded by the current uninterrupted phase.
    The default read-only policy protects editable source geometry.  A caller
    selecting ``working_copy`` declares that the supplied model is already an
    isolated mesh-job closure and may be finalized in place.
    """

    phase_seconds: dict[str, float] = {}
    _check_cancellation(cancellation_check, "hybrid generation start")
    target_size = float(target_size)
    if not np.isfinite(target_size) or target_size <= 0.0:
        raise MeshError("target_size must be finite and positive")
    native_options = NativeMeshingOptions.coerce(native_options)
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
    if type(qualified_s3) is not bool:
        raise MeshError("qualified_s3 must be Boolean")
    if layout_policy not in ("existing", "adaptive"):
        raise MeshError("layout_policy must be 'existing' or 'adaptive'")
    if layout_policy == "adaptive" and quad_options is None:
        raise MeshError("adaptive layout requires explicit quad_options")

    # Quad-first narrow dispatch: explicit ``quad_options`` is the only signal
    # that the quad-first contract applies.  Public integration validates the
    # scope (out-of-scope ``order`` raises a typed
    # :class:`QuadPublicUnsupported`).  ``require_workers=False`` keeps the
    # PQ-M1 public route self-contained: no worker probes run, and the
    # capability report truthfully reports both workers as NOT_INTEGRATED.
    # ``None`` leaves the legacy body byte-identical; ``None`` stays legacy.
    # When quad options are present the result is produced by the genuine
    # planar quad driver; residual faces remain on the legacy body and the
    # two routes are merged with deterministic conflict checks.
    _quad_normalized, _quad_capabilities = route_quad_first(
        quad_options, order=order, planar=True, require_workers=False
    )
    if quad_options is not None and _quad_normalized is None:
        raise MeshError("explicit quad_options must not coerce to None")

    source_geometry = geometry
    requested_beam_edges = tuple(int(item) for item in beam_edges)
    requested_member_ids = (
        None if member_ids is None else tuple(int(item) for item in member_ids)
    )
    requested_face_ids = (
        None if face_ids is None else tuple(int(item) for item in face_ids)
    )
    requested_seeding = seeding
    requested_refinements = tuple(refinements)
    refinements = requested_refinements

    preflight_started = perf_counter()
    source_view = GeometryMeshingView(source_geometry)
    source_faces = _face_ids(source_geometry, requested_face_ids)
    source_beams = _member_edges(
        source_view, requested_beam_edges, requested_member_ids
    )
    if not source_faces and not source_beams:
        raise MeshError("nothing to mesh: no faces and no beam edges")

    _quad_face_selector = _quad_first_normalize_face_ids(quad_face_ids)
    if _quad_normalized is not None and _quad_face_selector is not None:
        _selected_face_set = {int(face_id) for face_id in source_faces}
        _quad_selected = frozenset(_quad_face_selector)
        _missing = sorted(_quad_selected - _selected_face_set)
        if _missing:
            raise MeshError(
                "quad_face_ids must be a subset of the selected faces; "
                f"face {_missing[0]} is not selected {sorted(_selected_face_set)}"
            )
    if _quad_normalized is not None:
        _quad_scope_faces = (
            tuple(source_faces)
            if _quad_face_selector is None
            else tuple(_quad_face_selector)
        )
        if order == "quadratic" and source_beams:
            _cylindrical_boundary_edges = {
                int(use.edge)
                for face_id in _quad_scope_faces
                if isinstance(source_geometry.faces[int(face_id)].surface, Cylinder)
                for use in source_geometry.faces[int(face_id)].loop
            }
            _coowned_cylindrical_beams = sorted(
                set(map(int, source_beams)) & _cylindrical_boundary_edges
            )
            if _coowned_cylindrical_beams:
                raise QuadPublicUnsupported(
                    "quadratic cylindrical quad-first does not qualify beam ownership on a source-boundary edge"
                )
            _conical_boundary_edges = {
                int(use.edge)
                for face_id in _quad_scope_faces
                if isinstance(source_geometry.faces[int(face_id)].surface, Cone)
                for use in source_geometry.faces[int(face_id)].loop
            }
            _coowned_conical_beams = sorted(
                set(map(int, source_beams)) & _conical_boundary_edges
            )
            if _coowned_conical_beams:
                raise QuadPublicUnsupported(
                    "quadratic conical quad-first does not qualify beam ownership on a source-boundary edge"
                )
        for _quad_scope_face_id in _quad_scope_faces:
            _surface = getattr(
                source_geometry.faces[int(_quad_scope_face_id)],
                "surface",
                None,
            )
            if _surface is None or isinstance(_surface, Plane):
                continue
            if isinstance(_surface, Cylinder):
                continue
            if isinstance(_surface, Cone):
                continue
            if isinstance(_surface, ExtrudedSurface):
                if order == "quadratic" and source_beams:
                    _boundary_edges = {
                        int(use.edge)
                        for use in source_geometry.faces[int(_quad_scope_face_id)].loop
                    }
                    if set(map(int, source_beams)) & _boundary_edges:
                        raise QuadPublicUnsupported(
                            "quadratic extruded quad-first does not qualify beam ownership on a source-boundary edge"
                        )
                continue
            if isinstance(_surface, (RuledSurface, CoonsSurface)):
                try:
                    PlanarQuadDomain.from_geometry(
                        source_geometry, int(_quad_scope_face_id)
                    )
                except MeshError:
                    if order == "quadratic" and source_beams:
                        _metric_boundary_edges = {
                            int(use.edge)
                            for use in source_geometry.faces[int(_quad_scope_face_id)].loop
                        }
                        _coowned_metric_beams = sorted(
                            set(map(int, source_beams)) & _metric_boundary_edges
                        )
                        if _coowned_metric_beams:
                            raise QuadPublicUnsupported(
                                "quadratic metric-curved quad-first does not qualify beam ownership on a source-boundary edge"
                            )
                    continue
                else:
                    continue
            _corner_vertices = tuple(
                source_geometry.face_corner_vertices(int(_quad_scope_face_id))
            )
            if len(_corner_vertices) != 4:
                raise QuadPublicUnsupported(
                    "quad-first public route currently requires a four-corner planar face"
                )
            _corner_points = [
                np.asarray(source_geometry.vertex_position(vertex_id), dtype=float)
                for vertex_id in _corner_vertices
            ]
            _p0, _p1, _p2, _p3 = _corner_points
            _normal = np.cross(_p1 - _p0, _p3 - _p0)
            _normal_length = float(np.linalg.norm(_normal))
            if _normal_length <= 1.0e-14:
                raise QuadPublicUnsupported(
                    "quad-first public route currently requires a planar face"
                )
            _normal /= _normal_length
            _center = np.asarray(
                source_geometry.face_point(int(_quad_scope_face_id), 0.5, 0.5),
                dtype=float,
            )
            _scale = max(
                max(float(np.linalg.norm(point - _p0)) for point in _corner_points),
                1.0,
            )
            _residual = max(
                abs(float((point - _p0) @ _normal))
                for point in (*_corner_points, _center)
            )
            if _residual > max(1.0e-10, 1.0e-9 * _scale):
                raise QuadPublicUnsupported(
                    "quad-first public route currently requires a planar face"
                )
    if _quad_face_selector is not None and quad_options is None:
        raise MeshError(
            "quad_face_ids requires an explicit quad_options selector"
        )
    if _quad_normalized is not None and _quad_face_selector is not None:
        _selected_face_set = {int(face_id) for face_id in source_faces}
        _quad_selected = frozenset(_quad_face_selector)
        _missing = sorted(_quad_selected - _selected_face_set)
        if _missing:
            raise MeshError(
                "quad_face_ids must be a subset of the selected faces; "
                f"face {_missing[0]} is not selected {sorted(_selected_face_set)}"
            )
        _legacy_selected = sorted(_selected_face_set - _quad_selected)
        _check_cancellation(cancellation_check, "quad-mixed:dispatch")
        quad_result = _quad_first_execute(
            geometry,
            face_ids=tuple(sorted(_quad_selected)),
            target_size=target_size,
            certification_mode=certification_mode,
            options=_quad_normalized,
            capabilities=_quad_capabilities,
            order=order,
            refinements=requested_refinements,
            overrides=overrides,
            layout_policy=layout_policy,
            cancellation_check=cancellation_check,
        )
        _check_cancellation(cancellation_check, "quad-mixed:legacy")
        legacy_result = generate_hybrid_mesh_result(
            source_geometry,
            target_size=target_size,
            strategy=strategy,
            overrides=overrides,
            beam_edges=requested_beam_edges,
            beam_offsets=beam_offsets,
            member_ids=requested_member_ids,
            face_ids=tuple(_legacy_selected),
            seeding=requested_seeding,
            refinements=requested_refinements,
            order=order,
            recombine=recombine,
            native_backend=native_backend,
            native_options=native_options,
            structured_options=structured_options,
            structural_preparation=structural_preparation,
            qualified_s3=qualified_s3,
            overlap_policy=overlap_policy,
            mutation_policy=mutation_policy,
            certification_mode=certification_mode,
            change_set=change_set,
            audit_policy=audit_policy,
            cancellation_check=cancellation_check,
            _native_surface_options=_native_surface_options,
            _evaluate_declared_junction_alignment=_evaluate_declared_junction_alignment,
            _refine_declared_junction_transition=_refine_declared_junction_transition,
        )
        merged = _merge_quad_first_and_legacy(
            source_geometry,
            quad_faces=frozenset(_quad_selected),
            legacy_faces=frozenset(_legacy_selected),
            quad_result=quad_result,
            legacy_result=legacy_result,
        )

        def _validate_merged(candidate: HybridMeshResult) -> None:
            if not candidate.mesh.quads:
                raise MeshError("quad-mixed result exposed no quad elements")
            for face_id in sorted(_selected_face_set):
                if not candidate.mesh.elements_of_face.get(face_id):
                    raise MeshError(
                        f"quad-mixed result has no elements for face {face_id}"
                    )
            coord_arrays = [
                np.asarray(pos, dtype=float)
                for pos in candidate.mesh.nodes.values()
            ]
            if any(
                not np.all(np.isfinite(arr)) for arr in coord_arrays
            ):
                raise MeshError("quad-mixed result exposed a non-finite mesh")

        return publish_atomically(
            merged,
            validate=_validate_merged,
            cancellation_check=cancellation_check,
            publish=lambda published: published,
        )
    if _quad_normalized is not None:
        # Explicit quad-first shell route.  Beam/member ownership is meshed by
        # the established legacy beam path, then connected to the final Q4 shell
        # by StructuralMeshingPipeline; Q3/Q4/Q5 still run exactly once.
        quad_result = _quad_first_execute(
            geometry,
            face_ids=source_faces,
            target_size=target_size,
            certification_mode=certification_mode,
            options=_quad_normalized,
            capabilities=_quad_capabilities,
            order=order,
            refinements=requested_refinements,
            overrides=overrides,
            layout_policy=layout_policy,
            cancellation_check=cancellation_check,
        )
        if not source_beams:
            if qualified_s3:
                quad_result = _quad_first_apply_qualified_s3(
                    quad_result, source_geometry, cancellation_check
                )
            return quad_result

        if order == "quadratic":
            _refuse_curved_beams(source_geometry, source_beams)

        _check_cancellation(cancellation_check, "quad-first:beam-generation")
        beam_result = generate_hybrid_mesh_result(
            source_geometry,
            target_size=target_size,
            strategy=strategy,
            overrides=overrides,
            beam_edges=requested_beam_edges,
            beam_offsets=beam_offsets,
            member_ids=requested_member_ids,
            face_ids=(),
            seeding=requested_seeding,
            refinements=requested_refinements,
            order=order,
            recombine=recombine,
            native_backend=native_backend,
            native_options=native_options,
            structured_options=structured_options,
            structural_preparation=structural_preparation,
            qualified_s3=False,
            overlap_policy=overlap_policy,
            mutation_policy=mutation_policy,
            certification_mode=certification_mode,
            change_set=change_set,
            audit_policy=audit_policy,
            cancellation_check=cancellation_check,
            _native_surface_options=_native_surface_options,
            _evaluate_declared_junction_alignment=_evaluate_declared_junction_alignment,
            _refine_declared_junction_transition=_refine_declared_junction_transition,
        )
        merged = _merge_quad_first_and_legacy(
            source_geometry,
            quad_faces=frozenset(int(face) for face in source_faces),
            legacy_faces=frozenset(),
            quad_result=quad_result,
            legacy_result=beam_result,
        )
        if qualified_s3:
            merged = _quad_first_apply_qualified_s3(
                merged, source_geometry, cancellation_check
            )

        active_sheets, active_members = _active_structural_owners(
            source_view, source_faces, source_beams
        )
        pipeline = StructuralMeshingPipeline(
            source_view,
            overlap_policy=overlap_policy,
            mutation_policy=mutation_policy,
            active_sheet_ids=active_sheets,
            active_member_ids=active_members,
        )
        for element_id in (*merged.mesh.quads, *merged.mesh.tris, *merged.mesh.beams):
            merged.mesh.activity.setdefault(int(element_id), 1.0)
        connectivity = pipeline.apply_connectivity(merged.mesh)
        merged = replace(
            merged,
            preflight=tuple(connectivity.states),
            connectivity=connectivity,
        )

        def _validate_quad_beam(candidate: HybridMeshResult) -> None:
            if not candidate.mesh.quads:
                raise MeshError("quad-first beam result exposed no quad elements")
            if not candidate.mesh.beams:
                raise MeshError("quad-first beam result exposed no beam elements")
            if candidate.connectivity is None or candidate.connectivity.issues:
                raise MeshError("quad-first beam connectivity is incomplete")

        return publish_atomically(
            merged,
            validate=_validate_quad_beam,
            cancellation_check=cancellation_check,
            publish=lambda published: published,
        )

    source_sheets, source_members = _active_structural_owners(
        source_view, source_faces, source_beams
    )
    source_pipeline = StructuralMeshingPipeline(
        source_view,
        overlap_policy=overlap_policy,
        mutation_policy=mutation_policy,
        active_sheet_ids=source_sheets,
        active_member_ids=source_members,
    )
    source_preflight = tuple(source_pipeline.preflight())
    blocked = _blocked_preflight(source_preflight)
    if blocked:
        detail = "; ".join(str(item) for item in blocked[:5])
        raise MeshError(f"structural meshing preflight blocked generation: {detail}")

    preparation_started = perf_counter()
    reuse_prepared_working_copy = (
        mutation_policy is GeometryMutationPolicy.WORKING_COPY
    )
    geometry, preparation_report = prepare_structural_closure(
        source_geometry,
        face_ids=source_faces,
        beam_edges=source_beams,
        options=structural_preparation,
        cancellation_check=cancellation_check,
        reuse_working_copy=reuse_prepared_working_copy,
    )
    phase_seconds["structural_preparation"] = perf_counter() - preparation_started
    if preparation_report is None:
        source_to_prepared_faces = {
            face_id: (face_id,) for face_id in source_geometry.faces
        }
        source_to_prepared_edges = {
            edge_id: (edge_id,) for edge_id in source_geometry.edges
        }
    else:
        source_to_prepared_faces = dict(
            preparation_report.source_to_working_faces
        )
        source_to_prepared_edges = dict(
            preparation_report.source_to_working_edges
        )
    prepared_faces = tuple(
        dict.fromkeys(
            face_id
            for source_face in source_faces
            for face_id in source_to_prepared_faces[source_face]
        )
    )
    prepared_beams = tuple(
        dict.fromkeys(
            edge_id
            for source_edge in source_beams
            for edge_id in source_to_prepared_edges[source_edge]
        )
    )
    prepared_overrides = _remap_edge_divisions(
        source_geometry,
        geometry,
        source_to_prepared_edges,
        overrides,
    )
    if seeding is not None and any(
        len(source_to_prepared_edges[edge_id]) != 1
        for edge_id in seeding.divisions
        if edge_id in source_to_prepared_edges
    ):
        raise MeshError(
            "a precomputed Seeding cannot be reused after immutable structural "
            "preparation splits a source edge; retain the edge pins and regenerate"
        )

    structured_report: StructuredLayoutReport | None = None
    if structured_options is not None and strategy is MeshingStrategy.NATIVE:
        raise MeshError(
            "structured_options cannot be combined with strategy='native'; "
            "choose 'auto' for quality-gated structured fallback or 'mapped' "
            "to require mapped blocks"
        )
    if structured_options is not None and prepared_faces:
        structured_started = perf_counter()
        plan = plan_structured_layout(
            geometry,
            target_size=target_size,
            face_ids=prepared_faces,
            options=structured_options,
            explicit_seeding=seeding is not None,
            overrides=prepared_overrides,
            protected_edge_ids=prepared_beams,
            allowed_non_manifold_edge_ids=tuple(
                sorted(
                    set(_geometry_plate_junction_edge_ids(geometry))
                    | (
                        set()
                        if preparation_report is None
                        else set(preparation_report.declared_face_connection_edges)
                    )
                )
            ),
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
            geometry,
            plan,
            cancellation_check=cancellation_check,
        )
        phase_seconds["structured_planning_and_application"] = (
            perf_counter() - structured_started
        )

    if structured_report is None:
        prepared_face_strategies = {
            face_id: (
                "mapped"
                if strategy is MeshingStrategy.MAPPED
                or (
                    strategy is MeshingStrategy.AUTO
                    and _mappable(geometry, face_id)
                )
                else "native"
            )
            for face_id in prepared_faces
        }
        prepared_to_final_faces = {
            prepared_face: (prepared_face,)
            for descendants in source_to_prepared_faces.values()
            for prepared_face in descendants
        }
        prepared_to_final_edges = {
            edge_id: (edge_id,) for edge_id in geometry.edges
        }
    else:
        decisions = {
            item.source_face_id: item for item in structured_report.plan.faces
        }
        prepared_face_strategies = {
            prepared_face: (
                "mapped" if decisions[prepared_face].structured else "native"
            )
            for prepared_face in prepared_faces
        }
        prepared_to_final_faces = dict(
            structured_report.source_to_working_faces
        )
        prepared_to_final_edges = dict(
            structured_report.source_to_working_edges
        )

    source_to_final_faces = _compose_descendants(
        source_to_prepared_faces,
        prepared_to_final_faces,
    )
    source_to_final_edges = _compose_descendants(
        source_to_prepared_edges,
        prepared_to_final_edges,
    )
    faces = tuple(
        dict.fromkeys(
            final_face
            for source_face in source_faces
            for final_face in source_to_final_faces[source_face]
        )
    )
    beams = tuple(
        dict.fromkeys(
            final_edge
            for source_edge in source_beams
            for final_edge in source_to_final_edges[source_edge]
        )
    )
    source_strategies: dict[int, str] = {}
    for source_face in source_faces:
        strategies = {
            prepared_face_strategies[prepared_face]
            for prepared_face in source_to_prepared_faces[source_face]
        }
        source_strategies[source_face] = (
            next(iter(strategies)) if len(strategies) == 1 else "mixed"
        )
    mapped_faces = tuple(
        final_face
        for prepared_face in prepared_faces
        if prepared_face_strategies[prepared_face] == "mapped"
        for final_face in prepared_to_final_faces[prepared_face]
    )
    native_faces = tuple(
        final_face
        for prepared_face in prepared_faces
        if prepared_face_strategies[prepared_face] == "native"
        for final_face in prepared_to_final_faces[prepared_face]
    )
    final_overrides = _remap_edge_divisions(
        source_geometry,
        geometry,
        source_to_final_edges,
        overrides,
    )
    final_beam_offsets = _remap_edge_values(
        source_geometry,
        source_to_final_edges,
        beam_offsets,
        label="beam offset",
    )
    refinements = _remap_refinements(
        geometry,
        source_to_final_faces,
        source_to_final_edges,
        refinements,
    )

    view = GeometryMeshingView(geometry)
    def finish_publication(stage: str) -> None:
        # Internal working-copy preparation may advance the source revision.
        # A user callback may not change either publication authority afterward.
        owners = (source_geometry,) if geometry is source_geometry else (source_geometry, geometry)
        bindings = tuple(
            (owner, owner.model_id, owner.revision, owner.tolerance)
            for owner in owners
        )
        _check_cancellation(cancellation_check, stage)
        if not reuse_prepared_working_copy:
            source_view.assert_current()
        view.assert_current()
        for owner, model_id, revision, tolerance in bindings:
            if getattr(owner, "_transaction_journal", None) is not None:
                raise MeshError("cannot publish a mesh during an open geometry transaction")
            if (owner.model_id, owner.revision, owner.tolerance) != (model_id, revision, tolerance):
                raise MeshError("cannot publish a stale geometry-bound mesh after the final callback")
    active_sheets, active_members = _active_structural_owners(view, faces, beams)
    pipeline = StructuralMeshingPipeline(
        view,
        overlap_policy=overlap_policy,
        mutation_policy=mutation_policy,
        active_sheet_ids=active_sheets,
        active_member_ids=active_members,
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
    supplied_seeding = seeding is not None
    size_field = (
        seeding.size_field
        if seeding is not None and seeding.size_field is not None
        else SizeField(geometry, target_size, refinements)
    )
    edges = _active_edges(geometry, faces, beams)
    if seeding is None:
        effective_overrides = dict(final_overrides or {})
        if structured_report is not None:
            mapped_edge_ids = set(_active_edges(geometry, mapped_faces, ()))
            for edge_id, divisions in structured_report.seed_solution.items():
                if edge_id not in mapped_edge_ids:
                    continue
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
            unstructured_face_ids=(native_faces if
                order == "linear" and native_options.point_placement == "frontal_delaunay" else ()),
            maximum_adjacent_growth=(None if structured_report is not None else
                (_native_surface_options.max_element_growth
                 if _native_surface_options is not None else None)),
            cancellation_check=cancellation_check,
        )
    phase_seconds["seeding"] = perf_counter() - seeding_started
    _check_cancellation(cancellation_check, "hybrid seeding complete")

    if mapped_faces or beams:
        mapped_started = perf_counter()
        _check_cancellation(cancellation_check, "mapped generation start")
        mesh = generate_mapped_mesh(
            geometry,
            target_size=target_size,
            overrides=final_overrides,
            beam_edges=beams,
            beam_offsets=final_beam_offsets,
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

    from ._material_region_binding import prepare_material_regions
    material_region_bindings = prepare_material_regions(
        geometry, native_faces, source_to_final_faces, order=order,
        native_options=native_options, supplied_seeding=supplied_seeding,
        edge_overrides=final_overrides or {}, cancellation_check=cancellation_check)
    cancelled_region_edges = {handle.id for binding in material_region_bindings.values()
                              for handle in binding.region.cancelled_seams}
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
        tuple(edge for edge in edges if edge not in cancelled_region_edges),
        seeding,
        size_field,
        order,
    )
    prepared_declared_junction_edges = set(
        _geometry_plate_junction_edge_ids(geometry)
    ) | (
        set()
        if preparation_report is None
        else set(preparation_report.declared_face_connection_edges)
    )
    native_face_set = set(native_faces)
    edge_faces: dict[int, set[int]] = {}
    for selected_face in faces:
        selected = geometry.faces[selected_face]
        for loop in (selected.loop, *selected.holes):
            for oriented in loop:
                edge_faces.setdefault(int(oriented.edge), set()).add(int(selected_face))
    automatically_seeded_shared_edges = frozenset(
        edge_id
        for edge_id, incident_faces in edge_faces.items()
        if (len(incident_faces) > 1 or all(
            isinstance(geometry.faces[face_id].surface, Cone)
            or isinstance(geometry.faces[face_id].surface, ExtrudedSurface)
            for face_id in incident_faces))
        and incident_faces.issubset(native_face_set)
        and edge_id not in set(final_overrides or {})
        and edge_id not in set(beams)
        and isinstance(geometry.edges[edge_id].curve, Straight)
        and not supplied_seeding
        and order == "linear"
    )
    protected_region_constraints = {
        int(path.source_edge) for binding in material_region_bindings.values()
        for path in binding.region.interior_constraints}
    # Physical intervals inside a union have two local incident cells. The
    # boundary-only shared-split propagator cannot represent such a split.
    automatically_seeded_shared_edges = automatically_seeded_shared_edges - protected_region_constraints
    shared_node_reservations = ComponentNodeReservationPool(mesh)
    component_seed_registry = ComponentSeedRegistry(
        _next_identifier(mesh.nodes),
        reservation_pool=shared_node_reservations,
    )
    component_seed_registry._material_region_representatives = {
        face: binding.representative for face, binding in material_region_bindings.items()}
    final_declared_junction_edges = frozenset(
        final_edge
        for prepared_edge in prepared_declared_junction_edges
        for final_edge in prepared_to_final_edges.get(
            int(prepared_edge), (int(prepared_edge),)
        )
    )
    from ._cylindrical_public import component_local_split_edges, prepare_bindings

    cylindrical_bindings = prepare_bindings(
        geometry, native_faces, native_options, cancellation_check
    )
    automatically_seeded_shared_edges = component_local_split_edges(
        geometry, automatically_seeded_shared_edges, cylindrical_bindings
    )
    material_input_eligibility = None
    if material_region_bindings:
        material_edges = {
            int(path.source_edge) for binding in material_region_bindings.values()
            for paths in binding.region.boundaries for path in paths
        } | protected_region_constraints
        edge_terms = {}
        for edge_id in sorted(material_edges):
            incident = edge_faces.get(edge_id, set())
            terms = {
                'shared_or_analytic_supports': bool(incident) and (
                    len(incident) > 1 or all(
                        isinstance(geometry.faces[face].surface, Cone)
                        or isinstance(geometry.faces[face].surface, ExtrudedSurface)
                        for face in incident)),
                'all_native_faces': bool(incident) and incident.issubset(native_face_set),
                'not_overridden': edge_id not in set(final_overrides or {}),
                'no_beam': edge_id not in set(beams),
                'straight_curve': isinstance(geometry.edges[edge_id].curve, Straight),
                'default_seeding': not supplied_seeding,
                'linear_order': order == 'linear',
                'not_region_interior': edge_id not in protected_region_constraints,
            }
            prior_eligible = all(terms.values())
            terms['component_local_allowed'] = (
                edge_id in automatically_seeded_shared_edges if prior_eligible else True)
            terms['selected'] = all(terms.values())
            if terms['selected'] != (edge_id in automatically_seeded_shared_edges):
                raise MeshError('material input eligibility differs from the selected route')
            edge_terms[edge_id] = terms
        material_input_eligibility = {
            'model_id': str(geometry.model_id), 'revision': geometry.revision,
            'edges': edge_terms,
        }
    for face_id in native_faces:
        material_region_binding = material_region_bindings.get(face_id)
        if material_region_binding is not None and face_id != material_region_binding.representative:
            continue
        quadratic_cylinder = order == "quadratic" and face_id in cylindrical_bindings
        face_diagnostics = _mesh_native_face(
            geometry,
            mesh,
            face_id,
            order=order,
            recombine=False if quadratic_cylinder else bool(recombine),
            native_backend=native_backend,
            native_options=NativeMeshingOptions() if quadratic_cylinder else native_options,
            size_field=size_field,
            metric_model_uuid=str(source_geometry.model_id),
            metric_geometry_revision=int(source_geometry.revision),
            boundary_registry=boundary_registry,
            automatically_seeded_shared_edges=automatically_seeded_shared_edges,
            component_seed_registry=component_seed_registry,
            quality_options=(
                _native_surface_options
                if _native_surface_options is not None
                else (
                    None
                    if structured_report is None
                    else structured_report.plan.options
                )
            ),
            declared_junction_edges=final_declared_junction_edges,
            evaluate_declared_junction_alignment=(
                _evaluate_declared_junction_alignment
            ),
            refine_declared_junction_transition=(
                _refine_declared_junction_transition
            ),
            cancellation_check=cancellation_check,
            _cylindrical_binding=None if quadratic_cylinder else cylindrical_bindings.get(face_id),
            _material_region_binding=material_region_binding,
            _material_input_eligibility=material_input_eligibility,
        )
        triangulation_backend_by_face[int(face_id)] = face_diagnostics
        if material_region_binding is not None:
            for consumed in material_region_binding.face_ids:
                triangulation_backend_by_face[consumed] = face_diagnostics
        _check_cancellation(cancellation_check, f"native face {face_id} complete")

    final_cylindrical_repairs = {}
    if order == "linear" and native_faces and cylindrical_bindings:
        from ._shared_triangle_split import repair_all_completed_cylindrical_neighbours
        final_cylindrical_repairs = repair_all_completed_cylindrical_neighbours(
            mesh,geometry,cylindrical_bindings,component_seed_registry,
            cancellation_check=cancellation_check,
            refinement_options=native_options or NativeMeshingOptions())

    if getattr(component_seed_registry, "_deferred_cylindrical_components", None):
        from ._cylindrical_recombine import finalize_components

        finalize_components(geometry, mesh, component_seed_registry, cancellation_check)

    if order == "quadratic" and native_faces and cylindrical_bindings:
        from ._cylindrical_public import finish_quadratic_components

        mesh = finish_quadratic_components(
            geometry, mesh, cylindrical_bindings, boundary_registry,
            native_options=native_options, size_field=size_field,
            quality_options=(_native_surface_options if _native_surface_options is not None
                             else None if structured_report is None else structured_report.plan.options),
            recombine=recombine, backend=native_backend,
            pinned_edges=final_overrides or (), beam_edges=beams,
            declared_junction_edges=final_declared_junction_edges,
            supplied_seeding=supplied_seeding,
            metric_model_uuid=str(source_geometry.model_id),
            metric_geometry_revision=int(source_geometry.revision),
            face_diagnostics=triangulation_backend_by_face,
            cancellation_check=cancellation_check,
        )

    for representative, settings in getattr(component_seed_registry, '_material_region_quality_settings', {}).items():
        _check_cancellation(cancellation_check, f'material region {representative} final quality')
        binding = material_region_bindings[representative]
        triangulation_backend_by_face[representative]['material_region_final_physical_quality'] = (
            binding.certify_published(mesh, boundary_registry, settings, cancellation_check))

    for sheet_id, sheet in geometry.sheets.items():
        element_ids = {
            int(element_id)
            for face_use_id in sheet.face_use_ids
            for element_id in mesh.elements_of_face.get(
                geometry.face_uses[face_use_id].face_id, ()
            )
        }
        mesh.elements_of_sheet[int(sheet_id)] = sorted(element_ids)

    # Qualified-S3 topology admission needs the same explicit junction authority
    # as the later whole-mesh quality gate.  Publish it before S3 preparation;
    # the post-remap assignment below remains the final canonicalization step.
    prepared_mesh_junction_edges = (
        set()
        if preparation_report is None
        else set(
            _prepared_plate_junction_edges(
                mesh,
                preparation_report,
                prepared_to_final_edges,
            )
        )
    )
    mesh.declared_plate_junction_edges = tuple(
        sorted(
            prepared_mesh_junction_edges
            | set(_topology_plate_junction_edges(mesh, geometry))
        )
    )

    qualified_s3_record: dict[str, Any] | None = None
    # Structured/hybrid generation has an established quality fallback below.
    # Do not ask the bounded S3 repairer to qualify a candidate that the
    # mesher already knows it will discard: complex junction transitions can
    # contain hundreds of poor triangles while the deterministic native
    # fallback is admissible without repair.  The recursive fallback has no
    # structured report and therefore reaches the S3 gate normally.
    defer_qualified_s3 = bool(
        qualified_s3
        and structured_report is not None
        and not _structured_quality_report(
            mesh, structured_report.plan.options
        )["accepted"]
    )
    if qualified_s3 and not defer_qualified_s3:
        qualified_s3_started = perf_counter()
        _check_cancellation(
            cancellation_check, "qualified S3 production preparation start"
        )
        mesh, qualified_s3_record = prepare_qualified_s3_mesh(mesh, geometry)
        qualified_s3_record["authority_model"].update(
            {
                "source_model_id": str(source_geometry.model_id),
                "source_revision": int(source_geometry.revision),
            }
        )
        _check_cancellation(
            cancellation_check, "qualified S3 production preparation complete"
        )
        phase_seconds["qualified_s3_preparation"] = (
            perf_counter() - qualified_s3_started
        )

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
    mapped_working_elements = {
        int(element_id)
        for face_id in mapped_faces
        for element_id in mesh.elements_of_face.get(face_id, ())
    }
    if preparation_report is not None:
        mesh.declared_plate_junction_edges = _prepared_plate_junction_edges(
            mesh,
            preparation_report,
            prepared_to_final_edges,
        )
    if preparation_report is not None or structured_report is not None:
        working_backend_diagnostics = triangulation_backend_by_face
        remap_prepared_mesh_associations(
            mesh,
            source_geometry,
            geometry,
            source_to_working_faces=source_to_final_faces,
            source_to_working_edges=source_to_final_edges,
        )
        triangulation_backend_by_face = _source_backend_diagnostics(
            {
                face_id: source_to_final_faces[face_id]
                for face_id in source_faces
            },
            source_strategies,
            working_backend_diagnostics,
        )
        boundary_registry = _published_boundary_registry(source_geometry, mesh)
    mesh.declared_plate_junction_edges = tuple(
        sorted(
            set(mesh.declared_plate_junction_edges)
            | set(_topology_plate_junction_edges(mesh, source_geometry))
        )
    )
    if preparation_report is not None:
        mesh.automatic_intersections = preparation_report.face_connections
        mesh.automatic_beam_connections = len(connectivity.actions)
    if structured_report is not None:
        quality = _structured_quality_report(
            mesh,
            structured_report.plan.options,
        )
        mapped_elements = {
            element_id for element_id in mapped_working_elements
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
                native_options=native_options,
                overrides=overrides,
                beam_edges=requested_beam_edges,
                beam_offsets=beam_offsets,
                member_ids=requested_member_ids,
                face_ids=requested_face_ids,
                seeding=requested_seeding,
                refinements=requested_refinements,
                order=order,
                recombine=recombine,
                native_backend=native_backend,
                structured_options=None,
                structural_preparation=structural_preparation,
                qualified_s3=qualified_s3,
                overlap_policy=overlap_policy,
                mutation_policy=mutation_policy,
                certification_mode=certification_mode,
                change_set=change_set,
                audit_policy=audit_policy,
                cancellation_check=cancellation_check,
                _native_surface_options=structured_report.plan.options,
            )
            fallback_quality = _structured_quality_report(
                fallback.mesh,
                structured_report.plan.options,
            )
            if (
                not fallback_quality["accepted"]
                and _evaluate_declared_junction_alignment
                and fallback.mesh.declared_plate_junction_edges
            ):
                aligned_quality = fallback_quality
                aligned_strategies = {
                    str(working_face_id): working_values.get(
                        "quality_optimization", {}
                    ).get("selected_strategy")
                    for _face_id, values in sorted(
                        fallback.triangulation_backend_by_face.items()
                    )
                    if isinstance(values, Mapping)
                    for working_face_id, working_values in zip(
                        values.get("working_face_ids", ()),
                        values.get("working_face_diagnostics", ()),
                    )
                }
                repaired_mesh, repaired_quality, repair_diagnostics = (
                    _junction_growth_repair(
                        source_geometry,
                        fallback.mesh,
                        aligned_quality,
                        structured_report.plan.options,
                    )
                )
                if repaired_mesh is not None:
                    fallback = replace(fallback, mesh=repaired_mesh)
                    fallback.mesh.hybrid_diagnostics[
                        "alignment_candidate_repaired"
                    ] = {
                        "reason": "declared_junction_growth_transition",
                        "aligned_selected_strategy_by_face": aligned_strategies,
                        "initial_quality": aligned_quality,
                        "accepted_quality": repaired_quality,
                        "repair": dict(repair_diagnostics),
                    }
                    fallback.mesh.hybrid_diagnostics[
                        "junction_growth_repair"
                    ] = dict(repair_diagnostics)
                    fallback_quality = repaired_quality
                else:
                    refined = generate_hybrid_mesh_result(
                        source_geometry,
                        target_size=target_size,
                        strategy=MeshingStrategy.AUTO,
                        native_options=native_options,
                        overrides=overrides,
                        beam_edges=requested_beam_edges,
                        beam_offsets=beam_offsets,
                        member_ids=requested_member_ids,
                        face_ids=requested_face_ids,
                        seeding=requested_seeding,
                        refinements=requested_refinements,
                        order=order,
                        recombine=recombine,
                        native_backend=native_backend,
                        structured_options=None,
                        structural_preparation=structural_preparation,
                        overlap_policy=overlap_policy,
                        mutation_policy=mutation_policy,
                        certification_mode=certification_mode,
                        change_set=change_set,
                        audit_policy=audit_policy,
                        cancellation_check=cancellation_check,
                        _native_surface_options=structured_report.plan.options,
                        _refine_declared_junction_transition=True,
                    )
                    refined_quality = _structured_quality_report(
                        refined.mesh,
                        structured_report.plan.options,
                    )
                    if refined_quality["accepted"]:
                        refined.mesh.hybrid_diagnostics[
                            "alignment_candidate_repaired"
                        ] = {
                            "reason": "local_collar_transition_refinement",
                            "aligned_selected_strategy_by_face": aligned_strategies,
                            "initial_quality": aligned_quality,
                            "accepted_quality": refined_quality,
                        }
                        fallback = refined
                        fallback_quality = refined_quality
                    else:
                        conservative = generate_hybrid_mesh_result(
                            source_geometry,
                            target_size=target_size,
                            strategy=MeshingStrategy.AUTO,
                            native_options=native_options,
                            overrides=overrides,
                            beam_edges=requested_beam_edges,
                            beam_offsets=beam_offsets,
                            member_ids=requested_member_ids,
                            face_ids=requested_face_ids,
                            seeding=requested_seeding,
                            refinements=requested_refinements,
                            order=order,
                            recombine=recombine,
                            native_backend=native_backend,
                            structured_options=None,
                            structural_preparation=structural_preparation,
                            overlap_policy=overlap_policy,
                            mutation_policy=mutation_policy,
                            certification_mode=certification_mode,
                            change_set=change_set,
                            audit_policy=audit_policy,
                            cancellation_check=cancellation_check,
                            _native_surface_options=structured_report.plan.options,
                            _evaluate_declared_junction_alignment=False,
                        )
                        conservative_quality = _structured_quality_report(
                            conservative.mesh,
                            structured_report.plan.options,
                        )
                        if conservative_quality["accepted"]:
                            conservative.mesh.hybrid_diagnostics[
                                "alignment_candidate_rejected"
                            ] = {
                                "reason": "whole_mesh_quality_regression",
                                "aligned_selected_strategy_by_face": (
                                    aligned_strategies
                                ),
                                "aligned_quality": aligned_quality,
                                "refined_aligned_quality": refined_quality,
                                "accepted_baseline_quality": conservative_quality,
                                "comparison": _quality_rejection_details(
                                    refined_quality, conservative_quality
                                ),
                                "repair": dict(repair_diagnostics),
                            }
                            fallback = conservative
                            fallback_quality = conservative_quality
            if not fallback_quality["accepted"]:
                repaired_mesh, repaired_quality, repair_diagnostics = (
                    _junction_growth_repair(
                    source_geometry,
                    fallback.mesh,
                    fallback_quality,
                    structured_report.plan.options,
                    )
                )
                if repaired_mesh is None:
                    fallback_message = _quality_rejection_message(fallback_quality)
                    raise StructuredQualityRejected(
                        f"{message}; automatic native fallback also rejected: "
                        f"{fallback_message}; declared-junction transition repair "
                        "did not produce an accepted candidate"
                    )
                fallback = replace(fallback, mesh=repaired_mesh)
                fallback.mesh.hybrid_diagnostics["junction_growth_repair"] = dict(
                    repair_diagnostics
                )
                fallback_quality = repaired_quality
                _check_cancellation(
                    cancellation_check,
                    "declared-junction transition repair accepted",
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
            fallback_preparation = fallback.structural_preparation
            fallback_faces = (
                {
                    face_id: (face_id,)
                    for face_id in source_geometry.faces
                }
                if fallback_preparation is None
                else dict(fallback_preparation.source_to_working_faces)
            )
            fallback_edges = (
                {
                    edge_id: (edge_id,)
                    for edge_id in source_geometry.edges
                }
                if fallback_preparation is None
                else dict(fallback_preparation.source_to_working_edges)
            )
            fallback_qualified_s3 = fallback.mesh.structural_preparation.get(
                "qualified_s3"
            )
            if qualified_s3 and fallback_qualified_s3 is None:
                # Alignment/refinement is allowed to replace the first native
                # fallback.  Those candidates are intentionally generated
                # without qualified-S3 preparation so they can be compared by
                # the whole-mesh policy first.  Qualify the candidate that was
                # actually selected; never publish an accepted fallback with
                # a missing (or candidate-stale) solver admission record.
                qualified_s3_started = perf_counter()
                _check_cancellation(
                    cancellation_check,
                    "selected fallback qualified S3 preparation start",
                )
                selected_seeding = fallback.mesh.seeding
                qualifying_mesh = mesh_from_dict(mesh_to_dict(fallback.mesh))
                fallback_mesh, fallback_qualified_s3 = prepare_qualified_s3_mesh(
                    qualifying_mesh,
                    source_geometry,
                )
                fallback_mesh.seeding = selected_seeding
                fallback_qualified_s3["authority_model"].update(
                    {
                        "source_model_id": str(source_geometry.model_id),
                        "source_revision": int(source_geometry.revision),
                    }
                )
                _check_cancellation(
                    cancellation_check,
                    "selected fallback qualified S3 policy validation start",
                )
                repaired_fallback_quality = _structured_quality_report(
                    fallback_mesh, structured_report.plan.options
                )
                if not repaired_fallback_quality["accepted"]:
                    raise MeshError(
                        "native fallback violates structured quality policy "
                        "after qualified S3 preparation: "
                        f"{repaired_fallback_quality}"
                    )
                repaired_fallback_metrics = regularity_metrics(
                    fallback_mesh,
                    target_size=target_size,
                    minimum_size_ratio=(
                        structured_report.plan.options.minimum_size_ratio
                    ),
                    maximum_size_ratio=(
                        structured_report.plan.options.maximum_size_ratio
                    ),
                    mapped_element_ids=(),
                )
                _check_cancellation(
                    cancellation_check,
                    "selected fallback qualified S3 policy validation complete",
                )
                structured_report = replace(
                    structured_report,
                    metrics={
                        **dict(structured_report.metrics),
                        "pre_s3_fallback": fallback_metrics,
                        "accepted_fallback": repaired_fallback_metrics,
                    },
                    quality={
                        **dict(structured_report.quality),
                        "pre_s3_fallback": fallback_quality,
                        "accepted_fallback": repaired_fallback_quality,
                    },
                )
                fallback = replace(fallback, mesh=fallback_mesh)
                fallback.mesh.hybrid_diagnostics[
                    "qualified_s3_preparation"
                ] = True
                fallback.mesh.hybrid_diagnostics.setdefault(
                    "phase_seconds", {}
                )["qualified_s3_preparation"] = (
                    perf_counter() - qualified_s3_started
                )
                _check_cancellation(
                    cancellation_check,
                    "selected fallback qualified S3 preparation complete",
                )
            fallback_payload = _preparation_payload(
                fallback_preparation,
                structured_report,
                fallback_faces,
                fallback_edges,
                source_model_id=str(source_geometry.model_id),
            )
            if fallback_qualified_s3 is not None:
                fallback_payload["qualified_s3"] = fallback_qualified_s3
            fallback.mesh.structural_preparation = fallback_payload
            fallback.mesh.hybrid_diagnostics.update(
                {
                    "structured_layout_status": structured_report.status,
                    "structured_plan_hash": structured_report.plan.plan_hash,
                    "structured_quality": structured_report.to_dict()["quality"],
                }
            )
            finish_publication("structured quality fallback accepted")
            return replace(fallback, structured_layout=structured_report)
    if defer_qualified_s3:
        # Reaching this point means the later authoritative quality check did
        # not take the fallback return.  Qualify the retained, now-published
        # mesh against the source-side associations.
        qualified_s3_started = perf_counter()
        _check_cancellation(
            cancellation_check, "qualified S3 production preparation start"
        )
        mesh, qualified_s3_record = prepare_qualified_s3_mesh(
            mesh, source_geometry
        )
        qualified_s3_record["authority_model"].update(
            {
                "source_model_id": str(source_geometry.model_id),
                "source_revision": int(source_geometry.revision),
            }
        )
        _check_cancellation(
            cancellation_check, "qualified S3 production preparation complete"
        )
        phase_seconds["qualified_s3_preparation"] = (
            perf_counter() - qualified_s3_started
        )
    preparation_payload = _preparation_payload(
        preparation_report,
        structured_report,
        source_to_final_faces,
        source_to_final_edges,
        source_model_id=str(source_geometry.model_id),
    )
    if qualified_s3_record is not None:
        preparation_payload["qualified_s3"] = qualified_s3_record
    mesh.structural_preparation = preparation_payload
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
        structural_preparation=preparation_report,
    )
    mesh.hybrid_diagnostics = {
        "requested_target_size": target_size,
        "complex_geometry": {
            "strategy_ladder": [
                "mapped_or_structured",
                "native",
                "detached_decomposition",
                "qualified_gmsh",
            ],
            "component_candidate_budget": 6,
            "source_face_count": len(source_faces),
            "final_face_count": len(faces),
            "declared_junction_edge_count": len(
                final_declared_junction_edges
            ),
            "source_strategy_by_face": {
                str(face_id): source_strategies[face_id]
                for face_id in sorted(source_strategies)
            },
        },
        "strategy_by_face": dict(strategies),
        "triangulation_backend_by_face": {
            int(face_id): _stable_diagnostic_record(values)
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
        "structural_preparation_hash": (
            None
            if preparation_report is None
            else preparation_report.preparation_hash
        ),
        "qualified_s3_preparation": (
            None
            if qualified_s3_record is None
            else {
                "contract_id": qualified_s3_record["contract_id"],
                "element_count": len(qualified_s3_record["element_ids"]),
                "formulation_id": qualified_s3_record["formulation_id"],
                "legacy_fallback": qualified_s3_record["legacy_fallback"],
                "status": qualified_s3_record["status"],
            }
        ),
        "reused_prepared_working_copy": reuse_prepared_working_copy,
        "phase_seconds": dict(phase_seconds),
        "completed_phases": sorted(phase_seconds),
    }
    if final_cylindrical_repairs:
        mesh.hybrid_diagnostics['cylindrical_shared_boundary_final_repair']=final_cylindrical_repairs
    mesh.boundary_registry = boundary_registry
    finish_publication("hybrid generation complete")
    return result


def generate_hybrid_mesh(geometry: GeometryModel, **options: Any) -> Mesh:
    """Return only the neutral mesh from :func:`generate_hybrid_mesh_result`."""

    return generate_hybrid_mesh_result(geometry, **options).mesh
