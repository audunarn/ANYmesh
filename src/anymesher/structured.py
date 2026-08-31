"""Global structure planning and atomic detached geometry application.

Plans are immutable semantic records bound to one exact geometry revision.
Application always works on a clone and partitions only qualified,
straight-edged convex planar rings.  No proximity repair is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from hashlib import sha256
import json
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from anygeometry.curves import Straight
from anygeometry.entities import EntityRef, OrientedEdge
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cylinder, Plane

from .decomposition import check_mappable
from .errors import MeshError
from .seeding import SeedingConflict, solve_seeding

__all__ = [
    "MeshQualityPolicy",
    "SeedEquation",
    "StructurePreference",
    "StructuredBlock",
    "StructuredFacePlan",
    "StructuredInterface",
    "StructuredLayoutPlan",
    "StructuredLayoutReport",
    "StructuredMeshingOptions",
    "apply_structured_layout",
    "commit_structured_layout",
    "plan_structured_layout",
    "regularity_metrics",
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


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class StructurePreference(StrEnum):
    BALANCED = "balanced"
    QUAD_FIRST = "quad_first"
    SIZE_FIRST = "size_first"


@dataclass(frozen=True, slots=True)
class MeshQualityPolicy:
    minimum_scaled_jacobian: float = 0.20
    maximum_aspect_ratio: float = 4.0
    minimum_angle: float = 30.0
    maximum_angle: float = 150.0
    maximum_warpage: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "minimum_scaled_jacobian", "maximum_aspect_ratio",
            "minimum_angle", "maximum_angle", "maximum_warpage",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise MeshError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0.0 < self.minimum_scaled_jacobian <= 1.0:
            raise MeshError("minimum_scaled_jacobian must be in (0, 1]")
        if self.maximum_aspect_ratio < 1.0:
            raise MeshError("maximum_aspect_ratio must be at least 1")
        if not 0.0 < self.minimum_angle < 90.0:
            raise MeshError("minimum_angle must be in (0, 90)")
        if not 90.0 < self.maximum_angle < 180.0:
            raise MeshError("maximum_angle must be in (90, 180)")
        if self.minimum_angle >= self.maximum_angle:
            raise MeshError("minimum_angle must be below maximum_angle")
        if not 0.0 <= self.maximum_warpage <= 1.0:
            raise MeshError("maximum_warpage must be in [0, 1]")

    @classmethod
    def create(cls, value: "MeshQualityPolicy | Mapping[str, Any] | None" = None) -> "MeshQualityPolicy":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise MeshError("quality_policy must be MeshQualityPolicy or a mapping")

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in (
            "minimum_scaled_jacobian", "maximum_aspect_ratio",
            "minimum_angle", "maximum_angle", "maximum_warpage",
        )}


@dataclass(frozen=True, slots=True)
class StructuredMeshingOptions:
    preference: StructurePreference | str = StructurePreference.BALANCED
    allow_detached_partition: bool = True
    max_element_growth: float = 1.25
    minimum_size_ratio: float = 0.67
    maximum_size_ratio: float = 1.50
    maximum_radial_sides: int = 8
    maximum_candidates_per_component: int = 256
    maximum_face_records: int = 100_000
    maximum_blocks: int = 100_000
    maximum_edge_records: int = 200_000
    maximum_estimated_elements: int = 2_000_000
    maximum_divisions_per_edge: int = 100_000
    quality_policy: MeshQualityPolicy | Mapping[str, Any] = field(default_factory=MeshQualityPolicy)

    def __post_init__(self) -> None:
        try:
            preference = StructurePreference(str(self.preference).strip().lower())
        except ValueError as error:
            raise MeshError(f"unknown structure preference {self.preference!r}") from error
        object.__setattr__(self, "preference", preference)
        for name in ("max_element_growth", "minimum_size_ratio", "maximum_size_ratio"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise MeshError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.max_element_growth < 1.0:
            raise MeshError("max_element_growth must be at least 1")
        if self.minimum_size_ratio > 1.0 or self.maximum_size_ratio < 1.0:
            raise MeshError("size ratios must bracket 1")
        for name, minimum in (
            ("maximum_radial_sides", 3),
            ("maximum_candidates_per_component", 1),
            ("maximum_face_records", 1),
            ("maximum_blocks", 1),
            ("maximum_edge_records", 1),
            ("maximum_estimated_elements", 1),
            ("maximum_divisions_per_edge", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < minimum:
                raise MeshError(f"{name} must be an integer of at least {minimum}")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "quality_policy", MeshQualityPolicy.create(self.quality_policy))

    @classmethod
    def create(cls, value: "StructuredMeshingOptions | Mapping[str, Any] | None" = None) -> "StructuredMeshingOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise MeshError("structured_options must be StructuredMeshingOptions or a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "preference": self.preference.value,
            "allow_detached_partition": self.allow_detached_partition,
            "max_element_growth": self.max_element_growth,
            "minimum_size_ratio": self.minimum_size_ratio,
            "maximum_size_ratio": self.maximum_size_ratio,
            "maximum_radial_sides": self.maximum_radial_sides,
            "maximum_candidates_per_component": self.maximum_candidates_per_component,
            "maximum_face_records": self.maximum_face_records,
            "maximum_blocks": self.maximum_blocks,
            "maximum_edge_records": self.maximum_edge_records,
            "maximum_estimated_elements": self.maximum_estimated_elements,
            "maximum_divisions_per_edge": self.maximum_divisions_per_edge,
            "quality_policy": self.quality_policy.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class StructuredBlock:
    block_id: int
    source_face_id: int
    sheet_id: int | None
    component_id: int
    action: str
    surface_family: str
    estimated_elements: int
    boundary_edges: tuple[int, ...] = ()
    output_key: str = ""
    working_face_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("block_id", "source_face_id", "component_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise MeshError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "boundary_edges", tuple(int(item) for item in self.boundary_edges))
        if self.working_face_id is not None:
            object.__setattr__(self, "working_face_id", int(self.working_face_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id, "source_face_id": self.source_face_id,
            "sheet_id": self.sheet_id, "component_id": self.component_id,
            "action": self.action, "surface_family": self.surface_family,
            "estimated_elements": self.estimated_elements,
            "boundary_edges": list(self.boundary_edges), "output_key": self.output_key,
            "working_face_id": self.working_face_id,
        }


@dataclass(frozen=True, slots=True)
class StructuredInterface:
    edge_id: int
    first_face_id: int
    second_face_id: int
    component_id: int
    planned_divisions: int
    conforming: bool
    reason: str = ""
    first_block_id: int | None = None
    second_block_id: int | None = None
    source_edge_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id, "first_face_id": self.first_face_id,
            "second_face_id": self.second_face_id, "component_id": self.component_id,
            "planned_divisions": self.planned_divisions, "conforming": self.conforming,
            "reason": self.reason, "first_block_id": self.first_block_id,
            "second_block_id": self.second_block_id, "source_edge_id": self.source_edge_id,
        }


@dataclass(frozen=True, slots=True)
class SeedEquation:
    face_id: int
    left_edges: tuple[int, ...]
    right_edges: tuple[int, ...]
    relation: str = "equal_sum"
    block_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "left_edges", tuple(int(item) for item in self.left_edges))
        object.__setattr__(self, "right_edges", tuple(int(item) for item in self.right_edges))

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_id": self.face_id, "left_edges": list(self.left_edges),
            "right_edges": list(self.right_edges), "relation": self.relation,
            "block_id": self.block_id,
        }


@dataclass(frozen=True, slots=True)
class StructuredFacePlan:
    source_face_id: int
    action: str
    reason: str
    boundary_sides: int
    estimated_blocks: int = 0
    sheet_id: int | None = None
    component_id: int = 0
    surface_family: str = "unsupported"
    block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_ids", tuple(int(item) for item in self.block_ids))

    @property
    def structured(self) -> bool:
        return self.action in {"mapped", "promote_quad", "radial_partition", "ogrid_partition"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_face_id": self.source_face_id, "action": self.action,
            "reason": self.reason, "boundary_sides": self.boundary_sides,
            "estimated_blocks": self.estimated_blocks, "structured": self.structured,
            "sheet_id": self.sheet_id, "component_id": self.component_id,
            "surface_family": self.surface_family, "block_ids": list(self.block_ids),
        }


@dataclass(frozen=True, slots=True)
class StructuredLayoutPlan:
    model_id: str
    revision: int
    target_size: float
    options: StructuredMeshingOptions
    faces: tuple[StructuredFacePlan, ...]
    blocks: tuple[StructuredBlock, ...] = ()
    interfaces: tuple[StructuredInterface, ...] = ()
    seed_equations: tuple[SeedEquation, ...] = ()
    blockers: tuple[str, ...] = ()
    estimated_element_count: int = 0
    seed_overrides: tuple[tuple[int, int], ...] = ()
    preserves_explicit_seeding: bool = False
    plan_hash: str = ""

    def __post_init__(self) -> None:
        target = float(self.target_size)
        if not isfinite(target) or target <= 0.0:
            raise MeshError("structured plan target_size must be finite and positive")
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "target_size", target)
        object.__setattr__(self, "options", StructuredMeshingOptions.create(self.options))
        object.__setattr__(self, "faces", tuple(self.faces))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "seed_equations", tuple(self.seed_equations))
        object.__setattr__(self, "blockers", tuple(str(item) for item in self.blockers))
        normalized = []
        for edge_id, divisions in self.seed_overrides:
            if isinstance(divisions, bool) or int(edge_id) < 1 or int(divisions) < 1:
                raise MeshError("structured plan seed overrides require positive integers")
            normalized.append((int(edge_id), int(divisions)))
        if len({item[0] for item in normalized}) != len(normalized):
            raise MeshError("structured plan contains duplicate seed overrides")
        object.__setattr__(self, "seed_overrides", tuple(sorted(normalized)))
        object.__setattr__(self, "preserves_explicit_seeding", bool(self.preserves_explicit_seeding))

    @property
    def requires_working_clone(self) -> bool:
        return any(item.action.endswith("partition") or item.action == "promote_quad" for item in self.faces)

    @property
    def structured_source_faces(self) -> tuple[int, ...]:
        return tuple(item.source_face_id for item in self.faces if item.structured)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id, "revision": self.revision,
            "target_size": self.target_size, "options": self.options.to_dict(),
            "requires_working_clone": self.requires_working_clone,
            "faces": [item.to_dict() for item in self.faces],
            "blocks": [item.to_dict() for item in self.blocks],
            "interfaces": [item.to_dict() for item in self.interfaces],
            "seed_equations": [item.to_dict() for item in self.seed_equations],
            "blockers": list(self.blockers),
            "estimated_element_count": self.estimated_element_count,
            "seed_overrides": [list(item) for item in self.seed_overrides],
            "preserves_explicit_seeding": self.preserves_explicit_seeding,
            "plan_hash": self.plan_hash,
        }

    def is_current(self, geometry: GeometryModel) -> bool:
        return str(geometry.model_id) == self.model_id and int(geometry.revision) == self.revision


@dataclass(frozen=True, slots=True)
class StructuredLayoutReport:
    plan: StructuredLayoutPlan
    source_to_working_faces: Mapping[int, tuple[int, ...]]
    source_to_working_edges: Mapping[int, tuple[int, ...]]
    diagnostics: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    seed_solution: Mapping[int, int] = field(default_factory=dict)
    quality: Mapping[str, Any] = field(default_factory=dict)
    status: str = "planned"
    blocks: tuple[StructuredBlock, ...] = ()
    interfaces: tuple[StructuredInterface, ...] = ()
    seed_equations: tuple[SeedEquation, ...] = ()
    working_model_id: str = ""
    working_revision: int = -1
    working_binding: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_to_working_faces", _freeze({
            int(key): tuple(int(item) for item in value)
            for key, value in self.source_to_working_faces.items()
        }))
        object.__setattr__(self, "source_to_working_edges", _freeze({
            int(key): tuple(int(item) for item in value)
            for key, value in self.source_to_working_edges.items()
        }))
        object.__setattr__(self, "diagnostics", tuple(str(item) for item in self.diagnostics))
        object.__setattr__(self, "metrics", _freeze(dict(self.metrics)))
        object.__setattr__(self, "seed_solution", _freeze({
            int(key): int(value) for key, value in self.seed_solution.items()
        }))
        object.__setattr__(self, "quality", _freeze(dict(self.quality)))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "interfaces", tuple(self.interfaces))
        object.__setattr__(self, "seed_equations", tuple(self.seed_equations))
        object.__setattr__(
            self,
            "working_model_id",
            str(self.working_model_id or self.plan.model_id),
        )
        object.__setattr__(
            self,
            "working_revision",
            self.plan.revision
            if int(self.working_revision) < 0
            else int(self.working_revision),
        )
        binding_payload = {
            "plan_hash": self.plan.plan_hash,
            "source_to_working_faces": _thaw(self.source_to_working_faces),
            "source_to_working_edges": _thaw(self.source_to_working_edges),
            "working_revision": self.working_revision,
        }
        binding = "sha256:" + sha256(
            json.dumps(
                binding_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if self.working_binding and self.working_binding != binding:
            raise MeshError("structured report working binding is inconsistent")
        object.__setattr__(self, "working_binding", binding)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "source_to_working_faces": {str(k): list(v) for k, v in sorted(self.source_to_working_faces.items())},
            "source_to_working_edges": {str(k): list(v) for k, v in sorted(self.source_to_working_edges.items())},
            "diagnostics": list(self.diagnostics), "metrics": _thaw(self.metrics),
            "seed_solution": {str(k): int(v) for k, v in sorted(self.seed_solution.items())},
            "quality": _thaw(self.quality), "status": self.status,
            "blocks": [item.to_dict() for item in self.blocks],
            "interfaces": [item.to_dict() for item in self.interfaces],
            "seed_equations": [item.to_dict() for item in self.seed_equations],
            "working_revision": self.working_revision,
            "working_binding": self.working_binding,
        }


class _UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != root:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            low, high = sorted((first_root, second_root))
            self.parent[high] = low


def _surface_family(geometry: GeometryModel, face_id: int) -> str:
    surface = geometry.faces[face_id].surface
    if isinstance(surface, Plane):
        return "plane"
    if isinstance(surface, Cylinder):
        return "cylinder"
    return "unsupported"


def _loop_vertices(geometry: GeometryModel, loop: Sequence[OrientedEdge]) -> tuple[int, ...]:
    return tuple(geometry.oriented_start_vertex(item) for item in loop)


def _face_sheet_membership(geometry: GeometryModel) -> dict[int, tuple[int, ...]]:
    result: dict[int, list[int]] = {}
    for sheet_id, sheet in sorted(geometry.sheets.items()):
        for use_id in sheet.face_use_ids:
            result.setdefault(geometry.face_uses[use_id].face_id, []).append(sheet_id)
    return {key: tuple(sorted(set(value))) for key, value in result.items()}


def _components(
    geometry: GeometryModel,
    selected: Sequence[int],
    membership: Mapping[int, tuple[int, ...]],
    cancellation_check: CancellationCheck | None = None,
) -> dict[int, int]:
    """True union through both Sheets and exact shared topology."""

    selected_set = set(selected)
    union = _UnionFind(selected)
    for sheet_id in sorted({item for values in membership.values() for item in values}):
        faces = [face for face in selected if sheet_id in membership.get(face, ())]
        for face in faces[1:]:
            union.union(faces[0], face)
    for position, edge_id in enumerate(sorted(geometry.edges)):
        if position % 512 == 0:
            _cancel(cancellation_check, "structured component edge scan")
        owners = [face for face in geometry.faces_using_edge(edge_id) if face in selected_set]
        for face in owners[1:]:
            union.union(owners[0], face)
    roots = sorted({union.find(face) for face in selected})
    numbers = {root: index + 1 for index, root in enumerate(roots)}
    return {face: numbers[union.find(face)] for face in selected}


def _plane_projection(
    geometry: GeometryModel, loop: Sequence[OrientedEdge]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(loop) < 3 or any(not isinstance(geometry.edges[item.edge].curve, Straight) for item in loop):
        return None
    points = np.asarray([geometry.vertex_position(item) for item in _loop_vertices(geometry, loop)], dtype=float)
    if len({tuple(point) for point in points}) != len(points):
        return None
    centre = points.mean(axis=0)
    _u, singular, axes = np.linalg.svd(points - centre, full_matrices=False)
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    if len(singular) < 2 or float(singular[1]) <= 1.0e-12 * scale:
        return None
    if len(singular) >= 3 and float(singular[2]) > 1.0e-9 * scale:
        return None
    return points, centre, axes[:2], (points - centre) @ axes[:2].T


def _signed_area(points: np.ndarray) -> float:
    return 0.5 * float(np.sum(
        points[:, 0] * np.roll(points[:, 1], -1)
        - points[:, 1] * np.roll(points[:, 0], -1)
    ))


def _strict_convex(points: np.ndarray) -> bool:
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    tolerance = 1.0e-12 * scale * scale
    turns = []
    for previous, current, following in zip(np.roll(points, 1, axis=0), points, np.roll(points, -1, axis=0)):
        first, second = current - previous, following - current
        turns.append(float(first[0] * second[1] - first[1] * second[0]))
    return bool(turns) and (all(value > tolerance for value in turns) or all(value < -tolerance for value in turns))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = map(float, point)
    inside = False
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        x1, y1 = map(float, first)
        x2, y2 = map(float, second)
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) <= 1.0e-12 and min(x1, x2) - 1.0e-12 <= x <= max(x1, x2) + 1.0e-12 and min(y1, y2) - 1.0e-12 <= y <= max(y1, y2) + 1.0e-12:
            return True
        if (y1 > y) != (y2 > y):
            crossing = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing:
                inside = not inside
    return inside


def _segments_cross(first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]) -> bool:
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    a, b = first
    c, d = second
    values = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return values[0] * values[1] < -1.0e-12 and values[2] * values[3] < -1.0e-12


def _convex_planar_face(geometry: GeometryModel, face_id: int) -> bool:
    face = geometry.faces[face_id]
    if not isinstance(face.surface, Plane) or face.holes:
        return False
    projected = _plane_projection(geometry, face.loop)
    return projected is not None and _strict_convex(projected[3])


def _ogrid_correspondence(geometry: GeometryModel, face_id: int) -> tuple[int, ...] | None:
    """Qualify a planar 4-by-4 annulus and return non-crossing pairing."""

    face = geometry.faces[face_id]
    if (
        not isinstance(face.surface, Plane)
        or len(face.loop) != 4
        or len(face.holes) != 1
        or len(face.holes[0]) != 4
    ):
        return None
    outer = _plane_projection(geometry, face.loop)
    hole = _plane_projection(geometry, face.holes[0])
    if outer is None or hole is None:
        return None
    outer_xyz, centre, axes, outer_uv = outer
    del outer_xyz
    hole_vertices = _loop_vertices(geometry, face.holes[0])
    hole_xyz = np.asarray([geometry.vertex_position(item) for item in hole_vertices], dtype=float)
    hole_uv = (hole_xyz - centre) @ axes.T
    if not _strict_convex(outer_uv) or not _strict_convex(hole_uv):
        return None
    if not all(_point_in_polygon(point, outer_uv) for point in hole_uv):
        return None
    outer_vertices = _loop_vertices(geometry, face.loop)
    candidates: list[tuple[float, tuple[int, ...]]] = []
    for sequence in (hole_vertices, tuple(reversed(hole_vertices))):
        for offset in range(4):
            pairing = sequence[offset:] + sequence[:offset]
            paired_uv = np.asarray([hole_uv[hole_vertices.index(item)] for item in pairing])
            connectors = tuple(zip(outer_uv, paired_uv))
            if any(
                _segments_cross(connectors[first], connectors[second])
                for first in range(4)
                for second in range(first + 1, 4)
            ):
                continue
            if not all(_point_in_polygon(point, outer_uv) for point in 0.5 * (outer_uv + paired_uv)):
                continue
            score = sum(
                float(np.linalg.norm(geometry.vertex_position(a) - geometry.vertex_position(b)))
                for a, b in zip(outer_vertices, pairing)
            )
            candidates.append((score, pairing))
    return None if not candidates else min(candidates, key=lambda item: (item[0], item[1]))[1]


def _planar_area(geometry: GeometryModel, face_id: int) -> float:
    face = geometry.faces[face_id]
    outer = _plane_projection(geometry, face.loop)
    if outer is None:
        return 0.0
    _xyz, centre, axes, outer_uv = outer
    area = abs(_signed_area(outer_uv))
    for loop in face.holes:
        xyz = np.asarray([geometry.vertex_position(item) for item in _loop_vertices(geometry, loop)])
        area -= abs(_signed_area((xyz - centre) @ axes.T))
    return max(0.0, area)


def _structural_blocker(
    geometry: GeometryModel,
    face_id: int,
    membership: Mapping[int, tuple[int, ...]],
    protected_edges: set[int],
    allowed_non_manifold_edges: set[int],
    selected_faces: set[int],
) -> str | None:
    sheets = membership.get(face_id, ())
    if len(sheets) > 1:
        return f"face belongs to multiple structural Sheets {list(sheets)}"
    face = geometry.faces[face_id]
    boundary = {
        item.edge for item in (*face.loop, *(item for loop in face.holes for item in loop))
    }
    for edge_id in sorted(boundary):
        owners = geometry.faces_using_edge(edge_id)
        if len(owners) > 2 and edge_id not in allowed_non_manifold_edges:
            return f"edge {edge_id} is non-manifold with face owners {owners}"
        outside = sorted(set(owners).difference(selected_faces))
        if outside:
            return (
                f"boundary edge {edge_id} also belongs to face(s) {outside} "
                "outside the global planning scope"
            )
        if edge_id in protected_edges:
            return f"boundary edge {edge_id} is protected by caller intent"
        members = geometry.members_using_edge(edge_id)
        if members:
            return f"boundary edge {edge_id} is a structural Member axis {list(members)}"
        attachments = [
            item.id for item in geometry.attachments.values()
            if getattr(item.target_kind, "value", item.target_kind) == "edge"
            and item.target_id == edge_id
        ]
        if attachments:
            return f"boundary edge {edge_id} carries attachments {attachments}"
    attachments = set(geometry.attachments_for_face(face_id))
    attachments.update(geometry.attachments_for_source("face", face_id))
    for sheet_id in sheets:
        attachments.update(geometry.attachments_for_sheet(sheet_id))
    if attachments:
        return f"face/Sheet carries attachments {sorted(attachments)} requiring exact remap"
    sheet_set = set(sheets)
    junctions = [
        item.id for item in geometry.junctions.values()
        if sheet_set.intersection(item.sheet_ids)
        or set(item.attachment_ids).intersection(attachments)
    ]
    if junctions:
        return f"face participates in structural junctions {junctions}"
    for sheet_id in sheets:
        if set(geometry.sheets[sheet_id].declared_non_manifold_edges).intersection(boundary):
            return f"Sheet {sheet_id} declares a non-manifold boundary edge"
    return None


def _plan_hash_payload(plan: StructuredLayoutPlan) -> str:
    payload = plan.to_dict()
    payload.pop("model_id", None)
    payload.pop("revision", None)
    payload.pop("plan_hash", None)
    return "sha256:" + sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def plan_structured_layout(
    geometry: GeometryModel,
    *,
    target_size: float,
    face_ids: Iterable[int] | None = None,
    options: StructuredMeshingOptions | Mapping[str, Any] | None = None,
    explicit_seeding: bool = False,
    overrides: Mapping[int, int] | None = None,
    protected_edge_ids: Iterable[int] = (),
    allowed_non_manifold_edge_ids: Iterable[int] = (),
    cancellation_check: CancellationCheck | None = None,
) -> StructuredLayoutPlan:
    """Plan globally connected structured blocks without editing the model."""

    _cancel(cancellation_check, "structured planning start")
    target = float(target_size)
    if not isfinite(target) or target <= 0.0:
        raise MeshError("target_size must be finite and positive")
    policy = StructuredMeshingOptions.create(options)
    selected = (
        tuple(sorted(geometry.faces))
        if face_ids is None
        else tuple(sorted(dict.fromkeys(int(item) for item in face_ids)))
    )
    missing = [item for item in selected if item not in geometry.faces]
    if missing:
        raise MeshError(f"no face {missing[0]}")
    if len(selected) > policy.maximum_face_records:
        raise MeshError(
            f"structured plan has {len(selected)} face records, exceeding "
            f"maximum_face_records={policy.maximum_face_records}"
        )
    membership = _face_sheet_membership(geometry)
    if len(geometry.edges) > policy.maximum_edge_records:
        raise MeshError(
            f"model has {len(geometry.edges)} edge records; structured planning "
            f"is bounded to {policy.maximum_edge_records}"
        )
    components = _components(
        geometry,
        selected,
        membership,
        cancellation_check,
    )
    selected_set = set(selected)
    normalized_overrides: dict[int, int] = {}
    for edge_id, divisions in (overrides or {}).items():
        if isinstance(divisions, bool) or int(divisions) < 1:
            raise MeshError(f"edge {edge_id} override must be a positive integer")
        normalized_overrides[int(edge_id)] = int(divisions)
    protected = {int(item) for item in protected_edge_ids}
    protected.update(normalized_overrides)
    allowed_non_manifold = {int(item) for item in allowed_non_manifold_edge_ids}
    unknown = sorted((protected | allowed_non_manifold).difference(geometry.edges))
    if unknown:
        raise MeshError(f"protected/overridden edge {unknown[0]} does not exist")

    decisions: list[StructuredFacePlan] = []
    for position, face_id in enumerate(selected):
        if position % 64 == 0:
            _cancel(cancellation_check, "structured face qualification")
        face = geometry.faces[face_id]
        side_count = len(face.loop)
        family = _surface_family(geometry, face_id)
        sheets = membership.get(face_id, ())
        common = {
            "sheet_id": sheets[0] if len(sheets) == 1 else None,
            "component_id": components[face_id],
            "surface_family": family,
        }
        owner_blocker = None
        for item in (*face.loop, *(edge for loop in face.holes for edge in loop)):
            owners = geometry.faces_using_edge(item.edge)
            if len(owners) > 2 and item.edge not in allowed_non_manifold:
                owner_blocker = f"edge {item.edge} is non-manifold with face owners {owners}"
                break
        if len(sheets) > 1:
            owner_blocker = f"face belongs to multiple structural Sheets {list(sheets)}"
        if owner_blocker:
            decisions.append(StructuredFacePlan(face_id, "native", owner_blocker, side_count, **common))
            continue

        if check_mappable(geometry, face_id).ok and not face.holes:
            decisions.append(StructuredFacePlan(
                face_id, "mapped", "existing four-sided mapped parameterization",
                side_count, 1, **common,
            ))
            continue

        blocker = _structural_blocker(
            geometry,
            face_id,
            membership,
            protected,
            allowed_non_manifold,
            selected_set,
        )
        if not policy.allow_detached_partition:
            blocker = "detached partitioning is disabled"
        elif explicit_seeding:
            blocker = "a caller-supplied Seeding must be preserved exactly"
        elif policy.preference is StructurePreference.SIZE_FIRST and side_count != 4:
            blocker = "size-first policy avoids element-adding radial partitions"
        elif family != "plane":
            blocker = (
                "automatic partitioning is qualified only for Plane surfaces; "
                f"{family} remains native"
            )
        if blocker:
            decisions.append(StructuredFacePlan(face_id, "native", blocker, side_count, **common))
            continue

        if face.holes:
            if _ogrid_correspondence(geometry, face_id) is not None:
                decisions.append(StructuredFacePlan(
                    face_id, "ogrid_partition",
                    "qualified straight-edged convex planar four-by-four ring",
                    side_count, 8, **common,
                ))
            else:
                decisions.append(StructuredFacePlan(
                    face_id, "native",
                    "trim is not a qualified straight-edged convex planar four-by-four O-grid",
                    side_count, **common,
                ))
            continue
        if not _convex_planar_face(geometry, face_id):
            decisions.append(StructuredFacePlan(
                face_id, "native",
                "automatic radial partition requires an exact straight-edged convex planar ring",
                side_count, **common,
            ))
        elif side_count == 4:
            decisions.append(StructuredFacePlan(
                face_id, "promote_quad",
                "qualified planar quadrilateral can be promoted to one mapped block",
                side_count, 1, **common,
            ))
        elif 3 <= side_count <= policy.maximum_radial_sides:
            decisions.append(StructuredFacePlan(
                face_id, "radial_partition",
                f"qualified convex planar {side_count}-sided ring forms {side_count} radial blocks",
                side_count, side_count, **common,
            ))
        else:
            decisions.append(StructuredFacePlan(
                face_id, "native",
                f"face has {side_count} sides; radial limit is {policy.maximum_radial_sides}",
                side_count, **common,
            ))

    # Propagate radial partitioning through adjacent promotable quads. This is
    # a fixed point, so reversing the caller's face order cannot change it.
    while True:
        radial_faces = {item.source_face_id for item in decisions if item.action == "radial_partition"}
        changed = False
        converted = []
        for item in decisions:
            if item.action == "promote_quad" and any(
                any(owner in radial_faces for owner in geometry.faces_using_edge(edge.edge))
                for edge in geometry.faces[item.source_face_id].loop
            ):
                item = replace(
                    item, action="radial_partition", estimated_blocks=4,
                    reason="shared radial interface requires a conforming four-block partition",
                )
                changed = True
            converted.append(item)
        decisions = converted
        if not changed:
            break
    radial_faces = {item.source_face_id for item in decisions if item.action == "radial_partition"}
    decisions = [
        replace(
            item, action="native", estimated_blocks=0,
            reason="O-grid boundary is split by an adjacent radial component",
        )
        if item.action == "ogrid_partition" and any(
            any(owner in radial_faces for owner in geometry.faces_using_edge(edge.edge))
            for edge in geometry.faces[item.source_face_id].loop
        ) else item
        for item in decisions
    ]

    bounded = list(decisions)
    for component_id in sorted({item.component_id for item in decisions}):
        candidates = [
            (index, item) for index, item in enumerate(bounded)
            if item.component_id == component_id
            and item.action in {"promote_quad", "radial_partition", "ogrid_partition"}
        ]
        candidates.sort(key=lambda pair: (
            pair[1].estimated_blocks, -_planar_area(geometry, pair[1].source_face_id),
            pair[1].source_face_id,
        ))
        for index, item in candidates[policy.maximum_candidates_per_component:]:
            bounded[index] = replace(
                item, action="native", estimated_blocks=0,
                reason=f"component structured-candidate budget reached ({policy.maximum_candidates_per_component})",
            )
    decisions = bounded
    blocks: list[StructuredBlock] = []
    assigned: list[StructuredFacePlan] = []
    next_block = 1
    estimated_total = 0
    for decision in decisions:
        count = decision.estimated_blocks if decision.structured else 0
        if len(blocks) + count > policy.maximum_blocks:
            raise MeshError(f"structured plan exceeds maximum_blocks={policy.maximum_blocks}")
        area = _planar_area(geometry, decision.source_face_id)
        estimate = (
            max(1, int(np.ceil(area / target**2 / count)))
            if count and area else (1 if count else 0)
        )
        block_ids = tuple(range(next_block, next_block + count))
        next_block += count
        assigned.append(replace(decision, block_ids=block_ids))
        boundary = tuple(item.edge for item in geometry.faces[decision.source_face_id].loop)
        for local, block_id in enumerate(block_ids, start=1):
            blocks.append(StructuredBlock(
                block_id, decision.source_face_id, decision.sheet_id,
                decision.component_id,
                decision.action if decision.structured else "native_residual",
                decision.surface_family, estimate, boundary,
                f"face_{decision.source_face_id}_block_{local}",
            ))
        estimated_total += estimate * count
    decisions = assigned
    if estimated_total > policy.maximum_estimated_elements:
        raise MeshError(
            f"estimated {estimated_total} shell elements exceed "
            f"maximum_estimated_elements={policy.maximum_estimated_elements}"
        )

    decision_by_face = {item.source_face_id: item for item in decisions}
    interfaces: list[StructuredInterface] = []
    for position, edge_id in enumerate(sorted(geometry.edges)):
        if position % 512 == 0:
            _cancel(cancellation_check, "structured source interface scan")
        owners = tuple(face for face in geometry.faces_using_edge(edge_id) if face in selected_set)
        if len(owners) != 2:
            continue
        first, second = owners
        conforming = decision_by_face[first].structured and decision_by_face[second].structured
        interfaces.append(StructuredInterface(
            edge_id, first, second, components[first],
            max(1, int(round(geometry.edge_length(edge_id) / target))), conforming,
            "exact shared topology" if conforming else "native transition on at least one side",
            source_edge_id=edge_id,
        ))

    equations: list[SeedEquation] = []
    for decision in decisions:
        if decision.action not in {"mapped", "promote_quad"}:
            continue
        face = geometry.faces[decision.source_face_id]
        sides = face.sides() if len(face.corners) == 4 else tuple((item,) for item in face.loop)
        if len(sides) == 4:
            equations.extend((
                SeedEquation(
                    decision.source_face_id, tuple(item.edge for item in sides[0]),
                    tuple(item.edge for item in sides[2]), block_id=decision.block_ids[0],
                ),
                SeedEquation(
                    decision.source_face_id, tuple(item.edge for item in sides[1]),
                    tuple(item.edge for item in sides[3]), block_id=decision.block_ids[0],
                ),
            ))
    blockers = tuple(
        f"face {item.source_face_id}: {item.reason}" for item in decisions if not item.structured
    )
    plan = StructuredLayoutPlan(
        str(geometry.model_id), geometry.revision, target, policy, tuple(decisions),
        tuple(blocks), tuple(interfaces), tuple(equations), blockers, estimated_total,
        tuple(sorted(normalized_overrides.items())), bool(explicit_seeding),
    )
    _cancel(cancellation_check, "structured planning complete")
    return replace(plan, plan_hash=_plan_hash_payload(plan))


def _radial_partition(
    geometry: GeometryModel,
    face_id: int,
    original_loop: Sequence[OrientedEdge],
    split_edges: Mapping[int, tuple[int, int, int]],
    centre: np.ndarray,
) -> tuple[int, ...]:
    source_face = geometry.faces[face_id]
    surface = source_face.surface
    parameterization = source_face.parameterization
    metadata = source_face.metadata.to_dict()
    halves: list[tuple[OrientedEdge, OrientedEdge]] = []
    midpoints: list[int] = []
    for item in original_loop:
        midpoint, first, second = split_edges[item.edge]
        midpoints.append(midpoint)
        halves.append(
            (OrientedEdge(first, True), OrientedEdge(second, True))
            if item.forward
            else (OrientedEdge(second, False), OrientedEdge(first, False))
        )
    geometry.remove_face(face_id, record=False)
    centre_vertex = geometry.add_point(*centre)
    spokes = [geometry.add_line(midpoint, centre_vertex) for midpoint in midpoints]
    made: list[int] = []
    for index in range(len(original_loop)):
        previous = (index - 1) % len(original_loop)
        made_face = geometry.add_face_from_loop(
            (
                halves[previous][1], halves[index][0],
                OrientedEdge(spokes[index], True),
                OrientedEdge(spokes[previous], False),
            ),
            (0, 1, 2, 3), surface=surface,
        )
        geometry.set_face_metadata(made_face, metadata)
        geometry.set_face_parameterization(made_face, parameterization)
        made.append(made_face)
    geometry.record_replacement(
        EntityRef("face", face_id), tuple(EntityRef("face", item) for item in made)
    )
    return tuple(made)


def _promote_quad(geometry: GeometryModel, face_id: int) -> tuple[int, ...]:
    face = geometry.faces[face_id]
    if len(face.loop) != 4 or face.holes:
        raise MeshError(f"face {face_id} is no longer a promotable quadrilateral")
    loop, surface = tuple(face.loop), face.surface
    geometry.remove_face(face_id, record=False)
    made = geometry.add_face_from_loop(loop, (0, 1, 2, 3), surface=surface)
    geometry.set_face_metadata(made, face.metadata.to_dict())
    geometry.set_face_parameterization(made, face.parameterization)
    geometry.record_replacement(EntityRef("face", face_id), (EntityRef("face", made),))
    return (made,)


def _ogrid_partition(geometry: GeometryModel, face_id: int) -> tuple[int, ...]:
    face = geometry.faces[face_id]
    pairing = _ogrid_correspondence(geometry, face_id)
    if pairing is None:
        raise MeshError(f"face {face_id} is no longer a qualified planar O-grid")
    outer, hole, surface = tuple(face.loop), tuple(face.holes[0]), face.surface
    outer_vertices = _loop_vertices(geometry, outer)
    hole_edge_by_vertices = {
        frozenset((geometry.edges[item.edge].start, geometry.edges[item.edge].end)): item.edge
        for item in hole
    }
    geometry.remove_face(face_id, record=False)
    ring_vertices = geometry.add_points(tuple(
        0.5 * (geometry.vertex_position(outer_vertex) + geometry.vertex_position(hole_vertex))
        for outer_vertex, hole_vertex in zip(outer_vertices, pairing)
    ))
    ring_edges = [geometry.add_line(ring_vertices[index], ring_vertices[(index + 1) % 4]) for index in range(4)]
    outer_connectors = [geometry.add_line(a, b) for a, b in zip(outer_vertices, ring_vertices)]
    inner_connectors = [geometry.add_line(a, b) for a, b in zip(ring_vertices, pairing)]
    made: list[int] = []
    for index, outer_item in enumerate(outer):
        following = (index + 1) % 4
        hole_edge = hole_edge_by_vertices.get(frozenset((pairing[index], pairing[following])))
        if hole_edge is None:
            raise MeshError("O-grid correspondence does not follow the trim loop")
        rings = (
            (outer_item.edge, outer_connectors[following], ring_edges[index], outer_connectors[index]),
            (ring_edges[index], inner_connectors[following], hole_edge, inner_connectors[index]),
        )
        for edges in rings:
            made_face = geometry.add_face_from_loop(
                geometry.order_loop(edges), (0, 1, 2, 3), surface=surface,
            )
            geometry.set_face_metadata(made_face, face.metadata.to_dict())
            geometry.set_face_parameterization(made_face, face.parameterization)
            made.append(made_face)
    geometry.record_replacement(
        EntityRef("face", face_id), tuple(EntityRef("face", item) for item in made)
    )
    return tuple(made)


def _actual_evidence(
    working: GeometryModel,
    plan: StructuredLayoutPlan,
    face_mapping: Mapping[int, tuple[int, ...]],
    edge_mapping: Mapping[int, tuple[int, ...]],
    cancellation_check: CancellationCheck | None,
) -> tuple[
    tuple[StructuredBlock, ...], tuple[StructuredInterface, ...],
    tuple[SeedEquation, ...], Mapping[int, int], int,
]:
    plan_blocks = {item.block_id: item for item in plan.blocks}
    blocks: list[StructuredBlock] = []
    by_working_face: dict[int, StructuredBlock] = {}
    for decision in plan.faces:
        if not decision.block_ids:
            continue
        descendants = face_mapping[decision.source_face_id]
        if len(descendants) != len(decision.block_ids):
            raise MeshError(
                f"face {decision.source_face_id} produced {len(descendants)} blocks; "
                f"the verified plan requires {len(decision.block_ids)}"
            )
        for block_id, working_face_id in zip(decision.block_ids, descendants):
            source = plan_blocks[block_id]
            actual = replace(
                source,
                boundary_edges=tuple(item.edge for item in working.faces[working_face_id].loop),
                working_face_id=working_face_id,
            )
            blocks.append(actual)
            by_working_face[working_face_id] = actual
    edge_ids = tuple(sorted({
        item.edge for block in blocks
        for item in working.faces[int(block.working_face_id)].loop
    }))
    _cancel(cancellation_check, "structured global seeding start")
    try:
        seeding = solve_seeding(
            working, target_size=plan.target_size, edge_ids=edge_ids,
            overrides=dict(plan.seed_overrides),
            max_divisions=plan.options.maximum_divisions_per_edge,
        )
    except (SeedingConflict, ValueError) as error:
        raise MeshError(f"global structured seeding failed: {error}") from error
    _cancel(cancellation_check, "structured global seeding complete")

    equations: list[SeedEquation] = []
    estimated = 0
    for block in blocks:
        face = working.faces[int(block.working_face_id)]
        if len(face.corners) != 4 or face.holes:
            continue
        sides = face.sides()
        for first, second in ((0, 2), (1, 3)):
            left = tuple(item.edge for item in sides[first])
            right = tuple(item.edge for item in sides[second])
            equations.append(SeedEquation(face.id, left, right, block_id=block.block_id))
            if sum(seeding.divisions[item] for item in left) != sum(seeding.divisions[item] for item in right):
                raise MeshError(f"global seed equation failed for block {block.block_id}")
        if block.action != "native_residual":
            estimated += (
                sum(seeding.divisions[item.edge] for item in sides[0])
                * sum(seeding.divisions[item.edge] for item in sides[1])
            )
    if estimated > plan.options.maximum_estimated_elements:
        raise MeshError(
            f"global seed solution estimates {estimated} structured elements, exceeding "
            f"maximum_estimated_elements={plan.options.maximum_estimated_elements}"
        )

    source_for_working_edge: dict[int, int] = {}
    for source_edge, descendants in sorted(edge_mapping.items()):
        for edge_id in descendants:
            source_for_working_edge.setdefault(edge_id, source_edge)
    interfaces: list[StructuredInterface] = []
    for position, edge_id in enumerate(sorted(working.edges)):
        if position % 512 == 0:
            _cancel(cancellation_check, "structured working interface scan")
        owners = [face for face in working.faces_using_edge(edge_id) if face in by_working_face]
        if len(owners) != 2:
            continue
        first_face, second_face = owners
        first, second = by_working_face[first_face], by_working_face[second_face]
        interfaces.append(StructuredInterface(
            edge_id, first_face, second_face, first.component_id,
            seeding.divisions[edge_id], True, "exact shared working topology",
            first.block_id, second.block_id, source_for_working_edge.get(edge_id),
        ))
    return (
        tuple(sorted(blocks, key=lambda item: item.block_id)), tuple(interfaces),
        tuple(equations), seeding.divisions, estimated,
    )


def apply_structured_layout(
    geometry: GeometryModel,
    plan: StructuredLayoutPlan,
    *,
    include_features: bool = False,
    cancellation_check: CancellationCheck | None = None,
) -> tuple[GeometryModel, StructuredLayoutReport]:
    """Verify and apply a plan on a detached deep clone."""

    _cancel(cancellation_check, "structured application start")
    if not isinstance(plan, StructuredLayoutPlan):
        raise MeshError("structured layout must be a StructuredLayoutPlan")
    if not plan.plan_hash or plan.plan_hash != _plan_hash_payload(plan):
        raise MeshError("structured layout plan hash is missing or has been tampered with")
    if not plan.is_current(geometry):
        raise MeshError("structured layout plan is stale or belongs to another model")
    face_ids = tuple(item.source_face_id for item in plan.faces)
    if len(face_ids) != len(set(face_ids)) or any(item not in geometry.faces for item in face_ids):
        raise MeshError("structured layout contains duplicate or missing source faces")
    planned_ids = tuple(item.block_id for item in plan.blocks)
    assigned_ids = tuple(block_id for face in plan.faces for block_id in face.block_ids)
    if len(planned_ids) != len(set(planned_ids)) or set(planned_ids) != set(assigned_ids):
        raise MeshError("structured layout block identities are inconsistent")
    if len(plan.blocks) > plan.options.maximum_blocks:
        raise MeshError("structured layout block count exceeds its verified resource policy")

    original_loops = {
        item.source_face_id: tuple(geometry.faces[item.source_face_id].loop)
        for item in plan.faces if item.action == "radial_partition"
    }
    original_centres = {
        face_id: np.mean(
            [
                geometry.vertex_position(item)
                for item in _loop_vertices(geometry, loop)
            ],
            axis=0,
        )
        for face_id, loop in original_loops.items()
    }
    working = geometry.clone(include_features=include_features)
    face_mapping: dict[int, tuple[int, ...]] = {}
    diagnostics: list[str] = []
    split_data: dict[int, tuple[int, int, int]] = {}
    radial_edges = sorted({item.edge for loop in original_loops.values() for item in loop})
    with working.transaction():
        for position, edge_id in enumerate(radial_edges):
            if position % 64 == 0:
                _cancel(cancellation_check, "structured shared-edge partition")
            midpoint, (first, second) = working.split_edge(edge_id, 0.5)
            split_data[edge_id] = (midpoint, first, second)
        for position, decision in enumerate(plan.faces):
            if position % 32 == 0:
                _cancel(cancellation_check, "structured block materialization")
            source = decision.source_face_id
            if decision.action in {"mapped", "native"}:
                face_mapping[source] = (source,)
            elif decision.action == "promote_quad":
                face_mapping[source] = _promote_quad(working, source)
            elif decision.action == "radial_partition":
                face_mapping[source] = _radial_partition(
                    working, source, original_loops[source], split_data,
                    original_centres[source],
                )
            elif decision.action == "ogrid_partition":
                face_mapping[source] = _ogrid_partition(working, source)
            else:
                raise MeshError(f"unknown structured face action {decision.action!r}")
            if decision.action != "mapped":
                diagnostics.append(f"face {source}: {decision.reason}")

    edge_mapping: dict[int, tuple[int, ...]] = {}
    for position, edge_id in enumerate(sorted(geometry.edges)):
        if position % 512 == 0:
            _cancel(cancellation_check, "structured edge lineage")
        resolved = tuple(
            reference.id for reference in working.resolve_ref(EntityRef("edge", edge_id))
            if reference.kind == "edge"
        )
        if not resolved and edge_id in working.edges:
            resolved = (edge_id,)
        edge_mapping[edge_id] = tuple(dict.fromkeys(resolved))
    blocks, interfaces, equations, seed_solution, estimated = _actual_evidence(
        working, plan, face_mapping, edge_mapping, cancellation_check
    )
    report = StructuredLayoutReport(
        plan, face_mapping, edge_mapping, tuple(diagnostics),
        metrics={
            "actual_block_count": len(blocks),
            "actual_interface_count": len(interfaces),
            "actual_seed_equation_count": len(equations),
            "estimated_structured_elements": estimated,
        },
        seed_solution=seed_solution, status="applied", blocks=blocks,
        interfaces=interfaces, seed_equations=equations,
        working_model_id=str(working.model_id),
        working_revision=working.revision,
    )
    _cancel(cancellation_check, "structured application complete")
    return working, report


def _upstream_feature_bindings(
    geometry: GeometryModel, plan: StructuredLayoutPlan
) -> tuple[tuple[int, ...], Mapping[str, tuple[Any, ...]]]:
    try:
        from anygeometry.features import FeatureOutputRef
    except ImportError:
        return (), {}
    changed = {
        EntityRef("face", item.source_face_id)
        for item in plan.faces
        if item.action in {"promote_quad", "radial_partition", "ogrid_partition"}
    }
    for item in plan.faces:
        if item.action in {"promote_quad", "radial_partition", "ogrid_partition"}:
            changed.update(
                EntityRef("edge", edge.edge)
                for edge in geometry.faces[item.source_face_id].loop
            )
    bindings = []
    for record in geometry.features.records:
        for output_key, reference in sorted(record.outputs.items()):
            if reference in changed:
                bindings.append(FeatureOutputRef(record.feature_id, output_key, reference.kind))
    bindings.sort(key=lambda item: (item.feature_id, item.output_key, item.kind))
    dependencies = tuple(sorted({item.feature_id for item in bindings}))
    return dependencies, ({"upstream": tuple(bindings)} if bindings else {})


def commit_structured_layout(
    geometry: GeometryModel,
    plan: StructuredLayoutPlan,
    *,
    name: str = "Structured mesh partitions",
    cancellation_check: CancellationCheck | None = None,
) -> tuple[GeometryModel, StructuredLayoutReport, Any]:
    """Return a detached committed clone with one exact frozen feature."""

    if not hasattr(geometry.features, "adopt_frozen"):
        raise MeshError("installed ANYgeometry lacks FeatureHistory.adopt_frozen")
    dependencies, inputs = _upstream_feature_bindings(geometry, plan)
    working, report = apply_structured_layout(
        geometry, plan, include_features=True, cancellation_check=cancellation_check
    )
    changed = [
        item for item in plan.faces
        if item.action in {"promote_quad", "radial_partition", "ogrid_partition"}
    ]
    if not changed:
        raise MeshError("structured layout has no geometry partitions to commit")
    outputs = {
        f"face_{item.source_face_id}_{offset + 1}": EntityRef("face", face_id)
        for item in changed
        for offset, face_id in enumerate(report.source_to_working_faces[item.source_face_id])
    }
    _cancel(cancellation_check, "structured frozen feature adoption")
    try:
        record = working.features.adopt_frozen(
            working,
            kind="geometry.mesh_partition.frozen",
            name=name,
            parameters={"structured_layout": plan.to_dict()},
            inputs=inputs,
            dependencies=dependencies,
            outputs=outputs,
            diagnostic=(
                "Exact whole-plan mesh partitions are frozen. Edit upstream geometry, "
                "preview a fresh plan, and commit it again."
            ),
        )
    except Exception as error:
        raise MeshError(f"frozen structured-layout commit failed atomically: {error}") from error
    return working, replace(report, status="committed"), record


def regularity_metrics(
    mesh: Any,
    *,
    target_size: float,
    minimum_size_ratio: float = 0.67,
    maximum_size_ratio: float = 1.50,
    mapped_element_ids: Iterable[int] = (),
) -> dict[str, float | int]:
    """Return deterministic element topology and edge-size metrics."""

    quads = {int(key): tuple(map(int, value[:4])) for key, value in mesh.quads.items()}
    tris = {int(key): tuple(map(int, value[:3])) for key, value in mesh.tris.items()}
    total = len(quads) + len(tris)
    lengths: list[float] = []
    valence: dict[int, set[int]] = {}
    for element_id, nodes in (*quads.items(), *tris.items()):
        for first, second in zip(nodes, nodes[1:] + nodes[:1]):
            lengths.append(float(np.linalg.norm(
                np.asarray(mesh.nodes[second]) - np.asarray(mesh.nodes[first])
            )))
            valence.setdefault(first, set()).add(element_id)
            valence.setdefault(second, set()).add(element_id)
    array = np.asarray(lengths)
    mean = float(array.mean()) if len(array) else 0.0
    coefficient = float(array.std() / mean) if mean > 0.0 else 0.0
    outside = int(np.count_nonzero(
        (array < minimum_size_ratio * target_size)
        | (array > maximum_size_ratio * target_size)
    )) if len(array) else 0
    all_quad = bool(quads) and not tris
    irregular = sum(
        1 for attached in valence.values()
        if len(attached) > 2 and len(attached) != (4 if all_quad else 6)
    )

    def area(nodes: Sequence[int]) -> float:
        points = np.asarray([mesh.nodes[item] for item in nodes])
        if len(points) == 3:
            return 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))
        return 0.5 * float(
            np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
            + np.linalg.norm(np.cross(points[2] - points[0], points[3] - points[0]))
        )

    mapped = {int(item) for item in mapped_element_ids}
    areas = {element_id: area(nodes) for element_id, nodes in (*quads.items(), *tris.items())}
    total_area = sum(areas.values())
    mapped_area = sum(value for key, value in areas.items() if key in mapped)
    return {
        "quad_fraction": 0.0 if total == 0 else len(quads) / total,
        "triangle_fraction": 0.0 if total == 0 else len(tris) / total,
        "mapped_area_fraction": 0.0 if total_area <= 0.0 else mapped_area / total_area,
        "size_coefficient_of_variation": coefficient,
        "size_uniformity_score": 1.0 / (1.0 + coefficient),
        "edges_outside_balanced_size_band": outside,
        "irregular_interior_valence_count": irregular,
        "valence_regularity_score": 1.0 if not valence else max(0.0, 1.0 - irregular / len(valence)),
        "shell_element_count": total,
    }
