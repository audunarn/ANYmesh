"""Vector-friendly shell quality metrics and non-advisory validity checks."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees
from typing import Any, Iterable, Sequence

import numpy as np

from .core import MeshCore, corner_edges
from .errors import MeshError

__all__ = [
    "ElementQuality",
    "MeshQualityV2",
    "MeshValidityError",
    "ValidityReport",
    "assert_valid_mesh",
    "evaluate_quality",
    "hard_validity_check",
    "mesh_quality",
    "quad_candidate_quality",
    "quad_quality",
    "triangle_quality",
    "validate_mesh",
]


class MeshValidityError(MeshError):
    """Raised when topology or element geometry is invalid for analysis."""


def _frozen(array: Any, dtype: Any = np.float64) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


def _normal(points: np.ndarray) -> np.ndarray:
    normal = np.zeros(3, dtype=np.float64)
    for index in range(len(points)):
        normal += np.cross(points[index], points[(index + 1) % len(points)])
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 0.0 else normal


def _corner_data(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(coordinates)
    lengths = np.array(
        [np.linalg.norm(coordinates[(index + 1) % count] - coordinates[index]) for index in range(count)],
        dtype=np.float64,
    )
    angles = np.empty(count, dtype=np.float64)
    jacobians = np.empty(count, dtype=np.float64)
    reference = _normal(coordinates)
    for index in range(count):
        outgoing = coordinates[(index + 1) % count] - coordinates[index]
        incoming = coordinates[(index - 1) % count] - coordinates[index]
        denominator = max(float(np.linalg.norm(outgoing) * np.linalg.norm(incoming)), 1.0e-300)
        cosine = float(np.clip(np.dot(outgoing, incoming) / denominator, -1.0, 1.0))
        angles[index] = degrees(acos(cosine))
        jacobians[index] = float(np.dot(np.cross(outgoing, incoming), reference) / denominator)
    return lengths, angles, jacobians


@dataclass(frozen=True)
class ElementQuality:
    element_ids: np.ndarray
    area: np.ndarray
    aspect_ratio: np.ndarray
    minimum_angle: np.ndarray
    maximum_angle: np.ndarray
    scaled_jacobian: np.ndarray
    warpage: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "element_ids", "area", "aspect_ratio", "minimum_angle", "maximum_angle",
            "scaled_jacobian", "warpage",
        ):
            dtype = np.int64 if name == "element_ids" else np.float64
            object.__setattr__(self, name, _frozen(getattr(self, name), dtype))

    def __len__(self) -> int:
        return int(self.element_ids.size)


@dataclass(frozen=True)
class ValidityReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def ok(self) -> bool:
        return self.valid

    def __bool__(self) -> bool:
        return self.valid


@dataclass(frozen=True)
class MeshQualityV2:
    triangles: ElementQuality
    quadrilaterals: ElementQuality
    validity: ValidityReport

    @property
    def num_elements(self) -> int:
        return len(self.triangles) + len(self.quadrilaterals)

    @property
    def minimum_scaled_jacobian(self) -> float:
        values = np.concatenate((self.triangles.scaled_jacobian, self.quadrilaterals.scaled_jacobian))
        return float(np.min(values)) if values.size else 1.0

    @property
    def maximum_aspect_ratio(self) -> float:
        values = np.concatenate((self.triangles.aspect_ratio, self.quadrilaterals.aspect_ratio))
        return float(np.max(values)) if values.size else 1.0

    @property
    def minimum_angle(self) -> float:
        values = np.concatenate((self.triangles.minimum_angle, self.quadrilaterals.minimum_angle))
        return float(np.min(values)) if values.size else 90.0

    @property
    def maximum_warpage(self) -> float:
        values = np.concatenate((self.triangles.warpage, self.quadrilaterals.warpage))
        return float(np.max(values)) if values.size else 0.0

    @property
    def ok(self) -> bool:
        return self.validity.valid


def triangle_quality(
    points: Any,
    triangles: Any,
    element_ids: Any | None = None,
) -> ElementQuality:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise MeshError("points must have shape (n, 2) or (n, 3)")
    if coordinates.shape[1] == 2:
        coordinates = np.column_stack((coordinates, np.zeros(len(coordinates))))
    connectivity = np.asarray(triangles, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] not in (3, 6):
        raise MeshError("triangle connectivity must be T3 or T6")
    count = len(connectivity)
    ids = np.arange(1, count + 1) if element_ids is None else np.asarray(element_ids, dtype=np.int64)
    area = np.empty(count)
    aspect = np.empty(count)
    min_angle = np.empty(count)
    max_angle = np.empty(count)
    jacobian = np.empty(count)
    for row, element in enumerate(connectivity):
        corners = coordinates[element[:3]]
        lengths, angles, jacobians = _corner_data(corners)
        area[row] = 0.5 * np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
        aspect[row] = float(np.max(lengths) / max(float(np.min(lengths)), 1.0e-300))
        min_angle[row] = float(np.min(angles))
        max_angle[row] = float(np.max(angles))
        jacobian[row] = float(np.min(jacobians))
    return ElementQuality(ids, area, aspect, min_angle, max_angle, jacobian, np.zeros(count))


def _quad_values(corners: np.ndarray) -> tuple[float, float, float, float, float, float]:
    lengths, angles, jacobians = _corner_data(corners)
    first_cross = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    second_cross = np.cross(corners[2] - corners[0], corners[3] - corners[0])
    area = 0.5 * (np.linalg.norm(first_cross) + np.linalg.norm(second_cross))
    first_norm = float(np.linalg.norm(first_cross))
    second_norm = float(np.linalg.norm(second_cross))
    if first_norm > 0.0 and second_norm > 0.0:
        cosine = float(np.clip(np.dot(first_cross, second_cross) / (first_norm * second_norm), -1.0, 1.0))
        warpage = degrees(acos(cosine)) / 180.0
    else:
        warpage = 1.0
    return (
        float(area),
        float(np.max(lengths) / max(float(np.min(lengths)), 1.0e-300)),
        float(np.min(angles)),
        float(np.max(angles)),
        float(np.min(jacobians)),
        float(warpage),
    )


def quad_candidate_quality(points: Any, connectivity: Sequence[int]) -> dict[str, float]:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.shape[1] == 2:
        coordinates = np.column_stack((coordinates, np.zeros(len(coordinates))))
    values = _quad_values(coordinates[np.asarray(connectivity, dtype=np.int64)[:4]])
    return dict(zip(
        ("area", "aspect_ratio", "minimum_angle", "maximum_angle", "scaled_jacobian", "warpage"),
        values,
    ))


def quad_quality(
    points: Any,
    quads: Any,
    element_ids: Any | None = None,
) -> ElementQuality:
    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise MeshError("points must have shape (n, 2) or (n, 3)")
    if coordinates.shape[1] == 2:
        coordinates = np.column_stack((coordinates, np.zeros(len(coordinates))))
    connectivity = np.asarray(quads, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] not in (4, 8):
        raise MeshError("quadrilateral connectivity must be Q4 or Q8")
    count = len(connectivity)
    ids = np.arange(1, count + 1) if element_ids is None else np.asarray(element_ids, dtype=np.int64)
    values = np.asarray([_quad_values(coordinates[element[:4]]) for element in connectivity], dtype=float)
    if count == 0:
        values = np.empty((0, 6), dtype=float)
    return ElementQuality(ids, *(values[:, column] for column in range(6)))


def validate_mesh(
    mesh: MeshCore,
    *,
    tolerance: float | None = None,
    raise_on_error: bool = False,
    declared_plate_junction_edges: Iterable[Sequence[int]] = (),
) -> ValidityReport:
    if not isinstance(mesh, MeshCore):
        raise TypeError("validate_mesh expects MeshCore")
    extent = max(float(np.ptp(mesh.node_coordinates, axis=0).max()) if mesh.num_nodes else 0.0, 1.0)
    area_tolerance = extent * extent * 1.0e-14 if tolerance is None else float(tolerance)
    errors: list[str] = []
    declared_junctions: set[tuple[int, int]] = set()
    for raw_edge in declared_plate_junction_edges:
        edge = tuple(int(value) for value in raw_edge)
        if len(edge) != 2 or edge[0] == edge[1]:
            raise MeshError(
                "declared_plate_junction_edges must contain distinct node-row pairs"
            )
        if min(edge) < 0 or max(edge) >= mesh.num_nodes:
            raise MeshError("declared plate-junction node row is outside the mesh")
        declared_junctions.add((min(edge), max(edge)))
    active_elements: list[tuple[str, int, np.ndarray]] = []
    for row in np.flatnonzero(mesh.triangle_active):
        active_elements.append(("triangle", int(mesh.triangle_ids[row]), mesh.triangle_connectivity[row]))
    for row in np.flatnonzero(mesh.quad_active):
        active_elements.append(("quadrilateral", int(mesh.quad_ids[row]), mesh.quad_connectivity[row]))

    duplicate_keys: dict[tuple[int, ...], int] = {}
    incidence: dict[tuple[int, int], list[int]] = {}
    midsides: dict[tuple[int, int], int] = {}
    midside_edges: dict[int, tuple[int, int]] = {}
    for kind, stable_id, connectivity in active_elements:
        corner_count = 3 if kind == "triangle" else 4
        corners = tuple(int(value) for value in connectivity[:corner_count])
        if len(set(corners)) != corner_count:
            errors.append(f"{kind} {stable_id} repeats a corner node")
            continue
        key = tuple(sorted(corners))
        if key in duplicate_keys:
            errors.append(f"elements {duplicate_keys[key]} and {stable_id} duplicate the same corners")
        else:
            duplicate_keys[key] = stable_id
        for edge in corner_edges(connectivity):
            incidence.setdefault(edge, []).append(stable_id)
        if len(connectivity) in (6, 8):
            for edge_index, edge in enumerate(corner_edges(connectivity)):
                midside = int(connectivity[corner_count + edge_index])
                if midside in edge:
                    errors.append(f"{kind} {stable_id} has a midside equal to an edge corner")
                previous = midsides.get(edge)
                if previous is not None and previous != midside:
                    errors.append(f"edge {edge} uses midside nodes {previous} and {midside}")
                midsides[edge] = midside
                other_edge = midside_edges.get(midside)
                if other_edge is not None and other_edge != edge:
                    errors.append(f"midside node row {midside} is reused by edges {other_edge} and {edge}")
                midside_edges[midside] = edge

    for edge, attached in incidence.items():
        if len(attached) > 2 and edge not in declared_junctions:
            errors.append(f"non-manifold edge {edge} belongs to {len(attached)} active elements")

    if mesh.active_triangle_count:
        rows = np.flatnonzero(mesh.triangle_active)
        metrics = triangle_quality(
            mesh.node_coordinates,
            mesh.triangle_connectivity[rows],
            mesh.triangle_ids[rows],
        )
        for row, stable_id in enumerate(metrics.element_ids):
            if not np.isfinite(metrics.area[row]) or metrics.area[row] <= area_tolerance:
                errors.append(f"triangle {int(stable_id)} has zero or non-finite area")
            if metrics.scaled_jacobian[row] <= 0.0:
                errors.append(f"triangle {int(stable_id)} has a non-positive corner Jacobian")
    if mesh.active_quad_count:
        rows = np.flatnonzero(mesh.quad_active)
        metrics = quad_quality(
            mesh.node_coordinates,
            mesh.quad_connectivity[rows],
            mesh.quad_ids[rows],
        )
        for row, stable_id in enumerate(metrics.element_ids):
            if not np.isfinite(metrics.area[row]) or metrics.area[row] <= area_tolerance:
                errors.append(f"quadrilateral {int(stable_id)} has zero or non-finite area")
            if not np.isfinite(metrics.scaled_jacobian[row]) or metrics.scaled_jacobian[row] <= 0.0:
                errors.append(f"quadrilateral {int(stable_id)} is inverted, concave, or self-intersecting")

    report = ValidityReport(tuple(dict.fromkeys(errors)))
    if raise_on_error and not report.valid:
        raise MeshValidityError("; ".join(report.errors))
    return report


def assert_valid_mesh(
    mesh: MeshCore,
    *,
    tolerance: float | None = None,
    declared_plate_junction_edges: Iterable[Sequence[int]] = (),
) -> ValidityReport:
    return validate_mesh(
        mesh,
        tolerance=tolerance,
        raise_on_error=True,
        declared_plate_junction_edges=declared_plate_junction_edges,
    )


hard_validity_check = assert_valid_mesh


def mesh_quality(
    mesh: MeshCore,
    *,
    check_validity: bool = True,
    declared_plate_junction_edges: Iterable[Sequence[int]] = (),
) -> MeshQualityV2:
    validity = validate_mesh(
        mesh,
        raise_on_error=check_validity,
        declared_plate_junction_edges=declared_plate_junction_edges,
    )
    triangle_rows = np.flatnonzero(mesh.triangle_active)
    quad_rows = np.flatnonzero(mesh.quad_active)
    triangles = triangle_quality(
        mesh.node_coordinates,
        mesh.triangle_connectivity[triangle_rows],
        mesh.triangle_ids[triangle_rows],
    )
    quadrilaterals = quad_quality(
        mesh.node_coordinates,
        mesh.quad_connectivity[quad_rows],
        mesh.quad_ids[quad_rows],
    )
    return MeshQualityV2(triangles, quadrilaterals, validity)


evaluate_quality = mesh_quality
