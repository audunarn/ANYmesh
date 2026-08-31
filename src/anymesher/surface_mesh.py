"""High-level native planar surface meshing and T6/Q8 enrichment."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, ceil, sqrt
from numbers import Integral
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .core import MeshCore, corner_edges
from .errors import MeshError
from .native import NativeBoundary
from .optimization import constrained_smoothing, local_edge_flip
from .quality_v2 import MeshQualityV2, assert_valid_mesh, evaluate_quality
from .recombine import recombine_triangles_with_report
from .triangulation import PlanarTriangulation, triangulate_polygon

__all__ = [
    "SurfaceMeshOptions",
    "generate_surface_mesh",
    "insert_midside_nodes",
    "mesh_planar_surface",
    "surface_mesh",
    "to_quadratic",
]


@dataclass(frozen=True)
class SurfaceMeshOptions:
    recombine: bool = True
    order: str | int = "linear"
    target_size: float | None = None
    backend: str | NativeBoundary | None = "auto"
    min_scaled_jacobian: float = 0.20
    max_aspect_ratio: float = 4.0
    min_angle: float = 30.0
    max_angle: float = 150.0
    max_warpage: float = 0.10
    max_element_growth: float = 1.5
    prefer_quality_policy: bool = False
    lattice_alignment: str = "chart"
    metric_tensor: Any | None = None
    enforce_quality: bool = False
    max_lattice_points: int = 1_000_000
    max_metric_aspect_ratio: float = 25.0
    max_recombination_work: int = 1_000_000

    @property
    def quadratic(self) -> bool:
        return self.order in (2, "2", "quadratic", "T6/Q8", "t6/q8")

    def __post_init__(self) -> None:
        if self.order not in (1, 2, "1", "2", "linear", "quadratic", "T3/Q4", "T6/Q8", "t3/q4", "t6/q8"):
            raise MeshError("order must select linear T3/Q4 or quadratic T6/Q8")
        if self.target_size is not None and (not np.isfinite(self.target_size) or self.target_size <= 0.0):
            raise MeshError("target_size must be positive and finite")
        if self.lattice_alignment not in {"chart", "dominant_boundary"}:
            raise MeshError(
                "lattice_alignment must be 'chart' or 'dominant_boundary'"
            )
        for name in ("max_lattice_points", "max_recombination_work"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) < 1
            ):
                raise MeshError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        metric_aspect = float(self.max_metric_aspect_ratio)
        if not np.isfinite(metric_aspect) or metric_aspect < 1.0:
            raise MeshError("max_metric_aspect_ratio must be finite and at least 1")
        object.__setattr__(self, "max_metric_aspect_ratio", metric_aspect)
        growth = float(self.max_element_growth)
        if not np.isfinite(growth) or growth < 1.0:
            raise MeshError("max_element_growth must be finite and at least 1")
        object.__setattr__(self, "max_element_growth", growth)
        if self.metric_tensor is not None:
            tensor = np.asarray(self.metric_tensor, dtype=float)
            if tensor.shape != (2, 2) or not np.all(np.isfinite(tensor)):
                raise MeshError("metric_tensor must be a finite 2-by-2 tensor")
            scale = max(float(np.max(np.abs(tensor))), 1.0)
            if not np.allclose(
                tensor,
                tensor.T,
                rtol=1.0e-12,
                atol=1.0e-14 * scale,
            ):
                raise MeshError("metric_tensor must be symmetric positive definite")
            eigenvalues = np.linalg.eigvalsh(tensor)
            if np.any(eigenvalues <= 0.0):
                raise MeshError("metric_tensor must be symmetric positive definite")
            aspect = sqrt(float(eigenvalues[-1] / eigenvalues[0]))
            if aspect > metric_aspect * (1.0 + 1.0e-12):
                raise MeshError(
                    "metric_tensor requests aspect ratio "
                    f"{aspect:.6g}, above max_metric_aspect_ratio "
                    f"{metric_aspect:.6g}"
                )


def _quality_threshold_report(
    quality: MeshQualityV2,
    settings: SurfaceMeshOptions,
) -> dict[str, Any]:
    groups = (quality.triangles, quality.quadrilaterals)
    values = {
        "scaled_jacobian": np.concatenate(
            [group.scaled_jacobian for group in groups]
        ),
        "aspect_ratio": np.concatenate([group.aspect_ratio for group in groups]),
        "minimum_angle": np.concatenate([group.minimum_angle for group in groups]),
        "maximum_angle": np.concatenate([group.maximum_angle for group in groups]),
        "warpage": np.concatenate([group.warpage for group in groups]),
    }
    element_ids = np.concatenate([group.element_ids for group in groups])
    masks = {
        "scaled_jacobian": values["scaled_jacobian"] < settings.min_scaled_jacobian,
        "aspect_ratio": values["aspect_ratio"] > settings.max_aspect_ratio,
        "minimum_angle": values["minimum_angle"] < settings.min_angle,
        "maximum_angle": values["maximum_angle"] > settings.max_angle,
        "warpage": values["warpage"] > settings.max_warpage,
    }
    failed = np.zeros(len(element_ids), dtype=bool)
    for mask in masks.values():
        failed |= mask
    return {
        "accepted": not bool(np.any(failed)),
        "thresholds": {
            "minimum_scaled_jacobian": settings.min_scaled_jacobian,
            "maximum_aspect_ratio": settings.max_aspect_ratio,
            "minimum_angle": settings.min_angle,
            "maximum_angle": settings.max_angle,
            "maximum_warpage": settings.max_warpage,
        },
        "worst": {
            "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
            "maximum_aspect_ratio": quality.maximum_aspect_ratio,
            "minimum_angle": quality.minimum_angle,
            "maximum_angle": (
                float(np.max(values["maximum_angle"]))
                if len(element_ids)
                else 90.0
            ),
            "maximum_warpage": quality.maximum_warpage,
        },
        "violation_counts": {
            name: int(np.count_nonzero(mask)) for name, mask in masks.items()
        },
        "poor_element_ids": [int(value) for value in element_ids[failed]],
    }


@dataclass(frozen=True)
class _Plane:
    origin: np.ndarray
    first: np.ndarray
    second: np.ndarray
    dimension: int

    def project(self, values: np.ndarray) -> np.ndarray:
        if self.dimension == 2:
            return np.asarray(values, dtype=np.float64)
        offsets = np.asarray(values, dtype=np.float64) - self.origin
        return np.column_stack((offsets @ self.first, offsets @ self.second))

    def lift(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self.dimension == 2:
            return np.column_stack((values, np.zeros(len(values))))
        return self.origin + values[:, :1] * self.first + values[:, 1:] * self.second


def _plane_from_outer(outer: np.ndarray) -> _Plane:
    if outer.ndim != 2 or outer.shape[1] not in (2, 3):
        raise MeshError("surface loops must have shape (n, 2) or (n, 3)")
    if outer.shape[1] == 2:
        return _Plane(np.zeros(3), np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)), 2)
    origin = outer[0]
    first_vector = next((point - origin for point in outer[1:] if np.linalg.norm(point - origin) > 0.0), None)
    if first_vector is None:
        raise MeshError("surface outer loop has no extent")
    first = first_vector / np.linalg.norm(first_vector)
    normal = next(
        (
            np.cross(first, point - origin)
            for point in outer[1:]
            if np.linalg.norm(np.cross(first, point - origin)) > 1.0e-14 * max(np.linalg.norm(point - origin), 1.0)
        ),
        None,
    )
    if normal is None:
        raise MeshError("surface outer loop is collinear")
    normal = normal / np.linalg.norm(normal)
    second = np.cross(normal, first)
    return _Plane(origin, first, second, 3)


def _as_loop(value: Any, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise MeshError(f"{name} must have shape (n, {dimension})")
    if not np.all(np.isfinite(array)):
        raise MeshError(f"{name} must be finite")
    return array


def _densify_open(values: np.ndarray, size: float) -> np.ndarray:
    result = [values[0]]
    for first, second in zip(values, values[1:]):
        divisions = max(1, int(ceil(np.linalg.norm(second - first) / size)))
        result.extend(first + (second - first) * index / divisions for index in range(1, divisions))
        result.append(second.copy())
    return np.asarray(result)


def _densify_loop(values: np.ndarray, size: float) -> np.ndarray:
    closed = np.vstack((values, values[0]))
    return _densify_open(closed, size)[:-1]


def _inside(point: np.ndarray, ring: np.ndarray) -> bool:
    inside = False
    x, y = point
    for first, second in zip(ring, np.vstack((ring[1:], ring[:1]))):
        if (first[1] > y) != (second[1] > y):
            crossing = first[0] + (y - first[1]) * (second[0] - first[0]) / (second[1] - first[1])
            if crossing > x:
                inside = not inside
    return inside


def _segment_distance(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    direction = second - first
    squared = float(direction @ direction)
    if squared == 0.0:
        return float(np.linalg.norm(point - first))
    fraction = float(np.clip(((point - first) @ direction) / squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + fraction * direction)))


def _target_points(
    outer: np.ndarray,
    holes: Sequence[np.ndarray],
    constraints: Sequence[np.ndarray],
    size: float,
    alignment: str = "chart",
    metric_tensor: Any | None = None,
    *,
    max_lattice_points: int = 1_000_000,
    cancellation_check: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Generate a deterministic triangular lattice away from protected lines."""

    origin = np.zeros(2, dtype=float)
    basis = np.eye(2, dtype=float)
    x_spacing = float(size)
    y_spacing = float(size)
    if metric_tensor is not None:
        tensor = np.asarray(metric_tensor, dtype=float)
        values, vectors = np.linalg.eigh(tensor)
        order = np.argsort(values)
        values = values[order]
        basis = vectors[:, order]
        # Eigenvector signs are mathematically arbitrary.  Anchor the first
        # metric axis to the first non-orthogonal oriented boundary segment so
        # rotating a model rotates the generated lattice instead of mirroring
        # or shifting its staggered rows.
        boundary_segments = np.roll(outer, -1, axis=0) - outer
        tolerance = 1.0e-14 * max(
            float(np.max(np.linalg.norm(boundary_segments, axis=1))), 1.0
        )
        for segment in boundary_segments:
            projection = float(segment @ basis[:, 0])
            if abs(projection) > tolerance:
                if projection < 0.0:
                    basis[:, 0] *= -1.0
                break
        if float(np.linalg.det(basis)) < 0.0:
            basis[:, 1] *= -1.0
        x_spacing, y_spacing = (
            float(1.0 / sqrt(value)) for value in values
        )
        origin = np.mean(outer, axis=0)
    elif alignment == "dominant_boundary":
        segments = np.roll(outer, -1, axis=0) - outer
        lengths = np.linalg.norm(segments, axis=1)
        if len(lengths) and float(np.max(lengths)) > 0.0:
            choices: list[
                tuple[tuple[float, float, tuple[float, ...]], np.ndarray]
            ] = []
            for index, (segment, length) in enumerate(zip(segments, lengths)):
                if float(length) <= 0.0:
                    continue
                axis = np.asarray(segment / length, dtype=float)
                if axis[0] < 0.0 or (
                    abs(float(axis[0])) <= 1.0e-15 and axis[1] < 0.0
                ):
                    axis = -axis
                angle = float(np.arctan2(axis[1], axis[0]))
                endpoints = tuple(
                    np.round(
                        np.sort(
                            np.vstack(
                                (outer[index], outer[(index + 1) % len(outer)])
                            ),
                            axis=0,
                        ).ravel(),
                        14,
                    )
                )
                choices.append(((-float(length), angle, endpoints), axis))
            axis = min(choices, key=lambda item: item[0])[1]
            basis = np.column_stack((axis, np.array((-axis[1], axis[0]))))
            origin = np.mean(outer, axis=0)

    def project(values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - origin) @ basis

    def lift(values: np.ndarray) -> np.ndarray:
        return origin + np.asarray(values, dtype=float) @ basis.T

    local_outer = project(outer)
    local_holes = [project(hole) for hole in holes]
    local_constraints = [project(segment) for segment in constraints]
    minimum = np.min(local_outer, axis=0)
    maximum = np.max(local_outer, axis=0)
    vertical = sqrt(3.0) * 0.5 * y_spacing
    row_count = max(
        0,
        int(
            ceil(
                max(0.0, float(maximum[1] - minimum[1]))
                / max(vertical, np.finfo(float).tiny)
            )
        ),
    )
    column_count = max(
        0,
        int(
            ceil(
                max(0.0, float(maximum[0] - minimum[0]))
                / max(x_spacing, np.finfo(float).tiny)
            )
        ),
    )
    lattice_budget = row_count * column_count
    if lattice_budget > int(max_lattice_points):
        raise MeshError(
            "target lattice would inspect "
            f"{lattice_budget} points, above max_lattice_points "
            f"{int(max_lattice_points)}; increase target size or the explicit budget"
        )
    ys = np.arange(minimum[1] + 0.5 * vertical, maximum[1], vertical)
    protected_coordinates = [
        (ring[index], ring[(index + 1) % len(ring)])
        for ring in (local_outer, *local_holes)
        for index in range(len(ring))
    ]
    protected_coordinates.extend(
        (segment[0], segment[1]) for segment in local_constraints
    )
    clearance = 0.25 * min(x_spacing, y_spacing)
    values: list[np.ndarray] = []
    for row, y in enumerate(ys):
        if cancellation_check is not None and row % 256 == 0:
            cancellation_check("native target lattice")
        offset = 0.5 * x_spacing if row % 2 else 0.0
        xs = np.arange(
            minimum[0] + 0.5 * x_spacing + offset,
            maximum[0],
            x_spacing,
        )
        for x in xs:
            point = np.array((x, y), dtype=float)
            if not _inside(point, local_outer) or any(
                _inside(point, hole) for hole in local_holes
            ):
                continue
            if any(
                _segment_distance(point, first, second) <= clearance
                for first, second in protected_coordinates
            ):
                continue
            values.append(point)
    local = np.asarray(values, dtype=float).reshape((-1, 2))
    return lift(local)


@dataclass(frozen=True)
class _QualityCandidate:
    points: np.ndarray
    triangles: np.ndarray
    report: dict[str, Any]
    score: tuple[int, int, int, float, float, float]
    aspect_ratios: np.ndarray
    flips: int = 0
    moved_nodes: tuple[int, ...] = ()
    added_points: int = 0
    rounds: int = 0


def _triangle_quality(
    points: np.ndarray,
    triangles: np.ndarray,
    settings: SurfaceMeshOptions | None = None,
) -> tuple[dict[str, Any], tuple[int, int, float, float, float], np.ndarray]:
    policy = settings or SurfaceMeshOptions()
    aspects: list[float] = []
    jacobians: list[float] = []
    minimum_angles: list[float] = []
    maximum_angles: list[float] = []
    characteristic: list[float] = []
    incidence: dict[tuple[int, int], list[int]] = {}
    invalid = 0
    scale = max(float(np.ptp(points, axis=0).max()), 1.0) if len(points) else 1.0
    area_tolerance = np.finfo(np.float64).eps * scale * scale * 32.0
    for row, triangle in enumerate(triangles):
        coordinates = points[np.asarray(triangle, dtype=np.int64)]
        lengths = np.asarray(
            (
                np.linalg.norm(coordinates[1] - coordinates[0]),
                np.linalg.norm(coordinates[2] - coordinates[1]),
                np.linalg.norm(coordinates[0] - coordinates[2]),
            ),
            dtype=np.float64,
        )
        first = coordinates[1] - coordinates[0]
        second = coordinates[2] - coordinates[0]
        double_area = float(first[0] * second[1] - first[1] * second[0])
        if double_area <= area_tolerance or float(np.min(lengths)) <= 0.0:
            invalid += 1
        aspects.append(float(np.max(lengths) / max(float(np.min(lengths)), 1.0e-15)))
        characteristic.append(float(np.mean(lengths)))
        for index in range(3):
            edge = tuple(
                sorted(
                    (
                        int(triangle[index]),
                        int(triangle[(index + 1) % 3]),
                    )
                )
            )
            incidence.setdefault(edge, []).append(row)
        angles: list[float] = []
        corner_jacobians: list[float] = []
        for corner in range(3):
            previous = coordinates[(corner - 1) % 3] - coordinates[corner]
            following = coordinates[(corner + 1) % 3] - coordinates[corner]
            denominator = max(float(np.linalg.norm(previous) * np.linalg.norm(following)), 1.0e-30)
            cosine = float(np.clip((previous @ following) / denominator, -1.0, 1.0))
            angles.append(float(np.degrees(acos(cosine))))
            corner_jacobians.append(double_area / denominator)
        jacobians.append(min(corner_jacobians))
        minimum_angles.append(min(angles))
        maximum_angles.append(max(angles))
    aspect_array = np.asarray(aspects, dtype=np.float64)
    jacobian_array = np.asarray(jacobians, dtype=np.float64)
    minimum_angle_array = np.asarray(minimum_angles, dtype=np.float64)
    maximum_angle_array = np.asarray(maximum_angles, dtype=np.float64)
    growth_array = np.ones(len(triangles), dtype=np.float64)
    growth_repair_rows: set[int] = set()
    for attached in incidence.values():
        if len(attached) != 2:
            continue
        first_row, second_row = attached
        small = min(characteristic[first_row], characteristic[second_row])
        ratio = (
            float("inf")
            if small <= 0.0
            else max(characteristic[first_row], characteristic[second_row]) / small
        )
        growth_array[first_row] = max(growth_array[first_row], ratio)
        growth_array[second_row] = max(growth_array[second_row], ratio)
        if ratio > policy.max_element_growth:
            if characteristic[first_row] > characteristic[second_row]:
                growth_repair_rows.add(first_row)
            elif characteristic[second_row] > characteristic[first_row]:
                growth_repair_rows.add(second_row)
    aspect_rows = np.flatnonzero(aspect_array > policy.max_aspect_ratio)
    jacobian_rows = np.flatnonzero(jacobian_array < policy.min_scaled_jacobian)
    minimum_angle_rows = np.flatnonzero(minimum_angle_array < policy.min_angle)
    maximum_angle_rows = np.flatnonzero(maximum_angle_array > policy.max_angle)
    growth_rows = np.flatnonzero(growth_array > policy.max_element_growth)
    poor_rows = np.asarray(
        sorted(
            set(map(int, aspect_rows))
            | set(map(int, jacobian_rows))
            | set(map(int, minimum_angle_rows))
            | set(map(int, maximum_angle_rows))
            | set(map(int, growth_rows))
        ),
        dtype=np.int64,
    )
    repair_rows = np.asarray(
        sorted(
            set(map(int, aspect_rows))
            | set(map(int, jacobian_rows))
            | set(map(int, minimum_angle_rows))
            | set(map(int, maximum_angle_rows))
            | growth_repair_rows
        ),
        dtype=np.int64,
    )
    maximum_aspect = float(np.max(aspect_array)) if len(aspect_array) else 1.0
    minimum_jacobian = float(np.min(jacobians)) if jacobians else 1.0
    minimum_angle = float(np.min(minimum_angles)) if minimum_angles else 60.0
    maximum_angle = float(np.max(maximum_angles)) if maximum_angles else 60.0
    report = {
        "invalid_element_count": invalid,
        "elements_above_aspect_ratio_5": int(
            np.count_nonzero(aspect_array > 5.0)
        ),
        "quality_violation_count": int(len(poor_rows)),
        "elements_below_minimum_angle": int(len(minimum_angle_rows)),
        "elements_above_maximum_angle": int(len(maximum_angle_rows)),
        "elements_below_minimum_scaled_jacobian": int(len(jacobian_rows)),
        "elements_above_maximum_growth": int(len(growth_rows)),
        "max_aspect_ratio": maximum_aspect,
        "min_scaled_jacobian": minimum_jacobian,
        "min_angle": minimum_angle,
        "max_angle": maximum_angle,
        "max_element_growth": float(np.max(growth_array)) if len(growth_array) else 1.0,
        "poor_element_ids": [int(row) + 1 for row in poor_rows],
        "repair_element_ids": [int(row) + 1 for row in repair_rows],
    }
    score = (
        invalid,
        int(report["quality_violation_count"])
        if settings is not None and settings.prefer_quality_policy
        else 0,
        int(np.count_nonzero(aspect_array > 5.0)),
        maximum_aspect,
        -minimum_jacobian,
        -minimum_angle,
    )
    return report, score, aspect_array


def _make_candidate(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    flips: int = 0,
    moved_nodes: tuple[int, ...] = (),
    added_points: int = 0,
    rounds: int = 0,
    settings: SurfaceMeshOptions | None = None,
) -> _QualityCandidate:
    report, score, aspects = _triangle_quality(points, triangles, settings)
    return _QualityCandidate(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(triangles, dtype=np.int64),
        report,
        score,
        aspects,
        flips=flips,
        moved_nodes=moved_nodes,
        added_points=added_points,
        rounds=rounds,
    )


def _fixed_rows(
    points: np.ndarray,
    protected_edges: np.ndarray,
    explicit_points: np.ndarray,
) -> tuple[int, ...]:
    fixed = {int(node) for edge in protected_edges for node in edge}
    for explicit in explicit_points:
        matches = np.flatnonzero(np.all(points == explicit, axis=1))
        fixed.update(int(row) for row in matches)
    return tuple(sorted(fixed))


def _optimize_candidate(
    candidate: _QualityCandidate,
    protected_edges: np.ndarray,
    explicit_points: np.ndarray,
    settings: SurfaceMeshOptions | None = None,
) -> _QualityCandidate:
    """Run the fixed flip/smooth/flip sequence and publish only improvements."""

    best = candidate
    first_flip = local_edge_flip(
        candidate.points,
        candidate.triangles,
        protected_edges=protected_edges,
    )
    flipped = _make_candidate(
        candidate.points,
        first_flip.triangles,
        flips=first_flip.flip_count,
        added_points=candidate.added_points,
        rounds=candidate.rounds,
        settings=settings,
    )
    if flipped.score < best.score and flipped.score[0] == 0:
        best = flipped
    smoothing = constrained_smoothing(
        candidate.points,
        first_flip.triangles,
        fixed_nodes=_fixed_rows(candidate.points, protected_edges, explicit_points),
        constrained_edges=protected_edges,
        iterations=4,
        relaxation=0.6,
    )
    moved_nodes = tuple(int(row) for row in smoothing.moved_nodes)
    smoothed = _make_candidate(
        smoothing.points,
        first_flip.triangles,
        flips=first_flip.flip_count,
        moved_nodes=moved_nodes,
        added_points=candidate.added_points,
        rounds=candidate.rounds,
        settings=settings,
    )
    if smoothed.score < best.score and smoothed.score[0] == 0:
        best = smoothed
    final_flip = local_edge_flip(
        smoothing.points,
        first_flip.triangles,
        protected_edges=protected_edges,
    )
    finished = _make_candidate(
        smoothing.points,
        final_flip.triangles,
        flips=first_flip.flip_count + final_flip.flip_count,
        moved_nodes=moved_nodes,
        added_points=candidate.added_points,
        rounds=candidate.rounds,
        settings=settings,
    )
    if finished.score < best.score and finished.score[0] == 0:
        best = finished
    return best


def _refinement_midpoints(
    candidate: _QualityCandidate,
    protected_edges: np.ndarray,
    limit: int,
) -> np.ndarray:
    if limit <= 0:
        return np.empty((0, 2), dtype=np.float64)
    protected = {tuple(sorted(map(int, edge))) for edge in protected_edges}
    poor_rows = [element_id - 1 for element_id in candidate.report["repair_element_ids"]]
    poor_rows.sort(
        key=lambda row: (
            -float(candidate.aspect_ratios[row]),
            tuple(sorted(map(int, candidate.triangles[row]))),
            tuple(map(int, candidate.triangles[row])),
        )
    )
    scale = max(float(np.ptp(candidate.points, axis=0).max()), 1.0)
    tolerance = scale * 1.0e-12
    added: list[np.ndarray] = []
    for row in poor_rows:
        triangle = candidate.triangles[row]
        edges = {
            tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
            for index in range(3)
        }
        available = [edge for edge in edges if edge not in protected]
        if not available:
            continue
        edge = min(
            available,
            key=lambda item: (-float(np.linalg.norm(candidate.points[item[1]] - candidate.points[item[0]])), item),
        )
        midpoint = 0.5 * (candidate.points[edge[0]] + candidate.points[edge[1]])
        if np.any(np.linalg.norm(candidate.points - midpoint, axis=1) <= tolerance):
            continue
        if added and any(float(np.linalg.norm(value - midpoint)) <= tolerance for value in added):
            continue
        added.append(midpoint)
        if len(added) == limit:
            break
    return np.asarray(added, dtype=np.float64).reshape((-1, 2))


def _next_ids(existing: np.ndarray, count: int) -> np.ndarray:
    used = {int(value) for value in existing}
    candidate = max(used, default=0) + 1
    result: list[int] = []
    while len(result) < count:
        if candidate not in used:
            result.append(candidate)
            used.add(candidate)
        candidate += 1
    return np.asarray(result, dtype=np.int64)


def insert_midside_nodes(
    mesh: MeshCore,
    *,
    projector: Callable[[np.ndarray, int, int], Sequence[float]] | None = None,
) -> MeshCore:
    """Promote every T3/Q4 record to T6/Q8 using one node per shared edge."""

    if mesh.is_quadratic:
        assert_valid_mesh(mesh)
        return mesh
    all_elements: list[tuple[str, int, np.ndarray, bool, int]] = []
    for row, connectivity in enumerate(mesh.triangle_connectivity):
        all_elements.append(("triangle", row, connectivity, bool(mesh.triangle_active[row]), int(mesh.triangle_owner_handles[row])))
    for row, connectivity in enumerate(mesh.quad_connectivity):
        all_elements.append(("quad", row, connectivity, bool(mesh.quad_active[row]), int(mesh.quad_owner_handles[row])))
    edge_data: dict[tuple[int, int], dict[str, Any]] = {}
    for _, _, connectivity, active, owner_handle in all_elements:
        for edge in corner_edges(connectivity):
            data = edge_data.setdefault(edge, {"active": False, "owners": []})
            data["active"] = data["active"] or active
            if active and owner_handle >= 0:
                data["owners"].append(owner_handle)
    ordered_edges = sorted(
        edge_data,
        key=lambda edge: (
            min(int(mesh.node_ids[edge[0]]), int(mesh.node_ids[edge[1]])),
            max(int(mesh.node_ids[edge[0]]), int(mesh.node_ids[edge[1]])),
        ),
    )
    edge_to_row: dict[tuple[int, int], int] = {}
    coordinates: list[np.ndarray] = []
    owner_handles: list[int] = []
    active_nodes: list[bool] = []
    new_ids = _next_ids(mesh.node_ids, len(ordered_edges))
    for offset, edge in enumerate(ordered_edges):
        midpoint = 0.5 * (mesh.node_coordinates[edge[0]] + mesh.node_coordinates[edge[1]])
        if projector is not None:
            midpoint = np.asarray(
                projector(midpoint.copy(), int(mesh.node_ids[edge[0]]), int(mesh.node_ids[edge[1]])),
                dtype=np.float64,
            )
            if midpoint.shape not in ((2,), (3,)) or not np.all(np.isfinite(midpoint)):
                raise MeshError("midside projector must return a finite 2-vector or 3-vector")
            if midpoint.shape == (2,):
                midpoint = np.array((midpoint[0], midpoint[1], 0.0))
        coordinates.append(midpoint)
        data = edge_data[edge]
        owners = data["owners"]
        if owners and all(owner == owners[0] for owner in owners):
            owner_handle = owners[0]
        else:
            first_owner, second_owner = int(mesh.node_owner_handles[edge[0]]), int(mesh.node_owner_handles[edge[1]])
            owner_handle = first_owner if first_owner == second_owner else -1
        owner_handles.append(owner_handle)
        active_nodes.append(bool(data["active"]))
        edge_to_row[edge] = mesh.num_nodes + offset

    triangle_connectivity = np.empty((mesh.num_triangles, 6), dtype=np.int64)
    for row, element in enumerate(mesh.triangle_connectivity):
        triangle_connectivity[row, :3] = element[:3]
        triangle_connectivity[row, 3:] = [edge_to_row[edge] for edge in corner_edges(element)]
    quad_connectivity = np.empty((mesh.num_quads, 8), dtype=np.int64)
    for row, element in enumerate(mesh.quad_connectivity):
        quad_connectivity[row, :4] = element[:4]
        quad_connectivity[row, 4:] = [edge_to_row[edge] for edge in corner_edges(element)]
    appended_coordinates = np.asarray(coordinates, dtype=np.float64).reshape((-1, 3))
    result = MeshCore(
        np.vstack((mesh.node_coordinates, appended_coordinates)),
        triangle_connectivity,
        quad_connectivity,
        node_ids=np.concatenate((mesh.node_ids, new_ids)),
        triangle_ids=mesh.triangle_ids,
        quad_ids=mesh.quad_ids,
        owner_table=mesh.owner_table,
        node_owner_handles=np.concatenate((mesh.node_owner_handles, np.asarray(owner_handles, dtype=np.int32))),
        triangle_owner_handles=mesh.triangle_owner_handles,
        quad_owner_handles=mesh.quad_owner_handles,
        node_active=np.concatenate((mesh.node_active, np.asarray(active_nodes, dtype=bool))),
        triangle_active=mesh.triangle_active,
        quad_active=mesh.quad_active,
    )
    assert_valid_mesh(result)
    return result


to_quadratic = insert_midside_nodes


def mesh_planar_surface(
    outer: Any,
    holes: Sequence[Any] = (),
    constraints: Sequence[Any] = (),
    *,
    interior_points: Any | None = None,
    recombine: bool = True,
    order: str | int = "linear",
    target_size: float | None = None,
    backend: str | NativeBoundary | None = "auto",
    lattice_alignment: str = "chart",
    owner: Any | None = None,
    options: SurfaceMeshOptions | None = None,
    cancellation_check: Callable[[str], None] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> MeshCore:
    """Build a valid hybrid mesh of a 2D polygon or a planar 3D surface.

    Cancellation phase names are diagnostic. The callback is checked only at
    safe boundaries, so latency is bounded by the current uninterrupted phase.
    """

    if cancellation_check is not None:
        cancellation_check("native surface preprocessing")

    phase_seconds: dict[str, float] = {}
    preparation_started = perf_counter()
    settings = options or SurfaceMeshOptions(
        recombine=recombine,
        order=order,
        target_size=target_size,
        backend=backend,
        lattice_alignment=lattice_alignment,
    )
    raw_outer = np.asarray(outer, dtype=np.float64)
    plane = _plane_from_outer(raw_outer)
    dimension = raw_outer.shape[1]
    raw_holes = [_as_loop(hole, dimension, f"hole {number}") for number, hole in enumerate(holes)]
    raw_constraints = [np.asarray(segment, dtype=np.float64) for segment in constraints]
    if any(segment.shape != (2, dimension) for segment in raw_constraints):
        raise MeshError(f"each surface constraint must have shape (2, {dimension})")
    raw_interior = np.empty((0, dimension), dtype=float) if interior_points is None else np.asarray(interior_points, dtype=float)
    if raw_interior.ndim == 1 and raw_interior.size == 0:
        raw_interior = np.empty((0, dimension), dtype=float)
    if raw_interior.ndim != 2 or raw_interior.shape[1] != dimension:
        raise MeshError(f"interior_points must have shape (n, {dimension})")
    all_values = [raw_outer, *raw_holes, raw_interior, *(segment for segment in raw_constraints)]
    if dimension == 3:
        scale = max(float(np.ptp(raw_outer, axis=0).max()), 1.0)
        normal = np.cross(plane.first, plane.second)
        for values in all_values:
            if len(values) and np.max(np.abs((values - plane.origin) @ normal)) > scale * 1.0e-10:
                raise MeshError("surface points are not coplanar")
    planar_outer = plane.project(raw_outer)
    planar_holes = [plane.project(hole) for hole in raw_holes]
    planar_constraints = [plane.project(segment) for segment in raw_constraints]
    planar_interior = plane.project(raw_interior)
    explicit_interior = planar_interior.copy()
    generated = np.empty((0, 2), dtype=np.float64)
    phase_seconds["chart_projection_and_preparation"] = (
        perf_counter() - preparation_started
    )
    if settings.target_size is not None:
        densification_started = perf_counter()
        planar_outer = _densify_loop(planar_outer, settings.target_size)
        planar_holes = [_densify_loop(hole, settings.target_size) for hole in planar_holes]
        planar_constraints = [_densify_open(segment, settings.target_size)[[0, -1]] for segment in planar_constraints]
        phase_seconds["boundary_densification"] = (
            perf_counter() - densification_started
        )
        target_points_started = perf_counter()
        generated = _target_points(
            planar_outer,
            planar_holes,
            planar_constraints,
            settings.target_size,
            settings.lattice_alignment,
            settings.metric_tensor,
            max_lattice_points=settings.max_lattice_points,
            cancellation_check=cancellation_check,
        )
        planar_interior = np.vstack((planar_interior, generated))
        phase_seconds["target_point_generation"] = (
            perf_counter() - target_points_started
        )
    if cancellation_check is not None:
        cancellation_check("native surface triangulation start")
    triangulation_started = perf_counter()
    triangulation: PlanarTriangulation = triangulate_polygon(
        planar_outer,
        planar_holes,
        planar_constraints,
        interior_points=planar_interior,
        backend=settings.backend,
        cancellation_check=cancellation_check,
    )
    phase_seconds["triangulation_and_strict_qualification"] = (
        perf_counter() - triangulation_started
    )
    if cancellation_check is not None:
        cancellation_check("native surface triangulation complete")
        cancellation_check("native surface quality optimization start")

    optimization_started = perf_counter()
    initial_candidate = _make_candidate(
        triangulation.points,
        triangulation.triangles,
        settings=settings,
    )
    current = _optimize_candidate(
        initial_candidate,
        triangulation.segments,
        explicit_interior,
        settings,
    )
    best = current if current.score < initial_candidate.score else initial_candidate
    best_triangulation = triangulation
    attempted_added_points = 0
    attempted_rounds = 0
    point_budget = int(0.5 * len(generated))

    for round_number in range(1, 3):
        if not best.report["poor_element_ids"] or attempted_added_points >= point_budget:
            break
        if cancellation_check is not None:
            cancellation_check(f"native surface quality refinement round {round_number} start")
        remaining_budget = point_budget - attempted_added_points
        remaining_rounds = 3 - round_number
        round_budget = max(
            1,
            (remaining_budget + remaining_rounds - 1) // remaining_rounds,
        )
        additions = _refinement_midpoints(
            current,
            triangulation.segments,
            round_budget,
        )
        if not len(additions):
            break
        protected_nodes = {int(node) for edge in triangulation.segments for node in edge}
        interior_rows = [row for row in range(len(current.points)) if row not in protected_nodes]
        retry_interior = np.vstack((current.points[interior_rows], additions))
        retry_triangulation = triangulate_polygon(
            planar_outer,
            planar_holes,
            planar_constraints,
            interior_points=retry_interior,
            backend=settings.backend,
            cancellation_check=cancellation_check,
        )
        attempted_added_points += len(additions)
        attempted_rounds += 1
        retry = _make_candidate(
            retry_triangulation.points,
            retry_triangulation.triangles,
            added_points=attempted_added_points,
            rounds=attempted_rounds,
            settings=settings,
        )
        retry = _optimize_candidate(
            retry,
            retry_triangulation.segments,
            explicit_interior,
            settings,
        )
        current = retry
        triangulation = retry_triangulation
        if retry.score < best.score and retry.score[0] == 0:
            best = retry
            best_triangulation = retry_triangulation
        if cancellation_check is not None:
            cancellation_check(f"native surface quality refinement round {round_number} complete")

    triangulation = best_triangulation
    phase_seconds["quality_optimization"] = perf_counter() - optimization_started
    target_met = (
        best.report["invalid_element_count"] == 0
        and not best.report["poor_element_ids"]
    )
    quality_diagnostics = {
        "initial_quality": dict(initial_candidate.report),
        "final_quality": dict(best.report),
        "flips": best.flips,
        "moved_nodes": len(best.moved_nodes),
        "moved_node_rows": list(best.moved_nodes),
        "added_points": best.added_points,
        "rounds": best.rounds,
        "attempted_added_points": attempted_added_points,
        "attempted_rounds": attempted_rounds,
        "initial_generated_points": len(generated),
        "point_budget": point_budget,
        "target_met": target_met,
        "budget_exhausted": not target_met and attempted_added_points >= point_budget,
    }
    if cancellation_check is not None:
        cancellation_check("native surface quality optimization complete")
    coordinates = plane.lift(best.points)
    owner_table = () if owner is None else (owner,)
    owner_handle = -1 if owner is None else 0
    core = MeshCore(
        coordinates,
        best.triangles,
        owner_table=owner_table,
        node_owner_handles=np.full(len(coordinates), owner_handle, dtype=np.int32),
        triangle_owner_handles=np.full(len(best.triangles), owner_handle, dtype=np.int32),
    )
    recombination_diagnostics: dict[str, int | bool] = {
        "candidate_count": 0,
        "rejected_candidate_count": 0,
        "exchange_count": 0,
        "exchange_work": 0,
        "exchange_truncated": False,
        "pair_count": 0,
    }
    if settings.recombine:
        recombination_started = perf_counter()
        recombination_report = recombine_triangles_with_report(
            core,
            protected_edges=triangulation.segments,
            min_scaled_jacobian=settings.min_scaled_jacobian,
            max_aspect_ratio=settings.max_aspect_ratio,
            min_angle=settings.min_angle,
            max_angle=settings.max_angle,
            max_warpage=settings.max_warpage,
            cancellation_check=cancellation_check,
            max_exchange_work=settings.max_recombination_work,
        )
        core = recombination_report.mesh
        recombination_diagnostics = {
            "candidate_count": recombination_report.candidate_count,
            "rejected_candidate_count": (
                recombination_report.rejected_candidate_count
            ),
            "exchange_count": recombination_report.exchange_count,
            "exchange_work": recombination_report.exchange_work,
            "exchange_truncated": recombination_report.exchange_truncated,
            "pair_count": recombination_report.pair_count,
        }
        phase_seconds["recombination"] = perf_counter() - recombination_started
        if cancellation_check is not None:
            cancellation_check("native surface recombination complete")
    if settings.quadratic:
        promotion_started = perf_counter()
        core = insert_midside_nodes(core)
        phase_seconds["quadratic_promotion"] = perf_counter() - promotion_started
        if cancellation_check is not None:
            cancellation_check("native surface quadratic promotion complete")
    validation_started = perf_counter()
    assert_valid_mesh(core)
    threshold_report = _quality_threshold_report(evaluate_quality(core), settings)
    if settings.enforce_quality and not threshold_report["accepted"]:
        counts = ", ".join(
            f"{name}={count}"
            for name, count in threshold_report["violation_counts"].items()
            if count
        )
        poor = threshold_report["poor_element_ids"][:12]
        suffix = "..." if len(threshold_report["poor_element_ids"]) > len(poor) else ""
        raise MeshError(
            "mesh quality policy rejected the generated surface: "
            f"{counts}; poor element IDs {poor}{suffix}. "
            "Refine or partition the face, or relax the explicit quality policy."
        )
    phase_seconds["surface_validation"] = perf_counter() - validation_started
    if diagnostics is not None:
        diagnostics.update(
            {
                "requested_backend": triangulation.requested_backend,
                "selected_backend": triangulation.selected_backend,
                "actual_backend": triangulation.actual_backend,
                "fallback_reason": triangulation.fallback_reason,
                "phase_seconds": phase_seconds,
                "native_diagnostics": dict(triangulation.native_diagnostics),
                "quality_optimization": quality_diagnostics,
                "quality_policy": threshold_report,
                "lattice_alignment": settings.lattice_alignment,
                "metric_tensor": (
                    None
                    if settings.metric_tensor is None
                    else np.asarray(settings.metric_tensor, dtype=float).tolist()
                ),
                "max_lattice_points": settings.max_lattice_points,
                "recombination": recombination_diagnostics,
            }
        )
    if cancellation_check is not None:
        cancellation_check("native surface validation complete")
    return core


surface_mesh = mesh_planar_surface
generate_surface_mesh = mesh_planar_surface
