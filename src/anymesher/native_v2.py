"""Opt-in native-v2 options, mutable T3 topology and planar front refinement."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from hashlib import sha256
import heapq
import json
from math import acos, sqrt
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MeshError
from .metric import (
    ExperimentalMetricProvider,
    MetricFieldSpec,
    SpatialMetricField,
    _compose_metric_tensors,
    _metric_lengths,
    limit_metric_gradation,
    pullback_metric,
)
from .native_cpp import native_mutable_t3_insert
from .triangulation import PlanarTriangulation, incircle, orient2d

__all__ = [
    "ComponentSeedRegistry",
    "MutableT3Topology",
    "NativeMeshingOptions",
    "frontal_delaunay_refine",
]

_MAXIMUM_METRIC_EDGE_LENGTH = 1.4


def _edge(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _canonical_triangle(values: Sequence[int], points: np.ndarray) -> tuple[int, int, int]:
    triangle = [int(value) for value in values]
    if orient2d(points[triangle[0]], points[triangle[1]], points[triangle[2]]) < 0.0:
        triangle[1], triangle[2] = triangle[2], triangle[1]
    start = min(range(3), key=lambda row: triangle[row])
    return tuple(triangle[start:] + triangle[:start])  # type: ignore[return-value]


class _GeometryLimited(MeshError):
    """A deterministic candidate rejection, distinct from an operational failure."""


@dataclass(frozen=True)
class NativeMeshingOptions:
    """Additive native-v2 controls; defaults preserve the legacy byte path."""

    point_placement: str = "legacy_lattice"
    metric_mode: str = "legacy"
    metric_field: MetricFieldSpec | None = None
    experimental_metric_provider: ExperimentalMetricProvider | None = field(
        default=None, repr=False, compare=False
    )
    max_insertions: int = 10_000
    max_topology_operations: int = 1_000_000
    cancellation_interval: int = 256

    def __post_init__(self) -> None:
        if self.point_placement not in {"legacy_lattice", "frontal_delaunay"}:
            raise MeshError("point_placement must be 'legacy_lattice' or 'frontal_delaunay'")
        if self.metric_mode not in {"legacy", "isotropic_spatial"}:
            raise MeshError("metric_mode must be 'legacy' or 'isotropic_spatial'")
        if self.point_placement == "legacy_lattice" and self.metric_mode != "legacy":
            raise MeshError("legacy_lattice cannot carry spatial metric controls")
        if self.metric_mode == "legacy" and (
            self.metric_field is not None or self.experimental_metric_provider is not None
        ):
            raise MeshError("legacy metric mode cannot carry a spatial metric provider")
        if self.metric_field is not None and not isinstance(self.metric_field, MetricFieldSpec):
            if isinstance(self.metric_field, Mapping):
                object.__setattr__(self, "metric_field", MetricFieldSpec.from_dict(self.metric_field))
            else:
                raise MeshError("metric_field must be MetricFieldSpec")
        if self.experimental_metric_provider is not None and not isinstance(
            self.experimental_metric_provider, ExperimentalMetricProvider
        ):
            raise MeshError("runtime metric provider must be ExperimentalMetricProvider")
        if self.metric_field is not None and self.experimental_metric_provider is not None:
            raise MeshError(
                "metric_field and experimental_metric_provider are mutually exclusive"
            )
        for name in ("max_insertions", "max_topology_operations", "cancellation_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
                raise MeshError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @classmethod
    def coerce(cls, value: "NativeMeshingOptions | Mapping[str, Any] | None") -> "NativeMeshingOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise MeshError("native_options must be NativeMeshingOptions or a mapping")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        if self.experimental_metric_provider is not None:
            raise MeshError("runtime metric providers are not serializable or automation-safe")
        return {
            "schema": "anymesher.native-meshing-options/1",
            "point_placement": self.point_placement,
            "metric_mode": self.metric_mode,
            "metric_field": None if self.metric_field is None else self.metric_field.to_dict(),
            "max_insertions": self.max_insertions,
            "max_topology_operations": self.max_topology_operations,
            "cancellation_interval": self.cancellation_interval,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NativeMeshingOptions":
        expected = {
            "schema", "point_placement", "metric_mode", "metric_field",
            "max_insertions", "max_topology_operations", "cancellation_interval",
        }
        if set(raw) != expected or raw.get("schema") != "anymesher.native-meshing-options/1":
            raise MeshError("native meshing options contain an unknown field or schema")
        metric = raw["metric_field"]
        return cls(
            point_placement=raw["point_placement"], metric_mode=raw["metric_mode"],
            metric_field=None if metric is None else MetricFieldSpec.from_dict(metric),
            max_insertions=raw["max_insertions"],
            max_topology_operations=raw["max_topology_operations"],
            cancellation_interval=raw["cancellation_interval"],
        )

    @property
    def canonical_digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


class ComponentSeedRegistry:
    """Component-owned deterministic identities for splittable shared seeds."""

    def __init__(
        self,
        first_node_id: int,
        *,
        node_id_allocator: Callable[[], int] | None = None,
    ) -> None:
        self._next = int(first_node_id)
        self._values: dict[tuple[int, int, int], int] = {}
        self._node_id_allocator = node_id_allocator

    def resolve(self, edge_id: int, numerator: int, denominator: int) -> int:
        if denominator <= 0 or numerator <= 0 or numerator >= denominator:
            raise MeshError("shared-edge split position must be strictly interior")
        common = np.gcd(numerator, denominator)
        key = (int(edge_id), int(numerator // common), int(denominator // common))
        if key not in self._values:
            node_id = (
                self._next
                if self._node_id_allocator is None
                else int(self._node_id_allocator())
            )
            if node_id < 1 or node_id in self._values.values():
                raise MeshError("component seed allocator returned an invalid node identity")
            self._values[key] = node_id
            self._next = max(self._next, node_id + 1)
        return self._values[key]

    @property
    def assigned_node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._values.values()))


class MutableT3Topology:
    """Atomic deterministic mutable T3 state used by the frontal queue."""

    def __init__(
        self,
        points: Any,
        triangles: Any,
        protected_edges: Any = (),
        *,
        node_owners: Any | None = None,
        triangle_owners: Any | None = None,
        splittable_edges: Mapping[
            tuple[int, int], int | tuple[int, Any, Any]
        ] | None = None,
        seed_registry: ComponentSeedRegistry | None = None,
    ) -> None:
        self._points = np.ascontiguousarray(points, dtype=np.float64).copy()
        self._triangles = np.ascontiguousarray(triangles, dtype=np.int64).reshape((-1, 3)).copy()
        self._protected_edges = set(_edge(int(a), int(b)) for a, b in np.asarray(protected_edges, dtype=np.int64).reshape((-1, 2)))
        self._seed_registry = seed_registry
        self._splittable_intervals: dict[
            tuple[int, int], tuple[int, Fraction, Fraction]
        ] = {}
        for edge, raw_interval in dict(splittable_edges or {}).items():
            if isinstance(raw_interval, Integral) and not isinstance(raw_interval, bool):
                edge_id, lower, upper = int(raw_interval), Fraction(0), Fraction(1)
            else:
                try:
                    edge_id, raw_lower, raw_upper = raw_interval  # type: ignore[misc]
                except (TypeError, ValueError) as error:
                    raise MeshError("splittable edge metadata is invalid") from error
                edge_id = int(edge_id)
                lower, upper = Fraction(raw_lower), Fraction(raw_upper)
            if edge_id < 1 or not Fraction(0) <= lower < upper <= Fraction(1):
                raise MeshError("splittable edge interval is invalid")
            self._splittable_intervals[_edge(*edge)] = (edge_id, lower, upper)
        if self._protected_edges.intersection(self._splittable_intervals):
            raise MeshError("a mutable T3 edge cannot be both protected and splittable")
        self.node_owners = np.full(len(self._points), -1, dtype=np.int64) if node_owners is None else np.ascontiguousarray(node_owners, dtype=np.int64).copy()
        self.triangle_owners = np.full(len(self._triangles), -1, dtype=np.int64) if triangle_owners is None else np.ascontiguousarray(triangle_owners, dtype=np.int64).copy()
        self.epoch = 0
        self.free_triangle_ids: list[int] = []
        self.quality_cache: dict[tuple[int, int, int], tuple[float, ...]] = {}
        self._shared_node_ids: dict[int, tuple[int, int, Fraction]] = {}
        self.validate()

    @property
    def protected_edges(self) -> frozenset[tuple[int, int]]:
        return frozenset(self._protected_edges)

    @property
    def splittable_edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._splittable_intervals))

    @property
    def constraint_edges(self) -> np.ndarray:
        return np.asarray(
            sorted(self._protected_edges | set(self._splittable_intervals)),
            dtype=np.int64,
        ).reshape((-1, 2))

    @property
    def shared_node_ids(self) -> Mapping[int, tuple[int, int, Fraction]]:
        return dict(self._shared_node_ids)

    @property
    def points(self) -> np.ndarray:
        result = self._points.copy()
        result.setflags(write=False)
        return result

    @property
    def triangles(self) -> np.ndarray:
        result = self._triangles.copy()
        result.setflags(write=False)
        return result

    def _incidence(
        self,
        triangles: np.ndarray | None = None,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> dict[tuple[int, int], list[int]]:
        rows = self._triangles if triangles is None else triangles
        result: dict[tuple[int, int], list[int]] = {}
        for row, triangle in enumerate(rows):
            if cancellation_check is not None and row % 4096 == 0:
                cancellation_check("native-v2 incidence scan")
            for index in range(3):
                result.setdefault(_edge(int(triangle[index]), int(triangle[(index + 1) % 3])), []).append(row)
        return result

    def validate(self) -> None:
        if self._points.ndim != 2 or self._points.shape[1] != 2 or not np.all(np.isfinite(self._points)):
            raise MeshError("mutable T3 points must be finite 2D rows")
        if self._triangles.ndim != 2 or self._triangles.shape[1] != 3:
            raise MeshError("mutable T3 connectivity must contain triangle rows")
        if len(self.node_owners) != len(self._points) or len(self.triangle_owners) != len(self._triangles):
            raise MeshError("mutable T3 owner arrays do not match topology")
        if np.any(self._triangles < 0) or np.any(self._triangles >= len(self._points)):
            raise MeshError("mutable T3 connectivity is out of range")
        canonical = [_canonical_triangle(row, self._points) for row in self._triangles]
        if len(canonical) != len(set(canonical)):
            raise MeshError("mutable T3 topology contains duplicate triangles")
        if any(orient2d(self._points[a], self._points[b], self._points[c]) <= 0.0 for a, b, c in canonical):
            raise MeshError("mutable T3 topology contains a non-positive triangle")
        incidence = self._incidence()
        if any(len(rows) > 2 for rows in incidence.values()):
            raise MeshError("mutable T3 topology contains a non-manifold edge")
        missing = sorted(self.protected_edges.difference(incidence))
        if missing:
            raise MeshError(f"mutable T3 topology lost protected edges: {missing[:3]}")
        missing_splittable = sorted(set(self._splittable_intervals).difference(incidence))
        if missing_splittable:
            raise MeshError(
                f"mutable T3 topology lost splittable edges: {missing_splittable[:3]}"
            )

    def locate(
        self,
        point: Sequence[float],
        *,
        cancellation_check: Callable[[str], None] | None = None,
        cancellation_interval: int = 4096,
    ) -> int | None:
        value = np.asarray(point, dtype=np.float64)
        for row, triangle in enumerate(self._triangles):
            if cancellation_check is not None and row % cancellation_interval == 0:
                cancellation_check("native-v2 fallback location scan")
            coordinates = self._points[triangle]
            signs = [orient2d(coordinates[index], coordinates[(index + 1) % 3], value) for index in range(3)]
            scale = max(float(np.ptp(coordinates, axis=0).max()), 1.0)
            if min(signs) >= -64.0 * np.finfo(float).eps * scale * scale:
                return row
        return None

    def _python_insert_with_owners(
        self,
        candidate: np.ndarray,
        *,
        owner: int = -1,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        scale = max(float(np.ptp(self._points, axis=0).max()), 1.0)
        tolerance = 64.0 * np.finfo(float).eps * scale
        if np.min(np.linalg.norm(self._points - candidate, axis=1)) <= tolerance:
            raise _GeometryLimited("frontal candidate duplicates an existing point")
        for edge_number, (first, second) in enumerate(self.protected_edges):
            if cancellation_check is not None and edge_number % 4096 == 0:
                cancellation_check("native-v2 protected-edge scan")
            edge_vector = self._points[second] - self._points[first]
            cross = abs(float(edge_vector[0] * (candidate[1] - self._points[first, 1]) - edge_vector[1] * (candidate[0] - self._points[first, 0])))
            if cross <= tolerance * max(float(np.linalg.norm(edge_vector)), 1.0) and np.all(candidate >= np.minimum(self._points[first], self._points[second]) - tolerance) and np.all(candidate <= np.maximum(self._points[first], self._points[second]) + tolerance):
                raise _GeometryLimited("frontal candidate encroaches an unsplittable protected edge")
        bad_triangles = []
        for row, triangle in enumerate(self._triangles):
            if cancellation_check is not None and row % 4096 == 0:
                cancellation_check("native-v2 cavity scan")
            a, b, c = (self._points[int(value)] for value in triangle)
            if incircle(a, b, c, candidate) > tolerance * tolerance:
                bad_triangles.append(row)
        seed = None
        for bad_number, row in enumerate(bad_triangles):
            if cancellation_check is not None and bad_number % 4096 == 0:
                cancellation_check("native-v2 cavity seed scan")
            if min(
                orient2d(
                    self._points[int(self._triangles[row, index])],
                    self._points[int(self._triangles[row, (index + 1) % 3])],
                    candidate,
                )
                for index in range(3)
            ) >= -tolerance:
                seed = row
                break
        if seed is None:
            located = self.locate(
                candidate, cancellation_check=cancellation_check
            )
            if located is None:
                raise _GeometryLimited("frontal candidate lies outside the mutable triangulation")
            cavity = [located]
        else:
            edge_rows: dict[tuple[int, int], list[int]] = {}
            for bad_number, row in enumerate(bad_triangles):
                if cancellation_check is not None and bad_number % 4096 == 0:
                    cancellation_check("native-v2 cavity adjacency scan")
                triangle = self._triangles[row]
                for index in range(3):
                    edge_rows.setdefault(
                        _edge(int(triangle[index]), int(triangle[(index + 1) % 3])),
                        [],
                    ).append(row)
            selected = {seed}
            frontier = [seed]
            for frontier_number, row in enumerate(frontier):
                if cancellation_check is not None and frontier_number % 4096 == 0:
                    cancellation_check("native-v2 cavity component scan")
                triangle = self._triangles[row]
                adjacent = {
                    other
                    for index in range(3)
                    for other in edge_rows[
                        _edge(int(triangle[index]), int(triangle[(index + 1) % 3]))
                    ]
                }
                for other in sorted(adjacent.difference(selected)):
                    selected.add(other)
                    frontier.append(other)
            cavity = sorted(selected)
        counts: dict[tuple[int, int], int] = {}
        edge_owners: dict[tuple[int, int], list[int]] = {}
        for row in cavity:
            triangle = self._triangles[row]
            for index in range(3):
                edge = _edge(int(triangle[index]), int(triangle[(index + 1) % 3]))
                counts[edge] = counts.get(edge, 0) + 1
                edge_owners.setdefault(edge, []).append(int(self.triangle_owners[row]))
        if any(count == 2 and edge in self.protected_edges for edge, count in counts.items()):
            raise _GeometryLimited("frontal cavity would remove an unsplittable protected edge")
        boundary = sorted(edge for edge, count in counts.items() if count == 1)
        new_id = len(self._points)
        extended = np.vstack((self._points, candidate))
        cavity_set = set(cavity)
        retained = [
            (tuple(map(int, triangle)), int(self.triangle_owners[row]))
            for row, triangle in enumerate(self._triangles)
            if row not in cavity_set
        ]
        retained.extend(
            (
                _canonical_triangle((a, b, new_id), extended),
                int(owner) if int(owner) != -1 else edge_owners[(a, b)][0],
            )
            for a, b in boundary
        )
        ordered = sorted(retained, key=lambda item: item[0])
        result = np.asarray([row for row, _ in ordered], dtype=np.int64).reshape((-1, 3))
        owners = np.asarray([value for _, value in ordered], dtype=np.int64)
        return result, owners, {"removed_triangles": len(cavity), "added_triangles": len(boundary), "native": False}

    def _python_insert(
        self, candidate: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the established two-value Python insertion oracle contract."""

        triangles, _owners, report = self._python_insert_with_owners(candidate)
        return triangles, report

    def _native_insert_owners(
        self,
        triangles: np.ndarray,
        *,
        inserted_node: int,
        owner: int,
    ) -> np.ndarray:
        retained_owners = {
            tuple(map(int, triangle)): int(self.triangle_owners[row])
            for row, triangle in enumerate(self._triangles)
        }
        result_identities = {tuple(map(int, triangle)) for triangle in triangles}
        edge_owners: dict[tuple[int, int], list[int]] = {}
        for row, triangle in enumerate(self._triangles):
            if tuple(map(int, triangle)) in result_identities:
                continue
            triangle_owner = int(self.triangle_owners[row])
            for index in range(3):
                edge = _edge(
                    int(triangle[index]), int(triangle[(index + 1) % 3])
                )
                edge_owners.setdefault(edge, []).append(triangle_owner)
        values: list[int] = []
        for triangle in triangles:
            identity = tuple(map(int, triangle))
            retained = retained_owners.get(identity)
            if retained is not None:
                values.append(retained)
                continue
            if inserted_node not in identity:
                raise MeshError("compiled mutable T3 insertion returned an unknown cell")
            opposite = [value for value in identity if value != inserted_node]
            sources = edge_owners.get(_edge(opposite[0], opposite[1]))
            if not sources:
                raise MeshError(
                    "compiled mutable T3 insertion returned an ownerless cavity cell"
                )
            values.append(int(owner) if int(owner) != -1 else sources[0])
        return np.asarray(values, dtype=np.int64)

    def insert_point(
        self,
        point: Sequence[float],
        *,
        owner: int = -1,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        candidate = np.asarray(point, dtype=np.float64)
        if candidate.shape != (2,) or not np.all(np.isfinite(candidate)):
            raise MeshError("frontal insertion point must be one finite 2D coordinate")
        if cancellation_check is not None:
            cancellation_check("native-v2 mutable insertion start")
        bounded_oracle = len(self._triangles) <= 4096
        reference: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None
        if bounded_oracle or cancellation_check is not None:
            reference = self._python_insert_with_owners(
                candidate,
                owner=owner,
                cancellation_check=cancellation_check,
            )
        native = (
            None
            if cancellation_check is not None and not bounded_oracle
            else native_mutable_t3_insert(
                self._points,
                self._triangles,
                np.asarray(sorted(self.protected_edges), dtype=np.int64).reshape((-1, 2)),
                candidate,
            )
        )
        if native is None:
            if reference is None:
                reference = self._python_insert_with_owners(candidate, owner=owner)
            new_triangles, reference_owners, report = reference
        else:
            new_triangles, report = native
            if reference is not None:
                reference_triangles, reference_owners, _ = reference
                if not np.array_equal(new_triangles, reference_triangles):
                    raise MeshError(
                        "compiled mutable T3 insertion disagrees with the Python oracle"
                    )
            else:
                reference_owners = self._native_insert_owners(
                    new_triangles,
                    inserted_node=len(self._points),
                    owner=owner,
                )
        if cancellation_check is not None:
            cancellation_check("native-v2 mutable insertion commit")
        old_points = self._points
        old_triangles = self._triangles
        old_node_owners = self.node_owners
        old_triangle_owners = self.triangle_owners
        try:
            self._points = np.vstack((self._points, candidate))
            self._triangles = np.ascontiguousarray(new_triangles, dtype=np.int64)
            self.node_owners = np.append(self.node_owners, int(owner))
            self.triangle_owners = reference_owners
            self.validate()
        except BaseException:
            self._points, self._triangles = old_points, old_triangles
            self.node_owners, self.triangle_owners = old_node_owners, old_triangle_owners
            raise
        self.epoch += 1
        self.quality_cache.clear()
        return {**report, "epoch": self.epoch, "point_id": len(self._points) - 1}

    def split_segment(
        self,
        edge: tuple[int, int],
        *,
        owner: int = -1,
        cancellation_check: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Split one registry-authorized automatically seeded shared edge."""

        target = _edge(*edge)
        if target in self._protected_edges:
            raise MeshError("protected or mandatory segments are unsplittable")
        interval = self._splittable_intervals.get(target)
        if interval is None or self._seed_registry is None:
            raise MeshError("segment is not an automatically seeded shared edge")
        attached = tuple(
            self._incidence(cancellation_check=cancellation_check).get(target, ())
        )
        if not attached or len(attached) > 2:
            raise MeshError("shared segment has invalid mutable incidence")
        if cancellation_check is not None:
            cancellation_check("native-v2 shared segment split start")
        edge_id, lower, upper = interval
        station = (lower + upper) / 2
        shared_node_id = self._seed_registry.resolve(
            edge_id, station.numerator, station.denominator
        )
        candidate = 0.5 * (self._points[target[0]] + self._points[target[1]])
        new_id = len(self._points)
        extended = np.vstack((self._points, candidate))
        retained_rows: list[tuple[int, int, int]] = []
        retained_owners: list[int] = []
        attached_set = set(attached)
        for row_id, triangle in enumerate(self._triangles):
            if cancellation_check is not None and row_id % 4096 == 0:
                cancellation_check("native-v2 shared segment reconstruction")
            if row_id not in attached_set:
                retained_rows.append(tuple(map(int, triangle)))
                retained_owners.append(int(self.triangle_owners[row_id]))
                continue
            opposite = next(int(node) for node in triangle if int(node) not in target)
            for values in (
                (target[0], new_id, opposite),
                (new_id, target[1], opposite),
            ):
                retained_rows.append(_canonical_triangle(values, extended))
                retained_owners.append(int(self.triangle_owners[row_id]))
        ordered = sorted(zip(retained_rows, retained_owners), key=lambda item: item[0])
        new_triangles = np.asarray([row for row, _ in ordered], dtype=np.int64)
        new_owners = np.asarray([value for _, value in ordered], dtype=np.int64)
        if cancellation_check is not None:
            cancellation_check("native-v2 shared segment split commit")
        old_state = (
            self._points,
            self._triangles,
            self.node_owners,
            self.triangle_owners,
            dict(self._splittable_intervals),
            dict(self._shared_node_ids),
        )
        try:
            self._points = extended
            self._triangles = new_triangles
            self.node_owners = np.append(self.node_owners, int(owner))
            self.triangle_owners = new_owners
            del self._splittable_intervals[target]
            self._splittable_intervals[_edge(target[0], new_id)] = (
                edge_id, lower, station
            )
            self._splittable_intervals[_edge(new_id, target[1])] = (
                edge_id, station, upper
            )
            self._shared_node_ids[new_id] = (shared_node_id, edge_id, station)
            self.validate()
        except BaseException:
            (
                self._points,
                self._triangles,
                self.node_owners,
                self.triangle_owners,
                self._splittable_intervals,
                self._shared_node_ids,
            ) = old_state
            raise
        self.epoch += 1
        self.quality_cache.clear()
        return {
            "epoch": self.epoch,
            "point_id": new_id,
            "shared_node_id": shared_node_id,
            "edge_id": edge_id,
            "station": (station.numerator, station.denominator),
        }

    def flip_edge(self, edge: tuple[int, int]) -> bool:
        target = _edge(*edge)
        if target in self.protected_edges:
            return False
        incidence = self._incidence()
        attached = incidence.get(target, ())
        if len(attached) != 2:
            return False
        first, second = (self._triangles[row] for row in attached)
        attached_owners = tuple(int(self.triangle_owners[row]) for row in attached)
        if attached_owners[0] != attached_owners[1]:
            return False
        opposite = [next(int(node) for node in triangle if int(node) not in target) for triangle in (first, second)]
        replacement_edge = _edge(*opposite)
        if replacement_edge in incidence:
            return False
        candidates = [
            _canonical_triangle((opposite[0], opposite[1], target[0]), self._points),
            _canonical_triangle((opposite[1], opposite[0], target[1]), self._points),
        ]
        if any(orient2d(self._points[a], self._points[b], self._points[c]) <= 0.0 for a, b, c in candidates):
            return False
        trial = [
            (tuple(map(int, row)), int(self.triangle_owners[number]))
            for number, row in enumerate(self._triangles)
            if number not in attached
        ]
        trial.extend((candidate, attached_owners[0]) for candidate in candidates)
        ordered = sorted(trial, key=lambda item: item[0])
        previous = self._triangles
        previous_owners = self.triangle_owners
        self._triangles = np.asarray([row for row, _ in ordered], dtype=np.int64)
        self.triangle_owners = np.asarray([value for _, value in ordered], dtype=np.int64)
        try:
            self.validate()
        except MeshError:
            self._triangles = previous
            self.triangle_owners = previous_owners
            return False
        self.epoch += 1
        self.quality_cache.clear()
        return True

    def canonical_export(self) -> tuple[np.ndarray, np.ndarray]:
        points = np.ascontiguousarray(self._points, dtype=np.float64)
        triangles = np.asarray(sorted(_canonical_triangle(row, points) for row in self._triangles), dtype=np.int64)
        points.setflags(write=False)
        triangles.setflags(write=False)
        return points, triangles


def _inside_ring(point: np.ndarray, points: np.ndarray, ring: np.ndarray) -> bool:
    inside = False
    x, y = float(point[0]), float(point[1])
    for index, first_id in enumerate(ring):
        second_id = ring[(index + 1) % len(ring)]
        first, second = points[int(first_id)], points[int(second_id)]
        if (first[1] > y) != (second[1] > y):
            crossing = float(first[0] + (y - first[1]) * (second[0] - first[0]) / (second[1] - first[1]))
            if crossing > x:
                inside = not inside
    return inside


def _angles(coordinates: np.ndarray) -> tuple[float, float, float]:
    result = []
    for index in range(3):
        first = coordinates[(index - 1) % 3] - coordinates[index]
        second = coordinates[(index + 1) % 3] - coordinates[index]
        denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1.0e-30)
        result.append(float(np.degrees(acos(float(np.clip(first @ second / denominator, -1.0, 1.0))))))
    return tuple(result)  # type: ignore[return-value]


def _offcentre(coordinates: np.ndarray, tensors: np.ndarray, fallback_size: float) -> np.ndarray:
    lengths = [float(np.linalg.norm(coordinates[(i + 1) % 3] - coordinates[i])) for i in range(3)]
    edge_index = min(range(3), key=lambda row: (lengths[row], row))
    first = coordinates[edge_index]
    second = coordinates[(edge_index + 1) % 3]
    opposite = coordinates[(edge_index + 2) % 3]
    midpoint = 0.5 * (first + second)
    direction = second - first
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
    if float((opposite - midpoint) @ normal) < 0.0:
        normal = -normal
    eigenvalue = float(np.max(np.linalg.eigvalsh(np.mean(tensors, axis=0))))
    target = min(fallback_size, 1.0 / sqrt(max(eigenvalue, 1.0e-30)))
    half = 0.5 * lengths[edge_index]
    height = sqrt(max(target * target - half * half, (0.35 * target) ** 2))
    candidate = midpoint + normal * height
    return candidate


def _circumcentre(coordinates: np.ndarray) -> np.ndarray | None:
    first, second, third = coordinates
    shifted_second = second - first
    shifted_third = third - first
    determinant = 2.0 * (
        shifted_second[0] * shifted_third[1]
        - shifted_second[1] * shifted_third[0]
    )
    scale = max(float(np.ptp(coordinates, axis=0).max()), 1.0)
    if abs(determinant) <= 64.0 * np.finfo(float).eps * scale * scale:
        return None
    rhs = np.asarray(
        (float(shifted_second @ shifted_second), float(shifted_third @ shifted_third))
    )
    matrix = np.asarray(
        ((shifted_second[0], shifted_second[1]), (shifted_third[0], shifted_third[1]))
    )
    return first + np.linalg.solve(2.0 * matrix, rhs)


def frontal_delaunay_refine(
    triangulation: PlanarTriangulation,
    options: NativeMeshingOptions,
    *,
    target_size: float,
    cancellation_check: Any = None,
    model_uuid: str | None = None,
    geometry_revision: int | None = None,
    metric_to_physical: Callable[[np.ndarray], np.ndarray] | None = None,
    metric_jacobian: np.ndarray | None = None,
    automatically_seeded_shared_segments: Mapping[
        tuple[int, int], int | tuple[int, Any, Any]
    ] | None = None,
    component_seed_registry: ComponentSeedRegistry | None = None,
    supplemental_metric_field: MetricFieldSpec | None = None,
    qualified_seed: bool = False,
) -> tuple[PlanarTriangulation, dict[str, Any]]:
    """Refine one qualified planar CDT through a deterministic bounded queue."""

    if options.point_placement != "frontal_delaunay":
        raise MeshError("frontal refinement requires point_placement='frontal_delaunay'")
    provider: Any
    metric_spec: MetricFieldSpec | None = None
    if supplemental_metric_field is not None and options.experimental_metric_provider is None:
        raise MeshError("supplemental metric field requires an experimental provider")
    if options.metric_mode == "legacy":
        metric_spec = MetricFieldSpec.uniform(target_size)
        provider = SpatialMetricField(metric_spec)
    elif options.experimental_metric_provider is not None:
        provider = options.experimental_metric_provider
        supplemental_metric_field = (
            supplemental_metric_field or MetricFieldSpec.uniform(target_size)
        )
    else:
        metric_spec = options.metric_field or MetricFieldSpec.uniform(target_size)
        provider = SpatialMetricField(
            metric_spec,
            model_uuid=model_uuid,
            geometry_revision=geometry_revision,
        )

    def evaluate_spec(
        specification: MetricFieldSpec,
        field: SpatialMetricField,
        points: np.ndarray,
    ) -> np.ndarray:
        if specification.spatial_dimension == 3:
            if metric_to_physical is None or metric_jacobian is None:
                raise MeshError("3D metric controls require a physical chart binding")
            physical_points = np.ascontiguousarray(metric_to_physical(points), dtype=np.float64)
            physical_tensors = field.evaluate(
                physical_points,
                cancellation_check=cancellation_check,
                cancellation_interval=options.cancellation_interval,
            )
            result = np.empty((len(physical_tensors), 2, 2), dtype=np.float64)
            for row, tensor in enumerate(physical_tensors):
                if cancellation_check is not None and row % options.cancellation_interval == 0:
                    cancellation_check("native-v2 metric pullback scan")
                result[row] = pullback_metric(tensor, metric_jacobian)
            return result
        return field.evaluate(
            points,
            cancellation_check=cancellation_check,
            cancellation_interval=options.cancellation_interval,
        )

    supplemental_provider = (
        None
        if supplemental_metric_field is None
        else SpatialMetricField(
            supplemental_metric_field,
            model_uuid=model_uuid,
            geometry_revision=geometry_revision,
        )
    )

    def evaluate_metric(points: np.ndarray) -> np.ndarray:
        if isinstance(provider, SpatialMetricField):
            assert metric_spec is not None
            return evaluate_spec(metric_spec, provider, points)
        if cancellation_check is not None:
            cancellation_check("native-v2 experimental metric evaluation start")
        result = provider.evaluate(points)
        if supplemental_provider is not None and supplemental_metric_field is not None:
            supplement = evaluate_spec(
                supplemental_metric_field, supplemental_provider, points
            )
            result = _compose_metric_tensors(
                result,
                supplement,
                cancellation_check=cancellation_check,
                cancellation_interval=options.cancellation_interval,
            )
        if cancellation_check is not None:
            cancellation_check("native-v2 experimental metric evaluation complete")
        return result

    splittable = {
        _edge(*edge): interval
        for edge, interval in dict(automatically_seeded_shared_segments or {}).items()
    }
    protected = [
        tuple(map(int, edge))
        for edge in np.asarray(triangulation.segments, dtype=np.int64)
        if _edge(*map(int, edge)) not in splittable
    ]
    topology = MutableT3Topology(
        triangulation.points,
        triangulation.triangles,
        protected,
        splittable_edges=splittable,
        seed_registry=component_seed_registry,
    )
    rejected: set[tuple[int, int]] = set()
    insertions = 0
    operations = 0
    geometry_limited = 0
    geometry_limited_triangles: set[tuple[int, int, int]] = set()
    stale_entries = 0
    shared_segment_splits = 0
    gradation_iterations = 0
    metric_minimum = float("inf")
    metric_maximum = 0.0
    route = "frontal_delaunay"
    while insertions < options.max_insertions and operations < options.max_topology_operations:
        if cancellation_check is not None and operations % options.cancellation_interval == 0:
            cancellation_check("native-v2 frontal queue")
        points, triangles = topology.canonical_export()
        tensors = evaluate_metric(points)
        topology_edge_set: set[tuple[int, int]] = set()
        for triangle_number, triangle in enumerate(triangles):
            if cancellation_check is not None and triangle_number % options.cancellation_interval == 0:
                cancellation_check("native-v2 topology-edge scan")
            for index in range(3):
                topology_edge_set.add(
                    _edge(int(triangle[index]), int(triangle[(index + 1) % 3]))
                )
        topology_edges = np.asarray(sorted(topology_edge_set), dtype=np.int64).reshape((-1, 2))
        target_lengths = 1.0 / np.sqrt(
            np.maximum(np.linalg.eigvalsh(tensors)[:, -1], 1.0e-30)
        )
        maximum_gradation = 1.5 if metric_spec is None else metric_spec.maximum_gradation
        limited_lengths, limited_iterations = limit_metric_gradation(
            points,
            topology_edges,
            target_lengths,
            maximum_gradation,
            cancellation_check=cancellation_check,
            cancellation_interval=options.cancellation_interval,
        )
        gradation_iterations = max(gradation_iterations, limited_iterations)
        factors = np.square(target_lengths / limited_lengths)
        tensors = tensors * factors[:, None, None]
        metric_minimum = min(metric_minimum, float(np.min(limited_lengths)))
        metric_maximum = max(metric_maximum, float(np.max(limited_lengths)))
        segment_queue: list[tuple[float, tuple[int, int]]] = []
        splittable_edges = np.asarray(
            topology.splittable_edges, dtype=np.int64
        ).reshape((-1, 2))
        splittable_lengths = _metric_lengths(
            points,
            splittable_edges,
            tensors,
            cancellation_check=cancellation_check,
            cancellation_interval=options.cancellation_interval,
        )
        for edge_number, (edge, length) in enumerate(
            zip(topology.splittable_edges, splittable_lengths)
        ):
            if cancellation_check is not None and edge_number % options.cancellation_interval == 0:
                cancellation_check("native-v2 segment queue scan")
            if length > _MAXIMUM_METRIC_EDGE_LENGTH * (1.0 + 1.0e-12):
                heapq.heappush(
                    segment_queue, (-length / _MAXIMUM_METRIC_EDGE_LENGTH, edge)
                )
        if segment_queue:
            _, edge = heapq.heappop(segment_queue)
            operations += 1
            topology.split_segment(
                edge,
                cancellation_check=cancellation_check,
            )
            insertions += 1
            shared_segment_splits += 1
            continue
        if qualified_seed and operations == 0:
            qualified_lengths = _metric_lengths(
                points,
                topology_edges,
                tensors,
                cancellation_check=cancellation_check,
                cancellation_interval=options.cancellation_interval,
            )
            if not len(qualified_lengths) or float(
                np.max(qualified_lengths)
            ) <= _MAXIMUM_METRIC_EDGE_LENGTH * (1.0 + 1.0e-12):
                route = "frontal_delaunay_baseline_satisfied"
                break
        queue: list[tuple[float, int, tuple[int, int, int], int]] = []
        for triangle_number, triangle in enumerate(triangles):
            if cancellation_check is not None and triangle_number % options.cancellation_interval == 0:
                cancellation_check("native-v2 triangle queue scan")
            coordinates = points[triangle]
            center_tensor = np.mean(tensors[triangle], axis=0)
            metric_lengths = []
            for index in range(3):
                delta = coordinates[(index + 1) % 3] - coordinates[index]
                metric_lengths.append(sqrt(max(float(delta @ center_tensor @ delta), 0.0)))
            angles = _angles(coordinates)
            maximum_metric_length = max(metric_lengths)
            minimum_angle = min(angles)
            minimum_angle_index = int(np.argmin(angles))
            corner = int(triangle[minimum_angle_index])
            prior = int(triangle[(minimum_angle_index - 1) % 3])
            following = int(triangle[(minimum_angle_index + 1) % 3])
            fixed_corner_edges = {_edge(corner, prior), _edge(corner, following)}
            identity = tuple(map(int, triangle))
            if (
                minimum_angle < 30.0
                and maximum_metric_length
                <= _MAXIMUM_METRIC_EDGE_LENGTH * (1.0 + 1.0e-12)
                and fixed_corner_edges.issubset(topology.protected_edges)
            ):
                if identity not in geometry_limited_triangles:
                    geometry_limited_triangles.add(identity)
                    geometry_limited += 1
                continue
            severity = max(
                maximum_metric_length / _MAXIMUM_METRIC_EDGE_LENGTH,
                30.0 / max(minimum_angle, 1.0e-12),
            )
            if severity > 1.0 + 1.0e-12:
                heapq.heappush(queue, (-severity, 1, identity, topology.epoch))
        if not queue:
            if geometry_limited:
                route = "frontal_delaunay_geometry_limited"
            break
        accepted = False
        triangle_lookup: dict[tuple[int, int, int], np.ndarray] = {}
        for lookup_number, row in enumerate(topology.triangles):
            if cancellation_check is not None and lookup_number % options.cancellation_interval == 0:
                cancellation_check("native-v2 triangle lookup scan")
            triangle_lookup[tuple(map(int, row))] = row
        while queue:
            if cancellation_check is not None and operations % options.cancellation_interval == 0:
                cancellation_check("native-v2 triangle queue processing")
            negative_severity, _, identity, generation = heapq.heappop(queue)
            operations += 1
            if generation != topology.epoch:
                stale_entries += 1
                continue
            triangle = triangle_lookup.get(identity)
            if triangle is None:
                stale_entries += 1
                continue
            coordinates = topology.points[triangle]
            local_tensors = evaluate_metric(coordinates)
            proposals = [_offcentre(coordinates, local_tensors, target_size)]
            circumcentre = _circumcentre(coordinates)
            if circumcentre is not None:
                proposals.append(circumcentre)
            for candidate in proposals:
                key = tuple(np.rint(candidate / max(target_size * 1.0e-8, 1.0e-14)).astype(np.int64))
                if key in rejected:
                    continue
                all_points = topology.points
                if not _inside_ring(candidate, all_points, triangulation.outer_loop) or any(
                    _inside_ring(candidate, all_points, ring) for ring in triangulation.hole_loops
                ):
                    rejected.add(key)
                    geometry_limited += 1
                    continue
                try:
                    topology.insert_point(
                        candidate,
                        cancellation_check=cancellation_check,
                    )
                except _GeometryLimited:
                    rejected.add(key)
                    geometry_limited += 1
                    continue
                insertions += 1
                accepted = True
                break
            if accepted:
                break
        if not accepted:
            route = "frontal_delaunay_geometry_limited"
            break
    if operations >= options.max_topology_operations or insertions >= options.max_insertions:
        route = "frontal_delaunay_budget_limited"
    elif insertions == 0 and route == "frontal_delaunay":
        route = "frontal_delaunay_baseline_satisfied"
    points, triangles = topology.canonical_export()
    native_diagnostics = dict(triangulation.native_diagnostics)
    report = {
        "selected_route": route,
        "metric_mode": options.metric_mode,
        "insertions": insertions,
        "topology_operations": operations,
        "geometry_limited_regions": geometry_limited,
        "rejection_cache_entries": len(rejected),
        "stale_queue_entries": stale_entries,
        "shared_segment_splits": shared_segment_splits,
        "metric_target_minimum": (
            None if not np.isfinite(metric_minimum) else metric_minimum
        ),
        "metric_target_maximum": metric_maximum,
        "gradation_iterations": gradation_iterations,
        "insertion_budget": options.max_insertions,
        "topology_budget": options.max_topology_operations,
        "cancelled": False,
        "shared_nodes": [
            {
                "local_node_id": int(local_node_id),
                "node_id": int(values[0]),
                "edge_id": int(values[1]),
                "station": (int(values[2].numerator), int(values[2].denominator)),
            }
            for local_node_id, values in sorted(topology.shared_node_ids.items())
        ],
    }
    native_diagnostics["native_v2"] = report
    return PlanarTriangulation(
        points=points, triangles=triangles, segments=topology.constraint_edges,
        boundary_segments=triangulation.boundary_segments,
        mandatory_segments=triangulation.mandatory_segments,
        outer_loop=triangulation.outer_loop, hole_loops=triangulation.hole_loops,
        backend=triangulation.backend, requested_backend=triangulation.requested_backend,
        selected_backend=triangulation.selected_backend, actual_backend=triangulation.actual_backend,
        fallback_reason=triangulation.fallback_reason, native_diagnostics=native_diagnostics,
    ), report
