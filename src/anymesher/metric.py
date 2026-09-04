"""Serializable spatial metric contracts for native-v2 surface meshing.

The public objects in this module contain data, not executable policy.  Runtime
callbacks are represented separately and deliberately refuse serialization so
automation and saved projects cannot accidentally acquire code execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import sqrt
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MeshError

__all__ = [
    "ExperimentalMetricProvider",
    "FeatureDistanceMetricControl",
    "ImportedMetricSamples",
    "IsotropicMetricControl",
    "MetricFieldSpec",
    "SpatialMetricField",
    "limit_metric_gradation",
    "metric_length",
    "pullback_metric",
]

_NATIVE_BATCH_THRESHOLD = 4096


def _positive(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MeshError(f"{label} must be a positive finite real")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise MeshError(f"{label} must be a positive finite real")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise MeshError(f"{label} must be a non-negative integer")
    return int(value)


def _points_tuple(values: Sequence[Sequence[float]], label: str) -> tuple[tuple[float, ...], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] not in (2, 3) or not np.all(np.isfinite(array)):
        raise MeshError(f"{label} must contain finite 2D or 3D point rows")
    return tuple(tuple(float(component) for component in row) for row in array)


def _tensor_tuple(values: Sequence[Sequence[Sequence[float]]], label: str) -> tuple[tuple[tuple[float, ...], ...], ...]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] not in ((2, 2), (3, 3)) or not np.all(np.isfinite(array)):
        raise MeshError(f"{label} must contain finite 2-by-2 or 3-by-3 tensors")
    result = []
    for number, tensor in enumerate(array):
        _validate_spd(tensor, f"{label}[{number}]")
        result.append(tuple(tuple(float(value) for value in row) for row in tensor))
    return tuple(result)


def _validate_spd(tensor: np.ndarray, label: str, maximum_anisotropy: float | None = None) -> np.ndarray:
    value = np.asarray(tensor, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] not in (2, 3):
        raise MeshError(f"{label} must be a 2D or 3D square tensor")
    scale = max(float(np.max(np.abs(value))), 1.0)
    if not np.all(np.isfinite(value)) or not np.allclose(
        value, value.T, rtol=1.0e-12, atol=1.0e-14 * scale
    ):
        raise MeshError(f"{label} must be symmetric positive definite")
    eigenvalues = np.linalg.eigvalsh(value)
    if float(eigenvalues[0]) <= 0.0:
        raise MeshError(f"{label} must be symmetric positive definite")
    if maximum_anisotropy is not None:
        anisotropy = sqrt(float(eigenvalues[-1] / eigenvalues[0]))
        if anisotropy > maximum_anisotropy * (1.0 + 1.0e-12):
            raise MeshError(
                f"{label} anisotropy {anisotropy:.6g} exceeds {maximum_anisotropy:.6g}"
            )
    return value


def _canonical_control_key(value: Any) -> str:
    return json.dumps(value.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _loewner_upper_bound(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return a deterministic SPD matrix that dominates both operands."""

    delta = 0.5 * ((second - first) + (second - first).T)
    eigenvalues, eigenvectors = np.linalg.eigh(delta)
    positive = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return 0.5 * ((first + positive) + (first + positive).T)


def _compose_metric_tensors(
    first: Any,
    second: Any,
    *,
    cancellation_check: Callable[[str], None] | None = None,
    cancellation_interval: int = 4096,
) -> np.ndarray:
    """Compose two equally shaped metric batches deterministically."""

    left = np.ascontiguousarray(first, dtype=np.float64)
    right = np.ascontiguousarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] not in (
        (2, 2),
        (3, 3),
    ):
        raise MeshError("metric batches must have the same 2D or 3D tensor shape")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise MeshError("metric batches must be finite")
    result = np.empty_like(left)
    for row in range(len(left)):
        if cancellation_check is not None and row % cancellation_interval == 0:
            cancellation_check("native-v2 metric composition")
        result[row] = _loewner_upper_bound(
            _validate_spd(left[row], "metric batch operand"),
            _validate_spd(right[row], "metric batch operand"),
        )
    return result


def _native_v2_available() -> bool:
    from . import native_cpp

    if not native_cpp.NATIVE_CPP_AVAILABLE:
        return False
    if not native_cpp.COMPILED_NATIVE_V2_AVAILABLE:
        raise MeshError("present ANYmesher native extension has an incomplete native-v2 ABI")
    return True


def _metric_lengths(
    points: Any,
    edges: Any,
    tensors: Any,
    *,
    cancellation_check: Callable[[str], None] | None = None,
    cancellation_interval: int = 4096,
) -> np.ndarray:
    """Return deterministic edge metric lengths, using native work when material."""

    coordinates = np.ascontiguousarray(points, dtype=np.float64)
    connections = np.ascontiguousarray(edges, dtype=np.int64)
    metrics = np.ascontiguousarray(tensors, dtype=np.float64)
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] not in (2, 3)
        or not np.all(np.isfinite(coordinates))
    ):
        raise MeshError("metric-length points must be finite 2D or 3D rows")
    if connections.ndim != 2 or connections.shape[1] != 2:
        raise MeshError("metric-length edges must have shape (n, 2)")
    dimension = coordinates.shape[1]
    if metrics.shape != (len(coordinates), dimension, dimension):
        raise MeshError("metric-length tensors do not match point coordinates")
    if isinstance(cancellation_interval, bool) or cancellation_interval < 1:
        raise MeshError("cancellation_interval must be a positive integer")
    if np.any(connections < 0) or np.any(connections >= len(coordinates)):
        raise MeshError("metric-length edge index is out of range")

    if (
        dimension == 2
        and len(connections) >= _NATIVE_BATCH_THRESHOLD
        and cancellation_check is None
        and _native_v2_available()
    ):
        from .native_cpp import native_metric_lengths

        compressed = np.ascontiguousarray(
            np.column_stack((metrics[:, 0, 0], metrics[:, 0, 1], metrics[:, 1, 1])),
            dtype=np.float64,
        )
        native = native_metric_lengths(coordinates, connections, compressed)
        if native is None:
            raise MeshError("complete native-v2 ABI did not execute metric lengths")
        result = np.ascontiguousarray(native, dtype=np.float64)
        if result.shape != (len(connections),) or np.any(result < 0.0) or not np.all(
            np.isfinite(result)
        ):
            raise MeshError("native-v2 metric lengths returned an invalid result")
        return result

    result = np.empty(len(connections), dtype=np.float64)
    for row, (first, second) in enumerate(connections):
        if cancellation_check is not None and row % cancellation_interval == 0:
            cancellation_check("native-v2 metric-length scan")
        tensor = 0.5 * (metrics[int(first)] + metrics[int(second)])
        delta = coordinates[int(second)] - coordinates[int(first)]
        result[row] = sqrt(max(float(delta @ tensor @ delta), 0.0))
    return result


def _limit_anisotropy(tensor: np.ndarray, maximum_anisotropy: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    floor = float(eigenvalues[-1]) / (maximum_anisotropy * maximum_anisotropy)
    bounded = np.maximum(eigenvalues, floor)
    result = eigenvectors @ np.diag(bounded) @ eigenvectors.T
    return 0.5 * (result + result.T)


@dataclass(frozen=True)
class IsotropicMetricControl:
    """Uniform target length in physical SI units."""

    target_size: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_size", _positive(self.target_size, "target_size"))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "isotropic", "target_size": self.target_size}


@dataclass(frozen=True)
class FeatureDistanceMetricControl:
    """Isotropic refinement that grows away from immutable feature samples."""

    feature_points: tuple[tuple[float, ...], ...]
    target_size: float
    influence_distance: float = 0.0
    growth: float = 1.5
    name: str = "feature"

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_points", _points_tuple(self.feature_points, "feature_points"))
        object.__setattr__(self, "target_size", _positive(self.target_size, "target_size"))
        distance = float(self.influence_distance)
        if not np.isfinite(distance) or distance < 0.0:
            raise MeshError("influence_distance must be finite and non-negative")
        object.__setattr__(self, "influence_distance", distance)
        growth = _positive(self.growth, "growth")
        if growth <= 1.0:
            raise MeshError("growth must exceed 1.0")
        object.__setattr__(self, "growth", growth)
        object.__setattr__(self, "name", str(self.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "feature_distance",
            "name": self.name,
            "feature_points": [list(row) for row in self.feature_points],
            "target_size": self.target_size,
            "influence_distance": self.influence_distance,
            "growth": self.growth,
        }


@dataclass(frozen=True)
class ImportedMetricSamples:
    """Revision-bound SPD samples supplied by an external deterministic source."""

    model_uuid: str
    geometry_revision: int
    points: tuple[tuple[float, ...], ...]
    tensors: tuple[tuple[tuple[float, ...], ...], ...]
    name: str = "imported"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_uuid", str(self.model_uuid))
        if not self.model_uuid:
            raise MeshError("model_uuid must not be empty")
        object.__setattr__(
            self, "geometry_revision", _nonnegative_int(self.geometry_revision, "geometry_revision")
        )
        points = _points_tuple(self.points, "imported metric points")
        tensors = _tensor_tuple(self.tensors, "imported metric tensors")
        if len(points) != len(tensors) or not points:
            raise MeshError("imported metric points and tensors need the same non-zero count")
        if len(points[0]) != len(tensors[0]):
            raise MeshError("imported metric point and tensor dimensions differ")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "tensors", tensors)
        object.__setattr__(self, "name", str(self.name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "imported_samples",
            "name": self.name,
            "model_uuid": self.model_uuid,
            "geometry_revision": self.geometry_revision,
            "points": [list(row) for row in self.points],
            "tensors": [[list(row) for row in tensor] for tensor in self.tensors],
        }


@dataclass(frozen=True)
class MetricFieldSpec:
    """Canonical serializable composition of deterministic metric controls."""

    global_control: IsotropicMetricControl
    feature_controls: tuple[FeatureDistanceMetricControl, ...] = ()
    imported_samples: tuple[ImportedMetricSamples, ...] = ()
    maximum_anisotropy: float = 25.0
    maximum_gradation: float = 1.5

    def __post_init__(self) -> None:
        if not isinstance(self.global_control, IsotropicMetricControl):
            raise MeshError("global_control must be IsotropicMetricControl")
        features = tuple(self.feature_controls)
        samples = tuple(self.imported_samples)
        if any(not isinstance(value, FeatureDistanceMetricControl) for value in features):
            raise MeshError("feature_controls contains an unsupported value")
        if any(not isinstance(value, ImportedMetricSamples) for value in samples):
            raise MeshError("imported_samples contains an unsupported value")
        dimensions = {
            len(value.feature_points[0]) for value in features
        } | {
            len(value.points[0]) for value in samples
        }
        if len(dimensions) > 1:
            raise MeshError("metric controls must use one common spatial dimension")
        object.__setattr__(self, "feature_controls", tuple(sorted(features, key=_canonical_control_key)))
        object.__setattr__(self, "imported_samples", tuple(sorted(samples, key=_canonical_control_key)))
        maximum_anisotropy = _positive(self.maximum_anisotropy, "maximum_anisotropy")
        if maximum_anisotropy < 1.0:
            raise MeshError("maximum_anisotropy must be at least 1")
        maximum_gradation = _positive(self.maximum_gradation, "maximum_gradation")
        if maximum_gradation <= 1.0:
            raise MeshError("maximum_gradation must exceed 1")
        object.__setattr__(self, "maximum_anisotropy", maximum_anisotropy)
        object.__setattr__(self, "maximum_gradation", maximum_gradation)

    @property
    def spatial_dimension(self) -> int | None:
        if self.feature_controls:
            return len(self.feature_controls[0].feature_points[0])
        if self.imported_samples:
            return len(self.imported_samples[0].points[0])
        return None

    @classmethod
    def uniform(cls, target_size: float) -> "MetricFieldSpec":
        return cls(IsotropicMetricControl(target_size))

    @classmethod
    def from_size_field(cls, size_field: Any) -> "MetricFieldSpec":
        controls = tuple(
            FeatureDistanceMetricControl(
                feature_points=tuple(tuple(float(x) for x in row) for row in sources),
                target_size=zone.size,
                influence_distance=zone.radius,
                growth=zone.growth,
                name=zone.name,
            )
            for zone, sources in zip(size_field.zones, size_field._sources)
        )
        return cls(IsotropicMetricControl(size_field.target_size), controls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "anymesher.metric-field/1",
            "global_control": self.global_control.to_dict(),
            "feature_controls": [value.to_dict() for value in self.feature_controls],
            "imported_samples": [value.to_dict() for value in self.imported_samples],
            "maximum_anisotropy": self.maximum_anisotropy,
            "maximum_gradation": self.maximum_gradation,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MetricFieldSpec":
        expected = {
            "schema", "global_control", "feature_controls", "imported_samples",
            "maximum_anisotropy", "maximum_gradation",
        }
        unknown = set(raw).difference(expected)
        if unknown or raw.get("schema") != "anymesher.metric-field/1":
            raise MeshError(f"invalid metric-field schema or fields: {sorted(unknown)}")
        global_raw = raw.get("global_control")
        if not isinstance(global_raw, Mapping) or set(global_raw) != {"type", "target_size"} or global_raw.get("type") != "isotropic":
            raise MeshError("global_control must be the canonical isotropic object")
        features = []
        for value in raw.get("feature_controls", ()):
            if not isinstance(value, Mapping) or value.get("type") != "feature_distance":
                raise MeshError("invalid feature-distance metric control")
            allowed = {"type", "name", "feature_points", "target_size", "influence_distance", "growth"}
            if set(value) != allowed:
                raise MeshError("feature-distance metric control has unknown fields")
            features.append(FeatureDistanceMetricControl(
                feature_points=value["feature_points"], target_size=value["target_size"],
                influence_distance=value["influence_distance"], growth=value["growth"], name=value["name"],
            ))
        samples = []
        for value in raw.get("imported_samples", ()):
            if not isinstance(value, Mapping) or value.get("type") != "imported_samples":
                raise MeshError("invalid imported metric samples")
            allowed = {"type", "name", "model_uuid", "geometry_revision", "points", "tensors"}
            if set(value) != allowed:
                raise MeshError("imported metric samples have unknown fields")
            samples.append(ImportedMetricSamples(
                model_uuid=value["model_uuid"], geometry_revision=value["geometry_revision"],
                points=value["points"], tensors=value["tensors"], name=value["name"],
            ))
        return cls(
            IsotropicMetricControl(global_raw["target_size"]), tuple(features), tuple(samples),
            raw.get("maximum_anisotropy", 25.0), raw.get("maximum_gradation", 1.5),
        )


@dataclass(frozen=True)
class ExperimentalMetricProvider:
    """Explicitly non-serializable in-memory metric evaluator."""

    evaluator: Callable[[np.ndarray], np.ndarray]
    label: str = "experimental"
    serializable: bool = field(default=False, init=False)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(self.evaluator(points), dtype=np.float64)
        if result.shape != (len(points), points.shape[1], points.shape[1]):
            raise MeshError("experimental metric provider returned the wrong tensor shape")
        for number, tensor in enumerate(result):
            _validate_spd(tensor, f"experimental metric tensor {number}")
        return result

    def to_dict(self) -> dict[str, Any]:
        raise MeshError("runtime metric providers are not serializable or automation-safe")


class SpatialMetricField:
    """Bounded deterministic evaluator for one :class:`MetricFieldSpec`."""

    def __init__(
        self,
        spec: MetricFieldSpec,
        *,
        model_uuid: str | None = None,
        geometry_revision: int | None = None,
    ) -> None:
        self.spec = spec
        self.model_uuid = None if model_uuid is None else str(model_uuid)
        self.geometry_revision = geometry_revision

    def evaluate(
        self,
        points: Any,
        *,
        cancellation_check: Callable[[str], None] | None = None,
        cancellation_interval: int = 4096,
    ) -> np.ndarray:
        values = np.ascontiguousarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] not in (2, 3) or not np.all(np.isfinite(values)):
            raise MeshError("metric query points must be finite 2D or 3D rows")
        if cancellation_interval < 1:
            raise MeshError("cancellation_interval must be a positive integer")

        def checkpoint(stage: str, index: int) -> None:
            if cancellation_check is not None and index % cancellation_interval == 0:
                cancellation_check(stage)

        dimension = values.shape[1]
        base = 1.0 / (self.spec.global_control.target_size ** 2)
        result = np.repeat(np.eye(dimension, dtype=np.float64)[None, :, :] * base, len(values), axis=0)
        sizes = np.full(len(values), self.spec.global_control.target_size, dtype=np.float64)
        for control in self.spec.feature_controls:
            sources = np.asarray(control.feature_points, dtype=np.float64)
            if sources.shape[1] != dimension:
                raise MeshError("feature metric dimension does not match query coordinates")
            distance = np.empty(len(values), dtype=np.float64)
            for start in range(0, len(values), cancellation_interval):
                checkpoint("native-v2 feature metric evaluation", start)
                stop = min(start + cancellation_interval, len(values))
                chunk = values[start:stop]
                distance[start:stop] = np.sqrt(
                    np.min(
                        np.sum(
                            (chunk[:, None, :] - sources[None, :, :]) ** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                )
            beyond = np.maximum(distance - control.influence_distance, 0.0)
            grown = control.target_size * control.growth ** (beyond / control.target_size)
            sizes = np.minimum(sizes, np.minimum(grown, self.spec.global_control.target_size))
        for row, size in enumerate(sizes):
            checkpoint("native-v2 isotropic metric assembly", row)
            result[row] = np.eye(dimension, dtype=np.float64) / (size * size)
        for imported in self.spec.imported_samples:
            if self.model_uuid is None or self.geometry_revision is None:
                raise MeshError("revision-bound imported metrics need model UUID and geometry revision")
            if imported.model_uuid != self.model_uuid or imported.geometry_revision != self.geometry_revision:
                raise MeshError("imported metric samples are stale or belong to another geometry model")
            samples = np.asarray(imported.points, dtype=np.float64)
            tensors = np.asarray(imported.tensors, dtype=np.float64)
            if samples.shape[1] != dimension:
                raise MeshError("imported metric dimension does not match query coordinates")
            nearest = np.empty(len(values), dtype=np.int64)
            for start in range(0, len(values), cancellation_interval):
                checkpoint("native-v2 imported metric lookup", start)
                stop = min(start + cancellation_interval, len(values))
                chunk = values[start:stop]
                nearest[start:stop] = np.argmin(
                    np.sum(
                        (chunk[:, None, :] - samples[None, :, :]) ** 2,
                        axis=2,
                    ),
                    axis=1,
                )
            for row, sample_row in enumerate(nearest):
                checkpoint("native-v2 imported metric composition", row)
                candidate = _validate_spd(
                    tensors[sample_row], "imported metric tensor", self.spec.maximum_anisotropy
                )
                result[row] = _loewner_upper_bound(result[row], candidate)
        for row in range(len(result)):
            checkpoint("native-v2 metric anisotropy limiting", row)
            result[row] = _limit_anisotropy(result[row], self.spec.maximum_anisotropy)
        if cancellation_check is not None:
            cancellation_check("native-v2 metric evaluation complete")
        return result


def metric_length(first: Sequence[float], second: Sequence[float], metric: Any) -> float:
    delta = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    tensor = _validate_spd(np.asarray(metric, dtype=np.float64), "metric")
    if tensor.shape != (len(delta), len(delta)):
        raise MeshError("metric dimension does not match edge coordinates")
    return sqrt(max(float(delta @ tensor @ delta), 0.0))


def pullback_metric(physical_metric: Any, jacobian: Any, *, max_condition: float = 1.0e12) -> np.ndarray:
    metric = _validate_spd(np.asarray(physical_metric, dtype=np.float64), "physical metric")
    derivative = np.asarray(jacobian, dtype=np.float64)
    if derivative.ndim != 2 or derivative.shape[0] != metric.shape[0] or derivative.shape[1] != 2:
        raise MeshError("surface derivative must map a 2D chart into the physical metric")
    gram = derivative.T @ derivative
    if not np.all(np.isfinite(gram)) or np.linalg.cond(gram) > max_condition:
        raise MeshError("surface chart is singular or ill-conditioned")
    return _validate_spd(derivative.T @ metric @ derivative, "chart metric")


def limit_metric_gradation(
    points: Any,
    edges: Any,
    target_lengths: Any,
    maximum_gradation: float,
    *,
    max_iterations: int = 64,
    cancellation_check: Callable[[str], None] | None = None,
    cancellation_interval: int = 4096,
) -> tuple[np.ndarray, int]:
    """Limit adjacent scalar target growth without changing point ordering."""

    coordinates = np.ascontiguousarray(points, dtype=np.float64)
    connections = np.ascontiguousarray(edges, dtype=np.int64)
    values = np.ascontiguousarray(target_lengths, dtype=np.float64).copy()
    growth = _positive(maximum_gradation, "maximum_gradation")
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] not in (2, 3)
        or not np.all(np.isfinite(coordinates))
    ):
        raise MeshError("gradation points must be finite 2D or 3D rows")
    if growth <= 1.0 or connections.ndim != 2 or connections.shape[1] != 2:
        raise MeshError("gradation needs an edge matrix and maximum_gradation above 1")
    if values.shape != (len(coordinates),) or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise MeshError("target_lengths must be one positive finite value per point")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise MeshError("max_iterations must be a positive integer")
    if isinstance(cancellation_interval, bool) or not isinstance(cancellation_interval, Integral) or cancellation_interval < 1:
        raise MeshError("cancellation_interval must be a positive integer")
    if np.any(connections < 0) or np.any(connections >= len(coordinates)):
        raise MeshError("gradation edge index is out of range")
    if (
        coordinates.shape[1] == 2
        and len(connections) >= _NATIVE_BATCH_THRESHOLD
        and cancellation_check is None
        and _native_v2_available()
    ):
        from .native_cpp import native_gradation_limit

        native = native_gradation_limit(
            coordinates, connections, values, growth, int(max_iterations)
        )
        if native is None:
            raise MeshError("complete native-v2 ABI did not execute metric gradation")
        limited, iterations = native
        limited = np.ascontiguousarray(limited, dtype=np.float64)
        if (
            limited.shape != values.shape
            or np.any(limited <= 0.0)
            or not np.all(np.isfinite(limited))
            or not 1 <= int(iterations) <= int(max_iterations)
        ):
            raise MeshError("native-v2 metric gradation returned an invalid result")
        return limited, int(iterations)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        if cancellation_check is not None:
            cancellation_check("native-v2 metric gradation sweep")
        changed = False
        for edge_number, (first, second) in enumerate(connections):
            if cancellation_check is not None and edge_number % cancellation_interval == 0:
                cancellation_check("native-v2 metric gradation scan")
            a, b = int(first), int(second)
            if a < 0 or b < 0 or a >= len(values) or b >= len(values):
                raise MeshError("gradation edge index is out of range")
            distance = float(np.linalg.norm(coordinates[b] - coordinates[a]))
            allowance = (growth - 1.0) * distance
            if values[b] > values[a] + allowance:
                values[b] = values[a] + allowance
                changed = True
            if values[a] > values[b] + allowance:
                values[a] = values[b] + allowance
                changed = True
        if not changed:
            return values, iterations
    for edge_number, (first, second) in enumerate(connections):
        if cancellation_check is not None and edge_number % cancellation_interval == 0:
            cancellation_check("native-v2 metric gradation validation")
        a, b = int(first), int(second)
        distance = float(np.linalg.norm(coordinates[b] - coordinates[a]))
        allowance = (growth - 1.0) * distance
        tolerance = 64.0 * np.finfo(float).eps * max(
            values[a], values[b], allowance, 1.0
        )
        if (
            values[b] > values[a] + allowance + tolerance
            or values[a] > values[b] + allowance + tolerance
        ):
            raise MeshError(
                "metric gradation did not converge within the bounded sweep budget"
            )
    return values, iterations
