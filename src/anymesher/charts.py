"""Deterministic UV adapters and metric fields for native surface meshing.

The adapter deliberately delegates geometry evaluation and projection to
ANYgeometry.  It feature-detects batch entry points as they become available,
while retaining the scalar public model API as a compatibility path.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import MeshError

__all__ = [
    "ChartProjection",
    "FaceChart",
    "MetricField",
    "metric_edge_lengths",
    "metric_tensors",
]


_MISSING = object()


def _readonly(value: Any, dtype: Any = np.float64) -> np.ndarray:
    made = np.ascontiguousarray(value, dtype=dtype)
    made.setflags(write=False)
    return made


def _rows(value: Any, width: int, name: str) -> np.ndarray:
    made = np.asarray(value, dtype=np.float64)
    if made.ndim == 1 and made.shape == (width,):
        made = made.reshape(1, width)
    if made.ndim != 2 or made.shape[1] != width:
        raise MeshError(f"{name} must have shape (n, {width})")
    if not np.all(np.isfinite(made)):
        raise MeshError(f"{name} must be finite")
    return made


def _uv_rows(value: Any) -> np.ndarray:
    made = _rows(value, 2, "UV coordinates")
    if np.any(made < 0.0) or np.any(made > 1.0):
        raise MeshError("UV coordinates must be in [0, 1]")
    return made


def _can_bind(function: Callable[..., Any], arguments: tuple[Any, ...]) -> bool:
    try:
        made = signature(function)
    except (TypeError, ValueError):
        return True
    try:
        made.bind(*arguments)
    except TypeError:
        return False
    return True


def _extract(value: Any, names: Sequence[str]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return _MISSING


@dataclass(frozen=True)
class ChartProjection:
    """A row-preserving bounded face projection result."""

    points: np.ndarray
    uv: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        points = _rows(self.points, 3, "projected points")
        uv = _rows(self.uv, 2, "projected UV coordinates")
        distances = np.asarray(self.distances, dtype=np.float64)
        if distances.ndim == 0:
            distances = distances.reshape(1)
        if distances.shape != (len(points),):
            raise MeshError("projection distances must have shape (n,)")
        if len(uv) != len(points) or not np.all(np.isfinite(distances)):
            raise MeshError("projection rows must have matching finite values")
        if np.any(uv < 0.0) or np.any(uv > 1.0) or np.any(distances < 0.0):
            raise MeshError("projected UV and distance values must be bounded")
        object.__setattr__(self, "points", _readonly(points))
        object.__setattr__(self, "uv", _readonly(uv))
        object.__setattr__(self, "distances", _readonly(distances))


class FaceChart:
    """A normalized ``[0, 1]^2`` chart over one ANYgeometry face.

    All methods accept one row or a batch and always return batched arrays.
    The row order is never changed.  A revision-bound meshing view is checked
    before every call and its guarded source is used only for APIs the view
    does not expose itself.
    """

    def __init__(
        self,
        geometry: object,
        face_id: int,
        *,
        difference_step: float | None = None,
    ) -> None:
        if isinstance(face_id, bool):
            raise MeshError("face ID must be a positive integer")
        self.geometry = geometry
        self.face_id = int(face_id)
        if self.face_id <= 0:
            raise MeshError("face ID must be a positive integer")
        step = (
            float(np.cbrt(np.finfo(np.float64).eps))
            if difference_step is None
            else float(difference_step)
        )
        if not np.isfinite(step) or not 0.0 < step <= 1.0:
            raise MeshError("difference_step must be finite and in (0, 1]")
        self.difference_step = step

    def _targets(self) -> tuple[tuple[object, bool], ...]:
        guard = getattr(self.geometry, "assert_current", None)
        owner = self.geometry
        if callable(guard):
            guard()
            owner = getattr(self.geometry, "source")
        targets: list[tuple[object, bool]] = []
        for value in (self.geometry, owner):
            if all(value is not existing for existing, _ in targets):
                targets.append((value, False))
        modules: list[object] = []
        try:
            modules.append(import_module("anygeometry"))
            modules.append(import_module("anygeometry.meshing"))
        except ModuleNotFoundError:
            pass
        for module in modules:
            if all(module is not existing for existing, _ in targets):
                targets.append((module, True))
        return tuple(targets)

    @staticmethod
    def _uses_face_ids(
        function: Callable[..., Any], name: str, module_function: bool
    ) -> bool:
        try:
            parameters = tuple(signature(function).parameters.values())
        except (TypeError, ValueError):
            parameters = ()
        offset = 1 if module_function else 0
        if len(parameters) > offset:
            parameter = parameters[offset].name.lower()
            if parameter.endswith("_ids") or parameter in ("faces", "face_handles"):
                return True
            if parameter.endswith("_id") or parameter in ("face", "face_handle"):
                return False
        return name in {
            "evaluate_faces",
            "face_derivatives",
            "face_normals",
            "project_faces",
            "project_to_faces",
        }

    def _batch_call(
        self, names: Sequence[str], values: np.ndarray
    ) -> Any:
        face_ids = np.full(len(values), self.face_id, dtype=np.int64)
        owner = (
            getattr(self.geometry, "source")
            if callable(getattr(self.geometry, "assert_current", None))
            else self.geometry
        )
        for target, module_function in self._targets():
            for name in names:
                function = getattr(target, name, None)
                if not callable(function):
                    continue
                face = (
                    face_ids
                    if self._uses_face_ids(function, name, module_function)
                    else self.face_id
                )
                arguments = (
                    (owner, face, values)
                    if module_function
                    else (face, values)
                )
                if _can_bind(function, arguments):
                    return function(*arguments)
        return _MISSING

    def _scalar_call(self, names: Sequence[str], arguments: tuple[Any, ...]) -> Any:
        owner = (
            getattr(self.geometry, "source")
            if callable(getattr(self.geometry, "assert_current", None))
            else self.geometry
        )
        for target, module_function in self._targets():
            for name in names:
                function = getattr(target, name, None)
                if not callable(function):
                    continue
                made_arguments = (
                    (owner, *arguments) if module_function else arguments
                )
                if _can_bind(function, made_arguments):
                    return function(*made_arguments)
        return _MISSING

    @staticmethod
    def _point_batch(value: Any, count: int) -> np.ndarray:
        selected = _extract(value, ("points", "positions", "xyz", "values"))
        if selected is _MISSING and isinstance(value, tuple) and value:
            selected = value[0]
        if selected is _MISSING:
            selected = value
        made = _rows(selected, 3, "evaluated face points")
        if len(made) != count:
            raise MeshError("face evaluation changed the batch row count")
        return made

    def evaluate(self, uv: Any) -> np.ndarray:
        """Evaluate world points for normalized UV rows."""

        parameters = _uv_rows(uv)
        batched = self._batch_call(
            ("evaluate_faces", "evaluate_face_batch", "face_points_batch", "face_points"),
            parameters,
        )
        if batched is not _MISSING:
            return _readonly(self._point_batch(batched, len(parameters)))
        points = []
        for u, v in parameters:
            value = self._scalar_call(
                ("face_point", "evaluate_face"),
                (self.face_id, float(u), float(v)),
            )
            if value is _MISSING:
                raise MeshError("geometry owner exposes no face evaluation API")
            points.append(value)
        return _readonly(_rows(points, 3, "evaluated face points"))

    @staticmethod
    def _derivative_batch(value: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
        du = _extract(value, ("du", "u_derivatives", "derivative_u"))
        dv = _extract(value, ("dv", "v_derivatives", "derivative_v"))
        if (du is _MISSING or dv is _MISSING) and isinstance(value, tuple) and len(value) >= 2:
            du, dv = value[0], value[1]
        if du is _MISSING or dv is _MISSING:
            combined = _extract(value, ("derivatives", "jacobians", "jacobian"))
            if combined is _MISSING:
                combined = value
            array = np.asarray(combined, dtype=np.float64)
            if array.shape == (count, 2, 3):
                du, dv = array[:, 0], array[:, 1]
            elif array.shape == (count, 3, 2):
                du, dv = array[:, :, 0], array[:, :, 1]
            else:
                raise MeshError("face derivatives must have shape (n, 2, 3)")
        made_du = _rows(du, 3, "u derivatives")
        made_dv = _rows(dv, 3, "v derivatives")
        if len(made_du) != count or len(made_dv) != count:
            raise MeshError("face derivatives changed the batch row count")
        return made_du, made_dv

    def derivatives(self, uv: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(dX/du, dX/dv)`` for each UV row."""

        parameters = _uv_rows(uv)
        batched = self._batch_call(
            ("face_derivatives", "evaluate_face_derivatives", "face_derivatives_batch"),
            parameters,
        )
        if batched is not _MISSING:
            du, dv = self._derivative_batch(batched, len(parameters))
            return _readonly(du), _readonly(dv)

        scalar: list[tuple[Any, Any]] = []
        for u, v in parameters:
            value = self._scalar_call(
                ("face_derivative", "face_derivatives"),
                (self.face_id, float(u), float(v)),
            )
            if value is _MISSING:
                scalar = []
                break
            scalar.append((value[0], value[1]))
        if scalar:
            du, dv = self._derivative_batch(
                (np.asarray([item[0] for item in scalar]), np.asarray([item[1] for item in scalar])),
                len(parameters),
            )
            return _readonly(du), _readonly(dv)

        step = self.difference_step
        lower_u = parameters.copy()
        upper_u = parameters.copy()
        lower_v = parameters.copy()
        upper_v = parameters.copy()
        lower_u[:, 0] = np.maximum(0.0, parameters[:, 0] - step)
        upper_u[:, 0] = np.minimum(1.0, parameters[:, 0] + step)
        lower_v[:, 1] = np.maximum(0.0, parameters[:, 1] - step)
        upper_v[:, 1] = np.minimum(1.0, parameters[:, 1] + step)
        du = (self.evaluate(upper_u) - self.evaluate(lower_u)) / (
            upper_u[:, 0] - lower_u[:, 0]
        )[:, None]
        dv = (self.evaluate(upper_v) - self.evaluate(lower_v)) / (
            upper_v[:, 1] - lower_v[:, 1]
        )[:, None]
        return _readonly(du), _readonly(dv)

    def normals(self, uv: Any) -> np.ndarray:
        """Return deterministic unit normals for each UV row."""

        parameters = _uv_rows(uv)
        batched = self._batch_call(
            ("face_normals", "evaluate_face_normals", "face_normals_batch"),
            parameters,
        )
        if batched is not _MISSING:
            selected = _extract(batched, ("normals", "values"))
            normals = self._point_batch(
                batched if selected is _MISSING else selected, len(parameters)
            )
        else:
            values: list[Any] = []
            for u, v in parameters:
                value = self._scalar_call(
                    ("face_normal",),
                    (self.face_id, float(u), float(v)),
                )
                if value is _MISSING:
                    values = []
                    break
                values.append(value)
            if values:
                normals = _rows(values, 3, "face normals")
            else:
                du, dv = self.derivatives(parameters)
                normals = np.cross(du, dv)
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0) or not np.all(np.isfinite(lengths)):
            raise MeshError("face chart has a degenerate normal")
        return _readonly(normals / lengths[:, None])

    @staticmethod
    def _projection_batch(value: Any, count: int) -> ChartProjection:
        points = _extract(value, ("points", "projected", "projected_points", "positions"))
        uv = _extract(value, ("uv", "parameters", "face_parameters"))
        distances = _extract(value, ("distances", "distance", "residuals"))
        if (
            (points is _MISSING or uv is _MISSING or distances is _MISSING)
            and isinstance(value, tuple)
            and len(value) >= 3
        ):
            points, uv, distances = value[:3]
        if points is _MISSING or uv is _MISSING or distances is _MISSING:
            raise MeshError("face projection must return points, UV, and distances")
        made = ChartProjection(points, uv, distances)
        if len(made.points) != count:
            raise MeshError("face projection changed the batch row count")
        return made

    def project(self, points: Any) -> ChartProjection:
        """Project world points to this bounded, trim-aware face chart."""

        targets = _rows(points, 3, "projection points")
        batched = self._batch_call(
            ("project_faces", "project_to_faces", "project_face_batch", "project_faces_batch"),
            targets,
        )
        if batched is not _MISSING:
            return self._projection_batch(batched, len(targets))
        projected: list[Any] = []
        uv: list[Any] = []
        distances: list[float] = []
        for point in targets:
            value = self._scalar_call(
                ("project_to_face", "project_face"),
                (self.face_id, point),
            )
            if value is _MISSING:
                raise MeshError("geometry owner exposes no bounded face projection API")
            made_point, made_uv, made_distance = value
            projected.append(made_point)
            uv.append(made_uv)
            distances.append(float(made_distance))
        return ChartProjection(projected, uv, distances)

    def metric(
        self,
        uv: Any,
        size: Any,
        *,
        direction: Any | None = None,
        aspect_ratio: Any = 1.0,
    ) -> np.ndarray:
        """Build desired UV metric tensors from chart derivatives."""

        parameters = _uv_rows(uv)
        du, dv = self.derivatives(parameters)
        return metric_tensors(
            du,
            dv,
            size,
            direction=direction,
            aspect_ratio=aspect_ratio,
        )


def _row_scalars(value: Any, count: int, name: str, minimum: float) -> np.ndarray:
    made = np.asarray(value, dtype=np.float64)
    if made.ndim == 0:
        made = np.full(count, float(made), dtype=np.float64)
    if made.shape != (count,) or not np.all(np.isfinite(made)):
        raise MeshError(f"{name} must be a finite scalar or have shape (n,)")
    if np.any(made < minimum) or (minimum > 0.0 and np.any(made == 0.0)):
        raise MeshError(f"{name} must be at least {minimum:g}")
    return made


def metric_tensors(
    du: Any,
    dv: Any,
    size: Any,
    *,
    direction: Any | None = None,
    aspect_ratio: Any = 1.0,
) -> np.ndarray:
    """Return desired SPD metric tensors in UV coordinates.

    ``size`` is the target transverse world-space length.  With a preferred
    tangent ``direction``, ``aspect_ratio`` permits elements that many times
    longer along that direction.  Directions may be world 3-vectors or UV
    2-vectors and may be supplied once or per row.
    """

    made_du = _rows(du, 3, "u derivatives")
    made_dv = _rows(dv, 3, "v derivatives")
    if len(made_du) != len(made_dv):
        raise MeshError("u and v derivatives must have matching rows")
    count = len(made_du)
    sizes = _row_scalars(size, count, "metric size", np.finfo(float).tiny)
    ratios = _row_scalars(aspect_ratio, count, "metric aspect ratio", 1.0)
    jacobian = np.stack((made_du, made_dv), axis=2)
    normals = np.cross(made_du, made_dv)
    normal_lengths = np.linalg.norm(normals, axis=1)
    if np.any(normal_lengths <= 0.0):
        raise MeshError("surface derivatives are linearly dependent")

    if direction is None:
        metric = np.einsum("nki,nkj->nij", jacobian, jacobian)
        metric /= sizes[:, None, None] ** 2
        return _readonly(metric)

    directions = np.asarray(direction, dtype=np.float64)
    if directions.ndim == 1 and directions.shape[0] in (2, 3):
        directions = np.repeat(directions[None, :], count, axis=0)
    if directions.ndim != 2 or directions.shape not in ((count, 2), (count, 3)):
        raise MeshError("metric directions must have shape (2,), (3,), (n, 2), or (n, 3)")
    if not np.all(np.isfinite(directions)):
        raise MeshError("metric directions must be finite")
    unit_normals = normals / normal_lengths[:, None]
    if directions.shape[1] == 2:
        tangent = (
            made_du * directions[:, :1]
            + made_dv * directions[:, 1:2]
        )
    else:
        tangent = directions - np.sum(directions * unit_normals, axis=1)[:, None] * unit_normals
    tangent_lengths = np.linalg.norm(tangent, axis=1)
    if np.any(tangent_lengths <= 0.0):
        raise MeshError("metric direction must have a non-zero surface tangent")
    parallel = tangent / tangent_lengths[:, None]
    transverse = np.cross(unit_normals, parallel)
    parallel_uv = np.stack(
        (np.sum(made_du * parallel, axis=1), np.sum(made_dv * parallel, axis=1)),
        axis=1,
    )
    transverse_uv = np.stack(
        (np.sum(made_du * transverse, axis=1), np.sum(made_dv * transverse, axis=1)),
        axis=1,
    )
    parallel_size = sizes * ratios
    metric = (
        np.einsum("ni,nj->nij", parallel_uv, parallel_uv)
        / parallel_size[:, None, None] ** 2
        + np.einsum("ni,nj->nij", transverse_uv, transverse_uv)
        / sizes[:, None, None] ** 2
    )
    return _readonly(metric)


def metric_edge_lengths(uv: Any, edges: Any, metrics: Any) -> np.ndarray:
    """Measure UV edges with the arithmetic mean endpoint metric."""

    points = _rows(uv, 2, "UV coordinates")
    connectivity = np.asarray(edges, dtype=np.int64)
    if connectivity.ndim == 1 and connectivity.size == 0:
        connectivity = np.empty((0, 2), dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.shape[1] != 2:
        raise MeshError("metric edges must have shape (m, 2)")
    if connectivity.size and (
        int(np.min(connectivity)) < 0 or int(np.max(connectivity)) >= len(points)
    ):
        raise MeshError("metric edge references an invalid UV row")
    tensors = np.asarray(metrics, dtype=np.float64)
    if tensors.shape == (2, 2):
        tensors = np.repeat(tensors[None, :, :], len(points), axis=0)
    if tensors.shape != (len(points), 2, 2) or not np.all(np.isfinite(tensors)):
        raise MeshError("metrics must have shape (2, 2) or (n, 2, 2)")
    differences = points[connectivity[:, 1]] - points[connectivity[:, 0]]
    edge_metrics = 0.5 * (
        tensors[connectivity[:, 0]] + tensors[connectivity[:, 1]]
    )
    squared = np.einsum("ni,nij,nj->n", differences, edge_metrics, differences)
    if np.any(squared < 0.0):
        raise MeshError("metric tensor is not positive semidefinite")
    return _readonly(np.sqrt(squared))


def _field_value(
    value: Any,
    method_name: str,
    uv: np.ndarray,
    xyz: np.ndarray,
) -> Any:
    function = getattr(value, method_name, None)
    if not callable(function) and callable(value):
        function = value
    if not callable(function):
        return value
    for arguments in ((uv, xyz), (xyz,)):
        if _can_bind(function, arguments):
            return function(*arguments)
    raise MeshError(f"{method_name} field must accept xyz or (uv, xyz)")


@dataclass(frozen=True)
class MetricField:
    """Composable size/direction field producing UV metric tensors."""

    size: Any
    direction: Any | None = None
    aspect_ratio: Any = 1.0

    def tensors(self, chart: FaceChart, uv: Any) -> np.ndarray:
        parameters = _uv_rows(uv)
        xyz = chart.evaluate(parameters)
        size = _field_value(self.size, "size_at", parameters, xyz)
        direction = (
            None
            if self.direction is None
            else _field_value(self.direction, "direction_at", parameters, xyz)
        )
        ratio = _field_value(
            self.aspect_ratio, "aspect_ratio_at", parameters, xyz
        )
        return chart.metric(
            parameters,
            size,
            direction=direction,
            aspect_ratio=ratio,
        )

    __call__ = tensors
