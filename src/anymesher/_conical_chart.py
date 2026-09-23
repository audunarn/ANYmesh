"""Revision-bound isometric chart for analytic conical frusta."""

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt

import numpy as np
from anygeometry.surfaces import Cone

from .charts import FaceChart
from .errors import MeshError


def _rows2(values):
    points = np.asarray(values, dtype=np.float64)
    if points.shape == (2,):
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise MeshError("cone chart coordinates must be finite (n, 2) rows")
    return points


@dataclass(frozen=True, slots=True)
class ConicalMetricChart:
    """Exact developable unroll of one owner Cone face."""

    face_chart: FaceChart
    model_id: object
    revision: int
    radius_start: float
    radius_end: float
    height: float
    sweep_angle: float
    slant_length: float
    phi_span: float

    @classmethod
    def from_geometry(cls, geometry, face_id):
        guard = getattr(geometry, "assert_current", None)
        if callable(guard):
            guard()
        owner = geometry.source if callable(guard) else geometry
        chart = FaceChart(geometry, face_id)
        face = owner.faces[chart.face_id]
        surface = face.surface
        if not isinstance(surface, Cone):
            raise MeshError("physical cone chart requires an owner Cone")
        if face.parameterization is not None and face.parameterization is not surface:
            raise MeshError("physical cone chart does not support a distinct face parameterization")
        r0 = float(surface.radius_start)
        r1 = float(surface.radius_end)
        height = float(surface.height)
        sweep = float(surface.sweep_angle)
        dr = r1 - r0
        slant = sqrt(height * height + dr * dr)
        scale = max(abs(r0), abs(r1), abs(height), 1.0)
        if not all(np.isfinite(v) for v in (r0, r1, height, sweep, slant)):
            raise MeshError("physical cone chart parameters must be finite")
        if min(r0, r1, abs(height), abs(sweep), slant) <= 0.0:
            raise MeshError("physical cone chart has a degenerate extent")
        if abs(dr) <= 1.0e-12 * scale:
            raise MeshError("near-cylindrical Cone is outside the conical quad-first scope")
        phi_span = sweep * abs(dr) / slant
        if abs(phi_span) <= 1.0e-14 or abs(phi_span) >= 2.0 * pi:
            raise MeshError("physical cone chart has an invalid unrolled angle")
        return cls(chart, owner.model_id, owner.revision, r0, r1, height, sweep, slant, phi_span)

    @property
    def dr(self) -> float:
        return self.radius_end - self.radius_start

    def _current(self):
        geometry = self.face_chart.geometry
        guard = getattr(geometry, "assert_current", None)
        if callable(guard):
            guard()
        owner = geometry.source if callable(guard) else geometry
        if owner.model_id != self.model_id or owner.revision != self.revision:
            raise MeshError("cone chart binding is stale or belongs to another model")

    def to_chart(self, uv):
        self._current()
        parameters = _rows2(uv)
        if np.any(parameters < 0.0) or np.any(parameters > 1.0):
            raise MeshError("owner cone parameters must be in [0, 1]")
        u = parameters[:, 0]
        v = parameters[:, 1]
        radius = self.radius_start + self.dr * v
        rho = radius * self.slant_length / abs(self.dr)
        phi = u * self.phi_span
        return np.column_stack((rho * np.cos(phi), rho * np.sin(phi)))

    def _unwrap_angle(self, angle: float) -> float:
        lo = min(0.0, self.phi_span)
        hi = max(0.0, self.phi_span)
        tol = 1.0e-12 * max(1.0, abs(self.phi_span))
        candidates = (angle - 2.0 * pi, angle, angle + 2.0 * pi)
        inside = [value for value in candidates if lo - tol <= value <= hi + tol]
        if len(inside) != 1:
            raise MeshError("cone chart point is outside its owning angular sector")
        return float(np.clip(inside[0], lo, hi))

    def to_parameters(self, points):
        self._current()
        values = _rows2(points)
        out = np.empty_like(values)
        scale = abs(self.dr) / self.slant_length
        for index, (x, y) in enumerate(values):
            rho = float(np.hypot(x, y))
            phi = self._unwrap_angle(float(atan2(y, x)))
            u = phi / self.phi_span
            radius = rho * scale
            v = (radius - self.radius_start) / self.dr
            out[index] = (u, v)
        tol = 1.0e-10
        if np.any(out < -tol) or np.any(out > 1.0 + tol):
            raise MeshError("cone chart point is outside its bound patch")
        return np.clip(out, 0.0, 1.0)

    def evaluate(self, points):
        parameters = self.to_parameters(points)
        values = np.asarray(self.face_chart.evaluate(parameters), dtype=np.float64)
        if values.shape != (len(parameters), 3) or not np.all(np.isfinite(values)):
            raise MeshError("evaluated cone face points must be finite (n, 3) rows")
        self._current()
        return values
