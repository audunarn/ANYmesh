"""Revision-bound physical charts, not a periodic trim/topology certificate.

This adapter deliberately does not unwrap trims, infer seam equivalence, or
authorize cylindrical Frontal-Delaunay activation. Those require the separate
owner occurrence contract and component topology integration.
"""

from dataclasses import dataclass

import numpy as np
from anygeometry.surfaces import Cylinder

from .charts import FaceChart
from .errors import MeshError


def _coordinates(values):
    points = np.asarray(values, dtype=np.float64)
    if points.shape == (2,):
        points = points.reshape(1, 2)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise MeshError("cylinder chart coordinates must be finite (n, 2) rows")
    return points


@dataclass(frozen=True, slots=True)
class CylindricalMetricChart:
    """One owner's normalized cylinder patch expressed in physical metres."""

    face_chart: FaceChart
    model_id: object
    revision: int
    circumferential_length: float
    axial_length: float
    # MESH-B A2: set only for a distinct face parameterization admitted on the
    # owner-certified trimmed route. All chart queries then use the support
    # frame and never the parameterization (see _cylindrical_support_frame).
    support_view: object = None

    @classmethod
    def from_geometry(cls, geometry, face_id, *, support_frame_parameterization=False):
        guard = getattr(geometry, "assert_current", None)
        if callable(guard):
            guard()
        owner = geometry.source if callable(guard) else geometry
        chart = FaceChart(geometry, face_id)
        face = owner.faces[chart.face_id]
        surface = face.surface
        if not isinstance(surface, Cylinder):
            raise MeshError("physical cylinder chart requires an owner Cylinder")
        # Supported-family restriction, not a test for geometric inequality:
        # FaceChart must evaluate the exact support used to scale this chart.
        support_view = None
        if face.parameterization is not None and face.parameterization is not surface:
            if not support_frame_parameterization:
                raise MeshError("physical cylinder chart does not support a distinct face parameterization")
            from ._cylindrical_support_frame import CylinderSupportFrame
            support_view = CylinderSupportFrame(owner, chart.face_id)
        circumference = float(surface.radius * abs(surface.sweep_angle))
        axial = float(abs(surface.height))
        if not np.isfinite(circumference) or not np.isfinite(axial) or min(circumference, axial) <= 0:
            raise MeshError("physical cylinder chart has a degenerate extent")
        return cls(chart, owner.model_id, owner.revision, circumference, axial, support_view)

    def frame_owner(self, geometry):
        """The object answering face-uv queries in this chart's frame."""
        return geometry if self.support_view is None else self.support_view

    def project(self, points):
        """Trim-aware projection to normalized chart parameters (FaceChart.project)."""
        if self.support_view is None:
            return self.face_chart.project(points)
        from .charts import ChartProjection, _rows

        targets = _rows(points, 3, "projection points")
        rows = [self.support_view.project_to_face(self.face_chart.face_id, point) for point in targets]
        return ChartProjection([row[0] for row in rows], [row[1] for row in rows],
                               [float(row[2]) for row in rows])

    def _current(self):
        geometry = self.face_chart.geometry
        guard = getattr(geometry, "assert_current", None)
        if callable(guard):
            guard()
        owner = geometry.source if callable(guard) else geometry
        if owner.model_id != self.model_id or owner.revision != self.revision:
            raise MeshError("cylinder chart binding is stale or belongs to another model")

    def to_chart(self, uv):
        self._current()
        parameters = _coordinates(uv)
        if np.any(parameters < 0.) or np.any(parameters > 1.):
            raise MeshError("owner cylinder parameters must be in [0, 1]")
        return parameters * (self.circumferential_length, self.axial_length)

    def to_parameters(self, points):
        self._current()
        parameters = _coordinates(points) / (self.circumferential_length, self.axial_length)
        if np.any(parameters < 0.) or np.any(parameters > 1.):
            raise MeshError("cylinder chart point is outside its bound patch")
        return parameters

    def evaluate(self, points):
        parameters = self.to_parameters(points)
        if not len(parameters):
            # Preserve FaceChart's existing empty-input rejection.
            return self.face_chart.evaluate(parameters)
        geometry = self.face_chart.geometry
        guard = getattr(geometry, "assert_current", None)
        if callable(guard):
            guard()
        owner = geometry.source if callable(guard) else geometry
        # This private chart is restricted to the exact owner Cylinder support.
        # Do not discover alternative APIs or hide operational batch failures.
        source = owner if self.support_view is None else self.support_view
        values = np.array(
            source.evaluate_face_many(self.face_chart.face_id, parameters),
            dtype=np.float64, order="C", copy=True,
        )
        if values.shape != (len(parameters), 3):
            raise MeshError("evaluated face points must have shape (n, 3) with unchanged row count")
        if not np.all(np.isfinite(values)):
            raise MeshError("evaluated face points must be finite")
        values.setflags(write=False)
        self._current()
        return values

    def jacobians(self, points):
        from anygeometry import face_derivatives_many

        parameters = self.to_parameters(points)
        geometry = self.face_chart.geometry
        guard = getattr(geometry, "assert_current", None)
        owner = geometry.source if callable(guard) else geometry
        # Owner derivatives are with respect to normalized face (u, v). The
        # constructor restricts evaluation to that exact Cylinder support.
        if self.support_view is None:
            du, dv = face_derivatives_many(owner, self.face_chart.face_id, parameters)
        else:
            du, dv = self.support_view.face_derivatives_many(self.face_chart.face_id, parameters)
        result = np.stack((du / self.circumferential_length, dv / self.axial_length), axis=2)
        self._current()
        if result.shape != (len(parameters), 3, 2) or not np.all(np.isfinite(result)):
            raise MeshError("owner cylinder derivatives must produce finite (n, 3, 2) rows")
        return result
