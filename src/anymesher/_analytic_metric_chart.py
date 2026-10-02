"""Owner-certified metric evaluation for general analytic material faces."""
import numpy as np
from anygeometry import (Cone, query_trimmed_surface_charts,
                         validate_trimmed_surface_charts_binding, face_derivatives_many)
try:
    from anygeometry import ExtrudedSurface
except ImportError:
    ExtrudedSurface = ()
from .errors import MeshError


class AnalyticMetricChart:
    def __init__(self, geometry, face_id, cancellation_check=None, *, region_binding=None):
        guard = getattr(geometry, 'assert_current', None)
        if callable(guard):
            guard()
        self.owner = geometry.source if callable(guard) else geometry
        self.face_id = int(face_id)
        face = self.owner.faces[self.face_id]
        if type(face.surface) not in tuple(t for t in (Cone, ExtrudedSurface) if isinstance(t,type)) or face.parameterization is not None:
            raise MeshError('general analytic metric chart requires an unmodified owner support')
        self.check = cancellation_check
        self.region_binding = region_binding
        if region_binding is None:
            self.binding = query_trimmed_surface_charts(self.owner, (self.face_id,),
                                                       cancellation_check=cancellation_check)
            du, dv = face_derivatives_many(self.owner, self.face_id, [[.5,.5]])
        else:
            if region_binding.geometry is not self.owner or self.face_id not in region_binding.face_ids:
                raise MeshError('analytic region chart does not bind this working face')
            self.binding = region_binding.collection
            self._current()
            from anygeometry import evaluate_material_surface_region
            du, dv = evaluate_material_surface_region(
                self.owner, self.binding, self.face_id, [[.5,.5]], derivatives=True,
                cancellation_check=self.check)
        derivative = np.column_stack((du[0],dv[0]))
        try:
            self.transform = np.linalg.cholesky(derivative.T @ derivative)
            self.inverse = np.linalg.inv(self.transform)
        except np.linalg.LinAlgError as error:
            raise MeshError("analytic metric chart has a singular reference differential") from error

    def _current(self):
        if self.region_binding is None:
            validate_trimmed_surface_charts_binding(self.owner, self.binding,
                                                   cancellation_check=self.check)
        else:
            self.region_binding.validate(self.check)

    def _rows(self, points):
        self._current()
        rows = np.asarray(points, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != 2 or not np.all(np.isfinite(rows)):
            raise MeshError('analytic chart requires finite (n, 2) parameters')
        return rows

    def evaluate(self, points):
        rows = self._rows(points)
        if self.region_binding is None:
            values = np.asarray(self.owner.evaluate_face_many(self.face_id, rows @ self.inverse), dtype=float)
        else:
            from anygeometry import evaluate_material_surface_region
            values = np.asarray(evaluate_material_surface_region(
                self.owner, self.binding, self.face_id, rows @ self.inverse,
                cancellation_check=self.check), dtype=float)
        self._current()
        if values.shape != (len(rows), 3) or not np.all(np.isfinite(values)):
            raise MeshError('owner analytic chart evaluation is invalid')
        return values

    def jacobians(self, points):
        rows = self._rows(points)
        if self.region_binding is None:
            du, dv = face_derivatives_many(self.owner, self.face_id, rows @ self.inverse)
        else:
            from anygeometry import evaluate_material_surface_region
            du, dv = evaluate_material_surface_region(
                self.owner, self.binding, self.face_id, rows @ self.inverse,
                derivatives=True, cancellation_check=self.check)
        values = np.stack((du, dv), axis=2) @ self.inverse.T
        self._current()
        if values.shape != (len(rows), 3, 2) or not np.all(np.isfinite(values)):
            raise MeshError('owner analytic chart derivatives are invalid')
        return values

    def certify_core(self, core, settings, boundary_points, *, registered_rows=None):
        from .core import MeshCore
        xyz = self.evaluate(core.node_coordinates[:, :2]).copy()
        xyz[:len(boundary_points)] = boundary_points
        for row, point in (registered_rows or {}).items():
            xyz[int(row)] = point
        physical = MeshCore(xyz, core.triangle_connectivity, core.quad_connectivity,
                            triangle_active=core.triangle_active, quad_active=core.quad_active)
        return self.certify_physical_core(physical, settings)

    def certify_physical_core(self, core, settings):
        from .quality_v2 import evaluate_quality
        from .surface_mesh import _quality_threshold_report
        from .errors import StructuredQualityRejected
        xyz = np.asarray(core.node_coordinates, dtype=float)
        physical = core
        if self.check is not None:
            self.check('analytic physical quality start')
        report = _quality_threshold_report(evaluate_quality(physical), settings)
        if self.check is not None:
            self.check('analytic physical quality complete')
        perimeters = []
        incidence = {}
        for rows in (core.triangle_connectivity[core.triangle_active],
                     core.quad_connectivity[core.quad_active]):
            for index, cell in enumerate(rows):
                if self.check is not None and index % 256 == 0:
                    self.check('analytic physical growth incidence')
                identifier = len(perimeters)
                perimeters.append(float(np.linalg.norm(np.roll(xyz[cell], -1, axis=0)-xyz[cell], axis=1).mean()))
                for a,b in zip(cell,np.roll(cell,-1)):
                    incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append(identifier)
        growth = 1.
        for index, owners in enumerate(incidence.values()):
            if self.check is not None and index % 256 == 0:
                self.check('analytic physical growth quality')
            if len(owners) == 2:
                a, b = owners
                growth = max(growth, perimeters[a]/perimeters[b], perimeters[b]/perimeters[a])
        report['maximum_element_growth'] = growth
        if not report['accepted'] or growth > settings.max_element_growth:
            raise StructuredQualityRejected(
                f'analytic face {self.face_id} failed physical quality: '
                f"{report['violation_counts']}, growth={growth}")
        return report
