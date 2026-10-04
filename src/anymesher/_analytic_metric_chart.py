"""Owner-certified metric evaluation for general analytic material faces."""
import numpy as np
from anygeometry import (Cone, query_trimmed_surface_charts,
                         validate_trimmed_surface_charts_binding, face_derivatives_many,
                         chart_definition_fingerprint)
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
        owner_identity = (self.owner, str(self.owner.model_id), self.owner.revision,
                          self.face_id, region_binding)
        if region_binding is None:
            self.binding = query_trimmed_surface_charts(self.owner, (self.face_id,),
                                                       cancellation_check=cancellation_check)
        else:
            if region_binding.geometry is not self.owner or self.face_id not in region_binding.face_ids:
                raise MeshError('analytic region chart does not bind this working face')
            self.binding = region_binding.collection
        source = self.binding if region_binding is None else self.binding.source
        region = None if region_binding is None else region_binding.region
        collection_entry = (self.binding, source, source.source_checksum,
                            chart_definition_fingerprint(self.binding), region,
                            None if region is None else chart_definition_fingerprint(region))
        if region_binding is None:
            du, dv = face_derivatives_many(self.owner, self.face_id, [[.5,.5]])
        else:
            from anygeometry import evaluate_material_surface_region
            du, dv = evaluate_material_surface_region(
                self.owner, self.binding, self.face_id, [[.5,.5]], derivatives=True,
                cancellation_check=self.check)
        if (self.owner is not owner_identity[0]
                or str(self.owner.model_id) != owner_identity[1]
                or self.owner.revision != owner_identity[2]
                or self.face_id != owner_identity[3]
                or self.region_binding is not owner_identity[4]):
            raise MeshError('analytic chart owner changed during reference selection')
        if (self.binding is not collection_entry[0]
                or (self.binding if region_binding is None else self.binding.source)
                    is not collection_entry[1]
                or collection_entry[1].source_checksum != collection_entry[2]
                or chart_definition_fingerprint(self.binding) != collection_entry[3]
                or (None if region_binding is None else region_binding.region)
                    is not collection_entry[4]
                or (region_binding is not None and
                    chart_definition_fingerprint(region_binding.region) != collection_entry[5])):
            raise MeshError('analytic chart collection changed during reference selection')
        du, dv = np.asarray(du, dtype=np.float64), np.asarray(dv, dtype=np.float64)
        if (du.shape != (1, 3) or dv.shape != (1, 3)
                or not np.isfinite(du).all() or not np.isfinite(dv).all()):
            raise MeshError('analytic chart reference derivative is invalid')
        derivative = np.column_stack((du[0],dv[0]))
        try:
            self.transform = np.linalg.cholesky(derivative.T @ derivative)
            self.inverse = np.linalg.inv(self.transform)
        except np.linalg.LinAlgError as error:
            raise MeshError("analytic metric chart has a singular reference differential") from error
        if (not np.isfinite(self.transform).all() or not np.isfinite(self.inverse).all()
                or not np.allclose(self.transform @ self.transform.T,
                                   derivative.T @ derivative, rtol=64*np.finfo(float).eps, atol=0.)
                or not np.allclose(self.transform @ self.inverse, np.eye(2),
                                   rtol=64*np.finfo(float).eps,
                                   atol=64*np.finfo(float).eps)):
            raise MeshError('analytic chart reference metric or inverse is invalid')
        self._reference_derivative = derivative.copy()
        self._reference_derivative.setflags(write=False)
        self._authority = (owner_identity, *collection_entry,
                           self._reference_derivative.tobytes(),
                           self.transform.tobytes(), self.inverse.tobytes())
        self.validate_reference()

    def _check_reference(self, entry, *, check_binding=True, check_revision=True):
        (identity, binding, source, source_checksum, binding_checksum,
         region, region_checksum, derivative_bytes, transform_bytes,
         inverse_bytes) = entry
        if (self._authority is not entry or self.owner is not identity[0]
                or str(self.owner.model_id) != identity[1]
                or (check_revision and self.owner.revision != identity[2])
                or self.face_id != identity[3]
                or self.region_binding is not identity[4]
                or (check_binding and self.binding is not binding)):
            raise MeshError('analytic chart owner or entry authority changed')
        if check_binding:
            current_source = self.binding if self.region_binding is None else self.binding.source
            current_region = None if self.region_binding is None else self.region_binding.region
            if (current_source is not source
                    or current_source.source_checksum != source_checksum
                    or chart_definition_fingerprint(self.binding) != binding_checksum
                    or current_region is not region
                    or (current_region is not None
                        and chart_definition_fingerprint(current_region) != region_checksum)):
                raise MeshError('analytic chart original collection or region changed')
        derivative = np.asarray(self._reference_derivative, dtype=np.float64)
        transform = np.asarray(self.transform, dtype=np.float64)
        inverse = np.asarray(self.inverse, dtype=np.float64)
        if (derivative.shape != (3, 2) or transform.shape != (2, 2)
                or inverse.shape != (2, 2)
                or not all(np.isfinite(array).all() for array in
                           (derivative, transform, inverse))
                or derivative.tobytes() != derivative_bytes
                or transform.tobytes() != transform_bytes
                or inverse.tobytes() != inverse_bytes):
            raise MeshError('analytic chart reference metric or inverse changed')

    def validate_reference(self):
        """Check the owner-bound reference choice without serializing the model."""
        entry = self._authority
        self._current()
        self._check_reference(entry)
        return self

    def _current(self):
        if self.region_binding is None:
            validate_trimmed_surface_charts_binding(self.owner, self.binding,
                                                   cancellation_check=self.check)
        else:
            self.region_binding.validate(self.check)

    def _rows(self, points):
        rows = np.array(points, dtype=float, copy=True)
        if rows.ndim != 2 or rows.shape[1] != 2 or not np.all(np.isfinite(rows)):
            raise MeshError('analytic chart requires finite (n, 2) parameters')
        # Region evaluation performs owner binding checks before and after the
        # calculation itself. Do not repeat them around that public operation.
        if self.region_binding is None:
            self._current()
        return rows

    def evaluate(self, points):
        entry = self._authority
        rows = self._rows(points)
        self._check_reference(entry, check_binding=False, check_revision=False)
        inverse = self.inverse.copy()
        if self.region_binding is None:
            values = np.asarray(self.owner.evaluate_face_many(self.face_id, rows @ inverse), dtype=float)
        else:
            from anygeometry import evaluate_material_surface_region
            values = np.asarray(evaluate_material_surface_region(
                self.owner, self.binding, self.face_id, rows @ inverse,
                cancellation_check=self.check), dtype=float)
        if self.region_binding is None:
            self._current()
        self._check_reference(entry)
        if values.shape != (len(rows), 3) or not np.all(np.isfinite(values)):
            raise MeshError('owner analytic chart evaluation is invalid')
        return values

    def jacobians(self, points):
        entry = self._authority
        rows = self._rows(points)
        self._check_reference(entry, check_binding=False, check_revision=False)
        inverse = self.inverse.copy()
        if self.region_binding is None:
            du, dv = face_derivatives_many(self.owner, self.face_id, rows @ inverse)
        else:
            from anygeometry import evaluate_material_surface_region
            du, dv = evaluate_material_surface_region(
                self.owner, self.binding, self.face_id, rows @ inverse,
                derivatives=True, cancellation_check=self.check)
        values = np.stack((du, dv), axis=2) @ inverse.T
        if self.region_binding is None:
            self._current()
        self._check_reference(entry)
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
