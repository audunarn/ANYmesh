"""Invocation-local physical arithmetic for append-only split proposals."""
import numpy as np
from . import quality_v2 as quality


class PhysicalSplitTrials:
    def __init__(self, settings):
        self.settings = settings
        self.current = None
        self.pending = None
        self.used = False

    def available(self):
        from . import surface_mesh as surface
        return (surface._combined_quality_context() == surface._COMBINED_QUALITY_CONTEXT
                and surface._REFERENCE_PHYSICAL_FULL_SERVICES is not None
                and surface._physical_full_quality_services() == surface._REFERENCE_PHYSICAL_FULL_SERVICES
                and type(self.settings) is surface.SurfaceMeshOptions
                and all(type(getattr(self.settings,name)) in (int,float) for name in
                    ('min_angle','max_angle','min_scaled_jacobian','max_aspect_ratio','max_warpage','max_element_growth'))
                and tuple((function,function.__code__) for function in
                    (surface._published_quality_report,surface._quality_threshold_report,surface._physical_chart_invalid_count))
                    == surface._REFERENCE_PHYSICAL_REPORT_SERVICES
                and surface._quality_patch_reuse_available()
                and quality.triangle_quality is surface._REFERENCE_PHYSICAL_TRIANGLE_QUALITY[0]
                and quality.triangle_quality.__code__ is surface._REFERENCE_PHYSICAL_TRIANGLE_QUALITY[1]
                and quality.native_element_quality is surface._PATCH_QUALITY_SERVICE)

    @staticmethod
    def _keys(cells):
        # Compare ordered connectivity bytes, never row positions or unordered corners.
        return np.ascontiguousarray(cells,dtype=np.int64).view(np.dtype((np.void,24))).reshape(-1)

    def rebase(self, candidate, xyz):
        if self.current is candidate:
            return
        if self.pending is not None and self.pending[0] is candidate:
            self.current,self.points,self.xyz,self.keys,self.values,self.lengths = self.pending
            self.pending = None
            return
        rows = quality.triangle_quality(xyz,candidate.triangles)
        names = ('area','aspect_ratio','minimum_angle','maximum_angle','scaled_jacobian','warpage')
        values = np.column_stack([getattr(rows,name) for name in names])
        lengths = np.linalg.norm(np.roll(xyz[candidate.triangles],-1,axis=1)-xyz[candidate.triangles],axis=2).mean(axis=1)
        keys = self._keys(candidate.triangles)
        order = np.argsort(keys)
        self.keys,self.values,self.lengths = keys[order],values[order],lengths[order]
        self.current,self.points,self.xyz = candidate,candidate.points.copy(),xyz.copy()
        self.pending = None

    @staticmethod
    def _growth(cells,lengths,limit):
        # Same edge ordering and binary64 ratio arithmetic as _published_growth.
        edges = np.sort(np.stack((cells,np.roll(cells,-1,axis=1)),axis=2).reshape(-1,2),axis=1)
        owners = np.repeat(np.arange(len(cells)),3)
        order = np.lexsort((edges[:,1],edges[:,0]))
        sorted_edges = edges[order]
        starts = np.r_[0,1+np.flatnonzero(np.any(sorted_edges[1:]!=sorted_edges[:-1],axis=1))]
        counts = np.diff(np.r_[starts,len(order)])
        starts = starts[counts==2]
        first,second = owners[order[starts]],owners[order[starts+1]]
        with np.errstate(divide='ignore',invalid='ignore'):
            ratios = np.where(np.minimum(lengths[first],lengths[second])>0,
                np.maximum(lengths[first]/lengths[second],lengths[second]/lengths[first]),np.inf)
        failed = ratios>limit
        poor = set(map(int,1+np.r_[first[failed],second[failed]]))
        return max(1.,float(np.max(ratios,initial=1.))),poor

    def trial(self, points, cells, xyz):
        from . import surface_mesh as surface
        from .core import MeshCore
        if (not self.available() or len(points)!=len(self.points)+1
                or xyz.shape!=(len(points),3) or not np.isfinite(xyz).all()
                or not np.array_equal(points[:-1],self.points)
                or not np.array_equal(xyz[:-1],self.xyz)):
            return None
        old_scale=max(float(np.ptp(self.points,axis=0).max()),1.)
        if max(float(np.ptp(points,axis=0).max()),1.) != old_scale:
            return None
        if cells.ndim!=2 or cells.shape[1]!=3 or np.any(cells<0) or np.any(cells>=len(points)):
            return None
        keys = self._keys(cells)
        order = np.argsort(keys)
        if np.any(keys[order][1:]==keys[order][:-1]) or not len(self.keys):
            return None
        positions = np.searchsorted(self.keys,keys)
        safe = np.minimum(positions,len(self.keys)-1)
        unchanged = (positions<len(self.keys)) & (self.keys[safe]==keys)
        added = np.flatnonzero(~unchanged)
        if not len(added) or np.any(~np.any(cells[added]==len(points)-1,axis=1)):
            return None
        rows = cells[added]
        metrics = quality.triangle_quality(xyz,rows)
        names = ('area','aspect_ratio','minimum_angle','maximum_angle','scaled_jacobian','warpage')
        values = np.empty((len(cells),6))
        values[unchanged] = self.values[positions[unchanged]]
        values[added] = np.column_stack([getattr(metrics,name) for name in names])
        lengths = np.empty(len(cells))
        lengths[unchanged] = self.lengths[positions[unchanged]]
        lengths[added] = np.linalg.norm(np.roll(xyz[rows],-1,axis=1)-xyz[rows],axis=2).mean(axis=1)
        triangles = quality.ElementQuality(np.arange(1,len(cells)+1),*(values[:,i] for i in range(6)))
        empty = quality.ElementQuality(np.empty(0,dtype=np.int64),*(np.empty(0) for _ in range(6)))
        core = MeshCore(xyz,cells)
        validity = quality._validate_mesh_with_triangle_metrics(core,triangles)
        complete = quality.MeshQualityV2(triangles,empty,validity)
        thresholds = surface._quality_threshold_report(complete,self.settings)
        maximum_growth,growth_poor = self._growth(cells,lengths,self.settings.max_element_growth)
        poor = sorted(set(thresholds['poor_element_ids'])|growth_poor)
        counts,worst=thresholds['violation_counts'],thresholds['worst']
        invalid=surface._physical_chart_invalid_count(points,cells)+len(validity.errors)
        report=dict(quality_scope='published_elements',invalid_element_count=invalid,
            elements_above_aspect_ratio_5=int(np.count_nonzero(triangles.aspect_ratio>5.)),
            quality_violation_count=len(poor),elements_below_minimum_angle=counts['minimum_angle'],
            elements_above_maximum_angle=counts['maximum_angle'],
            elements_below_minimum_scaled_jacobian=counts['scaled_jacobian'],
            elements_above_maximum_growth=len(growth_poor),max_aspect_ratio=worst['maximum_aspect_ratio'],
            min_scaled_jacobian=worst['minimum_scaled_jacobian'],min_angle=worst['minimum_angle'],
            max_angle=worst['maximum_angle'],max_element_growth=maximum_growth,
            poor_element_ids=poor,repair_element_ids=poor,violation_counts=dict(counts))
        score=(invalid,len(poor),report['elements_above_aspect_ratio_5'],report['max_aspect_ratio'],
               -report['min_scaled_jacobian'],-report['min_angle'])
        candidate=surface._QualityCandidate(np.ascontiguousarray(points),np.ascontiguousarray(cells),
                                           report,score,triangles.aspect_ratio)
        self.pending=(candidate,points.copy(),xyz.copy(),keys[order],values[order],lengths[order])
        self.used=True
        return candidate
