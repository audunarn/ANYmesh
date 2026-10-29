"""Discretization adapters for geometry-owner qualified material regions."""
from dataclasses import dataclass, replace, field
from fractions import Fraction
from numbers import Integral
import numpy as np

from .errors import MeshError


from contextlib import contextmanager
from contextvars import ContextVar
from threading import get_ident
from anygeometry._detached_geometry_read import MaterialNumericalRead as _CanonicalMaterialRead
_CANONICAL_MATERIAL_EVALUATE = _CanonicalMaterialRead.evaluate

_MATERIAL_READ_SCOPE = ContextVar('anymesher_material_read_scope', default=None)
_MATERIAL_SNAPSHOT_READS = {}


def _material_snapshot_runtime():
    from ._analytic_metric_chart import AnalyticMetricChart
    descriptor = vars(MaterialRegionBinding).get('requires_curvilinear')
    if type(descriptor) is not property:
        return None
    pins = tuple(_boundary_callable_pin(function) for function in (
        _material_snapshot_runtime,_material_binding_snapshot,_material_chart_snapshot,_material_read_view,
        _material_frontal_driver_pin,_material_native_dispatch_guard,
        vars(MaterialRegionBinding).get('_require_construction'),descriptor.fget,
        *(vars(_CanonicalMaterialRead).get(name) for name in ('evaluate','check_open','validate','close')),
        *(vars(AnalyticMetricChart).get(name) for name in ('evaluate','jacobians','_rows','_numeric_view','_check_reference'))))
    layouts = tuple((id(kind),tuple(id(vars(kind).get(name)) for name in names)) for kind,names in (
        (MaterialRegionBinding,('geometry','collection','region','authored_face','geometry_approximation_error','__getattribute__')),
        (AnalyticMetricChart,('owner','binding','region_binding','_authority','_reference_uv','transform','inverse','_reference_derivative','__getattribute__'))))
    from types import FunctionType
    driver_check=_material_frontal_driver_pin
    if (driver_check is not _MATERIAL_DRIVER_CHECK or type(driver_check) is not FunctionType
            or driver_check.__code__ is not _MATERIAL_DRIVER_CHECK_CODE
            or driver_check.__defaults__ is not None or driver_check.__kwdefaults__ is not None):return None
    driver=_MATERIAL_DRIVER_CHECK()
    return None if driver is None or any(pin is None for pin in pins) else (pins,layouts,driver)


def _material_binding_snapshot(binding):
    state = _MATERIAL_READ_SCOPE.get()
    record = _MATERIAL_SNAPSHOT_READS.get(id(state))
    if record is None or record['state'] is not state or record['binding'] is not binding or record['validating']:
        return None
    if (not state['active'] or record['thread'] != get_ident()
            or state['view'] is not record['view'] or _material_snapshot_runtime() != record['runtime']):
        raise MeshError('material snapshot identity, thread or implementation changed')
    data = record['binding_dict']
    if any(data.get(name) is not value for name,value in record['binding_identity']):
        raise MeshError('material snapshot construction identity changed')
    return record


def _material_chart_snapshot(chart, *, create=False):
    state = _MATERIAL_READ_SCOPE.get()
    record = _MATERIAL_SNAPSHOT_READS.get(id(state))
    if record is None or record['state'] is not state or record['validating']:
        return None
    _material_binding_snapshot(record['binding'])
    saved = record['charts'].get(id(chart))
    if saved is None:
        if not create:
            return None
        from ._analytic_metric_chart import AnalyticMetricChart
        if type(chart) is not AnalyticMetricChart:
            return None
        data = object.__getattribute__(chart,'__dict__')
        if data.get('region_binding') is not record['binding'] or data.get('check') is not None:
            return None
        record['validating'] = True
        try:
            record['binding'].validate(None)
            chart.validate_reference()
        finally:
            record['validating'] = False
        saved = dict(chart=chart,data=data,identity=tuple((name,data[name]) for name in
            ('owner','face_id','region_binding','binding','check','_authority','_reference_uv','transform','inverse','_reference_derivative')),
            inverse=np.frombuffer(chart.inverse.tobytes(),dtype=chart.inverse.dtype).reshape(chart.inverse.shape),
            transform=np.frombuffer(chart.transform.tobytes(),dtype=chart.transform.dtype).reshape(chart.transform.shape),
            reference=tuple(chart._reference_uv))
        from types import MappingProxyType
        saved = MappingProxyType(saved)
        record['charts'][id(chart)] = saved
    if saved['chart'] is not chart or any(saved['data'].get(name) is not value for name,value in saved['identity']):
        raise MeshError('material chart snapshot identity changed')
    return saved


def _validate_material_snapshot(record):
    record['validating'] = True
    try:
        record['binding'].validate(None)
        for saved in record['charts'].values():
            saved['chart'].validate_reference()
    finally:
        record['validating'] = False


def _material_scope_backend(options):
    # Unknown/user driver or provider callbacks retain full per-read checks.
    from types import FunctionType
    driver_check=_material_frontal_driver_pin
    if (driver_check is not _MATERIAL_DRIVER_CHECK or type(driver_check) is not FunctionType
            or driver_check.__code__ is not _MATERIAL_DRIVER_CHECK_CODE
            or driver_check.__defaults__ is not None or driver_check.__kwdefaults__ is not None):return None
    if _MATERIAL_DRIVER_CHECK() is None:return None
    from .native_v2 import NativeMeshingOptions
    from .metric import MetricFieldSpec
    from .refinement import SizeField
    native = options.get('native_options')
    if (options.get('cancellation_check') is not None
            or type(native) is not NativeMeshingOptions
            or native.experimental_metric_provider is not None
            or native.metric_field is not None and type(native.metric_field) is not MetricFieldSpec
            or type(options.get('size_field')) is not SizeField):
        return None
    # Only data-only canonical controls. Exact outer types alone do not rule
    # out a custom control nested inside an otherwise valid specification.
    from .metric import IsotropicMetricControl, FeatureDistanceMetricControl, ImportedMetricSamples
    from .refinement import Refinement
    from dataclasses import fields, is_dataclass
    def numeric_data(value):
        if value is None or type(value) in (str,int,float,bool): return True
        if type(value) in (tuple,list): return all(numeric_data(x) for x in value)
        if type(value) is np.ndarray: return value.dtype.kind in 'biuf' and np.isfinite(value).all()
        if type(value) in (IsotropicMetricControl,FeatureDistanceMetricControl,ImportedMetricSamples):
            return all(numeric_data(getattr(value,f.name)) for f in fields(value))
        return False
    metric = native.metric_field
    if metric is not None and not all(numeric_data(getattr(metric,f.name)) for f in fields(metric)):
        return None
    size = options['size_field']
    if any(type(zone) is not Refinement for zone in (*size.zones,*size.overrides)):
        return None
    if any(callable(value) for value in vars(size).values()): return None
    from anygeometry._detached_geometry_read import canonical_material_context
    if not canonical_material_context(): return None
    backend = options.get('native_backend')
    if backend is None or type(backend) is str and backend == 'python':
        return ('python',)
    if type(backend) is not str or backend not in ('auto','native'):
        return None
    from .native import snapshot_native_boundary
    selection = snapshot_native_boundary()
    if selection is None:
        return ('auto-python',) if backend == 'auto' else None
    if selection.source != 'built_in' or selection.name != 'anymesher-cpp17':
        return None
    kind = type(selection.boundary)
    methods = tuple((name,getattr(kind,name,None),getattr(getattr(kind,name,None),'__code__',None))
                    for name in ('triangulate','triangulate_cancellable'))
    return ('built_in',selection.boundary,kind,methods,
            tuple((getattr(selection.boundary,name,None),
                   getattr(getattr(kind,name,None),'__defaults__',None),
                   repr(getattr(getattr(kind,name,None),'__kwdefaults__',None)))
                  for name in ('triangulate','triangulate_cancellable')))


@contextmanager
def material_read_scope(binding, options):
    backend = _material_scope_backend(options)
    if type(binding) is not MaterialRegionBinding or backend is None:
        yield None
        return
    previous = _MATERIAL_SNAPSHOT_READS.get(id(_MATERIAL_READ_SCOPE.get()))
    if previous is not None:
        previous['validating'] = True
    try:
        binding.validate(None)
    finally:
        if previous is not None:
            previous['validating'] = False
    state = {'binding':binding,'view':None,'thread':get_ident(),'evaluations':0,
             'backend':backend,'active':True,'options':options}
    from anygeometry._detached_geometry_read import detach_numeric,DetachedOperandUnavailable
    try:
        construction = detach_numeric((binding.collection,binding.region,binding.authored_face,binding.geometry_approximation_error))
    except DetachedOperandUnavailable:
        yield None
        return
    runtime = _material_snapshot_runtime()
    if runtime is None:
        yield None
        return
    record = dict(state=state,binding=binding,binding_dict=object.__getattribute__(binding,'__dict__'),
        binding_identity=tuple((name,getattr(binding,name)) for name in
            ('geometry','collection','region','authored_face','geometry_approximation_error')),
        construction=construction,curvilinear=binding.requires_curvilinear,face=binding.representative,
        support_type=type(binding.region.support),
        thread=get_ident(),runtime=runtime,view=None,charts={},validating=False)
    chords = binding._curvilinear_state.get('chords')
    if chords is None or not record['curvilinear']:
        yield None
        return
    # Freeze the actual authenticated support at entry, before any private
    # caller work. Detached construction copies are never used as receipts.
    view = _CanonicalMaterialRead(binding.geometry,binding.collection,record['face'],chords,binding._local_allocation())
    state['view'] = record['view'] = view
    _MATERIAL_SNAPSHOT_READS[id(state)] = record
    token = _MATERIAL_READ_SCOPE.set(state)
    failure = None
    try:
        yield state
        if _material_scope_backend(options) != backend:
            raise MeshError('native material read backend changed during attempt')
        _material_binding_snapshot(binding)
        _validate_material_snapshot(record)
        if state['view'] is not None:
            if (binding._curvilinear_state.get('chords') is not state['view'].chords
                    or binding._local_allocation() is not state['view'].allocation):
                raise MeshError('material read authority changed after final numerical evaluation')
            state['view'].validate()
            if binding.reference_uv(None) != state['view'].reference_uv:
                raise MeshError('material regular anchor changed during attempt')
            state['evaluations'] += state['view'].evaluations
    except BaseException as error:
        failure = error
        raise
    finally:
        state['active'] = False
        cleanup_error = None
        try:
            if record['view'] is not None:
                object.__setattr__(record['view'],'_active',False)
        except BaseException as error:
            cleanup_error = error
        finally:
            _MATERIAL_SNAPSHOT_READS.pop(id(state),None)
            _MATERIAL_READ_SCOPE.reset(token)
        if failure is None and cleanup_error is not None:
            raise cleanup_error


def _material_read_view(binding):
    state = _MATERIAL_READ_SCOPE.get()
    if state is None or state['binding'] is not binding:
        return None
    record = _MATERIAL_SNAPSHOT_READS.get(id(state))
    if record is not None and record['validating']:
        return None
    if not state['active'] or state['thread'] != get_ident():
        raise MeshError('material read scope is closed or belongs to another thread')
    snapshot = _material_binding_snapshot(binding)
    if not (snapshot['curvilinear'] if snapshot is not None else binding.requires_curvilinear):
        return None
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.surfaces import Plane
    support_type = snapshot['support_type'] if snapshot is not None else type(binding.region.support)
    if support_type not in (NurbsSurface,Plane):
        return None
    from anygeometry._detached_geometry_read import canonical_material_context
    if not canonical_material_context():
        if state['view'] is not None:
            raise MeshError('material numerical implementation changed during attempt')
        return None
    chords = binding._curvilinear_state.get('chords')
    if chords is None: return None
    allocation = binding._local_allocation()
    view = state['view']
    if view is not None and (view.chords is not chords or view.allocation is not allocation):
        if snapshot is not None:
            _validate_material_snapshot(snapshot)
        view.validate()  # Real station/authority transition, never a revision shortcut.
        state['evaluations'] += view.evaluations
        object.__setattr__(view,'_active',False)
        view = None
    if view is None:
        from anygeometry._detached_geometry_read import MaterialNumericalRead
        view = MaterialNumericalRead(binding.geometry,binding.collection,
            snapshot['face'] if snapshot is not None else binding.representative,chords,allocation)
        state['view'] = view
        if snapshot is not None:
            snapshot['view'] = view
    view.check_open()
    return view


# Invocation-local candidates only. Mesh schema/serde contains no proof objects.
# Mesh is an unhashable dataclass, so identity keys have explicit weakref guards.
_CURVILINEAR_MESH_REFERENCES = {}
_CURVILINEAR_BINDING_IDENTITIES = {}


class _SourceCellResolutionRequired(MeshError):
    """Invocation-local retry request, issued only at the actual owner proof."""


def _is_source_cell_resolution_request(error, cells, error_limit, *, geometry=None, chords=None, local_error_limit=None):
    # Older geometry owners retain their existing fail-closed behavior. Never
    # infer refinement authority from a message or a generic GeometryError.
    import anygeometry
    signal = getattr(anygeometry, 'SourceCellRefinementRequired', None)
    inconclusive = getattr(anygeometry, 'SourceCellApproximationInconclusive', None)
    if type(error) not in tuple(kind for kind in (signal,inconclusive) if kind is not None):
        return False
    face = getattr(error, 'face_id', None)
    nodes = getattr(error, 'node_ids', None)
    limit = getattr(error, 'error_limit', None)
    requested = getattr(error, 'requested_error', None)
    if local_error_limit is not None and (type(limit) is not Fraction or limit>local_error_limit):
        return False
    matching = (type(face) is int and type(nodes) is tuple
            and all(type(node) is int for node in nodes)
            and type(requested) is Fraction and requested == error_limit
            and type(limit) is Fraction and 0 < limit <= requested
            and any(cell.face.id == face and cell.node_ids == nodes for cell in cells))
    if not matching:
        return False
    if type(error) is signal:
        lower = getattr(error, 'lower_error_squared', None)
        return type(lower) is Fraction and lower > limit * limit
    upper,wedge = getattr(error,'upper_error',None),getattr(error,'jacobian_lower',None)
    box,depth = getattr(error,'proof_box',None),getattr(error,'proof_depth',None)
    if (type(upper) is not Fraction or upper <= limit or type(wedge) is not Fraction or wedge <= 0
            or type(depth) is not int or depth < 0 or type(box) is not tuple or len(box) != 2
            or any(type(row) is not tuple or len(row) != 2 or any(type(x) is not Fraction for x in row)
                   or not 0 <= row[0] < row[1] <= 1 for row in box)
            or geometry is None or chords is None):
        return False
    from anygeometry.definition_binding import definition_checksum
    from uuid import UUID
    return (type(getattr(error,'source_model_id',None)) is UUID
            and error.source_model_id == geometry.model_id
            and type(getattr(error,'source_revision',None)) is int
            and error.source_revision == geometry.revision
            and type(getattr(error,'source_checksum',None)) is str
            and type(getattr(error,'chord_checksum',None)) is str
            and getattr(error,'source_checksum',None) == chords.source_checksum
            and getattr(error,'chord_checksum',None) == definition_checksum(chords)
            and any(cell.face.id == face and cell.node_ids == nodes
                    and cell.face.model_id == geometry.model_id for cell in cells))


def face_requires_curvilinear(geometry, face_id):
    """Routing hint only; the geometry owner still proves the material chart."""
    from anygeometry.nurbs import NurbsSurface, NurbsCurve
    from anygeometry import Plane, CoonsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    face = geometry.faces[face_id]
    if isinstance(face.surface, NurbsSurface) or isinstance(face.parameterization, NurbsSurface):
        return True
    # Raw authored spline sweeps acquire their effective chart only through the
    # owner's read-only qualifier. Boundary type routes there; it proves nothing.
    if isinstance(face.surface, CoonsSurface) and any(
            isinstance(geometry.edges[use.edge].curve, NurbsCurve)
            for loop in (face.loop, *face.holes) for use in loop):
        return True
    return isinstance(face.surface, Plane) and any(
        isinstance(geometry.edges[use.edge].curve, PlaneNurbsCurve)
        for loop in (face.loop, *face.holes) for use in loop)


def region_requires_curvilinear(region):
    from anygeometry.nurbs import NurbsSurface
    from anygeometry import Plane
    from anygeometry.support_curves import PlaneNurbsCurve
    support = getattr(region, 'support', None)
    return (isinstance(support, NurbsSurface) or
            isinstance(support, Plane) and any(isinstance(path.curve, PlaneNurbsCurve)
                for loop in region.boundaries for path in loop))


def _check_cancellation(check, phase):
    if check is not None and check(phase):
        raise MeshError(f'{phase} cancelled')


def boundary_registry_pin(registry):
    """Literal stations plus the actual immutable-view owner of those stations."""
    from ._authored_component_stage import _registry_receipt
    from .boundary import BoundaryKey, BoundaryEntry
    expected = {}
    if type(registry._entries) is not dict or type(registry._by_edge) is not dict:
        raise MeshError('NURBS boundary registry indexes disagree')
    for key, entry in registry._entries.items():
        if type(key) is not BoundaryKey or type(entry) is not BoundaryEntry or entry.key != key:
            raise MeshError('NURBS boundary registry indexes disagree')
        expected.setdefault(key.edge_id, {})[key.parameter] = entry
    if set(registry._by_edge) != set(expected):
        raise MeshError('NURBS boundary registry indexes disagree')
    for edge, entries in registry._by_edge.items():
        if (type(edge) is not int or type(entries) is not dict or set(entries) != set(expected[edge])
                or any(type(parameter) is not float or entry is not expected[edge][parameter]
                       for parameter, entry in entries.items())):
            raise MeshError('NURBS boundary registry indexes disagree')
    view = registry.view
    mappings = ('vertices', 'edges', 'faces', 'parts', 'sheets', 'face_uses', 'coedges',
                'members', 'member_edge_uses', 'attachments', 'junctions', 'sheet_faces',
                'member_uses', 'face_sheet_owners', 'edge_sheet_owners', 'edge_member_owners',
                'face_use_sheet_owner')
    return (id(view), id(view.source), str(view.model_id), view.revision, repr(view.tolerance),
            tuple((name, id(getattr(view, name))) for name in mappings), repr(_registry_receipt(registry)),
            id(registry._entries), id(registry._by_edge),
            tuple((edge, id(entries)) for edge, entries in sorted(registry._by_edge.items())))


def retain_curvilinear_reference_candidates(mesh, candidates, *, boundary_registry=None):
    """Retain owner receipts without extending canonical Mesh fields.

    Retrieval grants no authority: source/model checksums and literal cells must
    be freshly validated by the owner before any use or two-model transport.
    """
    from weakref import ref
    key = id(mesh)
    candidates = tuple(candidates)
    if not candidates:
        entry = _CURVILINEAR_MESH_REFERENCES.get(key)
        if entry is not None and entry[0]() is mesh:
            _CURVILINEAR_MESH_REFERENCES.pop(key, None)
        return
    def cleanup(reference):
        current = _CURVILINEAR_MESH_REFERENCES.get(key)
        if current is not None and current[0] is reference:
            _CURVILINEAR_MESH_REFERENCES.pop(key, None)
    registry_authority = None if boundary_registry is None else (
        ref(boundary_registry), boundary_registry_pin(boundary_registry),
        curvilinear_boundary_associations(mesh))
    _CURVILINEAR_MESH_REFERENCES[key] = (ref(mesh, cleanup), candidates, registry_authority)


def curvilinear_reference_candidates(mesh):
    """Return candidates only for this live object, never clones or serde."""
    entry = _CURVILINEAR_MESH_REFERENCES.get(id(mesh))
    return () if entry is None or entry[0]() is not mesh else entry[1]


def curvilinear_boundary_associations(mesh):
    return (tuple((edge, tuple(nodes)) for edge, nodes in sorted(mesh.nodes_of_edge.items())),
            tuple((edge, tuple(nodes)) for edge, nodes in sorted(mesh.offset_nodes_of_edge.items())),
            tuple(sorted(mesh.node_of_vertex.items())))


def curvilinear_reference_registry(mesh):
    entry = _CURVILINEAR_MESH_REFERENCES.get(id(mesh))
    return None if entry is None or entry[0]() is not mesh or len(entry) < 3 else entry[2]


from weakref import ref as _local_binding_ref
_LOCAL_ALLOCATION_BINDINGS={}

def checked_geometry_approximation_error(value):
    """Validate explicit source/linear-FE discrepancy authority."""
    if value is None:
        return None
    from numbers import Real
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MeshError('geometry_approximation_error must be a finite positive real')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise MeshError('geometry_approximation_error must be a finite positive real') from error
    if not np.isfinite(value) or value <= 0.:
        raise MeshError('geometry_approximation_error must be a finite positive real')
    return value


@dataclass(frozen=True)
class BoundaryIntervalReceipt:
    """Source identity of one owner boundary occurrence, never split permission."""
    binding: object
    loop_index: int
    path_index: int
    interval_index: int
    source_edge: int
    nodes: tuple[int, int]
    parameters: tuple[Fraction, Fraction]
    uv: tuple[tuple[float, float], tuple[float, float]]
    xyz: tuple[tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class MaterialRegionBinding:
    geometry: object
    collection: object
    region: object
    authored_face: int
    geometry_approximation_error: float | None = None
    _curvilinear_state: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        entry = _CURVILINEAR_BINDING_IDENTITIES.get(id(self))
        if entry is not None and entry[0]() is self:
            self._require_construction()
            return
        if not region_requires_curvilinear(self.region):
            return
        from anygeometry import chart_definition_fingerprint
        from weakref import ref
        key = id(self)
        def cleanup(reference):
            entry = _CURVILINEAR_BINDING_IDENTITIES.get(key)
            if entry is not None and entry[0] is reference:
                _CURVILINEAR_BINDING_IDENTITIES.pop(key, None)
        _CURVILINEAR_BINDING_IDENTITIES[key] = (ref(self, cleanup), self.geometry,
            self.collection, self.region, (type(self.authored_face), self.authored_face),
            (type(self.geometry_approximation_error), self.geometry_approximation_error),
            chart_definition_fingerprint(self.collection), chart_definition_fingerprint(self.region))

    def _require_construction(self):
        """Candidate caches cannot replace the original owner or error limit."""
        from anygeometry import chart_definition_fingerprint
        if _material_binding_snapshot(self) is not None:
            return
        entry = _CURVILINEAR_BINDING_IDENTITIES.get(id(self))
        if entry is None or entry[0]() is not self:
            if region_requires_curvilinear(self.region):
                raise MeshError('NURBS material binding lacks construction authority')
            return
        if (self.geometry is not entry[1] or self.collection is not entry[2]
                or self.region is not entry[3]
                or (type(self.authored_face), self.authored_face) != entry[4]
                or (type(self.geometry_approximation_error), self.geometry_approximation_error) != entry[5]
                or chart_definition_fingerprint(self.collection) != entry[6]
                or chart_definition_fingerprint(self.region) != entry[7]):
            raise MeshError('NURBS material binding changed from its construction authority')

    @property
    def requires_curvilinear(self):
        snapshot = _material_binding_snapshot(self)
        if snapshot is not None:
            return snapshot['curvilinear']
        self._require_construction()
        return region_requires_curvilinear(self.region)

    def _local_allocation(self):
        entry=_LOCAL_ALLOCATION_BINDINGS.get(id(self))
        return entry[1] if entry is not None and entry[0]() is self else None

    def _local_error(self,error):
        allocation=self._local_allocation()
        if allocation is None:return error
        from anygeometry.local_approximation_field import validate_local_approximation_allocation
        from anygeometry.nurbs import _float_down
        _,limits=validate_local_approximation_allocation(self.geometry,allocation)
        return min(error,_float_down(min(value for _,value in limits)))

    def _query_curvilinear_chords(self, intervals, cancellation_check=None):
        self._require_construction()
        from anygeometry import (query_prepared_material_boundary_chords,
            validate_prepared_material_boundary_chords_binding)
        error = checked_geometry_approximation_error(self.geometry_approximation_error)
        if error is None:
            raise MeshError('NURBS approximation requires explicit geometry_approximation_error')
        intervals = tuple(intervals)
        expected_field=()
        if self._local_allocation() is not None:
            from anygeometry.local_approximation_field import validate_local_approximation_allocation
            expected_field,_=validate_local_approximation_allocation(self.geometry,self._local_allocation())
        result = self._curvilinear_state.get('chords')
        bound = (() if result is None else tuple(
            (int(item.edge.id), *item.parameter_range) for item in result.intervals))
        if result is None or bound != intervals or result.chord_error != error or result.local_field_fingerprint != expected_field:
            result = query_prepared_material_boundary_chords(
                self.geometry, self.collection,
                tuple((self.geometry.handle('edge', edge), lo, hi) for edge, lo, hi in intervals),
                expected_revision=self.geometry.revision, chord_error=error,
                cancellation_check=cancellation_check, local_approximation=self._local_allocation())
        validate_prepared_material_boundary_chords_binding(
            self.geometry, self.collection, result, expected_revision=self.geometry.revision,
            cancellation_check=cancellation_check)
        return result

    def _uses_pair_boundary_sampling(self):
        """Exact-family proposal hint; owner queries still qualify every input."""
        self._require_construction()
        from anygeometry.nurbs import NurbsSurface
        if type(self.region.support) is not NurbsSurface:
            return False
        try:
            from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
        except ModuleNotFoundError as exc:
            if exc.name != 'anygeometry.nurbs_intersection_curve':
                raise
            return False
        if not any(type(path.curve) is NurbsSurfaceIntersectionCurve
                   for loop in self.region.boundaries for path in loop):
            return False
        return True

    def _boundary_sampling_error(self, error):
        """Proposal slack only; final receipts retain construction-bound error."""
        if not self._uses_pair_boundary_sampling():
            return error
        return checked_geometry_approximation_error(error/4.)

    def _propose_pair_boundary_parameters(self, parameters_by_edge, error, cancellation_check=None):
        """Owner-generated discretization proposals, never chord certificates."""
        if not self._uses_pair_boundary_sampling():
            return parameters_by_edge
        try:
            from anygeometry.nurbs_surface_intersection._mesh_proposals import propose_c2_boundary_parameters
        except ModuleNotFoundError as exc:
            if exc.name != 'anygeometry.nurbs_surface_intersection._mesh_proposals':
                raise
            return parameters_by_edge
        from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
        self.validate(cancellation_check)
        proposed=dict(parameters_by_edge)
        for edge in sorted(parameters_by_edge):
            curve=self.geometry.edges[edge].curve
            if type(curve) is NurbsSurfaceIntersectionCurve:
                parameters=tuple(Fraction(float(t)) for t in parameters_by_edge[edge])
                proposed[edge]=propose_c2_boundary_parameters(curve,parameters,
                    chord_error=error,cancellation_check=cancellation_check)
        self.validate(cancellation_check)
        return proposed

    def prepare_curvilinear_seed(self, parameters_by_edge, cancellation_check=None, *, refine=True):
        """Get owner literal targets before allocating any new boundary nodes."""
        if not self.requires_curvilinear:
            return {}
        edges = {int(path.source_edge) for loop in self.region.boundaries for path in loop}
        if refine:
            from anygeometry import (query_prepared_material_boundary_sampling,
                validate_prepared_material_boundary_sampling_binding)
            error = checked_geometry_approximation_error(self.geometry_approximation_error)
            if error is None:
                raise MeshError('NURBS approximation requires explicit geometry_approximation_error')
            sampling_error = self._boundary_sampling_error(error)
            parameters_by_edge = self._propose_pair_boundary_parameters(
                parameters_by_edge,sampling_error,cancellation_check)
            sampling = query_prepared_material_boundary_sampling(
                self.geometry, self.collection,
                tuple((self.geometry.handle('edge', edge), tuple(Fraction(float(t)) for t in
                    parameters_by_edge[edge])) for edge in sorted(edges)),
                expected_revision=self.geometry.revision, chord_error=error,
                cancellation_check=cancellation_check,local_approximation=self._local_allocation(),
                error_measure="cell_reference", proposal_fraction=.25)
            validate_prepared_material_boundary_sampling_binding(
                self.geometry, self.collection, sampling, expected_revision=self.geometry.revision,
                cancellation_check=cancellation_check)
            parameters_by_edge = {int(handle.id): parameters
                                 for handle, parameters in sampling.edge_parameters}
            self._curvilinear_state['chords'] = sampling.chords
            self._curvilinear_state['sampling'] = sampling
        intervals = tuple((edge, Fraction(float(a)), Fraction(float(b)))
            for edge in sorted(edges) for a, b in zip(
                parameters_by_edge[edge][:-1], parameters_by_edge[edge][1:]))
        result = self._query_curvilinear_chords(intervals, cancellation_check)
        targets = {}
        for interval in result.intervals:
            edge = targets.setdefault(int(interval.edge.id), {})
            for parameter, xyz in zip(interval.parameter_range, interval.xyz):
                previous = edge.setdefault(float(parameter), tuple(xyz))
                if previous != tuple(xyz):
                    raise MeshError('owner source station has conflicting literal targets')
        self._curvilinear_state['chords'] = result
        return targets

    def prepare_curvilinear_chords(self, mesh, registry, cancellation_check=None):
        """Qualify changed intervals before consuming their canonical UV/XYZ."""
        if not self.requires_curvilinear:
            return None
        self._validate_boundary_source_owner(mesh, registry)
        intervals = []
        for edge in sorted({int(path.source_edge) for loop in self.region.boundaries for path in loop}):
            entries = registry.entries(edge)
            by_node = {int(entry.node_id): Fraction(float(entry.key.parameter))
                       for entry in entries if entry.node_id is not None}
            nodes = tuple(mesh.nodes_of_edge.get(edge, ()))
            if (len(nodes) < 2 or len(by_node) != len(entries) or len(set(nodes)) != len(nodes)
                    or set(nodes) != set(by_node)):
                raise MeshError('NURBS boundary registry has incomplete station identities')
            parameters = tuple(by_node[node] for node in nodes)
            if parameters[0] != 0 or parameters[-1] != 1 or any(
                    a >= b for a, b in zip(parameters[:-1], parameters[1:])):
                raise MeshError('NURBS boundary stations do not follow their source edge')
            intervals.extend((edge, a, b) for a, b in zip(parameters[:-1], parameters[1:]))
        intervals = tuple(intervals)
        result = self._query_curvilinear_chords(intervals, cancellation_check)
        self._curvilinear_state['chords'] = result
        receipts = self.boundary_intervals(mesh, registry, cancellation_check)
        actual_intervals = tuple(sorted({(row.source_edge, *sorted(row.parameters)) for row in receipts}))
        if actual_intervals != intervals:
            raise MeshError('NURBS station intervals changed during owner qualification')
        targets = {(int(item.edge.id), parameter): tuple(xyz)
                   for item in result.intervals
                   for parameter, xyz in zip(item.parameter_range, item.xyz)}
        boundary_nodes = set()
        for row in receipts:
            for parameter, node in zip(row.parameters, row.nodes):
                boundary_nodes.add(node)
                if tuple(map(float, mesh.nodes[node])) != targets[(row.source_edge, parameter)]:
                    raise MeshError('registered NURBS source station differs from the owner literal target')
        self._curvilinear_state['boundary_nodes'] = frozenset(boundary_nodes)
        for node in boundary_nodes:
            self._curvilinear_state.get('uv_by_node', {}).pop(node, None)
        return result

    def curvilinear_station(self, edge_id, parameter, cancellation_check=None):
        """Query a new station by declared source parameter, never by projection."""
        if not self.requires_curvilinear or not isinstance(parameter, Fraction):
            raise MeshError('NURBS station requires its exact source parameter')
        if Fraction(float(parameter)) != parameter or not 0 < parameter < 1:
            raise MeshError('NURBS station must be exactly representable inside its edge')
        chords = self._curvilinear_state.get('chords')
        if chords is None:
            raise MeshError('NURBS station lacks initial owner boundary authority')
        existing = [(int(item.edge.id), *item.parameter_range) for item in chords.intervals]
        brackets = [(edge, a, b) for edge, a, b in existing
                    if edge == int(edge_id) and a < parameter < b]
        if len(brackets) != 1:
            raise MeshError('new NURBS station lacks one declared source interval')
        edge, a, b = brackets[0]
        intervals = tuple(sorted([item for item in existing if item != brackets[0]]
                                + [(edge, a, parameter), (edge, parameter, b)]))
        result = self._query_curvilinear_chords(intervals, cancellation_check)
        targets = []
        for interval in result.intervals:
            if interval.edge.id != edge_id or parameter not in interval.parameter_range:
                continue
            occurrence = [row for row in interval.occurrences if row.face.id == self.representative]
            if len(occurrence) != 1:
                raise MeshError('new NURBS station lacks a unique owner occurrence')
            index = interval.parameter_range.index(parameter)
            targets.append((tuple(occurrence[0].uv[index]), tuple(interval.xyz[index])))
        if not targets or any(row != targets[0] for row in targets[1:]):
            raise MeshError('new NURBS station has conflicting owner targets')
        return np.asarray(targets[0][0]), np.asarray(targets[0][1])

    def reference_uv(self, cancellation_check=None):
        """Require the owner's current regular anchor, never a sampled seed."""
        self._require_construction()
        from anygeometry import validate_prepared_material_boundary_chords_binding
        result = self._curvilinear_state.get('chords')
        if result is None:
            raise MeshError('NURBS metric chart lacks owner boundary authority')
        validate_prepared_material_boundary_chords_binding(
            self.geometry, self.collection, result, expected_revision=self.geometry.revision,
            cancellation_check=cancellation_check)
        if result.chord_error != self.geometry_approximation_error:
            raise MeshError('NURBS boundary authority binds another approximation limit')
        anchors = [uv for face, uv in result.reference_uv_by_face
                   if face == self.geometry.handle('face', self.representative)]
        if len(anchors) != 1:
            raise MeshError('NURBS metric chart lacks one certified regular anchor')
        return tuple(anchors[0])

    def initial_curvature_size_zones(self, requested, cancellation_check=None):
        """Refine-only proposals; explicit scalar and all final proofs unchanged."""
        self.validate(cancellation_check)
        from anygeometry._local_size_hints import curvature_size_hints
        from .refinement import Refinement
        import math
        error=checked_geometry_approximation_error(self.geometry_approximation_error)
        if error is None:return ()
        rows=curvature_size_hints(self.geometry,self.representative,error,
            cancellation_check=cancellation_check,local_approximation=self._local_allocation())
        zones=[]
        for box,size,curvature in rows:
            proposed=min(float(requested),float(size))
            if not math.isfinite(proposed) or proposed<=0 or proposed>=requested:continue
            centre=tuple(float((a+b)/2) for a,b in box)
            radius=math.sqrt(sum(max(abs(float(a)-c),abs(float(b)-c))**2
                                 for (a,b),c in zip(box,centre)))
            if math.isfinite(radius) and radius>0:
                zones.append(Refinement(size=proposed,radius=radius,center=centre,
                    name='source curvature proposal'))
        self.validate(cancellation_check)
        return tuple(zones)

    def _has_globally_affine_support(self, cancellation_check=None):
        """Use the owner's exact full-chart Hessian bound, never physical flatness."""
        from anygeometry import _local_curvature as curvature
        from anygeometry import nurbs
        from anygeometry.nurbs_surface_intersection import _bernstein as bernstein, _intervals as intervals
        from anygeometry.native_support_snapshots import capture_native_supports
        from anygeometry.nurbs_runtime_binding import validate_model_curve_runtimes
        from anygeometry.definition_binding import definition_checksum
        self.validate(None)
        support = self.region.support
        # This only selects a cheap sufficient family. It grants no affine
        # truth; even a planar bilinear chart can have a nonzero mixed Hessian.
        if (type(support) is not nurbs.NurbsSurface or support.degree_u != 1
                or support.degree_v != 1 or len(support.controls) != 2
                or any(len(row) != 2 for row in support.controls)
                or any(len(knots) != 4 or knots[0] != knots[1]
                       or knots[2] != knots[3] for knots in (support.knots_u, support.knots_v))
                or support.weights is not None and len({w for row in support.weights for w in row}) != 1):
            return False

        def calculators():
            from dataclasses import fields
            from types import ModuleType
            from anygeometry.predicates import IntersectionQualificationPolicy
            inventory = (
                (curvature, ('bound_surface_curvature','_bound','_jet','_planar','_divide',
                    '_cross','_square_upper','_square_lower','_pieces','bezier_piece',
                    'derivative_grid','grid_ranges','subbox','_sqrt_up','definition_checksum',
                    'charge_box','charge_split')),
                (bernstein, ('bezier_piece','derivative_grid','grid_ranges','subbox',
                    '_lerp','_blossom_axis_reference','_split_axis','_blossom_axis','_exact_refine')),
                (intervals.Interval, ('__init__','__add__','__sub__','__neg__','__mul__')),
                (nurbs.NurbsSurface, ('exact_net',)),
                (nurbs, ('_exact_refine','_exact_insert','_exact_span','_sqrt_up')),
                (nurbs.math, ('isqrt',)),
                (curvature.CurvatureTile, ('__init__',)),
                (curvature.SurfaceCurvatureBounds, ('__init__',)),
            )
            functions = tuple((id(module),name,_boundary_callable_pin(vars(module).get(name))
                               if type(module) is ModuleType or type(module) is type else None)
                              for module,names in inventory for name in names)
            helpers = tuple(_boundary_callable_pin(function) for function in
                            (_boundary_callable_pin, _boundary_literal_pin))
            policy_object = curvature.DEFAULT_INTERSECTION_QUALIFICATION_POLICY
            if type(policy_object) is not IntersectionQualificationPolicy:
                return None
            policy = _boundary_literal_pin(tuple((f.name,getattr(policy_object,f.name))
                                                 for f in fields(IntersectionQualificationPolicy)))
            if (any(value is None for _,_,value in functions)
                    or any(value is None for value in helpers) or policy is None):
                return None
            aliases = tuple(id(value) for value in (curvature.I, curvature.Q,
                bernstein.Interval, bernstein.Fraction, intervals.Interval, intervals.Fraction,
                nurbs.NurbsSurface, curvature.NurbsSurface, nurbs.math,
                curvature.CurvatureTile, curvature.SurfaceCurvatureBounds,
                curvature.LocalApproximationNotCertified))
            return functions,helpers,aliases,policy

        context = calculators()
        if context is None:
            return False  # Unknown initial calculator defaults cannot grant affine eligibility.
        native = capture_native_supports(self.geometry)
        signature = definition_checksum(support)

        def guard():
            current = calculators()
            if current is None or current != context:
                raise MeshError('affine support calculator changed during proposal')
            self.validate(None)
            validate_model_curve_runtimes(self.geometry)
            if (self.region.support is not support or definition_checksum(support) != signature
                    or capture_native_supports(self.geometry) != native):
                raise MeshError('affine support source changed during proposal')

        callback_errors = []
        def checked(phase):
            guard()
            try:
                _check_cancellation(cancellation_check, phase)
            except BaseException as error:
                callback_errors.append(error)
                raise
            guard()
            return False

        guard()
        try:
            bounds = curvature.bound_surface_curvature(support, cancellation_check=checked)
        except curvature.LocalApproximationNotCertified as error:
            if any(error is original for original in callback_errors):
                raise
            guard()
            return False
        guard()
        full = ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
        return (bounds.definition_checksum == signature and bounds.chart_box == full
                and len(bounds.tiles) == 1 and bounds.tiles[0].chart_box == full
                and bounds.tiles[0].hessian_norm_upper == 0
                and bounds.tiles[0].sigma_min_squared_lower > 0)

    def interior_size_proposal(self, requested, cancellation_check=None, *, _boundary_graded=False,
                               _with_affine=False, _with_graded=False):
        """Propose interior spacing consistent with the error-refined boundary.

        This is a mesher sizing heuristic, never a surface-error certificate.
        The final actual cells still require the unchanged geometry-owner proof.
        Exact shared boundary stations and the user's nominal maximum survive.
        """
        self._require_construction()
        if any(type(value) is not bool for value in (_boundary_graded,_with_affine,_with_graded)):
            raise MeshError('boundary graded proposal requires boolean options')
        def result(value, affine=False, graded=False):
            # Invocation-local proposal metadata, never geometry/proof authority.
            # The legacy pair's affinity meaning is deliberately unchanged.
            if _with_graded:
                return float(value), affine, graded
            return (float(value), affine) if _with_affine else value
        from anygeometry.nurbs import NurbsSurface
        if not self.requires_curvilinear or type(self.region.support) is not NurbsSurface:
            return result(requested)
        if isinstance(requested, (bool, np.bool_)) or not np.isfinite(requested) or requested <= 0:
            raise MeshError('NURBS interior size proposal requires a positive finite size')
        # This checks the live owner binding and approximation setting before
        # reading any literal chord target. A stale stored result grants nothing.
        # Capture the affine calculator context before its first external
        # callback. Legacy/ineligible paths retain their original callback law.
        self.reference_uv(None if _boundary_graded else cancellation_check)
        chords = self._curvilinear_state['chords']
        lengths = [float(np.linalg.norm(np.asarray(interval.xyz[1])-interval.xyz[0]))
                   for interval in chords.intervals
                   if any(occurrence.face.id in self.face_ids
                          for occurrence in interval.occurrences)]
        if not lengths or any(not np.isfinite(length) or length <= 0 for length in lengths):
            raise MeshError('NURBS interior size proposal lacks noncollapsed owner chords')
        if _boundary_graded and self._has_globally_affine_support(cancellation_check):
            # Boundary distances still resolve the protected chord scale. Only
            # the heuristic global chord ceiling is omitted for this exact map.
            self.reference_uv(None)
            if self._curvilinear_state.get('chords') is not chords:
                raise MeshError('NURBS affine spacing proposal receipt changed')
            return result(float(requested), True, True)
        if _boundary_graded and _with_graded:
            # Explicit canonical graded caller: chord resolution is local, not
            # a global interior requirement. Curvature/user controls and final
            # whole-cell owner error/physical gates still constrain recovery.
            self.reference_uv(cancellation_check)
            if self._curvilinear_state.get('chords') is not chords:
                raise MeshError('NURBS graded spacing proposal receipt changed')
            self.reference_uv(None)
            return result(float(requested), False, True)
        if _boundary_graded:
            self.reference_uv(cancellation_check)
        return result(min(float(requested), max(lengths)))

    def boundary_size_metric(self, spec, cancellation_check=None):
        """Compose refine-only physical spacing from current owner chords.

        A midpoint ball of half the chord length covers the entire protected
        segment. Its spacing grows using the existing metric gradation. This
        proposal changes neither stations nor any final quality/error gate.
        """
        self._require_construction()
        from anygeometry.nurbs import NurbsSurface
        from .metric import FeatureDistanceMetricControl
        from dataclasses import replace
        if not self.requires_curvilinear or type(self.region.support) is not NurbsSurface:
            return spec
        self.reference_uv(cancellation_check)
        chords = self._curvilinear_state['chords']
        groups = {}
        for interval in chords.intervals:
            if not any(occurrence.face.id in self.face_ids
                       for occurrence in interval.occurrences):
                continue
            first, second = np.asarray(interval.xyz, dtype=float)
            length = float(np.linalg.norm(second-first))
            if not np.isfinite(length) or length <= 0:
                raise MeshError('NURBS boundary size proposal lacks noncollapsed owner chords')
            target = min(spec.global_control.target_size, length)
            if target == spec.global_control.target_size:
                continue
            midpoint = tuple(first + (second-first)*.5)
            groups.setdefault((target, length*.5), []).append(midpoint)
        controls = tuple(FeatureDistanceMetricControl(
            tuple(points), target, radius, spec.maximum_gradation,
            name='owner chord spacing proposal')
            for (target, radius), points in groups.items())
        result = replace(spec, feature_controls=(*spec.feature_controls, *controls))
        # Requalify after literal conversions and any external callbacks. The
        # proposal never survives a replaced/stale receipt or source definition.
        self.reference_uv(cancellation_check)
        if self._curvilinear_state.get('chords') is not chords:
            raise MeshError('NURBS boundary size proposal receipt changed')
        self.reference_uv(None)
        return result

    def record_curvilinear_rows(self, rows, row_to_node):
        if not self.requires_curvilinear:
            return
        stored = self._curvilinear_state.setdefault('uv_by_node', {})
        for row, node in row_to_node.items():
            if node in self._curvilinear_state.get('boundary_nodes', ()):
                continue
            value = tuple(map(float, rows[int(row)]))
            previous = stored.setdefault(int(node), value)
            if previous != value:
                raise MeshError('NURBS source node changed its chart identity')

    def certify_curvilinear_published(self, mesh, registry, cancellation_check=None):
        """Requery final shared stations and prove actual published connectivity."""
        if not self.requires_curvilinear:
            return None
        from ._authored_component_stage import _mesh_digest
        entry_mesh = _mesh_digest(mesh)
        entry_registry = boundary_registry_pin(registry)
        entry_attached_registry = getattr(mesh, 'boundary_registry', None)
        element_ids = tuple(mesh.elements_of_face.get(self.representative, ()))
        if len(set(element_ids)) != len(element_ids) or any(
                (element in mesh.tris) == (element in mesh.quads) for element in element_ids):
            raise MeshError('NURBS publication has ambiguous cell identities')
        connectivity = tuple(tuple(mesh.tris.get(element, mesh.quads.get(element)))
                             for element in element_ids)
        def current():
            self._require_construction()
            if (_mesh_digest(mesh) != entry_mesh or boundary_registry_pin(registry) != entry_registry
                    or getattr(mesh, 'boundary_registry', None) is not entry_attached_registry):
                raise MeshError('NURBS publication mesh or stations changed during owner proof')
            self._validate_boundary_source_owner(mesh, registry)
        original_check = cancellation_check
        callback_error = None
        def checked(phase):
            nonlocal callback_error
            try:
                result = original_check(phase)
            except BaseException as error:
                callback_error = error
                raise
            current()
            return result
        from anygeometry import is_cancellation_signal
        # An owner-issued passive signal cannot run caller code or mutate
        # mesh inputs. Keep the original signal through the owner proof;
        # arbitrary callables retain every per-frontier mesh/registry check.
        cancellation_check = (original_check if is_cancellation_signal(original_check)
                              else checked if original_check is not None else None)
        from anygeometry import (CurvilinearMaterialCell,
            validate_prepared_material_curvilinear_cells)
        rows = {node: tuple(map(float, uv)) for node, uv in
                self._curvilinear_state.get('uv_by_node', {}).items()}
        chords = self.prepare_curvilinear_chords(mesh, registry, cancellation_check)
        current()
        receipts = self.boundary_intervals(mesh, registry, cancellation_check)
        sides = {}
        interval_indices = {(int(item.edge.id), tuple(item.parameter_range)): index
                            for index, item in enumerate(chords.intervals)}
        for receipt in receipts:
            index = interval_indices.get((receipt.source_edge, tuple(sorted(receipt.parameters))))
            if index is None:
                raise MeshError('NURBS final boundary interval has no owner receipt')
            a, b = receipt.nodes
            forward = receipt.parameters[0] < receipt.parameters[1]
            sides[(a, b)] = (index, forward)
            sides[(b, a)] = (index, not forward)
            for node, uv in zip(receipt.nodes, receipt.uv):
                rows[node] = uv
        cells = []
        for element, nodes in zip(element_ids, connectivity):
            _check_cancellation(cancellation_check, 'NURBS curvilinear published cells')
            if nodes is None or any(node not in rows for node in nodes):
                raise MeshError('NURBS published cell lost source chart correspondence')
            boundaries = tuple((side, *sides[(a, b)]) for side, (a, b) in enumerate(
                zip(nodes, (*nodes[1:], nodes[0]))) if (a, b) in sides)
            cells.append(CurvilinearMaterialCell(
                self.geometry.handle('face', self.representative), tuple(nodes),
                tuple(rows[node] for node in nodes),
                tuple(tuple(map(float, mesh.nodes[node])) for node in nodes), boundaries))
        cells = tuple(cells)
        if not cells:
            raise MeshError('NURBS publication has no cells')
        try:
            result = validate_prepared_material_curvilinear_cells(
                self.geometry, self.collection, chords, cells,
                expected_revision=self.geometry.revision,
                surface_error=self.geometry_approximation_error,
                previous=self._curvilinear_state.get('cells'),
                cancellation_check=cancellation_check, local_approximation=self._local_allocation())
        except Exception as error:
            if error is callback_error:
                raise
            current()
            local_ceiling=None
            if self._local_allocation() is not None:
                from anygeometry.prepared_material_curvilinear import _local_resolution_failure_limit
                from anygeometry import SourceCellRefinementRequired, SourceCellApproximationInconclusive
                if type(error) not in (SourceCellRefinementRequired, SourceCellApproximationInconclusive):
                    raise
                local_ceiling=_local_resolution_failure_limit(self.geometry,self._local_allocation(),error,cells,chords)
            if _is_source_cell_resolution_request(error, cells,
                    checked_geometry_approximation_error(self.geometry_approximation_error),
                    geometry=self.geometry, chords=chords, local_error_limit=local_ceiling):
                request = _SourceCellResolutionRequired(
                    'owner whole-cell proof requires finer source-cell resolution')
                matched = next(cell for cell in cells if cell.face.id == error.face_id and cell.node_ids == error.node_ids)
                request.cell_xyz = matched.xyz
                request.cell_node_ids = matched.node_ids
                request.face_id = error.face_id
                # Only a fresh owner boundary-discrepancy witness requires
                # shared station refinement. Incidence alone is not causality.
                request.boundary_failure_kind = getattr(error,"boundary_failure_kind","unresolved")
                request.boundary_intervals = matched.boundary_intervals
                raise request from error
            raise
        current()
        # Callbacks cannot alter the literal mesh after the owner examined it.
        for cell in cells:
            if tuple(tuple(map(float, mesh.nodes[node])) for node in cell.node_ids) != cell.xyz:
                raise MeshError('NURBS publication nodes changed during owner proof')
        if (tuple(mesh.elements_of_face.get(self.representative, ())) != element_ids
                or tuple(tuple(mesh.tris.get(element, mesh.quads.get(element, ())))
                         for element in element_ids) != connectivity):
            raise MeshError('NURBS publication cells changed during owner proof')
        from anygeometry import validate_prepared_material_curvilinear_cells_binding
        validate_prepared_material_curvilinear_cells_binding(
            self.geometry, self.collection, result, expected_revision=self.geometry.revision)
        self._curvilinear_state['cells'] = result
        self._curvilinear_state['published_elements'] = tuple(zip(element_ids, connectivity))
        return result

    def validate_curvilinear_publication(self, mesh):
        """No-callback final gate after association remapping and finalization."""
        if not self.requires_curvilinear:
            return
        from anygeometry import validate_prepared_material_curvilinear_cells_binding
        receipt = self._curvilinear_state.get('cells')
        chords = self._curvilinear_state.get('chords')
        if receipt is None or chords is None:
            raise MeshError('NURBS final publication lacks source-cell authority')
        published = self._curvilinear_state.get('published_elements', ())
        self._require_construction()
        if tuple(mesh.elements_of_face.get(self.representative, ())) != tuple(
                element for element, _nodes in published):
            raise MeshError('NURBS final face cell membership changed after source-cell proof')
        if (len(published) != len(receipt.cells)
                or len({element for element, _nodes in published}) != len(published)
                or any(cell.face != self.geometry.handle('face', self.representative)
                       or tuple(nodes) != cell.node_ids
                       for (_element, nodes), cell in zip(published, receipt.cells))):
            raise MeshError('NURBS final publication differs from owner-proved cells')
        for element, nodes in published:
            if ((element in mesh.tris) == (element in mesh.quads)
                    or (element in mesh.tris and len(nodes) != 3)
                    or (element in mesh.quads and len(nodes) != 4)):
                raise MeshError('NURBS final publication has ambiguous element kinds')
            if tuple(mesh.tris.get(element, mesh.quads.get(element, ()))) != nodes:
                raise MeshError('NURBS final connectivity changed after source-cell proof')
        for cell in receipt.cells:
            if tuple(tuple(map(float, mesh.nodes[node])) for node in cell.node_ids) != cell.xyz:
                raise MeshError('NURBS final XYZ changed after source-cell proof')
        if receipt.surface_error != self.geometry_approximation_error:
            raise MeshError('NURBS source cells bind another approximation limit')
        validate_prepared_material_curvilinear_cells_binding(
            self.geometry, self.collection, receipt,
            expected_revision=self.geometry.revision, cancellation_check=None)

    @property
    def face_ids(self):
        return frozenset(int(face.id) for face in self.region.faces)

    @property
    def representative(self):
        return min(self.face_ids)

    def validate(self, cancellation_check=None):
        self._require_construction()
        from anygeometry import validate_material_surface_regions_binding
        validate_material_surface_regions_binding(
            self.geometry, self.collection, cancellation_check=cancellation_check)

    def path_chain(self, path, mesh, registry):
        """Reuse source-edge station identities in the owner's path direction."""
        edge_id = path.source_edge
        if edge_id is None or edge_id not in mesh.nodes_of_edge:
            raise MeshError("material region path has no registered source edge")
        entries = registry.entries(edge_id)
        by_node = {int(entry.node_id): float(entry.key.parameter)
                   for entry in entries if entry.node_id is not None}
        nodes = list(mesh.nodes_of_edge[edge_id])
        if any(node not in by_node for node in nodes):
            raise MeshError("material region source station has no registry receipt")
        nodes.sort(key=by_node.__getitem__)
        tolerance = self.region.world_tolerance
        if self.requires_curvilinear:
            face = self.geometry.faces[self.representative]
            uses = [use for loop in (face.loop, *face.holes) for use in loop
                    if int(use.edge) == int(edge_id)]
            if len(uses) != 1:
                raise MeshError('NURBS boundary occurrence is not unique')
            forward = bool(uses[0].forward)
            reverse = not forward
        else:
            ends = np.asarray(path.curve.evaluate(np.asarray((0., 1.))), dtype=float)
            source_ends = self.geometry.sample_edge(edge_id, np.asarray((0., 1.)))
            forward = np.allclose(ends, source_ends, rtol=0., atol=tolerance)
            reverse = np.allclose(ends, source_ends[::-1], rtol=0., atol=tolerance)
        if forward == reverse:
            raise MeshError("material region path/source interval is ambiguous")
        if reverse:
            nodes.reverse()
        parameters = np.asarray([by_node[node] for node in nodes])
        path_parameters = parameters if forward else 1. - parameters
        # The owner domain supplies periodic branch conventions for each exact path.
        uv = None
        expected = None
        if self.requires_curvilinear:
            if self._curvilinear_state.get('chords') is None:
                raise MeshError('NURBS source chain requires owner literal station authority')
            chords = self._curvilinear_state['chords']
            from anygeometry import validate_prepared_material_boundary_chords_binding
            validate_prepared_material_boundary_chords_binding(
                self.geometry, self.collection, chords, expected_revision=self.geometry.revision)
            declared, literal_points = {}, {}
            for interval in chords.intervals:
                if interval.edge.id != edge_id:
                    continue
                occurrences = [occurrence for occurrence in interval.occurrences
                               if occurrence.face.id == self.representative]
                if len(occurrences) != 1:
                    raise MeshError('NURBS source interval lacks one bound chart occurrence')
                for parameter, row, point in zip(interval.parameter_range, occurrences[0].uv, interval.xyz):
                    key = Fraction(parameter)
                    old_point = literal_points.setdefault(key, tuple(point))
                    if old_point != tuple(point):
                        raise MeshError('NURBS owner supplied conflicting literal targets')
                    previous = declared.setdefault(key, tuple(row))
                    if previous != tuple(row):
                        raise MeshError('NURBS owner supplied conflicting canonical station rows')
            keys = [Fraction(float(parameter)) for parameter in parameters]
            if not all(key in declared for key in keys):
                raise MeshError('NURBS source station lacks current owner literal authority')
            uv = np.asarray([declared[key] for key in keys], dtype=float)
            expected = np.asarray([literal_points[key] for key in keys], dtype=float)
        else:
            uv = np.asarray([self.region.domain.uv(path.curve, float(t))
                             for t in path_parameters], dtype=float)
            expected = np.asarray(path.curve.evaluate(path_parameters), dtype=float)
        registered = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
        if (not np.array_equal(expected, registered) if self.requires_curvilinear else
                not np.allclose(expected, registered, rtol=0., atol=tolerance)):
            raise MeshError("material region path does not bind its registered stations")
        if uv.shape != (len(nodes), 2) or not np.isfinite(uv).all():
            raise MeshError("material region returned invalid station parameters")
        return nodes, uv, parameters

    def _validate_boundary_source_owner(self, mesh, registry):
        # Coincident coordinates and reusable local IDs cannot transfer receipts
        # between geometry owners, including independently loaded copies.
        view = registry.view
        if view.source is not self.geometry:
            raise MeshError('material boundary registry belongs to a different geometry owner')
        view.assert_current(self.geometry)
        mesh_owner = getattr(mesh, 'geometry_model_id', None)
        if mesh_owner is not None and str(mesh_owner) != str(self.geometry.model_id):
            raise MeshError('material boundary mesh belongs to a different geometry owner')

    def _boundary_source_chain(self, path, mesh, registry):
        self._validate_boundary_source_owner(mesh, registry)
        nodes, uv, parameters = self.path_chain(path, mesh, registry)
        entries = registry.entries(path.source_edge)
        selected = set(nodes)
        seen = set()
        for entry in entries:
            if entry.node_id not in selected:
                raise MeshError('material boundary chain omitted a registered source station')
            if entry.node_id in seen:
                raise MeshError('material boundary node has multiple source stations')
            seen.add(entry.node_id)
        if self.requires_curvilinear:
            # path_chain already checked exact owner receipt targets; never
            # reevaluate old algebraic endpoint representatives here.
            return nodes, uv, parameters
        source_points = np.asarray(self.geometry.sample_edge(path.source_edge, parameters), dtype=float)
        registered = np.asarray([mesh.nodes[node] for node in nodes], dtype=float)
        if (source_points.shape != registered.shape or not np.isfinite(source_points).all()
                or not np.allclose(source_points, registered, rtol=0., atol=self.region.world_tolerance)):
            raise MeshError('material boundary registry does not bind source parameters')
        return nodes, uv, parameters

    def _validate_boundary_owner(self, cancellation_check):
        if not any(region is self.region for region in self.collection.regions):
            raise MeshError('material boundary region is not its validated owner collection')
        self.validate(cancellation_check)

    def _prepare_boundary_source_chains(self, mesh, registry, cancellation_check=None):
        """Operation-local literal extraction; full entry/final owner gates remain.

        Unknown/custom/callback inputs retain the original fresh per-path route.
        No result or qualification survives this invocation.
        """
        # This closure is local to ONE callback-free native extraction. Public
        # path_chain/_boundary_source_chain keep their original full frontiers.
        extract = self._boundary_source_chain
        finish = None
        if cancellation_check is None and type(self) is MaterialRegionBinding:
            from .mesh import Mesh
            from .boundary import GlobalEdgeBoundaryRegistry,BoundaryEntry,BoundaryKey
            from .meshing_view import GeometryMeshingView
            from anygeometry.material_arrangement import ArrangementPath
            from anygeometry.identity import EntityHandle
            from uuid import UUID
            def native_owner(value,depth=0):
                if type(value) in (type(None),bool,int,float,str,bytes): return True
                if type(value) is UUID: return type(value.int) is int
                if type(value) is EntityHandle:
                    return (type(value.model_id) is UUID and type(value.model_id.int) is int
                        and type(value.kind) is str and type(value.id) is int)
                return (depth<8 and type(value) is tuple
                    and all(native_owner(item,depth+1) for item in value))
            def native_entry(entry):
                return (type(entry) is BoundaryEntry and type(entry.key) is BoundaryKey
                    and type(entry.key.edge_id) is int and type(entry.key.parameter) is float
                    and type(entry.node_id) in (int,type(None))
                    and type(entry.point) is np.ndarray and entry.point.dtype==np.dtype(float)
                    and entry.point.shape==(3,) and type(entry.owners) is tuple
                    and all(native_owner(item) for item in entry.owners))
            def native_registry():
                return (type(registry._entries) is dict and type(registry._by_edge) is dict
                    and all(type(key) is BoundaryKey and type(key.edge_id) is int
                        and type(key.parameter) is float and native_entry(entry)
                        for key,entry in registry._entries.items())
                    and all(type(edge) is int and type(entries) is dict
                        and all(type(parameter) is float and native_entry(entry)
                            for parameter,entry in entries.items())
                        for edge,entries in registry._by_edge.items()))
            native = (type(mesh) is Mesh and type(registry) is GlobalEdgeBoundaryRegistry
                and type(registry.view) is GeometryMeshingView
                and getattr(registry.entries,'__func__',None) is GlobalEdgeBoundaryRegistry.entries
                and getattr(registry.view.assert_current,'__func__',None) is GeometryMeshingView.assert_current
                and type(mesh.nodes) is dict and type(mesh.nodes_of_edge) is dict
                and all(type(node) in (int,np.int64) for node in mesh.nodes)
                and all(type(edge) is int for edge in mesh.nodes_of_edge)
                and type(self.region.boundaries) in (tuple,list)
                and all(type(loop) in (tuple,list) for loop in self.region.boundaries)
                and type(mesh.geometry_model_id) in (str,type(self.geometry.model_id),type(None))
                and type(self._curvilinear_state) is dict
                and all(type(path) is ArrangementPath and type(path.source_edge) is int
                        for loop in self.region.boundaries for path in loop)
                and all(type(nodes) in (list,tuple) and all(type(node) in (int,np.int64) for node in nodes)
                        for nodes in mesh.nodes_of_edge.values())
                and all(type(point) in (tuple,list,np.ndarray)
                        and (type(point) is not np.ndarray or
                             point.dtype == np.dtype(float) and point.shape == (3,))
                        and all(type(value) in (int,float,np.float64,np.int64) for value in point)
                        for point in mesh.nodes.values()))
            if (native and native_registry() and self.requires_curvilinear
                    and _boundary_methods_current()):
                import anygeometry as owner
                from anygeometry import validate_prepared_material_boundary_chords_binding
                pin_function, pin_literal = _boundary_callable_pin, _boundary_literal_pin
                helper_context = tuple((f.__code__, f.__defaults__, f.__kwdefaults__)
                    for f in (pin_function, pin_literal))
                def runtime():
                    if (_boundary_callable_pin is not pin_function
                            or _boundary_literal_pin is not pin_literal
                            or any(f.__code__ is not code or f.__defaults__ is not defaults
                                   or f.__kwdefaults__ is not keywords
                                for f,(code,defaults,keywords) in zip(
                                    (pin_function,pin_literal),helper_context))):
                        return (None,)
                    return tuple(pin_function(value) for value in (
                        *(vars(MaterialRegionBinding).get(name) for name in _BOUNDARY_METHOD_NAMES),
                        boundary_registry_pin, vars(GlobalEdgeBoundaryRegistry).get('entries'),
                        vars(GeometryMeshingView).get('assert_current'),
                        vars(owner).get('validate_prepared_material_boundary_chords_binding'),
                        vars(np).get('asarray'), vars(np).get('array_equal'), vars(np).get('isfinite')))
                entry_runtime = runtime()
                if any(pin is None for pin in entry_runtime):
                    return extract, finish
                def mesh_state():
                    return (tuple((edge, tuple(mesh.nodes_of_edge[edge]))
                        for edge in sorted(mesh.nodes_of_edge)),
                        tuple((int(node), tuple(point)) for node, point in sorted(mesh.nodes.items())))
                mesh_entry = mesh_state()
                self._validate_boundary_source_owner(mesh,registry)
                registry_entry = boundary_registry_pin(registry)
                chords = self._curvilinear_state.get('chords')
                if chords is None:
                    raise MeshError('NURBS source chain requires owner literal station authority')
                validate_prepared_material_boundary_chords_binding(
                    self.geometry,self.collection,chords,expected_revision=self.geometry.revision)
                literal = {}
                boundary_edges={int(path.source_edge) for loop in self.region.boundaries for path in loop}
                for interval in chords.intervals:
                    if int(interval.edge.id) not in boundary_edges:
                        continue
                    occurrences=[row for row in interval.occurrences if row.face.id==self.representative]
                    if len(occurrences)!=1:
                        raise MeshError('NURBS source interval lacks one bound chart occurrence')
                    declared,targets=literal.setdefault(int(interval.edge.id),({},{}))
                    for parameter,row,point in zip(interval.parameter_range,occurrences[0].uv,interval.xyz):
                        key=Fraction(parameter); row=tuple(map(float,row)); point=tuple(map(float,point))
                        if targets.setdefault(key,point)!=point:
                            raise MeshError('NURBS owner supplied conflicting literal targets')
                        if declared.setdefault(key,row)!=row:
                            raise MeshError('NURBS owner supplied conflicting canonical station rows')
                directions={}
                face=self.geometry.faces[self.representative]
                for paths in self.region.boundaries:
                    for path in paths:
                        edge=path.source_edge
                        uses=[use for loop in (face.loop,*face.holes) for use in loop if int(use.edge)==int(edge)]
                        if len(uses)!=1:
                            raise MeshError('NURBS boundary occurrence is not unique')
                        directions[edge]=bool(uses[0].forward)
                def extract(path,mesh,registry):
                    edge=path.source_edge
                    if edge is None or edge not in mesh.nodes_of_edge:
                        raise MeshError('material region path has no registered source edge')
                    entries=registry.entries(edge)
                    by_node={int(entry.node_id):float(entry.key.parameter) for entry in entries if entry.node_id is not None}
                    nodes=list(mesh.nodes_of_edge[edge])
                    if any(node not in by_node for node in nodes):
                        raise MeshError('material region source station has no registry receipt')
                    nodes.sort(key=by_node.__getitem__)
                    if not directions[edge]: nodes.reverse()
                    selected=set(nodes); seen=set()
                    for entry in entries:
                        if entry.node_id not in selected:
                            raise MeshError('material boundary chain omitted a registered source station')
                        if entry.node_id in seen:
                            raise MeshError('material boundary node has multiple source stations')
                        seen.add(entry.node_id)
                    parameters=np.asarray([by_node[node] for node in nodes])
                    declared,targets=literal.get(edge,({},{}))
                    keys=[Fraction(float(value)) for value in parameters]
                    if not all(key in declared for key in keys):
                        raise MeshError('NURBS source station lacks current owner literal authority')
                    uv=np.asarray([declared[key] for key in keys],dtype=float)
                    expected=np.asarray([targets[key] for key in keys],dtype=float)
                    registered=np.asarray([mesh.nodes[node] for node in nodes],dtype=float)
                    if not np.array_equal(expected,registered):
                        raise MeshError('material region path does not bind its registered stations')
                    if uv.shape!=(len(nodes),2) or not np.isfinite(uv).all():
                        raise MeshError('material region returned invalid station parameters')
                    return nodes,uv,parameters
                def finish():
                    if runtime() != entry_runtime:
                        raise MeshError('NURBS boundary extraction runtime changed')
                    if mesh_state() != mesh_entry:
                        raise MeshError('NURBS boundary extraction mesh changed')
                    if self._curvilinear_state.get('chords') is not chords:
                        raise MeshError('NURBS boundary chord receipt replaced during extraction')
                    validate_prepared_material_boundary_chords_binding(
                        self.geometry,self.collection,chords,expected_revision=self.geometry.revision)
                    if self._curvilinear_state.get('chords') is not chords:
                        raise MeshError('NURBS boundary chord receipt replaced during extraction')
                    if boundary_registry_pin(registry)!=registry_entry:
                        raise MeshError('NURBS boundary registry changed during extraction')
                    if runtime() != entry_runtime or mesh_state() != mesh_entry:
                        raise MeshError('NURBS boundary extraction context changed')
        return extract, finish

    def boundary_intervals(self, mesh, registry, cancellation_check=None):
        """Retain exact registered stations even on protected curved paths.

        Eligibility is deliberately absent. A caller must separately qualify its
        owner, curve-error, resource and whole-component publication contracts.
        """
        self._validate_boundary_owner(cancellation_check)
        extract, finish = self._prepare_boundary_source_chains(
            mesh, registry, cancellation_check)
        result = []
        for loop_index, paths in enumerate(self.region.boundaries):
            for path_index, path in enumerate(paths):
                _check_cancellation(cancellation_check, 'material boundary source receipts')
                nodes, uv, parameters = extract(path, mesh, registry)
                if (len(nodes) < 2 or len(set(nodes)) != len(nodes)
                        or any(isinstance(node, (bool, np.bool_))
                               or not isinstance(node, Integral) or node < 1 for node in nodes)):
                    raise MeshError('material boundary has ambiguous station identities')
                differences = np.diff(parameters)
                if not (np.all(differences > 0.) or np.all(differences < 0.)):
                    raise MeshError('material boundary has ambiguous source intervals')
                for index in range(len(nodes)-1):
                    pair = tuple(nodes[index:index+2])
                    result.append(BoundaryIntervalReceipt(
                        self, loop_index, path_index, index, int(path.source_edge), pair,
                        tuple(Fraction(float(t)) for t in parameters[index:index+2]),
                        tuple(tuple(map(float, row)) for row in uv[index:index+2]),
                        tuple(tuple(map(float, mesh.nodes[node])) for node in pair)))
        self._validate_boundary_owner(cancellation_check)
        # No caller callback follows these final source/chain checks. Cache by owner
        # occurrence so a long seeded boundary is checked once, not per interval.
        self._validate_boundary_source_owner(mesh, registry)
        if finish is not None:
            finish()
        chains = {}
        for receipt in result:
            occurrence = (receipt.loop_index, receipt.path_index)
            if occurrence not in chains:
                path = self.region.boundaries[receipt.loop_index][receipt.path_index]
                chains[occurrence] = extract(path, mesh, registry)
            self._assert_boundary_receipt_current(receipt, mesh, chains[occurrence])
        return tuple(result)

    @staticmethod
    def _assert_boundary_receipt_current(receipt, mesh, chain):
        nodes, uv, parameters = chain
        index = receipt.interval_index
        current_nodes = tuple(nodes[index:index+2])
        current_parameters = tuple(Fraction(float(t)) for t in parameters[index:index+2])
        if (current_nodes != receipt.nodes or current_parameters != receipt.parameters
                or tuple(tuple(map(float, row)) for row in uv[index:index+2]) != receipt.uv
                or tuple(tuple(map(float, mesh.nodes[node])) for node in current_nodes) != receipt.xyz):
            raise MeshError('material boundary station receipt is no longer current')

    def boundary_station(self, receipt, mesh, registry, parameter, cancellation_check=None):
        """Evaluate a still-bound exact source station in its owner occurrence.

        This read-only query grants no refinement permission and registers no
        station. Nonrepresentable rational parameters fail instead of silently
        changing their shared identity to a nearby binary64 value.
        """
        if not isinstance(receipt, BoundaryIntervalReceipt) or receipt.binding is not self:
            raise MeshError('material boundary station requires its source receipt')
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               or value < 0 for value in
               (receipt.loop_index, receipt.path_index, receipt.interval_index)):
            raise MeshError('material boundary source occurrence is unavailable')
        if not isinstance(parameter, Fraction):
            raise MeshError('material boundary station requires an exact rational parameter')
        lower, upper = sorted(receipt.parameters)
        if not lower < parameter < upper:
            raise MeshError('material boundary station must be inside its original interval')
        value = float(parameter)
        if not np.isfinite(value) or Fraction(value) != parameter:
            raise MeshError('material boundary station is not exactly representable')
        self._validate_boundary_owner(cancellation_check)
        # Reconstruct only the declared occurrence; no coordinate identity search.
        try:
            path = self.region.boundaries[receipt.loop_index][receipt.path_index]
        except (IndexError, TypeError):
            raise MeshError('material boundary source occurrence is unavailable') from None
        if (receipt.loop_index < 0 or receipt.path_index < 0 or receipt.interval_index < 0
                or path.source_edge != receipt.source_edge):
            raise MeshError('material boundary source occurrence changed')
        self._assert_boundary_receipt_current(
            receipt, mesh, self._boundary_source_chain(path, mesh, registry))
        if self.requires_curvilinear:
            row, point = self.curvilinear_station(receipt.source_edge, parameter, cancellation_check)
            self._validate_boundary_owner(cancellation_check)
            self._assert_boundary_receipt_current(
                receipt, mesh, self._boundary_source_chain(path, mesh, registry))
            return row.copy(), point.copy()
        path_parameter = value if receipt.parameters[0] < receipt.parameters[1] else 1.-value
        row = np.asarray(self.region.domain.uv(path.curve, path_parameter), dtype=float)
        point = np.asarray(self.geometry.sample_edge(receipt.source_edge, np.asarray((value,))), dtype=float)
        expected = np.asarray(path.curve.evaluate(path_parameter), dtype=float)
        if (row.shape != (2,) or point.shape != (1, 3) or expected.shape != (3,)
                or not np.isfinite(row).all() or not np.isfinite(point).all()
                or not np.isfinite(expected).all()
                or not np.allclose(point[0], expected, rtol=0., atol=self.region.world_tolerance)):
            raise MeshError('material boundary station lost its exact owner correspondence')
        from anygeometry import evaluate_material_surface_region
        owner_point = evaluate_material_surface_region(
            self.geometry, self.collection, self.representative, row[None, :],
            require_material=True, cancellation_check=cancellation_check)
        if not np.allclose(owner_point[0], point[0], rtol=0., atol=self.region.world_tolerance):
            raise MeshError('material boundary station does not bind its owner support')
        self._validate_boundary_owner(cancellation_check)
        self._assert_boundary_receipt_current(
            receipt, mesh, self._boundary_source_chain(path, mesh, registry))
        return row.copy(), point[0].copy()

    def loops(self, mesh, registry, splittable):
        result = []
        for paths in self.region.boundaries:
            nodes, rows, specifications = [], [], []
            previous = None
            for path in paths:
                chain, uv, parameters = self.path_chain(path, mesh, registry)
                if previous is not None and previous != chain[0]:
                    raise MeshError("material region boundary lost source vertex identity")
                nodes.extend(chain[:-1]); rows.extend(uv[:-1])
                specifications.extend(
                    (int(path.source_edge), float(min(a, b)), float(max(a, b)))
                    if int(path.source_edge) in splittable else None
                    for a, b in zip(parameters[:-1], parameters[1:]))
                previous = chain[-1]
            if not nodes or previous != nodes[0] or len(set(nodes)) != len(nodes):
                raise MeshError("material region has an ambiguous station cycle")
            result.append((tuple(nodes), np.asarray(rows), tuple(specifications)))
        return tuple(result)

    def input_provenance(self, mesh, registry, loops, pinned_ids, eligibility_context,
                         emitted_inputs,
                         cancellation_check=None):
        """Bind pre-native input identities without granting split permission."""
        from ._material_input_provenance import make_material_input_provenance

        return make_material_input_provenance(
            self, mesh, registry, loops, pinned_ids, eligibility_context,
            emitted_inputs,
            cancellation_check)

    def interior(self, mesh, registry, *, boundary_rows=None):
        segments, pinned = [], {}
        if boundary_rows is None:
            boundary_rows = {node: row for nodes, uv, _ in self.loops(mesh, registry, ())
                             for node, row in zip(nodes, uv)}
        canonical = {int(node): np.asarray(row, dtype=float).copy()
                     for node, row in boundary_rows.items()}
        boundary_nodes = set(canonical)

        def receipt(node, row):
            # Canonicalize only by an already known registry node identity.
            # Keep the submitted boundary/first interior row bit-for-bit.
            previous = canonical.setdefault(int(node), np.asarray(row, dtype=float).copy())
            if not np.allclose(previous, row, rtol=0., atol=2e-14):
                raise MeshError("material constraint has ambiguous chart correspondence")
            return previous

        for path in self.region.interior_constraints:
            nodes, uv, _ = self.path_chain(path, mesh, registry)
            uv = np.asarray([receipt(node, row) for node, row in zip(nodes, uv)])
            segments.extend(np.asarray((a, b)) for a, b in zip(uv[:-1], uv[1:]))
            for node, row in zip(nodes, uv):
                if node not in boundary_nodes:
                    pinned[node] = row
        for handle in self.region.retained_vertices:
            node = mesh.node_of_vertex.get(handle.id)
            if node is None:
                node = max(mesh.nodes, default=0) + 1
                mesh.node_of_vertex[handle.id] = node
                mesh.nodes[node] = np.asarray(self.geometry.vertex_position(handle.id), dtype=float)
            if node not in boundary_nodes:
                if node in canonical:
                    # A physical source path already supplied this vertex's
                    # owner-qualified branch coordinate. Reprojection is both
                    # redundant and may choose a different inverse branch.
                    pinned[node] = canonical[node]
                else:
                    row = self.region.support.local_uv_many([mesh.nodes[node]])[0]
                    pinned[node] = receipt(node, row)
        ids = tuple(sorted(pinned))
        return tuple(segments), ids, np.asarray([pinned[node] for node in ids]).reshape(-1, 2)

    def validate_constraints(self, core, mesh, registry, row_to_node, cancellation_check=None):
        """Require each protected source-station interval in active connectivity."""
        node_to_row = {node: row for row, node in row_to_node.items()}
        incidence = {}
        active_rows = set()
        for connectivity, activity in ((core.triangle_connectivity, core.triangle_active),
                                       (core.quad_connectivity, core.quad_active)):
            for index, cell in enumerate(connectivity[activity]):
                if index % 256 == 0:
                    _check_cancellation(cancellation_check, 'material region constraint incidence')
                active_rows.update(map(int, cell))
                for first, second in zip(cell, np.roll(cell, -1)):
                    key = tuple(sorted((int(first), int(second))))
                    incidence[key] = incidence.get(key, 0) + 1
        for path in self.region.interior_constraints:
            _check_cancellation(cancellation_check, 'material region constraint stations')
            nodes, _, _ = self.path_chain(path, mesh, registry)
            for first, second in zip(nodes[:-1], nodes[1:]):
                if first not in node_to_row or second not in node_to_row:
                    raise MeshError('material constraint lost a registered source station')
                key = tuple(sorted((node_to_row[first], node_to_row[second])))
                if incidence.get(key) != 2:
                    raise MeshError('material constraint station interval was lost or split without owner identity')
        for handle in self.region.retained_vertices:
            _check_cancellation(cancellation_check, 'material region retained vertices')
            node = mesh.node_of_vertex.get(handle.id)
            if node not in node_to_row or node_to_row[node] not in active_rows:
                raise MeshError('material region retained vertex is absent from active connectivity')

    def certify_published(self, mesh, registry, settings, cancellation_check=None):
        """Check the final detached region after other faces complete shared splits."""
        from .core import MeshCore
        from ._analytic_metric_chart import AnalyticMetricChart
        quality_settings = settings
        current = None
        if self.requires_curvilinear:
            from ._authored_component_stage import _mesh_digest
            from anygeometry import to_dict
            from .surface_mesh import SurfaceMeshOptions
            if type(settings) is not SurfaceMeshOptions:
                raise MeshError('NURBS physical quality requires canonical surface settings')
            quality_fields = ('min_scaled_jacobian', 'max_aspect_ratio', 'min_angle',
                              'max_angle', 'max_warpage', 'max_element_growth')
            def quality_pin():
                try:
                    values = tuple(float(getattr(settings, name)) for name in quality_fields)
                except (TypeError, ValueError, OverflowError) as error:
                    raise MeshError('NURBS physical quality controls require finite scalars') from error
                if not all(np.isfinite(value) for value in values):
                    raise MeshError('NURBS physical quality controls require finite scalars')
                return values
            entry_quality = quality_pin()
            quality_settings = replace(settings, **dict(zip(quality_fields, entry_quality)))
            entry_mesh = _mesh_digest(mesh)
            entry_registry = boundary_registry_pin(registry)
            entry_attached = getattr(mesh, 'boundary_registry', None)
            entry_geometry = to_dict(self.geometry)
            def current():
                self._require_construction()
                if (quality_pin() != entry_quality or _mesh_digest(mesh) != entry_mesh
                        or boundary_registry_pin(registry) != entry_registry
                        or getattr(mesh, 'boundary_registry', None) is not entry_attached
                        or to_dict(self.geometry) != entry_geometry):
                    raise MeshError('NURBS physical quality inputs changed during publication proof')
            original_check = cancellation_check
            if original_check is not None:
                def checked(phase):
                    result = original_check(phase)
                    current()
                    return result
                cancellation_check = checked
        self.validate(cancellation_check)
        elements = mesh.elements_of_face.get(self.representative, ())
        triangles, quads, used = [], [], set()
        for index, element in enumerate(elements):
            if index % 256 == 0:
                _check_cancellation(cancellation_check, 'material region published cells')
            if element in mesh.tris:
                cell = mesh.tris[element]
                triangles.append(cell)
            elif element in mesh.quads:
                cell = mesh.quads[element]
                quads.append(cell)
            else:
                raise MeshError('material region publication refers to a missing cell')
            used.update(cell)
        nodes = sorted(used)
        local = {node: row for row, node in enumerate(nodes)}
        core = MeshCore(np.asarray([mesh.nodes[node] for node in nodes]),
                        np.asarray([[local[node] for node in cell] for cell in triangles], dtype=int).reshape(-1, 3),
                        np.asarray([[local[node] for node in cell] for cell in quads], dtype=int).reshape(-1, 4))
        self.validate_constraints(core, mesh, registry, {row: node for node, row in local.items()}, cancellation_check)
        chart = AnalyticMetricChart(self.geometry, self.representative, cancellation_check,
                                    region_binding=self)
        self.certify_curvilinear_published(mesh, registry, cancellation_check)
        report = chart.certify_physical_core(core, quality_settings)
        if current is not None:
            current()
        return report


def prepare_material_regions(geometry, native_faces, source_to_final_faces, *,
                             order, native_options, supplied_seeding=False,
                             edge_overrides=(), cancellation_check=None,
                             geometry_approximation_error=None):
    """Select unions only when publication has one exact authored owner."""
    geometry_approximation_error = checked_geometry_approximation_error(geometry_approximation_error)
    if order != "linear" or native_options.point_placement != "frontal_delaunay" or supplied_seeding:
        return {}
    try:
        from anygeometry import ExtrudedSurface, query_material_surface_regions, MaterialSurfaceRegions
        from anygeometry.nurbs import NurbsSurface
    except ImportError:
        return {}
    selected = tuple(face for face in native_faces
                     if (face_requires_curvilinear(geometry, face)
                         or (isinstance(geometry.faces[face].surface, ExtrudedSurface)
                             and geometry.faces[face].parameterization is None)))
    if not selected:
        return {}
    preimages = {}
    for authored, descendants in source_to_final_faces.items():
        for face in descendants:
            preimages.setdefault(int(face), set()).add(int(authored))
    collection = query_material_surface_regions(
        geometry, selected, expected_revision=geometry.revision,
        cancellation_check=cancellation_check)
    from .approximation_policy import _ACTIVE_LOCAL_FIELD
    active=_ACTIVE_LOCAL_FIELD.get()
    allocation=None
    if active is not None:
        from anygeometry.local_approximation_field import bind_local_approximation_field
        allocation=bind_local_approximation_field(active[0],geometry,active[1],collection,cancellation_check=cancellation_check)
    bindings = {}
    for region in collection.regions:
        faces = frozenset(int(handle.id) for handle in region.faces)
        nurbs = region_requires_curvilinear(region)
        if not faces.issubset(selected):
            continue
        if nurbs:
            if len(faces) != 1 or region.cancelled_seams:
                raise MeshError('NURBS material route requires a singleton source region')
            if geometry_approximation_error is None:
                raise MeshError('NURBS approximation requires explicit geometry_approximation_error')
        elif len(faces) < 2 or not region.cancelled_seams:
            continue
        owners = {frozenset(preimages.get(face, ())) for face in faces}
        if len(owners) != 1 or len(next(iter(owners))) != 1:
            continue
        if any(handle.id in edge_overrides for handle in region.cancelled_seams):
            continue
        if region.source_attachments:
            raise MeshError("coalesced material region attachments require a qualified consumer")
        scoped = MaterialSurfaceRegions(replace(collection.source, charts=region.sources), (region,))
        binding = MaterialRegionBinding(geometry, scoped, region, next(iter(next(iter(owners)))),
                                        geometry_approximation_error=geometry_approximation_error if nurbs else None)
        if allocation is not None:
            key=id(binding)
            _LOCAL_ALLOCATION_BINDINGS[key]=(_local_binding_ref(binding,
                lambda reference,key=key:_LOCAL_ALLOCATION_BINDINGS.pop(key,None)),allocation)
        binding.validate(cancellation_check)
        for face in faces:
            bindings[face] = binding
    return bindings


def registered_core_rows(core, receipts):
    """Map unchanged submitted chart rows, without discovering geometry identity."""
    lookup = {}
    for row, point in enumerate(core.node_coordinates[:, :2]):
        lookup.setdefault(tuple(map(float, point)), []).append(row)
    result = {}
    node_rows = {}
    active_rows = set(map(int, core.triangle_connectivity[core.triangle_active].ravel()))
    active_rows.update(map(int, core.quad_connectivity[core.quad_active].ravel()))
    for point, node in receipts:
        matches = lookup.get(tuple(map(float, point)), ())
        if len(matches) != 1:
            raise MeshError("material region registered input was moved, lost or merged")
        row = matches[0]
        if row not in active_rows:
            raise MeshError('material region registered input is absent from active connectivity')
        previous = result.setdefault(row, int(node))
        if previous != int(node):
            raise MeshError("material region registered inputs have conflicting source identities")
        previous_row = node_rows.setdefault(int(node), row)
        if previous_row != row:
            raise MeshError("material region source station has ambiguous chart rows")
    return result


def _pair_sampling_order(bindings):
    """Seed simpler new-pair boundaries first; preserve every other position."""
    result=list(bindings)
    positions=[index for index,binding in enumerate(result) if binding._uses_pair_boundary_sampling()]
    ordered=sorted((result[index] for index in positions),key=lambda binding:
        (sum(len(loop) for loop in binding.region.boundaries),binding.representative))
    for index,binding in zip(positions,ordered): result[index]=binding
    return tuple(result)


def _pair_sampling_parameters(binding,own,previous):
    """Monotone station proposals only; no cached owner receipt is transferred."""
    if not binding._uses_pair_boundary_sampling():
        return own
    return {edge:tuple(sorted(set(parameters)|set(previous.get(edge,()))))
            for edge,parameters in own.items()}


def _station_balance_splits(geometry, owner_stations, maximum_gradation):
    """Numerical refine-only proposals from authenticated chord endpoints.

    Adjacency uses source edge/vertex identity, never coordinate proximity.
    No ratio here certifies cell growth or changes an approximation allowance.
    """
    import math
    if type(maximum_gradation) not in (int,float) or not math.isfinite(maximum_gradation) or maximum_gradation <= 1:
        raise MeshError('boundary station gradation must be finite and exceed one')
    threshold = Fraction(float(maximum_gradation))**2
    vertices, pairs = {}, []
    for edge, targets in sorted(owner_stations.items()):
        parameters = tuple(sorted(targets))
        intervals = []
        for a,b in zip(parameters,parameters[1:]):
            squared = sum((Fraction(float(y))-Fraction(float(x)))**2
                for x,y in zip(targets[a],targets[b]))
            if squared <= 0:
                raise MeshError('boundary station balancing requires noncollapsed owner chords')
            intervals.append((int(edge),a,b,squared))
        pairs.extend(zip(intervals,intervals[1:]))
        if intervals:
            source = geometry.edges[edge]
            vertices.setdefault(source.start,[]).append(intervals[0])
            vertices.setdefault(source.end,[]).append(intervals[-1])
    from itertools import combinations
    for rows in vertices.values(): pairs.extend(combinations(rows,2))
    splits = set()
    for first,second in pairs:
        longer,shorter = (first,second) if first[3] >= second[3] else (second,first)
        if longer[3] > threshold*shorter[3]:
            edge,a,b,_ = longer
            midpoint = float((Fraction(float(a))+Fraction(float(b)))/2)
            if not a < midpoint < b:
                raise MeshError('boundary station balancing has no representable midpoint')
            splits.add((edge,midpoint))
    return tuple(sorted(splits))


def _prepare_curvilinear_station_union(geometry,bindings,seeding,size_field,cancellation_check=None,*,maximum_gradation=None):
    """Authenticate a complete shared plan before allocating any boundary IDs."""
    from .hybrid import _station_parameters
    from contextlib import nullcontext
    from anygeometry.prepared_material_curvilinear import prepared_material_boundary_sampling_operation
    bindings=tuple(bindings)
    scope = (nullcontext(None) if maximum_gradation is None else
             prepared_material_boundary_sampling_operation())
    with scope as depth_limit:
        shared_parameters={}
        for binding in _pair_sampling_order(bindings):
            if not binding.requires_curvilinear:
                continue
            own={int(path.source_edge):_station_parameters(geometry,int(path.source_edge),
                int(seeding[int(path.source_edge)]),size_field)
                for loop in binding.region.boundaries for path in loop}
            requested=_pair_sampling_parameters(binding,own,shared_parameters)
            for edge,targets in binding.prepare_curvilinear_seed(requested,cancellation_check).items():
                shared_parameters.setdefault(edge,set()).update(targets)
        for round_number in range(1 if maximum_gradation is None else depth_limit+1):
            owner_stations={}
            for binding in bindings:
                for edge,targets in binding.prepare_curvilinear_seed(
                        {edge:tuple(sorted(values)) for edge,values in shared_parameters.items()},
                        cancellation_check,refine=False).items():
                    previous=owner_stations.setdefault(edge,targets)
                    if previous != targets:
                        raise MeshError('incident NURBS faces disagree on owner source stations')
            if maximum_gradation is None:
                return owner_stations
            splits=_station_balance_splits(geometry,owner_stations,maximum_gradation)
            if not splits:
                for binding in bindings: binding.validate(None)
                return owner_stations
            if round_number == depth_limit:
                raise MeshError('boundary station balancing qualification depth exhausted')
            for edge,parameter in splits: shared_parameters[edge].add(parameter)
            # Every numerical split is processed by fresh owner sampling under
            # this ONE operation allowance, before any shared registry exists.
            for binding in _pair_sampling_order(bindings):
                requested={edge:tuple(sorted(values)) for edge,values in shared_parameters.items()}
                for edge,targets in binding.prepare_curvilinear_seed(requested,cancellation_check).items():
                    shared_parameters.setdefault(edge,set()).update(targets)
        raise MeshError('boundary station balancing did not complete')


# No user formatting/equality runs while fingerprinting callable defaults.
# Unknown objects make the numerical extraction ineligible, never a stable pin.
def _boundary_literal_pin(value, depth=0):
    if depth > 16:
        return None
    kind = type(value)
    if kind is type(None): return ('none',)
    if kind is bool: return ('bool', value)
    if kind is int: return ('int', value)
    if kind is str: return ('str', value)
    if kind is bytes: return ('bytes', value)
    if kind is float: return ('float', value.hex())
    if kind in (tuple, list):
        parts = tuple(_boundary_literal_pin(item, depth+1) for item in value)
        return None if any(item is None for item in parts) else (
            'tuple' if kind is tuple else 'list', parts)
    if kind is dict:
        parts = tuple((_boundary_literal_pin(key, depth+1),
                       _boundary_literal_pin(item, depth+1)) for key,item in value.items())
        return None if any(a is None or b is None for a,b in parts) else ('dict', parts)
    return None


def _boundary_callable_pin(value):
    from types import FunctionType, MethodType, BuiltinFunctionType
    if type(value) is MethodType:
        value = value.__func__
    if type(value) is FunctionType:
        defaults = _boundary_literal_pin(value.__defaults__)
        keywords = _boundary_literal_pin(value.__kwdefaults__)
        if defaults is None or keywords is None:
            return None
        return ('python', id(value), id(value.__code__), defaults, keywords)
    if type(value) in (BuiltinFunctionType, np.ufunc):
        return ('native', id(value), id(type(value)))
    if type(value) is _BOUNDARY_ARRAY_DISPATCH_TYPE:
        wrapped = object.__getattribute__(value, '__wrapped__')
        pin = _boundary_callable_pin(wrapped)
        return None if pin is None else ('array_dispatch', id(value), pin)
    return None


_BOUNDARY_ARRAY_DISPATCH_TYPE = type(np.array_equal)
_BOUNDARY_METHOD_NAMES = ('_prepare_boundary_source_chains', '_boundary_source_chain',
    'path_chain', '_validate_boundary_source_owner', 'interior')
_BOUNDARY_METHOD_PINS = tuple(_boundary_callable_pin(vars(MaterialRegionBinding)[name])
    for name in _BOUNDARY_METHOD_NAMES)
_CANONICAL_MATERIAL_EVALUATOR_PIN = _boundary_callable_pin(_CANONICAL_MATERIAL_EVALUATE)


def _boundary_methods_current():
    current = tuple(_boundary_callable_pin(vars(MaterialRegionBinding).get(name))
        for name in _BOUNDARY_METHOD_NAMES)
    return all(pin is not None for pin in current) and current == _BOUNDARY_METHOD_PINS


# This finite trusted driver is captured independently of surface scheduling
# declarations. Caller-set replacement declarations do not renew admission.
from .native_v2 import frontal_delaunay_refine as _MATERIAL_FRONTAL_DRIVER
_MATERIAL_FRONTAL_DRIVER_CODE = _MATERIAL_FRONTAL_DRIVER.__code__
_MATERIAL_FRONTAL_DRIVER_PIN = _boundary_callable_pin(_MATERIAL_FRONTAL_DRIVER)


def _material_frontal_driver_pin():
    from types import FunctionType
    from . import surface_mesh
    value=vars(surface_mesh).get('frontal_delaunay_refine')
    if (value is not _MATERIAL_FRONTAL_DRIVER or type(value) is not FunctionType
            or value.__code__ is not _MATERIAL_FRONTAL_DRIVER_CODE):return None
    pin=_boundary_callable_pin(value)
    return pin if pin is not None and pin==_MATERIAL_FRONTAL_DRIVER_PIN else None


def _material_native_dispatch_guard():
    state=_MATERIAL_READ_SCOPE.get()
    record=_MATERIAL_SNAPSHOT_READS.get(id(state))
    if record is not None:_material_binding_snapshot(record['binding'])

_MATERIAL_DRIVER_CHECK = _material_frontal_driver_pin
_MATERIAL_DRIVER_CHECK_CODE = _material_frontal_driver_pin.__code__
