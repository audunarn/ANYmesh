"""Automatic approximation policy, separate from numerical geometry tolerances."""
from dataclasses import dataclass, field
from functools import wraps
from contextvars import ContextVar
_ACTIVE_LOCAL_FIELD = ContextVar("anymesher_local_field", default=None)
import math
from numbers import Real
from .errors import MeshError

def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MeshError(f'{name} must be a finite positive real')
    try:
        value = float(value)
    except (ValueError, OverflowError):
        raise MeshError(f'{name} must be a finite positive real') from None
    if not math.isfinite(value) or value <= 0:
        raise MeshError(f'{name} must be a finite positive real')
    return value

@dataclass(frozen=True, slots=True)
class GeometryApproximationPolicy:
    """Relative to selected trim-loop enclosures, not curvature or clearance.

    The prospective default is 0.1 percent of the smallest loop enclosure span.
    A cap and the existing owner's numerical policy floor are absolute lengths.
    """
    relative_accuracy: float = 1e-3
    maximum_error: float | None = None
    adaptation: str = "local"

    def __post_init__(self):
        if self.adaptation not in ('local','scale'):
            raise MeshError('adaptation must be local or scale')
        object.__setattr__(self, 'relative_accuracy', _positive(self.relative_accuracy, 'relative_accuracy'))
        if self.maximum_error is not None:
            object.__setattr__(self, 'maximum_error', _positive(self.maximum_error, 'maximum_error'))

    def to_dict(self):
        return dict(relative_accuracy=self.relative_accuracy, maximum_error=self.maximum_error, adaptation=self.adaptation)

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value)-{'relative_accuracy', 'maximum_error', 'adaptation'}:
            raise MeshError('invalid automatic geometry approximation policy settings')
        return cls(**value)

@dataclass(frozen=True, slots=True)
class GeometryApproximationResolution:
    effective_error: float | None
    mode: str
    face_ids: tuple
    relative_accuracy: float | None
    reference_scale: float | None
    maximum_error: float | None
    policy_floor: float | None
    floor_limited: bool
    cap_limited: bool
    source: object
    _scale: object = field(repr=False, compare=False, default=None)
    adaptation: str = "scale"
    clearance_face_ids: tuple = ()
    connect_intent: bool = False
    _field: object = field(repr=False, compare=False, default=None)

    def to_dict(self):
        from anygeometry.local_approximation_field import local_approximation_field_fingerprint
        local_identity=None if self._field is None else local_approximation_field_fingerprint(self._field)
        return dict(local_field=local_identity,connect_intent=self.connect_intent,mode=self.mode, adaptation=self.adaptation, effective_error=self.effective_error,
            face_ids=list(self.face_ids), clearance_face_ids=list(self.clearance_face_ids), relative_accuracy=self.relative_accuracy,
            reference_scale=self.reference_scale, maximum_error=self.maximum_error,
            policy_floor=self.policy_floor, floor_limited=self.floor_limited,
            cap_limited=self.cap_limited,
            scale_definition=None if self._scale is None else self._scale.scale_definition,
            source=dict(model_id=str(self.source.model_id), revision=self.source.revision,
                        source_checksum=self.source.source_checksum))

def _automatic(policy, scale):
    relative = policy.relative_accuracy * scale.reference_scale
    if not math.isfinite(relative) or relative <= 0 or not math.isfinite(scale.policy_floor):
        raise MeshError('automatic approximation resolution is nonfinite or underflowed')
    if policy.maximum_error is not None and policy.maximum_error < scale.policy_floor:
        raise MeshError('automatic approximation maximum_error conflicts with owner policy floor')
    effective = max(relative, scale.policy_floor)
    capped = policy.maximum_error is not None and policy.maximum_error < effective
    if capped:
        effective = policy.maximum_error
    return GeometryApproximationResolution(effective, 'automatic', scale.face_ids,
        policy.relative_accuracy, scale.reference_scale, policy.maximum_error,
        scale.policy_floor, relative < scale.policy_floor, capped, scale.source, scale)

def resolve_geometry_approximation(geometry, faces=None, *, geometry_approximation_error=None,
                                   policy=None, cancellation_check=None, clearance_faces=None, connect_intent=False):
    """Resolve on original selected source faces before generation/qualification.

    Explicit scalar preserves legacy behavior; scalar plus policy is ambiguous.
    Automatic resolution only measures selected curvilinear faces. Uncurved-only
    requests have mode ``not_applicable`` and effective_error None.
    """
    if type(connect_intent) is not bool:raise MeshError('CONNECT intent must be an explicit boolean')
    import anygeometry as owner
    from ._material_region_binding import face_requires_curvilinear, checked_geometry_approximation_error
    if geometry_approximation_error is not None and policy is not None:
        raise MeshError('geometry_approximation_error and automatic policy are mutually exclusive')
    scalar = checked_geometry_approximation_error(geometry_approximation_error)
    if type(policy) is dict:
        policy = GeometryApproximationPolicy.from_dict(policy)
    if policy is not None and type(policy) is not GeometryApproximationPolicy:
        raise MeshError('policy must be GeometryApproximationPolicy or its settings dictionary')
    from .hybrid import _face_ids
    selected = _face_ids(geometry, faces)
    clearance = selected if clearance_faces is None else _face_ids(geometry,clearance_faces)
    if not set(selected)<=set(clearance):raise MeshError("clearance context must contain generation faces")
    source = owner.query_approximation_source_binding(geometry, cancellation_check=cancellation_check)
    if scalar is not None:
        result = GeometryApproximationResolution(scalar, 'explicit', selected,
            None, None, None, None, False, False, source)
    else:
        curved = tuple(face for face in selected if face_requires_curvilinear(geometry, face))
        if curved:
            scale = owner.query_trim_enclosure_scale(geometry, curved, cancellation_check=cancellation_check)
            if scale.source != source:
                raise MeshError('source changed during automatic approximation resolution')
            from dataclasses import replace
            chosen=policy or GeometryApproximationPolicy()
            result = _automatic(chosen, scale)
            if chosen.adaptation=='local':
                from anygeometry.local_approximation_field import query_local_approximation_field
                local=query_local_approximation_field(geometry,clearance,relative_accuracy=chosen.relative_accuracy,
                    scale_error=result.effective_error,policy_floor=scale.policy_floor,cancellation_check=cancellation_check,
                    connect_intent=connect_intent)
                result=replace(result,adaptation='local',clearance_face_ids=clearance,connect_intent=connect_intent,_field=local)
        else:
            result = GeometryApproximationResolution(None, 'not_applicable', selected,
                None, None, None, None, False, False, source)
    owner.validate_approximation_source_binding(geometry, source, cancellation_check=cancellation_check)
    return result

def validate_geometry_approximation_binding(geometry, resolution, *, cancellation_check=None):
    import anygeometry as owner
    if type(resolution) is not GeometryApproximationResolution:
        raise MeshError('invalid geometry approximation resolution')
    owner.validate_approximation_source_binding(geometry, resolution.source,
                                              cancellation_check=cancellation_check)
    if resolution.mode == 'automatic':
        owner.validate_trim_enclosure_scale_binding(geometry, resolution._scale,
                                                   cancellation_check=cancellation_check)
        expected = _automatic(GeometryApproximationPolicy(resolution.relative_accuracy,
                              resolution.maximum_error), resolution._scale)
        from dataclasses import replace
        if resolution.adaptation=='local':
            from anygeometry.local_approximation_field import validate_local_approximation_field
            validate_local_approximation_field(geometry,resolution._field,cancellation_check=cancellation_check)
            from fractions import Fraction
            if (resolution._field.relative_accuracy!=Fraction(resolution.relative_accuracy)
                    or resolution._field.scale_error!=Fraction(resolution.effective_error)
                    or resolution._field.policy_floor!=Fraction(resolution.policy_floor)):
                raise MeshError('local field policy changed from resolution')
            if (not set(resolution.face_ids)<=set(resolution.clearance_face_ids)
                    or resolution.clearance_face_ids!=resolution._field.face_ids):
                raise MeshError('local clearance scope changed')
            if type(resolution.connect_intent) is not bool or resolution.connect_intent!=resolution._field.contact_intent:
                raise MeshError('local structural contact intent changed')
            expected=replace(expected,adaptation='local',clearance_face_ids=resolution.clearance_face_ids,
                connect_intent=resolution.connect_intent,_field=resolution._field)
        elif resolution.adaptation!='scale' or resolution._field is not None or resolution.clearance_face_ids or resolution.connect_intent:
            raise MeshError('invalid approximation adaptation context')
        if expected != resolution:
            raise MeshError('geometry approximation resolution changed')
    elif resolution.mode == 'explicit':
        if resolution.connect_intent is not False:raise MeshError('fixed approximation has no conditional contact intent')
        _positive(resolution.effective_error, 'geometry_approximation_error')
    elif resolution.mode != 'not_applicable' or resolution.effective_error is not None or resolution.connect_intent is not False:
        raise MeshError('invalid geometry approximation resolution mode')

def _resolve_public_approximation(function):
    """Bind automatic resolution across every return and user callback."""
    @wraps(function)
    def wrapped(geometry, **options):
        policy = options.pop('geometry_approximation_policy', None)
        supplied=options.pop('geometry_approximation_resolution',None)
        original=options.pop('geometry_approximation_source',None)
        if supplied is not None:
            if original is None or policy is not None:
                raise MeshError('live approximation resolution requires original source and no new policy')
            validate_geometry_approximation_binding(original,supplied,cancellation_check=options.get('cancellation_check'))
            if options.get('geometry_approximation_error') != supplied.effective_error:
                raise MeshError('resolved approximation scalar does not match live resolution')
            token=_ACTIVE_LOCAL_FIELD.set((original,supplied._field) if supplied._field is not None else None)
            try: result=function(geometry,**options)
            finally: _ACTIVE_LOCAL_FIELD.reset(token)
            validate_geometry_approximation_binding(original,supplied,cancellation_check=options.get('cancellation_check'))
            if hasattr(result,'extra_diagnostics'):result.extra_diagnostics['geometry_approximation']=supplied.to_dict()
            else:result.mesh.hybrid_diagnostics['geometry_approximation']=supplied.to_dict()
            return result
        if original is not None:raise MeshError('approximation source requires live resolution')
        scalar = options.get('geometry_approximation_error')
        if scalar is not None and policy is not None:
            raise MeshError('geometry_approximation_error and automatic policy are mutually exclusive')
        if scalar is not None:
            return function(geometry, **options)
        # Materialize once so generators select the same faces in both layers.
        faces = options.get('face_ids')
        if faces is not None:
            faces = tuple(faces)
            options['face_ids'] = faces
        from .hybrid import _face_ids
        from ._material_region_binding import face_requires_curvilinear
        selected = _face_ids(geometry, faces)
        if not any(face_requires_curvilinear(geometry, face) for face in selected):
            if policy is not None:
                if type(policy) is dict:
                    GeometryApproximationPolicy.from_dict(policy)
                elif type(policy) is not GeometryApproximationPolicy:
                    raise MeshError('policy must be GeometryApproximationPolicy or its settings dictionary')
            return function(geometry, **options)
        callback = options.get('cancellation_check')
        resolution = resolve_geometry_approximation(geometry, faces, policy=policy,
                                                    cancellation_check=callback)
        options['geometry_approximation_error'] = resolution.effective_error
        token=_ACTIVE_LOCAL_FIELD.set((geometry,resolution._field) if resolution._field is not None else None)
        try: result = function(geometry, **options)
        finally: _ACTIVE_LOCAL_FIELD.reset(token)
        validate_geometry_approximation_binding(geometry, resolution, cancellation_check=callback)
        if resolution.mode == 'automatic':
            diagnostics = resolution.to_dict()
            if hasattr(result, 'extra_diagnostics'):
                result.extra_diagnostics['geometry_approximation'] = diagnostics
            else:
                result.mesh.hybrid_diagnostics['geometry_approximation'] = diagnostics
        return result
    return wrapped
