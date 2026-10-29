"""The one place that says which request fields the NURBS tensor route honours.

Every public parameter of ``generate_hybrid_mesh_result`` is classified here. Option OBJECTS (native, quad,
structured, quality-policy, preparation options and anything nested in them) are never walked attribute by
attribute: the whole object must be EXACTLY the registered class (type identity, never a name or subclass) and structurally the
same as a default of the REAL class (no foreign ``__eq__`` is ever called; exact scalar types only), except for the scalar fields on
``SCALAR_ALLOWLIST``; the first differing dotted field path names the refusal. Parameter rules:

``('neutral', why)``   cannot change the mesh the tensor produces (selection, budgets, post-processing, cancellation);
``('equals', values)`` honoured only for these values;
``('empty',)``         must be None or empty;
``('none',)``          must be None;
``('stations', why)``  acts through the boundary stations, which are the tensor authority and are verified
                       (equal opposite counts and fractions) by the core, so it is honoured or refused there;
``('custom', name)``   checked by the gate itself (``native_options`` / ``quad_options`` field by field below,
                       ``quad_face_ids`` against the selected faces).

A parameter that is non-default and not honoured makes ``request_refusal`` return a message naming it; a
parameter that is not classified is refused too. Equality covers every option field, containers,
Optionals and unions included, so a future field can never be silently ignored.
"""
from __future__ import annotations

import dataclasses
import enum

PARAMETERS = {
    'target_size': ('neutral', 'uniform target; the stations decide the grid'),
    'strategy': ('neutral', 'selects the route, validated by the gate'),
    'overrides': ('stations', 'edge division overrides change stations'),
    'beam_edges': ('empty',),
    'beam_offsets': ('empty',),
    'member_ids': ('empty',),  # None selects all members: the gate separately refuses when any member exists
    'face_ids': ('neutral', 'selection'),
    'seeding': ('stations', 'seeding changes stations'),
    'refinements': ('empty',),
    'order': ('equals', ('linear',)),
    'recombine': ('equals', (True,)),
    'native_backend': ('neutral', 'backend of the native route, unused by the tensor'),
    'native_options': ('custom', 'whole-object equality'),
    'geometry_approximation_error': ('neutral', 'checked by the owner proof'),
    'geometry_approximation_policy': ('neutral', 'resolved before the gate'),
    'geometry_approximation_resolution': ('neutral', 'resolved before the gate'),
    'geometry_approximation_source': ('neutral', 'resolved before the gate'),
    'structured_options': ('none',),
    'structural_preparation': ('object',),
    'qualified_s3': ('neutral', 'post-mesh admission'),
    'quad_options': ('custom', 'whole-object equality'),
    'quad_t3_minimum_angle': ('neutral', 'applies to T3 cells; the tensor emits none'),
    'quad_t3_angle_action': ('neutral', 'applies to T3 cells; the tensor emits none'),
    'layout_policy': ('equals', (None, 'existing')),
    'quad_face_ids': ('custom', 'compared with the selected faces by the gate'),
    'overlap_policy': ('neutral', 'junction policy; the tensor admits no declared junction edges'),
    'mutation_policy': ('neutral', 'ownership of the input model'),
    'certification_mode': ('neutral', 'audit scope'),
    'change_set': ('neutral', 'audit'),
    'audit_policy': ('neutral', 'audit'),
    'cancellation_check': ('neutral', 'cooperative cancellation'),
    '_native_surface_options': ('object',),
    '_evaluate_declared_junction_alignment': ('neutral', 'junction edges are refused by the tensor'),
    '_refine_declared_junction_transition': ('equals', (False,)),
    '_defer_finalization': ('neutral', 'finalization timing'),
    '_respect_native_growth_limit': ('neutral', 'the tensor enforces the growth gate itself'),
    '_independent_beams': ('neutral', 'beams are refused'),
    '_nurbs_tensor': ('neutral', 'route selector'),
}

# Scalar fields of option classes that may differ from the default, with why. Anything else must be equal.
SCALAR_ALLOWLIST = {
    ('NativeMeshingOptions', 'point_placement'): 'the NURBS pipeline forces frontal_delaunay',
    ('NativeMeshingOptions', 'metric_mode'): 'the NURBS pipeline forces isotropic_spatial',
    ('NativeMeshingOptions', 'max_insertions'): 'insertion budget; the tensor inserts nothing',
    ('NativeMeshingOptions', 'max_topology_operations'): 'topology budget; the tensor performs none',
    ('NativeMeshingOptions', 'cancellation_interval'): 'cancellation cadence',
    ('QuadMeshingOptions', 'cancellation_interval'): 'cancellation cadence',
    ('StructuredMeshingOptions', 'preference'): 'structured layout planning only',
    ('StructuredMeshingOptions', 'allow_detached_partition'): 'structured layout planning only',
    ('StructuredMeshingOptions', 'max_element_growth'): 'honoured: growth gate of the tensor candidate',
    ('StructuredMeshingOptions', 'minimum_size_ratio'): 'structured layout planning only',
    ('StructuredMeshingOptions', 'maximum_size_ratio'): 'structured layout planning only',
    ('StructuredMeshingOptions', 'maximum_radial_sides'): 'structured layout budget',
    ('StructuredMeshingOptions', 'maximum_candidates_per_component'): 'structured layout budget',
    ('StructuredMeshingOptions', 'maximum_face_records'): 'structured layout budget',
    ('StructuredMeshingOptions', 'maximum_blocks'): 'structured layout budget',
    ('StructuredMeshingOptions', 'maximum_edge_records'): 'structured layout budget',
    ('StructuredMeshingOptions', 'maximum_estimated_elements'): 'structured layout budget',
    ('StructuredMeshingOptions', 'maximum_divisions_per_edge'): 'structured layout budget',
    ('MeshQualityPolicy', 'minimum_scaled_jacobian'): 'honoured: final gate of the tensor candidate',
    ('MeshQualityPolicy', 'maximum_aspect_ratio'): 'honoured: final gate of the tensor candidate',
    ('MeshQualityPolicy', 'minimum_angle'): 'honoured: final gate of the tensor candidate',
    ('MeshQualityPolicy', 'maximum_angle'): 'honoured: final gate of the tensor candidate',
    ('MeshQualityPolicy', 'maximum_warpage'): 'honoured: final gate of the tensor candidate',
    ('StructuralPreparationOptions', 'automatic_face_connections'): 'a declared connection becomes a junction edge, which the tensor refuses',
    ('StructuralPreparationOptions', 'automatic_member_connections'): 'members are refused',
    ('StructuralPreparationOptions', 'automatic_member_sheet_connections'): 'members are refused',
    ('StructuralPreparationOptions', 'declare_missing_owners'): 'ownership bookkeeping',
    ('StructuralPreparationOptions', 'maximum_candidate_pairs'): 'preparation budget',
    ('StructuralPreparationOptions', 'maximum_applications'): 'preparation budget',
    ('StructuralPreparationOptions', 'maximum_face_records'): 'preparation budget',
    ('StructuralPreparationOptions', 'maximum_edge_records'): 'preparation budget',
}

# Contract boundary (read with the module docstring): the checks below never evaluate anything a caller can override
# on a foreign object. Types are compared with ``is`` only (never ``in``, ``==`` or hashing on a foreign type, so a
# metaclass ``__eq__`` / ``__hash__`` / ``__instancecheck__`` is never called); ``type(x)`` is used instead of
# ``x.__class__`` and ``isinstance`` on foreign objects; fields come from the exact registered CLASS, never from the
# instance; an instance carrying a name that is not a declared field is refused; messages use the type name read from the
# class dictionary and never ``repr`` a foreign value. This protects against ordinary and accidental misuse and the
# hostile cases in the tests; it is not a security boundary against arbitrary Python object tampering (for example
# monkeypatching this module or the registered classes themselves).
_NoneType = type(None)


def _is_type(t, *classes):
    """``t`` is one of ``classes`` by identity (no ``in``: that would call ``==`` on a foreign type)."""
    for cls in classes:
        if t is cls:
            return True
    return False


def _scalar_type(t):
    return _is_type(t, bool, int, float, str, _NoneType)


def _type_name(value):
    """Name of ``type(value)`` without any attribute hook of a foreign metaclass."""
    try:
        return str(type.__dict__['__name__'].__get__(type(value)))
    except Exception:
        return '?'


def _registered_class(t):
    for cls in _registry_tuple():
        if t is cls:
            return cls
    return None


def _registry_tuple():
    from .native_v2 import NativeMeshingOptions
    from .quad.options import QuadMeshingOptions
    from .structured import MeshQualityPolicy, StructuredMeshingOptions
    from .preparation import StructuralPreparationOptions
    return (NativeMeshingOptions, QuadMeshingOptions, StructuredMeshingOptions, MeshQualityPolicy,
            StructuralPreparationOptions)


def _declared(cls):
    """Field names of an exact registered class, taken from the class (not from any instance)."""
    return tuple(f.name for f in dataclasses.fields(cls))


def _undeclared(value, cls):
    """Names stored on the instance that are not declared fields of its exact class (shadowing attempts)."""
    declared = _declared(cls)
    try:
        stored = object.__getattribute__(value, '__dict__')
    except AttributeError:
        return []
    return [k if _is_type(type(k), str) else '<non-str key>' for k in list(stored)
            if not (_is_type(type(k), str) and k in declared)]


def _reference(cls):
    """Default instance of the REAL registered class ``cls`` (an exact type), else None."""
    return cls() if _registered_class(cls) is not None else None


def _is_scalar(value, default=None):
    """Exact scalar type (no numpy scalars, no subclasses); an Enum member only of the default's own enum type."""
    if _scalar_type(type(value)):
        return True
    return isinstance(default, enum.Enum) and type(value) is type(default)


def _same(value, default):
    """Structural sameness with NO call to a foreign ``__eq__`` / hash / isinstance: exact types only, plain == only on
    exact scalars, identity for everything else that is not a registered option dataclass or a plain container."""
    t = type(value)
    if t is not type(default):
        return False
    if _scalar_type(t):
        return value == default or (value != value and default != default)
    if isinstance(default, enum.Enum):
        return value is default
    cls = _registered_class(t)
    if cls is not None:
        return not _undeclared(value, cls) and all(
            _same(getattr(value, name), getattr(default, name)) for name in _declared(cls))
    if _is_type(t, tuple, list):
        return len(value) == len(default) and all(_same(a, b) for a, b in zip(value, default))
    if _is_type(t, dict, set, frozenset):
        return len(value) == 0 and len(default) == 0
    return value is default


def _lookalike_extra(value):
    """Best-effort name of a field a look-alike class adds (for the message only; never decides acceptance)."""
    try:
        fields = type.__dict__['__dict__'].__get__(type(value)).get('__dataclass_fields__')
        mro = type.__dict__['__mro__'].__get__(type(value))
        base = next(c for c in _registry_tuple() for m in mro if m is c)
        known = _declared(base)
        return next((str(n) for n in fields if str(n) not in known), None)
    except Exception:
        return None


def _differences(value, default, path, owner, name):
    """Yield ``(path, class_name, field, value, default)`` for each leaf that is not the default. The dataclass recursion
    only builds the dotted path; sameness is ``_same``. A value that is not exactly the expected type is a leaf."""
    if _same(value, default):
        return
    cls = _registered_class(type(value))
    if cls is not None and type(default) is type(value):
        for extra in _undeclared(value, cls):
            yield (f'{path}.{extra}' if path else extra), None, None, value, default
            return
        for name_ in _declared(cls):
            yield from _differences(getattr(value, name_), getattr(default, name_),
                                    f'{path}.{name_}' if path else name_, cls.__name__, name_)
        return
    extra = _lookalike_extra(value) if cls is None else None
    if extra is not None:
        yield (f'{path}.{extra}' if path else extra), None, None, value, default
        return
    yield path, owner, name, value, default


def _option_refusal(prefix, obj, allow_bool=False):
    """First reason an option object differs from its REAL default beyond the scalar allowlist (dotted path)."""
    if obj is None or (allow_bool and type(obj) is bool):
        return None
    if type(obj) is dict and prefix in _MAPPING_CLASSES:
        # ANYfem passes the preparation options as a mapping of field names: read it as the real class.
        if not all(_is_type(type(k), str) for k in obj):
            return f'{prefix} has non-string keys'
        try:
            obj = _mapping_class(prefix)(**obj)
        except Exception as error:
            return f'{prefix} cannot be read as {_MAPPING_CLASSES[prefix]}: {error}'
    reference = _reference(type(obj))
    if reference is None:
        extra = _lookalike_extra(obj)
        where = f'{prefix}.{extra}' if extra else prefix
        return f'{where} is not an exact option object (got {_type_name(obj)}) for the tensor route'
    for path, cls_name, field, value, default in _differences(obj, reference, prefix, type(obj).__name__, None):
        if cls_name is not None and (cls_name, field) in SCALAR_ALLOWLIST and _is_scalar(value, default):
            continue
        return f'{path} differs from its default and is not honoured by the tensor route'
    return None


_MAPPING_CLASSES = {'structural_preparation': 'StructuralPreparationOptions'}


def _mapping_class(prefix):
    from .preparation import StructuralPreparationOptions
    return {'StructuralPreparationOptions': StructuralPreparationOptions}[_MAPPING_CLASSES[prefix]]


def _check(name, value, rule):
    kind = rule[0]
    if kind in ('neutral', 'stations', 'honoured'):
        return None
    if kind == 'equals':
        # exact scalar type and plain ==: a look-alike with a lax __eq__ never matches
        for allowed in rule[1]:
            if value is allowed or (type(value) is type(allowed) and _scalar_type(type(value)) and value == allowed):
                return None
        shown = repr(value) if _scalar_type(type(value)) else f'<{_type_name(value)}>'
        return f'{name}={shown} is not honoured (allowed: {", ".join(map(repr, rule[1]))})'
    if kind == 'none':
        return None if value is None else f'{name} is not honoured by the tensor route'
    if kind == 'empty':
        if value is None or (_is_type(type(value), tuple, list, dict, set, frozenset) and len(value) == 0):
            return None
        return f'{name} is not honoured by the tensor route'
    return f'{name} has an unusable rule {rule!r}'


def request_refusal(request, native_options, quad_options):
    """First reason the request cannot be honoured, naming the field; ``None`` when within the allowlist.

    ``request`` maps parameter names to the caller values (``layout_policy`` before the gate resolves it);
    ``native_options`` / ``quad_options`` are the coerced option objects (or None).
    """
    for name, value in request.items():
        if name == 'geometry':
            continue
        rule = PARAMETERS.get(name)
        if rule is None:
            return f'{name} is not classified for the tensor route'
        if rule[0] == 'custom':
            continue
        if rule[0] == 'object':
            reason = _option_refusal(name, value, allow_bool=name in _BOOL_PLACEHOLDERS)
        else:
            reason = _check(name, value, rule)
        if reason:
            return reason
    for prefix, options in (('native_options', native_options), ('quad_options', quad_options)):
        reason = _option_refusal(prefix, options)
        if reason:
            return reason
    return None


# Parameters whose bool form is a legitimate placeholder (structural_preparation=False means "none"). A structured
# option has no bool form: ``_native_surface_options=False`` is a typed refusal, not a downstream AttributeError.
_BOOL_PLACEHOLDERS = ('structural_preparation',)


def non_default_gate_options(options):
    """Dotted names and values of the honoured gate options (quality policy, growth) that differ from their default."""
    reference = _reference(type(options)) if options is not None else None
    if reference is None:
        return ''
    return ', '.join(f'{path}={value!r}' for path, cls_name, field, value, default
                     in _differences(options, reference, 'quality_options', type(options).__name__, None)
                     if cls_name is not None and (cls_name, field) in SCALAR_ALLOWLIST and _is_scalar(value, default))
