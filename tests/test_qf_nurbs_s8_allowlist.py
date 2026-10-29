"""Fixes-2: the tensor route honours an explicit ALLOWLIST of request fields; everything else non-default refuses."""
import dataclasses
import inspect

import pytest

import anymesher.hybrid as hybrid
from anygeometry import GeometryModel
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.metric import FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec
from anymesher.native_v2 import NativeMeshingOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher._parametric_tensor import nurbs_tensor_admission

from test_qf_nurbs_s0_characterisation import nurbs_patch
from test_qf_nurbs_s2_route import request
from test_qf_nurbs_s7_option_guards import refusal_reasons, tensor_used


def spec(global_size=.5, feature=None):
    features = () if feature is None else (FeatureDistanceMetricControl(((.5, .5, 0.),), .1, feature, 1.5, 'zone'),)
    return NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial',
                                metric_field=MetricFieldSpec(IsotropicMetricControl(global_size), features))


# ---- contract completeness: a new option can never be silently ignored ---------------------------------------
def test_every_request_parameter_and_option_field_is_classified():
    from anymesher import _tensor_contract as contract
    function = hybrid.generate_hybrid_mesh_result
    while hasattr(function, '__wrapped__'):
        function = function.__wrapped__
    parameters = set(inspect.signature(function).parameters) - {'geometry'}
    assert parameters <= set(contract.PARAMETERS), sorted(parameters - set(contract.PARAMETERS))
    assert all(rule[0] in ('neutral', 'equals', 'empty', 'none', 'stations', 'custom', 'object', 'honoured')
               for rule in contract.PARAMETERS.values())


def test_unclassified_field_is_refused_by_name():
    from anymesher import _tensor_contract as contract
    assert 'brand_new' in contract.request_refusal({'brand_new': 1}, NativeMeshingOptions(), QuadMeshingOptions())


# ---- findings 1-4 --------------------------------------------------------------------------------------------
@pytest.mark.parametrize('native', [spec(.05), spec(.5, feature=.01)], ids=['global_metric', 'feature_metric'])
def test_metric_field_is_refused_explicit_and_not_served_under_auto(native):
    model, face = nurbs_patch(warp=0.)
    with pytest.raises(QuadPublicUnsupported, match='metric_field'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
            **request(native_options=native))
    model, face = nurbs_patch(warp=0.)
    mesh = generate_hybrid_mesh_result(model, strategy='auto', **request(native_options=native)).mesh
    assert not tensor_used(mesh) and any('metric_field' in r for r in refusal_reasons(mesh))


def test_member_ids_and_implicit_members_are_refused():
    model, face = nurbs_patch()
    edge = model.faces[face].loop[0].edge
    member = model.add_member((edge,))
    with pytest.raises(QuadPublicUnsupported, match='member'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                    member_ids=(member,), **request())
    with pytest.raises(QuadPublicUnsupported, match='member'):  # implicit: member_ids omitted selects all members
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request())


def subdomain_face():
    """Bottom pcurve: degree 1, knots (0,0,.5,1,1), controls (0,0),(1,0),(1,1), domain (0,.5) -> the unit bottom edge."""
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    line = lambda a, b: NurbsCurve(1, (0., 0., 1., 1.), (a, b))
    sub = NurbsCurve(1, (0., 0., .5, 1., 1.), ((0., 0.), (1., 0.), (1., 1.)), domain=(0., .5))
    pcurves = (sub, line((1., 0.), (1., 1.)), line((1., 1.), (0., 1.)), line((0., 1.), (0., 0.)))
    curves = tuple(CurveOnSurface(support, p) for p in pcurves)
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], c) for i, c in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face


def test_valid_subdomain_pcurve_is_admitted():
    model, face = subdomain_face()
    assert nurbs_tensor_admission(model, face).eligible


def test_valid_subdomain_pcurve_meshes_with_explicit_quad_options():
    model, face = subdomain_face()
    mesh = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request()).mesh
    assert tensor_used(mesh)


def test_explicit_adaptive_layout_policy_is_refused_and_existing_is_accepted():
    model, face = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported, match='layout_policy'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                    layout_policy='adaptive', **request())
    model, face = nurbs_patch()
    mesh = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                       layout_policy='existing', **request()).mesh
    assert tensor_used(mesh)


# ---- guard strength: a narrow bulge BETWEEN the 129 samples must be caught by the exact check ----------------
def narrow_bulge_face(peak=.2):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    line = lambda a, b: NurbsCurve(1, (0., 0., 1., 1.), (a, b))
    bulge = NurbsCurve(1, (0., 0., .0004, .001, .0016, 1., 1.),
                       ((0., 0.), (.0004, 0.), (.0007, peak), (.001, 0.), (1., 0.)))
    pcurves = (bulge, line((1., 0.), (1., 1.)), line((1., 1.), (0., 1.)), line((0., 1.), (0., 0.)))
    curves = tuple(CurveOnSurface(support, p) for p in pcurves)
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], c) for i, c in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face


def test_bulge_between_samples_is_missed_by_sampling_but_caught_exactly():
    model, face = narrow_bulge_face()
    import numpy as np
    uv = np.asarray(model.face_boundary_uv(face, model.faces[face].loop[0].edge, np.linspace(0., 1., 129)))
    assert np.ptp(uv[:, 1]) < 1e-12  # the 129 samples really do miss the bulge (guard is the only defence)
    result = nurbs_tensor_admission(model, face)
    assert not result.eligible and result.code == 'not_iso_parametric_rectangle'


def test_empty_member_selection_is_accepted():
    """ANYfem passes member_ids=() for a model without members; that is not a request for members."""
    from anymesher import _tensor_contract as contract
    assert contract.request_refusal({'member_ids': ()}, None, None) is None
    assert contract.request_refusal({'member_ids': (3,)}, None, None) is not None


def weighted_subdomain_face(weights=(1., 3., 1.)):
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    line = lambda a, b: NurbsCurve(1, (0., 0., 1., 1.), (a, b))
    sub = NurbsCurve(1, (0., 0., .5, 1., 1.), ((0., 0.), (1., 0.), (1., 1.)), weights=weights, domain=(0., .5))
    pcurves = (sub, line((1., 0.), (1., 1.)), line((1., 1.), (0., 1.)), line((0., 1.), (0., 0.)))
    curves = tuple(CurveOnSurface(support, p) for p in pcurves)
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i], vertices[(i + 1) % 4], c) for i, c in enumerate(curves))
    face = model.add_face(edges, surface=support)
    model.add_sheet((face,))
    return model, face


def test_weighted_subdomain_pcurve_ends_in_a_fast_typed_refusal_or_a_valid_mesh():
    import time
    model, face = weighted_subdomain_face()
    t = time.time()
    try:
        mesh = generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request()).mesh
    except QuadPublicUnsupported:
        pass
    else:
        assert tensor_used(mesh) and len(mesh.tris) == 0
    assert time.time() - t < 60


# ---- fixes-3: nested option/policy dataclasses are classified recursively ------------------------------------
def with_extra_field(base, name='maximum_element_area', value=.001):
    return dataclasses.make_dataclass('Extended' + base.__name__, [(name, float, value)], bases=(base,), frozen=True)


def test_nested_quality_policy_field_added_in_memory_is_refused_by_name():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    extended = with_extra_field(MeshQualityPolicy)()
    options = StructuredMeshingOptions(quality_policy=extended)
    reason = contract.request_refusal({'_native_surface_options': options}, None, None)
    assert reason and 'maximum_element_area' in reason


def test_structured_options_field_added_in_memory_is_refused_by_name():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import StructuredMeshingOptions
    reason = contract.request_refusal({'_native_surface_options': with_extra_field(StructuredMeshingOptions)()}, None, None)
    assert reason and 'maximum_element_area' in reason


def test_structural_preparation_field_added_in_memory_is_refused_by_name():
    from anymesher import _tensor_contract as contract
    from anymesher.preparation import StructuralPreparationOptions
    reason = contract.request_refusal({'structural_preparation': with_extra_field(StructuralPreparationOptions)()}, None, None)
    assert reason and 'maximum_element_area' in reason
    assert contract.request_refusal({'structural_preparation': {'brand_new': 1}}, None, None)
    assert contract.request_refusal({'structural_preparation': {'maximum_applications': 5}}, None, None) is None
    assert contract.request_refusal({'structural_preparation': {}}, None, None) is None
    assert contract.request_refusal({'structural_preparation': StructuralPreparationOptions()}, None, None) is None
    assert contract.request_refusal({'structural_preparation': True}, None, None) is None


def test_default_nested_options_are_within_the_allowlist():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import StructuredMeshingOptions
    assert contract.request_refusal({'_native_surface_options': StructuredMeshingOptions()}, None, None) is None


def test_weighted_subdomain_pcurve_is_a_fast_typed_refusal():
    import time
    model, face = weighted_subdomain_face()
    assert nurbs_tensor_admission(model, face).code == 'weighted_pcurve'
    t = time.time()
    with pytest.raises(QuadPublicUnsupported, match='weighted_pcurve'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request())
    assert time.time() - t < 10


# ---- fixes-4: whole-object equality contract, scale-invariant weights, named quality refusal ---------------------
def _option_classes():
    from anymesher.preparation import StructuralPreparationOptions
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    return (NativeMeshingOptions, QuadMeshingOptions, StructuredMeshingOptions, MeshQualityPolicy,
            StructuralPreparationOptions)


def _slot(cls):
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    return {NativeMeshingOptions: 'native', QuadMeshingOptions: 'quad', StructuredMeshingOptions: 'surface',
            MeshQualityPolicy: 'policy'}.get(cls, 'preparation')


def _request_with(cls, instance):
    """Route ``instance`` of option class ``cls`` through the public parameter that carries it."""
    from anymesher import _tensor_contract as contract
    from anymesher.structured import StructuredMeshingOptions
    slot = _slot(cls)
    if slot == 'native':
        return contract.request_refusal({}, instance, None)
    if slot == 'quad':
        return contract.request_refusal({}, None, instance)
    if slot == 'surface':
        return contract.request_refusal({'_native_surface_options': instance}, None, None)
    if slot == 'policy':
        return contract.request_refusal({'_native_surface_options': StructuredMeshingOptions(quality_policy=instance)}, None, None)
    return contract.request_refusal({'structural_preparation': instance}, None, None)


def _extended(base, annotation, default):
    factory = default if callable(default) else (lambda value=default: value)
    field = dataclasses.field(default_factory=factory)
    return dataclasses.make_dataclass('Extended' + base.__name__, [('brand_new', annotation, field)], bases=(base,), frozen=True)


_SHAPES = {
    'list': (list, list), 'dict': (dict, dict), 'tuple': (tuple, tuple),
    'optional_none': ('Optional[float]', lambda: None), 'union_scalar': ('int | str', lambda: 3),
    'nested_dataclass': (object, lambda: __import__('anymesher.structured', fromlist=['x']).MeshQualityPolicy()),
}


@pytest.mark.parametrize('shape', sorted(_SHAPES))
@pytest.mark.parametrize('cls', _option_classes(), ids=lambda c: c.__name__)
def test_any_added_field_of_any_shape_is_refused_by_name_in_every_option_class(cls, shape):
    annotation, factory = _SHAPES[shape]
    instance = _extended(cls, annotation, factory)()
    reason = _request_with(cls, instance)
    assert reason and 'brand_new' in reason, (cls.__name__, shape, reason)


def test_the_equality_guard_is_what_refuses_added_fields(monkeypatch):
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy
    instance = _extended(MeshQualityPolicy, list, list)()
    assert _request_with(MeshQualityPolicy, instance)
    monkeypatch.setattr(contract, '_option_refusal', lambda *args, **kwargs: None)
    assert _request_with(MeshQualityPolicy, instance) is None  # without the guard the field slips through


def test_changed_non_allowlisted_values_are_refused_with_their_dotted_path():
    from anymesher import _tensor_contract as contract
    reason = contract.request_refusal({}, None, QuadMeshingOptions(max_front_iterations=7))
    assert reason and 'max_front_iterations' in reason
    native = NativeMeshingOptions()
    object.__setattr__(native, 'metric_field', object())  # set in memory: the constructor would validate it
    reason = contract.request_refusal({}, native, None)
    assert reason and 'metric_field' in reason


def test_allowlist_entries_name_real_scalar_fields_and_defaults_pass():
    from anymesher import _tensor_contract as contract
    real = {cls.__name__: {f.name for f in dataclasses.fields(cls)} for cls in _option_classes()}
    for cls_name, field in contract.SCALAR_ALLOWLIST:
        assert field in real[cls_name], (cls_name, field)
    for cls in _option_classes():
        assert _request_with(cls, cls()) is None


def test_non_scalar_value_on_an_allowlisted_field_is_refused():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    options = StructuredMeshingOptions()
    object.__setattr__(options, 'maximum_blocks', [1, 2])  # set in memory: the constructor would validate it
    reason = contract.request_refusal({'_native_surface_options': options}, None, None)
    assert reason and 'maximum_blocks' in reason
    options = StructuredMeshingOptions(quality_policy=MeshQualityPolicy(maximum_aspect_ratio=3.))
    assert contract.request_refusal({'_native_surface_options': options}, None, None) is None


def test_quality_gate_refusal_names_the_non_default_nested_field():
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    options = StructuredMeshingOptions(quality_policy=MeshQualityPolicy(maximum_aspect_ratio=1.001))
    model, _ = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported, match='quality_policy.maximum_aspect_ratio'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                    _native_surface_options=options, **request())


@pytest.mark.parametrize('scale', [1e-13, 1., 1e13])
def test_weighted_subdomain_refusal_is_scale_invariant(scale):
    import time
    weights = tuple(scale * w for w in (1., 3., 1.))
    model, face = weighted_subdomain_face(weights)
    assert nurbs_tensor_admission(model, face).code == 'weighted_pcurve'
    t = time.time()
    with pytest.raises(QuadPublicUnsupported, match='weighted_pcurve'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(), **request())
    assert time.time() - t < 10


@pytest.mark.parametrize('scale', [1e-13, 1e13])
def test_uniformly_scaled_weights_are_still_admitted(scale):
    model, face = weighted_subdomain_face((scale, scale, scale))
    assert nurbs_tensor_admission(model, face).eligible


def test_non_uniform_weight_outside_the_active_part_of_a_subdomain_curve_is_refused():
    model, face = weighted_subdomain_face((1., 1., 5.))
    assert nurbs_tensor_admission(model, face).code == 'weighted_pcurve'


# ---- fixes-5: exact-type contract; look-alikes, lax __eq__, numpy and callables are never trusted ---------------
class Lax:
    """Compares equal to anything and sizes as empty: the classic look-alike."""
    def __eq__(self, other):
        return True
    def __ne__(self, other):
        return False
    __hash__ = None
    def __len__(self):
        return 0
    def __iter__(self):
        return iter(())
    def __call__(self, *args, **kwargs):
        return None


def _lookalike(base, **new_defaults):
    """Same-named frozen subclass of ``base`` whose defaults differ (its own reference default would be accepted)."""
    fields = [(name, type(value), value) for name, value in new_defaults.items()]
    return dataclasses.make_dataclass(base.__name__, fields, bases=(base,), frozen=True)


def test_same_named_subclass_with_altered_defaults_is_refused_for_native_and_quad():
    from anymesher import _tensor_contract as contract
    for base, other in ((NativeMeshingOptions, 'native'), (QuadMeshingOptions, 'quad')):
        name = 'max_insertions' if other == 'native' else 'max_front_iterations'
        instance = _lookalike(base, **{name: getattr(base(), name) + 1})()
        reason = (contract.request_refusal({}, instance, None) if other == 'native'
                  else contract.request_refusal({}, None, instance))
        assert reason, other


def test_same_named_quad_subclass_is_refused_by_the_public_call_not_served_as_default():
    instance = _lookalike(QuadMeshingOptions, max_front_iterations=QuadMeshingOptions().max_front_iterations + 1)()
    model, _ = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=instance, **request())


def test_nested_policy_subclass_with_lax_eq_and_extra_field_is_refused_by_name():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    lax = dataclasses.make_dataclass('MeshQualityPolicy', [('maximum_element_area', float, .001)],
                                     bases=(MeshQualityPolicy,), frozen=True,
                                     namespace={'__eq__': lambda self, other: True, '__hash__': None})
    options = StructuredMeshingOptions(quality_policy=lax())
    reason = contract.request_refusal({'_native_surface_options': options}, None, None)
    assert reason and 'maximum_element_area' in reason


def test_numpy_array_and_scalars_on_allowlisted_fields_are_refused():
    import numpy as np
    from anymesher import _tensor_contract as contract
    from anymesher.structured import StructuredMeshingOptions
    for value in (np.array([3]), np.int64(3), np.float64(2.), Lax(), [3]):
        options = StructuredMeshingOptions()
        object.__setattr__(options, 'maximum_blocks', value)
        reason = contract.request_refusal({'_native_surface_options': options}, None, None)
        assert reason and 'maximum_blocks' in reason, repr(value)
    native = NativeMeshingOptions()
    object.__setattr__(native, 'max_insertions', np.int64(5))
    assert 'max_insertions' in contract.request_refusal({}, native, None)


def test_callable_or_lax_value_on_a_default_only_field_is_refused():
    from anymesher import _tensor_contract as contract
    for value in (Lax(), lambda *a: None, __import__('numpy').array([1])):
        quad = QuadMeshingOptions()
        object.__setattr__(quad, 'max_front_iterations', value)
        reason = contract.request_refusal({}, None, quad)
        assert reason and 'max_front_iterations' in reason
        native = NativeMeshingOptions()
        object.__setattr__(native, 'metric_field', value)
        assert 'metric_field' in contract.request_refusal({}, native, None)


def test_lax_values_never_satisfy_equals_empty_or_none_parameters():
    import numpy as np
    from anymesher import _tensor_contract as contract
    for name in ('order', 'recombine', 'layout_policy', 'beam_edges', 'beam_offsets', 'member_ids', 'refinements',
                 'structured_options', '_refine_declared_junction_transition'):
        assert contract.request_refusal({name: Lax()}, None, None), name
    assert contract.request_refusal({'recombine': np.True_}, None, None)
    assert contract.request_refusal({'order': 'linear'}, None, None) is None
    assert contract.request_refusal({'member_ids': ()}, None, None) is None
    assert contract.request_refusal({'beam_edges': []}, None, None) is None


def test_mapping_subclass_and_non_exact_placeholders_are_refused():
    from anymesher import _tensor_contract as contract
    class D(dict):
        pass
    assert contract.request_refusal({'structural_preparation': D()}, None, None)
    assert contract.request_refusal({'structural_preparation': Lax()}, None, None)
    assert contract.request_refusal({'structural_preparation': {}}, None, None) is None
    assert contract.request_refusal({'structural_preparation': False}, None, None) is None


def test_optional_child_default_none_replaced_by_dataclass_is_refused_by_path():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy
    native = NativeMeshingOptions()
    object.__setattr__(native, 'experimental_metric_provider', MeshQualityPolicy())
    assert 'experimental_metric_provider' in contract.request_refusal({}, native, None)


def test_ordinary_scalar_values_on_allowlisted_fields_are_still_accepted():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    options = StructuredMeshingOptions(max_element_growth=2., quality_policy=MeshQualityPolicy(maximum_aspect_ratio=3.))
    assert contract.request_refusal({'_native_surface_options': options}, None, None) is None
    assert contract.request_refusal({}, NativeMeshingOptions(max_insertions=10), QuadMeshingOptions()) is None


# ---- fixes-6: shadowed fields, metaclass tricks, spoofed __class__, non-dataclass placeholders ----------------
class _Meta(type):
    """Metaclass whose class objects claim equality with everything and refuse hashing."""
    def __eq__(cls, other):
        return True
    def __ne__(cls, other):
        return False
    def __hash__(cls):
        raise RuntimeError('hash of a foreign type must never be taken')
    def __instancecheck__(cls, instance):
        raise RuntimeError('isinstance against a foreign metaclass must never be evaluated')


class _V(metaclass=_Meta):
    def __call__(self, *args):
        return None
    def __repr__(self):
        raise RuntimeError('repr of a foreign object must never be evaluated')


def test_instance_shadowed_dataclass_fields_cannot_hide_a_changed_field():
    from anymesher import _tensor_contract as contract
    quad = QuadMeshingOptions(max_front_iterations=7)
    object.__setattr__(quad, '__dataclass_fields__', {})
    reason = contract.request_refusal({}, None, quad)
    assert reason and ('__dataclass_fields__' in reason or 'max_front_iterations' in reason)
    native = NativeMeshingOptions()
    object.__setattr__(native, '__dataclass_fields__', {})
    assert contract.request_refusal({}, native, None)  # undeclared instance attribute, even with default values


def test_undeclared_instance_attribute_is_refused_by_name_at_any_nesting_level():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import MeshQualityPolicy, StructuredMeshingOptions
    tried = 0
    for cls in _option_classes():  # slotted dataclasses cannot carry extra attributes at all; the rest must be refused
        instance = cls()
        try:
            object.__setattr__(instance, 'sneaky', 1)
        except AttributeError:
            continue
        tried += 1
        reason = _request_with(cls, instance)
        assert reason and 'sneaky' in reason, cls.__name__
    assert tried >= 1
    policy = MeshQualityPolicy()
    try:
        object.__setattr__(policy, 'sneaky', 1)
    except AttributeError:
        pass
    else:
        reason = contract.request_refusal({'_native_surface_options': StructuredMeshingOptions(quality_policy=policy)}, None, None)
        assert reason and 'sneaky' in reason


def test_foreign_metaclass_objects_are_never_taken_for_scalars_or_containers():
    from anymesher import _tensor_contract as contract
    from anymesher.structured import StructuredMeshingOptions
    for field in ('allow_detached_partition', 'maximum_blocks'):
        options = StructuredMeshingOptions()
        object.__setattr__(options, field, _V())
        reason = contract.request_refusal({'_native_surface_options': options}, None, None)
        assert isinstance(reason, str) and field in reason, field
    quad = QuadMeshingOptions()
    object.__setattr__(quad, 'max_front_iterations', _V())
    assert 'max_front_iterations' in contract.request_refusal({}, None, quad)
    for name in ('order', 'recombine', 'beam_edges', 'member_ids', 'layout_policy', 'structured_options'):
        reason = contract.request_refusal({name: _V()}, None, None)
        assert isinstance(reason, str) and name in reason, name
    assert isinstance(contract.request_refusal({'structural_preparation': _V()}, None, None), str)
    assert isinstance(contract.request_refusal({}, _V(), None), str)
    assert isinstance(contract.request_refusal({}, None, _V()), str)


def test_spoofed___class___and_getattribute_overrides_are_refused():
    from anymesher import _tensor_contract as contract
    class Spoof:
        @property
        def __class__(self):
            return QuadMeshingOptions
        def __getattribute__(self, name):
            return getattr(QuadMeshingOptions(), name)
    assert isinstance(contract.request_refusal({}, None, Spoof()), str)
    assert isinstance(contract.request_refusal({'_native_surface_options': Spoof()}, None, None), str)
    class Sub(QuadMeshingOptions):
        def __getattribute__(self, name):
            return getattr(QuadMeshingOptions(), name)
    assert isinstance(contract.request_refusal({}, None, Sub(max_front_iterations=7)), str)


@pytest.mark.parametrize('placeholder', [False, True, 0, 'x', 3.5, (), {}, [], object()])
def test_non_dataclass_placeholder_for_structured_options_is_a_typed_refusal_naming_the_option(placeholder):
    from anymesher import _tensor_contract as contract
    reason = contract.request_refusal({'_native_surface_options': placeholder}, None, None)
    assert isinstance(reason, str) and '_native_surface_options' in reason


def test_false_native_surface_options_is_typed_through_the_public_call():
    model, _ = nurbs_patch()
    with pytest.raises(QuadPublicUnsupported, match='_native_surface_options'):
        generate_hybrid_mesh_result(model, strategy='native', quad_options=QuadMeshingOptions(),
                                    _native_surface_options=False, **request())


def test_refusal_messages_never_evaluate_a_foreign_repr():
    from anymesher import _tensor_contract as contract
    assert isinstance(contract.request_refusal({'order': _V()}, None, None), str)
    assert isinstance(contract.request_refusal({'order': 'quadratic'}, None, None), str)
