"""Owner Hessian eligibility and real native metric composition regressions."""
from dataclasses import replace
import numpy as np
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher.errors import MeshError
from anymesher.metric import (MetricFieldSpec, IsotropicMetricControl,
                             FeatureDistanceMetricControl, ImportedMetricSamples, SpatialMetricField)
from test_nurbs_curvilinear_consumer import options


def support_case(kind):
    controls = (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.)))
    support = NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),controls)
    if kind == 'rotated':
        support = replace(support, controls=(((0.,0.,0.),(0.,2.,1.)),((1.,0.,1.),(1.,2.,2.))))
    elif kind == 'crop_reverse':
        support = replace(support, u_range=(.125,.875), v_range=(.875,.125))
    elif kind == 'uniform_weights':
        support = replace(support, weights=((2.,2.),(2.,2.)))
    elif kind == 'curved':
        support = replace(support, controls=(((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,.25))))
    elif kind == 'nonlinear_plane':
        support = replace(support, controls=(((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(2.,1.,0.))))
    elif kind == 'rational':
        support = replace(support, weights=((1.,2.),(1.,2.)))
    elif kind == 'multispan':
        support = replace(support, knots_u=(0.,0.,.5,1.,1.),
                          controls=(controls[0],((.5,0.,0.),(.5,1.,0.)),controls[1]))
    return support


def binding_case(kind='affine', *, divisions=16):
    support = support_case(kind)
    model = owner.GeometryModel()
    uv = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    curves = tuple(CurveOnSurface(support,NurbsCurve(1,(0.,0.,1.,1.),(uv[i],uv[(i+1)%4])))
                   for i in range(4))
    vertices = tuple(model.add_point(*curve.exact_point(0)) for curve in curves)
    edges = tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve) for i,curve in enumerate(curves))
    face = model.add_face(edges,surface=support)
    regions = owner.query_material_surface_regions(model,(face,))
    binding = MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=.01)
    intervals = tuple((model.handle('edge',edge),i/divisions,(i+1)/divisions)
                      for edge in edges for i in range(divisions))
    binding._curvilinear_state['chords'] = owner.query_prepared_material_boundary_chords(
        model,regions,intervals,chord_error=.01)
    return model,binding


@pytest.mark.parametrize('kind,expected', [('affine',True),('rotated',True),('crop_reverse',True),
    ('uniform_weights',True),('curved',False),('nonlinear_plane',False),('rational',False),('multispan',False)])
def test_owner_full_chart_hessian_controls_eligibility_and_legacy_fallback(kind,expected):
    model,binding = binding_case(kind)
    before = owner.to_dict(model)
    legacy = binding.interior_size_proposal(.5)
    assert 0 < legacy < .5
    assert binding._has_globally_affine_support() is expected
    assert binding.interior_size_proposal(.5,_boundary_graded=True) == (.5 if expected else legacy)
    proposal = binding.interior_size_proposal(.5,_boundary_graded=True,_with_affine=True)
    assert type(proposal) is tuple and proposal == (.5 if expected else legacy, expected)
    assert binding.interior_size_proposal(.5,_with_affine=True) == (legacy,False)
    assert owner.to_dict(model) == before
    if kind == 'nonlinear_plane':
        from anygeometry._local_curvature import bound_surface_curvature
        bound = bound_surface_curvature(binding.region.support)
        assert bound.tiles[0].curvature_upper == 0
        assert bound.tiles[0].hessian_norm_upper > 0


@pytest.mark.parametrize('action', ('source','calculator','cancel','exception'))
def test_affine_owner_proposal_brackets_callbacks(monkeypatch,action):
    from anygeometry import _local_curvature as curvature
    model,binding = binding_case()
    error = curvature.LocalApproximationNotCertified('original callback refusal')
    seen = []
    def callback(phase):
        if phase != 'local curvature bounds' or seen:
            return False
        seen.append(phase)
        if action == 'source':
            object.__setattr__(binding.region.support,'controls',support_case('curved').controls)
        elif action == 'calculator':
            monkeypatch.setattr(curvature,'_bound',lambda *args:None)
        elif action == 'cancel':
            return True
        else:
            raise error
        return False
    with pytest.raises((owner.GeometryError,MeshError)) as caught:
        binding.interior_size_proposal(.5,callback,_boundary_graded=True)
    assert seen
    if action == 'exception':
        assert caught.value is error


def test_effective_boundary_distance_field_grows_inward_and_preserves_existing_controls():
    model,binding = binding_case()
    feature = FeatureDistanceMetricControl(((.5,.5,0.),),.1,.01,1.3,name='user feature')
    original = MetricFieldSpec(IsotropicMetricControl(.5),(feature,),maximum_gradation=1.3,
                               maximum_anisotropy=4.)
    proposed = binding.interior_size_proposal(original.global_control.target_size,_boundary_graded=True)
    graded = binding.boundary_size_metric(replace(original,global_control=IsotropicMetricControl(proposed)))
    assert all(control in graded.feature_controls for control in original.feature_controls)
    assert graded.imported_samples == original.imported_samples
    assert graded.maximum_gradation == original.maximum_gradation
    assert graded.maximum_anisotropy == original.maximum_anisotropy
    points = np.array(((.03125,0.,0.),(.5,.5,0.)))
    tensors = SpatialMetricField(graded).evaluate(points)
    original_tensors = SpatialMetricField(original).evaluate(points)
    assert np.all(tensors[:,0,0] >= original_tensors[:,0,0])
    sizes = 1./np.sqrt(tensors[:,0,0])
    assert sizes[0] <= 1./16
    assert 1./16 < sizes[1] <= .1
    fine = binding.boundary_size_metric(MetricFieldSpec.uniform(.02))
    assert fine.global_control.target_size == .02
    assert np.all(1./np.sqrt(SpatialMetricField(fine).evaluate(points)[:,0,0]) <= .02)
    imported = ImportedMetricSamples(str(model.model_id), model.revision, ((.5,.5,0.),),
        (((400.,0.,0.),(0.,400.,0.),(0.,0.,400.)),))
    with_imported = replace(original, imported_samples=(imported,))
    composed = binding.boundary_size_metric(with_imported)
    assert composed.imported_samples == (imported,)
    tensors = SpatialMetricField(composed,model_uuid=str(model.model_id),
                                geometry_revision=model.revision).evaluate(points)
    assert np.all(1./np.sqrt(tensors[:,0,0]) <= .05)


@pytest.mark.parametrize('when', ('initial','callback','callback_exception'))
@pytest.mark.parametrize('attribute', ('__defaults__','__kwdefaults__'))
def test_affine_defaults_pins_never_invoke_hostile_repr_or_equality(monkeypatch,when,attribute):
    from anygeometry import _local_curvature as curvature
    model,binding = binding_case()
    legacy = binding.interior_size_proposal(.5)
    invoked = []
    class Hostile:
        def __repr__(self):
            invoked.append('repr')
            raise AssertionError('caller repr executed')
        def __eq__(self,other):
            invoked.append('eq')
            raise AssertionError('caller equality executed')
    value = (Hostile(),) if attribute=='__defaults__' else {'poison':Hostile()}
    error = RuntimeError('original callback exception after defaults mutation')
    def mutate():
        monkeypatch.setattr(curvature._bound,attribute,value)
    if when == 'initial':
        mutate()
        assert binding.interior_size_proposal(.5,_boundary_graded=True) == legacy
    else:
        seen = []
        def callback(phase):
            if phase=='local curvature bounds' and not seen:
                seen.append(phase)
                mutate()
                if when=='callback_exception':
                    raise error
            return False
        with pytest.raises(RuntimeError if when=='callback_exception' else MeshError) as caught:
            binding.interior_size_proposal(.5,callback,_boundary_graded=True)
        assert seen
        if when=='callback_exception':
            assert caught.value is error
    assert invoked == []


@pytest.mark.parametrize('target', ('policy','alias'))
@pytest.mark.parametrize('when', ('initial','callback'))
def test_affine_context_never_compares_unknown_policy_or_alias_values(monkeypatch,target,when):
    from anygeometry import _local_curvature as curvature
    model,binding = binding_case()
    legacy = binding.interior_size_proposal(.5)
    invoked = []
    policy = curvature.DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    original_boxes = policy.max_boxes_per_pair
    class Hostile:
        def __repr__(self):
            invoked.append('repr')
            raise AssertionError('caller repr executed')
        def __eq__(self,other):
            invoked.append('eq')
            raise AssertionError('caller equality executed')
        def __getattr__(self,name):
            invoked.append('getattr')
            raise AssertionError('caller attribute executed')
    def mutate():
        if target=='policy':
            object.__setattr__(policy,'max_boxes_per_pair',Hostile())
        else:
            monkeypatch.setattr(curvature,'CurvatureTile',Hostile())
    try:
        if when=='initial':
            mutate()
            assert binding.interior_size_proposal(.5,_boundary_graded=True) == legacy
        else:
            seen = []
            def callback(phase):
                if phase=='local curvature bounds' and not seen:
                    seen.append(phase)
                    mutate()
                return False
            with pytest.raises(MeshError):
                binding.interior_size_proposal(.5,callback,_boundary_graded=True)
            assert seen
    finally:
        object.__setattr__(policy,'max_boxes_per_pair',original_boxes)
    assert invoked == []


@pytest.mark.parametrize('explicit', (None,.02))
def test_actual_public_native_entry_uses_owner_eligible_graded_metric(monkeypatch,explicit):
    import anymesher.hybrid as hybrid
    from test_local_curved_mesh import curved_loop_model
    model,_,_,_ = curved_loop_model()
    request = options()
    request.update(mutation_policy='read_only')
    if explicit is not None:
        request['native_options'] = replace(request['native_options'],metric_field=MetricFieldSpec.uniform(explicit))
    captured = []
    class NativeEntryReached(BaseException):
        pass
    def inspect(*args,**kwargs):
        settings = kwargs.get('options')
        native = kwargs.get('native_options') if settings is None else settings.native_options
        target = kwargs['target_size'] if settings is None else settings.target_size
        jacobians = kwargs['_metric_jacobian'](np.vstack((args[0], *args[1])))
        stretch = float(np.max(np.linalg.svd(jacobians, compute_uv=False)))
        captured.append((native.metric_field,target,stretch))
        raise NativeEntryReached()
    monkeypatch.setattr(hybrid,'mesh_planar_surface',inspect)
    with pytest.raises(NativeEntryReached):
        hybrid.generate_hybrid_mesh_result(model,**request)
    assert len(captured) == 1
    field,target,stretch = captured[0]
    expected = .5 if explicit is None else explicit
    assert field.global_control.target_size == expected
    # The fallback is in chart units. Use the actual supplied Jacobian, not
    # an assumed unit frame; allow only roundoff of the repeated 3x2 SVD and
    # scalar division/product, without changing any geometry/quality limit.
    assert target * stretch == pytest.approx(expected, rel=64*np.finfo(float).eps, abs=0.)
    if explicit is None:
        assert any(row.name=='owner chord spacing proposal' for row in field.feature_controls)
