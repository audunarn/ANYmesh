"""Exact owner receipts shared only inside one frontal invocation."""
import json
import numpy as np
import pytest

from anymesher._owner_xyz_cache import OwnerXYZCache
from anymesher.errors import MeshError


def analytic_chart():
    from anygeometry import GeometryModel, BezierDirectrix, ExtrudedSurface
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model = GeometryModel()
    controls = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
    vertices = model.add_points(controls)
    edge = model.add_spline(vertices[0], vertices[1:-1], vertices[-1])
    face = model.extrude((edge,), (.25, 0., 1.5))[0]
    model.set_face_surface(face, ExtrudedSurface(BezierDirectrix(controls), (.25, 0., 1.5)))
    return model, AnalyticMetricChart(model, face)


def region_chart():
    from test_joint_triangle_owner_batches import scoped_material_region_chart
    return scoped_material_region_chart()


def coordinates(chart):
    return np.asarray(((0., 0.), (.113, .227), (.331, .517), (.707, .823), (1., 1.))) @ chart.transform


@pytest.mark.parametrize('factory', (analytic_chart, region_chart))
@pytest.mark.parametrize('batch', (1, 2, 3, 4096))
def test_owner_batch_grouping_and_reuse_are_byte_exact(factory, batch):
    from anygeometry import to_dict
    model, chart = factory()
    original = to_dict(model)
    points = coordinates(chart)
    expected = chart.evaluate(points)
    cache = OwnerXYZCache(chart.evaluate, max_rows=8, batch_rows=batch)
    cache.register(points)
    actual = cache.evaluate(points)
    assert actual.tobytes() == expected.tobytes()
    actual[:] = 99.
    subset = points[[3, 1, 3, 0]]
    assert cache.evaluate(subset).tobytes() == expected[[3, 1, 3, 0]].tobytes()
    assert cache.evaluated_rows == len(points)
    assert cache.reused_rows == len(subset)
    assert to_dict(model) == original
    receipt = cache.receipt()
    assert receipt['binding_checks'] == 7
    assert receipt['storage_bytes'] > expected.nbytes
    assert np.isfinite(receipt['evaluation_seconds']) and receipt['evaluation_seconds'] >= 0.
    print(json.dumps(dict(fixture=factory.__name__, batch=batch, **receipt), sort_keys=True))


def test_appends_exact_keys_duplicates_signed_zero_and_subset_slot_identity():
    _, chart = analytic_chart()
    points = np.asarray(((0., .2), (-0., .2), (.31, .4), (.31, .4)))
    expected = chart.evaluate(points)
    cache = OwnerXYZCache(chart.evaluate, max_rows=5, batch_rows=2)
    cache.register(points)
    # The first queried subset row is global slot 2, not global slot 0.
    assert cache.evaluate(points[[2, 3]]).tobytes() == expected[[2, 3]].tobytes()
    assert cache.evaluate(points).tobytes() == expected.tobytes()
    assert cache.evaluated_rows == 4 and len(cache._values) == 3
    appended = np.vstack((points, (.73, .8)))
    cache.register(appended)
    assert cache.evaluate(appended).tobytes() == chart.evaluate(appended).tobytes()
    assert cache.evaluated_rows == 5
    points[:] = 99.
    assert cache._points.tobytes() == appended.tobytes()


@pytest.mark.parametrize('change', ('reorder', 'coordinate', 'signedzero', 'shrink', 'capacity'))
def test_invalid_registration_is_atomic(change):
    _, chart = analytic_chart()
    cache = OwnerXYZCache(chart.evaluate, max_rows=3)
    points = np.asarray(((0., .2), (.3, .4)))
    cache.register(points)
    expected = cache.evaluate(points)
    altered = points.copy()
    if change == 'reorder': altered = altered[::-1]
    if change == 'coordinate': altered[1, 0] = np.nextafter(altered[1, 0], 1.)
    if change == 'signedzero': altered[0, 0] = -0.
    if change == 'shrink': altered = altered[:1]
    if change == 'capacity': altered = np.vstack((altered, (.5,.6), (.7,.8)))
    with pytest.raises(MeshError): cache.register(altered)
    assert cache.registrations == 1 and cache._points.tobytes() == points.tobytes()
    assert cache.evaluate(points).tobytes() == expected.tobytes()
    with pytest.raises(MeshError, match='not globally registered'):
        cache.evaluate(np.asarray(((.333, .4),)))


@pytest.mark.parametrize('attribute', ('transform', 'inverse'))
def test_transform_mutation_refuses_reuse(attribute):
    _, chart = analytic_chart()
    points = coordinates(chart)
    cache = OwnerXYZCache(chart.evaluate, max_rows=5)
    cache.register(points); cache.evaluate(points)
    original = getattr(chart, attribute).copy()
    getattr(chart, attribute)[0, 0] = np.nextafter(original[0, 0], np.inf)
    with pytest.raises(MeshError, match='chart binding changed'): cache.evaluate(points)
    assert cache.evaluated_rows == 5


@pytest.mark.parametrize('failure', ('exception', 'shape', 'nan', 'cancel', 'stale'))
def test_late_failure_never_publishes_pending_receipts(monkeypatch, failure):
    from anygeometry import GeometryError
    model, chart = analytic_chart()
    points = coordinates(chart)
    error = RuntimeError('late owner failure')
    enabled = False
    calls = 0
    original = model.evaluate_face_many
    def evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if enabled and calls == 3:
            if failure == 'exception': raise error
            if failure == 'shape': return np.zeros((1, 3))
            if failure == 'nan': return np.full((len(args[1]), 3), np.nan)
        result = original(*args, **kwargs)
        if enabled and calls == 3 and failure == 'stale': model.add_point(9.,9.,9.)
        return result
    monkeypatch.setattr(model, 'evaluate_face_many', evaluate)
    def cancel(phase):
        if enabled and failure == 'cancel' and phase == 'native-v2 owner XYZ commit': raise error
    cache = OwnerXYZCache(chart.evaluate, max_rows=5, batch_rows=2, cancellation_check=cancel)
    cache.register(points)
    old = cache.evaluate(points[:1])
    committed = dict(cache._values)
    enabled = True
    with pytest.raises((RuntimeError, MeshError, GeometryError)) as caught: cache.evaluate(points)
    if failure in ('exception', 'cancel'): assert caught.value is error
    assert cache._values.keys() == committed.keys()
    assert all(cache._values[key] is value for key, value in committed.items())
    enabled = False
    if failure != 'stale': assert cache.evaluate(points[:1]).tobytes() == old.tobytes()


@pytest.mark.parametrize('factory', (analytic_chart, region_chart))
def test_stale_binding_rejected_even_on_full_cache_hit(factory):
    from anygeometry import GeometryError
    model, chart = factory()
    points = coordinates(chart)
    cache = OwnerXYZCache(chart.evaluate, max_rows=5)
    cache.register(points); cache.evaluate(points)
    model.add_point(9.,9.,9.)
    with pytest.raises(GeometryError, match='stale'): cache.evaluate(points)
    assert cache.owner_calls == 1


@pytest.mark.parametrize('field,value', (('max_rows',0),('max_rows',True),('batch_rows',0),('batch_rows',4097)))
def test_bounds_and_generic_callbacks_refused(field, value):
    _, chart = analytic_chart()
    arguments = dict(max_rows=5, batch_rows=2); arguments[field] = value
    with pytest.raises(MeshError): OwnerXYZCache(chart.evaluate, **arguments)
    with pytest.raises(MeshError): OwnerXYZCache(lambda rows: chart.evaluate(rows), max_rows=5)


@pytest.mark.parametrize('factory', (analytic_chart, region_chart))
@pytest.mark.parametrize('mode', ('legacy', 'isotropic_spatial'))
@pytest.mark.parametrize('spatial_control', (False, True))
def test_complete_frontal_parity_and_finite_owner_work(monkeypatch, factory, mode, spatial_control):
    from dataclasses import replace
    from anygeometry import to_dict
    from anymesher import NativeMeshingOptions, FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec
    from anymesher.triangulation import triangulate_polygon
    import anymesher.native_v2 as native
    import anymesher._owner_xyz_cache as cache_module
    model, chart = factory()
    before = to_dict(model)
    original = triangulate_polygon(((0.,0.),(1.,0.),(1.,1.),(0.,1.)), backend='python')
    original = replace(original, points=original.points @ chart.transform)
    original_bytes = (original.points.tobytes(), original.triangles.tobytes(), original.segments.tobytes())
    if mode == 'legacy' and spatial_control:
        pytest.skip('legacy does not accept spatial controls')
    field = (MetricFieldSpec(IsotropicMetricControl(.5), feature_controls=(
        FeatureDistanceMetricControl(((.5,.2,.5),), .2, .3, 1.5, 'local'),)) if spatial_control else None)
    options = NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode=mode, metric_field=field,
        max_insertions=3, max_topology_operations=16, cancellation_interval=2)
    outcomes = []
    owner_queries = []
    if chart.region_binding is None:
        original_evaluator = model.evaluate_face_many
        def observed(face, rows, **kwargs):
            owner_queries.append(len(rows))
            return original_evaluator(face, rows, **kwargs)
        monkeypatch.setattr(model, 'evaluate_face_many', observed)
    else:
        import anygeometry
        original_evaluator = anygeometry.evaluate_material_surface_region
        def observed(owner, binding, face, rows, **kwargs):
            if not kwargs.get('derivatives', False): owner_queries.append(len(rows))
            return original_evaluator(owner, binding, face, rows, **kwargs)
        monkeypatch.setattr(anygeometry, 'evaluate_material_surface_region', observed)
    work = []
    for cache_type in (lambda *args, **kwargs: None, OwnerXYZCache):
        monkeypatch.setattr(cache_module, 'OwnerXYZCache', cache_type)
        owner_queries.clear()
        result, report = native.frontal_delaunay_refine(original, options, target_size=.5,
            metric_to_physical=chart.evaluate, metric_jacobian=chart.jacobians,
            short_edge_offcentre=True, minimum_angle_target=15.)
        receipt = report.pop('owner_xyz_cache', None)
        work.append(dict(calls=len(owner_queries),rows=sum(owner_queries)))
        outcomes.append((result.points.tobytes(),result.triangles.tobytes(),result.segments.tobytes(),report))
        if receipt is not None:
            assert receipt['reused_rows'] > 0
            assert receipt['evaluated_rows'] <= receipt['registered_rows']
            assert receipt['registered_rows'] <= len(original.points)+3
            assert receipt['binding_checks'] >= 5
            assert receipt['owner_calls'] == len(owner_queries)
            assert receipt['evaluated_rows'] == sum(owner_queries)
            print(json.dumps(dict(fixture=factory.__name__, mode=mode, **receipt),sort_keys=True))
    assert outcomes[0] == outcomes[1]
    assert work[1]['rows'] < work[0]['rows']
    print(json.dumps(dict(fixture=factory.__name__, mode=mode, spatial_control=spatial_control,
        uncached_owner_work=work[0],cached_owner_work=work[1]),sort_keys=True))
    assert to_dict(model) == before
    assert original_bytes == (original.points.tobytes(), original.triangles.tobytes(), original.segments.tobytes())


def test_generic_callback_sequence_and_receipt_unchanged(monkeypatch):
    from anymesher import NativeMeshingOptions
    from anymesher.triangulation import triangulate_polygon
    import anymesher.native_v2 as native
    import anymesher._owner_xyz_cache as cache_module
    _, chart = analytic_chart()
    original = triangulate_polygon(((0.,0.),(1.,0.),(1.,1.),(0.,1.)), backend='python')
    calls = []
    def generic(rows):
        calls.append(rows.tobytes()); return chart.evaluate(rows)
    def forbidden(*args, **kwargs): pytest.fail('generic callback enabled owner XYZ cache')
    monkeypatch.setattr(cache_module, 'OwnerXYZCache', forbidden)
    options = NativeMeshingOptions(point_placement='frontal_delaunay', metric_mode='isotropic_spatial',
        max_insertions=1,max_topology_operations=4,cancellation_interval=2)
    result, report = native.frontal_delaunay_refine(original,options,target_size=.5,
        metric_to_physical=generic,metric_jacobian=chart.jacobians)
    assert calls and 'owner_xyz_cache' not in report
    first = (result.points.tobytes(), result.triangles.tobytes(), report, tuple(calls))
    calls.clear()
    result, report = native.frontal_delaunay_refine(original,options,target_size=.5,
        metric_to_physical=generic,metric_jacobian=chart.jacobians)
    assert first == (result.points.tobytes(), result.triangles.tobytes(), report, tuple(calls))


@pytest.mark.parametrize('phase', ('native-v2 owner XYZ registration commit', 'native-v2 owner XYZ commit'))
def test_commit_cancellation_preserves_registration_values_and_identity(phase):
    _, chart = analytic_chart()
    points = coordinates(chart)
    error = MeshError('explicit cancellation')
    enabled = False
    def cancel(location):
        if enabled and location == phase: raise error
    cache = OwnerXYZCache(chart.evaluate,max_rows=5,batch_rows=2,cancellation_check=cancel)
    cache.register(points[:1]);cache.evaluate(points[:1])
    original = (cache._points.tobytes(), cache._registered.copy(), dict(cache._values))
    enabled = True
    with pytest.raises(MeshError) as caught:
        if phase.endswith('registration commit'): cache.register(points)
        else: cache.evaluate(points[:1])
    assert caught.value is error
    assert original == (cache._points.tobytes(), cache._registered, cache._values)


def test_post_reuse_public_validation_error_identity(monkeypatch):
    _, chart = analytic_chart()
    points = coordinates(chart)
    cache = OwnerXYZCache(chart.evaluate,max_rows=5)
    cache.register(points);cache.evaluate(points)
    error = RuntimeError('owner post-reuse validation')
    original = chart._current
    calls = 0
    def current():
        nonlocal calls
        calls += 1
        original()
        if calls == 2: raise error
    monkeypatch.setattr(chart,'_current',current)
    with pytest.raises(RuntimeError) as caught: cache.evaluate(points)
    assert caught.value is error and calls == 2 and cache.owner_calls == 1


def test_experimental_provider_excludes_cache_and_preserves_callback_sequence(monkeypatch):
    from anymesher import NativeMeshingOptions, ExperimentalMetricProvider
    from anymesher.triangulation import triangulate_polygon
    import anymesher.native_v2 as native
    import anymesher._owner_xyz_cache as cache_module
    _, chart = analytic_chart()
    seed = triangulate_polygon(((0.,0.),(1.,0.),(1.,1.),(0.,1.)),backend='python')
    metric_calls = []
    def field(rows):
        metric_calls.append(rows.tobytes())
        return np.broadcast_to(np.eye(2)*4.,(len(rows),2,2)).copy()
    def forbidden(*args,**kwargs): pytest.fail('experimental provider enabled owner XYZ cache')
    monkeypatch.setattr(cache_module,'OwnerXYZCache',forbidden)
    options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
        experimental_metric_provider=ExperimentalMetricProvider(field),max_insertions=1,
        max_topology_operations=4,cancellation_interval=2)
    outputs=[]
    for _ in range(2):
        metric_calls.clear()
        result, report=native.frontal_delaunay_refine(seed,options,target_size=.5,
            metric_to_physical=chart.evaluate,metric_jacobian=chart.jacobians)
        assert 'owner_xyz_cache' not in report
        outputs.append((result.points.tobytes(),result.triangles.tobytes(),report,tuple(metric_calls)))
    assert outputs[0] == outputs[1] and metric_calls


@pytest.mark.parametrize('points', (np.zeros(2), np.zeros((1,3)), np.array(((np.inf,0.),)), np.array(((np.nan,0.),))))
def test_invalid_coordinate_rows_do_not_query_owner(points):
    _,chart=analytic_chart()
    cache=OwnerXYZCache(chart.evaluate,max_rows=5)
    with pytest.raises(MeshError): cache.register(points)
    with pytest.raises(MeshError): cache.evaluate(points)
    assert not cache.owner_calls and not cache.registrations


@pytest.mark.parametrize('checkpoint_number', (1, 2, 3))
def test_registration_uses_entry_snapshot_when_callback_mutates_caller(checkpoint_number):
    _, chart = analytic_chart()
    caller = coordinates(chart)
    entry = caller.copy()
    expected = chart.evaluate(entry)
    enabled = True
    calls = 0
    def cancel(phase):
        nonlocal calls
        if enabled and phase == 'native-v2 owner XYZ registration':
            calls += 1
            if calls == checkpoint_number: caller[:] += .019
    cache = OwnerXYZCache(chart.evaluate,max_rows=5,batch_rows=2,cancellation_check=cancel)
    cache.register(caller)
    enabled = False
    assert caller.tobytes() != entry.tobytes()
    assert cache._points.tobytes() == entry.tobytes()
    assert cache._registered == set(cache._keys(entry))
    assert cache.evaluate(entry).tobytes() == expected.tobytes()
    with pytest.raises(MeshError, match='not globally registered'): cache.evaluate(caller)


@pytest.mark.parametrize('phase,checkpoint_number', (
    ('native-v2 owner XYZ lookup', 1), ('native-v2 owner XYZ evaluation', 1),
    ('native-v2 owner XYZ evaluation', 2), ('native-v2 owner XYZ commit', 1)))
def test_evaluation_uses_entry_snapshot_when_callback_mutates_caller(phase,checkpoint_number):
    _,chart=analytic_chart()
    caller=coordinates(chart)
    entry=caller.copy()
    expected=chart.evaluate(entry)
    changed=caller+.019
    changed_xyz=chart.evaluate(changed)
    enabled=False
    calls=0
    def cancel(location):
        nonlocal calls
        if enabled and location == phase:
            calls+=1
            if calls == checkpoint_number: caller[:]=changed
    cache=OwnerXYZCache(chart.evaluate,max_rows=5,batch_rows=2,cancellation_check=cancel)
    cache.register(caller)
    enabled=True
    actual=cache.evaluate(caller)
    enabled=False
    assert caller.tobytes() == changed.tobytes()
    assert actual.tobytes() == expected.tobytes()
    assert actual.tobytes() != changed_xyz.tobytes()
    assert cache._values.keys() == set(cache._keys(entry))
    assert cache.evaluate(entry).tobytes() == expected.tobytes()
    with pytest.raises(MeshError,match='not globally registered'): cache.evaluate(caller)
