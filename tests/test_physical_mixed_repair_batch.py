"""Compare batching at matching Jacobian capability; retain generic legacy controls."""
from copy import copy
from dataclasses import replace

import numpy as np
import pytest

import anygeometry as owner
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.nurbs_intersection_curve import intersection_curve_operation
from anymesher._analytic_metric_chart import AnalyticMetricChart
from anymesher._material_region_binding import MaterialRegionBinding
from anymesher._physical_mixed_repair import (_batch_chart, repair_mixed_coordinates,
                                             repair_mixed_core_coordinates)
from anymesher.core import MeshCore
from anymesher.errors import MeshError
from anymesher.recombine import RecombinationReport
from anymesher.surface_mesh import SurfaceMeshOptions, _published_growth, _quality_threshold_report
from anymesher.quality_v2 import evaluate_quality


@pytest.fixture(scope='module')
def chart_fixture():
    model = owner.GeometryModel()
    knots, heights = (0.,0.,0.,1.,1.,1.), (1.,-1.,1.)
    corners = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    parents = []
    for scale, offset in ((1.,0.),(2.,-.25)):
        surface = NurbsSurface(2,2,knots,knots,tuple(tuple(
            (float(i-1),float(j-1),scale*(heights[i]+heights[j])+offset)
            for j in range(3)) for i in range(3)))
        curves = tuple(CurveOnSurface(surface,NurbsCurve(1,(0.,0.,1.,1.),
            (corners[i],corners[(i+1)%4]))) for i in range(4))
        vertices = model.add_points(curve.exact_point(0) for curve in curves)
        edges = tuple(model.add_curve(vertices[i],vertices[(i+1)%4],curve)
                      for i,curve in enumerate(curves))
        parents.append(model.add_face(edges,surface=surface))
    with intersection_curve_operation():
        owner.apply_intersections(model,owner.plan_intersections(model,parents,policy='imprint'),policy='imprint')
        face = next(face.id for face in model.faces.values() if not face.holes
                    and face.surface == model.faces[min(model.faces)].surface)
        regions = owner.query_material_surface_regions(model,(face,))
        intervals = tuple((model.handle('edge',use.edge),i/4,(i+1)/4)
                          for use in model.faces[face].loop for i in range(4))
        chords = owner.query_prepared_material_boundary_chords(model,regions,intervals,chord_error=.1)
        binding = MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=.1)
        binding._curvilinear_state['chords'] = chords
        chart = AnalyticMetricChart(model,face,region_binding=binding)
        yield model, chart


def operands(chart, work):
    # One bad triangle fan and a disjoint Q4, all inside the true disk.
    uv = np.asarray(((.32,.45),(.48,.45),(.48,.55),(.32,.55),(.325,.452),
                     (.54,.46),(.60,.46),(.60,.54),(.54,.54))) @ chart.transform
    triangles = np.asarray(((0,1,4),(1,2,4),(2,3,4),(3,0,4)))
    core = MeshCore(chart.evaluate(uv),triangles,np.asarray(((5,6,7,8),)))
    report = RecombinationReport(core,np.empty((0,2),dtype=np.int64),
        np.empty(0,dtype=np.int64),np.empty(0),0,0)
    settings = SurfaceMeshOptions(min_angle=30.,max_recombination_work=work)
    return uv,triangles,report,settings,(0,1,2,3,5,6,7,8)


def snapshot(core):
    return tuple(getattr(core,name).tobytes() for name in (
        'node_coordinates','node_ids','triangle_connectivity','quad_connectivity',
        'triangle_active','quad_active','triangle_ids','quad_ids'))


@pytest.mark.parametrize('work',(1,5,7,31))
def test_owner_node_batches_match_serial_results_work_quality_and_fixed_rows(chart_fixture,monkeypatch,work):
    model,chart = chart_fixture
    assert _batch_chart(chart.evaluate,None) is chart
    uv,triangles,report,settings,fixed = operands(chart,work)
    before = uv.tobytes(),snapshot(report.mesh),owner.to_dict(model)
    calls = []
    evaluate_owner = owner.evaluate_material_surface_region
    def measured(*args,**kwargs):
        if not kwargs.get('derivatives',False):
            calls.append(len(args[3]))
        return evaluate_owner(*args,**kwargs)
    monkeypatch.setattr(owner,'evaluate_material_surface_region',measured)
    def forwarding(points):
        return chart.evaluate(points)
    # The canonical owner now supplies growth Jacobians. Compare execution
    # batching only after supplying that SAME capability to the serial oracle.
    serial,used,truncated = repair_mixed_core_coordinates(uv,triangles,report.mesh,settings,
        forwarding,fixed,exchange_work=report.exchange_work,evaluate_jacobians=chart.jacobians)
    sr=replace(report,exchange_work=report.exchange_work+used,
               exchange_truncated=report.exchange_truncated or truncated)
    serial_calls = tuple(calls)
    calls.clear()
    diagnostic={}
    batched,br = repair_mixed_coordinates(uv,triangles,report,settings,chart.evaluate,fixed,_diagnostics=diagnostic)
    batch_calls = tuple(calls)
    np.testing.assert_array_equal(batched,serial)
    np.testing.assert_array_equal(batched[list(fixed)],uv[list(fixed)])
    assert (br.exchange_work,br.exchange_truncated) == (sr.exchange_work,sr.exchange_truncated)
    assert br.exchange_work <= work
    assert before == (uv.tobytes(),snapshot(report.mesh),owner.to_dict(model))
    assert snapshot(br.mesh) == snapshot(sr.mesh) == snapshot(report.mesh)
    xyz = chart.evaluate(batched)
    physical = MeshCore(xyz,report.mesh.triangle_connectivity,report.mesh.quad_connectivity)
    serial_physical = MeshCore(chart.evaluate(serial),report.mesh.triangle_connectivity,report.mesh.quad_connectivity)
    assert _quality_threshold_report(evaluate_quality(physical),settings) == (
        _quality_threshold_report(evaluate_quality(serial_physical),settings))
    assert _published_growth(physical,settings.max_element_growth) == (
        _published_growth(serial_physical,settings.max_element_growth))
    assert batch_calls[0] == batch_calls[-1] == len(uv)  # final complete owner evaluation
    packets=diagnostic['evaluation_packets']
    assert tuple(packet['rows'] for packet in packets)==batch_calls
    assert packets[0]['kind']=='initial' and packets[-1]['kind']=='final'
    full_probes=[];node_packets=[]
    for packet in packets[1:-1]:
        assert 0<packet['used']<=br.exchange_work<=work
        if packet['kind']=='node_batch':
            # Ten legacy directions x3 plus 13 bounded growth fractions.
            assert packet['rows']<=min(work,43);node_packets.append(packet)
        else:
            assert packet['kind'] in ('constrained_patch','coupled_growth')
            assert packet['rows']==len(uv);full_probes.append(packet)
    assert len(full_probes)<=br.exchange_work
    if work==7:
        # One free node: baseline + four finite-difference probes fit this
        # pre-existing coupled solver budget, each needs complete physical XYZ.
        assert any(packet['kind']=='constrained_patch' for packet in full_probes)
        assert any(packet['proposal'][1]=='derivative' for packet in full_probes)
    if work==5:
        assert node_packets and sum(packet['rows'] for packet in node_packets)>1
        assert len(batch_calls)<len(serial_calls)
    if node_packets:
        assert sum(packet['rows'] for packet in node_packets)<=br.exchange_work
        assert len(batch_calls)<=len(serial_calls)
    assert diagnostic['trials']==br.exchange_work-report.exchange_work


def test_owner_callbacks_and_forwarders_keep_serial_path_and_exception(chart_fixture):
    _model,chart = chart_fixture
    uv,triangles,report,settings,fixed = operands(chart,7)
    assert _batch_chart(lambda rows:chart.evaluate(rows),None) is None
    assert _batch_chart(chart.evaluate,lambda phase:None) is None
    with_check = copy(chart)
    with_check.check = lambda phase:None
    assert _batch_chart(with_check.evaluate,None) is None
    class Cancelled(Exception):
        pass
    phases = []
    def cancel(phase):
        phases.append(phase)
        raise Cancelled()
    before = uv.tobytes(),snapshot(report.mesh)
    with pytest.raises(Cancelled):
        repair_mixed_coordinates(uv,triangles,report,settings,chart.evaluate,fixed,cancellation_check=cancel)
    assert phases == ['physical mixed-cell UV repair']
    assert before == (uv.tobytes(),snapshot(report.mesh))


@pytest.mark.parametrize('change',('inverse','missing','binding'))
def test_changed_chart_authority_refuses_before_any_trial(chart_fixture,change):
    _model,chart = chart_fixture
    uv,triangles,report,settings,fixed = operands(chart,7)
    forged = copy(chart)
    if change == 'inverse':
        forged.inverse = chart.inverse.copy()
        forged.inverse[0,0] *= 2.
    elif change == 'missing':
        forged._authority = None
    else:
        forged.binding = object()
    before = uv.tobytes(),snapshot(report.mesh)
    with pytest.raises(MeshError,match='inverse changed|missing owner|authority changed'):
        repair_mixed_coordinates(uv,triangles,report,settings,forged.evaluate,fixed)
    assert before == (uv.tobytes(),snapshot(report.mesh))


def test_ordinary_polynomial_nurbs_and_generic_evaluators_are_not_batch_eligible():
    # An existing owner fixture without a new-family trace keeps its route.
    from test_nurbs_curvilinear_consumer import patch_model
    model,face,edges = patch_model()
    regions = owner.query_material_surface_regions(model,(face,))
    chords = owner.query_prepared_material_boundary_chords(model,regions,
        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=.1)
    binding = MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=.1)
    binding._curvilinear_state['chords'] = chords
    chart = AnalyticMetricChart(model,face,region_binding=binding)
    assert _batch_chart(chart.evaluate,None) is None
    assert _batch_chart(model.faces[face].surface.evaluate_many,None) is None


def test_implicit_plane_owner_chart_keeps_serial_eligibility():
    from test_plane_curvilinear_consumer import plane_disk
    model,face,edges = plane_disk()
    regions = owner.query_material_surface_regions(model,(face,))
    chords = owner.query_prepared_material_boundary_chords(model,regions,
        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=1.)
    binding = MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=1.)
    binding._curvilinear_state['chords'] = chords
    chart = AnalyticMetricChart(model,face,region_binding=binding)
    assert _batch_chart(chart.evaluate,None) is None


def test_missing_nurbs_capability_keeps_legacy_plane_chart_serial(monkeypatch):
    import anymesher._analytic_metric_chart as charts
    from test_plane_curvilinear_consumer import plane_disk
    model,face,edges = plane_disk()
    regions = owner.query_material_surface_regions(model,(face,))
    chords = owner.query_prepared_material_boundary_chords(model,regions,
        tuple((model.handle('edge',edge),0.,1.) for edge in edges),chord_error=1.)
    binding = MaterialRegionBinding(model,regions,regions.regions[0],face,geometry_approximation_error=1.)
    binding._curvilinear_state['chords'] = chords
    chart = AnalyticMetricChart(model,face,region_binding=binding)
    # Model the capability sentinel already supplied by the legacy chart
    # module; eligibility must not import a newer owner dependency itself.
    monkeypatch.setattr(charts,'NurbsSurface',())
    assert _batch_chart(chart.evaluate,None) is None


@pytest.mark.parametrize('missing',('anygeometry.nurbs_intersection_curve','internal_owner_dependency'))
def test_optional_owner_module_absence_falls_back_but_nested_failures_propagate(chart_fixture,monkeypatch,missing):
    import builtins
    _model,chart = chart_fixture
    original = builtins.__import__
    def unavailable(name,*args,**kwargs):
        if name == 'anygeometry.nurbs_intersection_curve':
            raise ModuleNotFoundError('missing module sentinel',name=missing)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',unavailable)
    if missing == 'anygeometry.nurbs_intersection_curve':
        assert _batch_chart(chart.evaluate,None) is None
    else:
        with pytest.raises(ModuleNotFoundError,match='missing module sentinel'):
            _batch_chart(chart.evaluate,None)


def test_outer_owner_checker_introduced_during_admission_keeps_serial_path(chart_fixture,monkeypatch):
    import anygeometry.nurbs_intersection_curve as curve_owner
    model,chart = chart_fixture
    before = owner.to_dict(model)
    validate = owner.validate_material_surface_regions_binding
    activated = []
    introduced = lambda phase:None
    def checker():
        if not activated:
            activated.append(True)
            monkeypatch.setattr(chart,'check',introduced)
    def owner_frontier(*args,**kwargs):
        result = validate(*args,**kwargs)
        # Exercise an outer operation checker at a real, completed owner
        # validation frontier; it changes no geometry or numerical results.
        curve_owner._operation_check()
        return result
    monkeypatch.setattr(owner,'validate_material_surface_regions_binding',owner_frontier)
    with intersection_curve_operation(checker):
        assert _batch_chart(chart.evaluate,None) is None
    assert activated == [True] and chart.check is introduced
    assert owner.to_dict(model) == before


@pytest.mark.parametrize('evaluation',(1,2,3))
def test_outer_owner_checker_cannot_add_callback_during_batched_evaluation(chart_fixture,monkeypatch,evaluation):
    import anygeometry.nurbs_intersection_curve as curve_owner
    model,chart = chart_fixture
    uv,triangles,report,settings,fixed = operands(chart,7)
    before = uv.tobytes(),snapshot(report.mesh),owner.to_dict(model)
    original = owner.evaluate_material_surface_region
    calls, activated, introduced_phases = [], [], []
    def checker():
        if len(calls) == evaluation and not activated:
            activated.append(True)
            monkeypatch.setattr(chart,'check',introduced_phases.append)
    def owner_frontier(*args,**kwargs):
        result = original(*args,**kwargs)
        if not kwargs.get('derivatives',False):
            calls.append(len(args[3]))
            curve_owner._operation_check()
        return result
    monkeypatch.setattr(owner,'evaluate_material_surface_region',owner_frontier)
    with intersection_curve_operation(checker),pytest.raises(MeshError,match='acquired an owner callback'):
        repair_mixed_coordinates(uv,triangles,report,settings,chart.evaluate,fixed)
    # Initial full evaluation, current-node batch and final full evaluation
    # each have the same refusal frontier, before any new callback is run.
    assert len(calls) == evaluation and activated == [True]
    assert introduced_phases == []
    assert before == (uv.tobytes(),snapshot(report.mesh),owner.to_dict(model))


def test_generic_forwarder_without_jacobian_keeps_legacy_first_trial(chart_fixture):
    model,chart=chart_fixture
    uv,triangles,report,settings,fixed=operands(chart,1)
    before=uv.tobytes(),snapshot(report.mesh),owner.to_dict(model)
    def forwarding(rows):return chart.evaluate(rows)
    assert _batch_chart(forwarding,None) is None
    made,updated=repair_mixed_coordinates(uv,triangles,report,settings,forwarding,fixed)
    # Existing no-Jacobian first proposal: the free fan node moves to its
    # neighbor centroid. The targeted owner proposal intentionally differs.
    expected=uv.copy()
    expected[4]+=(uv[[0,1,2,3]]-uv[4]).mean(axis=0)
    np.testing.assert_array_equal(made,expected)
    np.testing.assert_array_equal(made[list(fixed)],uv[list(fixed)])
    assert updated.exchange_work==1
    assert snapshot(updated.mesh)==snapshot(report.mesh)
    assert before==(uv.tobytes(),snapshot(report.mesh),owner.to_dict(model))
    actual=MeshCore(chart.evaluate(made),triangles,report.mesh.quad_connectivity)
    legacy=MeshCore(chart.evaluate(expected),triangles,report.mesh.quad_connectivity)
    assert _quality_threshold_report(evaluate_quality(actual),settings)==_quality_threshold_report(evaluate_quality(legacy),settings)
    assert _published_growth(actual,settings.max_element_growth)==_published_growth(legacy,settings.max_element_growth)
    final_quality=_quality_threshold_report(evaluate_quality(legacy),settings)
    _,growth_poor=_published_growth(legacy,settings.max_element_growth)
    assert updated.exchange_truncated==bool(final_quality['poor_element_ids'] or growth_poor)
