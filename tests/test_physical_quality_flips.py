"""Physical flips preserve the exact local material before publication."""
from fractions import Fraction

import numpy as np
import pytest

from anymesher.native_v2 import MutableT3Topology, ComponentSeedRegistry, NativeMeshingOptions, _GeometryLimited
from anymesher.errors import MeshError
from anymesher._physical_t3_refinement import refine_physical_candidate, _alternative_progress
from anymesher.surface_mesh import SurfaceMeshOptions, _make_candidate, _physical_quality_candidate
from anymesher.triangulation import PlanarTriangulation


def exact_double_area(points, cells):
    result = Fraction()
    for row in cells:
        a, b, c = [[Fraction(float(value)) for value in points[node]] for node in row]
        result += abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))
    return result


def test_concave_flip_refuses_exact_material_increase():
    points = np.asarray(((0., 0.), (2., 0.), (.5, .5), (0., 2.)))
    cells = np.asarray(((0, 1, 2), (0, 2, 3)))
    unsafe_replacement = np.asarray(((0, 1, 3), (1, 2, 3)))
    assert exact_double_area(points, cells) == 2
    assert exact_double_area(points, unsafe_replacement) == 6
    topology = MutableT3Topology(points, cells)
    before = topology.points.tobytes(), topology.triangles.tobytes()
    assert not topology.flip_edge((0, 2))
    assert (topology.points.tobytes(), topology.triangles.tobytes()) == before
    assert topology.epoch == 0


def convex(*, protected=(), splittable=None, owners=(17, 17)):
    registry = ComponentSeedRegistry(100)
    topology = MutableT3Topology(np.asarray(((0., 0.), (1., 0.), (1., 1.), (0., 1.))),
        np.asarray(((0, 1, 2), (0, 2, 3))), protected,
        splittable_edges=splittable, seed_registry=registry,
        node_owners=np.asarray((10, 11, 12, 13)), triangle_owners=np.asarray(owners))
    topology.quality_cache[(0, 1, 2)] = (1.,)
    topology._shared_node_ids = {0: (99, 7, Fraction(0))}
    return topology


def state(topology):
    return (topology.points.tobytes(), topology.triangles.tobytes(),
        topology.node_owners.tobytes(), topology.triangle_owners.tobytes(),
        topology.constraint_edges.tobytes(), dict(topology._splittable_intervals),
        dict(topology.shared_node_ids), topology.epoch, dict(topology.quality_cache),
        tuple(topology.free_triangle_ids), id(topology._points), id(topology._triangles),
        id(topology.node_owners), id(topology.triangle_owners), id(topology._topology_index),
        id(topology.quality_cache), dict(topology._seed_registry._values), topology._seed_registry._next)


def test_convex_flip_preserves_exact_area_points_owners_and_constraints():
    protected = ((0, 1), (1, 2), (2, 3))
    topology = convex(protected=protected, splittable={(0, 3): (7, 0, 1)})
    old = state(topology)
    area = exact_double_area(topology.points, topology.triangles)
    assert topology.flip_edge((0, 2))
    assert (1, 3) in topology._topology_index and (0, 2) not in topology._topology_index
    assert exact_double_area(topology.points, topology.triangles) == area
    assert topology.points.tobytes() == old[0] and topology.node_owners.tobytes() == old[2]
    assert topology.constraint_edges.tobytes() == old[4]
    assert topology.shared_node_ids == old[6] and topology._seed_registry._values == old[-2]
    assert topology._seed_registry._next == old[-1]
    assert topology.triangle_owners.tolist() == [17, 17] and topology.epoch == 1
    assert not topology.quality_cache


@pytest.mark.parametrize('kind', ('protected', 'splittable', 'owners', 'replacement'))
def test_ineligible_flip_refuses_without_callback_or_state_change(kind):
    if kind == 'protected':
        topology = convex(protected=((0, 2),))
    elif kind == 'splittable':
        topology = convex(splittable={(0, 2): (7, 0, 1)})
    elif kind == 'owners':
        topology = convex(owners=(17, 18))
    else:
        topology = convex()
        topology = MutableT3Topology(topology.points, ((0, 1, 2), (0, 2, 3), (1, 2, 3)),
                                    seed_registry=topology._seed_registry)
    before = state(topology)
    calls = []
    assert not topology.flip_edge((0, 2), _validate_candidate=lambda *args: calls.append(args))
    assert not calls and state(topology) == before


@pytest.mark.parametrize('points,cells', (
    (((0., 0.), (1., 1.), (0., 1.), (1., 0.)), ((0, 1, 2), (0, 3, 2))),
    (((0., 0.), (1., 0.), (0., 1.), (-1., 0.)), ((0, 1, 2), (0, 2, 3))),
))
def test_bowtie_and_collinear_diagonal_refused(points, cells):
    topology = MutableT3Topology(points, cells)
    before = topology.triangles.tobytes(), topology.points.tobytes(), topology.epoch
    assert not topology.flip_edge((0, 2))
    assert (topology.triangles.tobytes(), topology.points.tobytes(), topology.epoch) == before


def test_small_determinant_signs_are_not_multiplied():
    topology = convex()
    points = topology.points * 1e-100
    topology = MutableT3Topology(points, topology.triangles)
    assert topology.flip_edge((0, 2))
    assert exact_double_area(topology.points, topology.triangles) == exact_double_area(points, ((0, 1, 2), (0, 2, 3)))


@pytest.mark.parametrize('stage', ('validator', 'native-v2 edge flip start',
                                  'native-v2 incidence update', 'native-v2 edge flip commit'))
@pytest.mark.parametrize('error_type', (RuntimeError, MeshError, _GeometryLimited))
def test_callback_failure_preserves_identity_and_exact_state(stage, error_type):
    topology = convex(protected=((0, 1),), splittable={(0, 3): (7, 0, 1)})
    before = state(topology)
    error = error_type('cancel or reject flip')
    def validate(points, cells):
        assert state(topology) == before
        assert not points.flags.writeable and not cells.flags.writeable
        if stage == 'validator':
            raise error
    def cancel(phase):
        assert state(topology) == before
        if phase == stage:
            raise error
    with pytest.raises(error_type) as caught:
        topology.flip_edge((0, 2), _validate_candidate=validate, cancellation_check=cancel)
    assert caught.value is error and state(topology) == before


def test_mutated_validator_backing_cannot_change_publication():
    topology = convex()
    points = topology.points.tobytes()
    def validate(x, cells):
        x.setflags(write=True)
        cells.setflags(write=True)
        x[:] = -100
        cells[:] = 0
    assert topology.flip_edge((0, 2), _validate_candidate=validate)
    assert topology.points.tobytes() == points and (1, 3) in topology._topology_index


def test_native_validation_refusal_rolls_back_after_callbacks(monkeypatch):
    topology = convex()
    before = state(topology)
    def reject():
        raise MeshError('native validation refusal')
    monkeypatch.setattr(topology, 'validate', reject)
    assert not topology.flip_edge((0, 2))
    assert state(topology) == before


def physical_fixture(*, budget=2, two=True):
    points = np.asarray(((0., 0.), (1., 0.), (1., .1), (0., .1),
                         (3., 0.), (5., 0.), (5., 1.), (3., .55)))
    cells = np.asarray(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)))
    if not two:
        points, cells = points[4:].copy(), cells[2:].copy()-4
    incidence = {}
    for row in cells:
        for a,b in zip(row,np.roll(row,-1)):
            edge=tuple(sorted((int(a),int(b))))
            incidence[edge]=incidence.get(edge,0)+1
    boundary=np.asarray([edge for edge,count in sorted(incidence.items()) if count==1])
    triangulation=PlanarTriangulation(points,cells,boundary,boundary,
        np.empty((0,2),dtype=int),np.arange(len(points)),())
    calls=[]
    def owner(rows):
        calls.append(rows.copy())
        return np.column_stack((rows,np.zeros(len(rows))))
    settings=SurfaceMeshOptions(min_angle=15.,prefer_quality_policy=True,
        native_options=NativeMeshingOptions(max_topology_operations=budget+1,max_insertions=1))
    candidate=_physical_quality_candidate(_make_candidate(points,cells,settings=settings),settings,owner)
    calls.clear()
    return candidate,triangulation,settings,owner,calls


def work_report():
    # Public limits stay positive; an earlier insertion consumed all headroom.
    return dict(topology_operations=1,insertions=1,shared_segment_splits=0,shared_nodes=[])


@pytest.mark.parametrize('budget', (0, 1, 2, 3, 4))
def test_flip_budget_determinism_cache_reset_and_no_owner_queries(budget, monkeypatch):
    candidate, triangulation, settings, owner, calls=physical_fixture(budget=budget)
    before=candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()
    original=MutableT3Topology.flip_edge
    seen=[]
    def flip(topology,edge,**kwargs):
        seen.append(edge)
        return original(topology,edge,**kwargs)
    monkeypatch.setattr(MutableT3Topology,'flip_edge',flip)
    result, output, receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        owner,{},ComponentSeedRegistry(100),allow_physical_flips=True)
    expected=[(0,2),(4,6),(0,2),(5,7)][:budget]
    assert seen == expected
    assert receipt['topology_operations'] == budget+1
    assert receipt['physical_quality_flip_attempts'] == budget
    assert receipt['physical_quality_attempts'] == budget
    assert receipt['physical_quality_flips'] == int(budget>=2)
    assert receipt['physical_quality_flip_refused_attempts'] == budget-int(budget>=2)
    assert receipt['physical_quality_bisections'] == 0 and receipt['insertions'] == 1
    assert len(calls)==1 and len(calls[0])==len(candidate.points)
    assert result.points.tobytes()==candidate.points.tobytes()
    assert output.segments.tobytes()==triangulation.segments.tobytes()
    assert result.flips==candidate.flips+int(budget>=2)
    if budget>=2:
        assert _alternative_progress(candidate.report,result.report)
    else:
        assert result.triangles.tobytes()==candidate.triangles.tobytes()
    assert receipt['physical_quality_flip_candidates_exhausted'] == (budget==4)
    assert receipt['physical_quality_stop_reason'] == ('no_progress_for_flip_candidates_insertion_budget'
                                                      if budget==4 else 'topology_budget')
    assert (candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes())==before


def test_default_zero_insertion_route_is_unchanged():
    candidate,triangulation,settings,owner,calls=physical_fixture()
    result,_,report=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        owner,{},ComponentSeedRegistry(100))
    assert result.points.tobytes()==candidate.points.tobytes()
    assert result.triangles.tobytes()==candidate.triangles.tobytes()
    assert report['topology_operations']==1 and report['physical_quality_attempts']==0
    assert report['physical_quality_stop_reason']=='insertion_budget'
    assert not any(key.startswith('physical_quality_flip') for key in report)


@pytest.mark.parametrize('value', (1,0,None,'yes',np.bool_(True)))
def test_flip_option_is_strict_boolean(value):
    candidate,triangulation,settings,owner,calls=physical_fixture()
    with pytest.raises(MeshError,match='invalid physical flip option'):
        refine_physical_candidate(candidate,triangulation,settings,work_report(),
            owner,{},ComponentSeedRegistry(100),allow_physical_flips=value)
    assert not calls


def test_all_flip_refusals_preserve_constraints_registry_and_input_exactly():
    from dataclasses import replace
    candidate,triangulation,settings,owner,calls=physical_fixture(budget=2)
    # The only progressing diagonal is a mandatory internal physical trace.
    triangulation=replace(triangulation,segments=np.vstack((triangulation.segments,(4,6))))
    registry=ComponentSeedRegistry(100)
    before=candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()
    result,output,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        owner,{},registry,allow_physical_flips=True)
    assert result.points.tobytes()==before[0] and result.triangles.tobytes()==before[1]
    assert set(map(tuple,output.segments))==set(map(tuple,triangulation.segments))
    assert receipt['physical_quality_flip_attempts']==receipt['physical_quality_flip_refused_attempts']==1
    assert receipt['physical_quality_flips']==0 and receipt['physical_quality_flip_candidates_exhausted']
    assert receipt['physical_quality_stop_reason']=='no_progress_for_flip_candidates_insertion_budget'
    assert not registry.assigned_node_ids and len(calls)==1


@pytest.mark.parametrize('kind', ('cancel','scorer'))
@pytest.mark.parametrize('error_type', (RuntimeError,MeshError,_GeometryLimited))
def test_physical_flip_failure_after_prior_refusal_never_publishes(monkeypatch,kind,error_type):
    import anymesher._physical_t3_refinement as physical
    import anymesher.surface_mesh as surface
    candidate,triangulation,settings,owner,calls=physical_fixture(budget=2)
    registry=ComponentSeedRegistry(100)
    topologies=[]
    original_topology=physical.MutableT3Topology
    def create(*args,**kwargs):
        topology=original_topology(*args,**kwargs)
        topologies.append((topology,state(topology)))
        return topology
    monkeypatch.setattr(physical,'MutableT3Topology',create)
    error=error_type('physical flip stale or cancellation')
    original_score=surface._physical_quality_candidate_from_xyz
    scoring=[]
    def score(*args):
        scoring.append(True)
        if len(scoring)==3 and kind=='scorer':
            raise error
        return original_score(*args)
    monkeypatch.setattr(surface,'_physical_quality_candidate_from_xyz',score)
    def cancel(phase):
        if phase=='native-v2 edge flip commit' and kind=='cancel':
            raise error
    before=candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()
    with pytest.raises(error_type) as caught:
        physical.refine_physical_candidate(candidate,triangulation,settings,work_report(),
            owner,{},registry,cancel,allow_physical_flips=True)
    assert caught.value is error and len(topologies)==1
    topology,snapshot=topologies[0]
    assert state(topology)==snapshot and not registry.assigned_node_ids
    assert len(scoring)==3 and len(calls)==1
    assert (candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes())==before


@pytest.mark.parametrize('analytic', (False,True))
def test_surface_physical_flips_only_for_exact_bound_analytic_evaluator(monkeypatch,analytic):
    import anymesher.surface_mesh as surface
    import anymesher._physical_t3_refinement as physical
    import anymesher._frontal_transition_quality as transition
    from test_joint_triangle_owner_batches import analytic_chart
    from anygeometry import to_dict
    model,chart=analytic_chart()
    before=to_dict(model)
    seen=[]
    monkeypatch.setattr(transition,'repair_frontal_transition',
        lambda candidate,protected,settings,report,*args,**kwargs:(candidate,report))
    def capture(candidate,triangulation,settings,report,*args,**kwargs):
        seen.append(kwargs['allow_physical_flips'])
        return candidate,triangulation,report
    monkeypatch.setattr(physical,'refine_physical_candidate',capture)
    surface.mesh_planar_surface(((0.,0.),(1.,0.),(1.,1.),(0.,1.)),interior_points=np.asarray(((.37,.43),)),
        options=SurfaceMeshOptions(target_size=1.,recombine=False,backend='python',min_angle=45.,
            native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
                max_insertions=1,max_topology_operations=8)),
        _metric_to_physical=chart.evaluate if analytic else lambda rows:chart.evaluate(rows),
        _metric_jacobian=np.asarray(((1.,0.),(0.,1.),(0.,0.))),_preserve_spatial_refinement=True)
    assert seen==[analytic] and to_dict(model)==before


def test_stale_owner_after_cached_flip_cannot_return_a_candidate(monkeypatch):
    from anygeometry import GeometryModel,BezierDirectrix,ExtrudedSurface,GeometryError
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model=GeometryModel()
    vertices=model.add_points(((0.,0.,0.),(.5,0.,0.),(1.,.0001,0.)))
    edge=model.add_spline(vertices[0],vertices[1:-1],vertices[-1])
    face=model.extrude((edge,),(0.,0.,1.))[0]
    model.set_face_surface(face,ExtrudedSurface(BezierDirectrix(((0.,0.,0.),(.5,0.,0.),(1.,.0001,0.))),
                                              (0.,0.,1.)))
    chart=AnalyticMetricChart(model,face)
    candidate,triangulation,settings,_,_=physical_fixture(budget=1,two=False)
    from dataclasses import replace
    points=candidate.points*.4 @ chart.transform
    triangulation=replace(triangulation,points=points)
    owner_calls=[]
    original_evaluate=type(model).evaluate_face_many
    def evaluate(owner,*args,**kwargs):
        owner_calls.append(True)
        return original_evaluate(owner,*args,**kwargs)
    monkeypatch.setattr(type(model),'evaluate_face_many',evaluate)
    candidate=_physical_quality_candidate(_make_candidate(points,candidate.triangles,settings=settings),
                                          settings,chart.evaluate)
    original_flip=MutableT3Topology.flip_edge
    changed=[]
    def flip(topology,edge,**kwargs):
        result=original_flip(topology,edge,**kwargs)
        if result:
            changed.append(True)
            model.add_point(2.,2.,2.)
        return result
    monkeypatch.setattr(MutableT3Topology,'flip_edge',flip)
    before=candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes()
    registry=ComponentSeedRegistry(100)
    with pytest.raises(GeometryError,match='stale'):
        refine_physical_candidate(candidate,triangulation,settings,work_report(),
            chart.evaluate,{},registry,allow_physical_flips=True)
    assert changed==[True] and len(owner_calls)==2 and not registry.assigned_node_ids
    assert (candidate.points.tobytes(),candidate.triangles.tobytes(),triangulation.segments.tobytes())==before


def test_opt_in_preserves_successful_ordinary_bisection_before_any_flip():
    from dataclasses import replace
    from test_physical_primary_progress import progress_cavity
    candidate,triangulation,settings,owner,intervals,_=progress_cavity()
    triangulation=replace(triangulation,segments=triangulation.boundary_segments,
                          mandatory_segments=np.empty((0,2),dtype=int))
    settings=replace(settings,native_options=NativeMeshingOptions(max_topology_operations=1))
    report=dict(topology_operations=0,insertions=0,shared_segment_splits=0,shared_nodes=[])
    default,default_tri,default_report=refine_physical_candidate(candidate,triangulation,settings,report,
        owner,intervals,ComponentSeedRegistry(500))
    result,output,receipt=refine_physical_candidate(candidate,triangulation,settings,report,
        owner,intervals,ComponentSeedRegistry(500),allow_physical_flips=True)
    assert result.points.tobytes()==default.points.tobytes()
    assert result.triangles.tobytes()==default.triangles.tobytes() and result.report==default.report
    assert output.segments.tobytes()==default_tri.segments.tobytes()
    assert all(receipt[key]==value for key,value in default_report.items())
    assert receipt['physical_quality_flip_attempts']==receipt['physical_quality_flips']==0


def test_bisection_refusals_cannot_renew_exhausted_flip_pool(monkeypatch):
    from dataclasses import replace
    candidate,triangulation,settings,owner,calls=physical_fixture(budget=6)
    settings=replace(settings,native_options=NativeMeshingOptions(max_topology_operations=6,max_insertions=2))
    attempted=[]
    def refuse(topology,edge,**kwargs):
        attempted.append(edge)
        raise _GeometryLimited('bounded independent primitive refusal')
    monkeypatch.setattr(MutableT3Topology,'bisect_interior_edge',refuse)
    result,_,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        owner,{},ComponentSeedRegistry(100),allow_physical_flips=True)
    # Four protected boundary skips and one interior refusal consume the five
    # remaining operations; flip admission cannot renew that exhausted pool.
    assert receipt['topology_operations']==6
    assert receipt['physical_quality_flip_attempts']==0
    assert receipt['physical_quality_stop_reason']=='topology_budget'
    assert receipt['physical_quality_bisections']==0 and len(calls)==1
    assert result.triangles.tobytes()==candidate.triangles.tobytes()


def test_exhausted_bisection_candidates_allow_guarded_flip_with_remaining_work(monkeypatch):
    from dataclasses import replace
    candidate,triangulation,settings,owner,calls=physical_fixture(budget=10)
    settings=replace(settings,native_options=NativeMeshingOptions(max_topology_operations=11,max_insertions=2))
    attempted=[]
    def refuse(topology,edge,**kwargs):
        attempted.append(edge)
        raise _GeometryLimited('bounded independent primitive refusal')
    monkeypatch.setattr(MutableT3Topology,'bisect_interior_edge',refuse)
    result,_,receipt=refine_physical_candidate(candidate,triangulation,settings,work_report(),
        owner,{},ComponentSeedRegistry(100),allow_physical_flips=True)
    assert set(attempted)=={(0,2),(4,6)}
    assert receipt['topology_operations']==11 and receipt['physical_quality_attempts']==10
    assert receipt['physical_quality_flip_attempts']==2 and receipt['physical_quality_flips']==1
    assert receipt['physical_quality_refused_attempts']==9
    assert receipt['physical_quality_bisections']==0 and receipt['insertions']==1 and len(calls)==1
    assert _alternative_progress(candidate.report,result.report)
    assert not receipt['physical_quality_candidate_cells_exhausted']
