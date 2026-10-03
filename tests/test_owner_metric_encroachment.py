"""Metric encroachment must never remove a registered constraint."""
from fractions import Fraction
import numpy as np
import pytest

import anymesher.native_v2 as native
from anymesher.errors import MeshError
from anymesher.native_cpp import COMPILED_NATIVE_V2_AVAILABLE


def diamond():
    points=np.array(((-1.,0.),(1.,0.),(0.,1.),(0.,-1.),(-3.,0.),(3.,0.),(0.,3.),(0.,-3.)))
    triangles=[(0,1,2),(0,3,1)]
    inner=(0,3,1,2);outer=(4,7,5,6)
    for i in range(4):
        j=(i+1)%4
        triangles.extend(((inner[i],outer[i],outer[j]),(inner[i],outer[j],inner[j])))
    return native.MutableT3Topology(points,triangles,
        protected_edges=((4,7),(7,5),(5,6),(6,4)),splittable_edges={(0,1):(7,0,1)},
        seed_registry=native.ComponentSeedRegistry(100),
        node_owners=np.arange(8),triangle_owners=np.arange(10)+20)


def edges(cells):
    return {tuple(sorted((int(a),int(b)))) for row in cells for a,b in zip(row,np.roll(row,-1))}


def snapshot(topology):
    return (topology._points.tobytes(),topology._triangles.tobytes(),topology.node_owners.tobytes(),
        topology.triangle_owners.tobytes(),topology.epoch,dict(topology.quality_cache),
        dict(topology._splittable_intervals),dict(topology.shared_node_ids),
        topology._seed_registry.assigned_node_ids,id(topology._topology_index))


@pytest.mark.parametrize('compiled',(False,True))
def test_metric_permitted_diamond_cavity_refuses_without_losing_splittable_edge(monkeypatch,compiled):
    if compiled and not COMPILED_NATIVE_V2_AVAILABLE: pytest.skip('optional native-v2 extension absent')
    if not compiled: monkeypatch.setattr(native,'native_mutable_t3_insert',lambda *args,**kwargs:None)
    topology=diamond();candidate=np.array((0.,.75))
    topology.quality_cache[(0,1,2)]=(1.,)
    assert float((candidate-topology._points[0]) @ (candidate-topology._points[1])) == -.4375
    assert float((candidate-topology._points[0]) @ np.diag((1.,4.)) @ (candidate-topology._points[1])) == 1.25
    old_cells,_,_=topology._python_insert_with_owners(candidate)
    assert (0,1) not in edges(old_cells)
    before=snapshot(topology)
    with pytest.raises(native._GeometryLimited,match='cavity would remove'):
        topology.insert_point(candidate,preserve_splittable=True)
    assert snapshot(topology)==before


@pytest.mark.parametrize('scale',(1., .5, .25, 2.**-12, 2.**-24))
def test_metric_equilateral_child_base_does_not_self_encroach(scale):
    length=.0013*scale
    points=np.array(((0.,0.),(length,0.),(.3,.4)))
    metric=np.diag((1.,16.))
    tensors=np.broadcast_to(metric,(3,2,2)).copy()
    point=native._offcentre(points,tensors,.5,short_edge_metric=True)
    euclidean=float((point-points[0]) @ (point-points[1]))
    elliptic=native._metric_segment_penetration(point,*points[:2],tensors[:2])
    assert euclidean < 0. < elliptic
    assert elliptic == pytest.approx(length**2/2.,rel=2e-15,abs=0.)
    # h=sqrt(3)*L/(2k), hence metric diameter dot=L²/2 at every child scale.
    assert point[1] == pytest.approx(np.sqrt(3.)*length/8.,rel=2e-15,abs=0.)


def test_skew_metric_encroachment_and_placement_are_affine_consistent():
    points=np.array(((0.,0.),(.0013,0.),(.3,.4)))
    metric=np.array(((2.,.8),(.8,8.)))
    tensors=np.broadcast_to(metric,(3,2,2)).copy()
    point=native._offcentre(points,tensors,.5,short_edge_metric=True)
    transform=np.array(((1.2,.4),(-.3,.8)))
    inverse=np.linalg.inv(transform)
    transformed_metric=inverse @ metric @ inverse.T
    transformed_points=points @ transform
    transformed=native._offcentre(transformed_points,np.broadcast_to(transformed_metric,(3,2,2)),.5,
        short_edge_metric=True)
    np.testing.assert_allclose(transformed,point @ transform,rtol=2e-14,atol=1e-18)
    reference=native._metric_segment_penetration(point,*points[:2],tensors[:2])
    actual=native._metric_segment_penetration(transformed,*transformed_points[:2],
        np.broadcast_to(transformed_metric,(2,2,2)))
    assert actual == pytest.approx(reference,rel=2e-14,abs=0.) and reference>0.


@pytest.mark.parametrize('factor',(.25,1.,4.,64.))
@pytest.mark.parametrize('point',((0.,.75),(0.,1.25),(.3,.5),(2.,0.)))
def test_scalar_metric_preserves_euclidean_encroachment_sign(factor,point):
    a,b=np.array((-1.,0.)),np.array((1.,0.))
    euclidean=float((np.array(point)-a) @ (np.array(point)-b))
    actual=native._metric_segment_penetration(point,a,b,np.broadcast_to(np.eye(2)*factor,(2,2,2)))
    assert np.sign(actual)==np.sign(euclidean)
    assert actual == euclidean*factor


def test_variable_endpoint_mean_is_frozen_and_can_still_encroach_own_base():
    points=np.array(((0.,0.),(.0013,0.),(.3,.4)))
    tensors=np.array((np.diag((100.,1.)),np.diag((100.,1.)),np.diag((1.,400.))))
    before=(points.tobytes(),tensors.tobytes())
    point=native._offcentre(points,tensors,.5,short_edge_metric=True)
    endpoint=native._metric_segment_penetration(point,*points[:2],tensors[:2])
    cell_mean=np.broadcast_to(tensors.mean(axis=0),(2,2,2))
    assert endpoint<0.<native._metric_segment_penetration(point,*points[:2],cell_mean)
    assert endpoint == float((point-points[0]) @ (.5*tensors[0]+.5*tensors[1]) @ (point-points[1]))
    assert (points.tobytes(),tensors.tobytes())==before


@pytest.mark.parametrize('metric',(np.diag((0.,1.)),np.diag((-1.,1.)),np.array(((1.,np.nan),(np.nan,1.)))))
def test_invalid_metric_cannot_admit_candidate(metric):
    with pytest.raises(MeshError):
        native._metric_segment_penetration((0.,.75),(-1.,0.),(1.,0.),np.broadcast_to(metric,(2,2,2)))


def exact_area(topology):
    total=Fraction(0)
    for row in topology._triangles:
        a,b,c=[[Fraction(float(value)) for value in topology._points[node]] for node in row]
        total+=((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
    return total


@pytest.mark.parametrize('compiled',(False,True))
def test_preserving_success_retains_constraints_station_bytes_owners_and_exact_area(monkeypatch,compiled):
    if compiled and not COMPILED_NATIVE_V2_AVAILABLE: pytest.skip('optional native-v2 extension absent')
    if not compiled: monkeypatch.setattr(native,'native_mutable_t3_insert',lambda *args,**kwargs:None)
    topology=diamond();before=snapshot(topology);area=exact_area(topology)
    constraints=topology.constraint_edges.tobytes()
    report=topology.insert_point((0.,2.2),owner=91,preserve_splittable=True)
    assert topology.epoch==report['epoch']==1
    assert topology._points[:8].tobytes()==before[0] and topology.node_owners[:8].tobytes()==before[2]
    assert topology.node_owners[8]==91
    assert topology.constraint_edges.tobytes()==constraints and (0,1) in edges(topology._triangles)
    assert exact_area(topology)==area
    assert topology._splittable_intervals==before[6] and topology._seed_registry.assigned_node_ids==()
    assert all(owner==91 for row,owner in zip(topology._triangles,topology.triangle_owners) if 8 in row)


@pytest.mark.parametrize('phase',('native-v2 mutable insertion start','native-v2 mutable insertion commit'))
@pytest.mark.parametrize('compiled',(False,True))
def test_insertion_cancellation_is_exact_and_does_not_publish(monkeypatch,phase,compiled):
    if compiled and not COMPILED_NATIVE_V2_AVAILABLE: pytest.skip('optional native-v2 extension absent')
    if not compiled: monkeypatch.setattr(native,'native_mutable_t3_insert',lambda *args,**kwargs:None)
    topology=diamond();before=snapshot(topology)
    error=native._GeometryLimited('callback cancellation identity')
    def cancel(location):
        if location==phase: raise error
    with pytest.raises(native._GeometryLimited) as caught:
        topology.insert_point((0.,2.2),preserve_splittable=True,cancellation_check=cancel)
    assert caught.value is error and snapshot(topology)==before


def test_collinear_splittable_candidate_refuses_locally_before_publication(monkeypatch):
    monkeypatch.setattr(native,'native_mutable_t3_insert',lambda *args,**kwargs:None)
    topology=diamond();before=snapshot(topology)
    with pytest.raises(native._GeometryLimited,match='registered constraint'):
        topology.insert_point((0.,0.),preserve_splittable=True)
    assert snapshot(topology)==before


def test_compiled_dispatch_receives_all_constraints_only_when_opted_in(monkeypatch):
    submitted=[]
    def kernel(points,cells,protected,candidate,*args,**kwargs):
        submitted.append(protected.copy());return None
    monkeypatch.setattr(native,'native_mutable_t3_insert',kernel)
    for enabled in (False,True):
        topology=diamond()
        topology.insert_point((0.,2.2),preserve_splittable=enabled)
    assert (0,1) not in {tuple(edge) for edge in submitted[0]}
    assert {tuple(edge) for edge in submitted[1]}=={(0,1),(4,6),(4,7),(5,6),(5,7)}


def test_default_primitive_bytes_report_and_callback_sequence_unchanged(monkeypatch):
    monkeypatch.setattr(native,'native_mutable_t3_insert',lambda *args,**kwargs:None)
    outcomes=[]
    for supplied in ({},{'preserve_splittable':False}):
        topology=diamond();phases=[]
        report=topology.insert_point((0.,2.2),cancellation_check=phases.append,**supplied)
        outcomes.append((snapshot(topology)[:-1],report,phases))
    assert outcomes[0]==outcomes[1]


def owner_chart():
    from anygeometry import GeometryModel,BezierDirectrix,ExtrudedSurface
    from anymesher._analytic_metric_chart import AnalyticMetricChart
    model=GeometryModel();controls=((0.,0.,0.),(.5,.02,0.),(1.,0.,0.))
    vertices=model.add_points(controls)
    edge=model.add_spline(vertices[0],vertices[1:-1],vertices[-1])
    face=model.extrude((edge,),(0.,0.,1.))[0]
    model.set_face_surface(face,ExtrudedSurface(BezierDirectrix(controls),(0.,0.,1.)))
    return model,AnalyticMetricChart(model,face)


def frontal_seed():
    from anymesher.triangulation import PlanarTriangulation
    points=[(-.05,0.),(.05,0.),(0.,.05),(0.,-.05)]
    outer=np.array((( -1.,0.),(-1.,-1.),(0.,-1.),(1.,-1.),(1.,0.),(1.,1.),(0.,1.),(-1.,1.)))
    outer[[1,3,5,7]]/=np.sqrt(2.)
    points=np.vstack((points,outer*.3))+.5
    triangles=[(0,1,2),(0,3,1)];inner=(0,3,1,2)
    for i in range(4):
        a,b=inner[i],inner[(i+1)%4];x=4+2*i;y=4+(2*i+1)%8;z=4+(2*i+2)%8
        triangles.extend(((a,x,y),(a,y,z),(a,z,b)))
    boundary=np.array([tuple(sorted((4+i,4+(i+1)%8))) for i in range(8)],dtype=np.int64)
    return PlanarTriangulation(points=points,triangles=np.asarray(triangles,dtype=np.int64),
        segments=np.vstack((boundary,(0,1))),boundary_segments=boundary,
        mandatory_segments=np.array(((0,1),),dtype=np.int64),outer_loop=np.arange(4,12),hole_loops=())


def frontal(model,chart,*,short=True,generic=False,proposal=None,size=2.,shared=True,cancel=None,operations=12,
            seed_override=None,shared_edges=None,insertions=1):
    from anymesher import ImportedMetricSamples,MetricFieldSpec,IsotropicMetricControl
    sample=ImportedMetricSamples(str(model.model_id),model.revision,((.5,0.,.5),),
        (((1.,0.,0.),(0.,1.,0.),(0.,0.,4.)),))
    options=native.NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',
        metric_field=MetricFieldSpec(IsotropicMetricControl(size),imported_samples=(sample,)),
        max_insertions=insertions,max_topology_operations=operations,cancellation_interval=1)
    registry=native.ComponentSeedRegistry(100)
    evaluator=(lambda rows:chart.evaluate(rows)) if generic else chart.evaluate
    seed=frontal_seed() if seed_override is None else seed_override
    result,report=native.frontal_delaunay_refine(seed,options,target_size=size,
        model_uuid=str(model.model_id),geometry_revision=model.revision,
        metric_to_physical=evaluator,metric_jacobian=chart.jacobians,short_edge_offcentre=short,
        automatically_seeded_shared_segments=(shared_edges if shared_edges is not None else
            {(0,1):(7,0,1)} if shared else None),
        component_seed_registry=registry,_split_proposal=proposal,cancellation_check=cancel)
    return seed,result,report,registry


def fixed_proposal(monkeypatch,point):
    monkeypatch.setattr(native,'_offcentre',lambda *args,**kwargs:np.asarray(point))
    monkeypatch.setattr(native,'_circumcentre',lambda *args:None)


def test_owner_route_metric_permitted_cavity_refuses_instead_of_recursive_split(monkeypatch):
    from anygeometry import to_dict
    model,chart=owner_chart();source=to_dict(model)
    fixed_proposal(monkeypatch,(.5,.5375))
    original,result,report,registry=frontal(model,chart)
    assert report['owner_metric_encroachment'] is True
    assert report['topology_operations']>0 and report['geometry_limited_regions']>0
    assert report['shared_segment_splits']==report['insertions']==0 and registry.assigned_node_ids==()
    assert original.points.tobytes()==result.points.tobytes()
    assert (0,1) in edges(result.triangles)
    assert original.segments.tobytes()==result.segments.tobytes() or set(map(tuple,original.segments))==set(map(tuple,result.segments))
    assert to_dict(model)==source
    # The same candidate under the established Euclidean policy requests AB's station.
    _,_,legacy,old_registry=frontal(model,chart,short=False)
    assert 'owner_metric_encroachment' not in legacy
    assert legacy['shared_segment_splits']==1 and old_registry.assigned_node_ids==(100,)


def test_protected_segment_keeps_euclidean_blocker(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    calls=[]
    original=native._metric_segment_penetration
    def metric(*args,**kwargs): calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(native,'_metric_segment_penetration',metric)
    seed,result,report,registry=frontal(model,chart,shared=False)
    assert not calls and report['owner_metric_encroachment']
    assert report['insertions']==report['shared_segment_splits']==0
    assert seed.points.tobytes()==result.points.tobytes() and registry.assigned_node_ids==()


@pytest.mark.parametrize('size,point',((2.,(.5,.51)),(.02,(.5,.5375))))
def test_genuine_metric_encroachment_or_size_split_uses_exact_reserved_station(monkeypatch,size,point):
    model,chart=owner_chart();fixed_proposal(monkeypatch,point)
    proposals=[]
    def station(*args):
        proposals.append(args);return Fraction(1,2),(.5,.5),23
    seed,result,report,registry=frontal(model,chart,proposal=station,size=size)
    assert proposals==[(7,Fraction(0),Fraction(1))]
    assert registry.assigned_node_ids==(23,)
    assert report['shared_segment_splits']==report['reserved_node_reuses']==report['staged_point_insertions']==1
    assert report['insertions']==0 and report['topology_operations']==1
    assert result.points[:len(seed.points)].tobytes()==seed.points.tobytes()
    assert result.points[-1].tobytes()==np.array((.5,.5)).tobytes()
    assert (0,1) not in edges(result.triangles)
    assert (0,len(seed.points)) in edges(result.triangles) and (1,len(seed.points)) in edges(result.triangles)
    assert report['shared_nodes'][0]['station']==(1,2)


def test_refused_segment_keeps_euclidean_blocker_and_no_registry_publication(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    calls=[];metric_calls=[]
    def refuse(*args): calls.append(args);return None
    original=native._metric_segment_penetration
    def metric(*args,**kwargs):metric_calls.append(args);return original(*args,**kwargs)
    monkeypatch.setattr(native,'_metric_segment_penetration',metric)
    seed,result,report,registry=frontal(model,chart,proposal=refuse,size=.02)
    assert calls==[(7,Fraction(0),Fraction(1))] and not metric_calls
    assert report['refused_shared_edges']==[(0,1)]
    assert registry.assigned_node_ids==()
    assert seed.points.tobytes()==result.points.tobytes() and report['shared_segment_splits']==0


def test_generic_short_edge_path_keeps_euclidean_policy(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    _,_,report,registry=frontal(model,chart,generic=True)
    assert 'owner_xyz_cache' not in report and 'owner_metric_encroachment' not in report
    assert report['shared_segment_splits']==1 and registry.assigned_node_ids==(100,)


def test_owner_frontal_private_callback_error_propagates_from_insertion(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    error=native._GeometryLimited('owner route cancellation')
    def cancel(phase):
        if phase=='native-v2 mutable insertion start':raise error
    with pytest.raises(native._GeometryLimited) as caught:frontal(model,chart,cancel=cancel)
    assert caught.value is error


def test_validated_epoch_mean_keeps_exact_arithmetic_without_repeated_spd_solves(monkeypatch):
    import anymesher.metric as metric
    tensors=np.array((((2.,.8),(.8,8.)),((3.,.4),(.4,5.))))
    calls=[];validate=metric._validate_spd
    def observed(*args,**kwargs):calls.append(args);return validate(*args,**kwargs)
    monkeypatch.setattr(metric,'_validate_spd',observed)
    expected=native._metric_segment_penetration((.1,.7),(-1.,0.),(1.,0.),tensors)
    assert len(calls)==3;calls.clear()
    mean=native._metric_segment_mean(tensors)
    assert len(calls)==3 and not mean.flags.writeable;calls.clear()
    actual=native._metric_segment_penetration((.1,.7),(-1.,0.),(1.,0.),None,_mean=mean)
    assert actual==expected and not calls


def test_epoch_mean_validation_is_once_per_edge_not_per_proposal(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    means=[];products=[]
    old_mean=native._metric_segment_mean;old_product=native._metric_segment_penetration
    def mean(tensors):
        value=old_mean(tensors);means.append(value);return value
    def product(*args,**kwargs):
        products.append(kwargs['_mean']);return old_product(*args,**kwargs)
    monkeypatch.setattr(native,'_metric_segment_mean',mean)
    monkeypatch.setattr(native,'_metric_segment_penetration',product)
    _,_,report,_=frontal(model,chart)
    assert len(means)==1 and products and all(value is means[0] for value in products)
    assert report['shared_segment_splits']==0


@pytest.mark.parametrize('tied',(False,True))
def test_multiple_metric_encroachments_rank_metric_values_then_edge_ids(monkeypatch,tied):
    from dataclasses import replace
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.49,.505))
    original=frontal_seed()
    original=replace(original,segments=np.vstack((original.segments,(0,2))),
        mandatory_segments=np.vstack((original.mandatory_segments,(0,2))))
    if tied: monkeypatch.setattr(native,'_metric_segment_penetration',lambda *args,**kwargs:-1.)
    selected=[]
    def station(owner,lower,upper):
        selected.append(owner);return Fraction(1,2),(.5,.5),23
    outcomes=[]
    for segments in (original.segments,original.segments[::-1].copy()):
        selected.clear()
        _,result,report,registry=frontal(model,chart,proposal=station,
            seed_override=replace(original,segments=segments),shared_edges={(0,2):(8,0,1),(0,1):(7,0,1)})
        assert selected==[7] and registry.assigned_node_ids==(23,)
        outcomes.append((result.points.tobytes(),result.triangles.tobytes(),report['shared_nodes']))
    assert outcomes[0]==outcomes[1]


def test_cavity_refusal_respects_existing_single_operation_pool(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    seed,result,report,registry=frontal(model,chart,operations=1)
    assert report['topology_operations']==1 and report['shared_segment_splits']==report['insertions']==0
    assert registry.assigned_node_ids==() and seed.points.tobytes()==result.points.tobytes()


@pytest.mark.skipif(not COMPILED_NATIVE_V2_AVAILABLE,reason='optional native-v2 extension absent')
def test_constraint_preserving_compiled_and_python_success_bytes_match(monkeypatch):
    kernel=native.native_mutable_t3_insert
    outcomes=[]
    for implementation in (lambda *args,**kwargs:None,kernel):
        monkeypatch.setattr(native,'native_mutable_t3_insert',implementation)
        topology=diamond();topology.insert_point((0.,2.2),owner=91,preserve_splittable=True)
        outcomes.append((snapshot(topology)[:-1],topology.constraint_edges.tobytes(),exact_area(topology)))
    assert outcomes[0]==outcomes[1]


def test_size_required_children_rebuild_means_and_keep_exact_station_intervals(monkeypatch):
    model,chart=owner_chart();fixed_proposal(monkeypatch,(.5,.5375))
    old_mean=native._metric_segment_mean;means=[];stations=[]
    def mean(tensors):value=old_mean(tensors);means.append(value);return value
    monkeypatch.setattr(native,'_metric_segment_mean',mean)
    def station(owner,lower,upper):
        value=(lower+upper)/2
        stations.append((owner,lower,upper,value))
        return value,(.45+.1*float(value),.5),None
    seed,result,report,registry=frontal(model,chart,proposal=station,size=.02,insertions=2,operations=2)
    # The chart's binary decimal endpoints give the upper child the longer
    # physical metric edge. Its canonical endpoint order reverses the station.
    assert stations==[(7,Fraction(0),Fraction(1),Fraction(1,2)),
                      (7,Fraction(1),Fraction(1,2),Fraction(3,4))]
    assert len(means)==3 and len({id(value) for value in means})==3
    assert report['shared_segment_splits']==report['insertions']==report['topology_operations']==2
    assert registry.assigned_node_ids==(100,101)
    assert result.points[:len(seed.points)].tobytes()==seed.points.tobytes()


@pytest.mark.parametrize('phase,point',(
    ('native-v2 metric encroachment means',(.5,.5375)),
    ('native-v2 mutable insertion commit',(.5,.7)),
    ('native-v2 shared segment split commit',(.5,.51))))
def test_owner_route_cancellation_keeps_topology_and_registry_exact(monkeypatch,phase,point):
    from anygeometry import to_dict
    model,chart=owner_chart();source=to_dict(model);fixed_proposal(monkeypatch,point)
    instances=[];construct=native.MutableT3Topology.__init__
    def init(self,*args,**kwargs):
        construct(self,*args,**kwargs);instances.append((self,snapshot(self)))
    monkeypatch.setattr(native.MutableT3Topology,'__init__',init)
    error=native._GeometryLimited('owner route cancellation identity')
    def cancel(location):
        if location==phase:raise error
    with pytest.raises(native._GeometryLimited) as caught:frontal(model,chart,cancel=cancel)
    assert caught.value is error and len(instances)==1
    topology,before=instances[0]
    assert snapshot(topology)==before and to_dict(model)==source


def test_constraint_preserving_kernel_error_identity_and_state_are_unchanged(monkeypatch):
    topology=diamond();before=snapshot(topology)
    error=MeshError('native kernel failure')
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(native,'native_mutable_t3_insert',fail)
    with pytest.raises(MeshError) as caught:topology.insert_point((0.,2.2),preserve_splittable=True)
    assert caught.value is error and snapshot(topology)==before
