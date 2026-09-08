"""Linear atlas lifecycle and public recombination regressions.

Private lifecycle probes retain their detached fixtures. Public integration
probes consume the orchestrator's owner binding without injecting a second one.
Diagnostic receipts use pytest tmp_path, not external snapshot loaders.
"""
from copy import deepcopy
from dataclasses import replace
from contextlib import ExitStack
import json
import math
import pickle
from types import SimpleNamespace
import traceback
import numpy as np
import pytest
from anygeometry import CylinderAtlasError, CylinderAtlasErrorCode
from anymesher import hybrid, surface_mesh
from anymesher import FeatureDistanceMetricControl, IsotropicMetricControl, MetricFieldSpec, NativeMeshingOptions
from anymesher._cylindrical_atlas import prepare_cylindrical_atlas
from anymesher import _cylindrical_recombine as deferred
from anymesher._shared_triangle_split import propagate_triangle_split
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.surface_mesh import SurfaceMeshOptions
from anymesher.quality_v2 import assert_valid_mesh, evaluate_quality
from anymesher.refinement import Refinement
from anymesher.metric import SpatialMetricField
from anymesher.core import MeshCore
from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state



def mesh_payload(mesh):
    return {'nodes':{str(n):np.asarray(p).tolist() for n,p in mesh.nodes.items()},
        'tris':mesh.tris, 'quads':mesh.quads, 'node_of_vertex':mesh.node_of_vertex,
        'nodes_of_edge':mesh.nodes_of_edge, 'elements_of_face':mesh.elements_of_face,
        'elements_of_sheet':mesh.elements_of_sheet, 'activity':mesh.activity,
        'geometry_model_id':str(mesh.geometry_model_id), 'geometry_revision':mesh.geometry_revision}


def state_bytes(mesh):
    return pickle.dumps(mesh.__dict__, protocol=5)


def prepared(staged_faces=8):
    model, selected = _sector_model()
    binding = prepare_cylindrical_atlas(model, selected, reference_face_use=selected[0])
    mesh = Mesh(geometry_model_id=model.model_id, geometry_revision=model.revision)
    def node(i,z):
        return 1+3*(i%8)+z
    for i in range(8):
        for z in range(3):
            mesh.nodes[node(i,z)] = np.array((math.cos(i*math.pi/4),math.sin(i*math.pi/4),float(z)))
    def station_id(point):
        theta = math.atan2(point[1],point[0])%(2*math.pi)
        return node(int(round(theta/(math.pi/4)))%8, int(round(point[2])))
    for edge_id, edge in model.edges.items():
        a,b = (model.vertices[v].position for v in (edge.start,edge.end))
        count = 3 if abs(a[2]-b[2])>1 else 2
        mesh.nodes_of_edge[edge_id] = [station_id(p) for p in model.sample_edge(edge_id,np.linspace(0,1,count))]
        mesh.node_of_vertex[edge.start] = station_id(a)
        mesh.node_of_vertex[edge.end] = station_id(b)
    for i,face_id in enumerate(sorted(model.faces)):
        ids=[]
        for z in (0,1):
            a,b,c,d = node(i,z),node(i+1,z),node(i+1,z+1),node(i,z+1)
            for row in ((a,b,c),(a,c,d)):
                element=len(mesh.tris)+1
                mesh.tris[element]=row
                ids.append(element)
        mesh.elements_of_face[face_id]=ids
    for sheet_id,sheet in model.sheets.items():
        mesh.elements_of_sheet[sheet_id]=sorted(e for f in model.faces for e in mesh.elements_of_face[f])
    registry=SimpleNamespace(_published_triangle_incidence={})
    settings=SurfaceMeshOptions(target_size=3., recombine=True,
        native_options=NativeMeshingOptions(point_placement='frontal_delaunay'))
    for face_id in sorted(model.faces)[:staged_faces]:
        deferred.register_face(model,mesh,face_id,binding,registry,settings,
                               effective_metric_field=MetricFieldSpec.uniform(3.))
        deferred.mark_staged(registry,face_id,mesh,{'native_v2':{'published_insertions':0}})
    return model,mesh,binding,registry,settings


def test_incomplete_component_is_atomic():
    model,mesh,binding,registry,settings=prepared(7)
    before=state_bytes(mesh)
    with pytest.raises(MeshError,match='incomplete'):
        deferred.finalize_components(model,mesh,registry)
    assert state_bytes(mesh)==before


def test_cancellation_before_commit_is_atomic():
    model,mesh,binding,registry,settings=prepared()
    before=state_bytes(mesh)
    error=RuntimeError('cancel at component commit boundary')
    reached=[]
    def cancel(phase):
        if phase=='cylindrical deferred before commit':
            reached.append(phase)
            raise error
    with pytest.raises(RuntimeError) as caught:
        deferred.finalize_components(model,mesh,registry,cancel)
    assert caught.value is error
    assert reached==['cylindrical deferred before commit']
    assert state_bytes(mesh)==before
    assert len(next(iter(registry._deferred_cylindrical_components.values())).attempts)==8


def test_physical_quality_rejection_before_commit_is_atomic(monkeypatch):
    model,mesh,binding,registry,settings=prepared()
    for state in registry._deferred_cylindrical_components.values():
        state.settings = {face: replace(value, enforce_quality=True) for face,value in state.settings.items()}
    before=state_bytes(mesh)
    original=surface_mesh._quality_threshold_report
    calls=[]
    def rejected(quality,options):
        value=original(quality,options)
        calls.append(value['accepted'])
        if len(calls)==4:
            value=dict(value,accepted=False)
        return value
    monkeypatch.setattr(surface_mesh,'_quality_threshold_report',rejected)
    with pytest.raises(MeshError,match='physical/chart quality before commit'):
        deferred.finalize_components(model,mesh,registry)
    assert len(calls)==4
    assert state_bytes(mesh)==before


def test_physical_size_rejection_before_commit_is_atomic(monkeypatch):
    model,mesh,binding,registry,settings=prepared()
    before=state_bytes(mesh)
    original=deferred._physical_size
    def rejected(*args):
        return dict(original(*args),finite=False)
    monkeypatch.setattr(deferred,'_physical_size',rejected)
    with pytest.raises(MeshError,match='physical size before commit'):
        deferred.finalize_components(model,mesh,registry)
    assert state_bytes(mesh)==before


def test_duplicate_finalization_and_late_native_work_are_refused():
    model,mesh,binding,registry,settings=prepared()
    result=deferred.finalize_components(model,mesh,registry)
    assert len(result)==8 and mesh.quads
    before=state_bytes(mesh)
    with pytest.raises(MeshError,match='duplicate'):
        deferred.finalize_components(model,mesh,registry)
    face_id=min(model.faces)
    with pytest.raises(MeshError,match='late cylindrical'):
        hybrid._mesh_native_face(model,mesh,face_id,order='linear',recombine=True,
            native_backend='python',native_options=settings.native_options,size_field=None,
            metric_model_uuid=str(model.model_id),metric_geometry_revision=model.revision,
            boundary_registry=None,automatically_seeded_shared_edges=frozenset(),
            component_seed_registry=registry,quality_options=None,declared_junction_edges=(),
            evaluate_declared_junction_alignment=False,refine_declared_junction_transition=False,
            cancellation_check=None,_cylindrical_binding=binding)
    assert state_bytes(mesh)==before


def test_late_shared_split_keeps_existing_t3_guard():
    model,mesh,binding,registry,settings=prepared()
    deferred.finalize_components(model,mesh,registry)
    before=state_bytes(mesh)
    with pytest.raises(MeshError,match='staged linear T3 neighbours'):
        propagate_triangle_split(mesh,[min(model.faces)],(1,4),2,cache=registry._published_triangle_incidence)
    assert state_bytes(mesh)==before


def test_external_neighbours_are_refused_without_scope_expansion(monkeypatch):
    model,mesh,binding,registry,settings=prepared(0)
    before=state_bytes(mesh)
    original=model.faces_using_edge
    monkeypatch.setattr(model,'faces_using_edge',lambda edge:(*original(edge),999999))
    with pytest.raises(MeshError,match='external neighbours'):
        deferred.register_face(model,mesh,min(model.faces),binding,registry,settings,
                               effective_metric_field=MetricFieldSpec.uniform(3.))
    assert state_bytes(mesh)==before


def test_extracted_recombination_matches_existing_operation():
    # Portable operation oracle; cross-snapshot six-route hashes stay report evidence.
    points=np.asarray(((0.,0.),(1.3,0.),(1.3,1.),(0.,1.)))
    triangles=np.asarray(((0,1,2),(0,2,3)))
    protected=((0,1),(1,2),(2,3),(0,3))
    settings=SurfaceMeshOptions(target_size=.4,recombine=True)
    actual,quality=surface_mesh._qualified_recombination(
        MeshCore(points.copy(),triangles.copy()),protected,settings,None)
    expected=surface_mesh.recombine_triangles_with_report(
        MeshCore(points.copy(),triangles.copy()),protected_edges=protected,
        min_scaled_jacobian=settings.min_scaled_jacobian,
        max_aspect_ratio=settings.max_aspect_ratio,min_angle=settings.min_angle,
        max_angle=settings.max_angle,max_warpage=settings.max_warpage,
        max_exchange_work=settings.max_recombination_work,cancellation_check=None)
    assert actual.pair_count==expected.pair_count==1
    assert actual.exchange_truncated==expected.exchange_truncated
    for name in ('node_coordinates','node_ids','triangle_connectivity','quad_connectivity',
                 'triangle_ids','quad_ids','triangle_active','quad_active'):
        np.testing.assert_array_equal(getattr(actual.mesh,name),getattr(expected.mesh,name))
    assert_valid_mesh(actual.mesh)
    assert quality==surface_mesh._quality_threshold_report(evaluate_quality(expected.mesh),settings)


@pytest.mark.parametrize('enforce,prefer,overrides,expected', (
    (False,False,{},True), (True,True,{},True),
    (False,True,{'min_angle':91.},True), (True,False,{'min_angle':91.},False),
    (False,False,{'max_aspect_ratio':1.01},True), (True,True,{'max_aspect_ratio':1.01},False),
))
def test_requested_policy_binding(enforce,prefer,overrides,expected):
    model,mesh,binding,registry,settings=prepared()
    state=next(iter(registry._deferred_cylindrical_components.values()))
    state.settings={face:replace(value,enforce_quality=enforce,prefer_quality_policy=prefer,**overrides)
                    for face,value in state.settings.items()}
    before=state_bytes(mesh)
    if not expected:
        with pytest.raises(MeshError,match='physical/chart quality before commit'):
            deferred.finalize_components(model,mesh,registry)
        assert state_bytes(mesh)==before
        assert state.phase=='rejected'
    else:
        results=deferred.finalize_components(model,mesh,registry)
        assert state.phase=='committed'
        assert all(value['physical_size']['accepted'] for value in results.values())
        assert_valid_mesh(hybrid._neutral_shell_core(mesh))
        if overrides:
            assert not mesh.quads
            assert any(not value['physical_quality']['accepted'] for value in results.values())
        else:
            assert mesh.quads
    for value in state.attempts.values():
        assert value['requested_settings']['enforce_quality']==enforce
        assert value['requested_settings']['prefer_quality_policy']==prefer
        for key,requested in overrides.items():
            assert value['requested_settings'][key]==requested


def test_sizing_non_regression_rejects_larger_retained_edge():
    before={'finite':True,'edges':[[0,1]],'sampled_edge_metric_lengths':[[1.7,1.8,1.9]],
            'physical_edge_lengths':[.4],'maximum_sampled_edge_metric_length':1.9,'maximum_physical_edge_length':.4}
    assert deferred._sizing_non_regression(before,deepcopy(before))
    after=deepcopy(before)
    after['sampled_edge_metric_lengths'][0][1]=1.81
    assert not deferred._sizing_non_regression(before,after)


def test_hard_physical_validity_remains_atomic(monkeypatch):
    model,mesh,binding,registry,settings=prepared()
    original=deferred.assert_valid_mesh
    calls=[]
    def rejected(core):
        calls.append(core)
        if len(calls)==2:
            raise MeshError('injected post-recombination physical invalidity')
        return original(core)
    monkeypatch.setattr(deferred,'assert_valid_mesh',rejected)
    before=state_bytes(mesh)
    with pytest.raises(MeshError,match='physical invalidity'):
        deferred.finalize_components(model,mesh,registry)
    assert state_bytes(mesh)==before


@pytest.mark.parametrize('open_journal',(False,True))
def test_last_nonraising_callback_owner_mutation_is_atomic(open_journal):
    model,mesh,binding,registry,settings=prepared()
    before=state_bytes(mesh)
    before_revision=model.revision
    phases=[]
    caller_edits=[]
    with ExitStack() as journals:
        def callback(phase):
            phases.append(phase)
            if phase=='cylindrical deferred before commit':
                if open_journal:
                    journals.enter_context(model.transaction())
                    caller_edits.append(model.add_point(10.,11.,12.))
                else:
                    with model.transaction():
                        caller_edits.append(model.add_point(10.,11.,12.))
        with pytest.raises(CylinderAtlasError) as caught:
            deferred.finalize_components(model,mesh,registry,callback)
        assert caught.value.code is (CylinderAtlasErrorCode.BUSY_MODEL if open_journal else CylinderAtlasErrorCode.STALE_REVISION)
        assert state_bytes(mesh)==before
        assert phases[-1]=='cylindrical deferred before commit'
        assert phases.count('cylindrical deferred before commit')==1
        assert len(caller_edits)==1 and caller_edits[0] in model.vertices
        np.testing.assert_array_equal(model.vertices[caller_edits[0]].position,(10.,11.,12.))
        state=next(iter(registry._deferred_cylindrical_components.values()))
        assert state.phase=='rejected' and len(state.attempts)==8
    assert caller_edits[0] in model.vertices
    assert model.revision>before_revision
    assert state_bytes(mesh)==before


@pytest.mark.parametrize('combined',(False,True))
def test_effective_sizefield_receipt_and_evaluator(combined,monkeypatch,tmp_path):
    model,selected=_sector_model()
    center=(math.cos(math.pi/8),math.sin(math.pi/8),1.)
    refinement=Refinement(size=.6,radius=.2,center=center,name='sizefield-zone')
    explicit=(MetricFieldSpec(IsotropicMetricControl(2.5),feature_controls=(
        FeatureDistanceMetricControl(((1.,0.,1.),),.8,.1,1.5,'explicit-zone'),)) if combined else None)
    captured={}
    original_native=hybrid._mesh_native_face
    original_register=deferred.register_face
    class ReceiptCaptured(Exception):
        pass
    def native(geometry,mesh,face_id,**kwargs):
        assert kwargs['_cylindrical_binding'] is not None
        captured['size_field']=kwargs['size_field']
        return original_native(geometry,mesh,face_id,**kwargs)
    def register(geometry,mesh,face_id,binding,registry,settings,**kwargs):
        state=original_register(geometry,mesh,face_id,binding,registry,settings,**kwargs)
        captured.update(geometry=geometry,field=kwargs['effective_metric_field'],
                        supplemental=kwargs['supplemental_metric_field'],receipt=state.metric_fields[face_id]['receipt'])
        return state
    def surface(*args,**kwargs):
        assert kwargs['options'].native_options.metric_field is captured['field']
        assert kwargs['_supplemental_metric_field'] is captured['supplemental'] is None
        size_spec=MetricFieldSpec.from_size_field(captured['size_field'])
        expected=size_spec if explicit is None else MetricFieldSpec(
            IsotropicMetricControl(min(explicit.global_control.target_size,size_spec.global_control.target_size)),
            feature_controls=(*explicit.feature_controls,*size_spec.feature_controls),
            imported_samples=explicit.imported_samples,
            maximum_anisotropy=min(explicit.maximum_anisotropy,size_spec.maximum_anisotropy),
            maximum_gradation=min(explicit.maximum_gradation,size_spec.maximum_gradation))
        assert captured['field'].to_dict()==expected.to_dict()
        assert captured['receipt']['effective_metric_field']==expected.to_dict()
        assert {c.name for c in expected.feature_controls}==({'sizefield-zone','explicit-zone'} if combined else {'sizefield-zone'})
        points=np.asarray(center)+np.asarray(((0.,0.,0.),(.05,0.,0.),(0.,.05,0.)))
        core=MeshCore(points,np.asarray(((0,1,2),)))
        measured=deferred._physical_size(core,captured['field'],captured['geometry'],None)
        for edge,samples in zip(measured['edges'],measured['sampled_edge_metric_lengths']):
            a,b=points[edge]
            delta=b-a
            tensors=SpatialMetricField(expected).evaluate(a+np.asarray((0.,.5,1.))[:,None]*delta)
            np.testing.assert_array_equal(samples,np.sqrt(np.einsum('i,nij,j->n',delta,tensors,delta)))
        assert measured['maximum_sampled_edge_metric_length']>.1
        payload={'case':'combined' if combined else 'refinement-only','receipt':captured['receipt'],
                 'same_object_as_native_surface_field':True,'measurements':measured,
                 'boundary_only_probe':'stops deliberately before surface meshing; no publication claimed'}
        (tmp_path/('effective-field-combined.json' if combined else 'effective-field-refinement-only.json')).write_text(json.dumps(payload,indent=2),encoding='utf-8')
        raise ReceiptCaptured()
    monkeypatch.setattr(hybrid,'_mesh_native_face',native)
    monkeypatch.setattr(deferred,'register_face',register)
    monkeypatch.setattr(hybrid,'mesh_planar_surface',surface)
    with pytest.raises(ReceiptCaptured):
        hybrid.generate_hybrid_mesh_result(model,target_size=3.,refinements=(refinement,),strategy='native',native_backend='python',recombine=True,
            native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',metric_field=explicit,max_insertions=8))


def test_nonnull_supplement_preserves_external_provider_exclusion():
    model,mesh,binding,registry,settings=prepared(0)
    before=state_bytes(mesh)
    with pytest.raises(MeshError,match='excluded external provider'):
        deferred.register_face(model,mesh,min(model.faces),binding,registry,settings,
            effective_metric_field=MetricFieldSpec.uniform(3.),supplemental_metric_field=MetricFieldSpec.uniform(2.))
    assert state_bytes(mesh)==before


def test_actual_spatial_linear_recombination(monkeypatch,tmp_path):
    model,selected=_sector_model()
    before=_persistent_state(model)
    record={'status':'started','refinements':[],'faces':[], 'qualification':'not yet published'}
    bindings={}
    original_face=hybrid._mesh_native_face
    original_refine=surface_mesh.frontal_delaunay_refine
    original_finalize=deferred.finalize_components
    retained={}
    def refine(*args,**kwargs):
        item={'input_points':len(args[0].points),'completed':False}
        record['refinements'].append(item)
        result=original_refine(*args,**kwargs)
        item.update(completed=True,output_points=len(result[0].points),intermediate_insertions=len(result[0].points)-item['input_points'])
        return result
    def native(geometry,mesh,face_id,**kwargs):
        key=(geometry.model_id,geometry.revision)
        if key not in bindings:
            bindings[key]=kwargs['_cylindrical_binding']
            assert len(bindings[key].charts)==len(bindings[key].atlas.sectors)==8
            retained['protected']={n:np.asarray(mesh.nodes[n]).copy() for ids in mesh.nodes_of_edge.values() for n in ids}
        assert kwargs['_cylindrical_binding'] is bindings[key]
        item={'face_id':face_id,'completed':False}
        record['faces'].append(item)
        value=original_face(geometry,mesh,face_id,**kwargs)
        item.update(completed=True,diagnostics=value)
        return value
    def finalize(geometry,mesh,registry,cancellation_check=None):
        retained['mesh']=mesh
        retained['registry']=registry
        retained['precommit_state']=state_bytes(mesh)
        record['before_finalization']=mesh_payload(mesh)
        try:
            value=original_finalize(geometry,mesh,registry,cancellation_check)
            record['finalization']=value
            return value
        except BaseException:
            record['failure_atomicity_passed']=state_bytes(mesh)==retained['precommit_state']
            assert record['failure_atomicity_passed']
            raise
        finally:
            record['after_finalization']=mesh_payload(mesh)
            record['component_attempts']=[{'phase':s.phase,'attempts':s.attempts} for s in registry._deferred_cylindrical_components.values()]
    monkeypatch.setattr(hybrid,'_mesh_native_face',native)
    monkeypatch.setattr(surface_mesh,'frontal_delaunay_refine',refine)
    monkeypatch.setattr(deferred,'finalize_components',finalize)
    centers=tuple((float(np.cos((i+.5)*np.pi/4)),float(np.sin((i+.5)*np.pi/4)),1.) for i in range(8))
    field=MetricFieldSpec(IsotropicMetricControl(.4),feature_controls=tuple(FeatureDistanceMetricControl((center,),.12,.35,1.5,f'interior-{i}') for i,center in enumerate(centers)))
    try:
        result=hybrid.generate_hybrid_mesh_result(model,target_size=.4,strategy='native',native_backend='python',order='linear',recombine=True,
            native_options=NativeMeshingOptions(point_placement='frontal_delaunay',metric_mode='isotropic_spatial',metric_field=field,max_insertions=128,max_topology_operations=20000))
        mesh=result.mesh
        assert len(record['faces'])==8 and all(f['completed'] for f in record['faces'])
        assert mesh.quads and not mesh.is_quadratic
        assert set(result.strategy_by_face.values())=={'native'}
        committed=sum(v['published_insertions'] for v in record['finalization'].values())
        assert committed==581,'Published native refinement must match the exact spatial reference'
        assert len(record['before_finalization']['nodes'])==735
        assert len(record['before_finalization']['tris'])==1438
        assert len(mesh.nodes)==735
        assert len(mesh.tris)+2*len(mesh.quads)==1438
        assert all(v['physical_size']['accepted'] for v in record['finalization'].values())
        assert all(not v['requested_settings']['enforce_quality'] for v in record['finalization'].values())
        assert all(v['requested_settings']['min_angle']==30. for v in record['finalization'].values())
        for face_id,value in record['finalization'].items():
            pre=value['repaired_pre_pair']
            post=value['post_recombination_proposal']
            assert pre['source_node_ids']==post['source_node_ids']
            assert pre['physical_coordinates']==post['physical_coordinates']
            assert pre['chart_coordinates']==post['chart_coordinates']
            assert value['pair_count']==len(post['quads'])
            receipt=value['effective_physical_field_receipt']
            assert receipt['effective_metric_field']['feature_controls']==field.to_dict()['feature_controls']
            assert receipt['supplemental_metric_field'] is None
            cells=[mesh.tris.get(e,mesh.quads.get(e)) for e in mesh.elements_of_face[face_id]]
            expected=[tuple(post['source_node_ids'][i] for i in cell) for kind in ('triangles','quads') for cell in post[kind]]
            assert sorted(cells)==sorted(expected)
        assert sum(v['pair_count'] for v in record['finalization'].values())==len(mesh.quads)
        for node,point in retained['protected'].items():
            np.testing.assert_array_equal(mesh.nodes[node],point)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(tuple(mesh.nodes.values()))[:,:2],axis=1),1.,rtol=0.,atol=1.e-12)
        core=hybrid._neutral_shell_core(mesh)
        assert_valid_mesh(core)
        quality=evaluate_quality(core)
        assert quality.minimum_scaled_jacobian>0 and quality.maximum_aspect_ratio<=5.
        incidence={}
        cells=dict(mesh.tris)
        cells.update(mesh.quads)
        centroids=[]
        lengths=[]
        for cell in cells.values():
            xyz=np.asarray([mesh.nodes[n] for n in cell])
            centroids.append(xyz.mean(axis=0))
            lengths.append(np.max(np.linalg.norm(xyz-np.roll(xyz,-1,axis=0),axis=1)))
            for a,b in zip(cell,cell[1:]+cell[:1]):
                edge=tuple(sorted((a,b)))
                incidence[edge]=incidence.get(edge,0)+1
        assert max(incidence.values())==2
        for edge_id,ids in mesh.nodes_of_edge.items():
            assert len(ids)==len(set(ids))
            faces=model.faces_using_edge(edge_id)
            for a,b in zip(ids,ids[1:]):
                assert incidence[tuple(sorted((a,b)))]==len(faces)
            for face_id in faces:
                assert set(ids)<={n for e in mesh.elements_of_face[face_id] for n in cells[e]}
        for center in centers:
            near=np.linalg.norm(np.asarray(centroids)-center,axis=1)<.12
            assert np.any(near)
            assert np.max(np.asarray(lengths)[near])<.4
        record.update(status='passed',qualification='published refinement with post-recombination physical checks',published_insertions=committed,
            nodes=len(mesh.nodes),triangles=len(mesh.tris),quads=len(mesh.quads),minimum_scaled_jacobian=quality.minimum_scaled_jacobian,maximum_aspect_ratio=quality.maximum_aspect_ratio,published_mesh=mesh_payload(mesh))
    except BaseException as error:
        record.update(status='failed',exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc(),qualification='NOT QUALIFIED')
        raise
    finally:
        record['persistent_geometry_unchanged']=_persistent_state(model)==before
        record['dependency_scope']='repository imports; public owner-bound route; no snapshot loader'
        (tmp_path/'actual-spatial-result.json').write_text(json.dumps(record,indent=2,default=str),encoding='utf-8')
        print('ACTUAL_RESULT '+json.dumps({k:v for k,v in record.items() if k in ('status','qualification','exception','nodes','triangles','quads','published_insertions','failure_atomicity_passed')}),flush=True)
        assert record['persistent_geometry_unchanged']
