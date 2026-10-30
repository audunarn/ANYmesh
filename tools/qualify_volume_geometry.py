"""Finite Workflow 2 native geometry/refinement qualification; no FE or promotion."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from uuid import NAMESPACE_URL,uuid5
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    from collections.abc import Mapping
    if isinstance(value,np.ndarray):return plain(value.tolist())
    if isinstance(value,np.generic):return plain(value.item())
    if isinstance(value,Mapping):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):raise ValueError('nonfinite qualification result')
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise TypeError(f'unsupported qualification result type: {type(value).__name__}')


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(plain(value),allow_nan=False,sort_keys=True,indent=2)+'\n')
    temporary.replace(path)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def source_shape(case):
    """Explicit input-unit conversion precedes every GeometryModel API call."""
    if case['kind']=='pentagon':
        angles=np.arange(5)*2*np.pi/5
        r=case['radius'];height=case['height']
        profile=np.column_stack((r*np.cos(angles),r*np.sin(angles),np.zeros(5)))
        area=5*r*r*np.sin(2*np.pi/5)/2
        return profile,np.array([0.,0.,height]),0.,area,np.full(5,2*r*np.sin(np.pi/5)),[2*r,2*r,height]
    scale={'m':1.,'mm':.001}[case['input_units']]
    dims=np.asarray(case['input_dimensions'],float)*scale
    profile=np.array([[0,0,0],[dims[0],0,0],[dims[0],dims[1],0],[0,dims[1],0]],float)
    vector=np.array([0,0,dims[2]])
    offset=case.get('input_offset',0.)*scale
    if case.get('rotation'):
        z,x=case['rotation']['rz'],case['rotation']['rx']
        rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
        rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
        rotation=rz@rx
        profile=profile@rotation.T+np.asarray(case['input_origin'])*scale
        vector=vector@rotation.T
    return profile,vector,offset,float(dims[0]*dims[1]),np.array([dims[0],dims[1],dims[0],dims[1]]),np.abs(dims).tolist()


def build_source(case):
    from anygeometry import GeometryModel
    profile,vector,offset,area,edges,dimensions=source_shape(case)
    model=GeometryModel(model_id=str(uuid5(NAMESPACE_URL,'W2-geometry-20261028/'+case['id'])))
    rid=model.add_solid_prism(profile,vector,offset=offset,name=case['id'])
    before=model.solid_region_snapshot(rid)
    model.set_solid_face_group(rid,'load_face',(before.face_refs[1],))
    model.set_solid_face_group(rid,'shell_transition',(before.face_refs[0],before.face_refs[1]))
    model.set_solid_face_group(rid,'loaded_overlap',(before.face_refs[1],before.face_refs[2]))
    snapshot=model.solid_region_snapshot(rid)
    model.validate_solid_region_snapshot(snapshot)
    assert not snapshot.coordinates.flags.writeable
    height=float(np.linalg.norm(vector))
    assert np.isclose(snapshot.volume,area*height,rtol=1e-12,atol=0)
    analytics={'volume':float(area*height),'face_areas':[float(area),float(area),*(edges*height).tolist()],
               'extrusion_direction':(vector/height).tolist(),'thickness':height,'local_dimensions':dimensions}
    return model,snapshot,analytics


def triangles(snapshot):
    return np.array([(loop[0],loop[j],loop[j+1]) for loop in snapshot.face_loops for j in range(1,len(loop)-1)],dtype=np.int32)


def snapshot_record(snapshot):
    return {'schema_version':snapshot.schema_version,'model_id':snapshot.model_id,'revision':snapshot.revision,
            'region_id':snapshot.region_id,'units':snapshot.units,'geometry_class':snapshot.geometry_class,
            'coordinates':snapshot.coordinates,'face_loops':snapshot.face_loops,
            'face_refs':[(str(h.model_id),h.kind,int(h.id)) for h in snapshot.face_refs],
            'face_groups':{k:[(str(h.model_id),h.kind,int(h.id)) for h in v] for k,v in snapshot.face_groups.items()},
            'volume':snapshot.volume,'bounds':snapshot.bounds,'source_signature':snapshot.source_signature,'provenance':snapshot.provenance}


def backend_identity(manifest,workspace):
    import pytetwild
    from pytetwild import PyfTetWildWrapper as native
    version=importlib.metadata.version('pytetwild')
    wheel=workspace/manifest['backend_wheel']
    identity={'version':version,'native_path':str(Path(native.__file__).resolve()),'native_sha256':digest(native.__file__),
              'wheel_sha256':digest(wheel),'python':sys.version,'python_executable':sys.executable,
              'exact_envelope':getattr(native,'_workflow2_exact_envelope',False),
              'wrapper_sha256':digest(Path(pytetwild.__file__).with_name('pytetwild.py')),
              'init_sha256':digest(pytetwild.__file__)}
    if version!=manifest['backend'].split('==')[1] or identity['exact_envelope'] is not True:
        raise ValueError('wrong backend mode/version')
    for key in ('native_sha256','wheel_sha256'):
        if identity[key]!=manifest['backend_identity'][key]:raise ValueError('backend identity mismatch: '+key)
    return identity


def production_hashes(workspace):
    relative=['ANYmesh/src/anymesher/volume.py','ANYmesh/src/anymesher/_volume_validation.py',
              'ANYgeometry/src/anygeometry/model.py','ANYgeometry/src/anygeometry/solid_regions.py',
              'ANYgeometry/src/anygeometry/features.py','ANYgeometry/src/anygeometry/serialization.py']
    return {name:digest(workspace/name) for name in relative}


def prepare(manifest_path,workspace,output):
    from anygeometry import to_dict
    from anymesher._volume_validation import validate_source
    manifest=json.loads(manifest_path.read_text());output.mkdir(parents=True,exist_ok=False)
    frozen={'manifest':manifest,'manifest_sha256':digest(manifest_path),'runner_sha256':digest(__file__),
            'production_sha256':production_hashes(workspace),'backend':backend_identity(manifest,workspace),'cases':{}}
    for case in manifest['cases']:
        folder=output/case['id'];folder.mkdir()
        model,snapshot,analytics=build_source(case)
        validate_source(snapshot,case['boundary_tolerance'])
        assert model.strict_audit().certifiable
        write_json(folder/'geometry.json',to_dict(model))
        record=snapshot_record(snapshot);write_json(folder/'snapshot.json',record)
        np.savez_compressed(folder/'input.npz',coordinates=snapshot.coordinates,triangles=triangles(snapshot))
        frozen['cases'][case['id']]={'parameters':case,'analytics':analytics,'snapshot_sha256':canonical_hash(record),
            'geometry_sha256':digest(folder/'geometry.json'),'input_sha256':digest(folder/'input.npz')}
    write_json(output/'frozen.json',frozen)
    print(json.dumps({'status':'FROZEN_BEFORE_NATIVE_CALLS','frozen_sha256':digest(output/'frozen.json'),'cases':list(frozen['cases'])}),flush=True)


def checked_inputs(frozen_path,workspace,case_id):
    from anygeometry import from_dict
    frozen=json.loads(frozen_path.read_text())
    if frozen['runner_sha256']!=digest(__file__) or frozen['production_sha256']!=production_hashes(workspace):
        raise ValueError('runner/production source changed after prospective freeze')
    identity=backend_identity(frozen['manifest'],workspace)
    if identity!=frozen['backend']:raise ValueError('effective backend/runtime changed after freeze')
    case=frozen['cases'][case_id];folder=frozen_path.parent/case_id
    for filename,key in [('geometry.json','geometry_sha256'),('input.npz','input_sha256')]:
        if digest(folder/filename)!=case[key]:raise ValueError('frozen case input changed')
    model=from_dict(json.loads((folder/'geometry.json').read_text()))
    snapshot=model.solid_region_snapshot(next(iter(model.solid_regions)))
    if canonical_hash(snapshot_record(snapshot))!=case['snapshot_sha256']:raise ValueError('snapshot identity changed')
    return frozen,case,snapshot,folder


def generate(frozen_path,workspace,case_id,output):
    import pytetwild
    frozen,case,snapshot,folder=checked_inputs(frozen_path,workspace,case_id)
    output.mkdir(parents=True,exist_ok=False)
    params=case['parameters'];diagonal=float(np.linalg.norm(np.ptp(snapshot.coordinates,axis=0)))
    options={'edge_length_abs':params['target_edge_length'],'epsilon':params['boundary_tolerance']/diagonal,**frozen['manifest']['options']}
    ledger=frozen_path.parent/'native-attempts';ledger.mkdir(exist_ok=True)
    with (ledger/(case_id+'.json')).open('x') as claim:
        json.dump({'case':case_id,'output':str(output),'frozen_sha256':digest(frozen_path),'maximum_calls':1},claim,allow_nan=False)
    write_json(output/'native-attempt.json',{'case':case_id,'frozen_sha256':digest(frozen_path),'snapshot_sha256':case['snapshot_sha256'],
               'input_sha256':case['input_sha256'],'backend':frozen['backend'],'effective_options':options,'maximum_calls':1})
    with np.load(folder/'input.npz',allow_pickle=False) as data:x,f=data['coordinates'],data['triangles']
    os.chdir(output)
    started=time.perf_counter()
    coordinates,cells=pytetwild.tetrahedralize(x,f,**options)
    elapsed=time.perf_counter()-started
    temporary=output/'raw.tmp.npz';np.savez_compressed(temporary,coordinates=coordinates,cells=cells);temporary.replace(output/'raw.npz')
    cap=frozen['manifest']['limits']['maximum_cells']
    write_json(output/'result.json',{'status':'RAW_GENERATED' if len(cells)<=cap else 'FAILED_CELL_CAP','native_calls':1,
               'native_seconds':elapsed,'nodes':len(coordinates),'cells':len(cells),'raw_sha256':digest(output/'raw.npz')})
    if len(cells)>cap:raise ValueError('prospective cell cap exceeded')
    print(json.dumps({'case':case_id,'native_seconds':elapsed,'cells':len(cells)}),flush=True)


def distribution(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all() or not values.size:raise ValueError('invalid metric distribution')
    return dict(zip(('min','p01','p05','median','p95','p99','max'),map(float,np.quantile(values,[0,.01,.05,.5,.95,.99,1]))))


def measurements(mesh,snapshot,case):
    p=mesh.coordinates[mesh.cells];d=p[:,1:]-p[:,:1];det=np.linalg.det(d)
    edge=np.stack([np.linalg.norm(p[:,i]-p[:,j],axis=1) for i in range(4) for j in range(i+1,4)],axis=1)
    volume=det/6;quality=12*np.power(3*volume,2/3)/np.sum(edge**2,axis=1)
    h=case['parameters']['target_edge_length'];analytic=case['analytics'];axis=np.asarray(analytic['extrusion_direction'])
    spans=np.ptp(p@axis,axis=1)
    facets=mesh.coordinates[mesh.boundary_facets];cross=np.cross(facets[:,1]-facets[:,0],facets[:,2]-facets[:,0])
    areas=np.linalg.norm(cross,axis=1)/2;unit=cross/(2*areas[:,None]);face_areas=[];max_angle=0.;max_plane=0.
    owners=[mesh.face_owners[int(k)] for k in mesh.boundary_ids]
    if len(owners)!=len(facets) or set(owners)!=set(snapshot.face_refs):raise ValueError('incomplete source face ownership')
    for loop,ref in zip(snapshot.face_loops,snapshot.face_refs):
        polygon=snapshot.coordinates[list(loop)];n=np.cross(polygon[1]-polygon[0],polygon[2]-polygon[0]);n/=np.linalg.norm(n)
        selected=np.array([owner==ref for owner in owners]);face_areas.append(float(areas[selected].sum()))
        max_angle=max(max_angle,float(np.max(np.rad2deg(np.arccos(np.clip(unit[selected]@n,-1,1))))))
        max_plane=max(max_plane,float(np.max(np.abs((facets[selected]-polygon[0])@n))))
    return {'relative_volume_error':abs(float(volume.sum())/analytic['volume']-1),
            'maximum_relative_face_area_error':float(np.max(np.abs(np.asarray(face_areas)/analytic['face_areas']-1))),
            'maximum_face_normal_degrees':max_angle,'maximum_plane_distance':max_plane,'face_areas':face_areas,
            'mean_ratio':distribution(quality),'edge_length_over_h':distribution(edge.ravel()/h),
            'cell_axial_span_over_thickness':distribution(spans/analytic['thickness']),
            'cell_axial_span_over_h':distribution(spans/h),'nominal_local_extents_over_h':(np.asarray(analytic['local_dimensions'])/h).tolist(),
            'nominal_through_thickness_spacing_count':analytic['thickness']/h}


def admit(frozen_path,workspace,case_id,raw_folder,output):
    from anymesher._volume_validation import validate_source
    from anymesher.volume import _admit_provider_output
    frozen,case,snapshot,_=checked_inputs(frozen_path,workspace,case_id);output.mkdir(parents=True,exist_ok=False)
    result={'case':case_id,'status':'FAILED','native_calls':0,'predicates':{},'frozen_sha256':digest(frozen_path)}
    try:
        raw=json.loads((raw_folder/'result.json').read_text())
        if raw['status']!='RAW_GENERATED' or raw['raw_sha256']!=digest(raw_folder/'raw.npz'):raise ValueError('unverified retained raw arrays')
        with np.load(raw_folder/'raw.npz',allow_pickle=False) as data:x,t=data['coordinates'],data['cells']
        if len(t)>frozen['manifest']['limits']['maximum_cells']:raise ValueError('cell cap exceeded')
        source=validate_source(snapshot,case['parameters']['boundary_tolerance'])
        started=time.perf_counter()
        mesh=_admit_provider_output(source,x,t,{'backend':frozen['manifest']['backend'],'raw_sha256':raw['raw_sha256']},case['parameters']['target_edge_length'])
        result['admission_seconds']=time.perf_counter()-started
        metrics=measurements(mesh,snapshot,case);limits=frozen['manifest']['limits'];delta=case['parameters']['boundary_tolerance']
        bounds=np.asarray(snapshot.bounds)
        predicates={'production_topology_ownership_coverage':True,'cell_cap':len(mesh.cells)<=limits['maximum_cells'],
            'volume':metrics['relative_volume_error']<=limits['maximum_relative_volume_error'],
            'face_areas':metrics['maximum_relative_face_area_error']<=limits['maximum_relative_face_area_error'],
            'normals':metrics['maximum_face_normal_degrees']<=limits['maximum_face_normal_degrees'],
            'physical_plane_tolerance':metrics['maximum_plane_distance']<=delta,
            'bounds':bool(np.all(mesh.coordinates>=bounds[0]-delta) and np.all(mesh.coordinates<=bounds[1]+delta)),
            'persistent_face_groups':all(set(map(int,mesh.boundary_groups[name]))=={int(i) for i in mesh.boundary_ids if mesh.face_owners[int(i)] in handles} for name,handles in snapshot.face_groups.items()),
            'persistent_region_group':set(map(int,mesh.cell_groups[snapshot.region_id]))==set(map(int,mesh.cell_ids))}
        result.update(predicates=predicates,metrics=metrics,nodes=len(mesh.coordinates),cells=len(mesh.cells),boundary_facets=len(mesh.boundary_facets),validation=mesh.provenance['validation'])
        if not all(predicates.values()):raise ValueError('prospective qualification predicate failed')
        temporary=output/'mesh.tmp.npz';np.savez_compressed(temporary,coordinates=mesh.coordinates,cells=mesh.cells,node_ids=mesh.node_ids,cell_ids=mesh.cell_ids,boundary_facets=mesh.boundary_facets,boundary_ids=mesh.boundary_ids);temporary.replace(output/'mesh.npz')
        write_json(output/'ownership.json',{'face_owners':{int(k):(str(v.model_id),v.kind,int(v.id)) for k,v in mesh.face_owners.items()},'boundary_groups':mesh.boundary_groups,'cell_groups':mesh.cell_groups})
        result.update(status='PASSED',admitted_sha256=digest(output/'mesh.npz'))
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    write_json(output/'result.json',result)
    print(json.dumps(plain(result),allow_nan=False),flush=True)
    return 0 if result['status']=='PASSED' else 1


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('prepare','generate','admit'));p.add_argument('--workspace',type=Path,required=True);p.add_argument('--manifest',type=Path);p.add_argument('--frozen',type=Path);p.add_argument('--case');p.add_argument('--raw',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    workspace=a.workspace.resolve();output=a.output.resolve()
    if a.stage=='prepare':prepare(a.manifest.resolve(),workspace,output);return 0
    if a.stage=='generate':generate(a.frozen.resolve(),workspace,a.case,output);return 0
    return admit(a.frozen.resolve(),workspace,a.case,a.raw.resolve(),output)


if __name__=='__main__':raise SystemExit(main())
