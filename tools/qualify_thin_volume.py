"""Fresh, frozen thin-sampling/conditioning study. No provider promotion."""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[2]
ROOT=WORKSPACE/'evidence/thin-resolution-20261028-a1'
MANIFEST=Path(__file__).resolve().parents[1]/'docs/thin_resolution_manifest_20261028.json'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def helpers():
    return load('thin_geometry_reference',Path(__file__).with_name('qualify_volume_geometry.py'))


def stream_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def runtime_identity():
    import anygeometry,anymesher,anysolid,skfem,scipy.sparse.linalg,shapely,pytetwild,decimal
    from anymesher import _volume_validation
    result=dict(python=sys.version,executable=sys.executable,versions={},files={},external_shared_libraries={})
    for name in ('numpy','scipy','scikit-fem','anysolid','pytetwild','shapely'):
        distribution=importlib.metadata.distribution(name)
        result['versions'][name]=distribution.version
        for entry in distribution.files or ():
            path=distribution.locate_file(entry).resolve()
            if path.suffix=='.pyc' or '__pycache__' in path.parts or not path.is_file():continue
            result['files'][str(path)]=stream_sha(path)
    for path in (WORKSPACE/'ANYgeometry/src/anygeometry').rglob('*.py'):
        result['files'][str(path.resolve())]=stream_sha(path)
    result['origins']={module.__name__:str(Path(module.__file__).resolve())
        for module in (np,scipy,anysolid,skfem,pytetwild,shapely)}
    if any(path not in result['files'] for path in result['origins'].values()):
        raise ValueError('effective numerical implementation is outside hashed installed files')
    for module,repo in ((anygeometry,'ANYgeometry'),(anymesher,'ANYmesh')):
        expected=WORKSPACE/repo/'src'/module.__name__/'__init__.py'
        if Path(module.__file__).resolve()!=expected.resolve():
            raise ValueError('study imported a different owner candidate')
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/maps').read_text().splitlines():
            parts=line.split(maxsplit=5)
            if len(parts)!=6 or not parts[5].startswith('/') or '.so' not in parts[5]:continue
            path=Path(parts[5])
            if not path.is_file():raise ValueError('mapped shared library is not identifiable')
            resolved=str(path.resolve())
            if resolved not in result['files']:
                result['external_shared_libraries'][resolved]=stream_sha(path)
    return result


def regular_tet_condition(points):
    p=np.asarray(points,dtype=float)
    if p.ndim!=3 or p.shape[1:]!=(4,3) or not np.isfinite(p).all():
        raise ValueError('invalid tetrahedron coordinates')
    regular=np.array([[1.,.5,.5],[0.,np.sqrt(3)/2,np.sqrt(3)/6],[0.,0.,np.sqrt(2/3)]])
    jacobian=(p[:,1:]-p[:,:1]).swapaxes(1,2)
    singular=np.linalg.svd(jacobian@np.linalg.inv(regular),compute_uv=False)
    if not np.isfinite(singular).all() or np.any(singular[:,-1]<=0):
        raise ValueError('nonfinite or degenerate weighted Jacobian')
    result=singular[:,0]/singular[:,-1]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite shape condition')
    return result


def sampling_fraction(points,director,thickness):
    p=np.asarray(points,dtype=float);axis=np.asarray(director,dtype=float)
    if p.ndim!=3 or p.shape[1:]!=(4,3) or axis.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(axis).all():
        raise ValueError('invalid sampling input')
    if not np.isfinite(thickness) or thickness<=0 or abs(np.linalg.norm(axis)-1)>1e-12:
        raise ValueError('invalid physical director/thickness')
    fractions=np.ptp((p-p[:,:1])@axis,axis=1)/thickness
    if not np.isfinite(fractions).all():raise ValueError('nonfinite axial span')
    return fractions


def freeze():
    q=helpers();manifest=json.loads(MANIFEST.read_text())
    if manifest['status']!='READY_REVIEWED_PROSPECTIVE_THIN_STUDY':
        raise ValueError('thin criteria/runner not adopted for native execution')
    q.prepare(MANIFEST,WORKSPACE,ROOT/'frozen')
    paths=[Path(__file__),Path(__file__).with_name('volume_peak_worker.py'),
        Path(__file__).with_name('qualify_retained_volume.py'),
        Path(__file__).with_name('workflow2_process_supervisor.py'),
        WORKSPACE/'ANYmesh/src/anymesher/_volume_process.py',
        MANIFEST.with_name('thin_resolution_proposal_20261028.md')]
    record={str(p.relative_to(WORKSPACE)):q.digest(p) for p in paths}
    q.write_json(ROOT/'frozen/runtime-identity.json',runtime_identity())
    q.write_json(ROOT/'frozen/study-identity.json',dict(files=record,
        runtime_identity_sha256=q.digest(ROOT/'frozen/runtime-identity.json'),
        frozen_sha256=q.digest(ROOT/'frozen/frozen.json'),acceptance='all predicates and both P1/P2 results required'))


def verify_identity():
    q=helpers();identity=json.loads((ROOT/'frozen/study-identity.json').read_text())
    if q.digest(ROOT/'frozen/frozen.json')!=identity['frozen_sha256']:
        raise ValueError('frozen study changed')
    if any(q.digest(WORKSPACE/path)!=digest for path,digest in identity['files'].items()):
        raise ValueError('study runner/worker/reference/proposal changed after freeze')
    runtime_path=ROOT/'frozen/runtime-identity.json'
    if q.digest(runtime_path)!=identity['runtime_identity_sha256']:
        raise ValueError('frozen runtime identity changed')
    frozen_runtime=json.loads(runtime_path.read_text())
    if runtime_identity()!=frozen_runtime:
        raise ValueError('installed implementation or runtime changed after freeze')
    return q


def verify_inputs():
    q=verify_identity()
    manifest=json.loads((ROOT/'frozen/frozen.json').read_text())['manifest']
    for case in manifest['cases']:
        q.checked_inputs(ROOT/'frozen/frozen.json',WORKSPACE,case['id'])
    print(json.dumps(dict(status='FROZEN_IDENTITIES_VERIFIED',native_calls=0)),flush=True)


def case_run(case_id):
    q=verify_identity()
    frozen,case,source,input_folder=q.checked_inputs(ROOT/'frozen/frozen.json',WORKSPACE,case_id)
    manifest=frozen['manifest'];limits=manifest['limits'];params=case['parameters']
    directory=ROOT/'cases'/case_id;directory.mkdir(parents=True,exist_ok=False)
    result=dict(case=case_id,status='NO_GO',native_calls=0,phases={},orders=[])
    started=time.perf_counter()
    try:
        from anymesher._volume_process import _launch_worker
        from anymesher._volume_validation import validate_source
        from anymesher.volume import _admit_provider_output
        native_folder=directory/'native';native_folder.mkdir()
        shutil.copy2(input_folder/'input.npz',native_folder/'input.npz')
        options=dict(epsilon=params['boundary_tolerance']/np.linalg.norm(np.ptp(source.coordinates,axis=0)),
            edge_length_abs=params['target_edge_length'],edge_length_fac=.05,
            **manifest['options'],stop_energy=10.,num_opt_iter=80,loglevel=3,quiet=True,
            vtk_ordering=False,disable_filtering=False)
        q.write_json(native_folder/'request.json',dict(version=manifest['backend'].split('==')[1],
            native_sha256=frozen['backend']['native_sha256'],input_sha256=case['input_sha256'],options=options))
        claims=ROOT/'native-attempts';claims.mkdir(exist_ok=True)
        with (claims/(case_id+'.json')).open('x') as handle:
            json.dump(dict(case=case_id,frozen_sha256=q.digest(ROOT/'frozen/frozen.json'),maximum_native_calls=1),handle)
        if not sys.platform.startswith('linux'):
            raise ValueError('this registered lifetime-memory study requires Linux wait accounting')
        import resource
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        if usage.ru_maxrss or usage.ru_utime or usage.ru_stime:
            raise ValueError('case already reaped unrelated children; lifetime peak would be ambiguous')
        result['native_calls']=1
        process=_launch_worker([sys.executable,'-I',str(Path(__file__).with_name('volume_peak_worker.py')),str(native_folder)],
            native_folder,manifest['execution']['maximum_native_process_seconds'],None)
        if process['exit_code']!=0:raise ValueError('native worker did not complete')
        native=json.loads((native_folder/'native-result.json').read_text())
        lifetime_peak_bytes=int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*1024
        if native['raw_sha256']!=q.digest(native_folder/'raw.npz') or native['peak_rss_through_observation_bytes']<=0:
            raise ValueError('invalid native array/memory evidence')
        if lifetime_peak_bytes<native['peak_rss_through_observation_bytes']:
            raise ValueError('native lifetime peak contradicts the worker observation')
        native.update(lifetime_peak_rss_bytes=lifetime_peak_bytes,
            lifetime_peak_scope='Linux RUSAGE_CHILDREN after the only child is reaped; includes its full Python/NumPy/provider lifetime')
        result['native']=native;result['native_process']=process
        with np.load(native_folder/'raw.npz',allow_pickle=False) as data:x,t=data['coordinates'],data['cells']
        if len(t)>limits['maximum_cells']:raise ValueError('cell cap exceeded before admission')
        tick=time.perf_counter()
        mesh=_admit_provider_output(validate_source(source,params['boundary_tolerance']),x,t,
            {'backend':manifest['backend'],'raw_sha256':native['raw_sha256']},params['target_edge_length'])
        result['phases']['admission_seconds']=time.perf_counter()-tick
        points=mesh.coordinates[mesh.cells]
        condition=regular_tet_condition(points)
        fraction=sampling_fraction(points,case['analytics']['extrusion_direction'],case['analytics']['thickness'])
        metrics=q.measurements(mesh,source,case)
        metrics.update(regular_tet_condition=q.distribution(condition),axial_sampling_fraction=q.distribution(fraction))
        bounds=np.asarray(source.bounds);delta=params['boundary_tolerance']
        predicates=dict(production_geometry_topology_ownership=True,
            volume=metrics['relative_volume_error']<=limits['maximum_relative_volume_error'],
            face_areas=metrics['maximum_relative_face_area_error']<=limits['maximum_relative_face_area_error'],
            normals=metrics['maximum_face_normal_degrees']<=limits['maximum_face_normal_degrees'],
            plane=metrics['maximum_plane_distance']<=delta,
            bounds=bool(np.all(mesh.coordinates>=bounds[0]-delta) and np.all(mesh.coordinates<=bounds[1]+delta)),
            sampling=bool(np.max(fraction)<=limits['maximum_axial_span_fraction']),
            conditioning=bool(np.max(condition)<=limits['maximum_regular_tet_condition']),
            groups=all(set(map(int,mesh.boundary_groups[name]))=={int(i) for i in mesh.boundary_ids if mesh.face_owners[int(i)] in handles} for name,handles in source.face_groups.items()),
            region=set(map(int,mesh.cell_groups[source.region_id]))==set(map(int,mesh.cell_ids)))
        result.update(nodes=len(mesh.coordinates),cells=len(mesh.cells),metrics=metrics,predicates=predicates,
            mesh_validation=mesh.provenance['validation'])
        np.savez_compressed(directory/'admitted.npz',coordinates=mesh.coordinates,cells=mesh.cells,
            node_ids=mesh.node_ids,cell_ids=mesh.cell_ids,boundary_facets=mesh.boundary_facets,boundary_ids=mesh.boundary_ids)
        q.write_json(directory/'ownership.json',dict(face_owners={int(k):(str(v.model_id),v.kind,int(v.id)) for k,v in mesh.face_owners.items()},
            boundary_groups=mesh.boundary_groups,cell_groups=mesh.cell_groups))
        q.write_json(directory/'geometry-checks.json',dict(metrics=metrics,predicates=predicates))
        if not all(predicates.values()):raise ValueError('prospective geometric sampling/quality predicate failed')
        physics=load('thin_physics_reference',Path(__file__).with_name('qualify_retained_volume.py'))
        for order in (1,2):
            checked=physics.physical_checks(mesh,source,order,limits,directory)
            q.write_json(directory/f'order{order}-checks.json',checked)
            result['orders'].append(checked)
        if not all(order['status']=='PASS' for order in result['orders']):
            raise ValueError('P1/P2 physical predicate failed')
        result['status']='PASS'
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        traceback.print_exc()
    result['total_internal_seconds']=time.perf_counter()-started
    q.write_json(directory/'result.json',result)
    print(json.dumps(dict(case=case_id,status=result['status'],native_calls=result['native_calls'])),flush=True)
    return 0 if result['status']=='PASS' else 1


def supervise(name,command,timeout):
    q=helpers();ROOT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'budget.json'
    ledger=json.loads(path.read_text()) if path.exists() else dict(limit_seconds=1200,used_seconds=0.,attempts=[])
    remaining=1200-ledger['used_seconds']
    if remaining<=0:raise RuntimeError('thin-study budget exhausted')
    folder=ROOT/'processes'/name
    if folder.exists():raise FileExistsError(folder)
    helper=Path('C:/Github/ANYsolid/tools/adapter_platform_matrix.py') if os.name=='nt' else Path(__file__).with_name('workflow2_process_supervisor.py')
    module=load('thin_process_supervisor',helper)
    env=dict(os.environ)
    for key in ('PYTHONHOME','PYTHONOPTIMIZE','PYTEST_ADDOPTS','PYTEST_PLUGINS'):
        env.pop(key,None)
    env.update(PYTHONPATH=os.pathsep.join(str(WORKSPACE/repo/'src') for repo in ('ANYmesh','ANYgeometry')),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    timeout=min(timeout,remaining);receipt=None;started=time.perf_counter()
    try:
        if os.name=='nt':
            folder.mkdir(parents=True)
            receipt,_=module.windows_supervisor().run_process(command,folder,'stdout',timeout,env,Path(__file__).parents[1])
        else:
            receipt=module.run_linux(command,folder,timeout,env,Path(__file__).parents[1])
    finally:
        if receipt is None:
            receipt=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else dict(status='controller_error',process_wall_seconds=time.perf_counter()-started)
        ledger['used_seconds']+=receipt['process_wall_seconds'];ledger['remaining_seconds']=1200-ledger['used_seconds']
        entry=dict(name=name,**receipt);entry['command']=command
        ledger['attempts'].append(entry);q.write_json(path,ledger)
        q.write_json(folder/'study-identity.json',dict(command=command,runner_sha256=q.digest(__file__),
            manifest_sha256=q.digest(MANIFEST),supervisor_sha256=q.digest(helper),timeout_seconds=timeout,
            effective_environment={k:env.get(k) for k in ('PYTHONPATH','PYTHONOPTIMIZE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
    print(json.dumps(dict(name=name,status=receipt['status'],wall=receipt['process_wall_seconds'],remaining=ledger['remaining_seconds'])),flush=True)
    return 0 if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt.get('cleanup',{}).get('verified') else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('run-checks','run-linux-checks','run-freeze','run-verify','run-case','freeze','verify','case'))
    parser.add_argument('--case');args=parser.parse_args()
    if args.phase=='freeze':freeze();return 0
    if args.phase=='verify':verify_inputs();return 0
    if args.phase=='case':return case_run(args.case)
    command=[sys.executable,'-B',str(Path(__file__).resolve())]
    if args.phase=='run-checks':
        return supervise('controlled-tests',[sys.executable,'-B','-m','pytest','-q','tests/test_thin_volume_qualification.py',
            '--junitxml='+str(ROOT/'unit.xml')],60)
    if args.phase=='run-linux-checks':
        return supervise('linux-cleanup-and-metric-tests',[sys.executable,'-B','-m','pytest','-q',
            'tests/test_thin_volume_qualification.py','tests/test_workflow2_process_supervisor.py',
            '--junitxml='+str(ROOT/'linux-unit.xml'),'--basetemp=/tmp/anymesh-thin-study-units'],60)
    if args.phase=='run-freeze':return supervise('prospective-freeze',command+['freeze'],120)
    if args.phase=='run-verify':return supervise('frozen-identity-verification',command+['verify'],120)
    if args.case is None:parser.error('--case required')
    return supervise('case-'+args.case,command+['case','--case',args.case],480)


if __name__=='__main__':raise SystemExit(main())
