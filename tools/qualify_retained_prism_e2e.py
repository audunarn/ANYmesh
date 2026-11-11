"""Prospective retained-prism E2E: no native meshing execution, no promotion."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import zipfile
import numpy as np

WORKSPACE=Path(__file__).resolve().parents[2]
ROOT=WORKSPACE/'evidence/prism-e2e-20261028-a1'
MANIFEST=Path(__file__).resolve().parents[1]/'docs/prism_e2e_manifest_20261028.json'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def helpers():
    return load('prism_geometry_reference',Path(__file__).with_name('qualify_volume_geometry.py'))

def registration(manifest):
    execution=manifest['execution']
    if execution!={'maximum_native_calls':0,'maximum_physics_attempts':1,'process_budget_seconds':600,
                   'maximum_process_seconds':300,'verification_reserve_seconds':45,'retries':0}:
        raise ValueError('execution authority changed')
    expected=dict(maximum_cells=100000,maximum_coefficients=150000,maximum_relative_volume_error=.001,
        maximum_relative_face_area_error=.001,maximum_face_normal_degrees=1,maximum_regular_tet_condition=32,
        residual=1e-9,work_symmetry=1e-10,resultant=1e-8,patch_tensors=1e-10,solved_displacement=1e-7)
    if manifest['limits']!=expected or manifest['boundary_tolerance']!=6e-6:
        raise ValueError('scientific registration changed')
    if manifest['case_id']!='pentagon' or (manifest['expected_nodes'],manifest['expected_cells'])!=(463,1899):
        raise ValueError('retained case changed')
    if manifest['status']!='PROSPECTIVE_RETAINED_PRISM_HOLD_FOR_ROOT':raise ValueError('invalid registration state')
    q=helpers()
    for name,digest in manifest['helper_sha256'].items():
        if q.digest(Path(__file__).with_name(name))!=digest:raise ValueError('helper changed: '+name)

def checked_source(manifest):
    import anygeometry
    q=helpers();registration(manifest)
    frozen_path=WORKSPACE/manifest['original_frozen']
    if q.digest(frozen_path)!=manifest['original_frozen_sha256']:raise ValueError('original frozen record changed')
    frozen=json.loads(frozen_path.read_text());case=frozen['cases'][manifest['case_id']]
    folder=frozen_path.parent/manifest['case_id']
    for filename,key in [('geometry.json','geometry_sha256'),('input.npz','input_sha256')]:
        if q.digest(folder/filename)!=case[key]:raise ValueError('original source/input changed')
    for name,key in [('raw_path','raw_sha256'),('native_receipt','native_receipt_sha256')]:
        if q.digest(WORKSPACE/manifest[name])!=manifest[key]:raise ValueError('retained native evidence changed')
    receipt=json.loads((WORKSPACE/manifest['native_receipt']).read_text())
    if receipt['frozen_sha256']!=manifest['original_frozen_sha256'] or receipt['input_sha256']!=case['input_sha256'] or receipt['snapshot_sha256']!=case['snapshot_sha256']:
        raise ValueError('native receipt does not identify frozen input')
    backend=q.backend_identity(frozen['manifest'],WORKSPACE)
    if backend!=receipt['backend']:raise ValueError('native artifact identity changed')
    model=anygeometry.from_dict(json.loads((folder/'geometry.json').read_text()))
    source=model.solid_region_snapshot(manifest['expected_source']['region_id'])
    if not model.strict_audit().certifiable:raise ValueError('source is not certifiable')
    if q.canonical_hash(q.snapshot_record(source))!=case['snapshot_sha256']:raise ValueError('snapshot changed')
    for name,value in manifest['expected_source'].items():
        if getattr(source,name)!=value:raise ValueError('source identity changed: '+name)
    radius=manifest['analytics']['radius'];height=manifest['analytics']['height']
    area=5*radius**2*np.sin(2*np.pi/5)/2
    expected_areas=np.array([area,area,*([2*radius*np.sin(np.pi/5)*height]*5)])
    if not np.isclose(source.volume,area*height,rtol=1e-12,atol=0) or not np.allclose(case['analytics']['face_areas'],expected_areas,rtol=1e-12,atol=0):
        raise ValueError('independent regular-pentagon analytics disagree')
    cap=np.asarray(source.coordinates[:5]);center=cap.mean(axis=0)
    if not np.allclose(np.linalg.norm(cap-center,axis=1),radius,rtol=1e-12,atol=0) or not np.allclose(source.coordinates[5:]-cap,[0,0,height],rtol=1e-12,atol=1e-14):
        raise ValueError('source dimensions disagree')
    return source,case,backend

def runtime_identity():
    thin=load('prism_metrics_runtime',Path(__file__).with_name('qualify_thin_volume.py'))
    identity=thin.runtime_identity()
    import anysolid,importlib.metadata
    origin=Path(anysolid.__file__).resolve()
    runtime=WORKSPACE/'exact-runtime-linux312'
    if not origin.is_relative_to(runtime.resolve()):raise ValueError('ANYsolid is not installed in the registered runtime')
    if importlib.metadata.version('anysolid')!='0.1.0':raise ValueError('installed consumer version changed')
    manifest=json.loads(MANIFEST.read_text());wheel=WORKSPACE/manifest['installed_anysolid_wheel']
    if thin.stream_sha(wheel)!=manifest['installed_anysolid_wheel_sha256']:raise ValueError('registered consumer wheel changed')
    matched=0
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not name.startswith('anysolid/') or name.endswith('/'):continue
            installed=origin.parent.parent/name
            if not installed.is_file() or thin.stream_sha(installed)!=hashlib.sha256(archive.read(name)).hexdigest():
                raise ValueError('installed consumer differs from registered wheel: '+name)
            matched+=1
    if matched==0:raise ValueError('empty consumer wheel comparison')
    identity['consumer_wheel']=dict(sha256=thin.stream_sha(wheel),matched_package_files=matched)
    identity['origins']['anygeometry']=str((WORKSPACE/'ANYgeometry/src/anygeometry/__init__.py').resolve())
    identity['origins']['anymesher']=str((WORKSPACE/'ANYmesh/src/anymesher/__init__.py').resolve())
    for path in (WORKSPACE/'ANYmesh/src/anymesher').rglob('*.py'):
        identity['files'][str(path.resolve())]=thin.stream_sha(path)
    return identity

def freeze():
    q=helpers();manifest=json.loads(MANIFEST.read_text());source,case,backend=checked_source(manifest)
    folder=ROOT/'frozen';folder.mkdir(parents=True,exist_ok=False)
    q.write_json(folder/'runtime.json',runtime_identity())
    q.write_json(folder/'manifest.json',manifest)
    paths=[Path(__file__),MANIFEST,*[Path(__file__).with_name(name) for name in manifest['helper_sha256']]]
    record=dict(files={str(path.relative_to(WORKSPACE)):q.digest(path) for path in paths},
        runtime_sha256=q.digest(folder/'runtime.json'),manifest_sha256=q.digest(folder/'manifest.json'),
        original_case=case,backend=backend,source=q.snapshot_record(source),native_calls=0)
    q.write_json(folder/'identity.json',record)
    print(json.dumps(dict(status='FROZEN_HOLD_FOR_ROOT',identity_sha256=q.digest(folder/'identity.json'),native_calls=0)),flush=True)

def verify():
    q=helpers();folder=ROOT/'frozen';identity=json.loads((folder/'identity.json').read_text())
    for path,digest in identity['files'].items():
        if q.digest(WORKSPACE/path)!=digest:raise ValueError('runner/registration/helper changed after freeze: '+path)
    if q.digest(folder/'runtime.json')!=identity['runtime_sha256'] or q.digest(folder/'manifest.json')!=identity['manifest_sha256']:
        raise ValueError('frozen runtime/manifest changed')
    if runtime_identity()!=json.loads((folder/'runtime.json').read_text()):raise ValueError('runtime/source changed')
    manifest=json.loads((folder/'manifest.json').read_text());source,case,backend=checked_source(manifest)
    if q.canonical_hash(q.snapshot_record(source))!=q.canonical_hash(identity['source']):raise ValueError('source record changed')
    return manifest,source,case

def diagnostics(mesh,source,order,output,physics):
    import anysolid
    gradient=np.array([[.03,.08,-.02],[.04,-.01,.05],[.02,-.03,.06]])
    strain=(gradient+gradient.T)/2;E=1000.;nu=.3
    stress=2*(E/(2*(1+nu)))*strain+(E*nu/((1+nu)*(1-2*nu)))*np.trace(strain)*np.eye(3)
    prepared=anysolid.prepare(anysolid.SolidMesh.snapshot(mesh),{'elastic_modulus':E,'poisson_ratio':nu},order=order)
    values=prepared.coefficient_coordinates@gradient.T+[.1,-.2,.3]
    recovered=prepared.recover(values[np.arange(prepared.ndof),prepared.coefficient_components])
    resultants=[]
    for i,loop in enumerate(source.face_loops):
        area,center,normal=physics.face_measure(source.coordinates[list(loop)])
        group='start' if i==0 else 'end' if i==1 else f'side_{i-2}'
        force,moment=physics.vector_resultants(prepared,prepared.traction(group,stress@normal))
        resultants.append(dict(group=group,force=force,moment=moment,reference_force=area*(stress@normal),reference_moment=np.cross(center,area*(stress@normal))))
    with np.load(output/f'order{order}-solution.npz',allow_pickle=False) as solution:
        ndof=len(solution['coefficients']);fixed=len(solution['fixed'])
    return dict(affine_strain_mean=np.mean(recovered.strain,axis=tuple(range(recovered.strain.ndim-2))),affine_stress_mean=np.mean(recovered.stress,axis=tuple(range(recovered.stress.ndim-2))),
        reference_strain=strain,reference_stress=stress,reference_positive_strain_work=source.volume*float(np.sum(strain*stress)),
        face_resultants=resultants,tension_solver=dict(coefficients=ndof,fixed=fixed,free=ndof-fixed,sparse_lu_solves=1),
        assembly_calls_in_unchanged_physical_checks=2,recovery_calls_in_unchanged_physical_checks=1,additional_diagnostic_recovery_calls=1)

def case_run():
    q=helpers();output=ROOT/'case';output.mkdir(parents=True,exist_ok=False)
    result=dict(status='NO_GO',native_calls=0,orders=[]);started=time.perf_counter()
    try:
        manifest,source,case=verify()
        release=json.loads((ROOT/'root-release.json').read_text())
        if release.get('approved') is not True or release.get('identity_sha256')!=q.digest(ROOT/'frozen/identity.json'):
            raise ValueError('non-author root release missing or does not match frozen identity')
        with (ROOT/'physics-attempt.json').open('x') as handle:json.dump(dict(maximum_attempts=1,native_calls=0),handle)
        from anymesher._volume_validation import validate_source
        from anymesher.volume import _admit_provider_output,validate_volume_mesh
        import anysolid
        with np.load(WORKSPACE/manifest['raw_path'],allow_pickle=False) as data:x,t=data['coordinates'],data['cells']
        if (len(x),len(t))!=(manifest['expected_nodes'],manifest['expected_cells']):raise ValueError('raw mesh count changed')
        mesh=_admit_provider_output(validate_source(source,manifest['boundary_tolerance']),x,t,
            dict(backend='retained-original-exact1',retained_raw_sha256=manifest['raw_sha256']),case['parameters']['target_edge_length'])
        thin=load('prism_condition',Path(__file__).with_name('qualify_thin_volume.py'))
        metrics=q.measurements(mesh,source,case);condition=thin.regular_tet_condition(mesh.coordinates[mesh.cells]);limits=manifest['limits']
        delta=manifest['boundary_tolerance'];bounds=np.asarray(source.bounds)
        predicates=dict(volume=metrics['relative_volume_error']<=limits['maximum_relative_volume_error'],
            face_areas=metrics['maximum_relative_face_area_error']<=limits['maximum_relative_face_area_error'],
            normals=metrics['maximum_face_normal_degrees']<=limits['maximum_face_normal_degrees'],
            plane=metrics['maximum_plane_distance']<=delta,bounds=bool(np.all(mesh.coordinates>=bounds[0]-delta) and np.all(mesh.coordinates<=bounds[1]+delta)),
            conditioning=bool(np.max(condition)<=limits['maximum_regular_tet_condition']),cell_cap=len(mesh.cells)<=limits['maximum_cells'],
            groups=all(set(map(int,mesh.boundary_groups[name]))=={int(i) for i in mesh.boundary_ids if mesh.face_owners[int(i)] in handles} for name,handles in source.face_groups.items()),
            region=set(map(int,mesh.cell_groups[source.region_id]))==set(map(int,mesh.cell_ids)))
        metrics['regular_tet_condition']=q.distribution(condition)
        result.update(nodes=len(x),cells=len(t),geometry_metrics=metrics,geometry_predicates=predicates,source=q.snapshot_record(source))
        q.write_json(output/'geometry.json',dict(metrics=metrics,predicates=predicates))
        if not all(predicates.values()):raise ValueError('prospective geometry predicate failed')
        consumer=anysolid.SolidMesh.snapshot(mesh)
        result['consumer_snapshot_counts']=dict(nodes=len(consumer.coordinates),cells=len(consumer.cells))
        physics=load('prism_unchanged_physics',Path(__file__).with_name('qualify_retained_volume.py'))
        for order in (1,2):
            checked=physics.physical_checks(mesh,source,order,limits,output)
            q.write_json(output/f'order{order}-checks.json',checked);result['orders'].append(checked)
            if checked['status']!='PASS':raise ValueError(f'P{order} physical predicate failed')
            q.write_json(output/f'order{order}-diagnostics.json',diagnostics(mesh,source,order,output,physics))
        result['readmission']=validate_volume_mesh(mesh,source,boundary_tolerance=delta)
        q.write_json(output/'ownership.json',dict(face_owners={int(k):(str(v.model_id),v.kind,int(v.id)) for k,v in mesh.face_owners.items()},boundary_groups=mesh.boundary_groups,cell_groups=mesh.cell_groups))
        np.savez_compressed(output/'admitted.npz',coordinates=mesh.coordinates,cells=mesh.cells,node_ids=mesh.node_ids,cell_ids=mesh.cell_ids,boundary_facets=mesh.boundary_facets,boundary_ids=mesh.boundary_ids)
        result.update(status='PASS',physical_checks=sum(len(order['checks']) for order in result['orders']),native_calls=0)
    except Exception as error:
        result.update(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());traceback.print_exc()
    result['internal_seconds']=time.perf_counter()-started;q.write_json(output/'result.json',result)
    print(json.dumps(dict(status=result['status'],native_calls=0,orders=len(result['orders']))),flush=True)
    return 0 if result['status']=='PASS' else 1

def controls():
    import copy
    manifest=json.loads(MANIFEST.read_text());registration(manifest);checks=['accepted_registered_zero_native_case']
    for name,mutate in [('extra_native_call',lambda m:m['execution'].__setitem__('maximum_native_calls',1)),
        ('relaxed_numerics',lambda m:m['limits'].__setitem__('residual',1e-8)),
        ('wrong_retained_count',lambda m:m.__setitem__('expected_cells',1900)),
        ('changed_helper_identity',lambda m:m['helper_sha256'].__setitem__('qualify_retained_volume.py','0'*64))]:
        changed=copy.deepcopy(manifest);mutate(changed)
        try:registration(changed)
        except ValueError:checks.append(name);continue
        raise AssertionError('control failed: '+name)
    for key in ('original_frozen_sha256','raw_sha256'):
        changed=copy.deepcopy(manifest);changed[key]='0'*64
        try:checked_source(changed)
        except ValueError:checks.append('refuse_mutated_'+key);continue
        raise AssertionError('immutable input control failed: '+key)
    helpers().write_json(ROOT/'controls.json',dict(status='PASS',passed=len(checks),failed=0,checks=checks,native_calls=0))
    print(json.dumps(dict(status='PASS',passed=len(checks),failed=0,native_calls=0)),flush=True)

def supervise(name,command,timeout,reserve=0):
    q=helpers();ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'budget.json'
    ledger=json.loads(path.read_text()) if path.exists() else dict(limit_seconds=600,used_seconds=0.,attempts=[],native_calls=0)
    remaining=600-ledger['used_seconds'];timeout=min(timeout,300,remaining-reserve)
    if timeout<=0:raise RuntimeError('registered process budget/reserve exhausted')
    folder=ROOT/'processes'/name;helper=Path(__file__).with_name('workflow2_process_supervisor.py')
    module=load('prism_descendant_supervisor',helper);env=dict(os.environ)
    for key in ('PYTHONHOME','PYTHONOPTIMIZE','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(key,None)
    env.update(PYTHONPATH=os.pathsep.join(str(WORKSPACE/repo/'src') for repo in ('ANYmesh','ANYgeometry')),
        PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
    receipt=None;started=time.perf_counter()
    try:receipt=module.run_linux(command,folder,timeout,env,Path(__file__).parents[1])
    finally:
        if receipt is None:receipt=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else dict(status='controller_error',process_wall_seconds=time.perf_counter()-started)
        ledger['used_seconds']+=receipt['process_wall_seconds'];ledger['remaining_seconds']=600-ledger['used_seconds']
        entry=dict(receipt);entry.update(name=name,command=command)
        ledger['attempts'].append(entry);q.write_json(path,ledger)
        q.write_json(folder/'study-identity.json',dict(command=command,timeout_seconds=timeout,runner_sha256=q.digest(__file__),manifest_sha256=q.digest(MANIFEST),supervisor_sha256=q.digest(helper),effective_environment={key:env.get(key) for key in ('PYTHONPATH','PYTHONOPTIMIZE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},native_calls=0))
    print(json.dumps(dict(name=name,status=receipt['status'],wall=receipt['process_wall_seconds'],remaining=ledger['remaining_seconds'],native_calls=0)),flush=True)
    return 0 if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt.get('cleanup',{}).get('verified') else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('controls','freeze','verify','case','run-controls','run-freeze','run-verify','run-case'));args=p.parse_args()
    if args.phase.startswith('run-'):
        phase=args.phase[4:];return supervise(phase,[sys.executable,'-B',str(Path(__file__).resolve()),phase],300 if phase=='case' else 120,45 if phase=='case' else 0)
    if args.phase=='controls':controls();return 0
    if args.phase=='freeze':freeze();return 0
    if args.phase=='verify':verify();print(json.dumps(dict(status='VERIFIED',native_calls=0)));return 0
    return case_run()

if __name__=='__main__':raise SystemExit(main())
