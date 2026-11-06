"""Prospective, bounded installed-core matrix. No public-provider promotion.

Run freeze/build once on Windows, then run-platform separately on each host.
Every supervised process consumes the same durable ledger, including failures.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

WORKSPACE = Path(__file__).resolve().parents[2]
ROOT = WORKSPACE / 'evidence/installed-matrix-20261028-a1'
VERSION = '0.4.2+anyworkflow2.exact1'
WINDOWS = {'311':'C:/Users/AudunArnesenNyhus/AppData/Local/Programs/Python/Python311/python.exe',
 '312':'C:/Users/AudunArnesenNyhus/AppData/Roaming/uv/python/cpython-3.12-windows-x86_64-none/python.exe',
 '313':'C:/Python/Python313/python.exe','314':'C:/Python/Python314/python.exe'}
LINUX = {'311':'/mnt/c/Github/ANYsolid/evidence/w1/adapter-platform-20261027T122824Z-a8d13295/managed-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11',
         '312':'/usr/bin/python3.12','313':'/home/audunarn/.local/bin/python3.13',
         '314':'/mnt/c/Github/ANYsolid/evidence/w1/adapter-platform-20261027T122824Z-a8d13295/managed-python/cpython-3.14.3-linux-x86_64-gnu/bin/python3.14'}
TESTS = ('test_solid_regions.py','test_volume_mesh.py','test_volume_process.py',
         'test_volume_consumer.py','test_volume_validation_fastpath.py')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def load_tool(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def environment():
    env=dict(os.environ)
    temporary=ROOT/'temporary';temporary.mkdir(exist_ok=True)
    for key in ('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS','VIRTUAL_ENV'):
        env.pop(key,None)
    env.update(ANYGEOMETRY_BUILD_NATIVE='0',ANYGEOMETRY_REQUIRE_NATIVE='0',ANYGEOMETRY_DISABLE_NATIVE='1',
               ANYMESHER_DISABLE_NATIVE='1',ANYMESHER_REQUIRE_NATIVE='0',PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1',
               PIP_CACHE_DIR=str(ROOT/'cache/pip'),UV_CACHE_DIR=str(ROOT/'cache/uv'),UV_PYTHON_DOWNLOADS='never',TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary))
    return env
def supervise(name,command,cwd,limit):
    ledger=read(ROOT/'budget.json');remaining=2400-ledger['used_seconds']
    if remaining<=0: raise RuntimeError('2400s cumulative process budget exhausted')
    folder=ROOT/'processes'/name
    if folder.exists():raise FileExistsError(folder)
    timeout=min(float(limit),remaining)
    helper=Path('C:/Github/ANYsolid/tools/adapter_platform_matrix.py') if os.name=='nt' else Path('/mnt/c/Github/ANYsolid/tools/adapter_platform_matrix.py')
    module=load_tool(helper,'installed_matrix_supervisor')
    start=time.perf_counter();receipt=None
    try:
        if os.name=='nt':
            folder.mkdir(parents=True)
            receipt,_=module.windows_supervisor().run_process(list(map(str,command)),folder,'stdout',timeout,environment(),Path(cwd))
        else:
            if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError('subreaper unavailable')
            receipt=module.linux_process(list(map(str,command)),folder,timeout,environment(),Path(cwd))
    finally:
        if receipt is None:
            receipt=read(folder/'result.json') if (folder/'result.json').exists() else {'status':'controller_error','process_wall_seconds':time.perf_counter()-start}
        ledger['used_seconds']+=receipt['process_wall_seconds']
        ledger['remaining_seconds']=max(0.,2400-ledger['used_seconds'])
        ledger['attempts'].append({'name':name,'command':list(map(str,command)),'timeout_seconds':timeout,**receipt})
        save(ROOT/'budget.json',ledger)
        save(folder/'matrix-identity.json',{'runner_sha256':sha(__file__),'manifest_sha256':sha(ROOT/'manifest.json'),
             'supervisor_sha256':sha(helper),'environment':{k:v for k,v in environment().items() if k.startswith(('ANY','OMP','OPENBLAS','MKL','PYTHON','PIP_CACHE'))}})
    print(json.dumps({'process':name,'status':receipt['status'],'wall':receipt['process_wall_seconds'],'remaining':ledger['remaining_seconds']}),flush=True)
    if receipt['status']!='completed' or receipt.get('exit_code')!=0 or not receipt.get('cleanup',{}).get('verified'):
        raise RuntimeError('process failed; retained evidence: '+str(folder))
    return receipt

def freeze():
    if ROOT.exists():raise FileExistsError(ROOT)
    ROOT.mkdir(parents=True)
    backend=WORKSPACE/'evidence/exact-envelope-20261028-a1'
    records={}
    for key,rel in {'linux311':'linux311/import-only-smoke.json','linuxabi3':'import-only-smoke.json',
                    'windows311':'windows-import-only-smoke.json','windowsabi3':'windows312/import-only-smoke.json'}.items():
        record=read(backend/rel)
        wheel=Path(record['wheel'].replace('/mnt/c/','C:/'))
        assert sha(wheel)==record['wheel_sha256']
        native=record['native_sha256']
        if isinstance(native,str):native={'pytetwild/PyfTetWildWrapper.pyd':native}
        records[key]={'wheel':str(wheel.relative_to(WORKSPACE)).replace('\\','/'),'wheel_sha256':sha(wheel),
                      'native_sha256':native,'version':VERSION,'compiled_exact_envelope':True}
    solid=Path('C:/Github/ANYsolid/evidence/w1/platform-artifact-refresh-20261027/dist/anysolid-0.1.0-py3-none-any.whl')
    assert sha(solid)=='601bd500625e6cd95f3a741f4f19b87bdcb4a72a7c7650278fea8be3efc56c3c'
    (ROOT/'wheels').mkdir();shutil.copy2(solid,ROOT/'wheels'/solid.name)
    hashes={}
    for repo in ('ANYgeometry','ANYmesh'):
        source=WORKSPACE/repo;target=ROOT/'sources'/repo;target.mkdir(parents=True)
        for p in source.iterdir():
            if p.is_file() and p.suffix in ('.py','.toml','.md','.rst','.json','.txt',''):
                shutil.copy2(p,target/p.name)
        shutil.copytree(source/'src',target/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyd','*.so','*.dll','*.egg-info'))
        (target/'docs').mkdir(exist_ok=True)
        if (source/'docs/LICENSE.md').exists():shutil.copy2(source/'docs/LICENSE.md',target/'docs/LICENSE.md')
        for p in target.rglob('*'):
            if p.is_file():hashes[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    testdir=ROOT/'tests';testdir.mkdir()
    for name in TESTS:
        repo='ANYgeometry' if name=='test_solid_regions.py' else 'ANYmesh'
        shutil.copy2(WORKSPACE/repo/'tests'/name,testdir/name)
    (testdir/'tools').mkdir();(testdir/'tools/__init__.py').write_text('')
    for name in ('run_affected_tests.py','run_dev_smoke.py'):
        shutil.copy2(WORKSPACE/'ANYmesh/tools'/name,testdir/'tools'/name)
    shutil.copy2(WORKSPACE/'ANYmesh/tools/qualify_retained_volume.py',ROOT/'physics_reference.py')
    hashes['physics_reference.py']=sha(ROOT/'physics_reference.py')
    for p in testdir.rglob('*.py'):hashes[str(p.relative_to(ROOT)).replace('\\','/')]=sha(p)
    manifest={'schema_version':1,'status':'FROZEN_BEFORE_BUILDS_AND_NATIVE_CALLS','scope':'installed core-only; optional core extensions disabled; no public guard change',
      'cells':[p+v for p in ('windows','linux') for v in ('311','312','313','314')],
      'case':{'dimensions':[.1,.1,.1],'target_edge_length':.025,'boundary_tolerance':1e-5},
      'limits':{'maximum_cells':10000,'maximum_coefficients':150000,'residual':1e-9,'work_symmetry':1e-10,'resultant':1e-8,'patch_tensors':1e-10,'solved_displacement':1e-7,
                'relative_volume':1e-3,'relative_face_area':1e-3,'normal_degrees':1.},
      'execution':{'maximum_native_calls':8,'maximum_calls_per_cell':1,'native_timeout_seconds':120,'cell_timeout_seconds':300,'total_process_wall_seconds':2400},
      'backends':records,'source_hashes':hashes,'solid_wheel_sha256':sha(solid),'requested_dependencies':['numpy==2.4.6','scipy==1.17.1','scikit-fem==12.0.2'],
      'omissions':['optional compiled ANYgeometry/ANYmesher cores','MacOS','full repository/release gates','provider promotion']}
    save(ROOT/'manifest.json',manifest)
    save(WORKSPACE/'ANYmesh/docs/volume_installed_manifest_20261028.json',manifest)
    save(ROOT/'budget.json',{'limit_seconds':2400,'used_seconds':0.,'remaining_seconds':2400.,'attempts':[]})
    print(json.dumps({'frozen':str(ROOT),'manifest_sha256':sha(ROOT/'manifest.json'),'source_files':len(hashes)}))

def build():
    python=Path('C:/ANYworkspaceWork/workflow2-20261027/runtime/Scripts/python.exe')
    for repo in ('ANYgeometry','ANYmesh'):
        name='build-'+repo+('-owned-temp' if (ROOT/'processes'/('build-'+repo+'-offline')).exists() else '-offline' if (ROOT/'processes'/('build-'+repo)).exists() else '')
        supervise(name,[python,'-m','pip','wheel','--no-build-isolation','--no-deps','--wheel-dir',ROOT/'wheels',ROOT/'sources'/repo],ROOT,180)
    artifacts={p.name:sha(p) for p in (ROOT/'wheels').glob('*.whl')}
    for p in (ROOT/'wheels').glob('*.whl'):
        if p.name.lower().startswith(('anygeometry','anymesher')):
            with zipfile.ZipFile(p) as z:
                assert not any(n.endswith(('.pyd','.so','.dll')) for n in z.namelist())
    save(ROOT/'core-artifacts.json',artifacts)

def run_platform():
    platform='windows' if os.name=='nt' else 'linux'
    for version in ('311','312','313','314'):
        cell=platform+version;directory=ROOT/'cells'/cell
        if (directory/'native-attempt.json').exists():
            if (directory/'result.json').exists() and read(directory/'result.json')['status']=='PASS':continue
            raise RuntimeError('native attempt consumed; cannot retry '+cell)
        directory.mkdir(parents=True,exist_ok=True)
        ledger=read(ROOT/'budget.json')
        used_before=ledger['used_seconds']-sum(a['process_wall_seconds'] for a in ledger['attempts'] if a['name'].startswith(cell+'-'))
        def run(name,command,limit):
            prior=ROOT/'processes'/name/'result.json'
            if prior.exists():
                receipt=read(prior)
                if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt.get('cleanup',{}).get('verified'):return receipt
                name+='-repair'
                repaired=ROOT/'processes'/name/'result.json'
                if repaired.exists():
                    receipt=read(repaired)
                    if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt.get('cleanup',{}).get('verified'):return receipt
            remaining=300-(read(ROOT/'budget.json')['used_seconds']-used_before)
            if remaining<=0:raise RuntimeError('300s complete cell process budget exhausted')
            return supervise(name,command,directory,min(limit,remaining))
        if platform=='windows':
            base=Path(WINDOWS[version]);run(cell+'-venv',[base,'-m','venv',directory/'venv'],60)
            python=directory/'venv/Scripts/python.exe'
        else:
            uv='/home/audunarn/.local/bin/uv'
            run(cell+'-venv',[uv,'venv','--python',LINUX[version],'--seed',directory/'venv'],60)
            python=directory/'venv/bin/python'
        manifest=read(ROOT/'manifest.json');backend=manifest['backends'][platform+('311' if version=='311' else 'abi3')]
        wheel=WORKSPACE/backend['wheel'];assert sha(wheel)==backend['wheel_sha256']
        artifacts=read(ROOT/'core-artifacts.json')
        wheels=[ROOT/'wheels'/n for n in artifacts]
        for p in wheels:assert sha(p)==artifacts[p.name]
        run(cell+'-install',[python,'-m','pip','install',*wheels,wheel,*manifest['requested_dependencies'],'shapely','pytest','threadpoolctl'],180)
        run(cell+'-tests',[python,'-I','-m','pytest','-q','-o','pythonpath=','-k','not explicit_affected_test_mapping',*map(str,(ROOT/'tests'/n for n in TESTS)),
                    '--junitxml='+str(directory/'pytest.xml'),'-p','no:cacheprovider'],120)
        run(cell+'-native-physics',[python,'-I',Path(__file__),'cell','--cell',cell],300)

def cell_run(cell):
    import importlib.metadata as metadata
    import numpy as np
    import anygeometry,anymesher,anysolid
    manifest=read(ROOT/'manifest.json');directory=ROOT/'cells'/cell
    result={'cell':cell,'status':'FAILED','native_calls':0,'origins':{},'python':sys.version,'executable':sys.executable}
    started=time.perf_counter()
    try:
        assert sys.version_info[:2]==(3,int(cell[-2:]))
        for module in (anygeometry,anymesher,anysolid):
            origin=Path(module.__file__).resolve();assert origin.is_relative_to(directory/'venv')
            result['origins'][module.__name__]=str(origin)
        assert not any(n in sys.modules for n in ('pytetwild','pyvista'))
        assert not any(n.startswith(('anygeometry.','anymesher.')) and str(getattr(module,'__file__','')).endswith(('.pyd','.so')) for n,module in sys.modules.items())
        from anymesher._volume_process import _PROVIDER_QUALIFIED,_launch_worker
        assert _PROVIDER_QUALIFIED is False
        result['lazy_base_import']=True
        import pytetwild
        from pytetwild import PyfTetWildWrapper as native
        backend=manifest['backends'][('windows' if os.name=='nt' else 'linux')+('311' if cell.endswith('311') else 'abi3')]
        assert metadata.version('pytetwild')==VERSION and native._workflow2_exact_envelope is True
        assert Path(native.__file__).resolve().is_relative_to(directory/'venv')
        native_key='pytetwild/'+Path(native.__file__).name
        assert sha(native.__file__)==backend['native_sha256'][native_key]
        result['backend']={'version':VERSION,'native_sha256':sha(native.__file__),'exact_envelope':True,'wheel_sha256':backend['wheel_sha256']}
        result['versions']={name:metadata.version(name) for name in ('numpy','scipy','scikit-fem','ANYgeometry','ANYmesher','anysolid')}
        model=anygeometry.GeometryModel();rid=model.add_solid_box((0,0,0),manifest['case']['dimensions'])
        source=model.solid_region_snapshot(rid)
        model.set_solid_face_group(rid,'load_face',(source.face_refs[1],));model.set_solid_face_group(rid,'shell_transition',(source.face_refs[0],source.face_refs[1]))
        source=model.solid_region_snapshot(rid)
        facets=np.asarray([(loop[0],loop[k],loop[k+1]) for loop in source.face_loops for k in range(1,len(loop)-1)],dtype=np.int64)
        np.savez(directory/'input.npz',coordinates=source.coordinates,facets=facets)
        options=dict(epsilon=manifest['case']['boundary_tolerance']/np.linalg.norm(np.ptp(source.coordinates,axis=0)),edge_length_abs=.025,edge_length_fac=.05,
              simplify=False,coarsen=False,optimize=True,num_threads=1,stop_energy=10.,num_opt_iter=80,loglevel=3,quiet=True,vtk_ordering=False,disable_filtering=False)
        save(directory/'request.json',{'options':options,'backend':backend,'input_sha256':sha(directory/'input.npz')})
        with (directory/'native-attempt.json').open('x') as f:json.dump({'maximum_calls':1,'input_sha256':sha(directory/'input.npz')},f)
        worker=directory/'native_worker.py'
        worker.write_text("import sys\nassert sys.stdin.readline().strip()=='start'\nimport numpy as np,json,pytetwild\nfrom pathlib import Path\np=Path(sys.argv[1])\nd=np.load(p/'input.npz',allow_pickle=False)\nx,t=pytetwild.tetrahedralize(d['coordinates'],d['facets'],**json.loads((p/'request.json').read_text())['options'])\nnp.savez(p/'raw.npz',coordinates=x,cells=t)\n",encoding='utf-8')
        result['native_calls']=1
        receipt=_launch_worker([sys.executable,str(worker),str(directory)],directory,120,None)
        result['native_process']=receipt
        if receipt['exit_code']!=0:raise ValueError('native worker failed')
        with np.load(directory/'raw.npz',allow_pickle=False) as data:x,t=data['coordinates'],data['cells']
        if len(t)>manifest['limits']['maximum_cells']:raise ValueError('cell cap exceeded')
        from anymesher._volume_validation import validate_source
        from anymesher.volume import _admit_provider_output,validate_volume_mesh
        mesh=_admit_provider_output(validate_source(source,1e-5),x,t,{'backend':VERSION},.025)
        validate_volume_mesh(mesh,source,boundary_tolerance=1e-5)
        p=mesh.coordinates[mesh.cells];volume=np.linalg.det(p[:,1:]-p[:,:1]).sum()/6
        volume_error=abs(volume/.001-1);area_error=0.;normal_error=0.
        for i,handle in enumerate(source.face_refs):
            faces=mesh.boundary_facets[[mesh.face_owners[int(key)]==handle for key in mesh.boundary_ids]]
            pts=mesh.coordinates[faces];cross=np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0]);norm=np.linalg.norm(cross,axis=1)
            facepts=source.coordinates[list(source.face_loops[i])];n=np.cross(facepts[1]-facepts[0],facepts[2]-facepts[0]);n/=np.linalg.norm(n)
            area_error=max(area_error,abs(norm.sum()/2/.01-1))
            normal_error=max(normal_error,float(np.max(np.degrees(np.arccos(np.clip(cross@n/norm,-1,1))))))
        assert volume_error<=1e-3 and area_error<=1e-3 and normal_error<=1.
        result.update(nodes=len(mesh.coordinates),cells=len(mesh.cells),geometry={'relative_volume':float(volume_error),'relative_face_area':float(area_error),'normal_degrees':normal_error})
        physics=load_tool(ROOT/'physics_reference.py','installed_physics_reference')
        assert sha(ROOT/'physics_reference.py')==manifest['source_hashes']['physics_reference.py']
        result['orders']=[physics.physical_checks(mesh,source,order,manifest['limits'],directory) for order in (1,2)]
        assert all(r['status']=='PASS' for r in result['orders'])
        np.savez_compressed(directory/'admitted.npz',coordinates=mesh.coordinates,cells=mesh.cells,boundary_facets=mesh.boundary_facets)
        result['status']='PASS'
    except Exception as exc:
        result.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());traceback.print_exc()
    result['cell_internal_seconds']=time.perf_counter()-started
    if os.name!='nt':
        import resource
        result['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peak_rss_scope']='cell Python process, excludes separately owned native child'
    if (directory/'result.json').exists():
        shutil.copy2(directory/'result.json',directory/'pre-native-failed-result.json')
    save(directory/'result.json',result)
    print(json.dumps({'cell':cell,'status':result['status'],'native_calls':result['native_calls']}),flush=True)
    return 0 if result['status']=='PASS' else 1

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('freeze','build','run-platform','cell'));p.add_argument('--cell')
    a=p.parse_args()
    if a.phase=='cell':return cell_run(a.cell)
    {'freeze':freeze,'build':build,'run-platform':run_platform}[a.phase]()
    return 0
if __name__=='__main__':raise SystemExit(main())
