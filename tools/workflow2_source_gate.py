"""Fresh bounded W2 source-prerequisite checks; no full-suite authority."""
from __future__ import annotations
import argparse,ctypes,hashlib,importlib.util,json,os,subprocess,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'evidence/source-gate-repair-20261028-a1'
LIMIT=1200.

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(a):
    ROOT.mkdir(exist_ok=True,parents=True);ledger_path=ROOT/'budget.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text())
    else:ledger={'limit':LIMIT,'used':0.,'attempts':[]}
    remaining=LIMIT-ledger['used']
    if a.timeout>300 or a.timeout<=0 or remaining<=0:raise ValueError('execution bound unavailable')
    folder=ROOT/a.name;folder.mkdir(exist_ok=False)
    env=dict(os.environ)
    for key in ('PYTEST_ADDOPTS','PYTEST_PLUGINS','PYTHONHOME'):
        env.pop(key,None)
    env['PYTHONNOUSERSITE']='1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    env.update(PYTHONPATH=os.pathsep.join(a.pythonpath),PYTHONUTF8='1',PIP_CACHE_DIR=str(ROOT/'cache/pip'),UV_CACHE_DIR=str(ROOT/'cache/uv'),UV_PYTHON_DOWNLOADS='never')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if command[0]=='python':command[0]=sys.executable
    if '-m' in command and 'pytest' in command:command+=['--junitxml='+str(folder/'pytest.xml')]
    repo=Path(__file__).resolve().parents[1]
    identity={'command':command,'cwd':str(repo),'python':sys.version,'env':{k:env[k] for k in ('PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PIP_CACHE_DIR','UV_CACHE_DIR','UV_PYTHON_DOWNLOADS')},'source_sha256':{str(p.relative_to(repo)):sha(p) for p in [repo/'src/anymesher/quad/recovery.py',repo/'src/anymesher/quad/front.py',repo/'tools/workflow2_source_gate.py',repo/'tests/quad_first/test_q2_recovery.py']},'timeout':min(a.timeout,remaining),'budget_used_before':ledger['used']}
    save(folder/'identity.json',identity)
    helper=Path('C:/Github/ANYsolid/tools/adapter_platform_matrix.py') if os.name=='nt' else repo/'tools/workflow2_process_supervisor.py'
    identity.update(supervisor_path=str(helper),supervisor_sha256=sha(helper))
    save(folder/'identity.json',identity)
    spec=importlib.util.spec_from_file_location('source_gate_supervisor',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if os.name=='nt':receipt,_=module.windows_supervisor().run_process(command,folder,'stdout',identity['timeout'],env,repo)
    else:
        receipt=module.run_linux(command,folder/'process',identity['timeout'],env,repo)
    ledger['used']+=receipt['process_wall_seconds'];ledger['remaining']=LIMIT-ledger['used'];ledger['attempts'].append({'name':a.name,**receipt});save(ledger_path,ledger)
    print(json.dumps({'folder':str(folder),'remaining':ledger['remaining'],'status':receipt['status'],
        'exit_code':receipt['exit_code'],'process_wall_seconds':receipt['process_wall_seconds'],
        'cleanup_verified':receipt.get('cleanup',{}).get('verified')}),flush=True)
    return 0 if receipt['status']=='completed' and receipt['exit_code']==0 and receipt.get('cleanup',{}).get('verified') else 1

def export(a):
    ROOT.mkdir(exist_ok=True,parents=True);target=ROOT/'exports'/a.name;target.mkdir(parents=True,exist_ok=False)
    archive=target/'source.tar'
    paths=[]
    if a.runtime_only:
        for candidate in ('pyproject.toml','setup.py','MANIFEST.in','README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','dependency-licenses.json','docs/LICENSE.md','src','anysolver','packages/results'):
            if subprocess.run(['git','-C',a.repo,'cat-file','-e',a.ref+':'+candidate],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:paths.append(candidate)
    subprocess.run(['git','-C',a.repo,'archive','--format=tar','--output',str(archive),a.ref,*paths],check=True)
    source=target/'source';source.mkdir()
    with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
    save(target/'identity.json',{'repository':a.repo,'commit':a.ref,'archive_sha256':sha(archive),'selected_paths':paths,'source_files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}})
    print(str(source));return 0

def fingerprint():
    import importlib.metadata as metadata
    import anyfem,anygeometry,anymesher
    from anyfem.model.project import GeometryModel
    repo=Path(__file__).resolve().parents[1]
    assert Path(anygeometry.__file__).resolve()==repo.parent/'ANYgeometry/src/anygeometry/__init__.py'
    assert Path(anymesher.__file__).resolve()==repo/'src/anymesher/__init__.py'
    assert Path(anyfem.__file__).resolve().is_relative_to(ROOT/'consumer-runtime')
    assert GeometryModel is anygeometry.GeometryModel
    exported=ROOT/'exports/ANYfem-ed84589/source/src/anyfem'
    installed=Path(anyfem.__file__).parent
    matches={str(p.relative_to(exported)):sha(p) for p in exported.rglob('*.py')}
    assert all(sha(installed/name)==value for name,value in matches.items())
    distributions={d.metadata['Name']:{'version':d.version,'direct_url':d.read_text('direct_url.json'),'record_sha256':hashlib.sha256((d.read_text('RECORD') or '').encode()).hexdigest()} for d in metadata.distributions()}
    save(ROOT/'consumer-runtime-identity.json',{'python':sys.version,'executable':sys.executable,'origins':{m.__name__:str(Path(m.__file__).resolve()) for m in (anyfem,anygeometry,anymesher)},'anyfem_commit':'ed845899e7c358b46376bc3ad873327b0267e174','installed_anyfem_python_sha256':matches,'distributions':distributions,'candidate_geometry_identity_alias':True})
    (ROOT/'consumer-runtime.lock.txt').write_text(''.join(name+'=='+record['version']+'\n' for name,record in sorted(distributions.items())))
    print(json.dumps({'status':'PINNED_REAL_CONSUMER','installed_files':len(matches),'distributions':len(distributions)}))
    return 0

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('run');r.add_argument('--name',required=True);r.add_argument('--timeout',type=float,required=True);r.add_argument('--pythonpath',action='append',default=[]);r.add_argument('command',nargs=argparse.REMAINDER)
    e=sub.add_parser('export');e.add_argument('--name',required=True);e.add_argument('--repo',required=True);e.add_argument('--ref',required=True);e.add_argument('--runtime-only',action='store_true')
    sub.add_parser('fingerprint')
    a=p.parse_args();return run(a) if a.action=='run' else export(a) if a.action=='export' else fingerprint()
if __name__=='__main__':raise SystemExit(main())
