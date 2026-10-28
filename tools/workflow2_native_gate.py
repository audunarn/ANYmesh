"""Finite Linux3.12 installed-native gate with durable phase journals."""
from __future__ import annotations
import argparse,hashlib,importlib,importlib.util,json,os,shutil,subprocess,sys,time
from pathlib import Path
WORKSPACE=Path(__file__).resolve().parents[2]
ROOT=WORKSPACE/'evidence/native-core-gate-20261028-a1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(p)


def environment(folder):
    env=dict(os.environ)
    for key in ('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS','ANYGEOMETRY_TEST_SHARD','ANYGEOMETRY_SHARD_REPORT'):
        env.pop(key,None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONUTF8='1',
               ANYGEOMETRY_BUILD_NATIVE='1',ANYGEOMETRY_REQUIRE_NATIVE='1',ANYGEOMETRY_DISABLE_NATIVE='0',
               ANYMESHER_REQUIRE_NATIVE='1',ANYMESHER_DISABLE_NATIVE='0',ANYMESHER_REQUIRE_QUAD_WORKERS='1',
               ANYMESH_QUAD_MCF_WORKER=str(WORKSPACE/'ANYmesh/third_party/quad/worker/out/lemon/quad_mcf_worker'),
               ANYMESH_QUAD_TINYAD_WORKER=str(WORKSPACE/'ANYmesh/third_party/quad/worker/out/tinyad/quad_tinyad_optimizer'),
               PIP_CACHE_DIR=str(ROOT/'cache/pip'),UV_CACHE_DIR=str(ROOT/'cache/uv'),UV_PYTHON_DOWNLOADS='never',
               TMPDIR='/tmp/anymesh-native-core-gate-20261028-a1',W2_NATIVE_JOURNAL=str(folder/'journal.jsonl'))
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    return env


class Journal:
    """Flush every node phase/outcome; summary is reconstructible after kill."""
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=self.path.open('x');self.started=time.monotonic()
    def emit(self,event,**data):
        self.handle.write(json.dumps({'event':event,'elapsed':time.monotonic()-self.started,**data},allow_nan=False)+'\n');self.handle.flush()
    def pytest_sessionstart(self,session):self.emit('session_start')
    def pytest_collection_finish(self,session):self.emit('collection',nodeids=[i.nodeid for i in session.items])
    def pytest_collectreport(self,report):
        if report.failed:self.emit('collection_failure',nodeid=report.nodeid,longrepr=str(report.longrepr))
    def pytest_deselected(self,items):self.emit('deselected',nodeids=[i.nodeid for i in items])
    def pytest_runtest_logstart(self,nodeid,location):self.emit('node_start',nodeid=nodeid,phase='setup')
    def pytest_runtest_setup(self,item):self.emit('phase_start',nodeid=item.nodeid,phase='setup')
    def pytest_runtest_call(self,item):self.emit('phase_start',nodeid=item.nodeid,phase='call')
    def pytest_runtest_teardown(self,item,nextitem):self.emit('phase_start',nodeid=item.nodeid,phase='teardown')
    def pytest_runtest_logreport(self,report):
        self.emit('report',nodeid=report.nodeid,phase=report.when,outcome=report.outcome,duration=report.duration,longrepr=str(report.longrepr) if report.failed or report.skipped else None)
    def pytest_runtest_logfinish(self,nodeid,location):self.emit('node_finish',nodeid=nodeid)
    def pytest_sessionfinish(self,session,exitstatus):self.emit('session_finish',exitstatus=int(exitstatus));self.handle.close()


def journal_summary(path):
    result={'reports':[],'active_node':None,'active_phase':None,'finished_nodes':0,'session_finished':False,'selected':[],'deselected':[]}
    if not Path(path).exists():return result
    for line in Path(path).read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:result['partial_line']=True;continue
        event=r['event']
        if event=='collection':result['selected']=r['nodeids']
        if event=='deselected':result['deselected'].extend(r['nodeids'])
        if event in ('node_start','phase_start'):result.update(active_node=r['nodeid'],active_phase=r['phase'])
        if event=='report':result['reports'].append(r)
        if event=='collection_failure':result.setdefault('collection_failures',[]).append(r)
        if event=='node_finish':result.update(active_node=None,active_phase=None);result['finished_nodes']+=1
        if event=='session_finish':result.update(session_finished=True,exitstatus=r['exitstatus'])
    result['counts']={name:sum(r['outcome']==name and (r['phase']=='call' or name!='passed') for r in result['reports']) for name in ('passed','failed','skipped')}
    return result


def run(a):
    ROOT.mkdir(exist_ok=True,parents=True);ledger_path=ROOT/'budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limit':1800.,'used':0.,'attempts':[]}
    remaining=1800-ledger['used'];timeout=min(a.timeout,remaining-a.reserve)
    if not 0<timeout<=900:raise ValueError('process budget/reserve unavailable')
    folder=ROOT/a.name;folder.mkdir(exist_ok=False)
    env=environment(folder);command=a.command[1:] if a.command[:1]==['--'] else a.command
    if command[0]=='python':command[0]=sys.executable
    cwd=Path(a.cwd).resolve() if a.cwd else WORKSPACE/'ANYmesh'
    save(folder/'identity.json',{'command':command,'cwd':str(cwd),'timeout':timeout,'budget_before':ledger['used'],
         'runner_sha256':sha(__file__),'supervisor_sha256':sha(WORKSPACE/'ANYmesh/tools/workflow2_process_supervisor.py'),
         'environment':{k:v for k,v in env.items() if k.startswith(('ANY','OMP','OPENBLAS','MKL','PYTHON','UV_','PIP_','TMPDIR'))},'started_utc':time.time()})
    spec=importlib.util.spec_from_file_location('native_gate_supervisor',WORKSPACE/'ANYmesh/tools/workflow2_process_supervisor.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    receipt=helper.run_linux(command,folder/'process',timeout,env,cwd)
    ledger['used']+=receipt['process_wall_seconds'];ledger['remaining']=1800-ledger['used'];ledger['attempts'].append({'name':a.name,**receipt});save(ledger_path,ledger)
    if (folder/'journal.jsonl').exists():save(folder/'live-summary.json',journal_summary(folder/'journal.jsonl'))
    print(json.dumps({'evidence':str(folder),'remaining':ledger['remaining'],
        **{k:receipt[k] for k in ('status','exit_code','process_wall_seconds')},
        'cleanup_verified':receipt['cleanup']['verified']}),flush=True)
    return 0 if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt['cleanup']['verified'] else 1


def freeze():
    copied={};sources=ROOT/'sources';sources.mkdir(exist_ok=False)
    for repo_name in ('ANYgeometry','ANYmesh'):
        source=WORKSPACE/repo_name;target=sources/repo_name;target.mkdir()
        for name in ('pyproject.toml','setup.py','MANIFEST.in','README.md','LICENSE','NOTICE','THIRD_PARTY_NOTICES.md','dependency-licenses.json'):
            if (source/name).exists():shutil.copy2(source/name,target/name)
        for name in ('src','native'):
            if (source/name).exists():shutil.copytree(source/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.so','*.pyd','*.dll','*.egg-info'))
        (target/'docs').mkdir();shutil.copy2(source/'docs/LICENSE.md',target/'docs/LICENSE.md')
        copied.update({str(p.relative_to(sources)):sha(p) for p in target.rglob('*') if p.is_file()})
    consumer=WORKSPACE/'evidence/source-gate-repair-20261028-a1/consumer-runtime'
    shutil.copytree(consumer,ROOT/'runtime',symlinks=True)
    tests={str(p.relative_to(WORKSPACE/'ANYmesh')):sha(p) for p in (WORKSPACE/'ANYmesh/tests').rglob('*.py')}
    compiler=shutil.which('c++')
    dependencies=json.loads((WORKSPACE/'evidence/source-gate-repair-20261028-a1/consumer-runtime-identity.json').read_text())
    save(ROOT/'manifest.json',{'scope':'LinuxPython3.12installednativeonly','source_sha256':copied,'test_sha256':tests,
         'compiler':{'path':compiler,'sha256':sha(Path(compiler).resolve()),'version':subprocess.check_output([compiler,'--version'],text=True)},
         'dependency_closure':dependencies,'runner_sha256':sha(__file__),'supervisor_sha256':sha(WORKSPACE/'ANYmesh/tools/workflow2_process_supervisor.py'),
         'quad_workers':{str(p):sha(p) for p in (WORKSPACE/'ANYmesh/third_party/quad/worker/out').glob('*/*') if p.is_file()},
         'limits':{'total':1800,'per_process':900,'reserve':180},'full_command':['-I',str(Path(__file__).resolve()),'pytest-child','--','-q','-o','pythonpath=','--confcutdir=tests','--import-mode=importlib','tests','--durations=25']})
    print(json.dumps({'status':'FROZEN','package_files':len(copied),'test_files':len(tests)}))


def probe():
    import importlib.metadata as metadata
    modules={name:importlib.import_module(name) for name in ('anygeometry','anygeometry._core','anymesher','anymesher._native','anyfem')}
    for module in modules.values():assert Path(module.__file__).resolve().is_relative_to(ROOT/'runtime')
    import anygeometry
    assert anygeometry.backend_info()['backend']=='native'
    for name in ('anygeometry._core','anymesher._native'):assert str(modules[name].__file__).endswith('.so')
    save(ROOT/'installed-origins.json',{'origins':{k:str(v.__file__) for k,v in modules.items()},'native_sha256':{k:sha(v.__file__) for k,v in modules.items() if k.endswith(('_core','_native'))},
         'versions':{d.metadata['Name']:d.version for d in metadata.distributions()},'geometry_backend':anygeometry.backend_info(),'python':sys.version,'executable':sys.executable})
    print('required installed native modules verified')


def pytest_child(args):
    import pytest
    for name in ('pytest_runtest_setup','pytest_runtest_call','pytest_runtest_teardown'):
        pytest.hookimpl(tryfirst=True)(getattr(Journal,name))
    journal=Journal(os.environ['W2_NATIVE_JOURNAL'])
    return pytest.main(args,plugins=[journal])


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('run');r.add_argument('--name',required=True);r.add_argument('--timeout',type=float,required=True);r.add_argument('--reserve',type=float,default=180);r.add_argument('--cwd');r.add_argument('command',nargs=argparse.REMAINDER)
    sub.add_parser('freeze');sub.add_parser('probe');c=sub.add_parser('pytest-child');c.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.action=='run':return run(a)
    if a.action=='freeze':freeze();return 0
    if a.action=='probe':probe();return 0
    return pytest_child(a.args[1:] if a.args[:1]==['--'] else a.args)
if __name__=='__main__':raise SystemExit(main())
