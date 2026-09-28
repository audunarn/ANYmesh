"""Bounded serial SG1 supervisor; exclusively creates immutable attempt records."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid
from .contract import BASELINES,LIMITS,SCHEMA,inventory

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'reports/quad_first/structural-gate/attempts'


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def command(args,cwd=ROOT):
    if args[0]=='git':
        args=['git','-c','safe.directory='+Path(cwd).resolve().as_posix(),*args[1:]]
    return subprocess.run(args,cwd=cwd,text=True,capture_output=True,check=True).stdout.strip()


def identities():
    result=dict(python=sys.executable,python_version=sys.version,platform=platform.platform(),
                processor=platform.processor(),cpu_count=os.cpu_count(),modules={},sources={})
    for name in ('anygeometry','anymesher','anysolver','anymaterial','anyfileio','numpy','scipy','shapely'):
        module=importlib.import_module(name); path=Path(module.__file__).resolve()
        try: version=importlib.metadata.version('ANYmesher' if name=='anymesher' else name)
        except importlib.metadata.PackageNotFoundError: version=None
        result['modules'][name]=dict(path=str(path),version=version)
        repo=next((p for p in path.parents if (p/'.git').exists()),None)
        if repo is not None:
            files={}
            paths=command(['git','ls-files','src','pyproject.toml','setup.py','benchmarks/sg1','docs/SG1_ACCEPTANCE.md','tests/sg1'],repo).splitlines()
            # Imported production modules may be untracked on a candidate. Hash
            # every source Python file in every imported repository, not only
            # paths already present in the Git index.
            paths += [str(p.relative_to(repo)).replace('\\','/') for p in repo.glob('src/**/*.py')]
            # Include untracked gate-local implementation and tests as well.
            if repo==ROOT:
                paths+= [str(p.relative_to(repo)).replace('\\','/') for pattern in ('benchmarks/sg1/*.py','tests/sg1/*.py','docs/SG1_ACCEPTANCE.md') for p in repo.glob(pattern)]
                paths+= [str(p.relative_to(repo)).replace('\\','/') for p in repo.glob('third_party/quad/worker/*.cc')]
            for rel in sorted(set(paths)):
                p=repo/rel
                if p.is_file(): files[rel]=digest(p)
            result['sources'][repo.name]=dict(root=str(repo),head=command(['git','rev-parse','HEAD'],repo),
                status=command(['git','status','--porcelain'],repo),files=files)
    return result


def pin_workers(env):
    from anymesher.quad.quad_mcf_worker import find_worker
    from anymesher.quad.optimize import default_q5_worker_exe
    from anymesher.native_cpp import COMPILED_TRIANGULATION_AVAILABLE
    workers={}
    for label,lookup,disable,variable in (
        ('q4',find_worker,'ANYMESH_Q4_DISABLE_WORKER','ANYMESH_QUAD_MCF_WORKER'),
        ('q5',default_q5_worker_exe,'ANYMESH_Q5_DISABLE_WORKER','ANYMESH_QUAD_TINYAD_WORKER')):
        try: path=Path(lookup()).resolve()
        except (FileNotFoundError,RuntimeError) as exc: path=None
        available=path is not None and path.is_file()
        enabled=available and env.get(disable)!='1'
        if not enabled: env[disable]='1'
        elif label=='q4': env[variable]=str(path)
        workers[label]=dict(available=available,enabled=enabled,path=str(path) if available else None,
                           sha256=digest(path) if available else None)
    return dict(workers=workers,compiled_triangulation_available=bool(COMPILED_TRIANGULATION_AVAILABLE),selected_triangulation='python')


def stop_owned(process):
    if process.poll() is not None: return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
    else: process.kill()
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)


def combine_result(result,observation):
    if observation['status']=='incomplete':
        return {**result,**observation}
    if observation.get('exit_code')!=0 and result.get('status')=='passed':
        return {**result,'status':'failed','reason':'child exit disagrees with published result'}
    return result


def workers_stable(before):
    return all(not item['available'] or (Path(item['path']).is_file() and digest(item['path'])==item['sha256']) for item in before['workers'].values())


def supervise(args,folder,env,deadline,limit=600):
    remaining=min(limit,deadline-time.monotonic())
    if remaining<=0: return dict(status='incomplete',reason='attempt time budget exhausted')
    with (folder/'stdout.log').open('x',encoding='utf-8') as out,(folder/'stderr.log').open('x',encoding='utf-8') as err:
        start=time.monotonic()
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=out,stderr=err)
        try:
            code=process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            stop_owned(process)
            return dict(status='incomplete',reason='operation timeout',elapsed=time.monotonic()-start,pid=process.pid)
        except BaseException:
            stop_owned(process); raise
    return dict(status='passed' if code==0 else 'failed',exit_code=code,elapsed=time.monotonic()-start,pid=process.pid)


def compare_results(a,b):
    import numpy as np
    values={'energy':abs(a['energy']-b['energy'])/max(abs(b['energy']),1e-30)}
    for key,ref in b['probes'].items():
        x=np.asarray(a['probes'][key]); y=np.asarray(ref)
        # Zero reference probes must agree to absolute numerical displacement tolerance.
        values['probe/'+key]=float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-12))
    return values


def cross_checks(rows,operations):
    checks=[]
    def add(name,ok,**kw): checks.append(dict(name=name,status='passed' if ok else 'failed',**kw))
    def metric(row):
        op=operations.get(row['id']+'/mesh',{})
        file=Path(op.get('folder',''))/'mesh-metrics.json'
        return json.loads(file.read_text()) if file.is_file() else None
    grouped=defaultdict(list)
    for row in rows: grouped[row['case'],row['member'],row['graded']].append(row)
    for key,group in grouped.items():
        group=sorted(group,key=lambda x:x['level']); data=[metric(r) for r in group]
        name='resolution/'+str(key)
        if len(group)!=3 or any(x is None for x in data):
            checks.append(dict(name=name,status='incomplete',reason='missing resolution evidence')); continue
        counts=[d['counts']['equivalent'] for d in data]
        add(name+'/counts',counts[0]<counts[1]<counts[2],values=counts)
        errors=[d['max_geometry_error'] for d in data]
        add(name+'/geometry',errors[-1]<errors[0] or max(errors[0],errors[-1])<=data[-1]['support_tolerance'],values=errors)
    for row in rows:
        if not row['graded'] or row['case'].startswith('T-'): continue
        other=next((r for r in rows if r['case']==row['case'] and r['member']==row['member'] and r['level']==row['level'] and not r['graded']),None)
        a,b=metric(row),metric(other) if other else None
        name='refinement/'+row['id']
        if a is None or b is None or any(d[k] is None for d in (a,b) for k in ('core_median','remote_median')):
            checks.append(dict(name=name,status='incomplete',reason='missing core or remote sample')); continue
        add(name+'/core-count',a['core_corner_count']>b['core_corner_count'],graded=a['core_corner_count'],uniform=b['core_corner_count'])
        add(name+'/core-length',a['core_median']<b['core_median'])
        add(name+'/remote',abs(a['remote_median']/b['remote_median']-1)<=.2,value=a['remote_median']/b['remote_median'])
    for key,group in grouped.items():
        case,member,graded=key
        group=sorted(group,key=lambda x:x['level'])
        for load in (('pressure','member') if member else ('pressure',)):
            prefix=f'reference/{case}/b{int(member)}'
            names=[f'{prefix}/r{i}/{load}' for i in (0,1)]+[r['id']+'/'+load for r in (group[0],group[-1])]
            results=[operations.get(n,{}) for n in names]
            name=f'convergence/{key}/{load}'
            if any(x.get('status')!='passed' or 'energy' not in x for x in results):
                checks.append(dict(name=name,status='blocked',reason='missing or unsuccessful candidate/reference solve')); continue
            ref0,ref1,coarse,fine=results
            convergence=compare_results(ref0,ref1)
            add(name+'/reference',max(convergence.values())<=.01,errors=convergence)
            if max(convergence.values())>.01:
                checks.append(dict(name=name+'/candidate',status='blocked',reason='reference nonconvergence')); continue
            ce,fe=compare_results(coarse,ref1),compare_results(fine,ref1)
            add(name+'/fine',max(fe.values())<=.05,errors=fe)
            add(name+'/improves',all(fe[k]<ce[k] or max(fe[k],ce[k])<=.01 for k in fe),coarse=ce,fine=fe)
    return checks


def run(mode,selected=None):
    env=os.environ.copy()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'): env[name]='1'
    env['PYTHONHASHSEED']='0'; env['PYTHONUNBUFFERED']='1'
    os.environ.update({k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS')})
    root=OUTPUT/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+mode+'-'+uuid.uuid4().hex[:8]); root.mkdir(parents=True,exist_ok=False)
    print('ATTEMPT '+str(root),flush=True)
    started=time.monotonic(); deadline=started+LIMITS['attempt_seconds']
    try:
        sources=identities(); workers=pin_workers(env)
    except Exception as exc:
        import traceback
        (root/'startup-error.log').write_text(traceback.format_exc(),encoding='utf-8')
        write(root/'summary.json',dict(status='incomplete',mode=mode,reason='provenance preflight failed',exception=str(exc)))
        write(root/'manifest.json',{p.name:digest(p) for p in root.iterdir() if p.is_file()})
        print('INCOMPLETE: provenance preflight failed',flush=True)
        return root
    rows=inventory()
    if mode=='rehearsal': rows=[r for r in rows if r['level']==0]
    if selected: rows=[r for r in rows if r['case'] in selected]
    write(root/'contract.json',dict(schema=SCHEMA,limits=LIMITS,baselines=BASELINES,inventory=inventory(),selected=rows,mode=mode))
    write(root/'environment.json',dict(**sources,**workers,environment={k:v for k,v in env.items() if k.startswith(('ANYMESH_','OMP_','OPENBLAS_','MKL_','NUMBA_','NUMEXPR_','PYTHONHASH'))}))
    operations={}; sequence=0
    def operation(name,spec):
        nonlocal sequence
        sequence+=1; folder=root/f'{sequence:04d}-{name.replace("/","_")}'
        folder.mkdir(); write(folder/'request.json',spec)
        observation=supervise([sys.executable,'-m','benchmarks.sg1.operation',str(folder)],folder,env,deadline)
        file=folder/'result.json'
        result=json.loads(file.read_text()) if file.exists() else dict(status='incomplete',reason='child did not publish result')
        result=combine_result(result,observation)
        result={**result,'folder':str(folder),'supervisor':observation}
        operations[name]=result; write(folder/'supervisor.json',observation)
        print(name+': '+result['status'],flush=True)
        return result
    try:
        # Baseline mismatches are explicit prerequisites, not silently accepted candidates.
        for name,sha in BASELINES.items():
            if not sources['sources'][name]['head'].startswith(sha): raise RuntimeError(f'{name} baseline differs from frozen contract')
        operation('analytical',dict(operation='analytical'))
        operation('negative',dict(operation='negative'))
        for row in rows:
            result=operation(row['id']+'/mesh',dict(**row,operation='mesh'))
            mesh=Path(result['folder'])/'mesh.json'
            if mesh.exists() and (mesh.parent/'mesh-metrics.json').exists():
                # A quality-threshold failure is still informative for consumption;
                # invalid certificates or topology must not proceed into a solve.
                metrics=json.loads((mesh.parent/'mesh-metrics.json').read_text())
                invalid=[c for c in metrics['checks'] if c['status']=='failed' and ('certif' in c['name'] or 'incidence' in c['name'] or 'coverage' in c['name'] or 'midside' in c['name'] or '/segment/' in c['name'])]
                for load in (('pressure','member') if row['member'] else ('pressure',)):
                    name=row['id']+'/'+load
                    if invalid: operations[name]=dict(status='blocked',reason='invalid mesh topology or certificate')
                    else: operation(name,dict(**row,operation='solve',mesh=str(mesh),load=load))
            else:
                for load in (('pressure','member') if row['member'] else ('pressure',)):
                    operations[row['id']+'/'+load]=dict(status='blocked',reason='mesh was not published')
        if mode=='formal':
            for case,member in sorted({(r['case'],r['member']) for r in rows}):
                hf=.125 if case in ('P','F') else .15
                for level,h in enumerate((hf/2,hf/4)):
                    name=f'reference/{case}/b{int(member)}/r{level}'
                    result=operation(name,dict(operation='reference',case=case,member=member,h=h))
                    mesh=Path(result['folder'])/'mesh.json'
                    for load in (('pressure','member') if member else ('pressure',)):
                        if result['status']=='passed': operation(name+'/'+load,dict(operation='solve',case=case,member=member,mesh=str(mesh),load=load,measured=False))
                        else: operations[name+'/'+load]=dict(status='blocked',reason='reference generation failed')
        cross=cross_checks(rows,operations) if mode=='formal' else []
        folder=root/'regressions'; folder.mkdir()
        suites=[('sg1','tests/sg1')]
        if mode=='formal':
            suites.extend((('quad_first','tests/quad_first'),
                           ('quad_first_planar','tests/quad_first_planar'),
                           ('quad_first_curved','tests/quad_first_curved')))
        suite_results={}
        for name,path in suites:
            suite_folder=folder/name; suite_folder.mkdir()
            suite_results[name]={**supervise([sys.executable,'-m','pytest',path,'-q'],suite_folder,env,deadline,
                                             limit=LIMITS['operation_seconds']),
                                 'folder':str(suite_folder)}
            print('regressions/'+name+': '+suite_results[name]['status'],flush=True)
        suite_statuses={result['status'] for result in suite_results.values()}
        regression_status='failed' if 'failed' in suite_statuses else 'incomplete' if 'incomplete' in suite_statuses else 'passed'
        operations['regressions']=dict(status=regression_status,suites=suite_results,folder=str(folder))
    except BaseException as exc:
        import traceback
        (root/'supervisor-error.log').write_text(traceback.format_exc(),encoding='utf-8')
        operations['supervisor']=dict(status='incomplete',exception=type(exc).__name__,reason=str(exc)); cross=[]
    final=identities()
    stable=all(sources['sources'][k]['files']==final['sources'][k]['files'] and sources['sources'][k]['head']==final['sources'][k]['head'] for k in sources['sources'])
    operations['source-stability']=dict(status='passed' if stable else 'incomplete')
    operations['worker-stability']=dict(status='passed' if workers_stable(workers) else 'incomplete')
    write(root/'final-environment.json',dict(**final,workers={k:{**v,'final_sha256':digest(v['path']) if v['available'] and Path(v['path']).is_file() else None} for k,v in workers['workers'].items()}))
    checks=[c for r in operations.values() for c in r.get('checks',[])]+cross
    counts=Counter(c['status'] for c in checks)
    statuses=[r['status'] for r in operations.values()]+[c['status'] for c in cross]
    status='failed' if 'failed' in statuses else 'blocked' if 'blocked' in statuses else 'incomplete'
    summary=dict(schema=SCHEMA,mode=mode,status=status,elapsed_seconds=time.monotonic()-started,
        operation_counts=dict(Counter(r['status'] for r in operations.values())),check_counts=dict(counts),
        mandatory_inventory_complete=len(rows)==len(inventory()),independent_review='pending',operations=operations,cross_checks=cross)
    timing={}
    for name,op in operations.items():
        records=op.get('timings',[])
        if records:
            timing[name]={key:dict(median=statistics.median(values),minimum=min(values),maximum=max(values),runs=len(values))
                for key in records[0] if (values:=[r[key] for r in records if r.get(key) is not None])}
    scaling=[]
    for row in rows:
        if row['level']==0: continue
        previous=next((r for r in rows if r['case']==row['case'] and r['graded']==row['graded'] and r['member']==row['member'] and r['level']==row['level']-1),None)
        if previous is None: continue
        for stage in ('mesh','pressure','member'):
            a,b=timing.get(previous['id']+'/'+stage),timing.get(row['id']+'/'+stage)
            if a and b:
                scaling.append(dict(previous=previous['id'],current=row['id'],stage=stage,
                    ratios={k:b[k]['median']/a[k]['median'] for k in a.keys()&b.keys() if a[k]['median']>0}))
    write(root/'scaling.json',scaling)
    write(root/'timing-summary.json',timing); write(root/'summary.json',summary)
    failures=[f"- {name}: {r['status']} {r.get('message',r.get('reason',''))}" for name,r in operations.items() if r['status']!='passed']
    failures += [f"- {c['name']}: {c['status']}" for c in cross if c['status']!='passed']
    (root/'REPORT.md').write_text(f'# SG1 {mode}: {status.upper()}\n\nCheck counts: {dict(counts)}. Independent review: pending.\n\n'+ '\n'.join(failures)+'\n\nFull predicates are in summary.json; stage medians/ranges in timing-summary.json.\n',encoding='utf-8')
    write(root/'manifest.json',{str(p.relative_to(root)).replace('\\','/'):digest(p) for p in sorted(root.rglob('*')) if p.is_file()})
    print(json.dumps(dict(attempt=str(root),status=status,checks=dict(counts))),flush=True)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=('rehearsal','formal'),required=True)
    parser.add_argument('--cases',nargs='+',help='Subset is always incomplete evidence')
    args=parser.parse_args(); path=run(args.mode,args.cases)
    status=json.loads((path/'summary.json').read_text())['status']
    sys.exit(0 if status=='passed' else 1 if status=='failed' else 2)
