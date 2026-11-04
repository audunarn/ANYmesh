"""One-shot six-fixture source comparison; diagnostic, never a release verdict.

The parent imports no ANY package. Each source profile gets a fresh interpreter.
Only the declared Mac wheel extension/libraries augment exact Git source copies.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import time
import tomllib
import traceback
import zipfile

CASES=('planar','cylinder','cone','ruled','coons','native_nurbs')
FIXTURES=('test_qf_nurbs_s0_characterisation.py','test_curved_native_qualification.py','test_cylindrical_atlas_binding.py')
REFS={'old_g':'f645523c8006efe61e94678a5056fba3c9a6a39d',
      'current_g':'2d3b0f876976b4d2e970c531c4b71941f2bb78a6',
      'old_m':'aa19c0fa6b796a0d7cb1793a87840f52899dfa24',
      'current_m':'d4a8329997d8f700edca811f39b5a358950a6ba6'}
BUILD_FILES=('setup.py','MANIFEST.in','src/anymesher/_native.cpp',
    'src/anymesher/_triangulation_native.hpp','src/anymesher/_quality_pipeline_native.hpp',
    'src/anymesher/_native_v2.hpp','src/anymesher/_native_t3_storage.hpp',
    'src/anymesher/native.py','src/anymesher/native_cpp.py')
ENV={'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING':'0',
     'ANYMESH_Q4_DISABLE_WORKER':'1','ANYMESH_Q5_DISABLE_WORKER':'1',
     'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
     'VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,row):
    with Path(path).open('w',encoding='utf-8') as stream:
        json.dump(row,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())


def inventory(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def build_proof(roots):
    """Exact compiler inputs/adapters; sole runtime-dependency metadata exception."""
    roots=list(map(Path,roots));rows={}
    for name in BUILD_FILES:
        hashes=[sha(root/name) for root in roots]
        if len(set(hashes))!=1:raise ValueError('native build/adapter source mismatch: '+name)
        rows[name]=hashes[0]
    configs=[tomllib.loads((root/'pyproject.toml').read_text()) for root in roots]
    dependencies=[c['project'].pop('dependencies') for c in configs]
    if any(c!=configs[0] for c in configs):raise ValueError('pyproject build/version/config mismatch')
    allowed=[['numpy>=1.26','ANYgeometry[planar]>=0.4.5,<0.5'],
             ['numpy>=1.26','ANYgeometry[planar]>=0.4.7,<0.5']]
    if dependencies[0]!=allowed[0] or any(d!=allowed[1] for d in dependencies[1:]):
        raise ValueError('unexpected dependency-metadata difference')
    tree=ast.parse((roots[0]/'setup.py').read_text())
    imports=sorted({n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)} |
                   {a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names})
    if any(name and name.split('.')[0] not in ('__future__','os','shutil','sys','pathlib','setuptools') for name in imports):
        raise ValueError('unexpected setup import/build-derived owner constants')
    return dict(exact_inputs=rows,pyproject_sha256=[sha(root/'pyproject.toml') for root in roots],
        parsed_config_except_runtime_dependencies_equal=True,setup_imports=imports,
        declared_runtime_dependency_exception=dependencies,
        scope='same C++ inputs/setup/adapters, not a byte-identical whole pyproject or reproducible-build claim')


def native_payload(wheel,source=None):
    """No wheel Python bootstrap may silently replace a Git source module."""
    payload={}
    with zipfile.ZipFile(wheel) as archive:
        for item in archive.infolist():
            path=PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('unsafe wheel entry')
            if item.is_dir():continue
            name=path.as_posix()
            if source is not None and name.startswith('anymesher/') and path.suffix=='.py':
                original=Path(source)/'src'/name
                if not original.is_file() or original.read_bytes()!=archive.read(item):
                    raise ValueError('wheel Python bootstrap/source differs: '+name)
            if (path.parent==PurePosixPath('anymesher') and path.name.startswith('_native.') and path.suffix=='.so') or (
                    (name.startswith('anymesher/.dylibs/') or name.startswith('anymesher.libs/')) and path.suffix in ('.dylib','.so')):
                payload[name]=archive.read(item)
    if sum(PurePosixPath(name).name.startswith('_native.') for name in payload)!=1:
        raise ValueError('expected one Mac _native extension')
    return payload


def verify_git(root,ref):
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,timeout=10).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True,timeout=10)
    if actual!=ref or dirty:raise ValueError('checkout identity/cleanliness mismatch: '+str(root))
    return dict(commit=actual,source=inventory(root/'src'))


def kill_owned(proc,identity,actions,path):
    created,expected=identity;actual=proc.cmdline()
    action=dict(pid=proc.pid,created=proc.create_time(),expected=expected,actual=actual)
    actions.append(action);write(path,actions)
    if action['created']!=created or actual!=expected:raise RuntimeError('owned process identity/commandline changed; no kill')
    proc.kill()


def owned_group_state(pid,*,platform_name=None,probe=None):
    """Signal zero only: check this Popen's POSIX group without enumerating it."""
    if (os.name if platform_name is None else platform_name)=='nt':return dict(state='not_applicable')
    if probe is None:probe=os.killpg
    try:probe(pid,0)
    except ProcessLookupError:state='absent'
    except OSError:state='unknown'
    else:state='present'
    return dict(state=state,group=pid,signal=0)


def supervise(command,cwd,env,out,*,work_seconds=100.,total_seconds=110.):
    """Owned-PID identities and argv before kill; no unrelated process search.

    POSIX child session isolates this two-child diagnostic; psutil tracks only
    descendants. Cleanup is explicit sampled supervision, not crash-safe OS caps.
    """
    import psutil
    if not 0<work_seconds<total_seconds<=110:raise ValueError('invalid deadline')
    out=Path(out);out.mkdir();start=time.monotonic();identities={};actions=[];errors=[];reason='launch_error';child=None
    group_check=dict(state='not_launched')
    def discover():
        if child is None:return
        # Only retained root/descendant lineage. Never enumerate system PIDs,
        # even to filter by session or executable name.
        for pid,(created,argv) in list(identities.items()):
            try:
                proc=psutil.Process(pid)
                if proc.create_time()!=created:raise RuntimeError('owned PID reused')
                for descendant in proc.children(recursive=True):
                    identities.setdefault(descendant.pid,(descendant.create_time(),descendant.cmdline()))
            except psutil.NoSuchProcess:pass
    def live():
        result=[]
        for pid,(created,argv) in identities.items():
            try:
                proc=psutil.Process(pid)
                if proc.create_time()!=created:raise RuntimeError('owned PID reused')
                if proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:result.append(proc)
            except psutil.NoSuchProcess:pass
        return result
    with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
        try:
            child=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                start_new_session=os.name!='nt',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            proc=psutil.Process(child.pid);identities[child.pid]=(proc.create_time(),list(command))
            while True:
                discover()
                if child.poll() is not None:reason='completed';break
                if time.monotonic()-start>=work_seconds:reason='timeout';break
                time.sleep(.02)
        except BaseException as error:errors.append(type(error).__name__+': '+str(error))
        finally:
            try:
                discover()
                survivors=live()
                if reason=='completed' and survivors:reason='descendants_after_root_exit'
                for proc in reversed(survivors):
                    kill_owned(proc,identities[proc.pid],actions,out/'termination-command-checks.json')
                if child is not None:child.wait(timeout=max(.01,total_seconds-(time.monotonic()-start)))
                while live() and time.monotonic()-start<total_seconds:time.sleep(.02)
                remaining=[p.pid for p in live()]
                if child is not None:
                    group_check=owned_group_state(child.pid)
                    if group_check['state'] not in ('absent','not_applicable'):
                        remaining.append('owned POSIX group not proved absent; unrecorded members are not kill targets')
            except BaseException as error:
                errors.append(type(error).__name__+': '+str(error));remaining=['cleanup unverified']
    row=dict(command=list(command),status=reason,exit_code=None if child is None else child.returncode,
        process_wall_seconds=time.monotonic()-start,work_seconds=work_seconds,total_seconds=total_seconds,
        identities={str(pid):dict(created=data[0],command=data[1]) for pid,data in identities.items()},
        cleanup_verified=not remaining,remaining_owned=remaining,owned_group_check=group_check,
        cleanup_scope='recorded PID lineage plus POSIX owned-group absence only; no session-escape or crash-safe containment claim',
        termination_checks=actions,errors=errors)
    write(out/'result.json',row)
    return row


def child(config):
    row=json.loads(Path(config).read_text());out=Path(row['output']);out.mkdir()
    roots=[Path(row[k]) for k in ('geometry','mesher','fixtures')]
    before=[inventory(root) for root in roots];write(out/'source-before.json',before)
    if any(name.split('.')[0] in ('anygeometry','anymesher') for name in sys.modules):raise RuntimeError('ANY namespace already loaded')
    if any(os.environ.get(key)!=value for key,value in ENV.items()):raise RuntimeError('effective controls mismatch')
    sys.path[:0]=[str(p) for p in roots]
    complete=[];failure=None
    try:
        import anygeometry,anymesher,numpy as np,shapely
        extension=importlib.import_module('anymesher._native')
        fixture=importlib.import_module('test_qf_nurbs_s0_characterisation')
        for module,root in ((anygeometry,roots[0]),(anymesher,roots[1]),(fixture,roots[2]),(extension,roots[1])):
            if not Path(module.__file__).resolve().is_relative_to(root.resolve()):raise RuntimeError('unexpected import origin')
        from anymesher.native import _run_reference_native_seed
        observations=[];kernel=getattr(extension,'reference_delaunay',None)
        def observe(frame,event,arg):
            if event=='return' and frame.f_code is _run_reference_native_seed.__code__:
                observations.append(dict(event='reference_seed_result',accepted=arg is not None and arg[0] is not None,
                                         reason=None if arg is None else arg[1]))
            elif event in ('c_call','c_return','c_exception') and arg is kernel:
                observations.append(dict(event=event,function='anymesher._native.reference_delaunay'))
        write(out/'runtime.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),
            numpy=np.__version__,shapely=shapely.__version__,byteorder=sys.byteorder,effective_environment={key:os.environ.get(key) for key in ENV},
            geometry_origin=anygeometry.__file__,mesher_origin=anymesher.__file__,native_origin=extension.__file__,native_sha256=sha(extension.__file__),
            fixture_origin=fixture.__file__,meshing_execution='unchanged single-face in-process fixture, optional external Q4/Q5 disabled'))
        for name in CASES:
            write(out/'active.json',dict(case=name,completed=len(complete)))
            observations.clear();previous=sys.getprofile()
            try:sys.setprofile(observe);mesh=fixture.build(name)
            finally:sys.setprofile(previous)
            body=bytearray()
            for key in sorted(mesh.nodes):body.extend(np.int64(key).tobytes()+np.asarray(mesh.nodes[key],dtype=np.float64).tobytes())
            for kind in ('quads','tris'):
                body.extend(kind.encode())
                for key,values in sorted(getattr(mesh,kind).items()):body.extend(np.int64(key).tobytes()+np.asarray(values,dtype=np.int64).tobytes())
            (out/(name+'.bin')).write_bytes(body)
            digest=hashlib.sha256(body).hexdigest()
            if digest!=fixture.digest(mesh):raise RuntimeError('raw encoding differs from original digest')
            complete.append(dict(case=name,sha256=digest,bytes=len(body),nodes=len(mesh.nodes),quads=len(mesh.quads),tris=len(mesh.tris),
                historical_golden=fixture.GOLDEN[name],golden_equal=digest==fixture.GOLDEN[name],
                native_observations=list(observations)))
            write(out/'partial-cases.json',complete)
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    finally:
        after=[inventory(root) for root in roots];write(out/'source-after.json',after)
        unchanged=before==after
        write(out/'characterization.json',dict(scope='six S0 diagnostic fixtures only; no qualification or acceptance verdict',
            profile=row['profile'],cases=complete,source_unchanged=unchanged,failure=failure))
    return 0 if failure is None and unchanged and len(complete)==6 else 1


def compare(old,new,out):
    reports=[json.loads((Path(root)/'characterization.json').read_text()) for root in (old,new)]
    if any([row['case'] for row in report['cases']]!=list(CASES) for report in reports):raise ValueError('incomplete case inventory')
    rows=[]
    for index,name in enumerate(CASES):
        a=(Path(old)/(name+'.bin')).read_bytes();b=(Path(new)/(name+'.bin')).read_bytes()
        rows.append(dict(case=name,raw_bytes_equal=a==b,old_sha256=hashlib.sha256(a).hexdigest(),new_sha256=hashlib.sha256(b).hexdigest(),
            old_bytes=len(a),new_bytes=len(b),counts=[{key:report['cases'][index][key] for key in ('nodes','quads','tris')} for report in reports]))
    write(out,dict(scope='direct represented-coordinate/connectivity equality on one runtime; not a geometric accuracy or release verdict',cases=rows))
    return rows


def completed_results(results):
    return len(results)==2 and all(r['status']=='completed' and r['exit_code']==0 and
        r['cleanup_verified'] is True and r['remaining_owned']==[] and r['errors']==[] for r in results)


def run(args):
    if sys.platform!='darwin' or sys.version_info[:2]!=(3,14):raise RuntimeError('manual diagnostic requires Mac Python3.14')
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    out=Path(args.output).resolve();out.mkdir()
    roots={key:Path(getattr(args,key)).resolve() for key in REFS}
    identities={key:verify_git(root,REFS[key]) for key,root in roots.items()}
    wheel_run=json.loads(Path(args.wheel_run).read_text());wheel_source=Path(args.wheel_source).resolve()
    if wheel_run['id']!=41751327690 or wheel_run['repository']['full_name']!='audunarn/ANYmesh':raise ValueError('wrong artifact source run')
    identities['wheel_source']=verify_git(wheel_source,wheel_run['head_sha'])
    proof=build_proof([roots['old_m'],roots['current_m'],wheel_source]);write(out/'build-input-proof.json',proof)
    candidates=[p for p in Path(args.wheels).rglob('*.whl') if parse_wheel_filename(p.name)[3]&set(sys_tags())]
    if len(candidates)!=1:raise ValueError('expected exactly one runtime-compatible Mac artifact wheel')
    wheel=candidates[0];payload=native_payload(wheel,wheel_source)
    fixtures=out/'fixtures';fixtures.mkdir()
    for name in FIXTURES:shutil.copyfile(roots['current_m']/'tests'/name,fixtures/name)
    write(out/'identities.json',dict(checkouts=identities,wheel_run=wheel_run,wheel=str(wheel),wheel_sha256=sha(wheel),
        declared_non_git_payload={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()},fixtures=inventory(fixtures)))
    results=[]
    for profile,gkey,mkey in (('baseline','old_g','old_m'),('current','current_g','current_m')):
        base=out/profile;base.mkdir();g=base/'geometry';m=base/'mesher'
        shutil.copytree(roots[gkey]/'src',g);shutil.copytree(roots[mkey]/'src',m)
        for name,data in payload.items():
            path=m/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        config=base/'config.json';write(config,dict(profile=profile,geometry=str(g),mesher=str(m),fixtures=str(fixtures),output=str(base/'raw')))
        env=dict(os.environ);env.update(ENV);env.pop('PYTHONPATH',None);env.pop('ANYMESHER_DISABLE_NATIVE',None)
        result=supervise([sys.executable,'-B',str(Path(__file__).resolve()),'--child',str(config)],base,env,base/'process')
        results.append(result)
        if result['errors'] or not result['cleanup_verified'] or result['status']!='completed':break
    # A child failure is not a comparison result. Preserve all partial output.
    complete=completed_results(results)
    if complete:compare(out/'baseline/raw',out/'current/raw',out/'comparison.json')
    after={key:verify_git(root,REFS[key]) for key,root in roots.items()}
    write(out/'run-result.json',dict(scope='manual diagnostic only',processes=results,complete=complete,
        checkout_sources_unchanged=identities==dict(after,wheel_source=verify_git(wheel_source,wheel_run['head_sha']))))
    return 0 if complete else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--child')
    for key in REFS:parser.add_argument('--'+key.replace('_','-'))
    for key in ('wheel-source','wheel-run','wheels','output'):parser.add_argument('--'+key)
    args=parser.parse_args()
    return child(args.child) if args.child else run(args)


if __name__=='__main__':raise SystemExit(main())
