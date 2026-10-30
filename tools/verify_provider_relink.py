"""Recipient link/import proof only; new derivatives are never qualified wheels.

All work runs below a new evidence root. Historical payloads are read-only.
Use `run ACTION --abi ABI` for bounded, process-tree-supervised execution.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
WORK = REPO.parent
OLD = WORK/'evidence/exact-envelope-20261028-a1'
DELIVERY = WORK/'evidence/provider-notices-20261028-a1'
PRIOR = WORK/'evidence/provider-relink-20261028-a1'
OUT = WORK/'evidence/provider-relink-remaining-20261028-a1'
ABIS = ('windows312abi3', 'linux311')
PROFILE = 'remaining'
MATRIX = WORK/'evidence/installed-matrix-20261028-a1'

def select_profile(profile):
    if profile != 'remaining':
        raise ValueError('prior campaign is sealed; use its preserved executed runner for inspection')

def native_output(abi):
    if abi.startswith('windows'): return 'PyfTetWildWrapper.pyd'
    if abi == 'linux311': return 'PyfTetWildWrapper.cpython-311-x86_64-linux-gnu.so'
    raise ValueError('unregistered ABI')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')

def safe(name):
    name = str(name).replace('\\', '/')
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('unsafe member: '+name)
    return name

def identity(path):
    return dict(sha256=sha(path), bytes=Path(path).stat().st_size)

def verify_files(root, rows):
    for row in rows:
        path = root/safe(row['path'])
        if identity(path) != {key: row[key] for key in ('sha256','bytes')}:
            raise ValueError('input identity mismatch: '+str(path))

def proof(abi):
    return json.loads((DELIVERY/'proofs-final'/f'{abi}.json').read_text())

def prepare():
    manifest = json.loads((DELIVERY/'payload-final/relink-manifest.json').read_text())
    results = {}
    for abi in ABIS:
        source = DELIVERY/'payload-final/relink'/abi
        folder = OUT/'work'/abi
        folder.mkdir(parents=True, exist_ok=False)
        shutil.copytree(source, folder/'relink')
        rows = manifest[abi]['files']
        verify_files(source, rows); verify_files(folder/'relink', rows)
        recipe = manifest[abi]['command']
        if sha(source/recipe) != sha(folder/'relink'/recipe):
            raise ValueError('recipe copy mismatch')
        wheel_proof = proof(abi)
        wheel = DELIVERY/safe(wheel_proof['refreshed_wheel'])
        if sha(wheel) != wheel_proof['refreshed_sha256']:
            raise ValueError('delivered wheel mismatch')
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist(): safe(member.filename)
            archive.extractall(folder/'package')
        runtime = wheel_proof['preserved_runtime_members']
        verify_files(folder/'package', [dict(path=name, **row) for name,row in runtime.items()
                                      if not name.endswith('/')])
        results[abi] = dict(inputs=rows, recipe=identity(source/recipe),
            wheel=identity(wheel), original_native={name: row for name,row in runtime.items()
                if name.endswith(('.so','.pyd'))}, copied_recipe_unchanged=True)
    write(OUT/'input-identities.json',dict(abis=results, manifests={name:identity(DELIVERY/'payload-final'/name)
        for name in ('relink-manifest.json','component-manifest.json','README.md')}, native_calls=0))

def windows_environment():
    env = dict(os.environ)
    env.update(json.loads((OLD/'windows-prerequisite-probe-explicit-release/toolchain-environment.json').read_text()))
    # Match the already installed matrix 3.12 runtime, including stable ABI SDK.
    cfg = (MATRIX/'cells/windows312/venv/pyvenv.cfg').read_text()
    home = next(line.split(' = ',1)[1] for line in cfg.splitlines() if line.startswith('home = '))
    env['PYTHON_LIB_DIR'] = str(Path(home)/'libs')
    env['MPIR_LIB_DIR'] = str(OLD/'windows-sdk/Library/lib')
    return env

def repair():
    folder=OUT/'work/linux311/relink'
    original=(folder/'relink.sh').read_bytes()
    repaired=original.replace(b'\r\n',b'\n')
    if b'\r\n' not in original or b'\r' in repaired:raise ValueError('expected CRLF-only recipe')
    with (folder/'relink.lf.sh').open('xb') as stream:stream.write(repaired)
    write(OUT/'linux-recipe-repair.json',dict(original=identity(folder/'relink.sh'),
        repaired=identity(folder/'relink.lf.sh'),replacement=dict(old='\\r\\n',new='\\n'),
        occurrences=original.count(b'\r\n'),exact_transform='original.replace(b"\\r\\n", b"\\n")',
        text_identical_after_newline_normalization=True,original_preserved=True))

def recover():
    path=OUT/'budget.json';ledger=json.loads(path.read_text())
    receipt=json.loads((OUT/'processes/link-linux312abi3/result.json').read_text())
    if any(row['name']=='link-linux312abi3' for row in ledger['attempts']):raise ValueError('already recovered')
    entry=dict(receipt,name='link-linux312abi3',accounting_recovered_from_existing_receipt=True,
        controller_error='duplicate command keyword prevented ledger update; child receipt preserved',
        controller_executable='/mnt/c/ANYworkspaceWork/workflow2-20261027/linux-runtime/bin/python',
        runner_sha256_unavailable=True)
    ledger['attempts'].append(entry);ledger['used_seconds']+=receipt['process_wall_seconds']
    ledger['remaining_seconds']=600-ledger['used_seconds']
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(ledger,indent=2));temporary.replace(path)

def link(abi,repaired=False):
    folder = OUT/'work'/abi
    env = windows_environment() if abi.startswith('windows') else dict(os.environ)
    recipe = folder/'relink'/('relink.cmd' if os.name == 'nt' else 'relink.sh')
    if repaired:recipe=recipe.with_name('relink.lf.sh')
    if (os.name == 'nt') != abi.startswith('windows'):
        raise ValueError('ABI/platform mismatch')
    # Execute the delivered portable recipe without changing a byte.
    command = ['cmd.exe','/d','/c',str(recipe)] if os.name == 'nt' else ['sh',str(recipe)]
    tool = shutil.which('link.exe' if os.name == 'nt' else 'c++', path=env['PATH'])
    if tool is None: raise ValueError('linker/compiler missing')
    info = dict(command=command, cwd=str(recipe.parent), compiler=tool,
        compiler_identity=identity(tool), python=sys.version, executable=sys.executable,
        effective_environment={key: env.get(key) for key in
            ('LIB','INCLUDE','PYTHON_LIB_DIR','MPIR_LIB_DIR','VCToolsInstallDir','WindowsSDKVersion')})
    if os.name == 'nt':
        info['python_import_library_path'] = str(Path(env['PYTHON_LIB_DIR'])/'python3.lib')
        info['python_import_library'] = identity(Path(info['python_import_library_path']))
        info['gmp_import_library'] = identity(Path(env['MPIR_LIB_DIR'])/'gmp.lib')
        version = subprocess.run([tool,'/?'],env=env,capture_output=True,text=True)
    else:
        version = subprocess.run([tool,'--version'],env=env,capture_output=True,text=True)
        info['os_release'] = Path('/etc/os-release').read_text()
        info['system_libraries'] = {str(p):identity(p) for p in
            (Path('/usr/lib/x86_64-linux-gnu/libgmp.so.10'),Path('/usr/lib/x86_64-linux-gnu/libgomp.so.1'))}
    info['compiler_version_stdout'] = version.stdout
    info['compiler_version_stderr'] = version.stderr
    info['recipe_modified']=repaired
    info['recipe_identity']=identity(recipe)
    write(folder/('effective-link-repaired.json' if repaired else 'effective-link.json'),info)
    subprocess.run(command,cwd=recipe.parent,env=env,check=True)
    output = recipe.parent/native_output(abi)
    write(folder/'raw-link-output.json',dict(path=str(output),**identity(output), qualified=False,native_calls=0))

def package(abi):
    folder = OUT/'work'/abi
    raw = json.loads((folder/'raw-link-output.json').read_text())
    output = folder/'relink'/Path(raw['path']).name
    if sha(output) != raw['sha256']: raise ValueError('raw link output identity mismatch')
    native_names = [name for name in proof(abi)['preserved_runtime_members'] if name.endswith(('.pyd','.so'))]
    if len(native_names) != 1: raise ValueError('one native extension required')
    target = folder/'package'/native_names[0]
    shutil.copyfile(output,target)
    if os.name == 'nt':
        # The delivered wheel has hashed dependency names. Adapt only the NEW
        # extension, retaining every wrapper/bootstrap/dependency byte unchanged.
        sys.path.insert(0,str(OLD/'windows-build-venv/Lib/site-packages'))
        import pefile
        from delvewheel import _dll_utils
        with pefile.PE(str(target),fast_load=False) as pe:
            needed=[entry.dll.decode() for entry in pe.DIRECTORY_ENTRY_IMPORT]
        dlls = list((folder/'package/pytetwild.libs').glob('*.dll'))
        mapping = {}
        for name in needed:
            matches=[p.name for p in dlls if p.name.lower().startswith(Path(name).stem.lower()+'-')]
            if len(matches)>1: raise ValueError('ambiguous DLL adaptation')
            if matches: mapping[name]=matches[0]
        _dll_utils.replace_needed(str(target),list(mapping),mapping,strip=False,direct=False)
        with pefile.PE(str(target),fast_load=False) as pe:
            updated=[entry.dll.decode() for entry in pe.DIRECTORY_ENTRY_IMPORT]
        if updated != [mapping.get(name,name) for name in needed]:
            raise ValueError('DLL import-name adaptation mismatch')
        write(folder/'windows-native-adaptation.json',dict(mapping=mapping,before=needed,after=updated,
            tool_files={str(p.relative_to(OLD)):identity(p) for p in
                (OLD/'windows-build-venv/Lib/site-packages/delvewheel/_dll_utils.py',
                 OLD/'windows-build-venv/Lib/site-packages/pefile.py')},
            wrapper_bootstrap_dependencies_changed=False))
    unchanged = [dict(path=name,**row) for name,row in proof(abi)['preserved_runtime_members'].items()
                 if not name.endswith('/') and name not in native_names]
    verify_files(folder/'package',unchanged)
    write(folder/'derivative.json',dict(path=native_names[0],**identity(target),qualified=False,
        original_native=proof(abi)['preserved_runtime_members'][native_names[0]],
        unchanged_runtime_files=len(unchanged),recipe_modified=abi=='linux311',native_calls=0))

def smoke(abi):
    import importlib.metadata
    expected_version=(3,12) if abi=='windows312abi3' else (3,11)
    if sys.version_info[:2] != expected_version:raise ValueError('ABI/interpreter mismatch')
    folder=OUT/'work'/abi
    sys.path.insert(0,str(folder/'package'))
    import pytetwild
    from pytetwild import PyfTetWildWrapper as native
    expected=json.loads((folder/'derivative.json').read_text())
    if Path(native.__file__).resolve() != (folder/'package'/expected['path']).resolve():
        raise ValueError('wrong extension imported')
    if sha(native.__file__) != expected['sha256']:raise ValueError('extension identity mismatch')
    if getattr(native,'_workflow2_exact_envelope',False) is not True:raise ValueError('exact marker missing')
    exports=['tetrahedralize_csg','tetrahedralize_mesh']
    if not all(callable(getattr(native,name,None)) for name in exports):raise ValueError('exports missing')
    if 'pyvista' in sys.modules:raise ValueError('unexpected PyVista import')
    if importlib.metadata.version('pytetwild')!='0.4.2+anyworkflow2.exact1':raise ValueError('version mismatch')
    unchanged=[dict(path=name,**row) for name,row in proof(abi)['preserved_runtime_members'].items()
        if not name.endswith('/') and name!=expected['path']]
    verify_files(folder/'package',unchanged)
    write(folder/'import.json',dict(status='PASS',python=sys.version,executable=sys.executable,
        imported_from=str(native.__file__),native_sha256=sha(native.__file__),marker=True,
        exports=exports,exports_called=False,native_calls=0,qualified=False,
        unchanged_runtime_files=len(unchanged),version=importlib.metadata.version('pytetwild')))

def audit():
    inputs=json.loads((OUT/'input-identities.json').read_text())
    results={}
    for abi in ABIS:
        folder=OUT/'work'/abi
        verify_files(DELIVERY/'payload-final/relink'/abi,inputs['abis'][abi]['inputs'])
        verify_files(folder/'relink',inputs['abis'][abi]['inputs'])
        recipe='relink.cmd' if abi.startswith('windows') else 'relink.sh'
        if identity(folder/'relink'/recipe)!=inputs['abis'][abi]['recipe']:
            raise ValueError('original copied recipe changed')
        if sha(DELIVERY/safe(proof(abi)['refreshed_wheel']))!=proof(abi)['refreshed_sha256']:
            raise ValueError('original delivered wheel changed')
        derivative=json.loads((folder/'derivative.json').read_text())
        imported=json.loads((folder/'import.json').read_text())
        if imported['native_sha256']!=derivative['sha256'] or imported['status']!='PASS':
            raise ValueError('missing successful derivative import')
        verify_files(folder/'package',[dict(path=name,**row) for name,row in
            proof(abi)['preserved_runtime_members'].items() if not name.endswith('/') and name!=derivative['path']])
        results[abi]=dict(derivative=derivative,import_proof=imported,
            raw_link=json.loads((folder/'raw-link-output.json').read_text()))
    original=(OUT/'work/linux311/relink/relink.sh').read_bytes()
    repaired=(OUT/'work/linux311/relink/relink.lf.sh').read_bytes()
    if original.replace(b'\r\n',b'\n')!=repaired:raise ValueError('recipe repair changed more than newlines')
    checks=json.loads((OUT/'checks.json').read_text())
    if checks['failed'] or checks['passed']!=10:raise ValueError('refusal/registration checks missing')
    prior=json.loads((PRIOR/'handoff-final.json').read_text())
    if identity(REPO/'tools/package_exact_provider_notices.py') != prior['generator_comparison']['fixed_generator']:
        raise ValueError('generator changed; prior 12 checks cannot be reused')
    ledger=json.loads((OUT/'budget.json').read_text())
    if not all(row.get('cleanup',{}).get('verified') for row in ledger['attempts']):
        raise ValueError('cleanup not verified')
    write(OUT/'handoff.json',dict(state='verified against assigned technical link/import criteria; root review pending',
        results=results,focused_checks=checks,recipe_repair=json.loads((OUT/'linux-recipe-repair.json').read_text()),
        native_meshing_calls=0,qualification_reruns=0,qualified_derivatives=False,publication=False,
        budget_reference='budget.json (includes audit after its controller completes)',
        generator_tests_reused=dict(counts=prior['generator_tests'],source=identity(PRIOR/'handoff-final.json'),
            unchanged_generator=identity(REPO/'tools/package_exact_provider_notices.py')),
        failures_preserved=[row['name'] for row in ledger['attempts'] if row['status']!='completed' or row.get('exit_code')!=0],
        limits=[
            'No numerical/physics equivalence or provider/release promotion',
            'No final retained-symbol/header/source licensing or Microsoft redistribution adjudication',
            'Windows derivative DLL imports adapted with installed delvewheel; unchanged delivered bootstrap/dependencies',
            'Linux original CRLF recipe preserved; separate newline-normalized recipient recipe succeeds',
            'W2DEV001 exact-case waiver, original thresholds and raw NO_GO remain unchanged'],
        commands_reference='budget.json attempts[].command plus work/*/effective-link*.json'))

def checks():
    import tempfile
    failures=0; count=0
    for name in ('../bad','/bad','C:/bad','a/../../bad'):
        try:safe(name);failures+=1
        except ValueError:pass
        count+=1
    with tempfile.TemporaryDirectory(dir=OUT) as temporary:
        root=Path(temporary);(root/'input').write_bytes(b'original')
        rows=[dict(path='input',**identity(root/'input'))]
        verify_files(root,rows);count+=1
        (root/'input').write_bytes(b'mutated')
        try:verify_files(root,rows);failures+=1
        except ValueError:pass
        count+=1
    select_profile('remaining');count+=1
    try:select_profile('prior');failures+=1
    except ValueError:pass
    count+=1
    if native_output('windows312abi3')!='PyfTetWildWrapper.pyd':failures+=1
    count+=1
    if native_output('linux311')!='PyfTetWildWrapper.cpython-311-x86_64-linux-gnu.so':failures+=1
    count+=1
    write(OUT/'checks.json',dict(passed=count-failures,failed=failures,native_calls=0))
    if failures:raise ValueError('focused refusal checks failed')

def generator_check():
    spec=importlib.util.spec_from_file_location('relink_generator',REPO/'tools/package_exact_provider_notices.py')
    generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
    successful=(OUT/'work/linux312abi3/relink/relink.lf.sh').read_bytes()
    command=successful.decode('utf-8').splitlines(keepends=True)[-1]
    target=OUT/'generator-comparison/relink.sh';target.parent.mkdir(exist_ok=False)
    generator.write_posix_relink_recipe(target,command)
    if target.read_bytes()!=successful:raise ValueError('new generator differs from executed LF recipe')
    write(OUT/'generator-comparison/result.json',dict(status='PASS',generated=identity(target),
        successfully_linked=identity(OUT/'work/linux312abi3/relink/relink.lf.sh'),
        pre_fix_generator=identity(OUT/'package_exact_provider_notices.pre-fix.py'),
        fixed_generator=identity(REPO/'tools/package_exact_provider_notices.py'),native_calls=0))
    subprocess.run([sys.executable,'-I','-B','-m','pytest','-q',str(REPO/'tests/test_exact_provider_notices.py'),
        '--junitxml='+str(OUT/'generator-focused.xml')],cwd=REPO,check=True)

def finalize():
    import xml.etree.ElementTree as ET
    report=ET.parse(OUT/'generator-focused-repaired.xml').getroot()
    suites=[report] if report.tag=='testsuite' else list(report.findall('testsuite'))
    counts={key:sum(int(suite.get(key,'0')) for suite in suites)
            for key in ('tests','failures','errors','skipped')}
    if counts!={'tests':12,'failures':0,'errors':0,'skipped':0}:raise ValueError('focused checks incomplete')
    handoff=json.loads((OUT/'handoff.json').read_text())
    handoff.update(generator_comparison=json.loads((OUT/'generator-comparison/result.json').read_text()),
        generator_tests=counts,generator_tests_command_reference='budget.json attempts[name=focused-tests]',
        source_fix='write_posix_relink_recipe uses explicit UTF-8 and LF; export_relink calls it',
        historical_payloads_unchanged=True, review_status='root independent review pending')
    handoff['failures_preserved'].append('processes/generator-check (11 passed, 1 synthetic-fixture assertion failed; repaired to 12 passed)')
    write(OUT/'handoff-final.json',handoff)

def run(action,abi,child):
    OUT.mkdir(parents=True,exist_ok=True)
    ledger_path=OUT/'budget.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(limit_seconds=600.,used_seconds=0.,attempts=[])
    remaining=600-ledger['used_seconds']
    reserve=0 if action in ('smoke','checks','audit') else 90
    timeout=min(180.,remaining-reserve)
    if timeout<=0:raise RuntimeError('recipient check budget exhausted')
    name=action+('-'+abi if abi else '')
    folder=OUT/'processes'/name
    if folder.exists():raise FileExistsError(folder)
    snapshot=OUT/'runner-snapshots'/f'{name}.py';snapshot.parent.mkdir(exist_ok=True)
    with snapshot.open('xb') as stream:stream.write(Path(__file__).read_bytes())
    command=[child or sys.executable,'-I','-B',str(Path(__file__).resolve()),action,'--profile',PROFILE]
    if abi:command+=['--abi',abi]
    if action=='focused-tests':
        command=[sys.executable,'-I','-B','-m','pytest','-q',str(REPO/'tests/test_exact_provider_notices.py'),
            '--junitxml='+str(OUT/'generator-focused-repaired.xml')]
    helper=Path.home()/'bin/claude-review.py' if os.name=='nt' else REPO/'tools/workflow2_process_supervisor.py'
    # All identities are recorded before the supervised child is launched.
    write(OUT/'step-identities'/f'{name}.json',dict(profile=PROFILE,abis=ABIS,
        runner=identity(snapshot),supervisor=identity(helper),
        generator=identity(REPO/'tools/package_exact_provider_notices.py'),
        controller=dict(executable=sys.executable,identity=identity(sys.executable),version=sys.version),
        child=dict(executable=command[0],identity=identity(command[0])),
        delivery_manifest=identity(DELIVERY/'payload-final/relink-manifest.json'),
        input_identities=identity(OUT/'input-identities.json') if (OUT/'input-identities.json').exists() else None,
        proofs={name:identity(DELIVERY/'proofs-final'/f'{name}.json') for name in ABIS}))
    spec=importlib.util.spec_from_file_location('relink_supervisor',helper)
    supervisor=importlib.util.module_from_spec(spec);spec.loader.exec_module(supervisor)
    receipt=None;started=time.perf_counter()
    try:
        env=dict(os.environ);env['PYTHONDONTWRITEBYTECODE']='1'
        if os.name=='nt':
            folder.mkdir(parents=True,exist_ok=False)
            receipt,_=supervisor.run_process(command,folder,'stdout',timeout,env,REPO)
        else:receipt=supervisor.run_linux(command,folder,timeout,env,REPO)
    finally:
        if receipt is None:
            receipt=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else dict(status='controller_error',process_wall_seconds=time.perf_counter()-started)
        ledger['used_seconds']+=receipt['process_wall_seconds']
        ledger['remaining_seconds']=600-ledger['used_seconds']
        entry=dict(receipt)
        entry.update(name=name,command=command,runner_sha256=sha(snapshot),
            supervisor_sha256=sha(helper),controller_python=sys.version,controller_executable=sys.executable)
        ledger['attempts'].append(entry)
        temporary=ledger_path.with_suffix('.tmp');temporary.write_text(json.dumps(ledger,indent=2));temporary.replace(ledger_path)
    print(json.dumps(dict(action=name,status=receipt['status'],used_seconds=ledger['used_seconds'],remaining_seconds=ledger['remaining_seconds'])))
    return 0 if receipt['status']=='completed' and receipt.get('exit_code')==0 and receipt.get('cleanup',{}).get('verified') else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('run','prepare','link','link-repaired','repair','package','smoke','checks','audit'))
    parser.add_argument('action',nargs='?',choices=('prepare','link','link-repaired','repair','package','smoke','checks','audit'))
    parser.add_argument('--profile',choices=('remaining','prior'),default='remaining')
    parser.add_argument('--abi',choices=ABIS)
    parser.add_argument('--child-python')
    args=parser.parse_args()
    select_profile(args.profile)
    if args.phase=='run' and not args.action:parser.error('run requires action')
    operation=args.action if args.phase=='run' else args.phase
    if operation in ('link','link-repaired','package','smoke') and not args.abi:parser.error('action requires --abi')
    if operation=='link' and args.abi=='linux311':parser.error('use normalized copied recipe via link-repaired')
    if operation=='link-repaired' and args.abi!='linux311':parser.error('normalized recipe only registered for linux311')
    if args.phase=='run':return run(args.action,args.abi,args.child_python)
    if args.phase=='prepare':prepare()
    elif args.phase=='checks':checks()
    elif args.phase=='link':link(args.abi)
    elif args.phase=='link-repaired':link(args.abi,repaired=True)
    elif args.phase=='repair':repair()
    elif args.phase=='recover':recover()
    elif args.phase=='audit':audit()
    elif args.phase=='generator-check':generator_check()
    elif args.phase=='finalize':finalize()
    elif args.phase=='package':package(args.abi)
    else:smoke(args.abi)
    return 0

if __name__=='__main__':raise SystemExit(main())
