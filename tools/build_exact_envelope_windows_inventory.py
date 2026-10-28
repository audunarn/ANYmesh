"""Windows source/mode/dependency evidence and upstream-route DLL packaging."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import zipfile
from build_exact_envelope import save, sha


def inspect(args):
    root=args.root.resolve();build=root/'windows-build'
    wheels=list((root/'windows-wheels').glob('*.whl'))
    if len(wheels)!=1:raise ValueError('exactly one Windows source wheel required')
    wheel=wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        natives={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.endswith(('.pyd','.dll'))}
    env=dict(os.environ);env.update(json.loads((root/'windows-prerequisite-probe-explicit-release/toolchain-environment.json').read_text()))
    vs=Path(env['VCINSTALLDIR']).parent
    ninja=vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
    dumpbin=Path(env['VCToolsInstallDir'])/'bin/Hostx64/x64/dumpbin.exe'
    target='CMakeFiles/PyfTetWildWrapper.dir/src/FTetWildWrapper.cpp.obj'
    command=subprocess.check_output([str(ninja),'-C',str(build),'-t','commands',target],env=env,text=True)
    if 'NEW_ENVELOPE' not in command:raise ValueError('compiled exact-envelope macro missing')
    cache=(build/'CMakeCache.txt').read_text()
    flag=next(line for line in cache.splitlines() if line.startswith('FLOAT_TETWILD_WITH_EXACT_ENVELOPE:'))
    if flag!='FLOAT_TETWILD_WITH_EXACT_ENVELOPE:BOOL=ON':raise ValueError('effective mode not ON')
    cpp=root/'source/src/FTetWildWrapper.cpp'
    if sha(cpp)!=json.loads((root/'wrapper-glue-patch.json').read_text())['after_sha256']:
        raise ValueError('wrapper source changed after approved patch')
    directory=root/'windows-native-build-configuration';directory.mkdir(exist_ok=True)
    for name in ('CMakeCache.txt','compile_commands.json','build.ninja'):
        if (build/name).exists():(directory/name).write_bytes((build/name).read_bytes())
    (directory/'wrapper-compile.txt').write_text(command)
    link=subprocess.check_output([str(ninja),'-C',str(build),'-t','commands','PyfTetWildWrapper.pyd'],env=env,text=True)
    (directory/'relink-commands.txt').write_text(link)
    native=build/'PyfTetWildWrapper.pyd'
    if not native.exists():native=root/'source/src/pytetwild/PyfTetWildWrapper.pyd'
    dependencies=subprocess.check_output([str(dumpbin),'/DEPENDENTS',str(native)],env=env,text=True)
    (directory/'dumpbin-dependents.txt').write_text(dependencies)
    git_records=[]
    bases=[root/'source',*(p for p in (build/'_deps').glob('*-src'))]
    seen=set()
    for base in bases:
        for current,dirs,files in os.walk(base):
            if current in seen:dirs[:]=[];continue
            seen.add(current)
            if '.git' in dirs or '.git' in files:
                path=Path(current)
                commit=subprocess.check_output(['git','-c','safe.directory='+str(path),'-C',str(path),'rev-parse','HEAD'],text=True).strip()
                git_records.append(dict(path=current,commit=commit))
            dirs[:]=[name for name in dirs if name not in ('.git','.cache')]
    linux_commits={record['commit'] for record in json.loads((root/'recursive-git-closure.json').read_text())}
    if {record['commit'] for record in git_records}!=linux_commits:
        raise ValueError('Windows fetched Git closure differs from Linux pins')
    save(root/'windows-recursive-git-closure.json',git_records)
    result=dict(wheel=str(wheel),wheel_sha256=sha(wheel),native_sha256=natives,
         cmake_cache_flag=flag,compiled_macro_present=True,compiled_marker_import_verified=False,
         wrapper_source_sha256=sha(cpp),source_commits_match_linux=True,
         retained_build_root=str(build),static_libraries_sha256={str(p):sha(p) for p in build.rglob('*.lib')},
         known_compiler_warnings='MSVC ignores upstream GCC-specific -O3/-fsigned-char/-ffp-contract=off; effective MSVC flags retained',
         dependency_packaging='raw wheel requires upstream-route DLL repair or explicit SDK search directory',
         numerical_checks_invoked=False)
    save(root/'windows-wheel-build-proof.json',result)
    print(json.dumps(dict(wheel=str(wheel),sha256=result['wheel_sha256'],natives=natives,source_repositories=len(git_records))))


def repair(args):
    root=args.root.resolve();python=root/'windows-build-venv/Scripts/python.exe'
    tools=root/'windows-repair-tool-wheels';tools.mkdir()
    commands=[
        [str(python),'-m','pip','download','--only-binary=:all:','--dest',str(tools),'delvewheel'],
        [str(python),'-m','pip','install','--no-index','--find-links',str(tools),'delvewheel'],
    ]
    wheel=next((root/'windows-wheels').glob('*.whl'))
    commands.append([str(python),'-m','delvewheel','repair',str(wheel),'--add-path',str(root/'windows-sdk/Library/bin'),
                     '--wheel-dir',str(root/'windows-repaired-wheels')])
    save(root/'windows-dll-repair-commands.json',commands)
    for command in commands:subprocess.run(command,check=True)
    output=next((root/'windows-repaired-wheels').glob('*.whl'))
    with zipfile.ZipFile(output) as archive:
        native={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.endswith(('.pyd','.dll'))}
    save(root/'windows-repaired-wheel.json',dict(wheel=str(output),sha256=sha(output),native_sha256=native,
        repair_tool_wheels={p.name:sha(p) for p in tools.glob('*.whl')},meshing_invoked=False,
        license_distribution_gate='pending including MPIR LGPL3 and i_pred LGPL3-or-later; no publication'))
    print(json.dumps(dict(repaired_wheel=str(output),sha256=sha(output),native_files=len(native))))


def finalize(args):
    root=args.root.resolve();licenses=[]
    for wheel in (root/'windows-repair-tool-wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or not (any(term in name.lower() for term in ('license','copying')) or name.endswith('/METADATA')):continue
                target=root/'license-evidence/windows-repair-tools'/wheel.stem/name
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
                licenses.append(dict(wheel=wheel.name,path=name,sha256=sha(target),retained=str(target)))
    save(root/'windows-repair-tools-license-inventory.json',licenses)
    repaired=json.loads((root/'windows-repaired-wheel.json').read_text())
    origins=[]
    for name in repaired['native_sha256']:
        match=re.fullmatch(r'(.+)-([0-9a-f]{32})\.dll',Path(name).name)
        if match is None:continue
        original=match[1]+'.dll';hint=match[2]
        candidates=list((root/'windows-sdk').rglob(original))+[Path('C:/Windows/System32')/original]
        records=[]
        for path in candidates:
            if path.exists():
                digest=sha(path)
                records.append(dict(path=str(path),sha256=digest,matches_mangled_identity=digest.startswith(hint)))
        origins.append(dict(bundled=name,original_name=original,mangled_identity_hint=hint,source_candidates=records))
    save(root/'windows-bundled-dll-origin.json',origins)
    # Hash tracked files in nested Geogram submodules omitted by a parent's git
    # ls-files. The Git commits were already checked equal across both platforms.
    base_records=json.loads((root/'dependency-inventory.json').read_text())
    base_names={Path(record['path']).name for record in base_records}
    supplemental={}
    for record in json.loads((root/'windows-recursive-git-closure.json').read_text()):
        path=Path(record['path'])
        if path.name in base_names:continue
        names=subprocess.check_output(['git','-c','safe.directory='+str(path),'-C',str(path),'ls-files','-z'],text=True).split('\0')
        for name in names:
            source=path/name
            if name and source.is_file():supplemental[str(source)]=sha(source)
    save(root/'nested-source-files-sha256.json',supplemental)
    summary=dict(version='0.4.2+anyworkflow2.exact1',
        linux=json.loads((root/'import-only-smoke.json').read_text()),
        windows_raw=json.loads((root/'windows-wheel-build-proof.json').read_text()),
        windows_packaged=repaired,
        windows_import_executed=False,worker_meshing_executed=False,
        source_patch='candidate-source.patch',flag_proof=['native-flag-proof.json','windows-wheel-build-proof.json'],
        license_distribution_gate='pending non-MIT/non-MPL-only dependency assessment; includes LGPL3-or-later i_pred, LGPL3 MPIR, Microsoft runtime terms',
        retained_build_budget='build-budget.json')
    save(root/'candidate-build-summary.json',summary)
    print(json.dumps(dict(repair_tool_notices=len(licenses),bundled_dlls=len(origins),nested_source_files=len(supplemental))))


def attest(args):
    root=args.root.resolve()
    env=dict(os.environ);env.update(json.loads((root/'windows-prerequisite-probe-explicit-release/toolchain-environment.json').read_text()))
    directory=root/'windows-compiler-attestation';directory.mkdir()
    header=root/'windows-build/_deps/exact_envelope-src/3rdparty/i_pred/numerics.h'
    source=directory/'macro_probe.cpp'
    source.write_text('''#include <climits>
#include "'''+str(header).replace('\\','/')+'''"
#ifndef _MSC_VER
#error Required MSVC compiler macro absent
#endif
#ifndef ISVISUALSTUDIO
#error Required i_pred Visual Studio branch absent
#endif
#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)
#pragma message("WORKFLOW2_MSC_VER=" STRINGIFY(_MSC_VER))
static_assert(CHAR_MIN < 0, "signed char required by fTetWild");
int workflow2_macro_probe() { return _MSC_VER; }
''')
    compiler=Path(env['VCToolsInstallDir'])/'bin/Hostx64/x64/cl.exe'
    commands=[
        [str(compiler),'/nologo','/c','/EHsc','/MD','/O2','/DNDEBUG',str(source),'/Fo'+str(directory/'macro_probe.obj')],
        [str(compiler),'/nologo','/P','/EHsc',str(source),'/Fi'+str(directory/'macro_probe.i')],
    ]
    save(directory/'commands.json',commands)
    outputs=[]
    for index,command in enumerate(commands):
        result=subprocess.run(command,env=env,capture_output=True,text=True)
        (directory/f'command-{index}.stdout.txt').write_text(result.stdout)
        (directory/f'command-{index}.stderr.txt').write_text(result.stderr)
        if result.returncode:raise RuntimeError(f'MSVC macro/header evidence failed: {result.returncode}')
        outputs.append(result.stdout+result.stderr)
    match=re.search(r'WORKFLOW2_MSC_VER=(\d+)',outputs[0])
    if match is None:raise ValueError('MSVC version macro evidence missing')
    all_commands=(root/'windows-build/compile_commands.json').read_text()
    environment_flags={key:env.get(key,'') for key in ('CL','_CL_')}
    combined=all_commands+' '+str(environment_flags)
    if re.search(r'[/-]fp:(fast|contract)\b',combined,re.I):
        raise ValueError('unexpected fast/contract floating-point option')
    preprocessed=(directory/'macro_probe.i').read_text()
    if 'setFPUModeToRoundUP' not in preprocessed or '_controlfp' not in preprocessed:
        raise ValueError('expected MSVC rounding-control branch missing from preprocessed header')
    result=dict(msc_ver=int(match[1]),signed_char_static_assert_passed=True,
        i_pred_visual_studio_branch_preprocessed=True,header=str(header),header_sha256=sha(header),
        preprocessed_sha256=sha(directory/'macro_probe.i'),
        fast_or_contract_flags_present=False,observed_fp_flags=sorted(set(re.findall(r'[/-]fp:[A-Za-z_]+',all_commands))),
        compiler_environment_flags=environment_flags,numerical_execution=False,
        assumption_adjudication='root-owned; compiler/header evidence only')
    save(root/'windows-compiler-attestation.json',result)
    print(json.dumps(result))
    origins(args)


def origins(args):
    root=args.root.resolve()
    # Delvewheel seeds each DLL hash with hashes of its mangled dependencies.
    # Resolve origin with that actual algorithm, not an unseeded file-hash guess.
    sys.path.insert(0,str(root/'windows-build-venv/Lib/site-packages'))
    from delvewheel import _dll_utils
    from delvewheel._wheel_repair import WheelRepair
    repaired=json.loads((root/'windows-repaired-wheel.json').read_text())
    hints={}
    for name in repaired['native_sha256']:
        match=re.fullmatch(r'(.+)-([0-9a-f]{32})\.dll',Path(name).name)
        if match:hints[match[1].lower()+'.dll']=match[2]
    origins=[]
    for name,hint in hints.items():
        candidates=list((root/'windows-sdk').rglob(name))+[Path('C:/Windows/System32')/name]
        resolved=[]
        for candidate in candidates:
            if not candidate.exists():continue
            # WheelRepair excludes CPython-provided CRT DLLs according to the
            # wheel's minimum Python. Only dependencies actually mangled into
            # this wheel seed its graph hash.
            needed=[key for key in _dll_utils.get_direct_mangleable_needed(str(candidate),set(),set()) if key in hints]
            with candidate.open('rb') as handle:
                computed=WheelRepair._hashfile(handle,start=(hints[key] for key in needed))
            if computed==hint:resolved.append(dict(path=str(candidate),sha256=sha(candidate),dependent_mangle_hashes={key:hints[key] for key in needed}))
        if not resolved:raise ValueError(f'cannot prove bundled DLL origin: {name}')
        origins.append(dict(original_name=name,mangled_hash=hint,proven_sources=resolved))
    save(root/'windows-bundled-dll-origin-proven.json',origins)
    print(json.dumps(dict(proven_dll_origins=len(origins))))


def smoke(args):
    import ctypes
    from ctypes import wintypes
    root=args.root.resolve();wheel=next((root/'windows-repaired-wheels').glob('*.whl'))
    directory=root/'windows-import-only-smoke';directory.mkdir()
    with zipfile.ZipFile(wheel) as archive:archive.extractall(directory)
    sys.path.insert(0,str(directory))
    import pytetwild
    import pytetwild.PyfTetWildWrapper as native
    if pytetwild.__version__!='0.4.2+anyworkflow2.exact1' or native._workflow2_exact_envelope is not True:
        raise ValueError('Windows candidate version/compiled mode mismatch')
    if 'pyvista' in sys.modules:raise ValueError('unexpected PyVista import')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetModuleHandleW.argtypes=[wintypes.LPCWSTR];kernel.GetModuleHandleW.restype=wintypes.HMODULE
    kernel.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];kernel.GetModuleFileNameW.restype=wintypes.DWORD
    loaded={}
    manifest=json.loads((root/'windows-repaired-wheel.json').read_text())
    for name,expected in manifest['native_sha256'].items():
        if not name.endswith('.dll'):continue
        handle=kernel.GetModuleHandleW(Path(name).name)
        if not handle:raise ValueError(f'expected packaged DLL not loaded: {name}')
        buffer=ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(handle,buffer,len(buffer)):raise ctypes.WinError(ctypes.get_last_error())
        path=Path(buffer.value).resolve()
        if not path.is_relative_to(directory) or sha(path)!=expected:
            raise ValueError('loaded DLL does not match extracted wheel bytes')
        loaded[name]=dict(path=str(path),sha256=sha(path))
    result=dict(status='import_only_passed',version=pytetwild.__version__,compiled_exact_envelope_marker=True,
        imported_from=str(pytetwild.__file__),native_module=str(native.__file__),native_sha256=sha(native.__file__),
        wheel=str(wheel),wheel_sha256=sha(wheel),loaded_packaged_dlls=loaded,
        pyvista_imported=False,meshing_invoked=False,python=sys.version)
    save(root/'windows-import-only-smoke.json',result)
    summary=json.loads((root/'candidate-build-summary.json').read_text())
    summary.update(windows_import_executed=True,windows_import_only=result)
    save(root/'candidate-build-summary.json',summary)
    print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inspect','repair','finalize','attest','origins','smoke'))
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();globals()[args.phase](args)


if __name__=='__main__':main()
