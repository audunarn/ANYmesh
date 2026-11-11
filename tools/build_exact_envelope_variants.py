"""Missing ABI variants with measured native-object reuse and separate artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile
from build_exact_envelope import VERSION,save,sha

BASES={
 'linux311':'/mnt/c/Github/ANYsolid/evidence/w1/adapter-platform-20261027T122824Z-a8d13295/managed-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11',
 'windows312':'C:/Users/AudunArnesenNyhus/AppData/Roaming/uv/python/cpython-3.12-windows-x86_64-none/python.exe',
}


def paths(args):
    root=args.root.resolve();variant=root/args.variant
    build=root/('linux-build' if args.variant=='linux311' else 'windows-build')
    python=variant/('venv/bin/python' if args.variant=='linux311' else 'venv/Scripts/python.exe')
    return root,variant,build,python


def core_objects(build):
    return {str(path.relative_to(build)):dict(sha256=sha(path),mtime_ns=path.stat().st_mtime_ns)
        for path in build.rglob('*') if path.is_file() and path.suffix in ('.o','.obj')
        and 'nanobind' not in str(path).lower() and 'PyfTetWildWrapper' not in str(path)}


def prepare(args):
    root,variant,build,python=paths(args);variant.mkdir()
    base=Path(BASES[args.variant])
    if not base.exists():raise ValueError(f'installed ABI interpreter missing: {base}')
    save(variant/'native-core-before.json',core_objects(build))
    save(variant/'core-commands-before.json',json.loads((build/'compile_commands.json').read_text()))
    archived=[]
    with tarfile.open(variant/'previous-abi-relink-inputs.tar.gz','w:gz',compresslevel=1) as archive:
        for path in build.rglob('*'):
            if path.is_file() and (path.suffix in ('.o','.obj','.a','.lib','.so','.pyd','.exp') or path.name in ('CMakeCache.txt','compile_commands.json','build.ninja','CMakeInit.txt')):
                archive.add(path,arcname=str(path.relative_to(build)),recursive=False);archived.append(str(path.relative_to(build)))
    save(variant/'previous-abi-preservation.json',dict(archive_sha256=sha(variant/'previous-abi-relink-inputs.tar.gz'),files=archived,
          note='Preserves original ABI objects/libraries/configuration before reconfiguring the shared build directory.'))
    subprocess.run([str(base),'-m','venv','--without-pip',str(variant/'venv')],check=True)
    packages=['scikit-build-core','nanobind','setuptools-scm','wheel','pip==26.2.2']
    subprocess.run([sys.executable,'-m','pip','--python',str(python),'install','--no-index','--find-links',str(root/'build-tool-wheels'),*packages],check=True)
    runtime=variant/'runtime-wheels';runtime.mkdir()
    subprocess.run([str(python),'-m','pip','download','--only-binary=:all:','--dest',str(runtime),'numpy==2.4.6'],check=True)
    subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(runtime),'numpy==2.4.6'],check=True)
    identity=json.loads(subprocess.check_output([str(python),'-c','import sys,json;print(json.dumps(dict(version=list(sys.version_info[:3]),executable=sys.executable,bits=64 if sys.maxsize>2**32 else 32)))'],text=True))
    expected=[3,11] if args.variant=='linux311' else [3,12]
    if identity['version'][:2]!=expected or identity['bits']!=64:raise ValueError('ambiguous/wrong ABI interpreter')
    identity.update(base=str(base),numpy_wheels={p.name:sha(p) for p in runtime.glob('*.whl')})
    save(variant/'environment.json',identity);print(json.dumps(identity))


def child_environment(args):
    root,variant,build,python=paths(args);env=dict(os.environ)
    env.update(SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTETWILD=VERSION,CMAKE_GENERATOR='Ninja')
    if args.variant=='windows312':
        env.update(json.loads((root/'windows-prerequisite-probe-explicit-release/toolchain-environment.json').read_text()))
        sdk=root/'windows-sdk/Library';env.update(GMP_INC=str(sdk/'include'),GMP_LIB=str(sdk/'lib'),
            GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='core.longpaths',GIT_CONFIG_VALUE_0='true',GIT_CONFIG_KEY_1='core.autocrlf',GIT_CONFIG_VALUE_1='false')
        vs=Path(env['VCINSTALLDIR']).parent
        env['PATH']=str(vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin')+os.pathsep+str(vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja')+os.pathsep+env['PATH']
    return env


def build(args):
    root,variant,build_dir,python=paths(args);env=child_environment(args)
    command=[str(python),'-m','pip','wheel','.', '--no-deps','--no-build-isolation','--verbose','--wheel-dir',str(variant/'wheels'),
       '--config-settings=build-dir='+str(build_dir),'--config-settings=cmake.define.FLOAT_TETWILD_WITH_EXACT_ENVELOPE=ON',
       '--config-settings=cmake.define.Python_EXECUTABLE='+str(python)]
    if args.variant=='windows312':
        compiler=str(Path(env['VCToolsInstallDir'])/'bin/Hostx64/x64/cl.exe')
        command.extend(['--config-settings=cmake.define.CMAKE_TRY_COMPILE_CONFIGURATION=Release','--config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5',
                        '--config-settings=cmake.define.CMAKE_CXX_COMPILER='+compiler,'--config-settings=cmake.define.CMAKE_C_COMPILER='+compiler])
    save(variant/'build-command.json',dict(command=command,environment={key:env.get(key) for key in ('SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTETWILD','CMAKE_GENERATOR','GMP_INC','GMP_LIB')}))
    subprocess.run(command,cwd=root/'source',env=env,check=True)
    before=json.loads((variant/'native-core-before.json').read_text());after=core_objects(build_dir)
    unchanged=[name for name,value in before.items() if after.get(name)==value]
    changed=[name for name,value in before.items() if name in after and after[name]!=value]
    old=json.loads((variant/'core-commands-before.json').read_text());new=json.loads((build_dir/'compile_commands.json').read_text())
    filtered=lambda entries:{row['file']:row['command'] for row in entries if 'nanobind' not in row['file'].lower() and not row['file'].endswith('FTetWildWrapper.cpp')}
    save(variant/'native-object-reuse-proof.json',dict(before=before,after=after,unchanged_objects=len(unchanged),changed_objects=changed,
          core_compile_commands_unchanged=filtered(old)==filtered(new),claim='reuse only for exact unchanged object hashes, mtimes and compile commands'))
    config=variant/'build-configuration';config.mkdir()
    for name in ('CMakeCache.txt','compile_commands.json','build.ninja','CMakeInit.txt'):
        (config/name).write_bytes((build_dir/name).read_bytes())
    ninja=shutil.which('ninja',path=env['PATH'])
    object_name='CMakeFiles/PyfTetWildWrapper.dir/src/FTetWildWrapper.cpp.'+('o' if args.variant=='linux311' else 'obj')
    wrapper_command=subprocess.check_output([ninja,'-C',str(build_dir),'-t','commands',object_name],env=env,text=True)
    flag=next(line for line in (config/'CMakeCache.txt').read_text().splitlines() if line.startswith('FLOAT_TETWILD_WITH_EXACT_ENVELOPE:'))
    if flag!='FLOAT_TETWILD_WITH_EXACT_ENVELOPE:BOOL=ON' or 'NEW_ENVELOPE' not in wrapper_command:raise ValueError('effective exact-envelope mode absent')
    limited='Py_LIMITED_API=0x030C0000' in wrapper_command
    if limited!=(args.variant=='windows312'):raise ValueError('wrapper stable-ABI macro does not match requested Python')
    save(variant/'native-flags.json',dict(cmake_flag=flag,wrapper_compile_command=wrapper_command,python_executable=str(python),
        py_limited_api_cp312=limited,wrapper_source_sha256=sha(root/'source/src/FTetWildWrapper.cpp')))
    print(json.dumps(dict(unchanged_core_objects=len(unchanged),changed_core_objects=len(changed),core_commands_unchanged=filtered(old)==filtered(new))))


def repair(args):
    root,variant,_,python=paths(args)
    if args.variant!='windows312':raise ValueError('DLL repair is Windows only')
    subprocess.run([str(python),'-m','pip','install','--no-index','--find-links',str(root/'windows-repair-tool-wheels'),'delvewheel'],check=True)
    wheel=next((variant/'wheels').glob('*.whl'))
    subprocess.run([str(python),'-m','delvewheel','repair',str(wheel),'--add-path',str(root/'windows-sdk/Library/bin'),'--wheel-dir',str(variant/'repaired-wheels')],check=True)


def smoke(args):
    import importlib
    root,variant,build_dir,python=paths(args)
    expected=(3,11) if args.variant=='linux311' else (3,12)
    if sys.version_info[:2]!=expected:raise ValueError('smoke is not running the requested interpreter ABI')
    wheel=next((variant/('wheels' if args.variant=='linux311' else 'repaired-wheels')).glob('*.whl'))
    tag='cp311-cp311-linux_x86_64' if args.variant=='linux311' else 'cp312-abi3-win_amd64'
    if tag not in wheel.name:raise ValueError('unexpected candidate ABI tag')
    directory=variant/('import-only-normalized-paths' if args.variant=='windows312' else 'import-only');directory.mkdir()
    with zipfile.ZipFile(wheel) as archive:archive.extractall(directory)
    sys.path.insert(0,str(directory));provider=importlib.import_module('pytetwild');native=importlib.import_module('pytetwild.PyfTetWildWrapper')
    if provider.__version__!=VERSION or native._workflow2_exact_envelope is not True or 'pyvista' in sys.modules:raise ValueError('native mode/version/lazy PyVista mismatch')
    hashes={p.relative_to(directory).as_posix():sha(p) for p in directory.rglob('*') if p.is_file() and p.suffix in ('.so','.pyd','.dll')}
    loaded={}
    if args.variant=='windows312':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetModuleHandleW.argtypes=[wintypes.LPCWSTR];kernel.GetModuleHandleW.restype=wintypes.HMODULE
        kernel.GetModuleFileNameW.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];kernel.GetModuleFileNameW.restype=wintypes.DWORD
        baseline={name:value for name,value in json.loads((root/'windows-repaired-wheel.json').read_text())['native_sha256'].items() if name.endswith('.dll')}
        if {name:value for name,value in hashes.items() if name.endswith('.dll')}!=baseline:raise ValueError('variant DLL bytes differ from proven SDK origin inventory')
        for name,digest in baseline.items():
            handle=kernel.GetModuleHandleW(Path(name).name);buffer=ctypes.create_unicode_buffer(32768)
            if not handle or not kernel.GetModuleFileNameW(handle,buffer,len(buffer)):raise ValueError('packaged DLL not loaded')
            path=Path(buffer.value).resolve()
            if not path.is_relative_to(directory) or sha(path)!=digest:raise ValueError('loaded variant DLL bytes/path mismatch')
            loaded[name]=dict(path=str(path),sha256=digest)
    result=dict(status='import_only_passed',python=sys.version,executable=sys.executable,expected_abi=list(expected),wheel=str(wheel),wheel_sha256=sha(wheel),
        version=provider.__version__,compiled_exact_envelope_marker=True,native_sha256=hashes,meshing_invoked=False,pyvista_imported=False,
        loaded_packaged_dlls=loaded,dll_origin_inventory='windows-bundled-dll-origin-proven.json' if loaded else None)
    save(variant/'import-only-smoke.json',result);print(json.dumps(result))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','build','repair','smoke'));p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',choices=tuple(BASES),required=True)
    args=p.parse_args();globals()[args.phase](args)


if __name__=='__main__':main()
