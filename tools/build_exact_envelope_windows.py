"""Bounded Windows compiler/GMP prerequisite feasibility; never runs a mesher."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from build_exact_envelope import save, sha

ENV_NAMES = ('PATH','INCLUDE','LIB','LIBPATH','VCINSTALLDIR','VCToolsInstallDir',
             'WindowsSdkDir','WindowsSDKVersion','UniversalCRTSdkDir','UCRTVersion')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--dump-env', type=Path)
    p.add_argument('--work-name', default='windows-prerequisite-probe')
    args = p.parse_args()
    if args.dump_env is not None:
        save(args.dump_env,{key:os.environ[key] for key in ENV_NAMES if key in os.environ})
        return
    root = args.root.resolve()
    if not root.name.startswith('exact-envelope-'):
        raise ValueError('owned exact-envelope root required')
    work = root / args.work_name
    work.mkdir()
    vswhere = Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    installations = json.loads(subprocess.check_output([str(vswhere),'-latest','-products','*',
        '-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-format','json'],text=True))
    if len(installations) != 1:
        raise RuntimeError('one available MSVC installation required')
    vs = Path(installations[0]['installationPath'])
    vcvars = vs / 'VC/Auxiliary/Build/vcvars64.bat'
    cmake = vs / 'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    ninja = vs / 'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
    environment = work / 'toolchain-environment.json'
    script = work / 'activate-child.cmd'
    script.write_text('@echo off\ncall "'+str(vcvars)+'" >nul\nif errorlevel 1 exit /b %errorlevel%\n"'+sys.executable+'" "'+str(Path(__file__).resolve())+'" --dump-env "'+str(environment)+'"\n')
    subprocess.run(['cmd.exe','/d','/c',str(script)],check=True)
    child_env = dict(os.environ)
    child_env.update(json.loads(environment.read_text()))
    source = work / 'source';source.mkdir()
    source.joinpath('CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.20)
project(ExactEnvelopePrerequisites LANGUAGES CXX)
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("int main(){static_assert(sizeof(void*)==8);return 0;}" EXACT_MSVC_X64_COMPILES)
find_path(EXACT_GMP_INCLUDE gmp.h PATHS
  "C:/Users/AudunArnesenNyhus/miniconda3/Library/include"
  "C:/Users/AudunArnesenNyhus/miniconda3/envs/python312/Library/include")
find_library(EXACT_GMP_LIBRARY NAMES gmp libgmp PATHS
  "C:/Users/AudunArnesenNyhus/miniconda3/Library/lib"
  "C:/Users/AudunArnesenNyhus/miniconda3/envs/python312/Library/lib")
if(NOT EXACT_MSVC_X64_COMPILES)
 message(FATAL_ERROR "Required MSVC x64 C++ compile failed")
endif()
if(NOT EXACT_GMP_INCLUDE OR NOT EXACT_GMP_LIBRARY)
 message(FATAL_ERROR "Pinned fTetWild requires Windows GMP-compatible headers and import/static library; isolated provisioning is missing")
endif()
''')
    argv = [str(cmake),'-S',str(source),'-B',str(work/'build'),'-G','Ninja',
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_TRY_COMPILE_CONFIGURATION=Release',
            '-DCMAKE_MAKE_PROGRAM='+str(ninja)]
    with (work/'configure.stdout.txt').open('wb') as out, (work/'configure.stderr.txt').open('wb') as err:
        result = subprocess.run(argv,env=child_env,stdout=out,stderr=err)
    cache = (work/'build/CMakeCache.txt').read_text() if (work/'build/CMakeCache.txt').exists() else ''
    compiler = next((line for line in cache.splitlines() if line.startswith('CMAKE_CXX_COMPILER:')),None)
    compiled = 'EXACT_MSVC_X64_COMPILES:INTERNAL=1' in cache
    missing = 'EXACT_GMP_INCLUDE:PATH=EXACT_GMP_INCLUDE-NOTFOUND' in cache or 'EXACT_GMP_LIBRARY:FILEPATH=EXACT_GMP_LIBRARY-NOTFOUND' in cache
    receipt = dict(status='blocked_missing_gmp' if compiled and missing else ('prerequisites_available' if result.returncode == 0 else 'toolchain_failed'),
                   cmake_command=argv,cmake_exit_code=result.returncode,msvc_x64_compile_passed=compiled,
                   compiler=compiler,visual_studio_version=installations[0]['installationVersion'],
                   cmake_version=subprocess.check_output([str(cmake),'--version'],text=True).splitlines()[0],
                   ninja_version=subprocess.check_output([str(ninja),'--version'],text=True).strip(),
                   cmake_sha256=sha(cmake),ninja_sha256=sha(ninja),
                   candidate_build_invoked=False,meshing_invoked=False,
                   remaining_platform_gate='isolated GMP/MPIR provisioning, full exact-envelope configure/build, wheel dependency packaging and import smoke')
    save(root/'windows-build-feasibility.json',receipt)
    print(json.dumps(receipt))
    if receipt['status'] == 'toolchain_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
