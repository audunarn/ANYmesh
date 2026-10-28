"""Build-only Windows candidate with child-local MSVC and isolated MPIR SDK."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from build_exact_envelope import VERSION, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    env=dict(os.environ)
    env.update(json.loads((root/'windows-prerequisite-probe-explicit-release/toolchain-environment.json').read_text()))
    sdk=root/'windows-sdk/Library'
    if not (sdk/'include/gmp.h').exists() or not list((sdk/'lib').glob('*gmp*.lib')):
        raise ValueError('isolated upstream-CI GMP-compatible MPIR SDK required')
    env.update(GMP_INC=str(sdk/'include'),GMP_LIB=str(sdk/'lib'),
               SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTETWILD=VERSION,CMAKE_GENERATOR='Ninja',
               GIT_CONFIG_COUNT='2',GIT_CONFIG_KEY_0='core.longpaths',GIT_CONFIG_VALUE_0='true',
               GIT_CONFIG_KEY_1='core.autocrlf',GIT_CONFIG_VALUE_1='false',GIT_TERMINAL_PROMPT='0')
    vs=Path(env['VCINSTALLDIR']).parent
    cmake=vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    ninja=vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
    env['PATH']=str(cmake.parent)+os.pathsep+str(ninja.parent)+os.pathsep+env['PATH']
    python=root/'windows-build-venv/Scripts/python.exe'
    command=[str(python),'-m','pip','wheel','.', '--no-deps','--no-build-isolation','--verbose',
             '--wheel-dir',str(root/'windows-wheels'),
             '--config-settings=build-dir='+str(root/'windows-build'),
             '--config-settings=cmake.define.FLOAT_TETWILD_WITH_EXACT_ENVELOPE=ON',
             '--config-settings=cmake.define.CMAKE_TRY_COMPILE_CONFIGURATION=Release',
             '--config-settings=cmake.define.CMAKE_POLICY_VERSION_MINIMUM=3.5']
    save(root/'windows-wheel-command.json',dict(command=command,cwd=str(root/'source'),
         child_only_environment={key:env[key] for key in ('GMP_INC','GMP_LIB','SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTETWILD',
            'CMAKE_GENERATOR','GIT_CONFIG_COUNT','GIT_CONFIG_KEY_0','GIT_CONFIG_VALUE_0','GIT_CONFIG_KEY_1','GIT_CONFIG_VALUE_1')},
         existing_runtime_mutated=False,meshing_invoked=False))
    raise SystemExit(subprocess.run(command,cwd=root/'source',env=env).returncode)


if __name__=='__main__':
    main()
