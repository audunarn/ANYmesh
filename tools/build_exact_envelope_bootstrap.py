"""Provision only an owned candidate build venv and retain build-tool wheels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import venv
from build_exact_envelope import save, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--venv', required=True, type=Path)
    p.add_argument('--reuse-wheels', action='store_true')
    p.add_argument('--record-prefix', default='')
    args = p.parse_args()
    if args.venv.exists():
        raise FileExistsError(args.venv)
    # Debian's interpreter need not ship ensurepip. The already installed pip
    # provisions this new venv via --python; no host package is changed.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(args.venv)
    python = args.venv / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    wheels = args.root / 'build-tool-wheels'
    if not args.reuse_wheels:
        wheels.mkdir()
    elif not list(wheels.glob('*.whl')):
        raise ValueError('retained build-tool wheels required')
    packages = ['scikit-build-core','nanobind','setuptools-scm','wheel','pip==26.2.2']
    commands = [] if args.reuse_wheels else [[sys.executable,'-m','pip','download','--only-binary=:all:','--dest',str(wheels),*packages]]
    commands.append([sys.executable,'-m','pip','--python',str(python),'install','--no-index','--find-links',str(wheels),*packages])
    prior = args.root / (args.record_prefix+'bootstrap-commands.json')
    if prior.exists():
        save(args.root / (args.record_prefix+'bootstrap-commands-original.json'), json.loads(prior.read_text()))
    save(prior, commands)
    for command in commands:
        subprocess.run(command, check=True)
    installed = subprocess.check_output([str(python),'-m','pip','list','--format=json'], text=True)
    result = dict(python=str(python), python_version=sys.version,
                  inherited_base_site_packages=False,
                  installed_packages=json.loads(installed),
                  build_tool_wheels={path.name:sha(path) for path in wheels.glob('*.whl')})
    prior = args.root / (args.record_prefix+'build-environment.json')
    if prior.exists():
        save(args.root / (args.record_prefix+'build-environment-original.json'), json.loads(prior.read_text()))
    save(prior, result)
    print(json.dumps(dict(python=str(python), downloaded_wheels=len(result['build_tool_wheels']))))


if __name__ == '__main__':
    main()
