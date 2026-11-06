"""Supervised source-build tooling for the unqualified exact-envelope candidate.

This tool does not run meshing, install into an existing runtime, or promote a
provider. All source/build artifacts and receipts live in a fresh evidence root.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

VERSION = '0.4.2+anyworkflow2.exact1'
FTETWILD_COMMIT = 'd7d99bb4387a07895b9adce058dc7305f6b6e5ab'
FAST_ENVELOPE_COMMIT = '520ee04b6c69a802db31d1fd3a3e6e382d10ef98'
BUDGET = 2400.


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supervisor():
    source = Path('C:/Github/ANYsolid/tools/adapter_platform_matrix.py') if os.name == 'nt' else Path('/mnt/c/Github/ANYsolid/tools/adapter_platform_matrix.py')
    spec = importlib.util.spec_from_file_location('exact_build_supervisor', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.name != 'nt':
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(36, 1, 0, 0, 0):  # PR_SET_CHILD_SUBREAPER
            raise OSError(ctypes.get_errno(), 'cannot enable owned-child subreaper')
    return module, source


def command(args):
    root = args.root.resolve()
    evidence_parent = Path(__file__).resolve().parents[2] / 'evidence'
    if not root.name.startswith('exact-envelope-') or root.parent != evidence_parent.resolve():
        raise ValueError('root must be a unique exact-envelope-* directory in the sibling evidence folder')
    if not args.name or Path(args.name).name != args.name or any(value in args.name for value in ('/', '\\')) or args.name in ('.', '..'):
        raise ValueError('phase name must be one relative directory component')
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / 'build-budget.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(limit_seconds=BUDGET, used_seconds=0., attempts=[])
    remaining = BUDGET - ledger['used_seconds']
    timeout = min(float(args.timeout), 1200., remaining)
    if timeout <= 0:
        raise RuntimeError('cumulative build budget exhausted')
    folder = root / args.name
    if folder.exists():
        raise FileExistsError(folder)
    env = dict(os.environ)
    cache = root / 'caches' / args.name
    cache.mkdir(parents=True)
    for key in ('PYTHONPATH', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT'):
        env.pop(key, None)
    temporary = root / 'temporary' / args.name
    temporary.mkdir(parents=True)
    env.update(CMAKE_BUILD_PARALLEL_LEVEL='4', CTEST_PARALLEL_LEVEL='4',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PIP_CACHE_DIR=str(cache / 'pip'), UV_CACHE_DIR=str(cache / 'uv'),
               TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
               PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    for item in args.env:
        key, value = item.split('=', 1)
        if key not in {'SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTETWILD', 'CMAKE_GENERATOR',
                       'GMP_INC', 'GMP_LIB', 'PATH', 'INCLUDE', 'LIB', 'LIBPATH'}:
            raise ValueError(f'unsupported child-only build variable: {key}')
        env[key] = value
    module, source = supervisor()
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv:
        raise ValueError('a concrete command is required')
    identity = dict(command=argv, cwd=str(args.cwd.resolve()), platform=sys.platform,
                    python=sys.version, executable=sys.executable, timeout_seconds=timeout,
                    cumulative_used_before_seconds=ledger['used_seconds'],
                    cumulative_remaining_before_seconds=remaining, compile_jobs=4,
                    runner_sha256=sha(__file__), supervisor_path=str(source),
                    supervisor_sha256=sha(source), effective_environment={k:env[k] for k in (
                        'CMAKE_BUILD_PARALLEL_LEVEL','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                        'MKL_NUM_THREADS','PIP_CACHE_DIR','UV_CACHE_DIR','TEMP','TMP','TMPDIR')},
                    explicit_child_environment=dict(item.split('=',1) for item in args.env))
    identity['command_file_sha256'] = {item:sha(item) for item in argv if Path(item).is_file() and Path(item).suffix in ('.py','.cmake','.txt','.toml')}
    # The shared supervisor owns folder creation on both platforms.
    started = time.perf_counter()
    receipt = None
    try:
        if os.name == 'nt':
            folder.mkdir()
            receipt, _ = module.windows_supervisor().run_process(argv, folder, 'stdout', timeout, env, args.cwd.resolve())
        else:
            receipt = module.linux_process(argv, folder, timeout, env, args.cwd.resolve())
    finally:
        if folder.exists():
            save(folder / 'build-identity.json', identity)
            if receipt is None and (folder / 'result.json').exists():
                receipt = json.loads((folder / 'result.json').read_text())
            if receipt is None:
                receipt = dict(status='controller_error', process_wall_seconds=time.perf_counter()-started)
            ledger['used_seconds'] += receipt['process_wall_seconds']
            ledger['attempts'].append(dict(name=args.name, **receipt))
            ledger['remaining_seconds'] = max(0., BUDGET-ledger['used_seconds'])
            save(ledger_path, ledger)
    print(json.dumps(dict(evidence=str(folder), **receipt, cumulative_used_seconds=ledger['used_seconds'], remaining_seconds=ledger['remaining_seconds'])), flush=True)
    return 0 if receipt['status'] == 'completed' and receipt.get('cleanup', {}).get('verified') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--cwd', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1200)
    parser.add_argument('--env', action='append', default=[])
    parser.add_argument('command', nargs=argparse.REMAINDER)
    return command(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
