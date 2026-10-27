"""Bounded local Workflow 2 checks; retains exact inputs and full diagnostics."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('--evidence-root', type=Path, required=True)
    p.add_argument('--pythonpath', action='append', default=[])
    p.add_argument('--source', action='append', type=Path, default=[])
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--linux-supervisor', type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        p.error('command is required')
    if command[0] == 'python':
        command[0] = sys.executable
    folder = a.evidence_root / (a.name + '-' + uuid.uuid4().hex[:10])
    folder.mkdir(parents=True)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(a.pythonpath)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    env['PYTHONUTF8'] = '1'
    if '-m' in command and 'pytest' in command:
        command += ['--junitxml=' + str(folder / 'pytest.xml')]
    hashes = {}
    for source in a.source:
        paths = [source] if source.is_file() else source.rglob('*')
        for path in paths:
            if path.is_file() and path.suffix in ('.py', '.toml', '.json', '.yml'):
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = dict(command=command, cwd=str(a.cwd.resolve()), python=sys.version,
                    pythonpath=env['PYTHONPATH'], timeout_seconds=a.timeout,
                    threads={k: env[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    source_hashes=hashes, started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    identity['native_flags'] = {key:env.get(key) for key in (
        'ANYMESHER_DISABLE_NATIVE','ANYMESHER_REQUIRE_NATIVE',
        'ANYGEOMETRY_DISABLE_NATIVE','ANYGEOMETRY_BUILD_NATIVE','ANYGEOMETRY_REQUIRE_NATIVE')}
    (folder / 'identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    # Reuse the existing Windows Job Object supervisor, without any inference.
    helper = a.linux_supervisor if os.name != 'nt' else Path.home() / 'bin' / 'claude-review.py'
    if helper is None:
        p.error('--linux-supervisor must identify the established local Linux process helper')
    spec = importlib.util.spec_from_file_location('workflow2_supervisor', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if os.name == 'nt':
        receipt, _ = module.run_process(command, folder, 'stdout', a.timeout, env, a.cwd)
    else:
        import ctypes
        if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0) != 0:
            raise OSError('Linux child-subreaper unavailable')
        receipt = module.linux_process(command, folder/'process', a.timeout, env, a.cwd)
    print(json.dumps({'evidence': str(folder), **receipt}))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
