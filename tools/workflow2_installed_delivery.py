"""Offline Windows delivery smoke, not provider or platform qualification.

Uses an existing Python 3.11 runtime's build tools/dependencies and a registered
unchanged ANYsolid wheel. Never installs into that runtime or invokes a provider.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SOLID_SHA = '601bd500625e6cd95f3a741f4f19b87bdcb4a72a7c7650278fea8be3efc56c3c'
DEPENDENCIES = ('numpy', 'scipy', 'scikit-fem', 'shapely', 'pytest', 'pluggy',
                'packaging', 'iniconfig', 'pygments', 'colorama')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def copy_dependencies(root):
    target = root/'venv/Lib/site-packages'
    records = {}
    for name in DEPENDENCIES:
        distribution = metadata.distribution(name)
        origins = sorted({str(p).replace('\\', '/').split('/')[0]
                          for p in distribution.files if '..' not in p.parts})
        files = {}
        for origin in origins:
            source = Path(distribution.locate_file(origin))
            destination = target/origin
            if source.is_dir():
                shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
            elif source.is_file():
                shutil.copy2(source, destination)
            paths = destination.rglob('*') if destination.is_dir() else [destination]
            for path in paths:
                if path.is_file():
                    files[str(path.relative_to(target))] = sha(path)
        records[name] = dict(version=distribution.version, source=str(distribution.locate_file('')),
                             files=files)
    save(root/'dependency-inputs.json', records)


def origins(root):
    # -I makes repository files and PYTHONPATH irrelevant to these imports.
    import anygeometry
    import anymesher
    forbidden = {'pytetwild', 'pyvista', 'vtk', 'anysolid', 'anysolver', 'ngsolve'}
    assert not forbidden.intersection(sys.modules)
    import anysolid
    import numpy
    import scipy
    import skfem
    modules = (anygeometry, anymesher, anysolid, numpy, scipy, skfem)
    expected = (root/'venv/Lib/site-packages').resolve()
    result = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(expected), str(path)
        result[module.__name__] = dict(path=str(path), sha256=sha(path))
    for wheel in (root/'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or '.data/' in name or name.endswith('RECORD'):
                    continue
                installed = expected/name
                assert installed.is_file() and sha(installed) == hashlib.sha256(archive.read(name)).hexdigest(), name
    save(root/'installed-origins.json', dict(status='PASS', modules=result,
         base_import_optional_backends_absent=True, wheel_payloads_match=True,
         executable=sys.executable, version=sys.version))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--solid-wheel', type=Path)
    parser.add_argument('--phase', choices=('dependencies', 'origins'))
    parser.add_argument('--resume', action='store_true', help='continue frozen evidence; preserve failures and budget')
    parser.add_argument('--refresh-example', action='store_true', help='retain and refresh example inputs after a documented repair')
    args = parser.parse_args()
    root = args.evidence.resolve()
    if args.phase:
        return copy_dependencies(root) if args.phase == 'dependencies' else origins(root)
    if os.name != 'nt':
        parser.error('this delivery smoke uses the existing Windows Job supervisor')
    if root.exists() and not args.resume:
        parser.error('evidence directory must be new; use --resume for retained frozen inputs')
    if args.resume and not (root/'manifest.json').is_file():
        parser.error('resume requires a frozen manifest')
    if args.refresh_example and not args.resume:
        parser.error('--refresh-example requires --resume')
    if not args.solid_wheel or sha(args.solid_wheel) != SOLID_SHA:
        parser.error('registered unchanged ANYsolid wheel required')
    if not args.resume:
        root.mkdir(parents=True)
        for name in ('sources', 'wheels', 'run', 'processes'):
            (root/name).mkdir()
        hashes = {}
        for name in ('ANYgeometry', 'ANYmesh'):
            source = WORKSPACE/name
            target = root/'sources'/name
            target.mkdir()
            for filename in ('pyproject.toml', 'setup.py', 'README.md', 'LICENSE', 'NOTICE',
                             'THIRD_PARTY_NOTICES.md', 'dependency-licenses.json', 'MANIFEST.in'):
                if (source/filename).is_file():
                    shutil.copy2(source/filename, target/filename)
            shutil.copytree(source/'src', target/'src',
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyd', '*.so', '*.dll', '*.egg-info'))
            (target/'docs').mkdir()
            shutil.copy2(source/'docs/LICENSE.md', target/'docs/LICENSE.md')
            hashes.update({str(p.relative_to(root)): sha(p) for p in target.rglob('*') if p.is_file()})
        for name in ('examples', 'tests'):
            (root/'run'/name).mkdir()
        for name in ('examples/solid_volume_controlled.py', 'tests/test_installed_volume_example.py'):
            shutil.copy2(REPO/name, root/'run'/name)
            hashes[name] = sha(root/'run'/name)
        shutil.copy2(args.solid_wheel, root/'wheels'/args.solid_wheel.name)
    supervisor = Path.home()/'bin/claude-review.py'
    spec = importlib.util.spec_from_file_location('delivery_job_supervisor', supervisor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = dict(os.environ)
    env.update(PYTHONPATH='', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               ANYGEOMETRY_DISABLE_NATIVE='1', ANYMESHER_DISABLE_NATIVE='1',
               ANYGEOMETRY_BUILD_NATIVE='0', ANYGEOMETRY_REQUIRE_NATIVE='0', ANYMESHER_REQUIRE_NATIVE='0',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    if not args.resume:
        save(root/'manifest.json', dict(scope='fresh installed controlled delivery; zero native calls',
         input_hashes=hashes, solid_wheel_sha256=SOLID_SHA, harness_sha256=sha(__file__),
         supervisor_path=str(supervisor), supervisor_sha256=sha(supervisor),
         effective_environment={k: env[k] for k in env if k.startswith(('ANY', 'PIP_', 'PYTHON')) or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
         runtime=sys.executable, version=sys.version))
    ledger = dict(limit_seconds=600, verification_reserve_seconds=60,
                  used_seconds=0., remaining_seconds=600., attempts=[])
    if args.resume:
        ledger = json.loads((root/'budget.json').read_text())
        if args.refresh_example and any(
                row['name'].split('-attempt')[0] in ('example', 'tests') and row['status'] == 'completed'
                for row in ledger['attempts']):
            parser.error('completed example inputs cannot be refreshed in this evidence run')
        frozen = json.loads((root/'manifest.json').read_text())
        revisions_path = root/'input-revisions.json'
        revisions = json.loads(revisions_path.read_text()) if revisions_path.exists() else []
        current = dict(frozen['input_hashes'])
        for revision in revisions:
            current.update(revision['new_hashes'])
        for name, expected in current.items():
            path = root/name if name.startswith('sources') else root/'run'/name
            assert sha(path) == expected, name
        if args.refresh_example:
            retained = root/'retained-inputs'/str(len(revisions)+1)
            retained.mkdir(parents=True)
            new_hashes = {}
            for name in ('examples/solid_volume_controlled.py', 'tests/test_installed_volume_example.py'):
                shutil.copy2(root/'run'/name, retained/Path(name).name)
                shutil.copy2(REPO/name, root/'run'/name)
                new_hashes[name] = sha(root/'run'/name)
            revisions.append(dict(retained_previous=str(retained), new_hashes=new_hashes))
            save(revisions_path, revisions)
    save(root/'budget.json', ledger)
    (root/'tmp').mkdir(exist_ok=True)
    env['TEMP'] = env['TMP'] = str(root/'tmp')

    def check(name, command, cwd=root, maximum=180):
        timeout = min(maximum, int(ledger['remaining_seconds']-60))
        if timeout <= 0:
            raise RuntimeError('delivery budget reserve reached')
        if any(row['name'].split('-attempt')[0] == name and row['status'] == 'completed' for row in ledger['attempts']):
            return
        original = name
        attempt = 1
        while (root/'processes'/name).exists():
            attempt += 1
            name = original+'-attempt'+str(attempt)
        folder = root/'processes'/name
        folder.mkdir()
        save(folder/'command.json', dict(command=list(map(str, command)), cwd=str(cwd), timeout=timeout, harness_sha256=sha(__file__),
             environment={k: env[k] for k in ('TEMP','TMP','PYTHONPATH','ANYGEOMETRY_DISABLE_NATIVE','ANYMESHER_DISABLE_NATIVE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
        receipt, _ = module.run_process(list(map(str, command)), folder, 'stdout', timeout, env, cwd)
        ledger['used_seconds'] += receipt['process_wall_seconds']
        ledger['remaining_seconds'] = 600-ledger['used_seconds']
        ledger['attempts'].append(dict(name=name, **receipt))
        save(root/'budget.json', ledger)
        print(json.dumps(dict(name=name, status=receipt['status'], seconds=receipt['process_wall_seconds'])), flush=True)
        if receipt['status'] != 'completed' or not receipt['cleanup']['verified']:
            raise RuntimeError('failed delivery step: '+name)

    for name in ('ANYgeometry', 'ANYmesh'):
        check('build-'+name, [sys.executable, '-m', 'pip', 'wheel', '--no-build-isolation',
              '--no-deps', '--no-index', '--wheel-dir', root/'wheels', root/'sources'/name])
    check('create-venv', [sys.executable, '-m', 'venv', '--without-pip', root/'venv'])
    check('copy-dependencies', [sys.executable, '-I', __file__, '--evidence', root, '--phase', 'dependencies'])
    python = root/'venv/Scripts/python.exe'
    wheels = sorted((root/'wheels').glob('*.whl'))
    save(root/'wheel-hashes.json', {p.name: sha(p) for p in wheels})
    check('install', [sys.executable, '-m', 'pip', '--python', python, 'install', '--no-index', '--no-deps', *wheels])
    check('origins', [python, '-I', __file__, '--evidence', root, '--phase', 'origins'])
    check('pip-check', [sys.executable, '-m', 'pip', '--python', python, 'check'])
    check('example', [python, '-I', root/'run/examples/solid_volume_controlled.py', '--output', root/'example.json'], cwd=root/'run')
    check('tests', [python, '-I', '-m', 'pytest', '-q', '-o', 'pythonpath=',
          root/'run/tests/test_installed_volume_example.py', '--junitxml='+str(root/'junit.xml')], cwd=root/'run')
    save(root/'handoff.json', dict(state='verified against assigned controlled delivery criteria',
         native_meshing_calls=0, wheel_hashes=json.loads((root/'wheel-hashes.json').read_text()),
         budget=ledger, review_status='root review pending',
         omissions=['native/provider qualification', 'broad regressions', 'platform matrix', 'release acceptance'],
         retained=['copied inputs', 'fresh venv', 'wheel artifacts', 'all process stdout/stderr/terminal receipts']))


if __name__ == '__main__':
    main()
