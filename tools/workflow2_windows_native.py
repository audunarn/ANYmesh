"""Offline, bounded Windows native W2 owner/consumer compatibility profile.

`prepare` freezes copied inputs and builds a private installed runtime, then only
collects tests. `verify` requires a root release receipt for the frozen manifest.
Passed steps cannot be replayed. This does not qualify a real volume provider.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
SOLID_SHA = '601bd500625e6cd95f3a741f4f19b87bdcb4a72a7c7650278fea8be3efc56c3c'
MESH_TESTS = ('test_volume_mesh.py', 'test_volume_process.py', 'test_volume_consumer.py',
              'test_installed_volume_example.py', 'test_workflow2_native_operations.py')
OWNER_TESTS = ('test_solid_regions.py', 'test_serialization.py', 'test_serialization_v3.py')
SERIAL_NODES = (
    'test_serialization.py::test_round_trip_preserves_complete_document_and_id_allocation',
    'test_serialization.py::test_json_and_gzip_files_are_deterministic_and_equivalent',
    'test_serialization_v3.py::test_schema_4_round_trip_preserves_identity_coordinates_and_structural_uses',
    'test_serialization_v3.py::test_checksum_is_canonical_and_detects_payload_corruption')
FLAGS = dict(ANYGEOMETRY_DISABLE_NATIVE='0', ANYMESHER_DISABLE_NATIVE='0',
             ANYGEOMETRY_BUILD_NATIVE='1', ANYGEOMETRY_REQUIRE_NATIVE='1', ANYMESHER_REQUIRE_NATIVE='1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment(root):
    env = dict(os.environ)
    env.update(FLAGS, PYTHONPATH='', PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
               PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TEMP=str(root/'tmp'), TMP=str(root/'tmp'))
    # Owner test shard settings must not silently narrow this fixed selection.
    env.pop('ANYGEOMETRY_TEST_SHARD', None)
    env.pop('ANYGEOMETRY_SHARD_REPORT', None)
    for key in ('PYTEST_ADDOPTS', 'PYTEST_PLUGINS', 'PYTHONOPTIMIZE', 'PYTHONHOME',
                'PYTHONSTARTUP', 'PYTHONINSPECT'):
        env.pop(key, None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return env


class Inventory:
    """Terminal inventory plus flushed phase journal; no successful partial runs."""
    def __init__(self, root, collect):
        self.root, self.collect = root, collect
        self.value = dict(selected=[], deselected=[], reports=[], completed=False)
        self.name = 'collection-inventory.json' if collect else 'test-inventory.json'

    def pytest_collection_modifyitems(self, items):
        self.value['selected'] = [item.nodeid for item in items]

    def pytest_deselected(self, items):
        self.value['deselected'].extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report):
        row = dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome)
        self.value['reports'].append(row)
        with (self.root/'test-phases.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
            stream.flush()

    def pytest_sessionfinish(self, session, exitstatus):
        self.value.update(completed=True, exitstatus=int(exitstatus))
        save(self.root/self.name, self.value)


def validate_outcomes(expected, actual, junit):
    """Validate exact independent terminal evidence; raises on every unmet gate."""
    witnesses = {
        'tests/test_workflow2_native_operations.py::test_required_geometry_native_plane_kernel',
        'tests/test_workflow2_native_operations.py::test_required_mesher_native_orientation_kernel'}
    if len(expected) != 118 or len(set(expected)) != 118 or not witnesses.issubset(expected):
        raise ValueError('invalid prospective 118-node inventory or missing witness')
    if not actual.get('completed') or actual.get('exitstatus') != 0:
        raise ValueError('missing or incomplete terminal session')
    if actual.get('deselected') or sorted(actual.get('selected', [])) != sorted(expected):
        raise ValueError('deselected or changed selection')
    reports = actual.get('reports', [])
    wanted = {(node, phase) for node in expected for phase in ('setup', 'call', 'teardown')}
    observed = [(row['nodeid'],row['when']) for row in reports]
    if len(observed) != len(wanted) or set(observed) != wanted:
        raise ValueError('missing, extra, or duplicate execution phases')
    if any(row['outcome'] != 'passed' for row in reports):
        raise ValueError('failed or skipped execution phase')
    tree = ET.parse(junit)
    cases = tree.findall('.//testcase')
    if len(cases) != 118 or any(case.find(tag) is not None for case in cases
                              for tag in ('failure', 'error', 'skipped')):
        raise ValueError('incomplete or unsuccessful JUnit evidence')
    for suite in tree.findall('.//testsuite'):
        if any(int(suite.get(key, '0')) for key in ('failures','errors','skipped')):
            raise ValueError('unsuccessful JUnit suite')
    return dict(passed=118, failed=0, skipped=0, deselected=0, native_witnesses=sorted(witnesses))


def freeze(root):
    for name in ('sources', 'wheels', 'run/tests', 'run/examples', 'run/tools', 'processes', 'tmp'):
        (root/name).mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyd', '*.so', '*.dll', '*.egg-info')
    for name in ('ANYgeometry', 'ANYmesh'):
        source, target = WORKSPACE/name, root/'sources'/name
        target.mkdir()
        for filename in ('pyproject.toml', 'setup.py', 'README.md', 'LICENSE', 'NOTICE',
                         'THIRD_PARTY_NOTICES.md', 'dependency-licenses.json', 'MANIFEST.in'):
            if (source/filename).is_file():
                shutil.copy2(source/filename, target/filename)
        for folder in ('src', 'native'):
            if (source/folder).is_dir():
                shutil.copytree(source/folder, target/folder, ignore=ignore)
        (target/'docs').mkdir()
        shutil.copy2(source/'docs/LICENSE.md', target/'docs/LICENSE.md')
    for name in MESH_TESTS:
        shutil.copy2(REPO/'tests'/name, root/'run/tests'/name)
    for name in OWNER_TESTS:
        shutil.copy2(WORKSPACE/'ANYgeometry/tests'/name, root/'run/tests'/name)
    shutil.copy2(REPO/'tests/conftest.py', root/'run/tests/conftest.py')
    shutil.copy2(REPO/'examples/solid_volume_controlled.py', root/'run/examples/solid_volume_controlled.py')
    for name in ('run_affected_tests.py', 'run_dev_smoke.py'):
        shutil.copy2(REPO/'tools'/name, root/'run/tools'/name)
    # The affected-test mapping test parses these bytes; src is never on sys.path.
    shutil.copytree(root/'sources/ANYmesh/src', root/'run/src')
    solid = WORKSPACE/'evidence/installed-delivery-20261028-a1/wheels/anysolid-0.1.0-py3-none-any.whl'
    assert sha(solid) == SOLID_SHA
    shutil.copy2(solid, root/'wheels'/solid.name)
    shutil.copy2(__file__, root/'runner.py')
    shutil.copy2(REPO/'tools/workflow2_installed_delivery.py', root/'dependencies.py')
    selections = ['tests/test_solid_regions.py', *('tests/'+name for name in SERIAL_NODES),
                  *('tests/'+name for name in MESH_TESTS)]
    frozen = {str(p.relative_to(root)):sha(p) for name in ('sources', 'run')
              for p in (root/name).rglob('*') if p.is_file()}
    supervisor = Path.home()/'bin/claude-review.py'
    save(root/'manifest.json', dict(scope='Windows CPython3.11 native W2 installed compatibility',
        source_commits={'ANYmesh':'2a9268e43e31fc56b5aee0a9f9c1f55533e7bd0c',
                        'ANYgeometry':'6a72e9d6eecf4db2aca5a5c4c37badfe4eb3a3ee'},
        input_hashes=frozen, selections=selections, harness_sha256=sha(root/'runner.py'),
        dependency_helper_sha256=sha(root/'dependencies.py'), supervisor_path=str(supervisor),
        supervisor_sha256=sha(supervisor), solid_wheel_sha256=SOLID_SHA,
        runtime=sys.executable, version=sys.version, effective_environment={
            k:v for k,v in environment(root).items() if k.startswith(('ANY','PIP_','PYTHON'))
            or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TEMP','TMP')},
        limits={'supervised_process_seconds':1200, 'each_process_seconds':300,
                'reserve_seconds':120, 'real_native_provider_calls':0},
        commands=commands(root, selections)))


def origins(root):
    import anygeometry
    import anymesher
    assert not {'pytetwild','pyvista','vtk','anysolid','anysolver','ngsolve'}.intersection(sys.modules)
    import anygeometry._core as core
    import anymesher._native as native
    import anysolid
    assert anygeometry.backend_info()['backend'] == 'native'
    expected = (root/'venv/Lib/site-packages').resolve()
    results = {}
    for module in (anygeometry, anymesher, anysolid, core, native):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(expected), str(path)
        results[module.__name__] = dict(path=str(path), sha256=sha(path))
    assert Path(core.__file__).suffix == '.pyd' and Path(native.__file__).suffix == '.pyd'
    for wheel in (root/'wheels').glob('*.whl'):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or '.data/' in name or name.endswith('RECORD'):
                    continue
                path = expected/name
                assert path.is_file() and sha(path) == hashlib.sha256(archive.read(name)).hexdigest(), name
    assert all(os.environ[k] == v for k,v in FLAGS.items())
    save(root/'installed-origins.json', dict(status='PASS', modules=results,
        executable=sys.executable, python=sys.version, backend=anygeometry.backend_info(),
        flags={k:os.environ[k] for k in FLAGS}, wheel_payloads_match=True,
        base_import_optional_backends_absent=True))


def commands(root, selections):
    py = str(root/'venv/Scripts/python.exe')
    base = [py, '-I', str(root/'runner.py'), 'pytest-child', '--evidence',str(root),
            '--', '-q', '-o', 'pythonpath=', '-p', 'no:cacheprovider']
    return dict(
        build_geometry=[sys.executable,'-m','pip','wheel','--no-build-isolation','--no-deps','--no-index',
                        '--wheel-dir',str(root/'wheels'),str(root/'sources/ANYgeometry')],
        build_mesh=[sys.executable,'-m','pip','wheel','--no-build-isolation','--no-deps','--no-index',
                    '--wheel-dir',str(root/'wheels'),str(root/'sources/ANYmesh')],
        create_venv=[sys.executable,'-m','venv','--without-pip',str(root/'venv')],
        dependencies=[sys.executable,'-I',str(root/'dependencies.py'),'--evidence',str(root),'--phase','dependencies'],
        install=[sys.executable,'-m','pip','--python',py,'install','--no-index','--no-deps','<frozen-wheel-paths>'],
        origins=[py,'-I',str(root/'runner.py'),'origins','--evidence',str(root)],
        pip_check=[sys.executable,'-m','pip','--python',py,'check'],
        collect=[*base,'--collect-only',*selections],
        tests=[*base,*selections,'--junitxml='+str(root/'pytest.xml')],
        example=[py,'-I',str(root/'run/examples/solid_volume_controlled.py'),'--output',str(root/'example.json')])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare','freeze','verify','origins','amend-native-operations','pytest-child','guard-controls'))
    p.add_argument('--evidence', type=Path, required=True)
    args, extra = p.parse_known_args()
    root = args.evidence.resolve()
    if args.phase == 'pytest-child':
        import pytest
        selected = extra[1:] if extra[:1] == ['--'] else extra
        raise SystemExit(pytest.main(selected, plugins=[Inventory(root, '--collect-only' in selected)]))
    if extra:
        p.error('unexpected arguments')
    if args.phase == 'origins':
        return origins(root)
    if args.phase == 'freeze':
        return freeze(root)
    assert os.name == 'nt' and sys.version_info[:2] == (3,11)
    ledger_path = root/'budget.json'
    if args.phase == 'prepare':
        assert not root.exists(), 'fresh evidence required'
        root.mkdir(parents=True)
        (root/'processes').mkdir()
        (root/'tmp').mkdir()
        ledger = dict(limit_seconds=1200, reserve_seconds=120, used_seconds=0., remaining_seconds=1200., attempts=[])
    else:
        ledger = json.loads(ledger_path.read_text())
    supervisor = load_module(Path.home()/'bin/claude-review.py','w2_windows_supervisor')

    def check(name, command, cwd=root, maximum=300):
        assert not any(r['name']==name and r['status']=='completed' for r in ledger['attempts']), 'passed step replay refused'
        assert sum(r['name'].split('-attempt')[0]==name for r in ledger['attempts']) < 3
        timeout = min(maximum, int(ledger['remaining_seconds']-120))
        assert timeout > 0, 'budget reserve reached'
        actual = name
        attempt = 1
        while (root/'processes'/actual).exists():
            attempt += 1
            actual = name+'-attempt'+str(attempt)
        folder = root/'processes'/actual
        folder.mkdir()
        save(folder/'command.json',dict(command=command,cwd=str(cwd),timeout_seconds=timeout,
             environment={k:environment(root)[k] for k in (*FLAGS,'PYTHONPATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
             harness_sha256=sha(__file__)))
        receipt,_ = supervisor.run_process(command,folder,'stdout',timeout,environment(root),cwd)
        ledger['used_seconds'] += receipt['process_wall_seconds']
        ledger['remaining_seconds'] = 1200-ledger['used_seconds']
        ledger['attempts'].append(dict(name=actual,**receipt))
        save(ledger_path,ledger)
        print(json.dumps(dict(name=actual,status=receipt['status'],seconds=receipt['process_wall_seconds'])),flush=True)
        assert receipt['status']=='completed' and receipt['cleanup']['verified'], actual

    if args.phase == 'prepare':
        check('freeze',[sys.executable,'-I',str(Path(__file__).resolve()),'freeze','--evidence',str(root)])
    if args.phase == 'amend-native-operations':
        assert not (root/'root-release.json').exists(), 'amend before release only'
        assert not any(r['name']=='tests' for r in ledger['attempts']), 'no post-test amendments'
        revision = 1
        while (root/f'amendment-{revision}.json').exists():
            revision += 1
        original = root/('initial-freeze' if revision == 1 else f'pre-amendment-{revision}')
        original.mkdir()
        for name in ('manifest.json','runner.py','ready.json','collection-inventory.json'):
            if (root/name).exists():
                shutil.copy2(root/name,original/name)
        target = root/'run/tests/test_workflow2_native_operations.py'
        if target.exists():
            shutil.copy2(target,original/target.name)
        shutil.copy2(REPO/'tests/test_workflow2_native_operations.py',target)
        shutil.copy2(__file__,root/'runner.py')
        amended = json.loads((root/'manifest.json').read_text())
        amended['input_hashes'][str(target.relative_to(root))] = sha(target)
        if 'tests/test_workflow2_native_operations.py' not in amended['selections']:
            amended['selections'].append('tests/test_workflow2_native_operations.py')
        amended['commands'] = commands(root, amended['selections'])
        amended['harness_sha256'] = sha(root/'runner.py')
        (root/'guard-tests').mkdir(exist_ok=True)
        guard = root/'guard-tests/test_workflow2_windows_native_guard.py'
        shutil.copy2(REPO/'tests/test_workflow2_windows_native_guard.py',guard)
        amended['input_hashes'][str(guard.relative_to(root))] = sha(guard)
        amended['effective_environment'] = {k:v for k,v in environment(root).items()
             if k.startswith(('ANY','PIP_','PYTHON','PYTEST')) or k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TEMP','TMP')}
        save(root/'manifest.json',amended)
        save(root/f'amendment-{revision}.json',dict(reason='actual native operations; opt-in isolation from pure/default suites',
             state='prospective; initial collection only, no contract execution',
             previous_manifest_sha256=sha(original/'manifest.json'),
             manifest_sha256=sha(root/'manifest.json'),new_test_sha256=sha(target),
             previous_harness_sha256=sha(original/'runner.py'),harness_sha256=sha(root/'runner.py')))
        check(f'collect-amendment-{revision}',amended['commands']['collect'],root/'run')
        inventory = json.loads((root/'collection-inventory.json').read_text())
        assert inventory['completed'] and inventory['exitstatus'] == 0
        assert len(inventory['selected']) == 118 and not inventory['deselected']
        amended['expected_nodeids'] = inventory['selected']
        amended['collection_inventory_sha256'] = sha(root/'collection-inventory.json')
        save(root/'manifest.json', amended)
        amendment_path = root/f'amendment-{revision}.json'
        amendment = json.loads(amendment_path.read_text())
        amendment['manifest_sha256'] = sha(root/'manifest.json')
        amendment['reason'] = 'prospective exact-inventory guard, sanitized pytest environment and native operation witnesses'
        save(amendment_path, amendment)
        save(root/'ready.json',dict(state='amended installed profile collected; root release required',
             manifest_sha256=sha(root/'manifest.json'),wheel_hashes=json.loads((root/'wheel-hashes.json').read_text()),budget=ledger))
        return
    manifest = json.loads((root/'manifest.json').read_text())
    for name,expected in manifest['input_hashes'].items():
        assert sha(root/name)==expected, name
    assert sha(root/'runner.py')==manifest['harness_sha256']
    assert sha(__file__)==manifest['harness_sha256'], 'executing driver identity mismatch'
    assert sha(Path.home()/'bin/claude-review.py')==manifest['supervisor_sha256'], 'supervisor identity mismatch'
    cmd = manifest['commands']
    if args.phase == 'guard-controls':
        check('guard-controls',[str(root/'venv/Scripts/python.exe'),'-I','-m','pytest','-q',
            '-o','pythonpath=','-p','no:cacheprovider','--confcutdir='+str(root/'guard-tests'),
            str(root/'guard-tests/test_workflow2_windows_native_guard.py'),
            '--junitxml='+str(root/'guard-controls.xml')],root)
        return
    if args.phase == 'prepare':
        for name in ('build_geometry','build_mesh','create_venv','dependencies'):
            check(name,cmd[name])
        wheels = sorted((root/'wheels').glob('*.whl'))
        assert len(wheels)==3
        assert any('cp311-abi3-win_amd64' in w.name for w in wheels)
        assert any('cp311-cp311-win_amd64' in w.name for w in wheels)
        save(root/'wheel-hashes.json',{w.name:sha(w) for w in wheels})
        install = cmd['install'][:-1]+list(map(str,wheels))
        save(root/'install-command.json',install)
        check('install',install)
        for name in ('origins','pip_check','collect'):
            check(name,cmd[name],root/'run')
        save(root/'ready.json',dict(state='installed and collected; root release required',
            manifest_sha256=sha(root/'manifest.json'), wheel_hashes=json.loads((root/'wheel-hashes.json').read_text()),budget=ledger))
    else:
        release = json.loads((root/'root-release.json').read_text())
        assert release['manifest_sha256']==sha(root/'manifest.json') and release['approved'] is True
        check('tests',cmd['tests'],root/'run')
        validation = validate_outcomes(manifest['expected_nodeids'],
             json.loads((root/'test-inventory.json').read_text()),root/'pytest.xml')
        save(root/'validated-outcomes.json',validation)
        check('example',cmd['example'],root/'run')
        save(root/'handoff.json',dict(state='verified against assigned Windows native W2 criteria',
            review_status='root acceptance and non-author review pending',
            native_provider_calls=0, wheel_hashes=json.loads((root/'wheel-hashes.json').read_text()),
            manifest_sha256=sha(root/'manifest.json'), budget=ledger,
            counts=validation,
            omissions=['real native-provider meshing','large PDE','broad shell suite','remote CI','publication']))


if __name__ == '__main__':
    main()
