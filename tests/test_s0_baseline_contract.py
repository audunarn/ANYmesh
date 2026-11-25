"""Packet tests use synthetic bytes; real owner coverage remains in the six S0 cases."""
import copy
import hashlib
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

import _s0_baseline_contract as contract


def packet(tmp_path):
    raw = struct.pack('=qddd', 1, 0., 0., 0.) + b'quads' + b'tris' + struct.pack('=qqqq', 2, 1, 1, 1)
    rows = []
    for name in contract.CASES:
        (tmp_path / (name + '.bin')).write_bytes(raw)
        rows.append(dict(case=name, nodes=1, quads=0, tris=1, bytes=len(raw), sha256=contract.hash_bytes(raw)))
    value = dict(schema=1, refs=contract.REFS, order=list(contract.CASES),
                 historical_golden=contract.HISTORICAL, identity={'runtime': 'unit'}, fixtures={'fixture': 'unit'},
                 source_unchanged=True, completed=True, cases=rows,
                 terminal=dict(status='completed', exit_code=0, cleanup_verified=True, remaining_owned=[], errors=[]))
    return raw, value


def save(tmp_path, value):
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(value), encoding='utf8')
    return path, contract.hash_bytes(path.read_bytes())


def test_packet_exact_raw_and_counts(tmp_path):
    raw, value = packet(tmp_path)
    path, seal = save(tmp_path, value)
    result = contract.load_packet(path, seal, identity=value['identity'], fixtures=value['fixtures'])
    assert result['planar'] == raw


@pytest.mark.parametrize('field,replacement', [
    ('schema', 2), ('refs', {}), ('order', list(reversed(contract.CASES))),
    ('historical_golden', {}), ('identity', {}), ('fixtures', {}),
    ('source_unchanged', False), ('completed', False),
])
def test_forged_binding_refused(tmp_path, field, replacement):
    _, value = packet(tmp_path)
    expected = copy.deepcopy(value)
    value[field] = replacement
    path, seal = save(tmp_path, value)
    with pytest.raises(AssertionError):
        contract.load_packet(path, seal, identity=expected['identity'], fixtures=expected['fixtures'])


@pytest.mark.parametrize('field,replacement', [('status', 'timeout'), ('exit_code', 1),
    ('cleanup_verified', False), ('remaining_owned', [123]), ('errors', ['uncertain'])])
def test_incomplete_owned_child_never_publishes_success(tmp_path, field, replacement):
    _, value = packet(tmp_path)
    value['terminal'][field] = replacement
    path, seal = save(tmp_path, value)
    with pytest.raises(AssertionError):
        contract.load_packet(path, seal, identity=value['identity'], fixtures=value['fixtures'])


def test_seal_raw_corruption_count_and_traversal_fail_closed(tmp_path):
    _, value = packet(tmp_path)
    path, seal = save(tmp_path, value)
    with pytest.raises(AssertionError, match='manifest SHA'):
        contract.load_packet(path, '0' * 64, identity=value['identity'], fixtures=value['fixtures'])
    (tmp_path / 'planar.bin').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):
        contract.load_packet(path, seal, identity=value['identity'], fixtures=value['fixtures'])
    raw, value = packet(tmp_path)
    value['cases'][0]['nodes'] += 1
    path, seal = save(tmp_path, value)
    with pytest.raises(AssertionError):
        contract.load_packet(path, seal, identity=value['identity'], fixtures=value['fixtures'])


def test_raw_format_rejects_nan_unknown_node_duplicate_and_oversize():
    row = dict(nodes=1, quads=0, tris=1)
    prefix = struct.pack('=qddd', 1, 0., 0., 0.) + b'quads' + b'tris'
    contract.validate_raw(prefix + struct.pack('=qqqq', 2, 1, 1, 1), row)
    with pytest.raises(AssertionError):
        contract.validate_raw(prefix + struct.pack('=qqqq', 2, 9, 1, 1), row)
    bad = struct.pack('=qddd', 1, float('nan'), 0., 0.) + prefix[32:] + struct.pack('=qqqq', 2, 1, 1, 1)
    with pytest.raises(AssertionError):
        contract.validate_raw(bad, row)
    with pytest.raises(AssertionError):
        contract.validate_raw(b'0' * (contract.MAX_BYTES + 1), row)


def test_required_or_partial_environment_never_uses_local_golden(monkeypatch):
    monkeypatch.delenv(contract.UNAVAILABLE_ENV, raising=False)
    monkeypatch.delenv(contract.PATH_ENV, raising=False)
    monkeypatch.delenv(contract.SHA_ENV, raising=False)
    monkeypatch.setenv(contract.REQUIRE_ENV, '1')
    with pytest.raises(AssertionError, match='required'):
        contract.baseline_environment()
    monkeypatch.delenv(contract.REQUIRE_ENV)
    assert contract.baseline_environment() is None
    monkeypatch.setenv(contract.PATH_ENV, '/not/real')
    with pytest.raises(AssertionError):
        contract.baseline_environment()


def test_historical_constants_and_builders_remain_unchanged():
    import ast
    source = Path(__file__).with_name('test_qf_nurbs_s0_characterisation.py').read_text()
    tree = ast.parse(source)
    value = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'GOLDEN' for t in n.targets))
    assert value == contract.HISTORICAL
    contract.validate_fixture_contract(Path(__file__).parent)


def test_exact_comparison_rejects_same_counts_different_bytes(tmp_path):
    raw, value = packet(tmp_path)
    path, seal = save(tmp_path, value)
    rows = contract.load_packet(path, seal, identity=value['identity'], fixtures=value['fixtures'])
    changed = bytearray(raw)
    changed[9] ^= 1
    assert len(changed) == len(raw)
    assert rows['planar'] != bytes(changed)


def producer():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'tools' / 'ci_s0_baseline.py'
    spec = importlib.util.spec_from_file_location('s0_producer_unit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_patch_is_explicit_and_no_other_python_drift_allowed(tmp_path):
    module = producer()
    source = tmp_path / 'source' / 'src' / 'anymesher'
    actual = tmp_path / 'actual'
    source.mkdir(parents=True)
    actual.mkdir()
    original = '"""owner"""\nfrom __future__ import annotations\n\nx = 1\n'
    (source / '__init__.py').write_text(original)
    (actual / '__init__.py').write_text(original.replace('annotations\n', 'annotations\n' + module.BOOTSTRAP))
    result = module.loader_payload(actual, tmp_path / 'source')
    assert result['bootstrap'] == module.BOOTSTRAP
    before = contract.file_hash(source / '__init__.py')
    adaptation = module.add_loader(source / '__init__.py', result['bootstrap'])
    assert adaptation['original_sha256'] == before
    assert adaptation['overlay_sha256'] != before
    assert 'non-Git' in adaptation['scope']
    (actual / '__init__.py').write_text(original + '\nx = 2\n')
    with pytest.raises(AssertionError, match='loader'):
        module.loader_payload(actual, tmp_path / 'source')


def test_ci_runs_and_reports_the_cold_baseline_in_both_full_profiles():
    # Owner decision 2026-11-25: S0 runs in both full profiles and its result
    # is reported, but a failed S0 (including its frozen checkouts and its
    # evidence upload) no longer stops the test legs.  The producer's binding
    # reaches pytest only through tools/ci_s0_report.py after success.
    source = (Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text()
    for name, following in (('pytest:', 'native-absent:'), ('wheel-full-linux:', 'qualified-caller-planning:')):
        job = source.split('\n  ' + name, 1)[1].split('\n  ' + following, 1)[0]
        assert 'ANYMESHER_REQUIRE_S0_BASELINE' not in job  # only ci_s0_report.py exports it
        for sid in ('s0_old_geometry', 's0_old_mesher'):
            block = job.split('id: ' + sid, 1)[1].split('\n      - ', 1)[0]
            assert 'continue-on-error: true' in block
        step = job.split('- name: Produce one cold frozen-source six-face S0 baseline', 1)[1].split('\n      - ', 1)[0]
        assert 'id: s0' in step and 'continue-on-error: true' in step
        assert "steps.s0_old_geometry.outcome == 'success'" in step and "steps.s0_old_mesher.outcome == 'success'" in step
        assert 'tools/ci_s0_baseline.py' in step and '--github-env s0-binding.env' in step
        assert 'GITHUB_ENV' not in step  # the producer never writes the job environment
        report = job.split('- name: Report the S0 result', 1)[1].split('\n      - ', 1)[0]
        assert 'if: always()' in report and 'continue-on-error' not in report
        assert '${{ steps.s0.outcome }}' in report
        assert 'tools/ci_s0_report.py' in report and '--binding s0-binding.env' in report
        token = step.split('ANYMESHER_S0_OWNER: ', 1)[1].split('\n', 1)[0]
        assert 'github.run_id' in token and '--owner "' + token + '"' in report
        assert '--github-env "$GITHUB_ENV"' in report
        upload = job.split('- name: Retain cold S0 identity and partial process evidence', 1)[1].split('\n      - ', 1)[0]
        assert 'continue-on-error: true' in upload
        assert job.index('Report the S0 result') < job.index('-m pytest')
        assert contract.REFS['geometry'] in job and contract.REFS['mesher'] in job
    assert 'python -I -B tools/ci_s0_baseline.py' in source


def _report(tmp_path, outcome, binding_lines, owner=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'ci_s0_report', Path(__file__).resolve().parents[1] / 'tools' / 'ci_s0_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binding = tmp_path / 's0-binding.env'
    if binding_lines is not None:
        binding.write_text(''.join(line + '\n' for line in binding_lines), encoding='utf8')
    env, summary = tmp_path / 'github.env', tmp_path / 'summary.md'
    output = tmp_path / 'ci-s0-baseline'
    output.mkdir(exist_ok=True)
    code = module.report(outcome, binding, output, env, summary, owner)
    exported = dict(line.split('=', 1) for line in env.read_text(encoding='utf8').splitlines())
    return code, exported, summary.read_text(encoding='utf8')


def _pair(tmp_path, body=b'{"schema": 1}'):
    output = tmp_path / 'ci-s0-baseline'
    output.mkdir(exist_ok=True)
    manifest = output / 'manifest.json'
    manifest.write_bytes(body)
    return [contract.PATH_ENV + '=' + str(manifest), contract.SHA_ENV + '=' + hashlib.sha256(body).hexdigest()]


@pytest.mark.parametrize('outcome', ['success', 'failure', 'skipped', 'cancelled'])
def test_report_exports_the_binding_only_after_success(tmp_path, outcome):
    pair = _pair(tmp_path)
    code, exported, summary = _report(tmp_path, outcome, pair)
    assert code == 0 and outcome in summary
    if outcome == 'success':
        assert exported == {contract.PATH_ENV: pair[0].split('=', 1)[1],
                            contract.SHA_ENV: pair[1].split('=', 1)[1], contract.REQUIRE_ENV: '1'}
    else:
        # A binding written by a producer that then failed never reaches pytest.
        assert exported == {contract.UNAVAILABLE_ENV: outcome}


@pytest.mark.parametrize('mutation', ['missing', 'path_only', 'extra_row', 'duplicate', 'bad_sha',
                                      'upper_sha', 'wrong_sha', 'outside', 'not_a_file', 'invalid_utf8',
                                      'other_file', 'loop', 'alias'])
def test_report_refuses_a_missing_or_malformed_binding_after_success(tmp_path, mutation):
    pair = _pair(tmp_path)
    lines = {
        'missing': None, 'path_only': pair[:1], 'extra_row': pair + [contract.REQUIRE_ENV + '=1'],
        'duplicate': [pair[0], pair[0]], 'bad_sha': [pair[0], contract.SHA_ENV + '=bad-sha'],
        'upper_sha': [pair[0], pair[1].upper().replace(contract.SHA_ENV.upper(), contract.SHA_ENV)],
        'wrong_sha': [pair[0], contract.SHA_ENV + '=' + '0' * 64],
        'outside': [contract.PATH_ENV + '=' + str(tmp_path / 'manifest.json'), pair[1]],
        'not_a_file': [contract.PATH_ENV + '=' + str(tmp_path / 'ci-s0-baseline'), pair[1]],
        'invalid_utf8': pair,
        'other_file': None, 'loop': None, 'alias': None,
    }[mutation]
    if mutation == 'alias':
        # A correctly sealed symlink alias of the manifest in another directory.
        alias = tmp_path / 'elsewhere.json'
        try:
            os.symlink(tmp_path / 'ci-s0-baseline' / 'manifest.json', alias)
        except (OSError, NotImplementedError):
            pytest.skip('symlinks unavailable')
        lines = [contract.PATH_ENV + '=' + str(alias), pair[1]]
    if mutation == 'other_file':
        # A correctly sealed file that is not the manifest.
        other = tmp_path / 'ci-s0-baseline' / 'provenance.json'
        other.write_bytes(b'{}')
        lines = [contract.PATH_ENV + '=' + str(other), contract.SHA_ENV + '=' + hashlib.sha256(b'{}').hexdigest()]
    if mutation == 'loop':
        loop = tmp_path / 'ci-s0-baseline' / 'loop'
        try:
            os.symlink(loop, loop)
        except (OSError, NotImplementedError):
            pytest.skip('symlinks unavailable')
        lines = [contract.PATH_ENV + '=' + str(loop), pair[1]]
    if mutation == 'outside':
        (tmp_path / 'manifest.json').write_bytes(b'{"schema": 1}')
    code, exported, _ = _report(tmp_path, 'success', lines) if mutation != 'invalid_utf8' else (None, None, None)
    if mutation == 'invalid_utf8':
        (tmp_path / 's0-binding.env').write_bytes(b'\xff\xfe' + b'not utf8')
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'ci_s0_report', Path(__file__).resolve().parents[1] / 'tools' / 'ci_s0_report.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        env = tmp_path / 'github.env'
        code = module.report('success', tmp_path / 's0-binding.env', tmp_path / 'ci-s0-baseline', env, None)
        exported = dict(line.split('=', 1) for line in env.read_text(encoding='utf8').splitlines())
    assert code == 0 and exported == {contract.UNAVAILABLE_ENV: 'success-without-binding'}


def test_report_summary_write_error_never_blocks(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'ci_s0_report', Path(__file__).resolve().parents[1] / 'tools' / 'ci_s0_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = tmp_path / 'github.env'
    (tmp_path / 'ci-s0-baseline').mkdir()
    code = module.report('failure', tmp_path / 'none.env', tmp_path / 'ci-s0-baseline', env,
                         tmp_path / 'missing-dir' / 'summary.md')
    assert code == 0
    assert env.read_text(encoding='utf8') == contract.UNAVAILABLE_ENV + '=failure\n'


def test_unavailable_flag_never_admits_a_partial_binding(monkeypatch):
    monkeypatch.setenv(contract.UNAVAILABLE_ENV, 'failure')
    monkeypatch.delenv(contract.REQUIRE_ENV, raising=False)
    monkeypatch.setenv(contract.PATH_ENV, '/not/real')
    monkeypatch.delenv(contract.SHA_ENV, raising=False)
    with pytest.raises(AssertionError, match='partial'):
        contract.baseline_environment()


def test_supervisor_never_enumerates_unrelated_processes():
    import inspect
    module = producer()
    source = inspect.getsource(module.supervise)
    assert 'process_iter(' not in source and 'killpg(' not in source
    assert 'kill_owned(' in source


@pytest.mark.parametrize('reason', ['compiled module not installed', 'disabled by ANYGEOMETRY_DISABLE_NATIVE'])
def test_python_geometry_backend_records_reason_without_comparing_load_cause(reason):
    report = contract.checked_geometry_backend(dict(backend='python', module=None, build=None, reason=reason))
    assert report == dict(effective=dict(backend='python', module=None, build=None), reason=reason)


@pytest.mark.parametrize('change', [dict(backend='native'), dict(module='/unexpected/core.so'),
                                  dict(build={'compiler': 'unexpected'})])
def test_geometry_native_or_unbound_backend_is_not_a_portable_baseline(change):
    report = dict(backend='python', module=None, build=None, reason='unit')
    report.update(change)
    with pytest.raises(AssertionError, match='actual Python geometry backend'):
        contract.checked_geometry_backend(report)


def test_assert_output_consumes_exact_packet_and_rejects_changed_warm_bytes(tmp_path, monkeypatch):
    # Synthetic packet/identity boundary only: no owner/numerical function mocked.
    _, value = packet(tmp_path)
    path, seal = save(tmp_path, value)
    monkeypatch.setenv(contract.PATH_ENV, str(path))
    monkeypatch.setenv(contract.SHA_ENV, seal)
    monkeypatch.setenv(contract.REQUIRE_ENV, '1')
    for key in (*contract.THREADS, 'ANYMESH_Q4_DISABLE_WORKER', 'ANYMESH_Q5_DISABLE_WORKER'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setattr(contract, 'current_identity', lambda: value['identity'])
    monkeypatch.setattr(contract, 'fixture_identity', lambda directory: value['fixtures'])
    mesh = SimpleNamespace(nodes={1: (0., 0., 0.)}, quads={}, tris={2: (1, 1, 1)})
    digest = lambda item: contract.hash_bytes(contract.encode_mesh(item)[0])
    contract.assert_output('planar', mesh, contract.HISTORICAL, digest=digest)
    mesh.nodes[1] = (0., 0., 1.)
    with pytest.raises(AssertionError, match='current warm S0 output differs'):
        contract.assert_output('planar', mesh, contract.HISTORICAL, digest=digest)
    monkeypatch.delenv(contract.PATH_ENV)
    monkeypatch.delenv(contract.SHA_ENV)
    with pytest.raises(AssertionError, match='required S0 cold baseline'):
        contract.assert_output('planar', mesh, contract.HISTORICAL, digest=digest)


def test_unavailable_baseline_skips_with_the_reported_reason(monkeypatch):
    monkeypatch.delenv(contract.PATH_ENV, raising=False)
    monkeypatch.delenv(contract.SHA_ENV, raising=False)
    monkeypatch.delenv(contract.REQUIRE_ENV, raising=False)
    monkeypatch.setenv(contract.UNAVAILABLE_ENV, 'failure')
    with pytest.raises(pytest.skip.Exception, match='unavailable.*failure'):
        contract.baseline_environment()
    # Never both: a required baseline that is unavailable still fails.
    monkeypatch.setenv(contract.REQUIRE_ENV, '1')
    with pytest.raises(AssertionError, match='required'):
        contract.baseline_environment()


def test_report_kills_a_stray_s0_process_before_pytest(tmp_path):
    psutil = pytest.importorskip('psutil')
    import subprocess
    import sys
    output = tmp_path / 'ci-s0-baseline'
    output.mkdir()
    # A stand-in for a surviving cold child: its command line names the output directory.
    stray = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)', str(output)])
    try:
        code, exported, _ = _report(tmp_path, 'failure', None)
        assert code == 0 and exported == {contract.UNAVAILABLE_ENV: 'failure'}
        assert stray.wait(timeout=30) is not None
        assert not psutil.pid_exists(stray.pid) or psutil.Process(stray.pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if stray.poll() is None:
            stray.kill()


def test_report_kills_an_owned_descendant_that_outlived_its_parent(tmp_path, monkeypatch):
    psutil = pytest.importorskip('psutil')
    import subprocess
    import sys
    import time
    owner = 's0-test-owner-%d' % os.getpid()
    pid_file = tmp_path / 'grandchild.pid'
    # The parent starts a grandchild with no S0 marker on its command line and
    # exits at once; only the inherited owner token ties it to S0.
    launcher = ('import subprocess, sys; '
                'p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]); '
                'open(sys.argv[1], "w").write(str(p.pid))')
    env = dict(os.environ, ANYMESHER_S0_OWNER=owner)
    subprocess.run([sys.executable, '-c', launcher, str(pid_file)], env=env, check=True, timeout=60)
    grandchild = int(pid_file.read_text())
    try:
        assert psutil.pid_exists(grandchild)
        code, exported, _ = _report(tmp_path, 'failure', None, owner=owner)
        assert code == 0 and exported == {contract.UNAVAILABLE_ENV: 'failure'}
        deadline = time.time() + 30
        while psutil.pid_exists(grandchild) and time.time() < deadline:
            try:
                if psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE:
                    break
            except psutil.Error:
                break
            time.sleep(.2)
        assert not psutil.pid_exists(grandchild) or psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE
    finally:
        try:
            psutil.Process(grandchild).kill()
        except psutil.Error:
            pass


def test_containment_rescans_until_no_owned_process_is_left(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'ci_s0_report', Path(__file__).resolve().parents[1] / 'tools' / 'ci_s0_report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    psutil = pytest.importorskip('psutil')

    class Fake:
        def __init__(self, pid):
            self.pid = pid

        def kill(self):
            pass

    import time
    # The first scan sees one process; killing it reveals a child started
    # meanwhile; one empty scan is not enough, a child appears in the next.
    scans = [[Fake(1)], [Fake(2)], [], [Fake(3)], [], []]
    monkeypatch.setattr(module, 'stray_processes', lambda output, owner: (psutil, scans.pop(0)))
    monkeypatch.setattr(psutil, 'wait_procs', lambda processes, timeout: (processes, []))
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)
    assert module.contain(tmp_path) == []
    assert scans == []
