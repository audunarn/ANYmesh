"""Packet tests use synthetic bytes; real owner coverage remains in the six S0 cases."""
import copy
import hashlib
import json
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


def test_ci_requires_cold_baseline_in_both_full_profiles():
    source = (Path(__file__).resolve().parents[1] / '.github/workflows/ci.yml').read_text()
    for name, following in (('pytest:', 'native-absent:'), ('wheel-full-linux:', 'qualified-caller-planning:')):
        job = source.split('\n  ' + name, 1)[1].split('\n  ' + following, 1)[0]
        assert 'ANYMESHER_REQUIRE_S0_BASELINE: "1"' in job
        assert 'tools/ci_s0_baseline.py' in job
        assert contract.REFS['geometry'] in job and contract.REFS['mesher'] in job
        assert '--github-env' in job
        assert 'if: always()' in job
    assert 'python -I -B tools/ci_s0_baseline.py' in source


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
