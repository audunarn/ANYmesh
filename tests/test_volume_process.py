"""Bounded controlled subprocess tests; no real tetrahedralizer invocation."""
from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import pytest

from anymesher._volume_process import _launch_worker, run_provider
from anymesher._volume_validation import validate_source
from anymesher.volume import (InvalidVolumeMesh, VolumeBackendUnavailable,
                             VolumeMeshingCancelled, VolumeMeshingTimeout,
                             generate_volume_mesh, validate_volume_mesh)
from test_volume_mesh import box


@pytest.fixture
def controlled_provider_capability(monkeypatch):
    # Call only after this test has installed its fake launcher/distribution.
    # This is not an environment switch or a supported public research bypass.
    return lambda: monkeypatch.setattr('anymesher._volume_process._PROVIDER_QUALIFIED', True)


def worker(tmp_path, body):
    script = tmp_path / 'controlled.py'
    script.write_text("import sys\nassert sys.stdin.readline().strip() == 'start'\n" + body, encoding='utf-8')
    return [sys.executable, str(script)]


def test_success_and_nonzero_exit_preserve_full_logs(tmp_path):
    command = worker(tmp_path, "print('controlled stdout')\nprint('controlled stderr', file=sys.stderr)\n")
    result = _launch_worker(command, tmp_path, 5, None)
    assert result['status'] == 'success'
    assert result['exit_code'] == 0
    assert result['process_wall_seconds'] > 0
    assert result['cleanup'] == 'owned_tree_terminated_and_parent_reaped'
    assert 'controlled stdout' in (tmp_path / 'stdout.txt').read_text()
    assert 'controlled stderr' in (tmp_path / 'stderr.txt').read_text()
    result = _launch_worker(worker(tmp_path, 'raise SystemExit(7)\n'), tmp_path, 5, None)
    assert result['exit_code'] == 7
    assert result['status'] == 'worker_failed'


def test_timeout_is_bounded_and_preserves_partial_output(tmp_path):
    command = worker(tmp_path, "import time\nprint('before timeout', flush=True)\ntime.sleep(20)\n")
    with pytest.raises(VolumeMeshingTimeout) as error:
        _launch_worker(command, tmp_path, .3, None)
    receipt = json.loads((tmp_path / 'process.json').read_text())
    assert receipt['status'] == 'timeout'
    assert .3 <= receipt['process_wall_seconds'] < 5
    assert receipt['exit_code'] is not None
    assert error.value.evidence_directory == str(tmp_path)
    assert 'before timeout' in (tmp_path / 'stdout.txt').read_text()


def test_cancel_is_bounded_and_reaps_process(tmp_path):
    command = worker(tmp_path, 'import time\ntime.sleep(20)\n')
    start = time.perf_counter()
    with pytest.raises(VolumeMeshingCancelled):
        _launch_worker(command, tmp_path, 5, lambda: time.perf_counter() - start > .2)
    receipt = json.loads((tmp_path / 'process.json').read_text())
    assert receipt['status'] == 'cancelled'
    assert receipt['cleanup'] == 'owned_tree_terminated_and_parent_reaped'
    assert receipt['process_wall_seconds'] < 5


def _process_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(handle, 0) == 0x102
        finally:
            kernel.CloseHandle(handle)
    try:
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists() and stat.read_text().rsplit(')', 1)[1].split()[0] in ('Z', 'X'):
            return False
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_worker_orphan_is_terminated_after_parent_success(tmp_path):
    body = """import subprocess, pathlib, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])
pathlib.Path('child.pid').write_text(str(child.pid))
"""
    result = _launch_worker(worker(tmp_path, body), tmp_path, 5, None)
    assert result['status'] == 'success'
    pid = int((tmp_path / 'child.pid').read_text())
    deadline = time.perf_counter() + 2
    while _process_alive(pid) and time.perf_counter() < deadline:
        time.sleep(.01)
    assert not _process_alive(pid)


@pytest.mark.parametrize('mode', ['unavailable', 'missing', 'partial', 'wrong_options', 'wrong_version', 'bad_output'])
def test_provider_protocol_failure_is_atomic(monkeypatch, mode, controlled_provider_capability):
    snapshot, cells = box()
    source = validate_source(snapshot, 1e-7)
    def fake(command, directory, timeout, cancel):
        request = json.loads((directory / 'request.json').read_text())
        response = dict(status='success', version='0.4.2', effective_options=request['options'])
        if mode == 'unavailable': response = dict(status='unavailable', detail='ModuleNotFoundError: pyvista')
        if mode == 'missing': return dict(exit_code=1)
        if mode == 'wrong_options': response['effective_options']['num_threads'] = 2
        if mode == 'wrong_version': response['version'] = '0.4.3'
        if mode == 'bad_output': (directory / 'output.npz').write_bytes(b'not an npz')
        (directory / 'response.json').write_text(json.dumps(response))
        return dict(exit_code=0)
    monkeypatch.setattr('anymesher._volume_process._launch_worker', fake)
    controlled_provider_capability()
    exc = VolumeBackendUnavailable if mode in ('unavailable', 'missing') else InvalidVolumeMesh
    with pytest.raises(exc):
        run_provider(source, .25, 1, None)


def test_pinned_effective_options_and_local_patch_identity(monkeypatch, controlled_provider_capability):
    snapshot, cells = box()
    source = validate_source(snapshot, 1e-7)
    def fake(command, directory, timeout, cancel):
        request = json.loads((directory / 'request.json').read_text())
        options = request['options']
        assert options['epsilon'] == pytest.approx(1e-7 / np.sqrt(3))
        assert options['edge_length_abs'] == .25
        assert options['simplify'] is False and options['coarsen'] is False
        assert options['optimize'] is True and options['num_threads'] == 1
        np.savez(directory / 'output.npz', coordinates=snapshot.coordinates, cells=cells)
        (directory / 'response.json').write_text(json.dumps(dict(status='success', version='0.4.2+anyworkflow2.1', effective_options=options)))
        return dict(exit_code=0, process_wall_seconds=.1)
    monkeypatch.setattr('anymesher._volume_process._launch_worker', fake)
    controlled_provider_capability()
    x, t, metadata = run_provider(source, .25, 1, None)
    assert np.array_equal(x, snapshot.coordinates) and np.array_equal(t, cells)
    assert metadata['provider_version'] == '0.4.2+anyworkflow2.1'
    assert len(metadata['input_sha256']) == 64


def test_public_generation_through_controlled_provider_subprocess(monkeypatch, tmp_path, controlled_provider_capability):
    # An explicitly controlled import distribution inside this test process's
    # child environment. This never imports or executes the real provider.
    # It tests the production file protocol, options, job ownership, raw array
    # transfer, canonicalization and admission together.
    distribution = tmp_path / 'pytetwild-0.4.2.dist-info'
    distribution.mkdir()
    (distribution / 'METADATA').write_text('Metadata-Version: 2.1\nName: pytetwild\nVersion: 0.4.2\n')
    (tmp_path / 'pytetwild.py').write_text('''import numpy as np
def tetrahedralize(coordinates, facets, **options):
    assert options['simplify'] is False
    assert options['coarsen'] is False
    assert options['optimize'] is True
    assert options['num_threads'] == 1
    assert options['edge_length_abs'] == .25
    assert len(facets) == 12
    return coordinates.copy(), np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],
                                        [0,7,4,6],[0,4,5,6],[0,5,1,6]])
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    controlled_provider_capability()
    snapshot, _ = box()
    mesh = generate_volume_mesh(snapshot, target_edge_length=.25, boundary_tolerance=1e-7, timeout=10)
    assert validate_volume_mesh(mesh, snapshot, boundary_tolerance=1e-7)['volume'] == pytest.approx(1.)
    assert mesh.provenance['provider_version'] == '0.4.2'
    assert mesh.provenance['process']['exit_code'] == 0
    assert mesh.provenance['process']['cleanup'] == 'owned_tree_terminated_and_parent_reaped'


def test_worker_import_failure_is_unavailable_without_real_provider(monkeypatch, tmp_path, controlled_provider_capability):
    distribution = tmp_path / 'pytetwild-0.4.2.dist-info'
    distribution.mkdir()
    (distribution / 'METADATA').write_text('Metadata-Version: 2.1\nName: pytetwild\nVersion: 0.4.2\n')
    (tmp_path / 'pytetwild.py').write_text("raise ModuleNotFoundError('controlled missing optional dependency')\n")
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    controlled_provider_capability()
    snapshot, _ = box()
    with pytest.raises(VolumeBackendUnavailable, match='controlled missing optional dependency') as error:
        generate_volume_mesh(snapshot, target_edge_length=.25, boundary_tolerance=1e-7, timeout=10)
    evidence = Path(error.value.evidence_directory)
    assert (evidence / 'request.json').exists()
    assert 'ModuleNotFoundError' in (evidence / 'stderr.txt').read_text()
    assert not (evidence / 'output.npz').exists()


def test_volume_subprocess_worker_has_explicit_affected_test_mapping():
    from tools.run_affected_tests import select_tests
    selectors, reasons = select_tests(['src/anymesher/_volume_worker.py'])
    assert 'tests/test_volume_process.py' in selectors
    assert 'tests/test_volume_mesh.py' in selectors
    assert 'tests/test_volume_process.py' in reasons


def test_unqualified_real_provider_is_refused_before_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('the default unqualified provider must never launch')
    monkeypatch.setattr('anymesher._volume_process._launch_worker', forbidden)
    snapshot, _ = box()
    with pytest.raises(VolumeBackendUnavailable, match='UNQUALIFIED'):
        generate_volume_mesh(snapshot, target_edge_length=.25, boundary_tolerance=1e-7)


def test_linux_group_accounting_excludes_terminated_and_unrelated_processes(monkeypatch, tmp_path):
    import anymesher._volume_process as module
    for pid, state, group in ((1, 'S', 71), (2, 'Z', 71), (3, 'R', 99), (4, 'X', 71)):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / 'stat').write_text(f'{pid} (test process (name)) {state} 10 {group} 10 0 0')
    original_path = module.Path
    monkeypatch.setattr(module, 'Path', lambda value: tmp_path if value == '/proc' else original_path(value))
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    assert module._active_posix_group(71) == 1
    assert module._active_posix_group(99) == 1
    assert module._active_posix_group(42) == 0
