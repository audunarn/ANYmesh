"""Owned provider subprocess and retained diagnostics; no optional imports here."""
from __future__ import annotations

from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import numpy as np

from .volume import (InvalidVolumeMesh, VolumeBackendUnavailable,
                     VolumeMeshingCancelled, VolumeMeshingTimeout, _cancelled)


# W2.0 has not established a viable/qualified real provider. Root's raw
# feasibility tool is the separately governed research route. No environment,
# public parameter or fallback bypasses this capability decision.
_PROVIDER_QUALIFIED = False
_PROVIDER_DISPOSITION = (
    'UNQUALIFIED: Workflow 2 real-provider feasibility is blocked; '
    'pytetwild public volume generation is disabled. '
    'See docs/volume_mesh.md and the root Workflow 2 handoff. '
    'Raw tools/volume_feasibility.py is a separately governed research route.'
)


def _active_posix_group(group_id):
    """Count running owned group members; terminated zombies are not active."""
    proc = Path('/proc')
    if sys.platform.startswith('linux') and proc.is_dir():
        active = 0
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                # comm may contain spaces/parentheses; the last ')' ends it.
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(fields[2]) == group_id and fields[0] not in ('Z', 'X'):
                active += 1
        return active
    # POSIX without procfs: use the platform process table, with a bounded
    # command and no shell. Failure to inspect is a cleanup failure.
    result = subprocess.run(['ps', '-axo', 'pid=,pgid=,stat='], capture_output=True,
                            text=True, check=True, timeout=2)
    return sum(1 for line in result.stdout.splitlines()
               if (fields := line.split()) and int(fields[1]) == group_id
               and not fields[2].startswith(('Z', 'X')))


class _WindowsJob:
    """Kill-on-close job assigned before the worker receives its start signal."""
    def __init__(self, process):
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64),
                        ('PerJobUserTimeLimit', ctypes.c_int64),
                        ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
                         'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.TerminateJobObject.restype = wintypes.BOOL
        kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        kernel.QueryInformationJobObject.restype = wintypes.BOOL
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.kernel, self.handle = kernel, handle
        self.ctypes = ctypes
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle, wintypes.HANDLE(int(process._handle))):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def terminate_and_verify(self):
        ctypes = self.ctypes
        from ctypes import wintypes
        class Accounting(ctypes.Structure):
            _fields_ = [('TotalUserTime', ctypes.c_int64), ('TotalKernelTime', ctypes.c_int64),
                        ('ThisPeriodTotalUserTime', ctypes.c_int64), ('ThisPeriodTotalKernelTime', ctypes.c_int64),
                        ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                        ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.perf_counter() + 5
        while True:
            accounting = Accounting()
            if not self.kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(accounting), ctypes.sizeof(accounting), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if accounting.ActiveProcesses == 0:
                return 0
            if time.perf_counter() >= deadline:
                raise OSError('provider job still has active owned processes after termination')
            time.sleep(.01)

    def close(self):
        if self.handle:
            if not self.kernel.CloseHandle(self.handle):
                raise OSError('failed to close provider process-tree job')
            self.handle = None


def _launch_worker(command, directory, timeout, cancel):
    """Run a stdin-gated worker and always reap its owned process tree.

    The command is an internal adapter seam used by controlled process tests;
    callers cannot select arbitrary commands through the public volume API.
    """
    directory = Path(directory)
    started = time.perf_counter()
    process, job = None, None
    failure = None
    status = 'startup_failed'
    result = dict(command=list(map(str, command)), python=sys.version, platform=sys.platform,
                  pid=None, exit_code=None, process_wall_seconds=None, cleanup='not_started')
    try:
        with (directory / 'stdout.txt').open('wb') as stdout, (directory / 'stderr.txt').open('wb') as stderr:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                       cwd=directory, start_new_session=os.name != 'nt',
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            result['pid'] = process.pid
            if os.name == 'nt':
                job = _WindowsJob(process)
            process.stdin.write(b'start\n')
            process.stdin.close()
            status = 'running'
            while process.poll() is None:
                elapsed = time.perf_counter() - started
                if _cancelled(cancel):
                    status = 'cancelled'
                    failure = VolumeMeshingCancelled('provider cancelled')
                    break
                if elapsed >= timeout:
                    status = 'timeout'
                    failure = VolumeMeshingTimeout(f'provider exceeded {timeout:g}s deadline')
                    break
                time.sleep(min(.025, max(.001, timeout - elapsed)))
            if failure is None:
                status = 'success' if process.returncode == 0 else 'worker_failed'
    except Exception as exc:
        failure = VolumeBackendUnavailable(f'provider process could not start: {exc}')
    finally:
        try:
            # Close the job even after success: a worker may have left children.
            if job is not None:
                try:
                    result['active_owned_processes_after_cleanup'] = job.terminate_and_verify()
                finally:
                    job.close()
            elif process is not None and os.name != 'nt':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process is not None and process.poll() is None:
                process.kill()
            if process is not None:
                process.wait(timeout=5)
                result['exit_code'] = process.returncode
                if os.name != 'nt':
                    deadline = time.perf_counter() + 5
                    while (remaining := _active_posix_group(process.pid)):
                        if time.perf_counter() >= deadline:
                            raise OSError(f'{remaining} active owned provider group members remain after termination')
                        time.sleep(.01)
                    result['active_owned_processes_after_cleanup'] = 0
            result['cleanup'] = 'owned_tree_terminated_and_parent_reaped'
        except Exception as exc:
            status = 'cleanup_failed'
            failure = VolumeBackendUnavailable(f'provider process cleanup failed: {exc}')
        result.update(status=status, process_wall_seconds=time.perf_counter() - started)
        (directory / 'process.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if failure is not None:
        failure.evidence_directory = str(directory)
        raise failure
    return result


def run_provider(source, target_edge_length, timeout, cancel):
    if not _PROVIDER_QUALIFIED:
        raise VolumeBackendUnavailable(_PROVIDER_DISPOSITION)
    directory = Path(tempfile.mkdtemp(prefix='anymesher-volume-'))
    surface = np.asarray([(loop[0], loop[k], loop[k + 1])
                          for loop in source.loops for k in range(1, len(loop) - 1)], dtype=np.int64)
    np.savez(directory / 'input.npz', coordinates=source.coordinates, facets=surface)
    options = dict(epsilon=source.delta / source.diagonal,
                   edge_length_abs=target_edge_length, edge_length_fac=0.05,
                   simplify=False, coarsen=False, optimize=True, num_threads=1,
                   stop_energy=10.0, num_opt_iter=80, loglevel=3, quiet=True,
                   vtk_ordering=False, disable_filtering=False)
    request = dict(provider='pytetwild', required_version='0.4.2', options=options,
                   source_signature=source.signature, model_id=source.model_id,
                   revision=source.revision, region_id=source.region_id)
    import hashlib
    request['input_sha256'] = hashlib.sha256((directory / 'input.npz').read_bytes()).hexdigest()
    (directory / 'request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    command = [sys.executable, str(Path(__file__).with_name('_volume_worker.py')), str(directory)]
    result = _launch_worker(command, directory, timeout, cancel)
    response_path = directory / 'response.json'
    if not response_path.exists():
        error = VolumeBackendUnavailable(f'provider failed before returning metadata; diagnostics: {directory}')
        error.evidence_directory = str(directory)
        raise error
    try:
        response = json.loads(response_path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise InvalidVolumeMesh(f'invalid provider response; diagnostics: {directory}') from exc
    if response.get('status') == 'unavailable':
        error = VolumeBackendUnavailable(f"pytetwild 0.4.2 unavailable: {response.get('detail', '')}; diagnostics: {directory}")
        error.evidence_directory = str(directory)
        raise error
    if result['exit_code'] != 0 or response.get('status') != 'success':
        raise InvalidVolumeMesh(f"provider did not produce a complete mesh: {response.get('detail', '')}; diagnostics: {directory}")
    if response.get('version') not in ('0.4.2', '0.4.2+anyworkflow2.1') or response.get('effective_options') != options:
        raise InvalidVolumeMesh(f'provider metadata does not match required settings; diagnostics: {directory}')
    try:
        with np.load(directory / 'output.npz', allow_pickle=False) as output:
            x, t = output['coordinates'].copy(), output['cells'].copy()
    except (OSError, ValueError, KeyError) as exc:
        raise InvalidVolumeMesh(f'invalid or incomplete raw provider arrays; diagnostics: {directory}') from exc
    return x, t, dict(backend='pytetwild', provider_version=response['version'],
                      effective_options=options, input_sha256=request['input_sha256'],
                      evidence_directory=str(directory), process=result)
