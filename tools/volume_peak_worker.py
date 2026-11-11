"""Stdin-gated raw-array study worker; reports its own kernel peak memory."""
from __future__ import annotations
import sys


def main():
    if sys.stdin.readline().strip() != 'start':
        return 2
    import hashlib
    import importlib.metadata
    import json
    import os
    from pathlib import Path
    import time
    import numpy as np
    import pytetwild
    from pytetwild import PyfTetWildWrapper as native
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory/'request.json').read_text())
    if importlib.metadata.version('pytetwild') != request['version']:
        raise ValueError('native version differs from frozen request')
    if getattr(native,'_workflow2_exact_envelope',False) is not True:
        raise ValueError('compiled exact-envelope mode missing')
    native_sha = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
    if native_sha != request['native_sha256']:
        raise ValueError('native bytes differ from frozen request')
    path = directory/'input.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != request['input_sha256']:
        raise ValueError('native input changed')
    with np.load(path,allow_pickle=False) as data:
        coordinates,facets = data['coordinates'],data['triangles']
    started = time.perf_counter()
    x,t = pytetwild.tetrahedralize(coordinates,facets,**request['options'])
    native_seconds = time.perf_counter()-started
    np.savez_compressed(directory/'raw.npz',coordinates=x,cells=t)
    digest=hashlib.sha256()
    with (directory/'raw.npz').open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    raw_sha=digest.hexdigest()
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        peak_bytes=int(counters.PeakWorkingSetSize)
    else:
        import resource
        peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    result=dict(status='RAW_GENERATED',native_seconds=native_seconds,peak_rss_through_observation_bytes=peak_bytes,
        peak_scope='kernel peak through observation after raw output serialization and streaming hash; not full child lifetime',
        coordinates=len(x),cells=len(t),native_sha256=native_sha,
        raw_sha256=raw_sha,
        effective_options=request['options'],python=sys.version,
        threads={name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    (directory/'native-result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
