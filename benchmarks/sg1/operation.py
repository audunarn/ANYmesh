"""One owned, supervised SG1 operation. Invoked only by runner."""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path


def save(path,value):
    encoded=json.dumps(value,sort_keys=True,allow_nan=False,indent=2)
    with Path(path).open('x',encoding='utf-8') as stream:
        stream.write(encoded)


def peak_memory():
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        c=Counters(); c.cb=ctypes.sizeof(c)
        kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi'); psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb): raise ctypes.WinError()
        return int(c.PeakWorkingSetSize)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def main(spec,folder):
    start=time.perf_counter()
    from .fixtures import build,generate
    from .measure import audit,promotion_checks,mesh_signature,geometry_signature,check
    from anymesher.serialize import mesh_to_dict,mesh_from_dict
    from anygeometry import to_dict
    import numpy as np
    cold=time.perf_counter()-start
    checks=[]; timings=[]
    if spec['operation'] in ('analytical','negative'):
        from . import controls
        result=getattr(controls,spec['operation'])()
    else:
        fixture=build(spec['case'],spec.get('member',False))
        before=geometry_signature(fixture)
        save(folder/'source.json',to_dict(fixture.geometry))
        if spec['operation']=='mesh':
            from anymesher.errors import MeshError
            linear=None
            try:
                linear=generate(fixture,spec['h'],spec['graded'],order='linear')
            except MeshError as exc:
                # Do not waive linear acceptance or discard independently runnable
                # quadratic evidence when linear attachment admission fails.
                (folder/'linear-failure.log').write_text(traceback.format_exc(),encoding='utf-8')
                checks.append(dict(name='linear-generation',status='blocked',exception=type(exc).__name__,message=str(exc)))
                checks.append(dict(name='promotion-comparison',status='blocked',reason='linear generation did not publish'))
            if linear is not None:
                save(folder/'linear.json',mesh_to_dict(linear))
                checks.extend(audit(fixture,linear,spec['h'],spec['graded'])['checks'])
            t=time.perf_counter(); warm=generate(fixture,spec['h'],spec['graded']); warm_seconds=time.perf_counter()-t
            signature=mesh_signature(warm)
            for iteration in range(3):
                t=time.perf_counter(); mesh=generate(fixture,spec['h'],spec['graded']); meshing=time.perf_counter()-t
                t=time.perf_counter(); metrics=audit(fixture,mesh,spec['h'],spec['graded']); validation=time.perf_counter()-t
                t=time.perf_counter(); payload=mesh_to_dict(mesh); restored=mesh_from_dict(payload); serialization=time.perf_counter()-t
                check(checks,f'repeat/{iteration}',mesh_signature(mesh)==signature)
                check(checks,f'roundtrip/{iteration}',mesh_signature(restored)==mesh_signature(mesh))
                checks.extend({**c,'name':f'run/{iteration}/'+c['name']} for c in metrics['checks'])
                timings.append(dict(meshing=meshing,high_order_validation=validation,serialization=serialization))
                if iteration==0:
                    save(folder/'mesh.json',payload)
                    save(folder/'mesh-metrics.json',metrics)
                    if linear is not None: checks.extend(promotion_checks(linear,mesh))
            class Cancelled(Exception): pass
            stages=[]
            def cancel(stage):
                stages.append(str(stage))
                if len(stages)>=3: raise Cancelled(stage)
            try: generate(fixture,spec['h'],spec['graded'],cancellation_check=cancel)
            except Cancelled: check(checks,'cancellation/raised',True,stages=stages)
            else: check(checks,'cancellation/raised',False,stages=stages)
            check(checks,'cancellation/existing-mesh-unchanged',mesh_signature(warm)==signature)
            check(checks,'source/immutable',geometry_signature(fixture)==before)
            result=dict(checks=checks,timings=timings,warmup_seconds=warm_seconds,mesh_signature=signature,
                        diagnostics=payload['hybrid_diagnostics'])
            plot(mesh,metrics,folder/'mesh-errors.png')
        elif spec['operation']=='reference':
            from .reference import structured
            from anymesher.quad.high_order import certify_mapping_validity
            mesh=structured(fixture,spec['h'])
            for e,b in mesh.quads.items():
                report=certify_mapping_validity(np.array([mesh.nodes[n] for n in b]),'Q8')
                check(checks,f'reference/{e}/validity',report.status.value=='CERTIFIED_POSITIVE')
            save(folder/'mesh.json',mesh_to_dict(mesh))
            check(checks,'source/immutable',geometry_signature(fixture)==before)
            result=dict(checks=checks,counts=dict(nodes=len(mesh.nodes),quads=len(mesh.quads),beams=len(mesh.beams)))
        elif spec['operation']=='solve':
            from .consumer import solve
            mesh=mesh_from_dict(json.loads(Path(spec['mesh']).read_text(encoding='utf-8')))
            warm=solve(fixture,mesh,spec['load'])
            iterations=3 if spec.get('measured',True) else 1
            for i in range(iterations):
                result=solve(fixture,mesh,spec['load'])
                timings.append(result['timings'])
                checks.extend({**c,'name':f'run/{i}/'+c['name']} for c in result['checks'])
                check(checks,f'solve-repeat/{i}',abs(result['energy']-warm['energy'])<=1e-12*max(1.,abs(warm['energy'])))
            result={**result,'checks':checks,'timings':timings,'warmup_timings':warm['timings']}
        else: raise ValueError('unknown operation')
    result['cold_import_seconds']=cold
    result['peak_process_bytes']=peak_memory()
    result['status']='failed' if any(c['status']=='failed' for c in result['checks']) else 'blocked' if any(c['status']=='blocked' for c in result['checks']) else 'passed'
    return result


def plot(mesh,metrics,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    polys=[]; errors=[]
    for e,b in {**mesh.quads,**mesh.tris}.items():
        nc=4 if e in mesh.quads else 3
        polys.append([mesh.nodes[n] for n in b[:nc]])
        errors.append(metrics['geometry_by_cell'][e])
    fig=plt.figure(figsize=(9,5)); ax=fig.add_subplot(121,projection='3d')
    collection=Poly3DCollection(polys,edgecolors='k',linewidths=.15,cmap='viridis')
    collection.set_array(np_array(errors)); ax.add_collection3d(collection)
    xyz=np_array(list(mesh.nodes.values())); ax.auto_scale_xyz(*xyz.T); ax.set_title('Sampled surface error [m]')
    fig.colorbar(collection,ax=ax,shrink=.5)
    ratios=metrics['size_ratios']; lo,hi=min(ratios),max(ratios)
    if hi-lo<1e-10: lo,hi=lo-.1,hi+.1
    hist=fig.add_subplot(122); hist.hist(ratios,bins=24,range=(lo,hi))
    hist.axvline(.25,color='r'); hist.axvline(2,color='r'); hist.set_xlabel('Integrated edge / local size'); hist.set_ylabel('Unique edges')
    fig.tight_layout(); fig.savefig(path,dpi=130); plt.close(fig)


def np_array(x):
    import numpy as np
    return np.asarray(x)


if __name__=='__main__':
    folder=Path(sys.argv[1]); spec=json.loads((folder/'request.json').read_text())
    try:
        result=main(spec,folder)
    except Exception as exc:
        traceback.print_exc()
        result=dict(status='blocked' if spec['operation']=='solve' or type(exc).__name__=='CylinderAtlasError' else 'failed',
                    exception=type(exc).__name__,message=str(exc),checks=[])
    save(folder/'result.json',result)
    print(json.dumps({k:result[k] for k in ('status','exception','message') if k in result}))
    sys.exit(0 if result['status']=='passed' else 1)
