import sys, time
import anyfem.model.project as pm
from anyfem.model.project import Project
import anymesher
from anymesher import generate_hybrid_mesh_result_parallel

def build(n):
    p = Project("many")
    side = int(n ** .5) + 1
    for i in range(n):
        x, y = 10.0 * (i % side), 10.0 * (i // side)
        v = p.geometry.add_points(((x,y,0),(x+2,y,0),(x+2,y+1,0),(x,y+1,0)))
        f = p.geometry.add_plate(v)
        p.geometry.add_sheet([f], name=f"plate{i}")
    return p

import numpy as np
def sig(m):
    pos = lambda n: tuple(np.round(m.nodes[n], 9))
    cen = lambda e: tuple(np.round(np.mean([m.nodes[n] for n in (m.quads.get(e) or m.tris.get(e) or m.beams[e])], axis=0), 9))
    return (sorted(pos(n) for n in m.nodes), {f: sorted(cen(e) for e in els) for f, els in m.elements_of_face.items()},
            {e: sorted(pos(n) for n in ns) for e, ns in m.nodes_of_edge.items()}, len(m.quads), len(m.tris), len(m.beams), len(m.couplings))
calls = []
orig = pm.generate_hybrid_mesh
def wrapped(working, **kw):
    calls.append(sorted(kw))
    result = generate_hybrid_mesh_result_parallel(working, **kw)
    info = result.mesh.hybrid_diagnostics["parallel"]
    print("   parallel info:", {k: info[k] for k in info if k in ("used", "reason", "components", "workers")}, flush=True)
    return result.mesh

TS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
if __name__ == "__main__":
    n = int(sys.argv[1]); strategy = sys.argv[2] if len(sys.argv) > 2 else None
    p = build(n)
    t = time.perf_counter()
    called = []
    pm.generate_hybrid_mesh = lambda *a, **k: (called.append(1), orig(*a, **k))[1]
    m = p.generate_mesh(TS, strategy=strategy) if strategy else p.generate_mesh(TS)
    print(f"unpatched n={n} strategy={strategy}: {time.perf_counter()-t:.2f}s elems={m.num_elements} hybrid_called={bool(called)}", flush=True)
    p = build(n)
    pm.generate_hybrid_mesh = wrapped
    t = time.perf_counter()
    m2 = p.generate_mesh(TS, strategy=strategy) if strategy else p.generate_mesh(TS)
    print("   mesh identical up to numbering:", sig(m) == sig(m2))
    print(f"parallel-wrapped n={n}: {time.perf_counter()-t:.2f}s elems={m2.num_elements} wrapper_called={bool(calls)}", flush=True)
    if calls: print("   hybrid kwargs from ANYfem:", calls[0])
