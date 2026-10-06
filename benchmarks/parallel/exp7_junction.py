"""Exp 7: crossing-grid connectivity phase + a numbering-exact mesh digest (A/B by PYTHONPATH)."""
import hashlib, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import anymesher
from exp3_coldstart_heavy_grid import grid
from anymesher import generate_hybrid_mesh_result

def digest(m):
    h = hashlib.sha256()
    for k in sorted(m.nodes): h.update(repr((k, tuple(np.round(m.nodes[k], 9)))).encode())
    for t in (m.quads, m.tris, m.beams):
        for k in sorted(t): h.update(repr((k, t[k])).encode())
    for k in sorted(m.couplings): h.update(repr((k, m.couplings[k])).encode())
    return h.hexdigest()[:16]

if __name__ == "__main__":
    print(anymesher.__file__)
    for n in [int(x) for x in sys.argv[1:]] or [16, 32, 48]:
        g = grid(n)
        t = time.perf_counter()
        r = generate_hybrid_mesh_result(g, target_size=0.25)
        d = r.mesh.hybrid_diagnostics["phase_seconds"]
        print(f"n={n} wall={time.perf_counter()-t:.2f}s connectivity={d['structural_connectivity']:.2f}s "
              f"elems={r.mesh.num_elements} digest={digest(r.mesh)} actions={r.connectivity.connected}", flush=True)
