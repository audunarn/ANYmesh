"""Exp 5: parallel route vs serial route on N disjoint components."""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from exp2_pools import build
from anymesher import generate_hybrid_mesh_result
from anymesher.component_parallel import (
    generate_hybrid_mesh_result_parallel, ParallelOptions, plan_independent_components)

def signature(mesh):
    """Id-independent: per-face sorted element centroids and node positions."""
    pos = lambda n: tuple(np.round(mesh.nodes[n], 9))
    nodes = sorted(pos(n) for n in mesh.nodes)
    per_face = {f: sorted(tuple(np.round(np.mean([mesh.nodes[n] for n in (mesh.quads.get(e) or mesh.tris.get(e))], axis=0), 9))
                          for e in els) for f, els in mesh.elements_of_face.items()}
    edges = {e: sorted(pos(n) for n in ns) for e, ns in mesh.nodes_of_edge.items()}
    return nodes, per_face, edges, len(mesh.quads), len(mesh.tris), len(mesh.beams), len(mesh.couplings)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    g, _ = build(n)
    plan = plan_independent_components(g, pad=0.25)
    print("components:", len(plan.components), plan.fallback_reason)
    t = time.perf_counter(); s = generate_hybrid_mesh_result(g, target_size=0.25); ts = time.perf_counter() - t
    t = time.perf_counter(); p = generate_hybrid_mesh_result_parallel(g, target_size=0.25, parallel=ParallelOptions(workers=8)); tp = time.perf_counter() - t
    print(f"serial {ts:.2f}s parallel {tp:.2f}s", {k: (round(v,3) if isinstance(v,float) else v) for k, v in p.mesh.hybrid_diagnostics["parallel"].items() if k != "component_seconds"})
    a, b = signature(s.mesh), signature(p.mesh)
    print("nodes equal", a[0] == b[0], "per-face equal", a[1] == b[1], "edge nodes equal", a[2] == b[2], "counts", a[3:], b[3:])
    print("strategy equal", dict(s.strategy_by_face) == dict(p.strategy_by_face))
    print("registry entries", len(p.mesh.boundary_registry), len(s.mesh.boundary_registry))
    print("sheet elems equal", {k: len(v) for k, v in s.mesh.elements_of_sheet.items()} == {k: len(v) for k, v in p.mesh.elements_of_sheet.items()})
    print("junction edges", len(s.mesh.declared_plate_junction_edges), len(p.mesh.declared_plate_junction_edges))
    print("preflight", len(s.preflight), len(p.preflight), "conn", s.connectivity and s.connectivity.connected, p.connectivity and p.connectivity.connected)
