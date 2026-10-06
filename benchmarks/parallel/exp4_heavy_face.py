"""Exp 4: element-heavy single faces -- where does time go when meshing dominates?"""
import math, sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from native_hybrid_performance import _pentagon, _rectangle, _plate_with_hole
from anymesher import generate_hybrid_mesh_result, NativeMeshingOptions

cases = {"rectangle(mapped)": (_rectangle, 1.0), "pentagon(native)": (_pentagon, 2.3776),
         "plate_with_hole(native)": (_plate_with_hole, 16 - math.pi * .25)}
for name, (mk, area) in cases.items():
    for n in (10_000, 100_000):
        g = mk()
        target = 1.65 * math.sqrt(area / n)
        t = time.perf_counter()
        r = generate_hybrid_mesh_result(g, target_size=target)
        dt = time.perf_counter() - t
        d = r.mesh.hybrid_diagnostics["phase_seconds"]
        print(f"{name} ~{n}: wall={dt:.2f}s elems={r.mesh.num_elements} "
              + " ".join(f"{k}={v:.2f}" for k, v in d.items()), flush=True)
