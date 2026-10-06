"""Exp 6: serial vs component-parallel route, cold and warm pool, by N and workers."""
import multiprocessing as mp, os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2_pools import build
from anymesher import generate_hybrid_mesh_result
from anymesher.component_parallel import generate_hybrid_mesh_result_parallel, ParallelOptions

if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[1:]] or [16, 32, 64]
    for n in ns:
        g, _ = build(n)
        t = time.perf_counter(); generate_hybrid_mesh_result(g, target_size=0.25); ts = time.perf_counter() - t
        row = [f"n={n} serial={ts:.2f}s"]
        for w in (4, 8, 16):
            t = time.perf_counter()
            generate_hybrid_mesh_result_parallel(g, target_size=0.25, parallel=ParallelOptions(workers=w))
            cold = time.perf_counter() - t
            with ProcessPoolExecutor(w, mp_context=mp.get_context("spawn")) as ex:
                generate_hybrid_mesh_result_parallel(g, target_size=0.25, parallel=ParallelOptions(workers=w, executor=ex))  # warm
                t = time.perf_counter()
                r = generate_hybrid_mesh_result_parallel(g, target_size=0.25, parallel=ParallelOptions(workers=w, executor=ex))
                warm = time.perf_counter() - t
            info = r.mesh.hybrid_diagnostics["parallel"]
            row.append(f"w={w}: cold={cold:.2f} warm={warm:.2f} (x{ts/warm:.1f}; mesh={info['mesh_seconds']:.2f} prep={info['prepare_seconds']:.2f} join={info['join_seconds']:.2f})")
        print(" | ".join(row), flush=True)
