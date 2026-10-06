"""Exp 3: (a) spawn cold-start cost, (b) connected crossing-grid phase profile.

(b) is the case per-component parallelism cannot help: one connected model of
n/2 x n/2 crossing webs on a base plate.  We time the pipeline phases and the
top cProfile cumulative entries, to bound what parallelising each phase could
give (Amdahl) before any design work.
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from anygeometry import GeometryModel, from_dict, to_dict
from anymesher import generate_hybrid_mesh_result


def grid(n: int, pitch: float = 1.0):
    """Base plate with n/2 webs along x and n/2 along y, all crossing."""
    g = GeometryModel()
    m = n // 2
    L = pitch * (m + 1)
    h = 0.5
    base = g.add_points(((0, 0, 0), (L, 0, 0), (L, L, 0), (0, L, 0)))
    g.add_sheet((g.add_plate(tuple(base)),))
    for i in range(m):
        c = pitch * (i + 1)
        wx = g.add_points(((0, c, 0), (L, c, 0), (L, c, h), (0, c, h)))
        g.add_sheet((g.add_plate(tuple(wx)),))
        wy = g.add_points(((c, 0, 0), (c, L, 0), (c, L, h), (c, 0, h)))
        g.add_sheet((g.add_plate(tuple(wy)),))
    return g


def noop(x):
    return x


def cold_start(workers: int):
    t = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(import_and_mesh, [None] * workers))
    return time.perf_counter() - t


def import_and_mesh(_):
    from anygeometry import GeometryModel
    from anymesher import generate_hybrid_mesh_result
    g = GeometryModel()
    p = g.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    g.add_sheet((g.add_plate(tuple(p)),))
    return generate_hybrid_mesh_result(g, target_size=0.25).mesh.num_elements


def phases(n: int, profile: bool):
    g = grid(n)
    pr = cProfile.Profile() if profile else None
    t = time.perf_counter()
    if pr:
        pr.enable()
    r = generate_hybrid_mesh_result(g, target_size=0.25)
    if pr:
        pr.disable()
    dt = time.perf_counter() - t
    d = r.mesh.hybrid_diagnostics["phase_seconds"]
    print(f"grid n={n}: wall={dt:.2f}s elems={r.mesh.num_elements} "
          + " ".join(f"{k}={v:.2f}" for k, v in d.items()))
    if pr:
        st = pstats.Stats(pr)
        st.sort_stats("cumulative").print_stats(28)


if __name__ == "__main__":
    for w in (4, 8, 16):
        print(f"spawn cold start, {w} workers, import+mesh tiny plate: {cold_start(w):.2f}s")
    for n in (16, 32):
        phases(n, profile=False)
    phases(32, profile=True)
