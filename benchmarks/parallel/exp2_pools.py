"""Exp 2: per-component meshing via extract_model_closure, serial/thread/process.

Reports extraction, mesh, pickle-return and a naive-merge-size cost so the
decision "is per-component parallelism worth it" rests on numbers.
"""
from __future__ import annotations

import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from anygeometry import GeometryModel, from_dict, to_dict

from anymesher import generate_hybrid_mesh_result

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_scaling import add_component  # noqa: E402

TARGET = 0.25


def build(n: int, size_scale: float = 1.0):
    g = GeometryModel()
    side = int(n ** 0.5) + 1
    groups = []
    for i in range(n):
        before = (set(g.faces), set(g.sheets))
        add_component(g, 10.0 * (i % side), 10.0 * (i // side))
        groups.append(
            [("face", f) for f in set(g.faces) - before[0]]
            + [("sheet", s) for s in set(g.sheets) - before[1]]
        )
    return g, groups


def mesh_one(model):
    t = time.perf_counter()
    if isinstance(model, dict):
        model = from_dict(model)
    r = generate_hybrid_mesh_result(model, target_size=TARGET)
    mesh = r.mesh
    mesh.boundary_registry = None  # live-geometry bound; not transportable
    return mesh, time.perf_counter() - t


def main(n: int, workers: int):
    g, groups = build(n)
    t = time.perf_counter()
    whole = generate_hybrid_mesh_result(g, target_size=TARGET).mesh
    t_whole = time.perf_counter() - t

    t = time.perf_counter()
    closures = [g.extract_model_closure(h) for h in groups]
    t_extract = time.perf_counter() - t
    models = [c.working_model for c in closures]

    t = time.perf_counter()
    serial = [mesh_one(m) for m in models]
    t_serial = time.perf_counter() - t

    t = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        thr = list(ex.map(mesh_one, models))
    t_thread = time.perf_counter() - t

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        dicts = [to_dict(m) for m in models]
        list(ex.map(mesh_one, dicts[:workers]))  # warm the workers (imports)
        t = time.perf_counter()
        proc = list(ex.map(mesh_one, dicts))
        t_proc = time.perf_counter() - t

    pk = sum(len(pickle.dumps(m)) for m, _ in serial) / 1e6
    nodes = sum(m.num_nodes for m, _ in serial)
    elems = sum(m.num_elements for m, _ in serial)
    print(
        f"n={n} workers={workers} whole={t_whole:.2f}s extract={t_extract:.2f}s "
        f"serial_parts={t_serial:.2f}s threads={t_thread:.2f}s procs={t_proc:.2f}s "
        f"| nodes whole/parts={whole.num_nodes}/{nodes} elems={whole.num_elements}/{elems} "
        f"| result pickle={pk:.2f}MB"
    )


if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [8, 16]:
        main(n, workers=8)
