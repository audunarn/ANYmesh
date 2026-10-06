"""Exp 1: does whole-model time scale linearly with N disjoint components?

Builds N translated copies of a small structural model (two plates, one
extruded wall crossing them, declared sheets) in ONE GeometryModel and times
generate_hybrid_mesh_result on (a) the whole model, (b) each component alone.
"""
from __future__ import annotations

import json
import sys
import time

from anygeometry import GeometryModel

from anymesher import generate_hybrid_mesh_result


def add_component(g: GeometryModel, ox: float, oy: float):
    p = g.add_points(
        (
            (ox + 0.0, oy + 0.0, 0.0), (ox + 2.0, oy + 0.0, 0.0),
            (ox + 0.0, oy + 2.0, 0.0), (ox + 2.0, oy + 2.0, 0.0),
            (ox + 0.5, oy + 0.5, 0.5), (ox + 1.5, oy + 0.5, 0.5),
            (ox + 0.5, oy + 1.5, 0.5), (ox + 1.5, oy + 1.5, 0.5),
        )
    )
    support = g.add_plate((p[0], p[1], p[3], p[2]))
    floating = g.add_plate((p[4], p[5], p[7], p[6]))
    diag = g.add_line(p[1], p[2])
    g.extrude((diag,), (0.0, 0.0, 1.0))
    g.add_sheet((support,))
    g.add_sheet((floating,))
    return support, floating


def build(n: int) -> GeometryModel:
    g = GeometryModel()
    side = int(n ** 0.5) + 1
    for i in range(n):
        add_component(g, 10.0 * (i % side), 10.0 * (i // side))
    return g


def timed(g, **kw):
    t = time.perf_counter()
    r = generate_hybrid_mesh_result(g, target_size=0.25, **kw)
    return time.perf_counter() - t, r


if __name__ == "__main__":
    out = {}
    warm = build(1)
    timed(warm)
    for n in [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8, 16]:
        g = build(n)
        dt, r = timed(g)
        m = r.mesh
        out[n] = {
            "wall": round(dt, 3),
            "per_comp": round(dt / n, 3),
            "nodes": m.num_nodes,
            "elems": m.num_elements,
            "phases": {k: round(v, 3) for k, v in m.hybrid_diagnostics["phase_seconds"].items()},
        }
        print(n, json.dumps(out[n]), flush=True)
