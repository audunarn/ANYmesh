"""Development scaling probe for the experimental quad-first route.

Meshes a 10 m x 6 m plate (optionally with an off-centre circular hole) at a
sequence of target sizes and prints one JSON line per run with element counts,
wall time and per-stage seconds.  This is a single unrepeated measurement per
size, not a formal timing qualification.

    python benchmarks/quad_first_scaling.py --sizes 0.5 0.25 0.14 0.1
    python benchmarks/quad_first_scaling.py --hole --sizes 0.4 0.2
"""
from __future__ import annotations

import argparse
import json
import math
import time

from anygeometry import GeometryModel, punch_hole

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.timing import collect_quad_stage_timings


def _plate(hole: bool) -> tuple[GeometryModel, int]:
    model = GeometryModel()
    vertices = model.add_points(((0, 0, 0), (10, 0, 0), (10, 6, 0), (0, 6, 0)))
    face = model.add_face(model.add_polyline(vertices, close=True), surface=None)
    if hole:
        face, _ = punch_hole(model, face, (3.2, 2.4, 0.0), 1.2)
    return model, face


def _run(size: float, hole: bool, layout: str) -> dict:
    model, face = _plate(hole)
    options = QuadMeshingOptions(max_front_iterations=10_000_000)
    with collect_quad_stage_timings() as stages:
        started = time.perf_counter()
        mesh = generate_hybrid_mesh_result(
            model, target_size=size, face_ids=(face,),
            quad_options=options, layout_policy=layout,
        ).mesh
        elapsed = time.perf_counter() - started
    return {
        "shape": "plate_hole" if hole else "plate",
        "layout": layout,
        "target_size": size,
        "q4": len(mesh.quads),
        "t3": len(mesh.tris),
        "seconds": round(elapsed, 3),
        "stages": {name: round(value, 3) for name, value in sorted(stages.items())},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=float, nargs="+", default=[0.5, 0.25, 0.14, 0.1])
    parser.add_argument("--hole", action="store_true")
    parser.add_argument("--layout", choices=("existing", "adaptive"), default="existing")
    args = parser.parse_args(argv)
    rows = []
    for size in args.sizes:
        row = _run(size, args.hole, args.layout)
        if rows:
            previous = rows[-1]
            growth = math.log(row["seconds"] / previous["seconds"])
            growth /= math.log(max(row["q4"] + row["t3"], 2) / max(previous["q4"] + previous["t3"], 1))
            row["empirical_exponent"] = round(growth, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
