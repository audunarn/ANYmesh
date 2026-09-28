"""One isolated IS1 mesh operation against a frozen source model."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from anygeometry import from_dict, to_dict
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.timing import collect_quad_stage_timings
from anymesher.refinement import Refinement
from anymesher.serialize import mesh_to_dict
from benchmarks.is1.measure import audit_irregular
from benchmarks.sg1.fixtures import Fixture, Patch
from benchmarks.sg1.measure import audit, mesh_signature


def _write(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def _sg1_fixture(case, geometry, faces, center):
    face_edges = {face: tuple((int(use.edge), bool(use.forward))
                             for use in geometry.faces[face].loop) for face in faces}
    counts = {}
    for uses in face_edges.values():
        for edge, _ in uses:
            counts[edge] = counts.get(edge, 0) + 1
    sheets = tuple(geometry.sheets)
    return Fixture({"cone_unequal": "CO", "ruled": "R", "coons": "C"}[case],
                   geometry, {face: Patch(geometry.faces[face].surface) for face in faces},
                   face_edges, tuple(edge for edge, count in counts.items() if count == 2),
                   tuple(edge for edge, count in counts.items() if count == 1),
                   np.asarray(center, dtype=float), sheet=sheets[0] if len(sheets) == 1 else None)


def main(folder: Path):
    spec = json.loads((folder / "input.json").read_text(encoding="utf-8"))
    frozen = json.loads(Path(spec["fixture"]).read_text(encoding="utf-8"))
    geometry = from_dict(frozen["geometry"])
    faces = tuple(int(item) for item in frozen["faces"])
    before = to_dict(geometry)
    h = float(spec["h"])
    refined = bool(spec["refined"])
    zones = (Refinement(size=h/3, radius=.35, center=tuple(frozen["center"]),
                        growth=1.5, name=f"IS1/{spec['case']}"),) if refined else ()
    _write(folder / "progress.json", {"stage": "mesh"})
    started = perf_counter()
    with collect_quad_stage_timings() as stages:
        result = generate_hybrid_mesh_result(
            geometry, face_ids=faces, target_size=h, order=spec["order"],
            strategy="native", native_backend="python",
            quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
            layout_policy=spec["layout"],
            overrides={int(k): int(v) for k, v in spec["overrides"].items()},
            refinements=zones,
        )
    mesh_seconds = perf_counter() - started
    mesh = result.mesh
    shell_count = len(mesh.quads) + len(mesh.tris)
    dofs = 6 * len(mesh.nodes)
    if shell_count > 1500 or dofs > 15000:
        _write(folder / "result.json", {"status": "incomplete",
               "reason": f"resource counts {shell_count} shells/{dofs} DOFs"})
        return
    _write(folder / "progress.json", {"stage": "validation"})
    metrics = (audit(_sg1_fixture(spec["case"], geometry, faces, frozen["center"]),
                     mesh, h, refined)
               if spec["case"] in ("cone_unequal", "ruled", "coons")
               else audit_irregular(geometry, faces, mesh, h, zones, frozen["center"]))
    metrics["checks"].append({"name": "source-unchanged",
                              "status": "passed" if to_dict(geometry) == before else "failed"})
    metrics["checks"].append({"name": "effective-layout",
                              "status": "passed" if mesh.hybrid_diagnostics.get("layout_policy") == spec["layout"] else "failed"})
    _write(folder / "metrics.json", metrics)
    _write(folder / "mesh.json", mesh_to_dict(mesh))
    _write(folder / "diagnostics.json", mesh.hybrid_diagnostics)
    _write(folder / "result.json", {
        "status": "passed" if all(item["status"] == "passed" for item in metrics["checks"]) else "failed",
        "failed_checks": [item["name"] for item in metrics["checks"] if item["status"] == "failed"],
        "mesh_seconds": mesh_seconds, "stage_seconds": stages,
        "mesh_signature": mesh_signature(mesh), "shells": shell_count, "dofs": dofs,
    })


if __name__ == "__main__":
    main(Path(sys.argv[1]))
