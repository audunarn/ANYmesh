"""Non-acceptance trial of declared-attachment local sizing at SG1 h=0.15 m."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import numpy as np

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.refinement import Refinement
from anymesher.serialize import mesh_to_dict

from .consumer import solve
from .fixtures import build
from .measure import audit
from .point_coupling_diagnostic import _artifact, _counts
from .runner import ROOT, compare_results, digest, write


OUTPUT = ROOT / "reports/quad_first/structural-gate/attachment-refinement-trial"


def trial(case: str, graded: bool, baseline: Path, folder: Path,
          radius: float = .35, size_factor: float = 1/3,
          orientation: str = "cross_4theta", extra_refinement: bool = True,
          edge_divisions: dict[int, int] | None = None,
          target_size_m: float = .15):
    fixture = build(case, True)
    h = float(target_size_m)
    face = fixture.faces[0]
    center = fixture.patches[face].point(.5, .5)
    refinement = Refinement(size=h*size_factor, radius=radius, center=tuple(center),
                            growth=1.5, name="attachment-trial")
    start = time.perf_counter()
    mesh = generate_hybrid_mesh_result(
        fixture.geometry, face_ids=fixture.faces, member_ids=fixture.members,
        target_size=h, strategy="native", native_backend="python",
        recombine=True, order="quadratic",
        quad_options=QuadMeshingOptions(quality_model="shape_jacobian",
                                        orientation=orientation),
        refinements=fixture.refinements(h, graded) + ((refinement,) if extra_refinement else ()),
        overrides=edge_divisions,
    ).mesh
    meshing_seconds = time.perf_counter() - start
    write(folder / "mesh.json", mesh_to_dict(mesh))
    measurements = audit(fixture, mesh, h, graded)
    write(folder / "audit.json", measurements)
    result = solve(fixture, mesh, "member")
    write(folder / "result.json", result)
    reference_path = _artifact(baseline, f"reference_{case}_b1_r1_member",
                               "result.json")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    errors = compare_results(result, reference)
    summary = {
        "case": case, "graded": graded, "kind": "diagnostic-not-SG1-acceptance",
        "target_size_m": h, "attachment_local_size_m": h*size_factor,
        "attachment_radius_m": radius, "attachment_growth": 1.5,
        "orientation": orientation, "extra_attachment_refinement": extra_refinement,
        "edge_divisions": edge_divisions or {},
        "source_attachment_xyz_m": np.asarray(center).tolist(),
        "meshing_seconds": meshing_seconds, "counts": _counts(mesh),
        "reference_result_sha256": digest(reference_path),
        "max_relative_error": max(errors.values()), "errors": errors,
        "audit_failed_checks": [item["name"] for item in measurements["checks"]
                                if item["status"] != "passed"],
        "solver_failed_checks": [item["name"] for item in result["checks"]
                                 if item["status"] != "passed"],
    }
    write(folder / "summary.json", summary)
    write(folder / "manifest.json", {
        path.name: digest(path) for path in sorted(folder.iterdir())
        if path.is_file() and path.name != "manifest.json"
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("CY", "CO", "R", "C"), required=True)
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--radius", type=float, default=.35)
    parser.add_argument("--size-factor", type=float, default=1/3)
    parser.add_argument("--orientation", choices=("cross_4theta", "boundary_tangent"),
                        default="cross_4theta")
    parser.add_argument("--no-extra-refinement", action="store_true")
    parser.add_argument("--edge-divisions", help="diagnostic source-edge counts, e.g. 1:12,2:22")
    parser.add_argument("--h", type=float, default=.15)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    if not (args.radius > 0 and 0 < args.size_factor < 1 and args.h > 0):
        parser.error("radius must be positive and size-factor must be between 0 and 1")
    edge_divisions = None
    if args.edge_divisions:
        try:
            edge_divisions = dict(
                (int(edge), int(count))
                for edge, count in (pair.split(":") for pair in args.edge_divisions.split(","))
            )
        except ValueError:
            parser.error("edge-divisions must be comma-separated edge:count pairs")
        if not edge_divisions or any(edge <= 0 or count <= 0 for edge, count in edge_divisions.items()):
            parser.error("edge-divisions requires positive edge IDs and counts")
    root = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                     + "-" + args.case + f"-g{int(args.graded)}-"
                     + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    print("TRIAL " + str(root), flush=True)
    result = trial(args.case, args.graded, args.baseline.resolve(), root,
                   args.radius, args.size_factor, args.orientation,
                   not args.no_extra_refinement, edge_divisions, args.h)
    print(json.dumps({"max_relative_error": result["max_relative_error"],
                      "audit_failures": len(result["audit_failed_checks"]),
                      "solver_failures": len(result["solver_failed_checks"])}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
