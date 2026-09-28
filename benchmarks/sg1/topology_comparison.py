"""Non-acceptance comparison of sealed quad-first and same-size Q8 topology."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

from anymesher.serialize import mesh_from_dict, mesh_to_dict

from .fixtures import build
from .measure import audit
from .point_coupling_diagnostic import _artifact
from .reference import structured
from .runner import ROOT, digest, identities, write


OUTPUT = ROOT / "reports/quad_first/structural-gate/topology-comparison"
CASES = ("CY", "CO", "R", "C")


def _stats(fixture, mesh, h: float) -> dict:
    measurements = audit(fixture, mesh, h, False)
    centre = fixture.patches[fixture.faces[0]].point(.5, .5)
    corner_nodes = {
        node
        for cell in mesh.elements_of_face[fixture.faces[0]]
        for node in (mesh.quads.get(cell) or mesh.tris[cell])[:(4 if cell in mesh.quads else 3)]
    }
    distances = np.asarray([np.linalg.norm(mesh.nodes[node] - centre) for node in corner_nodes])
    coupling = next(iter(mesh.couplings.values()))
    weights = np.asarray(coupling.weights)
    ratios = np.asarray(measurements["size_ratios"])
    core_quality = []
    core_triangles = 0
    for cell in mesh.elements_of_face[fixture.faces[0]]:
        corners = (mesh.quads.get(cell) or mesh.tris[cell])[:(4 if cell in mesh.quads else 3)]
        centroid = np.mean([mesh.nodes[node] for node in corners], axis=0)
        if np.linalg.norm(centroid - centre) <= .35:
            core_quality.append(float(measurements["quality_by_cell"][cell]))
            core_triangles += int(cell in mesh.tris)
    host_id = next(cell for cell in mesh.elements_of_face[fixture.faces[0]]
                   if tuple(mesh.quads.get(cell) or mesh.tris[cell]) == coupling.plate_nodes)
    return {
        "equivalent_shell_cells": len(mesh.quads) + .5 * len(mesh.tris),
        "q8": len(mesh.quads), "t6": len(mesh.tris),
        "attachment_core_corner_count_0p35m": int(np.count_nonzero(distances <= .35)),
        "attachment_neighbour_corner_count_0p70m": int(np.count_nonzero(distances <= .70)),
        "nearest_corner_distance_m": float(distances.min()),
        "coupling_nonzero_weights": int(np.count_nonzero(np.abs(weights) > 1e-10)),
        "coupling_weight_l2": float(np.linalg.norm(weights)),
        "coupling_eccentricity_m": list(coupling.eccentricity),
        "attachment_host_normalized_jacobian": float(measurements["quality_by_cell"][host_id]),
        "core_cell_count_0p35m": len(core_quality),
        "core_t6_count_0p35m": core_triangles,
        "core_normalized_jacobian_min": float(min(core_quality)),
        "core_normalized_jacobian_median": float(np.median(core_quality)),
        "size_ratio_median": float(np.median(ratios)),
        "size_ratio_p95": float(np.quantile(ratios, .95)),
        "size_ratio_max": float(np.max(ratios)),
        "audit_failures": [row["name"] for row in measurements["checks"]
                           if row["status"] != "passed"],
    }


def run(baseline: Path) -> Path:
    if not (baseline / "manifest.json").is_file():
        raise ValueError("baseline must have a sealed manifest")
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, expected in manifest.items()
           if not (baseline / name).is_file() or digest(baseline / name) != expected]
    if bad:
        raise ValueError(f"sealed baseline has {len(bad)} missing or changed files")
    folder = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                       + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    before = identities()
    write(folder / "environment.json", before)
    rows = {}
    for case in CASES:
        fixture = build(case, True)
        candidate_path = _artifact(baseline, f"{case}-L2-g0-b1_mesh", "mesh.json")
        candidate = mesh_from_dict(json.loads(candidate_path.read_text(encoding="utf-8")))
        reference = structured(fixture, .15)
        reference_path = folder / f"reference-{case}.json"
        write(reference_path, mesh_to_dict(reference))
        rows[case] = {
            "candidate_mesh_path": str(candidate_path),
            "candidate_mesh_sha256": digest(candidate_path),
            "same_size_reference_mesh_sha256": digest(reference_path),
            "candidate": _stats(fixture, candidate, .15),
            "same_size_reference": _stats(fixture, reference, .15),
        }
    after = identities()
    write(folder / "final-environment.json", after)
    source_stable = all(
        before["sources"][name]["head"] == after["sources"][name]["head"]
        and before["sources"][name]["files"] == after["sources"][name]["files"]
        for name in before["sources"]
    )
    write(folder / "summary.json", {
        "kind": "diagnostic-not-SG1-acceptance", "baseline": str(baseline),
        "baseline_manifest_sha256": digest(baseline / "manifest.json"),
        "target_size_m": .15, "source_stable": source_stable, "cases": rows,
    })
    write(folder / "manifest.json", {
        path.name: digest(path) for path in sorted(folder.iterdir())
        if path.is_file() and path.name != "manifest.json"
    })
    return folder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    folder = run(args.baseline.resolve())
    print(folder, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
