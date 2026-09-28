"""Full B3/MPC point-placement control on a fixed structured shell topology.

Development experiment only. This keeps every shell and beam cell identity and
boundary node while moving interior shell nodes, then reconstructs the point
coupling from the declared source location before the legacy solver consumes it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np

from anymesher.mesh import Coupling
from anymesher.mesh_bvh import MeshElementBVH
from anymesher.serialize import mesh_to_dict

from .attachment_accuracy_study import _shifted_interior
from .consumer import solve
from .fixtures import build
from .measure import audit
from .reference import structured
from .runner import ROOT, compare_results, digest, identities, write


OUTPUT = ROOT / "reports/quad_first/structural-gate/matched-member-topology"
CASES = ("CY", "CO", "R", "C")


def run(case: str, baseline: Path) -> Path:
    baseline = baseline.resolve()
    sealed = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if sealed["status"] != "failed" or sealed["mode"] != "formal":
        raise ValueError("requires the failed sealed formal attempt")
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, expected in manifest.items()
           if not (baseline / name).is_file() or digest(baseline / name) != expected]
    if bad:
        raise ValueError(f"baseline has {len(bad)} missing or changed files")
    before = identities()
    fixture = build(case, True)
    original = structured(fixture, .15)
    shifted, moved = _shifted_interior(fixture, original)
    if (original.quads != shifted.quads or original.tris != shifted.tris
            or original.beams != shifted.beams
            or original.nodes_of_edge != shifted.nodes_of_edge):
        raise ValueError("matched-topology control changed connectivity or boundary chains")
    coupling_id, old = next(iter(shifted.couplings.items()))
    source_point = fixture.patches[fixture.faces[0]].point(.5, .5)
    face_elements = shifted.elements_of_face[fixture.faces[0]]
    hit = MeshElementBVH(shifted, element_ids=face_elements).locate(
        source_point, element_ids=face_elements, tolerance=.003,
    )
    if hit is None:
        raise ValueError("shifted shell does not support declared attachment point")
    owner, body, weights, projected = (
        hit.element_id, hit.node_ids, hit.weights, hit.point,
    )
    target = np.asarray(shifted.nodes[old.beam_node])
    shifted.couplings[coupling_id] = Coupling(
        old.beam_node, tuple(body), tuple(float(w) for w in weights),
        tuple(float(x) for x in target - projected),
    )
    anchor_distance = min(
        float(np.linalg.norm(shifted.nodes[node] - source_point))
        for node in shifted.quads[owner][:4]
    )
    if anchor_distance <= 1e-6 or max(abs(np.asarray(weights))) >= 1 - 1e-9:
        raise ValueError("shift did not move the point inside a Q8")
    unshifted_audit = audit(fixture, original, .15, False)
    shifted_audit = audit(fixture, shifted, .15, False)
    original_result = solve(fixture, original, "member")
    shifted_result = solve(fixture, shifted, "member")
    fine_paths = tuple(baseline.glob(f"*-reference_{case}_b1_r1_member/result.json"))
    if len(fine_paths) != 1:
        raise ValueError("missing fine sealed reference")
    fine = json.loads(fine_paths[0].read_text(encoding="utf-8"))
    after = identities()
    if before["sources"] != after["sources"]:
        raise RuntimeError("source files changed during matched-topology control")
    folder = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                       + "-" + case + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "environment.json", before)
    write(folder / "baseline.json", {
        "path": str(baseline), "verified_manifest_entries": len(manifest),
        "manifest_sha256": digest(baseline / "manifest.json"),
        "fine_result_sha256": digest(fine_paths[0]),
    })
    write(folder / "unshifted-mesh.json", mesh_to_dict(original))
    write(folder / "shifted-mesh.json", mesh_to_dict(shifted))
    write(folder / "unshifted-audit.json", unshifted_audit)
    write(folder / "shifted-audit.json", shifted_audit)
    write(folder / "unshifted-result.json", original_result)
    write(folder / "shifted-result.json", shifted_result)
    write(folder / "summary.json", {
        "kind": "diagnostic-not-SG1-acceptance", "case": case,
        "target_size_m": .15, "moved_interior_shell_nodes": moved,
        "shell_cells": len(original.quads) + .5 * len(original.tris),
        "attachment_host": owner, "attachment_nearest_corner_m": anchor_distance,
        "attachment_weights": list(weights),
        "unshifted_errors": compare_results(original_result, fine),
        "shifted_errors": compare_results(shifted_result, fine),
        "unshifted_mesh_failures": [c["name"] for c in unshifted_audit["checks"]
                                    if c["status"] != "passed"],
        "shifted_mesh_failures": [c["name"] for c in shifted_audit["checks"]
                                  if c["status"] != "passed"],
        "unshifted_solver_failures": [c["name"] for c in original_result["checks"]
                                      if c["status"] != "passed"],
        "shifted_solver_failures": [c["name"] for c in shifted_result["checks"]
                                    if c["status"] != "passed"],
    })
    write(folder / "final-environment.json", after)
    write(folder / "manifest.json", {
        path.name: digest(path) for path in sorted(folder.iterdir()) if path.is_file()
    })
    return folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.case, args.baseline))


if __name__ == "__main__":
    main()
