"""Replay the sealed SG1 point-attachment failures without new solves.

This is development evidence, not an SG1 acceptance operation. In particular,
the beam-tip decomposition uses stored six-DOF solutions and does not infer
accuracy from the MPC equations used to obtain them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.optimize import least_squares

from anymesher.quad.high_order import shape_values
from anymesher.serialize import mesh_from_dict

from .fixtures import build
from .runner import ROOT, digest, identities, write


CASES = ("CY", "CO", "R", "C")
OUTPUT = ROOT / "reports/quad_first/structural-gate/causal-replay"


def _read(root: Path, suffix: str, filename: str):
    paths = tuple(root.glob(f"*-{suffix}/{filename}"))
    if len(paths) != 1:
        raise ValueError(f"expected one sealed {suffix}/{filename}; found {len(paths)}")
    return json.loads(paths[0].read_text(encoding="utf-8")), digest(paths[0])


def _components(fixture, mesh, result):
    if len(mesh.couplings) != 1:
        raise ValueError("point replay requires exactly one coupling")
    coupling = next(iter(mesh.couplings.values()))
    displacement = result["displacement"]
    weights = np.asarray(coupling.weights, dtype=float)
    shell = np.asarray([displacement[str(n)] for n in coupling.plate_nodes])
    shell_u = weights @ shell[:, :3]
    shell_theta = weights @ shell[:, 3:]
    eccentricity = np.asarray(coupling.eccentricity)
    slave = np.asarray(displacement[str(coupling.beam_node)])
    end_node = mesh.node_of_vertex[fixture.member_end]
    end = np.asarray(displacement[str(end_node)])
    lever = np.asarray(mesh.nodes[end_node]) - np.asarray(mesh.nodes[coupling.beam_node])
    point_u = shell_u + np.cross(shell_theta, eccentricity)
    rigid_arm_u = np.cross(shell_theta, lever)
    beam_deformation = end[:3] - slave[:3] - np.cross(slave[3:], lever)
    reconstructed = point_u + rigid_arm_u + beam_deformation
    return {
        "shell_translation_m": shell_u.tolist(),
        "shell_rotation_rad": shell_theta.tolist(),
        "eccentric_translation_m": np.cross(shell_theta, eccentricity).tolist(),
        "attachment_translation_m": point_u.tolist(),
        "member_rigid_arm_translation_m": rigid_arm_u.tolist(),
        "member_deformation_m": beam_deformation.tolist(),
        "tip_translation_m": end[:3].tolist(),
        "tip_reconstruction_error_m": float(np.linalg.norm(reconstructed - end[:3])),
        "mpc_translation_error_m": float(np.linalg.norm(point_u - slave[:3])),
        "mpc_rotation_error_rad": float(np.linalg.norm(shell_theta - slave[3:])),
        "member_axial_deformation_m": float(np.dot(beam_deformation, fixture.member_axis)),
        "member_transverse_deformation_m": float(np.linalg.norm(
            beam_deformation - np.dot(beam_deformation, fixture.member_axis) * fixture.member_axis
        )),
        "positive_member_length_m": float(np.linalg.norm(lever)),
    }


def _probe_geometry(fixture, mesh, result):
    """Bound effect of evaluating source UV versus closest interpolated XYZ."""
    values = {}
    for face_id, patch in fixture.patches.items():
        point = np.asarray(patch.point(.5, .5))
        candidates = []
        for eid in mesh.elements_of_face[face_id]:
            body = mesh.quads.get(eid, mesh.tris.get(eid))
            family = "Q8" if len(body) == 8 else "T6"
            xyz = np.asarray([mesh.nodes[n] for n in body])
            if min(np.linalg.norm(node - point) for node in xyz) > .5:
                continue
            fit = least_squares(
                lambda q: shape_values(family, q) @ xyz - point,
                (0., 0.) if family == "Q8" else (1 / 3, 1 / 3),
                gtol=1e-13, xtol=1e-13, ftol=1e-13,
            )
            inside = (max(abs(fit.x)) <= 1 + 1e-8 if family == "Q8"
                      else min(fit.x) >= -1e-8 and sum(fit.x) <= 1 + 1e-8)
            if inside:
                weights = shape_values(family, fit.x)
                displacement = weights @ np.asarray(
                    [result["displacement"][str(n)][:3] for n in body]
                )
                candidates.append((float(np.linalg.norm(fit.fun)), int(eid), displacement))
        if not candidates:
            raise ValueError(f"no physical probe candidate for face {face_id}")
        distance, eid, physical_u = min(candidates, key=lambda row: (row[0], row[1]))
        source_u = np.asarray(result["probes"][str(face_id)])
        values[str(face_id)] = {
            "element": eid,
            "element_to_source_point_m": distance,
            "physical_minus_source_probe_m": float(np.linalg.norm(physical_u - source_u)),
            "physical_minus_source_probe_relative": float(
                np.linalg.norm(physical_u - source_u) / max(np.linalg.norm(source_u), 1e-12)
            ),
        }
    return values


def _delta(candidate, reference):
    ref_tip = np.asarray(reference["tip_translation_m"])
    denominator = max(float(np.linalg.norm(ref_tip)), 1e-12)
    fields = ("shell_translation_m", "eccentric_translation_m",
              "member_rigid_arm_translation_m", "member_deformation_m",
              "tip_translation_m")
    return {field: {
        "absolute_m": float(np.linalg.norm(np.asarray(candidate[field]) - reference[field])),
        "relative_to_reference_tip": float(
            np.linalg.norm(np.asarray(candidate[field]) - reference[field]) / denominator
        ),
    } for field in fields}


def _algebra_checks(component):
    expected_axial = 100.0 * component["positive_member_length_m"] / (
        210e9 * .05**2
    )
    return {
        "tip_reconstruction": component["tip_reconstruction_error_m"] <= 1e-12,
        "mpc_translation": component["mpc_translation_error_m"] <= 1e-12,
        "mpc_rotation": component["mpc_rotation_error_rad"] <= 1e-12,
        "beam_axial_control": abs(
            component["member_axial_deformation_m"] / expected_axial - 1
        ) <= 1e-6,
        "beam_transverse_control": component["member_transverse_deformation_m"] <= 1e-12,
    }


def replay(baseline: Path):
    baseline = baseline.resolve()
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "failed":
        raise ValueError("baseline must be a failed sealed formal attempt")
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, sha in manifest.items()
           if not (baseline / name).is_file() or digest(baseline / name) != sha]
    if bad:
        raise ValueError(f"baseline has {len(bad)} missing or changed files")
    before = identities()
    cases = {}
    artifact_hashes = {}
    for case in CASES:
        fixture = build(case, True)
        raw, artifact_hashes[f"{case}/reference_mesh"] = _read(
            baseline, f"reference_{case}_b1_r1", "mesh.json")
        result, artifact_hashes[f"{case}/reference_result"] = _read(
            baseline, f"reference_{case}_b1_r1_member", "result.json")
        reference_mesh = mesh_from_dict(raw)
        reference = _components(fixture, reference_mesh, result)
        rows = {}
        for graded in (0, 1):
            key = f"{case}-L2-g{graded}-b1"
            raw, artifact_hashes[f"{key}/mesh"] = _read(baseline, key + "_mesh", "mesh.json")
            result, artifact_hashes[f"{key}/result"] = _read(baseline, key + "_member", "result.json")
            mesh = mesh_from_dict(raw)
            component = _components(fixture, mesh, result)
            rows[str(graded)] = {
                "components": component,
                "algebra_checks": _algebra_checks(component),
                "difference_from_reference": _delta(component, reference),
                "source_vs_physical_probes": _probe_geometry(fixture, mesh, result),
            }
        cases[case] = {
            "reference": reference,
            "reference_algebra_checks": _algebra_checks(reference),
            "candidate": rows,
        }
    failed_checks = [f"{case}/{kind}/{name}"
                     for case, body in cases.items()
                     for kind, checks in (
                         ("reference", body["reference_algebra_checks"]),
                         *((f"candidate_g{g}", row["algebra_checks"])
                           for g, row in body["candidate"].items()),
                     )
                     for name, passed in checks.items() if not passed]
    after = identities()
    if before["sources"] != after["sources"]:
        raise RuntimeError("source files changed during causal replay")
    folder = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                       + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "environment.json", before)
    write(folder / "baseline.json", {
        "path": str(baseline), "manifest_sha256": digest(baseline / "manifest.json"),
        "verified_manifest_entries": len(manifest), "artifact_hashes": artifact_hashes,
    })
    write(folder / "results.json", cases)
    write(folder / "checks.json", {
        "status": "passed" if not failed_checks else "failed",
        "count": sum(len(body["reference_algebra_checks"])
                     + sum(len(row["algebra_checks"])
                           for row in body["candidate"].values())
                     for body in cases.values()),
        "failed": failed_checks,
    })
    write(folder / "final-environment.json", after)
    write(folder / "manifest.json", {
        str(path.relative_to(folder)).replace("\\", "/"): digest(path)
        for path in sorted(folder.iterdir()) if path.is_file()
    })
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    folder = replay(args.baseline)
    print(folder)
    checks = json.loads((folder / "checks.json").read_text(encoding="utf-8"))
    if checks["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
