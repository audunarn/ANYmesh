"""Bounded, non-acceptance SG1 point-coupling attribution experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from .consumer import import_model, probe, solve
from .fixtures import build, generate
from .measure import audit, physical_edges, samples
from .reference import structured
from .runner import (
    ROOT, compare_results, digest, identities, pin_workers, supervise, write,
    workers_stable,
)


OUTPUT = ROOT / "reports/quad_first/structural-gate/point-coupling/attempts"
CASES = ("CY", "CO", "R", "C")


def _artifact(root: Path, suffix: str, filename: str) -> Path:
    matches = list(root.glob(f"*-{suffix}/{filename}"))
    if len(matches) != 1:
        raise ValueError(f"expected one sealed artifact for {suffix}/{filename}")
    return matches[0]


def _stored(root: Path, suffix: str, filename: str) -> tuple[object, str]:
    path = _artifact(root, suffix, filename)
    return json.loads(path.read_text(encoding="utf-8")), digest(path)


def _energy_parts(fixture, mesh, result) -> dict[str, float]:
    model = import_model(fixture, mesh)
    dof_count = max(dof for node in model.mesh.nodes.values() for dof in node.dofs) + 1
    displacement = np.zeros(dof_count)
    for node_id, node in model.mesh.nodes.items():
        displacement[node.dofs] = result["displacement"][str(node_id)]
    parts = {"shell": 0.0, "beam": 0.0}
    for element_id, element in model.mesh.elements.items():
        if element_id in mesh.couplings:
            continue
        stiffness = element.compute_stiffness_matrix(
            model.mesh, model.get_material(element.material_name)
        )
        element_dofs = element.get_dof_mapping(model.mesh)
        local = displacement[element_dofs]
        energy = 0.5 * float(local @ stiffness @ local)
        parts["beam" if element_id in mesh.beams else "shell"] += energy
    parts["sum"] = parts["shell"] + parts["beam"]
    parts["sum_relative_error"] = abs(parts["sum"] - result["energy"]) / max(
        abs(result["energy"]), 1e-30
    )
    return parts


def _equivalent_shell_force(fixture, mesh, member_result) -> dict[str, object]:
    from anysolver import ResourceConfig, solve_linear
    from anysolver.boundary import LoadCase

    coupling = next(iter(mesh.couplings.values()))
    force = 100.0 * fixture.member_axis
    moment = np.cross(np.asarray(coupling.eccentricity), force)
    load = LoadCase("diagnostic equivalent shell load")
    for node_id, weight in zip(coupling.plate_nodes, coupling.weights):
        load.add_nodal_load(
            node_id, [*(weight * force), *(weight * moment)]
        )
    model = import_model(fixture, mesh)
    displacement, info = solve_linear(
        model, load,
        resource_config=ResourceConfig(solver_threads=1, assembly_threads=1),
    )
    actual = probe(fixture, mesh, model, displacement)
    deltas = {}
    for face_id in fixture.faces:
        name = str(face_id)
        baseline = np.asarray(member_result["probes"][name])
        observed = np.asarray(actual[name])
        absolute = float(np.linalg.norm(observed - baseline))
        deltas[name] = {
            "absolute_m": absolute,
            "relative": absolute / max(float(np.linalg.norm(baseline)), 1e-12),
        }
    return {
        "probe_deltas": deltas,
        "converged": info["convergence_info"]["status"] == "converged",
        "max_relative": max(item["relative"] for item in deltas.values()),
    }


def _host(fixture, mesh, graded: bool) -> dict[str, object]:
    from anymesher.quad.high_order import evaluate_mapping

    coupling = next(iter(mesh.couplings.values()))
    owner = next(
        element_id for element_id, body in {**mesh.quads, **mesh.tris}.items()
        if tuple(body) == tuple(coupling.plate_nodes)
    )
    body = mesh.quads.get(owner, mesh.tris.get(owner))
    corner_count = 4 if len(body) == 8 else 3
    xyz = np.asarray([mesh.nodes[node] for node in body])
    lengths = [item[0] for item in physical_edges(
        xyz, corner_count, fixture.field(0.15, graded)
    )]
    quality = evaluate_mapping(
        xyz, "Q8" if corner_count == 4 else "T6",
        samples(corner_count == 3),
    )
    return {
        "element_id": owner,
        "family": "Q8" if corner_count == 4 else "T6",
        "physical_edge_lengths_m": lengths,
        "minimum_normalized_jacobian": float(np.min(quality.normalized_quality)),
        "weights": list(coupling.weights),
        "eccentricity_m": list(coupling.eccentricity),
    }


def _absolute_probe_errors(candidate, reference) -> dict[str, dict[str, float]]:
    result = {}
    for name, expected in reference["probes"].items():
        observed = np.asarray(candidate["probes"][name])
        expected = np.asarray(expected)
        absolute = float(np.linalg.norm(observed - expected))
        result[name] = {
            "absolute_m": absolute,
            "reference_norm_m": float(np.linalg.norm(expected)),
            "relative": absolute / max(float(np.linalg.norm(expected)), 1e-12),
        }
    return result


def _counts(mesh) -> dict[str, float | int]:
    return {
        "quads": len(mesh.quads), "tris": len(mesh.tris),
        "equivalent_shell_cells": len(mesh.quads) + 0.5 * len(mesh.tris),
        "beams": len(mesh.beams), "couplings": len(mesh.couplings),
        "nodes": len(mesh.nodes),
    }


def diagnose_case(case: str, baseline: Path) -> dict[str, object]:
    from anymesher.serialize import mesh_from_dict

    fixture = build(case, True)
    reference, reference_hash = _stored(
        baseline, f"reference_{case}_b1_r1_member", "result.json"
    )
    reference_mesh_raw, reference_mesh_hash = _stored(
        baseline, f"reference_{case}_b1_r1", "mesh.json"
    )
    reference_mesh = mesh_from_dict(reference_mesh_raw)
    reference_parts = _energy_parts(fixture, reference_mesh, reference)
    same_h_mesh = structured(fixture, 0.15)
    same_h_result = solve(fixture, same_h_mesh, "member")
    same_h_parts = _energy_parts(fixture, same_h_mesh, same_h_result)
    rows = {}
    inputs = {
        "fine_reference_result_sha256": reference_hash,
        "fine_reference_mesh_sha256": reference_mesh_hash,
    }
    for graded in (False, True):
        row = f"{case}-L2-g{int(graded)}-b1"
        mesh_raw, mesh_hash = _stored(baseline, row + "_mesh", "mesh.json")
        result, result_hash = _stored(baseline, row + "_member", "result.json")
        mesh = mesh_from_dict(mesh_raw)
        parts = _energy_parts(fixture, mesh, result)
        equivalent = _equivalent_shell_force(fixture, mesh, result)
        rows[str(int(graded))] = {
            "counts": _counts(mesh), "host": _host(fixture, mesh, graded),
            "fine_reference_errors": compare_results(result, reference),
            "absolute_probe_errors": _absolute_probe_errors(result, reference),
            "energy_parts_j": parts,
            "equivalent_shell_force": equivalent,
            "source_hashes": {
                "candidate_mesh_sha256": mesh_hash,
                "candidate_result_sha256": result_hash,
            },
        }
    diagnostic_mesh = generate(fixture, 0.075, False)
    diagnostic_audit = audit(fixture, diagnostic_mesh, 0.075, False)
    diagnostic_result = solve(fixture, diagnostic_mesh, "member")
    audit_failures = [
        item["name"] for item in diagnostic_audit["checks"]
        if item["status"] != "passed"
    ]
    checks = {
        "fine_reference_energy_decomposes": reference_parts["sum_relative_error"] <= 1e-8,
        "same_h_energy_decomposes": same_h_parts["sum_relative_error"] <= 1e-8,
        "smaller_quad_first_audit": not audit_failures,
    }
    for graded in (False, True):
        row = rows[str(int(graded))]
        checks[f"candidate_g{int(graded)}_energy_decomposes"] = (
            row["energy_parts_j"]["sum_relative_error"] <= 1e-8
        )
        checks[f"candidate_g{int(graded)}_equivalent_shell_transfer"] = (
            row["equivalent_shell_force"]["converged"]
            and row["equivalent_shell_force"]["max_relative"] <= 1e-8
        )
    return {
        "case": case,
        "diagnostic_status": "passed" if all(checks.values()) else "failed",
        "diagnostic_checks": checks,
        "reference": {"counts": _counts(reference_mesh), "energy_parts_j": reference_parts},
        "same_h_structured": {
            "counts": _counts(same_h_mesh), "energy_parts_j": same_h_parts,
            "fine_reference_errors": compare_results(same_h_result, reference),
            "absolute_probe_errors": _absolute_probe_errors(same_h_result, reference),
        },
        "candidate": rows,
        "smaller_quad_first_diagnostic": {
            "size_m": 0.075, "counts": _counts(diagnostic_mesh),
            "fine_reference_errors": compare_results(diagnostic_result, reference),
            "audit_failed_checks": audit_failures,
        },
        "baseline_hashes": inputs,
    }


def run(baseline: Path) -> Path:
    if not (baseline / "summary.json").is_file() or not (baseline / "manifest.json").is_file():
        raise ValueError("baseline must be a sealed SG1 attempt")
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "failed":
        raise ValueError("diagnostic baseline must be a failed formal attempt")
    baseline_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    bad_baseline_files = [
        name for name, expected in baseline_manifest.items()
        if not (baseline / name).is_file() or digest(baseline / name) != expected
    ]
    if bad_baseline_files:
        raise ValueError(f"sealed baseline has {len(bad_baseline_files)} missing or changed files")
    root = OUTPUT / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-diagnostic-" + uuid.uuid4().hex[:8]
    )
    root.mkdir(parents=True, exist_ok=False)
    print("ATTEMPT " + str(root), flush=True)
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONHASHSEED"] = "0"
    before = identities()
    workers = pin_workers(env)
    write(root / "environment.json", {**before, **workers})
    write(root / "baseline.json", {
        "path": str(baseline),
        "verified_manifest_entries": len(baseline_manifest),
        "summary_sha256": digest(baseline / "summary.json"),
        "manifest_sha256": digest(baseline / "manifest.json"),
    })
    deadline = time.monotonic() + 3600
    observations = {}
    for case in CASES:
        folder = root / case
        folder.mkdir()
        observation = supervise(
            [sys.executable, "-m", "benchmarks.sg1.point_coupling_diagnostic",
             "--child-case", case, "--baseline", str(baseline),
             "--output", str(folder / "result.json")],
            folder, env, deadline, limit=600,
        )
        observations[case] = observation
        write(folder / "supervisor.json", observation)
        print(case + ": " + observation["status"], flush=True)
    after = identities()
    stable = all(
        before["sources"][name]["head"] == after["sources"][name]["head"]
        and before["sources"][name]["files"] == after["sources"][name]["files"]
        for name in before["sources"]
    )
    write(root / "final-environment.json", after)
    status = (
        "incomplete" if any(item["status"] == "incomplete" for item in observations.values())
        else "failed" if not stable or not workers_stable(workers)
        or any(item["status"] != "passed" for item in observations.values())
        else "complete"
    )
    write(root / "summary.json", {
        "status": status, "kind": "diagnostic-not-SG1-acceptance",
        "baseline_status": summary["status"], "observations": observations,
        "source_stable": stable, "workers_stable": workers_stable(workers),
    })
    write(root / "manifest.json", {
        str(path.relative_to(root)).replace("\\", "/"): digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    })
    print(json.dumps({"attempt": str(root), "status": status}), flush=True)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--child-case", choices=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    if args.child_case:
        if args.output is None:
            parser.error("--output is required for a child")
        result = diagnose_case(args.child_case, baseline)
        write(args.output, result)
        return 0 if result["diagnostic_status"] == "passed" else 1
    root = run(baseline)
    return 0 if json.loads((root / "summary.json").read_text())["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
