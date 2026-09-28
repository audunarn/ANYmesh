"""Separate, non-acceptance study of shell topology and localized loading."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from scipy.optimize import least_squares

from anymesher.quad.high_order import (
    certify_mapping_validity, evaluate_mapping, shape_values,
)
from anymesher.serialize import mesh_from_dict

from .consumer import import_model, probe
from .fixtures import build
from .measure import quadrature
from .point_coupling_diagnostic import _counts
from .reference import structured
from .runner import ROOT, digest, identities, pin_workers, supervise, write, workers_stable


CASES = ("CY", "CO", "R", "C")
LOADS = {"point_center": (.5, .5), "point_offset": (.57, .43),
         "gaussian_020m": (.5, .5)}
OUTPUT = ROOT / "reports/quad_first/structural-gate/attachment-accuracy/attempts"


def _shape_at(fixture, mesh, uv):
    """Locate a declared source-chart point; never select the nearest node."""
    face = fixture.faces[0]
    patch = fixture.patches[face]
    target = np.asarray(uv, dtype=float)
    hits = []
    for eid in mesh.elements_of_face[face]:
        body = mesh.quads.get(eid, mesh.tris.get(eid))
        family = "Q8" if len(body) == 8 else "T6"
        node_uv = np.asarray([patch.uv(mesh.nodes[node]) for node in body])
        if np.any(target < node_uv.min(axis=0) - 1e-8) or np.any(target > node_uv.max(axis=0) + 1e-8):
            continue
        initial = [0., 0.] if family == "Q8" else [1/3, 1/3]
        fit = least_squares(lambda q: shape_values(family, q) @ node_uv - target,
                            initial, gtol=1e-13, xtol=1e-13, ftol=1e-13)
        inside = (np.max(np.abs(fit.x)) <= 1 + 1e-8 if family == "Q8"
                  else np.min(fit.x) >= -1e-8 and np.sum(fit.x) <= 1 + 1e-8)
        if inside and np.linalg.norm(fit.fun) <= 1e-9:
            weights = np.asarray(shape_values(family, fit.x), dtype=float).reshape(-1)
            xyz = weights @ np.asarray([mesh.nodes[node] for node in body])
            hits.append((eid, body, weights, xyz))
    if not hits:
        raise ValueError(f"no containing shell cell for {uv}")
    if any(np.linalg.norm(hit[3] - hits[0][3]) > 1e-8 for hit in hits[1:]):
        raise ValueError("discontinuous source point geometry")
    return hits[0]


def _load(fixture, mesh, kind):
    face = fixture.faces[0]
    patch = fixture.patches[face]
    uv = LOADS[kind]
    target = patch.point(*uv)
    direction = patch.normal(*uv)
    force = 100.0 * direction
    nodal = {node: np.zeros(6) for node in mesh.nodes}
    eid, body, weights, located = _shape_at(fixture, mesh, uv)
    support_error = float(np.linalg.norm(located - target))
    if support_error > .003:
        raise ValueError("sampled shell misses declared source load point")
    if kind.startswith("point_"):
        for node, weight in zip(body, weights):
            nodal[node][:3] += weight * force
        quadrature_relative_l1 = None
    else:
        sigma = .2
        def integrate(order):
            values = {node: 0.0 for node in mesh.nodes}
            integral = 0.0
            for cell_id in mesh.elements_of_face[face]:
                cell = mesh.quads.get(cell_id, mesh.tris.get(cell_id))
                family = "Q8" if len(cell) == 8 else "T6"
                q, qw = quadrature(len(cell) == 6, order=order)
                xyz = np.asarray([mesh.nodes[node] for node in cell])
                ev = evaluate_mapping(xyz, family, q)
                kernel = np.exp(-.5 * np.sum((ev.points - target)**2, axis=1) / sigma**2)
                coefficients = qw * ev.jacobian_magnitude * kernel
                contributions = shape_values(family, q).T @ coefficients
                for node, amount in zip(cell, contributions):
                    values[node] += float(amount)
                integral += float(np.sum(coefficients))
            if integral <= 0:
                raise ValueError("empty physical load footprint")
            return {node: 100.0 * amount / integral for node, amount in values.items()}

        coarse, fine = integrate(10), integrate(16)
        quadrature_relative_l1 = float(
            sum(abs(fine[node] - coarse[node]) for node in fine)
            / max(sum(abs(value) for value in fine.values()), 1e-30)
        )
        if quadrature_relative_l1 > 1e-5:
            raise ValueError("physical-footprint load quadrature did not converge")
        for node, amount in fine.items():
            nodal[node][:3] = amount * direction
    resultant = sum((value[:3] for value in nodal.values()), np.zeros(3))
    if np.linalg.norm(resultant - force) > 1e-8:
        raise ValueError("shell load resultant differs from 100 N")
    active = int(sum(np.linalg.norm(value[:3]) > 1e-9 for value in nodal.values()))
    centroid = sum((mesh.nodes[node] * float(value[:3] @ direction)
                    for node, value in nodal.items()), np.zeros(3)) / 100.0
    return nodal, (eid, body, weights), {
        "source_uv": list(uv), "source_xyz_m": target.tolist(),
        "mapped_source_error_m": support_error,
        "direction": direction.tolist(), "resultant_n": resultant.tolist(),
        "active_nodes": active, "host_family": "Q8" if len(body) == 8 else "T6",
        "force_centroid_m": centroid.tolist(),
        "quadrature_relative_l1": quadrature_relative_l1,
        "host_cell": eid, "host_weight_l1": float(np.sum(np.abs(weights))),
        "host_nonzero_weights": int(np.sum(np.abs(weights) > 1e-8)),
    }


def _solve(fixture, mesh, kind):
    from anysolver import ResourceConfig, solve_linear
    from anysolver.boundary import LoadCase

    model = import_model(fixture, mesh)
    nodal, host, load_info = _load(fixture, mesh, kind)
    load = LoadCase("attachment accuracy " + kind)
    for node, value in nodal.items():
        if np.any(value):
            load.add_nodal_load(node, value.tolist())
    u, info = solve_linear(model, load,
                           resource_config=ResourceConfig(solver_threads=1,
                                                          assembly_threads=1))
    if info["convergence_info"]["status"] != "converged" or not np.isfinite(u).all():
        raise ValueError("shell solve did not converge to finite displacement")
    work = sum(float(nodal[node] @ u[model.mesh.nodes[node].dofs]) for node in mesh.nodes)
    energy = .5 * work
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("non-positive shell energy")
    _, body, weights = host
    loaded = sum((weight * u[model.mesh.nodes[node].dofs[:3]]
                  for node, weight in zip(body, weights)), np.zeros(3))
    return {"energy_j": energy, "loaded_point_displacement_m": loaded.tolist(),
            "fixed_source_probes_m": probe(fixture, mesh, model, u),
            "dofs": len(u), "load": load_info}


def _errors(candidate, reference):
    result = {"energy": abs(candidate["energy_j"] - reference["energy_j"])
              / max(abs(reference["energy_j"]), 1e-30)}
    for key in ("loaded_point_displacement_m",):
        a, b = np.asarray(candidate[key]), np.asarray(reference[key])
        result[key] = float(np.linalg.norm(a-b) / max(np.linalg.norm(b), 1e-12))
    for name, value in reference["fixed_source_probes_m"].items():
        a, b = np.asarray(candidate["fixed_source_probes_m"][name]), np.asarray(value)
        result["probe/" + name] = float(np.linalg.norm(a-b)
                                       / max(np.linalg.norm(b), 1e-12))
    return result


def _shifted_interior(fixture, mesh):
    """Keep connectivity and source-boundary stations; move interior Q8 nodes."""
    shifted = deepcopy(mesh)
    moved = set()
    for face, patch in fixture.patches.items():
        nodes = {node for eid in mesh.elements_of_face[face]
                 for node in mesh.quads.get(eid, mesh.tris.get(eid))}
        for node in nodes:
            if node in moved:
                continue
            u, v = patch.uv(mesh.nodes[node])
            if not (1e-8 < u < 1-1e-8 and 1e-8 < v < 1-1e-8):
                continue
            bump = .025 * np.sin(np.pi*u) * np.sin(np.pi*v)
            shifted.nodes[node] = patch.point(u+bump, v+bump)
            moved.add(node)
    if not moved:
        raise ValueError("matched-topology perturbation moved no interior nodes")
    for body in shifted.quads.values():
        certificate = certify_mapping_validity(
            np.asarray([shifted.nodes[node] for node in body]), "Q8")
        if getattr(certificate.status, "value", certificate.status) != "CERTIFIED_POSITIVE":
            raise ValueError("shifted structured Q8 lost positive mapping")
    return shifted, len(moved)


def diagnose(case, baseline):
    fixture = build(case, False)
    meshes = {}
    hashes = {}
    for graded in (False, True):
        row = f"{case}-L2-g{int(graded)}-b0_mesh"
        paths = list(baseline.glob(f"[0-9][0-9][0-9][0-9]-{row}/mesh.json"))
        if len(paths) != 1:
            raise ValueError(f"expected exactly one sealed mesh for {row}")
        raw = json.loads(paths[0].read_text(encoding="utf-8"))
        sha = digest(paths[0])
        meshes[f"quad_g{int(graded)}"] = mesh_from_dict(raw)
        hashes[row] = sha
    for size, label in ((.15, "structured_015"), (.075, "structured_0075"),
                        (.0375, "structured_00375")):
        meshes[label] = structured(fixture, size)
    meshes["structured_shifted_015"], moved = _shifted_interior(
        fixture, meshes["structured_015"])
    results = {}
    for name, mesh in meshes.items():
        results[name] = {"counts": _counts(mesh),
                         "loads": {kind: _solve(fixture, mesh, kind) for kind in LOADS}}
    reference = results["structured_00375"]["loads"]
    comparisons = {}
    for name, row in results.items():
        comparisons[name] = {kind: _errors(row["loads"][kind], reference[kind])
                             for kind in LOADS}
    return {"case": case, "status": "complete", "loads": list(LOADS),
            "sigma_m": .2, "force_n": 100., "source_mesh_hashes": hashes,
            "structured_shifted_interior_nodes": moved,
            "results": results, "relative_to_structured_00375": comparisons}


def run(baseline, cases):
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "failed":
        raise ValueError("expected failed sealed SG1 baseline")
    bad = [name for name, expected in manifest.items()
           if not (baseline / name).is_file() or digest(baseline / name) != expected]
    if bad:
        raise ValueError(f"baseline has {len(bad)} changed or missing files")
    root = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                     + "-study-" + uuid.uuid4().hex[:8])
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
    write(root / "baseline.json", {"path": str(baseline),
                                   "manifest_entries_verified": len(manifest),
                                   "manifest_sha256": digest(baseline / "manifest.json"),
                                   "summary_sha256": digest(baseline / "summary.json")})
    deadline = time.monotonic() + 5400
    observations = {}
    for case in cases:
        folder = root / case
        folder.mkdir()
        observation = supervise(
            [sys.executable, "-m", "benchmarks.sg1.attachment_accuracy_study",
             "--child-case", case, "--baseline", str(baseline),
             "--output", str(folder / "result.json")],
            folder, env, deadline, limit=1200)
        observations[case] = observation
        write(folder / "supervisor.json", observation)
        print(case + ": " + observation["status"], flush=True)
    after = identities()
    stable = all(before["sources"][name]["head"] == after["sources"][name]["head"]
                 and before["sources"][name]["files"] == after["sources"][name]["files"]
                 for name in before["sources"])
    write(root / "final-environment.json", after)
    status = ("incomplete" if any(x["status"] == "incomplete" for x in observations.values())
              else "failed" if not stable or not workers_stable(workers)
              or any(x["status"] != "passed" for x in observations.values())
              else "complete")
    write(root / "summary.json", {"status": status,
                                  "kind": "attachment-accuracy-study-not-SG1-acceptance",
                                  "cases": list(cases), "observations": observations,
                                  "source_stable": stable,
                                  "workers_stable": workers_stable(workers)})
    write(root / "manifest.json", {
        str(path.relative_to(root)).replace("\\", "/"): digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()})
    print(json.dumps({"attempt": str(root), "status": status}), flush=True)
    return root, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--child-case", choices=CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    if args.child_case:
        if args.output is None:
            parser.error("--output is required for a child")
        write(args.output, diagnose(args.child_case, baseline))
        return 0
    return 0 if run(baseline, args.cases)[1] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
