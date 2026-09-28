"""Serial, bounded and evidence-preserving IS1 acceptance attempt."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import psutil

from anygeometry import to_dict
from anymesher.serialize import mesh_from_dict
from benchmarks.is1.fixtures import CASES, SIZES, build
from benchmarks.sg1.measure import promotion_checks

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "quad_first" / "is1" / "attempts"
MESH_SECONDS = 120.
ATTEMPT_SECONDS = 30 * 60.
MEMORY_BYTES = 2 * 1024**3


def save(path, data):
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def _identity():
    repos = ("ANYmesh", "ANYgeometry", "ANYfem", "ANYstructure")
    revisions = {}
    for name in repos:
        repo = ROOT.parent / name
        git = ["git", "-c", f"safe.directory={repo.as_posix()}"]
        head = subprocess.run([*git, "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                              text=True, check=True).stdout.strip()
        changed = subprocess.run([*git, "status", "--porcelain"], cwd=repo,
                                 capture_output=True, text=True, check=True).stdout.splitlines()
        revisions[name] = {"head": head, "dirty_entries": len(changed)}
    source_hashes = {
        f"{name}/{path.relative_to(ROOT.parent / name).as_posix()}":
        hashlib.sha256(path.read_bytes()).hexdigest()
        for name in repos
        for path in sorted((ROOT.parent / name / ("anystruct" if name == "ANYstructure" else "src")).rglob("*.py"))
    }
    for path in (*sorted((ROOT / "benchmarks" / "is1").glob("*.py")),
                 ROOT / "benchmarks" / "sg1" / "fixtures.py",
                 ROOT / "benchmarks" / "sg1" / "measure.py"):
        source_hashes[f"ANYmesh/{path.relative_to(ROOT).as_posix()}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    imported = {name: str(importlib.import_module(name).__file__)
                for name in ("anygeometry", "anymesher")}
    workers = {}
    for name, relative in (
        ("ANYMESH_QUAD_MCF_WORKER", "third_party/quad/worker/out/lemon/quad_mcf_worker.exe"),
        ("ANYMESH_QUAD_TINYAD_WORKER", "third_party/quad/worker/out/tinyad/quad_tinyad_optimizer.exe"),
    ):
        path = ROOT / relative
        workers[name] = {"path": str(path), "available": path.is_file(),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version, "logical_cpus": os.cpu_count(),
            "revisions": revisions, "source_sha256": source_hashes,
            "imported": imported, "workers": workers,
            "limits": {"mesh_seconds": MESH_SECONDS, "memory_bytes": MEMORY_BYTES,
                       "shells": 1500, "dofs": 15000, "attempt_seconds": ATTEMPT_SECONDS,
                       "numerical_threads": 1}}


def _run(folder, environment, deadline):
    started = stage_started = time.monotonic()
    stage = None
    peak = 0
    reason = None
    with (folder / "stdout.log").open("x", encoding="utf-8") as stdout, (
         folder / "stderr.log").open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.is1.operation", str(folder)],
            cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        owned = psutil.Process(process.pid)
        while process.poll() is None:
            now = time.monotonic()
            progress = folder / "progress.json"
            if progress.exists():
                try:
                    current = json.loads(progress.read_text(encoding="utf-8"))["stage"]
                    if current != stage:
                        stage, stage_started = current, now
                except (OSError, ValueError, KeyError):
                    pass
            try:
                peak = max(peak, owned.memory_info().rss + sum(
                    child.memory_info().rss for child in owned.children(recursive=True)
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if peak > MEMORY_BYTES:
                reason = "2 GiB process memory limit"
            elif now - stage_started > MESH_SECONDS:
                reason = f"120 second limit in {stage or 'startup'}"
            elif now > deadline:
                reason = "30 minute attempt limit"
            if reason:
                for child in owned.children(recursive=True):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
                break
            time.sleep(.1)
    result_file = folder / "result.json"
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        result = {"status": "incomplete" if reason else "failed",
                  "reason": reason or f"worker exited {process.returncode}"}
    if reason:
        result = {**result, "status": "incomplete", "reason": reason}
    save(folder / "supervisor.json", {"exit_code": process.returncode,
         "reason": reason, "peak_rss_bytes": peak,
         "elapsed_seconds": time.monotonic()-started, "last_stage": stage})
    return result


def _checks(attempt, inventory):
    checks = []
    for case in CASES:
        for h in SIZES[case]:
            linear = attempt / f"{case}-h{h}-linear-uniform-adaptive" / "mesh.json"
            quadratic = attempt / f"{case}-h{h}-quadratic-uniform-adaptive" / "mesh.json"
            if not linear.is_file() or not quadratic.is_file():
                checks.append({"case": case, "h": h, "name": "promotion", "status": "blocked"})
            else:
                checks.extend({"case": case, "h": h, **item}
                              for item in promotion_checks(
                                  mesh_from_dict(json.loads(linear.read_text(encoding="utf-8"))),
                                  mesh_from_dict(json.loads(quadratic.read_text(encoding="utf-8")))
                              ))
        coarse, fine = SIZES[case]
        uniform = attempt / f"{case}-h{fine}-quadratic-uniform-adaptive" / "metrics.json"
        refined = attempt / f"{case}-h{fine}-quadratic-refined-adaptive" / "metrics.json"
        coarse_path = attempt / f"{case}-h{coarse}-quadratic-uniform-adaptive" / "metrics.json"
        if all(path.is_file() for path in (coarse_path, uniform, refined)):
            a, b, c = [json.loads(path.read_text(encoding="utf-8")) for path in (coarse_path, uniform, refined)]
            checks.append({"case": case, "name": "resolution-count",
                           "status": "passed" if b["counts"]["equivalent"] > a["counts"]["equivalent"] else "failed"})
            checks.append({"case": case, "name": "geometry-response",
                           "status": "passed" if a["max_geometry_error"] <= 1e-10 or b["max_geometry_error"] < a["max_geometry_error"] else "failed"})
            more = (c.get("core_corner_count") is not None and b.get("core_corner_count") is not None
                    and c["core_corner_count"] > b["core_corner_count"])
            smaller = (c.get("core_median") is not None and b.get("core_median") is not None
                       and c["core_median"] < b["core_median"])
            remote = (c.get("remote_median") is not None and b.get("remote_median") is not None
                      and abs(c["remote_median"] / b["remote_median"] - 1.) <= .2)
            checks.append({"case": case, "name": "refinement-causality",
                           "status": "passed" if more and smaller and remote else "failed",
                           "core_nodes": (b.get("core_corner_count"), c.get("core_corner_count")),
                           "core_medians": (b.get("core_median"), c.get("core_median")),
                           "remote_medians": (b.get("remote_median"), c.get("remote_median"))})
        else:
            checks.append({"case": case, "name": "resolution-and-refinement", "status": "blocked"})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehearsal-limit", type=int, default=None)
    arguments = parser.parse_args()
    identity = _identity()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = REPORTS / f"{timestamp}-adaptive-{hashlib.sha256(json.dumps(identity['source_sha256'],sort_keys=True).encode()).hexdigest()[:8]}"
    attempt.mkdir(parents=True, exist_ok=False)
    save(attempt / "environment.json", identity)
    fixture_root = attempt / "fixtures"
    fixture_root.mkdir()
    frozen = {}
    for case in CASES:
        geometry, faces, center, _ = build(case, SIZES[case][0])
        path = fixture_root / f"{case}.json"
        save(path, {"geometry": to_dict(geometry), "faces": faces, "center": center})
        frozen[case] = path
    inventory = []
    for case in CASES:
        for h in SIZES[case]:
            for order in ("linear", "quadratic"):
                inventory.append((case, h, order, False, "adaptive"))
        inventory.append((case, SIZES[case][1], "quadratic", True, "adaptive"))
        inventory.append((case, SIZES[case][1], "quadratic", False, "existing"))
    if arguments.rehearsal_limit is not None:
        if arguments.rehearsal_limit < 1:
            parser.error("rehearsal limit must be positive")
        inventory = inventory[:arguments.rehearsal_limit]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((
        str(ROOT), str(ROOT / "src"), str(ROOT.parent / "ANYgeometry" / "src"),
        environment.get("PYTHONPATH", ""),
    ))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    for name, worker in identity["workers"].items():
        if worker["available"]:
            environment[name] = worker["path"]
    deadline = time.monotonic() + ATTEMPT_SECONDS
    dispositions = {}
    for case, h, order, refined, layout in inventory:
        label = f"{case}-h{h}-{order}-{'refined' if refined else 'uniform'}-{layout}"
        folder = attempt / label
        spec = {"case": case, "h": h, "order": order, "refined": refined,
                "layout": layout, "fixture": str(frozen[case]),
                "overrides": build(case, h)[3]}
        folder.mkdir()
        save(folder / "input.json", spec)
        result = (_run(folder, environment, deadline) if time.monotonic() < deadline
                  else {"status": "incomplete", "reason": "30 minute attempt limit before start"})
        dispositions[label] = result
        print(f"{label}: {result['status']}", flush=True)
    checks = ([] if arguments.rehearsal_limit is not None
              else _checks(attempt, inventory))
    save(attempt / "crosschecks.json", checks)
    save(attempt / "case-dispositions.json", dispositions)
    manifest = {str(path.relative_to(attempt)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(attempt.rglob("*")) if path.is_file() and path.name != "hashes.json"}
    save(attempt / "hashes.json", manifest)
    counts = {status: sum(item["status"] == status for item in dispositions.values())
              for status in ("passed", "failed", "blocked", "incomplete")}
    print(json.dumps({"attempt": str(attempt), "counts": counts,
                      "crosscheck_failures": sum(item["status"] != "passed" for item in checks)}), flush=True)


if __name__ == "__main__":
    main()
