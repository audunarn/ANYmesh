"""Bounded development preflight for source-aligned point Q8 neighbourhoods."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .attachment_refinement_trial import trial
from .runner import ROOT, digest, identities, pin_workers, supervise, workers_stable, write


OUTPUT = ROOT / "reports/quad_first/structural-gate/point-seed-preflight"
CASES = ("CY", "CO", "R", "C")
SNAPSHOT = (
    "src/anymesher/hybrid.py",
    "src/anymesher/quad/seed.py",
    "src/anymesher/quad/validate.py",
    "benchmarks/sg1/attachment_refinement_trial.py",
    "benchmarks/sg1/point_seed_preflight.py",
)


def run(baseline: Path, sizes: tuple[float, ...] = (.15,)) -> Path:
    baseline = baseline.resolve()
    sealed = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if sealed.get("status") != "failed" or sealed.get("mode") != "formal":
        raise ValueError("requires the sealed failed formal SG1 attempt")
    baseline_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    bad = [name for name, sha in baseline_manifest.items()
           if not (baseline / name).is_file() or digest(baseline / name) != sha]
    if bad:
        raise ValueError(f"baseline has {len(bad)} missing or altered artifacts")
    root = OUTPUT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                     + "-" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    print("PREFLIGHT " + str(root), flush=True)
    source_dir = root / "source"
    for relative in SNAPSHOT:
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    before = identities()
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONHASHSEED"] = "0"
    workers = pin_workers(env)
    write(root / "environment.json", {**before, **workers})
    write(root / "baseline.json", {
        "path": str(baseline), "manifest_sha256": digest(baseline / "manifest.json"),
        "verified_manifest_entries": len(baseline_manifest),
    })
    deadline = time.monotonic() + 7200
    observations = {}
    for case in CASES:
        for h in sizes:
            for graded in (False, True):
                key = f"{case}-h{h:g}-g{int(graded)}"
                folder = root / key
                folder.mkdir()
                args = [sys.executable, "-m", "benchmarks.sg1.point_seed_preflight",
                        "--child-case", case, "--baseline", str(baseline),
                        "--output", str(folder), "--child-h", str(h)]
                if graded:
                    args.append("--graded")
                observation = supervise(args, folder, env, deadline, limit=600)
                observations[key] = observation
                write(folder / "supervisor.json", observation)
                print(key + ": " + observation["status"], flush=True)
    checks = []

    def check(name, passed, **values):
        checks.append({"name": name, "status": "passed" if passed else "failed", **values})

    for case in CASES:
        for h in sizes:
            results = {}
            audits = {}
            for graded in (False, True):
                key = f"{case}-h{h:g}-g{int(graded)}"
                folder = root / key
                if observations[key]["status"] != "passed":
                    continue
                results[graded] = json.loads((folder / "summary.json").read_text())
                audits[graded] = json.loads((folder / "audit.json").read_text())
                row = results[graded]
                check(key + "/mesh", not row["audit_failed_checks"],
                      failures=row["audit_failed_checks"])
                check(key + "/solver", not row["solver_failed_checks"],
                      failures=row["solver_failed_checks"])
                if h == .15:
                    check(key + "/reference", row["max_relative_error"] <= .05,
                          value=row["max_relative_error"])
            if len(audits) == 2:
                uniform, refined = audits[False], audits[True]
                check(f"{case}-h{h:g}/core-count",
                      refined["core_corner_count"] > uniform["core_corner_count"])
                check(f"{case}-h{h:g}/core-size",
                      refined["core_median"] < uniform["core_median"])
                ratio = refined["remote_median"] / uniform["remote_median"]
                check(f"{case}-h{h:g}/remote", abs(ratio - 1) <= .2, ratio=ratio)
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
        or not checks or any(item["status"] != "passed" for item in checks)
        else "passed"
    )
    write(root / "summary.json", {
        "kind": "development-preflight-not-SG1-acceptance", "status": status,
        "sizes_m": list(sizes),
        "observations": observations, "checks": checks,
        "source_stable": stable, "workers_stable": workers_stable(workers),
    })
    write(root / "manifest.json", {
        str(path.relative_to(root)).replace("\\", "/"): digest(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    })
    print(json.dumps({"path": str(root), "status": status,
                      "checks": len(checks)}), flush=True)
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--child-case", choices=CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--graded", action="store_true")
    parser.add_argument("--child-h", type=float, default=.15)
    parser.add_argument("--sizes", type=float, nargs="+", default=[.15])
    args = parser.parse_args()
    if args.child_case:
        if args.output is None:
            parser.error("--output is required for a child")
        trial(args.child_case, args.graded, args.baseline, args.output,
              extra_refinement=False, target_size_m=args.child_h)
        return 0
    if not args.sizes or any(h <= 0 for h in args.sizes):
        parser.error("--sizes must contain positive physical sizes")
    root = run(args.baseline, tuple(args.sizes))
    return 0 if json.loads((root / "summary.json").read_text())["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
