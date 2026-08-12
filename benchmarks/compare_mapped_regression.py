"""Drift-controlled comparison of frozen baseline and current mapped paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_state(root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"sha": sha, "clean": not bool(dirty), "status": dirty.splitlines()}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fraction(current: float, baseline: float) -> float:
    return current / baseline - 1.0


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--harness",
        type=Path,
        default=Path(__file__).with_name("mapped_regression_baseline.py"),
    )
    parser.add_argument("--baseline-anymesh-root", type=Path, required=True)
    parser.add_argument("--baseline-anygeometry-root", type=Path, required=True)
    parser.add_argument("--current-anymesh-root", type=Path, required=True)
    parser.add_argument("--current-anygeometry-root", type=Path, required=True)
    parser.add_argument("--baseline-anymesh-sha", required=True)
    parser.add_argument("--baseline-anygeometry-sha", required=True)
    parser.add_argument("--current-anymesh-sha", required=True)
    parser.add_argument("--current-anygeometry-sha", required=True)
    parser.add_argument("--expected-harness-sha256", required=True)
    parser.add_argument("--expected-comparator-sha256", required=True)
    parser.add_argument("--elements", type=int, default=10_000)
    parser.add_argument("--leg-repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--generation-limit", type=float, default=0.03)
    parser.add_argument("--memory-limit", type=float, default=0.05)
    parser.add_argument("--serialization-limit", type=float, default=0.05)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.elements <= 0 or args.leg_repeats <= 0 or args.warmup < 0:
        parser.error("elements/repeats must be positive and warmup non-negative")

    harness_path = args.harness.resolve()
    comparator_path = Path(__file__).resolve()
    tooling_before = {
        "harness": {
            "path": str(harness_path),
            "sha256": _file_hash(harness_path),
            "expected_sha256": args.expected_harness_sha256.upper(),
        },
        "comparator": {
            "path": str(comparator_path),
            "sha256": _file_hash(comparator_path),
            "expected_sha256": args.expected_comparator_sha256.upper(),
        },
    }
    failures = []
    for name, item in tooling_before.items():
        if item["sha256"] != item["expected_sha256"]:
            failures.append(f"{name} hash differs from expected value")

    variants = {
        "baseline": {
            "anymesh_root": args.baseline_anymesh_root.resolve(),
            "anygeometry_root": args.baseline_anygeometry_root.resolve(),
            "expected_anymesh_sha": args.baseline_anymesh_sha,
            "expected_anygeometry_sha": args.baseline_anygeometry_sha,
        },
        "current": {
            "anymesh_root": args.current_anymesh_root.resolve(),
            "anygeometry_root": args.current_anygeometry_root.resolve(),
            "expected_anymesh_sha": args.current_anymesh_sha,
            "expected_anygeometry_sha": args.current_anygeometry_sha,
        },
    }
    for name, variant in variants.items():
        variant["anymesh_git"] = _git_state(variant["anymesh_root"])
        variant["anygeometry_git"] = _git_state(variant["anygeometry_root"])
        for package in ("anymesh", "anygeometry"):
            state = variant[f"{package}_git"]
            expected = variant[f"expected_{package}_sha"]
            if state["sha"] != expected:
                failures.append(
                    f"{name} {package} SHA {state['sha']} != expected {expected}"
                )
            if not state["clean"]:
                failures.append(f"{name} {package} worktree is dirty")

    args.samples_dir.mkdir(parents=True, exist_ok=True)
    order = ("baseline", "current", "current", "baseline")
    environment_overrides = {
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    legs = []
    if not failures:
        for number, name in enumerate(order, start=1):
            variant = variants[name]
            output = args.samples_dir / f"{number:02d}-{name}.json"
            command = [
                args.python,
                str(harness_path),
                "--elements",
                str(args.elements),
                "--repeats",
                str(args.leg_repeats),
                "--warmup",
                str(args.warmup),
                "--label",
                f"{number:02d}-{name}",
                "--expected-anymesh-root",
                str(variant["anymesh_root"]),
                "--expected-anygeometry-root",
                str(variant["anygeometry_root"]),
                "--output",
                str(output),
            ]
            environment = os.environ.copy()
            environment.update(environment_overrides)
            environment["PYTHONPATH"] = os.pathsep.join(
                (
                    str(variant["anymesh_root"] / "src"),
                    str(variant["anygeometry_root"] / "src"),
                )
            )
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=args.samples_dir,
                env=environment,
                capture_output=True,
                text=True,
            )
            elapsed = time.perf_counter() - started
            leg = {
                "sequence": number,
                "variant": name,
                "command": command,
                "command_text": _command_text(command),
                "elapsed_seconds": elapsed,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "output": str(output),
            }
            if completed.returncode == 0 and output.exists():
                leg["result"] = json.loads(output.read_text())
            else:
                failures.append(f"leg {number} ({name}) failed")
            legs.append(leg)

    post_run_variants = {}
    for name, variant in variants.items():
        post_run_variants[name] = {
            "anymesh_git": _git_state(variant["anymesh_root"]),
            "anygeometry_git": _git_state(variant["anygeometry_root"]),
        }
        for package in ("anymesh", "anygeometry"):
            before = variant[f"{package}_git"]
            after = post_run_variants[name][f"{package}_git"]
            if after["sha"] != before["sha"]:
                failures.append(f"{name} {package} SHA changed during comparison")
            if not after["clean"]:
                failures.append(f"{name} {package} worktree mutated during comparison")

    tooling_after = {
        "harness": _file_hash(harness_path),
        "comparator": _file_hash(comparator_path),
    }
    for name, digest in tooling_after.items():
        if digest != tooling_before[name]["sha256"]:
            failures.append(f"{name} changed during comparison")

    samples: dict[str, dict[str, list[Any]]] = {
        name: {
            "generation_seconds": [],
            "generation_peaks": [],
            "serialization_seconds": [],
            "serialization_peaks": [],
            "nodes": [],
            "elements": [],
            "topology_hashes": [],
            "serialized_hashes": [],
        }
        for name in variants
    }
    for leg in legs:
        result = leg.get("result")
        if result is None:
            continue
        target = samples[leg["variant"]]
        target["generation_seconds"].extend(result["generation_seconds"])
        target["generation_peaks"].append(result["generation_peak_traced_bytes"])
        target["serialization_seconds"].extend(result["serialization_seconds"])
        target["serialization_peaks"].append(result["serialization_peak_traced_bytes"])
        target["nodes"].append(result["nodes"])
        target["elements"].append(result["elements"])
        target["topology_hashes"].extend(result["topology_hashes"])
        target["serialized_hashes"].extend(result["serialized_hashes"])

    summary = {}
    environment_fingerprints = []
    for leg in legs:
        result = leg.get("result")
        if result is None:
            continue
        environment_fingerprints.append(
            {
                "platform": result["platform"],
                "processor": result["processor"],
                "machine": result["machine"],
                "processor_identifier": result["processor_identifier"],
                "logical_processors": result["logical_processors"],
                "python": result["python"],
                "non_owned_dependencies": {
                    key: value
                    for key, value in result["dependencies"].items()
                    if key not in {
                        "anygeometry",
                        "anymesher",
                        "anygeometry_module",
                        "anymesher_module",
                    }
                },
            }
        )
    if environment_fingerprints:
        canonical_environments = {
            json.dumps(value, sort_keys=True) for value in environment_fingerprints
        }
        if len(canonical_environments) != 1:
            failures.append("isolated legs used different host/dependency environments")

    if not failures:
        for name, values in samples.items():
            summary[name] = {
                "generation_median_seconds": statistics.median(
                    values["generation_seconds"]
                ),
                "generation_peak_traced_bytes": max(values["generation_peaks"]),
                "serialization_median_seconds": statistics.median(
                    values["serialization_seconds"]
                ),
                "serialization_peak_traced_bytes": max(
                    values["serialization_peaks"]
                ),
                "nodes": sorted(set(values["nodes"])),
                "elements": sorted(set(values["elements"])),
                "topology_hashes": sorted(set(values["topology_hashes"])),
                "serialized_hashes": sorted(set(values["serialized_hashes"])),
            }
        baseline = summary["baseline"]
        current = summary["current"]
        ratios = {
            "generation_median_regression_fraction": _fraction(
                current["generation_median_seconds"],
                baseline["generation_median_seconds"],
            ),
            "generation_peak_memory_regression_fraction": _fraction(
                current["generation_peak_traced_bytes"],
                baseline["generation_peak_traced_bytes"],
            ),
            "serialization_median_regression_fraction": _fraction(
                current["serialization_median_seconds"],
                baseline["serialization_median_seconds"],
            ),
            "serialization_peak_memory_regression_fraction": _fraction(
                current["serialization_peak_traced_bytes"],
                baseline["serialization_peak_traced_bytes"],
            ),
        }
        if baseline["nodes"] != current["nodes"] or baseline["elements"] != current["elements"]:
            failures.append("mapped topology counts differ")
        if len(baseline["topology_hashes"]) != 1 or baseline["topology_hashes"] != current["topology_hashes"]:
            failures.append("mapped topology hashes differ or are nondeterministic")
        if len(baseline["serialized_hashes"]) != 1 or baseline["serialized_hashes"] != current["serialized_hashes"]:
            failures.append("mapped serialized hashes differ or are nondeterministic")
        if ratios["generation_median_regression_fraction"] > args.generation_limit:
            failures.append("mapped generation median regression exceeds gate")
        if ratios["generation_peak_memory_regression_fraction"] > args.memory_limit:
            failures.append("mapped peak traced-memory regression exceeds gate")
        if ratios["serialization_median_regression_fraction"] > args.serialization_limit:
            failures.append("mapped serialization median regression exceeds gate")
    else:
        ratios = {}

    manifest_variants = {
        name: {
            **variant,
            "anymesh_root": str(variant["anymesh_root"]),
            "anygeometry_root": str(variant["anygeometry_root"]),
        }
        for name, variant in variants.items()
    }
    manifest = {
        "schema": "anymesher.mapped_regression_comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_processors": os.cpu_count(),
            "python": platform.python_version(),
            "executable": args.python,
        },
        "method": {
            "order": list(order),
            "warmup_per_leg": args.warmup,
            "timed_repeats_per_leg": args.leg_repeats,
            "timed_samples_per_variant": 2 * args.leg_repeats,
            "isolated_process_per_leg": True,
            "environment_overrides": environment_overrides,
            "elements": args.elements,
        },
        "variants": manifest_variants,
        "post_run_variants": post_run_variants,
        "tooling_before": tooling_before,
        "tooling_after": tooling_after,
        "legs": legs,
        "environment_fingerprints": environment_fingerprints,
        "raw_samples": samples,
        "summary": summary,
        "ratios": ratios,
        "gates": {
            "generation_median_regression_max_fraction": args.generation_limit,
            "generation_peak_memory_regression_max_fraction": args.memory_limit,
            "serialization_median_regression_max_fraction": args.serialization_limit,
            "passed": not failures,
            "failures": failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
