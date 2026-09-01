"""Compare two ANYmesher native/hybrid performance reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _case_key(case: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(case["family"]),
        str(case["backend"]),
        int(case["requested_elements"]),
    )


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    improvement_gate: float = 0.15,
    regression_gate: float = 0.05,
    memory_gate: float = 0.10,
) -> dict[str, Any]:
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("both reports must be complete")
    baseline_cases = {_case_key(item): item for item in baseline["cases"]}
    candidate_cases = {_case_key(item): item for item in candidate["cases"]}
    if baseline_cases.keys() != candidate_cases.keys():
        raise ValueError("baseline and candidate case inventories differ")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    target_improvements: list[float] = []
    for key in sorted(baseline_cases):
        before = baseline_cases[key]
        after = candidate_cases[key]
        before_seconds = float(before["generation_median_seconds"])
        after_seconds = float(after["generation_median_seconds"])
        improvement = 1.0 - after_seconds / max(before_seconds, 1.0e-12)
        regression = -improvement
        before_memory = max(
            int(before["generation_peak_traced_bytes"]),
            int(before["generation_peak_process_rss_bytes"]),
        )
        after_memory = max(
            int(after["generation_peak_traced_bytes"]),
            int(after["generation_peak_process_rss_bytes"]),
        )
        memory_regression = after_memory / max(before_memory, 1) - 1.0
        hash_equal = before["mesh_hash"] == after["mesh_hash"]
        strategy_equal = before.get("generation_strategy") == after.get(
            "generation_strategy"
        )
        if regression > regression_gate:
            failures.append(f"{key}: median regression {regression:.3%}")
        if memory_regression > memory_gate:
            failures.append(f"{key}: peak-memory regression {memory_regression:.3%}")
        if not hash_equal:
            failures.append(f"{key}: canonical mesh hash changed")
        if not strategy_equal:
            failures.append(f"{key}: selected strategy or candidate count changed")
        if key[0] == "plate_hole":
            phase_before = before.get("generation_phase_median_seconds", {})
            phase_after = after.get("generation_phase_median_seconds", {})
            shared = set(phase_before) & set(phase_after)
            meaningful = [
                1.0 - float(phase_after[name]) / max(float(phase_before[name]), 1.0e-12)
                for name in shared
                if float(phase_before[name]) >= 0.05 * before_seconds
            ]
            target_improvements.extend((improvement, *meaningful))
        rows.append(
            {
                "family": key[0],
                "backend": key[1],
                "requested_elements": key[2],
                "median_improvement_fraction": improvement,
                "peak_memory_regression_fraction": memory_regression,
                "mesh_hash_equal": hash_equal,
                "strategy_equal": strategy_equal,
            }
        )
    best_target_improvement = max(target_improvements, default=float("-inf"))
    if best_target_improvement < improvement_gate:
        failures.append(
            "plate_hole total or dominant phase did not improve by "
            f"{improvement_gate:.0%}"
        )
    return {
        "schema": "anymesher.native_hybrid.performance_comparison",
        "version": 1,
        "passed": not failures,
        "gates": {
            "minimum_target_improvement_fraction": improvement_gate,
            "maximum_median_regression_fraction": regression_gate,
            "maximum_peak_memory_regression_fraction": memory_gate,
        },
        "best_target_improvement_fraction": best_target_improvement,
        "failures": failures,
        "cases": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_reports(baseline, candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
