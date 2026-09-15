"""Compare frozen, same-route native-v2 records from two source commits.

This verifier does not run benchmarks. It requires equivalent published work
while reporting internal topology operations as a separate performance measure.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _load(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")
    row = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(row, dict) or row.get("schema") != "anymesher.native-v2-baseline/3":
        raise ValueError("unsupported native-v2 benchmark record")
    return row, sha256(raw).hexdigest()


def _require_equal(before: dict[str, Any], after: dict[str, Any], field: str) -> None:
    if before.get(field) != after.get(field):
        raise ValueError(f"before/after {field} differs")


def _work(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("native_work_samples")
    if not isinstance(samples, list) or len(samples) != 7:
        raise ValueError("seven native work samples are required")
    first = samples[0]
    if any(sample != first for sample in samples[1:]):
        raise ValueError("native work changed between repetitions")
    return first


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    for field in (
        "case", "scale", "requested_elements", "route", "backend",
        "benchmark_configuration", "semantic_contract", "quality_policy",
        "warmups", "repetitions", "install_kind",
    ):
        _require_equal(before, after, field)
    if before.get("source_commit") == after.get("source_commit"):
        raise ValueError("before/after records must name distinct source commits")
    if before.get("route") not in ("legacy", "frontal"):
        raise ValueError("unknown meshing route")
    old_provenance = dict(before.get("provenance") or {})
    new_provenance = dict(after.get("provenance") or {})
    for key in ("wheel_sha256", "commit_binding"):
        old_provenance.pop(key, None)
        new_provenance.pop(key, None)
    if old_provenance != new_provenance:
        raise ValueError("runtime, dependency, machine or harness provenance differs")
    for field in (
        "mesh_digest", "association_digest", "actual_elements", "triangles", "quadrilaterals",
        "q4_fraction", "quality", "serialization_bytes",
    ):
        _require_equal(before, after, field)
    old_work, new_work = _work(before), _work(after)
    for field in ("published_insertions",):
        if old_work.get(field) != new_work.get(field):
            raise ValueError(f"accepted refinement work differs: {field}")
    old_samples = before.get("durations_seconds")
    new_samples = after.get("durations_seconds")
    if not all(isinstance(rows, list) and len(rows) == 7 for rows in (old_samples, new_samples)):
        raise ValueError("seven measured repetitions are required")
    if any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
           for rows in (old_samples, new_samples) for value in rows):
        raise ValueError("non-positive or non-finite timing")
    old_median, new_median = statistics.median(old_samples), statistics.median(new_samples)
    old_peak, new_peak = before.get("peak_rss_bytes"), after.get("peak_rss_bytes")
    if any(type(value) is not int or value <= 0 for value in (old_peak, new_peak)):
        raise ValueError("fresh-process peak RSS evidence is required")
    return {
        "schema": "anymesher.performance-before-after/1",
        "case": before["case"], "route": before["route"],
        "before_commit": before["source_commit"],
        "after_commit": after["source_commit"],
        "actual_elements": before["actual_elements"],
        "published_insertions": old_work["published_insertions"],
        "before_attempted_insertions": old_work.get("insertions"),
        "after_attempted_insertions": new_work.get("insertions"),
        "before_topology_operations": old_work.get("topology_operations"),
        "after_topology_operations": new_work.get("topology_operations"),
        "before_samples_seconds": old_samples,
        "after_samples_seconds": new_samples,
        "before_median_seconds": old_median,
        "after_median_seconds": new_median,
        "runtime_ratio": new_median / old_median,
        "before_peak_rss_bytes": old_peak,
        "after_peak_rss_bytes": new_peak,
        "peak_rss_ratio": new_peak / old_peak,
        "mesh_digest": before["mesh_digest"],
        "association_digest": before.get("association_digest"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    before, before_sha = _load(args.before)
    after, after_sha = _load(args.after)
    result = compare(before, after)
    result["before_record_sha256"] = before_sha
    result["after_record_sha256"] = after_sha
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
