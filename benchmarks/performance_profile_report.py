"""Preserve a bounded pstats attribution summary without running a mesh."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import pstats

from performance_dev_measure import _utc, _write_new


def summarize(profile: Path, limit: int = 30) -> dict:
    if limit < 1 or limit > 100:
        raise ValueError("profile summary limit must be in 1..100")
    raw = profile.read_bytes()
    stats = pstats.Stats(str(profile))
    rows = []
    for (filename, line, function), (primitive, calls, self_seconds,
                                     cumulative_seconds, _callers) in stats.stats.items():
        rows.append({
            "filename": filename, "line": line, "function": function,
            "primitive_calls": primitive, "calls": calls,
            "self_seconds": self_seconds,
            "cumulative_seconds": cumulative_seconds,
        })
    return {
        "schema": "anymesher.performance-profile-attribution/1",
        "created_utc": _utc(), "profile_path": str(profile.resolve()),
        "profile_bytes": len(raw), "profile_sha256": sha256(raw).hexdigest(),
        "total_self_seconds": stats.total_tt,
        "top_cumulative": sorted(
            rows, key=lambda row: (-row["cumulative_seconds"],
                                   row["filename"], row["line"], row["function"])
        )[:limit],
        "top_self": sorted(
            rows, key=lambda row: (-row["self_seconds"],
                                   row["filename"], row["line"], row["function"])
        )[:limit],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "reports" / "performance_dev"
    if (not args.profile.resolve().is_relative_to(root.resolve())
            or not args.output.resolve().is_relative_to(root.resolve())):
        raise ValueError("profile and summary must stay in isolated development evidence")
    _write_new(args.output, summarize(args.profile, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
