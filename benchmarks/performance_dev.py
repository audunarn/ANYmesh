"""Bounded native-v2 development benchmark transport.

Historical large-scale records remain readable by native_v2_baseline, but this
controller and its worker never launch them during the performance programme.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


MAX_REQUESTED_ELEMENTS = 100_000
DEVELOPMENT_SCALES = {"1k": 1_000, "10k": 10_000, "100k": 100_000}
EXCLUDED_SCALES = {"500k", "workstation"}
DEVELOPMENT_CASES = frozenset({
    "planar_callback_present", "planar_callback_absent",
    "cylindrical_callback_present", "cylindrical_callback_absent",
    "mapped_zero_use", "structural_three_plate", "plate_with_hole",
    "planar_insertion_work",
})


def guard_request(args: argparse.Namespace) -> int:
    """Reject excluded aggregate work before imports, allocation or spawning."""
    if str(args.case) not in DEVELOPMENT_CASES:
        raise ValueError("EXCLUDED_BY_SCOPE: unregistered development case")
    scale = str(args.scale)
    requested = int(args.requested_elements)
    if (
        scale in EXCLUDED_SCALES
        or requested > MAX_REQUESTED_ELEMENTS
        or args.allow_large
        or args.workstation_elements is not None
    ):
        raise ValueError("EXCLUDED_BY_SCOPE: 500k/workstation development work")
    if scale not in DEVELOPMENT_SCALES:
        raise ValueError(f"unknown development scale: {scale}")
    if requested != DEVELOPMENT_SCALES[scale]:
        raise ValueError("requested element count does not match the named scale")
    if int(args.worker_count) != 1:
        raise ValueError("development benchmarks require one bounded worker")
    if scale == "100k" and not args.checkpoint:
        raise ValueError("100k requires an explicit checkpoint request")
    install_kind = getattr(args, "install_kind", None)
    wheel_sha = getattr(args, "wheel_sha256", None)
    if install_kind == "wheel" and not (
        isinstance(wheel_sha, str) and re.fullmatch(r"[0-9a-f]{64}", wheel_sha)
    ):
        raise ValueError("wheel development requests require a wheel SHA-256")
    if install_kind == "source" and wheel_sha is not None:
        raise ValueError("source development requests must not carry a wheel SHA-256")
    return requested


def _benchmark_argv(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("performance_dev_measure.py")),
        "--case", args.case, "--scale", args.scale,
        "--requested-elements", str(args.requested_elements),
        "--route", args.route, "--backend", args.backend,
        "--install-kind", args.install_kind,
        "--source-commit", args.source_commit,
        "--compiler-id", args.compiler_id,
        "--output", str(args.output),
        "--measurements", str(args.measurements),
    ]
    if args.checkpoint:
        command.append("--checkpoint")
    if args.wheel_sha256 is not None:
        command.extend(("--wheel-sha256", args.wheel_sha256))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor", choices=("run", "_worker"))
    parser.add_argument("--case", required=True)
    parser.add_argument("--scale", default="1k")
    parser.add_argument("--requested-elements", type=int, default=1_000)
    parser.add_argument("--route", choices=("legacy", "frontal"), required=True)
    parser.add_argument("--backend", choices=("auto", "python", "compiled"), required=True)
    parser.add_argument("--install-kind", choices=("source", "wheel"), required=True)
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurements", type=int, choices=(1, 7), default=7)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--workstation-elements", type=int)
    args = parser.parse_args(argv)
    guard_request(args)
    if args.actor == "_worker":
        return subprocess.call(_benchmark_argv(args))
    worker_argv = [
        sys.executable, str(Path(__file__)), "_worker",
        "--case", args.case, "--scale", args.scale,
        "--requested-elements", str(args.requested_elements),
        "--route", args.route, "--backend", args.backend,
        "--install-kind", args.install_kind,
        "--source-commit", args.source_commit,
        "--compiler-id", args.compiler_id,
        "--output", str(args.output),
        "--measurements", str(args.measurements),
    ]
    if args.checkpoint:
        worker_argv.append("--checkpoint")
    if args.wheel_sha256 is not None:
        worker_argv.extend(("--wheel-sha256", args.wheel_sha256))
    return subprocess.call(worker_argv)


if __name__ == "__main__":
    raise SystemExit(main())
