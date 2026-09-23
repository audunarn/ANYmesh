"""Governed cache/queue scaling microbenchmark, not end-to-end mesher acceptance.

Run both modes in fresh processes. The reference reconstructs geometric records
and the work heap each round; the incremental mode refreshes the same priorities.
No result from this harness establishes Frontal-Delaunay or mapped timing gates.
"""
from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import struct
import subprocess
import sys
import time

import numpy as np

from anymesher._t3_runtime import CellGeometry, TriangleWorkQueue
from anymesher.native_v2 import _angles


def _rss():
    if os.name != "nt":
        import resource
        amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(amount if sys.platform == "darwin" else amount * 1024)

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def _write_new(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _identity(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": sha256(data).hexdigest()}


def _worker(count, mode, deadline):
    def checkpoint(number=0):
        if number % 1024 == 0 and time.monotonic() >= deadline:
            raise TimeoutError("benchmark worker shared deadline exhausted")

    checkpoint()
    width = math.ceil(math.sqrt(count / 2))
    xx, yy = np.meshgrid(np.arange(width + 1, dtype=float),
                         np.arange(width + 1, dtype=float))
    points = np.column_stack((xx.ravel(), yy.ravel()))
    cells = []
    for row in range(width):
        checkpoint(row)
        for column in range(width):
            a = row * (width + 1) + column
            b = a + 1
            d = a + width + 1
            cells.extend(((a, b, d + 1), (a, d + 1, d)))
    cells = cells[:count]
    records = {}
    queue = TriangleWorkQueue()
    base_tensor = np.repeat((np.eye(2) * 4)[None], 3, axis=0)
    changed_tensor = base_tensor * 1.25
    signatures = []
    timings = []
    evaluations = []
    hits = []
    initialization = None
    # Initial state, one warmup update, seven measured updates.
    for round_number in range(9):
        checkpoint()
        if mode == "reference":
            records = {}
            queue = TriangleWorkQueue()
        before_evaluations = queue.evaluations
        before_hits = queue.cache_hits
        start = time.perf_counter()
        queue.begin_round()
        digest = sha256()
        changed = 0
        for number, cell in enumerate(cells):
            checkpoint(number)
            record = records.get(cell)
            if record is None:
                record = CellGeometry.prepare(points, cell)
                records[cell] = record
            if record.angles is None:
                record.angles = _angles(record.coordinates)
            active_change = number % 100 == round_number % 100
            changed += int(active_change)
            tensors = changed_tensor if active_change else base_tensor

            def evaluate():
                center = np.mean(tensors, axis=0)
                lengths = []
                for index in range(3):
                    delta = record.coordinates[(index + 1) % 3] - record.coordinates[index]
                    lengths.append(math.sqrt(max(float(delta @ center @ delta), 0.)))
                return max(max(lengths) / math.sqrt(2),
                           30. / max(min(record.angles), 1.e-12)), False

            severity, limited = queue.refresh(cell, record, tensors, False, evaluate)
            digest.update(struct.pack("<3qd?", *cell, severity, limited))
        queue.finish_round()
        # Exercise consumption and next-round reactivation without draining the heap.
        popped = []
        for _ in range(min(8, count)):
            if not queue:
                break
            entry = queue.pop()
            popped.append(entry[:3])
            digest.update(struct.pack("<d3q", entry[0], *entry[2]))
        elapsed = time.perf_counter() - start
        signatures.append(digest.hexdigest())
        if round_number == 0:
            initialization = elapsed
        elif round_number >= 2:
            timings.append(elapsed)
            evaluations.append(queue.evaluations - before_evaluations)
            hits.append(queue.cache_hits - before_hits)
    return {
        "schema": "anymesher.t3-runtime-scaling-worker/1",
        "mode": mode, "triangles": count, "points": len(points),
        "initialization_seconds": initialization,
        "warmups": 1, "measurements": 7,
        "seconds": timings, "median_seconds": statistics.median(timings),
        "peak_rss_bytes": _rss(), "round_signatures": signatures,
        "quality_evaluations": evaluations, "quality_cache_hits": hits,
        "cells_perturbed_from_baseline_per_round": changed,
        "metric_transition_fraction_per_measured_round": 0.02,
        "popped_cells_per_round": len(popped),
        "full_mesh_scans_remaining": True,
        "end_to_end_qualification": False,
        "parity_scope": "all refreshed priorities and first eight pops per round",
        "timing_includes": ["refresh", "signature hashing", "first eight pops"],
        "rss_scope": "whole-process peak including initialization",
        "topology_mutation_and_location_exercised": False,
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _ledger_approves(text, request_id):
    """Read timestamp/request/status columns, retaining the leading empty field."""
    return any(
        len(fields := [part.strip() for part in line.split("|")]) >= 5
        and fields[2:4] == [request_id, "APPROVED"]
        for line in text.splitlines()
    )


def _authorize(args):
    if not args.request_id or re.fullmatch(r"[0-9a-f]{32}", args.request_id) is None:
        raise RuntimeError("registered resource request required for all modes")
    manager = Path("C:/Github/.resource-manager")
    owner = json.loads((manager / "active-lock" / "owner.json").read_text(encoding="utf-8-sig"))
    owner = {key.lower(): value for key, value in owner.items()}
    if owner.get("requestid", owner.get("request_id")) != args.request_id:
        raise RuntimeError("resource lock does not belong to this request")
    approved = _ledger_approves(
        (manager / "ledger.md").read_text(encoding="utf-8-sig"), args.request_id
    )
    if not approved:
        raise RuntimeError("request is not administrator-approved")
    request = json.loads((manager / "requests" / (args.request_id + ".json")).read_text(
        encoding="utf-8-sig"))
    configuration = request.get("benchmark_configuration", {})
    expected = {"scales": [10000, 100000, 500000], "warmups": 1, "measurements": 7,
                "max_seconds": 1200, "worker_modes": ["reference", "incremental"]}
    if configuration != expected:
        raise RuntimeError("approved benchmark scope/configuration mismatch")
    if request.get("benchmark_sha256", "").lower() != _identity(__file__)["sha256"]:
        raise RuntimeError("benchmark source differs from registered identity")
    if str(Path(__file__).resolve()) not in request.get("command", ""):
        raise RuntimeError("approved command does not bind this benchmark")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[10000, 100000, 500000])
    parser.add_argument("--output")
    parser.add_argument("--request-id")
    parser.add_argument("--worker", choices=["reference", "incremental"])
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--deadline-monotonic", type=float)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    request = _authorize(args)
    if args.worker:
        if (args.worker_count not in request["benchmark_configuration"]["scales"]
                or args.deadline_monotonic is None
                or not 0 < args.deadline_monotonic - time.monotonic() <= 1200):
            parser.error("bounded worker count required")
        sys.stdout.buffer.write(_encoded(_worker(args.worker_count, args.worker,
                                                 args.deadline_monotonic)))
        return
    if (not args.output or not args.request_id
            or any(count not in (10000, 100000, 500000) for count in args.counts)
            or len(set(args.counts)) != len(args.counts)
            or not 1 <= args.timeout_seconds <= 1200):
        parser.error("fresh output, approved request, and unique registered scales required")
    if str(Path(args.output)) not in request["command"]:
        raise RuntimeError("output path is not bound by approved command")
    root = Path(args.output)
    root.mkdir(parents=False, exist_ok=False)
    import anymesher._t3_runtime as runtime
    import anymesher._t3_incidence as incidence
    import anymesher.native_v2 as native_v2
    artifacts = [_identity(path) for path in
                 (__file__, runtime.__file__, incidence.__file__, native_v2.__file__)]
    _write_new(root / "intent.json", _encoded({
        "schema": "anymesher.t3-runtime-scaling-intent/1",
        "request_id": args.request_id, "argv": sys.argv,
        "source_artifacts": artifacts, "python": _identity(sys.executable),
        "end_to_end_qualification": False,
    }))
    deadline = time.monotonic() + args.timeout_seconds
    results = []
    for count in args.counts:
        pair = {}
        for mode in ("reference", "incremental"):
            remaining = deadline - time.monotonic()
            if remaining <= 10:
                raise TimeoutError("shared deadline reserve exhausted")
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--worker", mode, "--worker-count", str(count),
                       "--request-id", args.request_id,
                       "--deadline-monotonic", str(deadline - 10)]
            stamp = root / (str(count) + "-" + mode)
            try:
                process = subprocess.run(command, capture_output=True, timeout=remaining - 10,
                                         check=False)
            except subprocess.TimeoutExpired as error:
                _write_new(str(stamp) + ".stdout.partial", error.stdout or b"")
                _write_new(str(stamp) + ".stderr.partial", error.stderr or b"")
                _write_new(str(stamp) + ".exit.json", _encoded({
                    "timeout": True, "child_killed_and_waited_by_subprocess": True,
                    "command": command, "retry": False}))
                raise
            _write_new(str(stamp) + ".stdout", process.stdout)
            _write_new(str(stamp) + ".stderr", process.stderr)
            _write_new(str(stamp) + ".exit.json", _encoded({
                "returncode": process.returncode, "command": command, "timeout": False}))
            if process.returncode:
                raise RuntimeError("benchmark worker failed; preserved first failure")
            pair[mode] = json.loads(process.stdout)
        if pair["reference"]["round_signatures"] != pair["incremental"]["round_signatures"]:
            raise RuntimeError("reference/incremental decision parity failed")
        results.append({
            "triangles": count, "parity": True,
            "median_ratio": pair["incremental"]["median_seconds"] /
                            pair["reference"]["median_seconds"],
            "peak_rss_ratio": pair["incremental"]["peak_rss_bytes"] /
                              pair["reference"]["peak_rss_bytes"],
            "reference": pair["reference"], "incremental": pair["incremental"],
        })
    if artifacts != [_identity(row["path"]) for row in artifacts]:
        raise RuntimeError("source changed during benchmark")
    _write_new(root / "result.json", _encoded({
        "schema": "anymesher.t3-runtime-scaling/1",
        "results": results, "source_artifacts": artifacts,
        "end_to_end_qualification": False,
        "remaining_gates": ["actual front insertion scaling", "mapped zero-use timing/RSS",
                            "source/wheel cross-platform corpus"],
    }))


if __name__ == "__main__":
    main()
