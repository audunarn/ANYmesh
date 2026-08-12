"""Like-for-like mapped compatibility baseline for leased branch comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

import anygeometry
import anymesher
from anygeometry import GeometryModel
from anymesher.mapped import generate_mesh
from anymesher.serialize import mesh_to_dict

MODEL_ID = UUID("4f88d65f-68a2-4ffc-a0b0-5755a48aa65f")


def _rectangle() -> GeometryModel:
    geometry = GeometryModel(model_id=MODEL_ID)
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    geometry.add_plate(vertices)
    return geometry


def _time_peak(call: Callable[[], Any]) -> tuple[Any, float, int]:
    tracemalloc.start()
    started = time.perf_counter()
    try:
        value = call()
        elapsed = time.perf_counter() - started
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, elapsed, int(peak)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _topology_hash(mesh: Any) -> str:
    return _canonical_hash(
        {
            "nodes": [
                [int(identifier), *(float(value) for value in mesh.nodes[identifier])]
                for identifier in sorted(mesh.nodes)
            ],
            "quads": [
                [int(identifier), *(int(value) for value in mesh.quads[identifier])]
                for identifier in sorted(mesh.quads)
            ],
            "tris": [
                [int(identifier), *(int(value) for value in mesh.tris[identifier])]
                for identifier in sorted(mesh.tris)
            ],
            "beams": [
                [int(identifier), *(int(value) for value in mesh.beams[identifier])]
                for identifier in sorted(mesh.beams)
            ],
        }
    )


def _module_path(module: Any) -> str:
    return str(Path(module.__file__).resolve())


def _dependency_versions() -> dict[str, str]:
    result = {}
    for name in ("numpy", "scipy", "shapely", "meshio", "anygeometry", "anymesher"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _require_source(module: Any, expected_root: Path, name: str) -> str:
    source = Path(module.__file__).resolve()
    root = expected_root.resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f"{name} imported from {source}, outside expected {root}")
    return str(source)


def run(requested_elements: int, repeats: int, warmup: int, label: str) -> dict[str, Any]:
    target_size = math.sqrt(1.0 / requested_elements)
    for _ in range(warmup):
        mesh_to_dict(generate_mesh(_rectangle(), target_size=target_size))
    generation_seconds = []
    generation_peaks = []
    topology_hashes = []
    serialized_hashes = []
    mesh = None
    for _ in range(repeats):
        mesh, elapsed, peak = _time_peak(
            lambda: generate_mesh(_rectangle(), target_size=target_size)
        )
        generation_seconds.append(elapsed)
        generation_peaks.append(peak)
        topology_hashes.append(_topology_hash(mesh))
        serialized_hashes.append(_canonical_hash(mesh_to_dict(mesh)))
    if mesh is None:
        raise RuntimeError("mapped baseline generated no mesh")

    serialization_seconds = []
    serialization_peaks = []
    payload = None
    for _ in range(repeats):
        payload, elapsed, peak = _time_peak(lambda: mesh_to_dict(mesh))
        serialization_seconds.append(elapsed)
        serialization_peaks.append(peak)
    return {
        "schema": "anymesher.mapped_regression_baseline.v1",
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_processors": os.cpu_count(),
        "python": platform.python_version(),
        "dependencies": {
            **_dependency_versions(),
            "anygeometry_module": str(getattr(anygeometry, "__version__", "unknown")),
            "anymesher_module": str(getattr(anymesher, "__version__", "unknown")),
        },
        "sources": {
            "anygeometry": _module_path(anygeometry),
            "anymesher": _module_path(anymesher),
        },
        "model_id": str(MODEL_ID),
        "requested_elements": requested_elements,
        "target_size": target_size,
        "repeats": repeats,
        "warmup": warmup,
        "nodes": len(mesh.nodes),
        "elements": len(mesh.quads) + len(mesh.tris) + len(mesh.beams),
        "generation_seconds": generation_seconds,
        "generation_median_seconds": statistics.median(generation_seconds),
        "generation_peak_traced_bytes": max(generation_peaks),
        "serialization_seconds": serialization_seconds,
        "serialization_median_seconds": statistics.median(serialization_seconds),
        "serialization_peak_traced_bytes": max(serialization_peaks),
        "topology_hashes": topology_hashes,
        "serialized_hashes": serialized_hashes,
        "topology_sha256": topology_hashes[-1],
        "serialized_sha256": _canonical_hash(payload),
        "repeated_topology_identical": len(set(topology_hashes)) == 1,
        "repeated_serialization_identical": len(set(serialized_hashes)) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--label", default="mapped-regression")
    parser.add_argument("--expected-anymesh-root", type=Path, required=True)
    parser.add_argument("--expected-anygeometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.elements <= 0 or args.repeats <= 0 or args.warmup < 0:
        parser.error("--elements/repeats must be positive and --warmup non-negative")
    _require_source(anymesher, args.expected_anymesh_root, "anymesher")
    _require_source(anygeometry, args.expected_anygeometry_root, "anygeometry")
    result = run(args.elements, args.repeats, args.warmup, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
