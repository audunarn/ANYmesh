"""Reproducible native-hybrid performance and scaling qualification.

This module is intentionally outside the test suite.  Its larger cases require
an ecosystem performance lease and write one machine-readable report for the
release evidence packet.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from anygeometry import Cylinder, GeometryModel, Plane
from anymesher.core import MeshCore
from anymesher.hybrid import generate_hybrid_mesh
from anymesher.mapped import generate_mesh as generate_mapped_mesh
from anymesher.quality import verify_mesh_quality
from anymesher.quality_v2 import evaluate_quality
from anymesher.serialize import mesh_to_dict


def _planar_model(vertices: Iterable[tuple[float, float, float]]) -> GeometryModel:
    coordinates = np.asarray(tuple(vertices), dtype=float)
    minimum = np.min(coordinates, axis=0)
    span = np.ptp(coordinates, axis=0)
    geometry = GeometryModel()
    points = geometry.add_points(coordinates)
    geometry.add_face(
        geometry.add_polyline(points, close=True),
        surface=Plane(
            minimum,
            np.asarray((span[0], 0.0, 0.0)),
            np.asarray((0.0, span[1], 0.0)),
        ),
    )
    return geometry


def _rectangle() -> GeometryModel:
    return _planar_model(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )


def _pentagon() -> GeometryModel:
    angles = np.linspace(0.0, 2.0 * np.pi, 6)[:-1] + np.pi / 2.0
    return _planar_model(
        tuple((float(np.cos(a)), float(np.sin(a)), 0.0) for a in angles)
    )


def _cylinder() -> GeometryModel:
    geometry = GeometryModel()
    radius = 1.0
    surface = Cylinder(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        radius,
        math.pi / 2.0,
        0.0,
        math.pi / 2.0,
    )
    points = [
        geometry.add_point(*surface.evaluate(u, v))
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    ]
    low_mid = geometry.add_point(*surface.evaluate(0.5, 0.0))
    high_mid = geometry.add_point(*surface.evaluate(0.5, 1.0))
    edges = (
        geometry.add_arc(points[0], low_mid, points[1]),
        geometry.add_line(points[1], points[2]),
        geometry.add_arc(points[2], high_mid, points[3]),
        geometry.add_line(points[3], points[0]),
    )
    geometry.add_face(edges, corners=(0, 1, 2, 3), surface=surface)
    return geometry


def _target_size(family: str, requested_elements: int) -> float:
    area = 1.0 if family == "mapped" else 2.377641290737884
    if family == "cylinder":
        area = math.pi * math.pi / 4.0
    return math.sqrt(area / float(requested_elements))


def _mesh_factory(family: str, requested_elements: int) -> Callable[[], Any]:
    target_size = _target_size(family, requested_elements)
    if family == "mapped":
        return lambda: generate_mapped_mesh(_rectangle(), target_size=target_size)
    if family == "native":
        return lambda: generate_hybrid_mesh(
            _pentagon(), target_size=target_size, strategy="native",
            native_backend="native"
        )
    if family == "cylinder":
        return lambda: generate_hybrid_mesh(
            _cylinder(), target_size=target_size, strategy="native",
            native_backend="native"
        )
    raise ValueError(f"unknown benchmark family {family!r}")


def _process_peak_rss_bytes() -> int:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _time_peak(call: Callable[[], Any]) -> tuple[Any, float, int, int]:
    tracemalloc.start()
    started = time.perf_counter()
    value = call()
    seconds = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, float(seconds), int(peak), _process_peak_rss_bytes()


def _mesh_arrays(mesh) -> dict[str, np.ndarray]:
    node_ids = np.asarray(sorted(mesh.nodes), dtype=np.int64)
    coordinates = np.asarray([mesh.nodes[int(item)] for item in node_ids], dtype=np.float64)
    triangle_ids = np.asarray(sorted(mesh.tris), dtype=np.int64)
    triangles = np.asarray(
        [mesh.tris[int(item)] for item in triangle_ids], dtype=np.int64
    ).reshape((-1, 3))
    quad_ids = np.asarray(sorted(mesh.quads), dtype=np.int64)
    quads = np.asarray(
        [mesh.quads[int(item)] for item in quad_ids], dtype=np.int64
    ).reshape((-1, 4))
    return {
        "node_ids": node_ids,
        "coordinates": coordinates,
        "triangle_ids": triangle_ids,
        "triangles": triangles,
        "quad_ids": quad_ids,
        "quads": quads,
    }


def _mesh_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in (
        "node_ids",
        "coordinates",
        "triangle_ids",
        "triangles",
        "quad_ids",
        "quads",
    ):
        values = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii"))
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _core_from(arrays: dict[str, np.ndarray]) -> MeshCore:
    return MeshCore.from_id_connectivity(
        arrays["coordinates"],
        node_ids=arrays["node_ids"],
        triangles=arrays["triangles"],
        triangle_ids=arrays["triangle_ids"],
        quadrilaterals=arrays["quads"],
        quad_ids=arrays["quad_ids"],
    )


def _element_quality_summary(values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"elements": int(len(values.element_ids))}
    for name in (
        "area",
        "aspect_ratio",
        "minimum_angle",
        "maximum_angle",
        "scaled_jacobian",
        "warpage",
    ):
        array = np.asarray(getattr(values, name), dtype=np.float64)
        result[name] = (
            None
            if array.size == 0
            else {"minimum": float(np.min(array)), "maximum": float(np.max(array))}
        )
    return result


def _native_quality_summary(values: Any) -> dict[str, Any]:
    return {
        "triangles": _element_quality_summary(values.triangles),
        "quadrilaterals": _element_quality_summary(values.quadrilaterals),
        "validity": {
            "errors": list(values.validity.errors),
            "warnings": list(values.validity.warnings),
        },
    }


def _measure_case(
    family: str,
    requested_elements: int,
    *,
    repeats: int,
    serialization_limit: int,
    deterministic_repeat_limit: int,
) -> dict[str, Any]:
    factory = _mesh_factory(family, requested_elements)
    generation_seconds: list[float] = []
    generation_peaks: list[int] = []
    generation_rss_peaks: list[int] = []
    generation_phase_samples: list[dict[str, Any]] = []
    mesh = None
    for _ in range(repeats):
        mesh, seconds, peak, rss_peak = _time_peak(factory)
        generation_seconds.append(seconds)
        generation_peaks.append(peak)
        generation_rss_peaks.append(rss_peak)
        generation_phase_samples.append(
            dict(getattr(mesh, "hybrid_diagnostics", {}))
        )
    assert mesh is not None

    arrays, array_seconds, array_peak, array_rss = _time_peak(lambda: _mesh_arrays(mesh))
    mesh_hash = _mesh_hash(arrays)
    core, core_seconds, core_peak, core_rss = _time_peak(lambda: _core_from(arrays))
    native_quality, native_quality_seconds, native_quality_peak, native_quality_rss = _time_peak(
        lambda: evaluate_quality(core)
    )
    quality, quality_seconds, quality_peak, quality_rss = _time_peak(lambda: verify_mesh_quality(mesh))

    element_ids = core.element_ids
    victims = element_ids[::100] if len(element_ids) else element_ids
    _damaged, damage_seconds, damage_peak, damage_rss = _time_peak(
        lambda: core.deactivate_elements(victims)
    )

    serialization: dict[str, Any] | None = None
    if mesh.num_elements <= serialization_limit:
        payload, seconds, peak, rss_peak = _time_peak(lambda: mesh_to_dict(mesh))
        serialization = {
            "seconds": seconds,
            "peak_traced_bytes": peak,
            "peak_process_rss_bytes": rss_peak,
            "top_level_keys": len(payload),
        }

    repeated_hash: str | None = None
    if mesh.num_elements <= deterministic_repeat_limit:
        repeated_hash = _mesh_hash(_mesh_arrays(factory()))

    return {
        "family": family,
        "requested_elements": int(requested_elements),
        "target_size": _target_size(family, requested_elements),
        "nodes": int(mesh.num_nodes),
        "elements": int(mesh.num_elements),
        "triangles": int(len(mesh.tris)),
        "quadrilaterals": int(len(mesh.quads)),
        "generation_seconds": generation_seconds,
        "generation_median_seconds": statistics.median(generation_seconds),
        "generation_peak_traced_bytes": max(generation_peaks),
        "generation_peak_process_rss_bytes": max(generation_rss_peaks),
        "generation_phase_samples": generation_phase_samples,
        "compatibility_array_conversion_seconds": array_seconds,
        "compatibility_array_peak_traced_bytes": array_peak,
        "compatibility_array_peak_process_rss_bytes": array_rss,
        "compact_core_conversion_seconds": core_seconds,
        "compact_core_peak_traced_bytes": core_peak,
        "compact_core_peak_process_rss_bytes": core_rss,
        "compact_core_memory_bytes": int(core.memory_bytes),
        "native_quality_seconds": native_quality_seconds,
        "native_quality_peak_traced_bytes": native_quality_peak,
        "native_quality_peak_process_rss_bytes": native_quality_rss,
        "native_quality": _native_quality_summary(native_quality),
        "quality_seconds": quality_seconds,
        "quality_peak_traced_bytes": quality_peak,
        "quality_peak_process_rss_bytes": quality_rss,
        "quality": quality.as_dict(),
        "damage_one_percent_seconds": damage_seconds,
        "damage_one_percent_peak_traced_bytes": damage_peak,
        "damage_one_percent_peak_process_rss_bytes": damage_rss,
        "serialization": serialization,
        "mesh_hash": mesh_hash,
        "repeated_mesh_hash": repeated_hash,
        "repeated_hash_identical": (
            None if repeated_hash is None else repeated_hash == mesh_hash
        ),
    }


def _scaling(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_family.setdefault(str(case["family"]), []).append(case)
    for family, values in by_family.items():
        values.sort(key=lambda item: int(item["elements"]))
        for first, second in zip(values, values[1:]):
            element_ratio = float(second["elements"]) / float(first["elements"])
            time_ratio = float(second["generation_median_seconds"]) / max(
                float(first["generation_median_seconds"]), 1.0e-12
            )
            memory_ratio = float(second["compact_core_memory_bytes"]) / max(
                float(first["compact_core_memory_bytes"]), 1.0
            )
            rows.append(
                {
                    "family": family,
                    "from_elements": int(first["elements"]),
                    "to_elements": int(second["elements"]),
                    "element_ratio": element_ratio,
                    "generation_time_ratio": time_ratio,
                    "generation_log_slope": math.log(time_ratio) / math.log(element_ratio),
                    "compact_memory_ratio": memory_ratio,
                    "compact_memory_log_slope": math.log(memory_ratio)
                    / math.log(element_ratio),
                }
            )
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(10_000, 100_000, 500_000))
    parser.add_argument(
        "--families", nargs="+", choices=("mapped", "native", "cylinder"), default=("mapped", "native", "cylinder")
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--serialization-limit", type=int, default=100_000)
    parser.add_argument("--deterministic-repeat-limit", type=int, default=100_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/native_hybrid/performance_results.json"),
    )
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.sizes):
        parser.error("all sizes must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    return args


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases: list[dict[str, Any]] = []
    started = time.perf_counter()
    report = {
        "schema": "anymesher.native_hybrid.performance",
        "version": 2,
        "status": "incomplete",
        "current_case": None,
        "failure": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "configuration": {
            "sizes": list(args.sizes),
            "families": list(args.families),
            "repeats": args.repeats,
            "serialization_limit": args.serialization_limit,
            "deterministic_repeat_limit": args.deterministic_repeat_limit,
        },
        "wall_seconds": 0.0,
        "cases": cases,
        "scaling": [],
    }
    _write_report(args.output, report)
    try:
        for family in args.families:
            for size in args.sizes:
                report["current_case"] = {
                    "family": family,
                    "requested_elements": int(size),
                    "phase": "generation",
                }
                report["wall_seconds"] = time.perf_counter() - started
                _write_report(args.output, report)
                print(f"measuring {family} at approximately {size:,} elements", flush=True)
                cases.append(
                    _measure_case(
                        family,
                        size,
                        repeats=args.repeats,
                        serialization_limit=args.serialization_limit,
                        deterministic_repeat_limit=args.deterministic_repeat_limit,
                    )
                )
                report["current_case"] = None
                report["wall_seconds"] = time.perf_counter() - started
                _write_report(args.output, report)
    except BaseException as error:
        report["status"] = "failed"
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "last_completed_cases": len(cases),
        }
        report["wall_seconds"] = time.perf_counter() - started
        _write_report(args.output, report)
        raise
    report["status"] = "complete"
    report["current_case"] = None
    report["wall_seconds"] = time.perf_counter() - started
    report["scaling"] = _scaling(cases)
    _write_report(args.output, report)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
