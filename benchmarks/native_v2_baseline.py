"""Content-addressed native-v2 benchmark and acceptance contract.

Heavy scales are never run by CI. ``check-evidence`` executes a fixed bounded
native corpus; ``run`` performs one warmup and exactly seven samples.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


WARMUPS = 1
MEASUREMENTS = 7
SCALES = {"10k": 10_000, "100k": 100_000, "500k": 500_000}
PERFORMANCE_CASES = (
    "mapped_zero_use",
    "planar",
    "hole",
    "narrow_ligament",
    "intersection",
    "declared_junction",
    "rotated",
)
FROZEN_EVIDENCE_TESTS = (
    "tests/test_native_backend_defaults.py::test_present_partial_native_v2_abi_fails_hard",
    "tests/test_native_v2_foundation.py::test_loaded_extension_with_zero_native_v2_symbols_fails_hard",
    "tests/test_native_v2_foundation.py::test_large_native_metric_and_gradation_kernels_match_python_oracle",
    "tests/test_native_v2_foundation.py::test_mutable_t3_python_cpp_parity",
    "tests/test_native_v2_foundation.py::test_native_v2_physical_gradation_matches_python_oracle",
    "tests/test_native_v2_foundation.py::test_native_v2_uncertain_orientation_and_near_cocircle_use_or_match_oracle",
    "tests/test_native_v2_foundation.py::test_native_v2_material_loop_honors_python_signal",
    "tests/test_native_v2_foundation.py::test_mutable_t3_preserves_retained_owners_and_fails_hard_on_native_error",
)
QUALIFICATION_CORPUS = {
    "planar": "tests/test_frontal_delaunay.py",
    "curved": "tests/test_curved_native_qualification.py",
    "intersection": "tests/test_intersection_meshing.py",
    "declared_junction": "tests/test_s3_production.py",
    "hole": "tests/test_planar_native_qualification.py",
    "narrow_ligament": "tests/test_planar_native_qualification.py",
    "mixed_mapped_native": "tests/test_hybrid.py",
    "activity": "tests/test_mesh_persistence_contract.py",
    "incremental_component": "tests/test_structural_pipeline.py",
}
REQUIRED_CORPUS = frozenset(
    {
        "planar",
        "curved",
        "intersection",
        "declared_junction",
        "hole",
        "narrow_ligament",
        "mixed_mapped_native",
        "activity",
        "incremental_component",
    }
)
SOURCE_COMMIT_LENGTH = 40
MAX_COMPARABLE_ELEMENT_RATIO = 1.20


class _CancellationProbe(RuntimeError):
    pass


def _case(
    name: str,
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...], bool]:
    outer = np.asarray(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)))
    holes: tuple[np.ndarray, ...] = ()
    constraints: tuple[np.ndarray, ...] = ()
    declared = False
    if name == "hole":
        holes = (
            np.asarray(((1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5))),
        )
    elif name == "narrow_ligament":
        holes = (
            np.asarray(((0.4, 0.4), (3.6, 0.4), (3.6, 3.4), (0.4, 3.4))),
        )
    elif name in {"intersection", "declared_junction"}:
        constraints = (np.asarray(((0.0, 2.0), (4.0, 2.0))),)
        declared = name == "declared_junction"
    elif name == "rotated":
        angle = np.deg2rad(31.0)
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        outer = outer @ rotation.T
    elif name not in {"planar", "mapped_zero_use"}:
        raise ValueError(name)
    return outer, holes, constraints, declared


def _mesh_digest(mesh: Any) -> str:
    return sha256(
        b"".join(
            (
                mesh.node_coordinates.tobytes(),
                mesh.triangle_connectivity.tobytes(),
                mesh.quad_connectivity.tobytes(),
            )
        )
    ).hexdigest()


def _canonical_contract_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _canonical_contract_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_contract_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_canonical_contract_value(item) for item in value]
        return sorted(rows, key=lambda item: json.dumps(item, sort_keys=True))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_dict"):
        return _canonical_contract_value(value.to_dict())
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _digest_contract(value: Any) -> str:
    encoded = json.dumps(
        _canonical_contract_value(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _semantic_contract(
    mesh: Any,
    outer: np.ndarray,
    holes: tuple[np.ndarray, ...],
    constraints: tuple[np.ndarray, ...],
    declared: bool,
) -> dict[str, str]:
    state = vars(mesh) if hasattr(mesh, "__dict__") else {}

    def selected(*tokens: str) -> dict[str, Any]:
        return {
            name: value
            for name, value in sorted(state.items())
            if any(token in name.lower() for token in tokens)
        }

    protected_input = {
        "outer": outer,
        "holes": holes,
        "constraints": constraints,
        "declared_junction": declared,
        "mesh_protected_state": selected("protected", "boundary", "mandatory"),
    }
    return {
        "protected_topology": _digest_contract(protected_input),
        "associations": _digest_contract(selected("association", "_of_")),
        "ownership": _digest_contract(selected("owner", "elements_of_", "nodes_of_")),
        "activity": _digest_contract(selected("active", "activity")),
    }


def _serialization_bytes(mesh: Any) -> int:
    return sum(
        value.nbytes
        for value in (
            mesh.node_ids,
            mesh.node_coordinates,
            mesh.triangle_ids,
            mesh.triangle_connectivity,
            mesh.quad_ids,
            mesh.quad_connectivity,
        )
    )


def _peak_rss() -> int | None:
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                ] + [
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return None
            return int(counters.PeakWorkingSetSize)
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == "darwin" else 1024))
    except Exception:
        return None


def _dependency_versions() -> dict[str, str]:
    result = {}
    for distribution in ("ANYmesher", "ANYgeometry", "numpy"):
        try:
            result[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            result[distribution] = "source-uninstalled"
    return result


def _validate_hex(value: str, length: int, label: str) -> str:
    lowered = value.lower()
    if len(lowered) != length or any(
        character not in "0123456789abcdef" for character in lowered
    ):
        raise ValueError(f"{label} must be exactly {length} hexadecimal characters")
    return lowered


def _run(args: argparse.Namespace) -> int:
    from anygeometry import GeometryModel
    from anymesher import MetricFieldSpec, NativeMeshingOptions, __version__
    from anymesher.hybrid import _neutral_shell_core, generate_hybrid_mesh_result
    from anymesher.quality_v2 import evaluate_quality
    from anymesher.surface_mesh import SurfaceMeshOptions, mesh_planar_surface

    if args.scale in {"500k", "workstation"} and not args.allow_large:
        raise ValueError("500k/workstation evidence requires --allow-large")
    if args.scale == "workstation":
        if args.workstation_elements is None or args.workstation_elements < 500_000:
            raise ValueError("workstation-elements must be at least 500000")
        requested_elements = args.workstation_elements
    else:
        requested_elements = SCALES[args.scale]
    source_commit = _validate_hex(
        args.source_commit, SOURCE_COMMIT_LENGTH, "source commit"
    )
    wheel_sha256 = None
    if args.install_kind == "wheel":
        if not args.wheel_sha256:
            raise ValueError("wheel runs require --wheel-sha256")
        wheel_sha256 = _validate_hex(args.wheel_sha256, 64, "wheel SHA-256")
    elif args.wheel_sha256:
        raise ValueError("source runs must not carry a wheel SHA-256")

    outer, holes, constraints, declared = _case(args.case)
    target = (16.0 / requested_elements) ** 0.5
    native = NativeMeshingOptions()
    if args.route == "frontal":
        native = NativeMeshingOptions(
            point_placement="frontal_delaunay",
            metric_mode="isotropic_spatial",
            metric_field=MetricFieldSpec.uniform(target),
            max_insertions=max(100, requested_elements // 2),
        )
    options = SurfaceMeshOptions(
        target_size=target,
        backend=args.backend,
        recombine=True,
        declared_junction=declared,
        native_options=native,
    )

    mapped_geometry = None
    mapped_face = None
    if args.case == "mapped_zero_use":
        mapped_geometry = GeometryModel()
        mapped_points = mapped_geometry.add_points(
            (
                (0.0, 0.0, 0.0),
                (4.0, 0.0, 0.0),
                (4.0, 4.0, 0.0),
                (0.0, 4.0, 0.0),
            )
        )
        mapped_face = mapped_geometry.add_plate(mapped_points)
        mapped_geometry.add_sheet((mapped_face,))

    semantic_mesh = None

    def generate() -> tuple[Any, dict[str, Any]]:
        nonlocal semantic_mesh
        if mapped_geometry is not None and mapped_face is not None:
            result = generate_hybrid_mesh_result(
                mapped_geometry,
                target_size=target,
                strategy="mapped",
                recombine=True,
                native_backend=args.backend,
                native_options=native,
            )
            semantic_mesh = result.mesh
            diagnostics = {
                "requested_route": "mapped",
                "selected_route": result.strategy_by_face[mapped_face],
                "triangulation_backend": result.triangulation_backend_by_face.get(
                    mapped_face
                ),
            }
            return _neutral_shell_core(result.mesh), diagnostics
        diagnostics: dict[str, Any] = {}
        mesh = mesh_planar_surface(
            outer,
            holes,
            constraints,
            options=options,
            diagnostics=diagnostics,
        )
        semantic_mesh = mesh
        return mesh, diagnostics

    for _ in range(WARMUPS):
        generate()
    durations: list[float] = []
    digests: list[str] = []
    counts: list[int] = []
    mesh = None
    diagnostics: dict[str, Any] = {}
    rss_before = _peak_rss()
    for _ in range(MEASUREMENTS):
        started = perf_counter()
        mesh, diagnostics = generate()
        durations.append(perf_counter() - started)
        digests.append(_mesh_digest(mesh))
        counts.append(mesh.num_triangles + mesh.num_quads)
    rss_after = _peak_rss()
    if len(set(digests)) != 1 or len(set(counts)) != 1:
        raise RuntimeError("benchmark repetitions are not canonically deterministic")
    assert mesh is not None
    quality = evaluate_quality(mesh)
    cancellation_phases: list[str] = []
    cancellation_started = perf_counter()

    def cancel(phase: str) -> None:
        cancellation_phases.append(phase)
        if phase == "native surface triangulation start":
            raise _CancellationProbe("bounded benchmark cancellation")

    cancellation_observed = False
    try:
        mesh_planar_surface(
            outer,
            holes,
            constraints,
            options=options,
            cancellation_check=cancel,
        )
    except _CancellationProbe:
        cancellation_observed = True
    if not cancellation_observed:
        raise RuntimeError("cancellation probe missed the registered safe boundary")

    total_elements = counts[-1]
    record = {
        "schema": "anymesher.native-v2-baseline/2",
        "source_commit": source_commit,
        "package_version": __version__,
        "case": args.case,
        "scale": args.scale,
        "requested_elements": requested_elements,
        "actual_elements": total_elements,
        "element_count_ratio": total_elements / requested_elements,
        "route": args.route,
        "warmups": WARMUPS,
        "repetitions": MEASUREMENTS,
        "durations_seconds": durations,
        "median_seconds": statistics.median(durations),
        "peak_rss_before_bytes": rss_before,
        "peak_rss_bytes": rss_after,
        "mesh_digest": digests[0],
        "semantic_contract": _semantic_contract(
            semantic_mesh if semantic_mesh is not None else mesh,
            outer,
            holes,
            constraints,
            declared,
        ),
        "repetition_digests": digests,
        "nodes": mesh.num_nodes,
        "triangles": mesh.num_triangles,
        "quadrilaterals": mesh.num_quads,
        "q4_fraction": mesh.num_quads / max(total_elements, 1),
        "serialization_bytes": _serialization_bytes(mesh),
        "quality": {
            "minimum_scaled_jacobian": quality.minimum_scaled_jacobian,
            "maximum_aspect_ratio": quality.maximum_aspect_ratio,
            "minimum_angle": quality.minimum_angle,
            "maximum_angle": quality.maximum_angle,
            "maximum_warpage": quality.maximum_warpage,
        },
        "quality_policy": {
            "minimum_scaled_jacobian": "no_decrease",
            "maximum_aspect_ratio": "no_increase",
            "minimum_angle": "no_decrease",
            "maximum_angle": "no_increase",
            "maximum_warpage": "no_increase",
        },
        "alignment": {
            "lattice_alignment": diagnostics.get("lattice_alignment"),
            "selected_strategy": diagnostics.get("quality_optimization", {}).get(
                "selected_strategy"
            ),
            "boundary_alignment": diagnostics.get("quality_optimization", {}).get(
                "boundary_alignment"
            ),
        },
        "backend": {
            "requested": diagnostics.get("requested_backend"),
            "selected": diagnostics.get("selected_backend"),
            "actual": diagnostics.get("actual_backend"),
        },
        "cancellation": {
            "observed": cancellation_observed,
            "latency_seconds": perf_counter() - cancellation_started,
            "phases": cancellation_phases,
        },
        "provenance": {
            "install_kind": args.install_kind,
            "wheel_sha256": wheel_sha256,
            "compiler_id": args.compiler_id,
            "dependencies": _dependency_versions(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "benchmark_configuration": {
            "target_size": target,
            "allow_large": bool(args.allow_large),
            "fixed_warmups": WARMUPS,
            "fixed_measurements": MEASUREMENTS,
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _load_record(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "anymesher.native-v2-baseline/2":
        raise ValueError(f"unsupported benchmark evidence: {path}")
    if value.get("warmups") != WARMUPS or value.get("repetitions") != MEASUREMENTS:
        raise ValueError(f"non-canonical repetition contract: {path}")
    if len(value.get("repetition_digests", ())) != MEASUREMENTS or len(
        set(value["repetition_digests"])
    ) != 1:
        raise ValueError(f"non-deterministic benchmark evidence: {path}")
    return value


def _rss_ratio_exceeds(candidate: Any, baseline: Any, limit: float) -> bool:
    return candidate is None or baseline is None or candidate > limit * baseline


def _compare(args: argparse.Namespace) -> int:
    legacy = _load_record(args.legacy)
    frontal = _load_record(args.frontal)
    for field in (
        "case",
        "scale",
        "source_commit",
        "requested_elements",
        "backend",
        "provenance",
        "benchmark_configuration",
        "quality_policy",
        "semantic_contract",
    ):
        if legacy[field] != frontal[field]:
            raise ValueError(f"benchmark evidence differs in {field}")
    counts = (legacy["actual_elements"], frontal["actual_elements"])
    ratio = max(counts) / max(min(counts), 1)
    failures = []
    quality_directions = {
        "minimum_scaled_jacobian": 1,
        "maximum_aspect_ratio": -1,
        "minimum_angle": 1,
        "maximum_angle": -1,
        "maximum_warpage": -1,
    }
    for metric, direction in quality_directions.items():
        baseline_value = float(legacy["quality"][metric])
        candidate_value = float(frontal["quality"][metric])
        tolerance = 1.0e-12 * max(1.0, abs(baseline_value))
        if direction * (candidate_value - baseline_value) < -tolerance:
            failures.append(f"quality regression in {metric}")
    if ratio > MAX_COMPARABLE_ELEMENT_RATIO:
        failures.append(
            f"element-count ratio {ratio:.6g} exceeds {MAX_COMPARABLE_ELEMENT_RATIO}"
        )
    if (
        legacy["scale"] == "100k"
        and frontal["median_seconds"] > 1.25 * legacy["median_seconds"]
    ):
        failures.append("100k Frontal-Delaunay median exceeds 1.25x legacy")
    if legacy["case"] == "mapped_zero_use":
        if frontal["median_seconds"] > 1.03 * legacy["median_seconds"]:
            failures.append("mapped zero-use median regression exceeds 3%")
        if _rss_ratio_exceeds(
            frontal["peak_rss_bytes"], legacy["peak_rss_bytes"], 1.05
        ):
            failures.append("mapped zero-use peak-RSS regression exceeds 5%")
    if _rss_ratio_exceeds(
        frontal["peak_rss_bytes"], legacy["peak_rss_bytes"], 2.0
    ):
        failures.append("native-v2 peak RSS exceeds 2x legacy")
    result = {
        "schema": "anymesher.native-v2-acceptance/1",
        "legacy": str(args.legacy),
        "frontal": str(args.frontal),
        "element_count_ratio": ratio,
        "accepted": not failures,
        "failures": failures,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not failures else 2


def _check_contract() -> int:
    if frozenset(QUALIFICATION_CORPUS) != REQUIRED_CORPUS:
        raise RuntimeError("native-v2 qualification corpus drift")
    if WARMUPS != 1 or MEASUREMENTS != 7 or SCALES != {
        "10k": 10_000,
        "100k": 100_000,
        "500k": 500_000,
    }:
        raise RuntimeError("native-v2 benchmark scale/repetition contract drift")
    print(
        json.dumps(
            {
                "schema": "anymesher.native-v2-contract/1",
                "corpus": QUALIFICATION_CORPUS,
                "performance_cases": PERFORMANCE_CASES,
                "scales": SCALES,
                "warmups": WARMUPS,
                "measurements": MEASUREMENTS,
                "thresholds": {
                    "frontal_100k_runtime_ratio": 1.25,
                    "mapped_zero_use_runtime_ratio": 1.03,
                    "mapped_zero_use_peak_rss_ratio": 1.05,
                    "native_v2_peak_rss_ratio": 2.0,
                    "comparable_element_ratio": MAX_COMPARABLE_ELEMENT_RATIO,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _check_evidence(evidence_pairs: list[tuple[Path, Path]]) -> int:
    _check_contract()
    root = Path(__file__).resolve().parents[1]
    missing = [
        f"{label}:{relative}"
        for label, relative in QUALIFICATION_CORPUS.items()
        if not (root / relative).is_file()
    ]
    if missing:
        raise RuntimeError(f"native-v2 qualification corpus paths are absent: {missing}")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *FROZEN_EVIDENCE_TESTS],
        cwd=root,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"bounded native-v2 evidence corpus failed with exit {completed.returncode}"
        )
    outcomes = []
    for legacy_path, frontal_path in evidence_pairs:
        status = _compare(
            argparse.Namespace(legacy=legacy_path, frontal=frontal_path)
        )
        outcomes.append(
            {
                "legacy": str(legacy_path),
                "frontal": str(frontal_path),
                "accepted": status == 0,
            }
        )
        if status != 0:
            return status
    print(
        json.dumps(
            {
                "status": "evidence_contract_passed",
                "corpus_paths": QUALIFICATION_CORPUS,
                "performance_evidence": (
                    outcomes if outcomes else "not_supplied_merge_blocker"
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check-contract")
    evidence = subparsers.add_parser("check-evidence")
    evidence.add_argument(
        "--evidence-pair",
        action="append",
        nargs=2,
        type=Path,
        default=[],
        metavar=("LEGACY_JSON", "FRONTAL_JSON"),
    )
    run = subparsers.add_parser("run")
    run.add_argument("--case", choices=PERFORMANCE_CASES, required=True)
    run.add_argument("--scale", choices=(*SCALES, "workstation"), required=True)
    run.add_argument("--workstation-elements", type=int)
    run.add_argument("--allow-large", action="store_true")
    run.add_argument("--route", choices=("legacy", "frontal"), required=True)
    run.add_argument(
        "--backend", choices=("auto", "python", "compiled"), required=True
    )
    run.add_argument("--install-kind", choices=("source", "wheel"), required=True)
    run.add_argument("--source-commit", required=True)
    run.add_argument("--wheel-sha256")
    run.add_argument("--compiler-id", required=True)
    run.add_argument("--output", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--legacy", type=Path, required=True)
    compare.add_argument("--frontal", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "check-contract":
        return _check_contract()
    if args.command == "check-evidence":
        return _check_evidence(args.evidence_pair)
    if args.command == "compare":
        return _compare(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
