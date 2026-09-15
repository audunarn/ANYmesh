"""One guarded native-v2 development case with retained per-sample evidence.

The default is one warmup and seven unprofiled samples. A single sample is an
attribution probe only and is never a formal before/after comparison record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tomllib
from typing import Any

from performance_dev import guard_request


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), "rev-parse", "HEAD"),
        capture_output=True, text=True, check=True, timeout=10,
    )
    return result.stdout.strip()


def _source_version(root: Path) -> str:
    with (root / "pyproject.toml").open("rb") as stream:
        value = tomllib.load(stream)["project"]["version"]
    if not isinstance(value, str) or not value:
        raise RuntimeError("isolated source version is missing")
    return value


def _bind_imports(root: Path, install_kind: str) -> dict[str, str]:
    if install_kind == "source":
        sys.path.insert(0, str(root / "src"))
    import anymesher
    import anygeometry
    import numpy

    origin = Path(anymesher.__file__).resolve()
    source = (root / "src" / "anymesher").resolve()
    if install_kind == "source" and not origin.is_relative_to(source):
        raise RuntimeError("source benchmark imported ANYmesher outside the isolated worktree")
    if install_kind == "wheel" and ("Github" in origin.parts or "github" in origin.parts):
        raise RuntimeError("wheel benchmark imported an editable source checkout")
    return {
        "anymesher_origin": str(origin),
        "anygeometry_origin": str(Path(anygeometry.__file__).resolve()),
        "numpy_version": numpy.__version__,
        "anymesher_version": (
            _source_version(root) if install_kind == "source" else
            importlib.metadata.version("ANYmesher")
        ),
        "installed_distribution_metadata_version": importlib.metadata.version("ANYmesher"),
        "anygeometry_version": importlib.metadata.version("ANYgeometry"),
    }


def _named_counts(value: Any, name: str) -> list[int]:
    result: list[int] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == name and type(nested) is int and nested >= 0:
                result.append(nested)
            else:
                result.extend(_named_counts(nested, name))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            result.extend(_named_counts(nested, name))
    return result


def _work(diag: Any, route: str) -> dict[str, int | None]:
    backends = diag.get("triangulation_backend_by_face", {}) if isinstance(diag, dict) else {}
    face_rows = [value["native_v2"] for _, value in sorted(backends.items())
                 if isinstance(value, dict) and isinstance(value.get("native_v2"), dict)]

    def first_available(*names: str) -> int | None:
        for name in names:
            if face_rows:
                values = [row.get(name) for row in face_rows]
                if all(type(value) is int and value >= 0 for value in values):
                    return sum(values)
                continue
            values = _named_counts(diag, name)
            if values:
                return sum(values)
        return None

    published = first_available("published_insertions", "accepted_insertions")
    attempted = first_available("insertions")
    if route == "legacy":
        published = attempted = 0
    return {
        "published_insertions": published,
        "insertions": attempted,
        "topology_operations": first_available("topology_operations"),
    }


def _association_digest(mesh: Any) -> str:
    payload = {
        "nodes_of_edge": [
            [int(edge), [int(node) for node in mesh.nodes_of_edge[edge]]]
            for edge in sorted(mesh.nodes_of_edge)
        ],
        "elements_of_face": [
            [int(face), [int(element) for element in mesh.elements_of_face[face]]]
            for face in sorted(mesh.elements_of_face)
        ],
        "node_of_vertex": [
            [int(vertex), int(mesh.node_of_vertex[vertex])]
            for vertex in sorted(mesh.node_of_vertex)
        ],
        "declared_junction_edges": sorted(
            [int(first), int(second)]
            for first, second in getattr(mesh, "declared_plate_junction_edges", ())
        ),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def _sample_mesh(mesh: Any, route: str, stages: list[str]) -> dict[str, Any]:
    from native_hybrid_performance import (
        _core_from, _flatten_phase_seconds, _mesh_arrays, _mesh_hash,
        _process_peak_rss_bytes,
    )
    from anymesher.quality import verify_mesh_quality

    arrays = _mesh_arrays(mesh)
    core = _core_from(arrays)
    triangle_storage = int(len(core.triangle_active))
    quad_storage = int(len(core.quad_active))
    triangle_active = int(sum(bool(item) for item in core.triangle_active))
    quad_active = int(sum(bool(item) for item in core.quad_active))
    if triangle_active + quad_active != mesh.num_elements:
        raise RuntimeError("active output count differs from published mesh")
    quality = verify_mesh_quality(mesh).as_dict()
    diag = dict(getattr(mesh, "hybrid_diagnostics", {}))
    return {
        "mesh_digest": _mesh_hash(arrays),
        "association_digest": _association_digest(mesh),
        "actual_elements": triangle_active + quad_active,
        "triangles": triangle_active, "quadrilaterals": quad_active,
        "stored_triangle_rows": triangle_storage,
        "stored_quad_rows": quad_storage,
        "q4_fraction": quad_active / max(triangle_active + quad_active, 1),
        "quality": quality,
        "stage_seconds": _flatten_phase_seconds(diag),
        "native_diagnostics": diag,
        "native_work": _work(diag, route),
        "callback_checkpoints": len(stages),
        "peak_rss_bytes": _process_peak_rss_bytes(),
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--requested-elements", type=int, required=True)
    parser.add_argument("--route", choices=("legacy", "frontal"), required=True)
    parser.add_argument("--backend", choices=("auto", "python", "compiled"), required=True)
    parser.add_argument("--install-kind", choices=("source", "wheel"), required=True)
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurements", type=int, choices=(1, 7), default=7)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args(argv)
    args.allow_large = False
    args.workstation_elements = None
    args.worker_count = 1
    guard_request(args)
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    root = Path(__file__).resolve().parents[1]
    report_root = root / "reports" / "performance_dev"
    output = args.output.resolve()
    if not output.is_relative_to(report_root.resolve()):
        raise ValueError("development reports must stay in the isolated evidence root")
    if output.exists():
        raise FileExistsError(output)
    phase = "preflight"
    samples: list[dict[str, Any]] = []
    try:
        actual_commit = _git_head(root)
        if actual_commit != args.source_commit:
            raise RuntimeError("declared source commit does not match worktree HEAD")
        origins = _bind_imports(root, args.install_kind)
        from performance_dev_cases import prepare_case, generate_once
        from anymesher.serialize import mesh_to_dict

        prepared = prepare_case(args.case, args.requested_elements, args.route)
        phase = "warmup"
        generate_once(prepared, args.backend)
        last_mesh = None
        for ordinal in range(1, args.measurements + 1):
            phase = f"measurement-{ordinal}"
            stages: list[str] = []
            started = time.perf_counter()
            mesh = generate_once(prepared, args.backend, cancellation_stages=stages)
            seconds = time.perf_counter() - started
            sample = _sample_mesh(mesh, args.route, stages)
            sample["duration_seconds"] = seconds
            sample["ordinal"] = ordinal
            if prepared.callback_present and not stages:
                raise RuntimeError("callback-present lane executed no cancellation checkpoints")
            _write_new(output.with_name(output.name + f".sample-{ordinal:02d}.json"), sample)
            samples.append(sample)
            last_mesh = mesh
        assert last_mesh is not None
        phase = "serialization"
        started = time.perf_counter()
        serialization = json.dumps(mesh_to_dict(last_mesh), sort_keys=True,
                                   separators=(",", ":"), allow_nan=False).encode("utf-8")
        serialization_seconds = time.perf_counter() - started
        hashes = {(item["mesh_digest"], item["association_digest"])
                  for item in samples}
        if len(hashes) != 1:
            raise RuntimeError("repeated development mesh output changed")
        if args.route == "frontal" and any(
            item["native_work"]["published_insertions"] is None for item in samples
        ):
            raise RuntimeError("frontal insertion work is not evidenced by diagnostics")
        if args.case == "planar_insertion_work" and any(
            item["native_work"]["published_insertions"] < 1 for item in samples
        ):
            raise RuntimeError("registered insertion-work fixture did not insert a point")
        representative = samples[0]
        provenance = {
            **origins,
            "platform": platform.platform(), "machine": platform.node(),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "compiler_id": args.compiler_id,
            "wheel_sha256": args.wheel_sha256,
            "commit_binding": actual_commit,
        }
        stage_names = sorted({name for item in samples for name in item["stage_seconds"]})
        report = {
            "schema": "anymesher.native-v2-baseline/3",
            "status": "complete", "created_utc": _utc(),
            "case": args.case, "scale": args.scale,
            "requested_elements": args.requested_elements,
            "route": args.route, "backend": args.backend,
            "install_kind": args.install_kind,
            "source_commit": actual_commit,
            "provenance": provenance,
            "benchmark_configuration": {
                "target_size": prepared.target_size,
                "callback_present": prepared.callback_present,
                "recombine": prepared.recombine,
                "strategy": prepared.strategy,
            },
            "semantic_contract": "published-mesh-and-protected-ownership/1",
            "quality_policy": "established-default",
            "warmups": 1, "repetitions": args.measurements,
            "mesh_digest": representative["mesh_digest"],
            "association_digest": representative["association_digest"],
            "actual_elements": representative["actual_elements"],
            "triangles": representative["triangles"],
            "quadrilaterals": representative["quadrilaterals"],
            "stored_triangle_rows": representative["stored_triangle_rows"],
            "stored_quad_rows": representative["stored_quad_rows"],
            "q4_fraction": representative["q4_fraction"],
            "quality": representative["quality"],
            "serialization_bytes": len(serialization),
            "serialization_seconds": serialization_seconds,
            "durations_seconds": [item["duration_seconds"] for item in samples],
            "generation_median_seconds": statistics.median(
                item["duration_seconds"] for item in samples
            ),
            "peak_rss_bytes": max(item["peak_rss_bytes"] for item in samples),
            "stage_median_seconds": {
                name: statistics.median(item["stage_seconds"].get(name, math.nan)
                                        for item in samples)
                for name in stage_names
                if all(name in item["stage_seconds"] for item in samples)
            },
            "native_work_samples": [item["native_work"] for item in samples],
            "callback_checkpoint_counts": [item["callback_checkpoints"] for item in samples],
            "sample_paths": [str(output.with_name(output.name + f".sample-{i:02d}.json"))
                             for i in range(1, len(samples) + 1)],
        }
        _write_new(output, report)
        return 0
    except BaseException as error:
        failure = {
            "schema": "anymesher.performance-dev-failure/1",
            "created_utc": _utc(), "phase": phase,
            "error_type": type(error).__name__, "error": str(error),
            "completed_samples": len(samples),
            "sample_paths": [str(output.with_name(output.name + f".sample-{i:02d}.json"))
                             for i in range(1, len(samples) + 1)],
        }
        try:
            _write_new(output.with_name(output.name + ".failure.json"), failure)
        except BaseException:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
