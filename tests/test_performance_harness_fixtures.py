"""Light regression checks for benchmark geometry fixtures."""

from __future__ import annotations

import copy
import runpy
from pathlib import Path

from anygeometry import EntityRef
from anymesher.hybrid import generate_hybrid_mesh


def test_native_pentagon_fixture_preserves_registered_boundary_order() -> None:
    harness = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "native_hybrid_performance.py")
    )
    geometry = harness["_pentagon"]()

    mesh = generate_hybrid_mesh(
        geometry,
        target_size=0.2,
        strategy="native",
        native_backend="python",
    )

    assert mesh.shells
    face = geometry.faces[min(geometry.faces)]
    for oriented in face.loop:
        nodes = mesh.nodes_on(EntityRef("edge", oriented.edge))
        assert len(nodes) >= 2
        assert all(node in mesh.nodes for node in nodes)


def test_plate_hole_fixture_is_deterministic_and_reports_collars() -> None:
    harness = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks" / "native_hybrid_performance.py")
    )
    geometry = harness["_plate_with_hole"]()
    first = generate_hybrid_mesh(
        geometry, target_size=0.25, strategy="native", native_backend="python",
        recombine=True,
    )
    second = generate_hybrid_mesh(
        harness["_plate_with_hole"](), target_size=0.25, strategy="native",
        native_backend="python", recombine=True,
    )

    assert len(geometry.faces[min(geometry.faces)].holes) == 1
    assert harness["_mesh_hash"](harness["_mesh_arrays"](first)) == harness[
        "_mesh_hash"
    ](harness["_mesh_arrays"](second))
    summary = harness["_strategy_summary"](first.hybrid_diagnostics)
    assert "outer_hole_collar" in summary["selected_strategies"]
    assert 3 in summary["candidate_counts"]


def test_performance_comparator_enforces_speed_memory_and_hash_gates() -> None:
    comparator = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "benchmarks"
            / "compare_native_hybrid_performance.py"
        )
    )
    case = {
        "family": "plate_hole",
        "backend": "python",
        "requested_elements": 2000,
        "generation_median_seconds": 10.0,
        "generation_peak_traced_bytes": 100,
        "generation_peak_process_rss_bytes": 1000,
        "generation_phase_median_seconds": {"surface.collar": 5.0},
        "generation_strategy": {
            "selected_strategies": ["outer_hole_collar"],
            "candidate_counts": [3],
        },
        "mesh_hash": "a" * 64,
    }
    baseline = {"status": "complete", "cases": [case]}
    improved = copy.deepcopy(case)
    improved["generation_median_seconds"] = 8.0
    improved["generation_phase_median_seconds"]["surface.collar"] = 4.0
    candidate = {"status": "complete", "cases": [improved]}

    accepted = comparator["compare_reports"](baseline, candidate)
    assert accepted["passed"] is True
    changed = copy.deepcopy(candidate)
    changed["cases"][0]["mesh_hash"] = "b" * 64
    rejected = comparator["compare_reports"](baseline, changed)
    assert rejected["passed"] is False
    assert any("mesh hash changed" in item for item in rejected["failures"])
