from copy import deepcopy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def comparison(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("performance_before_after").compare


def report(commit):
    return {
        "schema": "anymesher.native-v2-baseline/3", "case": "planar",
        "scale": "10k", "requested_elements": 10_000, "route": "frontal",
        "backend": {"actual": "anymesher-cpp17"},
        "benchmark_configuration": {"target_size": 0.02},
        "semantic_contract": {"protected": "abc"},
        "quality_policy": {"minimum_angle": "no_decrease"},
        "warmups": 1, "repetitions": 7, "source_commit": commit,
        "provenance": {"machine": "AMD64", "numpy": "2.4.6", "wheel_sha256": None},
        "mesh_digest": "d" * 64, "actual_elements": 9_500,
        "triangles": 100, "quadrilaterals": 9_400, "q4_fraction": 9_400 / 9_500,
        "quality": {"minimum_scaled_jacobian": 0.5},
        "serialization_bytes": 100_000,
        "native_work_samples": [{"insertions": 20, "published_insertions": 20,
                                 "topology_operations": 24} for _ in range(7)],
        "durations_seconds": [2.0] * 7, "peak_rss_bytes": 1_000_000,
    }


def test_same_route_distinct_commit_improvement_is_measurable(comparison):
    before, after = report("a" * 40), report("b" * 40)
    after["durations_seconds"] = [1.5] * 7
    result = comparison(before, after)
    assert result["runtime_ratio"] == 0.75
    assert result["before_commit"] != result["after_commit"]


@pytest.mark.parametrize("field", (
    "mesh_digest", "actual_elements", "benchmark_configuration", "quality",
))
def test_changed_work_or_mesh_cannot_claim_performance_speedup(comparison, field):
    before, after = report("a" * 40), report("b" * 40)
    after[field] = "changed"
    with pytest.raises(ValueError):
        comparison(before, after)


def test_fewer_accepted_insertions_cannot_claim_speedup(comparison):
    before, after = report("a" * 40), report("b" * 40)
    after["native_work_samples"] = deepcopy(after["native_work_samples"])
    for row in after["native_work_samples"]:
        row["published_insertions"] = 19
    with pytest.raises(ValueError, match="accepted refinement work differs"):
        comparison(before, after)
