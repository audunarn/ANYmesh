"""Focused tests for the runtime operation-count contract (M1/M2/M6 plumbing)."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import anygeometry
from anygeometry.generators import plate

import anymesher as am
from anymesher._runtime_counters import (
    geometry_runtime_diagnostics,
    merge_operation_counts,
    new_operation_counts,
    operation_counts_scope,
)
from anymesher.preparation import _report_hash, prepare_structural_closure


def test_scope_accumulates_and_isolates_counts() -> None:
    first = new_operation_counts()
    second = new_operation_counts()
    with operation_counts_scope(first):
        from anymesher._runtime_counters import add_operation_count

        add_operation_count("mesher.bvh_builds")
        add_operation_count("mesher.bvh_builds")
        add_operation_count("mesher.bvh_lookups", 3)
        with operation_counts_scope(second):
            add_operation_count("mesher.bvh_lookups")
    assert first == {"mesher.bvh_builds": 2, "mesher.bvh_lookups": 3}
    assert second == {"mesher.bvh_lookups": 1}


def test_merge_sums_integers_and_applies_prefixes() -> None:
    target = {"mesher.bvh_builds": 1}
    merge_operation_counts(
        target,
        {"bvh_builds": 2, "attachment_clip_calls": 4, "ignored": "x", "flag": True},
        prefix="geometry.",
    )
    assert target == {
        "mesher.bvh_builds": 1,
        "geometry.bvh_builds": 2,
        "geometry.attachment_clip_calls": 4,
    }


def test_merge_without_prefix_sums_candidate_chain_work() -> None:
    chain = {"mesher.bvh_builds": 1}
    merge_operation_counts(chain, {"mesher.bvh_builds": 2, "mesher.bvh_lookups": 5})
    assert chain == {"mesher.bvh_builds": 3, "mesher.bvh_lookups": 5}


def test_geometry_diagnostics_fallback_yields_empty_mapping() -> None:
    if hasattr(anygeometry, "runtime_diagnostics"):  # pragma: no cover
        return
    with geometry_runtime_diagnostics() as counts:
        assert counts == {}
    assert counts == {}


def test_geometry_diagnostics_context_merges_prefixed_counts(
    monkeypatch,
) -> None:
    module = SimpleNamespace()

    @contextmanager
    def collect():
        yield {"attachment_clip_calls": 2, "attachment_bounds_tests": 5,
                "attachment_bounds_pruned": 1}

    module.collect_runtime_diagnostics = collect
    monkeypatch.setitem(
        __import__("sys").modules, "anygeometry.runtime_diagnostics", module
    )
    counts = new_operation_counts()
    with geometry_runtime_diagnostics() as geometry_counts:
        merge_operation_counts(counts, geometry_counts, prefix="geometry.")
    assert counts == {
        "geometry.attachment_clip_calls": 2,
        "geometry.attachment_bounds_tests": 5,
        "geometry.attachment_bounds_pruned": 1,
    }


def test_counters_never_enter_preparation_records() -> None:
    from anygeometry import GeometryModel

    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    second = geometry.add_plate(
        geometry.add_points(
            ((3.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 1.0, 0.0), (3.0, 1.0, 0.0))
        )
    )
    counts = new_operation_counts()
    with operation_counts_scope(counts):
        working, report = prepare_structural_closure(
            geometry, face_ids=(first, second), beam_edges=()
        )
    assert report is not None
    payload = json.dumps(report.to_dict(), sort_keys=True)
    assert "operation_counts" not in payload
    assert "bvh_builds" not in payload
    assert "chart_query_batches" not in payload
    assert _report_hash(report) == report.preparation_hash
    # The batched chart qualification of the two selected faces was counted.
    assert counts.get("mesher.chart_query_batches") == 1
    assert counts.get("mesher.chart_query_faces") == 2


def test_published_runtime_carries_operation_counts() -> None:
    geometry = plate(2.0, 1.0, semantic_group="runtime_counts_plate")
    result = am.generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        strategy="auto",
        structural_preparation=False,
        certification_mode="interactive",
    )
    runtime = result.mesh.hybrid_diagnostics["runtime"]
    assert runtime["candidate_count"] == 1
    assert runtime["connectivity_application_count"] == 1
    counts = runtime["operation_counts"]
    assert isinstance(counts, dict)
    assert all(isinstance(value, int) for value in counts.values())
    assert all(
        key.startswith(("mesher.", "geometry.")) for key in counts
    )
    # Runtime counters live only under the runtime block.
    assert "operation_counts" not in result.mesh.hybrid_diagnostics
    assert am.verify_mesh_quality(result.mesh).num_shell_elements > 0


def test_geometry_diagnostics_broken_module_propagates(monkeypatch) -> None:
    # A present but broken runtime_diagnostics module (missing entry point)
    # must propagate its import failure, not fall back silently.
    module = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "anygeometry.runtime_diagnostics", module)
    with pytest.raises(ImportError):
        geometry_runtime_diagnostics()
