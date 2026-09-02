"""Regression for fine cylinder-through-plate automatic meshing."""

from __future__ import annotations

import pytest

from anygeometry.generators.structural import cylinder
from anymesher.hybrid import generate_hybrid_mesh


_POLICY = {
    "minimum_scaled_jacobian": 0.1,
    "maximum_aspect_ratio": 5.0,
    "minimum_angle": 20.0,
    "maximum_angle": 160.0,
    "maximum_warpage": 0.1,
}


def _cylinder_through_plate():
    geometry = cylinder(
        0.5,
        2.0,
        circumferential_segments=12,
        longitudinal_spacing=0.5,
        ring_spacing=1.0,
    )
    geometry.add_plate(
        geometry.add_points(
            (
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            )
        )
    )
    return geometry


def _quality_reports(value: object) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    if isinstance(value, dict):
        quality = value.get("quality_optimization")
        if isinstance(quality, dict):
            reports.append(quality)
        for item in value.values():
            reports.extend(_quality_reports(item))
    if isinstance(value, (list, tuple)):
        for item in value:
            reports.extend(_quality_reports(item))
    return reports


def _alignment_scopes(value: object) -> list[dict[str, object]]:
    scopes: list[dict[str, object]] = []
    if isinstance(value, dict):
        scope = value.get("declared_junction_alignment_scope")
        if isinstance(scope, dict):
            scopes.append(scope)
        for item in value.values():
            scopes.extend(_alignment_scopes(item))
    if isinstance(value, (list, tuple)):
        for item in value:
            scopes.extend(_alignment_scopes(item))
    return scopes


@pytest.mark.parametrize("target_size", (0.25, 0.20, 0.15))
def test_automatic_fine_cylinder_plate_evaluates_component_alignment_safely(
    target_size: float,
) -> None:
    mesh = generate_hybrid_mesh(
        _cylinder_through_plate(),
        target_size=target_size,
        strategy="auto",
        beam_edges=(),
        member_ids=(),
        structured_options={"quality_policy": _POLICY},
        structural_preparation=True,
        native_backend="python",
    )

    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert mesh.hybrid_diagnostics["requested_target_size"] == target_size
    assert mesh.declared_plate_junction_edges
    assert mesh.hybrid_diagnostics["structured_quality"]["accepted"] is True
    reports = _quality_reports(
        mesh.hybrid_diagnostics["triangulation_backend_by_face"]
    )
    junction_reports = [
        report
        for report in reports
        if report.get("complex_geometry", {}).get("declared_junction") is True
    ]
    assert len(junction_reports) == 2
    rejection = mesh.hybrid_diagnostics.get("alignment_candidate_rejected")
    if target_size in {0.25, 0.20}:
        assert rejection is None
    if rejection is None:
        scopes = _alignment_scopes(
            mesh.hybrid_diagnostics["triangulation_backend_by_face"]
        )
        assert len(scopes) == 2
        assert sum(scope["outer_boundary"] == "evaluated" for scope in scopes) == 1
        assert sum(
            scope["outer_boundary"]
            == "skipped_declared_junction_only_boundary"
            for scope in scopes
        ) == 1
        assert all(
            scope["hole_boundary"] == "skipped_declared_junction_interface"
            for scope in scopes
        )
        assert any(
            report["selected_strategy"].startswith("outer_boundary_collar")
            for report in junction_reports
        )
    else:
        assert rejection["reason"] == "whole_mesh_quality_regression"
        assert rejection["aligned_quality"]["accepted"] is False
        assert rejection["accepted_baseline_quality"]["accepted"] is True
        comparison = rejection["comparison"]
        assert comparison["rejection_metrics"]
        assert comparison["aligned_growth_violation_count"] >= 0
        assert comparison["baseline_growth_violation_count"] == 0
        assert all(
            report["complex_geometry"]["alignment_evaluation"]
            == "whole_mesh_quality_fallback"
            for report in junction_reports
        )
        assert all(
            report["boundary_collar_skip_reason"]
            == "whole_mesh_quality_fallback"
            for report in junction_reports
        )
    assert all(
        report["final_quality"]["invalid_element_count"] == 0
        for report in junction_reports
    )
