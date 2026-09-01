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


def _contains_component_alignment_defer(value: object) -> bool:
    if isinstance(value, dict):
        if (
            value.get("boundary_collar_skip_reason")
            == "declared_junction_requires_component_aligned_transition"
        ):
            return True
        return any(
            _contains_component_alignment_defer(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_component_alignment_defer(item) for item in value)
    return False


@pytest.mark.parametrize("target_size", (0.20, 0.15))
def test_automatic_fine_cylinder_plate_defers_component_alignment_safely(
    target_size: float,
) -> None:
    mesh = generate_hybrid_mesh(
        _cylinder_through_plate(),
        target_size=target_size,
        strategy="auto",
        beam_edges=(),
        member_ids=(),
        structured_options={"quality_policy": _POLICY},
        native_backend="python",
    )

    assert mesh.num_nodes > 0
    assert mesh.num_elements > 0
    assert mesh.declared_plate_junction_edges
    assert mesh.hybrid_diagnostics["structured_quality"]["accepted"] is True
    assert _contains_component_alignment_defer(
        mesh.hybrid_diagnostics["triangulation_backend_by_face"]
    )
