"""Residual triangle angle actions retain hard topology checks and diagnostics."""

import pytest

from anygeometry.model import GeometryModel

from anymesher.errors import MeshError
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.state import QuadMeshState
from anymesher.quad.validate import QuadQualityRejected, validate_planar_quad_result


def _thin_triangle() -> QuadMeshState:
    return QuadMeshState(
        {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.1)},
        {1: (1, 2, 3)},
    )


def test_triangle_angle_warning_keeps_default_rejection_and_reports_cells():
    state = _thin_triangle()
    with pytest.raises(QuadQualityRejected, match="corner angle"):
        validate_planar_quad_result(state)

    report = validate_planar_quad_result(
        state, t3_minimum_angle=15.0, t3_angle_action="warning",
    )
    record = report.to_dict()
    assert record["tri_count"] == 1
    assert record["t3_angle_warning_count"] == 1
    assert record["t3_angle_warning_cells"] == [1]
    assert record["min_t3_angle"] < 15.0
    assert record["quality_gates"]["t3_angle_action"] == "warning"
    assert validate_planar_quad_result(state, t3_minimum_angle=5.0).t3_angle_warning_count == 0


def test_angle_warning_does_not_admit_degenerate_triangle():
    state = QuadMeshState(
        {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (2.0, 0.0)},
        {1: (1, 2, 3)},
    )
    with pytest.raises(MeshError, match="not positively oriented"):
        validate_planar_quad_result(state, t3_angle_action="warning")


def test_public_angle_action_is_quad_only_and_not_strict_certification():
    geometry = GeometryModel()
    geometry.add_plate(geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ))
    with pytest.raises(MeshError, match="require quad_options"):
        generate_hybrid_mesh_result(
            geometry, target_size=0.5, quad_t3_angle_action="warning",
        )
    with pytest.raises(MeshError, match="strict certification"):
        generate_hybrid_mesh_result(
            geometry, target_size=0.5, quad_options=QuadMeshingOptions(),
            certification_mode="strict", quad_t3_angle_action="warning",
        )
    with pytest.raises(MeshError, match="finite angle"):
        generate_hybrid_mesh_result(
            geometry, target_size=0.5, quad_options=QuadMeshingOptions(),
            quad_t3_minimum_angle=float("nan"),
        )


def test_public_quad_mesh_records_effective_angle_action():
    geometry = GeometryModel()
    geometry.add_plate(geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0),
         (2.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    ))
    result = generate_hybrid_mesh_result(
        geometry, target_size=0.5, strategy="native",
        quad_options=QuadMeshingOptions(),
        quad_t3_minimum_angle=12.0, quad_t3_angle_action="warning",
    )
    rule = result.mesh.hybrid_diagnostics["t3_angle_rule"]
    assert rule == {"minimum_angle": 12.0, "action": "warning", "warning_count": 0}
    assert result.mesh.hybrid_diagnostics["certifiable"] is True
