"""Quad-first triangle-angle controls do not reach the legacy fallback routes.

Regression: the ANYfem drawing barge at target 2.0 m requests quad-first with
``quad_t3_angle_action="warning"``. Quad-first refused the layout (Q4 aspect
ratio), the automatic route dropped ``quad_options`` but kept the angle
controls, and the fallback raised "quad triangle angle controls require
quad_options" instead of meshing or reporting the real refusal.
"""
import pytest
from anygeometry import GeometryModel
from anymesher.recovery import generate_automatic_mesh_result, MeshAutomationOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.validate import QuadQualityRejected


def plate():
    geometry = GeometryModel()
    geometry.add_plate(geometry.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))))
    return geometry


@pytest.mark.parametrize("controls", [{"quad_t3_angle_action": "warning"},
                                      {"quad_t3_minimum_angle": 20.0},
                                      {"quad_t3_minimum_angle": 20.0, "quad_t3_angle_action": "warning"}])
@pytest.mark.parametrize("strategy", ["auto", "native"])
def test_quad_first_refusal_falls_back_without_angle_controls(monkeypatch, strategy, controls):
    import anymesher.recovery as recovery
    original = recovery.generate_hybrid_mesh_result
    calls = []

    def reject(geometry, **options):
        calls.append(dict(options))
        if options.get("quad_options") is not None:
            raise QuadQualityRejected("owned quality refusal")
        return original(geometry, **options)

    monkeypatch.setattr(recovery, "generate_hybrid_mesh_result", reject)
    options = dict(strategy=strategy, target_size=.5, quad_options=QuadMeshingOptions(),
                   layout_policy="adaptive", **controls)
    result = generate_automatic_mesh_result(plate(), **options)

    assert result.attempts[0]["status"] == "rejected"
    assert result.selected_method in ("auto", "native")
    assert all(key in calls[0] for key in controls)
    for fallback in calls[1:]:
        assert fallback["quad_options"] is None
        assert "quad_t3_minimum_angle" not in fallback and "quad_t3_angle_action" not in fallback
    with pytest.raises(QuadQualityRejected, match="owned quality refusal"):
        generate_automatic_mesh_result(
            plate(), automation=MeshAutomationOptions(strict_method=True), **options)


def test_default_angle_controls_are_unchanged_on_the_preferred_route():
    from anymesher.recovery import _attempt_options
    first = dict(strategy="auto", target_size=.5, quad_options=QuadMeshingOptions(),
                 quad_t3_minimum_angle=20.0, quad_t3_angle_action="warning")
    recipes = _attempt_options(first, None)
    assert recipes[0][1]["quad_t3_minimum_angle"] == 20.0
    assert recipes[0][1]["quad_t3_angle_action"] == "warning"
    assert [method for method, _ in recipes] == ["quad_first", "auto", "native"]
