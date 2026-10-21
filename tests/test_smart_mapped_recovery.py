"""SmartMapped retains identity through the GUI's existing recovery wrapper."""
import pytest
from anygeometry import GeometryModel
from anymesher.recovery import generate_automatic_mesh_result, MeshAutomationOptions
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.validate import QuadQualityRejected


def plate():
    geometry = GeometryModel()
    geometry.add_plate(geometry.add_points(((0, 0, 0), (2, 0, 0),
                                          (2, 2, 0), (0, 2, 0))))
    return geometry


def test_actual_recovery_wrapper_names_smart_mapped():
    result = generate_automatic_mesh_result(
        plate(), strategy="smart_mapped", target_size=.5,
        quad_options=QuadMeshingOptions(quality_model="shape_jacobian"),
        automation=MeshAutomationOptions(strict_method=True))
    assert result.selected_method == "smart_mapped"
    assert result.attempts == ({"method": "smart_mapped", "status": "selected"},)
    assert set(result.result.strategy_by_face.values()) == {"smart_mapped"}


def test_quality_refusal_keeps_existing_recovery_and_strict_policy(monkeypatch):
    import anymesher.recovery as recovery
    original = recovery.generate_hybrid_mesh_result
    calls = []

    def reject(geometry, **options):
        calls.append(dict(options))
        if options["strategy"] == "smart_mapped":
            raise QuadQualityRejected("owned quality refusal")
        return original(geometry, **options)

    monkeypatch.setattr(recovery, "generate_hybrid_mesh_result", reject)
    options = dict(strategy="smart_mapped", target_size=.5,
                   quad_options=QuadMeshingOptions(), quad_t3_angle_action="warning")
    result = generate_automatic_mesh_result(plate(), **options)
    assert result.selected_method == "auto"
    assert result.attempts[0]["method"] == "smart_mapped"
    assert result.attempts[0]["status"] == "rejected"
    assert "quad_t3_angle_action" not in calls[1]
    with pytest.raises(QuadQualityRejected, match="owned quality refusal"):
        generate_automatic_mesh_result(
            plate(), automation=MeshAutomationOptions(strict_method=True), **options)
