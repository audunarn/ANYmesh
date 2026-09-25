"""Small controls for the opt-in automatic recovery contract."""

from anygeometry import GeometryModel
import pytest

from anymesher import MeshAutomationOptions, generate_automatic_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadCapabilityMissing


def _plate() -> GeometryModel:
    geometry = GeometryModel()
    geometry.add_plate(geometry.add_points((
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
    )))
    return geometry


def test_missing_preferred_capability_uses_existing_automatic_route(monkeypatch):
    import anymesher.recovery as recovery

    generate = recovery.generate_hybrid_mesh_result

    def without_quad_worker(geometry, **options):
        if options.get("quad_options") is not None:
            raise QuadCapabilityMissing("test worker unavailable")
        return generate(geometry, **options)

    monkeypatch.setattr(recovery, "generate_hybrid_mesh_result", without_quad_worker)
    geometry = _plate()
    before = (str(geometry.model_id), geometry.revision)
    result = generate_automatic_mesh_result(
        geometry,
        automation=MeshAutomationOptions(),
        strategy="native", quad_options=QuadMeshingOptions(),
        target_size=0.25, order="linear", native_backend="python",
    )
    assert result.selected_method == "auto"
    assert result.status == "ready"
    assert [item["status"] for item in result.attempts] == ["rejected", "selected"]
    assert result.mesh.num_elements > 0
    assert (str(geometry.model_id), geometry.revision) == before
    with pytest.raises(QuadCapabilityMissing, match="worker unavailable"):
        generate_automatic_mesh_result(
            geometry,
            automation=MeshAutomationOptions(strict_method=True),
            strategy="native", quad_options=QuadMeshingOptions(),
            target_size=0.25, order="linear", native_backend="python",
        )
