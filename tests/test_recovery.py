"""Small controls for the opt-in automatic recovery contract."""

from anygeometry import GeometryModel
import pytest

from anymesher import MeshAutomationOptions, generate_automatic_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadCapabilityMissing


@pytest.mark.parametrize('kind',('quad','structured'))
def test_completed_quality_refusal_can_try_the_native_triangle_recipe(monkeypatch,kind):
    import anymesher.recovery as recovery
    from anymesher.errors import StructuredQualityRejected
    from anymesher.quad.validate import QuadQualityRejected
    generate=recovery.generate_hybrid_mesh_result
    calls=[]
    def reject_completed_candidate(geometry,**options):
        calls.append((options.get('strategy'),options.get('recombine')))
        if options.get('strategy')=='auto':
            error=QuadQualityRejected if kind=='quad' else StructuredQualityRejected
            raise error('completed candidate failed its existing quality gate')
        return generate(geometry,**options)
    monkeypatch.setattr(recovery,'generate_hybrid_mesh_result',reject_completed_candidate)
    result=generate_automatic_mesh_result(_plate(),automation=MeshAutomationOptions(),
        strategy='auto',target_size=.25,order='linear',native_backend='python')
    assert result.status=='ready' and result.selected_method=='native'
    assert calls[-1]==('native',False)
    assert result.mesh.tris and not result.mesh.quads
    assert result.attempts[0]['status']=='rejected'
    with pytest.raises((QuadQualityRejected,StructuredQualityRejected)):
        generate_automatic_mesh_result(_plate(),automation=MeshAutomationOptions(strict_method=True),
            strategy='auto',target_size=.25,order='linear',native_backend='python')


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


def test_automatic_analytic_trim_selects_chart_recipe_and_retains_budgets():
    from types import SimpleNamespace
    from anygeometry import EllipticArc
    from anymesher.native_v2 import NativeMeshingOptions
    from anymesher.recovery import _attempt_options,_prefer_analytic_trim_recipe
    arc=EllipticArc((0.,0.,0.),(2.,0.,0.),(0.,1.,0.),0.,1.)
    geometry=SimpleNamespace(edges={1:SimpleNamespace(curve=arc)})
    settings=NativeMeshingOptions(max_insertions=37,max_topology_operations=129,
                                  cancellation_interval=7)
    first={'strategy':'auto','order':'linear','native_options':settings,'recombine':True}
    recipes=_attempt_options(first,None)
    chosen=_prefer_analytic_trim_recipe(geometry,first,recipes)
    assert [method for method,_ in chosen]==['native','auto']
    native=chosen[0][1]['native_options']
    assert native.point_placement=='frontal_delaunay' and native.metric_mode=='isotropic_spatial'
    assert (native.max_insertions,native.max_topology_operations,native.cancellation_interval)==(37,129,7)
    assert chosen[0][1]['recombine'] is False
    assert first['native_options'] is settings and recipes[0][1]['native_options'] is settings
    for changes in ({'strategy':'native'},{'strategy':'mapped'},{'order':'quadratic'},
                    {'native_options':NativeMeshingOptions(point_placement='frontal_delaunay',
                                                           metric_mode='isotropic_spatial')}):
        requested={**first,**changes}
        original=_attempt_options(requested,None)
        assert _prefer_analytic_trim_recipe(geometry,requested,original) is original
    assert _prefer_analytic_trim_recipe(_plate(),first,recipes) is recipes


def test_strict_automatic_request_does_not_select_a_material_recipe(monkeypatch):
    import anymesher.recovery as recovery
    def forbidden(*args):
        raise AssertionError('strict request must not change discretization preference')
    monkeypatch.setattr(recovery,'_prefer_analytic_trim_recipe',forbidden)
    result=generate_automatic_mesh_result(_plate(),automation=MeshAutomationOptions(strict_method=True),
        strategy='auto',target_size=.25,order='linear',native_backend='python')
    assert result.status=='ready' and result.selected_method=='auto'
