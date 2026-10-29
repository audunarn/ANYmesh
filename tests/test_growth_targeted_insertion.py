from dataclasses import dataclass
import numpy as np
import pytest
from anymesher.errors import MeshError
from anymesher.native_v2 import _GeometryLimited,NativeMeshingOptions
from anymesher.surface_mesh import SurfaceMeshOptions,_make_candidate,_physical_quality_candidate_from_xyz
from anymesher._physical_growth_insertion import refine_interior_growth
from test_t3_insertion_transaction import topology,snapshot,assert_unchanged

@dataclass(frozen=True)
class Packet:
    points: object
    triangles: object
    segments: object

@pytest.mark.parametrize('kind',['refusal','cancel','failure'])
def test_private_insertion_validator_rolls_back_all_owned_state(kind):
    mesh=topology();mesh.quality_cache[(0,1,5)]=(1.,);before=snapshot(mesh)
    error={'refusal':_GeometryLimited('physical refusal'),'cancel':KeyboardInterrupt(),'failure':RuntimeError('owner failure')}[kind]
    def reject(points,cells):raise error
    with pytest.raises(type(error)) as caught:mesh.insert_point((.21,.13),_validate_candidate=reject)
    assert caught.value is error;assert_unchanged(mesh,before)


def test_candidate_mutation_still_passes_original_transaction_refusal():
    mesh=topology();before=snapshot(mesh)
    def mutate(points,cells):points[0,0]=.01
    with pytest.raises(MeshError):mesh.insert_point((.21,.13),_validate_candidate=mutate)
    assert_unchanged(mesh,before)


def operands():
    mesh=topology();points=mesh.points.copy()
    # A genuine nonuniform tensor grid with interior growth transitions.
    levels=np.array((0.,.2,.7,1.))
    points[:,0]=levels[np.rint(points[:,0]*3).astype(int)]
    points[:,1]=levels[np.rint(points[:,1]*3).astype(int)]
    settings=SurfaceMeshOptions(native_options=NativeMeshingOptions(
        point_placement='frontal_delaunay',metric_mode='isotropic_spatial',max_topology_operations=32,max_insertions=16))
    evaluate=lambda values:np.column_stack((values,np.zeros(len(values))))
    candidate=_physical_quality_candidate_from_xyz(_make_candidate(points,mesh.triangles,settings=settings),settings,evaluate(points))
    packet=Packet(points,mesh.triangles,np.array(sorted(mesh.protected_edges)))
    return candidate,packet,settings,evaluate


def test_genuine_growth_route_respects_remaining_budget_and_original_points():
    candidate,packet,settings,evaluate=operands()
    report={'topology_operations':31,'insertions':0,'shared_segment_splits':0}
    result,updated,receipt=refine_interior_growth(candidate,packet,settings,report,evaluate)
    assert receipt['topology_operations']<=32 and receipt['insertions']<=16
    assert receipt['growth_targeted_insertion']['attempts']<=1
    np.testing.assert_array_equal(result.points[:len(candidate.points)],candidate.points)
    np.testing.assert_array_equal(updated.segments,packet.segments)
    assert result.report['invalid_element_count']==0
    assert result.report['elements_above_maximum_growth']<=candidate.report['elements_above_maximum_growth']


def test_genuine_evaluator_policy_mutation_refuses():
    candidate,packet,settings,evaluate=operands()
    def changed(points):
        object.__setattr__(settings,'max_element_growth',2.)
        return evaluate(points)
    with pytest.raises(MeshError,match='changed'):
        refine_interior_growth(candidate,packet,settings,{'topology_operations':0,'insertions':0},changed)


def test_truthy_cancellation_and_evaluator_exception_are_not_geometry_refusals():
    candidate,packet,settings,evaluate=operands();before=candidate.points.copy()
    with pytest.raises(MeshError):
        refine_interior_growth(candidate,packet,settings,{'topology_operations':0,'insertions':0},evaluate,lambda phase:True)
    error=_GeometryLimited('actual evaluator exception')
    def failure(points):raise error
    with pytest.raises(_GeometryLimited) as caught:
        refine_interior_growth(candidate,packet,settings,{'topology_operations':0,'insertions':0},failure)
    assert caught.value is error
    np.testing.assert_array_equal(candidate.points,before)


def test_plain_public_route_does_not_select_owner_growth_insertion(monkeypatch):
    from importlib import import_module
    module=import_module('anymesher._physical_growth_insertion')
    surface=import_module('anymesher.surface_mesh')
    def forbidden(*args,**kwargs):raise AssertionError('owner-only growth route selected')
    monkeypatch.setattr(module,'refine_interior_growth',forbidden)
    result=surface.mesh_planar_surface(((0.,0.),(1.,0.),(1.,1.),(0.,1.)),
        target_size=.5,recombine=False,backend='python',native_options=NativeMeshingOptions(
            point_placement='frontal_delaunay',metric_mode='isotropic_spatial'))
    assert len(result.node_coordinates)>0


def test_owner_frontal_route_reaches_mixed_repair_without_centroid_insertion(monkeypatch):
    """Exercise the formerly eligible dispatch before the next repair stage."""
    from dataclasses import replace
    from importlib import import_module
    module = import_module('anymesher._physical_growth_insertion')
    surface = import_module('anymesher.surface_mesh')
    candidate, packet, settings, evaluate = operands()
    settings = replace(settings, target_size=.5, recombine=True)
    assert candidate.report['elements_above_maximum_growth'] > 0

    def forbidden(*args, **kwargs):
        raise AssertionError('production triangle-centroid insertion selected')

    class ReachedMixedRepair(Exception):
        pass

    reached = ReachedMixedRepair()

    def mixed_repair(best, triangulation, *args, **kwargs):
        assert best is candidate and triangulation is packet
        raise reached

    monkeypatch.setattr(module, 'refine_interior_growth', forbidden)
    monkeypatch.setattr(surface, '_analytic_preview_chart', lambda evaluator: object())
    monkeypatch.setattr(surface, 'frontal_delaunay_refine',
                        lambda *args, **kwargs: (packet, {'topology_operations': 0}))
    monkeypatch.setattr(surface, '_make_candidate', lambda *args, **kwargs: candidate)
    monkeypatch.setattr(surface, '_optimize_candidate', lambda *args, **kwargs: candidate)
    monkeypatch.setattr(surface, '_physical_quality_candidate', lambda *args, **kwargs: candidate)
    monkeypatch.setattr(surface, '_curvilinear_mixed_preview', mixed_repair)
    with pytest.raises(ReachedMixedRepair) as caught:
        surface._run_frontal_quality_path(
            {'best': candidate, 'triangulation': packet}, settings, None,
            model_uuid=None, geometry_revision=None, metric_to_physical=evaluate,
            metric_jacobian=None, automatically_seeded_shared_segments=None,
            component_seed_registry=None, supplemental_metric_field=None,
            preserve_spatial_refinement=True)
    assert caught.value is reached
