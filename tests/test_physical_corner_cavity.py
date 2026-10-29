"""Detached corner proposals use real owner admission and unchanged global gates."""
from copy import copy
from dataclasses import replace
import numpy as np
import pytest
from test_physical_mixed_repair_batch import chart_fixture, snapshot
from anymesher._physical_corner_cavity import (_fans, _positive, _valid_star,
    _closed_mixed_link, _ray_points, _score, repair_corner_cavities)
from anymesher._physical_t3_refinement import _alternative_progress
from anymesher.core import MeshCore
from anymesher.errors import MeshError
from anymesher.recombine import RecombinationReport
from anymesher.surface_mesh import SurfaceMeshOptions


def operands(chart):
    # Two free nodes have complete mixed disk links. The fixed corner has
    # three triangles, and each free node starts inside its fixed corner ear.
    rows = np.asarray(((0,0),(1,0),(.6,.2),(.2,.6),(0,1),(1,1)),dtype=float)
    uv = (.42+.10*rows) @ chart.transform
    triangles = np.asarray(((0,1,2),(0,2,3),(0,3,4),(1,5,2),(5,3,2),(5,4,3)))
    mesh = MeshCore(chart.evaluate(uv),triangles)
    report = RecombinationReport(mesh,np.empty((0,2),dtype=np.int64),
        np.empty(0,dtype=np.int64),np.empty(0),0,0)
    edges = ((0,1),(1,5),(5,4),(4,0))
    return uv,report,SurfaceMeshOptions(min_angle=30.,max_recombination_work=128),edges


def test_real_owner_complete_proposals_and_atomic_full_quality(chart_fixture):
    model,chart = chart_fixture
    import anygeometry as owner
    uv,report,settings,edges = operands(chart)
    before = uv.tobytes(),snapshot(report.mesh),owner.to_dict(model)
    assert _closed_mixed_link(report.mesh,2) and _closed_mixed_link(report.mesh,3)
    assert list(_fans(report.mesh,{0,1,4,5},set(tuple(sorted(e)) for e in edges),
        set(map(int,report.mesh.triangle_ids))))
    result,updated,work,facts = repair_corner_cavities(uv,report,settings,
        chart.evaluate,(0,1,4,5),edges,topology_remaining=256)
    assert facts['enabled'] and facts['complete_proposals'] > 0
    assert facts['relocation_trials'] <= 128 and work <= 256
    assert updated.exchange_work == facts['relocation_trials']+facts['polish_work']
    assert updated.exchange_work <= settings.max_recombination_work
    assert all(seed['status']!='not_started' for seed in facts.get('compound_seeds',()))
    assert before == (uv.tobytes(),snapshot(report.mesh),owner.to_dict(model))
    np.testing.assert_array_equal(result[[0,1,4,5]],uv[[0,1,4,5]])
    for name in ('node_ids','triangle_ids','node_owner_handles','triangle_owner_handles'):
        np.testing.assert_array_equal(getattr(updated.mesh,name),getattr(report.mesh,name))
    assert _valid_star(result,updated.mesh,{2,3})
    if facts['commits']:
        assert _alternative_progress(_score(report.mesh,settings),_score(updated.mesh,settings))
        np.testing.assert_array_equal(updated.mesh.node_coordinates,chart.evaluate(result))
    else:
        assert updated.mesh is report.mesh and result is uv


def test_generic_and_callback_routes_do_not_propose(chart_fixture):
    _model,chart = chart_fixture
    uv,report,settings,edges = operands(chart)
    for evaluate,check in ((lambda points:chart.evaluate(points),None),
                           (chart.evaluate,lambda phase:None)):
        result,updated,work,facts = repair_corner_cavities(uv,report,settings,evaluate,
            (0,1,4,5),edges,topology_remaining=256,cancellation_check=check)
        assert result is uv and updated is report and work == 0 and not facts['enabled']


def test_zero_allowances_and_forged_owner_do_not_publish(chart_fixture):
    _model,chart = chart_fixture
    uv,report,settings,edges = operands(chart)
    before = uv.tobytes(),snapshot(report.mesh)
    exhausted = replace(report,exchange_work=settings.max_recombination_work)
    result,updated,work,facts = repair_corner_cavities(uv,exhausted,
        settings,chart.evaluate,(0,1,4,5),edges,
        topology_remaining=0)
    assert result is uv and updated.mesh is report.mesh and work == 0 and not facts['commits']
    assert updated.exchange_truncated
    forged = copy(chart); forged._authority = None
    with pytest.raises(MeshError,match='missing owner'):
        repair_corner_cavities(uv,report,settings,forged.evaluate,(0,1,4,5),edges,
            topology_remaining=256)
    assert before == (uv.tobytes(),snapshot(report.mesh))


def test_original_signed_orientation_and_mixed_interface_are_required():
    points = np.asarray(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    mesh = MeshCore(np.column_stack((points,np.zeros(4))),np.empty((0,3),dtype=int),
                    np.asarray(((0,1,2,3),)))
    assert _positive(points,(0,1,2)) and not _positive(points,(0,2,1))
    trial = points.copy(); trial[2] = (-1.,-.1)
    assert not _valid_star(trial,mesh,{2})
    assert not _closed_mixed_link(mesh,2)  # boundary/interface row cannot move


def test_real_owner_operation_exception_identity_and_immutability(chart_fixture):
    import anygeometry as owner
    from anygeometry.nurbs_intersection_curve import intersection_curve_operation
    _model,chart = chart_fixture
    uv,report,settings,edges = operands(chart)
    before = uv.tobytes(),snapshot(report.mesh)
    failure = RuntimeError('owner operation cancellation')
    def stop():
        raise failure
    with pytest.raises(RuntimeError) as caught:
        with intersection_curve_operation(check=stop):
            repair_corner_cavities(uv,report,settings,chart.evaluate,(0,1,4,5),edges,
                topology_remaining=256)
    assert caught.value is failure
    assert before == (uv.tobytes(),snapshot(report.mesh))


def test_production_hook_ignores_irrelevant_legacy_topology_receipt(monkeypatch):
    from types import SimpleNamespace
    from importlib import import_module
    surface = import_module('anymesher.surface_mesh')
    import anymesher._physical_corner_cavity as cavity
    points=np.asarray(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    triangles=np.asarray(((0,1,2),(0,2,3)))
    segments=np.asarray(((0,1),(1,2),(2,3),(3,0)))
    settings=SurfaceMeshOptions(min_angle=15.)
    def evaluate(rows):
        return np.column_stack((rows,np.zeros(len(rows))))
    def forbidden(*args,**kwargs):
        raise AssertionError('legacy path entered new corner authority')
    monkeypatch.setattr(cavity,'repair_corner_cavities',forbidden)
    monkeypatch.setattr(surface,'_fixed_rows',forbidden)
    path=dict(best=surface._make_candidate(points,triangles,settings=settings),
        triangulation=SimpleNamespace(points=points,triangles=triangles,segments=segments),
        native_v2={'topology_operations':object()})
    before=points.tobytes(),triangles.tobytes(),segments.tobytes()
    result=surface._prepare_recombined_path(path,SimpleNamespace(lift=evaluate),
        points,(),settings,None,physical_evaluator=evaluate)
    assert result['path'] is path
    assert result['quality_policy']['accepted']
    assert before==(points.tobytes(),triangles.tobytes(),segments.tobytes())


@pytest.mark.parametrize('incoming',(False,True))
def test_global_exhaustion_receipt_preserves_truncation(chart_fixture,incoming):
    _model,chart = chart_fixture
    uv,report,settings,edges = operands(chart)
    exhausted = replace(report,exchange_work=settings.max_recombination_work,
                        exchange_truncated=incoming)
    result,updated,work,facts = repair_corner_cavities(uv,exhausted,settings,chart.evaluate,
        (0,1,4,5),edges,topology_remaining=256)
    assert facts['final_quality']['poor_element_ids']
    assert updated.exchange_truncated
    assert updated.exchange_work == exhausted.exchange_work
    assert result is uv and updated.mesh is exhausted.mesh and work == 0


def test_private_polish_guard_refuses_charged_trials_before_scoring(chart_fixture):
    from anymesher._physical_mixed_repair import repair_mixed_coordinates
    from test_physical_mixed_repair_batch import operands as batch_operands
    _model,chart = chart_fixture
    uv,triangles,report,settings,fixed = batch_operands(chart,7)
    before=uv.tobytes(),snapshot(report.mesh)
    seen=[]
    def refuse(proposed):
        seen.append(proposed.copy())
        return False
    result,updated=repair_mixed_coordinates(uv,triangles,report,settings,chart.evaluate,
        fixed,_trial_guard=refuse)
    assert seen and updated.exchange_work == 7
    np.testing.assert_array_equal(result,uv)
    assert before==(uv.tobytes(),snapshot(report.mesh))
    failure=RuntimeError('polish guard failure')
    def stop(proposed):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        repair_mixed_coordinates(uv,triangles,report,settings,chart.evaluate,
            fixed,_trial_guard=stop)
    assert caught.value is failure
    assert before==(uv.tobytes(),snapshot(report.mesh))
