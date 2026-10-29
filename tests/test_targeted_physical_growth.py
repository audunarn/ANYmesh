"""Numerical growth proposals; final physical gates remain the oracle."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from anymesher.core import MeshCore
from anymesher.surface_mesh import SurfaceMeshOptions,_published_growth,_quality_threshold_report
from anymesher.quality_v2 import evaluate_quality
from anymesher._physical_mixed_repair import (_growth_descent,repair_mixed_core_coordinates,
                                             repair_mixed_coordinates)
from anymesher.errors import MeshError


def physical(rows):return np.column_stack((rows,np.zeros(len(rows))))
def jacobians(rows):return np.repeat(np.array((((1.,0.),(0.,1.),(0.,0.)),)),len(rows),axis=0)


def with_xyz(core,xyz):
    return MeshCore(xyz,core.triangle_connectivity,core.quad_connectivity,
        node_ids=core.node_ids,triangle_ids=core.triangle_ids,quad_ids=core.quad_ids,
        node_active=core.node_active,triangle_active=core.triangle_active,quad_active=core.quad_active)


def boundary_rows(core):
    incidence={}
    for rows,active in ((core.triangle_connectivity,core.triangle_active),(core.quad_connectivity,core.quad_active)):
        for cell in rows[active]:
            for a,b in zip(cell,np.roll(cell,-1)):
                edge=tuple(sorted((int(a),int(b))));incidence[edge]=incidence.get(edge,0)+1
    return sorted({node for edge,count in incidence.items() if count==1 for node in edge})


def test_analytic_growth_proposal_matches_independent_scalar_difference():
    xyz=np.array(((0.,0.,0.),(1.,0.,0.),(2.6,0.,0.),(0.,1.,0.),(1.,1.,0.),(2.6,1.,0.)))
    core=MeshCore(xyz,(),((0,1,4,3),(1,2,5,4)))
    limit=1.1;gradient=_growth_descent(core,xyz,limit)
    def objective(values):
        growth,_=_published_growth(with_xyz(core,values),limit)
        return .5*max(0.,growth-limit)**2
    for node in range(len(xyz)):
        for axis in range(3):
            plus=xyz.copy();minus=xyz.copy();plus[node,axis]+=1.e-6;minus[node,axis]-=1.e-6
            derivative=(objective(plus)-objective(minus))/2.e-6
            assert derivative==pytest.approx(gradient.get(node,np.zeros(3))[axis],rel=1.e-6,abs=1.e-9)
    assert _growth_descent(core,xyz,10.)=={}


def test_actual_captured_rounded_mixed_core_meets_unchanged_physical_policy():
    path=Path(__file__).parent/'data/rounded_growth_failure_20261025.npz'
    with np.load(path,allow_pickle=False) as saved:
        names=('node_coordinates','triangle_connectivity','quad_connectivity','node_ids',
            'triangle_ids','quad_ids','node_active','triangle_active','quad_active')
        values={name:saved['6_physical_'+name].copy() for name in names}
    core=MeshCore(**values);assert np.all(core.node_coordinates[:,2]==0.)
    points=core.node_coordinates[:,:2].copy();fixed=boundary_rows(core)
    before=points.tobytes(),core.node_coordinates.tobytes()
    settings=SurfaceMeshOptions()
    initial,_=_published_growth(core,settings.max_element_growth)
    assert initial==pytest.approx(1.5463398398527914,rel=1.e-13)
    assert _quality_threshold_report(evaluate_quality(core),settings)['accepted']
    diagnostic={}
    made,used,truncated=repair_mixed_core_coordinates(points,core.triangle_connectivity,core,
        settings,physical,fixed,exchange_work=9112,evaluate_jacobians=jacobians,_diagnostics=diagnostic)
    final=with_xyz(core,physical(made));quality=evaluate_quality(final)
    growth,poor=_published_growth(final,settings.max_element_growth)
    print('CAPTURED_GROWTH_REPLAY',dict(initial=initial,final=growth,trials=used,truncated=truncated),flush=True)
    import json
    print('CAPTURED_GROWTH_PROPOSALS',json.dumps(diagnostic,sort_keys=True),flush=True)
    assert not quality.validity.errors and _quality_threshold_report(quality,settings)['accepted']
    assert growth<=1.5 and not poor and 0<used<=512
    assert made[fixed].tobytes()==points[fixed].tobytes()
    assert before==(points.tobytes(),core.node_coordinates.tobytes())
    # This replay proves numerical quality only, never source/cell/full-mesh acceptance.


def test_actual_captured_fresh_rejected_rounded_mixed_core_meets_unchanged_physical_policy():
    saved_path=Path(__file__).parent/'data/rounded_growth_failure_20261025.npz'
    rejected_path=Path(__file__).parent/'data/rounded_coupled_failure_20261025.npz'
    names=('node_coordinates','triangle_connectivity','quad_connectivity','node_ids',
        'triangle_ids','quad_ids','node_active','triangle_active','quad_active')
    with np.load(saved_path,allow_pickle=False) as saved:
        saved_values={name:saved['6_physical_'+name].copy() for name in names}
    with np.load(rejected_path,allow_pickle=False) as rejected:
        rejected_values={name:rejected['6_physical_'+name].copy() for name in names}
    for name in names[1:]:
        assert np.array_equal(saved_values[name],rejected_values[name])
    differing=np.flatnonzero(np.any(saved_values['node_coordinates']!=rejected_values['node_coordinates'],axis=1))
    assert differing.tolist()==[1063,1202,1230,1257]
    pre_repair_xyz=rejected_values['node_coordinates'].copy()
    pre_repair_xyz[1202]=saved_values['node_coordinates'][1202]
    settings=SurfaceMeshOptions()
    for label,xyz,expected_growth,expected_poor in (
            ('fresh_rejected',rejected_values['node_coordinates'],1.531890384936561,[149,2257,2646,3670]),
            ('pre_repair',pre_repair_xyz,1.5548383281731009,[147,149,2257,2646,3670])):
        core=MeshCore(**{**rejected_values,'node_coordinates':xyz.copy()})
        assert np.all(core.node_coordinates[:,2]==0.)
        points=core.node_coordinates[:,:2].copy();fixed=boundary_rows(core)
        before=points.tobytes(),core.node_coordinates.tobytes()
        initial,initial_poor=_published_growth(core,settings.max_element_growth)
        initial_quality=_quality_threshold_report(evaluate_quality(core),settings)
        assert initial==pytest.approx(expected_growth,rel=1.e-13)
        assert sorted(initial_poor)==expected_poor
        assert not initial_quality['accepted']
        assert initial_quality['violation_counts']=={'scaled_jacobian':0,'aspect_ratio':0,'minimum_angle':1,'maximum_angle':0,'warpage':0}
        assert initial_quality['poor_element_ids']==[147]
        assert initial_quality['worst']['minimum_angle']==pytest.approx(29.859506156979286,rel=1.e-13)
        diagnostic={}
        made,used,truncated=repair_mixed_core_coordinates(points,core.triangle_connectivity,core,
            settings,physical,fixed,exchange_work=9112,evaluate_jacobians=jacobians,_diagnostics=diagnostic)
        final=with_xyz(core,physical(made));quality=evaluate_quality(final)
        growth,poor=_published_growth(final,settings.max_element_growth)
        print('CAPTURED_COUPLED_REPLAY',label,dict(initial=initial,final=growth,trials=used,truncated=truncated),flush=True)
        import json
        print('CAPTURED_COUPLED_PROPOSALS',label,json.dumps(diagnostic,sort_keys=True),flush=True)
        assert not quality.validity.errors and _quality_threshold_report(quality,settings)['accepted']
        assert growth<=1.5 and not poor and 0<used<=512
        assert made[fixed].tobytes()==points[fixed].tobytes()
        assert before==(points.tobytes(),core.node_coordinates.tobytes())


def test_real_report_wrapper_keeps_pairing_identity_and_shared_budget():
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case()
    report=replace(report,exchange_work=settings.max_recombination_work-2)
    fixed=(0,2,3,5)
    made,updated=repair_mixed_coordinates(points,cells,report,settings,evaluate,fixed)
    reference,used,truncated=repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,
        fixed,exchange_work=report.exchange_work)
    np.testing.assert_array_equal(made,reference)
    assert updated.exchange_work==report.exchange_work+used<=settings.max_recombination_work
    assert updated.paired_triangle_ids is report.paired_triangle_ids
    assert updated.created_quad_ids is report.created_quad_ids and updated.mesh is report.mesh
    assert updated.exchange_truncated==(report.exchange_truncated or truncated)


@pytest.mark.parametrize('failure',('truthy','exception','jacobian'))
def test_targeted_repair_refuses_cancellation_and_bad_jacobian(failure):
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case();error=RuntimeError('original cancellation')
    def check(phase):
        if failure=='exception':raise error
        return failure=='truthy'
    def bad(rows):return np.full((len(rows),3,2),np.nan)
    with pytest.raises(RuntimeError if failure=='exception' else MeshError) as caught:
        repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,(),
            cancellation_check=check,evaluate_jacobians=bad if failure=='jacobian' else jacobians)
    if failure=='exception':assert caught.value is error


def test_zero_remaining_work_never_proposes_or_calls_jacobians():
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case()
    def forbidden(rows):raise AssertionError('no proposal budget')
    made,used,truncated=repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,(),
        exchange_work=settings.max_recombination_work,evaluate_jacobians=forbidden)
    assert used==0 and truncated and made is points


def test_coupled_proposals_keep_fixed_signed_zero_rows_byte_exact():
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case();points=points.copy();points[0,0]=-0.
    before=points[0].tobytes()
    made,used,_=repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,(0,),
        exchange_work=settings.max_recombination_work-3,evaluate_jacobians=jacobians)
    assert made[0].tobytes()==before and used<=3


def test_targeted_backtracking_reaches_smaller_admissible_steps_without_new_budget():
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case()
    settings=replace(settings,max_recombination_work=64)
    diagnostic={};fixed=(0,2,3,5);before=points.tobytes()
    def guard(proposed):
        return np.max(np.linalg.norm(proposed-points,axis=1))<=1.e-4
    made,used,_=repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,fixed,
        evaluate_jacobians=jacobians,_trial_guard=guard,_diagnostics=diagnostic)
    initial,_=_published_growth(report.mesh,settings.max_element_growth)
    final,_=_published_growth(with_xyz(report.mesh,evaluate(made)),settings.max_element_growth)
    assert diagnostic['caller_refusals']>0 and diagnostic['commits']>0
    assert final<initial and 0<used<=64
    assert diagnostic['scored_trials']+diagnostic['constraint_probes']+diagnostic['chart_refusals']+diagnostic['caller_refusals']==used
    np.testing.assert_array_equal(made[list(fixed)],points[list(fixed)])
    assert points.tobytes()==before
    # Only a proposal-size guard is tightened in this isolated regression;
    # physical quality tolerances and production policy are untouched.


def test_halfspace_proposal_respects_active_constraint_and_radius():
    from anymesher._physical_mixed_repair import _project_growth_halfspaces
    # Growth needs x>=.2; active angle needs y>=0; another quality gate x+y<=.3.
    values=np.array((-.2,0.,.3));matrix=np.array(((1.,0.),(0.,1.),(-1.,-1.)))
    proposed,steps=_project_growth_halfspaces(values,matrix,np.zeros(2))
    np.testing.assert_allclose(proposed,(.2,0.),rtol=0.,atol=1.e-6)
    # The solver aims just inside movable inequalities; this is a proposal
    # buffer, not relaxation of any physical gate.
    assert np.min(values+matrix@proposed)>=-1.e-10 and np.linalg.norm(proposed)<=1.
    assert steps<=2048
    # A violated invariant with no free derivative is explicitly unavailable.
    refused,count=_project_growth_halfspaces(np.array((-.1,)),np.zeros((1,2)),np.zeros(2))
    assert refused is None and count==0


def test_physical_constraint_signs_match_real_growth_and_quality():
    from anymesher._physical_mixed_repair import _physical_constraint_values
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case()
    quality=evaluate_quality(report.mesh)
    assert _quality_threshold_report(quality,settings)['accepted']
    values=_physical_constraint_values(report.mesh,quality,settings)
    assert np.any(values<0)  # Exactly the actual mixed growth violation remains.
    accepted=replace(settings,max_element_growth=1.5)
    assert np.all(_physical_constraint_values(report.mesh,quality,accepted)>=0)


def test_constraint_probe_budget_is_shared_and_fixed_rows_are_preserved():
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case()
    settings=replace(settings,max_recombination_work=24);fixed=(0,2,3,5);diagnostic={}
    made,used,_=repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,fixed,
        evaluate_jacobians=jacobians,_diagnostics=diagnostic)
    assert 0<diagnostic['constraint_probes']<=used<=24
    assert diagnostic['patch_projection_steps']<=2048*3*diagnostic['patch_attempts']
    assert diagnostic['scored_trials']+diagnostic['constraint_probes']+diagnostic['chart_refusals']+diagnostic['caller_refusals']==used
    np.testing.assert_array_equal(made[list(fixed)],points[list(fixed)])


def test_rejected_constraint_probes_never_escape_or_exceed_budget():
    from test_physical_growth_retention import case
    points,cells,evaluate,report,settings=case();settings=replace(settings,max_recombination_work=32)
    diagnostic={};before=points.tobytes()
    made,used,truncated=repair_mixed_core_coordinates(points,cells,report.mesh,settings,evaluate,(),
        evaluate_jacobians=jacobians,_trial_guard=lambda rows:False,_diagnostics=diagnostic)
    assert made is points and made.tobytes()==before
    assert diagnostic['commits']==0 and diagnostic['caller_refusals']==used<=32
    assert diagnostic['scored_trials']==0 and diagnostic['constraint_probes']==0
