"""Conservative rejection proofs and real owner-route/reference parity."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import anygeometry as owner
from anymesher._physical_flip_rejection import prepare_flip_rejection
from anymesher._physical_t3_refinement import refine_physical_candidate, _alternative_progress
from anymesher.core import MeshCore
from anymesher.native_v2 import MutableT3Topology, NativeMeshingOptions, ComponentSeedRegistry
from anymesher.surface_mesh import (SurfaceMeshOptions, _make_candidate,
    _physical_quality_candidate_from_xyz, _physical_quality_candidate)
from anymesher.triangulation import PlanarTriangulation
from test_physical_mixed_repair_batch import chart_fixture


def baseline(kind='floor'):
    points = [(0.,0.),(1.,0.),(1.,1.),(0.,1.)]
    cells = [(0,1,2),(0,2,3)]
    if kind == 'floor':
        points += [(3.,0.),(4.,0.),(3.,.001)]
        cells += [(4,5,6)]
        settings = SurfaceMeshOptions()
    else:
        for offset in (3.,7.) if kind == 'ties' else (3.,):
            n = len(points)
            points += [(offset,0.),(offset+1.,0.),(offset,1.),(offset,-5.)]
            cells += [(n,n+1,n+2),(n+1,n,n+3)]
        settings = SurfaceMeshOptions(min_angle=.01,max_angle=179.99,
            min_scaled_jacobian=.00001,max_aspect_ratio=10000.,max_element_growth=100.)
    topology = MutableT3Topology(np.asarray(points),np.asarray(cells))
    points,cells = topology.canonical_export()
    xyz = np.column_stack((points,np.zeros(len(points))))
    current = _physical_quality_candidate_from_xyz(_make_candidate(points,cells,settings=settings),settings,xyz)
    return current,xyz,settings,topology


@pytest.mark.parametrize('kind',('floor','witness','ties'))
def test_rejection_preserves_actual_global_progress_guard(kind):
    current,xyz,settings,topology = baseline(kind)
    frame = prepare_flip_rejection(current,xyz,settings,topology.canonical_export())
    assert frame is not None and frame.rejects((0,2),current,xyz)
    before = topology.points.tobytes(),topology.triangles.tobytes(),deepcopy(current.report)
    proposals = []
    def validate(points,cells):
        proposed = _physical_quality_candidate_from_xyz(_make_candidate(points,cells,settings=settings),settings,xyz)
        proposals.append(proposed)
        assert not _alternative_progress(current.report,proposed.report)
        raise RuntimeError('reference rejects nonprogress')
    with pytest.raises(RuntimeError,match='reference rejects'):
        topology.flip_edge((0,2),_validate_candidate=validate)
    assert proposals
    assert before == (topology.points.tobytes(),topology.triangles.tobytes(),current.report)


def test_affected_maximum_witness_retains_full_primitive_and_ties_can_certify_other_witness():
    current,xyz,settings,topology = baseline('witness')
    frame = prepare_flip_rejection(current,xyz,settings,topology.canonical_export())
    assert current.report['max_element_growth'] > 1.
    assert frame is not None and not frame.rejects((4,5),current,xyz)
    current,xyz,settings,topology = baseline('ties')
    frame = prepare_flip_rejection(current,xyz,settings,topology.canonical_export())
    assert frame is not None and frame.rejects((4,5),current,xyz)


def test_changed_poor_cells_are_never_pruned():
    current,xyz,settings,topology = baseline('floor')
    settings = replace(settings,min_angle=50.)
    current = _physical_quality_candidate_from_xyz(_make_candidate(current.points,current.triangles,settings=settings),settings,xyz)
    frame = prepare_flip_rejection(current,xyz,settings,topology.canonical_export())
    assert frame is not None and not frame.rejects((0,2),current,xyz)


@pytest.mark.parametrize('change',('unknown_metric','missing_metric','wrong_id','wrong_count','wrong_max',
                                 'invalid','nonfinite','zero_size','canonical','malformed','threshold'))
def test_incomplete_or_mismatched_evidence_retains_full_validation(change):
    current,xyz,settings,topology = baseline()
    report = deepcopy(current.report)
    canonical = topology.canonical_export()
    if change == 'unknown_metric': report['violation_counts']['unknown'] = 0
    elif change == 'missing_metric': del report['violation_counts']['warpage']
    elif change == 'wrong_id': report['poor_element_ids'] = [99999]; report['quality_violation_count'] = 1
    elif change == 'wrong_count': report['violation_counts']['minimum_angle'] += 1
    elif change == 'wrong_max': report['max_element_growth'] = 123.
    elif change == 'invalid': report['invalid_element_count'] = 1
    elif change == 'nonfinite': xyz = xyz.copy(); xyz[0,0] = np.nan
    elif change == 'zero_size': xyz = np.zeros_like(xyz)
    elif change == 'canonical': canonical = (canonical[0],canonical[1][::-1])
    elif change == 'threshold': object.__setattr__(settings,'min_angle',float('nan'))
    else: canonical = ()
    assert prepare_flip_rejection(replace(current,report=report),xyz,settings,canonical) is None


@pytest.mark.parametrize('change',('points','cells','xyz','report','threshold'))
def test_frame_is_bound_to_old_canonical_geometry_and_report(change):
    current,xyz,settings,topology = baseline()
    frame = prepare_flip_rejection(current,xyz,settings,topology.canonical_export())
    assert frame is not None
    if change == 'points':
        points = current.points.copy(); points[0,0] += .125
        current = replace(current,points=points)
    elif change == 'cells': current = replace(current,triangles=current.triangles[::-1].copy())
    elif change == 'xyz': xyz = xyz.copy(); xyz[0,2] += .125
    elif change == 'report':
        report = deepcopy(current.report); report['max_element_growth'] += .125
        current = replace(current,report=report)
    else: object.__setattr__(settings,'min_angle',settings.min_angle+1.)
    assert not frame.rejects((0,2),current,xyz)


def owner_operands(chart,budget,*,noncanonical=False):
    uv = np.asarray(((.54,.46),(.60,.46),(.60,.54),(.54,.54),
        (.32,.45),(.48,.45),(.48,.55),(.32,.55),(.325,.452))) @ chart.transform
    cells = np.asarray(((0,1,2),(0,2,3),(4,5,8),(5,6,8),(6,7,8),(7,4,8)))
    topology = MutableT3Topology(uv,cells)
    uv,cells = topology.canonical_export()
    if noncanonical:
        cells = np.roll(cells[::-1],1,axis=1).copy()
    incidence = {}
    for cell in cells:
        for a,b in zip(cell,np.roll(cell,-1)):
            edge = tuple(sorted((int(a),int(b))))
            incidence[edge] = incidence.get(edge,0)+1
    boundary = np.asarray([edge for edge,count in sorted(incidence.items()) if count == 1])
    triangulation = PlanarTriangulation(uv,cells,boundary,boundary,np.empty((0,2),dtype=np.int64),np.arange(len(uv)),())
    settings = SurfaceMeshOptions(min_angle=30.,native_options=NativeMeshingOptions(
        max_topology_operations=budget,max_insertions=1))
    candidate = _physical_quality_candidate(_make_candidate(uv,cells,settings=settings),settings,chart.evaluate)
    report = dict(topology_operations=0,insertions=1,shared_segment_splits=0,shared_nodes=[])
    return candidate,triangulation,settings,report


@pytest.mark.parametrize('budget',(1,9,30))
@pytest.mark.parametrize('noncanonical',(False,True))
def test_real_owner_pruning_matches_noop_callback_reference_with_owner_cavities(chart_fixture,monkeypatch,budget,noncanonical):
    model,chart = chart_fixture
    candidate,triangulation,settings,report = owner_operands(chart,budget,noncanonical=noncanonical)
    before = candidate.points.tobytes(),candidate.triangles.tobytes(),deepcopy(candidate.report),deepcopy(report),owner.to_dict(model)
    counts = []
    original = MutableT3Topology.flip_edge
    def observed(self,*args,**kwargs):
        counts.append(1)
        return original(self,*args,**kwargs)
    monkeypatch.setattr(MutableT3Topology,'flip_edge',observed)
    reference,reference_tri,reference_work = refine_physical_candidate(candidate,triangulation,settings,report,
        chart.evaluate,{},ComponentSeedRegistry(100),cancellation_check=lambda phase:None,allow_physical_flips=True)
    conservative_calls = len(counts); counts.clear()
    actual,actual_tri,actual_work = refine_physical_candidate(candidate,triangulation,settings,report,
        chart.evaluate,{},ComponentSeedRegistry(100),allow_physical_flips=True)
    assert actual_work == reference_work
    assert actual_work['physical_quality_two_flip_cavity_enabled'] is True
    assert actual.report == reference.report
    np.testing.assert_array_equal(actual.points,reference.points)
    np.testing.assert_array_equal(actual.triangles,reference.triangles)
    np.testing.assert_array_equal(actual_tri.segments,reference_tri.segments)
    assert len(counts) <= conservative_calls
    if not noncanonical:
        assert len(counts) < conservative_calls
    assert before == (candidate.points.tobytes(),candidate.triangles.tobytes(),candidate.report,report,owner.to_dict(model))


def test_initial_representation_mismatch_retains_original_flip_validation():
    current,xyz,settings,topology = baseline('witness')
    # Build the mismatch explicitly: canonical topology storage is allowed to
    # share export order. Cyclic rotation preserves each oriented physical cell.
    current = _physical_quality_candidate_from_xyz(_make_candidate(
        current.points,np.roll(current.triangles,1,axis=1),settings=settings),settings,xyz)
    assert not np.array_equal(current.triangles,topology.triangles)
    before = topology.points.tobytes(),topology.triangles.tobytes(),deepcopy(current.report)
    assert prepare_flip_rejection(current,xyz,settings,(topology.points,topology.triangles)) is None
    assert before == (topology.points.tobytes(),topology.triangles.tobytes(),current.report)


def test_real_bisection_issued_receipt_and_canonical_export_keep_rejection_enabled():
    # The genuine callback receipt and canonical storage describe the same
    # oriented cells; an explicitly different representation must still refuse.
    points = np.asarray(((0.,0.),(1.,0.),(1.,1.),(0.,1.),
        (3.,0.),(4.,0.),(4.,1.),(3.,1.),(6.,0.),(7.,0.),(6.,.001)))
    cells = np.asarray(((0,1,2),(0,2,3),(4,5,6),(4,6,7),(9,10,8)))
    original = points.tobytes(),cells.tobytes()
    topology = MutableT3Topology(points,cells)
    settings = SurfaceMeshOptions()
    issued = []
    def score_committed_trial(trial_points,trial_cells):
        xyz = np.column_stack((trial_points,np.zeros(len(trial_points))))
        current = _physical_quality_candidate_from_xyz(
            _make_candidate(trial_points,trial_cells,settings=settings),settings,xyz)
        issued.append((current,xyz))
    split = topology.bisect_interior_edge((4,6),_validate_candidate=score_committed_trial)
    assert topology.epoch == 1 and len(issued) == 1
    current,xyz = issued[0]
    assert np.array_equal(current.points,topology.points)
    assert np.array_equal(current.triangles,topology.triangles)
    assert np.array_equal(current.triangles,topology.canonical_export()[1])
    before = current.points.tobytes(),current.triangles.tobytes(),deepcopy(current.report)
    frame = prepare_flip_rejection(current,xyz,settings,(topology.points,topology.triangles))
    assert frame is not None
    # Either independent good component supplies an untouched growth witness;
    # do not assume floating mean-edge sums remain exactly tied after splitting.
    assert any(frame.rejects(edge,current,xyz) for edge in ((0,2),(4,split['point_id'])))
    assert prepare_flip_rejection(current,xyz,settings,topology.canonical_export()) is not None
    assert prepare_flip_rejection(current,xyz,settings,
        (topology.points,np.roll(topology.triangles,1,axis=1))) is None
    assert before == (current.points.tobytes(),current.triangles.tobytes(),current.report)
    assert original == (points.tobytes(),cells.tobytes())


def test_real_owner_callback_cancellation_does_not_mutate_inputs(chart_fixture):
    model,chart = chart_fixture
    candidate,triangulation,settings,report = owner_operands(chart,9)
    before = candidate.points.tobytes(),candidate.triangles.tobytes(),deepcopy(candidate.report),deepcopy(report),owner.to_dict(model)
    class Cancelled(Exception): pass
    def cancel(phase): raise Cancelled()
    with pytest.raises(Cancelled):
        refine_physical_candidate(candidate,triangulation,settings,report,chart.evaluate,{},
            ComponentSeedRegistry(100),cancellation_check=cancel,allow_physical_flips=True)
    assert before == (candidate.points.tobytes(),candidate.triangles.tobytes(),candidate.report,report,owner.to_dict(model))
