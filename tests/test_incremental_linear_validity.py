"""Reference parity and call inventory for private linear-validity reuse."""
import sys
import numpy as np
import pytest
from anymesher import quality_v2 as quality
from anymesher.core import MeshCore
from anymesher.native_v2 import MutableT3Topology
from anymesher.surface_mesh import SurfaceMeshOptions
from anymesher._physical_split_trials import PhysicalSplitTrials
from test_incremental_physical_split_trials import score


def fixture():
    points=np.array(((0.,0.),(1.,0.),(1.,1.),(0.,1.)))
    cells=np.array(((0,1,2),(0,2,3)),dtype=np.int64)
    xyz=np.column_stack((points,np.zeros(4)))
    settings=SurfaceMeshOptions()
    current=score(points,cells,xyz,settings)
    trials=PhysicalSplitTrials(settings);trials.rebase(current,xyz)
    topology=MutableT3Topology(points,cells);topology.bisect_interior_edge((0,2))
    made,split=topology.canonical_export()
    return trials,settings,made,split,np.column_stack((made,np.zeros(5)))


def test_eligible_trial_computes_only_new_triangle_metrics_and_matches_full_report():
    trials,settings,points,cells,xyz=fixture()
    counts=[];code=quality.triangle_quality.__code__
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is code:
            counts.append(len(frame.f_locals['triangles']))
    sys.setprofile(profile)
    try:actual=trials.trial(points,cells,xyz)
    finally:sys.setprofile(None)
    expected=score(points,cells,xyz,settings)
    assert actual.report==expected.report and actual.score==expected.score
    assert counts==[4]  # one added-cell batch, no full validity batch


@pytest.mark.parametrize('kind',('valid','duplicate','repeated','nonmanifold','degenerate','world_scale','orientation'))
def test_reused_linear_validity_matches_complete_validator(kind):
    points=np.array(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.),(.5,.5,0.)))
    cells=np.array(((0,1,4),(1,2,4),(2,3,4),(3,0,4)),dtype=np.int64)
    if kind=='duplicate':cells=np.vstack((cells,cells[:1]))
    if kind=='repeated':cells[0]=(0,0,4)
    if kind=='nonmanifold':cells=np.vstack((cells,(0,4,2)))
    if kind=='degenerate':points[4]=points[0]
    if kind=='world_scale':points[4,2]=1e8
    if kind=='orientation':cells=cells[:,::-1].copy()
    core=MeshCore(points,cells)
    metrics=quality.triangle_quality(points,cells)
    actual=quality._validate_mesh_with_triangle_metrics(core,metrics)
    expected=quality.validate_mesh(core,raise_on_error=False)
    assert actual==expected


def test_world_extent_change_rescores_validity_without_reusing_previous_validity():
    trials,settings,points,cells,xyz=fixture();xyz[-1,2]=1e16
    actual=trials.trial(points,cells,xyz);expected=score(points,cells,xyz,settings)
    assert actual.report==expected.report and actual.score==expected.score
    assert actual.report['invalid_element_count']>0


@pytest.mark.parametrize('target',('_certified_linear_validity','_validate_mesh_with_triangle_metrics'))
@pytest.mark.parametrize('kind',('wrapper','defaults'))
def test_reuse_service_mutation_disables_cache_before_rebase(monkeypatch,target,kind):
    trials=PhysicalSplitTrials(SurfaceMeshOptions());assert trials.available()
    original=getattr(quality,target)
    if kind=='wrapper':
        def unknown(*args,**kwargs):raise AssertionError('must not dispatch unknown service')
        monkeypatch.setattr(quality,target,unknown)
    else:
        monkeypatch.setattr(original,'__kwdefaults__',{'_unknown':object()})
    assert not trials.available()


def test_ordered_row_mapping_preserves_unchanged_metrics_across_two_splits():
    trials,settings,points,cells,xyz=fixture()
    first=trials.trial(points,cells[::-1].copy(),xyz)
    assert first.report==score(points,cells[::-1].copy(),xyz,settings).report
    trials.rebase(first,xyz)
    topology=MutableT3Topology(points,cells)
    topology.bisect_interior_edge((0,4))
    points,cells=topology.canonical_export();cells=cells[::-1].copy()
    xyz=np.column_stack((points,.2*points[:,0]*points[:,1]))
    # A changed old XYZ prefix cannot be silently matched through connectivity.
    assert trials.trial(points,cells,xyz) is None
    xyz=np.column_stack((points,np.zeros(len(points))))
    actual=trials.trial(points,cells,xyz);expected=score(points,cells,xyz,settings)
    assert actual.report==expected.report and actual.score==expected.score


def test_world_bbox_expansion_invalidates_reused_unchanged_cells_too():
    trials,settings,points,cells,xyz=fixture()
    first=trials.trial(points,cells,xyz);trials.rebase(first,xyz)
    topology=MutableT3Topology(points,cells);topology.bisect_interior_edge((0,4))
    made,split=topology.canonical_export()
    # Two unchanged flat cells become invalid under the new global world tolerance.
    world=np.column_stack((made,np.zeros(len(made))));world[-1,2]=1e8
    actual=trials.trial(made,split,world);expected=score(made,split,world,settings)
    assert actual.report==expected.report and actual.score==expected.score
    assert actual.report['invalid_element_count']>0


def test_unknown_metric_row_correspondence_uses_original_full_validation():
    core=MeshCore(np.array(((0.,0.,0.),(1.,0.,0.),(0.,1.,0.))),np.array(((0,1,2),)))
    metrics=quality.triangle_quality(core.node_coordinates,core.triangle_connectivity,np.array((99,)))
    assert quality._validate_mesh_with_triangle_metrics(core,metrics)==quality.validate_mesh(core)
