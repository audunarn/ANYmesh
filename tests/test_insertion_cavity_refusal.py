"""Actual epoch4491 invalid cavity, refusal provenance and atomic rollback."""
from pathlib import Path
import numpy as np
import pytest
from anymesher import native_cpp,native_v2
from anymesher import _t3_insertion_result as result
from anymesher import _t3_local_validation as local
from anymesher import _t3_correspondence as correspondence
from anymesher.errors import MeshError
from anymesher.triangulation import orient2d


def witness():
    with np.load(Path(__file__).parent/'fixtures/insertion_cavity_4491.npz',allow_pickle=False) as saved:
        return tuple(saved[name].copy() for name in ('points','before','protected','candidate','packet'))


def test_actual_packet_is_rejected_by_both_full_and_qualified_manifold_proofs():
    points,before,protected,candidate,packet=witness()
    options=dict(removed_triangles=7,added_triangles=7,native=True)
    with pytest.raises(result._RejectedInsertionCavity):
        result.validate_insertion(points,before,protected,candidate,packet,options,orient2d)
    subject=native_v2.MutableT3Topology(points,before,protected)
    local.enable_local_validation(subject)
    @correspondence.insertion_correspondence_scope
    def validate(subject):return result.validate_insertion(points,before,protected,candidate,packet,options,orient2d)
    with pytest.raises(result._RejectedInsertionCavity):validate(subject)
    assert correspondence._ACTIVE.get() is None


@pytest.mark.skipif(not native_cpp.NATIVE_CPP_AVAILABLE,reason='compiled insertion witness')
def test_actual_compiled_cavity_refuses_without_mutating_or_committing_source():
    points,before,protected,candidate,packet=witness()
    subject=native_v2.MutableT3Topology(points,before,protected)
    local.enable_local_validation(subject);qualified=subject._local_validation_source
    index=subject._topology_index;counts=dict(qualified.counts);epoch=subject.epoch
    with pytest.raises(native_v2._GeometryLimited,match='non-manifold'):
        subject.insert_point(candidate)
    np.testing.assert_array_equal(subject.points,points);np.testing.assert_array_equal(subject.triangles,before)
    assert subject._topology_index is index and subject.epoch==epoch
    assert subject._local_validation_source is qualified and dict(qualified.counts)==counts
    assert local.source_for_scope(subject) is qualified and correspondence._ACTIVE.get() is None


@pytest.mark.parametrize('kind',('user_typed','ordinary_source'))
def test_external_or_source_errors_never_become_candidate_refusals(monkeypatch,kind):
    from test_t3_insertion_result import fixture
    points,before,protected,candidate,packet,diagnostics=fixture()
    error=result._RejectedInsertionCavity('user') if kind=='user_typed' else MeshError('source inconsistency')
    if kind=='ordinary_source':
        def invalid(*args,**kwargs):raise error
        monkeypatch.setattr(result,'validate_insertion',invalid)
        callback=None
    else:
        def callback(phase):
            if phase=='native-v2 insertion result incidence validation':raise error
    with pytest.raises(type(error)) as caught:
        native_cpp.native_mutable_t3_insert(points,before,protected,candidate,cancellation_check=callback)
    assert caught.value is error and not isinstance(caught.value,native_v2._GeometryLimited)


def test_commit_source_contradiction_is_fatal_not_proposal_refusal():
    from types import SimpleNamespace
    with pytest.raises(MeshError) as caught:local._changed_counts(SimpleNamespace(counts={(0,1):2},protected=frozenset()),{(0,1):1})
    assert type(caught.value) is MeshError

def test_forged_private_exception_from_opaque_validator_is_original(monkeypatch):
    from test_t3_insertion_result import fixture
    points,before,protected,candidate,packet,diagnostics=fixture()
    error=result._RejectedInsertionCavity('forged service')
    def forged(*args,**kwargs):raise error
    monkeypatch.setattr(result,'validate_insertion',forged)
    with pytest.raises(result._RejectedInsertionCavity) as caught:
        native_cpp.native_mutable_t3_insert(points,before,protected,candidate)
    assert caught.value is error


def test_uncertain_orientation_oracle_private_exception_is_original(monkeypatch):
    from test_t3_insertion_result import fixture
    points,before,protected,_,_,_=fixture();candidate=np.array((.5,.5+np.spacing(.5)))
    packet=np.array(((0,1,2),(0,2,4),(0,4,3),(2,3,4)),dtype=np.int64)
    error=result._RejectedInsertionCavity('oracle')
    def oracle(*args):raise error
    import anymesher.triangulation as predicates
    monkeypatch.setattr(predicates,'orient2d',oracle)
    monkeypatch.setattr(native_cpp._compiled,'native_v2_mutable_t3_insert',
        lambda *args:(packet,dict(removed_triangles=1,added_triangles=3,native=True)))
    with pytest.raises(result._RejectedInsertionCavity) as caught:
        native_cpp.native_mutable_t3_insert(points,before,protected,candidate)
    assert caught.value is error


def test_bounded_frontal_retries_genuine_rejected_cavity_then_commits_valid_alternative(monkeypatch):
    from anymesher.triangulation import PlanarTriangulation
    points,before,protected,bad,packet=witness()
    adjacent={}
    for a,b in protected:
        adjacent.setdefault(int(a),[]).append(int(b));adjacent.setdefault(int(b),[]).append(int(a))
    assert all(len(v)==2 for v in adjacent.values())
    ring=[min(adjacent)];previous=None
    while len(ring)<len(adjacent):
        following=next(n for n in sorted(adjacent[ring[-1]]) if n!=previous and n!=ring[0])
        previous,following=ring[-1],following;ring.append(following)
    seed=PlanarTriangulation(points=points,triangles=before,segments=protected,
        boundary_segments=protected,mandatory_segments=np.empty((0,2),dtype=np.int64),
        outer_loop=np.array(ring),hole_loops=())
    original=native_v2._offcentre;proposed=[]
    def first_bad(*args,**kwargs):
        proposed.append(1)
        return bad.copy() if len(proposed)==1 else original(*args,**kwargs)
    monkeypatch.setattr(native_v2,'_offcentre',first_bad)
    output,report=native_v2.frontal_delaunay_refine(seed,
        native_v2.NativeMeshingOptions(point_placement='frontal_delaunay',max_insertions=1,max_topology_operations=8),target_size=.01)
    assert report['insertions']==1 and 1<=report['topology_operations']<=8
    assert report['geometry_limited_regions']>=1 and len(output.points)==len(points)+1
    np.testing.assert_array_equal(output.points[:-1],points)
    native_v2.MutableT3Topology(output.points,output.triangles,protected).validate()

@pytest.mark.parametrize('mutation',('renew_declarations','replace_helper','replace_helper_code','delete_helper'))
def test_actual_kernel_callback_cannot_renew_refusal_provenance(monkeypatch,mutation):
    from test_t3_insertion_result import fixture
    if mutation=='renew_declarations':points,before,protected,candidate,packet,diagnostics=fixture()
    else:points,before,protected,candidate,packet=witness()
    error=result._RejectedInsertionCavity('forged after actual kernel callback')
    changed=[];helper_calls=[];validator_calls=[]
    def forged(points,before,protected,candidate,raw_rows,diagnostics,oracle,callback=None):
        validator_calls.append(1);raise error
    def opaque_helper(*args):helper_calls.append(1);return True
    def callback(phase):
        if changed:return
        changed.append(phase)
        if mutation=='renew_declarations':
            monkeypatch.setattr(result,'validate_insertion',forged)
            monkeypatch.setattr(result,'_CAVITY_VALIDATOR',forged)
            monkeypatch.setattr(result,'_CAVITY_CODES',(forged.__code__,*result._CAVITY_CODES[1:]))
        elif mutation=='replace_helper':monkeypatch.setattr(result,'_genuine_cavity_rejection',opaque_helper)
        elif mutation=='delete_helper':monkeypatch.delattr(result,'_genuine_cavity_rejection')
        else:
            original=result._genuine_cavity_rejection
            monkeypatch.setattr(original,'__code__',(lambda *args:True).__code__)
    with pytest.raises(MeshError) as caught:
        native_cpp.native_mutable_t3_insert(points,before,protected,candidate,cancellation_check=callback)
    assert changed and not isinstance(caught.value,native_v2._GeometryLimited)
    assert not helper_calls and not validator_calls
    if mutation=='renew_declarations':assert str(caught.value)=='insertion validator service changed during callback'
    else:assert type(caught.value) is result._RejectedInsertionCavity
