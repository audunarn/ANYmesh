"""Fresh owner partition admission through the actual preparation audit."""
from dataclasses import replace

import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anymesher._owner_trim_domains import validated_prepared_partition_disjoint_interiors
from anymesher.preparation import prepare_structural_closure


def partition_pair(*, clone=True, extra_parent=False):
    source = owner.GeometryModel()
    support = NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))
    uv = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    for _index in range(2 if extra_parent else 1):
        points = [source.add_point(*support.evaluate(*p)) for p in uv]
        edges = [source.add_curve(points[i],points[(i+1)%4],CurveOnSurface(support,
            NurbsCurve(1,(0.,0.,1.,1.),(p,uv[(i+1)%4])))) for i,p in enumerate(uv)]
        face = source.add_face(edges,surface=support)
        source.add_sheet((face,))
    working = owner.clone_prepared_geometry(source,new_preparation_epoch=True) if clone else source
    parent = min(working.faces)
    result = owner.apply_plane_split(working,owner.plan_plane_split(working,[parent],
        owner.Plane((.6,0.,0.),(0.,1.,0.),(0.,0.,1.))),policy='imprint')
    children = tuple(child for _parent,children in result.parent_children for child in children)
    # Public result children are local integer IDs, not manufactured lineage.
    assert len(children) == 2
    return source,working,children


def test_actual_preparation_consumes_only_current_owner_sibling_partition(monkeypatch):
    import anymesher.preparation as preparation
    import anymesher._owner_trim_domains as trims
    source,working,pair = partition_pair()
    before = owner.to_dict(source),owner.to_dict(working)
    assert not owner.has_current_intersection_preparation(working)
    actual = owner.query_prepared_partition_disjoint_interiors
    calls = []
    def observe(*args,**kwargs):
        result = actual(*args,**kwargs)
        calls.append((args[1:3],result,kwargs.get('cancellation_check')))
        return result
    monkeypatch.setattr(owner,'query_prepared_partition_disjoint_interiors',observe)
    def forbidden(*_args,**_kwargs):
        raise AssertionError('qualified sibling partition reached unrelated overlap classifier')
    monkeypatch.setattr(preparation,'find_coplanar_overlaps',forbidden)
    monkeypatch.setattr(trims,'validated_complementary_trim_domains',forbidden)
    prepare_structural_closure(working,face_ids=pair,
        options={'automatic_face_connections':False,'automatic_member_connections':False,
                 'automatic_member_sheet_connections':False,'declare_missing_owners':False},
        reuse_working_copy=True)
    assert calls and all(result is True for _parents,result,_callback in calls)
    assert all(callback is None for _parents,_result,callback in calls)
    assert owner.to_dict(source) == before[0]


def test_partition_query_unknown_for_unqualified_direct_split_and_cross_parent():
    _source,direct,pair = partition_pair(clone=False)
    assert not validated_prepared_partition_disjoint_interiors(direct,*pair)
    _source,working,children = partition_pair(extra_parent=True)
    other = next(face for face in working.faces if face not in children)
    assert not validated_prepared_partition_disjoint_interiors(working,children[0],other)


def test_unknown_partition_still_executes_actual_overlap_audit(monkeypatch):
    import anymesher.preparation as preparation
    _source,working,pair = partition_pair(clone=False)
    called = []
    class AuditReached(BaseException):
        pass
    def observe(*args,**kwargs):
        called.append(kwargs.get('candidate_pairs'))
        raise AuditReached()  # Wiring evidence only, never a successful classifier.
    monkeypatch.setattr(preparation,'find_coplanar_overlaps',observe)
    # This tests unknown classification wiring, independently of either owner
    # capability's expanding supported domain. Neither fabricated False grants
    # scientific admission: the actual overlap audit must still be reached.
    monkeypatch.setattr(owner,'query_prepared_partition_disjoint_interiors',
                        lambda *args,**kwargs:False)
    monkeypatch.setattr(owner,'query_same_support_material_disjoint_interiors',
                        lambda *args,**kwargs:False)
    with pytest.raises(AuditReached):
        prepare_structural_closure(working,face_ids=pair,
            options={'automatic_face_connections':False,'automatic_member_connections':False,
                     'automatic_member_sheet_connections':False,'declare_missing_owners':False},
            reuse_working_copy=True)
    assert called


def test_partition_helper_propagates_actual_cancellation_and_raw_stale_refusal():
    _source,working,pair = partition_pair()
    signal = owner.CancellationSignal()
    signal.cancel()
    before = owner.to_dict(working)
    with pytest.raises(owner.GeometryError,match='cancel'):
        validated_prepared_partition_disjoint_interiors(working,*pair,cancellation_check=signal)
    assert owner.to_dict(working) == before
    face = pair[0]
    working._faces[face] = replace(working.faces[face],metadata={'external_raw_edit':True})
    changed = owner.to_dict(working)
    with pytest.raises(owner.GeometryError):
        validated_prepared_partition_disjoint_interiors(working,*pair)
    assert owner.to_dict(working) == changed


@pytest.mark.parametrize('edited', ['source', 'working'])
def test_revision_stale_partition_is_reclassified_without_erasing_explicit_stale_proof(monkeypatch, edited):
    import anymesher.preparation as preparation
    import anymesher._owner_trim_domains as trims
    from anygeometry.prepared_material_source_transfer import _LINKS

    source, working, pair = partition_pair()
    stale_links = _LINKS[working]
    (source if edited == 'source' else working).add_point(100, 100, 100)
    before = owner.to_dict(source), owner.to_dict(working)
    with pytest.raises(owner.GeometryError, match='stale'):
        validated_prepared_partition_disjoint_interiors(working, *pair)
    class CurrentAuditReached(BaseException):
        pass

    def audit(*args, **kwargs):
        assert args[0] is working
        raise CurrentAuditReached()  # Reclassification wiring, never admission.

    monkeypatch.setattr(preparation, 'find_coplanar_overlaps', audit)
    monkeypatch.setattr(trims, 'validated_same_support_material_disjoint_interiors',
                        lambda *args, **kwargs: False)
    monkeypatch.setattr(trims, 'validated_complementary_trim_domains',
                        lambda *args, **kwargs: False)
    with pytest.raises(CurrentAuditReached):
        prepare_structural_closure(working, face_ids=pair,
            options={'automatic_face_connections': False, 'automatic_member_connections': False,
                     'automatic_member_sheet_connections': False, 'declare_missing_owners': False},
            reuse_working_copy=True)
    assert (owner.to_dict(source), owner.to_dict(working)) == before
    assert _LINKS[working] is stale_links
    with pytest.raises(owner.GeometryError, match='stale'):
        validated_prepared_partition_disjoint_interiors(working, *pair)


def test_preparation_never_falls_back_from_malformed_readiness_or_current_native_refusal(monkeypatch):
    import anymesher.preparation as preparation
    source, working, pair = partition_pair()
    before = owner.to_dict(working)
    def forbidden(*args, **kwargs):
        raise AssertionError('owner refusal reached an unrelated fallback')
    monkeypatch.setattr(preparation, 'find_coplanar_overlaps', forbidden)
    actual = owner.has_current_prepared_partition_lineage
    monkeypatch.setattr(owner, 'has_current_prepared_partition_lineage', lambda model: 1)
    from anymesher.errors import MeshError
    with pytest.raises(MeshError, match='malformed classification'):
        prepare_structural_closure(working, face_ids=pair, options=False)
    error = owner.GeometryError('native lineage changed')
    def refuse(model):
        raise error
    monkeypatch.setattr(owner, 'has_current_prepared_partition_lineage', refuse)
    with pytest.raises(owner.GeometryError) as caught:
        prepare_structural_closure(working, face_ids=pair, options=False)
    assert caught.value is error
    monkeypatch.setattr(owner, 'has_current_prepared_partition_lineage', actual)
    assert owner.to_dict(working) == before


@pytest.mark.parametrize('attack', ['commit', 'raw', 'cancel'])
def test_current_readiness_does_not_hide_later_callback_mutation_or_cancellation(monkeypatch, attack):
    import anymesher.preparation as preparation
    from anygeometry.prepared_material_source_transfer import _LINKS
    source, working, pair = partition_pair()
    before = owner.to_dict(working)
    links = _LINKS[working]
    error = RuntimeError('cancel structural partition audit')
    def forbidden(*args, **kwargs):
        raise AssertionError('changed current proof fell through to another classifier')
    monkeypatch.setattr(preparation, 'find_coplanar_overlaps', forbidden)
    called = []
    def frontier(phase):
        if phase != 'structural preparation overlap narrow phase':
            return
        called.append(phase)
        if attack == 'cancel':
            raise error
        if attack == 'commit':
            working.add_point(100, 100, 100)
        else:
            face = pair[0]
            working._faces[face] = replace(working.faces[face], metadata={'callback_raw_edit': True})
    with pytest.raises(RuntimeError if attack == 'cancel' else owner.GeometryError) as caught:
        prepare_structural_closure(working, face_ids=pair,
            options={'automatic_face_connections': False, 'automatic_member_connections': False,
                     'automatic_member_sheet_connections': False, 'declare_missing_owners': False},
            reuse_working_copy=True, cancellation_check=frontier)
    assert called and _LINKS[working] is links
    if attack == 'cancel':
        assert caught.value is error and owner.to_dict(working) == before
    else:
        assert len(working.faces) == len(before['faces'])
        assert len(working.edges) == len(before['edges'])
