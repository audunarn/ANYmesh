"""Fresh owner material proof through the disabled-auto preparation overlap audit.

These tests confer no automatic batch, meshing, quality or source-S3 admission.
Positive classifications are always computed by the genuine geometry owner.
"""
from dataclasses import replace

import numpy as np
import pytest
import anygeometry as owner
from anygeometry.nurbs import NurbsSurface, NurbsCurve, CurveOnSurface
from anygeometry.serialization import to_dict, from_dict
from anymesher.errors import MeshError
from anymesher._owner_trim_domains import (
    validated_prepared_partition_disjoint_interiors,
    validated_same_support_material_disjoint_interiors,
)
from anymesher.preparation import prepare_structural_closure
from test_prepared_partition_admission import partition_pair


OPTIONS = dict(automatic_face_connections=False, automatic_member_connections=False,
    automatic_member_sheet_connections=False, declare_missing_owners=False)


def closed_pair():
    # Genuine public bowl/plane split, as in ANYgeometry test_plane_split.bowl.
    model = owner.GeometryModel()
    heights = (1., -1., 1.)
    support = NurbsSurface(2,2,(0.,0.,0.,1.,1.,1.),(0.,0.,0.,1.,1.,1.),
        tuple(tuple((i-1.,j-1.,heights[i]+heights[j]) for j in range(3)) for i in range(3)))
    uv = ((0.,0.),(1.,0.),(1.,1.),(0.,1.))
    vertices = [model.add_point(*support.evaluate(*point)) for point in uv]
    edges = [model.add_curve(vertices[i],vertices[(i+1)%4],CurveOnSurface(support,
        NurbsCurve(1,(0.,0.,1.,1.),(point,uv[(i+1)%4])))) for i,point in enumerate(uv)]
    face = model.add_face(edges,surface=support)
    model.add_sheet((face,))
    result = owner.apply_plane_split(model,owner.plan_plane_split(model,[face],
        owner.Plane((0.,0.,.5),(1.,0.,0.),(0.,1.,0.))),policy='imprint')
    pair = tuple(child for _parent,children in result.parent_children for child in children)
    assert len(pair) == 2
    return model,pair


def transformed_pair(kind, transport):
    if kind == 'flat':
        _source,model,pair = partition_pair(clone=False)
    else:
        model,pair = closed_pair()
    if transport == 'clone':
        model = model.clone()
    elif transport == 'load':
        model = from_dict(to_dict(model))
    return model,pair


@pytest.mark.parametrize('kind', ['flat','closed'])
@pytest.mark.parametrize('transport', ['direct','clone','load'])
def test_actual_disabled_auto_preparation_consumes_fresh_direct_clone_loaded_materials(
        monkeypatch, kind, transport):
    import anymesher.preparation as preparation
    import anymesher._owner_trim_domains as trims
    model,pair = transformed_pair(kind,transport)
    before = to_dict(model)
    assert not owner.has_current_intersection_preparation(model,face_ids=pair)
    assert validated_prepared_partition_disjoint_interiors(model,*pair) is False
    actual = owner.query_same_support_material_disjoint_interiors
    observed = []
    def observe(geometry,first,second,**kwargs):
        classification = actual(geometry,first,second,**kwargs)
        observed.append((geometry,first,second,classification,kwargs.get('cancellation_check')))
        return classification
    def forbidden(*args,**kwargs):
        raise AssertionError('fresh admitted material pair reached historical overlap fallback')
    monkeypatch.setattr(owner,'query_same_support_material_disjoint_interiors',observe)
    monkeypatch.setattr(trims,'validated_complementary_trim_domains',forbidden)
    monkeypatch.setattr(preparation,'find_coplanar_overlaps',forbidden)
    working,report = prepare_structural_closure(model,face_ids=pair,
        options=OPTIONS,reuse_working_copy=True)
    assert working is model and report.applications == 0
    assert observed and all(row[0] is model and row[3] is True for row in observed)
    assert all(row[1].kind == row[2].kind == 'face' and row[4] is None for row in observed)
    assert to_dict(model) == before


@pytest.mark.parametrize('forged_metadata', [False,True])
def test_real_overlapping_inputs_refuse_via_actual_fallback_despite_shared_support_metadata(monkeypatch,forged_metadata):
    import anymesher.preparation as preparation
    _source,model,children = partition_pair(clone=False,extra_parent=True)
    other = next(face for face in model.faces if face not in children)
    pair = children[0],other
    if forged_metadata:
        for face in pair:
            model.update_face_metadata(face,partition_parent=999,
                complementary=True,disjoint_interiors=True)
    before = to_dict(model)
    assert validated_same_support_material_disjoint_interiors(model,*pair) is False
    actual = preparation.find_coplanar_overlaps
    actual_plan = preparation.plan_intersections
    observed = []
    fallback = []
    def observe(*args,**kwargs):
        observed.append(kwargs.get('candidate_pairs'))
        return actual(*args,**kwargs)
    monkeypatch.setattr(preparation,'find_coplanar_overlaps',observe)
    def observe_fallback(*args,**kwargs):
        fallback.append(args[1])
        return actual_plan(*args,**kwargs)
    monkeypatch.setattr(preparation,'plan_intersections',observe_fallback)
    # These inputs really overlap, but the existing NURBS overlap backend
    # currently refuses qualification rather than proving positive area.
    # Require the actual conservative audit/fallback and a typed refusal;
    # neither generic failure wording nor forged metadata grants admission.
    with pytest.raises((MeshError,owner.GeometryError)):
        prepare_structural_closure(model,face_ids=pair,options=OPTIONS,reuse_working_copy=True)
    assert observed and fallback
    assert to_dict(model) == before


def test_older_owner_missing_fresh_capability_retains_actual_overlap_audit(monkeypatch):
    import anymesher.preparation as preparation
    _source,model,pair = partition_pair(clone=False)
    before = to_dict(model)
    monkeypatch.delattr(owner,'query_same_support_material_disjoint_interiors')
    assert validated_same_support_material_disjoint_interiors(model,*pair) is False
    class ActualAuditReached(BaseException):
        pass
    observed = []
    def observe(*args,**kwargs):
        observed.append(kwargs.get('candidate_pairs'))
        raise ActualAuditReached()
    monkeypatch.setattr(preparation,'find_coplanar_overlaps',observe)
    with pytest.raises(ActualAuditReached):
        prepare_structural_closure(model,face_ids=pair,options=OPTIONS,reuse_working_copy=True)
    assert observed and to_dict(model) == before


@pytest.mark.parametrize('classification', [None,0,1,np.bool_(True),(),object()])
def test_malformed_owner_classification_refuses_before_working_publication(monkeypatch,classification):
    model,pair = transformed_pair('flat','direct')
    before = to_dict(model)
    monkeypatch.setattr(owner,'query_same_support_material_disjoint_interiors',
                        lambda *args,**kwargs:classification)
    with pytest.raises(MeshError,match='malformed classification'):
        prepare_structural_closure(model,face_ids=pair,options=OPTIONS,reuse_working_copy=True)
    assert to_dict(model) == before


def test_fresh_owner_cancellation_and_callback_exception_identity(monkeypatch):
    model,pair = transformed_pair('flat','direct')
    before = to_dict(model)
    signal = owner.CancellationSignal()
    signal.cancel()
    with pytest.raises(owner.GeometryError,match='cancel'):
        validated_same_support_material_disjoint_interiors(model,*pair,cancellation_check=signal)
    failure = RuntimeError('fresh owner callback failure')
    def callback(phase):
        if phase.startswith('fresh material relation'):
            raise failure
    with pytest.raises(RuntimeError) as caught:
        prepare_structural_closure(model,face_ids=pair,options=OPTIONS,
            reuse_working_copy=True,cancellation_check=callback)
    assert caught.value is failure and to_dict(model) == before


def test_owner_refusal_identity_and_stale_raw_geometry_do_not_publish(monkeypatch):
    model,pair = transformed_pair('flat','direct')
    before = to_dict(model)
    actual = owner.query_same_support_material_disjoint_interiors
    failure = owner.GeometryError('fresh owner work/native refusal')
    def refuse(*args,**kwargs):
        raise failure
    monkeypatch.setattr(owner,'query_same_support_material_disjoint_interiors',refuse)
    with pytest.raises(owner.GeometryError) as caught:
        prepare_structural_closure(model,face_ids=pair,options=OPTIONS,reuse_working_copy=True)
    assert caught.value is failure and to_dict(model) == before
    monkeypatch.setattr(owner,'query_same_support_material_disjoint_interiors',actual)
    changed = []
    def mutate(phase):
        if phase.startswith('fresh material relation') and not changed:
            model._faces[pair[0]] = replace(model.faces[pair[0]],metadata={'raw_edit':True})
            changed.append(to_dict(model))
    with pytest.raises(owner.GeometryError):
        validated_same_support_material_disjoint_interiors(model,*pair,cancellation_check=mutate)
    assert changed and to_dict(model) == changed[0]
