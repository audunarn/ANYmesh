"""Narrow originally-unassigned sweeps use a fresh public whole-pair proof."""
import pytest
from anygeometry import GeometryModel,has_current_intersection_preparation
from anygeometry.nurbs import NurbsCurve
from anygeometry.serialization import to_dict
from anymesher.preparation import prepare_structural_closure
from anymesher.errors import MeshError


def pair(crossing=False):
    model=GeometryModel();vertices={};edges=[]
    controls=(((0.,0.,0.),(0.,1.,1.),(0.,2.,0.)),((0.,2.,0.),(0.,3.,-1.),(0.,4.,0.)))
    if crossing:controls=(((0.,-1.,-1.),(0.,0.,0.),(0.,1.,1.)),((0.,-1.,1.),(0.,0.,0.),(0.,1.,-1.)))
    for points in controls:
        curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),points);ends=[]
        for point in curve.evaluate([0.,1.]):
            key=tuple(point)
            if key not in vertices:vertices[key]=model.add_point(*point)
            ends.append(vertices[key])
        edges.append(model.add_curve(*ends,curve))
    model.extrude(edges,(3.,0.,0.));return model


def test_raw_scope_is_retained_across_temporary_owners_without_new_stamp():
    model=pair();before=to_dict(model)
    working,report=prepare_structural_closure(model,options={'declare_missing_owners':True})
    assert to_dict(model)==before and not model.sheets
    assert len(report.temporary_sheet_ids)==2 and len(working.sheets)==2
    assert report.applications==0 and report.candidate_queries==1
    assert any('strong rational sweep' in item for item in report.diagnostics)
    assert not has_current_intersection_preparation(working,face_ids=tuple(working.faces))


def test_uncertified_interior_profiles_are_not_omitted():
    model=pair(crossing=True);before=to_dict(model)
    with pytest.raises(MeshError,match='not strongly certified'):
        prepare_structural_closure(model,options={'declare_missing_owners':True})
    assert to_dict(model)==before


@pytest.mark.parametrize('limit',['maximum_candidate_pairs','maximum_face_records','maximum_edge_records'])
def test_explicit_budgets_keep_refusal_and_source(limit):
    model=pair();before=to_dict(model)
    with pytest.raises(MeshError):
        prepare_structural_closure(model,options={'declare_missing_owners':True,limit:0})
    assert to_dict(model)==before
