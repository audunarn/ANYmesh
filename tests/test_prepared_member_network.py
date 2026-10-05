"""Small geometry/accounting-only controls; no mesh engine or solver."""
from dataclasses import replace
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from anymesher.errors import MeshError
from anymesher._prepared_member_network import (
    query_prepared_member_network_accounting as query,
    validate_prepared_member_network_accounting as validate,
    query_prepared_member_network_stations as stations,
    validate_prepared_member_network_stations as validate_stations,
)

HANDOFF = Path('C:/Github/ANYgeometry/reports/member-network-4dd5c51/chain_network_handoff.py')
HANDOFF_SHA = '180a812625461a8f8f71c9b0c346bf08b56326da63dc4a1235aa0693691446ff'


def build(count=3, **options):
    assert hashlib.sha256(HANDOFF.read_bytes()).hexdigest() == HANDOFF_SHA
    spec = importlib.util.spec_from_file_location('network_handoff', HANDOFF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_chain_network(count, **options)


@pytest.mark.parametrize('count,options',[(3,{}),(5,{'central':True}),
                                        (10,{'mixed':True})])
def test_complete_native_multiuse_network(count,options):
    from anygeometry import to_dict
    model,joint,_original,fixture = build(count,**options)
    before=to_dict(model)
    receipt=query(model,joint)
    assert len(receipt.owner_receipt.relations['members']) == 2*(count-1)+1
    assert len(receipt.points) == 3
    assert {row[2] for row in receipt.traversal} == set(model.member_edge_uses)
    for member in receipt.owner_receipt.relations['members']:
        assert member['source_carrier'] is None
        for mapping in member['source_use_mappings']:
            p,q=(Fraction(*value) for value in mapping['source_span'])
            for use,certificate in zip(mapping['current_member_uses'],mapping['station_certificates']):
                u,v=(Fraction(float(value)) for value in use['parent_range'])
                expected=((1-(v-p)/(q-p),1-(u-p)/(q-p))
                    if mapping['orientation']=='reversed' else ((u-p)/(q-p),(v-p)/(q-p)))
                assert tuple(Fraction(*value) for value in certificate['native_parent_interval']) == expected
    result=stations(model,receipt,target_size=2.)
    assert len({(row[0],row[1]) for row in result.stations}) == len(result.stations)
    for point in result.point_intents:
        assert any(row[0]==point[2] and row[1]==point[4] for row in result.stations)
    for row,sequence in result.traversal_stations:
        global_values=[Fraction(*value[1]) for value in sequence]
        assert global_values == sorted(global_values)
        assert sequence[0][1] == row[6][0] and sequence[-1][1] == row[6][1]
    assert not receipt.mesh_qualified and not receipt.solver_admitted
    assert not result.mesh_qualified and not result.publication_qualified
    assert to_dict(model)==before


def test_network_omission_and_forgery_rejected():
    model,joint,*_=build()
    receipt=query(model,joint)
    with pytest.raises(MeshError,match='contents changed'):
        validate(model,replace(receipt,traversal=receipt.traversal[:-1]))
    payload=receipt.owner_receipt.relations
    payload['members'].pop()
    forged=replace(receipt.owner_receipt,relations_json=json.dumps(payload))
    with pytest.raises(MeshError):
        query(model,joint,owner_receipt=forged)
    other,other_joint,*_=build()
    with pytest.raises(MeshError):
        query(other,other_joint,owner_receipt=receipt.owner_receipt)


def test_original_budget_native_points_and_validation():
    model,joint,*_=build(3,mixed=True)
    receipt=query(model,joint)
    with pytest.raises(MeshError,match='budget exhausted'):
        stations(model,receipt,max_stations=1)
    result=stations(model,receipt,max_stations=100)
    validate_stations(model,result)
    with pytest.raises(MeshError,match='only decrease'):
        validate_stations(model,result,max_stations=101)
    with pytest.raises(MeshError,match='contents changed'):
        validate_stations(model,replace(result,point_intents=result.point_intents[:-1]))


def test_cancellation_and_unsupported_repeated_carrier():
    model,joint,*_=build()
    with pytest.raises(MeshError):
        query(model,joint,cancellation_check=lambda *_:True)
    from anygeometry import GeometryError
    with pytest.raises(GeometryError,match='more than once'):
        build(repeated_carrier=True)


def test_later_callback_cannot_replace_earlier_station_proof(monkeypatch):
    import anygeometry
    model,joint,*_=build()
    receipt=query(model,joint)
    original=anygeometry.query_prepared_authored_boundary_stations
    proofs=[]
    def corrupt(*args,**kwargs):
        result=original(*args,**kwargs)
        if proofs:
            object.__setattr__(proofs[0],'authored_points',())
        proofs.append(result)
        return result
    monkeypatch.setattr(anygeometry,'query_prepared_authored_boundary_stations',corrupt)
    with pytest.raises(MeshError,match='station proof changed'):
        stations(model,receipt)


def test_valid_proof_for_wrong_edge_is_rejected_before_coordinates(monkeypatch):
    import anygeometry
    model,joint,*_=build()
    receipt=query(model,joint)
    original=anygeometry.query_prepared_authored_boundary_stations
    substituted=[]
    def wrong_edge(model,correspondence,edge,parameters,**kwargs):
        other=next(child for loop in correspondence.exterior_loops
                   for _root,_forward,children in loop for child in children if child != edge)
        proof=original(model,correspondence,other,parameters,**kwargs)
        # This is a valid owner proof, not a corrupted record.
        coordinates=tuple(tuple(float(Fraction(*v)) for v in xyz)
                          for xyz in proof.current_polynomial_points)
        anygeometry.validate_prepared_authored_boundary_station_coordinates(model,proof,coordinates)
        substituted.append((edge,proof.edge_id))
        return proof
    monkeypatch.setattr(anygeometry,'query_prepared_authored_boundary_stations',wrong_edge)
    with pytest.raises(MeshError,match='requested edge or occurrence'):
        stations(model,receipt)
    assert substituted and substituted[0][0] != substituted[0][1]


def shared_occurrence_model():
    from anygeometry import GeometryModel,Plane,plan_intersections,apply_intersections
    model=GeometryModel()
    a,b,c,d,e,f=model.add_points(((-1,-1,0),(0,-1,0),(0,1,0),(-1,1,0),(1,-1,0),(1,1,0)))
    shared=model.add_line(b,c)
    left=model.add_face((model.add_line(a,b),shared,model.add_line(c,d),model.add_line(d,a)))
    right=model.add_face((model.add_line(b,e),model.add_line(e,f),model.add_line(f,c),shared))
    for face in (left,right):
        model.set_face_surface(face,Plane((0,0,0),(1,0,0),(0,1,0)))
    model.add_sheet((left,right))
    third=model.add_plate(model.add_points(((-1,0,-1),(1,0,-1),(1,0,1),(-1,0,1))))
    model.set_face_surface(third,Plane((0,0,0),(1,0,0),(0,0,1)))
    model.add_sheet((third,))
    model.add_member((shared,))
    apply_intersections(model,plan_intersections(model,tuple(model.faces),policy='connect'),policy='connect')
    joint=next(a.target_id for a in model.attachments.values() if a.kind=='sheet_on_joint')
    return model,joint


def test_valid_same_edge_wrong_occurrence_is_rejected(monkeypatch):
    import anygeometry
    model,joint=shared_occurrence_model()
    receipt=query(model,joint)
    original=anygeometry.query_prepared_authored_boundary_stations
    substituted=[]
    def wrong_occurrence(model,correspondence,edge,parameters,**kwargs):
        other=next(c for c in receipt.constraint_receipt.boundary_correspondences
            if c.authored_definition.face_id != correspondence.authored_definition.face_id
            and any(edge in children for loop in c.exterior_loops for _root,_forward,children in loop))
        proof=original(model,other,edge,parameters,**kwargs)
        coordinates=tuple(tuple(float(Fraction(*v)) for v in xyz)
                          for xyz in proof.current_polynomial_points)
        anygeometry.validate_prepared_authored_boundary_station_coordinates(model,proof,coordinates)
        assert proof.edge_id == edge
        substituted.append((correspondence.authored_definition.face_id,
                            proof.correspondence.authored_definition.face_id))
        return proof
    monkeypatch.setattr(anygeometry,'query_prepared_authored_boundary_stations',wrong_occurrence)
    with pytest.raises(MeshError,match='requested edge or occurrence'):
        stations(model,receipt)
    assert substituted and substituted[0][0] != substituted[0][1]


def test_valid_shared_owner_occurrences_are_preserved():
    model,joint=shared_occurrence_model()
    receipt=query(model,joint)
    result=stations(model,receipt)
    assert result.stations
    for row in result.stations:
        matches=[c for c in receipt.constraint_receipt.boundary_correspondences
            if any(row[0] in children for loop in c.exterior_loops
                   for _root,_forward,children in loop)]
        assert len(matches)==2
        assert row[4]==min(c.authored_definition.face_id for c in matches)


@pytest.mark.parametrize('capability',['AuthoredBoundaryStations',
    'query_prepared_authored_boundary_stations','validate_prepared_authored_boundary_station_coordinates'])
@pytest.mark.parametrize('missing',[True,False])
def test_missing_station_capability_is_typed(monkeypatch,capability,missing):
    import anygeometry
    if missing:
        monkeypatch.delattr(anygeometry,capability)
    else:
        monkeypatch.setattr(anygeometry,capability,None)
    with pytest.raises(MeshError,match='station capability unavailable'):
        stations(None,None)


def test_station_proof_subclass_is_rejected_before_field_access(monkeypatch):
    import anygeometry
    model,joint,*_=build()
    receipt=query(model,joint)
    class Forged(anygeometry.AuthoredBoundaryStations):
        def __getattribute__(self,name):
            raise AssertionError('untrusted proof field access')
    original=anygeometry.query_prepared_authored_boundary_stations
    def substituted(*args,**kwargs):
        proof=original(*args,**kwargs)
        return Forged(**{field: getattr(proof,field) for field in proof.__dataclass_fields__})
    monkeypatch.setattr(anygeometry,'query_prepared_authored_boundary_stations',substituted)
    with pytest.raises(MeshError,match='proof type'):
        stations(model,receipt)


@pytest.mark.parametrize('size',[False,0,float('nan'),float('inf')])
def test_invalid_size_without_owner_callback(size):
    with pytest.raises(MeshError,match='target size'):
        stations(None,None,target_size=size)
