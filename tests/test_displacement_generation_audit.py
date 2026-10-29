"""Source-only audit-retention tests: generation and orientation are both fake."""
from types import SimpleNamespace
import json

import pytest

from anymesher import Mesh, mesh_from_dict, mesh_to_dict
import anymesher.boundary_orientation as boundary
import anymesher.hybrid as hybrid


@pytest.mark.parametrize('certifiable,audit,expected',[
    (True, {'certifiable':True,'completed':True,'model_id':'working-clone','checksum':'actual-audit'}, 'RETAINED'),
    (False, {'certifiable':False,'completed':True}, 'RETAINED'),
    (True, None, 'MISSING_EVIDENCE'),
    (None, {'certifiable':True}, 'MISSING_EVIDENCE'),
    ('true', {'certifiable':True}, 'MISSING_EVIDENCE'),
    (True, {'certifiable':'true'}, 'MISSING_EVIDENCE'),
])
def test_actual_outcomes_are_retained_not_inferred(monkeypatch,certifiable,audit,expected):
    original=Mesh();oriented=Mesh()
    report=None if audit is None else SimpleNamespace(to_dict=lambda:dict(audit))
    seen={}
    def generate(geometry,**options):
        seen.update(options)
        return SimpleNamespace(mesh=original, audit_report=report,certifiable=certifiable,
                               certification_mode=hybrid.CertificationMode.STRICT,strategy_by_face={1:'native'})
    def orient(mesh,geometry,**options):
        assert mesh is original
        return boundary.BoundaryOrientationResult(oriented,{'input_mesh_sha256':boundary._digest(mesh),
             'oriented_mesh_sha256_before_receipt':'permuted-content'},original)
    monkeypatch.setattr(hybrid,'generate_hybrid_mesh_result',generate)
    monkeypatch.setattr(boundary,'orient_displacement_boundary',orient)
    result=boundary.generate_displacement_boundary_mesh_result(object(),target_size=.2,face_ids=(1,))
    record=result.generation_evidence
    assert record['status']==expected
    assert record['audit_report']==audit
    assert record['certifiable']==(certifiable if type(certifiable) is bool else None)
    assert record['original_mesh_sha256']==boundary._digest(original)
    assert record['original_generation_mesh_sha256']==record['original_mesh_sha256']
    assert record['oriented_mesh_sha256_before_receipt']=='permuted-content'
    assert record['generation_and_orientation_implementation']['wheel_sha256'] is None
    assert record['generation_request']['member_ids']==[]
    assert seen['member_ids']==() and seen['face_ids']==(1,)
    saved=mesh_to_dict(result.mesh)
    restored=mesh_from_dict(saved)
    assert restored.hybrid_diagnostics['boundary_orientation']['generation_audit']==record
    json.dumps(record,allow_nan=False)


def test_no_final_callback_after_orientation_binding(monkeypatch):
    finished=False
    def callback():assert not finished
    def generate(geometry,**options):
        options['cancellation_check']('start')
        return SimpleNamespace(mesh=Mesh(),audit_report=None,certifiable=None)
    def orient(*args,**kwargs):
        nonlocal finished
        finished=True
        return boundary.BoundaryOrientationResult(Mesh(),{'input_mesh_sha256':boundary._digest(args[0]),
             'oriented_mesh_sha256_before_receipt':'b'},Mesh())
    monkeypatch.setattr(hybrid,'generate_hybrid_mesh_result',generate)
    monkeypatch.setattr(boundary,'orient_displacement_boundary',orient)
    boundary.generate_displacement_boundary_mesh_result(object(),target_size=.2,face_ids=(1,),cancellation_check=callback)


@pytest.mark.parametrize('mutation_point',['after_generation','orientation_entry'])
def test_changed_generated_mesh_cannot_inherit_earlier_audit(monkeypatch,mutation_point):
    mesh=Mesh();generated=False
    def callback():
        if generated and mutation_point=='after_generation':mesh.activity[9]=.2
    def generate(*args,**kwargs):
        nonlocal generated
        generated=True
        return SimpleNamespace(mesh=mesh,certifiable=True,
                               audit_report=SimpleNamespace(to_dict=lambda:{'certifiable':True}))
    def orient(*args,**kwargs):
        if mutation_point=='orientation_entry':mesh.activity[9]=.2
        return boundary.BoundaryOrientationResult(Mesh(),{'input_mesh_sha256':boundary._digest(mesh),
                    'oriented_mesh_sha256_before_receipt':'b'},mesh)
    monkeypatch.setattr(hybrid,'generate_hybrid_mesh_result',generate)
    monkeypatch.setattr(boundary,'orient_displacement_boundary',orient)
    with pytest.raises(boundary.BoundaryOrientationError,match='generated mesh changed'):
        boundary.generate_displacement_boundary_mesh_result(object(),target_size=.2,face_ids=(1,),cancellation_check=callback)


def test_callback_cannot_rewrite_captured_producer_verdict(monkeypatch):
    producer=None
    def callback():
        if producer is not None:producer.certifiable=False
    def generate(*args,**kwargs):
        nonlocal producer
        producer=SimpleNamespace(mesh=Mesh(),certifiable=True,audit_report=SimpleNamespace(to_dict=lambda:{'certifiable':True}))
        return producer
    def orient(mesh,*args,**kwargs):
        return boundary.BoundaryOrientationResult(Mesh(),{'input_mesh_sha256':boundary._digest(mesh),
                    'oriented_mesh_sha256_before_receipt':'b'},mesh)
    monkeypatch.setattr(hybrid,'generate_hybrid_mesh_result',generate)
    monkeypatch.setattr(boundary,'orient_displacement_boundary',orient)
    result=boundary.generate_displacement_boundary_mesh_result(object(),target_size=.2,face_ids=(1,),cancellation_check=callback)
    assert producer.certifiable is False
    assert result.generation_evidence['certifiable'] is True
