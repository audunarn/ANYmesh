from __future__ import annotations

import dataclasses
import json

import pytest

from test_cylindrical_atlas_binding import _sector_model
from test_cylindrical_frontal_integration import _persistent_state
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.high_order import (
    HighOrderBoundaryMidside, HighOrderGeometryReport, HighOrderMeshCertificate, ValidityStatus,
)
from anymesher.quad.options import QuadMeshingOptions
from quad_first_planar.fixtures import p01_geometry
from quad_first_planar.test_pq4_general_domains import _p03_geometry

def _face_id(model, use) -> int:
    return int(model.face_uses[use.id].face_id)

def _generate(model, faces, h=0.5, *, order="quadratic", cancellation_check=None):
    return generate_hybrid_mesh_result(
        model, face_ids=tuple(faces), target_size=float(h), strategy="native",
        native_backend="python", recombine=True, order=order,
        quad_options=QuadMeshingOptions(), cancellation_check=cancellation_check,
    )

def _certificate(mesh):
    payload=mesh.hybrid_diagnostics["high_order_geometry"]
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    return payload

def test_ch5_report_types_are_frozen_and_json_safe():
    boundary=HighOrderBoundaryMidside((5,0.0,1.0),5,(0.0,1.0),0.5,40,0.0,"straight")
    report=HighOrderGeometryReport(
        model_id="m", revision=2, face_id=7, geometry_family="planar",
        chart_kind="PlanarQuadDomain", chart_origin=(0.0,0.0,0.0),
        boundary_projection="source-edge-parameter-midpoint", interior_projection="chord-midpoint",
        q8_count=1, t6_count=0, certified_elements=1, total_elements=1,
        boundary_midsides=(boundary,), edge_curvature_classes=("straight",),
        interior_midside_count=3, max_geometry_residual=0.0,
    )
    cert=HighOrderMeshCertificate(
        ValidityStatus.CERTIFIED_POSITIVE,"m",2,(report,),1,0,4,1,0.0,
    )
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        report.q8_count=99
    payload=cert.to_dict(); assert payload["status"]=="CERTIFIED_POSITIVE"
    assert payload["boundary_midsides"][0]["station_interval"] == [0.0,1.0]
    assert payload["boundary_midsides"][0]["curvature_class"] == "straight"
    assert json.loads(json.dumps(payload,sort_keys=True)) == payload

def test_ch5_p01_certificate_linear_truth_and_repeat_determinism():
    geometry, face_id=p01_geometry()
    linear=_generate(geometry,(face_id,),order="linear").mesh
    assert linear.hybrid_diagnostics["high_order_geometry"] == {"status":"NOT_APPLICABLE","reports":[]}
    first=_generate(geometry,(face_id,)).mesh
    payload=_certificate(first)
    assert payload["status"]=="CERTIFIED_POSITIVE"
    assert (payload["q8_count"],payload["t6_count"],payload["unique_midside_count"])==(240,0,512)
    assert payload["unique_boundary_midside_count"]==64
    face=payload["reports"][0]
    assert (face["model_id"],face["revision"],face["face_id"])==(str(geometry.model_id),int(geometry.revision),face_id)
    assert face["certified_elements"]==face["total_elements"]==240
    assert face["validity_status"]=="CERTIFIED_POSITIVE" and face["invalid_elements"]==0
    assert _certificate(_generate(geometry,(face_id,)).mesh) == payload

def test_ch5_p03_boundary_residual_and_station_provenance():
    p03=_p03_geometry(); geometry, face_id=p03[0], p03[1]
    payload=_certificate(_generate(geometry,(face_id,)).mesh)
    records=payload["reports"][0]["boundary_midsides"]
    assert records and max(float(item["residual"]) for item in records) <= 1.0e-10
    assert payload["max_geometry_residual"] <= 1.0e-10
    assert set(payload["reports"][0]["edge_curvature_classes"]) == {"straight", "analytic_curved"}
    assert len(payload["reports"][0]["chart_origin"]) == 3
    for item in records:
        assert len(item["station_interval"])==2
        lo,hi=item["station_interval"]; assert lo <= item["midpoint_parameter"] <= hi

def test_ch5_cylinder_sector_certificate_and_source_immutability():
    model, selected=_sector_model(False); before=_persistent_state(model)
    face=_face_id(model,selected[0]); mesh=_generate(model,(face,)).mesh
    payload=_certificate(mesh); report=payload["reports"][0]
    assert (payload["q8_count"],payload["t6_count"],payload["unique_midside_count"])==(8,2,26)
    assert report["geometry_family"]=="cylindrical" and report["chart_kind"]=="CylindricalQuadDomain"
    assert report["interior_projection"]=="owner-chart-midpoint"
    assert len(report["chart_origin"]) == 3
    assert "analytic_curved" in report["edge_curvature_classes"]
    assert payload["max_geometry_residual"] <= 1.0e-10
    assert _persistent_state(model)==before

def test_ch5_full_ring_canonical_boundary_ownership_is_unique():
    model, selected=_sector_model(False); before=_persistent_state(model)
    faces=tuple(_face_id(model,use) for use in selected)
    payload=_certificate(_generate(model,faces).mesh)
    assert (payload["q8_count"],payload["t6_count"],payload["unique_midside_count"])==(64,16,168)
    keys=[tuple(item["canonical_edge"]) for item in payload["boundary_midsides"]]
    assert len(keys)==len(set(keys))==payload["unique_boundary_midside_count"]
    assert max((float(item["residual"]) for item in payload["boundary_midsides"]),default=0.0) <= 1.0e-10
    assert set(item["face_id"] for item in payload["reports"])==set(faces)
    assert _persistent_state(model)==before

def test_ch5_positive_certificate_rejects_invalid_face_claim():
    report=HighOrderGeometryReport(
        model_id="m",revision=1,face_id=1,geometry_family="planar",chart_kind="PlanarQuadDomain",
        chart_origin=(0.0,0.0,0.0), boundary_projection="source-edge-parameter-midpoint",
        interior_projection="chord-midpoint", q8_count=1,t6_count=0,
        certified_elements=0,total_elements=1,boundary_midsides=(),edge_curvature_classes=(),
        interior_midside_count=4,max_geometry_residual=0.0,validity_status="INVALID",invalid_elements=1,
    )
    with pytest.raises(ValueError):
        HighOrderMeshCertificate(ValidityStatus.CERTIFIED_POSITIVE,"m",1,(report,),1,0,4,0,0.0)

def test_ch5_certificate_rejects_global_count_mismatch():
    boundary=HighOrderBoundaryMidside((5,0.0,1.0),5,(0.0,1.0),0.5,40,0.0,"straight")
    report=HighOrderGeometryReport(
        model_id="m", revision=2, face_id=7, geometry_family="planar",
        chart_kind="PlanarQuadDomain", chart_origin=(0.0,0.0,0.0),
        boundary_projection="source-edge-parameter-midpoint", interior_projection="chord-midpoint",
        q8_count=1, t6_count=0, certified_elements=1, total_elements=1,
        boundary_midsides=(boundary,), edge_curvature_classes=("straight",),
        interior_midside_count=4, max_geometry_residual=0.0,
    )
    with pytest.raises(ValueError, match="shell counts"):
        HighOrderMeshCertificate(ValidityStatus.CERTIFIED_POSITIVE,"m",2,(report,),2,0,4,1,0.0)


def test_ch5_cancellation_publishes_no_partial_certificate():
    model, selected=_sector_model(False); before=_persistent_state(model)
    face=_face_id(model,selected[0])
    class Cancelled(RuntimeError): pass
    def cancel(phase: str):
        if phase == "quad-first:quadratic-promotion-ready": raise Cancelled(phase)
    with pytest.raises(Cancelled):
        _generate(model,(face,),cancellation_check=cancel)
    assert _persistent_state(model)==before
