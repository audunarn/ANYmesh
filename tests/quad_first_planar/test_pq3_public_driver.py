from __future__ import annotations

import pytest

from anygeometry.model import GeometryModel
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.driver import run_planar_quad_driver
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.seed import build_planar_quad_seed
from anymesher.quad.validate import validate_planar_quad_result
from .fixtures import p01_geometry, p02_geometry


def _n_eq(mesh) -> float:
    return len(mesh.quads) + len(mesh.tris) / 2.0


def _diagnostics(mesh):
    return mesh.hybrid_diagnostics


def _run(fixture, h):
    geometry, face_id = fixture()
    return generate_hybrid_mesh_result(
        geometry,
        target_size=h,
        face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
    ).mesh


@pytest.mark.parametrize("h", [1.0, 0.5, 0.25])
def test_p01_public_driver_h_parametrized(h):
    h = float(h)
    mesh = _run(p01_geometry, h)
    n = _n_eq(mesh)
    scale = 60.0 / h**2
    assert 0.70 * scale <= n <= 1.40 * scale
    diag = _diagnostics(mesh)
    assert diag["route"] == "quad-first"
    assert diag["q4"]["status"] == "NOT_INTEGRATED"
    assert diag["q5"]["status"] in {"NOIMPROVE", "NO_ELIGIBLE", "APPLIED", "UNAVAILABLE_SKIPPED"}
    face_validation = diag["validation"]["faces"]
    assert len(face_validation) == 1
    validation = next(iter(face_validation.values()))
    assert validation["area_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert validation["q4_area_fraction"] >= 0.85
    assert validation["n_eq"] == pytest.approx(n, rel=1e-9)
    face_driver = diag["front"]["faces"]
    assert len(face_driver) == 1
    driver = next(iter(face_driver.values()))
    assert driver["final_q4"] + driver["final_t3"] == len(mesh.quads) + len(mesh.tris)
    assert driver["final_q4"] == len(mesh.quads)
    assert driver["final_t3"] == len(mesh.tris)


def test_p01_h_half_q4_dominant_fraction_and_validated_area():
    mesh = _run(p01_geometry, 0.5)
    total = len(mesh.quads) + len(mesh.tris)
    assert total > 0
    assert len(mesh.quads) / total >= 0.85
    validation = next(iter(_diagnostics(mesh)["validation"]["faces"].values()))
    assert validation["q4_area_fraction"] >= 0.85
    assert validation["q4_count_fraction"] >= 0.85


def test_p01_h_half_nodes_q4_fraction_station_chains():
    mesh = _run(p01_geometry, 0.5)
    assert len(mesh.nodes) > 4
    assert len(mesh.quads) > 4
    total = len(mesh.quads) + len(mesh.tris)
    assert len(mesh.quads) / total >= 0.85
    lengths = sorted(len(c) for c in mesh.nodes_of_edge.values())
    assert lengths == [13, 13, 21, 21]
    assert mesh.node_of_vertex
    for nodes in mesh.nodes_of_edge.values():
        assert nodes[0] in mesh.nodes
        assert nodes[-1] in mesh.nodes


def _signature(geometry):
    return (
        str(geometry.model_id), int(geometry.revision),
        tuple((i, tuple(map(float, geometry.vertex_position(i)))) for i in sorted(geometry.vertices)),
        tuple((i, geometry.edges[i].start, geometry.edges[i].end) for i in sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
    )


def test_p01_source_geometry_unchanged_after_meshing():
    geometry, face_id = p01_geometry()
    before = _signature(geometry)
    generate_hybrid_mesh_result(
        geometry,
        target_size=0.5,
        face_ids=(face_id,),
        quad_options=QuadMeshingOptions(),
    )
    assert _signature(geometry) == before


def test_p02_h_half_matches_p01_counts():
    a = _run(p01_geometry, 0.5)
    b = _run(p02_geometry, 0.5)
    assert (len(b.quads), len(b.tris), len(b.nodes)) == (
        len(a.quads),
        len(a.tris),
        len(a.nodes),
    )


def test_public_recovery_is_causal_on_small_trapezoid():
    geometry = GeometryModel()
    vertices = geometry.add_points(((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (3.0, 4.0, 0.0), (1.0, 4.0, 0.0)))
    face_id = geometry.add_plate(vertices)
    public = generate_hybrid_mesh_result(
        geometry, target_size=0.75, face_ids=(face_id,), quad_options=QuadMeshingOptions()
    ).mesh
    report = public.hybrid_diagnostics["front"]["faces"][face_id]
    assert report["recovery_accepts"] >= 1
    assert report["recovery_inserted_nodes"] >= 1

    with_recovery = run_planar_quad_driver(
        build_planar_quad_seed(geometry, face_id, 0.75), QuadMeshingOptions()
    )
    without_recovery = run_planar_quad_driver(
        build_planar_quad_seed(geometry, face_id, 0.75), QuadMeshingOptions(), allow_recovery=False
    )
    area = 12.0
    on = validate_planar_quad_result(with_recovery.state, face=face_id, reference_area=area)
    off = validate_planar_quad_result(without_recovery.state, face=face_id, reference_area=area)
    assert with_recovery.report.recovery_accepts >= 1
    assert with_recovery.report.final_t3 < without_recovery.report.final_t3
    assert on.q4_count_fraction > off.q4_count_fraction
    assert on.q4_area_fraction > off.q4_area_fraction
