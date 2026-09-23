from __future__ import annotations

import numpy as np
import pytest

from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.native_cpp import COMPILED_TRIANGULATION_AVAILABLE
from anymesher.quad.boundary import BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.seed import build_planar_quad_seed
from anymesher.quad.options import QuadMeshingOptions
from anymesher.refinement import Refinement, SizeField
from anymesher.triangulation import constrained_planar_triangulation


def _signature(geometry: GeometryModel) -> tuple:
    return (
        str(geometry.model_id),
        int(geometry.revision),
        tuple((i, tuple(map(float, geometry.vertex_position(i)))) for i in sorted(geometry.vertices)),
        tuple((i, geometry.edges[i].start, geometry.edges[i].end) for i in sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
    )


def _poly(points) -> tuple[GeometryModel, int]:
    geometry = GeometryModel()
    vertices = geometry.add_points(tuple(points))
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    return geometry, face


def _public(geometry: GeometryModel, faces, h: float, **kwargs):
    return generate_hybrid_mesh_result(
        geometry,
        target_size=float(h),
        face_ids=tuple(faces),
        quad_options=QuadMeshingOptions(),
        **kwargs,
    )


def _validation(result, face: int) -> dict:
    assert result.strategy_by_face[face] == "quad_first"
    assert result.mesh.hybrid_diagnostics["route"] == "quad-first"
    return result.mesh.hybrid_diagnostics["validation"]["faces"][face]


def _inside_connector(point) -> bool:
    x, y = map(float, np.asarray(point, dtype=float)[:2])
    return 3.0 < x < 5.0 and 1.625 < y < 2.375


def _p06_geometry() -> tuple[GeometryModel, int]:
    return _poly(
        (
            (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 1.625, 0.0),
            (5.0, 1.625, 0.0), (5.0, 0.0, 0.0), (8.0, 0.0, 0.0),
            (8.0, 4.0, 0.0), (5.0, 4.0, 0.0), (5.0, 2.375, 0.0),
            (3.0, 2.375, 0.0), (3.0, 4.0, 0.0), (0.0, 4.0, 0.0),
        )
    )


def test_p06_narrow_public_transition_heavy_domain() -> None:
    geometry, face = _p06_geometry()
    before = _signature(geometry)
    result = _public(geometry, (face,), 0.25)
    validation = _validation(result, face)
    assert _signature(geometry) == before
    assert validation["area_ratio"] == pytest.approx(1.0, rel=1.0e-10, abs=1.0e-10)
    assert validation["q4_count_fraction"] >= 0.75
    assert validation["q4_area_fraction"] >= 0.75
    assert any(_inside_connector(position) for position in result.mesh.nodes.values())
    centroids = (
        np.mean([np.asarray(result.mesh.nodes[n], dtype=float) for n in result.mesh.corners_of(eid)], axis=0)
        for eid in result.mesh.elements_of_face[face]
    )
    assert any(_inside_connector(point) for point in centroids)

    geometry2, face2 = _p06_geometry()
    repeat = _public(geometry2, (face2,), 0.25)
    assert (len(result.mesh.nodes), len(result.mesh.quads), len(result.mesh.tris)) == (
        len(repeat.mesh.nodes), len(repeat.mesh.quads), len(repeat.mesh.tris)
    )


def _p07_geometry() -> tuple[GeometryModel, int]:
    return _poly(((0.0, 0.0, 0.0), (8.0, 0.0, 0.0), (8.0, 4.0, 0.0), (0.0, 4.0, 0.0)))


def _near_count(mesh, center=(2.0, 2.0, 0.0), radius=1.25) -> int:
    c = np.asarray(center, dtype=float)
    return sum(float(np.linalg.norm(np.asarray(p, dtype=float) - c)) < radius for p in mesh.nodes.values())


def test_p07_public_refinement_causes_local_grading() -> None:
    uniform_geometry, uniform_face = _p07_geometry()
    graded_geometry, graded_face = _p07_geometry()
    uniform_before = _signature(uniform_geometry)
    graded_before = _signature(graded_geometry)

    uniform = _public(uniform_geometry, (uniform_face,), 1.0)
    graded = _public(
        graded_geometry,
        (graded_face,),
        1.0,
        refinements=(
            Refinement(size=0.25, radius=0.75, center=(2.0, 2.0, 0.0), growth=1.5, name="p07-local"),
        ),
    )
    assert _signature(uniform_geometry) == uniform_before
    assert _signature(graded_geometry) == graded_before
    assert _validation(uniform, uniform_face)["area_ratio"] == pytest.approx(1.0)
    assert _validation(graded, graded_face)["area_ratio"] == pytest.approx(1.0)
    assert len(graded.mesh.nodes) > len(uniform.mesh.nodes)
    assert _near_count(graded.mesh) > _near_count(uniform.mesh)


def _p09_geometry() -> tuple[GeometryModel, int, int, int]:
    g = GeometryModel()
    v = g.add_points(((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0), (4, 0, 0), (4, 1, 0)))
    e01 = g.add_line(v[0], v[1]); shared = g.add_line(v[1], v[2])
    e23 = g.add_line(v[2], v[3]); e30 = g.add_line(v[3], v[0])
    e14 = g.add_line(v[1], v[4]); e45 = g.add_line(v[4], v[5]); e52 = g.add_line(v[5], v[2])
    f1 = g.add_face_from_loop((OrientedEdge(e01, True), OrientedEdge(shared, True), OrientedEdge(e23, True), OrientedEdge(e30, True)))
    f2 = g.add_face_from_loop((OrientedEdge(e14, True), OrientedEdge(e45, True), OrientedEdge(e52, True), OrientedEdge(shared, False)))
    return g, f1, f2, shared


def test_p09_public_reuses_reversed_shared_edge_node_ids() -> None:
    geometry, f1, f2, shared = _p09_geometry()
    before = _signature(geometry)
    d1 = PlanarQuadDomain.from_geometry(geometry, f1)
    d2 = PlanarQuadDomain.from_geometry(geometry, f2)
    result = _public(geometry, (f1, f2), 0.5)
    assert _signature(geometry) == before
    for face in (f1, f2):
        assert _validation(result, face)["area_ratio"] == pytest.approx(1.0)
        assert result.mesh.elements_of_face[face]
    chain = list(result.mesh.nodes_of_edge[shared])
    assert len(chain) >= 3 and all(node in result.mesh.nodes for node in chain)
    use1 = dict(d1.edge_uses)[shared]; use2 = dict(d2.edge_uses)[shared]
    assert use1 != use2
    oriented1 = chain if use1 else list(reversed(chain))
    oriented2 = chain if use2 else list(reversed(chain))
    assert oriented1 == list(reversed(oriented2))

    geometry2, a, b, shared2 = _p09_geometry()
    repeat = _public(geometry2, (a, b), 0.5)
    assert (len(result.mesh.nodes), len(result.mesh.quads), len(result.mesh.tris), len(chain)) == (
        len(repeat.mesh.nodes), len(repeat.mesh.quads), len(repeat.mesh.tris), len(repeat.mesh.nodes_of_edge[shared2])
    )


def test_explicit_quad_first_qualified_s3_publishes_admission() -> None:
    geometry, face = _poly(((0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 2.0, 0.0), (0.0, 2.0, 0.0)))
    before = _signature(geometry)
    result = _public(geometry, (face,), 0.5, qualified_s3=True)
    assert _signature(geometry) == before
    assert result.strategy_by_face[face] == "quad_first"
    assert result.mesh.quads
    record = result.mesh.structural_preparation.get("qualified_s3")
    assert isinstance(record, dict)
    assert isinstance(record.get("admission"), dict) and record["admission"]


def _assert_seed_native_parity(
    geometry: GeometryModel,
    face: int,
    h: float,
    *,
    refinements=(),
    registry: BoundaryStationRegistry | None = None,
) -> None:
    before = _signature(geometry)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    field = SizeField(geometry, float(h), tuple(refinements))
    registry = registry or BoundaryStationRegistry.for_domain(
        geometry, domain, float(h), size_field=field
    )
    seed = build_planar_quad_seed(
        geometry, face, float(h), domain=domain, registry=registry, size_field=field
    )
    python = seed.triangulation
    native = constrained_planar_triangulation(
        python.points, python.outer_loop, holes=python.hole_loops, backend="native"
    )
    assert native.actual_backend == "anymesher-cpp17"
    assert native.points.tobytes() == python.points.tobytes()
    np.testing.assert_array_equal(native.segments, python.segments)
    np.testing.assert_array_equal(native.boundary_segments, python.boundary_segments)
    np.testing.assert_array_equal(native.mandatory_segments, python.mandatory_segments)
    np.testing.assert_array_equal(native.triangles, python.triangles)
    assert _signature(geometry) == before


@pytest.mark.skipif(
    not COMPILED_TRIANGULATION_AVAILABLE,
    reason="PQ4b compiled triangulation parity requires rebuilt native extension",
)
def test_compiled_triangulation_matches_p06_narrow_seed_exactly() -> None:
    geometry, face = _p06_geometry()
    _assert_seed_native_parity(geometry, face, 0.25)


@pytest.mark.skipif(
    not COMPILED_TRIANGULATION_AVAILABLE,
    reason="PQ4b compiled triangulation parity requires rebuilt native extension",
)
def test_compiled_triangulation_matches_p07_graded_seed_exactly() -> None:
    geometry, face = _p07_geometry()
    _assert_seed_native_parity(
        geometry,
        face,
        1.0,
        refinements=(
            Refinement(size=0.25, radius=0.75, center=(2.0, 2.0, 0.0), growth=1.5, name="p07-local"),
        ),
    )


@pytest.mark.skipif(
    not COMPILED_TRIANGULATION_AVAILABLE,
    reason="PQ4b compiled triangulation parity requires rebuilt native extension",
)
def test_compiled_triangulation_matches_p09_shared_edge_seeds_exactly() -> None:
    geometry, f1, f2, _shared = _p09_geometry()
    d1 = PlanarQuadDomain.from_geometry(geometry, f1)
    d2 = PlanarQuadDomain.from_geometry(geometry, f2)
    field = SizeField(geometry, 0.5)
    registry = BoundaryStationRegistry.for_domains(
        geometry, (d1, d2), 0.5, size_field=field
    )
    _assert_seed_native_parity(geometry, f1, 0.5, registry=registry)
    _assert_seed_native_parity(geometry, f2, 0.5, registry=registry)
