"""PQ7: quad-first element-shape gates.

Before PQ7 the explicit quad-first route admitted any strictly convex quad
under an absolute 1e-9 tolerance and published near-degenerate elements on
ordinary shapes (a 180-degree Q4 corner on a triangular plate, Q4 aspect
ratios above 300, residual T3 corners below 1 degree).  These tests pin the
admission gate, the residual-T3 improvement pass, the scale-free final
validator and the public outcome on that shape corpus, measuring the published
mesh independently through :mod:`anymesher.quality_v2`.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from anygeometry import GeometryModel, punch_hole

from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.boundary import BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.front import FrontNoCandidate, FrontRejected, front_step, make_quad
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.quality_gate import (
    QUAD_FIRST_Q4_POLICY,
    QUAD_FIRST_T3_MIN_ANGLE,
    corner_metrics,
)
from anymesher.quad.residual import improve_residual_triangles
from anymesher.quad.state import QuadMeshState
from anymesher.quad.validate import QuadQualityRejected, validate_planar_quad_result
from anymesher.quality_v2 import quad_candidate_quality, quad_quality, triangle_quality


def _polygon(points, hole=None):
    model = GeometryModel()
    vertices = model.add_points(tuple((x, y, 0.0) for x, y in points))
    face = model.add_face(model.add_polyline(vertices, close=True), surface=None)
    if hole is not None:
        face, _ = punch_hole(model, face, (hole[0], hole[1], 0.0), hole[2])
    return model, face


_C60, _S60 = math.cos(math.radians(60.0)), math.sin(math.radians(60.0))
SHAPES = {
    "parallelogram": (((0, 0), (6, 0), (6 + 4 * _C60, 4 * _S60), (4 * _C60, 4 * _S60)), None, 0.4),
    "trapezoid": (((0, 0), (4, 0), (3, 4), (1, 4)), None, 0.25),
    "l_shape": (((0, 0), (4, 0), (4, 1.2), (2.2, 1.2), (2.2, 2.6), (0, 2.6)), None, 0.2),
    "plate_hole": (((0, 0), (10, 0), (10, 6), (0, 6)), (3.2, 2.4, 1.2), 0.4),
    "pentagon": (((0, 0), (5, 0.8), (6, 4), (2.5, 6), (-0.5, 3.5)), None, 0.35),
    "triangle": (((0, 0), (6, 0), (2, 5)), None, 0.35),
}


def _assert_published_quality(mesh) -> None:
    ids = sorted(mesh.nodes)
    row = {node: index for index, node in enumerate(ids)}
    points = np.asarray([mesh.nodes[node] for node in ids], dtype=float)
    if mesh.quads:
        quads = np.asarray([[row[n] for n in body[:4]] for body in mesh.quads.values()])
        quality = quad_quality(points, quads)
        assert float(np.min(quality.scaled_jacobian)) >= QUAD_FIRST_Q4_POLICY.minimum_scaled_jacobian
        assert float(np.max(quality.maximum_angle)) <= QUAD_FIRST_Q4_POLICY.maximum_angle + 1e-9
        assert float(np.min(quality.minimum_angle)) >= QUAD_FIRST_Q4_POLICY.minimum_angle - 1e-9
        assert float(np.max(quality.aspect_ratio)) <= QUAD_FIRST_Q4_POLICY.maximum_aspect_ratio
    if mesh.tris:
        tris = np.asarray([[row[n] for n in body[:3]] for body in mesh.tris.values()])
        assert float(np.min(triangle_quality(points, tris).minimum_angle)) >= QUAD_FIRST_T3_MIN_ANGLE - 1e-9


@pytest.mark.parametrize("layout", ("existing", "adaptive"))
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_public_quad_first_publishes_only_gated_elements(shape: str, layout: str) -> None:
    outline, hole, size = SHAPES[shape]
    model, face = _polygon(outline, hole)
    result = generate_hybrid_mesh_result(
        model, target_size=size, face_ids=(face,),
        quad_options=QuadMeshingOptions(), layout_policy=layout,
    )
    mesh = result.mesh
    assert mesh.quads
    _assert_published_quality(mesh)
    report = mesh.hybrid_diagnostics["validation"]["faces"][face]
    assert report["quality_gates"]["q4_maximum_angle"] == QUAD_FIRST_Q4_POLICY.maximum_angle
    assert report["max_q4_angle"] <= QUAD_FIRST_Q4_POLICY.maximum_angle
    if mesh.tris:
        assert report["min_t3_angle"] >= QUAD_FIRST_T3_MIN_ANGLE


def test_triangular_plate_no_longer_publishes_a_straight_angle_quad() -> None:
    # Regression: this exact input published a Q4 with a 179.9999996 degree
    # corner (scaled Jacobian 7e-9) before the PQ7 gate.
    model, face = _polygon(SHAPES["triangle"][0])
    mesh = generate_hybrid_mesh_result(
        model, target_size=0.35, face_ids=(face,), quad_options=QuadMeshingOptions(),
    ).mesh
    _assert_published_quality(mesh)


def test_composite_mapped_side_does_not_overdivide_a_short_edge() -> None:
    # ANYgeometry declares four corners on this L-shaped polyline face, making
    # one mapped side of three edges.  Mapped opposite-side equality used to
    # put 31 divisions on the 1.2 m edge at h=0.2, fanning slivers off it.
    model, face = _polygon(SHAPES["l_shape"][0])
    domain = PlanarQuadDomain.from_geometry(model, face)
    registry = BoundaryStationRegistry.for_domains(model, (domain,), 0.2)
    short_edge = next(
        edge_id for edge_id in model.edges
        if abs(np.linalg.norm(np.subtract(*(model.vertex_position(v) for v in (
            model.edges[edge_id].start, model.edges[edge_id].end)))) - 1.2) < 1e-12
        and model.vertex_position(model.edges[edge_id].start)[0] == 4.0
    )
    assert registry.edge_divisions(short_edge) == 6


def _near_straight_pair() -> QuadMeshState:
    # T3 (0,1,3) + (1,2,3) form a convex quad whose corner at node 1 is
    # ~179.9 degrees: strictly convex, but not an admissible element.
    nodes = {0: (0.0, 0.0), 1: (1.0, -0.001), 2: (2.0, 0.0), 3: (1.0, 1.0)}
    cells = {0: (0, 1, 3), 1: (1, 2, 3)}
    front = ((0, 1), (1, 2), (2, 3), (0, 3))
    return QuadMeshState(nodes, cells, {0: "T3", 1: "T3"}, initial_front=front)


def test_make_quad_rejects_a_convex_but_near_straight_corner() -> None:
    state = _near_straight_pair()
    with pytest.raises(FrontRejected, match="shape gate"):
        make_quad(state, (0, 1, 2, 3))


def test_front_step_gate_rejection_leaves_state_unchanged() -> None:
    state = _near_straight_pair()
    digest, generation = state.digest(), state.generation
    with pytest.raises(FrontNoCandidate):
        front_step(state, (0, 1))
    assert state.digest() == digest
    assert state.generation == generation


def test_gate_is_scale_free() -> None:
    for scale in (1.0e-6, 1.0, 1.0e6):
        nodes = {0: (0.0, 0.0), 1: (scale, 0.0), 2: (scale, scale), 3: (0.0, scale)}
        state = QuadMeshState(nodes, {0: (0, 1, 3), 1: (1, 2, 3)}, {0: "T3", 1: "T3"},
                              initial_front=((0, 1), (1, 2), (2, 3), (0, 3)))
        assert make_quad(state, (0, 1, 2, 3)) == (0, 1, 2, 3)


def test_corner_metrics_match_quality_v2_definitions() -> None:
    rng = random.Random(7)
    for _ in range(50):
        base = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        quad = [(x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2)) for x, y in base]
        min_angle, max_angle, jacobian, aspect = corner_metrics(quad)
        reference = quad_candidate_quality(np.asarray(quad), (0, 1, 2, 3))
        assert min_angle == pytest.approx(reference["minimum_angle"], abs=1e-9)
        assert max_angle == pytest.approx(reference["maximum_angle"], abs=1e-9)
        assert jacobian == pytest.approx(reference["scaled_jacobian"], abs=1e-12)
        assert aspect == pytest.approx(reference["aspect_ratio"], rel=1e-12)


def _thin_pair(protect_diagonal: bool = False) -> QuadMeshState:
    # a=0, b=1 share the long diagonal; apexes c=2 and d=3 are close to it.
    nodes = {0: (0.0, 0.0), 1: (4.0, 0.0), 2: (2.0, 0.5), 3: (2.0, -0.5)}
    cells = {0: (2, 0, 1), 1: (3, 1, 0)}
    front = ((0, 2), (1, 2), (1, 3), (0, 3))
    return QuadMeshState(nodes, cells, {0: "T3", 1: "T3"}, initial_front=front,
                         protected_edges=((0, 1),) if protect_diagonal else ())


def test_residual_flip_raises_minimum_angle() -> None:
    state = _thin_pair()
    report = improve_residual_triangles(state)
    assert report.flips == 1
    assert report.min_t3_angle_after > report.min_t3_angle_before
    assert not state.edge_cells((0, 1))
    assert len(state.edge_cells((2, 3))) == 2
    assert sorted(state.front) == [(0, 2), (0, 3), (1, 2), (1, 3)]


def test_residual_flip_never_touches_a_protected_edge() -> None:
    state = _thin_pair(protect_diagonal=True)
    digest = state.digest()
    report = improve_residual_triangles(state)
    assert report.flips == 0
    assert state.digest() == digest


def test_validator_rejects_a_sliver_residual_triangle() -> None:
    nodes = {0: (0.0, 0.0), 1: (4.0, 0.0), 2: (2.0, 0.1)}
    state = QuadMeshState(nodes, {0: (0, 1, 2)}, {0: "T3"},
                          initial_front=((0, 1), (1, 2), (0, 2)))
    with pytest.raises(QuadQualityRejected, match="T3 cell 0"):
        validate_planar_quad_result(state)
