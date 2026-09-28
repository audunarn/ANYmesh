"""PQ5 contract: TinyAD Q5 optimization on the real generated quad mesh.

Mutation-first freeze. This file defines the acceptance contract for
``src/anymesher/quad/optimize.py`` and the public route integration.

Frozen statuses: DISABLED, NO_ELIGIBLE, NOIMPROVE, APPLIED, UNAVAILABLE_SKIPPED.
Q4 MCF is out of scope for this tranche (stays NOT_INTEGRATED).
"""
from __future__ import annotations

import json
import math

import pytest

from anygeometry.model import GeometryModel
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.options import QuadMeshingOptions
from anymesher.refinement import Refinement, SizeField

# New PQ5 optimizer module (to be implemented per this file).
from anymesher.quad.optimize import (
    QuadOptimizerReport,
    optimize_quad_state,
    default_q5_worker_exe,
    default_q5_worker_root,
)

# ---------------------------------------------------------------------------
# Geometry fixtures
# ---------------------------------------------------------------------------

CAUSAL_GRADED_POINTS = (
    (0.0, 0.0),
    (8.0, 0.0),
    (8.0, 4.0),
    (0.0, 4.0),
)
CAUSAL_REFINEMENTS = (
    Refinement(size=0.25, radius=0.75, center=(2.0, 2.0, 0.0), growth=1.5, name="pq5-local"),
)

P01_POINTS = (
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 6.0),
    (0.0, 6.0),
)


def _poly_area(points) -> float:
    total = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i][:2]
        x1, y1 = points[(i + 1) % n][:2]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _mesh(result):
    return result.mesh


def _node_positions(mesh) -> dict:
    return {nid: (p[0], p[1]) for nid, p in mesh.nodes.items()}


def _cell_topology(mesh) -> set:
    return (
        {(int(cid), tuple(map(int, body)), "Q4") for cid, body in mesh.quads.items()}
        | {(int(cid), tuple(map(int, body)), "T3") for cid, body in mesh.tris.items()}
    )


def _boundary_set(points) -> set:
    return frozenset((round(x, 9), round(y, 9)) for x, y in points)


def _boundary_node_ids(mesh) -> set[int]:
    return {int(nid) for chain in mesh.nodes_of_edge.values() for nid in chain}


def _boundary_displacement(disabled: dict, enabled: dict, source_points) -> float:
    src = _boundary_set(source_points)
    max_d = 0.0
    for nid, (x0, y0) in disabled.items():
        if (round(x0, 9), round(y0, 9)) in src:
            x1, y1 = enabled[nid]
            max_d = max(max_d, math.hypot(x1 - x0, y1 - y0))
    return max_d


def _q5_diagnostics(result) -> dict:
    diag = _mesh(result).hybrid_diagnostics
    assert "q5" in diag, "hybrid_diagnostics must expose a truthful 'q5' block"
    q5 = diag["q5"]
    assert isinstance(q5, dict), "diagnostics q5 must be a truthful dict, not a bare tag"
    return q5


def _public_result(points: tuple, *, h: float, max_local: int, refinements=()):
    options = QuadMeshingOptions(max_local_optimizations=max_local)
    geometry = GeometryModel()
    vertices = geometry.add_points(
        tuple((float(x), float(y), 0.0) for x, y in points)
    )
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    return generate_hybrid_mesh_result(
        geometry,
        target_size=float(h),
        face_ids=(face,),
        quad_options=options,
        refinements=tuple(refinements),
    )


# ===========================================================================
# A. CAUSAL GRADED REAL MESH
# ===========================================================================

@pytest.mark.quad_workers
def test_causal_graded_mesh_applies_q5_moves() -> None:
    disabled = _public_result(CAUSAL_GRADED_POINTS, h=1.0, max_local=0, refinements=CAUSAL_REFINEMENTS)
    enabled = _public_result(CAUSAL_GRADED_POINTS, h=1.0, max_local=4, refinements=CAUSAL_REFINEMENTS)

    md, me = _mesh(disabled), _mesh(enabled)

    # Strict validity + area preservation + source geometry preserved.
    face_validation = next(iter(me.hybrid_diagnostics["validation"]["faces"].values()))
    assert face_validation["area_ratio"] == pytest.approx(1.0, rel=1e-6)

    pos_d = _node_positions(md)
    pos_e = _node_positions(me)
    assert _boundary_displacement(pos_d, pos_e, CAUSAL_GRADED_POINTS) == 0.0
    assert _boundary_node_ids(md) == _boundary_node_ids(me)
    for nid in _boundary_node_ids(md):
        assert pos_d[nid] == pos_e[nid]

    q5 = _q5_diagnostics(enabled)
    assert q5["status"] == "APPLIED", f"expected APPLIED, got {q5}"
    assert q5["applied"] >= 1
    assert q5["attempts"] >= q5["applied"]
    assert q5["objective_initial_sum"] > q5["objective_final_sum"] > 0.0
    assert q5["worker_calls"] >= q5["applied"]
    assert q5["budget_cap"] == 8
    assert q5["budget"] == 4
    assert isinstance(q5["moved_node_ids"], list) and q5["moved_node_ids"]
    assert all(int(node) in me.nodes for node in q5["moved_node_ids"])
    face_q5 = next(iter(q5["faces"].values()))
    assert face_q5["resident_moved_node_ids"]

    q5_d = _q5_diagnostics(disabled)
    assert q5_d["status"] == "DISABLED"

    # Causality: at least one NON-boundary node moved.
    src = _boundary_set(CAUSAL_GRADED_POINTS)
    interior_moved = [
        nid for nid in pos_d
        if pos_d[nid] != pos_e[nid] and (round(pos_d[nid][0], 9), round(pos_d[nid][1], 9)) not in src
    ]
    assert interior_moved, "enabled mesh must move at least one interior node"

    # Optimizer-only gate: topology/connectivity identical, coordinates differ.
    assert _cell_topology(me) == _cell_topology(md)
    assert set(pos_d) == set(pos_e)
    assert pos_d != pos_e

    # PQ6 MCF runs on the same seed in both executions, so Q5 is the only
    # difference between them (the PQ7 lattice clearance lets MCF solve one
    # small component here; before PQ7 it was bounded out entirely).
    q4_enabled = _mesh(enabled).hybrid_diagnostics["q4"]
    assert q4_enabled["status"] in ("NO_ELIGIBLE", "APPLIED")
    assert q4_enabled == _mesh(disabled).hybrid_diagnostics["q4"]


# ===========================================================================
# B. PERFECT P01 NO-OP
# ===========================================================================

@pytest.mark.quad_workers
def test_perfect_p01_optimizer_noop() -> None:
    disabled = _public_result(P01_POINTS, h=0.5, max_local=0)
    enabled = _public_result(P01_POINTS, h=0.5, max_local=8)

    md, me = _mesh(disabled), _mesh(enabled)
    assert _cell_topology(me) == _cell_topology(md)
    pos_d = _node_positions(md)
    pos_e = _node_positions(me)
    assert _boundary_displacement(pos_d, pos_e, P01_POINTS) == 0.0
    for nid in _boundary_node_ids(md):
        assert pos_d[nid] == pos_e[nid]

    q5 = _q5_diagnostics(enabled)
    assert q5["status"] in ("NOIMPROVE", "NO_ELIGIBLE", "APPLIED")
    if q5["status"] == "APPLIED":
        assert q5["max_displacement"] < 1e-6
    else:
        assert q5["applied"] == 0
        assert pos_e == pos_d


# ===========================================================================
# C. PROTECTED / BOUNDARY IMMUTABILITY (direct optimizer)
# ===========================================================================

@pytest.mark.quad_workers
def test_protected_nodes_never_free_and_never_move() -> None:
    """Direct optimizer on a built real state: protected nodes are immutable."""
    from anymesher.quad.driver import run_planar_quad_driver
    from anymesher.quad.seed import build_planar_quad_seed
    from anymesher.quad.domain import PlanarQuadDomain
    from anymesher.quad.boundary import BoundaryStationRegistry

    geometry = GeometryModel()
    vertices = geometry.add_points(
        tuple((float(x), float(y), 0.0) for x, y in CAUSAL_GRADED_POINTS)
    )
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    field = SizeField(geometry, 1.0, CAUSAL_REFINEMENTS)
    registry = BoundaryStationRegistry.for_domain(geometry, domain, 1.0, size_field=field)
    seed = build_planar_quad_seed(geometry, face, 1.0, domain=domain, registry=registry, size_field=field)
    driver_result = run_planar_quad_driver(seed, QuadMeshingOptions(max_local_optimizations=0))
    state = driver_result.state
    prot_before = state.protected_nodes
    report = optimize_quad_state(
        state,
        target_size=1.0,
        max_local_optimizations=4,
        size_field=field,
        domain=domain,
    )
    assert isinstance(report, QuadOptimizerReport)
    assert report.status == "APPLIED"
    assert state.protected_nodes == prot_before
    for nid in report.moved_node_ids:
        assert not state.is_protected_node(nid), f"moved node {nid} is protected"


def test_worker_notfound_returns_unavailable_without_mutation() -> None:
    """Missing worker is mapped truthfully to an unavailable no-mutation report."""
    from anymesher.quad.driver import run_planar_quad_driver
    from anymesher.quad.seed import build_planar_quad_seed

    geometry = GeometryModel()
    vertices = geometry.add_points(
        tuple((float(x), float(y), 0.0) for x, y in P01_POINTS)
    )
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    driver_result = run_planar_quad_driver(
        build_planar_quad_seed(geometry, face, 0.5),
        QuadMeshingOptions(max_local_optimizations=8),
    )
    report = optimize_quad_state(
        driver_result.state,
        target_size=0.5,
        max_local_optimizations=2,
        worker_exe="/C/Users/nobody/missing/quad_tinyad_optimizer.exe",
        worker_root="/C/Users/nobody/missing",
    )
    assert report.status == "UNAVAILABLE_SKIPPED"
    assert report.worker_calls == 0
    assert report.applied == 0


# ===========================================================================
# D. DISABLED CONTRACT
# ===========================================================================

def test_disabled_contract_zero_worker_calls() -> None:
    result = _public_result(CAUSAL_GRADED_POINTS, h=0.5, max_local=0)
    q5 = _q5_diagnostics(result)
    assert q5["status"] == "DISABLED", "must be DISABLED, never NOT_INTEGRATED"
    assert q5["worker_calls"] == 0
    assert q5["attempts"] == 0
    assert q5["applied"] == 0
    assert q5["moved_node_ids"] == []


# ===========================================================================
# E. WORKER UNAVAILABLE CONTRACT (public route, truthful status)
# ===========================================================================

def test_public_route_worker_unavailable_skipped_no_mutation(monkeypatch) -> None:
    """Public route must not silently claim optimization when the worker
    binary is missing: UNAVAILABLE_SKIPPED, no coordinate change."""
    monkeypatch.setenv("ANYMESH_Q5_DISABLE_WORKER", "1")
    disabled = _public_result(CAUSAL_GRADED_POINTS, h=1.0, max_local=0, refinements=CAUSAL_REFINEMENTS)
    result = _public_result(CAUSAL_GRADED_POINTS, h=1.0, max_local=4, refinements=CAUSAL_REFINEMENTS)
    q5 = _q5_diagnostics(result)
    assert q5["status"] == "UNAVAILABLE_SKIPPED"
    assert q5["worker_calls"] == 0
    assert q5["applied"] == 0
    assert _node_positions(_mesh(result)) == _node_positions(_mesh(disabled))


def test_report_to_dict_deterministic_json_safe() -> None:
    from anymesher.quad.driver import run_planar_quad_driver
    from anymesher.quad.seed import build_planar_quad_seed

    geometry = GeometryModel()
    vertices = geometry.add_points(
        tuple((float(x), float(y), 0.0) for x, y in P01_POINTS)
    )
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), surface=None)
    driver_result = run_planar_quad_driver(
        build_planar_quad_seed(geometry, face, 0.5),
        QuadMeshingOptions(max_local_optimizations=8),
    )
    r1 = optimize_quad_state(
        driver_result.state, target_size=0.5, max_local_optimizations=2,
    )
    assert default_q5_worker_root() == default_q5_worker_exe().parent
    d1 = r1.to_dict()
    d2 = r1.to_dict()
    assert d1 == d2
    json.dumps(d1)
    for key in (
        "status", "eligible_nodes", "attempts", "applied", "worker_calls",
        "objective_initial_sum", "objective_final_sum", "moved_node_ids",
        "max_displacement", "budget", "budget_cap",
    ):
        assert key in d1, f"report dict missing key {key!r}"
