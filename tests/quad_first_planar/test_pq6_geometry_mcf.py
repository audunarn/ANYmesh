"""PQ6 contract: geometry-derived Q4 MCF acts on real seed corridors."""
from __future__ import annotations

import pytest

from anygeometry.model import GeometryModel
import anymesher.quad.mcf_seed as mcf_seed_module
from anymesher.quad.count_mcf import CountInfeasible
from anymesher.quad.count_model import CountRejected
from anymesher.hybrid import generate_hybrid_mesh_result
from anymesher.quad.boundary import BoundaryStationRegistry
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.driver import run_planar_quad_driver
from anymesher.quad.mcf_seed import optimize_q4_seed_mcf
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.seed import build_planar_quad_seed
from anymesher.quad.state import QuadMeshState
from anymesher.quad.validate import validate_planar_quad_result
from anymesher.refinement import SizeField

SKEW = ((0.0, 0.0), (4.0, 0.0), (3.2, 2.5), (0.4, 2.5))
# The expectations below follow the complete finite-hull seed adopted by the
# exact completeness check.  The earlier Bowyer-Watson seed had 76 triangles and
# an incomplete boundary; the complete seed has 81 (2n - b - 2 with 21 boundary
# stations), and the public mesh still covers the full polygon area of 8.5.
P01 = ((0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0))


def _geometry(points):
    model = GeometryModel()
    vertices = model.add_points(tuple((x, y, 0.0) for x, y in points))
    face = model.add_face(model.add_polyline(vertices, close=True), surface=None)
    return model, face


def _seed(points=SKEW, h=0.5):
    model, face = _geometry(points)
    domain = PlanarQuadDomain.from_geometry(model, face)
    field = SizeField(model, h, ())
    registry = BoundaryStationRegistry.for_domain(
        model, domain, h, size_field=field
    )
    seed = build_planar_quad_seed(
        model, face, h, domain=domain, registry=registry, size_field=field
    )
    return model, face, seed


def _public(points=SKEW, h=0.5):
    model, face = _geometry(points)
    result = generate_hybrid_mesh_result(
        model,
        target_size=h,
        face_ids=(face,),
        quad_options=QuadMeshingOptions(max_local_optimizations=0),
    )
    return model, face, result


def _signature(model):
    return (
        str(model.model_id), int(model.revision),
        tuple((i, tuple(map(float, model.vertex_position(i)))) for i in sorted(model.vertices)),
        tuple((i, model.edges[i].start, model.edges[i].end) for i in sorted(model.edges)),
        tuple(sorted(model.faces)),
    )


@pytest.mark.quad_workers
def test_real_skew_corridor_is_solved_and_mutated_before_driver():
    _model, face, seed = _seed()
    state = seed.state
    protected_nodes = state.protected_nodes
    protected_edges = state.protected_edges

    report = optimize_q4_seed_mcf(state, target_size=0.5)
    assert report.status == "APPLIED"
    assert report.candidate_components == 3
    assert report.eligible_components == 2
    assert report.worker_calls == 2
    assert report.applied_pairs == 4
    assert report.initial_t3 == 76
    assert report.final_t3 == 68
    assert report.initial_q4 == 0
    assert report.final_q4 == 4
    assert len(report.selected_pairs) == 4
    assert report.generation_after == report.generation_before + 1
    assert report.generation_after == state.generation
    assert len(report.added_q4_ids) == 4
    assert all(state.cell_kind(cid) == "Q4" for cid in report.added_q4_ids)
    assert state.protected_nodes == protected_nodes
    assert state.protected_edges == protected_edges

    validation = validate_planar_quad_result(
        state, face=face, reference_area=seed.discrete_area, seed=seed
    )
    assert validation.area_ratio == pytest.approx(1.0, abs=1e-12)


@pytest.mark.quad_workers
def test_mcf_stage_causally_reduces_real_front_work():
    options = QuadMeshingOptions(max_local_optimizations=0)
    _m0, _f0, baseline = _seed()
    baseline_run = run_planar_quad_driver(baseline, options)

    _m1, _f1, staged = _seed()
    mcf = optimize_q4_seed_mcf(staged.state, target_size=0.5)
    staged_run = run_planar_quad_driver(staged, options)

    assert mcf.applied_pairs == 4
    assert baseline_run.report.initial_t3 == 76
    assert staged_run.report.initial_t3 == 68
    assert staged_run.report.attempts < baseline_run.report.attempts
    assert baseline_run.report.attempts == 47
    assert staged_run.report.attempts == 43
    assert baseline_run.report.final_q4 == staged_run.report.final_q4 == 37
    assert baseline_run.report.final_t3 == staged_run.report.final_t3 == 4


@pytest.mark.quad_workers
def test_public_route_reports_geometry_derived_mcf():
    _model, face, result = _public()
    q4 = result.mesh.hybrid_diagnostics["q4"]
    assert q4["status"] == "APPLIED"
    assert q4["worker_calls"] == 2
    assert q4["applied_pairs"] == 4

    face_q4 = q4["faces"][face]
    assert face_q4["status"] == "APPLIED"
    assert face_q4["worker_calls"] == 2
    assert face_q4["applied_pairs"] == 4
    assert result.mesh.hybrid_diagnostics["front"]["faces"][face]["initial_t3"] == 68
    assert len(result.mesh.quads) == 37
    assert len(result.mesh.tris) == 4


def test_uniform_p01_large_component_is_bounded_no_worker_call():
    _model, _face, result = _public(P01, h=0.5)
    q4 = result.mesh.hybrid_diagnostics["q4"]
    assert q4["status"] == "NO_ELIGIBLE"
    assert q4["worker_calls"] == 0
    assert q4["applied_pairs"] == 0
    assert q4["skipped_large"] >= 1
    assert len(result.mesh.quads) == 240
    assert len(result.mesh.tris) == 0


def test_disabled_worker_is_truthful_and_does_not_mutate(monkeypatch):
    _model, _face, seed = _seed()
    before = (seed.state.digest(), seed.state.generation)
    monkeypatch.setenv("ANYMESH_Q4_DISABLE_WORKER", "1")
    report = optimize_q4_seed_mcf(seed.state, target_size=0.5)
    assert report.status == "UNAVAILABLE_SKIPPED"
    assert report.worker_calls == 0
    assert report.applied_pairs == 0
    assert (seed.state.digest(), seed.state.generation) == before
    assert report.generation_before == report.generation_after == seed.state.generation
    assert report.added_q4_ids == ()


@pytest.mark.quad_workers
def test_mcf_report_is_deterministic():
    _m0, _f0, seed0 = _seed()
    _m1, _f1, seed1 = _seed()
    r0 = optimize_q4_seed_mcf(seed0.state, target_size=0.5)
    r1 = optimize_q4_seed_mcf(seed1.state, target_size=0.5)
    assert r0.to_dict() == r1.to_dict()
    assert r0.selected_pairs == ((0, 2), (37, 38), (39, 40), (41, 42))


def test_missing_worker_is_unavailable_without_mutation():
    _model, _face, seed = _seed()
    before = (seed.state.digest(), seed.state.generation)
    report = optimize_q4_seed_mcf(
        seed.state,
        target_size=0.5,
        worker="C:/definitely-missing-anymesh/quad_mcf_worker.exe",
    )
    assert report.status == "UNAVAILABLE_SKIPPED"
    assert report.worker_calls == 0
    assert report.applied_pairs == 0
    assert (seed.state.digest(), seed.state.generation) == before
    assert report.generation_before == report.generation_after == seed.state.generation
    assert report.added_q4_ids == ()


@pytest.mark.quad_workers
def test_cancellation_before_commit_rolls_back_without_consuming_ids():
    class Cancelled(RuntimeError):
        pass

    _model, _face, seed = _seed()
    state = seed.state
    before = (state.digest(), state.next_node_id, state.next_cell_id, state.generation)

    def check(stage: str) -> None:
        if stage == "quad-first:q4-mcf-commit":
            raise Cancelled(stage)

    with pytest.raises(Cancelled):
        optimize_q4_seed_mcf(
            state, target_size=0.5, cancellation_check=check
        )
    assert (state.digest(), state.next_node_id, state.next_cell_id, state.generation) == before


@pytest.mark.quad_workers
def test_count_rejected_propagates_without_mutation(monkeypatch):
    _model, _face, seed = _seed()
    state = seed.state
    before = (state.digest(), state.next_node_id, state.next_cell_id, state.generation)

    def rejected(*args, **kwargs):
        raise CountRejected("synthetic internal model rejection")

    monkeypatch.setattr(mcf_seed_module, "solve_count_instance", rejected)
    with pytest.raises(CountRejected):
        optimize_q4_seed_mcf(state, target_size=0.5)
    assert (state.digest(), state.next_node_id, state.next_cell_id, state.generation) == before


@pytest.mark.quad_workers
def test_public_enabled_vs_disabled_mcf_is_causal(monkeypatch):
    model_e, face_e = _geometry(SKEW)
    before_e = _signature(model_e)
    enabled = generate_hybrid_mesh_result(
        model_e, target_size=0.5, face_ids=(face_e,),
        quad_options=QuadMeshingOptions(max_local_optimizations=0),
    )
    q4e = enabled.mesh.hybrid_diagnostics["q4"]
    bodies_e = {tuple(body) for body in enabled.mesh.quads.values()}
    attempts_e = enabled.mesh.hybrid_diagnostics["front"]["faces"][face_e]["attempts"]
    assert _signature(model_e) == before_e

    monkeypatch.setenv("ANYMESH_Q4_DISABLE_WORKER", "1")
    model_d, face_d = _geometry(SKEW)
    before_d = _signature(model_d)
    disabled = generate_hybrid_mesh_result(
        model_d, target_size=0.5, face_ids=(face_d,),
        quad_options=QuadMeshingOptions(max_local_optimizations=0),
    )
    q4d = disabled.mesh.hybrid_diagnostics["q4"]
    bodies_d = {tuple(body) for body in disabled.mesh.quads.values()}
    attempts_d = disabled.mesh.hybrid_diagnostics["front"]["faces"][face_d]["attempts"]
    assert _signature(model_d) == before_d

    assert q4e["status"] == "APPLIED"
    assert q4e["worker_calls"] == 2
    assert q4e["applied_pairs"] == 4
    assert q4d["status"] == "UNAVAILABLE_SKIPPED"
    assert q4d["worker_calls"] == 0
    assert q4d["applied_pairs"] == 0
    assert len(enabled.mesh.quads) == len(disabled.mesh.quads) == 37
    assert len(enabled.mesh.tris) == len(disabled.mesh.tris) == 4
    assert bodies_e != bodies_d
    assert attempts_e == 43
    assert attempts_d == 47
    assert enabled.mesh.hybrid_diagnostics["validation"]["faces"][face_e]["area_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert disabled.mesh.hybrid_diagnostics["validation"]["faces"][face_d]["area_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_report_exposes_skip_and_provenance_contract():
    _model, _face, seed = _seed()
    report = optimize_q4_seed_mcf(seed.state, target_size=0.5)
    data = report.to_dict()
    for key in (
        "skipped_large", "skipped_unbalanced", "skipped_nonbipartite",
        "skipped_infeasible", "skipped_component_cap", "component_sizes", "arc_counts",
        "added_q4_ids", "generation_before", "generation_after",
    ):
        assert key in data


def test_protected_shared_diagonal_is_excluded_but_protected_endpoints_are_allowed():
    _model, _face, seed = _seed()
    state = seed.state
    pair = (0, 2)
    _adj, pair_data = mcf_seed_module._pair_graph(state)
    assert pair in pair_data
    shared = pair_data[pair][0]

    endpoints_protected = QuadMeshState(
        state.nodes, state.cells, state.cell_kinds,
        initial_front=state.front,
        protected_nodes=state.protected_nodes | frozenset(shared),
        protected_edges=state.protected_edges,
    )
    _adj_ep, pair_data_ep = mcf_seed_module._pair_graph(endpoints_protected)
    assert pair in pair_data_ep

    edge_protected = QuadMeshState(
        state.nodes, state.cells, state.cell_kinds,
        initial_front=state.front,
        protected_nodes=state.protected_nodes | frozenset(shared),
        protected_edges=state.protected_edges | frozenset((shared,)),
    )
    before = (edge_protected.digest(), edge_protected.generation)
    _adj_pe, pair_data_pe = mcf_seed_module._pair_graph(edge_protected)
    assert pair not in pair_data_pe
    assert (edge_protected.digest(), edge_protected.generation) == before
    protected_report = optimize_q4_seed_mcf(edge_protected, target_size=0.5)
    assert pair not in protected_report.selected_pairs


def test_geometry_perturbation_changes_real_derived_cost_signature():
    def signature(points):
        model, face, seed = _seed(points)
        domain = PlanarQuadDomain.from_geometry(model, face)
        field = SizeField(model, 0.5, ())
        adjacency, pair_data = mcf_seed_module._pair_graph(seed.state)
        components, eligible = mcf_seed_module._eligible_components(adjacency)
        assert eligible
        _component, left, right, _arcs = eligible[0]
        instance = mcf_seed_module._instance_for_component(
            seed.state, left, right, pair_data, 0.5, field, domain
        )
        component_signature = tuple(
            (len(component), sum(len(adjacency[node]) for node in component) // 2)
            for component in components
        )
        return component_signature, instance.cost

    perturbed = ((0.0, 0.0), (4.0, 0.0), (3.15, 2.5), (0.45, 2.5))
    assert signature(SKEW) != signature(perturbed)


@pytest.mark.quad_workers
def test_infeasible_component_is_explicit_skip_without_mutation(monkeypatch):
    _model, _face, seed = _seed()
    state = seed.state
    before = (state.digest(), state.next_node_id, state.next_cell_id, state.generation)

    def infeasible(*args, **kwargs):
        raise CountInfeasible("synthetic infeasible component")

    monkeypatch.setattr(mcf_seed_module, "solve_count_instance", infeasible)
    report = optimize_q4_seed_mcf(state, target_size=0.5)
    assert report.status == "NO_MATCH"
    assert report.skipped_infeasible == 2
    assert report.worker_calls == 2
    assert report.applied_pairs == 0
    assert report.added_q4_ids == ()
    assert report.generation_before == report.generation_after == state.generation
    assert (state.digest(), state.next_node_id, state.next_cell_id, state.generation) == before


def test_component_cap_has_distinct_truthful_skip_counter():
    adjacency = {}
    for i in range(5):
        a, b = 2 * i, 2 * i + 1
        adjacency[a] = {b}
        adjacency[b] = {a}
    classified = mcf_seed_module._classify_components(adjacency)
    _components, eligible, skipped_large, skipped_unbalanced, skipped_nonbipartite, skipped_cap, _sizes, _arcs = classified
    assert len(eligible) == 4
    assert skipped_cap == 1
    assert skipped_large == skipped_unbalanced == skipped_nonbipartite == 0
