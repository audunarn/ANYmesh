"""Public SmartMapped strategy: explicit dispatch, refusal, cancellation and
regular-rectangle preservation through the real quality repair.

No test patches ``anymesher.hybrid`` or any seed function: dispatch is the
public ``strategy="smart_mapped"`` argument and restoration is checked by
object identity of the untouched module global.
"""
from __future__ import annotations

import copy
import math
import types

import numpy as np
import pytest
from anygeometry.model import GeometryModel

from anymesher import hybrid
from anymesher.errors import MeshError
from anymesher.hybrid import MeshingStrategy, generate_hybrid_mesh_result
from anymesher.quad import seed as quad_seed
from anymesher.quad.domain import PlanarQuadDomain
from anymesher.quad.options import QuadMeshingOptions
from anymesher.quad.public_integration import QuadPublicUnsupported
from anymesher.quad.validate import QuadQualityRejected
from anymesher.smart_mapped import (
    SmartMappedAdapterError,
    SmartMappedRefusal,
    SmartMappedRun,
    prepare_smart_mapped_request,
    smart_mapped_eligibility,
)


def _plate(width: float = 4.0, height: float = 3.0):
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, height, 0.0), (0.0, height, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


def _smart(geometry, face, **overrides):
    options = dict(target_size=1.0, face_ids=(face,), strategy="smart_mapped")
    options.update(overrides)
    return generate_hybrid_mesh_result(geometry, **options)


def _assert_global_seed_untouched() -> None:
    assert hybrid.build_planar_quad_seed is quad_seed.build_planar_quad_seed


# -- eligibility ------------------------------------------------------------
def test_eligibility_is_cheap_conservative_and_side_effect_free() -> None:
    geometry, face = _plate()
    revision = geometry.revision
    verdict = smart_mapped_eligibility(geometry, (face,), 1.0)
    assert verdict.eligible and verdict.reasons == ()
    assert 12 <= verdict.estimated_cells <= verdict.max_cells
    assert verdict.estimated_work > 0
    assert verdict.to_dict()["eligible"] is True
    assert geometry.revision == revision
    _assert_global_seed_untouched()


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"target_size": 0.0}, "target_size"),
        ({"refinements": (object(),)}, "non-uniform"),
        ({"overrides": {1: 3}}, "overrides"),
        ({"seeding": object()}, "Seeding"),
        ({"beam_edges": (9999,)}, "beam"),
        ({"quad_face_ids": (1,)}, "quad_face_ids"),
        ({"order": "cubic"}, "order"),
        ({"max_cells": 3}, "estimated cells"),
        ({"max_work": 3}, "estimated work"),
    ],
)
def test_eligibility_refuses_unsupported_requests(kwargs, fragment) -> None:
    geometry, face = _plate()
    arguments = {"target_size": 1.0}
    arguments.update(kwargs)
    verdict = smart_mapped_eligibility(geometry, (face,), **arguments)
    assert not verdict.eligible
    assert any(fragment in reason for reason in verdict.reasons), verdict.reasons


def test_eligibility_refuses_unknown_face_without_raising() -> None:
    geometry, _face = _plate()
    verdict = smart_mapped_eligibility(geometry, (9999,), 1.0)
    assert not verdict.eligible and any("unknown face" in item for item in verdict.reasons)


# -- request preparation / refusal -----------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"quad_face_ids": (0,)},
        {"seeding": object()},
        {"refinements": (object(),)},
        {"overrides": {1: 2}},
        {"deferred": True},
    ],
)
def test_prepare_request_fails_closed(kwargs) -> None:
    arguments = dict(
        quad_options=None, quad_face_ids=None, structured_options=None,
        seeding=None, refinements=(), overrides=None, deferred=False,
    )
    arguments.update(kwargs)
    with pytest.raises(SmartMappedRefusal) as caught:
        prepare_smart_mapped_request(**arguments)
    assert isinstance(caught.value, MeshError)
    assert not isinstance(caught.value, QuadPublicUnsupported)


def test_public_strategy_refuses_unsupported_inputs_without_fallback() -> None:
    geometry, face = _plate()
    for extra in ({"quad_face_ids": (face,)}, {"overrides": {0: 2}},
                  {"structured_options": {}}):
        with pytest.raises(SmartMappedRefusal):
            _smart(geometry, face, **extra)
    _assert_global_seed_untouched()


# -- real public dispatch ----------------------------------------------------
def test_strategy_enum_and_public_dispatch_without_patching() -> None:
    assert MeshingStrategy("smart_mapped") is MeshingStrategy.SMART_MAPPED
    geometry, face = _plate()
    revision, model_id = geometry.revision, geometry.model_id
    _assert_global_seed_untouched()
    result = _smart(geometry, face)
    _assert_global_seed_untouched()
    assert (geometry.revision, geometry.model_id) == (revision, model_id)
    assert dict(result.strategy_by_face) == {face: "smart_mapped"}
    assert result.triangulation_backend_by_face[face]["backend"] == "smart_mapped"
    diagnostics = result.mesh.hybrid_diagnostics
    assert diagnostics["route"] == "quad-first"
    assert diagnostics["meshing_strategy"] == "smart_mapped"
    report = diagnostics["smart_mapped"]
    assert report["strategy"] == "smart_mapped"
    assert report["global_patch"] is False
    assert report["dispatch"] == "explicit_seed_builder_argument"
    assert report["faces"] == [face]
    assert report["pure_prototype_claim"] is False
    assert result.mesh.quads
    assert set(result.mesh.elements_of_face[face]) == set(result.mesh.quads) | set(result.mesh.tris)


def test_ordinary_quad_first_is_not_labelled_smart_mapped() -> None:
    geometry, face = _plate()
    result = generate_hybrid_mesh_result(
        geometry, target_size=1.0, face_ids=(face,), quad_options=QuadMeshingOptions())
    assert dict(result.strategy_by_face) == {face: "quad_first"}
    assert "smart_mapped" not in result.mesh.hybrid_diagnostics
    assert "meshing_strategy" not in result.mesh.hybrid_diagnostics


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_regular_footprints_are_preserved_through_every_stage(order) -> None:
    geometry, face = _plate()
    result = _smart(geometry, face, order=order)
    footprints = result.mesh.hybrid_diagnostics["smart_mapped"]["regular_footprints"]
    assert footprints["seeded"] > 0
    assert footprints["per_face"][face] == footprints["seeded"]
    assert set(footprints["stages"]) >= {"front_q5", "before_repair", "after_repair", "final"}
    for stage, counts in footprints["stages"].items():
        assert counts["lost"] == 0 and counts["moved"] == 0, stage
        assert counts["preserved"] == counts["regular_footprints"] == footprints["seeded"], stage
    assert footprints["final_lost"] == 0
    assert footprints["final_preserved"] == footprints["seeded"]
    repair = result.mesh.hybrid_diagnostics["quad_quality_repair"]
    guard = repair["regular_footprint_guard"]
    assert guard["regular_cells"] == footprints["seeded"]
    assert guard["blocked_triangles"] == []
    assert repair["unresolved_quads"] == []


def test_smart_mapped_is_deterministic() -> None:
    geometry, face = _plate()
    first = _smart(geometry, face).mesh
    second = _smart(geometry, face).mesh
    assert first.quads == second.quads and first.tris == second.tris
    assert set(first.nodes) == set(second.nodes)
    for node, position in first.nodes.items():
        np.testing.assert_array_equal(position, second.nodes[node])


# -- the real repair keeps regular footprints frozen -------------------------
def _repair_inputs(order: str = "linear"):
    geometry, face = _plate()
    mesh = copy.deepcopy(_smart(geometry, face, order=order).mesh)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    return geometry, face, mesh, {face: domain}


def test_repair_with_frozen_regular_cells_is_a_no_op_on_good_mesh() -> None:
    geometry, face, mesh, domains = _repair_inputs()
    quads = dict(mesh.quads)
    nodes = {node: np.array(pos, dtype=float) for node, pos in mesh.nodes.items()}
    regular = frozenset(quads)
    hybrid._repair_quad_first_quality(
        mesh, geometry, domains,
        protected_regular_nodes=frozenset(n for body in quads.values() for n in body),
        regular_cells=regular,
    )
    assert dict(mesh.quads) == quads
    for node, position in nodes.items():
        np.testing.assert_array_equal(mesh.nodes[node], position)
    assert mesh.hybrid_diagnostics["quad_quality_repair"]["regular_footprint_guard"][
        "regular_cells"] == len(regular)


def test_repair_reports_blocked_regular_cell_instead_of_splitting_or_moving() -> None:
    geometry, face, mesh, domains = _repair_inputs()
    boundary = {int(n) for chain in mesh.nodes_of_edge.values() for n in chain}
    target = None
    for element_id, body in sorted(mesh.quads.items()):
        free = [n for n in body if n not in boundary]
        if free:
            target = (element_id, body, free[0])
            break
    assert target is not None, "plate must have an interior regular quad"
    element_id, body, free_node = target
    opposite = body[(body.index(free_node) + 2) % 4]
    # Collapse the quad onto its diagonal: its Jacobian is far below the floor.
    mesh.nodes[free_node] = (
        np.asarray(mesh.nodes[opposite], dtype=float)
        + 1.0e-6 * (np.asarray(mesh.nodes[body[(body.index(free_node) + 1) % 4]], dtype=float)
                    - np.asarray(mesh.nodes[opposite], dtype=float))
    )
    collapsed = np.array(mesh.nodes[free_node], dtype=float)
    with pytest.raises(QuadQualityRejected, match="normalized-Jacobian floor"):
        hybrid._repair_quad_first_quality(
            mesh, geometry, domains,
            protected_regular_nodes=frozenset(body),
            regular_cells=frozenset({element_id}),
        )
    # The blocked regular cell is neither split nor relocated (neighbouring
    # irregular cells may still be repaired around its frozen nodes).
    assert element_id in mesh.quads
    for node in body:
        assert node in mesh.nodes
    np.testing.assert_array_equal(mesh.nodes[free_node], collapsed)


# -- seed-level footprint audit ---------------------------------------------
def test_seed_footprints_detect_a_moved_regular_node() -> None:
    geometry, face = _plate()
    run = SmartMappedRun()
    seed = run.seed_builder(geometry, face, 1.0)
    _assert_global_seed_untouched()
    footprints = run._faces[face].footprints
    assert footprints, "a 4x3 plate must seed regular rectangles"
    run.audit_state(face, seed.state, "unit")
    node = footprints[0].body[0]
    x, y = seed.state.position(node)
    transaction = seed.state.transaction()
    transaction.move_node(node, (x + 1.0e-3, y))
    transaction.commit()
    with pytest.raises(QuadQualityRejected, match="not preserved after unit"):
        run.audit_state(face, seed.state, "unit")
    with pytest.raises(Exception):
        run.seed_builder(geometry, face, 1.0)  # single use per face


# -- cancellation ------------------------------------------------------------
class _Stop(MeshError):
    pass


def test_cancellation_propagates_and_leaves_no_global_state() -> None:
    geometry, face = _plate()
    revision = geometry.revision
    phases: list[str] = []

    def cancel(phase: str) -> None:
        phases.append(phase)
        if phase == "quad-first:face-seed":
            raise _Stop("cancelled by test")

    with pytest.raises(_Stop):
        _smart(geometry, face, cancellation_check=cancel)
    assert "quad-first:face-seed" in phases
    _assert_global_seed_untouched()
    assert geometry.revision == revision
    # A fresh public call afterwards still succeeds (nothing was left locked).
    assert _smart(geometry, face).strategy_by_face[face] == "smart_mapped"


# -- eligibility: beams admitted, typed deterministic refusals ---------------
def test_eligibility_admits_supported_beams_and_labels_estimates() -> None:
    geometry, face = _plate()
    edge = min(geometry.edges)
    verdict = smart_mapped_eligibility(
        geometry, (face,), 1.0, beam_edges=(edge,), order="quadratic")
    assert verdict.eligible and verdict.reasons == () and verdict.codes == ()
    exported = verdict.to_dict()
    assert exported["estimate_kind"] == "heuristic_scale_estimate_not_a_proven_bound"
    assert exported["codes"] == []
    unknown = smart_mapped_eligibility(geometry, (face,), 1.0, beam_edges=(9999,))
    assert unknown.codes == ("unknown_beam_edge",)


def test_eligibility_early_exit_codes_are_typed_and_deterministic() -> None:
    geometry, face = _plate()
    first = smart_mapped_eligibility(
        geometry, (face,), 1.0, overrides={min(geometry.edges): 2}, refinements=(object(),))
    second = smart_mapped_eligibility(
        geometry, (face,), 1.0, overrides={min(geometry.edges): 2}, refinements=(object(),))
    assert first.codes == ("non_uniform_size_field", "edge_division_pins")
    assert (first.codes, first.reasons) == (second.codes, second.reasons)
    # Early exit: no chart was built, so no per-face estimate exists.
    assert first.faces == () and first.estimated_cells == 0 and not first.eligible
    capped = smart_mapped_eligibility(geometry, (face,), 1.0, max_cells=3, max_work=3)
    assert capped.codes == ("cell_cap", "work_cap")
    _assert_global_seed_untouched()


# -- boundary fidelity profile: analytic arc interpolation bound -------------------
def _arc_plate():
    try:
        from test_smart_mapped_workflow import _arc_hole_plate
    except ImportError:  # tests collected as a package
        from tests.test_smart_mapped_workflow import _arc_hole_plate
    return _arc_hole_plate()


def test_arc_span_formula_error_and_divisions() -> None:
    from anymesher.smart_mapped import (
        ARC_RADIAL_TOLERANCE, arc_divisions_for_tolerance, arc_interpolation_radial_error,
        arc_max_span,
    )
    assert ARC_RADIAL_TOLERANCE == 5.0e-5
    for radius in (0.05, 0.5, 2.0, 5.0, 100.0):
        span = arc_max_span(radius)
        epsilon = min(5.0e-5 / radius, 1.0)
        spec = 2.0 * math.acos(min(1.0, max(-1.0, 1.0 - 2.0 * math.sqrt(2 * epsilon - epsilon**2))))
        assert span == pytest.approx(spec, rel=1e-9)
        # The analytic error at the maximum span equals the bound ...
        assert arc_interpolation_radial_error(radius, span) == pytest.approx(5.0e-5, rel=1e-6)
        # ... and the chosen count keeps every span within it.
        sweep = math.pi / 2.0
        count = arc_divisions_for_tolerance(radius, sweep)
        assert count == max(1, math.ceil(sweep / span))
        assert arc_interpolation_radial_error(radius, sweep / count) <= 5.0e-5 * (1 + 1e-9)
    # Direct (unstable-form) formula agrees at a moderate span.
    delta, radius = 0.4, 1.5
    c = 1.0 - math.cos(delta / 2.0)
    assert arc_interpolation_radial_error(radius, delta) == pytest.approx(
        radius * (1.0 - math.sqrt(1.0 - c * c / 4.0)), rel=1e-9)
    # Tiny radius: epsilon clamps to 1, one full-turn span.
    assert arc_max_span(1.0e-6) == pytest.approx(2.0 * math.pi)
    assert arc_divisions_for_tolerance(1.0e-6, math.pi) == 1


@pytest.mark.parametrize(
    "radius, sweep, tolerance",
    [(float("nan"), 1.0, 5e-5), (0.0, 1.0, 5e-5), (-1.0, 1.0, 5e-5), (float("inf"), 1.0, 5e-5),
     (1.0, 0.0, 5e-5), (1.0, float("nan"), 5e-5), (1.0, 7.0, 5e-5), (1.0, "x", 5e-5),
     (1.0, 1.0, 0.0), (1.0, 1.0, float("nan")), (0.5, 3.0, 1.0e-300)],
)
def test_arc_division_refuses_invalid_input_with_a_typed_error(radius, sweep, tolerance) -> None:
    from anymesher.smart_mapped import arc_divisions_for_tolerance

    with pytest.raises(SmartMappedRefusal):
        arc_divisions_for_tolerance(radius, sweep, tolerance)


def test_boundary_overrides_derive_arc_counts_and_leave_plain_plates_alone() -> None:
    from anymesher.smart_mapped import arc_divisions_for_tolerance

    geometry, face = _plate()
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    pinned = {99: 3}
    assert SmartMappedRun().boundary_overrides(geometry, (domain,), 1.0) is None
    assert SmartMappedRun().boundary_overrides(geometry, (domain,), 1.0, overrides=pinned) is pinned

    geometry, face, arcs = _arc_plate()
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    expected = arc_divisions_for_tolerance(0.5, math.pi / 2.0)
    derived = SmartMappedRun().boundary_overrides(geometry, (domain,), 0.4)
    assert expected > 2  # beyond the 0.4 m target-sized count of a 0.785 m arc
    assert derived == {edge: expected for edge in arcs}
    first = min(arcs)
    kept = SmartMappedRun().boundary_overrides(geometry, (domain,), 0.4, overrides={first: 9})
    assert kept[first] == 9 and all(kept[e] == expected for e in arcs if e != first)
    with pytest.raises(SmartMappedRefusal):
        SmartMappedRun().boundary_overrides(geometry, (domain,), float("nan"))


def _arc_bound_error(geometry, mesh, arcs) -> float:
    """Independent check: parabola through two stations and the owner midpoint."""
    worst = 0.0
    t = np.linspace(0.0, 1.0, 401)
    basis = np.stack([(1 - t) * (1 - 2 * t), 4 * t * (1 - t), t * (2 * t - 1)], axis=1)
    for edge, (cx, cy, _t0) in arcs.items():
        chain = [np.asarray(mesh.nodes[n], dtype=float) for n in mesh.nodes_of_edge[edge]]
        params = [float(geometry.closest_edge_point(edge, p)[1]) for p in chain]
        for k in range(len(chain) - 1):
            mid = np.asarray(geometry.sample_edge(edge, [0.5 * (params[k] + params[k + 1])]),
                             dtype=float).reshape(3)
            curve = basis @ np.stack([chain[k], mid, chain[k + 1]])
            radial = np.hypot(curve[:, 0] - cx, curve[:, 1] - cy)
            worst = max(worst, float(np.max(np.abs(radial - 0.5))))
    return worst


def test_arc_hole_boundary_is_curve_bound_with_the_same_counts_in_both_orders() -> None:
    from anymesher.smart_mapped import arc_divisions_for_tolerance

    geometry, face, arcs = _arc_plate()
    expected = arc_divisions_for_tolerance(0.5, math.pi / 2.0)
    results = {}
    for order in ("linear", "quadratic"):
        results[order] = generate_hybrid_mesh_result(
            geometry, target_size=0.4, face_ids=(face,), strategy="smart_mapped", order=order)
        _assert_global_seed_untouched()
    reports = {order: item.mesh.hybrid_diagnostics["smart_mapped"]["boundary_fidelity"]
               for order, item in results.items()}
    for report in reports.values():
        assert report["profile"] == "smart_mapped_arc_radial_interpolation/1"
        assert report["arc_radial_tolerance_m"] == 5.0e-5
        assert report["geometry_tolerance_changed"] is False
        assert report["qualification_claim"] is False
        assert set(report["arc_divisions"]) == set(arcs)
        assert all(count >= expected for count in report["arc_divisions"].values())
        assert all(item["analytic_radial_error"] <= 5.0e-5 * (1 + 1e-9)
                   for item in report["arcs"].values())
    assert reports["linear"]["arc_divisions"] == reports["quadratic"]["arc_divisions"]
    assert len(results["linear"].mesh.quads) == len(results["quadratic"].mesh.quads)
    linear = results["linear"].mesh
    for edge in arcs:
        assert len(linear.nodes_of_edge[edge]) - 1 == reports["linear"]["arc_divisions"][edge]
    assert _arc_bound_error(geometry, linear, arcs) <= 5.0e-5 * (1 + 1e-3)


# -- optimization-only pins ------------------------------------------------------
def test_optimization_pins_are_exact_temporary_and_released_before_the_validator() -> None:
    geometry, face = _plate()
    run = SmartMappedRun()
    seed = run.seed_builder(geometry, face, 1.0)
    state = seed.state
    geometry_nodes = frozenset(int(n) for n in state.protected_nodes)
    geometry_edges = frozenset(state.protected_edges)
    footprint_nodes = {int(n) for item in run._faces[face].footprints for n in item.body}
    extras = footprint_nodes - geometry_nodes
    assert extras, "a 4x3 plate must carry interior regular nodes"

    run.protect_optimization_pins(face, state)
    assert frozenset(int(n) for n in state.protected_nodes) == geometry_nodes | extras
    assert frozenset(state.protected_edges) == geometry_edges
    with pytest.raises(SmartMappedAdapterError):
        run.protect_optimization_pins(face, state)  # not re-entrant
    node = min(extras)
    transaction = state.transaction()
    with pytest.raises(MeshError, match="protected"):
        transaction.move_node(node, state.position(node))
    transaction.close()
    run.audit_state(face, state, "front_q5")  # pins unchanged: audit passes

    run.release_optimization_pins(face, state)
    assert frozenset(int(n) for n in state.protected_nodes) == geometry_nodes
    assert frozenset(state.protected_edges) == geometry_edges
    run.release_optimization_pins(face, state)  # idempotent once released
    assert run.report()["optimization_pins"]["per_face"][face] == {
        "held": len(extras), "released": len(extras)}


def test_tampered_optimization_pins_fail_closed() -> None:
    geometry, face = _plate()
    run = SmartMappedRun()
    state = run.seed_builder(geometry, face, 1.0).state
    run.protect_optimization_pins(face, state)
    extra = min(int(n) for n in run._faces[face].footprints[0].body
                if int(n) in run._faces[face].pins[2])
    transaction = state.transaction()
    transaction.release_node(extra)
    transaction.commit()
    with pytest.raises(QuadQualityRejected, match="optimization pins"):
        run.audit_state(face, state, "front_q5")
    with pytest.raises(QuadQualityRejected, match="optimization pins"):
        run.release_optimization_pins(face, state)


def test_public_run_releases_every_optimization_pin_and_keeps_geometry_nodes() -> None:
    geometry, face = _plate()
    report = _smart(geometry, face).mesh.hybrid_diagnostics["smart_mapped"]
    pins = report["optimization_pins"]["per_face"][face]
    assert pins["held"] > 0 and pins["released"] == pins["held"]
    assert report["boundary_fidelity"]["arc_divisions"] == {}


# -- canonical boundary refinement beside a curved source edge -----------------
# -- bounded local irregular-cavity repair -------------------------------------
# Actual failing patch: face 248 of the ORIGINAL 15 m barge at h = 0.25.  The
# blocked T3 6895 sits beside source arc 24 (340 -> 13, exact midside below);
# the adjacent NON-regular Q4 6883 shares edge 13-7131; 7136, 7130 and 7131 are
# frozen regular-footprint nodes, 13 and 340 are boundary stations.  7200/7201
# and the rectangle 6001 stand for the regular footprint beside the cavity.
_CAVITY_CHART = {
    13: (0.7000000000000002, 0.95),
    7136: (0.4666666666666668, 1.0),
    7130: (0.4666666666666668, 0.75),
    7131: (0.7000000000000002, 0.75),
    340: (0.7732233047033636, 0.7732233047033633),
    7200: (0.2333333333333334, 0.75),
    7201: (0.2333333333333334, 1.0),
}
_ARC_EXACT_MIDPOINT_CHART = (0.7190301168722328, 0.8543291419085957)
_ARC_EDGE, _BLOCKED_T3, _IRREGULAR_Q4, _REGULAR_Q4 = 24, 6895, 6883, 6001
_CAVITY_RING = [13, 7136, 7130, 7131, 340]
_CHART_NODES = tuple(_CAVITY_CHART)


class _ArcEdgeGeometry:
    """Delegating geometry: source edge ``edge_id`` is the arc 340 -> 13.

    Only the chain-segment queries of the repair are needed: the station
    parameters (0 at 340, 1 at 13) and the exact parameter-midpoint.
    """

    def __init__(self, geometry, edge_id, start, end, midpoint):
        self._geometry = geometry
        self._edge = int(edge_id)
        self._start = np.asarray(start, dtype=float)
        self._end = np.asarray(end, dtype=float)
        self._midpoint = np.asarray(midpoint, dtype=float)

    def closest_edge_point(self, edge_id, point):
        if int(edge_id) != self._edge:
            return self._geometry.closest_edge_point(edge_id, point)
        point = np.asarray(point, dtype=float)
        for parameter, end in ((0.0, self._start), (1.0, self._end)):
            if np.allclose(point, end, rtol=0.0, atol=1.0e-12):
                return end.copy(), parameter, 0.0
        raise AssertionError("the repair queried a point that is not an arc station")

    def sample_edge(self, edge_id, parameters):
        if int(edge_id) != self._edge:
            return self._geometry.sample_edge(edge_id, parameters)
        t = np.asarray(parameters, dtype=float).reshape(-1)
        chord = self._start + t[:, None] * (self._end - self._start)
        return np.where(np.isclose(t, 0.5)[:, None], self._midpoint, chord)

    def __getattr__(self, name):
        return getattr(self._geometry, name)


def _cavity_case():
    from anymesher.mesh import Mesh

    geometry, face = _plate(1.0, 1.0)
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    xyz = {node: np.asarray(domain.lift(uv), dtype=float) for node, uv in _CAVITY_CHART.items()}
    midpoint = np.asarray(domain.lift(_ARC_EXACT_MIDPOINT_CHART), dtype=float)
    curved = _ArcEdgeGeometry(geometry, _ARC_EDGE, xyz[340], xyz[13], midpoint)
    mesh = Mesh(
        nodes=xyz,
        quads={_IRREGULAR_Q4: (13, 7136, 7130, 7131), _REGULAR_Q4: (7200, 7130, 7136, 7201)},
        tris={_BLOCKED_T3: (13, 7131, 340)},
        nodes_of_edge={_ARC_EDGE: [340, 13]},
        elements_of_face={face: [_IRREGULAR_Q4, _REGULAR_Q4, _BLOCKED_T3]},
    )
    return curved, face, mesh, {face: domain}, midpoint


def _t6_nodes(mesh, body, midpoint):
    """Three corners plus midsides: exact owner midside on arc 13-340, else chord."""
    sides = []
    for i in range(3):
        a, b = body[i], body[(i + 1) % 3]
        sides.append(midpoint if {a, b} == {13, 340} else 0.5 * (mesh.nodes[a] + mesh.nodes[b]))
    return np.asarray([mesh.nodes[n] for n in body] + sides, dtype=float)


def _t6_quality(mesh, domain, body, midpoint):
    nodes = _t6_nodes(mesh, body, midpoint)
    normal = np.asarray(domain.normal, dtype=float)
    samples = np.asarray([(i / 8, j / 8) for i in range(9) for j in range(9 - i)], dtype=float)
    sampled = hybrid.evaluate_mapping(nodes, "T6", samples, reference_normal=normal)
    status = hybrid.certify_mapping_validity(nodes, "T6", reference_normal=normal).status
    return float(np.min(sampled.normalized_quality)), status


def _min_corner_angle(mesh, body):
    points = [np.asarray(mesh.nodes[n], dtype=float) for n in body]
    angles = []
    for i in range(3):
        u, v = points[(i + 1) % 3] - points[i], points[(i + 2) % 3] - points[i]
        angles.append(float(np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))))
    return min(angles)


def test_cavity_repair_replaces_blocked_t3_and_irregular_q4_by_a_five_triangle_fan() -> None:
    """Face 248 of the original 15 m barge, h = 0.25: the T3 beside arc 24 is
    below the Jacobian floor; the T3 + adjacent non-regular Q4 form the simple
    pentagon [13, 7136, 7130, 7131, 340] that a vertex-mean centroid fans into
    five strictly certified T6.  No boundary node/station moves."""
    curved, face, mesh, domains, midpoint = _cavity_case()
    domain = domains[face]
    floor = hybrid._QUAD_FIRST_MIN_NORMALIZED_JACOBIAN
    before, _status = _t6_quality(mesh, domain, mesh.tris[_BLOCKED_T3], midpoint)
    assert before < floor, "the original patch must fail the unchanged floor"
    frozen = {n: np.array(mesh.nodes[n], dtype=float) for n in _CHART_NODES}
    regular_body = mesh.quads[_REGULAR_Q4]
    chain_before = list(mesh.nodes_of_edge[_ARC_EDGE])
    nodes_before = set(mesh.nodes)

    hybrid._repair_quad_first_quality(
        mesh, curved, domains,
        protected_regular_nodes=frozenset({7200, 7201, 7130, 7131, 7136}),
        regular_cells=frozenset({_REGULAR_Q4}),
    )

    guard = mesh.hybrid_diagnostics["quad_quality_repair"]["regular_footprint_guard"]
    assert guard["blocked_triangles"] == []
    assert "boundary_splits" not in guard
    (repair,) = guard["cavity_repairs"]
    assert repair["polygon"] == _CAVITY_RING
    assert (repair["triangle"], repair["quad"]) == (_BLOCKED_T3, _IRREGULAR_Q4)
    np.testing.assert_allclose(
        repair["centroid_chart"], (0.6213113276073395, 0.8446446609406728), rtol=0.0, atol=1.0e-12)
    # Complete replacement: the Q4 is gone, the regular rectangle is untouched.
    assert mesh.quads == {_REGULAR_Q4: regular_body}
    fan = [mesh.tris[cell] for cell in repair["cells"]]
    centre = repair["node"]
    assert len(fan) == 5 and len(set(repair["cells"])) == 5
    assert fan == [(centre, _CAVITY_RING[i], _CAVITY_RING[(i + 1) % 5]) for i in range(5)]
    assert set(mesh.tris) == set(repair["cells"])
    assert sorted(mesh.elements_of_face[face]) == sorted([_REGULAR_Q4, *repair["cells"]])
    # No boundary node, station or regular footprint moved; one new node only.
    assert set(mesh.nodes) == nodes_before | {centre}
    for node, position in frozen.items():
        np.testing.assert_array_equal(mesh.nodes[node], position)
    assert mesh.nodes_of_edge[_ARC_EDGE] == chain_before
    # Every new T6 (exact arc midside on 13-340, chords elsewhere) meets the
    # unchanged floor, strict certificate and 15 degree rule.
    qualities, angles = [], []
    for body in fan:
        value, status = _t6_quality(mesh, domain, body, midpoint)
        assert status is hybrid.ValidityStatus.CERTIFIED_POSITIVE
        assert value >= floor
        # The independent oracle used (boundary start, boundary end, centre).
        # The reference-coordinate Jacobian normalization depends on this
        # cyclic ordering; admission above checks the actual published order.
        qualities.append(_t6_quality(mesh, domain, body[1:] + body[:1], midpoint)[0])
        angles.append(_min_corner_angle(mesh, body))
    assert min(angles) >= 15.0
    np.testing.assert_allclose(
        angles, [33.0366, 44.8686, 31.4672, 25.0789, 42.3195], rtol=0.0, atol=2.0e-3)
    np.testing.assert_allclose(
        qualities, [0.90879, 0.70548, 0.52201, 0.92624, 0.35226], rtol=0.0, atol=2.0e-3)


def test_cavity_repair_stays_blocked_when_the_adjacent_quad_is_regular() -> None:
    """Fail closed: a regular neighbour is never consumed, so the original
    failing patch is reported and the mesh is left exactly as it was."""
    curved, face, mesh, domains, _midpoint = _cavity_case()
    tris_before, quads_before = dict(mesh.tris), dict(mesh.quads)
    nodes_before = {n: np.array(p, dtype=float) for n, p in mesh.nodes.items()}
    with pytest.raises(QuadQualityRejected, match="smart mapped repair is blocked"):
        hybrid._repair_quad_first_quality(
            mesh, curved, domains,
            protected_regular_nodes=frozenset({7200, 7201, 7130, 7131, 7136}),
            regular_cells=frozenset({_REGULAR_Q4, _IRREGULAR_Q4}),
        )
    assert dict(mesh.tris) == tris_before and dict(mesh.quads) == quads_before
    assert set(mesh.nodes) == set(nodes_before)
    for node, position in nodes_before.items():
        np.testing.assert_array_equal(mesh.nodes[node], position)
    guard = mesh.hybrid_diagnostics["quad_quality_repair"]["regular_footprint_guard"]
    assert guard["cavity_repairs"] == []
    assert [item[1] for item in guard["blocked_triangles"]] == [_BLOCKED_T3]


def test_linear_beam_pass_becomes_straight_sided_b3_with_chord_midsides() -> None:
    from anymesher.hybrid import CertificationMode, HybridMeshResult
    from anymesher.mesh import Mesh

    def result(mesh):
        return HybridMeshResult(
            mesh=mesh, strategy_by_face={}, triangulation_backend_by_face={},
            preflight=(), connectivity=None, audit_report=None,
            certification_mode=CertificationMode.NONE, certifiable=True)

    legacy = Mesh(
        nodes={1: np.array([0.0, 0.0, 0.0]), 2: np.array([1.0, 0.05, 0.0]),
               3: np.array([2.0, 0.0, 0.0])},
        beams={10: (1, 2), 11: (2, 3)}, nodes_of_edge={5: [1, 2, 3]},
        elements_of_edge={5: [10, 11]},
    )
    # Immutable reports/metadata cannot be deep-copied (mappingproxy); the
    # promotion must share them unchanged instead of failing or dropping them.
    report = types.MappingProxyType({"report": types.MappingProxyType({"kept": 1})})
    legacy.hybrid_diagnostics = report
    shell = Mesh(nodes={
        11: np.array([0.0, 0.0, 0.0]), 12: np.array([0.5, 0.1, 0.0]),
        13: np.array([1.0, 0.05, 0.0]), 14: np.array([1.5, 0.1, 0.0]),
        15: np.array([2.0, 0.0, 0.0])}, nodes_of_edge={5: [11, 12, 13, 14, 15]})
    promoted = hybrid._promote_linear_beam_result_to_b3(result(legacy), frozenset({5}), shell)
    mesh = promoted.mesh
    assert legacy.beams[10] == (1, 2) and not legacy.is_quadratic  # input untouched
    assert mesh.is_quadratic and set(len(body) for body in mesh.beams.values()) == {3}
    assert mesh.hybrid_diagnostics is report  # metadata preserved, not copied or lost
    # No station is relocated: the beam axis keeps its (matching) stations;
    # midsides are chords of the beam end nodes.
    np.testing.assert_array_equal(mesh.nodes[2], shell.nodes[13])
    assert set(legacy.nodes) == {1, 2, 3} and legacy.nodes_of_edge[5] == [1, 2, 3]
    assert legacy.offset_nodes_of_edge == {} and legacy.beams == {10: (1, 2), 11: (2, 3)}
    first, middle, second = mesh.beams[10]
    np.testing.assert_allclose(mesh.nodes[middle], 0.5 * (mesh.nodes[first] + mesh.nodes[second]))
    assert float(np.linalg.norm(mesh.nodes[middle] - shell.nodes[12])) > 1.0e-3  # chord != arc
    assert mesh.offset_nodes_of_edge[5] == mesh.nodes_of_edge[5] and len(mesh.nodes_of_edge[5]) == 5
