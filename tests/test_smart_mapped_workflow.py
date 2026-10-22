"""Private smart-mapped seed adapter: layout, refusals, restoration, integration.

Not executed by the author (no-commands brief); the parent runs this file.
Tests marked ``quad_workers`` need the compiled quad workers, like the existing
explicit quad-first public tests.
"""
from __future__ import annotations

import math
import os
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

# Resolve ``anymesher`` to this worktree's sources, as the quad-first suites do.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and (not sys.path or sys.path[0] != _SRC):
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import inspect  # noqa: E402

from anygeometry.curves import Arc  # noqa: E402
from anygeometry.entities import OrientedEdge  # noqa: E402
from anygeometry.model import GeometryModel  # noqa: E402
from anymesher import hybrid  # noqa: E402
from anymesher._smart_mapped_prototype import build_smart_mapped  # noqa: E402
from anymesher._smart_mapped_workflow import (  # noqa: E402
    ADAPTER_ID,
    SmartMappedAdapterError,
    SmartMappedRefusal,
    SmartMappedSeedContext,
    _assert_simple_loops,
    _assert_stations_in_plane,
    _layout_from_prototype,
    build_smart_mapped_planar_seed,
)
from anymesher.errors import MeshError  # noqa: E402
from anymesher.quad import seed as seed_module  # noqa: E402
from anymesher.quad.boundary import BoundaryStationRegistry  # noqa: E402
from anymesher.quad.domain import PlanarQuadDomain  # noqa: E402
from anymesher.quad.options import QuadMeshingOptions  # noqa: E402
from anymesher.quad.seed import PlanarQuadSeed, build_planar_quad_seed  # noqa: E402

ORIGINAL = seed_module.build_planar_quad_seed


def _plate(width: float = 6.0, height: float = 5.0):
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, height, 0.0), (0.0, height, 0.0))
    )
    return geometry, geometry.add_plate(vertices)


def _domain_and_registry(geometry, face, h=1.0):
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    return domain, BoundaryStationRegistry.for_domain(geometry, domain, h)


def _cell_area(state) -> float:
    total = 0.0
    for cid in state.cells:
        points = [state.position(n) for n in state.cell(cid)]
        total += 0.5 * sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
    return total


@pytest.fixture(autouse=True)
def _hybrid_is_pristine():
    assert hybrid.build_planar_quad_seed is ORIGINAL
    yield
    assert hybrid.build_planar_quad_seed is ORIGINAL


# --------------------------------------------------------------------------
# Pure layout: exact boundary and holes, invariants, front policy
# --------------------------------------------------------------------------

OUTER = ((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0))
HOLE = ((3.2, 3.2), (4.8, 3.2), (4.8, 4.8), (3.2, 4.8))


def test_layout_keeps_exact_outer_and_hole_rows_area_and_protection() -> None:
    prototype = build_smart_mapped(OUTER, (HOLE,), (), target_size=1.0)
    triangulation, state, order = _layout_from_prototype(prototype, OUTER, (HOLE,))
    n_outer, n_hole = len(OUTER), len(HOLE)
    assert np.array_equal(triangulation.points[:n_outer], np.asarray(OUTER))
    assert np.array_equal(triangulation.points[n_outer:n_outer + n_hole], np.asarray(HOLE))
    assert list(triangulation.outer_loop) == list(range(n_outer))
    assert [list(ring) for ring in triangulation.hole_loops] == [
        list(range(n_outer, n_outer + n_hole))
    ]
    assert sorted(order) == list(range(len(prototype.points)))
    assert state.count_kind("Q4") == len(prototype.quads) > 0
    assert state.count_kind("T3") == len(prototype.triangles)
    assert _cell_area(state) == pytest.approx(64.0 - 1.6 * 1.6, rel=1e-12)
    # Coverage triangulation (Q4 split in two) covers the same material.
    assert len(triangulation.triangles) == len(prototype.triangles) + 2 * len(prototype.quads)
    boundary = {tuple(sorted(map(int, e))) for e in triangulation.boundary_segments}
    assert len(boundary) == n_outer + n_hole
    assert state.protected_nodes == frozenset(range(n_outer + n_hole))
    assert state.protected_edges == frozenset(boundary)
    assert len(triangulation.mandatory_segments) == 0
    assert triangulation.actual_backend == "smart-mapped-prototype"


def test_layout_interface_front_is_boundary_of_residual_t3_region() -> None:
    prototype = build_smart_mapped(OUTER, (HOLE,), (), target_size=1.0)
    _, state, _ = _layout_from_prototype(prototype, OUTER, (HOLE,), initial_front="interface")
    assert state.front
    for edge in state.front:
        owners = state.edge_cells(edge)
        t3 = [c for c in owners if state.cell_kind(c) == "T3"]
        assert len(t3) == 1
        assert all(state.cell_kind(c) == "Q4" for c in owners if c != t3[0])
    for edge in state.edges - state.front:
        owners = state.edge_cells(edge)
        assert not (len(owners) == 1 and state.cell_kind(owners[0]) == "T3")
    _, empty, _ = _layout_from_prototype(prototype, OUTER, (HOLE,), initial_front="none")
    assert not empty.front
    with pytest.raises(SmartMappedRefusal):
        _layout_from_prototype(prototype, OUTER, (HOLE,), initial_front="everything")


def test_layout_refuses_changed_boundary_station_or_loop_shape() -> None:
    prototype = build_smart_mapped(OUTER, (HOLE,), (), target_size=1.0)
    moved = np.array(prototype.points, copy=True)
    moved[int(prototype.outer_loop[0])] += 1.0e-9
    with pytest.raises(SmartMappedRefusal, match="changed a canonical boundary station"):
        _layout_from_prototype(replace(prototype, points=moved), OUTER, (HOLE,))
    with pytest.raises(SmartMappedRefusal, match="differ from the canonical"):
        _layout_from_prototype(prototype, OUTER, ())
    with pytest.raises(SmartMappedRefusal, match="differ from the canonical"):
        _layout_from_prototype(prototype, OUTER[:3], (HOLE,))


# --------------------------------------------------------------------------
# Canonical registry contract against the baseline seed
# --------------------------------------------------------------------------

def test_seed_matches_baseline_station_registry_and_boundary() -> None:
    geometry, face = _plate()
    domain, registry = _domain_and_registry(geometry, face)
    baseline = build_planar_quad_seed(geometry, face, 1.0, domain=domain, registry=registry)
    seed, prototype, diagnostics = build_smart_mapped_planar_seed(
        geometry, face, 1.0, domain=domain, registry=registry)
    assert isinstance(seed, PlanarQuadSeed)
    assert seed.registry is registry and seed.domain is domain
    assert dict(seed.station_to_node) == dict(baseline.station_to_node)
    assert list(seed.station_to_node) == list(baseline.station_to_node)
    assert seed.boundary_keys == baseline.boundary_keys
    n = len(seed.station_to_node)
    assert np.array_equal(seed.triangulation.points[:n], baseline.triangulation.points[:n])
    assert ({tuple(sorted(map(int, e))) for e in seed.triangulation.boundary_segments}
            == {tuple(sorted(map(int, e))) for e in baseline.triangulation.boundary_segments})
    assert seed.state.protected_nodes == baseline.state.protected_nodes
    assert seed.state.protected_edges == baseline.state.protected_edges
    assert seed.discrete_area == pytest.approx(baseline.discrete_area, rel=1e-12)
    assert seed.discrete_area == pytest.approx(30.0, rel=1e-12)
    assert _cell_area(seed.state) == pytest.approx(30.0, rel=1e-12)
    assert seed.state.count_kind("Q4") == len(prototype.quads) > 0
    assert diagnostics["seed_quads"] == len(prototype.quads)
    assert diagnostics["prototype_diagnostics"]["prototype"] is True
    assert "retained_grid_indices" not in diagnostics["prototype_diagnostics"]


def test_seed_geometry_and_registry_are_not_mutated() -> None:
    geometry, face = _plate()
    revision, model_id = geometry.revision, geometry.model_id
    domain, registry = _domain_and_registry(geometry, face)
    build_smart_mapped_planar_seed(geometry, face, 1.0, domain=domain, registry=registry)
    assert (geometry.revision, geometry.model_id) == (revision, model_id)
    domain.assert_current(geometry)
    registry.assert_current(geometry)


# --------------------------------------------------------------------------
# Explicit refusals; the prototype is never reached
# --------------------------------------------------------------------------

def _must_not_run(*args, **kwargs):  # pragma: no cover - failure path
    raise AssertionError("prototype builder must not be reached for a refused request")


class _NonUniformField:
    is_uniform = False
    target_size = 1.0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"source_aligned_cut": (0.5, 0.5)}, "source-aligned"),
        ({"size_field": _NonUniformField()}, "uniform"),
        ({"domain": object()}, "PlanarQuadDomain"),
        ({"layout_policy": "other"}, "layout_policy"),
        ({"prototype_options": {"cancellation_check": None}}, "unknown smart mapped"),
        ({"prototype_options": {"bogus": 1}}, "unknown smart mapped"),
        ({"initial_front": "all"}, "initial_front"),
    ],
)
def test_unsupported_requests_are_refused_not_fallen_back(kwargs, match) -> None:
    geometry, face = _plate()
    with pytest.raises(SmartMappedRefusal, match=match):
        build_smart_mapped_planar_seed(
            geometry, face, 1.0, prototype_builder=_must_not_run, **kwargs)


def test_refusal_is_a_mesh_error_but_not_a_public_unsupported_fallback_signal() -> None:
    from anymesher.quad.public_integration import QuadPublicUnsupported

    assert issubclass(SmartMappedRefusal, MeshError)
    assert issubclass(SmartMappedAdapterError, MeshError)
    assert not issubclass(SmartMappedRefusal, QuadPublicUnsupported)


def test_nonpositive_or_nonfinite_target_size_refused() -> None:
    geometry, face = _plate()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(SmartMappedRefusal):
            build_smart_mapped_planar_seed(geometry, face, bad, prototype_builder=_must_not_run)


def test_context_rejects_bad_constructor_arguments() -> None:
    with pytest.raises(SmartMappedRefusal):
        SmartMappedSeedContext(prototype_options={"nope": 1})
    with pytest.raises(SmartMappedRefusal):
        SmartMappedSeedContext(initial_front="x")
    with pytest.raises(SmartMappedRefusal):
        SmartMappedSeedContext(seed_callback=3)
    with pytest.raises(SmartMappedRefusal):
        SmartMappedSeedContext(require_calls=1)


# --------------------------------------------------------------------------
# Context: patching, restoration, nesting, concurrency, truthful failure
# --------------------------------------------------------------------------

def test_context_patches_only_hybrid_seed_name_and_restores() -> None:
    with SmartMappedSeedContext(require_calls=False) as adapter:
        assert adapter.active
        assert hybrid.build_planar_quad_seed is not ORIGINAL
        assert seed_module.build_planar_quad_seed is ORIGINAL
    assert not adapter.active and adapter.restored
    assert hybrid.build_planar_quad_seed is ORIGINAL


def test_context_restores_after_body_exception_and_instance_is_single_use() -> None:
    adapter = SmartMappedSeedContext(require_calls=False)
    with pytest.raises(RuntimeError, match="boom"):
        with adapter:
            raise RuntimeError("boom")
    assert adapter.restored and hybrid.build_planar_quad_seed is ORIGINAL
    with pytest.raises(SmartMappedAdapterError, match="single-use"):
        adapter.__enter__()
    assert hybrid.build_planar_quad_seed is ORIGINAL
    with SmartMappedSeedContext(require_calls=False):  # lock was released
        pass


def test_nesting_is_refused_and_outer_context_survives() -> None:
    with SmartMappedSeedContext(require_calls=False) as outer:
        wrapper = hybrid.build_planar_quad_seed
        with pytest.raises(SmartMappedAdapterError, match="already active"):
            with SmartMappedSeedContext(require_calls=False):
                pass  # pragma: no cover
        assert outer.active and hybrid.build_planar_quad_seed is wrapper
    assert hybrid.build_planar_quad_seed is ORIGINAL


def test_second_thread_cannot_enter_while_active() -> None:
    errors: list[BaseException] = []

    def attempt() -> None:
        try:
            with SmartMappedSeedContext(require_calls=False):
                pass  # pragma: no cover
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    with SmartMappedSeedContext(require_calls=False):
        worker = threading.Thread(target=attempt)
        worker.start()
        worker.join(30)
        assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], SmartMappedAdapterError)


def test_foreign_thread_seed_call_is_refused_and_silent_success_is_not_allowed() -> None:
    geometry, face = _plate()
    errors: list[BaseException] = []
    adapter = SmartMappedSeedContext(require_calls=False)
    with pytest.raises(SmartMappedAdapterError, match="silent fallback"):
        with adapter:
            def foreign() -> None:
                try:
                    hybrid.build_planar_quad_seed(geometry, face, 1.0)
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)

            worker = threading.Thread(target=foreign)
            worker.start()
            worker.join(30)
    assert len(errors) == 1 and isinstance(errors[0], SmartMappedAdapterError)
    assert adapter.calls == 0 and len(adapter.failures) == 1
    assert hybrid.build_planar_quad_seed is ORIGINAL


def test_prepatched_hybrid_seed_is_refused_without_taking_the_lock(monkeypatch) -> None:
    monkeypatch.setattr(hybrid, "build_planar_quad_seed", lambda *a, **k: None)
    with pytest.raises(SmartMappedAdapterError, match="already replaced"):
        SmartMappedSeedContext(require_calls=False).__enter__()
    monkeypatch.setattr(hybrid, "build_planar_quad_seed", ORIGINAL)
    with SmartMappedSeedContext(require_calls=False):  # lock was released
        pass


def test_external_modification_during_context_is_detected_and_original_restored() -> None:
    adapter = SmartMappedSeedContext(require_calls=False)
    with pytest.raises(SmartMappedAdapterError, match="modified while"):
        with adapter:
            hybrid.build_planar_quad_seed = lambda *a, **k: None
    assert hybrid.build_planar_quad_seed is ORIGINAL and adapter.restored


def test_never_calling_the_seed_is_reported_not_claimed() -> None:
    with pytest.raises(SmartMappedAdapterError, match="never called"):
        with SmartMappedSeedContext():
            pass
    assert hybrid.build_planar_quad_seed is ORIGINAL


def test_swallowed_refusal_inside_context_fails_at_exit() -> None:
    geometry, face = _plate()
    adapter = SmartMappedSeedContext(require_calls=False)
    with pytest.raises(SmartMappedAdapterError, match="silent fallback"):
        with adapter:
            with pytest.raises(SmartMappedRefusal, match="source-aligned"):
                hybrid.build_planar_quad_seed(
                    geometry, face, 1.0, source_aligned_cut=(0.5, 0.5))
    assert adapter.calls == 0
    assert adapter.failures[0]["type"] == "SmartMappedRefusal"
    assert hybrid.build_planar_quad_seed is ORIGINAL


def test_direct_call_counts_faces_callback_and_truthful_report() -> None:
    geometry, face = _plate()
    domain, registry = _domain_and_registry(geometry, face)
    received = []
    with SmartMappedSeedContext(seed_callback=received.append) as adapter:
        seed = hybrid.build_planar_quad_seed(
            geometry, face, 1.0, domain=domain, registry=registry,
            size_field=None, source_aligned_cut=None, layout_policy="existing")
    assert isinstance(seed, PlanarQuadSeed)
    assert adapter.calls == 1 and adapter.faces == (face,)
    (record,) = received
    assert record is adapter.records[0]
    assert record.seed is seed and record.face_id == face
    assert record.seed_state_digest == seed.state.digest()
    assert len(record.seed_quads) == seed.state.count_kind("Q4")
    assert len(record.seed_triangles) == seed.state.count_kind("T3")
    assert not record.seed_points.flags.writeable and not record.seed_quads.flags.writeable
    # Snapshots survive later in-place edits to the resident state.
    tx = seed.state.transaction()
    movable = next(n for n in seed.state.node_ids if not seed.state.is_protected_node(n))
    x, y = seed.state.position(movable)
    tx.move_node(movable, (x + 1.0e-3, y))
    tx.commit()
    assert record.seed_state_digest != seed.state.digest()
    assert tuple(record.seed_points[movable]) == (x, y)
    report = adapter.report()
    assert report["adapter"] == ADAPTER_ID
    assert report["prototype_function"] == "build_smart_mapped"
    assert report["replaced"] == "anymesher.hybrid.build_planar_quad_seed"
    assert report["calls"] == 1 and report["faces"] == [face]
    assert report["restored"] is True and report["failures"] == []
    assert report["pure_prototype_claim"] is False
    assert report["downstream_may_modify_seed"] is True
    assert report["per_face"][0]["prototype_diagnostics"]["prototype"] is True


def test_callback_failure_is_a_failure_not_ignored() -> None:
    geometry, face = _plate()
    domain, registry = _domain_and_registry(geometry, face)

    def audit(record) -> None:
        raise MeshError("parent audit rejected the seed")

    adapter = SmartMappedSeedContext(seed_callback=audit)
    with pytest.raises(MeshError, match="parent audit rejected"):
        with adapter:
            hybrid.build_planar_quad_seed(geometry, face, 1.0, domain=domain, registry=registry)
    assert hybrid.build_planar_quad_seed is ORIGINAL
    assert adapter.failures[0]["message"] == "parent audit rejected the seed"


def test_prototype_cancellation_propagates_and_restores() -> None:
    geometry, face = _plate()
    domain, registry = _domain_and_registry(geometry, face)

    def cancel(phase: str) -> None:
        if phase.startswith("smart-mapped:"):
            raise MeshError(f"cancelled at {phase}")

    adapter = SmartMappedSeedContext()
    with pytest.raises(MeshError, match="cancelled at smart-mapped:start"):
        with adapter:
            hybrid.build_planar_quad_seed(
                geometry, face, 1.0, domain=domain, registry=registry,
                cancellation_check=cancel)
    assert hybrid.build_planar_quad_seed is ORIGINAL and adapter.restored
    assert adapter.calls == 0


# --------------------------------------------------------------------------
# Real downstream workflow (compiled quad workers required)
# --------------------------------------------------------------------------

@pytest.mark.quad_workers
def test_quad_first_cancellation_restores_through_real_entry_point() -> None:
    geometry, face = _plate()

    def cancel(phase: str) -> None:
        if phase == "quad-first:face-seed":
            raise MeshError("cancelled by test")

    adapter = SmartMappedSeedContext()
    with pytest.raises(MeshError, match="cancelled by test"):
        with adapter:
            hybrid.generate_hybrid_mesh_result(
                geometry, target_size=1.0, face_ids=(face,),
                quad_options=QuadMeshingOptions(), cancellation_check=cancel)
    assert adapter.calls == 0 and adapter.restored
    assert hybrid.build_planar_quad_seed is ORIGINAL


@pytest.mark.quad_workers
@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_real_downstream_validation_and_publication_use_prototype_seed(
    monkeypatch, order,
) -> None:
    geometry, face = _plate()
    validated = []
    real_validate = hybrid.validate_planar_quad_result

    def counting_validate(state, **kwargs):
        validated.append(kwargs.get("seed"))
        return real_validate(state, **kwargs)

    monkeypatch.setattr(hybrid, "validate_planar_quad_result", counting_validate)
    records = []
    with SmartMappedSeedContext(seed_callback=records.append) as adapter:
        result = hybrid.generate_hybrid_mesh_result(
            geometry, target_size=1.0, face_ids=(face,),
            quad_options=QuadMeshingOptions(), order=order)
        report = adapter.annotate(result.mesh)
    assert hybrid.build_planar_quad_seed is ORIGINAL
    assert adapter.calls == 1 and adapter.faces == (face,)
    assert validated and validated[0] is records[0].seed  # real validator saw our seed
    mesh = result.mesh
    assert mesh.hybrid_diagnostics["route"] == "quad-first"
    stored = mesh.hybrid_diagnostics["smart_mapped_seed_adapter"]
    assert stored == report and stored is not report
    assert report["context_active"] is True and report["restored"] is False
    final = adapter.annotate(mesh)  # after exit: records the restoration
    assert final["context_active"] is False and final["restored"] is True
    assert mesh.hybrid_diagnostics["smart_mapped_seed_adapter"] == final
    assert report["pure_prototype_claim"] is False
    assert report["downstream_may_modify_seed"] is True
    assert report["calls"] == 1 and report["failures"] == []
    assert mesh.order == order
    assert len(records[0].seed_quads) > 0
    area = 0.0
    for body in (*mesh.quads.values(), *mesh.tris.values()):
        corners = [mesh.nodes[n] for n in body[:4 if len(body) in (4, 8, 9) else 3]]
        area += 0.5 * sum(
            corners[i][0] * corners[(i + 1) % len(corners)][1]
            - corners[(i + 1) % len(corners)][0] * corners[i][1]
            for i in range(len(corners))
        )
    assert area == pytest.approx(30.0, rel=1e-9)
    # Canonical boundary chains are the registry's, shared by node id.
    for chain in mesh.nodes_of_edge.values():
        assert len(chain) == len(set(chain)) >= 2


# --------------------------------------------------------------------------
# Planar face with a rounded-rectangle (Arc-edged) access hole
# --------------------------------------------------------------------------

def _shoelace(points) -> float:
    pts = np.asarray(points, dtype=float)
    return 0.5 * abs(float(np.dot(pts[:, 0], np.roll(pts[:, 1], -1))
                           - np.dot(pts[:, 1], np.roll(pts[:, 0], -1))))


def _arc_hole_plate(*, bad_mid_z: float = 0.0, width: float = 8.0, height: float = 6.0,
                    cx: float = 4.0, cy: float = 3.0, a: float = 1.5, b: float = 1.0,
                    r: float = 0.5):
    """Rectangle plate with a CCW rounded-rectangle hole (4 lines + 4 Arc edges).

    Returns ``(geometry, face, arcs)`` with ``arcs = {edge_id: (cx, cy, theta0)}``
    for the quarter-circle Arc edges.  ``bad_mid_z`` lifts the first arc's mid
    vertex out of the face plane.
    """
    geometry = GeometryModel()
    ov = geometry.add_points(
        ((0.0, 0.0, 0.0), (width, 0.0, 0.0), (width, height, 0.0), (0.0, height, 0.0)))
    outer = tuple(
        OrientedEdge(geometry.add_line(ov[i], ov[(i + 1) % 4]), True) for i in range(4))
    corners = (
        (cx + a - r, cy - b + r, -math.pi / 2.0),
        (cx + a - r, cy + b - r, 0.0),
        (cx - a + r, cy + b - r, math.pi / 2.0),
        (cx - a + r, cy - b + r, math.pi),
    )
    points = []
    for index, (x0, y0, t0) in enumerate(corners):
        for k, angle in enumerate((t0, t0 + math.pi / 4.0, t0 + math.pi / 2.0)):
            z = bad_mid_z if (index == 0 and k == 1) else 0.0
            points.append((x0 + r * math.cos(angle), y0 + r * math.sin(angle), z))
    ids = geometry.add_points(tuple(points))
    arcs: dict[int, tuple[float, float, float]] = {}
    hole = []
    for index, (x0, y0, t0) in enumerate(corners):
        start, mid, end = ids[3 * index:3 * index + 3]
        arc_edge = geometry.add_arc(start, mid, end)
        arcs[int(arc_edge)] = (x0, y0, t0)
        hole.append(OrientedEdge(arc_edge, True))
        next_start = ids[3 * ((index + 1) % 4)]
        hole.append(OrientedEdge(geometry.add_line(end, next_start), True))
    from anygeometry.operations import trim_face
    face = geometry.add_face_from_loop(outer, (0, 1, 2, 3))
    trim_face(geometry, face, (tuple(hole),))
    for edge_id in arcs:
        assert type(geometry.edges[edge_id].curve) is Arc
    return geometry, face, arcs


def _station_xy(domain, stations):
    return np.asarray([domain.project(s.position) for s in stations], dtype=float)


def test_arc_hole_seed_adopts_arc_edges_with_exact_registry_stations() -> None:
    geometry, face, arcs = _arc_hole_plate()
    domain, registry = _domain_and_registry(geometry, face, 0.4)
    assert any(registry.edge_divisions(edge_id) >= 2 for edge_id in arcs)
    baseline = build_planar_quad_seed(geometry, face, 0.4, domain=domain, registry=registry)
    seed, prototype, diagnostics = build_smart_mapped_planar_seed(
        geometry, face, 0.4, domain=domain, registry=registry)
    assert isinstance(seed, PlanarQuadSeed) and seed.registry is registry
    assert list(seed.station_to_node) == list(baseline.station_to_node)
    assert dict(seed.station_to_node) == dict(baseline.station_to_node)
    outer = _station_xy(domain, registry.outer_stations(domain))
    hole = _station_xy(domain, registry.hole_stations(domain, 0))
    n_outer, n_hole = len(outer), len(hole)
    assert n_hole > 8  # arcs carry interior stations
    points = np.asarray(seed.triangulation.points, dtype=float)
    assert np.array_equal(points[:n_outer], outer)
    assert np.array_equal(points[n_outer:n_outer + n_hole], hole)
    assert seed.state.protected_nodes == baseline.state.protected_nodes
    assert seed.state.protected_edges == baseline.state.protected_edges
    # Hole stations of every Arc edge lie on the true source circle.
    for edge_id, (x0, y0, _t0) in arcs.items():
        for station in registry.chain(edge_id):
            x, y = domain.project(station.position)
            assert math.hypot(x - x0, y - y0) == pytest.approx(0.5, abs=1e-12)
    expected_area = _shoelace(outer) - _shoelace(hole)
    assert seed.discrete_area == pytest.approx(expected_area, rel=1e-12)
    assert _cell_area(seed.state) == pytest.approx(expected_area, rel=1e-12)
    assert seed.state.count_kind("Q4") == len(prototype.quads) > 0
    assert diagnostics["boundary_stations"] == n_outer + n_hole


def test_unsupported_source_edge_type_is_still_refused_before_the_prototype() -> None:
    geometry, face = _plate()
    domain = PlanarQuadDomain.from_geometry(geometry, face)

    class OddCurve:
        pass

    class Proxy:
        def __init__(self, base):
            self._base = base
            self.edges = {key: SimpleNamespace(curve=OddCurve()) for key in base.edges}

        def __getattr__(self, name):
            return getattr(self._base, name)

    with pytest.raises(SmartMappedRefusal, match=r"Straight and Arc.*OddCurve"):
        build_smart_mapped_planar_seed(
            Proxy(geometry), face, 1.0, domain=domain, prototype_builder=_must_not_run)


def test_out_of_plane_arc_is_refused_not_flattened() -> None:
    geometry, face, _arcs = _arc_hole_plate(bad_mid_z=0.3)
    with pytest.raises(SmartMappedRefusal, match="not in the face plane"):
        build_smart_mapped_planar_seed(geometry, face, 0.4, prototype_builder=_must_not_run)


def test_station_plane_and_loop_simplicity_checks() -> None:
    geometry, face = _plate()
    domain = PlanarQuadDomain.from_geometry(geometry, face)
    _assert_stations_in_plane(domain, [(1.0, 1.0, 0.0), (5.0, 4.0, 1.0e-12)])
    with pytest.raises(SmartMappedRefusal, match="not in the face plane"):
        _assert_stations_in_plane(domain, [(1.0, 1.0, 0.01)])
    square = ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))
    inside = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0))
    crossing = ((3.0, 3.0), (5.0, 3.0), (5.0, 5.0), (3.0, 5.0))
    touching = ((0.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
    overlapping_holes = ((1.5, 1.5), (3.0, 1.5), (3.0, 3.0), (1.5, 3.0))
    _assert_simple_loops((square, inside))
    for bad in (crossing, touching):
        with pytest.raises(SmartMappedRefusal, match="touch or cross"):
            _assert_simple_loops((square, bad))
    with pytest.raises(SmartMappedRefusal, match="touch or cross"):
        _assert_simple_loops((square, inside, overlapping_holes))
    bowtie = ((0.0, 0.0), (2.0, 2.0), (2.0, 0.0), (0.0, 2.0))
    with pytest.raises(SmartMappedRefusal, match="touch or cross"):
        _assert_simple_loops((bowtie,))


def test_annotate_is_truthful_independent_and_refuses_false_claims() -> None:
    geometry, face = _plate()
    domain, registry = _domain_and_registry(geometry, face)

    def mesh(route="quad-first", faces=(face,)):
        return SimpleNamespace(
            hybrid_diagnostics={"route": route} if route else {},
            elements_of_face={f: [1] for f in faces},
        )

    never = SmartMappedSeedContext(require_calls=False)
    with pytest.raises(SmartMappedAdapterError, match="never entered"):
        never.annotate(mesh())
    with SmartMappedSeedContext(require_calls=False) as empty:
        with pytest.raises(SmartMappedAdapterError, match="never called"):
            empty.annotate(mesh())
    adapter = SmartMappedSeedContext()
    with adapter:
        hybrid.build_planar_quad_seed(geometry, face, 1.0, domain=domain, registry=registry)
        with pytest.raises(SmartMappedAdapterError, match="quad-first"):
            adapter.annotate(mesh(route="mapped"))
        with pytest.raises(SmartMappedAdapterError, match="quad-first"):
            adapter.annotate(mesh(route=None))
        with pytest.raises(SmartMappedAdapterError, match="every face"):
            adapter.annotate(mesh(faces=(face + 100,)))
        with pytest.raises(SmartMappedAdapterError, match="hybrid_diagnostics"):
            adapter.annotate(SimpleNamespace())
        target = mesh()
        inside = adapter.annotate(target)
    stored = target.hybrid_diagnostics["smart_mapped_seed_adapter"]
    assert inside["context_active"] is True and inside["restored"] is False
    assert stored == inside and stored is not inside
    inside["calls"] = 99  # caller edits never reach the stored diagnostics
    assert stored["calls"] == 1
    outside = adapter.annotate(target)
    assert outside["context_active"] is False and outside["restored"] is True
    assert target.hybrid_diagnostics["smart_mapped_seed_adapter"] == outside
    assert outside["supported_source_edges"] == ["Straight", "Arc"]
    assert outside["pure_prototype_claim"] is False

    failing = SmartMappedSeedContext(require_calls=False)
    with pytest.raises(SmartMappedAdapterError, match="silent fallback"):
        with failing:
            with pytest.raises(SmartMappedRefusal):
                hybrid.build_planar_quad_seed(
                    geometry, face, 1.0, source_aligned_cut=(0.5, 0.5))
    with pytest.raises(SmartMappedAdapterError, match="failed"):
        failing.annotate(mesh())
    assert hybrid.build_planar_quad_seed is ORIGINAL


@pytest.mark.quad_workers
def test_real_downstream_linear_and_quadratic_arc_hole_nodes_lie_on_source_arcs() -> None:
    meshes = {}
    for order in ("linear", "quadratic"):
        geometry, face, arcs = _arc_hole_plate()
        records = []
        with SmartMappedSeedContext(seed_callback=records.append) as adapter:
            result = hybrid.generate_hybrid_mesh_result(
                geometry, target_size=0.4, face_ids=(face,),
                # Historical private-prototype curve-projection profile: keep the
                # original layout (adaptive is the public quad-first default).
                quad_options=QuadMeshingOptions(), order=order,
                layout_policy="existing")
        assert hybrid.build_planar_quad_seed is ORIGINAL and adapter.restored
        assert adapter.calls == 1 and adapter.failures == ()
        assert len(records[0].seed_quads) > 0
        mesh = result.mesh
        assert mesh.order == order and mesh.hybrid_diagnostics["route"] == "quad-first"
        meshes[order] = mesh
        report = adapter.annotate(mesh)
        assert report["restored"] is True and report["pure_prototype_claim"] is False
        # Corner-polygon area of the published cells vs the exact rounded-hole
        # plate area (chord error of the sampled arcs is well below 2e-3).
        area = 0.0
        for body in (*mesh.quads.values(), *mesh.tris.values()):
            corners = [mesh.nodes[n] for n in body[:4 if len(body) in (4, 8, 9) else 3]]
            area += 0.5 * sum(
                corners[i][0] * corners[(i + 1) % len(corners)][1]
                - corners[(i + 1) % len(corners)][0] * corners[i][1]
                for i in range(len(corners))
            )
        exact = 48.0 - (6.0 - (4.0 - math.pi) * 0.25)
        assert area == pytest.approx(exact, rel=2e-3)
        # Every mesh node in an Arc edge's sector is on the true circle; the
        # count is the stations (linear) or stations plus owner midpoints.
        for edge_id, (x0, y0, t0) in arcs.items():
            if order == "linear":
                divisions = len(meshes["linear"].nodes_of_edge[edge_id]) - 1
                assert divisions >= 2
            else:
                divisions = len(meshes["linear"].nodes_of_edge[edge_id]) - 1
            on_arc = set()
            for node_id, position in mesh.nodes.items():
                dx, dy = float(position[0]) - x0, float(position[1]) - y0
                if abs(math.hypot(dx, dy) - 0.5) > 1.0e-9:
                    continue
                turn = (math.atan2(dy, dx) - t0) % (2.0 * math.pi)
                if turn <= math.pi / 2.0 + 1.0e-9 or turn >= 2.0 * math.pi - 1.0e-9:
                    on_arc.add(node_id)
            expected = divisions + 1 if order == "linear" else 2 * divisions + 1
            assert len(on_arc) == expected, (order, edge_id)
            assert on_arc >= set(mesh.nodes_of_edge[edge_id])

