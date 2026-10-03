"""Provisional component changes stay detached until a validated return."""

import numpy as np
import pytest

import anygeometry as owner
from anymesher._authored_component_stage import (
    AuthoredComponentPublication, AuthoredComponentStage,
)
from anymesher._component_reservations import ComponentNodeReservationPool
from anymesher.boundary import GlobalEdgeBoundaryRegistry
from anymesher.errors import MeshError
from anymesher.mesh import Mesh
from anymesher.meshing_view import GeometryMeshingView
from anymesher.native_v2 import ComponentSeedRegistry


def inputs():
    model = owner.GeometryModel()
    face = model.add_plate(model.add_points(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    edge = model.faces[face].loop[0].edge
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    registry.register(edge, 0., node_id=1, owner="original")
    mesh = Mesh(nodes={1: np.array((0., 0., 0.))})
    seeds = ComponentSeedRegistry(2)
    return mesh, registry, seeds, edge


def test_failed_or_aborted_stage_leaves_all_published_inputs_unchanged():
    mesh, registry, seeds, edge = inputs()
    stage = AuthoredComponentStage(mesh, registry, seeds)
    stage.mesh.nodes[2] = np.array((.5, 0., 0.))
    stage.boundary_registry.register(edge, .5, node_id=2, owner="candidate")
    assert stage.seed_registry.resolve(edge, 1, 2) == 2
    with pytest.raises(MeshError, match="validation failed"):
        stage.finish(lambda *_: (_ for _ in ()).throw(MeshError("validation failed")))
    assert set(mesh.nodes) == {1}
    assert registry.parameters(edge) == (0.,)
    assert seeds.assigned_node_ids == ()
    with pytest.raises(MeshError, match="already closed"):
        stage.finish(lambda *_: None)
    another = AuthoredComponentStage(mesh, registry, seeds)
    another.abort()
    assert set(mesh.nodes) == {1}


def test_finish_returns_independent_result_and_preserves_station_owners():
    mesh, registry, seeds, edge = inputs()
    stage = AuthoredComponentStage(mesh, registry, seeds)
    assert stage.boundary_registry.require(edge, 0.).owners == ("original",)
    stage.mesh.nodes[2] = np.array((.5, 0., 0.))
    stage.boundary_registry.register(edge, .5, node_id=2, owner="candidate")
    def admit(candidate, stations, assigned):
        assert tuple(candidate.nodes[2]) == (.5, 0., 0.)
        assert stations.require(edge, .5).node_id == 2
        assert assigned.assigned_node_ids == ()
        return True

    candidate_mesh, candidate_edges, candidate_seeds = stage.finish(admit)
    assert candidate_mesh is not mesh
    assert candidate_edges is not registry
    assert candidate_seeds is not seeds
    assert registry.parameters(edge) == (0.,)


def test_external_allocator_and_missing_validator_refuse():
    mesh, registry, _seeds, edge = inputs()
    externally_reserved = []

    def allocate():
        externally_reserved.append(2)
        return 2

    seeds = ComponentSeedRegistry(2, node_id_allocator=allocate)
    with pytest.raises(MeshError, match="atomic allocator snapshot"):
        AuthoredComponentStage(mesh, registry, seeds)
    assert externally_reserved == []
    stage = AuthoredComponentStage(mesh, registry, ComponentSeedRegistry(2))
    with pytest.raises(MeshError, match="final validator"):
        stage.finish(None)
    with pytest.raises(MeshError, match="did not admit"):
        stage.finish(lambda *_: None)


def test_seed_fork_preserves_committed_station_ids_and_refuses_unknown_state():
    mesh, registry, seeds, edge = inputs()
    assert seeds.resolve(edge, 1, 4) == 2
    stage = AuthoredComponentStage(mesh, registry, seeds)
    assert stage.seed_registry.resolve(edge, 1, 4) == 2
    assert stage.seed_registry.resolve(edge, 1, 2) == 3
    assert seeds.assigned_node_ids == (2,)
    seeds._unqualified_extension = {"mutable": True}
    with pytest.raises(MeshError, match="no detached staging contract"):
        AuthoredComponentStage(mesh, registry, seeds)


def test_component_pool_fork_keeps_outstanding_reservations_detached():
    mesh, registry, _seeds, edge = inputs()
    pool = ComponentNodeReservationPool(mesh)
    assert pool.allocate() == 2  # A previous failed publication still owns 2.
    seeds = ComponentSeedRegistry(2, reservation_pool=pool)
    holder = AuthoredComponentPublication(mesh, registry, seeds)
    stage = holder.begin()
    assert stage.seed_registry.reservation_pool.mesh is stage.mesh
    assert stage.seed_registry.resolve(edge, 1, 2) == 3
    assert pool.snapshot().reserved_ids == (2,)
    assert stage.seed_registry.reservation_pool.snapshot().reserved_ids == (2, 3)
    with pytest.raises(MeshError, match="validator did not admit"):
        holder.publish(stage, lambda *_: False)
    assert holder.snapshot()[0] == 0
    assert pool.snapshot().reserved_ids == (2,)
    assert registry.parameters(edge) == (0.,)


def test_reservation_fork_rejects_stale_snapshot():
    mesh, _registry, _seeds, _edge = inputs()
    pool = ComponentNodeReservationPool(mesh)
    before = pool.snapshot()
    assert pool.allocate() == 2
    with pytest.raises(MeshError, match="source changed"):
        pool.fork_detached(Mesh(nodes={1: np.array((0., 0., 0.))}), expected=before)
    assert pool.snapshot().reserved_ids == (2,)


def test_cancellation_stale_stage_and_successful_triple_swap():
    mesh, registry, _seeds, edge = inputs()
    pool = ComponentNodeReservationPool(mesh)
    seeds = ComponentSeedRegistry(2, reservation_pool=pool)
    holder = AuthoredComponentPublication(mesh, registry, seeds)
    cancelled = holder.begin()
    cancelled.seed_registry.resolve(edge, 1, 2)

    def cancel(label):
        if label == "authored component before publication":
            raise MeshError("cancelled")

    with pytest.raises(MeshError, match="cancelled"):
        holder.publish(cancelled, lambda *_: True, cancellation_check=cancel)
    assert holder.snapshot() == (0, (mesh, registry, seeds))
    assert pool.snapshot().reserved_ids == ()

    stale = holder.begin()
    winning = holder.begin()
    winning.mesh.nodes[2] = np.array((.5, 0., 0.))
    winning.boundary_registry.register(edge, .5, node_id=2, owner="candidate")
    winning.seed_registry.resolve(edge, 1, 2)
    generation, candidate = holder.publish(winning, lambda *_: True)
    assert generation == 1
    assert candidate[0] is winning.mesh
    assert candidate[1] is winning.boundary_registry
    assert candidate[2] is winning.seed_registry
    assert candidate[2].reservation_pool.origin is None
    assert set(mesh.nodes) == {1}
    assert registry.parameters(edge) == (0.,)
    assert pool.snapshot().reserved_ids == ()
    with pytest.raises(MeshError, match="stale"):
        holder.publish(stale, lambda *_: True)


def test_source_change_during_stage_refuses_publication():
    mesh, registry, seeds, edge = inputs()
    holder = AuthoredComponentPublication(mesh, registry, seeds)
    stage = holder.begin()
    mesh.nodes[2] = np.array((.5, 0., 0.))
    with pytest.raises(MeshError, match="source changed"):
        holder.publish(stage, lambda *_: True)
    assert holder.snapshot()[0] == 0
    assert seeds.assigned_node_ids == ()


def test_stale_owner_validation_discards_candidate_and_reservations():
    model = owner.GeometryModel()
    model.add_plate(model.add_points(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    plan = owner.plan_intersections(model, tuple(model.faces), policy="connect")
    owner.apply_intersections(model, plan, policy="connect")
    scope = owner.query_prepared_model_scope(model)
    mesh = Mesh(nodes={1: np.array((0., 0., 0.))})
    registry = GlobalEdgeBoundaryRegistry(GeometryMeshingView(model))
    pool = ComponentNodeReservationPool(mesh)
    seeds = ComponentSeedRegistry(2, reservation_pool=pool)
    holder = AuthoredComponentPublication(mesh, registry, seeds)
    stage = holder.begin()
    stage.seed_registry.resolve(1, 1, 2)
    model.add_point(10, 10, 10)

    def validate(*_candidate):
        owner.validate_prepared_model_scope_binding(model, scope)
        return True

    with pytest.raises(owner.GeometryError):
        holder.publish(stage, validate)
    assert holder.snapshot()[0] == 0
    assert pool.snapshot().reserved_ids == ()
