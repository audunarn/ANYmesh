"""Provisional component changes stay detached until a validated return."""

import numpy as np
import pytest

import anygeometry as owner
from anymesher._authored_component_stage import AuthoredComponentStage
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
