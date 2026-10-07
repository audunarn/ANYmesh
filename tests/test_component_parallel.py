"""Contract tests for component-parallel meshing (research prototype)."""

from __future__ import annotations

import numpy as np
import pytest
from anygeometry import GeometryModel
from anygeometry.entities import OrientedEdge
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    ParameterRange,
)

from anymesher import generate_hybrid_mesh_result
import anygeometry
from anygeometry.errors import GeometryError
from anymesher import (
    Mesh,
    ParallelOptions,
    ParallelPoolLease,
    create_parallel_pool,
    generate_hybrid_mesh_parallel,
)
from anymesher.component_parallel import generate_hybrid_mesh_result_parallel
from anymesher._parallel_pool import _pid_alive
from anymesher.errors import MeshError

pytestmark = pytest.mark.skipif(
    not hasattr(anygeometry, "plan_independent_components"),
    reason="needs ANYgeometry with plan_independent_components",
)

TARGET = 0.25


def plan(model, *, separation):
    return anygeometry.plan_independent_components(model, separation=separation)


def _stiffened_plate(model: GeometryModel, x: float, y: float = 0.0) -> int:
    vertices = model.add_points(
        ((x, y, 0.0), (x + 2.0, y, 0.0), (x + 2.0, y + 1.0, 0.0), (x, y + 1.0, 0.0))
    )
    face = model.add_plate(vertices)
    model.add_sheet((face,))
    edge = model.faces[face].loop[0].edge
    member = model.add_member((OrientedEdge(edge, False),))
    model.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE_BOUNDARY,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
    )
    return face


def _pierced_plate(model: GeometryModel, x: float) -> int:
    """A plate with a member through its interior; meshing couples the nodes."""

    vertices = model.add_points(
        ((x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x + 1.0, 1.0, 0.0), (x, 1.0, 0.0))
    )
    face = model.add_plate(vertices)
    model.add_sheet((face,))
    ends = model.add_points(((x + 0.37, 0.41, -1.0), (x + 0.37, 0.41, 1.0)))
    edge = model.add_line(*ends)
    member = model.add_member((edge,))
    model.add_attachment(
        member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.37), ParameterRange.point(0.41)),
        metadata={"face_sequence": [face]},
    )
    return face


def _model(count: int, spacing: float) -> GeometryModel:
    model = GeometryModel()
    for index in range(count):
        _stiffened_plate(model, index * spacing)
    return model


@pytest.fixture(scope="module")
def pool():
    lease = create_parallel_pool(2)
    yield lease
    lease.close()


def _signature(mesh):
    """Numbering-independent description of a mesh."""

    def position(node):
        return tuple(np.round(mesh.nodes[node], 9))

    def centroid(element):
        nodes = mesh.shells.get(element) or mesh.beams[element]
        return tuple(np.round(np.mean([mesh.nodes[n] for n in nodes], axis=0), 9))

    return {
        "nodes": sorted(position(n) for n in mesh.nodes),
        "faces": {f: sorted(centroid(e) for e in els) for f, els in mesh.elements_of_face.items()},
        "edges": {e: sorted(position(n) for n in ns) for e, ns in mesh.nodes_of_edge.items()},
        "members": {m: sorted(centroid(e) for e in els) for m, els in mesh.elements_of_member.items()},
        "counts": (len(mesh.quads), len(mesh.tris), len(mesh.beams), len(mesh.couplings)),
        "vertices": {v: position(n) for v, n in mesh.node_of_vertex.items()},
    }


def test_owner_planner_separates_distant_components_and_keeps_structure_together():
    partition = plan(_model(3, spacing=10.0), separation=TARGET / 2)
    assert partition.certified
    assert len(partition.components) == 3
    for component in partition.components:
        assert len(component.face_ids) == len(component.member_ids) == 1


def test_owner_planner_merges_inside_twice_the_separation():
    # 0.1 m gap: a separation of 0.125 (gap up to 0.25 merges) must merge,
    # 0.025 (gap up to 0.05) must not.
    assert len(plan(_model(2, spacing=2.1), separation=TARGET / 2).components) == 1
    assert len(plan(_model(2, spacing=2.1), separation=0.025).components) == 2


def test_owner_planner_unites_components_sharing_a_vertex():
    model = GeometryModel()
    shared = model.add_points(((1.0, 0.0, 0.0), (1.0, 1.0, 0.0)))
    first = model.add_points(((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    second = model.add_points(((2.0, 0.0, 0.0), (2.0, 1.0, 0.0)))
    model.add_plate((first[0], shared[0], shared[1], first[1]))
    model.add_plate((shared[0], second[0], second[1], shared[1]))
    assert len(plan(model, separation=1.0e-9).components) == 1


def test_close_components_fall_back_to_serial_with_a_reason():
    # Gap 0.1 < target size 0.25: the mesher merges them, leaving one component.
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=2.1), target_size=TARGET
    )
    assert result.mesh.hybrid_diagnostics["parallel"] == {
        "used": False, "route": "serial", "reason": "1 component(s)",
    }


def test_refused_partition_falls_back_to_serial(monkeypatch):
    class Refusal:
        reason = "face support family lacks certified full trim-domain conservative bounds"

    class Refused:
        certified = False
        components = ()
        merge_reasons = ()
        refusals = (Refusal(),)

    monkeypatch.setattr(anygeometry, "plan_independent_components", lambda *a, **k: Refused())
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0), target_size=TARGET
    )
    info = result.mesh.hybrid_diagnostics["parallel"]
    assert info["used"] is False and info["reason"].startswith("partition refused: face support")


def test_missing_owner_planner_falls_back_to_serial(monkeypatch):
    monkeypatch.delattr(anygeometry, "plan_independent_components")
    result = generate_hybrid_mesh_result_parallel(
        _model(2, spacing=10.0), target_size=TARGET
    )
    assert result.mesh.hybrid_diagnostics["parallel"] == {
        "used": False, "route": "serial",
        "reason": "ANYgeometry has no plan_independent_components",
    }


def test_parallel_result_matches_serial_up_to_numbering(pool):
    model = _model(3, spacing=10.0)
    serial = generate_hybrid_mesh_result(model, target_size=TARGET)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool)
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert parallel.mesh.hybrid_diagnostics["parallel"]["components"] == 3
    assert _signature(parallel.mesh) == _signature(serial.mesh)
    assert dict(parallel.strategy_by_face) == dict(serial.strategy_by_face)
    assert parallel.mesh.order == serial.mesh.order
    assert parallel.mesh.geometry_model_id == model.model_id
    assert (
        sorted(parallel.mesh.declared_plate_junction_edges, key=str) != []
        or serial.mesh.declared_plate_junction_edges == ()
    )
    assert len(parallel.mesh.boundary_registry) == len(serial.mesh.boundary_registry)
    assert len(parallel.preflight) == len(serial.preflight)


def _couplings(mesh):
    def at(n):
        return tuple(np.round(mesh.nodes[n], 9))

    return sorted(
        (at(c.beam_node), tuple(sorted(at(n) for n in c.plate_nodes)),
         tuple(np.round(c.eccentricity, 9)))
        for c in mesh.couplings.values()
    )


def test_pierced_plate_components_match_serial(pool):
    model = GeometryModel()
    for index in range(2):
        _pierced_plate(model, index * 10.0)
    serial = generate_hybrid_mesh_result(model, target_size=TARGET)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool)
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert _signature(parallel.mesh) == _signature(serial.mesh)


def test_eccentric_stiffener_couplings_survive_the_join(pool):
    model = _model(2, spacing=10.0)
    offsets = {
        model.faces[face].loop[0].edge: 0.05 for face in sorted(model.faces)
    }
    options = {"beam_offsets": offsets}
    serial = generate_hybrid_mesh_result(model, target_size=TARGET, **options)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool), **options
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert len(serial.mesh.couplings) > 0
    assert _couplings(parallel.mesh) == _couplings(serial.mesh)
    assert sorted(parallel.mesh.offset_nodes_of_edge) == sorted(serial.mesh.offset_nodes_of_edge)
    assert _signature(parallel.mesh) == _signature(serial.mesh)
    assert parallel.connectivity.connected == serial.connectivity.connected


def test_refinements_and_quadratic_order_match_serial(pool):
    from anymesher.refinement import Refinement

    model = _model(2, spacing=10.0)
    options = {
        "order": "quadratic",
        "refinements": (
            Refinement(size=0.1, radius=0.5, center=(0.5, 0.5, 0.0), growth=1.5, name="r"),
        ),
    }
    serial = generate_hybrid_mesh_result(model, target_size=TARGET, **options)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool), **options
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert parallel.mesh.order == "quadratic"
    assert _signature(parallel.mesh) == _signature(serial.mesh)


def test_diagnostics_carry_the_keys_consumers_read(pool):
    model = _model(2, spacing=10.0)
    serial = generate_hybrid_mesh_result(model, target_size=TARGET).mesh.hybrid_diagnostics
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool)
    ).mesh.hybrid_diagnostics
    for key in (
        "requested_target_size", "strategy_by_face", "geometry_model_id",
        "geometry_revision", "certification_mode", "certifiable", "preflight_count",
    ):
        assert parallel[key] == serial[key], key
    assert parallel["complex_geometry"]["source_face_count"] == (
        serial["complex_geometry"]["source_face_count"]
    )
    assert set(parallel["triangulation_backend_by_face"]) == set(
        serial["triangulation_backend_by_face"]
    )


def test_a_dead_worker_is_reported_as_a_mesh_error(monkeypatch):
    """An actual worker crash mid-job is a MeshError; the lease invalidates."""

    monkeypatch.setenv("_ANYMESHER_PARALLEL_TEST_CRASH", "1")
    lease = create_parallel_pool(2)
    try:
        with pytest.raises(MeshError, match="worker process died"):
            generate_hybrid_mesh_result_parallel(
                _model(2, spacing=10.0),
                target_size=TARGET,
                parallel=ParallelOptions(pool_lease=lease),
            )
        assert not lease.valid
        assert not any(_pid_alive(pid) for pid in lease.worker_pids)
    finally:
        monkeypatch.delenv("_ANYMESHER_PARALLEL_TEST_CRASH", raising=False)
        lease.close()


def test_package_exports_the_parallel_entry_points():
    import anymesher

    for name in (
        "generate_hybrid_mesh_result_parallel",
        "generate_hybrid_mesh_parallel",
        "ParallelOptions",
        "ParallelPoolLease",
        "create_parallel_pool",
    ):
        assert name in anymesher.__all__ and hasattr(anymesher, name)


def _plain_plates(count: int) -> GeometryModel:
    model = GeometryModel()
    for index in range(count):
        x = index * 10.0
        vertices = model.add_points(
            ((x, 0.0, 0.0), (x + 2.0, 0.0, 0.0), (x + 2.0, 1.0, 0.0), (x, 1.0, 0.0))
        )
        model.add_sheet((model.add_plate(vertices),))
    return model


def test_native_strategy_single_face_components_match_the_whole_model_run(pool):
    """The planar chart metric depends on the model's face count; a component
    meshed alone must still produce the whole-model mesh."""

    model = _plain_plates(3)
    serial = generate_hybrid_mesh_result(model, target_size=TARGET, strategy="native")
    alone = generate_hybrid_mesh_result(
        _plain_plates(1), target_size=TARGET, strategy="native"
    )
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, strategy="native",
        parallel=ParallelOptions(pool_lease=pool),
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    # Premise: a plate meshed alone differs from the same plate in a larger model.
    assert alone.mesh.num_elements * 3 != serial.mesh.num_elements
    assert _signature(parallel.mesh) == _signature(serial.mesh)


def test_merged_ids_are_unique_and_references_resolve(pool):
    model = _model(3, spacing=10.0)
    mesh = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool)
    ).mesh
    tables = (mesh.quads, mesh.tris, mesh.beams, mesh.couplings)
    element_ids = [key for table in tables for key in table]
    assert len(element_ids) == len(set(element_ids))
    for nodes in (*mesh.quads.values(), *mesh.tris.values(), *mesh.beams.values()):
        assert all(node in mesh.nodes for node in nodes)
    assert set(mesh.elements_of_face) <= set(model.faces)
    assert set(mesh.elements_of_member) <= set(model.members)
    assert set(mesh.nodes_of_edge) <= set(model.edges)
    assert set(mesh.node_of_vertex) <= set(model.vertices)
    for ids in mesh.elements_of_face.values():
        assert all(i in mesh.quads or i in mesh.tris for i in ids)
    for ids in mesh.elements_of_member.values():
        assert all(i in mesh.beams or i in mesh.couplings for i in ids)


def test_overrides_are_routed_to_their_component(pool):
    model = _model(2, spacing=10.0)
    edge = model.faces[1].loop[1].edge
    options = {"overrides": {edge: 12}}
    serial = generate_hybrid_mesh_result(model, target_size=TARGET, **options)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool), **options
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert _signature(parallel.mesh) == _signature(serial.mesh)


@pytest.mark.parametrize(
    "extra, reason",
    [
        ({"face_ids": (1,)}, "partial face selection"),
        ({"member_ids": ()}, "partial member selection"),
        ({"certification_mode": "strict"}, "certification requested"),
        ({"change_set": object()}, "unsupported option change_set"),
    ],
)
def test_unsupported_options_fall_back_to_serial_with_a_reason(extra, reason, monkeypatch):
    import anymesher.component_parallel as module

    seen = []

    def record(geometry, options, why):
        seen.append(why)
        return "serial"

    monkeypatch.setattr(module, "_serial", record)
    model = _model(2, spacing=10.0)
    assert generate_hybrid_mesh_result_parallel(model, target_size=TARGET, **extra) == "serial"
    assert seen == [reason]


def test_interactive_certification_without_a_change_set_runs_in_parallel(pool):
    """ANYfem's default mode."""

    model = _model(2, spacing=10.0)
    options = {"certification_mode": "interactive"}
    serial = generate_hybrid_mesh_result(model, target_size=TARGET, **options)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, parallel=ParallelOptions(pool_lease=pool), **options
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert parallel.certification_mode == serial.certification_mode
    assert parallel.certifiable == serial.certifiable
    assert parallel.mesh.hybrid_diagnostics["certification_mode"] == "interactive"
    assert _signature(parallel.mesh) == _signature(serial.mesh)


def test_working_copy_and_full_selections_run_in_parallel(pool):
    """ANYfem calls the mesher with exactly these arguments."""

    model = _model(3, spacing=10.0)
    options = {
        "mutation_policy": "working_copy",
        "face_ids": tuple(sorted(model.faces)),
        "member_ids": tuple(sorted(model.members)),
    }
    serial = generate_hybrid_mesh_result(model.clone(), target_size=TARGET, **options)
    parallel = generate_hybrid_mesh_result_parallel(
        model.clone(), target_size=TARGET, parallel=ParallelOptions(pool_lease=pool),
        **options,
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert _signature(parallel.mesh) == _signature(serial.mesh)


def test_precomputed_seeding_is_split_per_component(pool):
    from anymesher import solve_seeding

    model = _model(3, spacing=10.0)
    seeding = solve_seeding(model, target_size=TARGET)
    serial = generate_hybrid_mesh_result(model, target_size=TARGET, seeding=seeding)
    parallel = generate_hybrid_mesh_result_parallel(
        model, target_size=TARGET, seeding=seeding,
        parallel=ParallelOptions(pool_lease=pool),
    )
    assert parallel.mesh.hybrid_diagnostics["parallel"]["used"] is True
    assert _signature(parallel.mesh) == _signature(serial.mesh)
    assert parallel.mesh.seeding.divisions == serial.mesh.seeding.divisions


def test_single_component_falls_back_to_serial():
    result = generate_hybrid_mesh_result_parallel(
        _model(1, spacing=10.0), target_size=TARGET
    )
    assert result.mesh.hybrid_diagnostics["parallel"]["used"] is False


def test_cancellation_check_is_honoured_between_components(pool):
    class Cancelled(RuntimeError):
        pass

    def stop(_phase: str) -> None:
        raise Cancelled

    with pytest.raises(Cancelled):
        generate_hybrid_mesh_result_parallel(
            _model(3, spacing=10.0),
            target_size=TARGET,
            parallel=ParallelOptions(pool_lease=pool),
            cancellation_check=stop,
        )
