"""Per-entity mesh sizes (mode="override") versus refine-only spatial zones."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Plane

from anymesher import EntityRef, Refinement, generate_hybrid_mesh, refine_around
from anymesher.mapped import generate_mesh as generate_mapped_mesh
from anymesher.refinement import SizeField, size_override


@pytest.mark.parametrize("size", [None, 0.5, 2.0])
def test_native_quadratic_publication_removes_only_inactive_recombination_nodes(monkeypatch, size):
    import anymesher.hybrid as hybrid
    from anymesher.quad.high_order import certify_mapping_validity

    def generate():
        geometry, first, _second, names = _two_plates(width=4., height=4.)
        controls = [] if size is None else [size_override(EntityRef("edge", names["a0"]), size)]
        return generate_hybrid_mesh(
            geometry, target_size=1., strategy="native", native_backend="python",
            order="quadratic", beam_edges=[names["a0"]],
            beam_offsets={names["a0"]: .1}, refinements=controls,
        )

    # Independent legacy storage oracle: receipts retain the inactive T6
    # diagonals, but the active element/coupling data must remain byte-identical.
    with monkeypatch.context() as legacy_context:
        legacy_context.setattr(hybrid, "_prune_native_recombination_nodes", lambda mesh: 0)
        legacy = generate()
    mesh = generate()
    assert mesh.quads == legacy.quads and mesh.tris == legacy.tris
    assert mesh.beams == legacy.beams and mesh.couplings == legacy.couplings
    assert len(mesh.nodes) < len(legacy.nodes)
    assert set(mesh.nodes) == _referenced(mesh)
    for node, point in mesh.nodes.items():
        assert point.tobytes() == legacy.nodes[node].tobytes()
    for elements, family in ((mesh.quads, "Q8"), (mesh.tris, "T6")):
        for body in elements.values():
            assert certify_mapping_validity([mesh.nodes[n] for n in body], family).status.value == "CERTIFIED_POSITIVE"


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_smart_mapped_thin_bay_local_size_repairs_internal_seam_circumcenters(order):
    from anymesher.quad.high_order import certify_mapping_validity
    from anymesher.quad.quality_gate import corner_metrics

    geometry = GeometryModel()
    points = geometry.add_points(((0, 0, 0), (8, 0, 0), (8, 2, 0), (0, 2, 0),
                                  (8, 4, 0), (0, 4, 0)))
    first = geometry.add_plate(points[:4])
    shared = geometry.faces[first].loop[2].edge
    second = geometry.add_face_from_loop((
        OrientedEdge(shared, False),
        OrientedEdge(geometry.add_line(points[2], points[4]), True),
        OrientedEdge(geometry.add_line(points[4], points[5]), True),
        OrientedEdge(geometry.add_line(points[5], points[3]), True),
    ), corners=(0, 1, 2, 3))
    revision = geometry.revision
    mesh = generate_hybrid_mesh(
        geometry, target_size=1., strategy="smart_mapped", order=order,
        beam_edges=[shared], beam_offsets={shared: .1},
        refinements=[size_override(EntityRef("face", first), .5)],
    )
    assert geometry.revision == revision
    assert set(mesh.nodes) == _referenced(mesh)
    assert len(mesh.elements_of_face[first]) == 64
    assert all(element in mesh.quads for element in mesh.elements_of_face[first])
    area = 0.
    for elements, family in ((mesh.quads, "Q8"), (mesh.tris, "T6")):
        for body in elements.values():
            corners = body[:4] if family == "Q8" else body[:3]
            xy = np.asarray([mesh.nodes[n][:2] for n in corners])
            area += .5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - xy[:, 1] * np.roll(xy[:, 0], -1))
            assert corner_metrics(xy)[0] >= 15.
            if order == "quadratic":
                assert certify_mapping_validity([mesh.nodes[n] for n in body], family).status.value == "CERTIFIED_POSITIVE"
    assert area == pytest.approx(32., abs=1.e-10)
    for element in mesh.elements_of_face[first]:
        xy = [mesh.nodes[n][:2] for n in mesh.quads[element][:4]]
        assert corner_metrics(xy)[:2] == pytest.approx((90., 90.))


def _plate(length: float = 4.0, width: float = 2.0):
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (length, 0.0, 0.0), (length, width, 0.0), (0.0, width, 0.0))
    )
    face_id = geometry.add_plate(vertices)
    return geometry, face_id


def _mixed_faces():
    """A mapped square sharing one edge with a native triangle."""

    geometry = GeometryModel()
    v0, v1, v2, v3, v4 = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
         (0.0, 1.0, 0.0), (1.8, 0.5, 0.0))
    )
    e0 = geometry.add_line(v0, v1)
    shared = geometry.add_line(v1, v2)
    e2 = geometry.add_line(v2, v3)
    e3 = geometry.add_line(v3, v0)
    e4 = geometry.add_line(v1, v4)
    e5 = geometry.add_line(v4, v2)
    mapped = geometry.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in (e0, shared, e2, e3)),
        corners=(0, 1, 2, 3),
    )
    native = geometry.add_face_from_loop(
        (OrientedEdge(shared, False), OrientedEdge(e4, True), OrientedEdge(e5, True)),
        surface=Plane(
            np.array((1.0, 0.0, 0.0)),
            np.array((0.8, 0.0, 0.0)),
            np.array((0.0, 1.0, 0.0)),
        ),
    )
    return geometry, mapped, native, shared


def _same_shell(first, second) -> None:
    assert first.quads == second.quads
    assert first.tris == second.tris
    assert tuple(first.nodes) == tuple(second.nodes)
    for node in first.nodes:
        assert first.nodes[node].tobytes() == second.nodes[node].tobytes()


def _referenced(mesh) -> set[int]:
    used: set[int] = set()
    for table in (mesh.quads, mesh.tris, mesh.beams):
        for connectivity in table.values():
            used.update(connectivity)
    for coupling in mesh.couplings.values():
        used.add(coupling.beam_node)
        used.update(coupling.plate_nodes)
    return used


# ----------------------------------------------------------------- contract
def test_default_mode_is_refine_only_and_still_refuses_coarse_zones():
    zone = Refinement(size=2.0, ref=EntityRef("face", 1))
    assert zone.mode == "refine"
    geometry, face_id = _plate()
    with pytest.raises(ValueError, match="coarser than the global target"):
        SizeField(geometry, 1.0, (Refinement(size=2.0, ref=EntityRef("face", face_id)),))
    with pytest.raises(ValueError, match="coarser than the global target"):
        SizeField(geometry, 1.0, (refine_around(EntityRef("face", face_id), 2.0),))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "sideways", "ref": EntityRef("face", 1)},
        {"mode": "override", "center": (0.0, 0.0, 0.0)},
        {"mode": "override", "ref": EntityRef("vertex", 1)},
    ],
)
def test_override_mode_validation(kwargs):
    with pytest.raises(ValueError):
        Refinement(size=1.0, **kwargs)


def test_coarse_override_is_entity_scoped_not_a_zone():
    geometry, face_id = _plate()
    field = SizeField(geometry, 0.5, (size_override(EntityRef("face", face_id), 2.0),))
    assert field.face_target(face_id) == 2.0
    assert field.zones == ()  # never a sampled distance zone
    assert field.has_entity_sizes and not field.is_uniform
    assert all(size == 2.0 for size in field.edge_sizes.values())
    assert set(field.edge_sizes) == {item.edge for item in geometry.faces[face_id].loop}
    view = field.face_view(face_id)
    assert view.target_size == 2.0 and view.is_uniform and not view.has_entity_sizes


def test_equal_override_is_a_noop_and_keeps_the_uniform_field():
    geometry, face_id = _plate()
    field = SizeField(geometry, 0.5, (size_override(EntityRef("face", face_id), 0.5),))
    assert field.is_uniform and not field.has_entity_sizes
    assert field.face_view(face_id) is field
    plain = generate_hybrid_mesh(geometry, target_size=0.5, strategy="mapped")
    noop = generate_hybrid_mesh(
        geometry, target_size=0.5, strategy="mapped",
        refinements=[size_override(EntityRef("face", face_id), 0.5)],
    )
    _same_shell(plain, noop)


def test_overlapping_overrides_resolve_to_the_finer_request():
    geometry, face_id = _plate()
    ref = EntityRef("face", face_id)
    first = SizeField(geometry, 1.0, (size_override(ref, 0.5), size_override(ref, 0.25)))
    second = SizeField(geometry, 1.0, (size_override(ref, 0.25), size_override(ref, 0.5)))
    assert first.face_sizes == second.face_sizes == {face_id: 0.25}


def test_spatial_zone_still_reduces_below_an_override_target():
    geometry, face_id = _plate()
    field = SizeField(
        geometry, 1.0,
        (size_override(EntityRef("face", face_id), 2.0),
         Refinement(size=0.25, radius=0.0, ref=EntityRef("edge", geometry.faces[face_id].loop[0].edge))),
    )
    view = field.face_view(face_id)
    assert len(view.zones) == 1
    assert float(view.size_at(np.array([[2.0, 0.0, 0.0]]))[0]) == pytest.approx(0.25)


# ------------------------------------------------- isolated entity equivalence
@pytest.mark.parametrize(("global_size", "local_size"), [(1.0, 0.25), (0.25, 1.0)])
def test_isolated_plate_override_matches_the_equivalent_global_target(global_size, local_size):
    geometry, face_id = _plate()
    local = generate_hybrid_mesh(
        geometry, target_size=global_size, strategy="mapped",
        refinements=[size_override(EntityRef("face", face_id), local_size)],
    )
    other, _ = _plate()
    expected = generate_hybrid_mesh(other, target_size=local_size, strategy="mapped")
    _same_shell(local, expected)


# ------------------------------------------------ adjacent coarse/fine plates
def test_adjacent_coarse_mapped_plate_keeps_the_finer_neighbour_interface():
    geometry, mapped, native, shared = _mixed_faces()
    baseline = generate_hybrid_mesh(
        geometry, target_size=0.28, strategy="auto", native_backend="python"
    )
    coarse = generate_hybrid_mesh(
        geometry, target_size=0.28, strategy="auto", native_backend="python",
        refinements=[size_override(EntityRef("face", mapped), 0.56)],
    )
    assert len(coarse.elements_of_face[mapped]) < len(baseline.elements_of_face[mapped])
    # The shared interface is the finest adjacent request, so it is unchanged.
    assert len(coarse.nodes_of_edge[shared]) == len(baseline.nodes_of_edge[shared])
    nodes = set(coarse.nodes_of_edge[shared])
    for face in (mapped, native):
        used = set()
        for element in coarse.elements_of_face[face]:
            used.update(coarse.quads.get(element, coarse.tris.get(element)))
        assert nodes <= used


def test_adjacent_fine_mapped_plate_refines_the_shared_interface_for_both():
    geometry, mapped, native, shared = _mixed_faces()
    baseline = generate_hybrid_mesh(
        geometry, target_size=0.28, strategy="auto", native_backend="python"
    )
    fine = generate_hybrid_mesh(
        geometry, target_size=0.28, strategy="auto", native_backend="python",
        refinements=[size_override(EntityRef("face", mapped), 0.14)],
    )
    assert len(fine.elements_of_face[mapped]) > len(baseline.elements_of_face[mapped])
    assert len(fine.nodes_of_edge[shared]) > len(baseline.nodes_of_edge[shared])
    used = set()
    for element in fine.elements_of_face[native]:
        used.update(fine.quads.get(element, fine.tris.get(element)))
    assert set(fine.nodes_of_edge[shared]) <= used


# ------------------------------------------------------------ beam own sizes
def _bottom_edge(geometry, face_id) -> int:
    return geometry.faces[face_id].loop[0].edge


def test_coarse_beam_on_a_fine_shell_edge_reuses_chain_nodes_without_couplings():
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    mesh = generate_mapped_mesh(
        geometry, target_size=0.5, beam_edges=[edge],
        refinements=[size_override(EntityRef("edge", edge), 1.0)],
    )
    chain = mesh.nodes_of_edge[edge]
    assert len(chain) == 9  # canonical shell stations stay at the shell size
    assert len(mesh.elements_of_edge[edge]) == 4
    beam_nodes = {node for element in mesh.elements_of_edge[edge] for node in mesh.beams[element]}
    assert beam_nodes <= set(chain) and len(beam_nodes) == 5
    assert not mesh.couplings and not mesh.offset_nodes_of_edge
    assert mesh.hybrid_diagnostics["independent_beam_edges"] == [edge]
    assert set(mesh.nodes) <= _referenced(mesh)


def test_fine_beam_gets_own_nodes_tied_by_interpolation_couplings():
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    mesh = generate_mapped_mesh(
        geometry, target_size=1.0, beam_edges=[edge],
        refinements=[size_override(EntityRef("edge", edge), 0.25)],
    )
    chain = mesh.nodes_of_edge[edge]
    assert len(chain) == 5 and len(mesh.elements_of_edge[edge]) == 16
    new_nodes = {
        node for element in mesh.elements_of_edge[edge] for node in mesh.beams[element]
    } - set(chain)
    assert len(new_nodes) == 12
    assert {item.beam_node for item in mesh.couplings.values()} == new_nodes
    for coupling in mesh.couplings.values():
        assert sum(coupling.weights) == pytest.approx(1.0)
        assert set(coupling.plate_nodes) <= set(chain)
        assert np.linalg.norm(coupling.eccentricity) < 1e-9
    assert set(mesh.nodes) <= _referenced(mesh)


@pytest.mark.parametrize(("beam_size", "expected_couplings"), [(1.0, 9), (4.0 / 3.0, 7)])
def test_quadratic_eccentric_coarse_beam_has_only_real_stations(beam_size, expected_couplings):
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    mesh = generate_mapped_mesh(
        geometry, target_size=0.5, beam_edges=[edge], beam_offsets={edge: 0.1},
        order="quadratic",
        refinements=[size_override(EntityRef("edge", edge), beam_size)],
    )
    divisions = max(1, round(4.0 / beam_size))
    elements = mesh.elements_of_edge[edge]
    assert len(elements) == divisions and all(len(mesh.beams[item]) == 3 for item in elements)
    beam_nodes = {node for item in elements for node in mesh.beams[item]}
    assert len(beam_nodes) == 2 * divisions + 1 == expected_couplings
    # one coupling per actual beam node, and no offset chain aligned to the shell chain
    assert {item.beam_node for item in mesh.couplings.values()} == beam_nodes
    assert not mesh.offset_nodes_of_edge
    chain = set(mesh.nodes_of_edge[edge])
    for coupling in mesh.couplings.values():
        assert sum(coupling.weights) == pytest.approx(1.0)
        assert set(coupling.plate_nodes) <= chain
        assert np.linalg.norm(coupling.eccentricity) == pytest.approx(0.1, abs=1e-9)
    assert set(mesh.nodes) <= _referenced(mesh)


def _two_plates(width: float = 2.0, height: float = 1.0):
    """Two coplanar mapped plates sharing one vertical edge, plus bottom beams."""

    geometry = GeometryModel()
    v0, v1, v2, v3, v4, v5 = geometry.add_points((
        (0.0, 0.0, 0.0), (width, 0.0, 0.0), (2 * width, 0.0, 0.0),
        (2 * width, height, 0.0), (width, height, 0.0), (0.0, height, 0.0),
    ))
    a0 = geometry.add_line(v0, v1)
    shared = geometry.add_line(v1, v4)
    a2 = geometry.add_line(v4, v5)
    a3 = geometry.add_line(v5, v0)
    b0 = geometry.add_line(v1, v2)
    b1 = geometry.add_line(v2, v3)
    b2 = geometry.add_line(v3, v4)
    first = geometry.add_face_from_loop(
        tuple(OrientedEdge(edge, True) for edge in (a0, shared, a2, a3)),
        corners=(0, 1, 2, 3),
    )
    second = geometry.add_face_from_loop(
        (OrientedEdge(b0, True), OrientedEdge(b1, True), OrientedEdge(b2, True),
         OrientedEdge(shared, False)),
        corners=(0, 1, 2, 3),
    )
    return geometry, first, second, {"a0": a0, "b0": b0, "shared": shared, "v1": v1}


@pytest.mark.parametrize("order", ["linear", "quadratic"])
@pytest.mark.parametrize("beam_size", [0.25, 1.0])
def test_resizing_one_of_two_adjacent_eccentric_beams_keeps_the_shared_joint(order, beam_size):
    geometry, _first, _second, names = _two_plates()
    mesh = generate_mapped_mesh(
        geometry, target_size=0.5, order=order,
        beam_edges=[names["a0"], names["b0"]],
        beam_offsets={names["a0"]: 0.1, names["b0"]: 0.1},
        refinements=[size_override(EntityRef("edge", names["a0"]), beam_size)],
    )
    nodes_a = {n for e in mesh.elements_of_edge[names["a0"]] for n in mesh.beams[e]}
    nodes_b = {n for e in mesh.elements_of_edge[names["b0"]] for n in mesh.beams[e]}
    joint = nodes_a & nodes_b
    # the two beams still share exactly the one joint node at the common vertex
    assert len(joint) == 1
    (joint_node,) = joint
    coupling_owners = [item.beam_node for item in mesh.couplings.values()]
    # ... which keeps its coupling to the shell, and nothing is coupled twice
    assert coupling_owners.count(joint_node) == 1
    assert len(coupling_owners) == len(set(coupling_owners))
    vertex_node = mesh.node_of_vertex[names["v1"]]
    joint_coupling = next(item for item in mesh.couplings.values() if item.beam_node == joint_node)
    assert joint_coupling.plate_nodes == (vertex_node,)
    # every beam node of both beams is coupled; no coupling dangles; no unused node
    assert (nodes_a | nodes_b) <= set(coupling_owners)
    assert all(owner in mesh.nodes for owner in coupling_owners)
    assert set(mesh.nodes) <= _referenced(mesh)
    # the untouched beam keeps its ordinary shared-chain ownership
    assert "independent_beam_edges" in mesh.hybrid_diagnostics
    assert mesh.hybrid_diagnostics["independent_beam_edges"] == [names["a0"]]


# ------------------------------------------- validation before the collapse
def test_collapse_never_swallows_a_missing_entity_reference():
    geometry, face_id = _plate()
    with pytest.raises(ValueError, match="999999"):
        generate_hybrid_mesh(
            geometry, target_size=1.0, strategy="mapped",
            refinements=[
                size_override(EntityRef("face", face_id), 0.5),
                size_override(EntityRef("face", 999999), 0.5),
            ],
        )


def test_collapse_keeps_the_original_global_contract_for_spatial_zones():
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    with pytest.raises(ValueError, match="coarser than the global target"):
        generate_hybrid_mesh(
            geometry, target_size=1.0, strategy="mapped",
            refinements=[
                size_override(EntityRef("face", face_id), 2.0),
                Refinement(size=1.5, ref=EntityRef("edge", edge)),
            ],
        )


def test_inert_spatial_zone_is_ignored_after_a_finer_whole_selection_override():
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    local = generate_hybrid_mesh(
        geometry, target_size=1.0, strategy="mapped",
        refinements=[
            size_override(EntityRef("face", face_id), 0.5),
            Refinement(size=0.75, ref=EntityRef("edge", edge)),
        ],
    )
    other, _ = _plate()
    _same_shell(local, generate_hybrid_mesh(other, target_size=0.5, strategy="mapped"))


# -------------------------------------------------- SmartMapped plate sizes
def _smart_plates():
    geometry, first, second, names = _two_plates(width=8.0, height=8.0)
    return geometry, first, second, names


def _smart(geometry, **options):
    return generate_hybrid_mesh(
        geometry, target_size=1.0, strategy="smart_mapped", native_backend="python",
        **options,
    )


def test_smart_mapped_equal_local_size_is_exactly_the_global_mesh():
    geometry, first, _second, _names = _smart_plates()
    plain = _smart(geometry)
    other, other_first, _s, _n = _smart_plates()
    local = _smart(other, refinements=[size_override(EntityRef("face", other_first), 1.0)])
    _same_shell(plain, local)


@pytest.mark.parametrize(("size", "direction"), [(0.5, "more"), (2.0, "fewer")])
def test_smart_mapped_mixed_plate_sizes_share_stations_and_cover_both_plates(size, direction):
    geometry, first, second, names = _smart_plates()
    baseline = _smart(geometry)
    other, other_first, other_second, other_names = _smart_plates()
    mixed = _smart(other, refinements=[size_override(EntityRef("face", other_first), size)])
    assert mixed.hybrid_diagnostics["meshing_strategy"] == "smart_mapped"
    count = lambda mesh, face: len(mesh.elements_of_face[face])
    if direction == "more":
        assert count(mixed, other_first) > count(baseline, first)
    else:
        assert count(mixed, other_first) < count(baseline, first)
    # one canonical interface chain, used by both plates
    chain = set(mixed.nodes_of_edge[other_names["shared"]])
    for face in (other_first, other_second):
        used = set()
        for element in mixed.elements_of_face[face]:
            used.update(mixed.quads.get(element, mixed.tris.get(element)))
        assert chain <= used
    assert set(mixed.nodes) <= _referenced(mixed)


@pytest.mark.parametrize("through_serialization", [False, True])
def test_independent_ownership_comes_from_the_mesh_not_from_diagnostics(through_serialization):
    from anymesher.meshing_view import GeometryMeshingView
    from anymesher.serialize import mesh_from_dict, mesh_to_dict
    from anymesher.structural_pipeline import StructuralMeshingPipeline

    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    coarse = generate_mapped_mesh(
        geometry, target_size=0.5, beam_edges=[edge], beam_offsets={edge: 0.1},
        refinements=[size_override(EntityRef("edge", edge), 1.0)],
    )
    ordinary = generate_mapped_mesh(
        geometry, target_size=0.5, beam_edges=[edge], beam_offsets={edge: 0.1},
    )
    if through_serialization:
        coarse = mesh_from_dict(mesh_to_dict(coarse))
        ordinary = mesh_from_dict(mesh_to_dict(ordinary))
    coarse.hybrid_diagnostics.pop("independent_beam_edges", None)
    pipeline = StructuralMeshingPipeline(
        GeometryMeshingView(geometry),
        overlap_policy="connect_declared", mutation_policy="working_copy",
    )
    axis = pipeline._independent_axis(coarse, edge)
    beam_nodes = {n for e in coarse.elements_of_edge[edge] for n in coarse.beams[e]}
    assert axis is not None and {node for _parameter, node in axis} == beam_nodes
    assert pipeline._independent_axis(ordinary, edge) is None


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_declared_off_grid_attachment_station_is_kept_on_the_independent_beam(order):
    """A real owner attachment point that no uniform beam station would hit."""

    from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange
    from anymesher.meshing_view import GeometryMeshingView
    from anymesher.structural_pipeline import StructuralMeshingPipeline

    geometry, face_id = _plate()  # 4 x 2
    geometry.add_sheet((face_id,))
    edge = _bottom_edge(geometry, face_id)
    member = geometry.add_member((OrientedEdge(edge, True),))
    point = 0.37  # uniform beam corners at 0.25 steps never land here
    geometry.add_attachment(
        member, AttachmentKind.MEMBER_ON_FACE_BOUNDARY, AttachmentTargetKind.EDGE,
        edge, ParameterRange.point(point), (ParameterRange.point(point),),
    )
    mesh = generate_mapped_mesh(
        geometry, target_size=0.5, order=order, beam_edges=[edge],
        refinements=[size_override(EntityRef("edge", edge), 1.0)],
    )
    elements = mesh.elements_of_edge[edge]
    assert len(elements) == 5  # four uniform spans plus the declared station
    beam_nodes = {n for e in elements for n in mesh.beams[e]}
    expected = np.array([point * 4.0, 0.0, 0.0])
    assert any(np.linalg.norm(mesh.nodes[n] - expected) < 1.0e-9 for n in beam_nodes)
    assert set(mesh.nodes) <= _referenced(mesh)
    # declared connectivity still resolves on the actual stations
    pipeline = StructuralMeshingPipeline(
        GeometryMeshingView(geometry),
        overlap_policy="connect_declared", mutation_policy="working_copy",
    )
    report = pipeline.apply_connectivity(mesh)
    assert not report.issues
    owners = [item.beam_node for item in mesh.couplings.values()]
    assert len(owners) == len(set(owners))


class _Cancelled(RuntimeError):
    pass


def test_cancellation_is_polled_during_independent_station_creation():
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    revision = geometry.revision
    phases: list[str] = []

    def cancel_late(phase: str) -> None:
        phases.append(phase)
        if "station 64" in phase:
            raise _Cancelled(phase)

    refinements = [size_override(EntityRef("edge", edge), 0.02)]  # 200 stations
    with pytest.raises(_Cancelled):
        generate_hybrid_mesh(
            geometry, target_size=1.0, strategy="mapped", beam_edges=[edge],
            refinements=refinements, cancellation_check=cancel_late,
        )
    # polled before the indexes, per edge, and inside the edge - not only at start
    assert "independent beams: start" in phases
    assert any(item.startswith("independent beams: edge") and "station" not in item for item in phases)
    assert geometry.revision == revision  # source untouched, nothing published
    # a later, uncancelled run is complete and valid
    mesh = generate_hybrid_mesh(
        geometry, target_size=1.0, strategy="mapped", beam_edges=[edge],
        refinements=refinements,
    )
    assert len(mesh.elements_of_edge[edge]) == 200
    assert set(mesh.nodes) <= _referenced(mesh)


def test_beam_division_count_above_the_seeding_bound_is_a_typed_refusal():
    from anymesher.errors import MeshError

    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    with pytest.raises(MeshError, match="per-edge bound"):
        generate_mapped_mesh(
            geometry, target_size=1.0, beam_edges=[edge],
            refinements=[size_override(EntityRef("edge", edge), 1.0e-6)],
        )


def test_beam_matching_its_chain_is_untouched_and_ordinary_models_never_flag_beams():
    geometry, face_id = _plate()
    edge = _bottom_edge(geometry, face_id)
    plain = generate_mapped_mesh(geometry, target_size=0.5, beam_edges=[edge])
    same = generate_mapped_mesh(
        geometry, target_size=0.5, beam_edges=[edge],
        refinements=[size_override(EntityRef("edge", edge), 0.5)],
    )
    assert "independent_beam_edges" not in plain.hybrid_diagnostics
    assert "independent_beam_edges" not in same.hybrid_diagnostics
    assert plain.beams == same.beams and plain.couplings == same.couplings
    _same_shell(plain, same)
